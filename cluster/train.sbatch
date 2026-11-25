#!/bin/bash
#SBATCH --job-name=amirgpt-train
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
# GPT-2 124M from scratch on FineWeb-Edu 10B, Karpathy's hyperparameters, 8x V100 32GB.
#   batch: 524,288 tokens = 32 seq x 1024 tok x 8 GPUs x 2 grad-accum steps (video uses 64 x 1024 x 8 x 1)
#   val:   40 steps x 32 x 1024 x 8 = first 10,485,760 val tokens, identical to the video's 20 x 64 x 1024 x 8
#   fp16 + GradScaler (V100 has no bf16), torch.compile on, document/shard shuffling on ([3:50:05])
# Re-submitting this script resumes from log_124M_10B/latest.pt.
set -uo pipefail
cd "$HOME/AmirGPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation between train/eval shapes
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
# pre-flight (already run once in job 98005): OpenAI GPT-2 val loss on our val shard = 3.2793
# (Karpathy play.ipynb baseline: 3.2924). Re-run with: venv/bin/python -m gpt2.eval_val --model gpt2
echo "=== training"
venv/bin/torchrun --standalone --nproc_per_node=8 -m gpt2.train_gpt2 \
  --data_root edu_fineweb10B --log_dir log_124M_10B \
  --micro_batch 32 --val_steps 40 --dtype auto --compile 1 --shuffle 1 --resume 1 --latest_interval 250
echo "TRAIN EXIT $?"
