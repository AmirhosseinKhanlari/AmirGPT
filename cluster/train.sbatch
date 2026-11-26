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
# GPT-2 pretraining from scratch on FineWeb-Edu 10B, Karpathy's hyperparameters, 8x V100 32GB.
#   batch: 524,288 tokens = MICRO_BATCH seq x 1024 tok x 8 GPUs x grad-accum (video: 64 x 1024 x 8 x 1)
#   val:   always the first 10,485,760 val tokens, identical to the video's 20 x 64 x 1024 x 8
#   fp16 + GradScaler (V100 has no bf16), torch.compile on, document/shard shuffling on ([3:50:05])
# Re-submitting with the same LOG_DIR resumes from $LOG_DIR/latest.pt.
#
# Runs (env vars; any extra arguments are passed to gpt2.train_gpt2):
#   sbatch cluster/train.sbatch                                     # 124M, 10B tokens (the video's main run)
set -uo pipefail
cd "$HOME/AmirGPT"
LOG_DIR=${LOG_DIR:-log_124M_10B}
MICRO_BATCH=${MICRO_BATCH:-32}
VAL_STEPS=$(( 10485760 / (MICRO_BATCH * 1024 * 8) ))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation between train/eval shapes
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "=== training: log_dir=$LOG_DIR micro_batch=$MICRO_BATCH val_steps=$VAL_STEPS extra: $*"
venv/bin/torchrun --standalone --nproc_per_node=8 -m gpt2.train_gpt2 \
  --data_root edu_fineweb10B --log_dir "$LOG_DIR" \
  --micro_batch "$MICRO_BATCH" --val_steps "$VAL_STEPS" --dtype auto --compile 1 --shuffle 1 --resume 1 --latest_interval 250 "$@"
echo "TRAIN EXIT $?"
