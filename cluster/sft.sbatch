#!/bin/bash
#SBATCH --job-name=amirgpt-sft
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Chat SFT of our pretrained GPT-2 (Karpathy [3:55:46]: "swap out the data set and continue training").
# Usage: [OUT=log_sft_x] sbatch cluster/sft.sbatch <pretrained checkpoint> [extra train_sft args]
set -uo pipefail
cd "$HOME/AmirGPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT=${1:-$(ls -1 log_124M_10B/model_*.pt | sort | tail -1)}
shift || true
OUT=${OUT:-log_sft}
echo "init from: $INIT -> $OUT"
venv/bin/torchrun --standalone --nproc_per_node=8 -m sft.train_sft --init "$INIT" --data sft_data --out "$OUT" \
  --epochs 2 --micro_batch 16 --grad_accum 1 --max_lr 1e-4 --warmup_steps 100 --eval_interval 250 "$@"
echo "SFT EXIT $?"
