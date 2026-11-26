#!/bin/bash
#SBATCH --job-name=amirgpt-dpo
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# DPO preference tuning of the AmirGPT chat model (one run, one learning rate).
# Usage: [OUT=log_dpo_x] sbatch cluster/dpo.sbatch [init checkpoint] [extra train_dpo args]
# Needs dpo_data/ (cluster/prep_dpo.sbatch) and sft_data/val_* (for the SFT val loss).
#
# Defaults, and why:
#   1 epoch of the filtered pairs (51,635), 8 pairs/GPU x 8 GPUs = 64 pairs/step -> 806 steps.
#   --max_lr 1e-6: from local MPS runs on the real chat model (1,600 pairs <= 512 tokens, 8 pairs/step,
#     200 steps, cosine; 200 held-out test pairs; 50-prompt quality eval). End of run:
#       lr     test loss  reward acc  margin  SFT val loss     mean reply tokens  replies hitting 256 cap
#       init   0.693      0.00        0.000   1.3175           188.6              42%
#       1e-6   0.673      0.62        0.053   1.3180           203.0              48%
#       3e-6   0.663      0.605       0.112   1.3203           214.3              62%
#       1e-5   0.700      0.565       0.162   1.3346           235.2              78%
#     Replies get longer as the LR grows (UltraFeedback's chosen replies are longer than the rejected
#     ones: 444.8 vs 399.3 tokens on average). This run takes 806 steps instead of 200, and AdamW moves
#     each weight by roughly lr per step, so the total movement scales with sum(lr): cluster 1e-6 ~ 4.4e-4,
#     between local 3e-6 (~3.3e-4) and local 1e-5 (~1.1e-3), while cluster 3e-6 (~1.3e-3) would be in the
#     regime where the local 1e-5 run lengthened replies most and raised the SFT val loss most.
#   --beta 0.1: the DPO default asked for; larger beta keeps the model closer to the SFT model.
#   cosine schedule with 10% warmup, weight decay 0 (we only want to move the model a little
#   away from the SFT reference), grad clip 1.0 (as in pretraining / SFT).
#   eval on all 1,660 test pairs every 100 steps; best checkpoint = highest test reward accuracy.
#   --time: ~6-8 min of work estimated from SFT's measured throughput (637.9M tokens in 35 min), plus margin.
# To use fewer pairs (shorter run), add e.g. --max_pairs 25600.
set -uo pipefail
cd "$HOME/AmirGPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT=${1:-log_sft/amirgpt_best.pt}
shift || true
OUT=${OUT:-log_dpo}
echo "init from: $INIT -> $OUT"
venv/bin/torchrun --standalone --nproc_per_node=8 -m sft.train_dpo --init "$INIT" --data dpo_data --sft_data sft_data \
  --out "$OUT" --epochs 1 --micro_batch 8 --grad_accum 1 --max_lr 1e-6 --beta 0.1 --schedule cosine \
  --warmup_ratio 0.1 --eval_interval 100 "$@"
STATUS=$?
echo "DPO EXIT $STATUS"
venv/bin/python -c "import sys; from sft.train_dpo import print_summary; print_summary(sys.argv[1:])" "$OUT"
exit $STATUS
