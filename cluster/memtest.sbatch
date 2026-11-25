#!/bin/bash
#SBATCH --job-name=amirgpt-memtest
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Reproduces the step-250 OOM scenario (compiled training -> eager eval) in 45 steps.
set -uo pipefail
cd "$HOME/AmirGPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
venv/bin/torchrun --standalone --nproc_per_node=8 -m gpt2.train_gpt2 --data_root edu_fineweb10B \
  --log_dir /tmp/amirgpt_memtest_$SLURM_JOB_ID --micro_batch 32 --val_steps 40 --dtype auto --compile 1 --shuffle 1 \
  --max_steps 45 --eval_interval 20 --sample_interval 20 --hellaswag_limit 400 --ckpt_interval 100000 --latest_interval 40 2>&1 \
  | grep -E "validation loss|HellaSwag|rank 0 sample|^step +(19|20|21|39|40|41|44) |Error|error|Traceback|memory"
echo "EXIT ${PIPESTATUS[0]}"
ls -la /tmp/amirgpt_memtest_$SLURM_JOB_ID; rm -rf /tmp/amirgpt_memtest_$SLURM_JOB_ID
