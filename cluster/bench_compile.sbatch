#!/bin/bash
#SBATCH --job-name=amirgpt-bench
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
# torch.compile throughput on 8xV100 using the uv-managed-Python venv (has Python.h for Triton).
set -uo pipefail
cd "$HOME/AmirGPT"
FAKE=/tmp/amirgpt_fake_$SLURM_JOB_ID
venv/bin/python - <<PYEOF
import numpy as np, os
os.makedirs("$FAKE", exist_ok=True)
rng = np.random.default_rng(0)
for i in range(3):
    np.save(f"$FAKE/edufineweb_{'val' if i == 0 else 'train'}_{i:06d}.npy", rng.integers(0, 50257, 30_000_000).astype(np.uint16))
PYEOF
bench () {
  echo "--- B=$1 compile=$2"
  timeout 1200 venv/bin/torchrun --standalone --nproc_per_node=8 -m gpt2.train_gpt2 --data_root $FAKE \
    --log_dir /tmp/amirgpt_bench_$SLURM_JOB_ID --micro_batch $1 --compile $2 --dtype auto --max_steps 15 \
    --hellaswag 0 --eval_interval 100000 --sample_interval 100000 --val_steps 2 --ckpt_interval 100000 \
    --latest_interval 100000 2>&1 | grep -E "step +(0|5|10|13) |accumulation|Error|out of memory" | head -8
}
bench 16 1
bench 32 1
rm -rf $FAKE /tmp/amirgpt_bench_$SLURM_JOB_ID
echo "BENCH DONE"
