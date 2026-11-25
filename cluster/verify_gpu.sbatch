#!/bin/bash
#SBATCH --job-name=amirgpt-verify
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Correctness + throughput checks on the 8xV100 node before the real run.
set -uo pipefail
cd "$HOME/AmirGPT"
PY=.venv/bin/python
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
$PY -c "import torch; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0))"

echo "=== 1) unit tests"
$PY -m pytest tests -q 2>&1 | tail -3

echo "=== 2) HellaSwag: OUR GPT class + OpenAI GPT-2 124M weights (expect acc_norm 0.2955)"
$PY -m gpt2.hellaswag -m gpt2 -d cuda 2>&1 | grep -v "it/s\]" | tail -2

echo "=== 3) throughput benchmarks (synthetic tokens; speed does not depend on content)"
FAKE=/tmp/amirgpt_fake_$SLURM_JOB_ID
$PY - <<PYEOF
import numpy as np, os
os.makedirs("$FAKE", exist_ok=True)
rng = np.random.default_rng(0)
for i in range(3):
    np.save(f"$FAKE/edufineweb_{'val' if i == 0 else 'train'}_{i:06d}.npy", rng.integers(0, 50257, 30_000_000).astype(np.uint16))
PYEOF
bench () {  # $1=micro_batch $2=compile
  echo "--- B=$1 compile=$2"
  timeout 900 .venv/bin/torchrun --standalone --nproc_per_node=8 -m gpt2.train_gpt2 --data_root $FAKE \
    --log_dir /tmp/amirgpt_bench_$SLURM_JOB_ID --micro_batch $1 --compile $2 --dtype auto --max_steps 12 \
    --hellaswag 0 --eval_interval 100000 --sample_interval 100000 --val_steps 2 --ckpt_interval 100000 \
    --latest_interval 100000 2>&1 | grep -E "step +(0|5|9|10|11) |mixed precision|accumulation|Error|error|out of memory" | head -12
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
}
bench 16 0
bench 16 1
bench 32 1
rm -rf $FAKE /tmp/amirgpt_bench_$SLURM_JOB_ID
echo "VERIFY DONE"
