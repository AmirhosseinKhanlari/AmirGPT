#!/bin/bash
#SBATCH --job-name=amirgpt-bench
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# (1) Data-parallel scaling: training throughput of GPT-2 124M on 1, 2, 4, 8 V100s
#     (same 524,288-token batch, micro-batch 32, fp16, torch.compile; steps 5..24 of 25 are timed).
set -uo pipefail
cd "$HOME/AmirGPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=bench_results; mkdir -p $OUT
COMMON="--data_root edu_fineweb10B --dtype auto --compile 1 --shuffle 1 --eval_interval 100000 --hellaswag 0
        --sample_interval 100000 --ckpt_interval 100000 --latest_interval 100000 --val_steps 1"
for N in 1 2 4 8; do
  echo "=== $N GPU(s)"
  venv/bin/torchrun --standalone --nproc_per_node=$N -m gpt2.train_gpt2 $COMMON --micro_batch 32 --max_steps 25 \
    --log_dir /tmp/amirgpt_bench_${SLURM_JOB_ID}_$N > $OUT/scaling_${N}gpu.log 2>&1
  echo "exit $?"; grep -E "^step +(5|24) " $OUT/scaling_${N}gpu.log
  rm -rf /tmp/amirgpt_bench_${SLURM_JOB_ID}_$N
done
venv/bin/python - <<'PY'
import re, json, statistics
res = {}
for n in (1, 2, 4, 8):
    ts = [float(m.group(2)) for m in re.finditer(r"^step +(\d+) .*tok/sec: ([\d.]+)", open(f"bench_results/scaling_{n}gpu.log").read(), re.M)
          if int(m.group(1)) >= 5]
    res[n] = statistics.median(ts) if ts else None
base = res[1]
out = {n: {"tok_per_sec": t, "speedup": t / base if t and base else None, "efficiency": t / base / n if t and base else None} for n, t in res.items()}
print(json.dumps(out, indent=1)); json.dump(out, open("bench_results/scaling.json", "w"), indent=1)
PY
nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1
echo "BENCH EXIT 0"
