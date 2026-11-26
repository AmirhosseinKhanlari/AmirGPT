#!/bin/bash
#SBATCH --job-name=amirgpt-eval
#SBATCH --partition=defq
#SBATCH --nodelist=tesla2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Independent final evaluation (fp32, outside the training loop) of the last pretraining checkpoint,
# next to OpenAI's GPT-2 124M on the exact same data:
#   val loss on the first 10,485,760 tokens of our FineWeb-Edu val shard, and HellaSwag (10,042 examples).
set -euo pipefail
cd ~/AmirGPT
CKPT=${1:-log_124M_10B/model_19072.pt}
echo "== val loss: ours ($CKPT)";  venv/bin/python -m gpt2.eval_val --checkpoint "$CKPT"
echo "== val loss: OpenAI GPT-2";  venv/bin/python -m gpt2.eval_val --model gpt2
echo "== HellaSwag: ours";         venv/bin/python -m gpt2.hellaswag --checkpoint "$CKPT" | tail -2
echo "EVAL EXIT 0"
