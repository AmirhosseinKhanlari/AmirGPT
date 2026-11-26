#!/bin/bash
#SBATCH --job-name=amirgpt-dpodata
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Download + tokenize UltraFeedback-binarized preference pairs for DPO (CPU only, no GPUs).
set -euo pipefail
cd "$HOME/AmirGPT"
venv/bin/python -m sft.prepare_dpo --out dpo_data
echo "DPO DATA DONE"
