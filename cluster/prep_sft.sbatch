#!/bin/bash
#SBATCH --job-name=amirgpt-sftdata
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=tesla2
#SBATCH --output=logs/%x-%j.out
# Download + tokenize the smol-smoltalk chat dataset for SFT (CPU only).
set -euo pipefail
cd "$HOME/AmirGPT"
venv/bin/python -m sft.prepare_sft --out sft_data
echo "SFT DATA DONE"
