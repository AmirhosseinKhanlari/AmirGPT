#!/bin/bash
#SBATCH --job-name=amirgpt-data
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=tesla2
#SBATCH --output=logs/%x-%j.out
# 1) download FineWeb-Edu sample-10BT parquet (HTTPS, verified)  2) tokenize into 100M-token uint16 shards
# 3) download the HellaSwag val set. CPU only.
set -euo pipefail
cd "$HOME/AmirGPT"
bash cluster/download_fineweb.sh "$HOME/AmirGPT/raw_fineweb"
.venv/bin/python -m gpt2.fineweb --parquet_dir raw_fineweb --out edu_fineweb10B --workers 22
.venv/bin/python -m gpt2.hellaswag --download_only
echo "PREP DONE"
