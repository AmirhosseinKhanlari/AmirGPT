#!/bin/bash
#SBATCH --job-name=amirgpt-datastats
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# CPU-only: token/document statistics of the FineWeb-Edu shards for the report.
set -euo pipefail
cd "$HOME/AmirGPT"
.venv/bin/python -m gpt2.data_stats --data_root edu_fineweb10B --out bench_results/data_stats.json --workers 16
echo "STATS EXIT 0"
