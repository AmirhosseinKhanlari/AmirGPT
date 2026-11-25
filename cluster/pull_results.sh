#!/bin/bash
# Copy training logs and final checkpoints from the cluster (ssh alias "sbu") to this Mac.
#   logs_from_cluster/pretrain/   metrics.jsonl, log.txt, run_config.json   (Lab tab)
#   logs_from_cluster/sft/        metrics.jsonl, sft_config.json
#   checkpoints/amirgpt_base.pt   newest pretraining snapshot (model weights only)
#   checkpoints/amirgpt_chat.pt   best SFT checkpoint (lowest val loss)
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${HOST:-sbu}
mkdir -p logs_from_cluster/pretrain logs_from_cluster/sft checkpoints
rsync -az "$HOST:AmirGPT/log_124M_10B/metrics.jsonl" "$HOST:AmirGPT/log_124M_10B/log.txt" \
  "$HOST:AmirGPT/log_124M_10B/run_config.json" logs_from_cluster/pretrain/ 2>/dev/null || echo "(pretrain logs not available yet)"
rsync -az "$HOST:AmirGPT/log_sft/metrics.jsonl" "$HOST:AmirGPT/log_sft/sft_config.json" logs_from_cluster/sft/ 2>/dev/null || echo "(sft logs not available yet)"
BASE=$(ssh "$HOST" 'ls -1 ~/AmirGPT/log_124M_10B/model_*.pt 2>/dev/null | sort | tail -1' || true)
if [ -n "$BASE" ]; then echo "base: $BASE"; rsync -az --progress "$HOST:$BASE" checkpoints/amirgpt_base.pt; fi
if ssh "$HOST" 'test -f ~/AmirGPT/log_sft/amirgpt_best.pt'; then
  rsync -az --progress "$HOST:AmirGPT/log_sft/amirgpt_best.pt" checkpoints/amirgpt_chat.pt
fi
ls -la checkpoints logs_from_cluster/*
