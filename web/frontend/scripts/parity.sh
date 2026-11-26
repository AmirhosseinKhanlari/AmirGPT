#!/usr/bin/env bash
# Engine parity test without a browser:
#   1. parity_reference.py runs the real server code (CPU) + tiktoken and dumps reference.json
#   2. esbuild bundles engine-parity.ts (which imports the real src/webgpu modules)
#   3. Node runs it with onnxruntime-web's wasm backend on checkpoints/onnx/*
# Usage (from web/frontend):  bash scripts/parity.sh [fp32] [fp16]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(cd ../.. && pwd)"
OUT="node_modules/.cache/amirgpt"
mkdir -p "$OUT"
AMIRGPT_DEVICE=cpu "$ROOT/.venv/bin/python" scripts/parity_reference.py "$OUT/reference.json"
node_modules/.bin/esbuild scripts/engine-parity.ts --bundle --platform=node --format=esm \
  --external:onnxruntime-web --log-level=warning --outfile="$OUT/engine-parity.mjs"
node "$OUT/engine-parity.mjs" "$ROOT" "$OUT/reference.json" "$@"
