#!/bin/bash
# Start AmirGPT locally: builds the frontend if needed, then serves UI + API on http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch numpy tiktoken tqdm requests transformers fastapi "uvicorn[standard]"
fi
if [ ! -f web/frontend/dist/index.html ]; then
  (cd web/frontend && npm install --no-audit --no-fund && npm run build)
fi
echo "AmirGPT -> http://127.0.0.1:${PORT:-8000}"
exec .venv/bin/python -m web.server
