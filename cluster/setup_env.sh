#!/bin/bash
# One-time environment setup on the Stony Brook AI cluster (run on submit.ai.stonybrook.edu).
# Creates ~/AmirGPT/.venv with PyTorch built for CUDA 12.4 (driver 550 on tesla/quadro/h100 nodes).
# torch 2.6 cu124 wheels include sm_70 kernels (V100) as well as sm_80/sm_90 (A100/H100).
set -euo pipefail
cd "$HOME/AmirGPT"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python numpy tiktoken tqdm requests datasets transformers pytest
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_arch_list() if torch.cuda.is_available() else '(no gpu on this node)')"
