"""
Validation loss on the first N tokens of the FineWeb-Edu val shard (edufineweb_val_000000.npy).

The training script's validation (video [3:23:58]) reads val_steps x B x T tokens per
rank, strided over ranks; with 8 GPUs, B=64, T=1024, 20 steps that is exactly the first
10,485,760 tokens of the val shard. This script evaluates the same tokens on one device,
so we can score any model on the identical set:

  python -m gpt2.eval_val --model gpt2                    # OpenAI GPT-2 124M (Karpathy's
                                                          #   play.ipynb baseline: 3.2924)
  python -m gpt2.eval_val --checkpoint log/model_19072.pt # one of ours
"""

import argparse
import os

import numpy as np
import torch

from gpt2.model import GPT, load_checkpoint


@torch.no_grad()
def val_loss(model, tokens, B=16, T=1024, device="cuda", dtype=torch.float32):
    model.eval()
    n_batches = (len(tokens) - 1) // (B * T)
    total = 0.0
    for b in range(n_batches):
        buf = torch.from_numpy(tokens[b * B * T: (b + 1) * B * T + 1].astype(np.int64)).to(device)
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
            _, loss = model(x, y)
        total += loss.float().item()
    return total / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="gpt2 | gpt2-medium ... (OpenAI weights)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data_root", default="edu_fineweb10B")
    ap.add_argument("--tokens", type=int, default=10_485_760)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    model = load_checkpoint(args.checkpoint)[0] if args.checkpoint else GPT.from_pretrained(args.model or "gpt2")
    model.to(args.device)
    toks = np.load(os.path.join(args.data_root, "edufineweb_val_000000.npy"))[: args.tokens + 1]
    loss = val_loss(model, toks, device=args.device)
    print(f"val loss over first {args.tokens:,} val tokens: {loss:.4f}")


if __name__ == "__main__":
    main()
