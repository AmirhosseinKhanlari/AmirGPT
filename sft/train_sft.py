"""
Supervised fine-tuning (SFT) of our pretrained GPT-2 into the AmirGPT chat model.

Karpathy, video [3:55:46]: "we swap out the data set and continue training". So this
script reuses the pretraining pieces unchanged -- the same GPT class, AdamW setup
(configure_optimizers), gradient clipping, warmup+cosine schedule (get_lr), mixed
precision and DDP -- and changes only:
  * the data: user/assistant conversations from sft/prepare_sft.py (one conversation
    per row, padded to the longest row in the batch; rows are grouped by length so
    little compute is wasted on padding);
  * the loss: cross-entropy is computed ONLY on assistant tokens (loss_mask) -- we
    "fill in the user tokens and sample the assistant tokens", so the model is never
    trained to imitate the user. The loss is normalised by the number of assistant
    tokens in the whole (gradient-accumulated, all-GPU) batch, so every assistant
    token has equal weight no matter how batches are split.
  * smaller learning rate (fine-tuning, not training from scratch).

Usage:
  python -m sft.train_sft --init log/model_19072.pt --data sft_data --out log_sft
  torchrun --standalone --nproc_per_node=8 -m sft.train_sft --init ... (DDP)
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.model import GPT, config_to_dict, load_checkpoint
from gpt2.train_gpt2 import get_lr
from sft.chat_format import ASSISTANT, BANNED_WHILE_ANSWERING, END, VOCAB_SIZE, decode, render_prompt


class ConversationData:
    """Conversations stored as concatenated tokens/mask + offsets (see prepare_sft.py)."""

    def __init__(self, root, split):
        self.tokens = np.load(os.path.join(root, f"{split}_tokens.npy"), mmap_mode="r")
        self.mask = np.load(os.path.join(root, f"{split}_mask.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(root, f"{split}_offsets.npy"))
        self.lengths = np.diff(self.offsets)

    def __len__(self):
        return len(self.lengths)

    def batch(self, ids, pad_to=None):
        """Rows -> (x, y, m): x inputs, y next-token targets, m loss mask aligned with y."""
        L = int(max(self.lengths[i] for i in ids)) if pad_to is None else pad_to
        x = np.zeros((len(ids), L - 1), dtype=np.int64)
        y = np.zeros((len(ids), L - 1), dtype=np.int64)
        m = np.zeros((len(ids), L - 1), dtype=np.int64)
        for r, i in enumerate(ids):
            a, b = self.offsets[i], self.offsets[i + 1]
            t = np.asarray(self.tokens[a:b], dtype=np.int64)
            k = np.asarray(self.mask[a:b], dtype=np.int64)
            n = len(t) - 1
            x[r, :n], y[r, :n], m[r, :n] = t[:-1], t[1:], k[1:]  # shift: mask labels the target token
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def length_grouped_batches(lengths, batch_size, seed, epoch, megabatch=64):
    """Shuffle, then sort by length inside 'megabatches' of megabatch*batch_size rows, then
    shuffle the batch order: random but with little padding. Same result on every rank."""
    rng = np.random.default_rng([seed, epoch])
    perm = rng.permutation(len(lengths))
    batches = []
    mb = batch_size * megabatch
    for s in range(0, len(perm), mb):
        chunk = perm[s:s + mb]
        chunk = chunk[np.argsort(-lengths[chunk], kind="stable")]
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size) if len(chunk[i:i + batch_size]) == batch_size]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="pretrained checkpoint (ours), or 'gpt2' for OpenAI weights")
    ap.add_argument("--data", default="sft_data")
    ap.add_argument("--out", default="log_sft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--micro_batch", type=int, default=16, help="conversations per GPU per micro step")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_lr", type=float, default=1e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_steps", type=int, default=-1, help="override (for smoke tests)")
    return ap.parse_args(argv)


SAMPLE_PROMPTS = ["Who are you?", "What is the capital of France?", "Give me three tips for staying focused while studying.",
                  "Write a short poem about the ocean."]


def main(argv=None):
    args = parse_args(argv)
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world = 0, 0, 1
        device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    master = rank == 0
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")
    log = print if master else (lambda *a, **k: None)

    dtype = args.dtype
    if dtype == "auto":
        dtype = ("bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16") if device_type == "cuda" else \
                ("bf16" if device_type == "mps" else "fp32")
    ptdtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    autocast = (lambda: torch.autocast(device_type=device_type, dtype=ptdtype)) if dtype != "fp32" else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "fp16" and device_type == "cuda"))
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    # ---- model: our pretrained checkpoint (vocab 50304 already includes the spare ids)
    if args.init == "gpt2":
        model = GPT.from_pretrained("gpt2", vocab_size=VOCAB_SIZE)
    else:
        model, _ = load_checkpoint(args.init)
    assert model.config.vocab_size >= VOCAB_SIZE, "vocab must include the chat special tokens"
    model.to(device)
    raw_model = model
    train_model = DDP(model, device_ids=[local_rank]) if ddp else model
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.max_lr, device_type, verbose=master)

    train = ConversationData(args.data, "train")
    val = ConversationData(args.data, "val")
    rows_per_step = args.micro_batch * args.grad_accum * world
    steps_per_epoch = len(train) // rows_per_step
    max_steps = args.max_steps if args.max_steps > 0 else int(args.epochs * steps_per_epoch)
    log(f"SFT: {len(train):,} train / {len(val):,} val conversations | {rows_per_step} conversations/step | "
        f"{steps_per_epoch} steps/epoch | {max_steps} steps | dtype {dtype}")
    min_lr = args.max_lr * args.min_lr_ratio

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    if master:
        open(metrics_path, "w").close()
        json.dump(vars(args), open(os.path.join(args.out, "sft_config.json"), "w"), indent=2)

    def metric(**kw):
        if master:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(kw) + "\n")

    @torch.no_grad()
    def evaluate():
        raw_model.eval()
        if device_type == "cuda":
            torch.cuda.empty_cache()
        # fixed, deterministic val batches (each rank takes a disjoint slice)
        ids = np.arange(len(val))[rank::world][: args.eval_batches * args.micro_batch]
        loss_sum = torch.zeros((), device=device)
        tok_sum = torch.zeros((), device=device)
        for s in range(0, len(ids), args.micro_batch):
            x, y, m = (t.to(device) for t in val.batch(ids[s:s + args.micro_batch]))
            with autocast():
                logits, _ = raw_model(x)
            per_tok = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none")
            loss_sum += (per_tok * m.view(-1)).sum()
            tok_sum += m.sum()
        if ddp:
            dist.all_reduce(loss_sum)
            dist.all_reduce(tok_sum)
        return (loss_sum / tok_sum).item()

    def save(step, val_loss, name):
        torch.save({"model": raw_model.state_dict(), "config": config_to_dict(raw_model.config), "step": step,
                    "val_loss": val_loss, "args": vars(args), "chat_format": "amirgpt-v1"},
                   os.path.join(args.out, name))

    @torch.no_grad()
    def show_samples(step):
        raw_model.eval()
        outs = []
        for p in SAMPLE_PROMPTS:
            toks = render_prompt([{"role": "user", "content": p}])
            gen = raw_model.generate(torch.tensor([toks], device=device), max_new_tokens=80, temperature=0.7,
                                     top_k=50, top_p=0.95, stop_tokens=(END,), banned_tokens=BANNED_WHILE_ANSWERING,
                                     generator=torch.Generator().manual_seed(step))
            reply = decode([t for t in gen if t != END])
            outs.append({"prompt": p, "reply": reply})
            print(f"  [user] {p}\n  [AmirGPT] {reply}")
        metric(step=step, kind="samples", value=outs)

    epoch, batches, bpos = 0, length_grouped_batches(train.lengths, args.micro_batch * world, args.seed, 0), 0
    best_val = float("inf")
    t_start = time.time()
    for step in range(max_steps):
        last = step == max_steps - 1
        if step % args.eval_interval == 0 or last:
            vl = evaluate()
            log(f"step {step} | val loss (assistant tokens) {vl:.4f}")
            metric(step=step, kind="val", value=vl)
            if master and step > 0:
                show_samples(step)
                if vl < best_val:
                    best_val = vl
                    save(step, vl, "amirgpt_best.pt")
        train_model.train()
        t0 = time.time()
        # gather this step's micro batches first, so the loss can be normalised by the
        # total number of assistant tokens in the whole step (across all GPUs)
        micro = []
        for _ in range(args.grad_accum):
            if bpos >= len(batches):
                epoch += 1
                batches, bpos = length_grouped_batches(train.lengths, args.micro_batch * world, args.seed, epoch), 0
            global_rows = batches[bpos]
            bpos += 1
            micro.append(train.batch(global_rows[rank::world]))  # this rank's share of the global batch
        n_tok = torch.tensor(float(sum(int(m.sum()) for _, _, m in micro)), device=device)
        if ddp:
            dist.all_reduce(n_tok)
        n_tok = n_tok.clamp(min=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss_report = torch.zeros((), device=device)
        for i, (x, y, m) in enumerate(micro):
            x, y, m = x.to(device), y.to(device), m.to(device)
            sync = train_model.no_sync() if (ddp and i < len(micro) - 1) else nullcontext()
            with sync:
                with autocast():
                    logits, _ = train_model(x)
                per_tok = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none")
                # DDP averages gradients over ranks, so multiply by world to get sum / global_token_count
                loss = (per_tok * m.view(-1)).sum() * world / n_tok
                loss_report += loss.detach() / world
                scaler.scale(loss).backward()
        if ddp:
            dist.all_reduce(loss_report)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        lr = get_lr(step, args.max_lr, min_lr, args.warmup_steps, max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if master and (step % 10 == 0 or last):
            eta = (max_steps - step - 1) * (time.time() - t_start) / (step + 1) / 3600
            print(f"step {step:5d}/{max_steps} | loss {loss_report.item():.4f} | lr {lr:.2e} | norm {norm:.3f} | "
                  f"dt {dt * 1000:.0f}ms | epoch {epoch} | eta {eta:.2f}h", flush=True)
            metric(step=step, kind="train", value=loss_report.item(), lr=lr, norm=float(norm))
    if master:
        save(max_steps - 1, best_val, "amirgpt_last.pt")
        log(f"done. best val loss {best_val:.4f} -> {os.path.join(args.out, 'amirgpt_best.pt')}")
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
