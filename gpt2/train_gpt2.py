"""
Pretrain GPT-2 (124M) from scratch -- Karpathy's "Let's reproduce GPT-2 (124M)".

Every default below is the value used in the video / build-nanogpt train_gpt2.py:
  total batch 524,288 tokens (2**19, ~0.5M, GPT-3 paper)             [2:35:28], [2:37:19]
  micro batch B=64, sequence length T=1024, grad accumulation         [2:36:41]
  max_lr 6e-4, min_lr = 0.1 * max_lr, cosine decay                    [2:21:12], [2:23:42]
  warmup 715 steps (= 375M tokens / 524,288)                          [3:19:57]
  max_steps 19,073 (~1 epoch of 10B tokens)
  AdamW betas (0.9, 0.95), eps 1e-8, weight decay 0.1, fused         [2:28:53], [2:31:43]
  gradient-norm clipping at 1.0                                      [2:17:22]
  vocab padded to 50,304 ("nice number")                             [2:07:26]
  TF32 matmuls, bf16 autocast                                         [1:25:27], [1:43:10]
  val loss every 250 steps (20 batches), HellaSwag every 250 steps,
  samples from "Hello, I'm a language model," every 250 steps         [3:39:40]
  DDP via torchrun                                                    [2:47:18], [2:54:57]

Deviations, all opt-in or forced by hardware (each is a CLI flag):
  --dtype fp16      V100 GPUs have no bfloat16, so fp16 autocast + GradScaler is used
                    there ("auto" picks bf16 on Ampere/Hopper, fp16 on Volta/Turing).
  --micro_batch     B=64 does not fit a 32GB V100; e.g. B=16 with 4x more gradient
                    accumulation gives the SAME 524,288-token batch and the same math.
  --shuffle         document/shard shuffling Karpathy asks for at [3:50:05] (default off).
  --compile         torch.compile for training. The video's final code turns compile
                    off because it broke HellaSwag/generation; here eval & sampling run
                    on the un-compiled module (same weights), so both can be on.
  --resume          checkpoints include optimizer/scaler/dataloader/RNG state so a run
                    can continue exactly (the video's checkpoints are model-only).
  MPS (Apple)       autocast uses device_type "mps" (README errata: the original mapped
                    it to "cpu").

Launch:
  python -m gpt2.train_gpt2                                   # single GPU / MPS / CPU
  torchrun --standalone --nproc_per_node=8 -m gpt2.train_gpt2 # 8 GPUs (DDP)
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.data import ShardedDataLoader
from gpt2.hellaswag import get_most_likely_row, iterate_examples, render_example
from gpt2.model import GPT, GPTConfig, config_to_dict


def get_lr(it, max_lr=6e-4, min_lr=6e-5, warmup_steps=715, max_steps=19073):
    """Linear warmup then cosine decay to min_lr (video [2:21:12])."""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / io
    ap.add_argument("--data_root", default="edu_fineweb10B")
    ap.add_argument("--log_dir", default="log")
    # batch
    ap.add_argument("--total_batch_size", type=int, default=524288)
    ap.add_argument("--micro_batch", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=1024)
    # optimisation
    ap.add_argument("--max_lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=715)
    ap.add_argument("--max_steps", type=int, default=19073)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # model
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--vocab_size", type=int, default=50304)
    ap.add_argument("--init_from", default="scratch", help="scratch | gpt2 (OpenAI weights, for experiments)")
    # eval / logging
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--val_steps", type=int, default=20)
    ap.add_argument("--eval_micro_batch", type=int, default=8, help="sequences per eval forward (memory only; same result)")
    ap.add_argument("--hellaswag", type=int, default=1)
    ap.add_argument("--hellaswag_limit", type=int, default=-1, help="-1 = all 10,042 val examples")
    ap.add_argument("--sample_interval", type=int, default=250)
    ap.add_argument("--ckpt_interval", type=int, default=5000)
    ap.add_argument("--latest_interval", type=int, default=500, help="steps between resumable 'latest.pt' saves")
    # systems
    ap.add_argument("--dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--resume", type=int, default=0, help="resume from <log_dir>/latest.pt if it exists")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)

    # ------------------------------------------------------------------ DDP setup
    # torchrun sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        master_process = True
        device = args.device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")
    log = print if master_process else (lambda *a, **k: None)
    log(f"using device: {device} (world size {ddp_world_size})")

    # mixed precision: bf16 on Ampere+/MPS, fp16 (+GradScaler) on Volta/Turing
    dtype = args.dtype
    if dtype == "auto":
        if device_type == "cuda":
            dtype = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
        elif device_type == "mps":
            dtype = "bf16"
        else:
            dtype = "fp32"
    ptdtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    autocast = (lambda: torch.autocast(device_type=device_type, dtype=ptdtype)) if dtype != "fp32" else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "fp16" and device_type == "cuda"))
    log(f"mixed precision: {dtype}{' + GradScaler' if scaler.is_enabled() else ''}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    enc = tiktoken.get_encoding("gpt2")

    total_batch_size, B, T = args.total_batch_size, args.micro_batch, args.seq_len
    assert total_batch_size % (B * T * ddp_world_size) == 0, \
        "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    log(f"total desired batch size: {total_batch_size}")
    log(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = ShardedDataLoader(B, T, ddp_rank, ddp_world_size, "train", args.data_root,
                                     shuffle=bool(args.shuffle), seed=args.seed, verbose=master_process)
    val_loader = ShardedDataLoader(B, T, ddp_rank, ddp_world_size, "val", args.data_root, verbose=master_process)

    torch.set_float32_matmul_precision("high")  # TF32

    # ------------------------------------------------------------------ model
    if args.init_from == "scratch":
        model = GPT(GPTConfig(block_size=T if T > 1024 else 1024, vocab_size=args.vocab_size,
                              n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd))
    else:
        model = GPT.from_pretrained(args.init_from, vocab_size=args.vocab_size)
    model.to(device)
    raw_model = model  # un-compiled, un-wrapped module: used for eval, sampling, checkpoints
    log(f"model parameters: {raw_model.num_params():,}")
    train_model = torch.compile(model) if args.compile else model
    if ddp:
        train_model = DDP(train_model, device_ids=[ddp_local_rank])

    min_lr = args.max_lr * args.min_lr_ratio
    lr_at = lambda it: get_lr(it, args.max_lr, min_lr, args.warmup_steps, args.max_steps)
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.max_lr,
                                               device_type=device_type, verbose=master_process)

    # ------------------------------------------------------------------ logging / resume
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "log.txt")
    metrics_file = os.path.join(args.log_dir, "metrics.jsonl")
    latest_path = os.path.join(args.log_dir, "latest.pt")
    start_step = 0
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        train_loader.load_state_dict(ckpt["train_loader"])
        torch.set_rng_state(ckpt["rng_cpu"])
        if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ckpt["rng_cuda"][min(ddp_rank, len(ckpt["rng_cuda"]) - 1)].cpu()
                                     if isinstance(ckpt["rng_cuda"], list) else ckpt["rng_cuda"].cpu())
        start_step = ckpt["step"] + 1
        log(f"resumed from {latest_path} at step {start_step}")
    elif master_process:
        open(log_file, "w").close()  # open for writing to clear the file
        open(metrics_file, "w").close()
        with open(os.path.join(args.log_dir, "run_config.json"), "w") as f:
            json.dump({**vars(args), "resolved_dtype": dtype, "world_size": ddp_world_size,
                       "grad_accum_steps": grad_accum_steps}, f, indent=2)

    def write_metric(**kw):
        if master_process:
            with open(metrics_file, "a") as f:
                f.write(json.dumps(kw) + "\n")

    def save_checkpoint(step, val_loss, path, full):
        ckpt = {"model": raw_model.state_dict(), "config": config_to_dict(raw_model.config),
                "step": step, "val_loss": val_loss, "args": vars(args)}
        if full:
            ckpt.update({"optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                         "train_loader": train_loader.state_dict(),
                         "rng_cpu": torch.get_rng_state(),
                         "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None})
        atomic_save(ckpt, path)

    t_start = time.time()
    val_loss_value = None
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # ---- once in a while evaluate our validation loss
        if step % args.eval_interval == 0 or last_step:
            raw_model.eval()
            val_loader.reset()
            if device_type == "cuda":
                torch.cuda.empty_cache()  # give back the compiled model's cached blocks before eager eval
            with torch.no_grad():
                val_loss_accum = torch.zeros((), device=device)
                for _ in range(args.val_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    # Evaluate in equal-size chunks: the eager model materialises full fp32 logits
                    # (B x T x 50304), which on a 32GB V100 does not fit next to the compiled model's
                    # cache for B=32. Equal chunks -> the mean of chunk means is exactly the batch mean.
                    chunks = max(1, B // args.eval_micro_batch) if B % args.eval_micro_batch == 0 else 1
                    for xc, yc in zip(x.chunk(chunks), y.chunk(chunks)):
                        with autocast():
                            _, loss = raw_model(xc, yc)
                        val_loss_accum += loss.detach().float() / (args.val_steps * chunks)
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            val_loss_value = val_loss_accum.item()
            log(f"validation loss: {val_loss_value:.4f}")
            if master_process:
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_value:.4f}\n")
                write_metric(step=step, kind="val", value=val_loss_value)
                if step > 0 and (step % args.ckpt_interval == 0 or last_step):
                    save_checkpoint(step, val_loss_value, os.path.join(args.log_dir, f"model_{step:05d}.pt"), full=False)

        # ---- once in a while evaluate hellaswag
        if args.hellaswag and (step % args.eval_interval == 0 or last_step):
            raw_model.eval()
            if device_type == "cuda":
                torch.cuda.empty_cache()
            num_correct_norm, num_total = 0, 0
            for i, example in enumerate(iterate_examples("val")):
                if 0 <= args.hellaswag_limit <= i:
                    break
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens, mask = tokens.to(device), mask.to(device)
                with torch.no_grad():
                    with autocast():
                        logits, _ = raw_model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                t = torch.tensor([num_total, num_correct_norm], dtype=torch.long, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                num_total, num_correct_norm = t.tolist()
            acc_norm = num_correct_norm / max(num_total, 1)
            log(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            if master_process:
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")
                write_metric(step=step, kind="hella", value=acc_norm)

        # ---- once in a while generate from the model (except step 0, which is noise)
        if master_process and ((step > 0 and step % args.sample_interval == 0) or last_step):
            raw_model.eval()
            prompt = torch.tensor([enc.encode("Hello, I'm a language model,")], dtype=torch.long, device=device)
            sample_rng = torch.Generator().manual_seed(42 + ddp_rank)
            samples = []
            for i in range(4):
                out = raw_model.generate(prompt, max_new_tokens=32 - prompt.size(1), temperature=1.0,
                                         top_k=50, generator=sample_rng, banned_tokens=range(50257, args.vocab_size))
                text = enc.decode(prompt[0].tolist() + [t for t in out if t < 50257])
                samples.append(text)
                print(f"rank {ddp_rank} sample {i}: {text}")
            write_metric(step=step, kind="samples", value=samples)

        # ---- one step of the optimization
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # only all-reduce gradients on the last micro step
            sync_ctx = train_model.no_sync() if (ddp and micro_step < grad_accum_steps - 1) else nullcontext()
            with sync_ctx:
                with autocast():
                    _, loss = train_model(x, y)
                # scale the loss: gradients add up over micro steps, we want the MEAN [2:41:28]
                loss = loss / grad_accum_steps
                loss_accum += loss.detach().float()
                scaler.scale(loss).backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        scaler.unscale_(optimizer)  # no-op when the scaler is disabled
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        lr = lr_at(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        tokens_per_sec = B * T * grad_accum_steps * ddp_world_size / dt
        if master_process:
            elapsed = time.time() - t_start
            done = step - start_step + 1
            eta_h = (args.max_steps - step - 1) * elapsed / done / 3600
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | "
                  f"dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | eta {eta_h:.2f}h", flush=True)
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
            write_metric(step=step, kind="train", value=loss_accum.item(), lr=lr, norm=float(norm),
                         dt=dt, tok_per_sec=tokens_per_sec)
            if (step > 0 and step % args.latest_interval == 0) or last_step:
                save_checkpoint(step, val_loss_value, latest_path, full=True)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
