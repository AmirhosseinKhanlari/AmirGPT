"""
DPO (Direct Preference Optimization, Rafailov et al. 2023) preference tuning of the
AmirGPT chat model, on UltraFeedback pairs prepared by sft/prepare_dpo.py.

For a prompt x with a preferred reply y_w ("chosen") and a dispreferred reply y_l
("rejected"), DPO minimises

    L = -log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )

where pi is the model being trained, ref is a FROZEN copy of the initial (SFT) model and
log pi(y|x) is the SUM of the token log-probabilities of the reply only (the prompt tokens
are masked out). beta * (log pi - log ref) is the "implicit reward" of a reply; we log the
mean chosen / rejected rewards, their margin, and the reward accuracy (fraction of pairs with
chosen reward > rejected reward; exactly 0 at step 0, where pi == ref and all rewards are 0).

The training loop mirrors sft/train_sft.py: DDP, fp16 autocast + GradScaler on V100
(--dtype auto), AdamW from GPT.configure_optimizers, length-grouped batches, gradient
clipping, the same checkpoint format (gpt2.model.load_checkpoint and the web server load
amirgpt_dpo_{best,last}.pt unchanged). Besides the DPO metrics it measures what we actually
want to improve and what we must not break:
  * quality eval: replies to QUALITY_PROMPTS (fixed seeds) -> repeated-4-gram rate and length,
    for the initial model (step 0) and the final model;
  * SFT val loss on assistant tokens (sft_data/val_*), at start and end.

Usage:
  python -m sft.train_dpo --init log_sft/amirgpt_best.pt --data dpo_data --sft_data sft_data --out log_dpo
  torchrun --standalone --nproc_per_node=8 -m sft.train_dpo ...   (DDP)
"""

import argparse
import json
import math
import os
import re
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.model import config_to_dict, load_checkpoint
from gpt2.train_gpt2 import get_lr
from sft.chat_format import BANNED_WHILE_ANSWERING, END, VOCAB_SIZE, decode, render_prompt
from sft.train_sft import ConversationData, length_grouped_batches

# Fixed prompts for the generation-based quality eval: everyday questions, explanations,
# lists (where repetition shows up, e.g. "Notre-Dame Cathedral" listed twice for Paris),
# how-to, writing, advice, simple math/reasoning, coding and identity.
QUALITY_PROMPTS = [
    "What are the best things to see in Paris?",
    "Tell me about Paris.",
    "What should I visit in Rome on a three-day trip?",
    "List five famous landmarks in London.",
    "What are some popular tourist attractions in Japan?",
    "Explain how photosynthesis works.",
    "Why is the sky blue?",
    "How does a vaccine train the immune system?",
    "What causes the seasons on Earth?",
    "Explain the difference between weather and climate.",
    "What is machine learning, in simple terms?",
    "How does the internet work?",
    "What is inflation and why does it happen?",
    "Explain what a black hole is to a ten-year-old.",
    "What were the main causes of World War I?",
    "Who was Marie Curie and what is she known for?",
    "Summarize the plot of Romeo and Juliet.",
    "What is the difference between a virus and a bacterium?",
    "Give me five tips for better sleep.",
    "List some healthy breakfast ideas.",
    "What are good ways to save money on groceries?",
    "Give me three tips for staying focused while studying.",
    "What are the benefits of regular exercise?",
    "Suggest some hobbies I could start this winter.",
    "How do I make a simple tomato pasta sauce?",
    "How do I change a flat tire?",
    "How can I improve my public speaking skills?",
    "What should I pack for a week-long camping trip?",
    "How do I start learning to play the guitar?",
    "Write a short poem about the ocean.",
    "Write a short story about a robot who learns to paint.",
    "Write a polite email asking my professor for an extension on an assignment.",
    "Write a product description for a reusable water bottle.",
    "Describe a sunset over the mountains.",
    "Write a haiku about autumn leaves.",
    "I feel stressed about my exams. Any advice?",
    "How do I deal with a coworker who keeps interrupting me?",
    "What should I consider when choosing a laptop?",
    "Should I learn Python or JavaScript first?",
    "What is 15% of 80?",
    "If a train travels 60 miles per hour for 2.5 hours, how far does it go?",
    "What is the difference between a prime number and a composite number?",
    "Write a Python function that checks whether a string is a palindrome.",
    "What does the 'git commit' command do?",
    "Explain what a for loop is in programming.",
    "What is the capital of Australia?",
    "Compare cats and dogs as pets.",
    "What are the pros and cons of electric cars?",
    "Who are you?",
    "What can you help me with?",
]


# ------------------------------------------------------------------ data
class PreferenceData:
    """Pairs from prepare_dpo.py: two aligned ConversationData (chosen / rejected)."""

    def __init__(self, root, split, max_pairs=0, max_len=0):
        """max_len > 0 keeps only pairs whose longer side has <= max_len tokens (low-memory local runs);
        max_pairs > 0 then keeps the first max_pairs of those (the HF split is already shuffled)."""
        self.chosen = ConversationData(root, f"{split}_chosen")
        self.rejected = ConversationData(root, f"{split}_rejected")
        assert len(self.chosen) == len(self.rejected), "chosen/rejected not aligned"
        full = np.maximum(self.chosen.lengths, self.rejected.lengths)
        idx = np.arange(len(full)) if max_len <= 0 else np.flatnonzero(full <= max_len)
        self.idx = idx if max_pairs <= 0 else idx[:max_pairs]  # local pair i -> stored pair self.idx[i]
        self.lengths = full[self.idx]

    def __len__(self):
        return len(self.idx)

    def batch(self, ids):
        """-> (x, y, m) with 2*len(ids) rows: chosen rows first, then rejected rows."""
        ids = np.asarray(ids)
        L = int(self.lengths[ids].max())
        rows = self.idx[ids]
        xc, yc, mc = self.chosen.batch(rows, pad_to=L)
        xr, yr, mr = self.rejected.batch(rows, pad_to=L)
        return torch.cat([xc, xr]), torch.cat([yc, yr]), torch.cat([mc, mr])


# ------------------------------------------------------------------ DPO math
def sequence_logps(logits, y, m):
    """Sum over the masked (reply) positions of log p(y_t | y_<t). logits (B,T,V), y/m (B,T) -> (B,)."""
    per_tok = -F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (per_tok.view(y.shape) * m.to(per_tok.dtype)).sum(dim=1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta):
    """Per-pair DPO loss and implicit rewards (all (B,) tensors)."""
    chosen_reward = beta * (pi_c - ref_c)
    rejected_reward = beta * (pi_r - ref_r)
    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    return loss, chosen_reward.detach(), rejected_reward.detach()


# ------------------------------------------------------------------ quality metrics
def words(text):
    return re.findall(r"\w+", text.lower())


def repeated_ngrams(text, n=4):
    """(number of word n-grams that already occurred earlier in the text, number of n-grams)."""
    w = words(text)
    seen, rep, total = set(), 0, 0
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        total += 1
        if g in seen:
            rep += 1
        seen.add(g)
    return rep, total


def quality_summary(replies):
    """replies: list of {"reply", "n_tokens", "stopped"}. rep4 pools all replies' 4-grams."""
    rep = sum(repeated_ngrams(r["reply"])[0] for r in replies)
    tot = sum(repeated_ngrams(r["reply"])[1] for r in replies)
    per = [a / b for a, b in (repeated_ngrams(r["reply"]) for r in replies) if b > 0]
    return {"rep4": rep / max(tot, 1), "rep4_mean_per_reply": float(np.mean(per)) if per else 0.0,
            "mean_tokens": float(np.mean([r["n_tokens"] for r in replies])),
            "mean_words": float(np.mean([len(words(r["reply"])) for r in replies])),
            "no_end_frac": float(np.mean([not r["stopped"] for r in replies])), "n": len(replies)}


@torch.no_grad()
def generate_replies(model, prompts, device, indices, max_new_tokens=256, temperature=0.7, repetition_penalty=1.0, seed=1337):
    """Sample one reply per prompt (top-k 50, top-p 0.95 as in the web UI; prompt i uses seed + i)."""
    model.eval()
    out = []
    for i in indices:
        toks = render_prompt([{"role": "user", "content": prompts[i]}])
        gen = model.generate(torch.tensor([toks], device=device), max_new_tokens=max_new_tokens, temperature=temperature,
                             top_k=50, top_p=0.95, repetition_penalty=repetition_penalty, stop_tokens=(END,),
                             banned_tokens=BANNED_WHILE_ANSWERING, generator=torch.Generator().manual_seed(seed + i))
        stopped = bool(gen) and gen[-1] == END
        body = gen[:-1] if stopped else gen
        out.append({"i": i, "prompt": prompts[i], "reply": decode(body), "n_tokens": len(body), "stopped": stopped})
    return out


# ------------------------------------------------------------------ args
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="SFT chat checkpoint; also the frozen reference model")
    ap.add_argument("--data", default="dpo_data")
    ap.add_argument("--sft_data", default="sft_data", help="for the SFT val loss ('' to skip)")
    ap.add_argument("--out", default="log_dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro_batch", type=int, default=8, help="PAIRS per GPU per micro step (2x rows)")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_lr", type=float, default=3e-6)
    ap.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_pairs", type=int, default=0, help="test pairs per eval (0 = all)")
    ap.add_argument("--sft_eval_batches", type=int, default=50)
    ap.add_argument("--sft_eval_micro", type=int, default=16, help="50 x 16 per GPU = train_sft's own val set")
    ap.add_argument("--quality_max_tokens", type=int, default=256)
    ap.add_argument("--quality_temperature", type=float, default=0.7)
    ap.add_argument("--quality_repetition_penalty", type=float, default=1.0,
                    help="1.0 = measure the model's own repetition (the web UI defaults to 1.1)")
    ap.add_argument("--quality_prompts", type=int, default=0, help="use only the first N prompts (0 = all)")
    ap.add_argument("--max_pairs", type=int, default=0, help="use only the first N training pairs (0 = all)")
    ap.add_argument("--max_len", type=int, default=0,
                    help="use only pairs with <= N tokens per side, train AND test (0 = all; for low-memory local runs)")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_steps", type=int, default=-1, help="override (for smoke tests)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        cuda = torch.cuda.is_available()  # gloo on CPU only for local tests of the DDP code path
        init_process_group(backend="nccl" if cuda else "gloo")
        rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}" if cuda else "cpu"
        if cuda:
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
    acc_dtype = torch.float32 if device_type == "mps" else torch.float64  # eval sums; MPS has no float64
    torch.set_float32_matmul_precision("high")

    # ---- policy and frozen reference: both start as the SFT checkpoint
    model, init_ckpt = load_checkpoint(args.init)
    ref_model, _ = load_checkpoint(args.init)
    assert model.config.vocab_size >= VOCAB_SIZE, "vocab must include the chat special tokens"
    model.to(device)
    ref_model.to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    raw_model = model
    train_model = (DDP(model, device_ids=[local_rank]) if device_type == "cuda" else DDP(model)) if ddp else model
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.max_lr, device_type, verbose=master)

    train = PreferenceData(args.data, "train", args.max_pairs, args.max_len)
    test = PreferenceData(args.data, "test", max_len=args.max_len)
    pairs_per_step = args.micro_batch * args.grad_accum * world
    steps_per_epoch = len(train) // pairs_per_step
    max_steps = args.max_steps if args.max_steps > 0 else max(1, int(args.epochs * steps_per_epoch))
    warmup_steps = int(round(args.warmup_ratio * max_steps))
    min_lr = args.max_lr * args.min_lr_ratio
    log(f"DPO: {len(train):,} train / {len(test):,} test pairs | {pairs_per_step} pairs/step | {steps_per_epoch} steps/epoch | "
        f"{max_steps} steps | warmup {warmup_steps} | beta {args.beta} | lr {args.max_lr:g} ({args.schedule}) | dtype {dtype}")

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    if master:
        open(metrics_path, "w").close()
        json.dump(vars(args), open(os.path.join(args.out, "dpo_config.json"), "w"), indent=2)

    def metric(**kw):
        if master:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(kw) + "\n")

    def all_sum(t):
        if ddp:
            dist.all_reduce(t)
        return t

    def free_cache():
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            torch.mps.empty_cache()

    def pair_logps(net, x, y, m):
        """-> (logp_chosen, logp_rejected), each (B,), from one forward over 2B rows."""
        with autocast():
            logits, _ = net(x)
        lp = sequence_logps(logits, y, m)
        B = x.size(0) // 2
        return lp[:B], lp[B:]

    @torch.no_grad()
    def evaluate_prefs():
        raw_model.eval()
        free_cache()
        ids = np.arange(len(test))
        if args.eval_pairs > 0:
            ids = ids[:args.eval_pairs]
        ids = ids[rank::world]
        s = torch.zeros(7, dtype=acc_dtype, device=device)  # loss, acc, margin, chosen_r, rejected_r, pi_c, n
        for i in range(0, len(ids), args.micro_batch):
            x, y, m = (t.to(device) for t in test.batch(ids[i:i + args.micro_batch]))
            ref_c, ref_r = pair_logps(ref_model, x, y, m)
            pi_c, pi_r = pair_logps(raw_model, x, y, m)
            loss, cr, rr = dpo_loss(pi_c, pi_r, ref_c, ref_r, args.beta)
            s += torch.stack([loss.sum(), (cr > rr).float().sum(), (cr - rr).sum(), cr.sum(), rr.sum(), pi_c.sum(),
                              torch.tensor(float(len(loss)), device=device)]).to(acc_dtype)
        s = all_sum(s)
        n = s[6].item()
        return {"value": s[0].item() / n, "reward_acc": s[1].item() / n, "reward_margin": s[2].item() / n,
                "chosen_reward": s[3].item() / n, "rejected_reward": s[4].item() / n, "chosen_logp": s[5].item() / n,
                "pairs": int(n)}

    sft_val = ConversationData(args.sft_data, "val") if args.sft_data else None

    @torch.no_grad()
    def evaluate_sft():
        """Assistant-token loss on SFT val; same rows and normalisation as train_sft.evaluate."""
        raw_model.eval()
        free_cache()
        ids = np.arange(len(sft_val))[rank::world][: args.sft_eval_batches * args.sft_eval_micro]
        s = torch.zeros(2, dtype=acc_dtype, device=device)
        for i in range(0, len(ids), args.sft_eval_micro):
            x, y, m = (t.to(device) for t in sft_val.batch(ids[i:i + args.sft_eval_micro]))
            with autocast():
                logits, _ = raw_model(x)
            per_tok = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none")
            s += torch.stack([(per_tok * m.view(-1)).sum(), m.sum().float()]).to(acc_dtype)
        s = all_sum(s)
        return s[0].item() / s[1].item()

    prompts = QUALITY_PROMPTS[:args.quality_prompts] if args.quality_prompts > 0 else QUALITY_PROMPTS

    @torch.no_grad()
    def quality_eval(tag):
        """Each rank generates its share of the prompts (fp32, fixed per-prompt seeds); results gathered."""
        mine = generate_replies(raw_model, prompts, device, range(rank, len(prompts), world), args.quality_max_tokens,
                                args.quality_temperature, args.quality_repetition_penalty, args.seed)
        if ddp:
            gathered = [None] * world
            dist.all_gather_object(gathered, mine)
            mine = [r for g in gathered for r in g]
        replies = sorted(mine, key=lambda r: r["i"])
        q = quality_summary(replies)
        if master:
            with open(os.path.join(args.out, f"quality_{tag}.json"), "w") as f:
                json.dump({"summary": q, "replies": replies}, f, indent=2)
        return q

    def save(step, name, test_metrics):
        torch.save({"model": raw_model.state_dict(), "config": config_to_dict(raw_model.config), "step": step,
                    "val_loss": test_metrics["value"], "args": vars(args), "chat_format": "amirgpt-v1",
                    "dpo": {"init": args.init, "init_step": init_ckpt.get("step"), "test": test_metrics}},
                   os.path.join(args.out, name))

    summary = {"init": args.init, "max_lr": args.max_lr, "beta": args.beta, "steps": max_steps,
               "train_pairs": len(train), "test_pairs": len(test)}
    t_start = time.time()
    # ---- step-0 measurements of the initial (SFT) model
    q0 = quality_eval("step0")
    log(f"quality (init): {q0}")
    metric(step=0, kind="quality", **q0)
    summary["quality_start"] = q0
    if sft_val is not None:
        sv0 = evaluate_sft()
        log(f"SFT val loss (init): {sv0:.4f}")
        metric(step=0, kind="sft_val", value=sv0)
        summary["sft_val_start"] = sv0

    epoch, batches, bpos = 0, length_grouped_batches(train.lengths, args.micro_batch * world, args.seed, 0), 0
    best_acc, best_step, ev = -1.0, None, None
    for step in range(max_steps):
        last = step == max_steps - 1
        if step % args.eval_interval == 0:
            ev = evaluate_prefs()
            log(f"step {step} | test {ev}")
            metric(step=step, kind="val", **ev)
            if step > 0 and ev["reward_acc"] > best_acc:
                best_acc, best_step = ev["reward_acc"], step
                if master:
                    save(step, "amirgpt_dpo_best.pt", ev)
        train_model.train()
        t0 = time.time()
        micro = []
        for _ in range(args.grad_accum):
            if bpos >= len(batches):
                epoch += 1
                batches, bpos = length_grouped_batches(train.lengths, args.micro_batch * world, args.seed, epoch), 0
            micro.append(train.batch(batches[bpos][rank::world]))  # this rank's share of the global batch
            bpos += 1
        optimizer.zero_grad(set_to_none=True)
        acc = torch.zeros(6, device=device)  # loss, acc, margin, chosen_r, rejected_r, pi_c (means over pairs)
        for i, (x, y, m) in enumerate(micro):
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.no_grad():  # reference first, so its activations are freed before the policy's
                ref_c, ref_r = pair_logps(ref_model, x, y, m)
            sync = train_model.no_sync() if (ddp and i < len(micro) - 1) else nullcontext()
            with sync:
                pi_c, pi_r = pair_logps(train_model, x, y, m)
                loss, cr, rr = dpo_loss(pi_c, pi_r, ref_c, ref_r, args.beta)
                # every rank has the same number of pairs and DDP averages over ranks -> global mean
                scaler.scale(loss.mean() / len(micro)).backward()
            acc += torch.stack([loss.detach().mean(), (cr > rr).float().mean(), (cr - rr).mean(), cr.mean(), rr.mean(),
                                pi_c.detach().mean()]) / len(micro)
        acc = all_sum(acc) / world
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        if args.schedule == "cosine":
            lr = get_lr(step, args.max_lr, min_lr, max(warmup_steps, 1), max(max_steps - 1, warmup_steps + 1))
        else:
            lr = args.max_lr * min(1.0, (step + 1) / max(warmup_steps, 1))
        for g in optimizer.param_groups:
            g["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.empty_cache()  # variable sequence lengths otherwise grow MPS's cache until the Mac swaps
        dt = time.time() - t0
        a = acc.tolist()
        if not all(math.isfinite(v) for v in a):
            log(f"warning: non-finite train metrics at step {step}: {a}")
        if master and (step % 10 == 0 or last):
            eta = (max_steps - step - 1) * (time.time() - t_start) / (step + 1) / 3600
            print(f"step {step:5d}/{max_steps} | loss {a[0]:.4f} | acc {a[1]:.3f} | margin {a[2]:.4f} | "
                  f"r_c {a[3]:.4f} r_r {a[4]:.4f} | lr {lr:.2e} | norm {norm:.3f} | dt {dt * 1000:.0f}ms | eta {eta:.2f}h",
                  flush=True)
            metric(step=step, kind="train", value=a[0], reward_acc=a[1], reward_margin=a[2], chosen_reward=a[3],
                   rejected_reward=a[4], chosen_logp=a[5], lr=lr, norm=float(norm))

    # ---- final measurements (after the last optimizer step)
    ev = evaluate_prefs()
    log(f"step {max_steps} | test {ev}")
    metric(step=max_steps, kind="val", **ev)
    if ev["reward_acc"] > best_acc:
        best_acc, best_step = ev["reward_acc"], max_steps
        if master:
            save(max_steps, "amirgpt_dpo_best.pt", ev)
    if master:
        save(max_steps, "amirgpt_dpo_last.pt", ev)
    q1 = quality_eval("final")
    log(f"quality (final): {q1}")
    metric(step=max_steps, kind="quality", **q1)
    summary.update(quality_end=q1, test_final=ev, best_reward_acc=best_acc, best_step=best_step)
    if sft_val is not None:
        sv1 = evaluate_sft()
        log(f"SFT val loss (final): {sv1:.4f}")
        metric(step=max_steps, kind="sft_val", value=sv1)
        summary["sft_val_end"] = sv1
    summary["minutes"] = (time.time() - t_start) / 60
    if master:
        json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
        log(f"done in {summary['minutes']:.1f} min. best test reward acc {best_acc:.4f} at step {best_step} -> "
            f"{os.path.join(args.out, 'amirgpt_dpo_best.pt')}")
    if ddp:
        destroy_process_group()


def print_summary(run_dirs):
    """Table of summary.json files (used at the end of cluster/dpo.sbatch)."""
    head = f"{'run':<28}{'lr':>8}{'acc_end':>9}{'acc_best':>9}{'margin':>8}{'sft_val':>17}{'rep4':>15}{'tokens':>13}{'no_end':>13}{'min':>6}"
    print(head)
    print("-" * len(head))
    for d in run_dirs:
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            print(f"{d:<28}  (no summary.json: run failed or unfinished)")
            continue
        s = json.load(open(p))
        q0, q1, t = s["quality_start"], s["quality_end"], s["test_final"]
        sv = f"{s.get('sft_val_start', float('nan')):.3f}->{s.get('sft_val_end', float('nan')):.3f}"
        print(f"{os.path.basename(d.rstrip('/')):<28}{s['max_lr']:>8.0e}{t['reward_acc']:>9.3f}{s['best_reward_acc']:>9.3f}"
              f"{t['reward_margin']:>8.3f}{sv:>17}{q0['rep4']:>7.3f}->{q1['rep4']:.3f}"
              f"{q0['mean_tokens']:>6.0f}->{q1['mean_tokens']:<4.0f}{q0['no_end_frac']:>6.2f}->{q1['no_end_frac']:.2f}"
              f"{s['minutes']:>6.1f}")


if __name__ == "__main__":
    main()
