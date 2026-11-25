"""
Prepare the conversational SFT dataset for AmirGPT.

Dataset: HuggingFaceTB/smol-smoltalk (Apache-2.0; 460,341 train / 24,229 test
conversations). Its card describes it as a subset of SmolTalk "adapted for smol
models with less than 1B parameters", used to build SmolLM2-135M-Instruct -- a
model of almost the same size as our GPT-2 124M.

Steps:
  1. download the parquet files over HTTPS (sizes verified against the HF API);
  2. encode every conversation with sft/chat_format.py (tokens + assistant-only loss mask);
  3. drop conversations with unknown roles or no assistant turn; truncate to 1024 tokens
     (keeping it only if some assistant tokens survive);
  4. add a small set of identity conversations so the model answers "who are you?"
     truthfully as AmirGPT (all facts in them are true for this project);
  5. write  <out>/{train,val}_{tokens,mask,offsets}.npy  + stats.json.

Usage: python -m sft.prepare_sft --out sft_data
"""

import argparse
import json
import os
import random

import numpy as np
import requests
from tqdm import tqdm

from sft.chat_format import encode_conversation

REPO = "HuggingFaceTB/smol-smoltalk"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
MAX_LEN = 1024

# ------------------------------------------------------------------ identity data
IDENTITY_QUESTIONS = [
    "Who are you?", "What is your name?", "What's your name?", "Introduce yourself.",
    "Tell me about yourself.", "What are you?", "Are you ChatGPT?", "Who made you?",
    "Who created you?", "Who built you?", "What model are you?", "How were you trained?",
    "Are you a human?", "hi, who am I talking to?", "What should I call you?",
    "Which company made you?", "How big are you?", "How many parameters do you have?",
    "What data were you trained on?", "Can you tell me what AmirGPT is?",
]
IDENTITY_ANSWERS = [
    "I'm AmirGPT, a small language model. I'm a GPT-2 (124M parameter) model that was trained from scratch "
    "on about 10 billion tokens of FineWeb-Edu web text and then fine-tuned to chat.",
    "My name is AmirGPT. I'm a 124-million-parameter GPT-2 style transformer. I was pretrained from scratch, "
    "following Andrej Karpathy's \"Let's reproduce GPT-2\" lecture, and then fine-tuned on conversations so I can chat.",
    "I'm AmirGPT! I'm a compact AI assistant built by reproducing GPT-2 (124M) from scratch and fine-tuning it "
    "for conversation. I'm much smaller than modern assistants, so I can make mistakes. Please double-check important facts.",
    "I'm AmirGPT, an AI language model, not a human. I have about 124 million parameters and was trained on "
    "educational web text (FineWeb-Edu), then taught to follow a user/assistant chat format.",
]
SPECIFIC = {
    "Are you ChatGPT?": "No, I'm AmirGPT. I'm a much smaller model: a GPT-2 (124M) reproduced from scratch and fine-tuned for chat.",
    "How many parameters do you have?": "I have about 124 million parameters, the same size as the smallest GPT-2 model.",
    "How big are you?": "I'm small for a language model: about 124 million parameters (GPT-2 small: 12 layers, 12 attention heads, 768-dimensional embeddings).",
    "What data were you trained on?": "I was pretrained on about 10 billion tokens of FineWeb-Edu, a dataset of educational web pages, and then fine-tuned on conversational data (smol-smoltalk).",
    "Are you a human?": "No, I'm AmirGPT, an AI language model with about 124 million parameters.",
}


def identity_conversations(n, seed):
    rng = random.Random(seed)
    convs = []
    for _ in range(n):
        q = rng.choice(IDENTITY_QUESTIONS)
        a = SPECIFIC.get(q) if q in SPECIFIC and rng.random() < 0.7 else rng.choice(IDENTITY_ANSWERS)
        convs.append([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    return convs


# ------------------------------------------------------------------ download
def download(out_dir):
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    listing = requests.get(API, timeout=60).json()
    files = []
    for item in listing:
        path, size = item["path"], item["size"]
        dst = os.path.join(raw, os.path.basename(path))
        if not (os.path.exists(dst) and os.path.getsize(dst) == size):
            with requests.get(RESOLVE + path, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dst + ".tmp", "wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=os.path.basename(path)) as bar:
                    for chunk in r.iter_content(1 << 20):
                        bar.update(f.write(chunk))
            os.replace(dst + ".tmp", dst)
        assert os.path.getsize(dst) == size, f"size mismatch for {dst}"
        files.append(dst)
    return sorted(files)


def iter_conversations(parquet_files):
    import pyarrow.parquet as pq
    for fname in parquet_files:
        pf = pq.ParquetFile(fname)
        for batch in pf.iter_batches(batch_size=2048, columns=["messages"]):
            for msgs in batch.column(0).to_pylist():
                yield [{"role": m["role"], "content": m["content"]} for m in msgs]


def build_split(convs, desc):
    all_tok, all_mask, offsets = [], [], [0]
    stats = {"kept": 0, "dropped_bad_role": 0, "dropped_no_assistant": 0, "dropped_other_identity": 0,
             "truncated": 0, "dropped_after_trunc": 0}
    for msgs in tqdm(convs, desc=desc):
        if any(m["role"] not in ("system", "user", "assistant") for m in msgs) or any(m["content"] is None for m in msgs):
            stats["dropped_bad_role"] += 1
            continue
        # the source data was written for SmolLM; an assistant naming itself SmolLM would teach the wrong identity
        if any(m["role"] == "assistant" and "smollm" in m["content"].lower() for m in msgs):
            stats["dropped_other_identity"] += 1
            continue
        if not any(m["role"] == "assistant" for m in msgs):
            stats["dropped_no_assistant"] += 1
            continue
        toks, mask = encode_conversation(msgs)
        if len(toks) > MAX_LEN:
            stats["truncated"] += 1
            toks, mask = toks[:MAX_LEN], mask[:MAX_LEN]
            if sum(mask[1:]) == 0:
                stats["dropped_after_trunc"] += 1
                continue
        all_tok.append(np.asarray(toks, dtype=np.uint16))
        all_mask.append(np.asarray(mask, dtype=np.uint8))
        offsets.append(offsets[-1] + len(toks))
        stats["kept"] += 1
    stats["tokens"] = int(offsets[-1])
    stats["assistant_tokens"] = int(sum(int(m.sum()) for m in all_mask))
    return np.concatenate(all_tok), np.concatenate(all_mask), np.asarray(offsets, dtype=np.int64), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft_data")
    ap.add_argument("--identity_train", type=int, default=1000)
    ap.add_argument("--identity_val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    files = download(args.out)
    train_files = [f for f in files if "train-" in os.path.basename(f)]
    test_files = [f for f in files if "test-" in os.path.basename(f)]
    assert train_files and test_files, files

    rng = random.Random(args.seed)
    train = list(iter_conversations(train_files)) + identity_conversations(args.identity_train, args.seed)
    rng.shuffle(train)
    val = list(iter_conversations(test_files)) + identity_conversations(args.identity_val, args.seed + 1)

    all_stats = {}
    for name, convs in (("train", train), ("val", val)):
        toks, mask, offs, stats = build_split(convs, name)
        np.save(os.path.join(args.out, f"{name}_tokens.npy"), toks)
        np.save(os.path.join(args.out, f"{name}_mask.npy"), mask)
        np.save(os.path.join(args.out, f"{name}_offsets.npy"), offs)
        all_stats[name] = stats
        print(name, stats)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump({"dataset": REPO, **all_stats}, f, indent=2)


if __name__ == "__main__":
    main()
