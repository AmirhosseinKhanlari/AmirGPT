"""
HellaSwag evaluation (video [3:28:35]). https://github.com/rowanz/hellaswag

Each example has a context and 4 candidate endings; exactly one is correct.
We score each ending by the model's average per-token cross-entropy over the
ending tokens only ("acc_norm" style, as in Karpathy's code) and predict the
ending with the lowest loss. This lets a small base model (no instruction
following) be evaluated with a multiple-choice benchmark.

Reference numbers (validation set, 10,042 examples), with sources:
  OpenAI GPT-2 124M, this script:  acc 0.2859, acc_norm 0.2955
      (video [3:38:06]: "our script reports 2955 and so that is the number that we'd
       like to beat"; also the header of build-nanogpt/hellaswag.py)
  OpenAI GPT-2 124M, as plotted in build-nanogpt/play.ipynb:  0.294463
  OpenAI GPT-3 124M (paper), as plotted in play.ipynb:          0.337
  random chance: 0.25 (video [3:37:19])

Improvement over the original: this module can evaluate OUR GPT class (with
OpenAI weights loaded via GPT.from_pretrained, or any of our checkpoints) as
well as the HuggingFace reference model, so the number doubles as an
end-to-end correctness check of the implementation.

Usage:
  python -m gpt2.hellaswag --download_only
  python -m gpt2.hellaswag --model gpt2                 # ours, OpenAI weights
  python -m gpt2.hellaswag --model gpt2 --hf            # HuggingFace reference
  python -m gpt2.hellaswag --checkpoint log/model_19072.pt
"""

import argparse
import json
import os

import requests
import tiktoken
import torch
from torch.nn import functional as F
from tqdm import tqdm

DATA_CACHE_DIR = os.environ.get("HELLASWAG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hellaswag"))
DATA_CACHE_DIR = os.path.abspath(DATA_CACHE_DIR)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = fname + ".tmp"
    with open(tmp, "wb") as file, tqdm(desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            bar.update(file.write(data))
    os.replace(tmp, fname)


# The original GitHub repo (rowanz/hellaswag) was blocked by a DMCA takedown on
# 2026-09-14, after the video, so the URLs above now return 404. Fallback: the
# author's copy on HuggingFace (Rowan/hellaswag), converted back to the original
# jsonl fields we use (ctx, endings, integer label). Verified by reproducing the
# reference numbers for OpenAI GPT-2 124M exactly (acc 0.2859, acc_norm 0.2955).
HF_MIRROR = {
    "train": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/train-00000-of-00001.parquet",
    "val": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/test-00000-of-00001.parquet",
}
EXPECTED_ROWS = {"val": 10042}


def _download_from_hf_mirror(split, data_filename):
    import pyarrow.parquet as pq
    pq_file = data_filename + ".parquet"
    download_file(HF_MIRROR[split], pq_file)
    rows = pq.read_table(pq_file).to_pylist()
    if split in EXPECTED_ROWS:
        assert len(rows) == EXPECTED_ROWS[split], f"expected {EXPECTED_ROWS[split]} rows, got {len(rows)}"
    tmp = data_filename + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            label = r["label"]
            f.write(json.dumps({"ind": r.get("ind"), "ctx": r["ctx"], "endings": list(r["endings"]),
                                "label": int(label) if label not in (None, "") else -1,
                                "activity_label": r.get("activity_label")}) + "\n")
    os.replace(tmp, data_filename)
    os.remove(pq_file)


def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {hellaswags[split]} to {data_filename}...")
        try:
            download_file(hellaswags[split], data_filename)
        except requests.HTTPError as e:
            print(f"original source unavailable ({e}); using HuggingFace mirror Rowan/hellaswag")
            _download_from_hf_mirror(split, data_filename)
    return data_filename


def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {"label": label, "ctx_tokens": None, "ending_tokens": []}

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows, mask_rows = [], []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    # there are 10,042 examples in total in val
    with open(download(split), "r") as f:
        for line in f:
            yield json.loads(line)


def get_most_likely_row(tokens, mask, logits):
    """Index of the completion with the lowest average loss over the completion tokens."""
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits.float(), flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # we must shift mask, so we start at the last prompt token
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()


@torch.no_grad()
def evaluate(model_name=None, checkpoint=None, use_hf=False, device="cuda", limit=-1):
    torch.set_float32_matmul_precision("high")
    if use_hf:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(f"openai-community/{model_name}")
        forward = lambda t: model(t).logits
    else:
        from gpt2.model import GPT, load_checkpoint
        model = load_checkpoint(checkpoint, map_location="cpu")[0] if checkpoint else GPT.from_pretrained(model_name)
        forward = lambda t: model(t)[0]
    model.to(device).eval()

    num_correct_norm, num_total = 0, 0
    for example in tqdm(iterate_examples("val"), total=10042 if limit < 0 else limit, desc="hellaswag"):
        if 0 <= limit <= num_total:
            break
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        logits = forward(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    acc = num_correct_norm / num_total
    print(f"HellaSwag acc_norm: {num_correct_norm}/{num_total} = {acc:.4f}")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download_only", action="store_true")
    ap.add_argument("-m", "--model", default="gpt2", help="gpt2 | gpt2-medium | gpt2-large | gpt2-xl")
    ap.add_argument("--checkpoint", default=None, help="evaluate one of our training checkpoints instead")
    ap.add_argument("--hf", action="store_true", help="evaluate the HuggingFace reference model")
    ap.add_argument("-d", "--device", default=None)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()
    if args.download_only:
        print(download("val"))
    else:
        dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        evaluate(args.model, args.checkpoint, args.hf, dev, args.limit)
