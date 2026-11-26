"""
Prepare the preference (DPO) dataset for AmirGPT.

Dataset: HuggingFaceH4/ultrafeedback_binarized (MIT), splits train_prefs (61,135 rows)
and test_prefs (2,000 rows). Each row is one user prompt with two assistant replies,
"chosen" and "rejected". Its create_dataset.py takes the completion with the highest
GPT-4 `overall_score` as chosen and a random OTHER completion as rejected -- which may
have the SAME score (a tie), in which case the label is arbitrary.

Steps:
  1. download the two parquet files over HTTPS (sizes checked against the HF tree API;
     row counts checked against the parquet metadata and the HF datasets-server API);
  2. encode prompt+chosen and prompt+rejected with sft/chat_format.py; the loss mask
     covers ONLY the final assistant reply (its text and its closing <|end|>);
  3. drop pairs that are malformed, have an empty reply, identical replies, tied scores
     (unless --keep_ties), a reply that names another assistant / model maker (so DPO does
     not undo AmirGPT's identity), or a prompt so long that fewer than --min_response
     tokens would be left for the reply; truncate the rest to 1024 tokens;
  4. write  <out>/{split}_{chosen,rejected}_{tokens,mask,offsets}.npy  (the same layout as
     sft/prepare_sft.py, so sft.train_sft.ConversationData reads each side directly),
     <out>/{split}_scores.npy  (float32 [N, 2]: score_chosen, score_rejected)  and stats.json.
  The splits are written as "train" (from train_prefs) and "test" (from test_prefs).

Usage: python -m sft.prepare_dpo --out dpo_data
"""

import argparse
import json
import os
import re

import numpy as np
import requests
from tqdm import tqdm

from sft.chat_format import ASSISTANT, encode_conversation

REPO = "HuggingFaceH4/ultrafeedback_binarized"
TREE_API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
SIZE_API = f"https://datasets-server.huggingface.co/size?dataset={REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
SPLITS = {"train": "train_prefs", "test": "test_prefs"}  # our name -> HF split name
MAX_LEN = 1024

# A reply that names another assistant or model maker would teach AmirGPT a wrong identity
# ("As an AI developed by OpenAI ..."). Case-insensitive patterns are distinctive words;
# LLaMA is matched case-sensitively so the animal ("llama") is not caught. Some false
# positives remain (e.g. "Claude Monet", "the Bard", the zodiac sign "Gemini"): dropping
# those pairs costs a little data but is harmless.
IDENTITY_PATTERNS = {
    "openai": re.compile(r"open\s?ai", re.I),
    "chatgpt": re.compile(r"chat\s?gpt", re.I),
    "gpt-3/4": re.compile(r"\bgpt[- ]?(3|4)", re.I),
    "anthropic": re.compile(r"\banthropic\b", re.I),
    "claude": re.compile(r"\bclaude\b", re.I),
    "llama": re.compile(r"\bLLaMA|\bLLAMA\b|\bLlama[- ]?\d"),
    "meta ai": re.compile(r"\bmeta ai\b", re.I),
    "bard": re.compile(r"\bbard\b", re.I),
    "gemini": re.compile(r"\bgemini\b", re.I),
    # other models whose completions are in UltraFeedback (see its dataset card)
    "other model names": re.compile(r"\b(vicuna|wizardlm|ultralm|starchat|lmsys|mosaicml)\b", re.I),
    "developed by": re.compile(r"\b(language model|AI model|AI assistant|assistant|AI)\b[^.\n]{0,40}?"
                               r"\b(developed|created|trained|built|made|designed)\s+by\b", re.I),
}


def identity_hit(text):
    """Name of the first identity pattern found in text, or None."""
    for name, pat in IDENTITY_PATTERNS.items():
        if pat.search(text):
            return name
    return None


# ------------------------------------------------------------------ download
def expected_rows():
    """{hf_split: num_rows} from the HF datasets-server API (None if unreachable)."""
    try:
        info = requests.get(SIZE_API, timeout=60).json()
        return {s["split"]: int(s["num_rows"]) for s in info["size"]["splits"]}
    except Exception as e:  # noqa: BLE001 -- the check is extra; the parquet metadata is still verified
        print(f"warning: could not query {SIZE_API}: {e}")
        return None


def download(out_dir):
    """Download the train_prefs / test_prefs parquet files. Returns {our_split: path}."""
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    listing = requests.get(TREE_API, timeout=60).json()
    paths = {}
    for ours, hf in SPLITS.items():
        items = [it for it in listing if os.path.basename(it["path"]).startswith(hf + "-")]
        assert len(items) == 1, f"expected one parquet file for {hf}, got {[it['path'] for it in items]}"
        path, size = items[0]["path"], items[0]["size"]
        dst = os.path.join(raw, os.path.basename(path))
        if not (os.path.exists(dst) and os.path.getsize(dst) == size):
            with requests.get(RESOLVE + path, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dst + ".tmp", "wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=os.path.basename(path)) as bar:
                    for chunk in r.iter_content(1 << 20):
                        bar.update(f.write(chunk))
            os.replace(dst + ".tmp", dst)
        assert os.path.getsize(dst) == size, f"size mismatch for {dst}"
        paths[ours] = dst
    return paths


def read_rows(parquet_file):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_file)
    cols = ["prompt_id", "chosen", "rejected", "score_chosen", "score_rejected"]
    rows = []
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        rows += batch.to_pylist()
    assert len(rows) == pf.metadata.num_rows, (len(rows), pf.metadata.num_rows)
    return rows


# ------------------------------------------------------------------ encoding
def encode_pair_side(prompt_msgs, reply, min_response=64, max_len=MAX_LEN):
    """Encode prompt + final assistant reply. Only the final reply is learned (mask=1 on its
    text and closing <|end|>), even if prompt_msgs contains earlier assistant turns.
    Returns (tokens, mask, truncated) or None if the prompt leaves < min_response tokens."""
    toks, mask = encode_conversation(prompt_msgs + [{"role": "assistant", "content": reply}])
    n_prompt = len(encode_conversation(prompt_msgs)[0]) + 1  # + the <|assistant|> role token
    assert toks[n_prompt - 1] == ASSISTANT
    mask = [0] * n_prompt + mask[n_prompt:]
    if max_len - n_prompt < min_response:
        return None
    truncated = len(toks) > max_len
    return toks[:max_len], mask[:max_len], truncated


def build_split(rows, desc, min_response=64, keep_ties=False):
    """rows: dicts with chosen/rejected message lists and scores. Returns (arrays, stats)."""
    out = {side: {"tok": [], "mask": [], "off": [0]} for side in ("chosen", "rejected")}
    scores = []
    stats = {"rows": len(rows), "kept": 0, "dropped_bad_format": 0, "dropped_empty_reply": 0,
             "dropped_identical": 0, "dropped_tie": 0, "dropped_score_inverted": 0,
             "dropped_other_identity": 0, "dropped_prompt_too_long": 0,
             "truncated_pairs": 0, "truncated_chosen": 0, "truncated_rejected": 0,
             "other_identity_by_pattern": {k: 0 for k in IDENTITY_PATTERNS}}
    for r in tqdm(rows, desc=desc):
        ch, rj = r["chosen"], r["rejected"]
        ok = (ch and rj and len(ch) == len(rj) and ch[-1]["role"] == "assistant" and rj[-1]["role"] == "assistant"
              and all(m["role"] in ("system", "user", "assistant") and m["content"] is not None for m in ch + rj)
              and [(m["role"], m["content"]) for m in ch[:-1]] == [(m["role"], m["content"]) for m in rj[:-1]]
              and any(m["role"] == "user" for m in ch[:-1]))
        if not ok:
            stats["dropped_bad_format"] += 1
            continue
        prompt = [{"role": m["role"], "content": m["content"]} for m in ch[:-1]]
        c, j = ch[-1]["content"], rj[-1]["content"]
        if not c.strip() or not j.strip():
            stats["dropped_empty_reply"] += 1
            continue
        if c.strip() == j.strip():
            stats["dropped_identical"] += 1
            continue
        sc, sr = r["score_chosen"], r["score_rejected"]
        if sc < sr:
            stats["dropped_score_inverted"] += 1
            continue
        if sc == sr and not keep_ties:
            stats["dropped_tie"] += 1
            continue
        hit = identity_hit(c) or identity_hit(j)
        if hit:
            stats["dropped_other_identity"] += 1
            stats["other_identity_by_pattern"][hit] += 1
            continue
        ec = encode_pair_side(prompt, c, min_response)
        ej = encode_pair_side(prompt, j, min_response)
        if ec is None or ej is None:  # same prompt, so both are None together
            stats["dropped_prompt_too_long"] += 1
            continue
        stats["truncated_chosen"] += ec[2]
        stats["truncated_rejected"] += ej[2]
        stats["truncated_pairs"] += ec[2] or ej[2]
        for side, (t, m, _) in (("chosen", ec), ("rejected", ej)):
            out[side]["tok"].append(np.asarray(t, dtype=np.uint16))
            out[side]["mask"].append(np.asarray(m, dtype=np.uint8))
            out[side]["off"].append(out[side]["off"][-1] + len(t))
        scores.append((sc, sr))
        stats["kept"] += 1
    arrays = {}
    for side, d in out.items():
        assert d["tok"], f"no pairs kept for {desc}"
        arrays[side] = (np.concatenate(d["tok"]), np.concatenate(d["mask"]), np.asarray(d["off"], dtype=np.int64))
        stats[f"{side}_tokens"] = int(d["off"][-1])
        stats[f"{side}_response_tokens"] = int(sum(int(m.sum()) for m in d["mask"]))
        stats[f"{side}_mean_len"] = round(d["off"][-1] / len(d["tok"]), 1)
    arrays["scores"] = np.asarray(scores, dtype=np.float32)
    return arrays, stats


def save_split(out_dir, name, arrays):
    for side in ("chosen", "rejected"):
        toks, mask, offs = arrays[side]
        np.save(os.path.join(out_dir, f"{name}_{side}_tokens.npy"), toks)
        np.save(os.path.join(out_dir, f"{name}_{side}_mask.npy"), mask)
        np.save(os.path.join(out_dir, f"{name}_{side}_offsets.npy"), offs)
    np.save(os.path.join(out_dir, f"{name}_scores.npy"), arrays["scores"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dpo_data")
    ap.add_argument("--min_response", type=int, default=64, help="drop pairs whose prompt leaves fewer tokens than this")
    ap.add_argument("--keep_ties", action="store_true", help="keep pairs with score_chosen == score_rejected")
    ap.add_argument("--max_rows", type=int, default=0, help="use only the first N rows of each split (local tests)")
    args = ap.parse_args(argv)

    paths = download(args.out)
    expected = expected_rows()
    all_stats = {"dataset": REPO, "max_len": MAX_LEN, "min_response": args.min_response, "keep_ties": args.keep_ties}
    prompt_ids = {}
    for name, path in paths.items():
        rows = read_rows(path)
        hf = SPLITS[name]
        if expected is not None:
            assert len(rows) == expected[hf], f"{hf}: file has {len(rows)} rows, HF API says {expected[hf]}"
        print(f"{hf}: {len(rows):,} rows in {os.path.basename(path)} (HF API: {expected[hf] if expected else 'n/a'})")
        if args.max_rows:
            rows = rows[:args.max_rows]
        prompt_ids[name] = {r["prompt_id"] for r in rows}
        arrays, stats = build_split(rows, name, args.min_response, args.keep_ties)
        stats["hf_split"] = hf
        stats["rows_expected_hf_api"] = expected[hf] if expected else None
        save_split(args.out, name, arrays)
        all_stats[name] = stats
        print(name, json.dumps(stats))
    # UltraFeedback was split by row, so a prompt can occur in both splits; report it (not removed)
    all_stats["test_prompt_ids_also_in_train_rows"] = len(prompt_ids["test"] & prompt_ids["train"])
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)
    print("test prompts also in train:", all_stats["test_prompt_ids_also_in_train_rows"])


if __name__ == "__main__":
    main()
