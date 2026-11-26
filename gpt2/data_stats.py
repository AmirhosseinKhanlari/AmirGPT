"""
Dataset statistics for the report: token and document counts per shard and the document-length
distribution of our tokenized FineWeb-Edu sample-10BT (every document starts with <|endoftext|>).

Usage: python -m gpt2.data_stats --data_root edu_fineweb10B --out bench_results/data_stats.json
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

EOT = 50256
# log-spaced histogram edges for document length in tokens (1 ... 2^20)
EDGES = np.unique(np.round(np.logspace(0, 20, 81, base=2)).astype(np.int64))


def shard_stats(path):
    t = np.load(path, mmap_mode="r")
    starts = np.flatnonzero(np.asarray(t) == EOT)
    # a document spans from its EOT to the token before the next EOT; the last one may continue in the next shard
    lengths = np.diff(np.concatenate((starts, [len(t)])))
    head = int(starts[0]) if len(starts) else len(t)   # tokens before the first EOT (tail of previous shard's doc)
    hist, _ = np.histogram(lengths, bins=EDGES)
    return {"shard": os.path.basename(path), "tokens": int(len(t)), "docs_starting": int(len(starts)),
            "head_tokens": head, "hist": hist.tolist(), "max_len": int(lengths.max()) if len(lengths) else 0,
            "lengths_sample": lengths[:: max(1, len(lengths) // 2000)].tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="edu_fineweb10B")
    ap.add_argument("--out", default="bench_results/data_stats.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    shards = sorted(os.path.join(args.data_root, f) for f in os.listdir(args.data_root)
                    if f.endswith(".npy") and ".tmp" not in f)
    with Pool(args.workers) as p:
        per = p.map(shard_stats, shards)
    sample = np.concatenate([np.array(s.pop("lengths_sample")) for s in per])
    hist = np.sum([s.pop("hist") for s in per], axis=0)
    train = [s for s in per if "train" in s["shard"]]
    val = [s for s in per if "val" in s["shard"]]
    out = {
        "shards": len(per), "train_shards": len(train), "val_shards": len(val),
        "tokens_total": sum(s["tokens"] for s in per),
        "tokens_train": sum(s["tokens"] for s in train), "tokens_val": sum(s["tokens"] for s in val),
        "documents": sum(s["docs_starting"] for s in per),
        "doc_len_tokens_percentiles_from_sample": {str(q): float(np.percentile(sample, q)) for q in (10, 25, 50, 75, 90, 99)},
        "doc_len_tokens_mean": sum(s["tokens"] for s in per) / max(1, sum(s["docs_starting"] for s in per)),
        "doc_len_max_within_shard": max(s["max_len"] for s in per),
        "hist_edges": EDGES.tolist(), "hist_counts": hist.tolist(),
        "per_shard": per,
        "notes": "Percentiles come from an evenly strided sample of about 2,000 documents per shard; "
                 "counts, mean and histogram use every document. A document cut at a shard boundary is counted in the shard where it starts.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_shard", "hist_edges", "hist_counts")}, indent=1))


if __name__ == "__main__":
    main()
