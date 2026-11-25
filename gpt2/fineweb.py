"""
FineWeb-Edu dataset (for pretraining) -> tokenized uint16 shards on disk.
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Follows Karpathy's build-nanogpt/fineweb.py (video [3:12:38]-[3:17:07]), with improvements:
  * --streaming / --max_shards: build only the first N shards without downloading
    the whole 10B-token sample (useful on a laptop, or for smoke tests).
  * output dir, dataset config, shard size, and worker count are CLI flags.
  * each shard is written atomically (tmp file + rename) so a killed job never
    leaves a half-written shard that the dataloader would silently read.
  * a final summary (shards, total tokens) is written to <out>/summary.json.
  * --parquet_dir: read already-downloaded parquet files (see cluster/download_fineweb.sh)
    instead of going through `datasets` (whose default transfer stalled on our cluster).
    Files are read in sorted order, which is the same document order load_dataset uses.

Format (unchanged from the video, so it is compatible with the dataloader):
  every document is prefixed with the <|endoftext|> token (id 50256) as a
  delimiter; tokens are stored as uint16 numpy arrays of exactly `shard_size`
  tokens (the last shard may be shorter). Shard 0 is the validation split.

Usage:
  python -m gpt2.fineweb                                  # full 10B sample (~100 shards)
  python -m gpt2.fineweb --streaming --max_shards 3       # quick local subset
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import tiktoken
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]  # 50256


def tokenize(doc):
    """Tokenize one document -> uint16 numpy array, prefixed with <|endoftext|>."""
    tokens = [EOT]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def iterate_parquet(parquet_dir):
    """Yield {"text": ...} rows from all *.parquet files in parquet_dir, in sorted file order."""
    import pyarrow.parquet as pq
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
    assert files, f"no parquet files in {parquet_dir}"
    for fname in files:
        pf = pq.ParquetFile(os.path.join(parquet_dir, fname))
        for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
            for text in batch.column(0).to_pylist():
                yield {"text": text}


def write_datafile(filename, tokens_np):
    tmp = filename + ".tmp.npy"
    np.save(tmp, tokens_np)
    os.replace(tmp, filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="edu_fineweb10B", help="output directory for shards")
    ap.add_argument("--remote_name", default="sample-10BT", help="fineweb-edu config name")
    ap.add_argument("--shard_size", type=int, default=int(1e8), help="tokens per shard (100M)")
    ap.add_argument("--streaming", action="store_true", help="stream instead of downloading everything")
    ap.add_argument("--max_shards", type=int, default=-1, help="stop after this many shards (-1 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--parquet_dir", default=None, help="read local parquet files instead of HF datasets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.parquet_dir:
        fw = iterate_parquet(args.parquet_dir)
    else:
        from datasets import load_dataset  # imported lazily: only needed for data prep
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=args.remote_name, split="train",
                          streaming=args.streaming)

    shard_index = 0
    total_tokens = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    done = False

    with mp.Pool(args.workers) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < args.shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(args.out, f"edufineweb_{split}_{shard_index:06d}.npy")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = args.shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                    progress_bar.close()
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                total_tokens += args.shard_size
                shard_index += 1
                progress_bar = None
                if args.max_shards > 0 and shard_index >= args.max_shards:
                    done = True
                    break
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

    # write any remaining tokens as the last shard
    if not done and token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(args.out, f"edufineweb_{split}_{shard_index:06d}.npy")
        write_datafile(filename, all_tokens_np[:token_count])
        total_tokens += token_count
        shard_index += 1

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"shards": shard_index, "total_tokens": int(total_tokens),
                   "remote_name": args.remote_name, "shard_size": args.shard_size}, f, indent=2)
    print(f"done: {shard_index} shards, {total_tokens:,} tokens -> {args.out}")


if __name__ == "__main__":
    main()
