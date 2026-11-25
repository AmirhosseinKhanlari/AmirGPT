#!/bin/bash
# Download the FineWeb-Edu sample-10BT parquet files (14 files, ~28.5 GB) over plain HTTPS.
# (The default HF "xet" transfer stalled on this cluster; the CDN URLs are fast: ~40 MB/s per stream.)
# Resumable (curl -C -), retried, and every file's size is verified against the HF API listing.
set -euo pipefail
OUT=${1:-$HOME/AmirGPT/raw_fineweb}
mkdir -p "$OUT"
API=https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT
curl -sf "$API" | python3 -c '
import json, sys
for x in json.load(sys.stdin):
    print(x["path"], x["size"])' > "$OUT/manifest.txt"
echo "files: $(wc -l < "$OUT/manifest.txt")"
cut -d" " -f1 "$OUT/manifest.txt" | xargs -P 7 -I{} bash -c '
  f="{}"; name=$(basename "$f")
  curl -sfL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "'"$OUT"'/$name" \
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/$f" && echo "got $name"'
# verify sizes
bad=0
while read -r path size; do
  name=$(basename "$path"); actual=$(stat -c %s "$OUT/$name" 2>/dev/null || echo 0)
  if [ "$actual" != "$size" ]; then echo "SIZE MISMATCH $name: $actual != $size"; bad=1; fi
done < "$OUT/manifest.txt"
[ $bad -eq 0 ] && echo "all files verified" || { echo "verification FAILED"; exit 1; }
