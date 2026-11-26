"""
Assemble the static GitHub Pages site around a static frontend build.

  VITE_STATIC=1 VITE_BASE=/AmirGPT/ npm run build          # in web/frontend -> web/frontend/dist
  python -m web.build_static_site --dist web/frontend/dist --onnx <dir with amirgpt-*-fp16.onnx + .meta.json>

Adds to the build:
  models/<name>/meta.json        the export's meta.json plus "parts"
  models/<name>/model.onnx.NNN   the ONNX file split into parts below GitHub's 100 MB file limit
  metrics.json                   the Lab tab's training curves (same as the server's /api/metrics)
  .nojekyll                      serve files as they are
The in-browser engine downloads the parts, joins them and checks the whole file's sha256.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from web.metrics import build_metrics

PART_BYTES = 95_000_000  # below GitHub's 100 MB per-file limit
MODELS = ["amirgpt-chat-fp16", "amirgpt-base-fp16"]


def add_model(onnx_dir: Path, out_dir: Path, name: str):
    src = onnx_dir / f"{name}.onnx"
    meta = json.loads((onnx_dir / f"{name}.meta.json").read_text())
    data = src.read_bytes()
    assert len(data) == meta["bytes"], f"{name}: size {len(data)} != meta {meta['bytes']}"
    assert hashlib.sha256(data).hexdigest() == meta["sha256"], f"{name}: sha256 does not match meta.json"
    dest = out_dir / "models" / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    parts = []
    for i in range(0, len(data), PART_BYTES):
        part = f"{meta['file']}.{i // PART_BYTES:03d}"
        (dest / part).write_bytes(data[i:i + PART_BYTES])
        parts.append(part)
    meta["parts"] = parts
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: {len(data):,} bytes in {len(parts)} parts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="web/frontend/dist")
    ap.add_argument("--onnx", required=True, help="folder with <name>.onnx and <name>.meta.json for each model")
    ap.add_argument("--logs", default="logs_from_cluster")
    args = ap.parse_args()
    dist, onnx_dir = Path(args.dist), Path(args.onnx)
    assert (dist / "index.html").exists(), f"no frontend build in {dist}"
    for name in MODELS:
        add_model(onnx_dir, dist, name)
    (dist / "metrics.json").write_text(json.dumps(build_metrics(Path(args.logs))))
    (dist / ".nojekyll").write_text("")
    total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    biggest = max((f for f in dist.rglob("*") if f.is_file()), key=lambda f: f.stat().st_size)
    print(f"site: {total / 1e6:.1f} MB, largest file {biggest.relative_to(dist)} ({biggest.stat().st_size / 1e6:.1f} MB)")
    assert total < 1_000_000_000, "GitHub Pages sites must stay under 1 GB"
    assert biggest.stat().st_size < 100_000_000, "a file is over 100 MB"


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
