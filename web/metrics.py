"""
Training curves and reference numbers for the Lab tab. Used by the server (/api/metrics) and by
web/build_static_site.py (metrics.json for the static GitHub Pages build). No torch import.
"""

import json
from pathlib import Path

# reference numbers, with sources (see gpt2/hellaswag.py and build-nanogpt/play.ipynb)
BASELINES = {
    "val_loss_openai_gpt2_124M": {"value": 3.2924, "source": "build-nanogpt play.ipynb: OpenAI GPT-2 (124M) checkpoint val loss on FineWeb-Edu val (Karpathy's shard)"},
    "val_loss_openai_gpt2_124M_ours": {"value": 3.2793, "source": "measured here (gpt2/eval_val.py, fp32): OpenAI GPT-2 (124M) on OUR val shard, same first 10,485,760 tokens our training evaluates"},
    "hella_openai_gpt2_124M": {"value": 0.2955, "source": "video [3:38:06] / build-nanogpt hellaswag.py: acc_norm 0.2955 (this eval; we reproduced 2967/10042)"},
    "hella_openai_gpt3_124M": {"value": 0.337, "source": "build-nanogpt play.ipynb: GPT-3 paper 124M HellaSwag"},
    "hella_random": {"value": 0.25, "source": "4-way multiple choice"},
}


def _read_jsonl(path):
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a line being written right now
    return rows


def build_metrics(logs: Path) -> dict:
    pre = _read_jsonl(logs / "pretrain" / "metrics.jsonl")
    sft = _read_jsonl(logs / "sft" / "metrics.jsonl")

    def series(rows, kind, every=1):
        pts = [(r["step"], r["value"]) for r in rows if r.get("kind") == kind]
        return pts[::every] if every > 1 else pts

    cfg_path = logs / "pretrain" / "run_config.json"
    return {
        "pretrain": {"train": series(pre, "train", 10), "val": series(pre, "val"), "hella": series(pre, "hella"),
                     "samples": [r for r in pre if r.get("kind") == "samples"][-5:],
                     "config": json.load(open(cfg_path)) if cfg_path.exists() else None},
        "sft": {"train": series(sft, "train"), "val": series(sft, "val"),
                "samples": [r for r in sft if r.get("kind") == "samples"][-3:]},
        "baselines": BASELINES,
    }
