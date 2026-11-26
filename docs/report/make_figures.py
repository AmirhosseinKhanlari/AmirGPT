"""
Figures for docs/report.md, built only from measured data in this repo:
  docs/report/scaling.json      (cluster/bench_scaling.sbatch)
  docs/report/data_stats.json   (cluster/data_stats.sbatch)
  logs_from_cluster/            (the training runs)
Run from the project root:  .venv/bin/python docs/report/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "report"
SERIES = "#257de0"      # categorical slot 1 (blue)
REF = "#8a8a8a"         # reference lines: neutral gray
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": TEXT, "axes.titlelocation": "left",
})


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def ref_line(ax, y, label, x_text, va="bottom"):
    ax.axhline(y, color=REF, lw=1.2, ls=(0, (4, 3)))
    ax.text(x_text, y, f" {label}", color=TEXT2, fontsize=8.5, va=va, ha="right")


# 1. data-parallel scaling ------------------------------------------------------------------
sc = json.load(open(OUT / "scaling.json"))
n = np.array(sorted(int(k) for k in sc))
tps = np.array([sc[str(k)]["tok_per_sec"] for k in n]) / 1000
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(n, tps[0] * n, color=REF, lw=1.2, ls=(0, (4, 3)))
ax.text(8, tps[0] * 8, "ideal linear scaling ", color=TEXT2, fontsize=8.5, ha="right", va="bottom")
ax.plot(n, tps, color=SERIES, lw=2, marker="o", ms=7, mec="white", mew=2)
for k, t in zip(n, tps):
    eff = sc[str(k)]["efficiency"] * 100
    ax.annotate(f"{t:,.0f}k tok/s\n{eff:.1f}% efficient" if k > 1 else f"{t:,.0f}k tok/s",
                (k, t), textcoords="offset points", xytext=(8, -26) if k > 1 else (10, -14), fontsize=8.5, color=TEXT)
ax.set_xticks(n)
ax.set_xlabel("V100 GPUs (DDP)")
ax.set_ylabel("training throughput (thousand tokens/s)")
ax.set_title("GPT-2 124M training throughput scales almost linearly")
ax.set_ylim(0, tps[0] * 8 * 1.08)
save(fig, "fig_scaling.png")

# 2. document length distribution -----------------------------------------------------------
ds = json.load(open(OUT / "data_stats.json"))
edges = np.array(ds["hist_edges"], dtype=float)
counts = np.array(ds["hist_counts"], dtype=float)
keep = counts > 0
lo, hi = edges[:-1][keep], edges[1:][keep]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.bar(lo, counts[keep] / 1e3, width=(hi - lo) * 0.88, align="edge", color=SERIES, edgecolor="white", linewidth=0.5)
ax.set_xscale("log")
med = ds["doc_len_tokens_percentiles_from_sample"]["50"]
top = ax.get_ylim()[1]
ax.axvline(med, color=REF, lw=1.2, ls=(0, (4, 3)))
ax.axvline(1024, color=REF, lw=1.2, ls=":")
ax.annotate(f"median: {med:,.0f} tokens", (med, top * 0.97), xytext=(9000, top * 0.86), color=TEXT2, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=REF, lw=0.8))
ax.annotate("context length: 1,024 tokens", (1024, top * 0.72), xytext=(9000, top * 0.70), color=TEXT2, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=REF, lw=0.8))
ax.set_xlabel("document length (GPT-2 tokens, log scale)")
ax.set_ylabel("documents (thousands)")
ax.set_title(f"FineWeb-Edu: {ds['documents']:,} documents, {ds['tokens_total'] / 1e9:.2f}B tokens")
save(fig, "fig_doc_lengths.png")

# 3. pretraining validation loss -------------------------------------------------------------
pre = read_jsonl(ROOT / "logs_from_cluster" / "pretrain" / "metrics.jsonl")
val = np.array([(r["step"], r["value"]) for r in pre if r.get("kind") == "val"])
hel = np.array([(r["step"], r["value"]) for r in pre if r.get("kind") == "hella"])
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(val[:, 0], val[:, 1], color=SERIES, lw=2)
ref_line(ax, 3.2793, "OpenAI GPT-2 124M on the same val tokens: 3.2793", val[-1, 0], va="top")
ax.set_ylim(2.95, 4.6)
ax.set_xlabel("training step (524,288 tokens each)")
ax.set_ylabel("validation loss")
ax.set_title(f"AmirGPT validation loss: {val[-1, 1]:.4f} at the last step")
ax.text(val[-1, 0], val[-1, 1] + 0.03, f"{val[-1, 1]:.4f}", color=TEXT, fontsize=8.5, ha="right", va="bottom")
save(fig, "fig_val_loss.png")

# 4. HellaSwag ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(hel[:, 0], hel[:, 1], color=SERIES, lw=2)
ax.axhline(0.2955, color=REF, lw=1.2, ls=(0, (4, 3)))
ax.text(200, 0.2955, "OpenAI GPT-2 124M: 0.2955", color=TEXT2, fontsize=8.5, va="bottom", ha="left")
ref_line(ax, 0.25, "random guessing: 0.25", hel[-1, 0], va="bottom")
ax.set_ylim(0.23, 0.32)
ax.set_xlabel("training step (524,288 tokens each)")
ax.set_ylabel("HellaSwag acc_norm")
ax.set_title(f"AmirGPT HellaSwag accuracy: {hel[-1, 1]:.4f} at the last step")
save(fig, "fig_hellaswag.png")
