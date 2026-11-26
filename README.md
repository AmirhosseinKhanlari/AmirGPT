# AmirGPT

> Course project for **Big Data** (Fall 2026).

**Live demo: https://amirhosseinkhanlari.github.io/AmirGPT/** runs AmirGPT Chat and Base entirely in
your browser (WebGPU; Chrome or Edge on a computer with a GPU works best). The first message downloads
the model, about 250 MB, which the browser then caches. The demo has no server, so the OpenAI GPT-2
and DPO comparison models are only in the local app (`./run.sh`).

GPT-2 (124M) **reproduced from scratch**, following Andrej Karpathy's
["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU)
([build-nanogpt](https://github.com/karpathy/build-nanogpt)). It is pretrained on 10B tokens of
FineWeb-Edu, **chat fine-tuned**, and served in a web app that shows how the model makes each choice.

* `gpt2/`: the model, data pipeline, training loop and evals, written step by step as in the video
  ([docs/walkthrough.md](docs/walkthrough.md) maps each video timestamp to the code)
* `sft/`: chat fine-tuning (the step Karpathy describes at [3:55:46] but does not cover) and an
  experimental preference-tuning (DPO) stage
* `web/`: the AmirGPT app (FastAPI + React), plus an ONNX export so the model can run in the browser
* `cluster/`: SLURM scripts for the Stony Brook AI cluster (8× V100)
* `tests/`: 32 Python tests (model vs HuggingFace logits, KV-cache, data loader, schedule, SFT, DPO, resume)
  and 31 frontend tests (markdown renderer)
* Documents: [docs/report.md](docs/report.md) (data and compute report for the course),
  [MODEL_CARD.md](MODEL_CARD.md), [docs/walkthrough.md](docs/walkthrough.md)

## Results

AmirGPT Base beats OpenAI's GPT-2 124M on both of the video's metrics after training on 10B tokens.
All numbers below come from `cluster/final_eval.sbatch`, an independent fp32 evaluation outside the
training loop, with both models evaluated on exactly the same data by the same code.

| Metric | OpenAI GPT-2 124M | **AmirGPT Base (ours)** |
|---|---|---|
| Val loss, first 10,485,760 tokens of our FineWeb-Edu val shard (lower is better) | 3.2793 | **3.0693** |
| HellaSwag acc_norm, 10,042 examples (higher is better) | 0.2955 (2967/10042) | **0.3047** (3060/10042) |

Training run (`cluster/train.sbatch`, job finished 2026-11-26):

* 19,073 steps × 524,288 tokens ≈ 10.0B tokens, 8× V100 (fp16 + GradScaler, `torch.compile`), 6 h 24 min wall clock
* Val loss: 10.95 at step 0 → below OpenAI GPT-2 from step 6,750 → **3.0693** at the last step (the training loop's fp16 eval gives the same value)
* HellaSwag: above OpenAI GPT-2 from step 12,000; the training loop's eval reads 0.3049 at the last step (fp16), the fp32 re-evaluation gives 0.3047
* The video's hyperparameters were used unchanged. Training differs in two ways: fp16 with loss scaling instead of bf16 (V100s have no bf16), and documents and shards are shuffled each epoch, which Karpathy suggests at [3:50:05]. The 10B training tokens are slightly more than one pass over the 99 training shards (≈ 9.85B tokens).

Chat fine-tuning (`cluster/sft.sbatch`, 2 epochs on smol-smoltalk plus AmirGPT identity conversations, 35 min):
loss on assistant tokens of the held-out set went from 2.456 to **1.351**.

| Prompt | AmirGPT Chat (seed 1, default settings) |
|---|---|
| Hello | Hello! How can I help you today? |
| Who are you? | I'm AmirGPT, a small language model. I'm a GPT-2 (124M parameter) model that was trained from scratch on about 10 billion tokens of FineWeb-Edu web text and then fine-tuned to chat. |
| What is the capital of France? | The capital of France is Paris. Paris is located in the northern part of the country and serves as the political, economic, and cultural center of France. … |

**Preference tuning (DPO), an experiment that did not help.** One epoch of DPO on 51,635
[UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) preference pairs
(5 min on 8× V100, `cluster/dpo.sbatch`) taught the model to prefer the better answer on 68.7% of
1,660 held-out pairs. But its replies got longer (193 → 224 tokens on 50 test prompts), slightly more
repetitive (repeated 4-grams 0.065 → 0.077), and more often ran to the length limit without finishing
(50% → 72%). AmirGPT Chat therefore stays the default; the DPO model is in the app as
"AmirGPT Chat + DPO" for comparison. Logs: `logs_from_cluster/dpo/`.

AmirGPT is a 124M-parameter model, so it makes factual mistakes and repeats itself; its full reply
about France, for example, lists Notre-Dame twice. Read its answers as a demonstration of what this scale can do,
not as reliable information.

## Correctness checks (all passing)

| Check | Expected | Ours |
|---|---|---|
| Parameter count (GPT-2's 50,257-token vocabulary) | 124,439,808 | 124,439,808 |
| Logits with OpenAI weights vs HuggingFace `GPT2LMHeadModel` | identical | match (atol 1e-3) |
| HellaSwag, our code + OpenAI GPT-2 124M weights | 0.2955 (video [3:38:06]) | **2967/10042 = 0.2955** |
| Loss at initialization | ≈ −ln(1/50257) = 10.82 | 10.95 (step 0) |
| Overfit a single batch (video [1:00:55]) | loss → ~0 | < 0.05 |
| KV-cached generation vs full recompute | identical tokens | identical |
| Tokenized shards vs reference tokenization | identical | identical |

## Run the app

```bash
./run.sh                     # http://127.0.0.1:8000
```

Models in the app:

* **AmirGPT Chat**: our model after chat fine-tuning (`checkpoints/amirgpt_chat.pt`)
* **AmirGPT Base**: our from-scratch pretrained model; it continues text (`checkpoints/amirgpt_base.pt`)
* **OpenAI GPT-2**: OpenAI's weights loaded into *our* model code, for comparison
* **AmirGPT Chat + DPO** (experimental): see the DPO result above (`checkpoints/amirgpt_chat_dpo.pt`)

Checkpoints are attached to the repository's [v1.0 release](https://github.com/AmirhosseinKhanlari/AmirGPT/releases/tag/v1.0);
put them in `checkpoints/`.

What's new in the UI:

* **Confidence lens:** each token is tinted by how unsure the model was (1 − p).
* **Token card:** hover over any token to see the top 5 tokens the model considered, with probabilities.
* **Fork:** click an alternative to regenerate the reply from that point with that token. Branches
  (from forks, regenerations and edits) are kept and can be browsed with ‹ ›.
* **Confidence river:** a sparkline of p for each token under every reply; click it to inspect a token.
* **Live sampling controls:** temperature, top-k, top-p, repetition penalty, length, seed.
* **Compare tab:** one prompt, two or three models side by side, same settings and seed, each column
  with the lens, token card and stats (tokens/s, mean probability, perplexity).
* **Formatted replies:** headings, lists, bold, italic and code are rendered without losing the
  per-token features; every visible character still belongs to its token. Toggle in Settings.
* **Runs in your browser:** Settings › Engine › "In your browser" runs AmirGPT Chat or Base with
  WebGPU (fp16, 249 MB, cached after the first load; WebAssembly fallback). Greedy output matches the
  server token for token. Export the models first with `.venv/bin/python -m web.export_onnx`.
* **Export / import:** a conversation as Markdown or JSON, all conversations as JSON, and JSON import.
* **Mobile layout:** drawer sidebar, bottom-sheet settings, tap for the token card.
* **Lab tab:** your training curves against Karpathy's reference lines (OpenAI GPT-2 val loss 3.2924,
  HellaSwag 0.2955, GPT-3 124M 0.337).

## Reproduce the training

```bash
# on the cluster (ssh alias "sbu"), from ~/AmirGPT
bash cluster/setup_env.sh                       # Python env (torch 2.6 + CUDA 12.4)
sbatch cluster/prep_data.sbatch                 # FineWeb-Edu 10B -> 100 shards (+ HellaSwag)
sbatch cluster/train.sbatch                     # pretraining, 8x V100, about 6 h (resumable)
sbatch cluster/prep_sft.sbatch                  # smol-smoltalk -> chat SFT data
sbatch cluster/sft.sbatch                       # chat fine-tuning
sbatch cluster/final_eval.sbatch                # fp32 val loss + HellaSwag, ours vs OpenAI GPT-2
sbatch cluster/prep_dpo.sbatch && sbatch cluster/dpo.sbatch   # optional: preference tuning (DPO)
sbatch cluster/bench_scaling.sbatch             # 1/2/4/8-GPU throughput (for docs/report.md)
sbatch cluster/data_stats.sbatch                # dataset statistics (for docs/report.md)
# on the Mac
bash cluster/pull_results.sh                    # logs + checkpoints -> logs_from_cluster/, checkpoints/
```

Single machine: `python -m gpt2.train_gpt2` · multi-GPU: `torchrun --standalone --nproc_per_node=8 -m gpt2.train_gpt2`.
All of the video's hyperparameters are the defaults. See `python -m gpt2.train_gpt2 --help`.

## Tests

```bash
.venv/bin/python -m pytest tests -q            # Python
(cd web/frontend && npm test)                   # markdown renderer
(cd web/frontend && bash scripts/parity.sh fp32 fp16)   # in-browser engine vs server (needs the ONNX export)
```

GitHub Actions runs the Python tests and the frontend build on every push, and redeploys the live demo
(`.github/workflows/pages.yml`, `web/build_static_site.py`).

## Repository layout notes

`logs_from_cluster/` holds the full training logs of the run above (they feed the app's Lab tab).
Not included (all can be regenerated with the scripts above): model checkpoints (≈ 500 MB each; see the release) and their ONNX exports,
FineWeb-Edu shards, SFT data, the HellaSwag file (downloaded by `gpt2/hellaswag.py`),
and the video transcript (copyrighted; the timestamps in [docs/walkthrough.md](docs/walkthrough.md)
were checked against it).

## Acknowledgements

* Andrej Karpathy, ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) and
  [build-nanogpt](https://github.com/karpathy/build-nanogpt). The model and training code follow them step by step.
* Data: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT),
  [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) for chat fine-tuning,
  [HellaSwag](https://rowanzellers.com/hellaswag/) for evaluation.
* Compute: the university AI cluster (8× V100).
