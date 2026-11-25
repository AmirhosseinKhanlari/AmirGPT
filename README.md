# AmirGPT

> Course project for **Big Data** (Fall 2026).

GPT-2 (124M) **reproduced from scratch**, following Andrej Karpathy's
["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU)
([build-nanogpt](https://github.com/karpathy/build-nanogpt)). It is pretrained on 10B tokens of
FineWeb-Edu, **chat fine-tuned**, and served in a web app that shows how the model makes each choice.

* `gpt2/`: the model, data pipeline, training loop and evals, written step by step as in the video
  ([docs/walkthrough.md](docs/walkthrough.md) maps each video timestamp to the code)
* `sft/`: chat fine-tuning (the step Karpathy describes at [3:55:46] but does not cover)
* `web/`: the AmirGPT app (FastAPI + React)
* `cluster/`: SLURM scripts for the Stony Brook AI cluster (8× V100)
* `tests/`: 25 tests (model vs HuggingFace logits, KV-cache, data loader, schedule, SFT, resume)

## Results

**Status: training in progress.** Pretraining (19,073 steps, about 10B tokens) is running on 8× V100,
and chat fine-tuning follows it. Final validation loss and HellaSwag numbers will be added here
when the runs finish. Until then, the targets to compare against are:

| Metric | OpenAI GPT-2 124M (reference) |
|---|---|
| Val loss on our FineWeb-Edu val shard | 3.2793 (measured with `gpt2/eval_val.py`) |
| HellaSwag acc_norm | 0.2955 (video [3:38:06]) |

## Correctness checks (all passing)

| Check | Expected | Ours |
|---|---|---|
| Parameter count | 124,439,808 | 124,439,808 |
| Logits with OpenAI weights vs HuggingFace `GPT2LMHeadModel` | identical | match (atol 1e-3) |
| HellaSwag, our code + OpenAI GPT-2 124M weights | 0.2955 (video [3:38:06]) | **2967/10042 = 0.2955** |
| Loss at initialization | ≈ ln(50257) = 10.82 | 10.98 (step 0) |
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

What's new in the UI:

* **Confidence lens:** each token is tinted by how unsure the model was (1 − p).
* **Token card:** hover over any token to see the top 5 tokens the model considered, with probabilities.
* **Fork:** click an alternative to regenerate the reply from that point with that token. Branches
  (from forks, regenerations and edits) are kept and can be browsed with ‹ ›.
* **Confidence river:** a sparkline of p for each token under every reply; click it to inspect a token.
* **Live sampling controls:** temperature, top-k, top-p, repetition penalty, length, seed.
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
# on the Mac
bash cluster/pull_results.sh                    # logs + checkpoints -> logs_from_cluster/, checkpoints/
```

Single machine: `python -m gpt2.train_gpt2` · multi-GPU: `torchrun --standalone --nproc_per_node=8 -m gpt2.train_gpt2`.
All of the video's hyperparameters are the defaults. See `python -m gpt2.train_gpt2 --help`.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

## Repository layout notes

Not included in the repo (all can be regenerated with the scripts above): model checkpoints,
FineWeb-Edu shards, SFT data, the HellaSwag file (downloaded by `gpt2/hellaswag.py`), training logs,
and the video transcript (copyrighted; the timestamps in [docs/walkthrough.md](docs/walkthrough.md)
were checked against it).

## Acknowledgements

* Andrej Karpathy, ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) and
  [build-nanogpt](https://github.com/karpathy/build-nanogpt). The model and training code follow them step by step.
* Data: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT),
  [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) for chat fine-tuning,
  [HellaSwag](https://rowanzellers.com/hellaswag/) for evaluation.
* Compute: the university AI cluster (8× V100).
