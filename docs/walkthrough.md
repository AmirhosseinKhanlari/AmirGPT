# From the video to the code: step-by-step walkthrough

This maps each step of Andrej Karpathy's **"Let's reproduce GPT-2 (124M)"**
(https://www.youtube.com/watch?v=l8pRSuU81PU, 4:01:25) to the code in this repo.
Every timestamp below was checked against the video's full transcript (not redistributed
in this repo because it is copyrighted).

## Part 1: The GPT-2 model (`gpt2/model.py`)

| Video | Topic | Where in the code |
|---|---|---|
| [0:04:23] | Look at OpenAI's GPT-2 checkpoint through HuggingFace Transformers | `GPT.from_pretrained` |
| [0:14:56] | Skeleton of the GPT `nn.Module`: `wte`, `wpe`, `h`, `ln_f`, `lm_head` | `GPT.__init__` |
| [0:21:06] | GELU with `approximate='tanh'`, as GPT-2 used | `MLP` |
| [0:25:07] | Load weights, transposing OpenAI's Conv1D matrices | `from_pretrained` (`transposed` list) |
| [0:11:44], [0:37:27] | Sampling with top-k = 50 | `GPT.stream`, `sample_from_probs` |
| [1:10:44] | Weight tying: `wte` is also the output classifier (saves ~30% of params, [1:13:12]) | `self.transformer.wte.weight = self.lm_head.weight` |
| [1:16:24] | Scale residual-projection init by 1/√(2·n_layer) | `_init_weights` + `NANOGPT_SCALE_INIT` |
| [2:00:26] | Flash attention via `F.scaled_dot_product_attention` | `CausalSelfAttention.forward` |

**Check:** `tests/test_model.py::test_matches_huggingface_logits`. Our module with OpenAI's
weights gives the same logits as HuggingFace's `GPT2LMHeadModel`. The parameter count is exactly 124,439,808.

## Part 2: Training loop and speed (`gpt2/train_gpt2.py`, `gpt2/data.py`)

| Video | Topic | Where |
|---|---|---|
| [0:46:06] | First dataset: tiny Shakespeare | (replaced by FineWeb-Edu in Part 3, as in the video) |
| [1:00:55] | Overfit a single batch as a sanity check | `tests/test_training.py::test_overfit_single_batch` |
| [1:02:01] | `DataLoaderLite` | `gpt2/data.py` (`DataLoaderLite = ShardedDataLoader`) |
| [1:25:27] | TF32 matmuls | `torch.set_float32_matmul_precision("high")` |
| [1:36:25] | Measure tokens/sec | training log line `tok/sec` |
| [1:40:30], [1:43:10] | bf16 with `torch.autocast` | `autocast()`; fp16 + GradScaler on V100 (no bf16 there) |
| [1:48:46] | `torch.compile` (2.3× in the video, [1:49:51]) | `--compile 1` |
| [2:07:26] | "Nice numbers": vocab 50257 → 50304 | `--vocab_size 50304` |
| [2:17:22] | Clip the global gradient norm at 1.0 | `clip_grad_norm_(…, 1.0)` |
| [2:21:12], [2:23:42] | Warmup + cosine LR schedule, max_lr 6e-4 | `get_lr` |
| [2:28:53], [2:31:43] | Weight decay 0.1 on 2D params only; fused AdamW | `GPT.configure_optimizers` |
| [2:35:28], [2:36:41] | 0.5M-token batch through gradient accumulation | `total_batch_size=524288`, `grad_accum_steps` |
| [2:47:18], [2:54:57] | DDP with `torchrun` | DDP block in `main()` |

## Part 3: Real data and evaluation

| Video | Topic | Where |
|---|---|---|
| [3:12:38] | FineWeb-Edu (sample-10BT) | `gpt2/fineweb.py` |
| [3:15:47] | Each document starts with `<\|endoftext\|>` (50256) | `tokenize()` in `fineweb.py` |
| [3:16:52] | 100M-token uint16 shards; shard 0 is validation | `fineweb.py`, `ShardedDataLoader` |
| [3:19:57] | 715 warmup steps = 375M tokens / 524,288 | `--warmup_steps 715` |
| [3:23:58] | Validation loss | eval block in `train_gpt2.py`, `gpt2/eval_val.py` |
| [3:28:35] | HellaSwag evaluation | `gpt2/hellaswag.py` |
| [3:38:06] | "our script reports 2955": the GPT-2 124M number to beat | we reproduce **2967/10042 = 0.2955** |

## Part 4: What Karpathy says should come next (our improvements)

| Video | What he says | What we did |
|---|---|---|
| [3:46:58], [3:50:05] | The loss shows periodicity because the data is never shuffled; "permute the documents around in every single shard on every single new epoch and possibly even permute the shards" | `--shuffle 1`: documents shuffled within each shard and shard order shuffled per epoch; validation stays fixed |
| [3:55:46] | To chat with it, "swap out the data set" for user/assistant conversations, "fill in the user tokens and sample the assistant tokens" | `sft/`: smol-smoltalk conversations, loss only on assistant tokens, role tokens in the spare vocab slots 50257–50260 |

## Changes forced by our hardware or setup (8× V100 32GB)

* **fp16 instead of bf16:** V100 has no bf16. We use fp16 autocast + `GradScaler`.
* **Micro-batch 32 instead of 64:** 32 × 1024 × 8 GPUs × 2 accumulation steps = the same 524,288-token batch.
  Validation uses 40 steps instead of 20, so it reads exactly the same first 10,485,760 val tokens.
* **HellaSwag source:** the original GitHub repository was taken down (DMCA, 2026-09-14), so the
  author's HuggingFace copy (`Rowan/hellaswag`) is used. It is checked by reproducing 0.2955 exactly.
* **FineWeb download:** the `datasets`/Xet transfer stalled on the cluster, so the 14 parquet files are fetched
  over HTTPS (with size checks) and read in the same order `load_dataset` uses.
