# Model card: AmirGPT

AmirGPT is a 124M-parameter GPT-2 language model trained from scratch for the **Big Data** course
(Fall 2026), following Andrej Karpathy's ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU).
There are two checkpoints:

| Checkpoint | File | What it does |
|---|---|---|
| AmirGPT Base | `amirgpt_base.pt` | Continues text. Pretrained only. |
| AmirGPT Chat | `amirgpt_chat.pt` | Answers in a conversation. AmirGPT Base after supervised chat fine-tuning (SFT). |

## Model details

* Architecture: GPT-2 small. 12 layers, 12 heads, 768-dim embeddings, 1,024-token context,
  GELU (tanh), pre-LayerNorm, token embedding tied to the output layer. 124,439,808 parameters
  at GPT-2's 50,257-token vocabulary, the same as OpenAI's. The checkpoints pad the embedding table
  to 50,304 rows for speed, as in the video, which makes 124,475,904.
* Tokenizer: OpenAI GPT-2 BPE (`tiktoken` "gpt2"). The chat model uses four extra tokens in the
  padded rows: `<|user|>` 50257, `<|assistant|>` 50258, `<|system|>` 50259, `<|end|>` 50260.
* Code: `gpt2/model.py`. Checkpoints load with `gpt2.model.load_checkpoint(path)`.

## Training

**Pretraining (AmirGPT Base).** FineWeb-Edu sample-10BT, tokenized into 100 shards of 100M tokens
(shard 0 is the validation split). 19,073 steps of 524,288 tokens (≈ 10.0B tokens), AdamW
(β = 0.9, 0.95, weight decay 0.1 on matrices only), max learning rate 6e-4 with 715 warmup steps
and cosine decay to 6e-5, gradient clipping at 1.0. These are the video's settings. Differences:
fp16 with loss scaling instead of bf16 (V100 GPUs have no bf16), and documents and shards are
shuffled every epoch (suggested in the video at [3:50:05]). 8× NVIDIA V100 32GB, 6 h 24 min.

**Chat fine-tuning (AmirGPT Chat).** Two epochs over
[smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) mixed with 1,000 short
conversations that teach the model its name and what it is: 455,391 conversations in total after
filtering, truncated to 1,024 tokens. The loss counts assistant tokens only. Learning rate 1e-4, 100 warmup steps,
cosine decay. 8× V100, 35 min.

Compute: about 51 V100-hours for pretraining and about 5 V100-hours for chat fine-tuning.

**Preference tuning (experimental, not the default).** `amirgpt_chat_dpo.pt` is AmirGPT Chat after one
epoch of DPO (β = 0.1, learning rate 1e-6) on 51,635 UltraFeedback preference pairs, 5 min on 8× V100.
It prefers the better answer on 68.7% of 1,660 held-out pairs, but its replies are longer and slightly
more repetitive, so it is not the default chat model (see the README).

## Evaluation

Independent fp32 evaluation (`cluster/final_eval.sbatch`), same data and code for both models:

| Metric | OpenAI GPT-2 124M | AmirGPT Base |
|---|---|---|
| Val loss, first 10,485,760 tokens of FineWeb-Edu val shard | 3.2793 | **3.0693** |
| HellaSwag acc_norm (10,042 validation examples) | 0.2955 | **0.3047** |

The val loss favours AmirGPT partly because it was trained on the same distribution
(FineWeb-Edu) that the val shard comes from; OpenAI GPT-2 was trained on WebText. HellaSwag is
the fairer comparison of the two. AmirGPT Chat's loss on held-out assistant tokens is 1.351.

## Intended use

Teaching and demonstration: showing how a small transformer language model is built, trained
and evaluated, and how a chat model differs from a base model. It is not suited to any task that
needs correct information.

## Limitations and risks

* At 124M parameters, the model often states false facts confidently, loses track of long
  conversations, and repeats itself.
* It has no safety training beyond what is in the chat data, and it can produce biased or
  inappropriate text that reflects its web training data.
* English only, in practice. FineWeb-Edu is filtered toward educational English web pages.
* The context is limited to 1,024 tokens; the app drops the oldest turns of long chats.

## Acknowledgements

Andrej Karpathy (video and [build-nanogpt](https://github.com/karpathy/build-nanogpt)),
HuggingFace (FineWeb-Edu, smol-smoltalk), Zellers et al. (HellaSwag), OpenAI (GPT-2 tokenizer and
reference weights).
