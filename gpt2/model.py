"""
GPT-2 (124M) model, written from scratch following Karpathy's "Let's reproduce GPT-2 (124M)".

Architecture is IDENTICAL to OpenAI GPT-2 so that the released weights load into it
(video [0:04:23] exploring the HuggingFace GPT-2 checkpoint, [0:14:56] the GPT nn.Module skeleton):
  * token + learned position embeddings (wte, wpe)
  * 12 pre-LayerNorm transformer blocks: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))
  * a final LayerNorm (ln_f) and an lm_head whose weight is TIED to wte (video [1:10:44])
  * GELU with the tanh approximation, as used by the original GPT-2 (video [0:21:06])
  * residual projections initialised with std 0.02 / sqrt(2 * n_layer) (video [1:16:24])
  * F.scaled_dot_product_attention -> flash / memory-efficient attention (video [2:00:26])

Improvements over the video's code (none change the math of the forward pass):
  * KV-cache for O(1)-per-token incremental decoding (used by the chat server).
  * `generate()` / `stream()` with temperature, top-k, top-p (nucleus) and repetition
    penalty, returning per-token probabilities and top alternatives (used by the
    AmirGPT UI to visualise the model's confidence and to fork generations).
  * the leftover causal-mask buffer from before flash attention is not created
    (README errata), and `from_pretrained` validates every key and shape.
  * `load_checkpoint` restores our training checkpoints regardless of whether they
    were saved from a torch.compile'd / DDP-wrapped model.
"""

import inspect
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, kv_cache=None):
        """
        x: (B, T, C). kv_cache: optional (k, v) of shape (B, nh, T_past, hs) holding
        keys/values of previous positions. Returns (y, (k, v)) where (k, v) now
        include the current positions; callers that do not use caching ignore it.
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        else:
            past_k, past_v = kv_cache
            T_past = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            if T == 1:
                # a single new query may attend to every cached position
                y = F.scaled_dot_product_attention(q, k, v)
            else:
                # several new queries: query i (absolute position T_past+i) sees keys <= T_past+i
                mask = torch.ones(T, T_past + T, dtype=torch.bool, device=x.device).tril(diagonal=T_past)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, (k, v)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        a, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing scheme (input embedding == output classifier), video [1:10:44]
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # residual stream variance grows with depth: 2 residual adds per block
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_caches=None, loss_mask=None):
        """
        idx: (B, T) token ids. targets: (B, T) next-token ids (optional).
        kv_caches: optional list (one per layer) of cached (k, v) for incremental decoding.
        loss_mask: optional (B, T) {0,1} mask; loss is averaged only where mask==1 (used by SFT).
        Returns (logits, loss) or, if kv_caches is given, (logits, loss, new_kv_caches).
        """
        B, T = idx.size()
        T_past = 0 if kv_caches is None else kv_caches[0][0].size(2)
        assert T_past + T <= self.config.block_size, \
            f"Cannot forward sequence of length {T_past + T}, block size is only {self.config.block_size}"
        pos = torch.arange(T_past, T_past + T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        new_caches = []
        for i, block in enumerate(self.transformer.h):
            x, c = block(x, None if kv_caches is None else kv_caches[i])
            new_caches.append(c)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            if loss_mask is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                m = loss_mask.view(-1).to(per_tok.dtype)
                loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)
        if kv_caches is not None:
            return logits, loss, new_caches
        return logits, loss

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, model_type, vocab_size=None):
        """Load OpenAI GPT-2 weights from HuggingFace into our module (video [0:25:07], used to verify our code).
        vocab_size: optionally pad the vocabulary (e.g. 50304); new rows start as the mean
        embedding (an "average token") until trained."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # use the fully-qualified Hub id: a bare "gpt2" would resolve to our local gpt2/ package dir
        model_hf = GPT2LMHeadModel.from_pretrained(f"openai-community/{model_type}")
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")]
        # OpenAI checkpoints use a "Conv1D" module; we use vanilla Linear -> transpose these
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        assert set(sd_keys_hf) == set(sd_keys), f"key names differ: {set(sd_keys_hf) ^ set(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        if vocab_size is not None and vocab_size > model.config.vocab_size:
            model.resize_vocab(vocab_size)
        return model

    def resize_vocab(self, new_vocab_size):
        """Pad the (tied) embedding / lm_head to new_vocab_size. New rows = mean embedding."""
        old = self.transformer.wte.weight.data
        if new_vocab_size <= old.size(0):
            return
        # new rows = mean of the existing embeddings (zeros would give logit 0, which beats
        # GPT-2's typical strongly-negative logits and makes untrained tokens dominate)
        new_w = old.mean(dim=0, keepdim=True).repeat(new_vocab_size, 1)
        new_w[: old.size(0)] = old
        self.transformer.wte = nn.Embedding(new_vocab_size, old.size(1)).to(old.device)
        self.lm_head = nn.Linear(old.size(1), new_vocab_size, bias=False).to(old.device)
        self.transformer.wte.weight = self.lm_head.weight
        with torch.no_grad():
            self.lm_head.weight.copy_(new_w)
        self.config.vocab_size = new_vocab_size

    # ---------------------------------------------------------------- optimizer
    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95), verbose=True):
        """AdamW; 2D params (matmuls, embeddings) decayed, biases & LayerNorms not (video [2:28:53]); fused kernel [2:31:43]."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
            print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused)

    # --------------------------------------------------------------- generation
    @torch.no_grad()
    def stream(self, idx, max_new_tokens=128, temperature=1.0, top_k=50, top_p=1.0,
               repetition_penalty=1.0, stop_tokens=(), n_alternatives=5, generator=None,
               banned_tokens=()):
        """
        Incremental (KV-cached) sampling for a single sequence. idx: (1, T) LongTensor.
        Yields one dict per generated token:
          {"id", "prob", "alternatives": [(id, prob), ...]}
        `prob`/`alternatives` are from the distribution after temperature / penalties but
        before top-k/top-p truncation, i.e. what the model "believes".
        Stops at a stop token (which is yielded, flagged "stop": True), at max_new_tokens,
        or when the context window is full.
        """
        assert idx.dim() == 2 and idx.size(0) == 1
        block = self.config.block_size
        if idx.size(1) >= block:
            idx = idx[:, -(block - 1):]
        logits, _, caches = self(idx, kv_caches=[
            (torch.empty(1, self.config.n_head, 0, self.config.n_embd // self.config.n_head,
                         device=idx.device, dtype=self.lm_head.weight.dtype),) * 2
            for _ in range(self.config.n_layer)])
        seen = idx[0].tolist()
        stop_set = set(stop_tokens)
        for _ in range(max_new_tokens):
            logits_last = logits[0, -1, :].float()
            if banned_tokens:
                logits_last[list(banned_tokens)] = -float("inf")
            if repetition_penalty != 1.0 and seen:
                ids = torch.tensor(sorted(set(seen)), device=logits_last.device)
                vals = logits_last[ids]
                logits_last[ids] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            if temperature <= 0:
                probs = F.softmax(logits_last, dim=-1)
                next_id = int(torch.argmax(probs))
            else:
                probs = F.softmax(logits_last / temperature, dim=-1)
                next_id = int(sample_from_probs(probs, top_k, top_p, generator))
            top_p_vals, top_i = torch.topk(probs, n_alternatives)
            alts = [(int(i), float(p)) for i, p in zip(top_i.tolist(), top_p_vals.tolist())]
            is_stop = next_id in stop_set
            yield {"id": next_id, "prob": float(probs[next_id]), "alternatives": alts, "stop": is_stop}
            if is_stop:
                return
            seen.append(next_id)
            if caches[0][0].size(2) >= block:
                return  # context window full
            logits, _, caches = self(torch.tensor([[next_id]], device=idx.device), kv_caches=caches)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, **kw):
        """Convenience wrapper: returns the list of generated token ids."""
        return [t["id"] for t in self.stream(idx, max_new_tokens=max_new_tokens, **kw)]


def sample_from_probs(probs, top_k=50, top_p=1.0, generator=None):
    """Sample one index from a 1D probability vector with optional top-k and top-p truncation."""
    if top_k is not None and 0 < top_k < probs.size(-1):
        vals, idxs = torch.topk(probs, top_k)
    else:
        vals, idxs = torch.sort(probs, descending=True)
    if top_p is not None and top_p < 1.0:
        cum = torch.cumsum(vals, dim=-1)
        # keep the smallest prefix whose cumulative probability >= top_p (always keep >= 1 token)
        keep = (cum - vals) < top_p
        keep[0] = True
        vals, idxs = vals[keep], idxs[keep]
    # multinomial does not require normalised input; sample on CPU for determinism across devices
    choice = torch.multinomial(vals.cpu(), 1, generator=generator)
    return idxs[choice.to(idxs.device)].item()


def load_checkpoint(path, map_location="cpu"):
    """Load one of our checkpoints -> (model, checkpoint_dict). Handles torch.compile/DDP prefixes."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = GPTConfig(**cfg)
    model = GPT(cfg)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # lm_head.weight is tied to wte.weight; either key may be absent depending on how it was saved
    missing = [m for m in missing if m not in ("lm_head.weight", "transformer.wte.weight")]
    assert not missing and not unexpected, f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
    return model, ckpt


def config_to_dict(cfg):
    return asdict(cfg)
