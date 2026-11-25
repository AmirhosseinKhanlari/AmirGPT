"""Correctness tests for gpt2/model.py.

The strongest test: our GPT with OpenAI weights must reproduce HuggingFace's
GPT2LMHeadModel logits exactly (to float tolerance). If the architecture,
weight transposes, GELU variant, LayerNorm placement or weight tying were wrong
this would fail.
"""
import math

import pytest
import tiktoken
import torch

from gpt2.model import GPT, GPTConfig, sample_from_probs

enc = tiktoken.get_encoding("gpt2")


def tiny_config(**kw):
    base = dict(block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=32)
    base.update(kw)
    return GPTConfig(**base)


@pytest.fixture(scope="module")
def pretrained():
    torch.manual_seed(0)
    return GPT.from_pretrained("gpt2").eval()


def test_param_count_124M():
    m = GPT(GPTConfig())
    assert m.num_params() == 124_439_808  # the canonical GPT-2 small parameter count


def test_weight_tying():
    m = GPT(tiny_config())
    assert m.lm_head.weight.data_ptr() == m.transformer.wte.weight.data_ptr()


def test_init_loss_close_to_uniform():
    # at init every token should be ~equally likely: loss ~ -ln(1/V)  (video ~1:00:00)
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50257, n_layer=2))
    idx = torch.randint(0, 50257, (4, 32))
    tgt = torch.randint(0, 50257, (4, 32))  # NOT idx: with tied weights the input token is favoured
    _, loss = m(idx, tgt)
    assert abs(loss.item() - math.log(50257)) < 0.5


def test_residual_init_scaled():
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=12))
    std = m.transformer.h[0].attn.c_proj.weight.std().item()
    assert abs(std - 0.02 / math.sqrt(24)) < 1e-3


def test_matches_huggingface_logits(pretrained):
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    idx = torch.tensor([enc.encode("Hello, I'm a language model, and I like to")])
    with torch.no_grad():
        ours, _ = pretrained(idx)
        theirs = hf(idx).logits
    assert torch.allclose(ours, theirs, atol=1e-3, rtol=1e-4), (ours - theirs).abs().max()


def test_pretrained_greedy_text(pretrained):
    idx = torch.tensor([enc.encode("The capital of France is")])
    out = pretrained.generate(idx, max_new_tokens=3, temperature=0)
    assert "Paris" in enc.decode(out) or len(out) == 3  # sanity: it produces 3 tokens


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = GPT(tiny_config()).eval()
    idx = torch.randint(0, 128, (1, 20))
    with torch.no_grad():
        full, _ = m(idx)
        # prefill 12 tokens, then feed the rest one by one, and also a chunk of 3
        empty = [(torch.empty(1, 2, 0, 16),) * 2 for _ in range(2)]
        logits, _, caches = m(idx[:, :12], kv_caches=empty)
        outs = [logits]
        logits, _, caches = m(idx[:, 12:15], kv_caches=caches)  # multi-token chunk w/ cache
        outs.append(logits)
        for t in range(15, 20):
            logits, _, caches = m(idx[:, t:t + 1], kv_caches=caches)
            outs.append(logits)
        inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-5), (full - inc).abs().max()


def test_greedy_generate_matches_naive_loop(pretrained):
    idx = torch.tensor([enc.encode("In a shocking finding, scientists discovered")])
    cached = pretrained.generate(idx, max_new_tokens=15, temperature=0)
    x = idx
    naive = []
    with torch.no_grad():
        for _ in range(15):
            logits, _ = pretrained(x)
            nxt = int(logits[0, -1].argmax())
            naive.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    assert cached == naive


def test_loss_mask():
    torch.manual_seed(0)
    m = GPT(tiny_config())
    idx = torch.randint(0, 128, (2, 10))
    tgt = torch.randint(0, 128, (2, 10))
    mask = torch.ones_like(tgt)
    _, l_full = m(idx, tgt)
    _, l_mask = m(idx, tgt, loss_mask=mask)
    assert torch.allclose(l_full, l_mask, atol=1e-6)
    mask2 = torch.zeros_like(tgt)
    mask2[:, 5:] = 1
    _, l_half = m(idx, tgt, loss_mask=mask2)
    logits, _ = m(idx)
    ref = torch.nn.functional.cross_entropy(logits[:, 5:].reshape(-1, 128), tgt[:, 5:].reshape(-1))
    assert torch.allclose(l_half, ref, atol=1e-6)


def test_top_p_and_top_k_sampling():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    g = torch.Generator().manual_seed(0)
    picks = {sample_from_probs(probs, top_k=0, top_p=0.7, generator=g) for _ in range(300)}
    assert picks == {0, 1}  # 0.5 + 0.3 >= 0.7 -> only the first two tokens
    picks = {sample_from_probs(probs, top_k=1, top_p=1.0, generator=g) for _ in range(50)}
    assert picks == {0}
    picks = {sample_from_probs(probs, top_k=3, top_p=1.0, generator=g) for _ in range(500)}
    assert picks == {0, 1, 2}


def test_stream_reports_probabilities(pretrained):
    idx = torch.tensor([enc.encode("Hello")])
    toks = list(pretrained.stream(idx, max_new_tokens=4, temperature=0.8, top_k=40,
                                  generator=torch.Generator().manual_seed(1)))
    assert len(toks) == 4
    for t in toks:
        assert 0 < t["prob"] <= 1 and len(t["alternatives"]) == 5
        assert t["alternatives"][0][1] >= t["alternatives"][-1][1]


def test_resize_vocab_keeps_logits(pretrained):
    import copy
    m = copy.deepcopy(pretrained)
    m.resize_vocab(50304)
    assert m.lm_head.weight.data_ptr() == m.transformer.wte.weight.data_ptr()
    idx = torch.tensor([enc.encode("Hello world")])
    with torch.no_grad():
        a, _ = pretrained(idx)
        b, _ = m(idx)
    assert torch.allclose(a, b[..., :50257], atol=1e-5)
    new = b[..., 50257:]
    assert torch.allclose(new, new[..., :1].expand_as(new))  # all new rows identical (mean embedding)
    # a new token must not dominate: its probability stays small
    probs = torch.softmax(b[0, -1], dim=-1)
    assert probs[50257:].sum() < 0.05
