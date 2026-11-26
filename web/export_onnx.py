"""
Export an AmirGPT checkpoint to ONNX (with a KV cache) for in-browser inference
(onnxruntime-web, WebGPU / wasm). gpt2/model.py is NOT modified: an export-friendly
wrapper (OnnxGPT) re-implements the forward pass with plain ops and reads the SAME
weights from a loaded gpt2.model.GPT.

Graph interface (one file per variant):
  inputs   input_ids   int64  [1, T]                               T >= 1 new tokens
           past_kv     kv     [n_layer, 2, 1, n_head, P, head_dim]  P >= 0 cached positions
  outputs  logits      fp32   [1, vocab_size]                      logits of the LAST new position only
           present_kv  kv     [n_layer, 2, 1, n_head, P+T, head_dim]
  T and P are dynamic; P may be 0 (first call). P + T must be <= block_size.
  Query i (absolute position P+i) attends to keys 0..P+i, so prefill-with-past is exact.

Variants:
  fp32  everything fp32 (kv = float32)
  fp16  weights fp16 and matmuls in fp16; LayerNorm, GELU, softmax and the residual
        stream stay fp32 (mixed precision); kv = float16; logits are cast to fp32.
        On WebGPU this needs the "shader-f16" adapter feature.

Output:  checkpoints/onnx/<model>-<dtype>/{model.onnx, meta.json}

Usage (from the repo root):
  .venv/bin/python -m web.export_onnx                       # chat + base, fp32 + fp16, verify
  .venv/bin/python -m web.export_onnx --models amirgpt-chat --dtypes fp16
  .venv/bin/python -m web.export_onnx --verify-only         # re-run the onnxruntime checks
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2.model import GPT, load_checkpoint
from sft.chat_format import (ASSISTANT, BANNED_WHILE_ANSWERING, END, EOT, SYSTEM, USER, enc,
                             render_prompt)

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = CKPT_DIR / "onnx"

MODELS = {
    "amirgpt-chat": {"path": CKPT_DIR / "amirgpt_chat.pt", "mode": "chat"},
    "amirgpt-base": {"path": CKPT_DIR / "amirgpt_base.pt", "mode": "complete"},
}
OPSET = 17  # LayerNormalization is opset 17; well supported by onnxruntime-web (WebGPU + wasm)


class OnnxGPT(nn.Module):
    """Export-friendly re-implementation of GPT.forward with a stacked KV cache.

    Mathematically identical to gpt2.model.GPT (pre-LN blocks, tanh-GELU, tied lm_head,
    causal attention with the same "query i sees keys <= P+i" rule as the kv_cache path).
    """

    def __init__(self, gpt: GPT, dtype: str):
        super().__init__()
        cfg = gpt.config
        self.n_layer, self.n_head, self.n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.wdt = torch.float16 if dtype == "fp16" else torch.float32  # weight / matmul dtype
        wdt = self.wdt

        def w(t):  # matmul weights / biases / embeddings -> wdt
            return nn.Parameter(t.detach().clone().to(wdt), requires_grad=False)

        def f32(t):  # LayerNorm params stay fp32
            return nn.Parameter(t.detach().clone().float(), requires_grad=False)

        tr = gpt.transformer
        self.wte = w(tr.wte.weight)  # [V, C], also the (tied) lm_head weight
        self.wpe = w(tr.wpe.weight)
        self.ln_f_w, self.ln_f_b = f32(tr.ln_f.weight), f32(tr.ln_f.bias)
        self.blocks = nn.ModuleList()
        for b in tr.h:
            m = nn.Module()
            m.ln1_w, m.ln1_b = f32(b.ln_1.weight), f32(b.ln_1.bias)
            m.attn_w, m.attn_b = w(b.attn.c_attn.weight), w(b.attn.c_attn.bias)
            m.proj_w, m.proj_b = w(b.attn.c_proj.weight), w(b.attn.c_proj.bias)
            m.ln2_w, m.ln2_b = f32(b.ln_2.weight), f32(b.ln_2.bias)
            m.fc_w, m.fc_b = w(b.mlp.c_fc.weight), w(b.mlp.c_fc.bias)
            m.fc2_w, m.fc2_b = w(b.mlp.c_proj.weight), w(b.mlp.c_proj.bias)
            self.blocks.append(m)

    @staticmethod
    def _gelu_tanh(x):
        # nn.GELU(approximate="tanh"), written out so it exports to basic ops on any opset
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

    def _ln(self, x, w, b):
        return F.layer_norm(x, (self.n_embd,), w, b, 1e-5)

    def forward(self, input_ids, past_kv):
        wdt, H, D, C = self.wdt, self.n_head, self.head_dim, self.n_embd
        T = input_ids.shape[1]
        P = past_kv.shape[4]
        pos = torch.arange(T, dtype=torch.long) + P                        # absolute positions [T]
        x = (F.embedding(input_ids, self.wte).float() + F.embedding(pos, self.wpe).float())  # [1,T,C] fp32
        k_pos = torch.arange(P + T, dtype=torch.long)                     # [P+T]
        allowed = k_pos.unsqueeze(0) <= pos.unsqueeze(1)                  # [T, P+T] causal w/ past
        neg_inf = torch.tensor(float("-inf"), dtype=torch.float32)
        presents = []
        for i, m in enumerate(self.blocks):
            h = self._ln(x, m.ln1_w, m.ln1_b).to(wdt)
            qkv = F.linear(h, m.attn_w, m.attn_b)                         # [1,T,3C]
            q, k, v = qkv.split(C, dim=2)
            q = q.reshape(1, T, H, D).transpose(1, 2)                     # [1,H,T,D]
            k = k.reshape(1, T, H, D).transpose(1, 2)
            v = v.reshape(1, T, H, D).transpose(1, 2)
            k = torch.cat([past_kv[i, 0], k], dim=2)                      # [1,H,P+T,D]
            v = torch.cat([past_kv[i, 1], v], dim=2)
            presents.append(torch.stack([k, v], dim=0))                   # [2,1,H,P+T,D]
            att = torch.matmul(q * (1.0 / math.sqrt(D)), k.transpose(2, 3)).float()  # [1,H,T,P+T]
            att = torch.where(allowed, att, neg_inf)
            att = torch.softmax(att, dim=-1).to(wdt)
            y = torch.matmul(att, v)                                      # [1,H,T,D]
            y = y.transpose(1, 2).reshape(1, T, C)
            x = x + F.linear(y, m.proj_w, m.proj_b).float()
            h = self._ln(x, m.ln2_w, m.ln2_b).to(wdt)
            h = self._gelu_tanh(F.linear(h, m.fc_w, m.fc_b).float()).to(wdt)
            x = x + F.linear(h, m.fc2_w, m.fc2_b).float()
        x_last = x[:, -1, :]                                              # [1,C]: only the last position
        x_last = self._ln(x_last, self.ln_f_w, self.ln_f_b).to(wdt)
        logits = F.linear(x_last, self.wte).float()                       # [1,V] (Gemm transB, shares wte)
        present_kv = torch.stack(presents, dim=0)                         # [L,2,1,H,P+T,D]
        return logits, present_kv


def untie_fix(m, wte):
    """The exporter turns the tied lm_head (F.linear(x, wte)) into MatMul(x, wte^T) with a
    constant-folded, TRANSPOSED COPY of the 50304x768 embedding (+154 MB fp32). Replace that
    node by Gemm(x, wte, transB=1) so the embedding Gather and the lm_head share one tensor."""
    from onnx import helper, numpy_helper
    inits = {t.name: t for t in m.graph.initializer}
    assert "wte" in inits, "expected the embedding initializer to be called 'wte'"
    V, C = wte.shape
    for idx, node in enumerate(m.graph.node):
        if node.op_type != "MatMul" or node.input[1] not in inits:
            continue
        t = inits[node.input[1]]
        if tuple(t.dims) != (C, V):
            continue
        arr = numpy_helper.to_array(t)
        assert np.array_equal(arr, wte.T), "transposed lm_head weight does not equal wte^T"
        gemm = helper.make_node("Gemm", [node.input[0], "wte"], list(node.output), name="lm_head_gemm", transB=1)
        m.graph.node.remove(node)
        m.graph.node.insert(idx, gemm)
        m.graph.initializer.remove(t)
        return
    raise RuntimeError("lm_head MatMul with a transposed wte copy not found")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kv_np_dtype(dtype):
    return np.float16 if dtype == "fp16" else np.float32


def export(model_id, dtype):
    spec = MODELS[model_id]
    gpt, ckpt = load_checkpoint(str(spec["path"]))
    gpt.eval()
    cfg = gpt.config
    wrapper = OnnxGPT(gpt, dtype).eval()
    L, H, D = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
    out_dir = OUT_DIR / f"{model_id}-{dtype}"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    # trace with P > 0 and T > 1 so neither dimension is specialised
    kvt = torch.float16 if dtype == "fp16" else torch.float32
    ids = torch.randint(0, 50257, (1, 5), dtype=torch.long)
    past = torch.zeros(L, 2, 1, H, 3, D, dtype=kvt)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (ids, past), str(onnx_path), dynamo=False, opset_version=OPSET,
            input_names=["input_ids", "past_kv"], output_names=["logits", "present_kv"],
            dynamic_axes={"input_ids": {1: "T"}, "past_kv": {4: "P"}, "present_kv": {4: "P_plus_T"}},
            do_constant_folding=True, external_data=False,
        )
    import onnx
    m = onnx.load(str(onnx_path))
    untie_fix(m, gpt.transformer.wte.weight.detach().to(kvt).numpy())
    onnx.checker.check_model(m)
    onnx.save(m, str(onnx_path))
    n_init = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
    meta = {
        "name": f"{model_id}-{dtype}",
        "model_id": model_id,
        "mode": spec["mode"],
        "dtype": dtype,
        "kv_dtype": "float16" if dtype == "fp16" else "float32",
        "file": "model.onnx",
        "bytes": onnx_path.stat().st_size,
        "sha256": sha256(onnx_path),
        "opset": OPSET,
        "vocab_size": cfg.vocab_size,
        "tokenizer_vocab_size": enc.n_vocab,  # 50257: ids >= this have no BPE bytes
        "n_layer": L, "n_head": H, "head_dim": D, "n_embd": cfg.n_embd,
        "block_size": cfg.block_size,
        "inputs": {"input_ids": "int64 [1,T]", "past_kv": f"{'float16' if dtype == 'fp16' else 'float32'} [n_layer,2,1,n_head,P,head_dim]"},
        "outputs": {"logits": "float32 [1,vocab_size] (last position)", "present_kv": "kv_dtype [n_layer,2,1,n_head,P+T,head_dim]"},
        "special_tokens": {"endoftext": EOT, "user": USER, "assistant": ASSISTANT, "system": SYSTEM, "end": END},
        "source_checkpoint": spec["path"].name,
        "source_step": ckpt.get("step"),
        "param_count_in_onnx": n_init,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] {meta['name']}: {meta['bytes'] / 1e6:.1f} MB ({n_init:,} initializer values, "
          f"model params {gpt.num_params():,}) in {time.time() - t0:.1f}s")
    return gpt


# ---------------------------------------------------------------------------- verification
def ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_run(sess, ids, past):
    logits, present = sess.run(None, {"input_ids": np.asarray([ids], dtype=np.int64), "past_kv": past})
    return logits[0], present


def ort_greedy(sess, prompt, n, banned, L, H, D, kvdt):
    past = np.zeros((L, 2, 1, H, 0, D), dtype=kvdt)
    logits, past = ort_run(sess, prompt, past)
    out = []
    for _ in range(n):
        lg = logits.astype(np.float32).copy()
        lg[list(banned)] = -np.inf
        nxt = int(np.argmax(lg))
        out.append(nxt)
        if len(out) == n:
            break  # like GPT.stream: no forward pass after the last token
        logits, past = ort_run(sess, [nxt], past)
    return out


def verify(model_id, dtype, gpt=None):
    spec = MODELS[model_id]
    if gpt is None:
        gpt, _ = load_checkpoint(str(spec["path"]))
    gpt.eval()
    cfg = gpt.config
    L, H, D = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
    kvdt = kv_np_dtype(dtype)
    d = OUT_DIR / f"{model_id}-{dtype}"
    sess = ort_session(d / "model.onnx")
    if spec["mode"] == "chat":
        prompt = render_prompt([{"role": "user", "content": "What is the capital of France? Explain briefly."}])
        banned = BANNED_WHILE_ANSWERING
    else:
        prompt = [EOT] + enc.encode_ordinary("The history of the Roman Empire begins with")
        banned = tuple(range(50257, cfg.vocab_size))
    res = {"model": f"{model_id}-{dtype}", "prompt_tokens": len(prompt)}

    # (a) logits of the last prompt position: PyTorch full forward (no cache) vs ONNX (P=0)
    with torch.no_grad():
        ref_logits = gpt(torch.tensor([prompt]))[0][0, -1].float().numpy()
    logits0, present0 = ort_run(sess, prompt, np.zeros((L, 2, 1, H, 0, D), dtype=kvdt))
    res["a_logits_max_abs_diff"] = float(np.abs(logits0 - ref_logits).max())
    res["a_logits_ref_range"] = [float(ref_logits.min()), float(ref_logits.max())]
    res["a_argmax_match"] = bool(int(np.argmax(logits0)) == int(np.argmax(ref_logits)))

    # also compare the KV cache with PyTorch's kv_cache path
    with torch.no_grad():
        empty = [(torch.empty(1, H, 0, D),) * 2 for _ in range(L)]
        _, _, caches = gpt(torch.tensor([prompt]), kv_caches=empty)
    ref_kv = np.stack([np.stack([k.numpy(), v.numpy()]) for k, v in caches])
    res["a_present_kv_max_abs_diff"] = float(np.abs(present0.astype(np.float32) - ref_kv).max())

    # (b) 50-token greedy continuation with the cache, vs PyTorch GPT.stream(temperature=0)
    n = 50
    with torch.no_grad():
        ref_ids = [t["id"] for t in gpt.stream(torch.tensor([prompt]), max_new_tokens=n, temperature=0.0,
                                               banned_tokens=banned, stop_tokens=())]
    got_ids = ort_greedy(sess, prompt, n, banned, L, H, D, kvdt)
    first_div = next((i for i, (a, b) in enumerate(zip(ref_ids, got_ids)) if a != b), None)
    res["b_greedy_identical"] = ref_ids == got_ids
    res["b_greedy_first_divergence"] = first_div
    res["b_greedy_text_torch"] = enc.decode([i for i in ref_ids if i < enc.n_vocab])
    if ref_ids != got_ids:
        res["b_greedy_text_onnx"] = enc.decode([i for i in got_ids if i < enc.n_vocab])

    # (c) prefill-with-past: run the prompt in two chunks (P>0, T>1) and compare with one pass
    seq = prompt + ref_ids[:20]
    split = len(seq) // 3
    _, pkv = ort_run(sess, seq[:split], np.zeros((L, 2, 1, H, 0, D), dtype=kvdt))
    lc, pc = ort_run(sess, seq[split:], pkv)
    lf, pf = ort_run(sess, seq, np.zeros((L, 2, 1, H, 0, D), dtype=kvdt))
    with torch.no_grad():
        ref_seq = gpt(torch.tensor([seq]))[0][0, -1].float().numpy()
    res["c_chunks"] = [split, len(seq) - split]
    res["c_chunked_vs_full_onnx_logits_max_abs_diff"] = float(np.abs(lc - lf).max())
    res["c_chunked_vs_full_onnx_kv_max_abs_diff"] = float(np.abs(pc.astype(np.float32) - pf.astype(np.float32)).max())
    res["c_chunked_vs_torch_logits_max_abs_diff"] = float(np.abs(lc - ref_seq).max())
    # three-way split including a single-token middle chunk
    s1, s2 = split, split + 1
    _, p1 = ort_run(sess, seq[:s1], np.zeros((L, 2, 1, H, 0, D), dtype=kvdt))
    _, p2 = ort_run(sess, seq[s1:s2], p1)
    l3, _ = ort_run(sess, seq[s2:], p2)
    res["c_3chunk_vs_torch_logits_max_abs_diff"] = float(np.abs(l3 - ref_seq).max())

    # (d) long context: ~950-token prompt, greedy until the 1024-token window is full
    long_prompt = ([EOT] + enc.encode_ordinary((ROOT / "README.md").read_text()))[:950]
    n_long = cfg.block_size - len(long_prompt)
    with torch.no_grad():
        ref_long = [t["id"] for t in gpt.stream(torch.tensor([long_prompt]), max_new_tokens=n_long, temperature=0.0,
                                                banned_tokens=banned, stop_tokens=())]
    got_long = ort_greedy(sess, long_prompt, len(ref_long), banned, L, H, D, kvdt)
    res["d_long_prompt_tokens"] = len(long_prompt)
    res["d_long_generated"] = len(ref_long)
    res["d_long_greedy_identical"] = ref_long == got_long
    res["d_long_first_divergence"] = next((i for i, (a, b) in enumerate(zip(ref_long, got_long)) if a != b), None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"])
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    results = []
    for mid in args.models:
        gpt = None
        for dt in args.dtypes:
            if not args.verify_only:
                gpt = export(mid, dt)
            if not args.no_verify:
                r = verify(mid, dt, gpt)
                results.append(r)
                print(json.dumps(r, indent=2))
    if results:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "verification.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
