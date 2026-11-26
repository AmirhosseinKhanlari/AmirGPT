"""
AmirGPT web server: FastAPI backend that streams tokens (with probabilities and top
alternatives) from our GPT-2 implementation, and serves the React frontend.

Models (each loaded lazily into OUR gpt2.model.GPT class):
  amirgpt-chat   checkpoints/amirgpt_chat.pt   SFT chat model (chat mode)
  amirgpt-base   checkpoints/amirgpt_base.pt   our from-scratch pretrained GPT-2 (completion mode)
  openai-gpt2    OpenAI GPT-2 124M weights loaded into our class (completion mode, for comparison)

Endpoints:
  GET  /api/models    -> models + availability
  POST /api/generate  -> text/event-stream of {"type":"token",...} events, then {"type":"done",...}
  GET  /api/metrics   -> training curves (log/metrics.jsonl, log_sft/metrics.jsonl) + reference baselines
  GET  /models/...    -> checkpoints/onnx/ (ONNX exports for the in-browser WebGPU engine; read-only)
  GET  /              -> built frontend (web/frontend/dist)

Run:  python -m web.server   (or ./run.sh)
"""

import codecs
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import tiktoken
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gpt2.model import GPT, load_checkpoint
from sft.chat_format import BANNED_WHILE_ANSWERING, END, EOT, SPECIAL_NAMES, render_prompt

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = Path(os.environ.get("AMIRGPT_CKPT_DIR", ROOT / "checkpoints"))
DIST = ROOT / "web" / "frontend" / "dist"
enc = tiktoken.get_encoding("gpt2")

DEVICE = os.environ.get("AMIRGPT_DEVICE") or ("cuda" if torch.cuda.is_available() else
                                              "mps" if torch.backends.mps.is_available() else "cpu")

MODELS = {
    "amirgpt-chat": {
        "name": "AmirGPT Chat", "mode": "chat", "path": CKPT_DIR / "amirgpt_chat.pt",
        "description": "Our GPT-2 124M, pretrained from scratch on 10B FineWeb-Edu tokens, then chat fine-tuned (SFT).",
    },
    "amirgpt-base": {
        "name": "AmirGPT Base", "mode": "complete", "path": CKPT_DIR / "amirgpt_base.pt",
        "description": "Our from-scratch pretrained GPT-2 124M before chat tuning. It continues text, it does not chat.",
    },
    "openai-gpt2": {
        "name": "OpenAI GPT-2", "mode": "complete", "path": None,
        "description": "OpenAI's original GPT-2 124M weights loaded into our model code, for comparison.",
    },
    "amirgpt-chat-dpo": {
        "name": "AmirGPT Chat + DPO", "mode": "chat", "path": CKPT_DIR / "amirgpt_chat_dpo.pt",
        "description": "Experimental: AmirGPT Chat after preference tuning (DPO). It prefers the better answer more often, "
                       "but its replies are longer and slightly more repetitive, so AmirGPT Chat stays the default.",
    },
}

# reference numbers, with sources (see gpt2/hellaswag.py and build-nanogpt/play.ipynb)
BASELINES = {
    "val_loss_openai_gpt2_124M": {"value": 3.2924, "source": "build-nanogpt play.ipynb: OpenAI GPT-2 (124M) checkpoint val loss on FineWeb-Edu val (Karpathy's shard)"},
    "val_loss_openai_gpt2_124M_ours": {"value": 3.2793, "source": "measured here (gpt2/eval_val.py, fp32): OpenAI GPT-2 (124M) on OUR val shard, same first 10,485,760 tokens our training evaluates"},
    "hella_openai_gpt2_124M": {"value": 0.2955, "source": "video [3:38:06] / build-nanogpt hellaswag.py: acc_norm 0.2955 (this eval; we reproduced 2967/10042)"},
    "hella_openai_gpt3_124M": {"value": 0.337, "source": "build-nanogpt play.ipynb: GPT-3 paper 124M HellaSwag"},
    "hella_random": {"value": 0.25, "source": "4-way multiple choice"},
}

_loaded = {}
_locks = {k: threading.Lock() for k in MODELS}
_load_lock = threading.Lock()


def available(key):
    m = MODELS[key]
    return m["path"] is None or Path(m["path"]).exists()


def get_model(key):
    if key not in MODELS:
        raise HTTPException(404, f"unknown model {key}")
    if not available(key):
        raise HTTPException(409, f"{MODELS[key]['name']} is not available yet (missing {MODELS[key]['path']})")
    with _load_lock:
        if key not in _loaded:
            if MODELS[key]["path"] is None:
                model = GPT.from_pretrained("gpt2")
            else:
                model, _ = load_checkpoint(str(MODELS[key]["path"]))
            model.to(DEVICE).eval()
            _loaded[key] = model
    return _loaded[key]


def token_text(i):
    if i in SPECIAL_NAMES:
        return SPECIAL_NAMES[i]
    if i >= enc.n_vocab:
        return f"<|unused_{i}|>"
    return enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")


class GenParams(BaseModel):
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=50304)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    max_new_tokens: int = Field(256, ge=1, le=1000)
    seed: Optional[int] = None


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = "amirgpt-chat"
    messages: List[Message] = []          # chat mode
    prompt: str = ""                      # completion mode
    system_prompt: Optional[str] = None
    prefix_ids: List[int] = []            # already-generated tokens to continue from (forking)
    params: GenParams = GenParams()


app = FastAPI(title="AmirGPT")


@app.get("/api/models")
def list_models():
    return [{"id": k, "name": v["name"], "mode": v["mode"], "description": v["description"],
             "available": available(k), "loaded": k in _loaded} for k, v in MODELS.items()]


@app.get("/api/health")
def health():
    return {"ok": True, "device": DEVICE}


def _build_prompt(req: GenerateRequest, model: GPT):
    block = model.config.block_size
    mode = MODELS[req.model]["mode"]
    reserve = min(req.params.max_new_tokens, block // 2)
    if mode == "chat":
        msgs = [m.model_dump() for m in req.messages if m.role in ("user", "assistant", "system")]
        if not msgs or msgs[-1]["role"] != "user":
            raise HTTPException(400, "chat mode needs the conversation to end with a user message")
        ids = render_prompt(msgs, system_prompt=req.system_prompt, max_tokens=block, reserve=reserve + len(req.prefix_ids))
        stop, banned = (END,), BANNED_WHILE_ANSWERING
    else:
        ids = [EOT] + enc.encode_ordinary(req.prompt)
        budget = block - reserve - len(req.prefix_ids)
        ids = ids[-budget:] if len(ids) > budget else ids
        stop, banned = (EOT,), tuple(range(50257, model.config.vocab_size))
    if any(not (0 <= t < model.config.vocab_size) for t in req.prefix_ids):
        raise HTTPException(400, "invalid prefix token id")
    return ids + list(req.prefix_ids), stop, banned


@app.post("/api/generate")
def generate(req: GenerateRequest):
    model = get_model(req.model)
    ids, stop, banned = _build_prompt(req, model)
    p = req.params
    gen = torch.Generator().manual_seed(p.seed if p.seed is not None else int(time.time_ns() % (2 ** 31)))

    def events():
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        # feed the prefix through the decoder so a fork in the middle of a multi-byte char stays correct
        for t in req.prefix_ids:
            if t < enc.n_vocab:
                decoder.decode(enc.decode_single_token_bytes(t))
        n, t0, reason = 0, time.time(), "length"
        idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        steps = model.stream(idx, max_new_tokens=p.max_new_tokens, temperature=p.temperature,
                             top_k=p.top_k, top_p=p.top_p, repetition_penalty=p.repetition_penalty,
                             stop_tokens=stop, banned_tokens=banned, generator=gen)
        while True:
            # Hold the model lock for one decoding step only, never across a `yield`: on a client
            # disconnect (Stop) Starlette abandons this generator without closing it, so a lock held
            # across the yield would block this model until the garbage collector freed it.
            with _locks[req.model]:
                tok = next(steps, None)
            if tok is None:
                break
            if tok["stop"]:
                reason = "stop"
                break
            n += 1
            text = decoder.decode(enc.decode_single_token_bytes(tok["id"])) if tok["id"] < enc.n_vocab else ""
            yield "data: " + json.dumps({
                "type": "token", "id": tok["id"], "text": text, "prob": tok["prob"],
                "alternatives": [{"id": i, "text": token_text(i), "prob": pr} for i, pr in tok["alternatives"]],
            }) + "\n\n"
        tail = decoder.decode(b"", final=True)
        dt = max(time.time() - t0, 1e-6)
        yield "data: " + json.dumps({"type": "done", "reason": reason, "tail": tail, "tokens": n,
                                     "tokens_per_sec": n / dt, "prompt_tokens": len(ids)}) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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


@app.get("/api/metrics")
def metrics():
    logs = Path(os.environ.get("AMIRGPT_LOG_DIR", ROOT / "logs_from_cluster"))
    pre = _read_jsonl(logs / "pretrain" / "metrics.jsonl")
    sft = _read_jsonl(logs / "sft" / "metrics.jsonl")

    def series(rows, kind, every=1):
        pts = [(r["step"], r["value"]) for r in rows if r.get("kind") == kind]
        return pts[::every] if every > 1 else pts

    cfg_path = logs / "pretrain" / "run_config.json"
    return JSONResponse({
        "pretrain": {"train": series(pre, "train", 10), "val": series(pre, "val"), "hella": series(pre, "hella"),
                     "samples": [r for r in pre if r.get("kind") == "samples"][-5:],
                     "config": json.load(open(cfg_path)) if cfg_path.exists() else None},
        "sft": {"train": series(sft, "train"), "val": series(sft, "val"),
                "samples": [r for r in sft if r.get("kind") == "samples"][-3:]},
        "baselines": BASELINES,
    })


# ONNX exports (web/export_onnx.py) for the in-browser engine (web/frontend/src/webgpu):
# read-only static files, .onnx / .json only, confined to checkpoints/onnx/.
ONNX_DIR = CKPT_DIR / "onnx"
_MODEL_FILE_TYPES = {".onnx": "application/octet-stream", ".json": "application/json"}


@app.get("/models/{path:path}")
def model_file(path: str):
    root = ONNX_DIR.resolve()
    f = (root / path).resolve()
    if root not in f.parents or f.suffix not in _MODEL_FILE_TYPES or not f.is_file():
        raise HTTPException(404, "not found")
    # no-cache = revalidate (ETag / Last-Modified -> 304); the engine itself keeps the model in Cache Storage
    return FileResponse(f, media_type=_MODEL_FILE_TYPES[f.suffix], headers={"Cache-Control": "no-cache"})


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
