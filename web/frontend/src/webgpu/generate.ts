// The generation loop of the in-browser engine: a port of web/server.py /api/generate +
// GPT.stream, independent of onnxruntime (it only needs a Runner), so it can be tested in Node.

import type { DoneEvent, GenerateRequest, Tok } from "../api";
import { buildPrompt } from "./prompt";
import { randomSeed, Rng, sampleStep } from "./sampling";
import type { Tokenizer } from "./tokenizer";
import { PyUtf8Decoder } from "./utf8";

/** A KV-cached forward pass over one sequence. */
export interface Runner {
  /** Feeds new tokens (all positions after the cache); returns the logits of the last one. */
  forward(ids: number[]): Promise<ArrayLike<number>>;
  /** Number of cached positions. */
  cacheLength(): number;
  /** Frees the cache (GPU buffers). The runner must not be used afterwards. */
  release(): void;
}

export type EngineModelInfo = { mode: "chat" | "complete"; blockSize: number; vocabSize: number };

const TOKENIZER_VOCAB = 50257;

export function abortError(): Error {
  return new DOMException("The generation was aborted.", "AbortError");
}

function checkParams(req: GenerateRequest): void {
  const p = req.params;
  const bad = (name: string, cond: boolean) => {
    if (!cond) throw new Error(`invalid parameter ${name}`);
  };
  bad("temperature", p.temperature >= 0 && p.temperature <= 2);
  bad("top_k", Number.isInteger(p.top_k) && p.top_k >= 0 && p.top_k <= 50304);
  bad("top_p", p.top_p > 0 && p.top_p <= 1);
  bad("repetition_penalty", p.repetition_penalty >= 1 && p.repetition_penalty <= 2);
  bad("max_new_tokens", Number.isInteger(p.max_new_tokens) && p.max_new_tokens >= 1 && p.max_new_tokens <= 1000);
}

/**
 * Same contract as api.ts streamGenerate: calls onToken for every generated token (the stop
 * token is not reported), resolves with the done event, rejects with an AbortError on abort.
 */
export async function generate(
  newRunner: () => Runner,
  tok: Tokenizer,
  info: EngineModelInfo,
  req: GenerateRequest,
  onToken: (t: Tok) => void,
  signal?: AbortSignal,
): Promise<DoneEvent> {
  checkParams(req);
  const p = req.params;
  const { ids, stop, banned } = buildPrompt(tok, req, info.mode, info.blockSize, info.vocabSize);
  const rng = new Rng(p.seed ?? randomSeed());

  // incremental UTF-8 decoding (codecs incremental decoder, errors="replace"); feed the prefix
  // through it so a fork in the middle of a multi-byte character stays correct
  const decoder = new PyUtf8Decoder();
  for (const t of req.prefix_ids ?? []) {
    if (t < TOKENIZER_VOCAB) decoder.decode(tok.tokenBytes(t));
  }

  const block = info.blockSize;
  let idx = ids;
  if (idx.length >= block) idx = idx.slice(-(block - 1)); // GPT.stream truncation
  const seen = [...idx];
  const stopSet = new Set(stop);
  let n = 0;
  let reason: "stop" | "length" = "length";
  const t0 = performance.now();

  if (signal?.aborted) throw abortError();
  const runner = newRunner();
  try {
    let logits = await runner.forward(idx);
    for (let step = 0; step < p.max_new_tokens; step++) {
      if (signal?.aborted) throw abortError();
      const s = sampleStep(logits, p, banned, seen, rng);
      if (stopSet.has(s.id)) {
        reason = "stop";
        break;
      }
      n++;
      const text = s.id < TOKENIZER_VOCAB ? decoder.decode(tok.tokenBytes(s.id)) : "";
      onToken({
        id: s.id,
        text,
        prob: s.prob,
        alternatives: s.alternatives.map(([i, pr]) => ({ id: i, text: tok.tokenText(i), prob: pr })),
      });
      seen.push(s.id);
      if (step === p.max_new_tokens - 1) break; // GPT.stream: no forward pass after the last token
      if (runner.cacheLength() >= block) break; // context window full
      logits = await runner.forward([s.id]);
    }
  } finally {
    runner.release();
  }
  const tail = decoder.flush();
  const dt = Math.max((performance.now() - t0) / 1000, 1e-6);
  return { type: "done", reason, tail, tokens: n, tokens_per_sec: n / dt, prompt_tokens: ids.length };
}
