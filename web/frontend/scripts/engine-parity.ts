// Node-side parity test for the in-browser engine (no browser needed).
// Runs the REAL engine modules (src/webgpu/*: tokenizer, prompt building, sampling, generation
// loop, ORT runner) with onnxruntime-web's wasm backend in Node, against the reference JSON made
// by parity_reference.py from the real server code. Run via scripts/parity.sh.
//
//   node engine-parity.mjs <repo root> <reference.json> [fp32] [fp16]

import * as fs from "node:fs";
import * as path from "node:path";
import * as ort from "onnxruntime-web";
import type { GenerateRequest, Tok } from "../src/api";
import { generate, type Runner } from "../src/webgpu/generate";
import { bannedWhileAnswering, buildPrompt } from "../src/webgpu/prompt";
import { processLogits, Rng, sampleFromProbs, softmax, truncateCandidates } from "../src/webgpu/sampling";
import { createRunner } from "../src/webgpu/session";
import { createTokenizer } from "../src/webgpu/tokenizer";
import { PyUtf8Decoder } from "../src/webgpu/utf8";
import { createHash } from "node:crypto";
import type { ModelMeta } from "../src/webgpu/types";

const [root, refPath, ...variantArgs] = process.argv.slice(2);
const variants = variantArgs.length ? variantArgs : ["fp32", "fp16"];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ref: any = JSON.parse(fs.readFileSync(refPath, "utf8"));

let failures = 0;
function check(name: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
}
const eq = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

const tok = createTokenizer();

// ------------------------------------------------------------------ tokenizer
{
  let bad = 0;
  for (const c of ref.encode) {
    const got = tok.encodeOrdinary(c.text);
    if (!eq(got, c.ids)) {
      bad++;
      console.log("  encode mismatch:", JSON.stringify(c.text), "\n   py:", c.ids.join(","), "\n   js:", got.join(","));
    }
  }
  check("tokenizer encode_ordinary parity (tricky multilingual/emoji/whitespace strings)", bad === 0, `${ref.encode.length - bad}/${ref.encode.length} strings identical`);

  let badText = 0;
  for (let i = 0; i < ref.token_text.length; i++) {
    if (tok.tokenText(i) !== ref.token_text[i]) {
      if (badText < 5) console.log("  token_text mismatch", i, JSON.stringify(tok.tokenText(i)), JSON.stringify(ref.token_text[i]));
      badText++;
    }
  }
  check("token_text parity for every id 0..50303", badText === 0, `${ref.token_text.length - badText}/${ref.token_text.length}`);

  let badInc = 0;
  let nChunks = 0;
  for (const c of ref.incremental) {
    const d = new PyUtf8Decoder();
    for (const t of c.ids.slice(0, c.prefix_n)) d.decode(tok.tokenBytes(t));
    const chunks = c.ids.slice(c.prefix_n).map((t: number) => d.decode(tok.tokenBytes(t)));
    const tail = d.flush();
    nChunks += chunks.length;
    if (!eq(chunks, c.chunks) || tail !== c.tail) badInc++;
  }
  check("incremental UTF-8 decoding parity on token sequences (Python codecs errors=replace vs PyUtf8Decoder)", badInc === 0,
    `${ref.incremental.length - badInc}/${ref.incremental.length} sequences, ${nChunks} chunks`);
  check("BANNED_WHILE_ANSWERING parity", eq(bannedWhileAnswering(50304), ref.banned_while_answering));
}

// ------------------------------------------------------------------ prompt building
{
  let bad = 0;
  for (const c of ref.prompts) {
    const req = c.request as GenerateRequest;
    req.params = { ...req.params, seed: req.params.seed ?? null };
    const mode = req.model === "amirgpt-chat" ? "chat" : "complete";
    const got = buildPrompt(tok, req, mode, 1024, 50304);
    const ok = eq(got.ids, c.ids) && eq(got.stop, c.stop) && got.banned.length === c.banned_len && eq(got.banned.slice(0, 8), c.banned_head);
    if (!ok) {
      bad++;
      console.log("  prompt mismatch:", JSON.stringify(req).slice(0, 120), got.ids.length, c.ids.length);
    }
  }
  check("_build_prompt / render_prompt parity (truncation, system prompt, prefix_ids edge cases)", bad === 0, `${ref.prompts.length - bad}/${ref.prompts.length} cases`);
  let threw = "";
  try {
    buildPrompt(tok, { model: "amirgpt-chat", messages: [{ role: "assistant", content: "x" }], params: ref.prompts[0].request.params }, "chat", 1024, 50304);
  } catch (e) {
    threw = (e as Error).message;
  }
  check("chat prompt not ending in a user message is rejected", threw === "chat mode needs the conversation to end with a user message");
}

// ------------------------------------------------------------------ sampling (pure JS)
{
  // multinomial correctness: empirical frequencies vs the kept, renormalised probabilities
  const V = 1000;
  const probs = new Float64Array(V);
  const r0 = new Rng(7);
  let s = 0;
  for (let i = 0; i < V; i++) {
    probs[i] = Math.pow(r0.random(), 4);
    s += probs[i];
  }
  for (let i = 0; i < V; i++) probs[i] /= s;
  const cand = truncateCandidates(probs, 50, 0.95);
  const kept = cand.reduce((a, i) => a + probs[i], 0);
  const counts = new Map<number, number>();
  const rng = new Rng(123);
  const N = 400000;
  for (let n = 0; n < N; n++) {
    const id = sampleFromProbs(probs, 50, 0.95, rng);
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  let chi2 = 0;
  let outside = 0;
  for (const [id, c] of counts) if (!cand.includes(id)) outside += c;
  for (const i of cand) {
    const e = (N * probs[i]) / kept;
    chi2 += ((counts.get(i) ?? 0) - e) ** 2 / e;
  }
  // chi-square with (|cand| - 1) dof: mean = dof, sd = sqrt(2 dof); accept < dof + 5 sd
  const dof = cand.length - 1;
  check("sampleFromProbs draws only from the top-k/top-p set, with the right frequencies", outside === 0 && chi2 < dof + 5 * Math.sqrt(2 * dof),
    `candidates=${cand.length}, chi2=${chi2.toFixed(1)} (dof ${dof}), draws outside set=${outside}`);
  // top-k/top-p candidate sets on PyTorch's exact probability vector (pure logic, no model numerics)
  {
    const c = ref.truncation;
    const probsPy = Float64Array.from(c.probs as number[]);
    let bad = 0;
    const sizes: string[] = [];
    for (const cfg of c.configs) {
      const cand2 = truncateCandidates(probsPy, cfg.top_k ?? 0, cfg.top_p ?? 1);
      // no truncation at all = a full sort: the tail is a sea of exact-zero (banned) ties in arbitrary order
      const full = !(cfg.top_k > 0 && cfg.top_k < 50304) && cfg.top_p >= 1;
      const ok = cand2.length === cfg.n && (full || eq(cand2.slice(0, 2000), cfg.ids));
      sizes.push(`k=${cfg.top_k},p=${cfg.top_p}: ${cand2.length}${ok ? "" : ` (py ${cfg.n})`}`);
      if (!ok) bad++;
    }
    check("top-k/top-p candidate sets identical to sample_from_probs (on PyTorch's probabilities)", bad === 0, sizes.join("; "));
  }
  const a = new Rng(42), b = new Rng(42), c2 = new Rng(43);
  const sa = Array.from({ length: 5 }, () => a.nextU32()), sb = Array.from({ length: 5 }, () => b.nextU32()), sc = Array.from({ length: 5 }, () => c2.nextU32());
  check("seeded PRNG is deterministic per seed", eq(sa, sb) && !eq(sa, sc));
}

function utf8Exhaustive() {
  // exhaustive byte-level check: every 4-byte combination over a set of boundary bytes, fed in chunks
  const ex = ref.utf8_exhaustive;
  const parts: string[] = [];
  let count = 0;
  for (const b1 of ex.first) for (const b2 of ex.rest) for (const b3 of ex.rest) for (const b4 of ex.rest) {
    const seq = [b1, b2, b3, b4];
    for (const split of ex.splits as number[][]) {
      const d = new PyUtf8Decoder();
      const chunks: string[] = [];
      let at = 0;
      for (const len of split) {
        chunks.push(d.decode(Uint8Array.from(seq.slice(at, at + len))));
        at += len;
      }
      parts.push(chunks.join("\ue000") + "\ue001" + d.flush());
      count++;
    }
  }
  const digest = createHash("sha256").update(parts.join("\ue002"), "utf8").digest("hex");
  check("UTF-8 decoder exhaustive parity with CPython (4-byte combinations x chunkings)", digest === ex.sha256 && count === ex.count, `${count} sequences`);
}

// ------------------------------------------------------------------ model runs (onnxruntime-web wasm in Node)
function loadMeta(name: string): ModelMeta {
  return JSON.parse(fs.readFileSync(path.join(root, "checkpoints/onnx", name, "meta.json"), "utf8"));
}

async function session(name: string) {
  const meta = loadMeta(name);
  const bytes = fs.readFileSync(path.join(root, "checkpoints/onnx", name, meta.file));
  const t0 = performance.now();
  const s = await ort.InferenceSession.create(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), {
    executionProviders: ["wasm"], graphOptimizationLevel: "all",
  });
  console.log(`  [${name}] session created in ${((performance.now() - t0) / 1000).toFixed(1)}s (wasm, numThreads=${ort.env.wasm.numThreads})`);
  return { meta, s };
}

async function runGen(meta: ModelMeta, s: ort.InferenceSession, req: GenerateRequest, signal?: AbortSignal) {
  const toks: Tok[] = [];
  const done = await generate(() => createRunner(ort, s, meta) as Runner, tok,
    { mode: meta.mode, blockSize: meta.block_size, vocabSize: meta.vocab_size }, req, (t) => toks.push(t), signal);
  return { toks, done };
}

const sessions = new Map<string, { meta: ModelMeta; s: ort.InferenceSession }>();
async function get(name: string) {
  if (!sessions.has(name)) sessions.set(name, await session(name));
  return sessions.get(name)!;
}

for (const dt of variants) {
  console.log(`\n=== variant ${dt} ===`);
  for (const [caseName, c] of Object.entries(ref.generate) as [string, any][]) {
    const name = `${c.request.model}-${dt}`;
    const { meta, s } = await get(name);
    const req = { ...c.request, params: { ...c.request.params } } as GenerateRequest;
    const t0 = performance.now();
    const { toks, done } = await runGen(meta, s, req);
    const secs = (performance.now() - t0) / 1000;
    const refIds = c.tokens.map((t: any) => t.id);
    const gotIds = toks.map((t) => t.id);
    const idsOk = eq(gotIds, refIds);
    const textOk = eq(toks.map((t) => t.text), c.tokens.map((t: any) => t.text)) && done.tail === c.done.tail;
    let maxProb = 0, maxAlt = 0, altIdsOk = true;
    for (let i = 0; i < Math.min(toks.length, c.tokens.length); i++) {
      maxProb = Math.max(maxProb, Math.abs(toks[i].prob - c.tokens[i].prob));
      const ra = c.tokens[i].alternatives, ga = toks[i].alternatives;
      if (!eq(ga.map((x) => x.id), ra.map((x: any) => x.id)) || !eq(ga.map((x) => x.text), ra.map((x: any) => x.text))) altIdsOk = false;
      for (let j = 0; j < Math.min(ga.length, ra.length); j++) maxAlt = Math.max(maxAlt, Math.abs(ga[j].prob - ra[j].prob));
    }
    const doneOk = done.reason === c.done.reason && done.tokens === c.done.tokens && done.prompt_tokens === c.done.prompt_tokens;
    const firstDiv = gotIds.findIndex((id, i) => id !== refIds[i]);
    check(`[${dt}] ${caseName}: token ids identical to server greedy`, idsOk,
      `${gotIds.length} vs ${refIds.length} tokens${idsOk ? "" : `, first divergence at ${firstDiv}`}; ${(done.tokens / secs).toFixed(1)} tok/s incl. prefill`);
    check(`[${dt}] ${caseName}: streamed text + tail identical`, textOk);
    check(`[${dt}] ${caseName}: done event (reason/tokens/prompt_tokens)`, doneOk, `${done.reason}/${done.tokens}/${done.prompt_tokens}`);
    check(`[${dt}] ${caseName}: alternatives (ids, texts) identical; prob diffs`, altIdsOk,
      `max |prob diff| ${maxProb.toExponential(2)}, max |alt prob diff| ${maxAlt.toExponential(2)}`);
    if (caseName === "chat_hello") console.log(`  text: ${JSON.stringify(toks.map((t) => t.text).join("") + done.tail)}`);
    if (caseName === "base_completion") console.log(`  text: ${JSON.stringify(toks.map((t) => t.text).join("").slice(0, 160))}...`);
  }

  // first sampled token: the distribution (alternatives) is deterministic even though the RNG differs
  {
    const c = ref.sampled_first;
    const { meta, s } = await get(`amirgpt-chat-${dt}`);
    const { toks } = await runGen(meta, s, { ...c.request, params: { ...c.request.params, max_new_tokens: 1 } });
    const ra = c.tokens[0].alternatives, ga = toks[0].alternatives;
    const d = Math.max(...ga.map((x, j) => Math.abs(x.prob - ra[j].prob)));
    check(`[${dt}] temperature 0.8 + rep 1.1: first-token top-5 distribution matches server`, eq(ga.map((x) => x.id), ra.map((x: any) => x.id)), `max |prob diff| ${d.toExponential(2)}`);
  }

  // the engine's probability distribution (ONNX logits -> penalties -> softmax) vs PyTorch
  {
    const c = ref.truncation;
    const { meta, s } = await get(`amirgpt-chat-${dt}`);
    const req = c.request as GenerateRequest;
    const { ids, banned } = buildPrompt(tok, req, "chat", 1024, 50304);
    const runner = createRunner(ort, s, meta);
    const logits = Float64Array.from(await runner.forward(ids));
    runner.release();
    processLogits(logits, banned, ids, c.repetition_penalty);
    const probs = softmax(logits, c.temperature);
    const pyTop = c.top5.map((x: number[]) => x[0]);
    const jsTop = Array.from(truncateCandidates(probs, 5, 1));
    let maxDiff = 0;
    for (let i = 0; i < probs.length; i++) maxDiff = Math.max(maxDiff, Math.abs(probs[i] - c.probs[i]));
    check(`[${dt}] engine distribution vs PyTorch on the same prompt (temperature 0.8, rep 1.1)`, eq(jsTop, pyTop),
      `top-5 identical; max |prob diff| over all 50304 ids ${maxDiff.toExponential(2)}`);
  }

  // seeded sampling is reproducible; abort works
  {
    const { meta, s } = await get(`amirgpt-chat-${dt}`);
    const req: GenerateRequest = { model: "amirgpt-chat", messages: [{ role: "user", content: "Tell me a story." }],
      params: { temperature: 0.8, top_k: 50, top_p: 0.95, repetition_penalty: 1.1, max_new_tokens: 30, seed: 42 } };
    const a = await runGen(meta, s, req);
    const b = await runGen(meta, s, req);
    const c3 = await runGen(meta, s, { ...req, params: { ...req.params, seed: 43 } });
    check(`[${dt}] same seed -> same sampled tokens; different seed -> different`, eq(a.toks.map((t) => t.id), b.toks.map((t) => t.id)) && !eq(a.toks.map((t) => t.id), c3.toks.map((t) => t.id)),
      JSON.stringify(a.toks.map((t) => t.text).join("") + a.done.tail).slice(0, 100));
    const ctrl = new AbortController();
    let n = 0, name = "";
    try {
      await generate(() => createRunner(ort, s, meta), tok, { mode: "chat", blockSize: 1024, vocabSize: 50304 },
        { ...req, params: { ...req.params, max_new_tokens: 200 } }, () => { if (++n === 3) ctrl.abort(); }, ctrl.signal);
    } catch (e) {
      name = (e as Error).name;
    }
    check(`[${dt}] AbortSignal stops generation with an AbortError`, name === "AbortError" && n === 3, `tokens before abort: ${n}`);
  }
  for (const [k, v] of sessions) {
    await v.s.release();
    sessions.delete(k);
  }
}

utf8Exhaustive();

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
