// Port of GPT.stream's per-token logic and sample_from_probs (gpt2/model.py).
//
// Semantics are identical to PyTorch: banned tokens -> -inf, repetition penalty on every id
// seen so far (prompt included; positive logits divided, negative multiplied), softmax with
// temperature (temperature <= 0 means greedy argmax over softmax(logits)), top-k then top-p
// (smallest prefix whose cumulative probability reaches top_p, always >= 1 token), sampling
// proportional to the kept (unnormalised) probabilities, and the reported prob/alternatives
// come from the distribution after temperature/penalties but BEFORE top-k/top-p truncation.
//
// Randomness: a seeded xoshiro128** PRNG. The same seed gives the same sample sequence in
// this engine, but NOT the same samples as torch.Generator on the server (bit-exact parity
// with torch's Mersenne-Twister multinomial is out of scope). Greedy decoding
// (temperature 0) is deterministic and matches the server token for token.

export type SamplingParams = {
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
};

/** xoshiro128** seeded through splitmix32; random() returns a double in [0, 1) with 53 random bits. */
export class Rng {
  private s = new Uint32Array(4);

  constructor(seed: number) {
    let x = (Math.floor(seed) >>> 0) ^ Math.imul(Math.floor(seed / 4294967296) >>> 0, 0x9e3779b9);
    for (let i = 0; i < 4; i++) {
      // splitmix32
      x = (x + 0x9e3779b9) | 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      this.s[i] = (z ^ (z >>> 16)) >>> 0;
    }
    if ((this.s[0] | this.s[1] | this.s[2] | this.s[3]) === 0) this.s[0] = 1;
  }

  nextU32(): number {
    const s = this.s;
    const r = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0;
    const t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return r;
  }

  random(): number {
    const hi = this.nextU32() >>> 5; // 27 bits
    const lo = this.nextU32() >>> 6; // 26 bits
    return (hi * 67108864 + lo) / 9007199254740992;
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

export function randomSeed(): number {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c?.getRandomValues) return c.getRandomValues(new Uint32Array(1))[0];
  return Math.floor(Math.random() * 4294967296);
}

/** Applies banned tokens and the repetition penalty in place (GPT.stream order). */
export function processLogits(logits: Float64Array, banned: readonly number[], seen: Iterable<number>, repetitionPenalty: number): void {
  for (const id of banned) logits[id] = -Infinity;
  if (repetitionPenalty !== 1.0) {
    for (const id of new Set(seen)) {
      const v = logits[id];
      logits[id] = v > 0 ? v / repetitionPenalty : v * repetitionPenalty;
    }
  }
}

/** softmax(logits / temperature) (temperature <= 0 -> plain softmax, as in GPT.stream). */
export function softmax(logits: Float64Array, temperature: number): Float64Array {
  const t = temperature <= 0 ? 1 : temperature;
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] / t > max) max = logits[i] / t;
  const out = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] / t - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function argmax(a: Float64Array): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i; // first index on ties, like torch.argmax
  return best;
}

/** Indices of all entries sorted by value, descending (ties: lower index first). */
export function argsortDesc(a: Float64Array): number[] {
  const idx = Array.from({ length: a.length }, (_, i) => i);
  idx.sort((i, j) => a[j] - a[i] || i - j);
  return idx;
}

/** Indices of the k largest entries, sorted descending (torch.topk). */
export function topK(a: Float64Array, k: number): number[] {
  const n = a.length;
  if (k >= n || k > 256) return argsortDesc(a).slice(0, Math.min(k, n));
  const idx: number[] = [];
  const val: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = a[i];
    if (idx.length === k && !(v > val[k - 1])) continue;
    let j = idx.length === k ? k - 1 : idx.length;
    while (j > 0 && val[j - 1] < v) {
      if (j < k) {
        idx[j] = idx[j - 1];
        val[j] = val[j - 1];
      }
      j--;
    }
    idx[j] = i;
    val[j] = v;
  }
  return idx;
}

/** The candidate set sample_from_probs draws from: top-k, then the top-p prefix. Sorted by prob, descending. */
export function truncateCandidates(probs: Float64Array, k: number, p: number): number[] {
  let idxs = k > 0 && k < probs.length ? topK(probs, k) : argsortDesc(probs);
  if (p < 1.0) {
    // keep = (cumsum - vals) < top_p, keep[0] = True  -> a prefix, since vals are sorted descending
    let excl = 0;
    let n = 0;
    while (n < idxs.length && (n === 0 || excl < p)) {
      excl += probs[idxs[n]];
      n++;
    }
    idxs = idxs.slice(0, n);
  }
  return idxs;
}

/** sample_from_probs: draw one index proportionally to the kept (unnormalised) probabilities. */
export function sampleFromProbs(probs: Float64Array, k: number, p: number, rng: Rng): number {
  const cand = truncateCandidates(probs, k, p);
  let total = 0;
  for (const i of cand) total += probs[i];
  const r = rng.random() * total;
  let acc = 0;
  let last = cand[0];
  for (const i of cand) {
    if (probs[i] <= 0) continue;
    acc += probs[i];
    last = i;
    if (r < acc) return i;
  }
  return last; // floating-point slack: fall back to the last positive-probability candidate
}

export type StepResult = { id: number; prob: number; alternatives: [number, number][] };

/** One GPT.stream step: logits of the last position -> (next id, its prob, top-n alternatives). */
export function sampleStep(
  rawLogits: ArrayLike<number>,
  params: SamplingParams,
  banned: readonly number[],
  seen: Iterable<number>,
  rng: Rng,
  nAlternatives = 5,
): StepResult {
  const logits = Float64Array.from(rawLogits as ArrayLike<number>);
  processLogits(logits, banned, seen, params.repetition_penalty);
  const probs = softmax(logits, params.temperature);
  const id = params.temperature <= 0 ? argmax(probs) : sampleFromProbs(probs, params.top_k, params.top_p, rng);
  const alternatives = topK(probs, nAlternatives).map((i) => [i, probs[i]] as [number, number]);
  return { id, prob: probs[id], alternatives };
}
