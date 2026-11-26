// Heavy half of the in-browser engine, loaded with dynamic import() from ./index.ts:
// onnxruntime-web (WebGPU + wasm), model download with progress + Cache Storage, sessions.

import * as ort from "onnxruntime-web";
import type { DoneEvent, GenerateRequest, Tok } from "../api";
import { generate } from "./generate";
import { createRunner } from "./session";
import type { Tokenizer } from "./tokenizer";
import type { Backend, LoadedModelInfo, LoadOptions, LoadProgress, ModelMeta } from "./types";
import { getWebGPUInfo, modelBaseUrl } from "./webgpu-info";

// onnxruntime prints informational warnings (e.g. shape ops placed on the CPU) via console.error; keep real errors only
ort.env.logLevel = "error";

const CACHE_NAME = "amirgpt-onnx-v1";

type Engine = {
  info: LoadedModelInfo;
  meta: ModelMeta;
  session: ort.InferenceSession;
  tokenizer: Tokenizer;
  queue: Promise<unknown>; // serialises generations (one session.run at a time)
};

type Entry = {
  requested: string; // the variant name that was resolved, e.g. "amirgpt-chat-fp16"
  promise: Promise<Engine>;
  listeners: Set<(p: LoadProgress) => void>;
};

const engines = new Map<string, Entry>(); // key: model id ("amirgpt-chat")
let tokenizerPromise: Promise<Tokenizer> | null = null;

function loadTokenizer(): Promise<Tokenizer> {
  if (!tokenizerPromise) {
    tokenizerPromise = import("./tokenizer").then((m) => m.createTokenizer());
    tokenizerPromise.catch(() => (tokenizerPromise = null));
  }
  return tokenizerPromise;
}

function splitName(name: string): { modelId: string; dtype: "fp16" | "fp32" | null } {
  const m = /^(.*)-(fp16|fp32)$/.exec(name);
  return m ? { modelId: m[1], dtype: m[2] as "fp16" | "fp32" } : { modelId: name, dtype: null };
}

async function resolveVariant(name: string, opts: LoadOptions): Promise<{ modelId: string; variant: string; device: Backend; why?: string }> {
  const { modelId, dtype } = splitName(name);
  const gpu = await getWebGPUInfo();
  let device: Backend;
  let why: string | undefined;
  if (opts.device === "wasm") {
    device = "wasm";
    why = "wasm requested";
  } else if (gpu.available) {
    device = "webgpu";
  } else {
    device = "wasm";
    why = gpu.reason ?? "WebGPU unavailable";
  }
  let dt = dtype ?? (opts.dtype && opts.dtype !== "auto" ? opts.dtype : null);
  if (!dt) dt = device === "webgpu" ? (gpu.shaderF16 ? "fp16" : "fp32") : "fp16";
  if (device === "webgpu" && dt === "fp16" && !gpu.shaderF16) {
    device = "wasm"; // an fp16 graph on WebGPU needs shader-f16
    why = "the WebGPU adapter lacks shader-f16 (needed by the fp16 model)";
  }
  return { modelId, variant: `${modelId}-${dt}`, device, why };
}

async function sha256Hex(bytes: Uint8Array): Promise<string | null> {
  const subtle = (globalThis as { crypto?: Crypto }).crypto?.subtle;
  if (!subtle) return null; // not a secure context: skip the check
  const h = new Uint8Array(await subtle.digest("SHA-256", bytes));
  return Array.from(h, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function openCache(): Promise<Cache | null> {
  try {
    return typeof caches !== "undefined" ? await caches.open(CACHE_NAME) : null;
  } catch {
    return null; // Cache Storage unavailable (insecure context, private mode, ...)
  }
}

/** Model bytes from Cache Storage, or downloaded with progress, verified and cached. */
async function fetchModelBytes(url: string, meta: ModelMeta, report: (p: Omit<LoadProgress, "name">) => void) {
  const key = `${url}?sha256=${meta.sha256}`; // a re-export changes the key
  const cache = await openCache();
  if (cache) {
    try {
      const hit = await cache.match(key);
      if (hit) {
        const bytes = new Uint8Array(await hit.arrayBuffer());
        if (bytes.byteLength === meta.bytes) {
          report({ phase: "download", loaded: meta.bytes, total: meta.bytes, fromCache: true });
          return { bytes, fromCache: true };
        }
        await cache.delete(key);
      }
    } catch {
      /* fall through to the network */
    }
  }

  // the model is one file, or (static hosting) several parts that are joined in order
  const urls = meta.parts?.length ? meta.parts.map((p) => url.replace(/[^/]*$/, p)) : [url];
  const total = meta.bytes;
  let buf = new Uint8Array(total);
  let loaded = 0;
  let lastReport = 0;
  report({ phase: "download", loaded: 0, total, fromCache: false });
  for (const partUrl of urls) {
    const r = await fetch(partUrl);
    if (!r.ok || !r.body) throw new Error(`model download failed: HTTP ${r.status} for ${partUrl}`);
    const reader = r.body.getReader();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (loaded + value.byteLength > buf.byteLength) {
        const grown = new Uint8Array(Math.max(buf.byteLength * 2, loaded + value.byteLength));
        grown.set(buf.subarray(0, loaded));
        buf = grown;
      }
      buf.set(value, loaded);
      loaded += value.byteLength;
      const now = Date.now();
      if (now - lastReport >= 100) {
        lastReport = now;
        report({ phase: "download", loaded, total, fromCache: false });
      }
    }
  }
  report({ phase: "download", loaded, total, fromCache: false });
  if (loaded !== total) throw new Error(`model download incomplete: got ${loaded} of ${total} bytes`);

  report({ phase: "verify", loaded, total, fromCache: false });
  const digest = await sha256Hex(buf);
  if (digest !== null && digest !== meta.sha256) throw new Error("model download corrupted (sha256 mismatch)");

  if (cache) {
    try {
      for (const req of await cache.keys()) {
        if (req.url.split("?")[0] === new URL(url, location.href).href) await cache.delete(req);
      }
      await cache.put(key, new Response(buf, { headers: { "Content-Type": "application/octet-stream" } }));
    } catch (e) {
      console.warn("[amirgpt] could not cache the model (quota?)", e);
    }
  }
  return { bytes: buf, fromCache: false };
}

async function createSession(bytes: Uint8Array, device: Backend): Promise<ort.InferenceSession> {
  if (device === "webgpu") {
    return ort.InferenceSession.create(bytes, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
      logSeverityLevel: 3, // errors only: skip the informational "nodes assigned to CPU" warning
      preferredOutputLocation: { present_kv: "gpu-buffer", logits: "cpu" },
    });
  }
  return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 });
}

async function loadEngine(name: string, opts: LoadOptions, entry: Entry): Promise<Engine> {
  const emit = (p: Omit<LoadProgress, "name">) => {
    for (const l of entry.listeners) {
      try {
        l({ ...p, name });
      } catch {
        /* a listener's error must not break loading */
      }
    }
  };
  const { device: wanted, why } = await resolveVariant(name, opts);
  const base = modelBaseUrl();
  emit({ phase: "meta", loaded: 0, total: 0, fromCache: false });
  const tokP = loadTokenizer(); // in parallel with the download
  const mr = await fetch(`${base}/${name}/meta.json`, { cache: "no-cache" });
  if (!mr.ok) throw new Error(`model ${name} is not available in-browser (HTTP ${mr.status} for meta.json)`);
  const meta = (await mr.json()) as ModelMeta;
  const { bytes, fromCache } = await fetchModelBytes(`${base}/${name}/${meta.file}`, meta, emit);

  emit({ phase: "init", loaded: meta.bytes, total: meta.bytes, fromCache, backend: wanted });
  let backend: Backend = wanted;
  let fallbackReason = why;
  let session: ort.InferenceSession;
  try {
    session = await createSession(bytes, wanted);
  } catch (e) {
    if (wanted !== "webgpu") throw e;
    console.warn("[amirgpt] WebGPU session failed, falling back to wasm", e);
    backend = "wasm";
    fallbackReason = `WebGPU session failed: ${(e as Error).message ?? e}`;
    emit({ phase: "init", loaded: meta.bytes, total: meta.bytes, fromCache, backend });
    session = await createSession(bytes, "wasm");
  }
  const tokenizer = await tokP;
  const info: LoadedModelInfo = {
    name, modelId: meta.model_id, mode: meta.mode, dtype: meta.dtype, backend, bytes: meta.bytes, fromCache,
    ...(backend === "wasm" && fallbackReason ? { fallbackReason } : {}),
  };
  emit({ phase: "ready", loaded: meta.bytes, total: meta.bytes, fromCache, backend });
  return { info, meta, session, tokenizer, queue: Promise.resolve() };
}

async function releaseEntry(entry: Entry): Promise<void> {
  try {
    const eng = await entry.promise;
    await eng.queue.catch(() => undefined);
    await eng.session.release();
  } catch {
    /* failed load: nothing to release */
  }
}

export async function loadModel(name: string, onProgress?: (p: LoadProgress) => void, opts: LoadOptions = {}): Promise<LoadedModelInfo> {
  const { modelId, variant } = await resolveVariant(name, opts);
  const existing = engines.get(modelId);
  if (existing && (existing.requested === variant || (splitName(name).dtype === null && !opts.dtype))) {
    if (onProgress) existing.listeners.add(onProgress);
    try {
      return (await existing.promise).info;
    } finally {
      if (onProgress) existing.listeners.delete(onProgress);
    }
  }
  if (existing) {
    engines.delete(modelId);
    await releaseEntry(existing); // a different variant of the same model: free it first
  }
  const entry: Entry = { requested: variant, promise: undefined as unknown as Promise<Engine>, listeners: new Set() };
  if (onProgress) entry.listeners.add(onProgress);
  entry.promise = loadEngine(variant, opts, entry);
  engines.set(modelId, entry);
  try {
    return (await entry.promise).info;
  } catch (e) {
    if (engines.get(modelId) === entry) engines.delete(modelId);
    throw e;
  } finally {
    if (onProgress) entry.listeners.delete(onProgress);
  }
}

export async function unloadModel(name: string): Promise<void> {
  const { modelId } = splitName(name);
  const entry = engines.get(modelId);
  if (!entry) return;
  engines.delete(modelId);
  await releaseEntry(entry);
}

export function loadedModels(): string[] {
  return [...engines.values()].map((e) => e.requested);
}

/** Generates with the engine loaded for req.model (loading it first if needed). */
export async function streamGenerateLocal(req: GenerateRequest, onToken: (t: Tok) => void, signal: AbortSignal): Promise<DoneEvent> {
  await loadModel(req.model);
  const entry = engines.get(splitName(req.model).modelId);
  if (!entry) throw new Error(`model ${req.model} is not loaded`);
  const eng = await entry.promise;
  const run = eng.queue.then(() =>
    generate(
      () => createRunner(ort, eng.session, eng.meta),
      eng.tokenizer,
      { mode: eng.meta.mode, blockSize: eng.meta.block_size, vocabSize: eng.meta.vocab_size },
      req,
      onToken,
      signal,
    ),
  );
  eng.queue = run.catch(() => undefined);
  return run;
}
