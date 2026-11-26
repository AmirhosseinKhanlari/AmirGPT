// Shared types for the in-browser (onnxruntime-web) engine. No runtime dependencies.

/** checkpoints/onnx/<name>/meta.json, written by web/export_onnx.py. */
export type ModelMeta = {
  name: string; // e.g. "amirgpt-chat-fp16"
  model_id: string; // e.g. "amirgpt-chat" (the server's model id)
  mode: "chat" | "complete";
  dtype: "fp32" | "fp16";
  kv_dtype: "float32" | "float16";
  file: string;
  /** optional: the file split into parts (static hosting with a per-file size limit); joined in order */
  parts?: string[];
  bytes: number;
  sha256: string;
  opset: number;
  vocab_size: number;
  tokenizer_vocab_size: number;
  n_layer: number;
  n_head: number;
  head_dim: number;
  n_embd: number;
  block_size: number;
  special_tokens: { endoftext: number; user: number; assistant: number; system: number; end: number };
};

export type Backend = "webgpu" | "wasm";

export type LoadProgress = {
  /** meta: fetching meta.json; download: model bytes (from network or Cache Storage);
   *  verify: sha256 check; init: creating the onnxruntime session; ready: done. */
  phase: "meta" | "download" | "verify" | "init" | "ready";
  loaded: number; // bytes
  total: number; // bytes (meta.bytes)
  fromCache: boolean;
  name: string;
  backend?: Backend;
};

export type LoadOptions = {
  /** "auto" (default): fp16 when the WebGPU adapter has shader-f16, else fp32. Ignored if the
   *  name already ends in -fp16 / -fp32. */
  dtype?: "auto" | "fp16" | "fp32";
  /** "auto" (default): WebGPU if available, else wasm. WebGPU failures fall back to wasm. */
  device?: "auto" | Backend;
};

export type LoadedModelInfo = {
  name: string; // the variant directory, e.g. "amirgpt-chat-fp16"
  modelId: string; // "amirgpt-chat"
  mode: "chat" | "complete";
  dtype: "fp32" | "fp16";
  backend: Backend;
  bytes: number;
  fromCache: boolean;
  /** why WebGPU was not used, when backend === "wasm" */
  fallbackReason?: string;
};
