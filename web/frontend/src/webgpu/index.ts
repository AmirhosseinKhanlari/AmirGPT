// In-browser AmirGPT engine (onnxruntime-web: WebGPU, falling back to wasm).
//
// Public API. This module is small: onnxruntime-web, the tokenizer ranks and the model are
// only fetched (dynamic import) when loadModel / streamGenerateLocal is first called.
//
//   const info = await loadModel("amirgpt-chat", (p) => console.log(p.phase, p.loaded / p.total));
//   const done = await streamGenerateLocal(req, onToken, abortController.signal);  // same as streamGenerate
//
// Models are served from /models/<name>/{meta.json,model.onnx} (web/server.py), where <name> is
// "<model id>-fp16" or "<model id>-fp32"; see web/export_onnx.py.

import type { DoneEvent, GenerateRequest, Tok } from "../api";
import type { LoadedModelInfo, LoadOptions, LoadProgress } from "./types";
import { getWebGPUInfo } from "./webgpu-info";

export type { Backend, LoadedModelInfo, LoadOptions, LoadProgress, ModelMeta } from "./types";
export type { WebGPUInfo } from "./webgpu-info";
export { getWebGPUInfo, setModelBaseUrl } from "./webgpu-info";

const runtime = () => import("./runtime");

/** true if navigator.gpu exists and an adapter can be obtained. */
export async function isWebGPUAvailable(): Promise<boolean> {
  return (await getWebGPUInfo()).available;
}

/**
 * Downloads (or reads from Cache Storage) and initialises a model. `name` is a server model id
 * ("amirgpt-chat", "amirgpt-base": the dtype is then chosen automatically) or an explicit
 * variant ("amirgpt-chat-fp32"). Resolves when the model is ready; repeated calls are cheap.
 */
export async function loadModel(
  name: string,
  onProgress?: (p: LoadProgress) => void,
  options?: LoadOptions,
): Promise<LoadedModelInfo> {
  return (await runtime()).loadModel(name, onProgress, options);
}

/** Frees the session (and its GPU memory) of a loaded model. */
export async function unloadModel(name: string): Promise<void> {
  return (await runtime()).unloadModel(name);
}

/**
 * Drop-in replacement for api.ts streamGenerate: same request, same token events
 * ({id, text, prob, alternatives[{id, text, prob}]}), same done event, AbortError on abort.
 * Loads req.model first if needed (without a progress callback; call loadModel for progress).
 */
export async function streamGenerateLocal(
  req: GenerateRequest,
  onToken: (t: Tok) => void,
  signal: AbortSignal,
): Promise<DoneEvent> {
  return (await runtime()).streamGenerateLocal(req, onToken, signal);
}
