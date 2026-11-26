// Light-weight helpers shared by index.ts and runtime.ts (no onnxruntime import).

type GPUAdapterLike = { features: { has(f: string): boolean } };
type GPULike = { requestAdapter(opts?: { powerPreference?: string }): Promise<GPUAdapterLike | null> };

export type WebGPUInfo = { available: boolean; shaderF16: boolean; reason?: string };

let infoPromise: Promise<WebGPUInfo> | null = null;

/** Probes navigator.gpu once (cached): is there a WebGPU adapter, and does it support shader-f16? */
export function getWebGPUInfo(): Promise<WebGPUInfo> {
  if (!infoPromise) {
    infoPromise = (async (): Promise<WebGPUInfo> => {
      const gpu = (globalThis.navigator as unknown as { gpu?: GPULike } | undefined)?.gpu;
      if (!gpu) return { available: false, shaderF16: false, reason: "navigator.gpu is not available in this browser" };
      try {
        const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) return { available: false, shaderF16: false, reason: "no WebGPU adapter" };
        return { available: true, shaderF16: adapter.features.has("shader-f16") };
      } catch (e) {
        return { available: false, shaderF16: false, reason: `requestAdapter failed: ${(e as Error).message ?? e}` };
      }
    })();
  }
  return infoPromise;
}

let baseUrl = "/models";

/** Where the exported models are served (default "/models", see web/server.py). */
export function setModelBaseUrl(url: string): void {
  baseUrl = url.replace(/\/+$/, "");
}

export function modelBaseUrl(): string {
  return baseUrl;
}
