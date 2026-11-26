import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: `npm run dev` proxies /api (generation) and /models (ONNX files for the in-browser
// engine, src/webgpu) to the FastAPI server on :8000
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/models": "http://127.0.0.1:8000",
    },
  },
  // onnxruntime-web locates its .wasm with new URL("ort-wasm-simd-threaded.jsep.wasm", import.meta.url):
  // Vite's dep pre-bundling would break that URL in dev, so serve the package as-is. In the
  // production build Vite sees the same pattern and emits the .wasm as a hashed asset.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: {
    // ort's bundle uses modern syntax (and top-level await in its worker glue)
    target: "es2022",
    // the lazily loaded engine chunks (onnxruntime-web ~420 kB, GPT-2 BPE ranks ~545 kB) are
    // only fetched when the in-browser engine is used
    chunkSizeWarningLimit: 1024,
  },
});
