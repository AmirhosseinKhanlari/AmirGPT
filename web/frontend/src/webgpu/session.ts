// KV-cached runner over an onnxruntime-web InferenceSession for the graph exported by
// web/export_onnx.py. The ort module is passed in (type-only import here), so the same code
// runs in the browser (WebGPU / wasm) and in the Node parity test.

import type * as OrtNS from "onnxruntime-web";
import type { Runner } from "./generate";
import type { ModelMeta } from "./types";

export type OrtModule = typeof OrtNS;

/**
 * With a WebGPU session created with preferredOutputLocation { present_kv: "gpu-buffer" },
 * the cache stays on the GPU between steps and only the logits ([1, vocab]) come back to the
 * CPU; GPU tensors are disposed as soon as they are replaced. With wasm everything is on the CPU.
 */
export function createRunner(ort: OrtModule, session: OrtNS.InferenceSession, meta: ModelMeta): Runner {
  const dims = [meta.n_layer, 2, 1, meta.n_head, 0, meta.head_dim];
  let past: OrtNS.Tensor =
    meta.kv_dtype === "float16"
      ? new ort.Tensor("float16", new Uint16Array(0), dims)
      : new ort.Tensor("float32", new Float32Array(0), dims);
  let len = 0;
  let released = false;

  const disposeIfGpu = (t: OrtNS.Tensor) => {
    if (t.location === "gpu-buffer") t.dispose();
  };

  return {
    async forward(ids: number[]) {
      if (released) throw new Error("runner already released");
      if (ids.length === 0) throw new Error("forward needs at least one token");
      if (len + ids.length > meta.block_size) {
        throw new Error(`context window exceeded: ${len + ids.length} > ${meta.block_size}`);
      }
      const inputIds = new ort.Tensor("int64", BigInt64Array.from(ids, (x) => BigInt(x)), [1, ids.length]);
      const out = await session.run({ input_ids: inputIds, past_kv: past });
      const logitsT = out.logits;
      const present = out.present_kv;
      const logits = (logitsT.location === "cpu" ? logitsT.data : await logitsT.getData(true)) as Float32Array;
      if (released) {
        // released while the run was in flight: drop the new cache too
        disposeIfGpu(present);
        return logits;
      }
      disposeIfGpu(past);
      past = present;
      len += ids.length;
      return logits;
    },
    cacheLength: () => len,
    release() {
      if (released) return;
      released = true;
      disposeIfGpu(past);
    },
  };
}
