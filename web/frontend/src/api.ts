// Types + streaming client for the AmirGPT FastAPI backend (web/server.py).

export type Alt = { id: number; text: string; prob: number };
export type Tok = { id: number; text: string; prob: number; alternatives: Alt[] };

export type ModelInfo = {
  id: string;
  name: string;
  mode: "chat" | "complete";
  description: string;
  available: boolean;
  loaded: boolean;
};

export type GenParams = {
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  max_new_tokens: number;
  seed: number | null;
};

export const DEFAULT_PARAMS: GenParams = {
  temperature: 0.7,
  top_k: 50,
  top_p: 0.95,
  repetition_penalty: 1.1,
  max_new_tokens: 256,
  seed: null,
};

export type DoneEvent = {
  type: "done";
  reason: "stop" | "length";
  tail: string;
  tokens: number;
  tokens_per_sec: number;
  prompt_tokens: number;
};

export type GenerateRequest = {
  model: string;
  messages?: { role: string; content: string }[];
  prompt?: string;
  system_prompt?: string | null;
  prefix_ids?: number[];
  params: GenParams;
};

export async function fetchModels(): Promise<ModelInfo[]> {
  const r = await fetch("/api/models");
  if (!r.ok) throw new Error(`models: HTTP ${r.status}`);
  return r.json();
}

export async function fetchMetrics(): Promise<Metrics> {
  const r = await fetch("/api/metrics");
  if (!r.ok) throw new Error(`metrics: HTTP ${r.status}`);
  return r.json();
}

/** POST /api/generate and call onToken for every streamed token. Resolves with the done event. */
export async function streamGenerate(
  req: GenerateRequest,
  onToken: (t: Tok) => void,
  signal: AbortSignal,
): Promise<DoneEvent> {
  const r = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!r.ok || !r.body) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let done: DoneEvent | null = null;
  for (;;) {
    const { value, done: eof } = await reader.read();
    if (eof) break;
    buf += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.type === "token") onToken({ id: ev.id, text: ev.text, prob: ev.prob, alternatives: ev.alternatives });
      else if (ev.type === "done") done = ev as DoneEvent;
    }
  }
  if (!done) throw new Error("stream ended unexpectedly");
  return done;
}

export type Point = [number, number];
export type Baseline = { value: number; source: string };
export type Metrics = {
  pretrain: {
    train: Point[];
    val: Point[];
    hella: Point[];
    samples: { step: number; value: string[] }[];
    config: Record<string, unknown> | null;
  };
  sft: { train: Point[]; val: Point[]; samples: { step: number; value: { prompt: string; reply: string }[] }[] };
  baselines: Record<string, Baseline>;
};
