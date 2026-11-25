import { useEffect, useState } from "react";
import type { Metrics, Point } from "./api";
import { fetchMetrics } from "./api";

type Series = { label: string; points: Point[]; cls: string };
type Ref = { label: string; value: number; cls: string };

function LineChart({ title, series, refs = [], yLabel, logY = false, yMax }: {
  title: string; series: Series[]; refs?: Ref[]; yLabel: string; logY?: boolean; yMax?: number;
}) {
  const [hover, setHover] = useState<{ x: number; step: number } | null>(null);
  const W = 640, H = 260, L = 52, R = 14, T = 14, B = 34;
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) {
    return (
      <figure className="chart">
        <figcaption>{title}</figcaption>
        <div className="chart-empty">No data yet. This fills in as training logs arrive.</div>
      </figure>
    );
  }
  const xs = all.map((p) => p[0]);
  const ys = [...all.map((p) => p[1]), ...refs.map((r) => r.value)];
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const f = logY ? Math.log : (v: number) => v;
  let y0 = Math.min(...ys), y1 = yMax ?? Math.max(...ys);
  if (y1 <= y0) y1 = y0 + 1;
  const pad = (f(y1) - f(y0)) * 0.06;
  const fy0 = f(y0) - pad, fy1 = f(y1) + pad;
  const sx = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const sy = (v: number) => T + (1 - (f(Math.min(v, y1 * 1.0001)) - fy0) / (fy1 - fy0)) * (H - T - B);
  const ticks = 5;
  const yt = Array.from({ length: ticks }, (_, i) => {
    const fv = fy0 + ((fy1 - fy0) * i) / (ticks - 1);
    return logY ? Math.exp(fv) : fv;
  });
  const xt = Array.from({ length: 5 }, (_, i) => x0 + ((x1 - x0) * i) / 4);
  const nearest = (s: Series, step: number) =>
    s.points.reduce<Point | null>((b, p) => (!b || Math.abs(p[0] - step) < Math.abs(b[0] - step) ? p : b), null);

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}
           onMouseMove={(e) => {
             const r = (e.currentTarget as SVGElement).getBoundingClientRect();
             const px = ((e.clientX - r.left) / r.width) * W;
             if (px < L || px > W - R) return setHover(null);
             setHover({ x: px, step: x0 + ((px - L) / (W - L - R)) * (x1 - x0) });
           }}
           onMouseLeave={() => setHover(null)}>
        {yt.map((v, i) => (
          <g key={i}>
            <line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} className="grid" />
            <text x={L - 6} y={sy(v) + 4} className="tick" textAnchor="end">{v < 10 ? v.toFixed(2) : v.toFixed(0)}</text>
          </g>
        ))}
        {xt.map((v, i) => (
          <text key={i} x={sx(v)} y={H - 12} className="tick" textAnchor="middle">{Math.round(v).toLocaleString()}</text>
        ))}
        <text x={14} y={T + (H - T - B) / 2} className="axis-label" transform={`rotate(-90 14 ${T + (H - T - B) / 2})`} textAnchor="middle">{yLabel}</text>
        {refs.map((r) => (
          <g key={r.label}>
            <line x1={L} x2={W - R} y1={sy(r.value)} y2={sy(r.value)} className={"ref " + r.cls} />
            <text x={W - R - 4} y={sy(r.value) - 4} className={"ref-label " + r.cls} textAnchor="end">{r.label} ({r.value})</text>
          </g>
        ))}
        {series.map((s) => (
          <polyline key={s.label} className={"series " + s.cls} fill="none"
                    points={s.points.map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ")} />
        ))}
        {hover && <line x1={hover.x} x2={hover.x} y1={T} y2={H - B} className="cursor" />}
      </svg>
      <div className="legend">
        {series.map((s) => {
          const p = hover ? nearest(s, hover.step) : s.points[s.points.length - 1];
          return (
            <span key={s.label} className="legend-item">
              <i className={"swatch " + s.cls} />{s.label}
              {p && <b> {p[1].toFixed(4)}</b>}{p && <span className="muted"> @ step {p[0].toLocaleString()}</span>}
            </span>
          );
        })}
      </div>
    </figure>
  );
}

export function Lab() {
  const [m, setM] = useState<Metrics | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => fetchMetrics().then((d) => alive && (setM(d), setErr(null))).catch((e) => alive && setErr(String(e)));
    load();
    const t = setInterval(load, 30000);
    return () => { alive = false; clearInterval(t); };
  }, []);
  if (err) return <div className="lab"><div className="error">Could not load metrics: {err}</div></div>;
  if (!m) return <div className="lab"><div className="muted">Loading training logs…</div></div>;
  const b = m.baselines;
  const lastVal = m.pretrain.val.at(-1);
  const lastHella = m.pretrain.hella.at(-1);
  const cfg = m.pretrain.config as Record<string, number | string> | null;
  return (
    <div className="lab">
      <header className="lab-head">
        <h2>The Lab</h2>
        <p className="muted">
          How AmirGPT was made: GPT-2 (124M) pretrained from scratch on FineWeb-Edu, following Karpathy's
          “Let's reproduce GPT-2 (124M)”, then chat fine-tuned. Dashed lines are reference numbers from the video and repo.
        </p>
        <div className="kpis">
          <div className="kpi"><span>val loss (latest)</span><b>{lastVal ? lastVal[1].toFixed(4) : "…"}</b><em>OpenAI GPT-2 on our val shard: {b.val_loss_openai_gpt2_124M_ours.value}</em></div>
          <div className="kpi"><span>HellaSwag (latest)</span><b>{lastHella ? (lastHella[1] * 100).toFixed(2) + "%" : "…"}</b><em>OpenAI GPT-2: {(b.hella_openai_gpt2_124M.value * 100).toFixed(2)}%</em></div>
          <div className="kpi"><span>tokens seen</span><b>{m.pretrain.train.length ? ((m.pretrain.train.at(-1)![0] + 1) * 524288 / 1e9).toFixed(2) + "B" : "…"}</b><em>batch 524,288 tokens/step</em></div>
          <div className="kpi"><span>hardware</span><b>{cfg ? `${cfg.world_size}× GPU` : "…"}</b><em>{cfg ? `${cfg.resolved_dtype}, micro-batch ${cfg.micro_batch}` : ""}</em></div>
        </div>
      </header>
      <div className="charts">
        <LineChart title="Pretraining loss" yLabel="loss" logY yMax={11}
                   series={[{ label: "train", points: m.pretrain.train, cls: "s1" }, { label: "val", points: m.pretrain.val, cls: "s2" }]}
                   refs={[{ label: "OpenAI GPT-2 124M, our val shard", value: b.val_loss_openai_gpt2_124M_ours.value, cls: "r1" },
                          { label: "OpenAI GPT-2 124M, Karpathy's notebook", value: b.val_loss_openai_gpt2_124M.value, cls: "r2" }]} />
        <LineChart title="HellaSwag accuracy (acc_norm)" yLabel="accuracy"
                   series={[{ label: "AmirGPT", points: m.pretrain.hella, cls: "s2" }]}
                   refs={[{ label: "GPT-2 124M", value: b.hella_openai_gpt2_124M.value, cls: "r1" },
                          { label: "GPT-3 124M", value: b.hella_openai_gpt3_124M.value, cls: "r2" },
                          { label: "random", value: b.hella_random.value, cls: "r3" }]} />
        <LineChart title="Chat fine-tuning (SFT) loss on assistant tokens" yLabel="loss"
                   series={[{ label: "train", points: m.sft.train, cls: "s1" }, { label: "val", points: m.sft.val, cls: "s2" }]} />
      </div>
      {m.pretrain.samples.length > 0 && (
        <section className="samples">
          <h3>“Hello, I'm a language model,”: samples during pretraining</h3>
          {m.pretrain.samples.map((s) => (
            <div key={s.step} className="sample-group">
              <div className="muted">step {s.step.toLocaleString()}</div>
              {s.value.map((t, i) => <p key={i} className="sample">{t}</p>)}
            </div>
          ))}
        </section>
      )}
      <section className="sources">
        <h3>Reference numbers</h3>
        <ul>{Object.entries(b).map(([k, v]) => <li key={k}><b>{v.value}</b>: {v.source}</li>)}</ul>
      </section>
    </div>
  );
}
