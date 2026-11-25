import { useMemo, useRef, useState } from "react";
import type { Alt, Tok } from "./api";
import type { ChatNode } from "./tree";
import { nodeText } from "./tree";

// ---------------------------------------------------------------- helpers
/** Low-probability tokens glow warm; confident tokens stay plain. */
function confidenceStyle(p: number): React.CSSProperties {
  const surprise = 1 - p; // 0 = model was certain
  // cubic: p=0.5 -> faint, p=0.3 -> light, p<0.1 -> strong
  const a = 0.6 * surprise ** 3;
  if (a < 0.05) return {};
  return { background: `rgba(var(--heat-rgb), ${a.toFixed(3)})` };
}

const showWs = (s: string) => s.replace(/\n/g, "⏎").replace(/ /g, "·") || "∅";

export function perplexity(tokens: Tok[]): number | null {
  if (!tokens.length) return null;
  const nll = tokens.reduce((s, t) => s - Math.log(Math.max(t.prob, 1e-12)), 0) / tokens.length;
  return Math.exp(nll);
}

// ---------------------------------------------------------------- token card
function TokenCard({ tok, x, y, onFork, onClose }: {
  tok: Tok; x: number; y: number; onFork: (alt: Alt) => void; onClose: () => void;
}) {
  const max = Math.max(...tok.alternatives.map((a) => a.prob), tok.prob);
  const inList = tok.alternatives.some((a) => a.id === tok.id);
  const rows = inList ? tok.alternatives : [...tok.alternatives, { id: tok.id, text: tok.text, prob: tok.prob }];
  return (
    <div className="token-card" style={{ left: x, top: y }} onMouseLeave={onClose} role="dialog" aria-label="Token alternatives">
      <div className="token-card-head">
        <span className="mono">“{showWs(tok.text)}”</span>
        <span className="muted">p = {(tok.prob * 100).toFixed(1)}%</span>
      </div>
      <div className="token-card-sub">What the model considered here. Click one to fork the reply.</div>
      {rows.map((a) => {
        const chosen = a.id === tok.id;
        return (
          <button key={a.id} className={"alt-row" + (chosen ? " chosen" : "")} disabled={chosen}
                  onClick={() => onFork(a)} title={chosen ? "the token that was sampled" : "fork from here with this token"}>
            <span className="alt-bar" style={{ width: `${(a.prob / max) * 100}%` }} />
            <span className="alt-text mono">{showWs(a.text)}</span>
            <span className="alt-prob">{(a.prob * 100).toFixed(1)}%</span>
            <span className="alt-action">{chosen ? "sampled" : "fork ↳"}</span>
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- confidence river
function River({ tokens, active, onPick }: { tokens: Tok[]; active: number | null; onPick: (i: number) => void }) {
  const W = 600, H = 34;
  if (tokens.length < 2) return null;
  const step = W / (tokens.length - 1);
  const pts = tokens.map((t, i) => `${(i * step).toFixed(1)},${(H - 3 - t.prob * (H - 6)).toFixed(1)}`).join(" ");
  return (
    <svg className="river" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
         aria-label="Model confidence for each token"
         onClick={(e) => {
           const r = (e.currentTarget as SVGElement).getBoundingClientRect();
           onPick(Math.round(((e.clientX - r.left) / r.width) * (tokens.length - 1)));
         }}>
      <polyline points={`0,${H} ${pts} ${W},${H}`} className="river-fill" />
      <polyline points={pts} className="river-line" />
      {active !== null && active < tokens.length && (
        <line x1={active * step} x2={active * step} y1={0} y2={H} className="river-cursor" />
      )}
    </svg>
  );
}

// ---------------------------------------------------------------- message
export type MessageProps = {
  node: ChatNode;
  mode: "chat" | "complete";
  lens: boolean;
  siblingIndex: number;
  siblingCount: number;
  busy: boolean;
  onSwitch: (delta: number) => void;
  onFork: (tokenIndex: number, alt: Alt) => void;
  onRegenerate: () => void;
  onContinue: () => void;
  onEdit: (text: string) => void;
  onStop: () => void;
};

export function Message(p: MessageProps) {
  const { node } = p;
  const [card, setCard] = useState<{ i: number; x: number; y: number } | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(node.content);
  const bodyRef = useRef<HTMLDivElement>(null);
  const ppl = useMemo(() => perplexity(node.tokens), [node.tokens]);

  const openCard = (i: number, el: HTMLElement) => {
    const host = bodyRef.current!.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    const x = Math.min(Math.max(0, r.left - host.left), Math.max(0, host.width - 300));
    setCard({ i, x, y: r.bottom - host.top + 6 });
  };

  const switcher = p.siblingCount > 1 && (
    <span className="switcher" aria-label="branches">
      <button onClick={() => p.onSwitch(-1)} disabled={p.siblingIndex === 0 || p.busy} aria-label="previous branch">‹</button>
      <span>{p.siblingIndex + 1}/{p.siblingCount}</span>
      <button onClick={() => p.onSwitch(1)} disabled={p.siblingIndex === p.siblingCount - 1 || p.busy} aria-label="next branch">›</button>
    </span>
  );

  if (node.role === "user") {
    return (
      <div className={"msg msg-user" + (p.mode === "complete" ? " msg-prompt" : "")}>
        {editing ? (
          <div className="edit-box">
            <textarea value={draft} onChange={(e) => setDraft(e.target.value)} rows={Math.min(10, draft.split("\n").length + 1)} autoFocus />
            <div className="edit-actions">
              <button className="btn ghost" onClick={() => setEditing(false)}>Cancel</button>
              <button className="btn primary" disabled={!draft.trim() || p.busy}
                      onClick={() => { setEditing(false); p.onEdit(draft); }}>Save &amp; branch</button>
            </div>
          </div>
        ) : (
          <>
            <div className="msg-body user-bubble">{node.content}</div>
            <div className="msg-foot">
              {switcher}
              <button className="link" disabled={p.busy} onClick={() => { setDraft(node.content); setEditing(true); }}>edit</button>
            </div>
          </>
        )}
      </div>
    );
  }

  const text = nodeText(node);
  return (
    <div className={"msg msg-assistant" + (p.mode === "complete" ? " msg-continuation" : "")}>
      <div className="msg-avatar" aria-hidden>{p.mode === "chat" ? "A" : "↳"}</div>
      <div className="msg-main">
        <div className="msg-body tokens" ref={bodyRef}>
          {node.tokens.map((t, i) => (
            <span key={i}
                  className={"tok" + (node.fork && i === node.fork.at ? " tok-fork" : "") + (card?.i === i ? " tok-active" : "")}
                  style={p.lens ? confidenceStyle(t.prob) : undefined}
                  onMouseEnter={(e) => p.lens && !p.busy && openCard(i, e.currentTarget)}
                  onClick={(e) => !p.busy && openCard(i, e.currentTarget)}>
              {t.text}
            </span>
          ))}
          {node.tail}
          {node.streaming && <span className="caret" />}
          {!node.streaming && node.tokens.length === 0 && !node.error && <span className="muted">(empty reply)</span>}
          {card && node.tokens[card.i] && !p.busy && (
            <TokenCard tok={node.tokens[card.i]} x={card.x} y={card.y} onClose={() => setCard(null)}
                       onFork={(alt) => { const i = card.i; setCard(null); p.onFork(i, alt); }} />
          )}
        </div>
        {node.error && <div className="error">⚠ {node.error}</div>}
        {p.lens && <River tokens={node.tokens} active={card?.i ?? null}
                          onPick={(i) => { const el = bodyRef.current?.querySelectorAll(".tok")[i] as HTMLElement | undefined; if (el) openCard(i, el); }} />}
        <div className="msg-foot">
          {switcher}
          {node.fork && <span className="chip" title="this reply was forked from another branch">forked at token {node.fork.at + 1}</span>}
          {node.streaming ? (
            <button className="link" onClick={p.onStop}>stop</button>
          ) : (
            <>
              <button className="link" disabled={p.busy} onClick={p.onRegenerate}>regenerate</button>
              {node.done?.reason === "length" && <button className="link" disabled={p.busy} onClick={p.onContinue}>continue</button>}
              <button className="link" onClick={() => navigator.clipboard?.writeText(text)}>copy</button>
            </>
          )}
          {node.done && (
            <span className="stats" title="tokens · speed · perplexity of the sampled tokens">
              {node.tokens.length} tok · {node.done.tokens_per_sec.toFixed(0)} tok/s{ppl !== null ? ` · ppl ${ppl.toFixed(1)}` : ""}
              {node.done.reason === "length" ? " · hit length limit" : ""}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
