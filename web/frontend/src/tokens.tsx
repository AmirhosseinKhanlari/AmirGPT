// Shared token rendering: confidence lens, token card (alternatives + fork), confidence river,
// and the raw / markdown-formatted views. Used by chat messages and Compare columns.
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Alt, Tok } from "./api";
import type { MdBlock, Run } from "./markdown";
import { BOLD, CODE, ITALIC, parseMarkdown } from "./markdown";

// ---------------------------------------------------------------- helpers
/** Low-probability tokens glow warm; confident tokens stay plain. */
export function confidenceStyle(p: number): React.CSSProperties {
  const surprise = 1 - p; // 0 = model was certain
  // cubic: p=0.5 -> faint, p=0.3 -> light, p<0.1 -> strong
  const a = 0.6 * surprise ** 3;
  if (a < 0.05) return {};
  return { background: `rgba(var(--heat-rgb), ${a.toFixed(3)})` };
}

export const showWs = (s: string) => s.replace(/\n/g, "⏎").replace(/ /g, "·") || "∅";

export function perplexity(tokens: Tok[]): number | null {
  if (!tokens.length) return null;
  const nll = tokens.reduce((s, t) => s - Math.log(Math.max(t.prob, 1e-12)), 0) / tokens.length;
  return Math.exp(nll);
}

export function meanProb(tokens: Tok[]): number | null {
  if (!tokens.length) return null;
  return tokens.reduce((s, t) => s + t.prob, 0) / tokens.length;
}

// ---------------------------------------------------------------- token card
export function TokenCard({ tok, x, y, onFork, onClose }: {
  tok: Tok; x: number; y: number; onFork?: (alt: Alt) => void; onClose: () => void;
}) {
  const max = Math.max(...tok.alternatives.map((a) => a.prob), tok.prob);
  const inList = tok.alternatives.some((a) => a.id === tok.id);
  const rows = inList ? tok.alternatives : [...tok.alternatives, { id: tok.id, text: tok.text, prob: tok.prob }];
  return (
    <div className="token-card" style={{ left: x, top: y }} onMouseLeave={onClose} role="dialog" aria-label="Token alternatives">
      <div className="token-card-head">
        <span className="mono">“{showWs(tok.text)}”</span>
        <span className="muted">p = {(tok.prob * 100).toFixed(1)}%</span>
        <button className="token-card-close" onClick={onClose} aria-label="close">×</button>
      </div>
      <div className="token-card-sub">What the model considered here.{onFork ? " Click one to fork the reply." : ""}</div>
      {rows.map((a) => {
        const chosen = a.id === tok.id;
        return (
          <button key={a.id} className={"alt-row" + (chosen ? " chosen" : "")} disabled={chosen || !onFork}
                  onClick={() => onFork?.(a)} title={chosen ? "the token that was sampled" : "fork from here with this token"}>
            <span className="alt-bar" style={{ width: `${(a.prob / max) * 100}%` }} />
            <span className="alt-text mono">{showWs(a.text)}</span>
            <span className="alt-prob">{(a.prob * 100).toFixed(1)}%</span>
            <span className="alt-action">{chosen ? "sampled" : onFork ? "fork ↳" : ""}</span>
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- confidence river
export function River({ tokens, active, onPick }: { tokens: Tok[]; active: number | null; onPick: (i: number) => void }) {
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

// ---------------------------------------------------------------- token view
export type TokenViewProps = {
  tokens: Tok[];
  tail: string;
  lens: boolean;
  /** render markdown structure (true) or the raw token stream (false) */
  markdown: boolean;
  /** false while generating: the token card is disabled, as in chat */
  interactive: boolean;
  streaming?: boolean;
  forkAt?: number;
  error?: string;
  empty?: ReactNode;
  className?: string;
  onFork?: (tokenIndex: number, alt: Alt) => void;
};

const styleClass = (st: number) =>
  (st & BOLD ? " md-b" : "") + (st & ITALIC ? " md-i" : "") + (st & CODE ? " md-code" : "");

/**
 * The body of a reply: token spans (tinted by the confidence lens), the token card, and the
 * confidence river underneath. In markdown mode each token may be drawn as several spans
 * (one per style run), all carrying the same data-i, so hover / card / fork work the same.
 */
export function TokenView(p: TokenViewProps) {
  const { tokens } = p;
  const [card, setCard] = useState<{ i: number; x: number; y: number } | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  const openCard = (i: number, el: HTMLElement) => {
    const host = bodyRef.current!.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    const w = Math.min(300, host.width);
    const x = Math.min(Math.max(0, r.left - host.left), Math.max(0, host.width - w));
    setCard({ i, x, y: r.bottom - host.top + 6 });
  };

  // touch has no hover: a tap outside the card (or Escape) closes it
  useEffect(() => {
    if (!card) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Element | null;
      if (!(t && bodyRef.current?.contains(t) && t.closest(".tok, .token-card"))) setCard(null);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setCard(null); };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("pointerdown", onDown); document.removeEventListener("keydown", onKey); };
  }, [card]);

  const blocks = useMemo(
    () => (p.markdown ? parseMarkdown([...tokens.map((t) => t.text), p.tail], { streaming: !!p.streaming }) : null),
    [p.markdown, tokens, p.tail, p.streaming],
  );

  const span = (i: number, text: string, key: string | number, extra = "") => {
    if (i >= tokens.length) return <span key={key} className={extra.trim() || undefined}>{text}</span>; // decoder tail
    const t = tokens[i];
    return (
      <span key={key} data-i={i}
            className={"tok" + extra + (p.forkAt === i ? " tok-fork" : "") + (card?.i === i ? " tok-active" : "")}
            style={p.lens ? confidenceStyle(t.prob) : undefined}
            onMouseEnter={(e) => p.lens && p.interactive && openCard(i, e.currentTarget)}
            onClick={(e) => p.interactive && openCard(i, e.currentTarget)}>
        {text}
      </span>
    );
  };

  const caret = p.streaming ? <span className="caret" /> : null;
  const showEmpty = !p.streaming && tokens.length === 0 && !p.error;

  let content: ReactNode;
  if (!blocks) {
    content = (
      <>
        {tokens.map((t, i) => span(i, t.text, i))}
        {p.tail}
        {caret}
      </>
    );
  } else {
    content = <>{renderBlocks(blocks, span, caret)}{blocks.length === 0 && caret}</>;
  }

  const pickFromRiver = (i: number) => {
    const body = bodyRef.current;
    if (!body) return;
    // markup tokens (e.g. "**") have no visible span: anchor the card on the nearest visible one
    for (let d = 0; d < tokens.length; d++) {
      for (const j of d === 0 ? [i] : [i - d, i + d]) {
        const el = body.querySelector(`.tok[data-i="${j}"]`) as HTMLElement | null;
        if (el) { openCard(i, el); return; }
      }
    }
  };

  return (
    <>
      <div className={(p.className ?? "tokens") + (blocks ? " md" : "")} ref={bodyRef}>
        {content}
        {showEmpty && (p.empty ?? <span className="muted">(empty reply)</span>)}
        {card && tokens[card.i] && p.interactive && (
          <TokenCard tok={tokens[card.i]} x={card.x} y={card.y} onClose={() => setCard(null)}
                     onFork={p.onFork && ((alt) => { const i = card.i; setCard(null); p.onFork!(i, alt); })} />
        )}
      </div>
      {p.error && <div className="error">⚠ {p.error}</div>}
      {p.lens && <River tokens={tokens} active={card?.i ?? null} onPick={pickFromRiver} />}
    </>
  );
}

type SpanFn = (i: number, text: string, key: string | number, extra?: string) => ReactNode;

function renderRuns(runs: Run[], span: SpanFn, prefix: string) {
  return runs.map((r, k) => span(r.tok, r.text, `${prefix}.${k}`, styleClass(r.style)));
}

/** Blocks -> elements. Consecutive list items of the same type and depth form one list. */
function renderBlocks(blocks: MdBlock[], span: SpanFn, caret: ReactNode): ReactNode[] {
  const out: ReactNode[] = [];
  const last = blocks.length - 1;
  for (let b = 0; b < blocks.length; b++) {
    const blk = blocks[b];
    const tail = b === last ? caret : null;
    if (blk.kind === "li") {
      const items: ReactNode[] = [];
      let e = b;
      for (; e < blocks.length; e++) {
        const it = blocks[e];
        if (it.kind !== "li" || it.ordered !== blk.ordered || it.indent !== blk.indent) break;
        items.push(
          <li key={e} value={it.ordered ? it.num : undefined}>
            {renderRuns(it.runs, span, String(e))}{e === last ? caret : null}
          </li>,
        );
      }
      const style = blk.indent ? { marginLeft: `${blk.indent * 1.4}em` } : undefined;
      out.push(blk.ordered ? <ol key={b} className="md-list" style={style}>{items}</ol>
                           : <ul key={b} className="md-list" style={style}>{items}</ul>);
      b = e - 1;
    } else if (blk.kind === "code") {
      out.push(
        <pre key={b} className="md-pre" data-lang={blk.lang || undefined}>
          <code>{renderRuns(blk.runs, span, String(b))}{tail}</code>
        </pre>,
      );
    } else if (blk.kind === "h") {
      const H = (["h1", "h2", "h3"] as const)[blk.level - 1];
      out.push(<H key={b} className="md-h">{renderRuns(blk.runs, span, String(b))}{tail}</H>);
    } else {
      out.push(<p key={b} className="md-p">{renderRuns(blk.runs, span, String(b))}{tail}</p>);
    }
  }
  return out;
}
