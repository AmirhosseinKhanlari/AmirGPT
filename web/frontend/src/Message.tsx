import { useMemo, useState } from "react";
import type { Alt } from "./api";
import type { ChatNode } from "./tree";
import { nodeText } from "./tree";
import { perplexity, TokenView } from "./tokens";

// ---------------------------------------------------------------- message
export type MessageProps = {
  node: ChatNode;
  mode: "chat" | "complete";
  lens: boolean;
  markdown: boolean;
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
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(node.content);
  const ppl = useMemo(() => perplexity(node.tokens), [node.tokens]);

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
        <TokenView className="msg-body tokens" tokens={node.tokens} tail={node.tail} lens={p.lens}
                   markdown={p.markdown} interactive={!p.busy} streaming={node.streaming}
                   forkAt={node.fork?.at} error={node.error} onFork={p.onFork} />
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
