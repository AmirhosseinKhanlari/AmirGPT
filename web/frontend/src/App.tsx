import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Alt, GenParams, ModelInfo, Tok } from "./api";
import { DEFAULT_PARAMS, fetchModels, streamGenerate } from "./api";
import { Lab } from "./Lab";
import { Message } from "./Message";
import type { ChatNode, Conversation, NodeId } from "./tree";
import {
  activePath, addChild, historyUpTo, loadConversations, makeNode, newConversation, nodeText,
  saveConversations, siblings, switchSibling, updateNode,
} from "./tree";

const SUGGESTIONS: Record<"chat" | "complete", string[]> = {
  chat: ["Who are you?", "Explain photosynthesis to a 10-year-old.", "Give me 3 ideas for a weekend project.", "Write a haiku about autumn."],
  complete: ["Hello, I'm a language model,", "The most important discovery in physics was", "Once upon a time, in a small village,", "In this lesson, we will learn how"],
};

function readPref<T>(k: string, d: T): T {
  try { const v = localStorage.getItem(k); return v ? (JSON.parse(v) as T) : d; } catch { return d; }
}
function writePref(k: string, v: unknown) {
  try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ }
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelErr, setModelErr] = useState<string | null>(null);
  const [convs, setConvs] = useState<Conversation[]>(() => loadConversations());
  const [currentId, setCurrentId] = useState<string | null>(() => readPref<string | null>("amirgpt.current", null));
  const [params, setParams] = useState<GenParams>(() => ({ ...DEFAULT_PARAMS, ...readPref("amirgpt.params", {}) }));
  const [lens, setLens] = useState<boolean>(() => readPref("amirgpt.lens", true));
  const [tab, setTab] = useState<"chat" | "lab">("chat");
  const [panel, setPanel] = useState<boolean>(() => readPref("amirgpt.panel", window.innerWidth > 1100));
  const [sidebar, setSidebar] = useState<boolean>(window.innerWidth > 800);
  const [theme, setTheme] = useState<string>(() => readPref("amirgpt.theme", ""));
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);

  useEffect(() => {
    fetchModels().then(setModels).catch((e) => setModelErr(String(e)));
    const t = setInterval(() => fetchModels().then(setModels).catch(() => {}), 60000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => { saveConversations(convs); }, [convs]);
  useEffect(() => writePref("amirgpt.current", currentId), [currentId]);
  useEffect(() => writePref("amirgpt.params", params), [params]);
  useEffect(() => writePref("amirgpt.lens", lens), [lens]);
  useEffect(() => writePref("amirgpt.panel", panel), [panel]);
  useEffect(() => {
    writePref("amirgpt.theme", theme);
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
  }, [theme]);

  const defaultModel = useMemo(() => {
    const chat = models.find((m) => m.id === "amirgpt-chat" && m.available);
    return (chat ?? models.find((m) => m.available) ?? models[0])?.id ?? "amirgpt-chat";
  }, [models]);

  const conv = convs.find((c) => c.id === currentId) ?? null;
  const modelInfo = models.find((m) => m.id === (conv?.model ?? defaultModel));
  const mode: "chat" | "complete" = modelInfo?.mode ?? "chat";
  const path = conv ? activePath(conv) : [];

  // keep the latest conversation state reachable from async stream callbacks
  const convsRef = useRef(convs);
  convsRef.current = convs;
  const mutate = useCallback((id: string, f: (c: Conversation) => Conversation) => {
    setConvs((list) => list.map((c) => (c.id === id ? f(c) : c)));
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  });

  // ------------------------------------------------------------------ generation
  const run = useCallback(async (convId: string, parentId: NodeId, node: ChatNode, prefix: Tok[]) => {
    const c0 = convsRef.current.find((c) => c.id === convId)!;
    const info = models.find((m) => m.id === c0.model);
    const history = historyUpTo(c0, parentId);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setBusy(true);
    stickRef.current = true;
    let pending: Tok[] = [];
    let raf = 0;
    const flush = () => {
      raf = 0;
      if (!pending.length) return;
      const add = pending;
      pending = [];
      mutate(convId, (c) => updateNode(c, node.id, { tokens: [...c.nodes[node.id].tokens, ...add] }));
    };
    try {
      const done = await streamGenerate({
        model: c0.model,
        ...(info?.mode === "complete"
          ? { prompt: history.filter((m) => m.role === "user").map((m) => m.content).at(-1) ?? "" }
          : { messages: history, system_prompt: c0.systemPrompt || null }),
        prefix_ids: prefix.map((t) => t.id),
        params,
      }, (t) => {
        pending.push(t);
        if (!raf) raf = requestAnimationFrame(flush);
      }, ctrl.signal);
      if (raf) cancelAnimationFrame(raf);
      flush();
      mutate(convId, (c) => updateNode(c, node.id, { streaming: false, done, tail: done.tail }));
    } catch (e) {
      if (raf) cancelAnimationFrame(raf);
      flush();
      const aborted = (e as Error).name === "AbortError";
      mutate(convId, (c) => updateNode(c, node.id, { streaming: false, error: aborted ? undefined : (e as Error).message }));
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }, [models, params, mutate]);

  const startAssistant = (c: Conversation, parentId: NodeId, prefix: Tok[] = [], fork?: ChatNode["fork"]) => {
    const node = makeNode("assistant", parentId, { tokens: prefix, streaming: true, fork });
    const next = addChild(c, parentId, node);
    setConvs((list) => list.map((x) => (x.id === c.id ? next : x)));
    convsRef.current = convsRef.current.map((x) => (x.id === c.id ? next : x));
    void run(c.id, parentId, node, prefix);
  };

  const send = (text: string) => {
    if (!text.trim() || busy) return;
    let c = conv;
    if (!c || (c.model !== (modelInfo?.id ?? defaultModel))) {
      c = newConversation(modelInfo?.id ?? defaultModel);
      setConvs((l) => [c!, ...l]);
      convsRef.current = [c, ...convsRef.current];
      setCurrentId(c.id);
    }
    const last = activePath(c).at(-1);
    const user = makeNode("user", last ? last.id : null, { content: text });
    let next = addChild(c, user.parent, user);
    if (next.title === "New chat") next = { ...next, title: text.slice(0, 48) };
    setConvs((l) => l.map((x) => (x.id === next.id ? next : x)));
    convsRef.current = convsRef.current.map((x) => (x.id === next.id ? next : x));
    setInput("");
    startAssistant(next, user.id);
  };

  const fork = (n: ChatNode, i: number, alt: Alt) => {
    const c = convsRef.current.find((x) => x.id === conv!.id)!;
    const orig = n.tokens[i];
    const forced: Tok = { id: alt.id, text: alt.text, prob: alt.prob, alternatives: orig.alternatives };
    startAssistant(c, n.parent!, [...n.tokens.slice(0, i), forced], { from: n.id, at: i });
  };
  const regenerate = (n: ChatNode) => startAssistant(convsRef.current.find((x) => x.id === conv!.id)!, n.parent!);
  const cont = (n: ChatNode) => startAssistant(convsRef.current.find((x) => x.id === conv!.id)!, n.parent!, n.tokens, { from: n.id, at: n.tokens.length });
  const edit = (n: ChatNode, text: string) => {
    const c = convsRef.current.find((x) => x.id === conv!.id)!;
    const user = makeNode("user", n.parent, { content: text });
    const next = addChild(c, n.parent, user);
    setConvs((l) => l.map((x) => (x.id === next.id ? next : x)));
    convsRef.current = convsRef.current.map((x) => (x.id === next.id ? next : x));
    startAssistant(next, user.id);
  };
  const stop = () => abortRef.current?.abort();

  const newChat = (modelId?: string) => {
    if (busy) return;
    const c = newConversation(modelId ?? modelInfo?.id ?? defaultModel);
    setConvs((l) => [c, ...l.filter((x) => Object.keys(x.nodes).length > 0)]);
    setCurrentId(c.id);
    setTab("chat");
    if (window.innerWidth <= 800) setSidebar(false);
  };
  const deleteChat = (id: string) => {
    setConvs((l) => l.filter((c) => c.id !== id));
    if (id === currentId) setCurrentId(null);
  };
  const pickModel = (id: string) => {
    if (busy) return;
    if (conv && Object.keys(conv.nodes).length === 0) mutate(conv.id, (c) => ({ ...c, model: id }));
    else newChat(id);
  };

  const setP = <K extends keyof GenParams>(k: K, v: GenParams[K]) => setParams((p) => ({ ...p, [k]: v }));
  const activeModelId = conv?.model ?? defaultModel;

  // ------------------------------------------------------------------ render
  return (
    <div className={"app" + (sidebar ? " with-sidebar" : "") + (panel && tab === "chat" ? " with-panel" : "")}>
      <aside className="sidebar" aria-label="Conversations">
        <div className="brand">
          <span className="logo" aria-hidden>
            <svg viewBox="0 0 32 32"><defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stopColor="var(--accent)" /><stop offset="1" stopColor="var(--accent-2)" /></linearGradient></defs>
              <path d="M16 3 L28 26 H4 Z" fill="none" stroke="url(#g)" strokeWidth="2.6" strokeLinejoin="round" /><circle cx="16" cy="19" r="3.2" fill="url(#g)" /></svg>
          </span>
          <span className="brand-name">Amir<b>GPT</b></span>
        </div>
        <button className="btn primary block" onClick={() => newChat()} disabled={busy}>+ New chat</button>
        <div className="model-list" role="radiogroup" aria-label="Model">
          {models.map((m) => (
            <button key={m.id} role="radio" aria-checked={activeModelId === m.id}
                    className={"model-item" + (activeModelId === m.id ? " active" : "")} disabled={!m.available || busy}
                    onClick={() => pickModel(m.id)} title={m.description}>
              <span className="model-name">{m.name}</span>
              <span className="model-tag">{!m.available ? "training…" : m.mode === "chat" ? "chat" : "completion"}</span>
            </button>
          ))}
          {modelErr && <div className="error small">Backend unreachable: {modelErr}</div>}
        </div>
        <div className="conv-list">
          {convs.filter((c) => Object.keys(c.nodes).length > 0).map((c) => (
            <div key={c.id} className={"conv-item" + (c.id === currentId ? " active" : "")}>
              <button className="conv-title" onClick={() => { setCurrentId(c.id); setTab("chat"); if (window.innerWidth <= 800) setSidebar(false); }}>
                {c.title}
              </button>
              <button className="conv-del" aria-label="delete conversation" onClick={() => deleteChat(c.id)}>×</button>
            </div>
          ))}
        </div>
        <div className="sidebar-foot muted small">GPT-2 124M · built from scratch</div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="icon-btn" onClick={() => setSidebar((s) => !s)} aria-label="toggle sidebar">☰</button>
          <nav className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "chat"} className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>Chat</button>
            <button role="tab" aria-selected={tab === "lab"} className={tab === "lab" ? "active" : ""} onClick={() => setTab("lab")}>Lab</button>
          </nav>
          <div className="topbar-right">
            {tab === "chat" && (
              <label className="toggle" title="Tint every token by how confident the model was, and show alternatives on hover">
                <input type="checkbox" checked={lens} onChange={(e) => setLens(e.target.checked)} />
                <span className="toggle-track"><span className="toggle-thumb" /></span>
                Confidence lens
              </label>
            )}
            <button className="icon-btn" aria-label="toggle theme" title="theme"
                    onClick={() => setTheme((t) => (t === "" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark") : t === "dark" ? "light" : "dark"))}>◐</button>
            {tab === "chat" && <button className="icon-btn" aria-label="toggle controls" onClick={() => setPanel((p) => !p)}>⚙</button>}
          </div>
        </header>

        {tab === "lab" ? <Lab /> : (
          <>
            <div className="scroll" ref={scrollRef}
                 onScroll={(e) => { const el = e.currentTarget; stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}>
              {path.length === 0 ? (
                <div className="hero">
                  <h1>Amir<b>GPT</b></h1>
                  <p className="muted">
                    {modelInfo ? modelInfo.description : "A GPT-2 (124M) reproduced from scratch."}
                    {mode === "complete" ? " Type the start of a text and it will continue it." : ""}
                  </p>
                  <p className="hint">With the <b>confidence lens</b> on, warm tokens are ones the model was unsure about.
                    Hover over any token to see what else it considered, and click an alternative to <b>fork</b> the reply from there.</p>
                  <div className="suggestions">
                    {SUGGESTIONS[mode].map((s) => (
                      <button key={s} className="suggestion" disabled={busy || !modelInfo?.available} onClick={() => send(s)}>{s}</button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className={"thread" + (mode === "complete" ? " doc" : "")}>
                  {path.map((n) => {
                    const sibs = siblings(conv!, n);
                    return (
                      <Message key={n.id} node={n} mode={mode} lens={lens} busy={busy}
                               siblingIndex={sibs.indexOf(n.id)} siblingCount={sibs.length}
                               onSwitch={(d) => mutate(conv!.id, (c) => switchSibling(c, c.nodes[n.id], d))}
                               onFork={(i, alt) => fork(n, i, alt)} onRegenerate={() => regenerate(n)}
                               onContinue={() => cont(n)} onEdit={(t) => edit(n, t)} onStop={stop} />
                    );
                  })}
                </div>
              )}
            </div>
            {mode === "complete" && modelInfo?.available && (
              <div className="mode-banner" role="note">
                <b>{modelInfo.name} is a base model.</b> It continues your text like an autocomplete; it does not answer you.
                {models.find((m) => m.mode === "chat")?.available
                  ? <> For a conversation, switch to <button className="link inline" onClick={() => pickModel(models.find((m) => m.mode === "chat")!.id)}>AmirGPT Chat</button>.</>
                  : <> Replying in conversation needs chat fine-tuning: <b>AmirGPT Chat</b> appears once it finishes training.</>}
              </div>
            )}
            <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input); }}>
              <textarea value={input} placeholder={!modelInfo?.available ? "This model is still training…" : mode === "chat" ? "Message AmirGPT…" : "Start a text for the model to continue…"}
                        rows={Math.min(8, Math.max(1, input.split("\n").length))} disabled={!modelInfo?.available}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(input); } }} />
              {busy ? <button type="button" className="btn" onClick={stop}>Stop</button>
                    : <button type="submit" className="btn primary" disabled={!input.trim() || !modelInfo?.available}>Send</button>}
            </form>
            <div className="disclaimer muted small">AmirGPT is a 124M-parameter model and often makes things up. Don't rely on it for facts.</div>
          </>
        )}
      </main>

      {tab === "chat" && (
        <aside className="panel" aria-label="Generation controls">
          <h3>Sampling</h3>
          <Slider label="Temperature" hint="0 = always the most likely token; higher = more random" min={0} max={2} step={0.05} value={params.temperature} onChange={(v) => setP("temperature", v)} />
          <Slider label="Top-k" hint="sample only from the k most likely tokens (0 = off)" min={0} max={200} step={1} value={params.top_k} onChange={(v) => setP("top_k", v)} />
          <Slider label="Top-p" hint="nucleus: smallest set of tokens with this total probability" min={0.05} max={1} step={0.01} value={params.top_p} onChange={(v) => setP("top_p", v)} />
          <Slider label="Repetition penalty" hint="discourage repeating earlier tokens" min={1} max={2} step={0.01} value={params.repetition_penalty} onChange={(v) => setP("repetition_penalty", v)} />
          <Slider label="Max new tokens" hint="upper limit on reply length" min={16} max={768} step={16} value={params.max_new_tokens} onChange={(v) => setP("max_new_tokens", v)} />
          <label className="field">
            <span>Seed <em className="muted">(fixed = reproducible)</em></span>
            <div className="row">
              <input type="number" value={params.seed ?? ""} placeholder="random" onChange={(e) => setP("seed", e.target.value === "" ? null : Number(e.target.value))} />
              <button className="btn ghost" onClick={() => setP("seed", null)}>random</button>
            </div>
          </label>
          <button className="btn ghost block" onClick={() => setParams(DEFAULT_PARAMS)}>Reset to defaults</button>
          {mode === "chat" && conv && (
            <label className="field">
              <span>System prompt <em className="muted">(this chat)</em></span>
              <textarea rows={3} value={conv.systemPrompt} placeholder="e.g. Answer in one short sentence."
                        onChange={(e) => mutate(conv.id, (c) => ({ ...c, systemPrompt: e.target.value }))} />
            </label>
          )}
          <h3>Legend</h3>
          <div className="legend-scale"><span>confident</span><i /><span>unsure</span></div>
          <p className="muted small">Each token's tint is 1 − p, where p is the probability the model gave to the token it produced.
            The line under each reply (the confidence river) plots p across the reply. Click it to inspect any token.</p>
          {path.length > 0 && (() => {
            const last = [...path].reverse().find((n) => n.role === "assistant");
            return last ? <p className="muted small">Visible reply: {nodeText(last).length} characters, {last.tokens.length} tokens.</p> : null;
          })()}
        </aside>
      )}
    </div>
  );
}

function Slider({ label, hint, min, max, step, value, onChange }: {
  label: string; hint: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="field" title={hint}>
      <span>{label} <b className="mono">{Number.isInteger(step) ? value : value.toFixed(2)}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <em className="muted small">{hint}</em>
    </label>
  );
}
