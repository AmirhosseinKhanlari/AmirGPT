// Export a conversation (active path as Markdown, whole tree as JSON), export all
// conversations, and import JSON with shape validation.

import type { Alt, DoneEvent, Tok } from "./api";
import type { ChatNode, Conversation, NodeId } from "./tree";
import { activePath, nodeText, uid } from "./tree";

export const FORMAT_ONE = "amirgpt.conversation";
export const FORMAT_ALL = "amirgpt.conversations";
export const FORMAT_VERSION = 1;

type ModelLite = { id: string; name: string; mode: "chat" | "complete" };

// ------------------------------------------------------------------ download
export function download(filename: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function fileStem(title: string): string {
  const slug = title.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
  const d = new Date();
  const ymd = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
  return `amirgpt-${slug || "chat"}-${ymd}`;
}

// ------------------------------------------------------------------ export
/** The visible (active) path of a conversation as a Markdown document. */
export function conversationToMarkdown(c: Conversation, model?: ModelLite): string {
  const mode = model?.mode ?? "chat";
  const name = model?.name ?? c.model;
  const lines: string[] = [`# ${c.title}`, ""];
  lines.push(`- Model: ${name} (\`${c.model}\`, ${mode === "chat" ? "chat" : "completion"} mode)`);
  lines.push(`- Exported: ${new Date().toISOString()}`);
  if (mode === "chat" && c.systemPrompt.trim()) lines.push(`- System prompt: ${c.systemPrompt.trim().replace(/\n+/g, " ")}`);
  lines.push("");
  for (const n of activePath(c)) {
    const heading = mode === "chat" ? (n.role === "user" ? "User" : name) : n.role === "user" ? "Prompt" : `Continuation (${name})`;
    lines.push(`## ${heading}`, "");
    lines.push(nodeText(n).trim() || (n.role === "assistant" ? "_(empty reply)_" : ""));
    if (n.role === "assistant" && n.error) lines.push("", `> Error: ${n.error}`);
    lines.push("");
  }
  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trimEnd() + "\n";
}

export function conversationToJSON(c: Conversation): string {
  return JSON.stringify({ format: FORMAT_ONE, version: FORMAT_VERSION, exportedAt: new Date().toISOString(), conversation: c }, null, 2);
}

export function allToJSON(list: Conversation[]): string {
  return JSON.stringify({ format: FORMAT_ALL, version: FORMAT_VERSION, exportedAt: new Date().toISOString(), conversations: list }, null, 2);
}

// ------------------------------------------------------------------ import / validation
class ShapeError extends Error {}
const fail = (where: string, what: string): never => { throw new ShapeError(`${where}: ${what}`); };
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isInt = (v: unknown): v is number => Number.isInteger(v);

function str(o: Record<string, unknown>, k: string, where: string, optional?: string): string {
  const v = o[k];
  if (v === undefined && optional !== undefined) return optional;
  return typeof v === "string" ? v : fail(where, `"${k}" must be a string`);
}

function alt(v: unknown, where: string): Alt {
  if (!isObj(v) || !isInt(v.id) || typeof v.text !== "string" || !isNum(v.prob) || v.prob < 0 || v.prob > 1)
    fail(where, "alternative must be {id: integer, text: string, prob: 0..1}");
  const o = v as Record<string, unknown>;
  return { id: o.id as number, text: o.text as string, prob: o.prob as number };
}

function tok(v: unknown, where: string): Tok {
  const t = alt(v, where); // same id/text/prob shape
  const o = v as Record<string, unknown>;
  if (!Array.isArray(o.alternatives)) fail(where, "token needs an alternatives array");
  return { ...t, alternatives: (o.alternatives as unknown[]).map((a, k) => alt(a, `${where}.alternatives[${k}]`)) };
}

function done(v: unknown, where: string): DoneEvent | undefined {
  if (v === undefined || v === null) return undefined;
  if (!isObj(v) || v.type !== "done" || (v.reason !== "stop" && v.reason !== "length") || typeof v.tail !== "string"
      || !isNum(v.tokens) || !isNum(v.tokens_per_sec) || !isNum(v.prompt_tokens))
    fail(where, "malformed done event");
  const o = v as Record<string, unknown>;
  return { type: "done", reason: o.reason as DoneEvent["reason"], tail: o.tail as string, tokens: o.tokens as number,
           tokens_per_sec: o.tokens_per_sec as number, prompt_tokens: o.prompt_tokens as number };
}

function node(key: string, v: unknown, where: string): ChatNode {
  if (!isObj(v)) fail(where, "node must be an object");
  const o = v as Record<string, unknown>;
  if (o.id !== key) fail(where, `node id ${JSON.stringify(o.id)} does not match its key`);
  if (o.role !== "user" && o.role !== "assistant") fail(where, `role must be "user" or "assistant"`);
  if (o.parent !== null && typeof o.parent !== "string") fail(where, "parent must be a node id or null");
  if (!Array.isArray(o.tokens)) fail(where, "tokens must be an array");
  if (!Array.isArray(o.children) || !(o.children as unknown[]).every((x) => typeof x === "string")) fail(where, "children must be an array of node ids");
  if (!isInt(o.active) || (o.active as number) < 0) fail(where, "active must be a non-negative integer");
  let fork: ChatNode["fork"];
  if (o.fork !== undefined && o.fork !== null) {
    if (!isObj(o.fork) || typeof o.fork.from !== "string" || !isInt(o.fork.at) || (o.fork.at as number) < 0) fail(where, "malformed fork");
    const f = o.fork as Record<string, unknown>;
    fork = { from: f.from as string, at: f.at as number };
  }
  const n: ChatNode = {
    id: key,
    parent: o.parent as NodeId | null,
    role: o.role as ChatNode["role"],
    content: str(o, "content", where, ""),
    tokens: (o.tokens as unknown[]).map((t, k) => tok(t, `${where}.tokens[${k}]`)),
    tail: str(o, "tail", where, ""),
    children: o.children as NodeId[],
    active: o.active as number,
    createdAt: isNum(o.createdAt) ? o.createdAt : Date.now(),
  };
  if (fork) n.fork = fork;
  const d = done(o.done, `${where}.done`);
  if (d) n.done = d;
  if (o.error !== undefined && o.error !== null) n.error = str(o, "error", where);
  return n;
}

/** Validate one conversation object. Returns a clean copy with a fresh id. Throws ShapeError. */
export function validateConversation(v: unknown, where = "conversation"): Conversation {
  if (!isObj(v)) fail(where, "must be an object");
  const o = v as Record<string, unknown>;
  const title = str(o, "title", where);
  const model = str(o, "model", where);
  if (!isObj(o.nodes)) fail(where, "nodes must be an object keyed by node id");
  if (!Array.isArray(o.roots) || !(o.roots as unknown[]).every((x) => typeof x === "string")) fail(where, "roots must be an array of node ids");
  const nodes: Record<NodeId, ChatNode> = {};
  for (const [k, nv] of Object.entries(o.nodes as Record<string, unknown>)) {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(k)) fail(where, `invalid node id ${JSON.stringify(k.slice(0, 70))}`);
    nodes[k] = node(k, nv, `${where}.nodes[${k}]`);
  }
  const roots = o.roots as NodeId[];
  // structure: parents and children agree, every node is reachable exactly once from the roots
  for (const r of roots) {
    if (!nodes[r]) fail(where, `root ${r} is not a node`);
    if (nodes[r].parent !== null) fail(where, `root ${r} has a parent`);
  }
  for (const n of Object.values(nodes)) {
    if (n.parent !== null && !nodes[n.parent]) fail(where, `node ${n.id} has a missing parent ${n.parent}`);
    for (const ch of n.children) {
      if (!nodes[ch]) fail(where, `node ${n.id} has a missing child ${ch}`);
      if (nodes[ch].parent !== n.id) fail(where, `child ${ch} does not point back to ${n.id}`);
    }
    if (n.children.length && n.active >= n.children.length) n.active = n.children.length - 1;
    if (!n.children.length) n.active = 0;
  }
  const seen = new Set<NodeId>();
  const stack = [...roots];
  while (stack.length) {
    const id = stack.pop()!;
    if (seen.has(id)) fail(where, `node ${id} is reachable twice (cycle or shared child)`);
    seen.add(id);
    stack.push(...nodes[id].children);
  }
  if (seen.size !== Object.keys(nodes).length) fail(where, "some nodes are not reachable from the roots");
  if (roots.length === 0) fail(where, "conversation is empty");
  const activeRoot = isInt(o.activeRoot) ? Math.max(0, Math.min(roots.length - 1, o.activeRoot as number)) : 0;
  const now = Date.now();
  return {
    id: uid(),
    title: title.trim() || "Imported chat",
    model,
    systemPrompt: str(o, "systemPrompt", where, ""),
    nodes,
    roots,
    activeRoot,
    createdAt: isNum(o.createdAt) ? o.createdAt : now,
    updatedAt: now,
  };
}

/**
 * Parse an exported file. Accepts a single-conversation export, an all-conversations export,
 * or a bare conversation / array of conversations. Throws an Error with a readable message.
 */
export function parseImport(text: string): Conversation[] {
  let data: unknown;
  try { data = JSON.parse(text); } catch { throw new Error("Not a JSON file."); }
  try {
    if (isObj(data) && data.format === FORMAT_ONE) return [validateConversation(data.conversation)];
    if (isObj(data) && data.format === FORMAT_ALL) {
      if (!Array.isArray(data.conversations)) fail("file", "conversations must be an array");
      const list = (data.conversations as unknown[]).map((c, i) => validateConversation(c, `conversations[${i}]`));
      if (!list.length) fail("file", "no conversations in this file");
      return list;
    }
    if (Array.isArray(data)) return data.map((c, i) => validateConversation(c, `conversations[${i}]`));
    if (isObj(data) && "nodes" in data) return [validateConversation(data)];
    throw new ShapeError("This is not an AmirGPT conversation export.");
  } catch (e) {
    if (e instanceof ShapeError) throw new Error(`Invalid file: ${e.message}`);
    throw e;
  }
}
