// Conversation tree. Every regenerate, user-message edit, or token-level fork creates a
// sibling node, so nothing is ever lost and every branch can be revisited.

import type { DoneEvent, Tok } from "./api";

export type NodeId = string;

export type ChatNode = {
  id: NodeId;
  parent: NodeId | null; // null => child of the conversation root
  role: "user" | "assistant";
  content: string; // user text (assistant text is derived from tokens)
  tokens: Tok[]; // assistant tokens
  tail: string; // trailing text flushed by the UTF-8 decoder at the end
  children: NodeId[];
  active: number; // index of the active child
  fork?: { from: NodeId; at: number }; // created by forking `from` at token index `at`
  done?: DoneEvent;
  error?: string;
  streaming?: boolean;
  createdAt: number;
};

export type Conversation = {
  id: string;
  title: string;
  model: string;
  systemPrompt: string;
  nodes: Record<NodeId, ChatNode>;
  roots: NodeId[];
  activeRoot: number;
  createdAt: number;
  updatedAt: number;
};

export const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);

export function newConversation(model: string): Conversation {
  const now = Date.now();
  return { id: uid(), title: "New chat", model, systemPrompt: "", nodes: {}, roots: [], activeRoot: 0, createdAt: now, updatedAt: now };
}

export function nodeText(n: ChatNode): string {
  return n.role === "user" ? n.content : n.tokens.map((t) => t.text).join("") + n.tail;
}

/** The currently visible path from the root, following each node's active child. */
export function activePath(c: Conversation): ChatNode[] {
  const path: ChatNode[] = [];
  let ids = c.roots;
  let idx = c.activeRoot;
  while (ids.length > 0) {
    const n = c.nodes[ids[Math.min(idx, ids.length - 1)]];
    if (!n) break;
    path.push(n);
    ids = n.children;
    idx = n.active;
  }
  return path;
}

export function siblings(c: Conversation, n: ChatNode): NodeId[] {
  return n.parent === null ? c.roots : c.nodes[n.parent].children;
}

/** Add `node` as a new child of `parent` (or root) and make it active. Returns a new conversation. */
export function addChild(c: Conversation, parent: NodeId | null, node: ChatNode): Conversation {
  const nodes = { ...c.nodes, [node.id]: node };
  let roots = c.roots;
  let activeRoot = c.activeRoot;
  if (parent === null) {
    roots = [...roots, node.id];
    activeRoot = roots.length - 1;
  } else {
    const p = nodes[parent];
    nodes[parent] = { ...p, children: [...p.children, node.id], active: p.children.length };
  }
  return { ...c, nodes, roots, activeRoot, updatedAt: Date.now() };
}

export function updateNode(c: Conversation, id: NodeId, patch: Partial<ChatNode>): Conversation {
  return { ...c, nodes: { ...c.nodes, [id]: { ...c.nodes[id], ...patch } }, updatedAt: Date.now() };
}

/** Switch which sibling of `n` is visible (delta = -1 / +1). */
export function switchSibling(c: Conversation, n: ChatNode, delta: number): Conversation {
  const sibs = siblings(c, n);
  const cur = sibs.indexOf(n.id);
  const next = Math.max(0, Math.min(sibs.length - 1, cur + delta));
  if (n.parent === null) return { ...c, activeRoot: next };
  return updateNode(c, n.parent, { active: next });
}

/** Messages (role/content) along the active path up to and including `uptoId`. */
export function historyUpTo(c: Conversation, uptoId: NodeId): { role: string; content: string }[] {
  const out: { role: string; content: string }[] = [];
  for (const n of activePath(c)) {
    out.push({ role: n.role, content: nodeText(n) });
    if (n.id === uptoId) break;
  }
  return out;
}

export function makeNode(role: ChatNode["role"], parent: NodeId | null, extra: Partial<ChatNode> = {}): ChatNode {
  return { id: uid(), parent, role, content: "", tokens: [], tail: "", children: [], active: 0, createdAt: Date.now(), ...extra };
}

const KEY = "amirgpt.conversations.v1";

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const list = JSON.parse(raw) as Conversation[];
    // a reload mid-stream leaves nodes flagged as streaming; clear the flag
    for (const c of list) for (const n of Object.values(c.nodes)) n.streaming = false;
    return list;
  } catch {
    return [];
  }
}

export function saveConversations(list: Conversation[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* storage full or unavailable: the app still works in memory */
  }
}
