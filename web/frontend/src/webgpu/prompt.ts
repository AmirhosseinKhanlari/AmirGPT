// Port of sft/chat_format.py (special tokens, render_prompt) and web/server.py _build_prompt.
// Slicing uses Array.prototype.slice(-n), which matches Python's lst[-n:] for every integer n
// (including n <= 0), so truncation edge cases behave exactly like the server.

import type { GenerateRequest } from "../api";
import type { Tokenizer } from "./tokenizer";

export const EOT = 50256;
export const USER = 50257;
export const ASSISTANT = 50258;
export const SYSTEM = 50259;
export const END = 50260;
export const VOCAB_SIZE = 50304;
export const TOKENIZER_VOCAB = 50257; // tiktoken gpt2 n_vocab; ids >= this have no BPE bytes

export const SPECIAL_NAMES: Record<number, string> = {
  [EOT]: "<|endoftext|>",
  [USER]: "<|user|>",
  [ASSISTANT]: "<|assistant|>",
  [SYSTEM]: "<|system|>",
  [END]: "<|end|>",
};

const ROLE_TOKENS: Record<string, number> = { user: USER, assistant: ASSISTANT, system: SYSTEM };

function range(a: number, b: number): number[] {
  const out: number[] = [];
  for (let i = a; i < b; i++) out.push(i);
  return out;
}

/** ids the assistant must never emit while answering (unused padding + other role markers). */
export function bannedWhileAnswering(vocabSize = VOCAB_SIZE): number[] {
  return [EOT, USER, ASSISTANT, SYSTEM, ...range(END + 1, vocabSize)];
}

export type ChatMessage = { role: string; content: string };

export function encodeMessage(tok: Tokenizer, role: string, content: string): number[] {
  const r = ROLE_TOKENS[role];
  if (r === undefined) throw new Error(`unknown role ${JSON.stringify(role)}`);
  return [r, ...tok.encodeOrdinary(content), END];
}

/** chat_format.render_prompt: conversation + "<|assistant|>", oldest turns dropped to fit max_tokens - reserve. */
export function renderPrompt(
  tok: Tokenizer,
  messages: ChatMessage[],
  systemPrompt: string | null | undefined,
  maxTokens = 1024,
  reserve = 256,
): number[] {
  let sysMsgs = messages.filter((m) => m.role === "system");
  if (systemPrompt) sysMsgs = [{ role: "system", content: systemPrompt }];
  const turns = messages.filter((m) => m.role !== "system");
  const budget = maxTokens - reserve;
  const head = [EOT];
  for (const m of sysMsgs) head.push(...encodeMessage(tok, "system", m.content));
  let encoded = turns.map((m) => encodeMessage(tok, m.role, m.content));
  let toks: number[];
  for (;;) {
    toks = [...head, ...encoded.flat(), ASSISTANT];
    if (toks.length <= budget || encoded.length <= 1) break;
    encoded = encoded.slice(1); // drop the oldest turn
  }
  if (toks.length > budget) {
    // a single giant message: keep its tail
    toks = [...toks.slice(0, head.length), ...toks.slice(-(budget - head.length))];
  }
  return toks;
}

export type BuiltPrompt = { ids: number[]; stop: number[]; banned: number[] };

/** web/server.py _build_prompt: prompt ids (+ prefix_ids for forking), stop and banned tokens. */
export function buildPrompt(
  tok: Tokenizer,
  req: GenerateRequest,
  mode: "chat" | "complete",
  blockSize: number,
  vocabSize: number,
): BuiltPrompt {
  const prefix = req.prefix_ids ?? [];
  const reserve = Math.min(req.params.max_new_tokens, Math.floor(blockSize / 2));
  let ids: number[];
  let stop: number[];
  let banned: number[];
  if (mode === "chat") {
    const msgs = (req.messages ?? [])
      .filter((m) => m.role === "user" || m.role === "assistant" || m.role === "system")
      .map((m) => ({ role: m.role, content: m.content }));
    if (msgs.length === 0 || msgs[msgs.length - 1].role !== "user") {
      throw new Error("chat mode needs the conversation to end with a user message");
    }
    ids = renderPrompt(tok, msgs, req.system_prompt, blockSize, reserve + prefix.length);
    stop = [END];
    banned = bannedWhileAnswering(vocabSize);
  } else {
    ids = [EOT, ...tok.encodeOrdinary(req.prompt ?? "")];
    const budget = blockSize - reserve - prefix.length;
    if (ids.length > budget) ids = ids.slice(-budget);
    stop = [EOT];
    banned = range(TOKENIZER_VOCAB, vocabSize);
  }
  if (prefix.some((t) => !Number.isInteger(t) || t < 0 || t >= vocabSize)) {
    throw new Error("invalid prefix token id");
  }
  return { ids: [...ids, ...prefix], stop, banned };
}
