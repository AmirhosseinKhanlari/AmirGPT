// GPT-2 BPE tokenizer for the in-browser engine: js-tiktoken (lite) + the gpt2 ranks,
// i.e. the same ranks/merges as Python `tiktoken.get_encoding("gpt2")`.
// This module is only reached through dynamic import() (see runtime.ts), so the ~545 kB
// ranks table is downloaded together with the model, not with the app.

import { Tiktoken } from "js-tiktoken/lite";
import gpt2 from "js-tiktoken/ranks/gpt2";
import { SPECIAL_NAMES, TOKENIZER_VOCAB } from "./prompt";

// tiktoken's gpt2 pattern uses \s / \S. In tiktoken (Rust regex) \s is the Unicode White_Space
// property; JavaScript's \s differs in two code points (adds U+FEFF, lacks U+0085). Spell the
// class out so JS splits text exactly like Python.
const WS = "\\t\\n\\v\\f\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PAT_STR =
  `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^${WS}\\p{L}\\p{N}]+|[${WS}]+(?![^${WS}])|[${WS}]+`;
const ORIGINAL_PAT = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

export interface Tokenizer {
  /** Python `enc.encode_ordinary(text)`: special-token strings are encoded as plain text. */
  encodeOrdinary(text: string): number[];
  /** Python `enc.decode_single_token_bytes(id)` for id < 50257 (50256 -> "<|endoftext|>"). */
  tokenBytes(id: number): Uint8Array;
  /** web/server.py token_text(): special names, "<|unused_N|>", else the token's bytes decoded with replacement. */
  tokenText(id: number): string;
}

export function createTokenizer(): Tokenizer {
  if (gpt2.pat_str !== ORIGINAL_PAT) {
    throw new Error("unexpected gpt2 pattern in js-tiktoken ranks; tokenizer parity is not guaranteed");
  }
  const enc = new Tiktoken({ ...gpt2, pat_str: PAT_STR });

  // id -> bytes table, parsed from the same compressed ranks js-tiktoken uses
  const bytes: Uint8Array[] = new Array(TOKENIZER_VOCAB);
  for (const line of gpt2.bpe_ranks.split("\n")) {
    if (!line) continue;
    const parts = line.split(" ");
    const offset = Number.parseInt(parts[1], 10);
    for (let i = 2; i < parts.length; i++) {
      const bin = atob(parts[i]);
      const b = new Uint8Array(bin.length);
      for (let j = 0; j < bin.length; j++) b[j] = bin.charCodeAt(j);
      bytes[offset + i - 2] = b;
    }
  }
  for (const [text, id] of Object.entries(gpt2.special_tokens)) bytes[id] = new TextEncoder().encode(text);
  for (let i = 0; i < TOKENIZER_VOCAB; i++) {
    if (!bytes[i]) throw new Error(`tokenizer: missing bytes for id ${i}`);
  }

  const single = new TextDecoder("utf-8"); // non-fatal: invalid bytes -> U+FFFD, like errors="replace"
  return {
    encodeOrdinary: (text) => enc.encode(text, [], []),
    tokenBytes: (id) => {
      const b = bytes[id];
      if (!b) throw new Error(`token id ${id} has no BPE bytes`);
      return b;
    },
    tokenText: (id) => {
      const name = SPECIAL_NAMES[id];
      if (name !== undefined) return name;
      if (id >= TOKENIZER_VOCAB) return `<|unused_${id}|>`;
      return single.decode(bytes[id]);
    },
  };
}
