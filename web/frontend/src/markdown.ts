// Tiny dependency-free markdown parser that keeps token identity.
//
// The input is the list of token texts of a reply. The output is a list of blocks
// (paragraph, heading, list item, fenced code) whose content is a list of runs. Every run
// covers visible characters of exactly ONE token with ONE style, so the renderer can draw
// each run as that token's span (hover, token card, fork and confidence lens keep working).
// A token whose text crosses a style boundary is split into several runs; markup characters
// (`**`, `#`, list markers, fences, ...) are hidden by simply not appearing in any run.
//
// Supported: paragraphs, soft line breaks, # to ### headings, -, *, + and 1. / 1) list items,
// **bold**, *italic*, `inline code`, fenced ``` / ~~~ code blocks, and backslash escapes.
// While streaming, unclosed constructs in the last block are treated as open (styled to the
// end, markers hidden) so formatting does not flicker; once the reply is final, unclosed
// delimiters are shown literally, as CommonMark would.

export const BOLD = 1;
export const ITALIC = 2;
export const CODE = 4;

/** Visible text of one token (`tok` = index into the input pieces) in one style. */
export type Run = { tok: number; text: string; style: number };

export type MdBlock =
  | { kind: "p"; runs: Run[] }
  | { kind: "h"; level: 1 | 2 | 3; runs: Run[] }
  | { kind: "li"; ordered: boolean; num: number; indent: number; runs: Run[] }
  | { kind: "code"; lang: string; runs: Run[] };

type Range = [number, number]; // [start, end) char offsets into the full text
type RawBlock =
  | { kind: "p"; ranges: Range[] }
  | { kind: "h"; level: 1 | 2 | 3; ranges: Range[] }
  | { kind: "li"; ordered: boolean; num: number; indent: number; ranges: Range[] }
  | { kind: "code"; lang: string; ranges: Range[] };

const FENCE_OPEN = /^( {0,3})(`{3,}|~{3,})(.*)$/;
const HEADING = /^ {0,3}(#{1,3})([ \t]+|$)/;
const LIST_ITEM = /^([ \t]*)([-*+]|\d{1,9}[.)])([ \t]+|$)/;
const BLANK = /^[ \t]*$/;
const PUNCT = /[\p{P}\p{S}]/u;
const WS = /\s/;
const ESCAPABLE = /[!-/:-@[-`{-~]/; // ASCII punctuation

const indentWidth = (s: string) => [...s].reduce((w, c) => w + (c === "\t" ? 4 : 1), 0);

/** Split the text into block-level structures (ranges of content characters). */
function parseBlocks(text: string, streaming: boolean): RawBlock[] {
  const lines: Range[] = [];
  for (let s = 0; ;) {
    const e = text.indexOf("\n", s);
    if (e < 0) { lines.push([s, text.length]); break; }
    lines.push([s, e]);
    s = e + 1;
  }
  const blocks: RawBlock[] = [];
  let cur: (RawBlock & { kind: "p" | "li" }) | null = null; // block accepting continuation lines
  let listIndents: number[] = [];
  const line = (i: number) => text.slice(lines[i][0], lines[i][1]);
  const leadingWs = (s: string) => s.length - s.replace(/^[ \t]+/, "").length;

  for (let i = 0; i < lines.length;) {
    const [ls, le] = lines[i];
    const s = line(i);

    // fenced code block: everything up to the closing fence (or the end, while it is open)
    const fm = FENCE_OPEN.exec(s);
    if (fm && !(fm[2][0] === "`" && fm[3].includes("`"))) {
      const ch = fm[2][0], len = fm[2].length;
      const close = new RegExp(`^ {0,3}\\${ch}{${len},}[ \\t]*$`);
      let j = i + 1;
      while (j < lines.length && !close.test(line(j))) j++;
      const closed = j < lines.length;
      const start = i + 1 < lines.length ? lines[i + 1][0] : text.length;
      let end = closed ? lines[j][0] - 1 : text.length;
      if (!closed && end > start && text[end - 1] === "\n") end--; // the newline before a fence still to come
      blocks.push({ kind: "code", lang: fm[3].trim(), ranges: [[start, Math.max(start, end)]] });
      cur = null; listIndents = [];
      i = closed ? j + 1 : lines.length;
      continue;
    }
    if (BLANK.test(s)) { cur = null; i++; continue; }

    const hm = HEADING.exec(s);
    if (hm) {
      blocks.push({ kind: "h", level: hm[1].length as 1 | 2 | 3, ranges: [[ls + hm[0].length, le]] });
      cur = null; listIndents = [];
      i++;
      continue;
    }

    const lm = LIST_ITEM.exec(s);
    if (lm) {
      const rest = s.slice(lm[0].length);
      const ordered = /\d/.test(lm[2][0]);
      const num = ordered ? parseInt(lm[2], 10) : 0;
      // a marker the model has just written, with its content still to come
      const typing = streaming && i === lines.length - 1 && BLANK.test(rest);
      // like CommonMark: a marker needs a following space; only "1." (or a bullet) may
      // interrupt a paragraph, and an empty item may not
      const interrupts = cur?.kind !== "p" || ((!ordered || num === 1) && (!BLANK.test(rest) || typing));
      if ((lm[3] !== "" || typing) && interrupts) {
        const w = indentWidth(lm[1]);
        while (listIndents.length && w < listIndents[listIndents.length - 1]) listIndents.pop();
        if (!listIndents.length || w > listIndents[listIndents.length - 1]) listIndents.push(w);
        cur = { kind: "li", ordered, num, indent: Math.min(listIndents.length - 1, 4), ranges: [[ls + lm[0].length, le]] };
        blocks.push(cur);
        i++;
        continue;
      }
    }

    // plain text: continues the open paragraph / list item (soft break), or starts a paragraph
    const start = ls + leadingWs(s);
    if (cur) {
      cur.ranges.push([ls - 1, ls]); // the newline between the lines stays visible as a line break
      cur.ranges.push([start, le]);
    } else {
      cur = { kind: "p", ranges: [[start, le]] };
      blocks.push(cur);
      listIndents = [];
    }
    i++;
  }
  return blocks;
}

/**
 * Resolve inline styles for the characters of one block. Returns, per visible-candidate
 * position, a style bitmask, or -1 for hidden markup.
 */
function parseInline(chars: string[], open: boolean): number[] {
  const n = chars.length;
  const style = new Array<number>(n).fill(0);
  const HIDDEN = -1;
  const isProtected = new Array<boolean>(n).fill(false);

  // pass 1: backslash escapes and code spans (they bind tighter than emphasis)
  for (let k = 0; k < n;) {
    if (chars[k] === "\\" && k + 1 < n && ESCAPABLE.test(chars[k + 1])) {
      style[k] = HIDDEN;
      isProtected[k] = isProtected[k + 1] = true;
      k += 2;
      continue;
    }
    if (chars[k] !== "`") { k++; continue; }
    let r = 0;
    while (k + r < n && chars[k + r] === "`") r++;
    let m = k + r, closeAt = -1;
    while (m < n) {
      if (chars[m] !== "`") { m++; continue; }
      let r2 = 0;
      while (m + r2 < n && chars[m + r2] === "`") r2++;
      if (r2 === r) { closeAt = m; break; }
      m += r2;
    }
    const end = closeAt >= 0 ? closeAt : open ? n : -1;
    if (end < 0) { // never closed: literal backticks
      for (let q = k; q < k + r; q++) isProtected[q] = true;
      k += r;
      continue;
    }
    for (let q = k; q < k + r; q++) { style[q] = HIDDEN; isProtected[q] = true; }
    for (let q = k + r; q < end; q++) { style[q] |= CODE; isProtected[q] = true; }
    if (closeAt >= 0) for (let q = closeAt; q < closeAt + r; q++) { style[q] = HIDDEN; isProtected[q] = true; }
    k = end + (closeAt >= 0 ? r : 0);
  }

  // pass 2: * emphasis, a simplified version of CommonMark's delimiter algorithm
  type Delim = { pos: number; len: number; orig: number; canOpen: boolean; canClose: boolean };
  const delims: Delim[] = [];
  for (let k = 0; k < n;) {
    if (chars[k] !== "*" || isProtected[k]) { k++; continue; }
    let r = 0;
    while (k + r < n && chars[k + r] === "*" && !isProtected[k + r]) r++;
    const prev = k > 0 ? chars[k - 1] : " ";
    const next = k + r < n ? chars[k + r] : " ";
    const leftFlanking = !WS.test(next) && (!PUNCT.test(next) || WS.test(prev) || PUNCT.test(prev));
    const rightFlanking = !WS.test(prev) && (!PUNCT.test(prev) || WS.test(next) || PUNCT.test(next));
    if (open && k + r === n) {
      // a run at the very end of a streaming reply is probably the start of a delimiter: hide it
      for (let q = k; q < n; q++) style[q] = HIDDEN;
      break;
    }
    delims.push({ pos: k, len: r, orig: r, canOpen: leftFlanking, canClose: rightFlanking });
    k += r;
  }
  const stack: Delim[] = [];
  const paint = (from: number, to: number, bit: number) => {
    for (let q = from; q < to; q++) if (style[q] !== HIDDEN) style[q] |= bit;
  };
  for (const d of delims) {
    if (d.canClose) {
      while (d.len > 0) {
        let j = stack.length - 1;
        for (; j >= 0; j--) {
          const o = stack[j];
          const ruleOf3 = (o.canClose || d.canOpen) && (o.orig + d.orig) % 3 === 0 && !(o.orig % 3 === 0 && d.orig % 3 === 0);
          if (o.len > 0 && !ruleOf3) break;
        }
        if (j < 0) break;
        const o = stack[j];
        const use = o.len >= 2 && d.len >= 2 ? 2 : 1;
        const innerStart = o.pos + o.len;
        for (let q = innerStart - use; q < innerStart; q++) style[q] = HIDDEN;
        for (let q = d.pos; q < d.pos + use; q++) style[q] = HIDDEN;
        paint(innerStart, d.pos, use === 2 ? BOLD : ITALIC);
        o.len -= use;
        d.pos += use;
        d.len -= use;
        stack.length = j + 1; // openers in between can no longer match
        if (o.len === 0) stack.pop();
      }
    }
    if (d.len > 0 && d.canOpen) stack.push(d);
  }
  if (open) {
    // still streaming: unmatched openers style everything after them
    for (const o of stack) {
      if (o.len <= 0) continue;
      for (let q = o.pos; q < o.pos + o.len; q++) style[q] = HIDDEN;
      paint(o.pos + o.len, n, o.len >= 3 ? BOLD | ITALIC : o.len === 2 ? BOLD : ITALIC);
    }
  }
  return style;
}

/**
 * Parse the concatenation of `pieces` (token texts) into blocks of token-attributed runs.
 * `streaming` = the text may still grow, so incomplete markup at the end is treated as open.
 */
export function parseMarkdown(pieces: string[], opts: { streaming?: boolean } = {}): MdBlock[] {
  const streaming = !!opts.streaming;
  const text = pieces.join("");
  const owner = new Int32Array(text.length);
  for (let i = 0, off = 0; i < pieces.length; i++) {
    owner.fill(i, off, off + pieces[i].length);
    off += pieces[i].length;
  }
  const raw = parseBlocks(text, streaming);
  return raw.map((b, bi) => {
    const idx: number[] = [];
    for (const [s, e] of b.ranges) for (let c = s; c < e; c++) idx.push(c);
    let styles: number[];
    if (b.kind === "code") {
      styles = idx.map(() => 0);
    } else {
      const lastEnd = b.ranges[b.ranges.length - 1][1];
      // the last block is "open" while streaming, unless a blank line already closed it
      const open = streaming && bi === raw.length - 1 && BLANK.test(text.slice(lastEnd).replace(/\n/, ""));
      styles = parseInline(idx.map((c) => text[c]), open);
    }
    const runs: Run[] = [];
    for (let k = 0; k < idx.length; k++) {
      const st = styles[k];
      if (st < 0) continue;
      const c = idx[k], tok = owner[c];
      const last = runs[runs.length - 1];
      if (last && last.tok === tok && last.style === st) last.text += text[c];
      else runs.push({ tok, text: text[c], style: st });
    }
    switch (b.kind) {
      case "p": return { kind: "p", runs };
      case "h": return { kind: "h", level: b.level, runs };
      case "li": return { kind: "li", ordered: b.ordered, num: b.num, indent: b.indent, runs };
      case "code": return { kind: "code", lang: b.lang, runs };
    }
  });
}
