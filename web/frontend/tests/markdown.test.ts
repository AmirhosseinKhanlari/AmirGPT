// Unit tests for src/markdown.ts. No test framework needed:
//   npm test        (= node --experimental-strip-types tests/markdown.test.ts)
import assert from "node:assert/strict";
import { BOLD, CODE, ITALIC, parseMarkdown } from "../src/markdown.ts";
import type { MdBlock, Run } from "../src/markdown.ts";

let passed = 0;
const failures: string[] = [];
function test(name: string, fn: () => void) {
  try { fn(); passed++; } catch (e) { failures.push(`${name}\n    ${(e as Error).message.split("\n").join("\n    ")}`); }
}
const md = (s: string | string[], streaming = false) => parseMarkdown(Array.isArray(s) ? s : [s], { streaming });
const visible = (b: MdBlock) => b.runs.map((r) => r.text).join("");
const styled = (b: MdBlock, st: number) => b.runs.filter((r) => r.style === st).map((r) => r.text).join("");
const shape = (bs: MdBlock[]) => bs.map((b) => (b.kind === "h" ? `h${b.level}` : b.kind === "li" ? (b.ordered ? `ol${b.num}` : "ul") + (b.indent ? `@${b.indent}` : "") : b.kind));

// ------------------------------------------------------------------ blocks
test("plain paragraph keeps every character and token", () => {
  const bs = md(["Hello", " world", "!"]);
  assert.deepEqual(shape(bs), ["p"]);
  assert.deepEqual(bs[0].runs, [{ tok: 0, text: "Hello", style: 0 }, { tok: 1, text: " world", style: 0 }, { tok: 2, text: "!", style: 0 }]);
});
test("soft line break stays inside the paragraph, owned by its token", () => {
  const bs = md(["line one", "\n", "line two"]);
  assert.deepEqual(shape(bs), ["p"]);
  assert.equal(visible(bs[0]), "line one\nline two");
  assert.deepEqual(bs[0].runs[1], { tok: 1, text: "\n", style: 0 });
});
test("blank line separates paragraphs and is hidden", () => {
  const bs = md("first\n\nsecond");
  assert.deepEqual(shape(bs), ["p", "p"]);
  assert.deepEqual(bs.map(visible), ["first", "second"]);
});
test("headings # to ###, markers hidden; #### is text", () => {
  const bs = md("# One\n## Two\n### Three\n#### Four");
  assert.deepEqual(shape(bs), ["h1", "h2", "h3", "p"]);
  assert.deepEqual(bs.map(visible), ["One", "Two", "Three", "#### Four"]);
});
test("#hashtag is not a heading", () => {
  assert.deepEqual(shape(md("#hashtag")), ["p"]);
});
test("bullet and numbered lists, markers hidden, numbers kept as data", () => {
  const bs = md("Tips:\n1. Sleep well\n2. Take breaks\n3) Review\n- dash\n* star\n+ plus");
  assert.deepEqual(shape(bs), ["p", "ol1", "ol2", "ol3", "ul", "ul", "ul"]);
  assert.deepEqual(bs.map(visible), ["Tips:", "Sleep well", "Take breaks", "Review", "dash", "star", "plus"]);
});
test("only 1. interrupts a paragraph (CommonMark)", () => {
  assert.deepEqual(shape(md("In\n2019. it rained")), ["p"]);
  assert.deepEqual(shape(md("Steps\n1. go")), ["p", "ol1"]);
});
test("lazy continuation line joins the list item", () => {
  const bs = md("- first line\ncontinued");
  assert.deepEqual(shape(bs), ["ul"]);
  assert.equal(visible(bs[0]), "first line\ncontinued");
});
test("nested list indentation levels", () => {
  assert.deepEqual(shape(md("- a\n  - b\n    - c\n- d")), ["ul", "ul@1", "ul@2", "ul"]);
});
test("*italic* at line start is not a bullet", () => {
  const bs = md("*note* this");
  assert.deepEqual(shape(bs), ["p"]);
  assert.equal(styled(bs[0], ITALIC), "note");
});
test("closed fenced code block: fences and language hidden, content verbatim, no inline parsing", () => {
  const bs = md("Code:\n```python\nx = 2 * 3 * 4\n**not bold**\n```\nafter");
  assert.deepEqual(shape(bs), ["p", "code", "p"]);
  const code = bs[1];
  assert.equal(code.kind === "code" && code.lang, "python");
  assert.equal(visible(code), "x = 2 * 3 * 4\n**not bold**");
  assert.ok(code.runs.every((r) => r.style === 0));
  assert.equal(visible(bs[2]), "after");
});
test("~~~ fences work; inline ``` is not a fence", () => {
  assert.deepEqual(shape(md("~~~\na\n~~~")), ["code"]);
  assert.deepEqual(shape(md("```inline``` text")), ["p"]);
});

// ------------------------------------------------------------------ inline
test("bold across tokens, and a token split at the style boundary", () => {
  const bs = md(["Hello", " **wor", "ld** and", " more"]);
  assert.deepEqual(bs[0].runs, [
    { tok: 0, text: "Hello", style: 0 },
    { tok: 1, text: " ", style: 0 },
    { tok: 1, text: "wor", style: BOLD },
    { tok: 2, text: "ld", style: BOLD },
    { tok: 2, text: " and", style: 0 },
    { tok: 3, text: " more", style: 0 },
  ]);
});
test("italic, bold and bold-italic", () => {
  const b = md("a *it* b **bo** c ***bi***")[0];
  assert.equal(visible(b), "a it b bo c bi");
  assert.equal(styled(b, ITALIC), "it");
  assert.equal(styled(b, BOLD), "bo");
  assert.equal(styled(b, BOLD | ITALIC), "bi");
});
test("nested emphasis", () => {
  const b = md("**bold *both* bold**")[0];
  assert.equal(visible(b), "bold both bold");
  assert.equal(styled(b, BOLD), "bold  bold");
  assert.equal(styled(b, BOLD | ITALIC), "both");
});
test("inline code: backticks hidden, content not parsed for emphasis", () => {
  const b = md("run `a * b *c*` now")[0];
  assert.equal(visible(b), "run a * b *c* now");
  assert.equal(styled(b, CODE), "a * b *c*");
});
test("double-backtick code span can contain a backtick", () => {
  const b = md("x ``a`b`` y")[0];
  assert.equal(styled(b, CODE), "a`b");
});
test("spaced asterisks are not emphasis", () => {
  const b = md("2 * 3 * 4")[0];
  assert.equal(visible(b), "2 * 3 * 4");
  assert.ok(b.runs.every((r) => r.style === 0));
});
test("backslash escapes show the literal character", () => {
  const b = md("\\*not italic\\*")[0];
  assert.equal(visible(b), "*not italic*");
  assert.ok(b.runs.every((r) => r.style === 0));
});
test("inline styles inside headings and list items", () => {
  const bs = md("## A **big** deal\n1. Use `git`");
  assert.equal(styled(bs[0], BOLD), "big");
  assert.equal(styled(bs[1], CODE), "git");
});
test("final text: unclosed delimiters are literal", () => {
  assert.equal(visible(md("a **b")[0]), "a **b");
  assert.equal(visible(md("a `b")[0]), "a `b");
});

// ------------------------------------------------------------------ streaming
test("streaming: unclosed bold styles to the end, markers hidden", () => {
  const b = md(["Some", " **bo", "ld"], true)[0];
  assert.equal(visible(b), "Some bold");
  assert.equal(styled(b, BOLD), "bold");
});
test("streaming: half-written closer is hidden, not shown as italic", () => {
  const b = md("x **bold*", true)[0];
  assert.equal(visible(b), "x bold");
  assert.equal(styled(b, BOLD), "bold");
});
test("streaming: trailing * is hidden until the next token", () => {
  assert.equal(visible(md("a *", true)[0]), "a ");
  assert.equal(visible(md("a *", false)[0]), "a *");
});
test("streaming: unclosed inline code styles to the end", () => {
  const b = md("use `np.ar", true)[0];
  assert.equal(styled(b, CODE), "np.ar");
});
test("streaming: unclosed fence is a code block", () => {
  const bs = md("```js\nconst a = 1;\n", true);
  assert.deepEqual(shape(bs), ["code"]);
  assert.equal(visible(bs[0]), "const a = 1;");
});
test("streaming: a just-written list marker is a list item", () => {
  assert.deepEqual(shape(md("Tips:\n1.", true)), ["p", "ol1"]);
  assert.deepEqual(shape(md("Tips:\n- ", true)), ["p", "ul"]);
});
test("streaming: an earlier, closed block is not treated as open", () => {
  const bs = md("a **b\n\nnext", true);
  assert.equal(visible(bs[0]), "a **b");
});
test("tail piece gets its own token index", () => {
  const bs = md(["ab", "c", "�"]);
  assert.deepEqual(bs[0].runs.at(-1), { tok: 2, text: "�", style: 0 });
});

// ------------------------------------------------------------------ invariants (fuzz)
// Every run must be non-empty, and the runs of each token must be an in-order subsequence
// of that token's text: visible characters always belong to the token that produced them.
function checkAttribution(pieces: string[], blocks: MdBlock[]) {
  const pos = pieces.map(() => 0);
  for (const b of blocks) {
    for (const r of b.runs as Run[]) {
      assert.ok(r.text.length > 0, "empty run");
      assert.ok(r.tok >= 0 && r.tok < pieces.length, `bad token index ${r.tok}`);
      // (a run may skip hidden markup inside its token, e.g. "a*b" -> "ab", so: subsequence)
      for (const ch of r.text) {
        const at = pieces[r.tok].indexOf(ch, pos[r.tok]);
        assert.ok(at >= 0, `run ${JSON.stringify(r.text)} is not in token ${r.tok} ${JSON.stringify(pieces[r.tok])} after ${pos[r.tok]}`);
        pos[r.tok] = at + ch.length;
      }
    }
  }
}
test("fuzz: token attribution invariant holds for random markdown token streams", () => {
  const frags = ["Hello", " world", "**", "*", "`", "```", "\n", "\n\n", "# ", "## ", "- ", "1. ", "2. ", " *", "* ",
    "code", " `x`", "\\*", "~~~", "  ", "py\n", "é", "😀", ".", "***", "###", "\t", "a*b", "**c", "d**"];
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let iter = 0; iter < 3000; iter++) {
    const n = 1 + Math.floor(rnd() * 25);
    const pieces = Array.from({ length: n }, () => frags[Math.floor(rnd() * frags.length)]);
    for (const streaming of [false, true]) {
      // every prefix, like a reply growing token by token while streaming
      for (let k = 1; k <= pieces.length; k++) {
        const p = pieces.slice(0, k);
        checkAttribution(p, parseMarkdown(p, { streaming }));
      }
    }
  }
});
test("text without markup is reproduced exactly", () => {
  const pieces = ["The", " quick", " brown", " fox", ".", " It", " jumps", "."];
  const bs = md(pieces);
  assert.equal(bs.map(visible).join(""), pieces.join(""));
});

console.log(`${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.error("FAIL " + f);
  process.exit(1);
}
