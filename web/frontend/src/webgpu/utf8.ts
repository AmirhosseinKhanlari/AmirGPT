// Incremental UTF-8 decoder with exactly the semantics of the server's
// codecs.getincrementaldecoder("utf-8")(errors="replace") (CPython unicode_decode_utf8):
//  * invalid bytes become U+FFFD per "maximal subpart" (same as WHATWG TextDecoder), and
//  * an incomplete sequence at the end of a chunk is kept for the next chunk, BUT
//  * unlike TextDecoder, CPython also holds back a trailing "ED A0..BF" (a truncated encoded
//    surrogate) until the next byte arrives, so the U+FFFDs land on the next token.
// Using this instead of TextDecoder keeps per-token text identical to /api/generate.

const REPLACEMENT = "�";

export class PyUtf8Decoder {
  private pending: number[] = [];

  /** Decodes the next chunk (final=false: incomplete trailing bytes are kept). */
  decode(chunk: Uint8Array, final = false): string {
    const buf = this.pending.length ? [...this.pending, ...chunk] : Array.from(chunk);
    this.pending = [];
    const n = buf.length;
    let out = "";
    let i = 0;
    while (i < n) {
      const c = buf[i];
      if (c < 0x80) {
        out += String.fromCharCode(c);
        i++;
        continue;
      }
      let need: number;
      if (c < 0xc2) need = 0; // 80..C1: invalid start byte
      else if (c < 0xe0) need = 1;
      else if (c < 0xf0) need = 2;
      else if (c < 0xf5) need = 3;
      else need = 0; // F5..FF: invalid start byte
      if (need === 0) {
        out += REPLACEMENT;
        i++;
        continue;
      }
      // allowed range of the first continuation byte (no overlongs, surrogates or > U+10FFFF)
      const lo = c === 0xe0 ? 0xa0 : c === 0xf0 ? 0x90 : 0x80;
      const hi = c === 0xed ? 0x9f : c === 0xf4 ? 0x8f : 0xbf;
      if (!final && c === 0xed && n - i === 2 && buf[i + 1] >= 0xa0 && buf[i + 1] <= 0xbf) {
        this.pending = buf.slice(i); // CPython: "Truncated surrogate code in range D800-DFFF"
        return out;
      }
      let j = i + 1;
      let valid = true;
      for (let k = 0; k < need && j < n; k++, j++) {
        const b = buf[j];
        if (k === 0 ? b < lo || b > hi : b < 0x80 || b > 0xbf) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        out += REPLACEMENT; // the lead byte + the valid continuations so far form one maximal subpart
        i = j;
        continue;
      }
      if (j - i === need + 1) {
        let cp = c & (need === 1 ? 0x1f : need === 2 ? 0x0f : 0x07);
        for (let k = i + 1; k < j; k++) cp = (cp << 6) | (buf[k] & 0x3f);
        out += String.fromCodePoint(cp);
        i = j;
        continue;
      }
      // a valid but incomplete sequence at the end of the data
      if (final) {
        out += REPLACEMENT; // "unexpected end of data": one U+FFFD for the remainder
        return out;
      }
      this.pending = buf.slice(i);
      return out;
    }
    return out;
  }

  /** decoder.decode(b"", final=True): flushes whatever is pending. */
  flush(): string {
    return this.decode(new Uint8Array(0), true);
  }
}
