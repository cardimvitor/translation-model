/* Byte-pair (word-piece) tokenizer used by the v5 students. Same algorithm as bpe.py (checked by test/parity_bpe.js).
 * Text is already normalized (NFKC, lowercase, single spaces). Each space-separated word starts with "▁"; words
 * longer than CHUNK code points are cut into chunks (only the first carries the mark). Merges are applied lowest
 * rank first, every non-overlapping occurrence left to right, until no ranked pair is left. */
(function (root) {
  "use strict";
  const MARK = "▁", CHUNK = 48;
  const cache = new WeakMap();
  function ranks(merges) {
    let R = cache.get(merges);
    if (!R) { R = new Map(); merges.forEach((m, i) => R.set(m[0] + "\u0000" + m[1], i)); cache.set(merges, R); }
    return R;
  }
  function word(sym, R) {
    while (sym.length > 1) {
      let best = -1, bi = -1;
      for (let i = 0; i < sym.length - 1; i++) {
        const r = R.get(sym[i] + "\u0000" + sym[i + 1]);
        if (r !== undefined && (best < 0 || r < best)) { best = r; bi = i; }
      }
      if (best < 0) break;
      const a = sym[bi], b = sym[bi + 1], out = [];
      for (let i = 0; i < sym.length; ) {
        if (i < sym.length - 1 && sym[i] === a && sym[i + 1] === b) { out.push(a + b); i += 2; }
        else { out.push(sym[i]); i += 1; }
      }
      sym = out;
    }
    return sym;
  }
  function encode(text, merges) {
    const R = ranks(merges); const toks = [];
    for (const w of text.split(" ")) {
      if (!w) continue;
      const cps = Array.from(w);
      for (let j = 0; j < cps.length; j += CHUNK) {
        let sym = cps.slice(j, j + CHUNK);
        if (j === 0) sym = [MARK].concat(sym);
        for (const t of word(sym, R)) toks.push(t);
      }
    }
    return toks;
  }
  function decode(tokens) {
    return tokens.join("").split(MARK).join(" ").split(/\s+/).filter(Boolean).join(" ");
  }
  const api = { encode, decode, MARK, CHUNK };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.BPE = api;
})(typeof self !== "undefined" ? self : this);
