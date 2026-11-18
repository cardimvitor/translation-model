/* SentencePiece unigram encoding in JS. Same algorithm as spm.py (checked by test/parity_spm.js).
 * model = {pieces: [...], scores: [...]}. The text must already be normalized (NFKC, lowercase, single spaces).
 * Scores are added in float32 (Math.fround), as SentencePiece does; on an exact tie the path found first wins. */
(function (root) {
  "use strict";
  const SP = "▁";
  const DROP = /[\u0000-\u0008\u000b\u000e-\u001f\u007f\u008f\u009f​‎‏‪-‮⁠-⁤﻿￾￿]/g;
  function create(model) {
    const score = new Map(); let maxLen = 1, min = Infinity;
    for (let i = 0; i < model.pieces.length; i++) {
      const sc = Math.fround(model.scores[i]); score.set(model.pieces[i], sc);
      const n = Array.from(model.pieces[i]).length; if (n > maxLen) maxLen = n; if (sc < min) min = sc;
    }
    const unk = Math.fround(min - 10.0);
    function encode(text) {
      text = text.replace(DROP, "").split(/\s+/).filter(Boolean).join(" ");
      if (!text) return [];
      const s = Array.from(SP + text.split(" ").join(SP)); const n = s.length;
      const best = new Array(n + 1).fill(-Infinity), back = new Array(n + 1).fill(0), known = new Array(n + 1).fill(true); best[0] = 0;
      for (let i = 0; i < n; i++) {
        const b = best[i]; if (b === -Infinity) continue;
        let hit = false, piece = "";
        const lim = Math.min(maxLen, n - i);
        for (let L = 1; L <= lim; L++) {
          piece += s[i + L - 1];
          const sc = score.get(piece);
          if (sc !== undefined) {
            if (L === 1) hit = true;
            const v = Math.fround(b + sc);
            if (v > best[i + L]) { best[i + L] = v; back[i + L] = i; known[i + L] = true; }
          }
        }
        if (!hit) { const v = Math.fround(b + unk); if (v > best[i + 1]) { best[i + 1] = v; back[i + 1] = i; known[i + 1] = false; } }
      }
      const out = []; let j = n;
      while (j > 0) { const i = back[j]; out.push([s.slice(i, j).join(""), known[j]]); j = i; }
      out.reverse(); const res = []; let prevUnk = false;
      for (const [p, k] of out) { if (!k && prevUnk) res[res.length - 1] += p; else res.push(p); prevUnk = !k; }
      return res;
    }
    return { encode };
  }
  const api = { create, SP };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.SPM = api;
})(typeof self !== "undefined" ? self : this);
