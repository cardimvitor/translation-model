/* Model 3: English check, JS runtime (same data/encheck.json, parity-tested with encheck.py).
 * await ENCheck.init(name => fetch("data/" + name).then(r => r.json())); ENCheck.isEnglish("black leather bag") */
(function (root) {
  "use strict";
  let G = null, W = null, WORDS = null, T = 0, R = 1;
  async function init(loader) { const m = await loader("encheck.json"); G = m.g; W = m.w; T = m.T; R = m.R; WORDS = new Set(m.words); }
  const ASCII_OK = new Set("abcdefghijklmnopqrstuvwxyz0123456789'&+");
  const isL = (c) => /^\p{L}$/u.test(c);
  function wordClass(w) {
    const a = Array.from(w);
    if (!a.some(isL)) return "N";
    if (WORDS.has(w)) return "E";
    if (a.some((c) => isL(c) && !ASCII_OK.has(c))) return "F";
    const gs = grams(w); const sc = gs.reduce((t, g) => t + (Object.prototype.hasOwnProperty.call(G, g) ? G[g] : 0), 0) / Math.max(1, gs.length);
    return sc < -T ? "F" : "N";
  }
  const SEP = /[\s\-'’‘`´.,()\[\]\/·・]+/gu;
  const normKey = (s) => s.normalize("NFKC").toLowerCase().split("ß").join("ss").split("ς").join("σ").replace(SEP, " ").trim();
  function isLatinLetter(ch) {
    const o = ch.codePointAt(0);
    return (o >= 0x41 && o <= 0x5A) || (o >= 0x61 && o <= 0x7A) || (o >= 0xC0 && o <= 0x24F) || (o >= 0x1E00 && o <= 0x1EFF);
  }
  function grams(w) {
    const a = Array.from(" " + w + " "); const out = [];
    for (const n of [1, 2, 3]) for (let i = 0; i + n <= a.length; i++) { const g = a.slice(i, i + n).join(""); if (g.trim()) out.push(g); }
    return out;
  }
  function isEnglish(text, mode = "strict") {
    const s = (text || "").normalize("NFKC");
    const letters = Array.from(s).filter((c) => /^\p{L}$/u.test(c));
    if (!letters.length) return { english: true, score: 1.0, reason: "no letters" };
    if (letters.some((c) => !isLatinLetter(c))) return { english: false, score: 0.0, reason: "non-Latin script" };
    const ws = normKey(s).split(/\s+/).filter(Boolean);
    if (mode === "relaxed") {
      const cl = ws.map(wordClass); const e = cl.filter((x) => x === "E").length, f = cl.filter((x) => x === "F").length;
      return { english: f === 0 || e >= R * f, score: Math.round((e + 1) / (e + f + 2) * 1000) / 1000, reason: `relaxed: ${e} English, ${f} foreign words` };
    }
    const gs = ws.flatMap(grams);
    const a = gs.reduce((t, g) => t + (Object.prototype.hasOwnProperty.call(G, g) ? G[g] : 0), 0) / Math.max(1, gs.length);
    const b = ws.filter((w) => WORDS.has(w)).length / Math.max(1, ws.length);
    const z = W[0] + W[1] * a + W[2] * b;
    const p = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
    const e = ws.filter((w) => WORDS.has(w)).length;
    const f = ws.filter((w) => !WORDS.has(w) && Array.from(w).some((c) => isL(c) && !ASCII_OK.has(c))).length;
    if (f && e < R * f) return { english: false, score: Math.round(Math.min(p, 0.1) * 1000) / 1000, reason: "non-English letters" };
    return { english: z > 0, score: Math.round(p * 1000) / 1000, reason: "model" };
  }
  const api = { init, isEnglish };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.ENCheck = api;
})(typeof self !== "undefined" ? self : this);
