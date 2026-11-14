/* Accurate model (JS): Fast front end + student. Same routing as accurate.py.
 * await Accurate.init(ENNormalizer, studentInstance, Student); await Accurate.process("translate", text) */
(function (root) {
  "use strict";
  let N, S, ST;
  // decision rule: "v2" (default, tuned on held-out names) or "v1" (v4 behaviour); same as accurate.py
  const POLICY = { name: "v2", tau: 0.0, digitsAreAddress: false };
  function init(normalizer, studentInstance, studentModule, policy) { N = normalizer; S = studentInstance; ST = studentModule; if (policy) Object.assign(POLICY, policy); }
  function addressLike(part, langs) {
    if (Array.from(part).some(N._.isDigit)) return true;
    return addressWords(part, langs);
  }
  function addressWords(part, langs) {
    const _ = N._;
    const keys = _.tokens(part).map(_.normKey);
    for (const l of langs) {
      const A = _.rules().address[l] || [];
      const spaced = new Set(); for (const [forms, , kind] of A) if (kind !== "s" && kind !== "p") for (const f of forms) spaced.add(f);
      if (keys.some((k) => spaced.has(k))) return true;
      if (["zh", "ja", "ko", "th"].includes(l)) for (const [forms, , kind] of A) if (kind === "s" || kind === "p") for (const f of forms) if (Array.from(f).length > 1 && part.includes(f)) return true;
    }
    return false;
  }
  let GEN = null;
  function glossEn() {
    if (!GEN) { GEN = new Set(); const g = N._.gloss(); for (const l of Object.keys(g)) if (l[0] !== "_") for (const v of Object.values(g[l])) for (const w of N._.lower(v).split(" ")) if (w) GEN.add(w); }
    return GEN;
  }
  function keepsKnown(studentOut, fastOut) {
    const have = new Set(N._.lower(studentOut).split(" ").filter(Boolean));
    for (const w of [...have]) if (w.endsWith("s")) have.add(w.slice(0, -1));
    return N._.lower(fastOut).split(" ").filter((w) => w && glossEn().has(w))
      .every((w) => have.has(w) || have.has(w + "s") || (w.endsWith("s") && have.has(w.slice(0, -1))));
  }
  let AEN = null;
  function addressEn() {
    if (!AEN) { AEN = new Set(); for (const A of Object.values(N._.rules().address)) for (const [, en] of A) for (const w of N._.lower(en).replace(/^-+/, "").split(" ")) if (w.length > 2) AEN.add(w); }
    return AEN;
  }
  function isAddressV2(s, langs, fast) {
    if (addressWords(s, langs)) return true;
    if (Array.from(s).some(N._.isDigit))
      return fast.method.includes("gazetteer") || N._.lower(fast.output).split(" ").some((w) => addressEn().has(w.replace(/^[,.]+|[,.]+$/g, "")));
    return false;
  }
  async function processV2(op, text, asciiOut) {
    const _ = N._; const fast = N.process(op, text, asciiOut);
    if (fast.operation !== "translate" || fast.method === "gazetteer" || fast.method === "already-english" || !fast.output) return Object.assign({}, fast, { model: "accurate" });
    const s = (text || "").normalize("NFKC").replace(/\s+/g, " ").trim();
    const langs = fast.language.map((x) => x[0]);
    if ((fast.coverage || 0) >= 1 || isAddressV2(s, langs, fast) || (POLICY.digitsAreAddress && addressLike(s, langs))) return Object.assign({}, fast, { model: "accurate" });
    const [o, c] = await S.translate(s);
    if (o && c >= POLICY.tau && ST.acceptable(o, 1.0, _.EC()) && keepsKnown(o, fast.output))
      return Object.assign({}, fast, { output: asciiOut ? _.trFallback(_.foldLatin(o)) : o, method: "neural", confidence: Math.round(c * 1000) / 1000, model: "accurate" });
    return Object.assign({}, fast, { method: fast.method + "(neural rejected)", model: "accurate" });
  }
  // rule v3 (strong neural model, v6): settings ship with the model (vocab.json "policy"); same as accurate.py
  function sim(a, b) {
    a = a.toLowerCase().split(/\s+/).filter(Boolean).join(" "); b = b.toLowerCase().split(/\s+/).filter(Boolean).join(" ");
    const A = new Set(), B = new Set();
    for (let i = 0; i < a.length - 1; i++) A.add(a.slice(i, i + 2));
    for (let i = 0; i < b.length - 1; i++) B.add(b.slice(i, i + 2));
    let n = 0; for (const x of A) if (B.has(x)) n++;
    return n / Math.max(1, A.size + B.size - n);
  }
  async function processV3(op, text, asciiOut, pol) {
    const _ = N._; const fast = N.process(op, text, asciiOut);
    if (fast.operation === "translate" && fast.method === "already-english" && fast.confidence < (pol.en_sure || 0)) {
      // English double check (same as accurate.py): confident and clearly different neural answer replaces the label
      const s0 = (text || "").normalize("NFKC").replace(/\s+/g, " ").trim();
      const [o0, c0] = await S.translate(s0);
      if (o0 && c0 >= (pol.en_tau === undefined ? 0.85 : pol.en_tau) && sim(o0, s0) < (pol.en_sim === undefined ? 0.5 : pol.en_sim))
        return Object.assign({}, fast, { output: asciiOut ? _.trFallback(_.foldLatin(o0)) : o0, method: "neural", confidence: Math.round(c0 * 1000) / 1000, model: "accurate" });
      return Object.assign({}, fast, { model: "accurate" });
    }
    if (fast.operation !== "translate" || fast.method === "gazetteer" || fast.method === "already-english" || !fast.output) return Object.assign({}, fast, { model: "accurate" });
    const s = (text || "").normalize("NFKC").replace(/\s+/g, " ").trim();
    const langs = fast.language.map((x) => x[0]);
    const longLabel = pol.addr_max_words && s.split(" ").length > pol.addr_max_words && (fast.coverage || 0) < (pol.addr_long_cov === undefined ? 0.6 : pol.addr_long_cov);
    const shortPlace = pol.keep_gaz && fast.method.includes("gazetteer") && s.split(" ").length <= (pol.addr_max_words === undefined ? 8 : pol.addr_max_words);
    if ((isAddressV2(s, langs, fast) && !longLabel) || (pol.keep_cov && (fast.coverage || 0) >= 1) || shortPlace) return Object.assign({}, fast, { model: "accurate" });
    const [o, c] = await S.translate(s);
    if (o && c >= (pol.tau || 0) && ST.acceptable(o, 1.0, _.EC(), pol.need_english !== false) && (!pol.need_known || keepsKnown(o, fast.output)))
      return Object.assign({}, fast, { output: asciiOut ? _.trFallback(_.foldLatin(o)) : o, method: "neural", confidence: Math.round(c * 1000) / 1000, model: "accurate" });
    return Object.assign({}, fast, { method: fast.method + "(neural rejected)", model: "accurate" });
  }
  // "reliable" flag (same as accurate.py): only when the model ships a calibrated threshold (policy.rel_tau)
  function reliable(r, pol) {
    if (pol.rel_tau !== undefined) {
      const m = r.method || "";
      r.reliable = !!(r.operation !== "translate" || m === "gazetteer" || (m === "already-english" && (r.confidence || 0) >= (pol.en_sure || 0))
        || (m === "neural" && (r.confidence || 0) >= pol.rel_tau)
        || (m !== "neural" && !m.includes("rejected") && (r.coverage || 0) >= 1));
    }
    return r;
  }
  async function process(op, text, asciiOut = false) {
    if (S && S.policy && POLICY.name !== "v1") return reliable(await processV3(op, text, asciiOut, S.policy), S.policy);
    if (POLICY.name === "v2") return processV2(op, text, asciiOut);
    const _ = N._; const fast = N.process(op, text, asciiOut);
    if (fast.operation !== "translate" || fast.method === "gazetteer" || fast.method === "already-english" || !fast.output) return Object.assign({}, fast, { model: "accurate" });
    const s = (text || "").normalize("NFKC").replace(/\s+/g, " ").trim();
    const langs = fast.language.map((x) => x[0]);
    const parts = s.split(/[,，、،]/).map((p) => p.trim()).filter(Boolean);
    if (addressLike(s, langs)) return Object.assign({}, fast, { model: "accurate" });
    const outs = [], methods = new Set(), confs = []; const G = _.gaz();
    const whole = fast.output.split(", ");
    const has = (k) => Object.prototype.hasOwnProperty.call(G, k);
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      const kp = _.normKey(p);
      if (has(kp) || has(_.foldLatin(kp))) { outs.push((has(kp) ? G[kp] : G[_.foldLatin(kp)])[0]); methods.add("gazetteer"); confs.push(0.95); continue; }
      let r = N.process("translate", p, asciiOut);
      if (whole.length === parts.length) r = Object.assign({}, r, { output: whole[i] });
      const known = _.lower(r.output).split(" ").filter((w) => w && glossEn().has(w));
      if (r.coverage >= 1 || !known.length) { outs.push(r.output); methods.add(r.method); confs.push(r.confidence); continue; }
      const [o, c] = await S.translate(p);
      if (!(ST.acceptable(o, c, _.EC()) && keepsKnown(o, r.output))) { outs.push(r.output); methods.add(r.method + "(neural rejected)"); confs.push(r.confidence); continue; }
      outs.push(asciiOut ? _.trFallback(_.foldLatin(o)) : o); methods.add("neural"); confs.push(c);
    }
    return Object.assign({}, fast, { output: outs.filter(Boolean).join(", "), method: [...methods].sort().join("+"),
      confidence: confs.length ? Math.round(Math.min(...confs) * 1000) / 1000 : 0.0, model: "accurate" });
  }
  async function normalize(items, asciiOut = false) {
    if (N.ensure) await N.ensure(items);           // lazy-loaded tables (no-op when everything is preloaded)
    const out = [];
    for (const it of items) out.push(typeof it === "string" ? await process("translate", it, asciiOut) : await process(it[0], it[1], asciiOut));
    return out;
  }
  const api = { init, process, normalize, POLICY };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Accurate = api;
})(typeof self !== "undefined" ? self : this);
