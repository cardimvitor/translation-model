/* en-normalizer: one entry point for the three models (browser or Node).
 *
 *   const n = await ENNormalizerKit.createNormalizer({ baseUrl: "https://your.cdn/en-normalizer/", model: "fast" });
 *   const results = await n.normalize([["translate", "Москва"], ["transliterate", "ул. Ленина 5"]]);
 *   n.isEnglish("leather bag")  // English check
 *
 * Options:
 *   baseUrl   where data/ and models/ live (URL in the browser, URL or folder path in Node)   (required)
 *   model     "fast" (default) or "accurate"
 *   variant   Accurate student folder under models/: "student_B" (commercial-friendly, default) | "student_A" | "student_C"
 *   ort       onnxruntime-web module (Accurate only); defaults to the global `ort` / require("onnxruntime-web")
 *   preload   "none" (default: load per-script tables on demand) | "all" | ["Latin", "Cyrillic", ...]
 *   timeoutMs per normalize() call (default 30000)
 * Errors are NormalizerError with code BAD_INPUT | LOAD_FAILED | NOT_LOADED | TIMEOUT.
 */
(function (root) {
  "use strict";
  const isNode = typeof process !== "undefined" && process.versions && process.versions.node && typeof window === "undefined";
  const req = (n) => (isNode ? require(n) : null);
  const mod = (local, global) => (isNode ? require(local) : root[global]);

  function makeLoaders(baseUrl) {
    const base = String(baseUrl || "").replace(/\/?$/, "/");
    if (isNode && !/^https?:/.test(base)) {
      const fs = req("fs"), path = req("path");
      return {
        json: async (p) => JSON.parse(fs.readFileSync(path.join(base, p), "utf8")),
        bin: async (p) => { const b = fs.readFileSync(path.join(base, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
      };
    }
    const get = async (p) => { const r = await fetch(base + p); if (!r.ok) throw new Error(p + " HTTP " + r.status); return r; };
    return { json: async (p) => (await get(p)).json(), bin: async (p) => (await get(p)).arrayBuffer() };
  }

  async function createNormalizer(opts) {
    opts = opts || {};
    const EC = mod("./encheck.js", "ENCheck"), N = mod("./normalizer.js", "ENNormalizer");
    const Err = N.NormalizerError;
    if (!opts.baseUrl) throw new Err("BAD_INPUT", "createNormalizer: baseUrl is required");
    const L = makeLoaders(opts.baseUrl); const model = opts.model || "fast";
    if (model !== "fast" && model !== "accurate") throw new Err("BAD_INPUT", "model must be \"fast\" or \"accurate\"");
    await N.init((n) => L.json("data/" + n), EC, { preload: opts.preload || "none" });
    let A = null, variant = null;
    if (model === "accurate") {
      const ST = mod("./student.js", "Student"); A = mod("./accurate.js", "Accurate");
      const ort = opts.ort || root.ort || req("onnxruntime-web");
      if (!ort) throw new Err("LOAD_FAILED", "Accurate needs onnxruntime-web: pass { ort }");
      variant = opts.variant || "student_B";
      let S;
      try { S = await ST.create(ort, (n) => L.bin("models/" + n), (n) => L.json("models/" + n), N._.lower, variant); }
      catch (e) { throw new Err("LOAD_FAILED", "could not load model " + variant + ": " + (e && e.message || e)); }
      A.init(N, S, ST);
    }
    const timeoutMs = opts.timeoutMs || 30000;
    function withTimeout(p) {
      let t; const to = new Promise((_, rej) => { t = setTimeout(() => rej(new Err("TIMEOUT", "normalize() took longer than " + timeoutMs + " ms")), timeoutMs); });
      return Promise.race([p, to]).finally(() => clearTimeout(t));
    }
    return {
      model, variant, version: N.version,
      /** items: Array<string | [operation, text]>; returns one result per item, same order */
      normalize: (items, o) => withTimeout(model === "accurate" ? A.normalize(items, !!(o && o.ascii)) : N.normalizeAsync(items, !!(o && o.ascii))),
      isEnglish: (text, mode) => EC.isEnglish(text, mode || "strict"),
      detect: (text) => N.detect(String(text || "").normalize("NFKC")),
    };
  }
  const api = { createNormalizer };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.ENNormalizerKit = api;
})(typeof self !== "undefined" ? self : this);
