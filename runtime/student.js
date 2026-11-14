/* Accurate model's neural part (student) in JS with onnxruntime-web. Same algorithm as student.py.
 * const S = await Student.create(ort, loadBinary, loadJson, lower, "student_B")   // loaders get "student_B/<file>"
 * Models exported by train_student2.py ("kv": true in vocab.json) decode one token per step with a key/value cache. */
(function (root) {
  "use strict";
  let BPE = typeof module !== "undefined" && module.exports ? require("./bpe.js") : root.BPE;
  let SPM = typeof module !== "undefined" && module.exports ? require("./spm.js") : root.SPM;
  async function create(ort, loadBinary, loadJson, lower, dir) {
    dir = dir || "student_V9";
    const v = await loadJson(dir + "/vocab.json");
    BPE = BPE || root.BPE;
    if (v.bpe && !BPE) throw new Error("this model splits words into pieces: load runtime/bpe.js before student.js");
    SPM = SPM || root.SPM;
    if (v.spm && !SPM) throw new Error("this model uses SentencePiece: load runtime/spm.js before student.js");
    const sp = v.spm ? SPM.create(await loadJson(dir + "/spm_src.json")) : null;
    // v7 language experts: one small file per language, fetched the first time that language is seen
    const NRM = typeof module !== "undefined" && module.exports ? require("./normalizer.js") : root.ENNormalizer;
    const EX = new Map();
    async function expert(text) {
      const e = v.experts; const r = NRM.process("translate", text).language;
      const lang = r && r.length && e.langs.includes(r[0][0]) && r[0][1] >= e.min_conf ? r[0][0] : null;
      if (!EX.has(lang)) {
        const shapes = [[e.enc_layers, e.d, e.rank], [e.enc_layers, e.rank, e.d], [e.dec_layers, e.d, e.rank], [e.dec_layers, e.rank, e.d]];
        const total = shapes.reduce((a, sh) => a + sh[0] * sh[1] * sh[2], 0);
        const a = lang ? new Float32Array(await loadBinary(dir + "/experts/" + lang + ".bin")) : new Float32Array(total);
        if (a.length !== total) throw new Error("bad expert file for " + lang);
        let o = 0; const out = shapes.map((sh) => { const n = sh[0] * sh[1] * sh[2]; const t = new ort.Tensor("float32", a.slice(o, o + n), sh); o += n; return t; });
        EX.set(lang, out);
      }
      return EX.get(lang);
    }
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    const enc = await ort.InferenceSession.create(new Uint8Array(await loadBinary(dir + "/encoder_int8.onnx")), opts);
    const dec = await ort.InferenceSession.create(new Uint8Array(await loadBinary(dir + "/decoder_int8.onnx")), opts);
    const si = new Map(v.src.map((c, i) => [c, i]));
    const prep = (s) => lower(s.normalize("NFKC")).replace(/\s+/g, " ").trim();
    const T = (arr, dims) => new ort.Tensor("int64", BigInt64Array.from(arr.map(BigInt)), dims);
    async function translate(text) {
      let ids;
      if (sp) {            // pretrained translator (v6): SentencePiece pieces, then the end mark
        ids = sp.encode(prep(text)).slice(0, v.max_src - 1).map((c) => si.has(c) ? si.get(c) : v.unk);
        if (ids.length) ids.push(v.eos);
      } else {
        const pieces = v.bpe ? BPE.encode(prep(text), v.bpe.src) : Array.from(prep(text));
        ids = pieces.slice(0, v.max_src).map((c) => si.has(c) ? si.get(c) : v.unk);
      }
      if (!ids.length) return ["", 0.0];
      const src = T(ids, [1, ids.length]);
      if (v.kv) return translateKv(src, v.experts ? await expert(prep(text)) : null);
      const mem = (await enc.run({ src })).mem;
      const out = [v.bos]; const lps = [];
      for (let k = 0; k < v.max_tgt - 1; k++) {
        const r = await dec.run({ tgt: T(out, [1, out.length]), mem, src });
        const lg = Array.from(r.logits.data, Number);
        lg[v.pad] = -1e9; lg[v.unk] = -1e9; lg[v.bos] = -1e9;
        let m = -Infinity, arg = 0; for (let i = 0; i < lg.length; i++) if (lg[i] > m) { m = lg[i]; arg = i; }
        let z = 0; for (let i = 0; i < lg.length; i++) z += Math.exp(lg[i] - m);
        if (arg === v.eos) break;
        out.push(arg); lps.push(lg[arg] - (m + Math.log(z)));
      }
      const txt = out.slice(1).map((i) => v.tgt[i]).join("");
      const conf = lps.length ? Math.exp(lps.reduce((a, b) => a + b, 0) / lps.length) : 0.0;
      return [txt, Math.round(conf * 1000) / 1000];
    }
    function pick(data) {
      const lg = Array.from(data, Number);
      lg[v.pad] = -1e9; lg[v.unk] = -1e9; lg[v.bos] = -1e9;
      let m = -Infinity, arg = 0; for (let i = 0; i < lg.length; i++) if (lg[i] > m) { m = lg[i]; arg = i; }
      let z = 0; for (let i = 0; i < lg.length; i++) z += Math.exp(lg[i] - m);
      return [arg, lg[arg] - (m + Math.log(z))];
    }
    async function translateKv(src, ex) {
      const e = await enc.run(ex ? { src, ed: ex[0], eu: ex[1] } : { src });
      const dex = ex && ex[2].dims[0] ? { dd: ex[2], du: ex[3] } : {};   // v8: experts in the encoder only
      let pk = new ort.Tensor("float32", new Float32Array(0), [v.layers, 1, v.heads, 0, v.head_dim]); let pv = pk;
      let tok = v.bos; const out = []; const lps = [];
      for (let t = 0; t < v.max_tgt - 1; t++) {
        const r = await dec.run({ tok: T([tok], [1, 1]), pos: T([t], [1, 1]), ck: e.ck, cv: e.cv, src_bias: e.src_bias, pk, pv, ...dex });
        const [arg, lp] = pick(r.logits.data);
        if (arg === v.eos) break;
        out.push(arg); lps.push(lp); tok = arg; pk = r.nk; pv = r.nv;
      }
      const txt = sp ? out.map((i) => v.tgt[i]).join("").split(SPM.SP).join(" ").split(/\s+/).filter(Boolean).join(" ")
        : v.bpe ? BPE.decode(out.map((i) => v.tgt[i])) : out.map((i) => v.tgt[i]).join("");
      const conf = lps.length ? Math.exp(lps.reduce((a, b) => a + b, 0) / lps.length) : 0.0;
      return [txt, Math.round(conf * 1000) / 1000];
    }
    return { translate, kv: !!v.kv, dir, policy: v.policy || null };
  }
  function acceptable(out, conf, EC, english) {
    if (!out || conf < 0.5) return false;
    const w = out.split(/\s+/).filter(Boolean);
    if (w.length >= 6 && new Set(w).size <= Math.floor(w.length / 2)) return false;
    return english === false ? true : EC.isEnglish(out, "relaxed").english;
  }
  const api = { create, acceptable };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Student = api;
})(typeof self !== "undefined" ? self : this);
