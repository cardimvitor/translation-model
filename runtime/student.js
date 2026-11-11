/* Accurate model's neural part (student) in JS with onnxruntime-web. Same algorithm as student.py.
 * const S = await Student.create(ort, loadBinary, loadJson, lower, "student_B")   // loaders get "student_B/<file>"
 * Models exported by train_student2.py ("kv": true in vocab.json) decode one token per step with a key/value cache. */
(function (root) {
  "use strict";
  let BPE = typeof module !== "undefined" && module.exports ? require("./bpe.js") : root.BPE;
  async function create(ort, loadBinary, loadJson, lower, dir) {
    dir = dir || "student_MAX";
    const v = await loadJson(dir + "/vocab.json");
    BPE = BPE || root.BPE;
    if (v.bpe && !BPE) throw new Error("this model splits words into pieces: load runtime/bpe.js before student.js");
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    const enc = await ort.InferenceSession.create(new Uint8Array(await loadBinary(dir + "/encoder_int8.onnx")), opts);
    const dec = await ort.InferenceSession.create(new Uint8Array(await loadBinary(dir + "/decoder_int8.onnx")), opts);
    const si = new Map(v.src.map((c, i) => [c, i]));
    const prep = (s) => lower(s.normalize("NFKC")).replace(/\s+/g, " ").trim();
    const T = (arr, dims) => new ort.Tensor("int64", BigInt64Array.from(arr.map(BigInt)), dims);
    async function translate(text) {
      const pieces = v.bpe ? BPE.encode(prep(text), v.bpe.src) : Array.from(prep(text));
      const ids = pieces.slice(0, v.max_src).map((c) => si.has(c) ? si.get(c) : v.unk);
      if (!ids.length) return ["", 0.0];
      const src = T(ids, [1, ids.length]);
      if (v.kv) return translateKv(src);
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
    async function translateKv(src) {
      const e = await enc.run({ src });
      let pk = new ort.Tensor("float32", new Float32Array(0), [v.layers, 1, v.heads, 0, v.head_dim]); let pv = pk;
      let tok = v.bos; const out = []; const lps = [];
      for (let t = 0; t < v.max_tgt - 1; t++) {
        const r = await dec.run({ tok: T([tok], [1, 1]), pos: T([t], [1, 1]), ck: e.ck, cv: e.cv, src_bias: e.src_bias, pk, pv });
        const [arg, lp] = pick(r.logits.data);
        if (arg === v.eos) break;
        out.push(arg); lps.push(lp); tok = arg; pk = r.nk; pv = r.nv;
      }
      const txt = v.bpe ? BPE.decode(out.map((i) => v.tgt[i])) : out.map((i) => v.tgt[i]).join("");
      const conf = lps.length ? Math.exp(lps.reduce((a, b) => a + b, 0) / lps.length) : 0.0;
      return [txt, Math.round(conf * 1000) / 1000];
    }
    return { translate, kv: !!v.kv, dir };
  }
  function acceptable(out, conf, EC) {
    if (!out || conf < 0.5) return false;
    const w = out.split(/\s+/).filter(Boolean);
    if (w.length >= 6 && new Set(w).size <= Math.floor(w.length / 2)) return false;
    return EC.isEnglish(out, "relaxed").english;
  }
  const api = { create, acceptable };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Student = api;
})(typeof self !== "undefined" ? self : this);
