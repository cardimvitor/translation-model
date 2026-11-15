/* Accurate model's neural part (student) in JS with onnxruntime-web. Same algorithm as student.py.
 * const S = await Student.create(ort, loadBinary, loadJson, lower, "student_B")   // loaders get "student_B/<file>"
 * Models exported by train_student2.py ("kv": true in vocab.json) decode one token per step with a key/value cache. */
(function (root) {
  "use strict";
  let BPE = typeof module !== "undefined" && module.exports ? require("./bpe.js") : root.BPE;
  let SPM = typeof module !== "undefined" && module.exports ? require("./spm.js") : root.SPM;
  async function create(ort, loadBinary, loadJson, lower, dir) {
    dir = dir || "student_V14";
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
    async function translate(text, opt) {
      opt = opt || {};
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
      if (v.kv) {
        const ex = v.experts ? await expert(prep(text)) : null;
        const pol = v.policy || {}; let K = v.batch ? (opt.beams || pol.beams || 1) : 1;
        if (pol.beam_max_src !== undefined && ids.length > pol.beam_max_src) K = 1;   // long inputs: greedy
        return K > 1 ? translateBeam(src, ex, K, opt.lp || pol.lp || 1.0) : translateKv(src, ex);
      }
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
    // Beam search, same algorithm as student.py: K partial outputs run through the decoder as one batch.
    async function translateBeam(src, ex, K, lp) {
      const e = await enc.run(ex ? { src, ed: ex[0], eu: ex[1] } : { src });
      const dex = ex && ex[2].dims[0] ? { dd: ex[2], du: ex[3] } : {};
      const L = v.layers, H = v.heads, D = v.head_dim, V = v.tgt.length;
      let pk = new ort.Tensor("float32", new Float32Array(0), [L, 1, H, 0, D]); let pv = pk;
      let beams = [[[], 0.0]]; const done = [];
      const take = (t, idx) => {                 // keep the cache rows of the surviving hypotheses (axis 1)
        const [l, b, h, T, d] = t.dims; const blk = h * T * d; const out = new Float32Array(l * idx.length * blk);
        for (let a = 0; a < l; a++) for (let k = 0; k < idx.length; k++) out.set(t.data.subarray((a * b + idx[k]) * blk, (a * b + idx[k] + 1) * blk), (a * idx.length + k) * blk);
        return new ort.Tensor("float32", out, [l, idx.length, h, T, d]);
      };
      for (let t = 0; t < v.max_tgt - 1; t++) {
        const toks = beams.map((b) => b[0].length ? b[0][b[0].length - 1] : v.bos);
        const r = await dec.run({ tok: T(toks, [toks.length, 1]), pos: T([t], [1, 1]), ck: e.ck, cv: e.cv, src_bias: e.src_bias, pk, pv, ...dex });
        const cand = [];
        for (let i = 0; i < beams.length; i++) {
          const lg = r.logits.data.subarray(i * V, (i + 1) * V); const off = (j) => j === v.pad || j === v.unk || j === v.bos;
          let m = -Infinity; for (let j = 0; j < V; j++) if (!off(j) && lg[j] > m) m = lg[j];
          let z = 0; for (let j = 0; j < V; j++) if (!off(j)) z += Math.exp(lg[j] - m);
          const norm = m + Math.log(z);
          const top = [];                        // the K best tokens in one pass (no sort of the vocabulary)
          for (let j = 0; j < V; j++) {
            if (off(j)) continue;
            if (top.length < K || lg[j] > lg[top[top.length - 1]]) {
              let p = top.length; top.push(j);
              while (p > 0 && lg[top[p - 1]] < lg[j]) { top[p] = top[p - 1]; p--; }
              top[p] = j; if (top.length > K) top.pop();
            }
          }
          for (const j of top) cand.push([beams[i][1] + (lg[j] - norm), i, j]);
        }
        cand.sort((a, b) => b[0] - a[0] || a[1] - b[1] || a[2] - b[2]);
        const nxt = [], idx = [];
        for (const [sc, i, j] of cand) {
          if (j === v.eos) done.push([sc / Math.pow(beams[i][0].length + 1, lp), beams[i][0], sc]);
          else { nxt.push([beams[i][0].concat([j]), sc]); idx.push(i); }
          if (nxt.length + done.length >= K) break;
        }
        if (done.length >= K || !nxt.length) break;
        beams = nxt; pk = take(r.nk, idx); pv = take(r.nv, idx);
      }
      if (!done.length) for (const [seq, sc] of beams) done.push([sc / Math.pow(Math.max(1, seq.length), lp), seq, sc]);
      done.sort((a, b) => b[0] - a[0] || a[1].length - b[1].length);
      const out = done[0][1], sc = done[0][2];
      const txt = sp ? out.map((i) => v.tgt[i]).join("").split(SPM.SP).join(" ").split(/\s+/).filter(Boolean).join(" ") : out.map((i) => v.tgt[i]).join("");
      const conf = out.length ? Math.exp(sc / (out.length + 1)) : 0.0;
      return [txt, Math.round(conf * 1000) / 1000];
    }
    return { translate, kv: !!v.kv, dir, policy: v.policy || null, enWords: v.en_words ? new Set(v.en_words) : null, xxWords: v.xx_words ? new Set(v.xx_words) : null };
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
