/* English normalizer, JS runtime. Same logic and data files as runtime/normalizer.py (parity-tested).
 *
 * Browser:  await ENNormalizer.init(name => fetch("data/" + name).then(r => r.json()));
 *           ENNormalizer.normalize([["translate", "ул. Ленина 5"], ["transliterate", "東京都"]])
 * Node:     const N = require("./normalizer.js"); await N.init(name => JSON.parse(fs.readFileSync(...)))
 */
(function (root) {
  "use strict";
  // Core files are loaded by init(); the per-script gazetteer parts and the zh / ja / th tables are loaded
  // on demand (ensure / normalizeAsync) or all at once (init option preload: "all", the default).
  const CORE = ["rules.json", "glossary.json", "lid.json", "encheck.json", "translit/fallback.json", "gazetteer/index.json"];
  const SCRIPT_TABLES = { Han: ["translit/zh.json", "translit/ja.json"], Kana: ["translit/ja.json"], Thai: ["translit/th.json"] };
  const D = {}; const GZ = {}; const GZT = {}; const loadedGaz = new Set();   // GZT: small towns, whole label / whole part only
  const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  const chars = (s) => Array.from(s);
  const len = (s) => chars(s).length;
  class NormalizerError extends Error {
    constructor(code, message) { super(message); this.name = "NormalizerError"; this.code = code; }
  }

  let ECmod = null, LOADER = null;
  const EC = () => ECmod;
  async function loadFile(f) {
    if (!has(D, f)) {
      try { D[f] = await LOADER(f); } catch (e) { throw new NormalizerError("LOAD_FAILED", "could not load data file " + f + ": " + (e && e.message || e)); }
      if (!D[f] || typeof D[f] !== "object") throw new NormalizerError("LOAD_FAILED", "data file " + f + " is empty or not JSON");
    }
    return D[f];
  }
  async function init(loader, encheckModule, opts) {
    LOADER = loader; opts = opts || {};
    await Promise.all(CORE.map(loadFile));
    ECmod = encheckModule || (typeof module !== "undefined" && module.exports ? require("./encheck.js") : root.ENCheck);
    await ECmod.init(async (n) => D[n]);
    const pre = opts.preload === undefined ? "all" : opts.preload;
    if (pre === "all") await ensureScripts(D["gazetteer/index.json"].parts.concat(Object.keys(SCRIPT_TABLES)));
    else if (Array.isArray(pre)) await ensureScripts(pre);
  }
  async function ensureScripts(list) {
    const jobs = []; const parts = D["gazetteer/index.json"].parts;
    for (const sc of list) {
      if (parts.includes(sc) && !loadedGaz.has(sc)) {
        loadedGaz.add(sc);
        jobs.push(loadFile("gazetteer/" + sc + ".json").then((d) => { Object.assign(GZ, d.k); if (d.t) Object.assign(GZT, d.t); }));
        for (const tn of (D["gazetteer/index.json"].towns || [])) if (tn.startsWith(sc + "_towns")) jobs.push(loadFile("gazetteer/" + tn + ".json").then((d) => { Object.assign(GZT, d.t); }));
      }
      for (const f of (SCRIPT_TABLES[sc] || [])) if (!has(D, f)) jobs.push(loadFile(f));
    }
    await Promise.all(jobs);
  }
  function scriptsNeeded(text) {
    const out = new Set(["Latin"]);
    for (const ch of String(text || "").normalize("NFKC")) if (isAlpha(ch)) out.add(scriptOf(ch) || "Other");
    return out;
  }
  async function ensure(items) {
    const need = new Set();
    for (const it of items) for (const sc of scriptsNeeded(typeof it === "string" ? it : it && it[1])) need.add(sc);
    await ensureScripts([...need]);
  }
  const gaz = () => GZ;
  const gazWhole = (k) => { const f = foldLatin(k); return has(GZ, k) ? GZ[k] : has(GZ, f) ? GZ[f] : has(GZT, k) ? GZT[k] : has(GZT, f) ? GZT[f] : null; };
  const rules = () => D["rules.json"];
  const gloss = () => D["glossary.json"];
  const tl = (n) => {
    const d = D["translit/" + n];
    if (!d) throw new NormalizerError("NOT_LOADED", "table translit/" + n + " is not loaded: call ensure(items) or normalizeAsync(items), or init with preload \"all\"");
    return d;
  };

  // ------------------------------------------------------------ text helpers
  const LATIN_SPECIAL = { "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ı": "i", "ß": "ss", "æ": "ae", "Æ": "AE",
    "ø": "o", "Ø": "O", "œ": "oe", "Œ": "OE", "þ": "th", "ð": "d", "ħ": "h" };
  const cp = (ch) => ch.codePointAt(0);
  const isLatin = (ch) => { const o = cp(ch); return o < 0x250 || (o >= 0x1E00 && o <= 0x1EFF); };
  function foldLatin(s) {
    let out = "";
    for (const ch of s) {
      if (has(LATIN_SPECIAL, ch)) { out += LATIN_SPECIAL[ch]; continue; }
      if (isLatin(ch) && cp(ch) > 0x7F) out += ch.normalize("NFD").replace(/\p{M}/gu, "");
      else out += ch;
    }
    return out;
  }
  const SEP = /[\s\-'’‘`´.,()\[\]\/·・]+/gu;
  const lower = (s) => s.toLowerCase().split("ß").join("ss").split("ς").join("σ");
  const isAlpha = (ch) => /^\p{L}$/u.test(ch);
  const isDigit = (ch) => /^\p{Nd}$/u.test(ch);
  const hasAlpha = (s) => chars(s).some(isAlpha);
  const allDigits = (s) => chars(s).every(isDigit);
  function normKey(s) { return lower(s.normalize("NFKC")).replace(SEP, " ").trim(); }

  const SCRIPTS = [["Cyrillic", 0x0400, 0x04FF], ["Arabic", 0x0600, 0x06FF], ["Devanagari", 0x0900, 0x097F],
    ["Thai", 0x0E00, 0x0E7F], ["Hangul", 0xAC00, 0xD7AF], ["Kana", 0x3040, 0x30FF], ["Han", 0x4E00, 0x9FFF],
    ["Greek", 0x0370, 0x03FF]];
  function scriptOf(ch) {
    const o = cp(ch);
    if ((o >= 0x41 && o <= 0x5A) || (o >= 0x61 && o <= 0x7A) || (o >= 0xC0 && o <= 0x24F && isAlpha(ch)) || (o >= 0x1E00 && o <= 0x1EFF)) return "Latin";
    for (const [n, a, b] of SCRIPTS) if (o >= a && o <= b) return n;
    if ((o >= 0x3400 && o <= 0x4DBF) || (o >= 0xF900 && o <= 0xFAFF)) return "Han";
    if ((o >= 0x1100 && o <= 0x11FF) || (o >= 0x3130 && o <= 0x318F)) return "Hangul";
    if ((o >= 0x31F0 && o <= 0x31FF) || (o >= 0xFF66 && o <= 0xFF9F)) return "Kana";
    return null;
  }
  function scriptsIn(s) {
    const c = {};
    for (const ch of s) { const sc = scriptOf(ch); if (sc) c[sc] = (c[sc] || 0) + 1; }
    return c;
  }
  const r3 = (x) => Math.round(x * 1000) / 1000;
  const cap = (w) => { const a = chars(w); return a.length ? a[0].toUpperCase() + a.slice(1).join("") : w; };
  const capWords = (s) => s.split(" ").map(cap).join(" ");

  // ------------------------------------------------------------ language identification
  const SCRIPT_LANG = { Cyrillic: "ru", Arabic: "ar", Devanagari: "hi", Thai: "th", Hangul: "ko", Kana: "ja" };
  function nb(group, s) {
    const m = D["lid.json"].groups[group]; const langs = m.langs, t = m.t, unk = m.unk;
    const k = normKey(s); const grams = [];
    if (group === "han") {
      const h = chars(k).filter((ch) => ch >= "㐀" && ch <= "鿿");
      for (const ch of h) grams.push(ch);
      for (let i = 0; i < h.length - 1; i++) grams.push(h[i] + h[i + 1]);
    } else {
      for (const w0 of k.split(/\s+/).filter(Boolean)) {
        const w = chars(" " + w0 + " ");
        for (let n = 1; n <= m.n; n++) for (let i = 0; i + n <= w.length; i++) {
          const g = w.slice(i, i + n).join(""); if (g.trim()) grams.push(g);
        }
      }
    }
    const sc = langs.map(() => 0);
    for (const g of grams) { const r = has(t, g) ? t[g] : null; for (let i = 0; i < langs.length; i++) sc[i] += r ? r[i] : unk[i]; }
    const mx = Math.max(...sc); const ex = sc.map((x) => Math.pow(2.718281828459045, x - mx));
    const z = ex.reduce((a, b) => a + b, 0);
    const ranked = langs.map((_, i) => i).sort((a, b) => (ex[b] - ex[a]) || (a - b));
    return ranked.slice(0, 3).map((i) => [langs[i], r3(ex[i] / z)]);
  }
  function detect(s) {
    const sc = scriptsIn(s); const keys = Object.keys(sc);
    if (!keys.length) return [null, [["und", 1.0]]];
    const main = keys.reduce((a, b) => (sc[b] > sc[a] || (sc[b] === sc[a] && b > a)) ? b : a);
    if (has(sc, "Kana")) return ["Kana", [["ja", 1.0]]];
    if (has(sc, "Hangul")) return ["Hangul", [["ko", 1.0]]];
    if (has(SCRIPT_LANG, main)) return [main, [[SCRIPT_LANG[main], 1.0]]];
    if (main === "Han") return ["Han", nb("han", s)];
    if (main === "Latin") return ["Latin", nb("latin", s)];
    return [main, [["und", 1.0]]];
  }

  // ------------------------------------------------------------ transliteration per script
  function trCyrillic(s) {
    if (UK_ONLY.test(s)) return trUkrainian(s);
    const R = rules().cyrillic, after = rules().cyr_ye_after; let out = ""; let prev = "";
    for (const ch of s) {
      const lo = lower(ch);
      if (has(R, lo)) {
        let r = R[lo];
        if (lo === "е" && (prev === "" || after.includes(prev) || !isAlpha(prev))) r = "ye";
        if (lo === "ё" && !(prev === "" || after.includes(prev) || !isAlpha(prev))) r = "yo";
        if (ch !== lo && r) r = r[0].toUpperCase() + r.slice(1);
        out += r;
      } else out += ch;
      prev = lo;
    }
    return out;
  }
  function trArabic(s) {
    const R = rules().arabic; const NM = rules().ar_names || {};
    return s.split(" ").map((w0) => {
      let w = w0, pre = "";
      if (has(NM, w)) return NM[w];
      if (w.startsWith("ال") && len(w) > 3) { pre = "al-"; w = chars(w).slice(2).join(""); }
      return pre + (has(NM, w) ? NM[w] : chars(w).map((ch) => has(R, ch) ? R[ch] : ch).join(""));
    }).join(" ");
  }
  const HI_LETTERS = [["डब्ल्यू", "W"], ["एक्स", "X"], ["क्यू", "Q"], ["जेड", "Z"], ["एच", "H"], ["एफ", "F"], ["एल", "L"], ["एम", "M"], ["एन", "N"], ["एस", "S"], ["आई", "I"], ["आर", "R"], ["वाई", "Y"], ["बी", "B"], ["सी", "C"], ["डी", "D"], ["जी", "G"], ["जे", "J"], ["के", "K"], ["पी", "P"], ["टी", "T"], ["यू", "U"], ["वी", "V"], ["ए", "A"], ["ई", "E"], ["ओ", "O"]];
  function hiInitials(w) {
    let out = ""; let i = 0;
    outer: while (i < w.length) {
      for (const [f, e] of HI_LETTERS) if (w.startsWith(f, i)) { out += e; i += f.length; continue outer; }
      return null;
    }
    return out.length >= 2 && out.length <= 4 ? out : null;
  }
  function trDevanagari(s) {
    const R = rules(); const C = R.deva_cons, V = R.deva_vowels, M = R.deva_matras, O = R.deva_other;
    const NMH = R.hi_names || {};
    return s.split(" ").map((w0) => {
      if (has(NMH, w0)) return NMH[w0];
      const ini = hiInitials(w0); if (ini) return ini;
      const w = chars(w0); const syl = []; let i = 0;
      while (i < w.length) {
        const ch = w[i]; const two = w.slice(i, i + 2).join("");
        if (w.length > i + 1 && has(C, two)) { syl.push([C[two], "a", true]); i += 2; continue; }
        if (has(C, ch)) syl.push([C[ch], "a", true]);
        else if (has(M, ch) && syl.length) { syl[syl.length - 1][1] = M[ch]; syl[syl.length - 1][2] = false; }
        else if (ch === "्" && syl.length) { syl[syl.length - 1][1] = ""; syl[syl.length - 1][2] = false; }
        else if (has(V, ch)) syl.push(["", V[ch], false]);
        else if (ch === "ं" || ch === "ँ") {
          const nc = i + 1 < w.length && has(C, w[i + 1]) ? C[w[i + 1]] : "";
          const nas = ["p", "b", "m"].includes(nc.slice(0, 1)) ? "m" : "n";
          if (syl.length) syl[syl.length - 1][1] += nas; else syl.push(["", "n", false]);
        } else if (has(O, ch)) syl.push(["", O[ch], false]);
        else syl.push(["", ch, false]);
        i += 1;
      }
      if (syl.length > 1 && syl[syl.length - 1][2]) syl[syl.length - 1][1] = "";
      for (let j = 1; j < syl.length - 1; j++)
        if (syl[j][2] && syl[j - 1][1] !== "" && syl[j + 1][0] !== "" && syl[j + 1][1] !== "") syl[j][1] = "";
      return syl.map(([c, v]) => c + v).join("");
    }).join(" ");
  }
  const KO_L = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"];
  const KO_V = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"];
  const KO_T = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t"];
  const KO_TL = ["", "g", "kk", "ks", "n", "nj", "n", "d", "r", "lg", "lm", "lb", "ls", "lt", "lp", "r", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", ""];
  function trHangul(s) {
    const syl = chars(s).map((ch) => { const o = cp(ch); if (o >= 0xAC00 && o <= 0xD7A3) { const S = o - 0xAC00; return [Math.floor(S / 588), Math.floor((S % 588) / 28), S % 28]; } return ch; });
    let out = "";
    syl.forEach((x, i) => {
      if (typeof x === "string") { out += x; return; }
      const [L, V, T] = x;
      const prev = i > 0 && typeof syl[i - 1] !== "string" ? syl[i - 1] : null;
      const nxt = i + 1 < syl.length && typeof syl[i + 1] !== "string" ? syl[i + 1] : null;
      let ini = KO_L[L];
      if (prev && prev[2] && L === 11 && prev[2] !== 21) ini = KO_TL[prev[2]];
      else if (prev && (prev[2] === 4 || prev[2] === 8) && L === 5) ini = "l";
      else if (prev && prev[2] === 8 && L === 2) ini = "l";
      let fin = KO_T[T];
      if (T === 4 && nxt && nxt[0] === 5) fin = "l";
      if (T && nxt && nxt[0] === 11 && T !== 21) fin = "";
      out += ini + KO_V[V] + fin;
    });
    return out;
  }
  function kanaToRomaji(s) {
    const K = rules().kana;
    const h = chars(s).map((c) => (cp(c) >= 0x30A1 && cp(c) <= 0x30F6) ? String.fromCodePoint(cp(c) - 0x60) : c);
    const out = []; let i = 0, gem = false;
    while (i < h.length) {
      if (h[i] === "っ") { gem = true; i += 1; continue; }
      if (h[i] === "ー") { i += 1; continue; }
      const two = h.slice(i, i + 2).join("");
      let r = (h.length > i + 1 && has(K, two)) ? K[two] : (has(K, h[i]) ? K[h[i]] : null);
      let step = (h.length > i + 1 && has(K, two)) ? 2 : 1;
      if (r === null) { r = h[i]; step = 1; }
      if (gem && r && isAlpha(r[0])) r = r.startsWith("ch") ? ("t" + r) : (r[0] + r);
      gem = false; out.push(r); i += step;
    }
    let s2 = out.join("").replace(/ou(?![aiueo])/g, "o");
    return s2.split("oo").join("o").split("uu").join("u");
  }
  function trPinyinWord(w) {
    const Z = tl("zh.json"), ov = rules().zh_override, F = tl("fallback.json");
    if (has(ov, w)) return ov[w].split(" ").join("");
    if (has(Z.w, w)) return Z.w[w].split(" ").join("");
    return chars(w).map((ch) => (has(Z.c, ch) ? Z.c[ch] : (has(F, ch) ? F[ch] : ch)).toLowerCase()).join("");
  }
  function trFallback(s) { const F = tl("fallback.json"); return chars(s).map((ch) => has(F, ch) ? F[ch] : ch).join(""); }

  // ------------------------------------------------------------ segmentation (zh / ja / th)
  const JA_SUFFIX_READ = { "都": "to", "道": "do", "府": "fu", "県": "ken", "市": "shi", "区": "ku", "町": "cho", "村": "mura",
    "郡": "gun", "丁目": "chome", "番地": "banchi", "号": "go", "番": "ban", "通り": "dori" };
  function wordDict(lang) { return lang === "zh" ? tl("zh.json").w : lang === "th" ? tl("th.json").w : null; }
  function segment(s, lang, singleGloss = true) {
    const G = gaz(), A = rules().address[lang] || [], GL = gloss()[lang] || {};
    const afforms = new Set(); for (const [forms] of A) for (const f of forms) afforms.add(f);
    const WD = wordDict(lang), JA = lang === "ja" ? tl("ja.json").w : null, ov = rules().zh_override;
    const a = chars(s); const n = a.length; const segs = []; let i = 0;
    while (i < n) {
      if (scriptOf(a[i]) === "Latin") {
        let j = i + 1;
        while (j < n && (scriptOf(a[j]) === "Latin" || isDigit(a[j]) || "-'.&+".includes(a[j]))) j++;
        segs.push(["other", a.slice(i, j).join("")]); i = j; continue;
      }
      let best = null;
      for (const kind of ["gaz", "addr", "gloss", "word"]) {
        for (let j = Math.min(n, i + 20); j > i; j--) {
          const sub = a.slice(i, j).join("");
          if ((kind === "gaz" || kind === "gloss") && !hasAlpha(sub)) continue;
          let ok;
          if (kind === "gaz") ok = j - i >= 2 && has(G, normKey(sub));
          else if (kind === "addr") ok = afforms.has(sub);
          else if (kind === "gloss") ok = has(GL, normKey(sub)) && (singleGloss || j - i >= 2);
          else ok = (WD !== null && has(WD, sub)) || (JA !== null && has(JA, a[i]) && has(JA[a[i]], sub)) || (lang === "zh" && has(ov, sub));
          if (ok) { best = [[kind, sub], j]; break; }
        }
        if (best) break;
      }
      if (best === null) {
        const ch = a[i]; let j = i + 1; const sc = scriptOf(ch);
        if (sc === "Kana") { while (j < n && scriptOf(a[j]) === "Kana") j++; }
        else if (sc === "Han" || sc === "Thai") {
          const last = segs[segs.length - 1];
          if (last && last[0] === "other") { const lc = chars(last[1]); if (scriptOf(lc[lc.length - 1]) === sc) { segs.pop(); segs.push(["other", last[1] + ch]); i = j; continue; } }
        } else if (isDigit(ch)) { while (j < n && isDigit(a[j])) j++; }
        segs.push(["other", a.slice(i, j).join("")]); i = j;
      } else { segs.push(best[0]); i = best[1]; }
    }
    return segs;
  }
  function suffixForms(lang) {
    const out = [];
    for (const [forms, en, kind] of (rules().address[lang] || [])) if (kind === "s") for (const f of forms) out.push([f, en]);
    return out.map((x, i) => [x, i]).sort((p, q) => (len(q[0][0]) - len(p[0][0])) || (p[1] - q[1])).map((x) => x[0]);
  }
  function segmentSplit(s, lang, depth = 0) {
    const out = []; const SF = suffixForms(lang);
    const first = segment(s, lang, false);
    const segs = first.some(([k]) => k === "gloss") ? segment(s, lang) : first;
    for (const [kind, t] of segs) {
      if ((kind === "gaz" || kind === "word") && depth < 3) {
        const suf = (SF.find(([f]) => t.endsWith(f) && len(t) > len(f)) || [null])[0];
        if (suf) {
          const base = chars(t).slice(0, len(t) - len(suf)).join("");
          if (len(base) >= 2 && has(gaz(), normKey(base))) { out.push(["gaz", base], ["addr", suf]); continue; }
          out.push(...segmentSplit(base, lang, depth + 1), ["addr", suf]); continue;
        }
      }
      out.push([kind, t]);
    }
    return out;
  }
  function jaBeforeSuffix(text, suffix) {
    const JA = tl("ja.json").w; const w = text + suffix; const sr = has(JA_SUFFIX_READ, suffix) ? JA_SUFFIX_READ[suffix] : null;
    const t0 = chars(text)[0];
    if (sr && has(JA, t0) && has(JA[t0], w)) {
      const r = kanaToRomaji(JA[t0][w]);
      if (r.endsWith(sr) && r.length > sr.length) return r.slice(0, r.length - sr.length);
    }
    return romanizeSegment(text, "ja");
  }
  function romanizeSegment(text, lang) {
    if (lang === "ja" && has(JA_SUFFIX_READ, text)) return JA_SUFFIX_READ[text];
    const sc = scriptsIn(text); const t0 = chars(text)[0];
    if (lang === "ja" || has(sc, "Kana")) {
      const JA = tl("ja.json").w;
      if (has(JA, t0) && has(JA[t0], text)) return kanaToRomaji(JA[t0][text]);
      if (has(sc, "Kana") && !has(sc, "Han")) return kanaToRomaji(text);
      return chars(text).map((c) => (has(JA, c) && has(JA[c], c)) ? kanaToRomaji(JA[c][c]) : trFallback(c).toLowerCase()).join("");
    }
    if (lang === "th") { const T = tl("th.json").w; return (has(T, text) && T[text]) || trFallback(text).toLowerCase(); }
    if (has(sc, "Han")) return trPinyinWord(text);
    return romanize(text, lang);
  }
  function group(ch) { const sc = scriptOf(ch); return (sc === "Han" || sc === "Kana") ? "CJK" : sc; }
  function greekDigraphs(t) {
    t = t.replace(/ου/g, "ou").replace(/ού/g, "ou").replace(/Ου/g, "Ou").replace(/Ού/g, "Ou");
    return t.replace(/([αεΑΕ])[υύ](?=([θκξπστφχψ])|.|$)/gu, (m, a, v) => ({ "α": "a", "ε": "e", "Α": "A", "Ε": "E" })[a] + (v ? "f" : "v"));
  }
  function romanize(s, lang = null) {
    const runs = [];
    for (const ch of s) { const g = group(ch); if (runs.length && runs[runs.length - 1][0] === g) runs[runs.length - 1][1] += ch; else runs.push([g, ch]); }
    let out = "";
    for (const [g, t] of runs) {
      if (g === "Cyrillic") out += trCyrillic(t);
      else if (g === "Arabic") out += trArabic(t);
      else if (g === "Devanagari") out += trDevanagari(t);
      else if (g === "Hangul") out += trHangul(t);
      else if (g === "CJK" || g === "Thai") {
        const l = g === "Thai" ? "th" : ((lang === "ja" || chars(t).some((c) => scriptOf(c) === "Kana")) ? "ja" : "zh");
        out += " " + renderCjk(segmentSplit(t, l), l, "transliterate") + " ";
      } else if (g === null || g === "Latin") out += t;
      else out += trFallback(greekDigraphs(t));
    }
    return out.replace(/\s+/g, " ").trim();
  }
  function reorderGloss(words) {
    const rank = gloss()._rank; const res = []; let run = [];
    const rk = (w) => has(rank, w) ? rank[w] : 5;
    const flush = () => { res.push(...run.map((w, i) => [w, i]).sort((p, q) => (rk(p[0]) - rk(q[0])) || (p[1] - q[1])).map((x) => x[0])); run = []; };
    for (const w of words) {
      if (w.startsWith("\x02") && rk(w.slice(1)) < 6) run.push(w.slice(1));
      else { flush(); res.push(w.startsWith("\x02") ? w.slice(1) : w); }
    }
    flush(); return res;
  }
  function renderCjk(segs, lang, op) {
    const G = gaz(), GL = gloss()[lang] || {}, A = rules().address[lang] || [];
    const aen = {}; for (const [forms, en] of A) for (const f of forms) aen[f] = en;
    let words = [];
    for (let i = 0; i < segs.length; i++) {
      const [kind, t] = segs[i];
      if (allDigits(t) || !t.trim()) { words.push(t); continue; }
      if (op === "translate" && kind === "gaz") { words.push("\x03" + G[normKey(t)][0]); continue; }
      if (op === "translate" && kind === "gloss") { words.push("\x02" + GL[normKey(t)]); continue; }
      if (kind === "addr") {
        if (op === "translate") {
          const en = has(aen, t) ? aen[t] : "";
          if (en.startsWith("-") && words.length) words[words.length - 1] += en; else if (en) words.push(en);
          continue;
        }
        const r = romanizeSegment(t, lang);
        if (lang === "ja" && words.length && !allDigits(words[words.length - 1])) words[words.length - 1] += "-" + r;
        else words.push(cap(r));
        continue;
      }
      const s0 = scriptOf(chars(t)[0]);
      if (s0 === null || s0 === "Latin") { words.push(t); continue; }
      if (lang === "ja" && i + 1 < segs.length && segs[i + 1][0] === "addr") { words.push(cap(jaBeforeSuffix(t, segs[i + 1][1]))); continue; }
      words.push(cap(romanizeSegment(t, lang)));
    }
    words = reorderGloss(words);
    if (op === "translate") words = moveTypeWords(words, lang);
    let out = words.filter(Boolean).join(" ").replace(/\s+/g, " ").trim().replace(/\x03/g, "");
    out = out.replace(/([0-9A-Za-z]) ?([-\/]) ?(?=[0-9])/g, "$1$2");
    return op === "translate" ? out.replace(/(\d+(?:-\d+)*) No\.(?= |$)/g, "No. $1") : out;
  }
  let ADDR_EN = null;
  function addrEnglish() {
    if (!ADDR_EN) { ADDR_EN = new Set(); for (const items of Object.values(rules().address)) for (const [, en, kind] of items) if (en && kind !== "title") ADDR_EN.add(en); }
    return ADDR_EN;
  }
  const TYPE_WORDS = new Set(["Street", "Road", "Avenue", "Lane", "Boulevard", "Square", "Highway", "Alley", "Promenade",
    "Embankment", "Passage", "Way", "Dead End", "Quay", "Causeway", "Ring Road"]);
  const PREFIX_TYPE_LANGS = new Set(["es", "pt", "fr", "it", "ru", "pl", "id", "vi", "ar", "th", "en"]);
  function moveTypeWords(words, lang) {
    if (!PREFIX_TYPE_LANGS.has(lang)) return words;
    const stop = addrEnglish(); const out = []; let held = null;
    for (const w of words) {
      if (TYPE_WORDS.has(w) && held === null) { held = w; continue; }
      const pl = w && w[0] === "\x03" && out.length && out[out.length - 1][0] !== "\x03";
      if (held !== null && ((w && isDigit(chars(w)[0])) || stop.has(w) || w.endsWith(",") || pl)) { out.push(held + (pl ? "," : "")); held = null; }
      out.push(w);
    }
    if (held !== null) out.push(held);
    return out;
  }

  // ------------------------------------------------------------ main pipeline
  const CJK_LANGS = new Set(["zh", "ja", "th"]);
  const POSTAL = /^(?:(\d{2}-\d{3}|\d{4}-\d{3}|\d{5}-\d{3}|\d{4,6})\s+(\D.*)|(\D.*?)\s+(\d{2}-\d{3}|\d{4}-\d{3}|\d{5}-\d{3}|\d{4,6}))$/;
  // personal names, used only when transliteration is asked for (same tables as normalizer.py)
  const KO_SURNAMES = {"김": "Kim", "이": "Lee", "박": "Park", "최": "Choi", "정": "Jung", "강": "Kang", "조": "Cho", "윤": "Yoon", "장": "Jang", "임": "Lim", "한": "Han", "오": "Oh", "서": "Seo", "신": "Shin", "권": "Kwon", "황": "Hwang", "안": "Ahn", "송": "Song", "류": "Ryu", "전": "Jeon", "홍": "Hong", "고": "Ko", "문": "Moon", "양": "Yang", "손": "Son", "배": "Bae", "백": "Baek", "허": "Heo", "유": "Yoo", "남": "Nam", "심": "Shim", "노": "Noh", "하": "Ha", "곽": "Kwak", "성": "Sung", "차": "Cha", "주": "Joo", "우": "Woo", "구": "Koo", "민": "Min"};
  const ZH_SURNAMES = "王李张張刘劉陈陳杨楊黄黃赵趙吴吳周徐孙孫马馬朱胡郭何罗羅郑鄭梁谢謝宋唐许許韩韓冯馮邓鄧曹彭曾肖董袁潘蒋蔣蔡余杜叶葉程苏蘇魏吕呂丁任沈姚卢盧姜崔钟鍾谭譚陆陸汪范廖贾賈夏韦韋付邹鄒孟熊秦邱尹薛闫段雷侯龙龍史陶黎贺賀顾顧毛郝龚龔邵孔";
  const JA_SURNAMES = {"佐藤": "Sato", "鈴木": "Suzuki", "高橋": "Takahashi", "田中": "Tanaka", "伊藤": "Ito", "渡辺": "Watanabe", "山本": "Yamamoto", "中村": "Nakamura", "小林": "Kobayashi", "加藤": "Kato", "吉田": "Yoshida", "山田": "Yamada", "佐々木": "Sasaki", "山口": "Yamaguchi", "松本": "Matsumoto", "井上": "Inoue", "木村": "Kimura", "斎藤": "Saito", "清水": "Shimizu", "山崎": "Yamazaki", "池田": "Ikeda", "橋本": "Hashimoto", "阿部": "Abe", "石川": "Ishikawa", "山下": "Yamashita", "中島": "Nakajima", "石井": "Ishii", "小川": "Ogawa", "前田": "Maeda", "岡田": "Okada", "長谷川": "Hasegawa", "藤田": "Fujita", "後藤": "Goto", "近藤": "Kondo", "村上": "Murakami", "遠藤": "Endo", "青木": "Aoki", "坂本": "Sakamoto", "福田": "Fukuda", "太田": "Ota", "西村": "Nishimura", "藤井": "Fujii", "岡本": "Okamoto", "松田": "Matsuda", "中川": "Nakagawa", "中野": "Nakano", "原田": "Harada", "小野": "Ono", "田村": "Tamura", "竹内": "Takeuchi", "金子": "Kaneko", "和田": "Wada", "中山": "Nakayama", "石田": "Ishida", "上田": "Ueda", "森田": "Morita", "酒井": "Sakai", "工藤": "Kudo", "横山": "Yokoyama", "宮崎": "Miyazaki", "宮本": "Miyamoto", "内田": "Uchida", "高木": "Takagi", "安藤": "Ando", "島田": "Shimada", "谷口": "Taniguchi", "大野": "Ono", "高田": "Takada", "丸山": "Maruyama", "今井": "Imai", "河野": "Kono", "藤原": "Fujiwara", "村田": "Murata", "武田": "Takeda", "上野": "Ueno", "杉山": "Sugiyama", "増田": "Masuda", "小島": "Kojima", "小山": "Koyama", "千葉": "Chiba", "久保": "Kubo", "松井": "Matsui", "岩崎": "Iwasaki", "野口": "Noguchi", "菊地": "Kikuchi", "木下": "Kinoshita", "野村": "Nomura", "新井": "Arai", "渡部": "Watanabe", "桜井": "Sakurai", "菅原": "Sugawara"};
  const JA_GIVEN = {"太郎": "Taro", "一郎": "Ichiro", "次郎": "Jiro", "健": "Ken", "誠": "Makoto", "大輔": "Daisuke", "翔": "Sho", "翔太": "Shota", "拓也": "Takuya", "健太": "Kenta", "大翔": "Hiroto", "蓮": "Ren", "悠真": "Yuma", "陽翔": "Haruto", "直樹": "Naoki", "和也": "Kazuya", "達也": "Tatsuya", "哲也": "Tetsuya", "浩": "Hiroshi", "隆": "Takashi", "勇": "Isamu", "茂": "Shigeru", "博": "Hiroshi", "清": "Kiyoshi", "花子": "Hanako", "陽子": "Yoko", "恵子": "Keiko", "美咲": "Misaki", "さくら": "Sakura", "愛": "Ai", "結衣": "Yui", "葵": "Aoi", "陽菜": "Hina", "美穂": "Miho", "由美": "Yumi", "直美": "Naomi", "真由美": "Mayumi", "裕子": "Yuko", "京子": "Kyoko", "洋子": "Yoko", "和子": "Kazuko", "幸子": "Sachiko", "明美": "Akemi", "智子": "Tomoko", "久美子": "Kumiko", "優子": "Yuko"};
  function personName(s) {
    const a = chars(s);
    if (s.includes(" ") || a.length < 2 || a.length > 5 || gazWhole(normKey(s))) return null;
    const sc = new Set(a.map(scriptOf));
    if (sc.size === 1 && sc.has("Hangul") && (a.length === 3 || a.length === 4) && has(KO_SURNAMES, a[0])) return KO_SURNAMES[a[0]] + " " + cap(trHangul(a.slice(1).join("")));
    if (sc.has("Han") && [...sc].every((x) => x === "Han" || x === "Kana")) {
      for (const n of [3, 2]) {
        const sn = a.slice(0, n).join("");
        if (has(JA_SURNAMES, sn) && a.length > n) { const g = a.slice(n).join(""); return JA_SURNAMES[sn] + " " + (has(JA_GIVEN, g) ? JA_GIVEN[g] : cap(romanizeSegment(g, "ja"))); }
      }
      if (sc.size === 1 && (a.length === 2 || a.length === 3) && ZH_SURNAMES.includes(a[0])) return cap(trPinyinWord(a[0])) + " " + cap(trPinyinWord(a.slice(1).join("")));
    }
    return null;
  }
  const UK_ONLY = /[їєіґЇЄІҐ]/;
  const UK = {"а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie", "ж": "zh", "з": "z", "и": "y", "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ь": "", "ю": "iu", "я": "ia", "'": "", "’": "", "ʼ": ""};
  const UK_START = {"є": "ye", "ї": "yi", "й": "y", "ю": "yu", "я": "ya"};
  function trUkrainian(s) {
    let out = ""; let start = true;
    for (const ch of s) {
      const lo = lower(ch);
      if (has(UK, lo)) { const r = start && has(UK_START, lo) ? UK_START[lo] : UK[lo]; out += (ch !== lo && r) ? cap(r) : r; start = false; }
      else { out += ch; start = !isAlpha(ch); }
    }
    return out;
  }
  const tokens = (part) => part.split(/\s+/).filter(Boolean);
  function merged(langs, getter) { const out = {}; for (const l of [...langs].reverse()) Object.assign(out, getter(l)); return out; }
  function stemLookup(k, table) {
    const a = chars(k);
    for (const cut of [1, 2, 3]) { const st = a.slice(0, a.length - cut).join(""); if (a.length - cut >= 5 && has(table, st)) return table[st]; }
    return null;
  }
  function addrMap(l, spaced) {
    const m = {};
    for (const [forms, en, kind] of (rules().address[l] || [])) if (!spaced || (kind !== "s" && kind !== "p")) for (const f of forms) m[f] = [en, kind];
    return m;
  }
  const COLON = /(?<=\p{L}):(?=\d)/gu;
  const ADDR_LATIN = ["es", "pt", "fr", "it", "de", "pl", "tr", "id", "vi"]; const TF = {};
  function typeForms(l) {
    if (!has(TF, l)) { TF[l] = new Set(); for (const [forms, , kind] of (rules().address[l] || [])) if (kind === "type") for (const f of forms) TF[l].add(f); }
    return TF[l];
  }
  function translateSpaced(part, lang0, op, candLangs) {
    let langs = [lang0, ...(candLangs || []).filter((l) => l !== lang0)];
    if (op === "translate" && chars(part).some(isDigit)) {
      const tk = tokens(part);
      for (const l of (langs.some((x) => tk.some((t) => typeForms(x).has(normKey(t)))) ? [] : ADDR_LATIN)) {
        if (!langs.includes(l) && tk.some((t) => typeForms(l).has(normKey(t)) && (len(normKey(t)) >= 5 || t.endsWith(".") || t.endsWith("/")))) { langs = [l, ...langs]; break; }
      }
    }
    const G = gaz();
    const GL = merged(langs, (l) => gloss()[l] || {});
    const addr = merged(langs, (l) => addrMap(l, true));
    const drop = new Set(); for (const l of langs) for (const w of (gloss()._drop[l] || [])) drop.add(w);
    const rank = gloss()._rank; const rk = (w) => has(rank, w) ? rank[w] : 5;
    let lang = lang0;
    const toks = tokens(part);
    for (const l of langs) {
      const gl = gloss()[l] || {}, am = addrMap(l, false);
      if (toks.some((t) => has(gl, normKey(t)) || has(am, normKey(t)))) { lang = l; break; }
    }
    const productLike = op === "translate" && toks.some((t) => has(GL, normKey(t)) || stemLookup(normKey(t), GL));
    const out = []; const stats = { known: 0, total: 0 };
    let i = 0, inStreet = false;
    while (i < toks.length) {
      let hit = null;
      for (let L = Math.min(5, toks.length - i); L > 0; L--) {
        const span = toks.slice(i, i + L).join(" "); const k = normKey(span); const f = foldLatin(k);
        if (!hasAlpha(k)) break;
        if (op === "translate") {
          const latin1 = L === 1 && chars(span).every((c) => { const s = scriptOf(c); return s === null || s === "Latin"; });
          if (has(addr, k) && !((k === "mah" || k === "m" || k === "d") && i > 0 && isDigit(chars(toks[i - 1]).slice(-1)[0] || "") && L === 1)) hit = ["addr", addr[k][0], L];
          else if (has(GL, k) && !inStreet) hit = ["gloss", GL[k], L];
          else if ((has(G, k) || (has(G, f) && f !== k)) && !inStreet && !latin1) hit = ["gaz", (has(G, k) ? G[k] : G[f])[0], L];
        }
        if (hit) break;
      }
      const tok = toks[i];
      if (!hit && op === "translate" && !inStreet) { const st = stemLookup(normKey(tok), GL); if (st) hit = ["gloss", st, 1]; }
      if (hit) {
        const [kind, en, L] = hit; i += L; stats.known += L; stats.total += L;
        inStreet = kind === "addr" && TYPE_WORDS.has(en);
        if (kind === "addr" && en === "") continue;
        out.push([kind, en]); continue;
      }
      i += 1; stats.total += 1;
      const k = normKey(tok);
      if (tok && isDigit(chars(tok)[0])) inStreet = false;
      if (productLike && drop.has(k)) { stats.known += 1; continue; }
      if (lang === "ko") { out.push(["name", romanizeKoreanToken(tok, op)]); continue; }
      if (!chars(tok).some((c) => { const s = scriptOf(c); return s !== null && s !== "Latin"; })) { out.push(["keep", tok]); continue; }
      out.push(["name", capWords(romanize(tok, lang))]);
    }
    let res = []; let run = [];
    const flush = () => { res.push(...run.map((w, j) => [w, j]).sort((p, q) => (rk(p[0]) - rk(q[0])) || (p[1] - q[1])).map((x) => x[0])); run = []; };
    for (const [kind, w] of out) { if (kind === "gloss" && rk(w) < 6) run.push(w); else { flush(); res.push(w); } }
    flush();
    if (op === "translate") res = moveTypeWords(res, lang);
    return [res.join(" "), stats];
  }
  const KO_STREET = new Set(["로", "길", "대로", "번길", "가"]);
  function romanizeKoreanToken(tok, op) {
    const G = gaz();
    for (const [f, en] of suffixForms("ko")) {
      if (tok.endsWith(f) && len(tok) > len(f)) {
        const base = chars(tok).slice(0, len(tok) - len(f)).join("");
        const b = (op === "translate" && has(G, normKey(base)) && !KO_STREET.has(f)) ? G[normKey(base)][0] : cap(trHangul(base));
        if (op === "translate") return (en.startsWith("-") || en === "F") ? b + en : (b + (en ? " " + en : ""));
        return b + "-" + trHangul(f);
      }
    }
    if (op === "translate" && has(G, normKey(tok))) return G[normKey(tok)][0];
    return cap(trHangul(tok));
  }
  const BASE_CONF = { Latin: 0.9, Cyrillic: 0.85, Hangul: 0.85, Kana: 0.85, Devanagari: 0.75, Han: 0.75, Arabic: 0.55, Thai: 0.6 };
  function processItem(op0, text, asciiOut = false) {
    const op = String(op0).toLowerCase().startsWith("translit") ? "transliterate" : "translate";
    const s = (text || "").normalize("NFKC").replace(/\s+/g, " ").trim().replace(COLON, ": ");
    const [script, langs] = detect(s);
    const lang = langs[0][0];
    const res = { input: text, operation: op, language: langs.map((x) => [x[0], x[1]]), script };
    if (!s) return Object.assign(res, { output: "", confidence: 0.0, method: "empty" });
    const G = gaz(); const k = normKey(s);
    let out, method, conf, cov = 1;
    let ec;
    if (op === "translate" && gazWhole(k)) {
      out = gazWhole(k)[0]; method = "gazetteer"; conf = 0.95;
    } else if (op === "transliterate" && personName(s)) {
      out = personName(s); method = "transliteration"; conf = 0.75;
    } else if (op === "translate" && (ec = EC().isEnglish(s, "strict")).english) {
      out = s; method = "already-english"; conf = ec.score;
    } else {
      const parts = s.split(/[,，、،]/).map((p) => p.trim());
      const outs = []; let known = 0, total = 0; const methods = new Set();
      for (const p of parts) {
        if (!p) continue;
        const kp = normKey(p);
        if (op === "translate" && gazWhole(kp)) { outs.push(gazWhole(kp)[0]); known++; total++; methods.add("gazetteer"); continue; }
        const pc = op === "translate" ? POSTAL.exec(p) : null;
        if (pc && gazWhole(normKey(pc[2] || pc[3]))) {
          const e = gazWhole(normKey(pc[2] || pc[3]))[0];
          outs.push(pc[1] ? pc[1] + " " + e : e + " " + pc[4]); known += 2; total += 2; methods.add("gazetteer"); continue;
        }
        if (CJK_LANGS.has(lang) || script === "Han" || script === "Kana" || script === "Thai") {
          let l = CJK_LANGS.has(lang) ? lang : (script === "Thai" ? "th" : script === "Kana" ? "ja" : "zh");
          let segs = segmentSplit(p, l);
          if (script === "Han" && langs[0][1] < 0.9) {
            const alt = l === "zh" ? "ja" : "zh"; const segs2 = segmentSplit(p, alt);
            const cov = (ss) => ss.filter(([kd]) => kd === "gaz" || kd === "addr" || kd === "gloss").reduce((a, [, t]) => a + len(t), 0);
            if (cov(segs2) > cov(segs)) {
              l = alt; segs = segs2;
              res.language = [[alt, langs[0][0] === alt ? langs[0][1] : r3(1 - langs[0][1])]].concat(res.language.filter((x) => x[0] !== alt).slice(0, 2));
            }
          }
          outs.push(renderCjk(segs, l, op));
          const kn = segs.filter(([kd]) => kd === "gaz" || kd === "addr" || kd === "gloss").length;
          known += op === "translate" ? kn : segs.length; total += segs.length;
          methods.add(op === "translate" && kn ? "dictionary" : "transliteration");
        } else {
          const [o, st] = translateSpaced(p, lang, op, langs.map((x) => x[0]));
          outs.push(o); known += op === "translate" ? st.known : st.total; total += st.total;
          methods.add(op === "translate" && st.known ? "dictionary" : "transliteration");
        }
      }
      out = outs.filter(Boolean).join(", ");
      cov = total ? known / total : 0;
      const base = has(BASE_CONF, script) ? BASE_CONF[script] : 0.5;
      conf = op === "translate" ? r3(base * (0.6 + 0.4 * cov)) : base;
      method = [...methods].sort().join("+") || "passthrough";
    }
    if (asciiOut) out = trFallback(foldLatin(out));
    return Object.assign(res, { output: out, confidence: conf, method, coverage: r3(cov) });
  }
  function checkItem(it) {
    if (typeof it === "string") return ["translate", it];
    if (!Array.isArray(it) || it.length < 2 || typeof it[1] !== "string") throw new NormalizerError("BAD_INPUT", "each item must be a string or an [operation, text] pair, got " + JSON.stringify(it));
    const op = String(it[0]).toLowerCase();
    if (!op.startsWith("translat") && !op.startsWith("translit")) throw new NormalizerError("BAD_INPUT", "operation must be \"translate\" or \"transliterate\", got " + JSON.stringify(it[0]));
    return [it[0], it[1]];
  }
  function normalize(items, asciiOut = false) {
    if (!Array.isArray(items)) throw new NormalizerError("BAD_INPUT", "normalize() takes an array of items");
    return items.map((it) => { const [op, t] = checkItem(it); return processItem(op, t, asciiOut); });
  }
  async function normalizeAsync(items, asciiOut = false) {
    if (!Array.isArray(items)) throw new NormalizerError("BAD_INPUT", "normalizeAsync() takes an array of items");
    await ensure(items.map(checkItem)); return normalize(items, asciiOut);
  }
  const api = { init, ensure, normalize, normalizeAsync, process: processItem, detect, romanize, normKey, NormalizerError, version: "fast-0.3",
    _: { gaz, rules, normKey, foldLatin, isDigit, tokens, trFallback, lower, EC: () => ECmod, gloss } };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.ENNormalizer = api;
})(typeof self !== "undefined" ? self : this);
