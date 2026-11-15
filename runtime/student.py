"""Accurate model's neural part (student), Python reference with onnxruntime. Same algorithm as student.js."""
import json, os, re, unicodedata
import numpy as np
import onnxruntime as ort
import normalizer as F
import bpe as BPE
import spm as SPM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "models", os.environ.get("NEURAL_DIR", "student_V12"))   # student_V12 (default) / student_V8F (fast) / older: _V9 _V8 _V7 _V6 ...


class Student:
    def __init__(self, path=MD, int8=True):
        o = ort.SessionOptions(); o.intra_op_num_threads = 1
        suf = "_int8" if int8 else ""
        self.enc = ort.InferenceSession(os.path.join(path, f"encoder{suf}.onnx"), o, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(path, f"decoder{suf}.onnx"), o, providers=["CPUExecutionProvider"])
        v = json.load(open(os.path.join(path, "vocab.json"), encoding="utf-8"))
        self.si = {c: i for i, c in enumerate(v["src"])}; self.tv = v["tgt"]; self.v = v
        self.spm = SPM.Unigram(json.load(open(os.path.join(path, "spm_src.json"), encoding="utf-8"))) if v.get("spm") else None
        self.path = path; self._ex = {}

    def expert(self, text):
        """v7 language experts: the detector's language picks one small expert file (loaded on first use);
        unsure, English or an unknown language -> zeros, which is the plain shared model."""
        e = self.v["experts"]; r = F.process("translate", text)["language"]
        lang = r[0][0] if r and r[0][0] in e["langs"] and r[0][1] >= e["min_conf"] else None
        if lang not in self._ex:
            d, k, ne, nd = e["d"], e["rank"], e["enc_layers"], e["dec_layers"]
            shapes = [(ne, d, k), (ne, k, d), (nd, d, k), (nd, k, d)]
            if lang is None:
                self._ex[lang] = [np.zeros(sh, np.float32) for sh in shapes]
            else:
                a = np.fromfile(os.path.join(self.path, "experts", lang + ".bin"), np.float32); out = []; o = 0
                for sh in shapes:
                    n = int(np.prod(sh)); out.append(a[o:o + n].reshape(sh)); o += n
                self._ex[lang] = out
        return self._ex[lang]

    @staticmethod
    def prep(s):
        return re.sub(r"\s+", " ", F.lower(unicodedata.normalize("NFKC", s))).strip()

    def translate(self, text):
        v = self.v
        if self.spm:                        # pretrained translator (v6): SentencePiece pieces, then the end mark
            ids = [self.si.get(c, v["unk"]) for c in self.spm.encode(self.prep(text))[: v["max_src"] - 1]]
            ids = ids + [v["eos"]] if ids else ids
        else:
            pieces = BPE.encode(self.prep(text), v["bpe"]["src"]) if v.get("bpe") else list(self.prep(text))
            ids = [self.si.get(c, v["unk"]) for c in pieces[: v["max_src"]]]
        if not ids:
            return "", 0.0
        src = np.array([ids], np.int64)
        if v.get("kv"):
            ex = self.expert(self.prep(text)) if v.get("experts") else None
            pol = v.get("policy") or {}
            beams = int(os.environ.get("NEURAL_BEAMS", pol.get("beams", 1))) if v.get("batch") else 1
            if len(ids) > pol.get("beam_max_src", 10 ** 9):
                beams = 1                        # long inputs (product names): greedy, where the beam costs 3x for little gain
            if beams > 1:
                return self._translate_beam(src, ex, beams, float(os.environ.get("NEURAL_LP", (v.get("policy") or {}).get("lp", 1.0))))
            return self._translate_kv(src, ex)
        mem = self.enc.run(None, {"src": src})[0]
        out = [v["bos"]]; lps = []
        for _ in range(v["max_tgt"] - 1):
            logits = self.dec.run(None, {"tgt": np.array([out], np.int64), "mem": mem, "src": src})[0][0].astype(np.float64)
            logits[v["pad"]] = -1e9; logits[v["unk"]] = -1e9; logits[v["bos"]] = -1e9
            m = logits.max(); lse = m + np.log(np.exp(logits - m).sum())
            nxt = int(logits.argmax())
            if nxt == v["eos"]:
                break
            out.append(nxt); lps.append(float(logits[nxt] - lse))
        txt = "".join(self.tv[i] for i in out[1:])
        conf = float(np.exp(np.mean(lps))) if lps else 0.0
        return txt, round(conf, 3)


def _pick(logits, v):
    logits = logits.astype(np.float64)
    logits[v["pad"]] = -1e9; logits[v["unk"]] = -1e9; logits[v["bos"]] = -1e9
    m = logits.max(); lse = m + np.log(np.exp(logits - m).sum()); nxt = int(logits.argmax())
    return nxt, float(logits[nxt] - lse)


def _translate_kv(self, src, ex=None):
    """Cached decoding: the encoder also returns each decoder layer's cross-attention keys/values; the decoder
    takes one token per step plus the past self-attention keys/values (models exported by train_student2.py)."""
    v = self.v
    ck, cv, sb = self.enc.run(None, {"src": src, **({"ed": ex[0], "eu": ex[1]} if ex else {})})
    dex = {"dd": ex[2], "du": ex[3]} if ex and ex[2].shape[0] else {}      # v8: experts in the encoder only
    pk = np.zeros((v["layers"], 1, v["heads"], 0, v["head_dim"]), np.float32); pv = pk
    tok = v["bos"]; out = []; lps = []
    for t in range(v["max_tgt"] - 1):
        logits, pk, pv = self.dec.run(None, {"tok": np.array([[tok]], np.int64), "pos": np.array([[t]], np.int64),
                                             "ck": ck, "cv": cv, "src_bias": sb, "pk": pk, "pv": pv, **dex})
        nxt, lp = _pick(logits[0], v)
        if nxt == v["eos"]:
            break
        out.append(nxt); lps.append(lp); tok = nxt
    if v.get("spm"):
        txt = " ".join("".join(self.tv[i] for i in out).replace(SPM.SP, " ").split())
    else:
        txt = BPE.decode([self.tv[i] for i in out]) if v.get("bpe") else "".join(self.tv[i] for i in out)
    conf = float(np.exp(np.mean(lps))) if lps else 0.0
    return txt, round(conf, 3)


def _translate_beam(self, src, ex, K, lp):
    """Beam search (same algorithm as student.js): keep the K best partial outputs, run them through the decoder as one
    batch, stop when K outputs have ended; pick the best by log-probability / (length ** lp)."""
    v = self.v
    ck, cv, sb = self.enc.run(None, {"src": src, **({"ed": ex[0], "eu": ex[1]} if ex else {})})
    dex = {"dd": ex[2], "du": ex[3]} if ex and ex[2].shape[0] else {}
    pk = np.zeros((v["layers"], 1, v["heads"], 0, v["head_dim"]), np.float32); pv = pk
    beams = [([], 0.0)]; done = []
    for t in range(v["max_tgt"] - 1):
        toks = np.array([[b[0][-1] if b[0] else v["bos"]] for b in beams], np.int64)
        logits, nk, nv = self.dec.run(None, {"tok": toks, "pos": np.array([[t]], np.int64), "ck": ck, "cv": cv, "src_bias": sb, "pk": pk, "pv": pv, **dex})
        cand = []
        for i, (seq, sc) in enumerate(beams):
            lg = logits[i].astype(np.float64); lg[v["pad"]] = -1e9; lg[v["unk"]] = -1e9
            if v["bos"] != v["pad"]:
                lg[v["bos"]] = -1e9
            m = lg.max(); norm = m + np.log(np.exp(lg - m).sum())
            top = np.argpartition(-lg, K)[:K]                    # the K best tokens without sorting the vocabulary
            for j in sorted(top.tolist(), key=lambda j: (-lg[j], j)):
                cand.append((sc + float(lg[j] - norm), i, int(j)))
        cand.sort(key=lambda c: (-c[0], c[1], c[2])); nxt = []; idx = []
        for sc, i, j in cand:
            if j == v["eos"]:
                done.append((sc / ((len(beams[i][0]) + 1) ** lp), beams[i][0], sc))
            else:
                nxt.append((beams[i][0] + [j], sc)); idx.append(i)
            if len(nxt) + len(done) >= K:
                break
        if len(done) >= K or not nxt:
            break
        beams = nxt; pk = nk[:, idx]; pv = nv[:, idx]
    if not done:
        done = [(sc / (max(1, len(seq)) ** lp), seq, sc) for seq, sc in beams]
    done.sort(key=lambda d: (-d[0], len(d[1]))); _, out, sc = done[0]
    txt = " ".join("".join(self.tv[i] for i in out).replace(SPM.SP, " ").split()) if v.get("spm") else "".join(self.tv[i] for i in out)
    conf = float(np.exp(sc / (len(out) + 1))) if out else 0.0
    return txt, round(conf, 3)


Student._translate_kv = _translate_kv
Student._translate_beam = _translate_beam


def acceptable(out, conf, english=True):
    """Reject empty, degenerate (same words repeated) or, unless english=False, non-English outputs."""
    import encheck
    if not out or conf < 0.5:
        return False
    w = out.split()
    if len(w) >= 6 and len(set(w)) <= len(w) // 2:
        return False
    return encheck.is_english(out, "relaxed")["english"] if english else True
