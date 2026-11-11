"""Accurate model's neural part (student), Python reference with onnxruntime. Same algorithm as student.js."""
import json, os, re, unicodedata
import numpy as np
import onnxruntime as ort
import normalizer as F
import bpe as BPE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "models", os.environ.get("NEURAL_DIR", "student_MAX"))   # student_MAX (default) / _B2 / _C2 / _A / _B / _C


class Student:
    def __init__(self, path=MD, int8=True):
        o = ort.SessionOptions(); o.intra_op_num_threads = 1
        suf = "_int8" if int8 else ""
        self.enc = ort.InferenceSession(os.path.join(path, f"encoder{suf}.onnx"), o, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(path, f"decoder{suf}.onnx"), o, providers=["CPUExecutionProvider"])
        v = json.load(open(os.path.join(path, "vocab.json"), encoding="utf-8"))
        self.si = {c: i for i, c in enumerate(v["src"])}; self.tv = v["tgt"]; self.v = v

    @staticmethod
    def prep(s):
        return re.sub(r"\s+", " ", F.lower(unicodedata.normalize("NFKC", s))).strip()

    def translate(self, text):
        v = self.v
        pieces = BPE.encode(self.prep(text), v["bpe"]["src"]) if v.get("bpe") else list(self.prep(text))
        ids = [self.si.get(c, v["unk"]) for c in pieces[: v["max_src"]]]
        if not ids:
            return "", 0.0
        src = np.array([ids], np.int64)
        if v.get("kv"):
            return self._translate_kv(src)
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


def _translate_kv(self, src):
    """Cached decoding: the encoder also returns each decoder layer's cross-attention keys/values; the decoder
    takes one token per step plus the past self-attention keys/values (models exported by train_student2.py)."""
    v = self.v
    ck, cv, sb = self.enc.run(None, {"src": src})
    pk = np.zeros((v["layers"], 1, v["heads"], 0, v["head_dim"]), np.float32); pv = pk
    tok = v["bos"]; out = []; lps = []
    for t in range(v["max_tgt"] - 1):
        logits, pk, pv = self.dec.run(None, {"tok": np.array([[tok]], np.int64), "pos": np.array([[t]], np.int64),
                                             "ck": ck, "cv": cv, "src_bias": sb, "pk": pk, "pv": pv})
        nxt, lp = _pick(logits[0], v)
        if nxt == v["eos"]:
            break
        out.append(nxt); lps.append(lp); tok = nxt
    txt = BPE.decode([self.tv[i] for i in out]) if v.get("bpe") else "".join(self.tv[i] for i in out)
    conf = float(np.exp(np.mean(lps))) if lps else 0.0
    return txt, round(conf, 3)


Student._translate_kv = _translate_kv


def acceptable(out, conf):
    """Reject empty, degenerate (same 3-gram repeated) or non-English outputs."""
    import encheck
    if not out or conf < 0.5:
        return False
    w = out.split()
    if len(w) >= 6 and len(set(w)) <= len(w) // 2:
        return False
    return encheck.is_english(out, "relaxed")["english"]
