"""Accurate model, Python reference: the Fast front end + a neural translator (ONNX, greedy decoding).

Routing per comma-separated part (translate operation):
  known place (gazetteer)          -> gazetteer
  address-like text (digits / address words anywhere) -> Fast rules (neural models translate street names)
  every word known to Fast, or no product word found -> Fast rules
  otherwise (products, free text)  -> student; kept only if it looks like English and keeps every
                                      dictionary word Fast found, else the Fast answer
transliterate operation            -> Fast rules
"""
import json, os, re, unicodedata, math
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import normalizer as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


NLLB_CODES = {"zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab", "ru": "rus_Cyrl", "hi": "hin_Deva",
              "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "tr": "tur_Latn", "pl": "pol_Latn", "it": "ita_Latn",
              "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "pt": "por_Latn", "en": "eng_Latn"}
MODEL_NAME = os.environ.get("NEURAL", "student")


class MT:
    def __init__(self, path=os.path.join(ROOT, "models", MODEL_NAME)):
        o = ort.SessionOptions(); o.intra_op_num_threads = 4
        self.enc = ort.InferenceSession(os.path.join(path, "onnx", "encoder_model_quantized.onnx"), o, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(path, "onnx", "decoder_model_merged_quantized.onnx"), o, providers=["CPUExecutionProvider"])
        tj = json.load(open(os.path.join(path, "tokenizer.json"), encoding="utf-8"))
        if tj.get("normalizer", {}) and tj["normalizer"].get("type") == "Precompiled" and not tj["normalizer"].get("precompiled_charsmap"):
            tj["normalizer"] = None            # empty charsmap = identity (input is already NFKC-normalized)
        self.tok = Tokenizer.from_str(json.dumps(tj))
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.start, self.eos, self.pad = cfg["decoder_start_token_id"], cfg["eos_token_id"], cfg["pad_token_id"]
        self.L, self.H = cfg["decoder_layers"], cfg["decoder_attention_heads"]
        self.hd = cfg["d_model"] // self.H
        self.nllb = "nllb" in cfg.get("_name_or_path", path).lower() or cfg.get("vocab_size", 0) > 200000
        self.vocab = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}

    def encode_ids(self, text, lang=None):
        if self.nllb:
            src = self.vocab[NLLB_CODES.get(lang, "eng_Latn")]
            return [src] + self.tok.encode(text, add_special_tokens=False).ids + [self.eos]
        ids = self.tok.encode(text).ids
        if not ids or ids[-1] != self.eos:
            ids = ids + [self.eos]
        return ids

    def translate(self, text, lang=None, max_new=48):
        ids = np.array([self.encode_ids(text, lang)], dtype=np.int64); mask = np.ones_like(ids)
        hs = self.enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        past = {}
        for l in range(self.L):
            for a in ("decoder", "encoder"):
                for kv in ("key", "value"):
                    past[f"past_key_values.{l}.{a}.{kv}"] = np.zeros((1, self.H, 0, self.hd), np.float32)
        cur = self.start; out = []; lps = []
        for step in range(max_new):
            feed = {"encoder_attention_mask": mask, "input_ids": np.array([[cur]], np.int64), "encoder_hidden_states": hs,
                    "use_cache_branch": np.array([step > 0])}
            feed.update(past)
            res = self.dec.run(None, feed)
            logits = res[0][0, -1].astype(np.float64)
            logits[self.pad] = -1e9
            m = logits.max(); lp = logits - m - math.log(np.exp(logits - m).sum())
            nxt = int(np.argmax(logits))
            if self.nllb and step == 0:
                nxt = self.vocab["eng_Latn"]          # forced target language token (not part of the output)
            names = [o.name for o in self.dec.get_outputs()]
            for name, val in zip(names[1:], res[1:]):
                key = name.replace("present.", "past_key_values.")
                if step == 0 or ".decoder." in key:
                    past[key] = val
            if nxt == self.eos:
                break
            if not (self.nllb and step == 0):
                out.append(nxt); lps.append(lp[nxt])
            cur = nxt
        text = self.tok.decode(out, skip_special_tokens=True)
        conf = float(math.exp(sum(lps) / len(lps))) if lps else 0.0
        return text.strip(), round(conf, 3)


_MT = {}


def mt(variant=None):
    """Neural part: a phone student (models/<variant>, default NEURAL_DIR or "student_B");
    NEURAL=nllb-600m uses NLLB instead, for comparison."""
    key = variant or "default"
    if key not in _MT:
        if MODEL_NAME == "student":
            import student
            _MT[key] = student.Student(os.path.join(ROOT, "models", variant)) if variant else student.Student()
        else:
            _MT[key] = MT()
    return _MT[key]


def address_like(part, langs):
    if any(F.is_digit(c) for c in part):
        return True
    toks = F._tokens(part); keys = [F.norm_key(t) for t in toks]
    for l in langs:
        A = F.rules()["address"].get(l, [])
        spaced = {f for forms, en, kind in A if kind not in ("s", "p") for f in forms}
        if any(k in spaced for k in keys):
            return True
        if l in ("zh", "ja", "ko", "th") and any(f in part for forms, en, kind in A if kind in ("s", "p") for f in forms if len(f) > 1):
            return True
    return False


_GEN = []


def gloss_en():
    """English words the glossary can produce (Fast output words that came from the dictionary)."""
    if not _GEN:
        g = F.gloss(); _GEN.append({w for l, d in g.items() if not l.startswith("_") for v in d.values() for w in F.lower(v).split()})
    return _GEN[0]


def keeps_known(student_out, fast_out):
    """The student may fix unknown words, but must keep every word Fast translated from the dictionary."""
    have = set(F.lower(student_out).split()); have |= {w[:-1] for w in have if w.endswith("s")}
    return all(w in have or w + "s" in have or (w.endswith("s") and w[:-1] in have)
               for w in F.lower(fast_out).split() if w in gloss_en())


def process(op, text, ascii_out=False, variant=None):
    fast = F.process(op, text, ascii_out)
    if fast["operation"] != "translate" or fast["method"] in ("gazetteer", "already-english") or not fast["output"]:
        return dict(fast, model="accurate")
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    langs = [l for l, _ in fast["language"]]
    parts = [p.strip() for p in re.split(r"[,，、،]", s) if p.strip()]
    if address_like(s, langs):
        return dict(fast, model="accurate")
    outs, methods, confs = [], set(), []
    G = F.gaz()
    whole = fast["output"].split(", ")
    for i, p in enumerate(parts):
        kp = F.norm_key(p)
        if kp in G or F.fold_latin(kp) in G:
            outs.append((G.get(kp) or G.get(F.fold_latin(kp)))[0]); methods.add("gazetteer"); confs.append(0.95); continue
        r = F.process("translate", p, ascii_out)
        if len(whole) == len(parts):
            r = dict(r, output=whole[i])                       # Fast's answer with the whole text's language context
        known = [w for w in F.lower(r["output"]).split() if w in gloss_en()]
        if r["coverage"] >= 1 or not known:                     # all words known, or not a product: Fast
            outs.append(r["output"]); methods.add(r["method"]); confs.append(r["confidence"]); continue
        if MODEL_NAME == "student":
            o, c = mt(variant).translate(p)
            import student
            if not (student.acceptable(o, c) and keeps_known(o, r["output"])):
                outs.append(r["output"]); methods.add(r["method"] + "(neural rejected)"); confs.append(r["confidence"]); continue
        else:
            o, c = mt().translate(p, langs[0])
        outs.append(F.tr_fallback(F.fold_latin(o)) if ascii_out else o); methods.add("neural"); confs.append(c)
    out = ", ".join(x for x in outs if x)
    return dict(fast, output=out, method="+".join(sorted(methods)), confidence=F.r3(min(confs)) if confs else 0.0,
                model="accurate")


def normalize(items, ascii_out=False, variant=None):
    if not isinstance(items, (list, tuple)):
        raise F.NormalizerError("BAD_INPUT", "normalize() takes a list of items")
    return [process(*F.check_item(it), ascii_out, variant) for it in items]
