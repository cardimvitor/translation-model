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
    """Neural part: a phone student (models/<variant>, default NEURAL_DIR or "student");
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


POLICY = {"name": os.environ.get("ACC_POLICY", "v2"), "tau": float(os.environ.get("ACC_TAU", 0.0)),
          "digits_are_address": os.environ.get("ACC_DIGITS", "0") == "1"}


def address_words(part, langs):
    """address_like without the "any digit" rule: only real address words (street, district ...) count."""
    toks = F._tokens(part); keys = [F.norm_key(t) for t in toks]
    for l in langs:
        A = F.rules()["address"].get(l, [])
        spaced = {f for forms, en, kind in A if kind not in ("s", "p") for f in forms}
        if any(k in spaced for k in keys):
            return True
        if l in ("zh", "ja", "ko", "th") and any(f in part for forms, en, kind in A if kind in ("s", "p") for f in forms if len(f) > 1):
            return True
    return False


_AEN = []


def address_en():
    """English words the address rules produce (Street, Ward, District ...)."""
    if not _AEN:
        _AEN.append({w for A in F.rules()["address"].values() for forms, en, kind in A for w in F.lower(en).lstrip("-").split() if len(w) > 2})
    return _AEN[0]


_KO_ADDR = re.compile(r"[a-z]-(?:gu|ro|gil|daero|dong|do|si|gun|eup|myeon)\b")
_DE_STREET = re.compile(r"(?i)[^\W\d_](?:straße|strasse|str\.|gasse|allee|platz|weg|damm|ufer)\s+\d+\s?[a-z]?(?=$|[\s,])")     # Hauptstraße 5, Bahnhofstr. 12


def is_address_v2(s, langs, fast):
    """Address words in the input, or a number together with a place name / address word in Fast's answer
    (product sizes like "38.5 EU" have numbers but no place)."""
    if address_words(s, langs) or _DE_STREET.search(s):
        return True
    o = fast["output"]
    if len(_KO_ADDR.findall(o)) >= 2:                    # Gangnam-gu Teheran-ro: Korean address parts
        return True
    if "dictionary" in fast["method"] and re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af\u0e00-\u0e7f]", s) \
            and sum(w.strip(",.") in address_en() for w in F.lower(o).split()) >= 2:
        return True                                      # two address words (Ward, City, District ...) in a CJK or Thai label, with or without a number
    if any(F.is_digit(c) for c in s):
        return "gazetteer" in fast["method"] or any(w.strip(",.") in address_en() for w in F.lower(fast["output"]).split())
    return False


def process_v2(op, text, ascii_out=False, variant=None):
    """Decision rule v2 (tuned on held-out names): the student sees the whole label and its answer is used when
    it is confident (>= tau), English, and keeps every word Fast took from the dictionary. Places, addresses
    (address words; digits alone do not make an address) and labels already in English stay with Fast."""
    fast = F.process(op, text, ascii_out)
    if fast["operation"] != "translate" or fast["method"] in ("gazetteer", "already-english") or not fast["output"]:
        return dict(fast, model="accurate")
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    langs = [l for l, _ in fast["language"]]
    if fast.get("coverage", 0) >= 1 or is_address_v2(s, langs, fast) or (POLICY["digits_are_address"] and address_like(s, langs)):
        return dict(fast, model="accurate")               # every word known, or an address: Fast
    o, c = mt(variant).translate(s)
    import student
    if o and c >= POLICY["tau"] and student.acceptable(o, 1.0) and keeps_known(o, fast["output"]):
        out = F.tr_fallback(F.fold_latin(o)) if ascii_out else o
        return dict(fast, output=out, method="neural", confidence=F.r3(c), model="accurate")
    return dict(fast, method=fast["method"] + "(neural rejected)", model="accurate")


def is_address_v3(s, langs, fast, pol):
    if not is_address_v2(s, langs, fast):
        return False
    mw = pol.get("addr_max_words", 0)
    if re.search(r"[,，、،]", s) and "gazetteer" in fast["method"] and (address_words(s, langs) or _DE_STREET.search(s)):
        return True                               # a street word and a known place in separate comma parts: an address, however long
    return not (mw and len(s.split()) > mw and fast.get("coverage", 0) < pol.get("addr_long_cov", 0.6))


_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*$")
_EDGE = re.compile(r"^[^\w]+|[^\w]+$")


def _toks(s):
    """Lower-cased words of a label, punctuation at the edges removed."""
    return [t for t in (_EDGE.sub("", w).replace("’", "'") for w in s.lower().split()) if t]


def _sim(a, b):
    """Share of character pairs two texts have in common (0..1), lower-cased."""
    a = " ".join(a.lower().split()); b = " ".join(b.lower().split())
    A = {a[i:i + 2] for i in range(len(a) - 1)}; B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / max(1, len(A | B))


def process_v3(op, text, ascii_out, variant, pol):
    """Decision rule v3, for a strong neural model (v6). Its settings ship with the model (vocab.json "policy"):
      keep_cov    stay with Fast when Fast knew every word of the label
      need_known  the neural answer must keep the words Fast took from the dictionary
      tau         minimum neural confidence
      need_english    the neural answer must pass the English check
      keep_gaz        a short label in which Fast found a known place name stays with Fast
      addr_max_words  a label with more words than this counts as an address only if Fast knew at least
                      addr_long_cov of its words (long product names often contain a number and a place word)
    Places, addresses and labels already in English always stay with Fast."""
    fast = F.process(op, text, ascii_out)
    if fast["operation"] == "translate" and fast["method"] == "already-english" and "w_tau" in pol:
        # Word check: a label the English check accepts may still hold a foreign word ("led ampul"). Words that are not in the
        # model's English word list are suspects; the neural answer is taken only if it is confident, drops every suspect
        # and keeps every other word.
        M = mt(variant); EW = getattr(M, "_en_words", None)
        if EW is None:
            EW = M._en_words = set(M.v.get("en_words") or []); M._xx_words = set(M.v.get("xx_words") or [])
        s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip(); toks = _toks(s)
        sus = [w for w in toks if len(w) >= 3 and _WORD.match(w) and w not in EW] if EW and len(toks) <= pol.get("w_max_words", 8) else []
        if sus:
            o, c = M.translate(s); ot = set(_toks(o or ""))
            # a suspect seen in foreign training text needs less confidence (w_tau_known) than an unknown word (w_tau)
            if o and c >= (pol.get("w_tau_known", pol["w_tau"]) if any(w in M._xx_words for w in sus) else pol["w_tau"]) and len(_toks(o)) <= len(toks) + 2 and not any(w in ot for w in sus) and all(w in ot for w in toks if w not in sus and len(w) >= 2):
                return dict(fast, output=F.tr_fallback(F.fold_latin(o)) if ascii_out else o, method="neural", confidence=F.r3(c), model="accurate")
    if fast["operation"] == "translate" and fast["method"] == "already-english" and fast["confidence"] < pol.get("en_sure", 0):
        # English double check: the English check was not sure. The neural answer replaces the label only if it is confident
        # (en_tau) and clearly different text (en_sim), i.e. a translation and not a small edit of English.
        s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
        o, c = mt(variant).translate(s)
        if o and c >= pol.get("en_tau", 0.85) and _sim(o, s) < pol.get("en_sim", 0.5) and len(o.split()) <= 2 * len(s.split()) + 2:
            return dict(fast, output=F.tr_fallback(F.fold_latin(o)) if ascii_out else o, method="neural", confidence=F.r3(c), model="accurate")
        return dict(fast, model="accurate")
    if fast["operation"] != "translate" or fast["method"] in ("gazetteer", "already-english") or not fast["output"]:
        return dict(fast, model="accurate")
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    langs = [l for l, _ in fast["language"]]
    if is_address_v3(s, langs, fast, pol) or (pol.get("keep_cov") and fast.get("coverage", 0) >= 1) \
            or (pol.get("keep_gaz") and "gazetteer" in fast["method"] and len(s.split()) <= pol.get("addr_max_words", 8)):
        return dict(fast, model="accurate")               # address, every word known, or a short label with a known place
    o, c = mt(variant).translate(s)
    import student
    if " " not in s and fast["method"] == "transliteration" and c < pol.get("one_tau", 0.0):
        o = ""                                           # one unknown word and a hesitant translator: most likely a name, keep the romanization
    if o and c >= pol.get("tau", 0.0) and student.acceptable(o, 1.0, pol.get("need_english", True)) and (not pol.get("need_known") or keeps_known(o, fast["output"])):
        out = F.tr_fallback(F.fold_latin(o)) if ascii_out else o
        return dict(fast, output=out, method="neural", confidence=F.r3(c), model="accurate")
    return dict(fast, method=fast["method"] + "(neural rejected)", model="accurate")


def reliable(r, pol):
    """Adds "reliable": True / False when the model ships a calibrated threshold (policy "rel_tau").
    True: a known place, text already in English, every word known to Fast, or a neural answer whose confidence is at least
    rel_tau. False means "show it, but a person should check it"."""
    if "rel_tau" in pol:
        m = r.get("method", "")
        r["reliable"] = bool(r.get("operation") != "translate" or m == "gazetteer" or (m == "already-english" and r.get("confidence", 0) >= pol.get("en_sure", 0))
                             or (m == "neural" and r.get("confidence", 0) >= pol["rel_tau"])
                             or (m != "neural" and "rejected" not in m and r.get("coverage", 0) >= 1))
    return r


def process(op, text, ascii_out=False, variant=None):
    if MODEL_NAME == "student" and POLICY["name"] != "v1":
        pol = getattr(mt(variant), "v", {}).get("policy")
        if pol:
            return reliable(process_v3(op, text, ascii_out, variant, pol), pol)
    if POLICY["name"] == "v2" and MODEL_NAME == "student":
        return process_v2(op, text, ascii_out, variant)
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
