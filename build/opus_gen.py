"""OPUS-MT teachers (Helsinki-NLP, CC BY 4.0 / Apache-2.0) for variant B: English -> each of the 16 languages,
plus the reverse direction for round-trip filtering. PyTorch Marian on CPU, batched greedy decoding."""
import os, math, torch
from transformers import MarianMTModel, MarianTokenizer
torch.set_num_threads(4)
H = "Helsinki-NLP/"
# language -> (forward model en->xx, target token or None, reverse model xx->en, reverse token or None)
FWD = {"zh": ("opus-mt-en-zh", ">>cmn_Hans<<"), "es": ("opus-mt-en-es", None), "ar": ("opus-mt-en-ar", ">>ara<<"),
       "ru": ("opus-mt-en-ru", None), "de": ("opus-mt-en-de", None), "fr": ("opus-mt-en-fr", None),
       "hi": ("opus-mt-en-hi", None), "id": ("opus-mt-en-id", None), "vi": ("opus-mt-en-vi", ">>vie<<"),
       "it": ("opus-mt-en-it", None), "pt": ("opus-mt-en-ROMANCE", ">>pt<<"), "ko": ("m2m100", "ko"),
       "tr": ("opus-mt-tc-big-en-tr", None), "ja": ("opus-tatoeba-en-ja", None), "pl": ("opus-mt-en-sla", ">>pol<<"),
       "th": ("m2m100", "th")}       # no usable OPUS en->ko / en->th model: Meta M2M100-418M (MIT)
REV = {"zh": "opus-mt-zh-en", "es": "opus-mt-es-en", "ar": "opus-mt-ar-en", "ru": "opus-mt-ru-en", "de": "opus-mt-de-en",
       "fr": "opus-mt-fr-en", "hi": "opus-mt-hi-en", "id": "opus-mt-id-en", "vi": "opus-mt-vi-en", "it": "opus-mt-it-en",
       "pt": "opus-mt-ROMANCE-en", "ko": "opus-mt-ko-en", "tr": "opus-mt-tr-en", "ja": "opus-mt-ja-en", "pl": "opus-mt-pl-en",
       "th": "opus-mt-th-en"}
_cache = {}


def _load(name):
    if name not in _cache:
        if len(_cache) >= 2:                # keep the forward and reverse model of one language in memory
            _cache.pop(next(iter(_cache)))
        tok = MarianTokenizer.from_pretrained(H + name); m = MarianMTModel.from_pretrained(H + name).eval()
        _cache[name] = (tok, m)
    return _cache[name]


@torch.inference_mode()
def translate(texts, model, prefix=None, max_new=48):
    """Returns [(text, confidence)], confidence = exp(mean token log-prob) of the greedy output."""
    tok, m = _load(model)
    src = [f"{prefix} {t}" if prefix else t for t in texts]
    enc = tok(src, return_tensors="pt", padding=True, truncation=True, max_length=96)
    out = m.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=max_new, max_length=None, output_scores=True, return_dict_in_generate=True)
    return _with_conf(tok, m, out, len(texts))


@torch.inference_mode()
def m2m(texts, lang, max_new=48):
    if "m2m100" not in _cache:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        if len(_cache) >= 2: _cache.pop(next(iter(_cache)))
        tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M"); tok.src_lang = "en"
        _cache["m2m100"] = (tok, M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").eval())
    tok, m = _cache["m2m100"]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=96)
    out = m.generate(**enc, forced_bos_token_id=tok.get_lang_id(lang), num_beams=1, max_new_tokens=max_new, max_length=None,
                     output_scores=True, return_dict_in_generate=True)
    return _with_conf(tok, m, out, len(texts), skip=2)


def _with_conf(tok, m, out, n_items, skip=1):
    seqs = out.sequences; res = []
    lp = torch.stack([torch.log_softmax(s.float(), -1) for s in out.scores], 1)
    first = seqs.shape[1] - lp.shape[1]
    for i in range(n_items):
        n = 0; tot = 0.0
        for j, t in enumerate(seqs[i, first:].tolist()):
            if t == m.config.pad_token_id:
                break
            if j >= skip - first:
                tot += float(lp[i, j, t]); n += 1
            if t == m.config.eos_token_id:
                break
        res.append((tok.decode(seqs[i], skip_special_tokens=True).strip(), round(math.exp(tot / max(1, n)), 3)))
    return res


def fwd(texts, lang):
    name, p = FWD[lang]
    return m2m(texts, p) if name == "m2m100" else translate(texts, name, p)


def rev(texts, lang):
    return translate(texts, REV[lang])
