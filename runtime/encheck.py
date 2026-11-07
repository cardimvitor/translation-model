"""Model 3: English check, Python reference. is_english(text) -> {"english": bool, "score": float, "reason": str}"""
import json, math, os, unicodedata
from normalizer import norm_key, script_of, DATA

_M = {}


def _m():
    if not _M:
        m = json.load(open(os.path.join(DATA, "encheck.json"), encoding="utf-8"))
        _M.update(g=m["g"], w=m["w"], T=m["T"], R=m["R"], words=set(m["words"]))
    return _M


def _grams(w):
    w = " " + w + " "
    return [w[i:i + n] for n in (1, 2, 3) for i in range(len(w) - n + 1) if w[i:i + n].strip()]


ASCII_OK = set("abcdefghijklmnopqrstuvwxyz0123456789'&+")


def _word_class(w, m):
    if not any(unicodedata.category(c)[0] == "L" for c in w):
        return "N"
    if w in m["words"]:
        return "E"
    if any(unicodedata.category(c)[0] == "L" and c not in ASCII_OK for c in w):
        return "F"
    gs = _grams(w); sc = sum(m["g"].get(g, 0.0) for g in gs) / max(1, len(gs))
    return "F" if sc < -m["T"] else "N"


def is_english(text, mode="strict"):
    """strict: is the INPUT already English (safe to skip translation)? Names/brands may be refused.
    relaxed: does an OUTPUT look like English? Names and brands are neutral."""
    s = unicodedata.normalize("NFKC", text or "")
    letters = [c for c in s if unicodedata.category(c)[0] == "L"]
    if not letters:
        return {"english": True, "score": 1.0, "reason": "no letters"}
    if any(script_of(c) not in ("Latin",) for c in letters):
        return {"english": False, "score": 0.0, "reason": "non-Latin script"}
    m = _m(); k = norm_key(s); ws = k.split()
    if mode == "relaxed":
        cl = [_word_class(w, m) for w in ws]; e, f = cl.count("E"), cl.count("F")
        ok = f == 0 or e >= m["R"] * f
        return {"english": ok, "score": round((e + 1) / (e + f + 2), 3), "reason": f"relaxed: {e} English, {f} foreign words"}
    gs = [g for w in ws for g in _grams(w)]
    a = sum(m["g"].get(g, 0.0) for g in gs) / max(1, len(gs))
    b = sum(w in m["words"] for w in ws) / max(1, len(ws))
    z = m["w"][0] + m["w"][1] * a + m["w"][2] * b
    p = 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
    e = sum(w in m["words"] for w in ws)
    f = sum(1 for w in ws if w not in m["words"] and any(unicodedata.category(c)[0] == "L" and c not in ASCII_OK for c in w))
    if f and e < m["R"] * f:
        return {"english": False, "score": round(min(p, 0.1), 3), "reason": "non-English letters"}
    return {"english": z > 0, "score": round(p, 3), "reason": "model"}
