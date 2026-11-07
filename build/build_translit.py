"""Export transliteration data used by both runtimes (build time only).

data/translit/
  fallback.json  codepoint -> ASCII (any-ascii data) for the scripts we serve
  zh.json        {"c": char -> pinyin (toneless, most common), "w": word -> "syl syl"}  (pypinyin)
  ja.json        {"w": {first_char: {word: hiragana}}} (pykakasi kanwadict)
  th.json        {"w": word -> RTGS romanization}  (pythainlp words + royin, silent-letter fixes)
"""
import json, os, pickle, re, sys, unicodedata
from anyascii import anyascii
from pypinyin import pinyin_dict, phrases_dict
import pykakasi
from pythainlp.corpus import thai_words
from pythainlp.transliterate import romanize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "translit")
os.makedirs(OUT, exist_ok=True)


def dump(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"{name}: {os.path.getsize(p)/1e6:.2f} MB")


def strip_tones(p):
    p = unicodedata.normalize("NFD", p)
    p = "".join(c for c in p if not unicodedata.combining(c))
    return p.replace("ü", "u").replace("v", "u")


# ---- fallback: any-ascii for Latin ext, Greek, Cyrillic, Arabic, Devanagari, Thai, Kana, Han, fullwidth
RANGES = [(0x00A0, 0x024F), (0x1E00, 0x1EFF), (0x0370, 0x03FF), (0x0400, 0x04FF), (0x0600, 0x06FF),
          (0x0900, 0x097F), (0x0E00, 0x0E7F), (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
          (0xAC00, 0xD7AF), (0xFF00, 0xFFEF), (0x2000, 0x206F), (0x3000, 0x303F)]
fb = {}
for a, b in RANGES:
    for cp in range(a, b + 1):
        ch = chr(cp)
        r = anyascii(ch)
        if r and r != ch:
            fb[ch] = r
dump("fallback.json", fb)

# ---- Chinese
chars = {}
for cp, v in pinyin_dict.pinyin_dict.items():
    chars[chr(cp)] = strip_tones(v.split(",")[0])
words = {}
for w, syls in phrases_dict.phrases_dict.items():
    if 2 <= len(w) <= 6:
        words[w] = " ".join(strip_tones(s[0]) for s in syls)
dump("zh.json", {"c": chars, "w": words})

# ---- Japanese (kanji words -> hiragana; first reading)
P = os.path.join(os.path.dirname(pykakasi.__file__), "data")
kw = pickle.load(open(os.path.join(P, "kanwadict4.db"), "rb"))
ja = {}
for cp, table in kw.items():
    d = {}
    for w, readings in table.items():
        yomi = next((y for y, con in readings if con is None), readings[0][0])
        d[w] = yomi
    ja[chr(cp)] = d
dump("ja.json", {"w": ja})

# ---- Thai (dictionary words -> RTGS)


def th_fix(w):
    w = re.sub("ห(?=[งญนมยรลว])", "", w)       # silent leading ho hip
    w = re.sub("อ(?=ย[่้]?า)", "", w)           # silent o ang
    return w


th = {}
for w in thai_words():
    if 1 <= len(w) <= 20:
        try:
            th[w] = romanize(th_fix(w), engine="royin")
        except Exception:
            pass
dump("th.json", {"w": th})
