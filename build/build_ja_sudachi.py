"""Japanese readings from SudachiDict-core (Apache-2.0) instead of kakasi (GPL-3.0).

Word surfaces are collected by tokenizing (Sudachi split modes A and B):
  - Japanese product titles from Amazon ESCI (jp locale, Apache-2.0),
  - every gazetteer key written with kanji/kana, the glossary / address forms, the test inputs,
  - Japanese outputs of the OPUS teacher (bt_opus_v2.jsonl, if present),
plus every single kanji that Sudachi knows on its own. Reading = Sudachi reading_form, katakana -> hiragana.
Output: data/translit/ja.json  {"w": {first_char: {word: hiragana}}}  (same format as before)."""
import glob, json, os, re, sys
import pyarrow.parquet as pq
from sudachipy import Dictionary, SplitMode
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
KANJI = re.compile(r"[㐀-䶿一-鿿豈-﫿々〆ヶ]")
KATA = re.compile(r"^[゠-ヿー]+$")


def hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def texts():
    for f in sorted(glob.glob(os.path.join(HERE, "work", "dl_esci", "*.parquet"))):
        tb = pq.read_table(f, columns=["product_locale", "product_title"]).to_pydict()
        seen = set()
        for loc, t in zip(tb["product_locale"], tb["product_title"]):
            if loc == "jp" and t not in seen:
                seen.add(t); yield t
    sys.path.insert(0, os.path.join(ROOT, "runtime")); import normalizer
    for k in normalizer.gaz():
        if KANJI.search(k):
            yield k
    for f in ("glossary.json", "rules.json"):
        yield json.dumps(json.load(open(os.path.join(ROOT, "data", f), encoding="utf-8")), ensure_ascii=False)
    for f in ("cases.json", "blind.json"):
        for c in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8")):
            yield c["in"]
    p = os.path.join(HERE, "work", "bt_opus_v2.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["lang"] == "ja": yield r["src"]


def main():
    from collections import Counter, defaultdict
    tok = Dictionary(dict="core").tokenizer(); seen = defaultdict(Counter); n = 0
    def add(m, w=1):
        s, r = m.surface(), m.reading_form()
        if KANJI.search(s) and not m.is_oov() and KATA.match(r) and len(s) <= 12:
            seen[s][hira(r)] += w
    for t in texts():
        for chunk in re.findall(r"[^\s\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]{1,200}", t):
            for mode in (SplitMode.A, SplitMode.B, SplitMode.C):
                for m in tok.tokenize(chunk, mode):
                    add(m)
        n += 1
        if n % 50000 == 0: print("texts", n, "words", len(words), flush=True)
    for cp in list(range(0x4E00, 0x9FFF + 1)) + [ord("々"), ord("ヶ")]:
        for m in tok.tokenize(chr(cp), SplitMode.A):
            if m.surface() == chr(cp): add(m, 0.5)      # standalone reading: only a fallback / tie-break
    words = {w: c.most_common(1)[0][0] for w, c in seen.items()}      # most frequent reading in context
    ja = {}
    for w, y in words.items():
        ja.setdefault(w[0], {})[w] = y
    p = os.path.join(ROOT, "data", "translit", "ja.json")
    json.dump({"w": ja, "source": "SudachiDict-core (Apache-2.0)"}, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("words", len(words), "first chars", len(ja), round(os.path.getsize(p) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
