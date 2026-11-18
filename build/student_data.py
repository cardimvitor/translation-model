"""Training pairs (foreign -> English) for the Accurate student model. Output: build/work/pairs_{train,dev}.jsonl

  bt     back-translated product titles (NLLB, English side is the original clean title), filtered
  gloss  glossary forms -> English term
  gaz    place names in the 16 languages' scripts -> English name (5% of cities held out)
Inputs are normalized like the runtime does (student_input); targets are lowercase English.
"""
import json, os, random, re, sys, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, HERE)
import normalizer as F, encheck
import glossary_src as GS
W = os.path.join(HERE, "work")
SCRIPT_OF_LANG = {"zh": "Han", "ja": None, "ko": "Hangul", "ar": "Arabic", "ru": "Cyrillic", "hi": "Devanagari", "th": "Thai"}


def student_input(s):
    """Same normalization as the runtimes: NFKC, lowercase, collapsed spaces."""
    return re.sub(r"\s+", " ", F.lower(unicodedata.normalize("NFKC", s))).strip()


def good_bt(r):
    src, en, l = r["src"], r["en"], r["lang"]
    if not src or src.lower() == en.lower() or r["conf"] < 0.25:
        return False
    if not (0.3 <= len(src) / max(1, len(en)) <= 3.0):
        return False
    sc = F.scripts_in(src)
    want = SCRIPT_OF_LANG.get(l, "Latin")
    if l == "ja":
        if not ("Kana" in sc or "Han" in sc):
            return False
    elif want not in sc:
        return False
    if want == "Latin" and encheck.is_english(src, "strict")["english"]:
        return False                     # not actually translated
    return True


def main():
    random.seed(3)
    pairs = []
    bt = [json.loads(l) for l in open(os.path.join(W, "bt.jsonl"), encoding="utf-8")]
    titles = sorted({r["en"] for r in bt}); random.shuffle(titles)
    dev_titles = set(titles[: len(titles) // 20])
    kept = 0
    for r in bt:
        if good_bt(r):
            kept += 1
            pairs.append(("dev" if r["en"] in dev_titles else "train", "bt", r["lang"], student_input(r["src"]), r["en"].lower()))
    print(f"bt kept {kept}/{len(bt)}")
    for row in GS.G:
        for l, cell in zip(GS.LANGS, row[1:]):
            for f in filter(None, cell.split("|")):
                pairs.append(("train", "gloss", l, student_input(f), row[0].lower()))
    G = F.gaz()
    eng = sorted({v[0] for v in G.values() if v[1] == "city"}); random.shuffle(eng)
    dev_city = set(eng[: len(eng) // 20])
    by_script = {}
    for k, v in G.items():
        sc = F.scripts_in(k)
        if not sc:
            continue
        s = max(sc, key=sc.get)
        by_script.setdefault(s, []).append((k, v))
    for s, items in by_script.items():
        random.shuffle(items)
        for k, v in items[: (6000 if s == "Latin" else 2500)]:
            if v[1] == "country" or len(k) > 40:
                pass
            split = "dev" if v[0] in dev_city else "train"
            pairs.append((split, "gaz", s, k, v[0].lower()))
    random.shuffle(pairs)
    for split in ("train", "dev"):
        with open(os.path.join(W, f"pairs_{split}.jsonl"), "w", encoding="utf-8") as f:
            for p in pairs:
                if p[0] == split:
                    f.write(json.dumps({"kind": p[1], "lang": p[2], "src": p[3], "tgt": p[4]}, ensure_ascii=False) + "\n")
    from collections import Counter
    print("counts", Counter((p[0], p[1]) for p in pairs))


if __name__ == "__main__":
    main()
