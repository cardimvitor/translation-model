"""v4 training pairs per student variant. Output: build/work/pairs_<V>_{train,dev}.jsonl

  A  NLLB-200 teacher (non-commercial):   bt_nllb_v2.jsonl + the v3 NLLB pairs (bt.jsonl)
  B  OPUS-MT / M2M100 teachers (commercial OK): bt_opus_v2.jsonl, kept only if the round trip back to English
     keeps >= half of the title's content words
  C  A + B

Every variant also gets: glossary pairs (hand-written + round-trip-confirmed auto glossary), place names
(5% of cities held out) and a "noise" copy of 10% of the product pairs with accents removed (people often type
without them). Titles are held out by bt_v2.is_dev (the same 5% for every variant).
"""
import json, os, random, re, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, HERE)
import normalizer as F
import glossary_src as GS
from student_data import student_input, good_bt
from bt_v2 import is_dev
W = os.path.join(HERE, "work")
LATIN_LANGS = {"es", "pt", "de", "fr", "tr", "id", "vi", "it", "pl"}


def words(s):
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def round_trip_ok(r):
    a, b = words(r["en"]), words(r.get("back", ""))
    return not a or len(a & b) / len(a) >= 0.5


def read(name):
    p = os.path.join(W, name)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def bt_pairs(variant):
    rows = []
    if variant in ("A", "C"):
        rows += [dict(r, teacher="nllb") for r in read("bt_nllb_v2.jsonl") + read("bt.jsonl")]
    if variant in ("B", "C"):
        rows += [dict(r, teacher="opus") for r in read("bt_opus_v2.jsonl")]
    out, seen, stat = [], set(), Counter()
    for r in rows:
        stat["read_" + r["teacher"]] += 1
        if not good_bt(r) or (r["teacher"] == "opus" and not round_trip_ok(r)):
            continue
        src = student_input(r["src"]); key = (src, r["en"].lower())
        if key in seen:
            continue
        seen.add(key); stat["kept_" + r["teacher"]] += 1
        out.append(("dev" if is_dev(r["en"]) else "train", "bt", r["lang"], src, r["en"].lower()))
    print(variant, dict(stat))
    return out


def gloss_pairs():
    out = []
    for row in GS.G:
        for l, cell in zip(GS.LANGS, row[1:]):
            for f in filter(None, cell.split("|")):
                out.append(("train", "gloss", l, student_input(f), row[0].lower()))
    p = os.path.join(HERE, "glossary_auto.json")
    if os.path.exists(p):
        for row in json.load(open(p, encoding="utf-8"))["G"]:
            for l, cell in zip(GS.LANGS, row[1:]):
                if cell:
                    out.append(("train", "gloss", l, student_input(cell), row[0].lower()))
    return out


def gaz_pairs(rng):
    G = F.gaz(); out = []
    eng = sorted({v[0] for v in G.values() if v[1] == "city"}); rng.shuffle(eng)
    dev_city = set(eng[: len(eng) // 20])
    by_script = {}
    for k, v in G.items():
        sc = F.scripts_in(k)
        if sc:
            by_script.setdefault(max(sc, key=sc.get), []).append((k, v))
    for s, items in sorted(by_script.items()):
        rng.shuffle(items)
        for k, v in items[: (8000 if s == "Latin" else 3000)]:
            if len(k) <= 40:
                out.append(("dev" if v[0] in dev_city else "train", "gaz", s, k, v[0].lower()))
    return out


def main():
    for v in sys.argv[1:] or ["A", "B", "C"]:
        rng = random.Random(3)
        pairs = bt_pairs(v) + gloss_pairs() + gaz_pairs(rng)
        noise = [("train", "noise", l, F.fold_latin(s), t) for sp, k, l, s, t in pairs
                 if sp == "train" and k in ("bt", "gloss") and l in LATIN_LANGS and F.fold_latin(s) != s and rng.random() < 0.25]
        pairs += noise
        rng.shuffle(pairs)
        for split in ("train", "dev"):
            with open(os.path.join(W, f"pairs_{v}_{split}.jsonl"), "w", encoding="utf-8") as f:
                for p in pairs:
                    if p[0] == split:
                        f.write(json.dumps({"kind": p[1], "lang": p[2], "src": p[3], "tgt": p[4]}, ensure_ascii=False) + "\n")
        print(v, "counts", dict(Counter((p[0], p[1]) for p in pairs)))


if __name__ == "__main__":
    main()
