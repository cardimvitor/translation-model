"""Word-level romanization tables mined from GeoNames, for scripts that cannot be romanized letter by letter:
Arabic (no written short vowels) and Hindi (schwa deletion, English loan words).
A place with an Arabic (or Hindi) name of n words and a Latin name of n words gives n word pairs; a pair is kept when it was
seen at least MIN times, is the most common reading of that word, and its consonants agree with the letter-by-letter reading.
  python build/mine_translit.py    -> adds to data/rules.json "ar_names" and "hi_names" (existing entries are kept)"""
import difflib, json, os, re, sys, unicodedata
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); G = os.path.join(ROOT, "build", "work", "geonames"); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import normalizer as F
MIN = int(os.environ.get("MIN", 3)); ART = {"al", "ar", "as", "at", "ash", "ad", "an", "az", "adh", "ath", "el", "ed", "en", "er", "es", "ez", "et", "aṭ", "aş", "az̧", "aḑ"}
names = {}
for l in open(os.path.join(G, "alternateNamesV2.txt"), encoding="utf-8"):
    p = l.rstrip("\n").split("\t")
    if p[2] in ("ar", "hi") and not (len(p) > 7 and (p[7] == "1" or p[6] == "1")):
        names.setdefault((p[1], p[2]), p[3])
ids = {k[0] for k in names}; main = {}
for l in open(os.path.join(G, "places_all.tsv"), encoding="utf-8"):
    gid, name = l.split("\t", 2)[:2]
    if gid in ids:
        main[gid] = name
fold = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s.replace("ʻ", "").replace("’", "").replace("‘", "").replace("'", "").replace("`", "")) if not unicodedata.combining(c))
cons = lambda s: re.sub(r"[aeiouy\-' ]", "", fold(s).lower()).replace("kh", "x").replace("sh", "c").replace("th", "t").replace("dh", "d").replace("gh", "g").replace("q", "k").replace("j", "g").replace("z", "s").replace("w", "").replace("h", "")
cnt = {"ar": defaultdict(Counter), "hi": defaultdict(Counter)}
for (gid, lg), nat in names.items():
    if gid not in main:
        continue
    en = [t for t in re.split(r"[\s\-]+", fold(main[gid])) if t]; nt = [t for t in nat.split() if t]
    if lg == "ar":
        m = []; i = 0
        while i < len(en):
            if en[i].lower() in ART and i + 1 < len(en):
                m.append(("al", en[i + 1])); i += 2
            else:
                m.append(("", en[i])); i += 1
        if len(m) != len(nt):
            continue
        for (art, e), a in zip(m, nt):
            if a.startswith("ال") and len(a) > 3 and art:
                cnt["ar"][a[2:]][e.capitalize()] += 1
            elif not art and not a.startswith("ال"):
                cnt["ar"][a][e.capitalize()] += 1
    else:
        if len(en) == len(nt):
            for e, a in zip(en, nt):
                cnt["hi"][a][e.capitalize()] += 1
rp = os.path.join(ROOT, "data", "rules.json"); R = json.load(open(rp, encoding="utf-8")); st = {}
for lg, key in (("ar", "ar_names"), ("hi", "hi_names")):
    T = R.setdefault(key, {}); n0 = len(T)
    for w, c in cnt[lg].items():
        e, k = c.most_common(1)[0]
        if w in T or k < MIN or k < 0.6 * sum(c.values()) or len(w) < 3 or not re.fullmatch(r"[A-Za-z]{2,}", e):
            continue
        base = F.tr_arabic(w) if lg == "ar" else F.tr_devanagari(w)
        if difflib.SequenceMatcher(None, cons(base), cons(e)).ratio() < 0.7:
            continue
        T[w] = e
    st[key] = (n0, len(T))
json.dump(R, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=0); print(st)
for w in ("تحرير", "قاهرة", "رياض", "نصر", "جديدة", "بाग", "बाग", "नगर", "पुर", "रोड", "दिल्ली", "बेंगलुरु", "करोल"):
    print(w, R["ar_names"].get(w) or R["hi_names"].get(w))
