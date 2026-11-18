"""Model 3: English check. Answers "is this text already English (no translation needed)?"

  non-Latin letters present          -> not English
  Latin text -> z = w0 + w1 * mean n-gram log-ratio (English vs other Latin languages)
                    + w2 * share of words in the English word list
             -> english if z > 0
Brand names and Latin place names carry no foreign evidence, so they count as English.
Training text: Europarl + UDHR (en, es, pt, fr, it, de) + UDHR (pl, tr, id, vi), CLDR names,
glossary forms. English word list: most frequent Europarl/UDHR English words + glossary
English + CLDR English names. Output: data/encheck.json (target < 300 KB).
"""
import json, os, random, re, sys, math, glob
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, HERE)
from normalizer import norm_key
import build_lid as BL
import glossary_src as GS
from babel import Locale

N_GRAMS = 3000; N_WORDS = 6000
OTHERS = ["es", "pt", "fr", "it", "de", "pl", "tr", "id", "vi"]


def grams(w):
    w = " " + w + " "
    return [w[i:i + n] for n in (1, 2, 3) for i in range(len(w) - n + 1) if w[i:i + n].strip()]


def main():
    random.seed(0)
    words = {l: norm_key(BL.texts(l)).split() for l in ["en"] + OTHERS}
    split = {l: (w[: int(len(w) * .8)], w[int(len(w) * .8):]) for l, w in words.items()}
    en_cldr = [norm_key(x) for x in list(Locale.parse("en").territories.values()) + list(Locale.parse("en").languages.values())]
    en_gloss = [norm_key(r[0]) for r in GS.G]
    oth_gloss = [norm_key(f) for r in GS.G for l, c in zip(GS.LANGS, r[1:]) if l in OTHERS for f in c.split("|") if f]

    # n-gram counts
    ce, co = Counter(), Counter()
    for w in split["en"][0] + en_cldr * 3 + en_gloss * 5:
        ce.update(grams(w))
    for l in OTHERS:
        tr = split[l][0]; random.shuffle(tr)
        for w in tr[: len(split["en"][0]) // len(OTHERS) * 2]:
            co.update(grams(w))
    for w in oth_gloss * 5:
        co.update(grams(w))
    vocab = set(ce) | set(co); Te, To = sum(ce.values()) + len(vocab), sum(co.values()) + len(vocab)
    llr = {g: math.log((ce[g] + 1) / Te) - math.log((co[g] + 1) / To) for g in vocab if ce[g] + co[g] >= 5}
    top = sorted(llr, key=lambda g: -abs(llr[g]) * math.log(1 + ce[g] + co[g]))[:N_GRAMS]
    table = {g: round(llr[g], 2) for g in top}

    # English word list: frequent words (excluding words that are just as frequent in other languages)
    fe = Counter(split["en"][0]); fo = Counter(w for l in OTHERS for w in split[l][0])
    cand = [w for w, c in fe.most_common(40000) if w.isalpha() and c >= 3 and fe[w] / (fo[w] + 1) > 2]
    wl = []
    for w in en_gloss + [t for x in en_cldr for t in x.split()] + cand:
        if w not in wl:
            wl.append(w)
    wl = wl[:N_WORDS]
    wset = set(wl)

    ASCII_OK = set("abcdefghijklmnopqrstuvwxyz0123456789'&+")

    def word_class(w, T):
        if not any(c.isalpha() for c in w):
            return "N"
        if w in wset:
            return "E"
        if any(c.isalpha() and c not in ASCII_OK for c in w):
            return "F"                                    # diacritics / non-English letters
        gs = grams(w); sc = sum(table.get(g, 0.0) for g in gs) / max(1, len(gs))
        return "F" if sc < -T else "N"

    def decide(s, T, R):
        cl = [word_class(w, T) for w in norm_key(s).split()]
        e, f = cl.count("E"), cl.count("F")
        return f == 0 or e >= R * f

    def sample(l, part, n, k):
        h = split[l][part]; out = []
        for _ in range(n):
            i = random.randrange(0, len(h) - k); out.append(" ".join(h[i:i + k]))
        return out

    # positives: English text, English place names, Fast-style outputs (English words + names)
    import geonamescache
    places = [v["name"] for v in geonamescache.GeonamesCache().get_cities().values() if v["name"].isascii()]
    random.shuffle(places)
    from normalizer import process as fast_process
    syn = []
    for name in places[:300]:
        syn.append(f"{name} Street {random.randint(1, 99)}"); syn.append(f"women's leather bag {name}")
    def dataset(part):
        P, Nn = [], []
        for k in (1, 2, 3, 5, 8):
            P += sample("en", part, 300, k)
            for l in OTHERS:
                Nn += sample(l, part, 35, k)
        P += places[300:900] if part == 0 else places[900:1500]
        P += syn[:300] if part == 0 else syn[300:]
        return P, Nn
    devP, devN = dataset(0)
    best = None
    for T in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5):
        for R in (1, 2, 3):
            acc = (sum(decide(x, T, R) for x in devP) / len(devP) + sum(not decide(x, T, R) for x in devN) / len(devN)) / 2
            if best is None or acc > best[0]:
                best = (acc, T, R)
    _, T, R = best
    print("relaxed rule: T", T, "R", R, "dev balanced acc", round(best[0], 3))

    # strict mode: logistic regression on (mean n-gram log-ratio, English-word share), English text vs foreign
    def feats(x):
        ws = norm_key(x).split(); gs = [g for w_ in ws for g in grams(w_)]
        return (sum(table.get(g, 0.0) for g in gs) / max(1, len(gs)), sum(w_ in wset for w_ in ws) / max(1, len(ws)))
    X = [feats(x) for x in devP[:1500]] + [feats(x) for x in devN]; Y = [1] * len(devP[:1500]) + [0] * len(devN)
    W = [0.0, 1.0, 1.0]
    for _ in range(3000):
        g3 = [0.0, 0.0, 0.0]
        for (a, b), y in zip(X, Y):
            z = W[0] + W[1] * a + W[2] * b; pr = 1 / (1 + math.exp(-max(-30, min(30, z))))
            for j, v in enumerate((1, a, b)):
                g3[j] += (pr - y) * v
        W = [wi - 0.5 * gi / len(X) for wi, gi in zip(W, g3)]
    def strict(x):
        a, b = feats(x)
        ws = norm_key(x).split(); e = sum(w_ in wset for w_ in ws)
        f = sum(1 for w_ in ws if w_ not in wset and any(c.isalpha() and c not in ASCII_OK for c in w_))
        return W[0] + W[1] * a + W[2] * b > 0 and (f == 0 or e >= R * f)
    model = {"g": table, "w": [round(x, 4) for x in W], "T": T, "R": R, "words": wl}

    rep = {}
    testP_place = places[900:1500]
    for mode, fn in (("strict", strict), ("relaxed", lambda x: decide(x, T, R))):
        random.seed(5)
        for name, k in (("1 word", 1), ("2-3 words", 2), ("5 words", 5), ("10 words", 10)):
            en_ok = sum(fn(x) for x in sample("en", 1, 500, k)) / 500
            ot_ok = sum(not fn(x) for l in OTHERS for x in sample(l, 1, 60, k)) / (60 * len(OTHERS))
            rep[f"{mode} {name}"] = {"English recognized": round(en_ok, 3), "foreign rejected": round(ot_ok, 3)}
        rep[f"{mode} place names"] = {"English recognized": round(sum(fn(x) for x in testP_place) / len(testP_place), 3)}
        rep[f"{mode} Fast-style outputs"] = {"English recognized": round(sum(fn(x) for x in syn[300:]) / len(syn[300:]), 3)}
    model["eval"] = rep
    p = os.path.join(ROOT, "data", "encheck.json")
    json.dump(model, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("size", round(os.path.getsize(p) / 1000), "KB")
    for k, v in rep.items():
        print(f"  {k:12s}", " | ".join(f"{a} {b:.1%}" for a, b in v.items()))


if __name__ == "__main__":
    main()
