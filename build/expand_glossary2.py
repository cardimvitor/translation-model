"""Grow the product glossary further (v5): ~4,000 more frequent product words and two/three-word combinations
from the ESCI and ABO English titles, added to build/glossary_auto.json (existing rows kept; backup in work/).

Translations: OPUS-MT forward (14 languages) or Hy-MT2-1.8B forward (ko, th: no usable OPUS en->ko/th model),
each kept only when OPUS-MT translating it back gives the same English term (round trip, two models agree).
A term is kept when confirmed in at least 4 languages. Same filters as expand_glossary.py (stop words, brands).
  python expand_glossary2.py <hy.gguf>"""
import json, os, re, shutil, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import glossary_src as GS
from expand_glossary import STOP, FUNC, COLORS, MATERIALS, SIZES, QUALITY, same, norm
LANGS = GS.LANGS
GLUE = {"and", "for", "with", "of", "the", "to", "in", "on", "a"}


def terms(have, n_uni=2500, n_bi=1200, n_tri=300):
    import encheck
    words = encheck._m()["words"]
    T = json.load(open(os.path.join(HERE, "work", "titles_esci.json"), encoding="utf-8"))
    T += json.load(open(os.path.join(HERE, "work", "abo_titles.json"), encoding="utf-8"))
    uni, bi, tri, cap = Counter(), Counter(), Counter(), Counter()
    for t in T:
        w = re.findall(r"[A-Za-z][A-Za-z'-]+", t)
        for i, x in enumerate(w):
            lx = x.lower(); uni[lx] += 1
            if i and x[0].isupper():
                cap[lx] += 1
            if i + 1 < len(w):
                bi[(lx, w[i + 1].lower())] += 1
            if i + 2 < len(w):
                tri[(lx, w[i + 1].lower(), w[i + 2].lower())] += 1
    ok = lambda x: x in words and x not in STOP and x not in FUNC and len(x) >= 3
    U = [w for w, c in uni.most_common() if ok(w) and w not in have and cap[w] < 0.9 * c][:n_uni]
    B = [f"{a} {b}" for (a, b), c in bi.most_common(20000) if c >= 8 and ok(a) and ok(b) and a != b
         and f"{a} {b}" not in have][:n_bi]
    C = [f"{a} {b} {d}" for (a, b, d), c in tri.most_common(20000) if c >= 8 and ok(a) and ok(d) and (ok(b) or b in ("of",))
         and f"{a} {b} {d}" not in have][:n_tri]
    return U + B + C


def main(hy_path):
    import torch, opus_gen as O
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    p = os.path.join(HERE, "glossary_auto.json"); A = json.load(open(p, encoding="utf-8"))
    shutil.copy(p, os.path.join(HERE, "work", "glossary_auto_v4_backup.json"))
    have = {r[0] for r in GS.G} | {r[0] for r in A["G"]}
    T = terms(have); print("new candidate terms", len(T), T[:20], T[-10:], flush=True)
    cells = {t: [""] * len(LANGS) for t in T}; hy = None
    for li, l in enumerate(LANGS):
        if O.FWD[l][0] == "m2m100":
            from hy_gen import Hy
            hy = hy or Hy(hy_path); fw = hy.translate(T, "en", l)
        else:
            fw = []
            for i in range(0, len(T), 64):
                fw += O.fwd(T[i:i + 64], l)
        bk = []
        for i in range(0, len(fw), 64):
            bk += O.rev([x for x, _ in fw[i:i + 64]], l)
        k = 0
        for t, (x, _), (b, _) in zip(T, fw, bk):
            x = x.strip(" .。!?¡¿\"'")
            if x and same(b, t) and len(x) <= 3 * len(t) + 6 and x.lower() != t.lower():
                cells[t][li] = x.lower(); k += 1
        print(l, "confirmed", k, "/", len(T), flush=True)
    new = [[t] + cells[t] for t in T if sum(1 for c in cells[t] if c) >= 4]
    for t in T:
        A["rank"][t] = ("color" if t in COLORS else "material" if t in MATERIALS else "size" if t in SIZES
                        else "quality" if t in QUALITY else "item")
    A["G"] += new; A["source"] += "; v5: +ESCI/ABO terms, OPUS-MT or Hy-MT2-1.8B forward, OPUS-MT round trip"
    json.dump(A, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("added terms", len(new), "total auto terms", len(A["G"]))


if __name__ == "__main__":
    main(sys.argv[1])
