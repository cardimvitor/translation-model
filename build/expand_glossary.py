"""Grow the product glossary (91 hand-written terms) with ~1,000 frequent product words and compounds.

Terms: the most frequent words / two-word compounds in the English titles (ESCI, Shopify, H&M) that are
common English words (not brands). Translations come only from commercially licensed teachers
(OPUS-MT, M2M100 for ko/th) and are kept only when translating back gives the same English term
(round trip), so every entry is confirmed by two models. Ambiguous words (case, set, stand ...) are only
used inside compounds. Output: build/glossary_auto.json {"G": [[en, cell_zh, ... cell_th]], "rank": {en: category}}
"""
import json, os, re, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import glossary_src as GS
LANGS = GS.LANGS
STOP = set("""a an the and or of for with to in on at by from as is it its this that be are was your you our my
new pack set kit lot piece pieces count size sizes inch inches ft oz lb lbs mm cm ml pcs pc x free plus pro max mini one
two three four five six seven eight ten edition version series style compatible replacement premium quality made use
case stand top bar mount light cover clip holder sheet pad cap base plate ring band tie line block board box bag
not without how out over under down way can first back day life time year end self friends love jordan city show live true
great best good happy little old products company class number team world family heart magic women men man play drive
throw work products sets books tools feet american french western polish united states we are but you my no up on attack
""".split())
FUNC = set("not without you we are but on my no how out up attack in to of and for with the".split())
AMBIG_OK_IN_COMPOUND = True
COLORS = "beige purple orange navy gold silver khaki ivory cream tan teal turquoise burgundy maroon".split()
MATERIALS = "metal aluminum bamboo ceramic rubber nylon polyester linen denim canvas velvet crystal copper iron paper steel silicone fleece suede lace".split()
SIZES = "big long short wide tall extra".split()
QUALITY = "portable adjustable electric digital automatic rechargeable foldable organic natural handmade vintage soft heavy durable magnetic solar".split()


def norm(s):
    s = re.sub(r"[^\w\s'-]", " ", s.lower()); s = re.sub(r"^(the|a|an)\s+", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def same(back, term):
    b, t = norm(back), norm(term)
    sing = lambda x: " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in x.split())
    return b == t or sing(b) == sing(t)


def terms(n_uni=800, n_bi=250):
    import encheck
    words = encheck._m()["words"]
    T = json.load(open(os.path.join(HERE, "work", "titles_esci.json"), encoding="utf-8"))
    uni, bi, cap = Counter(), Counter(), Counter()
    for t in T:
        w = [x for x in re.findall(r"[A-Za-z][A-Za-z'-]+", t)]
        for i, x in enumerate(w):
            lx = x.lower(); uni[lx] += 1
            if i and x[0].isupper(): cap[lx] += 1
            if i + 1 < len(w):
                bi[(lx, w[i + 1].lower())] += 1
    have = {r[0] for r in GS.G}
    U = [w for w, c in uni.most_common() if w in words and w not in STOP and w not in have and len(w) >= 3
         and cap[w] < 0.95 * c][:n_uni]
    B = [f"{a} {b}" for (a, b), c in bi.most_common(4000) if a in words and b in words and a not in STOP - {"case", "light"}
         and b not in ("and", "for", "with", "of", "the", "to", "in") and a not in ("and", "for", "with", "of", "the", "to", "in")
         and f"{a} {b}" not in have and a not in FUNC and b not in FUNC and not (a in COLORS + ["black", "white", "red", "blue"] and b in COLORS + ["black", "white", "red", "blue"])
         and a != b][:n_bi]
    return U + B


def main():
    import opus_gen as O
    out_p = os.path.join(HERE, "glossary_auto.json")
    T = terms(); print("terms", len(T), T[:30], T[-20:], flush=True)
    cells = {t: [""] * len(LANGS) for t in T}; stats = {}
    for li, l in enumerate(LANGS):
        fw = []
        for i in range(0, len(T), 64):
            fw += O.fwd(T[i:i + 64], l)
        bk = []
        for i in range(0, len(fw), 64):
            bk += O.rev([x for x, _ in fw[i:i + 64]], l)
        ok = 0
        for t, (x, c), (b, _) in zip(T, fw, bk):
            x = x.strip(" .。!?¡¿\"'")
            if x and same(b, t) and len(x) <= 3 * len(t) + 6 and x.lower() != t.lower():
                cells[t][li] = x.lower() if l not in ("de",) else x.lower(); ok += 1
        stats[l] = ok; print(l, "confirmed", ok, "/", len(T), flush=True)
    rank = {}
    for t in T:
        last = t.split()[-1]
        rank[t] = ("color" if t in COLORS else "material" if t in MATERIALS else "size" if t in SIZES
                   else "quality" if t in QUALITY else "item")
    G = [[t] + cells[t] for t in T if sum(1 for c in cells[t] if c) >= 4]
    json.dump({"G": G, "rank": rank, "stats": stats, "source": "OPUS-MT / M2M100 with round-trip agreement"},
              open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("kept terms", len(G))


if __name__ == "__main__":
    if sys.argv[1:] == ["terms"]:
        T = terms(); print(len(T)); print(T[:200]); print(T[-250:])
    else:
        main()
