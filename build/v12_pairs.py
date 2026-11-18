"""v12: training pairs for labels that mix English with another language, all 16 languages. No teacher needed.
  A  foreign label, one known word swapped for English        "deri ceket" -> "leather ceket"           => leather jacket
  B  English label, one word swapped for the foreign word     "wireless headphones" -> "wireless kulaklık" => wireless headphones
  C  English modifier before or after a foreign label         "diş fırçası electric"                    => electric toothbrush
  D  size / pack / unit after a foreign label                 "spor ayakkabı size 42"                   => sneakers size 42
Word pairs come from the product word list and from one-word training pairs. Test sources and test answers are left out.
  python build/v12_pairs.py   -> build/work/pairs_NEW_v12.jsonl"""
import json, os, random, re, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
from student_data import student_input
rd = lambda n: [json.loads(l) for l in open(os.path.join(W, n), encoding="utf-8")] if os.path.exists(os.path.join(W, n)) else []
rng = random.Random(12); NOSPACE = {"zh", "ja", "th"}; LANGS = "zh es pt ar ru ja ko de fr hi tr id vi it pl th".split()
held = set()
for f in ("gold_products.json", "dev_labels.json", "mixed_labels.json", "mixed_tr.json"):
    for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]:
        held |= {student_input(x) for x in (g["src"].values() if isinstance(g["src"], dict) else [g["src"]])} | {student_input(e) for e in (g["en"] if isinstance(g["en"], list) else [g["en"]])}
drop = {(d["src"], d["tgt"]) for d in rd("drop_pairs.jsonl")}
P = [p for n in ("pairs_NEW_v10.jsonl", "pairs_HQ_train.jsonl", "pairs_Q_train.jsonl", "pairs_NEW_v9.jsonl") for p in rd(n)
     if (p["src"], p["tgt"]) not in drop and p["src"] != p["tgt"] and p["lang"] in LANGS] + [p for p in rd("pairs_NEW_v11.jsonl") if p.get("kind") in ("hy7", "hy7_real")]
GL = json.load(open(os.path.join(ROOT, "data", "glossary.json"), encoding="utf-8"))
latin = re.compile(r"[a-z][a-z'\-]+$")
# word pairs: foreign word -> English word
lex = {l: defaultdict(Counter) for l in LANGS}
for p in P:
    s, t = p["src"], p["tgt"]
    if len(t.split()) == 1 and latin.match(t) and (len(s.split()) == 1 and (len(s) <= 8 if p["lang"] in NOSPACE else True)):
        lex[p["lang"]][s][t] += 1
for l in LANGS:
    for f, e in GL.get(l, {}).items():
        if len(e.split()) == 1 and len(f.split()) == 1 and latin.match(e):
            lex[l][f][e] += 3
f2e = {l: {f: c.most_common(1)[0][0] for f, c in d.items()} for l, d in lex.items()}
e2f = {l: defaultdict(list) for l in LANGS}
for l in LANGS:
    for f, e in f2e[l].items():
        if f != e:
            e2f[l][e].append(f)
print("word pairs per language:", {l: len(f2e[l]) for l in LANGS})
MODS = ("black white red blue green yellow pink purple grey gray brown beige gold silver orange navy wireless bluetooth usb led mini portable waterproof foldable adjustable "
        "electric digital smart large small medium big xl xxl leather cotton wooden plastic metal glass steel silicone ceramic bamboo velvet wool denim nylon "
        "kids baby men's women's unisex girls boys new original premium professional pro classic modern vintage luxury cheap soft slim thin thick long short "
        "rechargeable magnetic transparent organic natural vegan gaming outdoor indoor travel office kitchen garden winter summer sport sports cordless "
        "handmade reusable disposable ergonomic lightweight heavy duty non-slip anti-slip 4k hd fast charging stainless steel").split()
MODS = [m for m in MODS if m not in ("duty", "charging", "fast", "heavy", "stainless", "steel")] + ["heavy duty", "fast charging", "stainless steel"]
SUF = ["size 42", "size 38", "size m", "size l", "size xl", "500 ml", "1 liter", "250 g", "1 kg", "2 pack", "3 pack", "6 pack", "set of 2", "set of 4", "10 pcs", "100 pcs", "128 gb", "64 gb", "2 m", "1.5 m", "20000 mah", "12v", "220v", "50 cm", "100 cm", "spf 50"]
eng = sorted({p["tgt"] for p in P if 2 <= len(p["tgt"].split()) <= 4 and all(latin.match(w) or w.isdigit() for w in p["tgt"].split()) and p["tgt"] not in held})
by = defaultdict(list)
for p in P:
    if 1 <= len(p["tgt"].split()) <= 4 and len(p["src"]) <= 40 and p["tgt"] not in held and p["src"] not in held and p.get("kind") != "mixed":
        by[p["lang"]].append(p)
NA, NB, NC, ND = (int(os.environ.get(k, v)) for k, v in (("NA", 600), ("NB", 1200), ("NC", 700), ("ND", 250)))
out = []; st = Counter()
def add(src, tgt, lang, kind):
    src = re.sub(r"\s+", " ", src).strip()
    if src != tgt and src not in held and tgt not in held:
        out.append({"src": src, "tgt": tgt, "lang": lang, "kind": "mixed"}); st[kind] += 1; return True
    return False
for l in LANGS:
    ps = by[l]; rng.shuffle(ps); g = f2e[l]; seen = set()
    # A: foreign label with one English word
    k = 0
    for p in ps:
        if k >= NA: break
        src, tw = p["src"], p["tgt"].split()
        if l in NOSPACE:
            hits = [(f, e) for f, e in g.items() if len(f) >= 2 and f in src and len(src) > len(f) and e in tw]
        else:
            hits = [(w, g[w]) for w in src.split() if w in g and g[w] in tw] if len(src.split()) >= 2 else []
        if hits:
            f, e = rng.choice(hits); m = src.replace(f, " " + e + " ", 1) if l in NOSPACE else " ".join(e if w == f else w for w in src.split())
            if m not in seen and add(m, p["tgt"], l, "A foreign label, one English word"): seen.add(m); k += 1
    # B: English label with one foreign word
    k = 0; es = eng[:]; rng.shuffle(es)
    for t in es:
        if k >= NB: break
        ws = t.split(); c = [i for i, w in enumerate(ws) if w in e2f[l]]
        if c:
            i = rng.choice(c); m = " ".join(ws[:i] + [rng.choice(e2f[l][ws[i]])] + ws[i + 1:])
            if m not in seen and add(m, t, l, "B English label, one foreign word"): seen.add(m); k += 1
    # C: English modifier before / after a foreign label; D: size or pack after it
    k = 0
    for p in ps:
        if k >= NC: break
        m = rng.choice(MODS)
        if m in p["tgt"].split() or len(p["tgt"].split()) > 3: continue
        s = (m + " " + p["src"]) if rng.random() < 0.55 else (p["src"] + " " + m)
        if s not in seen and add(s, m + " " + p["tgt"], l, "C English modifier + foreign label"): seen.add(s); k += 1
    k = 0
    for p in reversed(ps):
        if k >= ND: break
        if len(p["tgt"].split()) > 3 or re.search(r"\d", p["tgt"]): continue
        x = rng.choice(SUF); s = p["src"] + " " + x
        if s not in seen and add(s, p["tgt"] + " " + x, l, "D foreign label + size or pack"): seen.add(s); k += 1
with open(os.path.join(W, "pairs_NEW_v12.jsonl"), "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"kinds": dict(st), "by_lang": dict(Counter(r["lang"] for r in out)), "word_pairs": {l: len(f2e[l]) for l in LANGS}}, open(os.path.join(W, "v12_pairs_stats.json"), "w"), indent=1)
json.dump({l: sorted(f2e[l]) for l in LANGS}, open(os.path.join(W, "v12_foreign_words.json"), "w"), ensure_ascii=False)
print(len(out), dict(st)); print(dict(Counter(r["lang"] for r in out)))
