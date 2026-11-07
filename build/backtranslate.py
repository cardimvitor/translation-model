"""Back-translation: English titles -> 8 random languages each, with NLLB (resumable, jsonl output)."""
import json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runtime"))
from nllb_gen import NLLB
import glossary_src as GS
LANGS = ["zh", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th"]
W = os.path.join(HERE, "work"); OUT = os.path.join(W, "bt.jsonl")

def english_titles():
    T = json.load(open(os.path.join(W, "titles_en.json")))
    random.seed(1)
    hm = random.sample(T["hm"], min(1100, len(T["hm"])))
    cat = {c: terms for c, terms in GS.CATEGORY.items()}
    items = [r[0] for r in GS.G if r[0] not in sum(cat.values(), []) and GS.RANK.get("item")]
    items = [x for x in items if x not in ("with", "for", "and", "pieces", "size", "set", "new", "original", "documents", "food", "spare parts", "fragile", "cosmetics", "clothes", "gift", "box")]
    combo = set()
    while len(combo) < 1100:
        parts = []
        if random.random() < .5: parts.append(random.choice(cat["gender"]))
        if random.random() < .3: parts.append(random.choice(cat["quality"]))
        if random.random() < .6: parts.append(random.choice(cat["color"]))
        if random.random() < .5: parts.append(random.choice(cat["material"]))
        parts.append(random.choice(items))
        combo.add(" ".join(parts))
    return T["shopify"] + hm + sorted(combo)

def main():
    titles = english_titles(); random.seed(2)
    jobs = [(t, l) for t in titles for l in random.sample(LANGS, 8)]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line); done.add((r["en"], r["lang"]))
    jobs = [j for j in jobs if j not in done]
    print("titles", len(titles), "jobs left", len(jobs), flush=True)
    n = NLLB(); t0 = time.time(); k = 0
    by = {}
    for t, l in jobs:
        by.setdefault(l, []).append(t)
    with open(OUT, "a", encoding="utf-8") as f:
        for l, ts in by.items():
            ts.sort(key=len)
            for i in range(0, len(ts), 32):
                batch = ts[i:i + 32]
                for en, (src, conf) in zip(batch, n.translate(batch, "en", l)):
                    f.write(json.dumps({"en": en, "lang": l, "src": src, "conf": round(conf, 3)}, ensure_ascii=False) + "\n")
                f.flush(); k += len(batch)
                print(f"{k}/{len(jobs)} done, {(time.time() - t0) / k * 1000:.0f} ms/item, lang {l}", flush=True)

if __name__ == "__main__":
    main()
