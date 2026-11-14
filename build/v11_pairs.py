"""v11 step 3: training files from the 7B's judgments.
  pairs_NEW_v11.jsonl   source -> the 7B's English answer, for every missed pair and every real es / ja query
  drop_pairs.jsonl      old pairs whose English target the 7B contradicts (chrF < 60 against its answer): removed from replay
A 7B answer is used only if it is English, not a copy of the source and of sane length."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import encheck
from analyze_student import clean, chrf
from collections import Counter
rd = lambda n: [json.loads(l) for l in open(os.path.join(W, n), encoding="utf-8")] if os.path.exists(os.path.join(W, n)) else []


def ok(src, en):
    en = re.sub(r"\s+", " ", re.sub(r"[\"“”‘’`]", "", en.replace("’s", "'s"))).strip().strip(".").lower()
    if not en or en == src or not (0.2 <= len(en) / max(1, len(src)) <= 5) or len(en.split()) > 12 or re.search(r"[Ѐ-ӿ؀-ۿऀ-ॿ฀-๿぀-ヿ一-鿿가-힣]", en):
        return None
    return en if encheck.is_english(en, "relaxed")["english"] else None


new, drop, st = [], [], Counter()
for j in rd("v11_judged.jsonl"):
    en = ok(j["src"], j["hy7"])
    if not en:
        st["7B answer unusable"] += 1; continue
    agree_t = chrf(clean(en), clean(j["tgt_old"])) >= 60; agree_p = chrf(clean(en), clean(j["pred"])) >= 60
    st["7B agrees with the old target (hard example)" if agree_t else "7B agrees with the model (old target was wrong)" if agree_p else "7B gives a third answer"] += 1
    if not agree_t:
        drop.append({"src": j["src"], "tgt": j["tgt_old"]})
    if not agree_p:                                   # the model already gets these right; nothing to teach
        new.append({"src": j["src"], "tgt": en, "lang": j["lang"], "kind": "hy7"})
for j in rd("v11_real.jsonl"):
    en = ok(j["src"], j["hy7"])
    if en:
        new.append({"src": j["src"], "tgt": en, "lang": j["lang"], "kind": "hy7_real"}); st["real query pairs"] += 1
# Mixed labels (English + another language in one label), made by code-switching: in a known pair, one source word that the
# product word list knows is replaced by its English word; the English answer stays the same, so word order stays natural.
#   "deri ceket" -> "leather jacket"   gives   "leather ceket" -> "leather jacket"   and   "deri jacket" -> "leather jacket"
import random
from student_data import student_input
rng = random.Random(11); dropset = {(d["src"], d["tgt"]) for d in drop}; held = set()
for f in ("gold_products.json", "dev_labels.json"):
    for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]:
        held |= {student_input(x) for x in g["src"].values()} | {student_input(e) for e in (g["en"] if isinstance(g["en"], list) else [g["en"]])}
GL = json.load(open(os.path.join(ROOT, "data", "glossary.json"), encoding="utf-8")); M = int(os.environ.get("MIXED_N", 1200)); NOSPACE = {"zh", "ja", "th"}
cand = {}
for n in ("pairs_NEW_v10.jsonl", "pairs_HQ_train.jsonl", "pairs_Q_train.jsonl", "pairs_NEW_v9.jsonl"):
    for p in rd(n):
        if (p["src"], p["tgt"]) in dropset or p["tgt"] in held or p["src"] in held or not (2 <= len(p["tgt"].split()) <= 5) or len(p["src"]) > 40:
            continue
        g = GL.get(p["lang"], {}); tw = p["tgt"].split(); src = p["src"]; hits = []
        if p["lang"] in NOSPACE:
            hits = [(t, e) for t, e in g.items() if len(t) >= 2 and t in src and len(src) > len(t) and all(w in tw for w in e.split())]
        else:
            sw = src.split()
            if len(sw) >= 2:
                hits = [(w, g[w]) for w in sw if w in g and all(x in tw for x in g[w].split())]
        if hits:
            t, e = rng.choice(hits)
            mixed = re.sub(r"\s+", " ", src.replace(t, " " + e + " ", 1)).strip() if p["lang"] in NOSPACE else " ".join(e if w == t else w for w in src.split())
            if mixed != src and mixed != p["tgt"] and mixed not in held:
                cand.setdefault(p["lang"], {})[mixed] = p["tgt"]
for lang, d in sorted(cand.items()):
    ks = sorted(d); rng.shuffle(ks)
    for k in ks[:M]:
        new.append({"src": k, "tgt": d[k], "lang": lang, "kind": "mixed"}); st["mixed pairs"] += 1
for name, rows in (("pairs_NEW_v11.jsonl", new), ("drop_pairs.jsonl", drop)):
    with open(os.path.join(W, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(dict(st), open(os.path.join(W, "v11_judge_stats.json"), "w"), indent=1)
print("new pairs", len(new), "| dropped old pairs", len(drop), "|", dict(st), "|", dict(Counter(p["lang"] for p in new)))
