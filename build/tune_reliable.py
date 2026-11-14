"""Calibrates the "reliable" flag of a model on the TUNING labels only, then reports it on the test labels.
  python build/tune_reliable.py student_V11 [target]
Picks the lowest confidence threshold rel_tau at which the neural answers marked reliable are right (exact, or chrF >= 90)
at least `target` (default 0.97) of the time on test/dev_labels.json. Writes policy.rel_tau into the model's vocab.json and
build/work/reliable_<model>.json (share marked reliable and how often those are right, on tuning and on test labels)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
model = sys.argv[1]; target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.97; os.environ["NEURAL_DIR"] = model
vp = os.path.join(ROOT, "models", model, "vocab.json"); V = json.load(open(vp, encoding="utf-8")); V["policy"]["rel_tau"] = 0.0
json.dump(V, open(vp, "w", encoding="utf-8"), ensure_ascii=False)
import accurate as AC
from analyze_student import clean, chrf


def run(f):
    rows = []
    for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]:
        refs = g["en"] if isinstance(g["en"], list) else [g["en"]]
        for lang, s in (g["src"].items() if isinstance(g.get("src"), dict) else [(g["lang"], g["src"])]):
            r = AC.process("translate", s); o = clean(r["output"]); c = max(chrf(o, clean(x)) for x in refs)
            rows.append({"lang": lang, "src": s, "out": r["output"], "method": r["method"], "conf": r.get("confidence", 0), "base": r["reliable"],
                         "exact": o in [clean(x) for x in refs], "chrf": c, "good": o in [clean(x) for x in refs] or c >= 90})
    return rows


def stats(rows, tau):
    rel = [r for r in rows if (r["conf"] >= tau if r["method"] == "neural" else r["base"])]
    return {"n": len(rows), "reliable_share": round(len(rel) / len(rows), 4), "reliable_exact": round(sum(r["exact"] for r in rel) / max(1, len(rel)), 4),
            "reliable_good": round(sum(r["good"] for r in rel) / max(1, len(rel)), 4), "reliable_chrF": round(sum(r["chrf"] for r in rel) / max(1, len(rel)), 2),
            "rest_chrF": round(sum(r["chrf"] for r in rows if r not in rel) / max(1, len(rows) - len(rel)), 2), "all_chrF": round(sum(r["chrf"] for r in rows) / len(rows), 2)}


dev = run("dev_labels.json"); best = 0.99
for t in [x / 100 for x in range(99, 19, -1)]:
    n = [r for r in dev if r["method"] == "neural" and r["conf"] >= t]
    if n and sum(r["good"] for r in n) / len(n) >= target:
        best = t
    elif n and len(n) > 30:
        break                                   # stop at the first threshold that falls under the target
V["policy"]["rel_tau"] = best; json.dump(V, open(vp, "w", encoding="utf-8"), ensure_ascii=False)
test = run("gold_products.json"); res = {"model": model, "target": target, "rel_tau": best, "tuning": stats(dev, best), "test": stats(test, best),
       "test_rows": [{k: r[k] for k in ("lang", "src", "out", "exact", "chrf")} | {"reliable": (r["conf"] >= best if r["method"] == "neural" else r["base"])} for r in test]}
json.dump(res, open(os.path.join(W, f"reliable_{model}.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ("rel_tau", "tuning", "test")}, indent=1))
