import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
os.environ["NEURAL_DIR"] = sys.argv[1]
for kv in sys.argv[3:]:
    k, v = kv.split("="); os.environ[k] = v
import accurate as AC
from analyze_student import clean, chrf
out = {}
for f in ("gold_products.json", "mixed_labels.json", "mixed_tr.json", "dev_labels.json"):
    rows = []
    for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]:
        refs = g["en"] if isinstance(g["en"], list) else [g["en"]]
        for lang, s in (g["src"].items() if isinstance(g["src"], dict) else [(g["lang"], g["src"])]):
            r = AC.process("translate", s); o = clean(r["output"])
            rows.append({"lang": lang, "src": s, "out": r["output"], "m": r["method"], "conf": r.get("confidence", 0), "refs": refs, "exact": o in [clean(x) for x in refs], "chrf": max(chrf(o, clean(x)) for x in refs)})
    out[f] = rows; print(sys.argv[1], f, len(rows), f"exact {sum(r['exact'] for r in rows) / len(rows) * 100:.1f}% chrF {sum(r['chrf'] for r in rows) / len(rows):.1f}", flush=True)
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)
