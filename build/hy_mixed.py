"""Hy-MT2 (1.8B and 7B) on the mixed-language tests, model alone. -> build/work/teachers/mixed_<hy18|hy7>.json
  python build/hy_mixed.py hy7      (CPU only, nothing else running)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
from hy_gen import Hy
from analyze_student import clean, chrf
which = sys.argv[1]; h = Hy(os.path.join(ROOT, "models", "hy-mt2", {"hy7": "Hy-MT2-7B-Q4_K_M.gguf", "hy18": "Hy-MT2-1.8B-Q4_K_M.gguf"}[which])); out = {}
for f in ("mixed_tr.json", "mixed_labels.json"):
    rows = []
    for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]:
        o = h.one(g["src"], "en", max_new=48)[0]; c = clean(o); rf = [clean(x) for x in g["en"]]
        rows.append({"lang": g["lang"], "src": g["src"], "out": o, "refs": g["en"], "exact": c in rf, "chrf": max(chrf(c, x) for x in rf)})
    out[f] = rows; print(which, f, len(rows), f"exact {sum(r['exact'] for r in rows) / len(rows) * 100:.1f}% chrF {sum(r['chrf'] for r in rows) / len(rows):.1f}", flush=True)
json.dump(out, open(os.path.join(W, "teachers", f"mixed_{which}.json"), "w"), ensure_ascii=False, indent=1)
