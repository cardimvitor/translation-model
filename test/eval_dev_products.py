"""Fast vs Accurate on held-out back-translated product titles (titles never seen in training)."""
import json, os, sys, random, re
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime"))
import normalizer as F, accurate as AC
src = open(os.path.join(ROOT, "test", "score_products.py")).read(); exec(src[src.index("def clean"):src.index("cases = [")])
pairs = [json.loads(l) for l in open(os.path.join(ROOT, "build", "work", "pairs_dev.jsonl"), encoding="utf-8")]
pairs = [p for p in pairs if p["kind"] == "bt"]; random.seed(2); random.shuffle(pairs); pairs = pairs[:int(sys.argv[1]) if len(sys.argv) > 1 else 400]
agg = defaultdict(lambda: [0, 0, 0, 0.0, 0.0]); used = Counter()
for p in pairs:
    f = F.process("translate", p["src"])["output"]; a = AC.process("translate", p["src"])
    used[a["method"]] += 1
    for k in ("all", p["lang"]):
        g = agg[k]; g[0] += 1; g[1] += clean(f) == clean(p["tgt"]); g[2] += clean(a["output"]) == clean(p["tgt"])
        g[3] += chrf(clean(f), clean(p["tgt"])); g[4] += chrf(clean(a["output"]), clean(p["tgt"]))
for k in sorted(agg, key=lambda k: (k != "all", k)):
    n, ef, ea, cf, ca = agg[k]; print(f"{k:4s} n={n:3d} Fast exact {ef / n:5.1%} chrF {cf / n:5.1f} | Accurate exact {ea / n:5.1%} chrF {ca / n:5.1f}")
print("Accurate methods:", dict(used.most_common()))
