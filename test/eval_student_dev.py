"""Student alone on held-out dev pairs (unseen titles / unseen cities): exact match + chrF, by kind and language."""
import json, os, sys, time
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, os.path.join(ROOT, "test"))
import student
import re
from collections import Counter
def clean(s):
    s = s.lower().strip(); s = re.sub(r"\s+([.,!?])", r"\1", s); s = re.sub(r"[.!?]+$", "", s).strip()
    s = re.sub(r"^(the|a|an|-)\s+", "", s); s = s.replace(" '", "'").replace("' s", "'s")
    return re.sub(r"\s+", " ", s)

def chrf(hyp, ref, n=6, beta=2):
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", ""); P = R = 0; k = 0
    for i in range(1, n + 1):
        h = Counter(hyp[j:j + i] for j in range(len(hyp) - i + 1)); r = Counter(ref[j:j + i] for j in range(len(ref) - i + 1))
        if not h or not r: continue
        m = sum((h & r).values()); P += m / sum(h.values()); R += m / sum(r.values()); k += 1
    if not k: return 0.0
    P, R = P / k, R / k
    return 0.0 if P + R == 0 else (1 + beta**2) * P * R / (beta**2 * P + R) * 100

S = student.Student()
pairs = [json.loads(l) for l in open(os.path.join(ROOT, "build", "work", "pairs_dev.jsonl"), encoding="utf-8")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
import random; random.seed(1); random.shuffle(pairs); pairs = pairs[:N]
agg = defaultdict(lambda: [0, 0, 0.0, 0]); t0 = time.time(); ex = []
for p in pairs:
    o, c = S.translate(p["src"]); ok = clean(o) == clean(p["tgt"]); acc = student.acceptable(o, c)
    for k in (p["kind"], p["kind"] + ":" + p["lang"]):
        a = agg[k]; a[0] += 1; a[1] += ok; a[2] += chrf(clean(o), clean(p["tgt"])); a[3] += acc
    if len(ex) < 25: ex.append((p["src"], o, p["tgt"], c))
ms = (time.time() - t0) / len(pairs) * 1000
for k in sorted(agg):
    n, e, ch, ac = agg[k]; print(f"{k:22s} n={n:4d} exact={e / n:6.1%} chrF={ch / n:5.1f} accepted={ac / n:6.1%}")
print(f"{ms:.1f} ms/item (python, int8, 1 thread)")
for s, o, t, c in ex: print(f"  {s!r} -> {o!r} (ref {t!r}, conf {c})")
