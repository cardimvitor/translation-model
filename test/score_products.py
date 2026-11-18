"""Product-only comparison of Fast, NLLB-only and Hybrid on both test sets: normalized exact match + chrF."""
import json, os, re, sys
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime"))
import normalizer as F, accurate as AC

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

cases = [c for f in ("cases.json", "blind.json") for c in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8")) if c["type"] == "product"]
blind = {c["in"] for c in json.load(open(os.path.join(ROOT, "test", "blind.json"), encoding="utf-8"))}
rows = []
for c in cases:
    fast = F.process("translate", c["in"])
    langs = [l for l, _ in fast["language"]]
    nllb, conf = AC.mt().translate(c["in"], langs[0])
    covered = fast["method"] in ("dictionary", "gazetteer") and fast["confidence"] >= 0.84 * {"Latin": 0.9}.get(fast["script"], 0.85) + 0.0001
    hybrid = fast["output"] if covered else clean(nllb)
    rows.append((c, fast["output"], nllb, hybrid))
for name, idx in (("Fast", 1), ("NLLB only", 2), ("Hybrid", 3)):
    for split in ("tuned set", "blind set"):
        R = [r for r in rows if (r[0]["in"] in blind) == (split == "blind set")]
        em = sum(clean(r[idx]) == clean(r[0]["want"]) for r in R); cf = sum(chrf(clean(r[idx]), clean(r[0]["want"])) for r in R) / len(R)
        print(f"{name:10s} {split:9s}: exact {em}/{len(R)}  chrF {cf:5.1f}")
print()
for c, f, n, h in rows:
    if c["in"] in blind: print(f"{c['lang']} {c['in'][:26]:26s} | fast: {f[:30]:30s} | nllb: {n[:36]:36s} | want: {c['want']}")
