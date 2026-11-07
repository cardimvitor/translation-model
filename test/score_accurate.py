"""Product-only comparison of Fast, student alone and Accurate on both test sets: normalized exact match + chrF."""
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
import student
S = AC.mt(); rows = []
for c in cases:
    fast = F.process("translate", c["in"])["output"]
    st, conf = S.translate(c["in"])
    acc = AC.process("translate", c["in"])
    rows.append((c, fast, st, acc["output"], acc["method"]))
for name, idx in (("Fast", 1), ("Student only", 2), ("Accurate", 3)):
    for split in ("tuned set", "blind set"):
        R = [r for r in rows if (r[0]["in"] in blind) == (split == "blind set")]
        em = sum(clean(r[idx]) == clean(r[0]["want"]) for r in R); cf = sum(chrf(clean(r[idx]), clean(r[0]["want"])) for r in R) / len(R)
        print(f"{name:12s} {split:9s}: exact {em}/{len(R)}  chrF {cf:5.1f}")
print()
for c, f, s, a, m in rows:
    if c["in"] in blind: print(f"{c['lang']} {c['in'][:24]:24s} | fast: {f[:28]:28s} | student: {s[:30]:30s} | acc[{m}]: {a[:30]:30s} | want: {c['want']}")
