"""Regression gate: fails (exit 1) if any score drops below test/gate_baseline.json.
Checks Fast on the tuned and blind cases and on the gold product set, and Accurate (each shipped variant) on gold."""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, os.path.join(ROOT, "build"))
import normalizer as F
import re
from collections import Counter


def clean(s):
    s = s.lower().strip(); s = re.sub(r"\s+([.,!?])", r"\1", s); s = re.sub(r"[.!?]+$", "", s).strip()
    s = re.sub(r"^(the|a|an|-)\s+", "", s); return re.sub(r"\s+", " ", s)


def chrf(hyp, ref, n=6, beta=2):
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", ""); P = R = 0; k = 0
    for i in range(1, n + 1):
        h = Counter(hyp[j:j + i] for j in range(len(hyp) - i + 1)); r = Counter(ref[j:j + i] for j in range(len(ref) - i + 1))
        if not h or not r: continue
        m = sum((h & r).values()); P += m / sum(h.values()); R += m / sum(r.values()); k += 1
    if not k: return 0.0
    P, R = P / k, R / k
    return 0.0 if P + R == 0 else (1 + beta ** 2) * P * R / (beta ** 2 * P + R) * 100


def cases(f):
    C = json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))
    return sum(F.process(c["op"], c["in"])["output"].casefold() == c["want"].casefold() for c in C)


def gold(model, variant=None):
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    if model == "fast":
        fn = lambda s: F.process("translate", s)["output"]
    else:
        import accurate
        fn = lambda s: accurate.process("translate", s, variant=variant)["output"]
    ex = ch = n = 0
    for g in G:
        refs = [clean(r) for r in g["en"]]
        for s in g["src"].values():
            o = clean(fn(s)); ex += o in refs; ch += max(chrf(o, r) for r in refs); n += 1
    return round(ex / n, 4), round(ch / n, 2)


def main():
    base = json.load(open(os.path.join(ROOT, "test", "gate_baseline.json")))
    got = {"fast_tuned": cases("cases.json"), "fast_blind": cases("blind.json")}
    got["fast_gold_exact"], got["fast_gold_chrF"] = gold("fast")
    for v in base.get("variants", []):
        if os.path.exists(os.path.join(ROOT, "models", v, "vocab.json")):
            got[f"{v}_gold_exact"], got[f"{v}_gold_chrF"] = gold("accurate", v)
    bad = [k for k, v in base["scores"].items() if k in got and got[k] < v - 1e-9]
    for k in sorted(got):
        print(f"{k:28s} {got[k]:>8}   baseline {base['scores'].get(k, '-')}")
    if "--update" in sys.argv:
        base["scores"].update(got); json.dump(base, open(os.path.join(ROOT, "test", "gate_baseline.json"), "w"), indent=1); print("baseline updated")
    elif bad:
        print("REGRESSION:", bad); sys.exit(1)
    else:
        print("gate passed")


if __name__ == "__main__":
    main()
