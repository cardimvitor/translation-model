"""JEPA-Lookup: translate by finding the nearest known English name in JEPA's embedding space.

  python jepa_lookup.py models/jepa_main/jepa.pt [--data MAX] [--bank 150000]
Bank: English names the system already knows (training titles, real ABO names, word list, place names), embedded
once by the target encoder. Query: the predictor's global output for the foreign label. Output: the bank entry with
the highest cosine; the cosine is the confidence.
The confidence threshold is chosen on held-out training-split titles (pairs_<data>_dev), never on the test sets.
Reports, on the gold set and the ABO real-name test: Lookup alone, Fast, and Fast with Lookup above the threshold
(and the same on top of an Accurate model when NEURAL_DIR is set). -> build/work/jepa_lookup_<tag>.json"""
import argparse, json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime"), os.path.join(ROOT, "test")]
from jepa import JEPA
from analyze_student import clean, chrf
import normalizer as F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--data", default="MAX")
    ap.add_argument("--bank", type=int, default=150000); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    j, _ = JEPA.load(a.ckpt); j.E.eval(); j.P.eval(); j.T.eval()
    train = [json.loads(l) for l in open(os.path.join(W, f"pairs_{a.data}_train.jsonl"), encoding="utf-8")]
    from collections import Counter
    cnt = Counter(p["tgt"] for p in train)
    bank = [t for t, _ in cnt.most_common(a.bank)]
    B = j.embed_target(bank); print("bank", len(bank), flush=True)

    def look(srcs):
        q = j.embed_query(srcs); s, i = (q @ B.T).max(1)
        return [(bank[k], float(c)) for k, c in zip(i.tolist(), s.tolist())]

    dev = [p for p in (json.loads(l) for l in open(os.path.join(W, f"pairs_{a.data}_dev.jsonl"), encoding="utf-8")) if p["kind"] in ("bt", "real")][:1500]
    L = look([p["src"] for p in dev]); fast = [F.process("translate", p["src"])["output"] for p in dev]
    best = (0, 1.01)
    for tau in [x / 100 for x in range(50, 100, 2)] + [1.01]:
        sc = sum(chrf(clean(l if c >= tau else f), clean(p["tgt"])) for (l, c), f, p in zip(L, fast, dev)) / len(dev)
        best = max(best, (sc, tau))
    tau = best[1]; print("threshold from held-out dev:", tau, "dev chrF", round(best[0], 2), flush=True)

    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    sets = {"gold": [(g["en"], s) for g in G for s in g["src"].values()],
            "abo": [([p["en"]], p["src"]) for p in map(json.loads, open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8"))]}
    acc = None
    if os.environ.get("NEURAL_DIR"):
        import accurate
        acc = lambda s: accurate.process("translate", s)["output"]
    res = {"threshold": tau}
    for name, items in sets.items():
        L = look([s for _, s in items]); fast = [F.process("translate", s)["output"] for _, s in items]
        def score(outs):
            ex = sum(clean(o) in [clean(r) for r in refs] for o, (refs, _) in zip(outs, items)) / len(items)
            ch = sum(max(chrf(clean(o), clean(r)) for r in refs) for o, (refs, _) in zip(outs, items)) / len(items)
            return {"exact": round(ex, 4), "chrF": round(ch, 2)}
        r = {"lookup_alone": score([l for l, _ in L]), "fast": score(fast),
             "fast+lookup": score([l if c >= tau else f for (l, c), f in zip(L, fast)]),
             "lookup_used": round(sum(c >= tau for _, c in L) / len(L), 3)}
        if acc:
            A = [acc(s) for _, s in items]
            r["accurate"] = score(A); r["accurate+lookup"] = score([l if c >= tau else x for (l, c), x in zip(L, A)])
        res[name] = r; print(name, r, flush=True)
    tag = os.path.basename(os.path.dirname(a.ckpt))
    json.dump(res, open(os.path.join(W, f"jepa_lookup_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
