"""Tune Accurate's decision rule v2 (confidence threshold tau, whether digits count as an address) on held-out
names from the MAX dev split (back-translated titles + real ABO names, never trained on, not the test sets).
  NEURAL_DIR=student_MAX python tune_routing.py
Prints the dev score of every setting and the chosen one; then compares rule v1 and tuned v2 on the gold set and
the ABO real-name test. -> build/work/routing_<model>.json"""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import normalizer as F, accurate as AC, student
from analyze_student import clean, chrf


def decide(item, tau, digits):
    fast, addr_w, addr_d, o, c, ok_known = item
    if fast is None:
        return None
    if addr_w or (digits and addr_d):
        return fast
    return o if (o and c >= tau and ok_known) else fast


def prepare(srcs):
    S = AC.mt(); out = []
    for s in srcs:
        f = F.process("translate", s)
        if f["method"] in ("gazetteer", "already-english") or not f["output"]:
            out.append((f["output"], True, True, "", 0.0, False)); continue
        langs = [l for l, _ in f["language"]]; o, c = S.translate(s)
        ok = bool(o) and student.acceptable(o, 1.0) and AC.keeps_known(o, f["output"])
        out.append((f["output"], f.get("coverage", 0) >= 1 or AC.is_address_v2(s, langs, f), AC.address_like(s, langs), o, c, ok))
    return out


def score(outs, refs):
    ex = sum(clean(o) in [clean(r) for r in R] for o, R in zip(outs, refs)) / len(refs)
    ch = sum(max(chrf(clean(o), clean(r)) for r in R) for o, R in zip(outs, refs)) / len(refs)
    return {"exact": round(ex, 4), "chrF": round(ch, 2)}


def main():
    name = os.environ.get("NEURAL_DIR", "student")
    dev = [json.loads(l) for l in open(os.path.join(HERE, "work", "pairs_MAX_dev.jsonl"), encoding="utf-8")]
    rng = random.Random(9); real = [p for p in dev if p["kind"] == "real"]; bt = [p for p in dev if p["kind"] == "bt"]
    rng.shuffle(real); rng.shuffle(bt); dev = real[:600] + bt[:600]
    D = prepare([p["src"] for p in dev]); refs = [[p["tgt"]] for p in dev]
    grid = {}
    for digits in (0, 1):
        for tau in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            grid[(tau, digits)] = score([decide(x, tau, digits) for x in D], refs)
    for k, v in sorted(grid.items()):
        print("dev tau", k[0], "digits", k[1], v)
    best = max(grid, key=lambda k: grid[k]["chrF"]); print("chosen", best, grid[best], flush=True)
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    sets = {"gold": ([s for g in G for s in g["src"].values()], [g["en"] for g in G for _ in g["src"]]),
            "abo": tuple(zip(*[(p["src"], [p["en"]]) for p in map(json.loads, open(os.path.join(HERE, "work", "abo_test.jsonl"), encoding="utf-8"))]))}
    res = {"model": name, "chosen": {"tau": best[0], "digits_are_address": bool(best[1])}, "dev": grid[best]}
    for k, (srcs, R) in sets.items():
        AC.POLICY["name"] = "v1"; v1 = score([AC.process("translate", s)["output"] for s in srcs], R)
        X = prepare(srcs); v2 = score([decide(x, *best) for x in X], R)
        res[k] = {"rule_v1": v1, "rule_v2": v2}; print(k, res[k], flush=True)
    json.dump(res, open(os.path.join(HERE, "work", f"routing_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
