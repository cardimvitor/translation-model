"""Scores on test items whose source text also appears in the training data ("seen") vs the rest ("unseen"),
for the full Accurate system of each model given. Overlap is by coincidence (common labels such as "phone case" are
also in catalogues and the mined word list); the test files themselves were never used for training.
  python build/seen_split.py student_V6 student_MAX_edited ...  -> build/work/seen_split.json"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]


def one(model):
    os.environ["NEURAL_DIR"] = model
    import accurate as AC, normalizer as F
    from analyze_student import clean, chrf
    from student_data import student_input
    seen = set()
    for f in ("pairs_MAX_train.jsonl", "pairs_Q_train.jsonl"):
        for l in open(os.path.join(W, f), encoding="utf-8"):
            seen.add(json.loads(l)["src"])
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    sets = {"gold": [(g["en"], s) for g in G for s in g["src"].values()],
            "abo": [([p["en"]], p["src"]) for p in map(json.loads, open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8"))]}
    S = AC.mt() if model != "none" else None; res = {}
    for k, items in sets.items():
        rows = []
        for refs, s in items:
            outs = {"fast": F.process("translate", s)["output"]}
            if S:
                outs["accurate"] = AC.process("translate", s)["output"]; outs["alone"] = S.translate(s)[0]
            rows.append((student_input(s) in seen, {n: (clean(o) in [clean(r) for r in refs], max(chrf(clean(o), clean(r)) for r in refs)) for n, o in outs.items()}))
        for sub, pick in (("all", lambda x: True), ("unseen", lambda x: not x), ("seen", lambda x: x)):
            R = [r for sn, r in rows if pick(sn)]
            res[f"{k}_{sub}"] = {"n": len(R), **{n: {"exact": round(sum(r[n][0] for r in R) / len(R), 4), "chrF": round(sum(r[n][1] for r in R) / len(R), 2)} for n in R[0]}}
    print(json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        one(sys.argv[2])
    else:
        out = {}
        for m in sys.argv[1:]:
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--one", m], capture_output=True, text=True)
            out[m] = json.loads(r.stdout.strip().split("\n")[-1]); print(m, {k: v for k, v in out[m].items() if "gold" in k or k == "abo_unseen"}, flush=True)
        json.dump(out, open(os.path.join(W, "seen_split.json"), "w"), indent=1)
