"""Gate for post-training: the new model must not be worse than its parent.
Both run as full systems on the tuning labels, the test labels, the real-name test and the address cases.
  python build/compare_models.py student_V8 student_V9      -> build/work/compare_<new>.json, exit 1 if it regressed
Rule: no test may drop by more than 0.5 chrF, tuned address cases must stay 70/70, unseen cases may not drop."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
TOL = 0.5


def one(model):
    os.environ["NEURAL_DIR"] = model
    import accurate as AC
    from analyze_student import clean, chrf
    res = {}
    def labels(f):
        G = json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]; return [(g["en"], s) for g in G for s in g["src"].values()]
    sets = {"tuning_labels": labels("dev_labels.json"), "test_labels": labels("gold_products.json")}
    p = os.path.join(W, "abo_test.jsonl")
    if os.path.exists(p):
        sets["real_names"] = [([x["en"]], x["src"]) for x in map(json.loads, open(p, encoding="utf-8"))]
    for k, items in sets.items():
        outs = [AC.process("translate", s)["output"] for _, s in items]
        res[k] = {"exact": round(sum(clean(o) in [clean(r) for r in refs] for o, (refs, _) in zip(outs, items)) / len(items), 4),
                  "chrF": round(sum(max(chrf(clean(o), clean(r)) for r in refs) for o, (refs, _) in zip(outs, items)) / len(items), 2)}
    for f in ("cases.json", "blind.json"):
        C = json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))
        res[f[:-5]] = sum(AC.process(c["op"], c["in"])["output"].casefold() == c["want"].casefold() for c in C)
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2])
    else:
        old, new = sys.argv[1], sys.argv[2]; R = {}
        for m in (old, new):
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--one", m], capture_output=True, text=True)
            R[m] = json.loads(r.stdout.strip().split("\n")[-1])
        ok = True; print(f"{'test':16s} {old:>14s} {new:>14s}")
        for k in R[old]:
            a, b = R[old][k], R[new][k]
            if isinstance(a, dict):
                bad = b["chrF"] < a["chrF"] - TOL; print(f"{k:16s} {a['exact'] * 100:5.1f}% / {a['chrF']:5.1f} {b['exact'] * 100:5.1f}% / {b['chrF']:5.1f}  {'WORSE' if bad else 'ok'}")
            else:
                bad = b < a if k == "blind" else b < 70; print(f"{k:16s} {a:>14d} {b:>14d}  {'WORSE' if bad else 'ok'}")
            ok &= not bad
        R["verdict"] = "PASS" if ok else "FAIL"; json.dump(R, open(os.path.join(W, f"compare_{new}.json"), "w"), indent=1)
        print("VERDICT:", R["verdict"], "(new model kept)" if ok else "(keep using the parent model; the new one is left in models/ for inspection)")
        sys.exit(0 if ok else 1)
