"""Gold product test (hand-written, 20 products x 16 languages): Fast, student alone and Accurate.
  NEURAL_DIR=student_B python test/eval_gold.py        (default models/student)
Exact = matches any accepted answer; chrF = best over accepted answers."""
import json, os, re, sys, time
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, os.path.join(ROOT, "build"))
import normalizer as F
from analyze_student import clean, chrf


def run(name, fn, items):
    agg = defaultdict(lambda: [0, 0, 0.0]); t0 = time.time(); rows = []
    for refs, lang, src in items:
        out = fn(src); c = clean(out); R = [clean(r) for r in refs]
        e = c in R; ch = max(chrf(c, r) for r in R)
        for k in ("all", lang):
            agg[k][0] += 1; agg[k][1] += e; agg[k][2] += ch
        rows.append((lang, src, out, refs[0], e))
    ms = (time.time() - t0) / len(items) * 1000
    a = agg["all"]; print(f"{name:14s} exact {a[1]}/{a[0]} ({a[1] / a[0]:.1%})  chrF {a[2] / a[0]:5.1f}  {ms:.1f} ms/item")
    return {"exact": a[1] / a[0], "chrF": round(a[2] / a[0], 2), "by_lang": {k: {"exact": v[1] / v[0], "chrF": round(v[2] / v[0], 1)} for k, v in agg.items() if k != "all"}, "rows": rows}


def main():
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    items = [(g["en"], l, s) for g in G for l, s in g["src"].items()]
    res = {"fast": run("Fast", lambda s: F.process("translate", s)["output"], items)}
    if os.environ.get("NEURAL_DIR", "student") != "none":
        import student, accurate
        S = accurate.mt()
        res["student"] = run("Student only", lambda s: S.translate(s)[0], items)
        res["accurate"] = run("Accurate", lambda s: accurate.process("translate", s)["output"], items)
    name = os.environ.get("NEURAL_DIR", "student")
    json.dump(res, open(os.path.join(ROOT, "test", f"results_gold_{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    if "-v" in sys.argv:
        for k in res:
            print("==", k); [print("  ", r) for r in res[k]["rows"] if not r[4]]


if __name__ == "__main__":
    main()
