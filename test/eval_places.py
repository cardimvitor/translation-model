"""Places test: python test/eval_places.py [model]   (model: fast | accurate, default accurate with NEURAL_DIR)
Exact = output equals one accepted answer (case and accents ignored). Writes build/work/places_<tag>.json when TAG is set."""
import json, os, sys, unicodedata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [os.path.join(ROOT, "runtime"), os.path.join(ROOT, "build")]
which = sys.argv[1] if len(sys.argv) > 1 else "accurate"
import normalizer as F
if which == "accurate":
    import accurate as A
    run = A.process
else:
    run = F.process
def fold(s):
    s = unicodedata.normalize("NFKD", s.replace("ı", "i").replace("ł", "l").replace("Ł", "L").replace("đ", "d").replace("Đ", "D").replace("ß", "ss")); return " ".join("".join(c for c in s if not unicodedata.combining(c)).casefold().replace(",", " ").replace(".", " ").replace("-", " ").replace("’", "'").split())
T = json.load(open(os.path.join(ROOT, "test", os.environ.get("SET", "places_world.json")), encoding="utf-8"))["items"]; by = {}; rows = []
for t in T:
    r = run(t["op"], t["in"]); ok = fold(r["output"]) in [fold(w) for w in t["want"]]; by.setdefault(t["kind"], []).append(ok)
    rows.append({"kind": t["kind"], "in": t["in"], "out": r["output"], "want": t["want"][0], "ok": ok, "method": r["method"]})
for k, v in by.items():
    print(f"{k:9s} {sum(v):3d}/{len(v)}")
n = sum(map(sum, by.values())); print(f"TOTAL     {n}/{len(T)} ({n / len(T) * 100:.1f}%)")
if os.environ.get("TAG"):
    json.dump({"by": {k: [sum(v), len(v)] for k, v in by.items()}, "total": [n, len(T)], "rows": rows}, open(os.path.join(ROOT, "build", "work", f"places_{os.environ['TAG']}.json"), "w"), ensure_ascii=False, indent=1)
if os.environ.get("SHOW"):
    for r in rows:
        if not r["ok"]: print("  ", r["kind"], "|", r["in"], "->", r["out"], "| want", r["want"], "|", r["method"])
