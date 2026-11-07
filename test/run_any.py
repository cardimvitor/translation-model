"""Run the curated test set with the Python runtime; print failures and a per-type / per-language score.
Comparison is case-insensitive. Writes test/results_python.json (used by the JS parity check)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime"))
import normalizer as N

cases = json.load(open(os.path.join(ROOT, "test", sys.argv[1] if len(sys.argv) > 1 else "cases.json"), encoding="utf-8"))
res = N.normalize([(c["op"], c["in"]) for c in cases])
by_type, by_lang, fails = {}, {}, []
for c, r in zip(cases, res):
    ok = r["output"].casefold() == c["want"].casefold()
    for d, k in ((by_type, c["type"]), (by_lang, c["lang"])):
        d.setdefault(k, [0, 0]); d[k][0] += ok; d[k][1] += 1
    if not ok:
        fails.append((c["lang"], c["op"], c["in"], r["output"], c["want"]))
print(f"PASS {sum(v[0] for v in by_type.values())}/{len(cases)}")
print("by type:", {k: f"{a}/{b}" for k, (a, b) in sorted(by_type.items())})
print("by lang:", {k: f"{a}/{b}" for k, (a, b) in sorted(by_lang.items())})
for f in fails:
    print("  FAIL", f[0], f[1][:8], f[2], "->", repr(f[3]), "| want", repr(f[4]))
json.dump(res, open(os.path.join(ROOT, "test", ("results_python.json" if len(sys.argv) < 2 else "results_" + sys.argv[1])), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
