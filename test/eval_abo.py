"""Real-world test: held-out Amazon Berkeley Objects names (marketplace-written, CC BY 4.0), up to 100 per language
(build/work/abo_test.jsonl, never trained on). Marketplace names are not always word-for-word translations of the
English name, so scores are lower than on the gold set; use it to compare models, not as an absolute.
  NEURAL_DIR=student_B2 python test/eval_abo.py      (NEURAL_DIR=none: Fast only)"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, os.path.join(ROOT, "build")); sys.path.insert(0, os.path.join(ROOT, "test"))
import normalizer as F
from eval_gold import run

items = [([p["en"]], p["lang"], p["src"]) for p in map(json.loads, open(os.path.join(ROOT, "build", "work", "abo_test.jsonl"), encoding="utf-8"))]
res = {"fast": run("Fast", lambda s: F.process("translate", s)["output"], items)}
if os.environ.get("NEURAL_DIR", "student") != "none":
    import student, accurate
    S = accurate.mt()
    res["student"] = run("Student only", lambda s: S.translate(s)[0], items)
    res["accurate"] = run("Accurate", lambda s: accurate.process("translate", s)["output"], items)
name = os.environ.get("NEURAL_DIR", "none")
json.dump({k: {x: v[x] for x in ("exact", "chrF", "by_lang")} for k, v in res.items()}, open(os.path.join(ROOT, "test", f"results_abo_{name}.json"), "w"), indent=1)
