"""Python side of the student / Accurate parity check: writes test/parity_student_python.json and parity_accurate_python.json."""
import json, os, sys, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime"))
import accurate as AC
items = [(c["op"], c["in"]) for f in ("cases.json", "blind.json") for c in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))]
gold = [s for g in json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"] for s in g["src"].values()]
random.seed(3); random.shuffle(gold); items += [("translate", s) for s in gold[:200]]
items += [("translate", g["src"]) for f in ("mixed_labels.json", "mixed_tr.json") for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]]   # English mixed with another language
S = AC.mt()
json.dump([[t, *S.translate(t)] for _, t in items], open(os.path.join(ROOT, "test", "parity_student_python.json"), "w", encoding="utf-8"), ensure_ascii=False)
json.dump([[op, t, AC.process(op, t)] for op, t in items], open(os.path.join(ROOT, "test", "parity_accurate_python.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(len(items), "items")
