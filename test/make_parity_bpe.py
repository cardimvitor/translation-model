"""Python side of the word-piece parity check: writes test/parity_bpe_python.json for one bpe_<V>.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import bpe as BPE, student
bp = os.path.abspath(sys.argv[1]); B = json.load(open(bp, encoding="utf-8"))
G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
texts = [student.Student.prep(s) for g in G for s in g["src"].values()] + [student.Student.prep(e) for g in G for e in g["en"]]
texts += [student.Student.prep(c["in"]) for f in ("cases.json", "blind.json") for c in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))]
texts += ["a" * 130, "東" * 100 + " x", "", "  ", "x"]
items = [[side, t, BPE.encode(t, B[side]["merges"])] for t in texts for side in ("src", "tgt")]
json.dump({"bpe": bp, "items": items}, open(os.path.join(ROOT, "test", "parity_bpe_python.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(len(items), "items")
