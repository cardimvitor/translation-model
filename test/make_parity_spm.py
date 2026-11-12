"""Reference output for test/parity_spm.js: SentencePiece pieces from runtime/spm.py for mixed-language inputs."""
import json, os, random, sys, unicodedata, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import spm, normalizer as F
D = os.path.join(ROOT, "models", os.environ.get("NEURAL_DIR", "student_V6"))
U = spm.Unigram(json.load(open(os.path.join(D, "spm_src.json"), encoding="utf-8")))
items = [c["in"] for f in ("cases.json", "blind.json") for c in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))]
items += [s for g in json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"] for s in g["src"].values()]
p = os.path.join(ROOT, "build", "work", "abo_test.jsonl")
if os.path.exists(p):
    items += [json.loads(l)["src"] for l in open(p, encoding="utf-8")]
prep = lambda s: re.sub(r"\s+", " ", F.lower(unicodedata.normalize("NFKC", s))).strip()
json.dump([[t, U.encode(prep(t))] for t in items], open(os.path.join(ROOT, "test", "parity_spm_python.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(len(items), "items")
