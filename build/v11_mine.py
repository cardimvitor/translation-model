"""v11 step 1: where does the current model fail on teacher-written short labels?
Runs the parent model (full precision, greedy) over every Hy-MT2-1.8B short-label pair and writes each pair with the
model's answer and its chrF against the known English.  -> build/work/v11_pool.jsonl
  python build/v11_mine.py post_student_V10"""
import argparse, json, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import ft_v8 as V8, ft_moe as MOE
from analyze_student import clean, chrf
tag = sys.argv[1]; torch.set_num_threads(4)
a = argparse.Namespace(tag=tag, rank=64); tok, m, ex, _ = V8.load(a, os.path.join(V8.out_dir(tag), "latest.pt")); torch.set_num_threads(4)
P = []; seen = set()
for f in ("pairs_NEW_v10.jsonl", "pairs_HQ_train.jsonl"):
    for l in open(os.path.join(W, f), encoding="utf-8"):
        p = json.loads(l)
        if p.get("kind") == "short_hy" and (p["src"], p["tgt"]) not in seen:
            seen.add((p["src"], p["tgt"])); P.append(p)
R = json.load(open(os.path.join(W, "moe_routes.json"), encoding="utf-8"))
outs = V8.translate(tok, m, ex, [p["src"] for p in P], [R.get(p["src"], 0) or MOE.route(p["src"]) for p in P])
with open(os.path.join(W, "v11_pool.jsonl"), "w", encoding="utf-8") as f:
    for p, o in zip(P, outs):
        f.write(json.dumps({"src": p["src"], "tgt": p["tgt"], "lang": p["lang"], "pred": o, "chrf": round(chrf(clean(o), clean(p["tgt"])), 1)}, ensure_ascii=False) + "\n")
import collections
c = collections.Counter(("exact" if clean(o) == clean(p["tgt"]) else "close" if chrf(clean(o), clean(p["tgt"])) >= 60 else "miss") for p, o in zip(P, outs))
print("pool", len(P), dict(c))
