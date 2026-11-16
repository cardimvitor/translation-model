"""Weight blend of two fast models with the same shape: (1 - a) * v8 Fast + a * v14 Fast v2, scored on TUNING data only
(tuning labels + held-out product titles from pairs_MAX_dev).   python build/soup_fast.py [a]  (with a: writes models/v14f2/soup.pt)"""
import json, os, random, sys, types, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import ft_v8 as V8
from student_data import student_input
a0 = types.SimpleNamespace(tag="v14f2", rank=64); tok, m, ex, _ = V8.load(a0)
A = torch.load(os.path.join(ROOT, "models", "student_V8F", "student.pt"), map_location="cpu")["model"]
B = torch.load(os.path.join(V8.out_dir("v14f2"), "latest.pt"), map_location="cpu")
blend = lambda al: {k: ((1 - al) * A[k].float() + al * v.float()) if v.is_floating_point() else v for k, v in B["model"].items()}
if len(sys.argv) > 1:
    torch.save({"model": blend(float(sys.argv[1])), "experts": B["experts"], "step": B.get("step", 0)}, os.path.join(V8.out_dir("v14f2"), "soup.pt")); print("wrote soup", sys.argv[1]); sys.exit()
DL = json.load(open(os.path.join(ROOT, "test", "dev_labels.json"), encoding="utf-8"))["items"]
dev = [json.loads(l) for l in open(os.path.join(W, "pairs_MAX_dev.jsonl"), encoding="utf-8")]; random.Random(5).shuffle(dev)
sets = {"labels": [(g["en"], s) for g in DL for s in g["src"].values()], "titles": [([p["tgt"]], p["src"]) for p in dev if p["kind"] == "bt"][:400]}
torch.set_num_threads(4); res = []
for al in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
    m.load_state_dict(blend(al)); r = {"a": al}
    for k, it in sets.items():
        srcs = [student_input(s) for _, s in it]; outs = V8.translate(tok, m, ex, srcs, [0] * len(srcs)); r[k] = V8.score(outs, [x for x, _ in it])["chrF"]
    r["mean"] = round((r["labels"] + r["titles"]) / 2, 2); res.append(r); print(r, flush=True)
json.dump(res, open(os.path.join(W, "soup_fast.json"), "w"), indent=1)
