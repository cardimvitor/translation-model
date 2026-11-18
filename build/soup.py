"""Weight change after post-training: blend the parent's weights with the child's ("model soup").
  w = (1 - a) * parent + a * child,  for the model and the language experts.
  python build/soup.py post_student_V12            scores each blend on TUNING data only (tuning labels + mixed tuning labels)
  python build/soup.py post_student_V12 0.5        writes models/<tag>/soup.pt with that blend (export it with --ckpt)"""
import json, os, sys, types, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import ft_v8 as V8
from student_data import student_input
tag = sys.argv[1]; a = types.SimpleNamespace(tag=tag, rank=int(os.environ.get("RANK", 64))); d = V8.out_dir(tag)
tok, m, ex, base = V8.load(a); child = torch.load(os.path.join(d, "latest.pt"), map_location="cpu")
def blend(al):
    out = {}
    for part in ("model", "experts"):
        out[part] = {k: ((1 - al) * base[part][k].float() + al * v.float()) if v.is_floating_point() else v for k, v in child[part].items()}
    return out
if len(sys.argv) > 2:
    sd = blend(float(sys.argv[2])); sd["step"] = child.get("step", 0); torch.save(sd, os.path.join(d, "soup.pt")); print("wrote", os.path.join(d, "soup.pt"), "blend", sys.argv[2]); sys.exit()
def items(f):
    r = []
    for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]:
        refs = g["en"] if isinstance(g["en"], list) else [g["en"]]
        r += [(refs, s) for s in (g["src"].values() if isinstance(g["src"], dict) else [g["src"]])]
    return r
sets = {"tuning_labels": items("dev_labels.json"), "mixed_tuning": items("mixed_labels.json")}; res = []
torch.set_num_threads(4)
for al in [float(x) for x in os.environ.get("GRID", "0,0.25,0.4,0.5,0.6,0.75,1").split(",")]:
    sd = blend(al); m.load_state_dict(sd["model"]); ex.load_state_dict(sd["experts"]); r = {"blend": al}
    for k, it in sets.items():
        srcs = [student_input(s) for _, s in it]; outs = V8.translate(tok, m, ex, srcs, [V8.rt(s) for s in srcs]); sc = V8.score(outs, [x for x, _ in it]); r[k] = sc["chrF"]; r[k + "_exact"] = sc["exact"]
    r["mean"] = round((r["tuning_labels"] + r["mixed_tuning"]) / 2, 2); res.append(r); print(r, flush=True)
json.dump(res, open(os.path.join(W, f"soup_{tag}.json"), "w"), indent=1)
