"""Weight change after training: average the last saved checkpoints of a run (same idea as v4's checkpoint averaging).
  python build/avg_ckpt.py post_student_V12      -> models/post_student_V12/avg.pt   (model + language experts averaged)
Keep it only if `python build/ft_v8.py eval --tag <tag> --ckpt <avg.pt>` beats latest.pt on the tuning labels."""
import glob, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
d = os.path.join(ROOT, "models", sys.argv[1]); files = sorted(glob.glob(os.path.join(d, "step_*.pt")))[-3:] + [os.path.join(d, "latest.pt")]
cks = [torch.load(f, map_location="cpu") for f in files]; out = {"step": cks[-1].get("step", 0)}
for part in ("model", "experts"):
    out[part] = {}
    for k, v in cks[-1][part].items():
        out[part][k] = sum(c[part][k].float() for c in cks) / len(cks) if v.is_floating_point() else v
torch.save(out, os.path.join(d, "avg.pt")); print("averaged", [os.path.basename(f) for f in files], "->", os.path.join(d, "avg.pt"))
