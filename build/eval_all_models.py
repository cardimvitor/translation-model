"""Every model since the first version on the tests made later, all run on the same day with today's rules and place list:
English + Turkish labels (64), the two places tests (160 + 101), and the 330 real product names Hy-MT2-7B was run on.
  python build/eval_all_models.py [dir ...]   -> build/work/all_models_today.json (resumable: finished models are kept)"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
DIRS = ["none", "student_edited", "student_A", "student_B", "student_C", "student_B2", "student_C2_edited", "student_H", "student_ALL3_edited", "student_HY_edited", "student_MAX_edited",
        "student_JG", "student_BASE_edited", "student_V6", "student_V7", "student_V8", "student_V8F", "student_V9", "student_V10", "student_V11", "student_V12", "student_V13", "student_V14", "student_V14F1", "student_V14F2"]
CHILD = r'''
import json, os, sys, time, unicodedata
ROOT = sys.argv[1]; sys.path[:0] = [os.path.join(ROOT, "runtime"), os.path.join(ROOT, "build")]
from analyze_student import clean, chrf
if os.environ["NEURAL_DIR"] == "none":
    import normalizer as F; run = F.process
else:
    import accurate as A; run = A.process
def fold(s):
    s = unicodedata.normalize("NFKD", s.replace("ı", "i").replace("ł", "l").replace("Ł", "L").replace("đ", "d").replace("Đ", "D").replace("ß", "ss"))
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).casefold().replace(",", " ").replace(".", " ").replace("-", " ").replace("’", "'").split())
r = {}
M = json.load(open(os.path.join(ROOT, "test", "mixed_tr.json"), encoding="utf-8"))["items"]; ex = ch = 0
for g in M:
    o = clean(run("translate", g["src"])["output"]); rf = [clean(x) for x in g["en"]]; ex += o in rf; ch += max(chrf(o, x) for x in rf)
r["mixed_tr"] = [round(ex / len(M) * 100, 1), round(ch / len(M), 1)]
for k, f in (("places1", "places_world.json"), ("places2", "places_world2.json")):
    T = json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]
    r[k] = [sum(fold(run(t["op"], t["in"])["output"]) in [fold(w) for w in t["want"]] for t in T), len(T)]
W = os.path.join(ROOT, "build", "work"); srcs = [json.loads(l)["src"] for l in open(os.path.join(W, "teachers", "abo_hy7.jsonl"), encoding="utf-8")]
ref = {json.loads(l)["src"]: json.loads(l)["en"] for l in open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8")}
outs = [clean(run("translate", s)["output"]) for s in srcs]
r["abo330"] = [round(sum(o == clean(ref[s]) for o, s in zip(outs, srcs)) / len(srcs) * 100, 1), round(sum(chrf(o, clean(ref[s])) for o, s in zip(outs, srcs)) / len(srcs), 2)]
print("RESULT " + json.dumps(r))
'''
out = os.path.join(W, "all_models_today.json"); R = json.load(open(out)) if os.path.exists(out) else {}
for d in (sys.argv[1:] or DIRS):
    if d in R:
        continue
    env = dict(os.environ, NEURAL_DIR=d); p = subprocess.run([sys.executable, "-c", CHILD, ROOT], env=env, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(d, "FAILED", p.stderr[-600:], flush=True); continue
    R[d] = json.loads(line[0][7:]); json.dump(R, open(out, "w"), indent=1); print(d, R[d], flush=True)
