"""Choose the beam width (and length penalty) for a model with a batched decoder, on tuning data only: the
hand-written tuning labels and held-out titles, model alone, 8-bit, one CPU thread. Stores "beams" and "lp" in the
model's vocab.json policy. A wider beam costs time on every label, so it is kept only if it adds at least 0.5 to the mean chrF over the narrower one already chosen.
  NEURAL_DIR=student_V10 python build/tune_beams.py     -> build/work/beams_<model>.json"""
import json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import student
from analyze_student import clean, chrf


def main():
    name = os.environ.get("NEURAL_DIR", "student_V10"); md = os.path.join(ROOT, "models", name); S = student.Student(md)
    DL = json.load(open(os.path.join(ROOT, "test", "dev_labels.json"), encoding="utf-8"))["items"]
    dev = [json.loads(l) for l in open(os.path.join(W, "pairs_MAX_dev.jsonl"), encoding="utf-8")]; random.Random(5).shuffle(dev)
    sets = {"labels": [(g["en"], s) for g in DL for s in g["src"].values()], "titles": [([p["tgt"]], p["src"]) for p in dev if p["kind"] == "bt"][:300]}
    prev = os.path.join(W, f"beams_{name}.json"); res = json.load(open(prev))["all"] if os.environ.get("REUSE") and os.path.exists(prev) else []
    for K, lp in [] if res else [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (4, 0.6), (5, 1.0)]:
        os.environ["NEURAL_BEAMS"] = str(K); os.environ["NEURAL_LP"] = str(lp); r = {"beams": K, "lp": lp}
        for k, items in sets.items():
            t0 = time.time(); outs = [S.translate(s)[0] for _, s in items]
            r[k] = round(sum(max(chrf(clean(o), clean(x)) for x in refs) for o, (refs, _) in zip(outs, items)) / len(items), 2); r[k + "_ms"] = round(1000 * (time.time() - t0) / len(items), 1)
        r["mean"] = round((r["labels"] + r["titles"]) / 2, 2); res.append(r); print(r, flush=True)
    best = res[0]
    for r in sorted(res[1:], key=lambda x: (x["beams"], -x["lp"])):
        if r["mean"] >= best["mean"] + 0.5:
            best = r
    v = json.load(open(os.path.join(md, "vocab.json"), encoding="utf-8")); pol = v.get("policy") or {}; pol["beams"] = best["beams"]; pol["lp"] = best["lp"]; pol["beam_max_src"] = int(os.environ.get("BEAM_MAX_SRC", 16)); v["policy"] = pol   # beam only for inputs up to 16 word pieces (short labels)
    json.dump(v, open(os.path.join(md, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"model": name, "chosen": best, "all": res}, open(os.path.join(W, f"beams_{name}.json"), "w"), indent=1); print("chosen", best, flush=True)


if __name__ == "__main__":
    main()
