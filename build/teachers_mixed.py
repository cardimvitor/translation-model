"""The teachers and the untrained base on the two mixed-language tests, model alone.
  python build/teachers_mixed.py   -> build/work/teachers/mixed_<opus|nllb|mul>.json
opus = Helsinki-NLP/opus-mt-tr-en for the English + Turkish test (the per-language OPUS-MT model a teacher would use);
mul  = Helsinki-NLP/opus-mt-mul-en untrained (v6's start); nllb = NLLB-200 600M (8-bit ONNX), source language given."""
import json, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
from analyze_student import clean, chrf
torch.set_num_threads(4)
T = {f: json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"] for f in ("mixed_tr.json", "mixed_labels.json")}


def score(fn, tag, sets):
    out = {}
    for f in sets:
        rows = []
        for g in T[f]:
            o = fn(g["src"], g["lang"]); c = clean(o); rf = [clean(x) for x in g["en"]]
            rows.append({"lang": g["lang"], "src": g["src"], "out": o, "exact": c in rf, "chrf": max(chrf(c, x) for x in rf)})
        out[f] = rows; print(tag, f, f"{sum(r['exact'] for r in rows) / len(rows) * 100:.1f}% / {sum(r['chrf'] for r in rows) / len(rows):.1f}", flush=True)
    json.dump(out, open(os.path.join(W, "teachers", f"mixed_{tag}.json"), "w"), ensure_ascii=False, indent=1)


def marian(name):
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(name); m = MarianMTModel.from_pretrained(name).eval()
    def f(s, lang):
        with torch.no_grad():
            g = m.generate(**tok([s], return_tensors="pt"), num_beams=1, max_new_tokens=64)
        return tok.batch_decode(g, skip_special_tokens=True)[0]
    return f


which = sys.argv[1:] or ["mul", "opus", "nllb"]
if "mul" in which:
    score(marian("Helsinki-NLP/opus-mt-mul-en"), "mul", ["mixed_tr.json", "mixed_labels.json"])
if "opus" in which:
    score(marian("Helsinki-NLP/opus-mt-tr-en"), "opus", ["mixed_tr.json"])
if "nllb" in which:
    os.environ["NEURAL"] = "nllb-600m"
    import accurate as AC
    M = AC.MT(os.path.join(ROOT, "models", "nllb-600m"))
    score(lambda s, lang: M.translate(s, lang)[0], "nllb", ["mixed_tr.json", "mixed_labels.json"])
