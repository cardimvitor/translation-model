"""Score a Marian xx->en model (HF name or local dir) on the gold labels and the ABO real names, alone.
  python build/eval_marian.py Helsinki-NLP/opus-mt-mul-en [tag]  -> build/work/teachers/gold_<tag>.json"""
import json, os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
from analyze_student import clean, chrf
from transformers import MarianMTModel, MarianTokenizer
torch.set_num_threads(4)


def run(name, tag, beams=1):
    tok = MarianTokenizer.from_pretrained(name); m = MarianMTModel.from_pretrained(name).eval()
    def tr(texts):
        out = []
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                g = m.generate(**b, num_beams=beams, max_new_tokens=64)
            out += tok.batch_decode(g, skip_special_tokens=True)
        return out
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    sets = {"gold": [(g["en"], s, l) for g in G for l, s in g["src"].items()],
            "abo": [([p["en"]], p["src"], p["lang"]) for p in map(json.loads, open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8"))]}
    res = {"model": name, "beams": beams}
    for sname, items in sets.items():
        t0 = time.time(); outs = tr([s for _, s, _ in items]); dt = (time.time() - t0) / len(items)
        ex = sum(clean(o) in [clean(r) for r in refs] for o, (refs, _, _) in zip(outs, items)) / len(items)
        ch = sum(max(chrf(clean(o), clean(r)) for r in refs) for o, (refs, _, _) in zip(outs, items)) / len(items)
        by = {}
        for o, (refs, _, l) in zip(outs, items):
            by.setdefault(l, []).append(max(chrf(clean(o), clean(r)) for r in refs))
        res[sname] = {"exact": round(ex, 4), "chrF": round(ch, 2), "ms": round(dt * 1000, 1), "by_lang": {l: round(sum(v) / len(v), 1) for l, v in by.items()}}
        print(sname, res[sname], flush=True)
        json.dump([[s, o] for o, (_, s, _) in zip(outs, items)], open(os.path.join(W, f"marian_{tag}_{sname}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(res, open(os.path.join(W, "teachers", f"marian_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "mul", int(os.environ.get("BEAMS", 1)))
