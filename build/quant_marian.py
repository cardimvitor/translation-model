"""Find 8-bit settings for the v6 translator that keep full-precision quality. Scored on tuning data only
(hand-written dev labels + held-out titles), never on the test sets.
  python build/quant_marian.py student_V6        -> build/work/quant_<model>.json, best variant installed as *_int8.onnx"""
import json, os, random, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import student
from analyze_student import clean, chrf


def data():
    DL = json.load(open(os.path.join(ROOT, "test", "dev_labels.json"), encoding="utf-8"))["items"]
    dev = [json.loads(l) for l in open(os.path.join(W, "pairs_MAX_dev.jsonl"), encoding="utf-8")]; random.Random(5).shuffle(dev)
    bt = [p for p in dev if p["kind"] == "bt"][:150]; real = [p for p in dev if p["kind"] == "real"][:150]
    return {"labels": [(g["en"], s) for g in DL for s in g["src"].values()], "titles": [([p["tgt"]], p["src"]) for p in bt],
            **({"real": [([p["tgt"]], p["src"]) for p in real]} if real else {})}


def run(md, D):
    S = student.Student(md, int8=True); r = {}; t0 = time.time(); n = 0
    for k, items in D.items():
        outs = [S.translate(s)[0] for _, s in items]; n += len(items)
        r[k] = round(sum(max(chrf(clean(o), clean(x)) for x in refs) for o, (refs, _) in zip(outs, items)) / len(items), 2)
    r["ms"] = round(1000 * (time.time() - t0) / n, 1); return r


def main():
    name = sys.argv[1]; md = os.path.join(ROOT, "models", name); D = data(); tmp = os.path.join(md, "q"); os.makedirs(tmp, exist_ok=True)
    def nodes(part, pred):
        m = onnx.load(os.path.join(md, f"{part}_fp32.onnx.bak"), load_external_data=False)
        return [n.name for n in m.graph.node if pred(n)]
    big = lambda part: nodes(part, lambda n: n.op_type == "Gather" and "embed_tokens" in n.name) + nodes(part, lambda n: n.op_type == "MatMul" and "lm_head" in n.name)
    variants = {
        "default": dict(),
        "per_channel": dict(per_channel=True),
        "no_embed_lmhead": dict(skip="big"),
        "matmul_only": dict(op_types_to_quantize=["MatMul"]),
        "matmul_only_per_channel": dict(op_types_to_quantize=["MatMul"], per_channel=True),
        "unsigned": dict(weight_type=QuantType.QUInt8),
    }
    res = {}
    for vn, kw in variants.items():
        size = 0; vd = os.path.join(md, "q_" + vn)
        if os.path.exists(os.path.join(vd, "decoder_int8.onnx")):       # already quantized: score it again
            for f in ("vocab.json", "spm_src.json"):
                shutil.copy(os.path.join(md, f), vd)
            r = run(vd, D); r["MB"] = round(sum(os.path.getsize(os.path.join(vd, f"{p}_int8.onnx")) for p in ("encoder", "decoder")) / 1e6, 1)
            res[vn] = r; print(vn, r, flush=True); continue
        for part in ("encoder", "decoder"):
            k = dict(kw); skip = k.pop("skip", None)
            if skip:
                k["nodes_to_exclude"] = big(part)
            k.setdefault("weight_type", QuantType.QInt8)
            out = os.path.join(tmp, f"{part}_int8.onnx")
            quantize_dynamic(os.path.join(md, f"{part}_fp32.onnx.bak"), out, **k); size += os.path.getsize(out)
        for f in ("vocab.json", "spm_src.json"):
            shutil.copy(os.path.join(md, f), tmp)
        try:
            r = run(tmp, D)
        except Exception as e:
            r = {"error": str(e)[:200]}
        r["MB"] = round(size / 1e6, 1); res[vn] = r; print(vn, r, flush=True)
        vd = os.path.join(md, "q_" + vn); os.makedirs(vd, exist_ok=True)
        for part in ("encoder", "decoder"):
            shutil.move(os.path.join(tmp, f"{part}_int8.onnx"), os.path.join(vd, f"{part}_int8.onnx"))
    json.dump(res, open(os.path.join(W, f"quant_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
