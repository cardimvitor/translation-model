"""Base (teacher) models on the ABO real-name test, alone, for comparison with our models. Resumable per item.
  python build/eval_teachers.py opus | nllb | hy18 | hy7     -> build/work/teachers/abo_<name>.json (+ .jsonl outputs)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]


def main(name):
    import torch
    from analyze_student import clean, chrf
    torch.set_num_threads(4)
    A = [json.loads(l) for l in open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8")]
    outp = os.path.join(W, "teachers", f"abo_{name}.jsonl")
    done = {json.loads(l)["src"]: json.loads(l)["out"] for l in open(outp, encoding="utf-8")} if os.path.exists(outp) else {}
    todo = [p for p in A if p["src"] not in done]; print(name, "left", len(todo), flush=True)
    if todo:
        if name == "opus":
            import opus_gen as O
            tr = lambda ts, l: [o for o, _ in O.rev(ts, l)]
        elif name == "nllb":
            from nllb_gen import NLLB
            N = NLLB(); tr = lambda ts, l: [o if isinstance(o, str) else o[0] for o in N.translate(ts, l, "en", max_new=64)]
        else:
            from hy_gen import Hy
            H = Hy(os.path.join(ROOT, "models", "hy-mt2", "Hy-MT2-1.8B-Q4_K_M.gguf" if name == "hy18" else "Hy-MT2-7B-Q4_K_M.gguf"))
            tr = lambda ts, l: [H.one(t, "en", max_new=96)[0] for t in ts]
        by = {}
        for p in todo:
            by.setdefault(p["lang"], []).append(p["src"])
        B = 1 if name.startswith("hy") else 16; t0 = time.time(); k = 0
        with open(outp, "a", encoding="utf-8") as f:
            for l, ss in by.items():
                for i in range(0, len(ss), B):
                    for s, o in zip(ss[i:i + B], tr(ss[i:i + B], l)):
                        done[s] = o; f.write(json.dumps({"src": s, "lang": l, "out": o}, ensure_ascii=False) + "\n")
                    f.flush(); k += len(ss[i:i + B])
                    if k % 48 == 0 or B == 1 and k % 20 == 0:
                        print(f"{k}/{len(todo)} {(time.time() - t0) / k:.2f} s/item", flush=True)
        ms = round(1000 * (time.time() - t0) / max(1, len(todo)))
    else:
        ms = None
    sc = [(clean(done[p["src"]]) == clean(p["en"]), chrf(clean(done[p["src"]]), clean(p["en"])), p["lang"]) for p in A]
    by = {}
    for e, c, l in sc:
        by.setdefault(l, []).append(c)
    res = {"n": len(A), "exact": round(sum(e for e, _, _ in sc) / len(A), 4), "chrF": round(sum(c for _, c, _ in sc) / len(A), 2),
           "by_lang": {l: round(sum(v) / len(v), 1) for l, v in by.items()}}
    old = json.load(open(os.path.join(W, "teachers", f"abo_{name}.json"))) if os.path.exists(os.path.join(W, "teachers", f"abo_{name}.json")) else {}
    res["ms_per_item"] = ms or old.get("ms_per_item")
    json.dump(res, open(os.path.join(W, "teachers", f"abo_{name}.json"), "w"), indent=1); print(name, res, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
