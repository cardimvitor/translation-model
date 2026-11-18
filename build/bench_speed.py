"""Speed of every model on this CPU: median time per request, translations per second, output tokens per second.
Two request types: short labels (the 320 gold labels) and long product names (330 ABO names, 30 per language).
  python build/bench_speed.py ours            our models: model alone and full system, ONE CPU thread (as the runtimes run)
  python build/bench_speed.py base            base models on a smaller sample (64 labels, 44 names), 4 CPU threads
"Tokens" are each model's own decoding steps (characters for v3-v5, word pieces for v6/v7 and the base models).
-> build/work/speed_ours.json, build/work/speed_base.json"""
import json, os, statistics, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work")
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
OURS_ALL = ["none", "student_edited", "student_A", "student_B", "student_C", "student_B2", "student_C2_edited", "student_H", "student_ALL3_edited",
        "student_HY_edited", "student_MAX_edited", "student_JG", "student_BASE_edited", "student_V6", "student_V7"]


OURS = os.environ["BENCH_MODELS"].split(",") if os.environ.get("BENCH_MODELS") else OURS_ALL


def data(per_lang_labels=None, per_lang_names=30):
    G = json.load(open(os.path.join(ROOT, "test", "gold_products.json"), encoding="utf-8"))["items"]
    if per_lang_labels:
        G = G[:per_lang_labels]
    labels = [(s, l) for g in G for l, s in g["src"].items()]
    seen = {}; names = []
    for p in map(json.loads, open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8")):
        seen[p["lang"]] = seen.get(p["lang"], 0) + 1
        if seen[p["lang"]] <= per_lang_names:
            names.append((p["src"], p["lang"]))
    return {"labels": labels, "names": names}


def stats(ts, toks=None):
    r = {"n": len(ts), "median_ms": round(1000 * statistics.median(ts), 1), "p90_ms": round(1000 * sorted(ts)[int(0.9 * len(ts))], 1),
         "per_second": round(len(ts) / sum(ts), 1)}
    if toks is not None:
        r["tokens_per_second"] = round(sum(toks) / sum(ts), 1)
    return r


def one(model):
    os.environ["NEURAL_DIR"] = model
    import normalizer as F
    D = data(); res = {}
    if model == "none":
        for k, items in D.items():
            F.process("translate", items[0][0]); ts = []
            for s, _ in items:
                t = time.perf_counter(); F.process("translate", s); ts.append(time.perf_counter() - t)
            res[k] = {"system": stats(ts)}
        print(json.dumps(res)); return
    import accurate as AC
    S = AC.mt(); steps = [0]; run = S.dec.run
    def counted(*a, **kw):
        steps[0] += 1; return run(*a, **kw)
    class Dec:                                   # count decoder steps (= output tokens incl. the end mark)
        def run(self, *a, **kw): return counted(*a, **kw)
    S.dec = Dec()
    for k, items in D.items():
        for s, _ in items[:8]:                   # warm up (and load the language experts of v7)
            S.translate(s)
        ts, tk = [], []
        for s, _ in items:
            steps[0] = 0; t = time.perf_counter(); S.translate(s); ts.append(time.perf_counter() - t); tk.append(steps[0])
        ts2 = []
        for s, _ in items:
            t = time.perf_counter(); AC.process("translate", s); ts2.append(time.perf_counter() - t)
        res[k] = {"alone": stats(ts, tk), "system": stats(ts2)}
    print(json.dumps(res))


def base(name):
    import torch
    torch.set_num_threads(4); D = data(4, 4); res = {}
    if name == "opus":
        import opus_gen as O
        def tr(s, l):
            tok, m = O._load(O.REV[l]); enc = tok([s], return_tensors="pt", truncation=True, max_length=96)
            with torch.inference_mode():
                g = m.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=64)
            return int(g.shape[1]) - 1
        warm = lambda l: O._load(O.REV[l])
    elif name == "nllb":
        from nllb_gen import NLLB
        N = NLLB()
        def tr(s, l):
            o = N.translate([s], l, "en", max_new=64)[0]; o = o if isinstance(o, str) else o[0]
            return len(N.tok.encode(o, add_special_tokens=False).ids) + 1
        warm = lambda l: None
    else:
        from hy_gen import Hy
        H = Hy(os.path.join(ROOT, "models", "hy-mt2", "Hy-MT2-1.8B-Q4_K_M.gguf" if name == "hy18" else "Hy-MT2-7B-Q4_K_M.gguf"))
        def tr(s, l):
            r = H.m.create_chat_completion(messages=[{"role": "user", "content": H.prompt(s, "en")}], temperature=0.0, top_k=1, repeat_penalty=1.05, max_tokens=96)
            return r["usage"]["completion_tokens"]
        warm = lambda l: None
    for k, items in D.items():
        items = sorted(items, key=lambda x: x[1]); ts, tk = [], []; last = None
        for s, l in items:
            if l not in ("zh", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th"):
                continue
            if l != last:
                warm(l); tr(s, l); last = l          # model load and first call are not timed
            t = time.perf_counter(); n = tr(s, l); ts.append(time.perf_counter() - t); tk.append(n)
        res[k] = stats(ts, tk); print(name, k, res[k], file=sys.stderr, flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "--one":
        one(sys.argv[2])
    elif mode == "--base":
        base(sys.argv[2])
    else:
        names = OURS if mode == "ours" else ["opus", "nllb", "hy18", "hy7"]; outp = os.path.join(W, f"speed_{mode}.json")
        out = json.load(open(outp)) if os.path.exists(outp) else {}
        for m in names:
            if m in out:
                continue
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--one" if mode == "ours" else "--base", m], capture_output=True, text=True)
            try:
                out[m] = json.loads(r.stdout.strip().split("\n")[-1])
            except Exception:
                print("FAILED", m, r.stderr[-600:], flush=True); continue
            json.dump(out, open(outp, "w"), indent=1); print(m, json.dumps(out[m])[:330], flush=True)
