"""v8 data: short labels for the weakest languages, written by Hy-MT2-1.8B (Apache-2.0).
English ESCI search queries (same cleaning as bt_queries.py, test and tuning answers left out) are translated
en -> ko / th / vi / ja / zh. OPUS-MT has no en->ko / en->th model, so v6 had no short-label data for those two.
Resumable. Output: build/work/bt_hy_q.jsonl {"en", "lang", "src"}"""
import json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path.insert(0, HERE)
PLAN = [("ko", 5000), ("th", 4000), ("vi", 1500), ("ja", 1500), ("zh", 1500)]


def main():
    from bt_queries import queries
    from hy_gen import Hy
    Q = queries(); random.Random(31).shuffle(Q); J = []; o = 0
    for l, n in PLAN:
        J += [(q, l) for q in Q[o:o + n]]; o += n
    out = os.path.join(W, "bt_hy_q.jsonl")
    done = {(json.loads(x)["en"], json.loads(x)["lang"]) for x in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    J = [j for j in J if j not in done]; print("jobs left", len(J), flush=True)
    h = Hy(os.path.join(ROOT, "models", "hy-mt2", "Hy-MT2-1.8B-Q4_K_M.gguf")); t0 = time.time(); k = 0
    with open(out, "a", encoding="utf-8") as f:
        for en, l in J:
            s = h.one(en, l, max_new=40)[0]
            f.write(json.dumps({"en": en, "lang": l, "src": s}, ensure_ascii=False) + "\n"); f.flush(); k += 1
            if k % 200 == 0:
                print(f"{k}/{len(J)} {(time.time() - t0) / k:.2f} s/item lang {l}", flush=True)
    print("hy queries done", flush=True)


if __name__ == "__main__":
    main()
