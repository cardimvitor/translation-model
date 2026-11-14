"""v11 step 2: Hy-MT2-7B as judge and teacher (xx -> en), CPU, resumable.
  python build/hy7_judge.py judge     every pool pair the parent model missed (chrF < 60): the 7B translates the source.
                                      -> build/work/v11_judged.jsonl {"src","lang","tgt_old","pred","hy7"}
  python build/hy7_judge.py real      real Spanish and Japanese search queries typed by people (Amazon ESCI es / jp
                                      locales), translated by the 7B. -> build/work/v11_real.jsonl {"src","lang","hy7"}"""
import glob, hashlib, json, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
NAMES = {"zh": "Chinese", "es": "Spanish", "pt": "Portuguese", "ar": "Arabic", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
         "hi": "Hindi", "tr": "Turkish", "id": "Indonesian", "vi": "Vietnamese", "it": "Italian", "pl": "Polish", "th": "Thai"}


def run(jobs, out, key):
    from hy_gen import Hy
    done = {json.loads(l)["src"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    jobs = [j for j in jobs if j["src"] not in done]; print(key, "jobs left", len(jobs), flush=True)
    h = Hy(os.path.join(ROOT, "models", "hy-mt2", "Hy-MT2-7B-Q4_K_M.gguf")); t0 = time.time(); k = 0
    with open(out, "a", encoding="utf-8") as f:
        for j in jobs:
            j["hy7"] = h.one(j["src"], "en", max_new=48)[0]; f.write(json.dumps(j, ensure_ascii=False) + "\n"); f.flush(); k += 1
            if k % 200 == 0:
                print(f"{key} {k}/{len(jobs)} {(time.time() - t0) / k:.2f} s/item", flush=True)
    print(key, "done", flush=True)


def judge():
    P = [json.loads(l) for l in open(os.path.join(W, "v11_pool.jsonl"), encoding="utf-8")]
    miss = [{"src": p["src"], "lang": p["lang"], "tgt_old": p["tgt"], "pred": p["pred"]} for p in P if p["chrf"] < 60]
    from collections import Counter
    seen = set(); n = Counter(); cap = int(os.environ.get("JUDGE_CAP", 1500)); jobs = []      # at most cap per language (time budget)
    for j in miss:
        if j["src"] not in seen and n[j["lang"]] < cap:
            seen.add(j["src"]); n[j["lang"]] += 1; jobs.append(j)
    run(jobs, os.path.join(W, "v11_judged.jsonl"), "judge")


def real():
    import pyarrow.parquet as pq
    from collections import Counter
    from student_data import student_input
    held = set()
    for f in ("gold_products.json", "dev_labels.json"):
        for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]:
            held |= {student_input(s) for s in g["src"].values()}
    c = Counter()
    for f in sorted(glob.glob(os.path.join(W, "dl_esci", "*.parquet"))):
        t = pq.read_table(f, columns=["query", "product_locale"])
        for q, loc in zip(t.column("query").to_pylist(), t.column("product_locale").to_pylist()):
            if loc in ("es", "jp") and q:
                c[("es" if loc == "es" else "ja", student_input(q))] += 1
    jobs = []; n = int(os.environ.get("REAL_N", 900)); bad = re.compile(r"無修正|porn|sex|adult|xxx", re.I)
    for lang, pat in (("es", r"[a-zñáéíóúü' \-]+"), ("ja", r".*[぀-ヿ一-鿿].*")):
        qs = [q for (l, q), _ in sorted(c.items(), key=lambda x: (-x[1], hashlib.md5(x[0][1].encode()).hexdigest())) if l == lang and re.fullmatch(pat, q) and 4 <= len(q) <= 40
              and len(q.split()) <= 6 and q not in held and not bad.search(q) and (lang == "ja" or len(q.split()) >= 2)]
        jobs += [{"src": q, "lang": lang} for q in qs[:n]]
    run(jobs, os.path.join(W, "v11_real.jsonl"), "real")


if __name__ == "__main__":
    {"judge": judge, "real": real}[sys.argv[1]]()
