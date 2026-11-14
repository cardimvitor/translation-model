"""v10 data: the most frequent short search queries ("head terms"), written in ALL 16 languages by Hy-MT2-1.8B.
OPUS-MT often picks the wrong everyday noun (vacuum cleaner -> "Vakuum-Reiniger" instead of "Staubsauger");
Hy-MT2 gets these right. Frequency = how many catalogue rows the query has in Amazon ESCI (US).
Queries equal to a test or tuning answer are left out (bt_queries.queries). Resumable.
Output: build/work/bt_hy_head.jsonl {"en", "lang", "src"}; then pairs via ft_v8.hy_pairs-style filters."""
import glob, json, os, re, sys, time
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path.insert(0, HERE)
LANGS = ["de", "hi", "zh", "pt", "id", "es", "tr", "ru", "ar", "vi", "ja", "pl", "it", "fr", "ko", "th"]      # weakest first


def head(n):
    import pyarrow.parquet as pq
    from bt_queries import queries
    ok = set(queries()); c = Counter()
    for f in sorted(glob.glob(os.path.join(W, "dl_esci", "*.parquet"))):
        t = pq.read_table(f, columns=["query", "product_locale"])
        for q, loc in zip(t.column("query").to_pylist(), t.column("product_locale").to_pylist()):
            if loc == "us" and q:
                q = re.sub(r"\s+", " ", q.lower()).strip()
                if q in ok and re.fullmatch(r"[a-z '\-]+", q) and len(q.split()) <= 4:
                    c[q] += 1
    return [q for q, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:n]]


def main():
    from hy_gen import Hy
    T = head(int(os.environ.get("HEAD_N", 1200))); out = os.path.join(W, "bt_hy_head.jsonl")
    done = {(json.loads(x)["en"], json.loads(x)["lang"]) for x in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    J = [(q, l) for l in LANGS for q in T if (q, l) not in done]; print("head terms", len(T), "e.g.", T[:12], "| jobs left", len(J), flush=True)
    h = Hy(os.path.join(ROOT, "models", "hy-mt2", "Hy-MT2-1.8B-Q4_K_M.gguf")); t0 = time.time(); k = 0
    with open(out, "a", encoding="utf-8") as f:
        for en, l in J:
            s = h.one(en, l, max_new=40)[0]
            f.write(json.dumps({"en": en, "lang": l, "src": s}, ensure_ascii=False) + "\n"); f.flush(); k += 1
            if k % 300 == 0:
                print(f"{k}/{len(J)} {(time.time() - t0) / k:.2f} s/item lang {l}", flush=True)
    print("head terms done", flush=True)


if __name__ == "__main__":
    main()
