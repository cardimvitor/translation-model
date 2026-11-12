"""Short-label training data (v6): Amazon ESCI search queries (US, English; Apache-2.0) are short product labels
("yoga mat", "bathroom fan without light"). OPUS-MT translates each into Q_LANGS of its 14 languages (rotating) and
back again. Queries equal to an accepted answer of the gold test labels or the dev labels are left out.
Resumable. Output: build/work/bt_opus_q.jsonl (same format as bt_opus_v2.jsonl)."""
import glob, json, os, random, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path.insert(0, HERE)
B = 64


def queries():
    import pyarrow.parquet as pq
    held = set()
    for f in ("gold_products.json", "dev_labels.json"):
        for g in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))["items"]:
            held |= {e.lower() for e in g["en"]}
    seen, out = set(), []
    for f in sorted(glob.glob(os.path.join(W, "dl_esci", "*.parquet"))):
        t = pq.read_table(f, columns=["query", "product_locale"])
        for q, loc in zip(t.column("query").to_pylist(), t.column("product_locale").to_pylist()):
            if loc != "us" or not q:
                continue
            q = re.sub(r"\s+", " ", q.lower()).strip()
            w = q.split()
            if q in seen or q in held or not (1 <= len(w) <= 6) or not (4 <= len(q) <= 44):
                continue
            if not re.fullmatch(r"[a-z0-9 '&\-]+", q) or sum(any(c.isdigit() for c in x) for x in w) > 1 or not any(len(x) > 2 and x.isalpha() for x in w):
                continue
            seen.add(q); out.append(q)
    return out


def main():
    import torch, opus_gen as O
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    langs = [l for l, (m, _) in O.FWD.items() if m != "m2m100"]; n = int(os.environ.get("Q_LANGS", 4))
    Q = queries(); rng = random.Random(31); rng.shuffle(Q); Q = Q[: int(os.environ.get("Q_CAP", 45000))]
    J = [(q, langs[(i * n + k) % len(langs)]) for i, q in enumerate(Q) for k in range(n)]
    out = os.path.join(W, "bt_opus_q.jsonl")
    done = {(json.loads(l)["en"], json.loads(l)["lang"]) for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    J = [j for j in J if j not in done]; print("queries", len(Q), "jobs left", len(J), flush=True)
    by = {}
    for t, l in J:
        by.setdefault(l, []).append(t)
    t0 = time.time(); k = 0
    with open(out, "a", encoding="utf-8") as f:
        for l, ts in by.items():
            ts.sort(key=len)
            for i in range(0, len(ts), B):
                batch = ts[i:i + B]; res = O.fwd(batch, l); back = O.rev([s for s, _ in res], l)
                for en, (s, c), (b, bc) in zip(batch, res, back):
                    f.write(json.dumps({"en": en, "lang": l, "src": s, "conf": c, "back": b, "back_conf": bc}, ensure_ascii=False) + "\n")
                f.flush(); k += len(batch)
                if (i // B) % 20 == 0:
                    print(f"{k}/{len(J)} done, {(time.time() - t0) / k * 1000:.0f} ms/item, lang {l}", flush=True)


if __name__ == "__main__":
    main()
