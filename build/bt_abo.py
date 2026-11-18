"""OPUS-MT back-translation of the ABO English product names (more titles, more product vocabulary).
One language per title, rotating over the OPUS-MT forward languages (ko / th need M2M100, skipped here: ko has
real ABO pairs). Held-out ABO test titles are skipped. Resumable; at most ABO_BT_CAP titles (default 60000).
Output: build/work/bt_opus_abo.jsonl (same format as bt_opus_v2.jsonl)."""
import json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work"); sys.path.insert(0, HERE)
from abo_prep import is_test
B = 32


def main():
    import torch, opus_gen as O
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    langs = [l for l, (m, _) in O.FWD.items() if m != "m2m100"]
    T = [t for t in json.load(open(os.path.join(W, "abo_titles.json"), encoding="utf-8")) if not is_test(t)]
    have = set()
    for name in ("bt_opus_v2.jsonl",):
        for line in open(os.path.join(W, name), encoding="utf-8"):
            have.add(json.loads(line)["en"].lower())
    rng = random.Random(23); T = [t for t in T if t.lower() not in have]; rng.shuffle(T)
    T = T[: int(os.environ.get("ABO_BT_CAP", 60000))]
    J = [(t, langs[i % len(langs)]) for i, t in enumerate(T)]
    out = os.path.join(W, "bt_opus_abo.jsonl")
    done = {(json.loads(l)["en"], json.loads(l)["lang"]) for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    J = [j for j in J if j not in done]; print("abo bt jobs left", len(J), flush=True)
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
