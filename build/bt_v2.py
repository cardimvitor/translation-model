"""v4 back-translation. English titles -> 3 random languages each (same assignment for both teachers).
  python bt_v2.py nllb   -> build/work/bt_nllb_v2.jsonl   (variant A, NLLB-200 600M, CC BY-NC)
  python bt_v2.py opus   -> build/work/bt_opus_v2.jsonl   (variant B, OPUS-MT / M2M100, CC BY / MIT),
                            each pair also translated back to English for the round-trip filter.
Resumable. Titles: ESCI (Apache-2.0) sample + the v3 titles (Shopify, H&M, glossary combos)."""
import json, os, random, sys, time, zlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runtime"))
LANGS = ["zh", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th"]
W = os.path.join(HERE, "work")
N_ESCI = int(os.environ.get("N_ESCI", 36000)); PER = 3; B = 32


def titles():
    from backtranslate import english_titles
    esci = json.load(open(os.path.join(W, "titles_esci.json"), encoding="utf-8"))[:N_ESCI]
    seen = set(); out = []
    for t in english_titles() + esci:
        if t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out


def is_dev(title):
    """Fixed 5% title hold-out, the same for every variant."""
    return zlib.crc32(title.lower().encode()) % 20 == 0


def jobs():
    rng = random.Random(7)
    return [(t, l) for t in titles() for l in rng.sample(LANGS, PER)]


def main(teacher):
    out = os.path.join(W, f"bt_{teacher}_v2.jsonl"); J = jobs(); done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line); done.add((r["en"], r["lang"]))
    J = [j for j in J if j not in done]
    if teacher == "nllb":                          # titles already translated into 8 languages by v3 (bt.jsonl)
        v3 = {json.loads(l)["en"] for l in open(os.path.join(W, "bt.jsonl"), encoding="utf-8")}
        J = [j for j in J if j[0] not in v3]
    print("jobs left", len(J), flush=True)
    by = {}
    for t, l in J:
        by.setdefault(l, []).append(t)
    if teacher == "nllb":
        from nllb_gen import NLLB
        n = NLLB(threads=int(os.environ.get("THREADS", 4)))
        fwd = lambda ts, l: n.translate(ts, "en", l); rev = None
    else:
        import torch, opus_gen as O
        torch.set_num_threads(int(os.environ.get("THREADS", 4)))
        fwd, rev = O.fwd, O.rev
    t0 = time.time(); k = 0
    with open(out, "a", encoding="utf-8") as f:
        for l, ts in by.items():
            ts.sort(key=len)
            for i in range(0, len(ts), B):          # forward + reverse per batch, saved at once (restart-safe)
                batch = ts[i:i + B]; res = fwd(batch, l)
                back = rev([s for s, _ in res], l) if rev else None
                for j, (en, (s, c)) in enumerate(zip(batch, res)):
                    r = {"en": en, "lang": l, "src": s, "conf": c}
                    if back:
                        r["back"], r["back_conf"] = back[j]
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush(); k += len(batch)
                if (i // B) % 20 == 0:
                    print(f"{k}/{len(J)} done, {(time.time() - t0) / k * 1000:.0f} ms/item, lang {l}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
