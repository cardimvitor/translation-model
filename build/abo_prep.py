"""Split and filter the real ABO name pairs (build/work/abo_pairs.jsonl, from fetch_abo.py).

  test:  titles with crc32(english) % 20 == 0, at most 100 per language -> build/work/abo_test.jsonl (never trained on)
  train: the rest, kept when OPUS-MT's English for the source shares at least 40% of the English name's content
         words (marketplace names are often shorter or reordered, not word-for-word) -> build/work/abo_train.jsonl
Resumable per language (abo_train.jsonl is appended)."""
import json, os, random, re, sys, zlib
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work"); sys.path.insert(0, HERE)


def words(s):
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def is_test(en):
    return zlib.crc32(en.lower().encode()) % 20 == 0


def main():
    P = [json.loads(l) for l in open(os.path.join(W, "abo_pairs.jsonl"), encoding="utf-8")]
    rng = random.Random(21); test, by = {}, {}
    for p in P:
        (test if is_test(p["en"]) else by).setdefault(p["lang"], []).append(p)
    with open(os.path.join(W, "abo_test.jsonl"), "w", encoding="utf-8") as f:
        for l in sorted(test):
            xs = test[l]; rng.shuffle(xs)
            for p in xs[:100]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    outp = os.path.join(W, "abo_train.jsonl")
    done = {json.loads(l)["lang"] for l in open(outp, encoding="utf-8")} if os.path.exists(outp) else set()
    import torch, opus_gen as O
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    with open(outp, "a", encoding="utf-8") as f:
        for l in sorted(by):
            if l in done or l not in O.REV:
                continue
            xs = by[l]; keep = 0
            for i in range(0, len(xs), 32):
                batch = xs[i:i + 32]
                for p, (b, _) in zip(batch, O.rev([p["src"] for p in batch], l)):
                    a = words(p["en"]); ov = len(a & words(b)) / max(1, len(a))
                    if ov >= 0.4:
                        f.write(json.dumps(dict(p, back=b, overlap=round(ov, 2)), ensure_ascii=False) + "\n"); keep += 1
            f.flush(); print(l, "kept", keep, "of", len(xs), flush=True)


if __name__ == "__main__":
    main()
