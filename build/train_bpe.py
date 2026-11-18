"""Learn the v5 word-piece vocabularies for one variant from its training pairs.
  python train_bpe.py C2 [--src 16000] [--tgt 8000]   -> build/work/bpe_C2.json {"src": {tokens, merges}, "tgt": {...}}
Training uses HuggingFace tokenizers; encoding at train and run time uses runtime/bpe.py / bpe.js."""
import argparse, json, os, sys
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runtime"))
import bpe as BPE
SPECIAL = ["<pad>", "<unk>", "<s>", "</s>"]


def learn(texts, size, alphabet):
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement=BPE.MARK, prepend_scheme="always", split=True)
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=size, min_frequency=3, special_tokens=SPECIAL,
                                                      limit_alphabet=alphabet, show_progress=False))
    m = json.loads(tk.to_str())["model"]
    merges = [x.split(" ", 1) if isinstance(x, str) else list(x) for x in m["merges"]]
    tokens = [t for t, _ in sorted(m["vocab"].items(), key=lambda kv: kv[1])]
    assert tokens[:4] == SPECIAL, tokens[:4]
    return {"tokens": tokens, "merges": merges}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("variant"); ap.add_argument("--src", type=int, default=16000)
    ap.add_argument("--tgt", type=int, default=8000)
    ap.add_argument("--joint", type=int, default=0, help="one shared vocabulary of this size for input and output")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(W, f"pairs_{a.variant}_train.jsonl"), encoding="utf-8")]
    if a.joint:
        J = learn([r["src"] for r in rows] + [r["tgt"] for r in rows], a.joint, 5000); out = {"src": J, "tgt": J}
    else:
        out = {"src": learn([r["src"] for r in rows], a.src, 5000), "tgt": learn([r["tgt"] for r in rows], a.tgt, 300)}
    for k in ("src", "tgt"):
        V = set(out[k]["tokens"]); n = unk = 0
        for r in rows[:5000]:
            t = BPE.encode(r[k], out[k]["merges"]); n += len(t); unk += sum(x not in V for x in t)
        print(k, "vocab", len(out[k]["tokens"]), "merges", len(out[k]["merges"]), f"avg tokens {n / 5000:.1f}", f"unk {unk / max(1, n):.4f}")
    json.dump(out, open(os.path.join(W, f"bpe_{a.variant}.json"), "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
