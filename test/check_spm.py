"""Check runtime/spm.py against the real sentencepiece library on training sources."""
import json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import sentencepiece as spm, spm as mine
D = os.path.join(ROOT, "models", sys.argv[1] if len(sys.argv) > 1 else "marian_ft")
U = mine.Unigram(json.load(open(os.path.join(D, "spm_src.json"), encoding="utf-8")))
sp = spm.SentencePieceProcessor(model_file=os.path.join(D, "source.spm"))
S = [json.loads(l)["src"] for l in open(os.path.join(ROOT, "build", "work", "pairs_MAX_train.jsonl"), encoding="utf-8")]
random.Random(2).shuffle(S); S = S[:int(os.environ.get("N", 60000))]
for tie in (False, True):
    mine.TIE_LATER = tie; bad = 0; ex = []
    for s in S:
        a = sp.encode(s, out_type=str); b = U.encode(s)
        if a != b:
            bad += 1
            if len(ex) < 4:
                ex.append((s, a, b))
    print("tie_later", tie, "mismatch", bad, "/", len(S))
    for e in ex:
        print(" ", e)
