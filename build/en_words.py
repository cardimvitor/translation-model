"""English word list for the word check: every word seen at least MIN times on the English side of the training data
(product titles, search queries), so brand names that appear in English labels count as known. Stored in the model's
vocab.json as "en_words".   python build/en_words.py student_V12 [min_count]"""
import json, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
os.environ.setdefault("NEURAL_DIR", sys.argv[1])
import accurate as AC
MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 3; c = Counter(); x = Counter(); LAT = {"es", "pt", "de", "fr", "tr", "id", "vi", "it", "pl"}
for n in ("pairs_MAX_train.jsonl", "pairs_Q_train.jsonl", "pairs_HQ_train.jsonl", "pairs_NEW_v9.jsonl", "pairs_NEW_v10.jsonl", "pairs_NEW_v11.jsonl"):
    p = os.path.join(W, n)
    if os.path.exists(p):
        seen = set()
        for l in open(p, encoding="utf-8"):
            j = json.loads(l); t = j["tgt"]
            if j.get("lang") in LAT and j.get("kind") != "mixed":
                x.update(set(AC._toks(j["src"])))            # words seen in foreign-language text
            if t not in seen:                       # one English text counts once, however many languages it was translated into
                seen.add(t); c.update(set(AC._toks(t)))
words = sorted(w for w, k in c.items() if k >= MIN and len(w) >= 3 and AC._WORD.match(w))
ew = set(words); xx = sorted(w for w, k in x.items() if k >= MIN and len(w) >= 3 and AC._WORD.match(w) and w not in ew and c.get(w, 0) == 0)
vp = os.path.join(ROOT, "models", sys.argv[1], "vocab.json"); v = json.load(open(vp, encoding="utf-8")); v["en_words"] = words; v["xx_words"] = xx; print(len(xx), "foreign words")
json.dump(v, open(vp, "w", encoding="utf-8"), ensure_ascii=False); print(len(words), "English words stored in", vp, "|", round(len(json.dumps(words)) / 1e6, 2), "MB")
