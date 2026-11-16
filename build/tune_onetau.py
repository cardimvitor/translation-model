"""Tunes policy.one_tau (keep the romanized form of one unknown word when the translator is unsure), outside the tests.
Uses 900 small-town names (should keep romanization) and the single-word tuning labels (must still translate).
Picks the highest threshold that loses no correct single-word tuning label.   python build/tune_onetau.py <model>"""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
model = sys.argv[1]; os.environ["NEURAL_DIR"] = model
import accurate as AC, normalizer as F
from analyze_student import clean
F.gaz(); M = AC.mt(None); rng = random.Random(5)
towns = rng.sample(sorted(k for k in F._GAZT if " " not in k and 3 <= len(k) <= 14), 900)
dev = [(s, g["en"]) for g in json.load(open(os.path.join(ROOT, "test", "dev_labels.json"), encoding="utf-8"))["items"] for s in g["src"].values() if " " not in s]
N = [M.translate(s)[1] for s in towns]; D = [(M.translate(s), refs) for s, refs in dev]
res = []
for t in (0.2, 0.3, 0.4, 0.5, 0.6):
    lost = sum(1 for (o, c), refs in D if c < t and clean(o) in [clean(r) for r in refs])
    res.append({"one_tau": t, "town_names_still_translated": round(sum(c >= t for c in N) / len(N), 3), "correct_tuning_words_lost": lost}); print(res[-1], flush=True)
best = max([r for r in res if r["correct_tuning_words_lost"] == 0] or [res[0]], key=lambda r: r["one_tau"])
vp = os.path.join(ROOT, "models", model, "vocab.json"); V = json.load(open(vp, encoding="utf-8")); V["policy"]["one_tau"] = best["one_tau"]
json.dump(V, open(vp, "w", encoding="utf-8"), ensure_ascii=False); json.dump({"chosen": best, "all": res}, open(os.path.join(W, f"onetau_{model}.json"), "w"), indent=1); print("chosen", best)
