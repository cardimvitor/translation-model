"""Tunes the word check (policy.w_tau) on data outside the tests.
  harm   English text that must stay as it is: 2,000 real English search queries + 993 real English product names (first 4 words)
  catch  English labels with one foreign word (pairs_NEW_v12 kind B, never trained on), Latin-script languages
Picks the lowest w_tau that changes at most MAX_HARM (default 0.3%) of the English texts. python build/tune_wordcheck.py student_V12"""
import glob, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); W = os.path.join(HERE, "work"); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
model = sys.argv[1]; os.environ["NEURAL_DIR"] = model
vp = os.path.join(ROOT, "models", model, "vocab.json"); V = json.load(open(vp, encoding="utf-8")); V["policy"]["w_tau"] = 0.0; V["policy"].setdefault("w_max_words", 8)
V = V
import accurate as AC, normalizer as F
import pyarrow.parquet as pq
from analyze_student import clean
rng = random.Random(3); en = set()
for f in sorted(glob.glob(os.path.join(W, "dl_esci", "*.parquet")))[:2]:
    t = pq.read_table(f, columns=["query", "product_locale"])
    en |= {q.lower().strip() for q, l in zip(t.column("query").to_pylist(), t.column("product_locale").to_pylist()) if l == "us" and q and 4 <= len(q) <= 40 and q.isascii()}
en = rng.sample(sorted(en), 2000) + [" ".join(json.loads(l)["en"].split()[:4]) for l in open(os.path.join(W, "abo_test.jsonl"), encoding="utf-8")]
B = [json.loads(l) for l in open(os.path.join(W, "pairs_NEW_v12.jsonl"), encoding="utf-8")]
B = [p for p in B if p["lang"] in ("es", "pt", "de", "fr", "tr", "id", "vi", "it", "pl") and len(p["src"].split()) == len(p["tgt"].split()) and sum(a != b for a, b in zip(p["src"].split(), p["tgt"].split())) == 1]
rng.shuffle(B); B = B[:1500]
def run(items):
    rows = []
    for s, tgt in items:
        r = AC.process("translate", s); rows.append((s, tgt, r["output"], r["method"], r.get("confidence", 0), F.process("translate", s)["method"]))
    return rows
V["policy"]["w_tau_known"] = 0.0; json.dump(V, open(vp, "w", encoding="utf-8"), ensure_ascii=False)
E = run([(s, s) for s in en]); C = run([(p["src"], p["tgt"]) for p in B]); res = []
M = AC.mt(None); EW = set(M.v.get("en_words") or []); XX = set(M.v.get("xx_words") or [])
def known(s):
    return any(len(w) >= 3 and AC._WORD.match(w) and w not in EW and w in XX for w in AC._toks(s))
nC = sum(1 for r in C if r[5] == "already-english")
for tk in (0.4, 0.5, 0.6, 0.7, 0.8):
    for tau in (0.7, 0.8, 0.85):
        acc = lambda s, m, c, fm: fm == "already-english" and m == "neural" and c >= (tk if known(s) else tau)
        harm = [(s, o) for s, t, o, m, c, fm in E if acc(s, m, c, fm) and clean(o) != clean(s)]
        fx = [(s, t, o) for s, t, o, m, c, fm in C if acc(s, m, c, fm)]
        res.append({"w_tau_known": tk, "w_tau": tau, "english_changed": len(harm), "english_n": len(E), "foreign_flagged_english": nC, "caught": len(fx), "caught_right": sum(clean(o) == clean(t) for _, t, o in fx), "ex": harm[:30]})
        print({k: v for k, v in res[-1].items() if k != "ex"}, flush=True)
mx = float(os.environ.get("MAX_HARM", 0.003)) * len(E); ok = [r for r in res if r["english_changed"] <= mx]; best = max(ok, key=lambda r: (r["caught_right"], -r["english_changed"])) if ok else res[-1]
V["policy"]["w_tau"] = best["w_tau"]; V["policy"]["w_tau_known"] = best["w_tau_known"]; json.dump(V, open(vp, "w", encoding="utf-8"), ensure_ascii=False)
json.dump({"model": model, "chosen": best, "all": res}, open(os.path.join(W, f"wordcheck_{model}.json"), "w"), ensure_ascii=False, indent=1)
print("chosen", {k: v for k, v in best.items() if k != "ex"}); print(best["ex"])
