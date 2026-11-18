"""Tune decision rule v3 (runtime/accurate.py process_v3) for a strong neural model and store it in the model's
vocab.json ("policy"). Tuning data, none of it a test set: the hand-written dev labels (test/dev_labels.json),
held-out real ABO names and held-out back-translated titles from the MAX dev split. The three parts count equally.
  NEURAL_DIR=student_V6 python build/tune_routing3.py   -> build/work/routing3_<model>.json"""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
import normalizer as F, accurate as AC, student
from analyze_student import clean, chrf


def prepare(srcs):
    S = AC.mt(); out = []
    for s in srcs:
        f = F.process("translate", s)
        if f["method"] in ("gazetteer", "already-english") or not f["output"]:
            out.append({"fast": f["output"], "always": True}); continue
        langs = [l for l, _ in f["language"]]
        addr = AC.is_address_v2(s, langs, f); nw = len(s.split()); cov = f.get("coverage", 0)
        if addr and not (nw > 8 and cov < 0.6):
            out.append({"fast": f["output"], "always": True}); continue
        o, c = S.translate(s)
        out.append({"fast": f["output"], "always": False, "long_addr": addr, "gaz": "gazetteer" in f["method"] and nw <= 8, "cov": cov >= 1, "o": o, "c": c,
                    "ok": bool(o) and student.acceptable(o, 1.0, False), "eng": bool(o) and student.acceptable(o, 1.0),
                    "known": bool(o) and AC.keeps_known(o, f["output"])})
    return out


def decide(x, keep_cov, need_known, tau, need_eng=1, addr_mw=0, keep_gaz=0):
    if x["always"] or (keep_cov and x["cov"]) or (x["long_addr"] and not addr_mw) or (keep_gaz and x["gaz"]):
        return x["fast"]
    return x["o"] if ((x["eng"] if need_eng else x["ok"]) and x["c"] >= tau and (x["known"] or not need_known)) else x["fast"]


def score(outs, refs):
    ex = sum(clean(o) in [clean(r) for r in R] for o, R in zip(outs, refs)) / len(refs)
    ch = sum(max(chrf(clean(o), clean(r)) for r in R) for o, R in zip(outs, refs)) / len(refs)
    return {"exact": round(ex, 4), "chrF": round(ch, 2)}


def main():
    name = os.environ.get("NEURAL_DIR", "student_V6"); md = os.path.join(ROOT, "models", name)
    v = json.load(open(os.path.join(md, "vocab.json"), encoding="utf-8")); oldp = v.pop("policy", None) or {}
    keep = {k: oldp[k] for k in ("beams", "lp", "beam_max_src") if k in oldp}            # decoding settings (build/tune_beams.py) stay
    if keep:
        v["policy"] = dict(keep)
    json.dump(v, open(os.path.join(md, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    dev = [json.loads(l) for l in open(os.path.join(HERE, "work", "pairs_MAX_dev.jsonl"), encoding="utf-8")]
    rng = random.Random(9); real = [p for p in dev if p["kind"] == "real"]; bt = [p for p in dev if p["kind"] == "bt"]
    rng.shuffle(real); rng.shuffle(bt); gz = [p for p in dev if p["kind"] == "gaz"]; rng.shuffle(gz)
    DL = json.load(open(os.path.join(ROOT, "test", "dev_labels.json"), encoding="utf-8"))["items"]
    parts = {"labels": ([s for g in DL for s in g["src"].values()], [g["en"] for g in DL for _ in g["src"]]),
             "real": ([p["src"] for p in real[:500]], [[p["tgt"]] for p in real[:500]]),
             "places": ([p["src"] for p in gz[:400]], [[p["tgt"]] for p in gz[:400]]),
             "titles": ([p["src"] for p in bt[:1500]], [[p["tgt"]] for p in bt[:1500]])}
    P = {k: (prepare(s), r) for k, (s, r) in parts.items() if s}
    grid = {}
    for kc in (1, 0):
        for nk in (1, 0):
            for tau in (0.0, 0.1, 0.2, 0.3, 0.4):
                for ne in (1, 0):
                    for mw in (0, 8):
                      for kg in (0, 1):
                        sc = {k: score([decide(x, kc, nk, tau, ne, mw, kg) for x in X], R) for k, (X, R) in P.items()}
                        grid[(kc, nk, tau, ne, mw, kg)] = (round(sum(s["chrF"] for s in sc.values()) / len(sc), 3), sc)
                        print("keep_cov", kc, "need_known", nk, "tau", tau, "need_english", ne, "addr_max_words", mw, "keep_gaz", kg, grid[(kc, nk, tau, ne, mw, kg)][0], {k: s["chrF"] for k, s in sc.items()}, flush=True)
    # Best setting by tuning score that also keeps every tuned address/label case right (on a tie, the cautious
    # setting: places stay with Fast). A setting that scores higher but breaks a tuned case is never chosen.
    CASES = json.load(open(os.path.join(ROOT, "test", "cases.json"), encoding="utf-8")); S = AC.mt(); best = pol = None; top = 0
    def cases_ok(p):
        S.v["policy"] = dict(p, **keep)
        return sum(AC.process(c["op"], c["in"])["output"].casefold() == c["want"].casefold() for c in CASES)
    for k in sorted(grid, key=lambda k: (-grid[k][0], -k[5])):
        p = {"keep_cov": bool(k[0]), "need_known": bool(k[1]), "tau": k[2], "need_english": bool(k[3]), "addr_max_words": k[4], "addr_long_cov": 0.6, "keep_gaz": bool(k[5])}
        n = cases_ok(p)
        if n > top:
            top, best, pol = n, k, p
        if n == len(CASES):
            break
    print("tuned cases with the chosen setting:", top, "of", len(CASES), flush=True)
    print("chosen", pol, grid[best], flush=True)
    fast_only = {k: score([x["fast"] for x in X], R) for k, (X, R) in P.items()}
    neural_only = {k: score([x.get("o") or x["fast"] for x in X], R) for k, (X, R) in P.items()}
    pol.update(keep); v["policy"] = pol; json.dump(v, open(os.path.join(md, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"model": name, "policy": pol, "dev": grid[best][1], "dev_mean": grid[best][0], "dev_fast_only": fast_only, "dev_neural_where_allowed": neural_only,
               "grid": [{"keep_cov": k[0], "need_known": k[1], "tau": k[2], "need_english": k[3], "addr_max_words": k[4], "keep_gaz": k[5], "mean": g[0]} for k, g in sorted(grid.items())]},
              open(os.path.join(HERE, "work", f"routing3_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
