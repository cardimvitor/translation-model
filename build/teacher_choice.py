"""Which teacher translates each language back into English best? Measured on held-out titles (bt_v2.is_dev),
never on the gold set. For each language: 60 sources made by NLLB and 60 made by OPUS-MT are translated back to
English by both teachers and scored with chrF against the original English title.
Output: build/work/teacher_choice.json {lang: {"nllb": chrF, "opus": chrF, "best": "nllb"|"opus"}}"""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work"); sys.path.insert(0, HERE)
from bt_v2 import is_dev
from analyze_student import clean, chrf
import opus_gen as O
from nllb_gen import NLLB


def rows(name):
    return [json.loads(l) for l in open(os.path.join(W, name), encoding="utf-8")]


def samples(n=60):
    """{lang: rows}: n held-out sources made by NLLB and n made by OPUS-MT (fixed seed, so every teacher sees the same)."""
    rng = random.Random(7); by = {}
    for name in ("bt_nllb_v2.jsonl", "bt_opus_v2.jsonl"):
        for r in rows(name):
            if r.get("src") and is_dev(r["en"]) and len(r["en"]) > 8:
                by.setdefault(r["lang"], {}).setdefault(name[3:7], []).append(r)
    out = {}
    for lang, d in sorted(by.items()):
        sample = []
        for k in sorted(d):
            d[k].sort(key=lambda r: (r["en"], r["src"])); rng.shuffle(d[k]); sample += d[k][:n]
        out[lang] = sample
    return out


def three(path):
    """All three teachers on the same held-out sample -> teacher_choice3.json (used for the H and ALL3 models).
    A teacher translating its own wording back has an unfair advantage, so each teacher is scored only on sources
    the others wrote: OPUS-MT on NLLB-made, NLLB on OPUS-made, Hy-MT2 on both (it wrote none of them)."""
    from hy_gen import Hy
    import torch; torch.set_num_threads(4)
    S = samples(); C = {}; outp = os.path.join(W, "teacher_choice3.json")
    if os.path.exists(outp):
        C = json.load(open(outp))
    nl = h = None
    for lang, sample in S.items():
        if lang in C:
            continue
        nl = nl or NLLB(threads=4); h = h or Hy(path); srcs = [r["src"] for r in sample]
        made = ["nllb" if (r.get("back") is None) else "opus" for r in sample]
        hyp = {"nllb": [x[0] if isinstance(x, (list, tuple)) else x for x in nl.translate(srcs, lang, "en")],
               "opus": [x[0] for x in O.rev(srcs, lang)], "hy": [x[0] for x in h.translate(srcs, lang, "en")]}
        sc = {"n": len(sample)}
        for t, hs in hyp.items():
            for m in ("nllb", "opus"):
                xs = [chrf(clean(x), clean(r["en"])) for x, r, mm in zip(hs, sample, made) if mm == m]
                sc[f"{t}_on_{m}"] = round(sum(xs) / max(1, len(xs)), 2)
        sc["nllb"] = sc["nllb_on_opus"]; sc["opus"] = sc["opus_on_nllb"]
        sc["hy"] = round((sc["hy_on_nllb"] + sc["hy_on_opus"]) / 2, 2)
        # pairwise, each on sources neither of the two wrote when possible
        win = {"opus": sc["opus_on_nllb"] >= sc["hy_on_nllb"], "nllb": sc["nllb_on_opus"] >= sc["hy_on_opus"]}
        two = json.load(open(os.path.join(W, "teacher_choice.json"))).get(lang, {}).get("best", "opus")
        sc["best_opus_hy"] = "opus" if win["opus"] else "hy"
        sc["best"] = two if win[two] else "hy"
        C[lang] = sc; print(lang, sc, flush=True); json.dump(C, open(outp, "w"), indent=1)


def main(n=60):
    nl = NLLB(threads=4); out = {}
    for lang, sample in samples(n).items():
        srcs = [r["src"] for r in sample]
        hyp = {"nllb": [x[0] if isinstance(x, (list, tuple)) else x for x in nl.translate(srcs, lang, "en")],
               "opus": [x[0] for x in O.rev(srcs, lang)]}
        sc = {t: round(sum(chrf(clean(h), clean(r["en"])) for h, r in zip(hs, sample)) / len(sample), 2) for t, hs in hyp.items()}
        sc["best"] = max(("nllb", "opus"), key=lambda t: sc[t]); sc["n"] = len(sample); out[lang] = sc
        print(lang, sc, flush=True)
    json.dump(out, open(os.path.join(W, "teacher_choice.json"), "w"), indent=1)


if __name__ == "__main__":
    three(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "three" else main()
