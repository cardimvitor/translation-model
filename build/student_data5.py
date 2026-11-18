"""v5 training pairs: original English targets plus the teachers' own English answers (sequence-level distillation).

  B2    OPUS-MT sources; targets: original title + OPUS-MT answers                     (commercial OK)
  C2    OPUS-MT + NLLB sources; + answers of both                                        (non-commercial)
  H     OPUS-MT sources; + answers of OPUS-MT and Hy-MT2                                 (commercial OK)
  ALL3  OPUS-MT + NLLB sources; + answers of all three                                   (non-commercial)
  HY    Hy-MT2 writes the sources and answers                                          (commercial OK)

Every available teacher answer is used if it passes kd_ok (English, close to the real title). Picking one teacher
per language from held-out data was dropped: those held-out sources were written by OPUS-MT / NLLB, which favors
the teacher that wrote them. Dev pairs keep the original title as target.
Every variant also gets the glossary, place names and accent-free copies (same as v4).
Output: build/work/pairs_<V>_{train,dev}.jsonl"""
import json, os, random, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runtime")); sys.path.insert(0, HERE)
import normalizer as F, encheck
from student_data import student_input, good_bt
from student_data2 import round_trip_ok, gloss_pairs, gaz_pairs, LATIN_LANGS, read
from bt_v2 import is_dev
from analyze_student import clean, chrf

SPEC = {"MAX": (["opus", "hy", "abo"], ["opus", "hy"]),     # v5 final: commercial-friendly teachers + real ABO names
        "B2": (["opus"], ["opus"]), "C2": (["opus", "nllb"], ["opus", "nllb"]), "H": (["opus"], ["opus", "hy"]),
        "ALL3": (["opus", "nllb"], ["opus", "nllb", "hy"]), "HY": (["hy"], ["hy"])}


def kd_ok(src, back, en):
    """A teacher answer is kept when it is English, not a copy of the source, and close to the real title
    (chrF >= KD_MIN, default 45): the student learns the teacher's wording without its clear mistakes."""
    if not back or student_input(back) == src or not (0.3 <= len(back) / max(1, len(src)) <= 3.5):
        return False
    if chrf(clean(back), clean(en)) < float(os.environ.get("KD_MIN", 45)):
        return False
    return encheck.is_english(back, "relaxed")["english"]


def best_teacher(teachers):
    """{lang: teacher} among `teachers`, by held-out chrF (same sample for every teacher in that file)."""
    if len(teachers) == 1:
        return lambda l: teachers[0]
    if "hy" in teachers:        # teacher_choice3.json: each teacher scored on sources the others wrote
        C = json.load(open(os.path.join(W, "teacher_choice3.json")))
        key = "best" if len(teachers) == 3 else "best_opus_hy"
        pick = {l: v[key] for l, v in C.items()}
    else:
        C = json.load(open(os.path.join(W, "teacher_choice.json")))
        pick = {l: max(teachers, key=lambda t: v.get(t, -1)) for l, v in C.items()}
    print("teacher per language:", pick, flush=True)
    return lambda l: pick.get(l, teachers[0])


def bt_and_kd(variant):
    srcs, teachers = SPEC[variant]
    kd = {}
    for r in read("kd_v5.jsonl"):
        kd[(r["src"], r["lang"], r["teacher"])] = r["back"]
    rows = []
    if "opus" in srcs:
        rows += [dict(r, made="opus") for r in read("bt_opus_v2.jsonl")]
    if "nllb" in srcs:
        rows += [dict(r, made="nllb") for r in read("bt_nllb_v2.jsonl") + read("bt.jsonl")]
    if "hy" in srcs:
        rows += [dict(r, made="hy") for r in read("bt_hy_v5.jsonl")]
    if "abo" in srcs:                               # OPUS-MT translations of the ABO English names
        rows += [dict(r, made="opus") for r in read("bt_opus_abo.jsonl")]
    out, seen, stat = [], set(), Counter()
    for r in rows:
        stat["read_" + r["made"]] += 1
        if not good_bt(r) or (r["made"] in ("opus", "hy") and not round_trip_ok(r)):
            continue
        src = student_input(r["src"]); en = r["en"].lower(); split = "dev" if is_dev(r["en"]) else "train"
        if (src, en) not in seen:
            seen.add((src, en)); stat["bt_" + r["made"]] += 1
            out.append((split, "bt", r["lang"], src, en))
        if split == "dev":
            continue
        cands = []                                  # every teacher's answer for this source
        for t in teachers:
            back = r.get("back") if t == r["made"] and t in ("opus", "hy") else kd.get((r["src"], r["lang"], t))
            if back:
                cands.append((t, back))
        for t, back in cands:
            if kd_ok(src, back, en) and (src, back.lower()) not in seen and back.lower() != en:
                seen.add((src, back.lower())); stat["kd_" + t] += 1
                out.append(("train", "kd", r["lang"], src, back.lower()))
    if "abo" in srcs:                               # real marketplace names (ABO): human-written source and English
        for r in read("abo_train.jsonl"):
            src = student_input(r["src"]); en = r["en"].lower()
            if (src, en) not in seen and len(src) <= 96 and len(en) <= 96:
                seen.add((src, en)); stat["real_abo"] += 1
                out.append(("dev" if is_dev(r["en"]) else "train", "real", r["lang"], src, en))
    print(variant, dict(stat), flush=True)
    return out


def main():
    for v in sys.argv[1:] or list(SPEC):
        rng = random.Random(3)
        pairs = bt_and_kd(v) + gloss_pairs() + gaz_pairs(rng)
        pairs += [("train", "noise", l, F.fold_latin(s), t) for sp, k, l, s, t in pairs
                  if sp == "train" and k in ("bt", "gloss") and l in LATIN_LANGS and F.fold_latin(s) != s and rng.random() < 0.25]
        rng.shuffle(pairs)
        for split in ("train", "dev"):
            with open(os.path.join(W, f"pairs_{v}_{split}.jsonl"), "w", encoding="utf-8") as f:
                for p in pairs:
                    if p[0] == split:
                        f.write(json.dumps({"kind": p[1], "lang": p[2], "src": p[3], "tgt": p[4]}, ensure_ascii=False) + "\n")
        print(v, "counts", dict(Counter((p[0], p[1]) for p in pairs)), flush=True)


if __name__ == "__main__":
    main()
