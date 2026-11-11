"""v5 sequence-level distillation targets: translate the synthetic foreign sources back into English with the
teacher that is best for that language (build/work/teacher_choice.json, chosen on held-out titles).

  python kd_v5.py opus   OPUS-MT xx->en for NLLB-made sources in OPUS-best languages (OPUS-made rows already
                         carry their OPUS back-translation in "back")
  python kd_v5.py nllb   NLLB xx->en for all sources in NLLB-best languages (at most NLLB_CAP rows, default 50000)

Training rows only (dev titles are skipped). Resumable; appends to build/work/kd_v5.jsonl:
  {"src", "lang", "teacher", "back", "conf"}"""
import json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work"); sys.path.insert(0, HERE)
from bt_v2 import is_dev
B = 32


def read(name):
    p = os.path.join(W, name)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def jobs(teacher):
    best = {l: v["best"] for l, v in json.load(open(os.path.join(W, "teacher_choice.json"))).items()}
    nllb_rows = read("bt_nllb_v2.jsonl") + read("bt.jsonl"); opus_rows = read("bt_opus_v2.jsonl")
    if teacher == "opus":
        rows = [r for r in nllb_rows if best.get(r["lang"]) == "opus"]
    else:
        rows = [r for r in nllb_rows + opus_rows if best.get(r["lang"]) == "nllb"]
    seen, out = set(), []
    for r in rows:
        k = (r["src"], r["lang"])
        if r.get("src") and not is_dev(r["en"]) and k not in seen:
            seen.add(k); out.append(k)
    random.Random(11).shuffle(out)
    if teacher == "nllb":
        out = out[: int(os.environ.get("NLLB_CAP", 50000))]
    return out


def main(teacher):
    path = os.path.join(W, "kd_v5.jsonl")
    done = {(r["src"], r["lang"]) for r in read("kd_v5.jsonl") if r["teacher"] == teacher}
    J = [j for j in jobs(teacher) if j not in done]
    print(teacher, "jobs left", len(J), flush=True)
    by = {}
    for s, l in J:
        by.setdefault(l, []).append(s)
    if teacher == "nllb":
        from nllb_gen import NLLB
        n = NLLB(threads=int(os.environ.get("THREADS", 4)))
        rev = lambda ss, l: [x if isinstance(x, (list, tuple)) else (x, 0.0) for x in n.translate(ss, l, "en")]
    else:
        import torch, opus_gen as O
        torch.set_num_threads(int(os.environ.get("THREADS", 4))); rev = O.rev
    t0 = time.time(); k = 0
    with open(path, "a", encoding="utf-8") as f:
        for l, ss in by.items():
            ss.sort(key=len)
            for i in range(0, len(ss), B):
                batch = ss[i:i + B]; res = rev(batch, l)
                for s, (e, c) in zip(batch, res):
                    f.write(json.dumps({"src": s, "lang": l, "teacher": teacher, "back": e, "conf": round(float(c), 3)}, ensure_ascii=False) + "\n")
                f.flush(); k += len(batch)
                if (i // B) % 20 == 0:
                    print(f"{k}/{len(J)} done, {(time.time() - t0) / k * 1000:.0f} ms/item, lang {l}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
