"""Training data from Tencent Hy-MT2-1.8B (Apache-2.0) via llama.cpp. Resumable, written per item.

  python hy_data.py kd   <gguf>   Hy-MT2 English answers for OPUS-made training sources (balanced per language,
                                  HY_KD_CAP rows, default 30000) -> appends {"teacher": "hy"} rows to kd_v5.jsonl
  python hy_data.py pure <gguf>   Hy-MT2 writes the foreign source (en -> xx) and translates it back (xx -> en) for
                                  HY_PURE_CAP titles (default 12000, one language each; held-out titles included so the
                                  pure model has its own dev set) -> build/work/bt_hy_v5.jsonl"""
import json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work"); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runtime"))
from bt_v2 import is_dev, titles, LANGS
from student_data import good_bt
from hy_gen import Hy


def read(name):
    p = os.path.join(W, name)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def kd_jobs():
    cap = int(os.environ.get("HY_KD_CAP", 30000)); by = {}
    for r in read("bt_opus_v2.jsonl"):
        if r.get("src") and not is_dev(r["en"]) and good_bt(r):
            by.setdefault(r["lang"], {})[r["src"]] = 1
    rng = random.Random(13); per = cap // len(by); out = []
    for l in sorted(by):
        ss = sorted(by[l]); rng.shuffle(ss); out += [(s, l) for s in ss[:per]]
    return out


def main(mode, path):
    h = Hy(path); t0 = time.time(); k = 0
    if mode == "kd":
        done = {(r["src"], r["lang"]) for r in read("kd_v5.jsonl") if r["teacher"] == "hy"}
        J = [j for j in kd_jobs() if j not in done]; print("hy kd jobs left", len(J), flush=True)
        with open(os.path.join(W, "kd_v5.jsonl"), "a", encoding="utf-8") as f:
            for s, l in J:
                e, c = h.one(s, "en")
                f.write(json.dumps({"src": s, "lang": l, "teacher": "hy", "back": e, "conf": c}, ensure_ascii=False) + "\n"); f.flush()
                k += 1
                if k % 200 == 0:
                    print(f"{k}/{len(J)} {(time.time() - t0) / k:.2f} s/item", flush=True)
    else:
        cap = int(os.environ.get("HY_PURE_CAP", 12000)); rng = random.Random(17)
        T = titles(); rng.shuffle(T); J = [(t, rng.choice(LANGS)) for t in T[:cap]]
        done = {(r["en"], r["lang"]) for r in read("bt_hy_v5.jsonl")}
        J = [j for j in J if j not in done]; print("hy pure jobs left", len(J), flush=True)
        with open(os.path.join(W, "bt_hy_v5.jsonl"), "a", encoding="utf-8") as f:
            for en, l in J:
                s, c = h.one(en, l); b, bc = h.one(s, "en") if s else ("", 0.0)
                f.write(json.dumps({"en": en, "lang": l, "src": s, "conf": c, "back": b, "back_conf": bc}, ensure_ascii=False) + "\n"); f.flush()
                k += 1
                if k % 100 == 0:
                    print(f"{k}/{len(J)} {(time.time() - t0) / k:.2f} s/item", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
