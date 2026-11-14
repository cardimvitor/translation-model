"""Turn a list of English product names into training pairs, when you have no translations.
Hy-MT2-1.8B (Apache-2.0, models/hy-mt2/Hy-MT2-1.8B-Q4_K_M.gguf) writes each name in the languages you choose.
  python build/make_pairs.py names.txt --langs ko,th,de --out my_pairs.jsonl      (names.txt: one English name per line)
About 0.7 s per name and language on 4 CPU cores. Resumable. Output lines: {"src", "tgt", "lang"} for post_train.sh."""
import argparse, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path[:0] = [HERE, os.path.join(ROOT, "runtime")]
SCRIPT = {"ko": r"[가-힣]", "th": r"[฀-๿]", "ja": r"[぀-ヿ一-鿿]", "zh": r"[一-鿿]", "ru": r"[Ѐ-ӿ]",
          "ar": r"[؀-ۿ]", "hi": r"[ऀ-ॿ]"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names"); ap.add_argument("--langs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default=os.path.join(ROOT, "models", "hy-mt2", "Hy-MT2-1.8B-Q4_K_M.gguf"))
    a = ap.parse_args(); langs = a.langs.split(",")
    names = [l.strip() for l in open(a.names, encoding="utf-8") if l.strip()]
    done = {(json.loads(l)["tgt"], json.loads(l)["lang"]) for l in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
    from hy_gen import Hy
    h = Hy(a.gguf); kept = skipped = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for en in names:
            for l in langs:
                if (en.lower(), l) in done:
                    continue
                s = h.one(en, l, max_new=64)[0]
                ok = s and s.lower() != en.lower() and 0.2 <= len(s) / max(1, len(en)) <= 3.5 and (l not in SCRIPT or re.search(SCRIPT[l], s))
                if ok:
                    f.write(json.dumps({"src": s, "tgt": en.lower(), "lang": l}, ensure_ascii=False) + "\n"); f.flush(); kept += 1
                else:
                    skipped += 1
    print("pairs written", kept, "| skipped (empty, copy, wrong script or odd length)", skipped, "->", a.out)


if __name__ == "__main__":
    main()
