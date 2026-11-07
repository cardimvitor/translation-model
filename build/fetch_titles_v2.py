"""More English product titles for the v4 students, from Amazon ESCI (tasksource/esci, Apache-2.0), US locale.

Titles are shortened to label-like phrases: cut at separators, drop model codes, optionally drop the brand
(kept in ~30% so the student learns to copy brand names through). 2-8 words, strict English check.
Input: build/work/dl_esci/*.parquet (downloaded); output: build/work/titles_esci.json
"""
import glob, json, os, random, re, sys
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime"))
import encheck

CUT = re.compile(r"\s*(?:\s-\s|\s–\s|\s—\s|,|\||\(|\[|:|;| for (?:men|women|kids|boys|girls)\b.*$)")
CODE = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9\-/.]{4,}$|^[A-Z0-9\-]{5,}$|^\d{3,}[\d\-]*$")


def shorten(title, brand, rng):
    t = re.sub(r"[™®©]", "", title or "").strip()
    t = CUT.split(t, maxsplit=1)[0]
    b = (brand or "").strip()
    if b and b.lower() != "none" and t.lower().startswith(b.lower() + " ") and rng.random() > 0.3:
        t = t[len(b):].strip()
    w = [x for x in t.split() if not CODE.match(x)]
    w = [x for x in w if x not in ("&", "-", "/")] if len(w) > 8 else w
    t = " ".join(w).strip(" -/&")
    return t


def main():
    rng = random.Random(0); seen = set(); out = []
    for f in sorted(glob.glob(os.path.join(HERE, "work", "dl_esci", "*.parquet"))):
        tb = pq.read_table(f, columns=["product_locale", "product_title", "product_brand"]).to_pydict()
        for loc, title, brand in zip(tb["product_locale"], tb["product_title"], tb["product_brand"]):
            if loc != "us" or title in seen:
                continue
            seen.add(title)
            t = shorten(title, brand, rng); w = t.split()
            if not (2 <= len(w) <= 8) or len(t) > 60 or sum(c.isdigit() for c in t) > 4:
                continue
            if not encheck.is_english(t, "strict")["english"]:
                continue
            out.append(t)
        print(f, "products", len(seen), "kept", len(out), flush=True)
    low = {}
    for t in out:
        low.setdefault(t.lower(), t)
    out = sorted(low.values()); rng.shuffle(out)
    json.dump(out, open(os.path.join(HERE, "work", "titles_esci.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("distinct titles", len(out)); print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
