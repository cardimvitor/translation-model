"""Real multilingual product names from Amazon Berkeley Objects (ABO, CC BY 4.0, attribution required).
Each listing can carry the item name in several marketplace languages; those written next to an English name are
real (not machine) translations.

  python fetch_abo.py   (parquet files in build/work/dl_abo/, from huggingface/amazon-berkeley-objects listings/)
Writes build/work/abo_pairs.jsonl {"lang", "src", "en"} for our 16 languages and build/work/abo_titles.json
(English names, for back-translation). Names are cut at the first " - ", "|", "," or "(" when both sides have one,
and pairs longer than 96 characters or with very different lengths are dropped."""
import glob, json, os, re, sys
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work")
LANGS = {"zh", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th"}
CUT = re.compile(r"\s[-|–]\s|\s*\|\s*|,\s|\s\(|（|，")
MAXLEN = 96


def clean(s):
    s = re.sub(r"^\s*\[[^\]]{0,30}\]\s*", "", s or "")          # "[Find] ...", "[Amazon Collection] ..."
    s = re.sub(r"[™®©]", "", s)
    return re.sub(r"\s+", " ", s).strip(" -/|,")


def cut(s):
    return CUT.split(s, maxsplit=1)[0].strip(" -/|,")


def main():
    pairs, titles, seen = [], [], set()
    for f in sorted(glob.glob(os.path.join(W, "dl_abo", "*.parquet"))):
        for raw in pq.read_table(f, columns=["raw_listing_json"]).column(0).to_pylist():
            names = json.loads(raw).get("item_name", [])
            ens = [clean(x["value"]) for x in names if x["language_tag"].startswith("en_")]
            if not ens:
                continue
            en = ens[0]
            if en and en.lower() not in seen:
                seen.add(en.lower()); titles.append(cut(en) if len(en) > MAXLEN else en)
            for x in names:
                l = x["language_tag"][:2]
                if l not in LANGS or x["language_tag"] == "zh_TW":       # zh = simplified only (our zh)
                    continue
                s, e = clean(x["value"]), en
                if s.lower() == e.lower():
                    continue
                if len(s) > MAXLEN or len(e) > MAXLEN:
                    s, e = cut(s), cut(e)
                if not s or not e or len(s) > MAXLEN or len(e) > MAXLEN:
                    continue
                r = len(s) / max(1, len(e))
                if (l in ("zh", "ja", "ko") and not 0.15 <= r <= 1.5) or (l not in ("zh", "ja", "ko") and not 0.4 <= r <= 2.5):
                    continue
                pairs.append({"lang": l, "src": s, "en": e})
    uniq = {(p["lang"], p["src"].lower(), p["en"].lower()): p for p in pairs}
    with open(os.path.join(W, "abo_pairs.jsonl"), "w", encoding="utf-8") as f:
        for p in uniq.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    titles = [t for t in titles if 3 <= len(t) <= MAXLEN]
    json.dump(titles, open(os.path.join(W, "abo_titles.json"), "w", encoding="utf-8"), ensure_ascii=False)
    from collections import Counter
    print(len(uniq), "pairs", dict(Counter(p["lang"] for p in uniq.values()).most_common()), "|", len(titles), "English titles")


if __name__ == "__main__":
    main()
