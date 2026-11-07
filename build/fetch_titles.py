"""Collect clean English product titles for the student model (build time).

Sources (Hugging Face, via the datasets-server rows API, titles only, no images):
  - Shopify/product-catalogue (Apache-2.0): real titles; keep the ones the strict English check accepts,
    2-8 words, no long codes.
  - Qdrant/hm_ecommerce_products (CC-BY-4.0): composed as "<gender> <colour> <product type>",
    e.g. "women's black vest top".
Output: build/work/titles_en.json
"""
import json, os, random, re, sys, time, urllib.request, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime"))
import encheck
OUT = os.path.join(HERE, "work"); os.makedirs(OUT, exist_ok=True)


def rows(dataset, offset, length=100, split="train", config="default"):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception as e:
            time.sleep(2 ** attempt)
    return []


def clean_title(t):
    t = re.sub(r"[™®©]", "", t or "")
    t = re.sub(r"\s*[|\-–—:]\s*.*$", "", t) if len(t.split()) > 8 else t   # drop long tails after separators
    t = re.sub(r"\s+", " ", t).strip(" -|,")
    return t


def main():
    random.seed(0)
    shop = []
    offsets = random.sample(range(0, 300000, 100), 220)
    for i, off in enumerate(offsets):
        for r in rows("Shopify/product-catalogue", off):
            t = clean_title(r.get("product_title", ""))
            w = t.split()
            if not (2 <= len(w) <= 8) or len(t) > 60:
                continue
            if sum(ch.isdigit() for ch in t) > 3 or re.search(r"[A-Z0-9]{5,}", t):
                continue
            if encheck.is_english(t, "strict")["english"]:
                shop.append(t)
        if i % 40 == 0:
            print("shopify batches", i, "kept", len(shop), flush=True)
    shop = sorted(set(shop))
    hm = set()
    gender = {"Ladieswear": "women's", "Menswear": "men's", "Baby/Children": "children's", "Divided": "", "Sport": ""}
    for off in random.sample(range(0, 100000, 100), 60):
        for r in rows("Qdrant/hm_ecommerce_products", off):
            g = gender.get(r.get("index_group_name", ""), "")
            c = (r.get("colour_group_name") or "").lower()
            p = (r.get("product_type_name") or "").lower()
            if c and p and "unknown" not in c and "undefined" not in c:
                hm.add(" ".join(x for x in (g, c, p) if x))
    hm = sorted(hm)
    json.dump({"shopify": shop, "hm": hm}, open(os.path.join(OUT, "titles_en.json"), "w"), ensure_ascii=False, indent=0)
    print("shopify titles", len(shop), "| hm titles", len(hm))
    print("examples:", random.sample(shop, 8), random.sample(hm, 6))


if __name__ == "__main__":
    main()
