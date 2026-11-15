"""Place-name dictionary, version 2: any-language name -> English name, for the whole world.

Sources (build time only, in build/work/geonames/, from download.geonames.org/export/dump):
  cities500.txt           every populated place with 500+ inhabitants (about 200,000)
  admin1CodesASCII.txt    states, provinces and regions
  countryInfo.txt         countries (their GeoNames ids)
  alternateNamesV2.txt    names per language, with "preferred" / "historic" / "colloquial" flags
  CLDR (via Babel)        country names in every locale
English name of a place: the preferred English alternate name if GeoNames has one, else its main name.
Collisions keep the bigger place; countries win over everything. A small town (under 15,000 people) whose one-word
Latin name is also a word seen in product text is left out, so product labels are not mistaken for towns.
Output: data/gazetteer/<Script>.json {"k": {name: [english, kind, cc]}} + index.json
  MINPOP=1000 python build/build_gazetteer2.py"""
import json, os, re, sys, unicodedata
from babel import Locale, localedata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); G = os.path.join(ROOT, "build", "work", "geonames")
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, os.path.join(ROOT, "build"))
from normalizer import norm_key, fold_latin, script_of, is_alpha
from build_gazetteer import EXONYMS, NAME_OVERRIDES
KEEP = re.compile(r"^[\u0000-ɏḀ-ỿͰ-Ͽἀ-῿Ѐ-ӿ؀-ۿऀ-ॿ฀-๿぀-ヿㇰ-ㇿ一-鿿㐀-䶿가-힯ᄀ-ᇿ\s\-'’.()]+$")
LATIN = re.compile(r"^[\u0000-ɏḀ-ỿ\s\-'’.()]+$")
LANGS = {"zh", "zh-CN", "zh-TW", "zh-Hans", "zh-Hant", "zh-HK", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th", "en", "el", "uk", ""}
MINPOP = int(os.environ.get("MINPOP", 500)); SMALL = 15000
EX2 = dict(EXONYMS, **{"Irákleion": "Heraklion", "Athína": "Athens", "Thessaloníki": "Thessaloniki", "Kyiv": "Kyiv", "Bengaluru": "Bengaluru"})


def main():
    # ---- ids we care about
    city = {}
    for l in open(os.path.join(G, "cities500.txt"), encoding="utf-8"):
        p = l.rstrip("\n").split("\t"); pop = int(p[14] or 0)
        if pop >= MINPOP or p[7] in ("PPLC", "PPLA"):
            city[p[0]] = (p[1], p[8], pop, p[3].split(",") if p[3] else [])
    region = {}
    for l in open(os.path.join(G, "admin1CodesASCII.txt"), encoding="utf-8"):
        code, name, asc, gid = l.rstrip("\n").split("\t"); region[gid] = (name, code.split(".")[0])
    country = {}
    for l in open(os.path.join(G, "countryInfo.txt"), encoding="utf-8"):
        if not l.startswith("#"):
            p = l.rstrip("\n").split("\t")
            if len(p) > 16 and p[16]:
                country[p[16]] = p[0]
    want = set(city) | set(region) | set(country); alts = {}; en_pref = {}; en_any = {}
    for l in open(os.path.join(G, "alternateNamesV2.txt"), encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if p[1] not in want or p[2] not in LANGS:
            continue
        p += [""] * (10 - len(p)); name = p[3]; hist = p[7] == "1" or p[6] == "1"
        if p[2] == "en" and not hist:
            if p[4] == "1":
                en_pref.setdefault(p[1], name)
            en_any.setdefault(p[1], name)
        if not hist or p[2] == "":
            alts.setdefault(p[1], []).append((name, p[2]))
    # ---- words seen in product text (block list for small towns)
    block = set()
    try:
        v = json.load(open(os.path.join(ROOT, "models", os.environ.get("WORDS_FROM", "student_V12"), "vocab.json"), encoding="utf-8")); block |= set(v.get("en_words", [])) | set(v.get("xx_words", []))
    except Exception as e:
        print("no product word lists:", e)
    GL = json.load(open(os.path.join(ROOT, "data", "glossary.json"), encoding="utf-8"))
    for lang, d in GL.items():
        if isinstance(d, dict) and not lang.startswith("_"):
            for f, e in d.items():
                block.add(norm_key(f)); block |= set(str(e).lower().split()) if isinstance(e, str) else set()
    for n in ("pairs_Q_train.jsonl", "pairs_NEW_v10.jsonl", "pairs_HQ_train.jsonl"):          # short product labels in every script
        p = os.path.join(ROOT, "build", "work", n)
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                s = json.loads(l)["src"]
                block.add(norm_key(s)); block |= {norm_key(w) for w in s.split()}
    V1 = {}                                   # the first place list: its names and spellings stay as they were
    b = os.path.join(ROOT, "build", "work", "gazetteer_v1_backup")
    for f in os.listdir(b):
        if f != "index.json":
            V1.update(json.load(open(os.path.join(b, f), encoding="utf-8"))["k"])
    entries = {}; st = {"blocked_small": 0, "blocked_word": 0}
    SHORT = lambda k: len(k) < (5 if re.match(r"^[\u0000-\u024f\u0370-\u04ff\u1e00-\u1eff\s\-'’.()]+$", k) else 2 if re.match(r"^[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+$", k) else 3)

    def add1(k, english, kind, cc, prio):
        if not k or len(k) < 2 or not any(ch.isalpha() for ch in k):
            return
        cur = entries.get(k)
        if cur is None or prio > cur[3]:
            entries[k] = (english, kind, cc, prio)

    def add(name, english, kind, cc, prio, small=False):
        k = norm_key(name)
        if kind != "country" and k not in V1 and " " not in k and prio < 500000 and (k in block or fold_latin(k) in block or (SHORT(k) and prio < 100000 and not small)):
            st["blocked_word"] += 1; return           # a new name that is also a product word, or too short to trust
        if small and k != norm_key(english):
            if (k in block or fold_latin(k) in block) or (LATIN.match(k) and len(k) < 4):
                st["blocked_small"] += 1; return
        elif small and " " not in k and (k in block or (LATIN.match(k) and len(k) < 4)):
            st["blocked_small"] += 1; return
        add1(k, english, kind, cc, prio)
        f = fold_latin(k)
        if f != k and not (small and f in block):
            add1(f, english, kind, cc, prio - 1)

    # ---- countries: CLDR in every locale, then GeoNames names in our languages
    en = Locale.parse("en").territories; codes = [c for c in en if re.fullmatch(r"[A-Z]{2}", c)]; lang_names = set()
    for loc in localedata.locale_identifiers():
        try:
            L = Locale.parse(loc); T = L.territories; lang_names.update(norm_key(x) for x in L.languages.values())
        except Exception:
            continue
        for c in codes:
            if c in T:
                add(T[c], en[c], "country", c, 10 ** 12)
    for gid, cc in country.items():
        if cc in en:
            for name, lg in alts.get(gid, []):
                if lg and KEEP.match(name) and not re.fullmatch(r"[A-Z]{2,3}", name):
                    add(name, en[cc], "country", cc, 10 ** 12 - 5)
    n_country = len(entries)
    # ---- regions
    for gid, (name, cc) in region.items():
        english = EX2.get(name, en_pref.get(gid) or name)
        english = re.sub(r"^(Canton of|State of|Province of|Region of|Departamento de|Department of) ", "", re.sub(r" (Region|Province|Prefecture|Oblast|Voivodeship|Governorate|Department|County|Krai|State)$", "", english)).replace("’", "")
        add(english, english, "region", cc, 3 * 10 ** 4); add(name, english, "region", cc, 3 * 10 ** 4 - 2)
        for a, lg in alts.get(gid, []):
            if lg and KEEP.match(a) and not re.fullmatch(r"[A-Z0-9]{1,3}", a) and norm_key(a) not in lang_names:
                add(a, english, "region", cc, 3 * 10 ** 4 - 3)
    # ---- cities and towns
    city_en = {}
    for gid, (name, cc, pop, _) in city.items():
        e = en_pref.get(gid) or (en_any.get(gid) if not name.isascii() and en_any.get(gid) else None) or name
        e = EX2.get(name, EX2.get(e, e)).replace("’", "").replace("ʼ", "").replace("‘", "")
        if cc == "KR":
            e = re.sub(r"-(si|gun|eup)$", "", e)
        old = V1.get(norm_key(name))
        city_en[gid] = old[0] if old and old[1] == "city" and old[2] == cc else e
    own = {}
    for gid, e in city_en.items():
        k = norm_key(e)
        if city[gid][2] > own.get(k, (-1,))[0]:
            own[k] = (city[gid][2], gid)
    n_alt = 0
    for gid, (name, cc, pop, raw) in city.items():
        english = city_en[gid]; small = pop < SMALL
        add(english, english, "city", cc, pop, small); add(name, english, "city", cc, pop - 1, small)
        seen = set()
        cand = alts.get(gid, []) + ([(a, "") for a in raw] if not small else [(a, "") for a in raw if not LATIN.match(a)])
        for a, lg in cand:
            if a in seen or re.fullmatch(r"[A-Z]{3}", a) or not KEEP.match(a):
                continue
            seen.add(a); k = norm_key(a)
            if k in lang_names and k != norm_key(english):
                continue
            o = own.get(k)
            if o and o[1] != gid and k != norm_key(english):          # another city's own English name
                continue
            add(a, english, "city", cc, pop - 2, small); n_alt += 1
            # the same place written with or without its administrative word
            if cc == "JP" and re.fullmatch(r"[\u4e00-\u9fff\u3040-\u30ff]{2,}", a):
                for sx in ("市", "町", "村"):
                    add(a[:-1], english, "city", cc, pop - 3, small) if a.endswith(sx) and len(a) > 2 else add(a + sx, english, "city", cc, pop - 3, small)
            elif cc in ("CN", "TW") and re.fullmatch(r"[\u4e00-\u9fff]{2,}", a):
                for sx in ("市", "县", "縣", "镇", "鎮"):
                    add(a[:-1], english, "city", cc, pop - 3, small) if a.endswith(sx) and len(a) > 2 else add(a + sx, english, "city", cc, pop - 3, small)
            elif cc == "KR" and re.fullmatch(r"[\uac00-\ud7af]{2,}", a):
                for sx in ("시", "군", "읍"):
                    add(a[:-1], english, "city", cc, pop - 3, small) if a.endswith(sx) and len(a) > 2 else add(a + sx, english, "city", cc, pop - 3, small)
            elif cc == "TH":
                for px in ("อำเภอ", "จังหวัด", "เมือง", "เทศบาลนคร", "เทศบาลเมือง"):
                    if a.startswith(px) and len(a) > len(px) + 1:
                        add(a[len(px):], english, "city", cc, pop - 3, small)
    for name, english in NAME_OVERRIDES.items():
        add(name, english, "city", "IN", 10 ** 13)
    D = os.path.join(ROOT, "data", "gazetteer"); parts = {}
    for k, (e, kind, cc, prio) in entries.items():
        a = next((ch for ch in k if is_alpha(ch)), None)
        town = kind == "city" and prio < SMALL - 5 and k not in V1
        parts.setdefault((script_of(a) if a else None) or "Other", {"k": {}, "t": {}})["t" if town else "k"][k] = [e] if town else [e, kind, cc]
    for f in os.listdir(D):
        os.remove(os.path.join(D, f))
    sizes = {}
    towns = []                                   # small towns go to their own files (<Script>_towns.json), split so no file is over 12 MB
    for name, d in sorted(parts.items()):
        path = os.path.join(D, name + ".json"); json.dump({"k": d["k"]}, open(path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":")); sizes[name] = round(os.path.getsize(path) / 1e6, 2)
        ks = sorted(d["t"]); n = max(1, -(-len(json.dumps(d["t"], ensure_ascii=False).encode()) // 11000000))
        for i in range(n if ks else 0):
            tn = f"{name}_towns{i}"; chunk = {k: d["t"][k] for k in ks[i::n]}
            path = os.path.join(D, tn + ".json"); json.dump({"k": {}, "t": chunk}, open(path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":")); sizes[tn] = round(os.path.getsize(path) / 1e6, 2); towns.append(tn)
    meta = {"countries_keys": n_country, "entries": len(entries), "places": len(city), "regions": len(region), "city_alt_names_seen": n_alt, "min_population": MINPOP, "small_town_names_left_out": st["blocked_small"], "names_left_out_as_product_words": st["blocked_word"], "town_keys": sum(len(d["t"]) for d in parts.values()),
            "sources": ["CLDR (Unicode License) via Babel", "GeoNames (CC BY 4.0): cities500, admin1CodesASCII, countryInfo, alternateNamesV2"]}
    json.dump({"parts": sorted(parts), "towns": towns, "meta": meta, "mb": sizes}, open(os.path.join(D, "index.json"), "w"), indent=0)
    print(meta, sizes, "total MB", round(sum(sizes.values()), 1))


if __name__ == "__main__":
    main()
