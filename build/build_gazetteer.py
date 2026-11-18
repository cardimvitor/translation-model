"""Build the place-name dictionary (gazetteer): any-language name -> English name.

Sources (build time only):
  - CLDR territory names for all locales (via Babel)            -> countries
  - GeoNames cities with population >= ~15k and their alternate names (via geonamescache)
Output: data/gazetteer/<Script>.json  {"k": {normalized_name: [english, kind, cc]}} + data/gazetteer/index.json
Collisions keep the most populous place (countries always win over cities).
"""
import json, os, re, sys, unicodedata
import babel, geonamescache
from babel import Locale, localedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime"))
from normalizer import norm_key, fold_latin  # the SAME key function the runtimes use

# scripts we keep for city alternate names (the 16 languages) + Latin
KEEP_SCRIPT = re.compile(
    r"^[\u0000-ɏḀ-ỿ"        # Latin (incl. Vietnamese)
    r"Ѐ-ӿ"                         # Cyrillic
    r"؀-ۿ"                         # Arabic
    r"ऀ-ॿ"                         # Devanagari
    r"฀-๿"                         # Thai
    r"぀-ヿㇰ-ㇿ"            # Kana
    r"一-鿿㐀-䶿"            # Han
    r"가-힯ᄀ-ᇿ"            # Hangul
    r"\s\-'’.()]+$")
# English exonyms where GeoNames keeps the local form
EXONYMS = {"Köln": "Cologne", "Zürich": "Zurich", "Kraków": "Krakow", "Sevilla": "Seville",
           "Thessaloníki": "Thessaloniki", "Ar Riyāḑ": "Riyadh", "Makkah": "Mecca", "Al Madīnah": "Medina",
           "Hannover": "Hanover", "Frankfurt am Main": "Frankfurt", "Québec": "Quebec City",
           "Montréal": "Montreal", "Bogotá": "Bogota", "Brasília": "Brasilia", "İzmir": "Izmir"}
# native spellings GeoNames files under a bigger neighbour (e.g. New Delhi's names under Delhi)
NAME_OVERRIDES = {"नई दिल्ली": "New Delhi", "नयी दिल्ली": "New Delhi", "نيودلهي": "New Delhi", "نيو دلهي": "New Delhi",
                  "新德里": "New Delhi", "ニューデリー": "New Delhi", "뉴델리": "New Delhi", "Нью-Дели": "New Delhi",
                  "นิวเดลี": "New Delhi", "Nueva Delhi": "New Delhi", "Nova Deli": "New Delhi", "Nowe Delhi": "New Delhi",
                  "Yeni Delhi": "New Delhi", "Nuova Delhi": "New Delhi", "Neu-Delhi": "New Delhi", "New Delhi": "New Delhi"}
LOCALES_16 = ["zh_Hans", "zh_Hant", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th", "en"]


def main():
    entries = {}  # key -> (english, kind, cc, priority)

    def add1(k, english, kind, cc, prio):
        if not k or len(k) < 2 or not any(ch.isalpha() for ch in k):
            return
        cur = entries.get(k)
        if cur is None or prio > cur[3]:
            entries[k] = (english, kind, cc, prio)

    def add(name, english, kind, cc, prio):
        english = EXONYMS.get(english, english)
        k = norm_key(name)
        add1(k, english, kind, cc, prio)
        f = fold_latin(k)                      # accent-free spelling ("Lodz", "Sao Paulo")
        if f != k:
            add1(f, english, kind, cc, prio - 1)

    # countries / territories: all CLDR locales (cheap, maximizes coverage)
    en = Locale.parse("en").territories
    codes = [c for c in en if re.fullmatch(r"[A-Z]{2}", c)]
    all_locales = localedata.locale_identifiers()
    lang_names = set()                      # GeoNames has junk alternate names that are language names ("日本語", "kartuli")
    for loc in all_locales:
        try:
            L = Locale.parse(loc); T = L.territories
            lang_names.update(norm_key(x) for x in L.languages.values())
        except Exception:
            continue
        for c in codes:
            if c in T:
                add(T[c], en[c], "country", c, 10 ** 12)
    n_country = len(entries)

    # cities
    gc = geonamescache.GeonamesCache()
    n_alt = 0
    city_en = {norm_key(v["name"]) for v in gc.get_cities().values()}     # each city keeps its own English name
    for v in gc.get_cities().values():
        pop = int(v.get("population") or 0)
        english = v["name"]
        add(english, english, "city", v["countrycode"], pop)
        for a in v.get("alternatenames") or []:
            if re.fullmatch(r"[A-Z]{3}", a):          # IATA codes like MUC
                continue
            if not KEEP_SCRIPT.match(a):
                continue
            if norm_key(a) in lang_names and norm_key(a) != norm_key(english):
                continue
            if norm_key(a) in city_en and norm_key(a) != norm_key(english):   # "New Delhi" listed under Delhi
                continue
            add(a, english, "city", v["countrycode"], pop)
            n_alt += 1

    for name, english in NAME_OVERRIDES.items():
        add(name, english, "city", "IN", 10 ** 13)
    out = {"k": {k: [e, kind, cc] for k, (e, kind, cc, _) in entries.items()},
           "meta": {"countries_keys": n_country, "entries": len(entries), "city_alt_names_seen": n_alt,
                    "sources": ["CLDR (Unicode License) via Babel", "GeoNames (CC BY 4.0) via geonamescache"]}}
    # one file per writing system of the key's first letter, so an app only loads what its text needs
    from normalizer import script_of, is_alpha
    D = os.path.join(ROOT, "data", "gazetteer"); os.makedirs(D, exist_ok=True)
    for f in os.listdir(D):
        os.remove(os.path.join(D, f))
    parts = {}
    for k, v in out["k"].items():
        a = next((ch for ch in k if is_alpha(ch)), None)
        parts.setdefault((script_of(a) if a else None) or "Other", {})[k] = v
    sizes = {}
    for name, d in sorted(parts.items()):
        path = os.path.join(D, name + ".json")
        json.dump({"k": d}, open(path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
        sizes[name] = round(os.path.getsize(path) / 1e6, 2)
    json.dump({"parts": sorted(parts), "meta": out["meta"], "mb": sizes}, open(os.path.join(D, "index.json"), "w"), indent=0)
    old = os.path.join(ROOT, "data", "gazetteer.json")
    if os.path.exists(old):
        os.remove(old)
    print(out["meta"], sizes)


if __name__ == "__main__":
    main()
