"""Export hand-written rules + glossary to data/rules.json and data/glossary.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "runtime"))
import rules_src as R, glossary_src as GS
from normalizer import norm_key

# kana table: hiragana + katakana, with yoon (きゃ) and sokuon handled in the runtime
kana = dict(R.KANA_BASE)
for base, rom in list(R.KANA_BASE.items()):
    if rom.endswith("i") and len(rom) >= 2 and base not in "いぃ":
        stem = rom[:-1]
        if stem in ("sh", "ch", "j"):
            pre = stem
        else:
            pre = stem + "y"
        for small, v in R.KANA_YOON.items():
            kana[base + small] = pre + v
# katakana extras for loanwords
extra = {"ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo", "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
         "うぃ": "wi", "うぇ": "we", "うぉ": "wo", "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo", "しぇ": "she",
         "じぇ": "je", "ちぇ": "che", "つぁ": "tsa", "いぇ": "ye"}
kana.update(extra)

rules = {
    "cyrillic": R.CYRILLIC, "cyr_ye_after": R.CYR_YE_AFTER,
    "deva_cons": R.DEVA_CONS, "deva_vowels": R.DEVA_VOWELS, "deva_matras": R.DEVA_MATRAS, "deva_other": R.DEVA_OTHER,
    "arabic": R.ARABIC, "ar_names": R.ARABIC_NAMES, "kana": kana, "zh_override": R.ZH_OVERRIDE,
    "address": {lang: [[[norm_key(f) if kind not in ("s", "p") else f for f in forms], en, kind]
                       for forms, en, kind in items] for lang, items in R.ADDRESS.items()},
}
json.dump(rules, open(os.path.join(ROOT, "data", "rules.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)

glossary = {}
for row in GS.G:
    assert len(row) == 1 + len(GS.LANGS), (row[0], len(row))
    en = row[0]
    for lang, cell in zip(GS.LANGS, row[1:]):
        for form in filter(None, cell.split("|")):
            glossary.setdefault(lang, {})[norm_key(form)] = en
cat = {en: c for c, terms in GS.CATEGORY.items() for en in terms}
glossary["_rank"] = {row[0]: GS.RANK[cat.get(row[0], "item")] for row in GS.G}
# auto glossary (build/expand_glossary.py): round-trip-confirmed OPUS / M2M100 translations; hand entries win
AUTO = os.path.join(ROOT, "build", "glossary_auto.json"); n_auto = 0
if os.path.exists(AUTO):
    A = json.load(open(AUTO, encoding="utf-8"))
    for row in A["G"]:
        en = row[0]
        for lang, form in zip(GS.LANGS, row[1:]):
            k = norm_key(form) if form else ""
            if k and k not in glossary.setdefault(lang, {}):
                glossary[lang][k] = en; n_auto += 1
        glossary["_rank"].setdefault(en, GS.RANK[A["rank"].get(en, "item")])
    # a multi-word auto term ("black leather") takes its head word's position, so "bag + black leather" sorts as
    # "black leather bag" instead of staying in source order
    R = glossary["_rank"]
    for row in A["G"]:
        en = row[0]; head = en.split()[-1]
        if " " in en and A["rank"].get(en, "item") == "item" and head in R and R[en] == GS.RANK["item"]:
            R[en] = R[head]
    print("auto glossary terms", len(A["G"]), "entries added", n_auto)
glossary["_rank"].update({en: GS.RANK["function"] for en in ("with", "for", "and")})
glossary["_drop"] = {l: [norm_key(w) for w in ws.split()] for l, ws in GS.FUNCTION_WORDS.items()}
json.dump(glossary, open(os.path.join(ROOT, "data", "glossary.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("kana", len(kana), "| address langs", len(rules["address"]), "| glossary terms", len(GS.G),
      "| entries per lang", {l: len(v) for l, v in glossary.items()})
