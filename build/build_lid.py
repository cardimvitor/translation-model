"""Train the language identifier (naive Bayes over character n-grams), export data/lid.json.

Script decides most languages; two n-gram groups resolve the rest:
  latin: es pt fr it de pl tr id vi en
  han:   zh ja   (strings written only in Han characters)
Training text: UDHR (all 16 + English), Europarl (es pt fr it de en), CLDR display names
(languages, territories, months, days) for short strings, glossary forms.
Evaluation on held-out UDHR/Europarl text cut into 1-word, 2-3-word and sentence pieces.
"""
import json, os, random, re, sys, unicodedata, math, glob
from collections import Counter, defaultdict
from babel import Locale
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime")); sys.path.insert(0, HERE)
from normalizer import norm_key
import glossary_src as GS

SRC = "/tmp/claude-0/-home-user/dd1f1d9e-08f9-5014-8bb7-06f12abeac7a/scratchpad/data"
UDHR = {"es": "spa", "pt": ["por_BR", "por_PT"], "fr": "fra", "it": "ita", "de": "deu", "pl": "pol", "tr": "tur",
        "id": "ind", "vi": "vie", "en": "eng", "zh": ["cmn_hans", "cmn_hant"], "ja": "jpn"}
EUROPARL = {"es": "spanish", "pt": "portuguese", "fr": "french", "it": "italian", "de": "german", "en": "english"}
CLDR = {"es": "es", "pt": "pt", "fr": "fr", "it": "it", "de": "de", "pl": "pl", "tr": "tr", "id": "id", "vi": "vi",
        "en": "en", "zh": "zh_Hans", "ja": "ja"}
GROUPS = {"latin": ["es", "pt", "fr", "it", "de", "pl", "tr", "id", "vi", "en"], "han": ["zh", "ja"]}
N_MAX = {"latin": 3, "han": 2}
TOP = 4000


def ngrams(s, group):
    s = norm_key(s)
    if group == "han":
        s = "".join(ch for ch in s if "㐀" <= ch <= "鿿")
        out = list(s) + [s[i:i + 2] for i in range(len(s) - 1)]
        return out
    out = []
    for w in s.split():
        w = f" {w} "
        for n in range(1, N_MAX[group] + 1):
            out += [w[i:i + n] for i in range(len(w) - n + 1)]
    return [g for g in out if g.strip()]


def texts(lang):
    t = []
    u = UDHR[lang]
    for f in (u if isinstance(u, list) else [u]):
        t.append(open(f"{SRC}/udhr2/{f}.txt", encoding="utf-8", errors="ignore").read())
    if lang in EUROPARL:
        fs = sorted(glob.glob(f"{SRC}/europarl_raw/{EUROPARL[lang]}/*"))
        t.append("\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in fs)[:3_000_000])
    return "\n".join(t)


def cldr_strings(lang):
    L = Locale.parse(CLDR[lang]); out = list(L.territories.values()) + list(L.languages.values())
    for ctx in ("format", "stand-alone"):
        try:
            out += list(L.months[ctx]["wide"].values()) + list(L.days[ctx]["wide"].values())
        except Exception:
            pass
    return out


def han_extra():
    """Labeled Han-only strings: Japanese dictionary words + JP city names vs Chinese words + CN/TW city names."""
    import pickle, pykakasi, geonamescache
    from pypinyin import phrases_dict
    han = re.compile(r"^[\u3400-\u9fff]{2,8}$")
    kw = pickle.load(open(os.path.join(os.path.dirname(pykakasi.__file__), "data", "kanwadict4.db"), "rb"))
    ja = [w for t in kw.values() for w in t if han.match(w)]
    zh = [w for w in phrases_dict.phrases_dict if han.match(w)]
    for v in geonamescache.GeonamesCache().get_cities().values():
        for a in v.get("alternatenames") or []:
            if han.match(a):
                if v["countrycode"] == "JP": ja.append(a)
                elif v["countrycode"] in ("CN", "TW"): zh.append(a)
    random.seed(1); random.shuffle(ja); random.shuffle(zh)
    n = min(len(ja), len(zh), 60000)
    return {"ja": ja[:n], "zh": zh[:n]}


def main():
    random.seed(0)
    global HAN_EXTRA
    HAN_EXTRA = han_extra()
    model = {"groups": {}}
    report = {}
    for g, langs in GROUPS.items():
        counts = {l: Counter() for l in langs}; test = {}
        for l in langs:
            words = norm_key(texts(l)).split() if g == "latin" else list(re.findall(r"[㐀-鿿]+", texts(l)))
            cut = int(len(words) * 0.9); train, held = words[:cut], words[cut:]
            for w in train:
                counts[l].update(ngrams(w, g))
            for s in cldr_strings(l):
                counts[l].update(ngrams(s, g) * 3)
            if g == "latin":
                for row in GS.G:
                    for f in filter(None, row[1 + GS.LANGS.index(l)].split("|")) if l in GS.LANGS else []:
                        counts[l].update(ngrams(f, g) * 3)
            if g == "han":
                extra = HAN_EXTRA[l]; random.shuffle(extra); c2 = int(len(extra) * 0.9)
                for w in extra[:c2]:
                    counts[l].update(ngrams(w, g))
                held = held + extra[c2:]
            test[l] = held
        vocab = set()
        for l in langs:
            vocab |= {k for k, _ in counts[l].most_common(TOP)}
        table, unk = {}, []
        for l in langs:
            tot = sum(counts[l][k] for k in vocab) + len(vocab)
            unk.append(round(math.log(1 / tot), 3))
        for k in vocab:
            row = []
            for l in langs:
                tot = sum(counts[l][kk] for kk in vocab) if False else None
            table[k] = None
        tots = {l: sum(counts[l][k] for k in vocab) + len(vocab) for l in langs}
        for k in vocab:
            table[k] = [round(math.log((counts[l][k] + 1) / tots[l]), 3) for l in langs]
        model["groups"][g] = {"langs": langs, "n": N_MAX[g], "t": table, "unk": [round(math.log(1 / tots[l]), 3) for l in langs]}

        # evaluation by input length
        def predict(s):
            sc = [0.0] * len(langs)
            for x in ngrams(s, g):
                r = table.get(x)
                for i in range(len(langs)):
                    sc[i] += r[i] if r else model["groups"][g]["unk"][i]
            return langs[max(range(len(langs)), key=lambda i: sc[i])]
        rep = {}
        for name, k in (("1 word", 1), ("2-3 words", 2), ("sentence (10 words)", 10)):
            ok = n = 0
            for l in langs:
                h = test[l]
                for _ in range(300):
                    i = random.randrange(0, max(1, len(h) - k))
                    piece = (" " if g == "latin" else "").join(h[i:i + k])
                    if not piece.strip():
                        continue
                    n += 1; ok += predict(piece) == l
            rep[name] = round(ok / n, 3)
        report[g] = rep
        print(g, "vocab", len(vocab), "held-out accuracy:", rep)
    model["eval"] = report
    p = os.path.join(ROOT, "data", "lid.json")
    json.dump(model, open(p, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"lid.json {os.path.getsize(p)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
