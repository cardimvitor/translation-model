"""English normalizer, Python reference runtime.

API:  normalize(items, model="fast") -> list of dicts
      items: list of (operation, text); operation in {"translate", "transliterate"} (default translate)

The JS runtime (normalizer.js) implements the same logic on the same data files and must
produce identical outputs (checked by test/parity).
"""
import json, math, os, re, unicodedata

DATA = os.environ.get("EN_NORMALIZER_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
VERSION = "fast-0.3"


class NormalizerError(ValueError):
    """code: BAD_INPUT (wrong item shape / operation) or LOAD_FAILED (missing / unreadable data file)."""
    def __init__(self, code, message):
        super().__init__(message); self.code = code

# ---------------------------------------------------------------- text helpers
LATIN_SPECIAL = {"ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ı": "i", "ß": "ss", "æ": "ae", "Æ": "AE",
                 "ø": "o", "Ø": "O", "œ": "oe", "Œ": "OE", "þ": "th", "ð": "d", "ħ": "h"}


def is_latin(ch):
    o = ord(ch)
    return o < 0x250 or 0x1E00 <= o <= 0x1EFF


def fold_latin(s):
    """Remove diacritics from Latin letters only (keeps Japanese dakuten, Hangul, etc.)."""
    out = []
    for ch in s:
        if ch in LATIN_SPECIAL:
            out.append(LATIN_SPECIAL[ch]); continue
        if is_latin(ch) and ord(ch) > 0x7F:
            d = unicodedata.normalize("NFD", ch)
            out.append("".join(c for c in d if not unicodedata.combining(c)))
        else:
            out.append(ch)
    return "".join(out)


_SEP = re.compile(r"[\s\-'’‘`´.,()\[\]/·・]+")


def r3(x):
    """Round to 3 decimals exactly like the JS runtime (Math.round(x * 1000) / 1000, halves up)."""
    return math.floor(x * 1000 + 0.5) / 1000


def lower(s):
    """Same result as the JS runtime: toLowerCase() + the two casefold mappings that matter here."""
    return s.lower().replace("ß", "ss").replace("ς", "σ")


def is_alpha(ch):
    return unicodedata.category(ch)[0] == "L"


def is_digit(ch):
    return unicodedata.category(ch) == "Nd"


def has_alpha(s):
    return any(is_alpha(c) for c in s)


def norm_key(s):
    s = lower(unicodedata.normalize("NFKC", s))
    s = _SEP.sub(" ", s).strip()
    return s


SCRIPTS = [("Latin", None), ("Cyrillic", (0x0400, 0x04FF)), ("Arabic", (0x0600, 0x06FF)),
           ("Devanagari", (0x0900, 0x097F)), ("Thai", (0x0E00, 0x0E7F)), ("Hangul", (0xAC00, 0xD7AF)),
           ("Kana", (0x3040, 0x30FF)), ("Han", (0x4E00, 0x9FFF)), ("Greek", (0x0370, 0x03FF))]


def script_of(ch):
    o = ord(ch)
    if (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A) or (0xC0 <= o <= 0x24F and is_alpha(ch)) or 0x1E00 <= o <= 0x1EFF:
        return "Latin"
    for name, rng in SCRIPTS[1:]:
        if rng[0] <= o <= rng[1]:
            return name
    if 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
        return "Han"
    if 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "Hangul"
    if 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "Kana"
    return None


def scripts_in(s):
    c = {}
    for ch in s:
        sc = script_of(ch)
        if sc:
            c[sc] = c.get(sc, 0) + 1
    return c


# ---------------------------------------------------------------- data
_D = {}


def _load(name):
    if name not in _D:
        try:
            _D[name] = json.load(open(os.path.join(DATA, name), encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise NormalizerError("LOAD_FAILED", f"could not load data file {name} from {DATA}: {e}") from e
    return _D[name]


_GAZ = {}


def gaz():
    """Place-name gazetteer: all per-script parts merged (data/gazetteer/<Script>.json)."""
    if not _GAZ:
        for part in _load(os.path.join("gazetteer", "index.json"))["parts"]:
            _GAZ.update(_load(os.path.join("gazetteer", part + ".json"))["k"])
    return _GAZ


def rules():
    return _load("rules.json")


def gloss():
    return _load("glossary.json")


def tl(name):
    return _load(os.path.join("translit", name))


# ---------------------------------------------------------------- language identification
SCRIPT_LANG = {"Cyrillic": "ru", "Arabic": "ar", "Devanagari": "hi", "Thai": "th", "Hangul": "ko", "Kana": "ja"}


def _nb(group, s):
    m = _load("lid.json")["groups"][group]
    langs, t, unk = m["langs"], m["t"], m["unk"]
    k = norm_key(s)
    grams = []
    if group == "han":
        h = "".join(ch for ch in k if "㐀" <= ch <= "鿿")
        grams = list(h) + [h[i:i + 2] for i in range(len(h) - 1)]
    else:
        for w in k.split():
            w = " " + w + " "
            for n in range(1, m["n"] + 1):
                for i in range(len(w) - n + 1):
                    g = w[i:i + n]
                    if g.strip():
                        grams.append(g)
    sc = [0.0] * len(langs)
    for g in grams:
        r = t.get(g)
        for i in range(len(langs)):
            sc[i] += r[i] if r else unk[i]
    mx = max(sc)
    ex = [pow(2.718281828459045, x - mx) for x in sc]
    z = sum(ex)
    ranked = sorted(range(len(langs)), key=lambda i: (-ex[i], i))
    return [(langs[i], r3(ex[i] / z)) for i in ranked[:3]]


def detect(s):
    """-> (main script, [(lang, prob), ...top 3])"""
    sc = scripts_in(s)
    if not sc:
        return None, [("und", 1.0)]
    main = max(sc, key=lambda k: (sc[k], k))
    if "Kana" in sc:
        return "Kana", [("ja", 1.0)]
    if "Hangul" in sc:
        return "Hangul", [("ko", 1.0)]
    if main in SCRIPT_LANG:
        return main, [(SCRIPT_LANG[main], 1.0)]
    if main == "Han":
        return "Han", _nb("han", s)
    if main == "Latin":
        return "Latin", _nb("latin", s)
    return main, [("und", 1.0)]


# ---------------------------------------------------------------- transliteration per script
def _cap(w):
    return w[:1].upper() + w[1:] if w else w


def tr_cyrillic(s):
    R = rules()["cyrillic"]; after = rules()["cyr_ye_after"]
    out = []; prev = ""
    for ch in s:
        lo = lower(ch)
        if lo in R:
            r = R[lo]
            if lo == "е" and (prev == "" or prev in after or not is_alpha(prev)):
                r = "ye"
            if lo == "ё" and not (prev == "" or prev in after or not is_alpha(prev)):
                r = "yo"
            if ch != lo and r:
                r = r[0].upper() + r[1:]
            out.append(r)
        else:
            out.append(ch)
        prev = lo
    return "".join(out)


def tr_arabic(s):
    R = rules()["arabic"]; NM = rules().get("ar_names", {}); out = []
    for w in s.split(" "):
        pre = ""
        if w in NM:
            out.append(NM[w]); continue
        if w.startswith("ال") and len(w) > 3:
            pre, w = "al-", w[2:]
        out.append(pre + (NM[w] if w in NM else "".join(R.get(ch, ch) for ch in w)))
    return " ".join(out)


def tr_devanagari(s):
    R = rules(); C, V, M, O = R["deva_cons"], R["deva_vowels"], R["deva_matras"], R["deva_other"]
    words = []
    for w in s.split(" "):
        syl = []  # [consonant, vowel, inherent]
        i = 0
        while i < len(w):
            ch = w[i]
            two = w[i:i + 2]
            if two in C:
                syl.append([C[two], "a", True]); i += 2; continue
            if ch in C:
                syl.append([C[ch], "a", True])
            elif ch in M and syl:
                syl[-1][1] = M[ch]; syl[-1][2] = False
            elif ch == "्" and syl:
                syl[-1][1] = ""; syl[-1][2] = False
            elif ch in V:
                syl.append(["", V[ch], False])
            elif ch in ("ं", "ँ"):
                nxt_c = C.get(w[i + 1], "") if i + 1 < len(w) else ""
                nas = "m" if nxt_c[:1] in ("p", "b", "m") else "n"
                if syl:
                    syl[-1][1] += nas
                else:
                    syl.append(["", "n", False])
            elif ch in O:
                syl.append(["", O[ch], False])
            else:
                syl.append(["", ch, False])
            i += 1
        # schwa deletion: word-final, and medial in V C a C V
        if len(syl) > 1 and syl[-1][2]:
            syl[-1][1] = ""
        for j in range(1, len(syl) - 1):
            if syl[j][2] and syl[j - 1][1] != "" and syl[j + 1][0] != "" and syl[j + 1][1] != "":
                syl[j][1] = ""
        words.append("".join(c + v for c, v, _ in syl))
    return " ".join(words)


_KO_L = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_KO_V = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu",
         "eu", "ui", "i"]
_KO_T = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng",
         "t", "t", "k", "t", "p", "t"]
_KO_TL = ["", "g", "kk", "ks", "n", "nj", "n", "d", "r", "lg", "lm", "lb", "ls", "lt", "lp", "r", "m", "b", "bs", "s",
          "ss", "ng", "j", "ch", "k", "t", "p", ""]


def tr_hangul(s):
    """Revised Romanization with liaison (final consonant moves before a silent initial) and l+r -> ll."""
    syl = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            S = o - 0xAC00; syl.append((S // 588, (S % 588) // 28, S % 28))
        else:
            syl.append(ch)
    out = []
    for i, x in enumerate(syl):
        if isinstance(x, str):
            out.append(x); continue
        L, V, T = x
        prev = syl[i - 1] if i > 0 and not isinstance(syl[i - 1], str) else None
        nxt = syl[i + 1] if i + 1 < len(syl) and not isinstance(syl[i + 1], str) else None
        ini = _KO_L[L]
        if prev and prev[2] and L == 11 and prev[2] != 21:
            ini = _KO_TL[prev[2]]                      # liaison
        elif prev and prev[2] in (4, 8) and L == 5:
            ini = "l"                                  # n/l + r -> ll
        elif prev and prev[2] == 8 and L == 2:
            ini = "l"                                  # l + n -> ll
        fin = _KO_T[T]
        if T == 4 and nxt and nxt[0] == 5:
            fin = "l"
        if T and nxt and nxt[0] == 11 and T != 21:
            fin = ""                                   # moved to next syllable
        out.append(ini + _KO_V[V] + fin)
    return "".join(out)


def kana_to_romaji(s):
    K = rules()["kana"]
    h = "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)
    out = []; i = 0; gem = False
    while i < len(h):
        if h[i] in "っ":
            gem = True; i += 1; continue
        if h[i] == "ー":
            i += 1; continue                      # long vowel mark: dropped (postal style)
        r = K.get(h[i:i + 2]) or K.get(h[i])
        step = 2 if K.get(h[i:i + 2]) else 1
        if r is None:
            r = h[i]; step = 1
        if gem and r and is_alpha(r[0]):
            r = ("t" + r) if r.startswith("ch") else (r[0] + r)
        gem = False
        out.append(r); i += step
    s2 = "".join(out)
    # postal-style long vowels: ou -> o, oo -> o, uu -> u
    s2 = re.sub(r"ou(?![aiueo])", "o", s2)
    s2 = s2.replace("oo", "o").replace("uu", "u")
    return s2


def tr_pinyin_word(w):
    Z = tl("zh.json")
    ov = rules()["zh_override"]
    if w in ov:
        return ov[w].replace(" ", "")
    if w in Z["w"]:
        return Z["w"][w].replace(" ", "")
    return "".join(Z["c"].get(ch, tl("fallback.json").get(ch, ch)).lower() for ch in w)


def tr_fallback(s):
    F = tl("fallback.json")
    return "".join(F.get(ch, ch) for ch in s)


# ---------------------------------------------------------------- segmentation (zh / ja / th)
def _word_dict(lang):
    if lang == "zh":
        return tl("zh.json")["w"]
    if lang == "th":
        return tl("th.json")["w"]
    return None


JA_SUFFIX_READ = {"都": "to", "道": "do", "府": "fu", "県": "ken", "市": "shi", "区": "ku", "町": "cho", "村": "mura",
                  "郡": "gun", "丁目": "chome", "番地": "banchi", "号": "go", "番": "ban", "通り": "dori"}


def segment(s, lang, single_gloss=True):
    """Greedy segmentation. At each position the first kind that matches wins, in priority order
    gazetteer > address word > glossary > dictionary word; each kind takes its longest match."""
    G = gaz(); A = rules()["address"].get(lang, []); GL = gloss().get(lang, {})
    afforms = {f for forms, en, kind in A for f in forms}
    WD = _word_dict(lang); JA = tl("ja.json")["w"] if lang == "ja" else None
    segs = []; i = 0; n = len(s)
    while i < n:
        if script_of(s[i]) == "Latin":            # Latin words inside CJK / Thai text stay whole (brands, English words)
            j = i + 1
            while j < n and (script_of(s[j]) == "Latin" or is_digit(s[j]) or s[j] in "-'.&+"):
                j += 1
            segs.append(("other", s[i:j])); i = j; continue
        best = None
        for kind in ("gaz", "addr", "gloss", "word"):
            for j in range(min(n, i + 20), i, -1):
                sub = s[i:j]
                if kind in ("gaz", "gloss") and not has_alpha(sub):
                    continue
                if kind == "gaz":
                    ok = j - i >= 2 and norm_key(sub) in G
                elif kind == "addr":
                    ok = sub in afforms
                elif kind == "gloss":
                    ok = norm_key(sub) in GL and (single_gloss or j - i >= 2)
                else:
                    ok = (WD is not None and sub in WD) or (JA is not None and sub[0] in JA and sub in JA[sub[0]]) \
                        or (lang == "zh" and sub in rules()["zh_override"])
                if ok:
                    best = ((kind, sub), j); break
            if best:
                break
        if best is None:
            ch = s[i]; j = i + 1
            if script_of(ch) == "Kana":           # group kana runs
                while j < n and script_of(s[j]) == "Kana":
                    j += 1
            elif script_of(ch) in ("Han", "Thai"):
                if segs and segs[-1][0] == "other" and script_of(segs[-1][1][-1]) == script_of(ch):
                    prev = segs.pop(); segs.append(("other", prev[1] + ch)); i = j; continue
            elif is_digit(ch):
                while j < n and is_digit(s[j]):
                    j += 1
            segs.append(("other", s[i:j])); i = j
        else:
            segs.append(best[0]); i = best[1]
    return segs


def suffix_forms(lang):
    return sorted([(f, en) for forms, en, kind in rules()["address"].get(lang, []) if kind == "s" for f in forms],
                  key=lambda x: -len(x[0]))


def segment_split(s, lang, depth=0):
    """segment(), then split address suffixes off longer gazetteer/word segments (東京都新宿区 -> 東京都新宿 + 区)."""
    out = []
    SF = suffix_forms(lang)
    first = segment(s, lang, single_gloss=False)
    segs = segment(s, lang) if any(k == "gloss" for k, _ in first) else first
    for kind, t in segs:
        if kind in ("gaz", "word") and depth < 3:
            suf = next((f for f, _ in SF if t.endswith(f) and len(t) > len(f)), None)
            if suf:
                base = t[:-len(suf)]
                inner = segment_split_limited(base, lang, depth + 1)
                out += inner + [("addr", suf)]
                continue
        out.append((kind, t))
    return out


def segment_split_limited(base, lang, depth):
    segs = segment_split(base, lang, depth)
    if len(segs) == 1 and segs[0][1] == base:
        return segs
    return segs


def ja_before_suffix(text, suffix):
    """Reading of a Japanese word that comes right before an address suffix, taken from the compound when the
    dictionary has it (中 + 区 -> 中区 'nakaku' -> 'naka'); otherwise the word's own reading."""
    JA = tl("ja.json")["w"]; w = text + suffix; sr = JA_SUFFIX_READ.get(suffix)
    if sr and text[0] in JA and w in JA[text[0]]:
        r = kana_to_romaji(JA[text[0]][w])
        if r.endswith(sr) and len(r) > len(sr):
            return r[: len(r) - len(sr)]
    return romanize_segment(text, lang="ja")


def romanize_segment(text, lang):
    if lang == "ja" and text in JA_SUFFIX_READ:
        return JA_SUFFIX_READ[text]
    sc = scripts_in(text)
    if lang == "ja" or "Kana" in sc:
        JA = tl("ja.json")["w"]
        if text[0] in JA and text in JA[text[0]]:
            return kana_to_romaji(JA[text[0]][text])
        if "Kana" in sc and "Han" not in sc:
            return kana_to_romaji(text)
        return "".join(kana_to_romaji(JA[c][c]) if c in JA and c in JA[c] else tr_fallback(c).lower() for c in text)
    if lang == "th":
        return tl("th.json")["w"].get(text) or tr_fallback(text).lower()
    if "Han" in sc:
        return tr_pinyin_word(text)
    return romanize(text, lang)


def _group(ch):
    sc = script_of(ch)
    if sc in ("Han", "Kana"):
        return "CJK"
    return sc


def romanize(s, lang=None):
    """Romanize a string that may mix scripts; Latin, digits and punctuation pass through."""
    runs = []
    for ch in s:
        g = _group(ch)
        if runs and runs[-1][0] == g:
            runs[-1][1] += ch
        else:
            runs.append([g, ch])
    out = []
    for g, t in runs:
        if g == "Cyrillic":
            out.append(tr_cyrillic(t))
        elif g == "Arabic":
            out.append(tr_arabic(t))
        elif g == "Devanagari":
            out.append(tr_devanagari(t))
        elif g == "Hangul":
            out.append(tr_hangul(t))
        elif g in ("CJK", "Thai"):
            l = "th" if g == "Thai" else ("ja" if lang == "ja" or any(script_of(c) == "Kana" for c in t) else "zh")
            out.append(" " + render_cjk(segment_split(t, l), l, "transliterate") + " ")
        elif g in (None, "Latin"):
            out.append(t)
        else:
            out.append(tr_fallback(t))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def reorder_gloss(words):
    """Words marked \x02 are glossary terms: sort each run into English adjective order."""
    rank = gloss()["_rank"]; res = []; run = []
    def flush():
        res.extend(sorted(run, key=lambda w: rank.get(w, 5))); run.clear()
    for w in words:
        if w.startswith("\x02") and rank.get(w[1:], 5) < 6:
            run.append(w[1:])
        else:
            flush(); res.append(w[1:] if w.startswith("\x02") else w)
    flush()
    return res


def render_cjk(segs, lang, op):
    """Render zh / ja / th segments. translate: places and known words in English, addresses in English.
    transliterate: everything romanized; Japanese suffixes hyphen-attached (Shinjuku-ku)."""
    G = gaz(); GL = gloss().get(lang, {}); A = rules()["address"].get(lang, [])
    aen = {f: en for forms, en, kind in A for f in forms}
    words = []
    for i, (kind, t) in enumerate(segs):
        if all(is_digit(c) for c in t) or not t.strip():
            words.append(t); continue
        if op == "translate" and kind == "gaz":
            words.append(G[norm_key(t)][0]); continue
        if op == "translate" and kind == "gloss":
            words.append("\x02" + GL[norm_key(t)]); continue
        if kind == "addr":
            if op == "translate":
                en = aen.get(t, "")
                if en.startswith("-") and words:
                    words[-1] += en
                elif en:
                    words.append(en)
                continue
            r = romanize_segment(t, lang)
            if lang == "ja" and words and not all(is_digit(c) for c in words[-1]):
                words[-1] += "-" + r
            else:
                words.append(_cap(r))
            continue
        if script_of(t[0]) in (None, "Latin"):
            words.append(t); continue
        if lang == "ja" and i + 1 < len(segs) and segs[i + 1][0] == "addr":
            words.append(_cap(ja_before_suffix(t, segs[i + 1][1]))); continue
        words.append(_cap(romanize_segment(t, lang)))
    words = reorder_gloss(words)
    if op == "translate":
        words = move_type_words(words, lang)
    out = re.sub(r"\s+", " ", " ".join(w for w in words if w)).strip()
    return re.sub(r"([0-9A-Za-z]) ?([-/]) ?(?=[0-9])", r"\1\2", out)


_ADDR_EN = []


def addr_english():
    if not _ADDR_EN:
        _ADDR_EN.append({en for items in rules()["address"].values() for forms, en, kind in items if en and kind != "title"})
    return _ADDR_EN[0]


TYPE_WORDS = {"Street", "Road", "Avenue", "Lane", "Boulevard", "Square", "Highway", "Alley", "Promenade",
              "Embankment", "Passage", "Way", "Dead End", "Quay", "Causeway", "Ring Road"}
PREFIX_TYPE_LANGS = {"es", "pt", "fr", "it", "ru", "pl", "id", "vi", "ar", "th", "en"}


def move_type_words(words, lang):
    """Prefix-type languages put the street type first (calle Mayor, ул. Ленина, ถนน สุขุมวิท);
    English puts it after the name: move it behind the following name words (stops at numbers / commas)."""
    if lang not in PREFIX_TYPE_LANGS:
        return words
    stop = addr_english()
    out = []; held = None
    for w in words:
        if w in TYPE_WORDS and held is None:
            held = w; continue
        if held is not None and ((w[:1] and is_digit(w[0])) or w in stop or w.endswith(",")):
            out.append(held); held = None
        out.append(w)
    if held is not None:
        out.append(held)
    return out


# ---------------------------------------------------------------- main pipeline
CJK_LANGS = {"zh", "ja", "th"}


def _tokens(part):
    return [t for t in re.split(r"\s+", part) if t]


def _merged(langs, getter):
    """Merge per-language dicts; earlier (more likely) languages win."""
    out = {}
    for l in reversed(langs):
        out.update(getter(l))
    return out


def _stem_lookup(k, table):
    """Inflected forms: a dictionary key of >= 5 letters followed by an ending of <= 3 letters (kabellose -> kabellos)."""
    for cut in (1, 2, 3):
        if len(k) - cut >= 5 and k[:-cut] in table:
            return table[k[:-cut]]
    return None


def translate_spaced(part, lang, op, cand_langs=None):
    """Languages written with spaces (all except zh / ja / th). Longest span first (5 words)."""
    langs = [lang] + [l for l in (cand_langs or []) if l != lang]
    G = gaz()
    GL = _merged(langs, lambda l: gloss().get(l, {}))
    addr = _merged(langs, lambda l: {f: (en, kind) for forms, en, kind in rules()["address"].get(l, [])
                                     if kind not in ("s", "p") for f in forms})
    drop = set(w for l in langs for w in gloss()["_drop"].get(l, [])); rank = gloss()["_rank"]
    lang = next((l for l in langs if any(norm_key(t) in gloss().get(l, {}) or norm_key(t) in
                 {f for forms, en, kind in rules()["address"].get(l, []) for f in forms} for t in _tokens(part))), lang)
    toks = _tokens(part)
    product_like = op == "translate" and any(norm_key(t) in GL or _stem_lookup(norm_key(t), GL) for t in toks)
    out = []; stats = {"known": 0, "total": 0}
    i = 0; in_street = False
    while i < len(toks):
        hit = None
        for L in range(min(5, len(toks) - i), 0, -1):
            span = " ".join(toks[i:i + L]); k = norm_key(span); f = fold_latin(k)
            if not has_alpha(k):
                break
            if op == "translate":
                latin1 = L == 1 and all(script_of(c) in (None, "Latin") for c in span)
                if k in addr:
                    hit = ("addr", addr[k][0], L)
                elif k in GL and not in_street:
                    hit = ("gloss", GL[k], L)
                elif (k in G or (f in G and f != k)) and not in_street and not latin1:
                    hit = ("gaz", (G.get(k) or G.get(f))[0], L)
            if hit:
                break
        tok = toks[i]
        if not hit and op == "translate" and not in_street:
            st = _stem_lookup(norm_key(tok), GL)
            if st:
                hit = ("gloss", st, 1)
        if hit:
            kind, en, L = hit; i += L; stats["known"] += L; stats["total"] += L
            in_street = kind == "addr" and en in TYPE_WORDS
            if kind == "addr" and en == "":
                continue
            out.append((kind, en)); continue
        i += 1; stats["total"] += 1
        k = norm_key(tok)
        if tok[:1] and is_digit(tok[0]):
            in_street = False
        if product_like and k in drop:
            stats["known"] += 1; continue
        if lang == "ko":
            out.append(("name", romanize_korean_token(tok, op))); continue
        if not any(script_of(c) not in (None, "Latin") for c in tok):
            out.append(("keep", tok)); continue
        out.append(("name", _cap_words(romanize(tok, lang))))
    # English adjective order inside runs of glossary words
    res = []; run = []
    def flush():
        res.extend(sorted(run, key=lambda w: rank.get(w, 5))); run.clear()
    for kind, w in out:
        if kind == "gloss" and rank.get(w, 5) < 6:
            run.append(w)
        else:
            flush(); res.append(w)
    flush()
    if op == "translate":
        res = move_type_words(res, lang)
    return " ".join(res), stats


def _cap_words(s):
    return " ".join(_cap(w) for w in s.split(" "))


def romanize_korean_token(tok, op):
    """Korean token: split address suffix (서울특별시 -> 서울 + 특별시) and hyphen-attach it (Sejong-daero)."""
    G = gaz()
    for f, en in suffix_forms("ko"):
        if tok.endswith(f) and len(tok) > len(f):
            base = tok[:-len(f)]
            if op == "translate" and norm_key(base) in G:
                b = G[norm_key(base)][0]
            else:
                b = _cap(tr_hangul(base))
            if op == "translate":
                return b + en if en.startswith("-") or en == "F" else (b + (" " + en if en else ""))
            return b + "-" + tr_hangul(f)
    if op == "translate" and norm_key(tok) in G:
        return G[norm_key(tok)][0]
    return _cap(tr_hangul(tok))


def process(op, text, ascii_out=False):
    op = "transliterate" if str(op).lower().startswith("translit") else "translate"
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()
    script, langs = detect(s)
    lang = langs[0][0]
    res = {"input": text, "operation": op, "language": [list(x) for x in langs], "script": script}
    if not s:
        res.update(output="", confidence=0.0, method="empty"); return res
    G = gaz(); k = norm_key(s); cov = 1.0
    if op == "translate" and (k in G or fold_latin(k) in G):
        e = G.get(k) or G.get(fold_latin(k))
        out, method, conf = e[0], "gazetteer", 0.95
    elif op == "translate" and _encheck()(s, "strict")["english"]:
        out, method, conf = s, "already-english", _encheck()(s, "strict")["score"]
    else:
        parts = [p.strip() for p in re.split(r"[,，、،]", s)]
        outs = []; known = total = 0; methods = set()
        for p in parts:
            if not p:
                continue
            kp = norm_key(p)
            if op == "translate" and (kp in G or fold_latin(kp) in G):
                outs.append((G.get(kp) or G.get(fold_latin(kp)))[0]); known += 1; total += 1; methods.add("gazetteer"); continue
            if lang in CJK_LANGS or script in ("Han", "Kana", "Thai"):
                l = lang if lang in CJK_LANGS else ("th" if script == "Thai" else "ja" if script == "Kana" else "zh")
                segs = segment_split(p, l)
                if script == "Han" and langs[0][1] < 0.9:      # zh vs ja unsure: keep the better-covered reading
                    alt = "ja" if l == "zh" else "zh"
                    segs2 = segment_split(p, alt)
                    cov = lambda ss: sum(len(t) for kd, t in ss if kd in ("gaz", "addr", "gloss"))
                    if cov(segs2) > cov(segs):
                        l, segs = alt, segs2
                        res["language"] = [[alt, langs[0][1] if langs[0][0] == alt else r3(1 - langs[0][1])]] + [x for x in res["language"] if x[0] != alt][:2]
                outs.append(render_cjk(segs, l, op))
                kn = sum(1 for kd, _ in segs if kd in ("gaz", "addr", "gloss"))
                known += kn if op == "translate" else len(segs); total += len(segs)
                methods.add("dictionary" if op == "translate" and kn else "transliteration")
            else:
                o, st = translate_spaced(p, lang, op, [l for l, _ in langs])
                outs.append(o); known += st["known"] if op == "translate" else st["total"]; total += st["total"]
                methods.add("dictionary" if op == "translate" and st["known"] else "transliteration")
        out = ", ".join(x for x in outs if x)
        cov = known / total if total else 0
        base = {"Latin": 0.9, "Cyrillic": 0.85, "Hangul": 0.85, "Kana": 0.85, "Devanagari": 0.75, "Han": 0.75,
                "Arabic": 0.55, "Thai": 0.6}.get(script, 0.5)
        conf = r3(base * (0.6 + 0.4 * cov)) if op == "translate" else base
        method = "+".join(sorted(methods)) or "passthrough"
    if ascii_out:
        out = tr_fallback(fold_latin(out))
    res.update(output=out, confidence=conf, method=method, coverage=r3(cov))
    return res


def _encheck():
    import encheck
    return encheck.is_english


def check_item(it):
    """Accepts "text" or (operation, text); operation is translate / transliterate (prefixes accepted)."""
    if isinstance(it, str):
        return "translate", it
    if not isinstance(it, (list, tuple)) or len(it) < 2 or not isinstance(it[1], str):
        raise NormalizerError("BAD_INPUT", f"each item must be a string or an (operation, text) pair, got {it!r}")
    op = str(it[0]).lower()
    if not (op.startswith("translat") or op.startswith("translit")):
        raise NormalizerError("BAD_INPUT", f'operation must be "translate" or "transliterate", got {it[0]!r}')
    return it[0], it[1]


def normalize(items, ascii_out=False):
    """items: list of (operation, text) or plain strings (translate)."""
    if not isinstance(items, (list, tuple)):
        raise NormalizerError("BAD_INPUT", "normalize() takes a list of items")
    return [process(*check_item(it), ascii_out) for it in items]
