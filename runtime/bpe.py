"""Byte-pair (word-piece) tokenizer used by the v5 students. Same algorithm as bpe.js (checked by test/parity_bpe.js).

Text is already normalized (NFKC, lowercase, single spaces). Each space-separated word starts with "▁"; words longer
than CHUNK code points are cut into chunks (only the first carries "▁"). Merges are applied lowest rank first,
every non-overlapping occurrence left to right, until no ranked pair is left."""
MARK, CHUNK = "▁", 48
_ranks = {}


def ranks(merges):
    key = id(merges)
    if key not in _ranks:
        _ranks[key] = {a + "\x00" + b: i for i, (a, b) in enumerate(merges)}
    return _ranks[key]


def _word(sym, R):
    while len(sym) > 1:
        best, bi = None, -1
        for i in range(len(sym) - 1):
            r = R.get(sym[i] + "\x00" + sym[i + 1])
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break
        a, b = sym[bi], sym[bi + 1]; out = []; i = 0
        while i < len(sym):
            if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(sym[i]); i += 1
        sym = out
    return sym


def encode(text, merges):
    R = ranks(merges); toks = []
    for w in text.split(" "):
        if not w:
            continue
        cps = list(w)
        for j in range(0, len(cps), CHUNK):
            sym = cps[j:j + CHUNK]
            if j == 0:
                sym = [MARK] + sym
            toks += _word(sym, R)
    return toks


def decode(tokens):
    return " ".join("".join(tokens).replace(MARK, " ").split())
