"""SentencePiece unigram encoding in pure Python (same algorithm as runtime/spm.js; no sentencepiece dependency).
Model file: {"pieces": [...], "scores": [...]} exported from the .spm by build/export_marian.py.
The input must already be normalized (NFKC, lowercase, single spaces), as both runtimes do before the neural model."""
import re, struct
SP = chr(0x2581)
_f = struct.Struct("f")
# characters SentencePiece's nmt_nfkc normalization removes (controls, zero-width space, direction marks, BOM)
_DROP = re.compile("[%s]" % "".join(chr(a) if a == b else chr(a) + "-" + chr(b) for a, b in (
    (0x0, 0x8), (0xB, 0xB), (0xE, 0x1F), (0x7F, 0x7F), (0x8F, 0x8F), (0x9F, 0x9F), (0x200B, 0x200B), (0x200E, 0x200F),
    (0x202A, 0x202E), (0x2060, 0x2064), (0xFEFF, 0xFEFF), (0xFFFE, 0xFFFF))))
TIE_LATER = False       # on an exact float32 tie, keep the path found first


def fround(x):
    """float32 rounding, as Math.fround in JS (SentencePiece adds scores in float32)."""
    return _f.unpack(_f.pack(x))[0]


class Unigram:
    def __init__(self, model):
        self.score = {p: fround(x) for p, x in zip(model["pieces"], model["scores"])}
        self.max_len = max(len(p) for p in self.score)
        self.unk = fround(min(self.score.values()) - 10.0)

    def encode(self, text):
        """Pieces (best-scoring segmentation). Characters with no piece come out as their own text, runs merged."""
        text = " ".join(_DROP.sub("", text).split())
        if not text:
            return []
        s = SP + text.replace(" ", SP); n = len(s); NEG = float("-inf")
        best = [NEG] * (n + 1); back = [0] * (n + 1); known = [True] * (n + 1); best[0] = 0.0
        for i in range(n):
            b = best[i]
            if b == NEG:
                continue
            hit = False
            for L in range(1, min(self.max_len, n - i) + 1):
                sc = self.score.get(s[i:i + L])
                if sc is not None:
                    if L == 1:
                        hit = True
                    v = fround(b + sc)
                    if v > best[i + L] or (TIE_LATER and v == best[i + L]):
                        best[i + L] = v; back[i + L] = i; known[i + L] = True
            if not hit:
                v = fround(b + self.unk)
                if v > best[i + 1]:
                    best[i + 1] = v; back[i + 1] = i; known[i + 1] = False
        out = []; j = n
        while j > 0:
            i = back[j]; out.append((s[i:j], known[j])); j = i
        out.reverse(); res = []; prev_unk = False
        for p, k in out:
            if not k and prev_unk:
                res[-1] += p
            else:
                res.append(p)
            prev_unk = not k
        return res
