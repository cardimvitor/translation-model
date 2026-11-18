"""en_normalizer: translate / transliterate e-commerce and shipping labels into English (16 languages).

    import en_normalizer as E
    E.normalize([("translate", "Москва"), ("transliterate", "ул. Ленина 5")])            # Fast
    E.normalize(["schwarze Lederjacke"], model="accurate", variant="student_B")           # Accurate
    E.is_english("leather bag")                                                          # English check

Data files are read from EN_NORMALIZER_DATA (default: ../data next to this package); models from ../models.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)        # the modules import each other by plain name (same files as the scripts use)
import normalizer as _N
from normalizer import NormalizerError, VERSION

__all__ = ["normalize", "is_english", "detect", "NormalizerError", "VERSION"]


def normalize(items, model="fast", ascii_out=False, variant=None):
    """items: list of "text" or (operation, text); operation "translate" (default) or "transliterate"."""
    if model == "fast":
        return _N.normalize(items, ascii_out)
    if model == "accurate":
        import accurate
        return accurate.normalize(items, ascii_out, variant)
    raise NormalizerError("BAD_INPUT", 'model must be "fast" or "accurate"')


def is_english(text, mode="strict"):
    import encheck
    return encheck.is_english(text, mode)


def detect(text):
    return _N.detect(text)
