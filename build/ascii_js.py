"""Rewrite a JS file so it is pure ASCII (non-ASCII chars -> \\uXXXX / \\u{XXXXX}); behavior is identical."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
out = []
for ch in s:
    o = ord(ch)
    out.append(ch if o < 128 else (f"\\u{o:04x}" if o <= 0xFFFF else f"\\u{{{o:x}}}"))
open(dst, "w", encoding="ascii").write("".join(out))
