"""Command line: normalize a JSON array of [operation, text] pairs (or plain strings).

  python cli.py input.json [--model fast|accurate] [--ascii] [--variant A|B|C] > output.json
  echo '[["translate","Москва"]]' | python cli.py -
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Translate / transliterate labels into English.")
    ap.add_argument("input", help="JSON file with a list of [operation, text] pairs, or - for stdin")
    ap.add_argument("--model", choices=["fast", "accurate"], default="fast")
    ap.add_argument("--variant", default=None, help="Accurate student variant (folder under models/), e.g. student_B")
    ap.add_argument("--ascii", action="store_true", help="plain ASCII output")
    a = ap.parse_args(argv)
    if a.variant:
        os.environ["NEURAL_DIR"] = a.variant
    items = json.load(sys.stdin if a.input == "-" else open(a.input, encoding="utf-8"))
    import normalizer
    try:
        if a.model == "accurate":
            import accurate
            res = accurate.normalize(items, a.ascii)
        else:
            res = normalizer.normalize(items, a.ascii)
    except normalizer.NormalizerError as e:
        print(f"error {e.code}: {e}", file=sys.stderr); return 2
    json.dump(res, sys.stdout, ensure_ascii=False, indent=1); print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
