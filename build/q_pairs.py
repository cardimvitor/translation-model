"""Short-label pairs for v6 from build/work/bt_opus_q.jsonl (ESCI queries translated by OPUS-MT), same filters as
the title data (script, confidence, round trip). -> build/work/pairs_Q_train.jsonl, kind "short"."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runtime")); sys.path.insert(0, HERE)
from student_data import student_input, good_bt
from student_data2 import round_trip_ok
n = k = 0; seen = set()
with open(os.path.join(W, "pairs_Q_train.jsonl"), "w", encoding="utf-8") as f:
    for l in open(os.path.join(W, "bt_opus_q.jsonl"), encoding="utf-8"):
        r = json.loads(l); n += 1
        if not good_bt(r) or not round_trip_ok(r):
            continue
        src = student_input(r["src"]); en = r["en"].lower()
        if (src, en) in seen:
            continue
        seen.add((src, en)); k += 1
        f.write(json.dumps({"kind": "short", "lang": r["lang"], "src": src, "tgt": en}, ensure_ascii=False) + "\n")
print("short pairs kept", k, "of", n)
