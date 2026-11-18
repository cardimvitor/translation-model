#!/bin/sh
# v10 (restart-safe): Hy-MT2 head terms in 16 languages -> post-train v9 -> v10 (beam search) -> tests and speed.
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; PYI="./.venv/bin/python -I"; W=build/work
if [ ! -f $W/v10_hy.done ]; then
  echo "[$(date -u +%H:%M)] Hy-MT2 head terms"; $PY build/hy_head.py >> $W/hy_head.log 2>&1 || exit 1; touch $W/v10_hy.done
fi
if [ ! -f $W/v10_new.done ]; then
  $PY - > $W/v10_new.log 2>&1 <<'PYEOF' || exit 1
import json, os, re, sys
sys.path[:0] = ["build", "runtime"]
from student_data import student_input
from ft_v8 import SCRIPT
SC = dict(SCRIPT, ru=r"[Ѐ-ӿ]", ar=r"[؀-ۿ]", hi=r"[ऀ-ॿ]"); W = "build/work"; seen = set(); n = 0
old = set()
for f in ("pairs_MAX_train.jsonl", "pairs_Q_train.jsonl", "pairs_HQ_train.jsonl", "pairs_NEW_v9.jsonl"):
    for l in open(os.path.join(W, f), encoding="utf-8"):
        p = json.loads(l); old.add((p["src"], p["tgt"]))
from collections import Counter
c = Counter()
with open(os.path.join(W, "pairs_NEW_v10.jsonl"), "w", encoding="utf-8") as f:
    for l in open(os.path.join(W, "bt_hy_head.jsonl"), encoding="utf-8"):
        r = json.loads(l); src = student_input(r["src"]); en = r["en"].lower(); n += 1
        if not src or src == en or (src, en) in seen or (src, en) in old or not (0.2 <= len(src) / max(1, len(en)) <= 3.5):
            continue
        if r["lang"] in SC and not re.search(SC[r["lang"]], src):
            continue
        seen.add((src, en)); c[r["lang"]] += 1
        f.write(json.dumps({"src": src, "tgt": en, "lang": r["lang"], "kind": "short_hy"}, ensure_ascii=False) + "\n")
print("new pairs for v10:", len(seen), "of", n, dict(c))
PYEOF
  touch $W/v10_new.done
fi
if [ ! -f $W/v10_post.done ]; then
  echo "[$(date -u +%H:%M)] post-training v9 -> v10"
  sh build/post_train.sh student_V9 student_V10 ${V10_MIN:-180} pairs_NEW_v9.jsonl pairs_NEW_v10.jsonl >> $W/v10_post.log 2>&1; echo "post_train exit $?" >> $W/v10_post.log; touch $W/v10_post.done
fi
if [ ! -f $W/v10_final.done ]; then
  E=student_V10; { echo "== $E"; NEURAL_DIR=$E $PYI test/eval_gold.py 2>/dev/null | grep exact | sed 's/^/gold /'; NEURAL_DIR=$E $PYI test/eval_abo.py 2>/dev/null | grep exact | sed 's/^/abo  /'
  echo "cases $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py cases.json | head -1) | blind $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py blind.json | head -1)"
  du -sh models/$E/encoder_int8.onnx models/$E/decoder_int8.onnx | awk '{s=s $1 " "} END {print "size " s}'; } > $W/final_eval_V10.txt
  $PY build/ft_v8.py eval --tag post_student_V10 >> $W/v10_eval_alone.log 2>&1
  rm -f $W/speed_ours.json; BENCH_MODELS=none,student_MAX_edited,student_V6,student_V7,student_V8,student_V8F,student_V9,student_V10 $PY build/bench_speed.py ours >> $W/bench_ours.log 2>&1
  $PY build/seen_split.py student_V10 > /dev/null 2>&1; cp $W/seen_split.json $W/seen_split_V10.json; touch $W/v10_final.done
fi
echo "[$(date -u +%H:%M)] pipeline 15 done"
