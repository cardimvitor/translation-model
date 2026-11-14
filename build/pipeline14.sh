#!/bin/sh
# v9 = v8 Complete post-trained with the new short labels, made with the generic tool build/post_train.sh.
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; PYI="./.venv/bin/python -I"; W=build/work
while [ ! -f $W/v9_hy.done ]; do sleep 30; done
if [ ! -f $W/v9_new.done ]; then
  echo "[$(date -u +%H:%M)] new pairs file"
  cp $W/pairs_Q_train.jsonl $W/pairs_Q_train_v8.jsonl; cp $W/pairs_HQ_train.jsonl $W/pairs_HQ_train_v8.jsonl
  $PY build/q_pairs.py >> $W/v9_new.log 2>&1 || exit 1
  $PY - >> $W/v9_new.log 2>&1 <<'PYEOF' || exit 1
import json, os, sys
sys.path[:0] = ["build", "runtime"]
import ft_v8 as V8
W = "build/work"; rd = lambda n: [json.loads(l) for l in open(os.path.join(W, n), encoding="utf-8")]
old = {(p["src"], p["tgt"]) for p in rd("pairs_Q_train_v8.jsonl") + rd("pairs_HQ_train_v8.jsonl")}
new = [p for p in rd("pairs_Q_train.jsonl") + V8.hy_pairs() if (p["src"], p["tgt"]) not in old]
with open(os.path.join(W, "pairs_NEW_v9.jsonl"), "w", encoding="utf-8") as f:
    for p in new:
        f.write(json.dumps({"src": p["src"], "tgt": p["tgt"], "lang": p["lang"], "kind": p["kind"]}, ensure_ascii=False) + "\n")
from collections import Counter
print("new pairs for v9:", len(new), dict(Counter(p["lang"] for p in new)))
PYEOF
  cp $W/pairs_Q_train_v8.jsonl $W/pairs_Q_train.jsonl; touch $W/v9_new.done
fi
if [ ! -f $W/v9_post.done ]; then
  echo "[$(date -u +%H:%M)] post-training v8 -> v9"
  sh build/post_train.sh student_V8 student_V9 ${V9_MIN:-300} pairs_NEW_v9.jsonl >> $W/v9_post.log 2>&1; echo "post_train exit $?" >> $W/v9_post.log; touch $W/v9_post.done
fi
if [ ! -f $W/v9_final.done ]; then
  E=student_V9; { echo "== $E"; NEURAL_DIR=$E $PYI test/eval_gold.py 2>/dev/null | grep exact | sed 's/^/gold /'; NEURAL_DIR=$E $PYI test/eval_abo.py 2>/dev/null | grep exact | sed 's/^/abo  /'
  echo "cases $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py cases.json | head -1) | blind $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py blind.json | head -1)"
  du -sh models/$E/encoder_int8.onnx models/$E/decoder_int8.onnx | awk '{s=s $1 " "} END {print "size " s}'; } > $W/final_eval_V9.txt
  $PY build/ft_v8.py eval --tag post_student_V9 >> $W/v9_eval_alone.log 2>&1
  $PY - <<'PYEOF'
import json
p = "build/work/speed_ours.json"; d = json.load(open(p)); d.pop("student_V9", None); json.dump(d, open(p, "w"), indent=1)
PYEOF
  BENCH_MODELS=student_V9 $PY build/bench_speed.py ours >> $W/bench_ours.log 2>&1; touch $W/v9_final.done
fi
echo "[$(date -u +%H:%M)] pipeline 14 done"
