#!/bin/sh
# v11 (restart-safe): 7B judges v10's misses and translates real queries -> post-train v10 -> v11 -> tests.
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; PYI="./.venv/bin/python -I"; W=build/work
while [ ! -f models/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf ]; do sleep 20; done
if [ ! -f $W/v11_judge.done ]; then echo "[$(date -u +%H:%M)] 7B judge"; $PY build/hy7_judge.py judge >> $W/hy7_judge.log 2>&1 || exit 1; touch $W/v11_judge.done; fi
if [ ! -f $W/v11_real.done ]; then echo "[$(date -u +%H:%M)] 7B real queries"; $PY build/hy7_judge.py real >> $W/hy7_judge.log 2>&1 || exit 1; touch $W/v11_real.done; fi
if [ ! -f $W/v11_new.done ]; then $PY build/v11_pairs.py > $W/v11_new.log 2>&1 || exit 1; touch $W/v11_new.done; fi
if [ ! -f $W/v11_post.done ]; then
  echo "[$(date -u +%H:%M)] post-training v10 -> v11"
  sh build/post_train.sh student_V10 student_V11 ${V11_MIN:-150} pairs_NEW_v11.jsonl >> $W/v11_post.log 2>&1; echo "post_train exit $?" >> $W/v11_post.log; touch $W/v11_post.done
fi
echo "[$(date -u +%H:%M)] pipeline 16 done"
