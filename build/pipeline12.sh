#!/bin/sh
# After the complete model (v8) is trained: export, rule tuning, tests; then the fast model (cut from v8), same steps;
# then speed of the old and new models. Restart-safe; waits for pipeline11.
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; PYI="./.venv/bin/python -I"; W=build/work; LITE_MIN=${LITE_MIN:-180}
while [ ! -f $W/v8_eval.done ]; do sleep 30; done
final() {   # $1 = model dir name
  E=$1; { echo "== $E"; NEURAL_DIR=$E $PYI test/eval_gold.py 2>/dev/null | grep exact | sed 's/^/gold /'; NEURAL_DIR=$E $PYI test/eval_abo.py 2>/dev/null | grep exact | sed 's/^/abo  /'
  echo "cases $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py cases.json | head -1) | blind $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py blind.json | head -1)"
  du -sh models/$E/encoder_int8.onnx models/$E/decoder_int8.onnx | awk '{s=s $1 " "} END {print "size " s}'; } > $W/final_eval_$2.txt
}
if [ ! -f $W/v8_export.done ]; then
  echo "[$(date -u +%H:%M)] export + tune + test the complete model"
  $PY build/export_v8.py --out student_V8 > $W/export_v8.log 2>&1 || exit 1
  NEURAL_DIR=student_V8 $PY build/tune_routing3.py > $W/tune_v8.log 2>&1 || exit 1
  final student_V8 V8; touch $W/v8_export.done
fi
if [ ! -f $W/v8f_pick.done ]; then
  echo "[$(date -u +%H:%M)] fast model: which layers to keep"
  : > $W/v8f_pick.txt
  for K in 0,2,5:0 0,2,5:1 1,3,5:1 0,3,5:1; do
    $PY build/ft_v8.py prep_lite --tag v8f_pick --enc-keep ${K%%:*} --dec-keep ${K#*:} > $W/v8f_prep_pick.log 2>&1 || exit 1
    $PY build/ft_v8.py train --tag v8f_pick --minutes 0.01 --ckpt-min 999 2>/dev/null | grep "^dev" | head -1 | sed "s/^/$K /" >> $W/v8f_pick.txt
  done
  rm -rf models/v8f_pick; touch $W/v8f_pick.done
fi
if [ ! -f $W/v8f_prep.done ]; then
  K=$($PY -c "
import re
best=max(open('$W/v8f_pick.txt').read().strip().split('\n'), key=lambda l: sum(map(float, re.findall(r': ([\d.]+)', l)[:2])))
print(best.split()[0])")
  echo "[$(date -u +%H:%M)] fast model layers: $K"
  $PY build/ft_v8.py prep_lite --tag v8f --enc-keep ${K%%:*} --dec-keep ${K#*:} > $W/v8f_prep.log 2>&1 || exit 1; touch $W/v8f_prep.done
fi
if [ ! -f $W/v8f_train.done ]; then
  echo "[$(date -u +%H:%M)] fast model fine-tune ($LITE_MIN min)"
  $PY build/ft_v8.py train --tag v8f --minutes $LITE_MIN --total-min $LITE_MIN --resume >> $W/ft_v8f.log 2>&1 || exit 1; touch $W/v8f_train.done
fi
if [ ! -f $W/v8f_export.done ]; then
  echo "[$(date -u +%H:%M)] export + tune + test the fast model"
  $PY build/ft_v8.py eval --tag v8f >> $W/ft_v8f_eval.log 2>&1
  $PY build/export_v8.py --tag v8f --out student_V8F > $W/export_v8f.log 2>&1 || exit 1
  NEURAL_DIR=student_V8F $PY build/tune_routing3.py > $W/tune_v8f.log 2>&1 || exit 1
  final student_V8F V8F; touch $W/v8f_export.done
fi
if [ ! -f $W/v8_bench.done ]; then
  echo "[$(date -u +%H:%M)] speed"
  BENCH_MODELS=none,student_MAX_edited,student_V6,student_V7,student_V8,student_V8F $PY build/bench_speed.py ours > $W/bench_ours.log 2>&1; touch $W/v8_bench.done
fi
echo "[$(date -u +%H:%M)] pipeline 12 done"
