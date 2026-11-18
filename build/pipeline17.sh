#!/bin/sh
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; PYI="./.venv/bin/python -I"; W=build/work
E=student_V11; { echo "== $E"; NEURAL_DIR=$E $PYI test/eval_gold.py 2>/dev/null | grep exact | sed 's/^/gold /'; NEURAL_DIR=$E $PYI test/eval_abo.py 2>/dev/null | grep exact | sed 's/^/abo  /'
  NEURAL_DIR=$E $PYI test/run_model.py accurate 2>/dev/null | tail -1 | sed 's/^/cases /'
  du -sh models/$E/encoder_int8.onnx models/$E/decoder_int8.onnx | awk '{s=s $1 " "} END {print "size " s}'; } > $W/final_eval_V11.txt
$PY build/ft_v8.py eval --tag post_student_V11 >> $W/v11_eval_alone.log 2>&1
cp $W/speed_ours.json $W/speed_ours_v10run.json
rm -f $W/speed_ours.json; BENCH_MODELS=student_V8F,student_V10,student_V11 $PY build/bench_speed.py ours >> $W/bench_ours.log 2>&1
cp $W/seen_split.json $W/seen_split_before_v11.json; $PY build/seen_split.py student_V11 > /dev/null 2>&1; cp $W/seen_split.json $W/seen_split_V11.json
touch $W/v11_final.done
