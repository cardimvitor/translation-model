#!/bin/sh
# Every model on: gold (320 hand-written), ABO real names (held out), tuned and blind cases. -> build/work/final_eval.json
cd "$(dirname "$0")/.."
PY="./.venv/bin/python -I"; W=build/work; OUT=$W/final_eval.txt; : > $OUT
for D in none student_edited student_A student_B student_C student_B2 student_C2 student_H student_ALL3 student_HY student_MAX student_JG student_BASE; do
  [ "$D" = none ] || [ -d models/$D ] || continue
  E=$D; [ -d models/${D}_edited ] && [ "$D" != student_edited ] && E=${D}_edited
  [ "$E" = none ] || [ -f models/$E/encoder_int8.onnx ] || continue
  echo "== $E" >> $OUT
  NEURAL_DIR=$E $PY test/eval_gold.py 2>/dev/null | grep exact | sed 's/^/gold /' >> $OUT
  NEURAL_DIR=$E $PY test/eval_abo.py 2>/dev/null | grep exact | sed 's/^/abo  /' >> $OUT
  if [ "$E" != none ]; then
    echo "cases $(MODEL=accurate NEURAL_DIR=$E $PY test/run_model.py cases.json | head -1) | blind $(MODEL=accurate NEURAL_DIR=$E $PY test/run_model.py blind.json | head -1)" >> $OUT
    du -sh models/$E/encoder_int8.onnx models/$E/decoder_int8.onnx | awk '{s=s $1 " "} END {print "size " s}' >> $OUT
  else
    echo "cases $($PY test/run_model.py cases.json | head -1) | blind $($PY test/run_model.py blind.json | head -1)" >> $OUT
  fi
done
cat $OUT
