#!/bin/sh
# v13 final measurements (v13 = v11 weights + word check; no new training kept)
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; PYI="./.venv/bin/python -I"; W=build/work
E=student_V13; { echo "== $E"; NEURAL_DIR=$E $PYI test/eval_gold.py 2>/dev/null | grep exact | sed 's/^/gold /'; NEURAL_DIR=$E $PYI test/eval_abo.py 2>/dev/null | grep exact | sed 's/^/abo  /'
  echo "cases $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py cases.json 2>/dev/null | head -1) | blind $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py blind.json 2>/dev/null | head -1)"
  du -sh models/$E/encoder_int8.onnx models/$E/decoder_int8.onnx | awk '{s=s $1 " "} END {print "size " s}'; } > $W/final_eval_V13.txt
$PY build/tune_reliable.py $E 0.97 > $W/reliable_V13.log 2>&1
cp $W/speed_ours.json $W/speed_ours_v11run.json
rm -f $W/speed_ours.json; BENCH_MODELS=student_V8F,student_V12,student_V13 $PY build/bench_speed.py ours >> $W/bench_ours.log 2>&1
cp $W/seen_split.json $W/seen_split_before_v13.json; $PY build/seen_split.py $E > /dev/null 2>&1
NEURAL_DIR=$E $PY test/make_parity_accurate.py > $W/parity_V13.log 2>&1; NEURAL_DIR=$E node test/parity_accurate.js 2>&1 | grep -v '^{"t"' >> $W/parity_V13.log
touch $W/v13_final.done
NEURAL_DIR=student_V13 TAG=v13 ./.venv/bin/python test/eval_places.py accurate > build/work/places_V13.log 2>&1; ./.venv/bin/python build/v11_diff.py student_V13 build/work/diff_student_V13.json > build/work/diff_V13.log 2>&1; ./.venv/bin/python build/compare_models.py student_V12 student_V13 > build/work/compare_V13.log 2>&1; touch build/work/v13_all.done
