#!/bin/sh
# v14 Fast: v1 = v8 Fast's weights re-exported (batched decoder for beam search) + v14 rules tuned for it;
#           v2 = cut from v14's weights (encoder layers 1,3,5, decoder layer 1, no experts), retrained 180 min on all current data, + v14 rules.
cd "$(dirname "$0")/.."; PY=./.venv/bin/python; PYI="./.venv/bin/python -I"; W=build/work
if [ ! -f $W/v14f1.done ]; then
  echo "[$(date -u +%H:%M)] v1: re-export v8 Fast and give it the v14 rules"
  mkdir -p models/v14f1 && cp models/v8f/vocab8.json models/v14f1/
  $PY - <<'PYEOF'
import torch
s = torch.load("models/student_V8F/student.pt", map_location="cpu")
torch.save({"model": s["model"], "dec_keep": [1], "enc_layers": 3, "enc_keep": [1, 3, 5], "lite": True}, "models/v14f1/base.pt")
torch.save({"model": s["model"], "experts": s["experts"], "step": 0}, "models/v14f1/latest.pt"); print("v14f1 checkpoint ready")
PYEOF
  $PY build/export_v8.py --tag v14f1 --out student_V14F1 > $W/export_v14f1.log 2>&1 || exit 1
  sh build/v14fast_rules.sh student_V14F1 > $W/rules_v14f1.log 2>&1
  touch $W/v14f1.done
fi
if [ ! -f $W/v14f2_prep.done ]; then
  echo "[$(date -u +%H:%M)] v2: cut from v14 weights"
  $PY build/ft_v8.py prep_lite --tag v14f2 --from-tag post_student_V11 --enc-keep 1,3,5 --dec-keep 1 > $W/v14f2_prep.log 2>&1 || exit 1; touch $W/v14f2_prep.done
fi
if [ ! -f $W/v14f2_train.done ]; then
  echo "[$(date -u +%H:%M)] v2: train ${V2_MIN:-180} min"
  $PY build/ft_v8.py train --tag v14f2 --minutes ${V2_MIN:-180} --total-min ${V2_MIN:-180} --resume >> $W/ft_v14f2.log 2>&1 || exit 1; touch $W/v14f2_train.done
fi
if [ ! -f $W/v14f2.done ]; then
  echo "[$(date -u +%H:%M)] v2: export + rules"
  $PY build/export_v8.py --tag v14f2 --out student_V14F2 > $W/export_v14f2.log 2>&1 || exit 1
  sh build/v14fast_rules.sh student_V14F2 > $W/rules_v14f2.log 2>&1
  touch $W/v14f2.done
fi
if [ ! -f $W/v14f_eval.done ]; then
  echo "[$(date -u +%H:%M)] evaluation"
  for E in student_V14F1 student_V14F2; do T=${E#student_}
    { echo "== $E"; NEURAL_DIR=$E $PYI test/eval_gold.py 2>/dev/null | grep exact | sed 's/^/gold /'; NEURAL_DIR=$E $PYI test/eval_abo.py 2>/dev/null | grep exact | sed 's/^/abo  /'
      echo "cases $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py cases.json 2>/dev/null | head -1) | blind $(MODEL=accurate NEURAL_DIR=$E $PYI test/run_model.py blind.json 2>/dev/null | head -1)"
      du -sh models/$E/encoder_int8.onnx models/$E/decoder_int8.onnx | awk '{s=s $1 " "} END {print "size " s}'; } > $W/final_eval_$T.txt
    $PY build/v11_diff.py $E $W/diff_$E.json > $W/diff_$T.log 2>&1
    NEURAL_DIR=$E TAG=$T $PY test/eval_places.py accurate > $W/places_$T.log 2>&1; SET=places_world2.json NEURAL_DIR=$E TAG=2_$T $PY test/eval_places.py accurate >> $W/places_$T.log 2>&1
    $PY build/compare_models.py student_V8F $E > $W/compare_$T.log 2>&1
  done
  cp $W/seen_split.json $W/seen_split_before_v14f.json; $PY build/seen_split.py student_V14F1 student_V14F2 > /dev/null 2>&1; cp $W/seen_split.json $W/seen_split_v14f.json
  cp $W/speed_ours.json $W/speed_ours_before_v14f.json; rm -f $W/speed_ours.json
  BENCH_MODELS=student_V14,student_V8F,student_V14F1,student_V14F2 $PY build/bench_speed.py ours >> $W/bench_ours.log 2>&1
  touch $W/v14f_eval.done
fi
echo "[$(date -u +%H:%M)] pipeline 22 done"
