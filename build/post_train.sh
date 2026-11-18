#!/bin/sh
# Post-training: teach an existing model new things without starting over. Restart-safe (run the same command again).
#   sh build/post_train.sh <parent_model> <new_model> <minutes> <pairs.jsonl> [more.jsonl ...]
#   e.g. sh build/post_train.sh student_V8 student_V9 120 my_new_pairs.jsonl
# A pairs file has one JSON object per line: {"src": "foreign label", "tgt": "english label", "lang": "ko"}
# Steps: 1 start from the parent's weights  2 fine-tune on the new pairs mixed with the old data (so nothing is
# forgotten)  3 export 8-bit  4 retune the decision rule  5 run all tests against the parent  6 PASS or FAIL.
set -e
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; [ -x $PY ] || PY=python3
PARENT=$1; NEW=$2; MIN=$3; shift 3; W=build/work; TAG=post_$NEW; mkdir -p $W
[ -d models/$PARENT ] || { echo "no such model: models/$PARENT"; exit 1; }
[ -f models/$PARENT/student.pt ] || { echo "models/$PARENT/student.pt is missing: rebuild it from weights/${PARENT}_fp32 (cat student.pt.part* > models/$PARENT/student.pt)"; exit 1; }
if [ ! -f $W/$TAG.prep.done ]; then
  echo "[1/6] start from $PARENT"; $PY build/ft_v8.py prep_post --tag $TAG --from-model $PARENT --extra "$@" 2>&1 | grep -v "Fetching\|Warn" ; touch $W/$TAG.prep.done
fi
if [ ! -f $W/$TAG.train.done ]; then
  echo "[2/6] fine-tune $MIN minutes (log: $W/$TAG.log)"
  $PY build/ft_v8.py train --tag $TAG --minutes $MIN --total-min $MIN --lr ${LR:-5e-5} --ex-lr ${EX_LR:-5e-4} --warmup 100 --resume --extra "$@" >> $W/$TAG.log 2>&1; touch $W/$TAG.train.done
fi
if [ ! -f $W/$TAG.export.done ]; then
  echo "[3/6] export 8-bit"; $PY build/export_v8.py --tag $TAG --out $NEW > $W/$TAG.export.log 2>&1; grep -E "step loop|int8" $W/$TAG.export.log
  echo "[4/6] choose the beam width, retune the decision rule"
  NEURAL_DIR=$NEW $PY build/tune_beams.py 2>&1 | grep chosen | cut -c1-160
  NEURAL_DIR=$NEW $PY build/tune_routing3.py 2>&1 | grep chosen | cut -c1-200; touch $W/$TAG.export.done
fi
echo "[5/6] tests against $PARENT"
if $PY build/compare_models.py $PARENT $NEW; then echo "[6/6] done: models/$NEW is ready"; else echo "[6/6] the new model is worse somewhere: keep $PARENT, or train longer / check the new pairs"; exit 1; fi
