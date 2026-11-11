#!/bin/sh
# Final stage: more product data, pure Hy-MT2 model, MAX model, final evaluation. Restart-safe.
#   setsid nohup sh build/pipeline6.sh >> build/work/pipeline6.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY="./.venv/bin/python -I"; export HF_HUB_DISABLE_PROGRESS_BARS=1 THREADS=4
W=build/work; HY=models/hy-mt2/Hy-MT2-1.8B-Q4_K_M.gguf; say() { echo "[$(date '+%H:%M')] $*"; }
step() { N=$1; [ -f $W/v5_$N.done ] && return 0; say "$1: $2"; shift 2; "$@" && touch $W/v5_$N.done; }
CFG="--d 384 --enc 6 --dec 4 --heads 6 --ff 1536 --lr 8e-4 --tokens 6000 --max-len 96 --ckpt-min 30 --latest-min 5"
CFG_MAX="--d 384 --enc 6 --dec 4 --heads 6 --ff 1536 --lr 8e-4 --tokens 6000 --max-len 96 --ckpt-min 30 --latest-min 5"   # 52M tried first: too slow on 4 CPU cores

train() {   # variant minutes config
  V=$1; MIN=$2; C=$3; M=models/student_$V
  [ -f $W/pairs_${V}_train.jsonl ] || { say "data $V"; $PY build/student_data5.py $V >> $W/data_v5.log 2>&1 || return 1; }
  if [ ! -f $M/student.pt ]; then
    if ls $M/ckpt/step_*.pt >/dev/null 2>&1 || [ -f $M/ckpt/latest.pt ]; then
      DONE=$($PY -c "import json;print(int(sum(r['minutes'] for r in json.load(open('$M/train_log.json'))['runs'])))" 2>/dev/null || echo 0)
      LEFT=$((MIN - DONE)); [ $LEFT -lt 5 ] && LEFT=5
      say "train $V: resume for $LEFT min"; $PY build/train_student2.py --variant $V $C --minutes $LEFT --total-min $MIN --resume >> $W/train_$V.log 2>&1 || return 1
    else
      say "train $V: $MIN min"; $PY build/train_student2.py --variant $V $C --minutes $MIN --total-min $MIN >> $W/train_$V.log 2>&1 || return 1
    fi
    rm -f $M/ckpt/step_*.pt $M/ckpt/latest.pt     # exported; checkpoints only take disk
  fi
  [ -f $W/analysis_student_$V.json ] || { say "weight analysis $V"; DEV=pairs_${V}_dev.jsonl $PY build/analyze_student.py $M 500 > $W/analysis_$V.log 2>&1; }
  [ -f $W/edits_student_$V.json ] || { say "weight edits $V"; $PY build/weight_edits.py $M --dev pairs_${V}_dev.jsonl --train pairs_${V}_train.jsonl --finetune-min 15 --threads 4 > $W/edits_$V.log 2>&1; say "   $V: $(grep DECISION $W/edits_$V.log)"; }
  return 0
}

glossary() {   # grow the word list, keep it only if no test case gets worse
  B=$($PY test/run_model.py blind.json | head -1)
  $PY build/expand_glossary2.py $HY >> $W/glossary2.log 2>&1 || return 1
  $PY build/build_rules.py >> $W/glossary2.log
  T2=$($PY test/run_model.py cases.json | head -1); B2=$($PY test/run_model.py blind.json | head -1)
  say "   bigger word list: tuned $T2 (need PASS 70/70), blind $B2 (before $B)"
  if [ "$T2" != "PASS 70/70" ] || [ "${B2#PASS }" \< "${B#PASS }" ]; then
    say "   regression: bigger word list rejected"; cp $W/glossary_auto_v4_backup.json build/glossary_auto.json; $PY build/build_rules.py >> $W/glossary2.log
  fi
  NEURAL_DIR=none $PY test/eval_gold.py >> $W/glossary2.log 2>&1; say "   $(grep Fast $W/glossary2.log | tail -1)"
}

train ALL3 300 "$CFG" || exit 1
step u1 "ABO real names: held-out test + OPUS-MT filter" sh -c "$PY build/abo_prep.py >> $W/abo.log 2>&1" || exit 1
step u2 "OPUS-MT translations of 60k ABO names" sh -c "$PY build/bt_abo.py >> $W/abo.log 2>&1" || exit 1
step u3 "bigger product word list (ESCI + ABO terms)" glossary || exit 1
step t3 "pure Hy-MT2 data (12k titles)" sh -c "$PY build/hy_data.py pure $HY >> $W/hy_data.log 2>&1" || exit 1
train HY 180 "$CFG" || exit 1
train MAX 600 "$CFG_MAX" || exit 1
step f1 "final evaluation of every model" sh build/final_eval.sh || exit 1
say "pipeline 6 done"; sh build/pipeline7.sh >> $W/pipeline7.log 2>&1
