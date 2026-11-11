#!/bin/sh
# v5 part 2 (called at the end of pipeline5.sh). Restart-safe like part 1.
cd "$(dirname "$0")/.."
PY="./.venv/bin/python -I"; export HF_HUB_DISABLE_PROGRESS_BARS=1 THREADS=4
W=build/work; HY=models/hy-mt2/Hy-MT2-1.8B-Q4_K_M.gguf; say() { echo "[$(date '+%H:%M')] $*"; }
step() { N=$1; [ -f $W/v5_$N.done ] && return 0; say "$1: $2"; shift 2; "$@" && touch $W/v5_$N.done; }
# model: characters (word pieces did not learn fast enough on 4 CPU cores, see reports), d=384, 6 encoder /
# 4 decoder layers, 6 heads, ff 1536, inputs and outputs up to 96 characters (v4: 64)
CFG="--d 384 --enc 6 --dec 4 --heads 6 --ff 1536 --lr 8e-4 --tokens 6000 --max-len 96 --ckpt-min 30 --latest-min 5"

train() {   # variant minutes
  V=$1; MIN=$2; M=models/student_$V
  [ -f $W/pairs_${V}_train.jsonl ] || { say "data $V"; $PY build/student_data5.py $V >> $W/data_v5.log 2>&1 || return 1; }
  if [ ! -f $M/student.pt ]; then
    if ls $M/ckpt/step_*.pt >/dev/null 2>&1 || [ -f $M/ckpt/latest.pt ]; then
      DONE=$($PY -c "import json;print(int(sum(r['minutes'] for r in json.load(open('$M/train_log.json'))['runs'])))" 2>/dev/null || echo 0)
      LEFT=$((MIN - DONE)); [ $LEFT -lt 5 ] && LEFT=5
      say "train $V: resume for $LEFT min"; $PY build/train_student2.py --variant $V $CFG --minutes $LEFT --total-min $MIN --resume >> $W/train_$V.log 2>&1 || return 1
    else
      say "train $V: $MIN min"; $PY build/train_student2.py --variant $V $CFG --minutes $MIN --total-min $MIN >> $W/train_$V.log 2>&1 || return 1
    fi
  fi
  [ -f $W/analysis_student_$V.json ] || { say "weight analysis $V"; DEV=pairs_${V}_dev.jsonl $PY build/analyze_student.py $M 500 > $W/analysis_$V.log 2>&1; }
  [ -f $W/edits_student_$V.json ] || { say "weight edits $V"; $PY build/weight_edits.py $M --dev pairs_${V}_dev.jsonl --train pairs_${V}_train.jsonl --finetune-min 15 --threads 4 > $W/edits_$V.log 2>&1; say "   $V: $(grep DECISION $W/edits_$V.log)"; }
  return 0
}

step t1 "three teachers on the same held-out sample" sh -c "$PY build/teacher_choice.py three $HY >> $W/kd_v5.log 2>&1" || exit 1
train B2 300 || exit 1
train C2 300 || exit 1
step t2 "Hy-MT2 English answers for 30k OPUS-made sources" sh -c "$PY build/hy_data.py kd $HY >> $W/hy_data.log 2>&1" || exit 1
train H 300 || exit 1
train ALL3 300 || exit 1
step t3 "pure Hy-MT2 data (20k titles)" sh -c "$PY build/hy_data.py pure $HY >> $W/hy_data.log 2>&1" || exit 1
train HY 180 || exit 1
say "v5 pipeline done"
