#!/bin/sh
# JEPA stage (after pipeline6). Restart-safe.
#   setsid nohup sh build/pipeline7.sh >> build/work/pipeline7.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY="./.venv/bin/python -I"; export THREADS=4
W=build/work; say() { echo "[$(date '+%H:%M')] $*"; }
step() { N=$1; [ -f $W/v5_$N.done ] && return 0; say "$1: $2"; shift 2; "$@" && touch $W/v5_$N.done; }
CFG="--d 384 --enc 6 --dec 4 --heads 6 --ff 1536 --lr 8e-4 --tokens 6000 --max-len 96 --ckpt-min 30 --latest-min 5"
for V in JG BASE; do for s in train dev; do [ -e $W/pairs_${V}_$s.jsonl ] || ln -s pairs_MAX_$s.jsonl $W/pairs_${V}_$s.jsonl; done; done

tune() {    # 4 short JEPA runs (30 min each); the best held-out retrieval@1 sets the full run's settings
  for T in "t_base:--ema 0.996 --mask 0.3 --vic 0.1" "t_ema99:--ema 0.99 --mask 0.3 --vic 0.1" \
           "t_mask5:--ema 0.996 --mask 0.5 --vic 0.1" "t_pred4:--ema 0.996 --mask 0.3 --vic 0.1 --pred-layers 4"; do
    N=${T%%:*}; A=${T#*:}
    [ -f models/jepa_$N/jepa_log.json ] || $PY build/jepa.py pretrain --data MAX --tag $N --minutes 30 $A >> $W/jepa_$N.log 2>&1 || return 1
  done
  $PY -c "
import json,glob
best=max(glob.glob('models/jepa_t_*/jepa_log.json'), key=lambda p: json.load(open(p))['evals'][-1]['retrieval@1'])
for p in sorted(glob.glob('models/jepa_t_*/jepa_log.json')): print(p, json.load(open(p))['evals'][-1])
a=json.load(open(best))['args']; open('$W/jepa_best_args.txt','w').write(' '.join(f'--{k} {a[k]}' for k in ('ema','mask','vic')) + f' --pred-layers {a[\"pred_layers\"]}')
print('best', best)"
}

step j1 "JEPA tuning: 4 short runs" tune || exit 1
step j2 "JEPA pretraining, 180 min, best settings" sh -c "$PY build/jepa.py pretrain --data MAX --tag main --minutes 180 --resume \$(cat $W/jepa_best_args.txt) >> $W/jepa_main.log 2>&1" || exit 1
tr() {   # variant minutes extra-args-for-a-fresh-run; resumes after a restart
  V=$1; MIN=$2; M=models/student_$V; [ -f $M/student.pt ] && return 0
  if ls $M/ckpt/step_*.pt >/dev/null 2>&1 || [ -f $M/ckpt/latest.pt ]; then
    DONE=$($PY -c "import json;print(int(sum(r['minutes'] for r in json.load(open('$M/train_log.json'))['runs'])))" 2>/dev/null || echo 0)
    LEFT=$((MIN - DONE)); [ $LEFT -lt 5 ] && LEFT=5
    $PY build/train_student2.py --variant $V $CFG --minutes $LEFT --total-min $MIN --resume >> $W/train_$V.log 2>&1
  else
    $PY build/train_student2.py --variant $V $CFG $3 --minutes $MIN --total-min $MIN >> $W/train_$V.log 2>&1
  fi
}
step j3 "JEPA-Gen: decoder on the JEPA encoder, 180 min" tr JG 180 "--init-enc models/jepa_main/jepa.pt --enc-lr-scale 0.3" || exit 1
step j4 "same-size baseline on the same data, 360 min" tr BASE 360 "" || exit 1
for V in HY JG BASE; do
  M=models/student_$V; rm -f $M/ckpt/step_*.pt $M/ckpt/latest.pt
  [ -f $W/analysis_student_$V.json ] || { say "weight analysis $V"; DEV=pairs_${V}_dev.jsonl $PY build/analyze_student.py $M 500 > $W/analysis_$V.log 2>&1; }
  [ -f $W/edits_student_$V.json ] || { say "weight edits $V"; $PY build/weight_edits.py $M --dev pairs_${V}_dev.jsonl --train pairs_${V}_train.jsonl --finetune-min 15 --threads 4 > $W/edits_$V.log 2>&1; say "   $V: $(grep DECISION $W/edits_$V.log)"; }
done
step j5 "JEPA-Lookup evaluation" sh -c "NEURAL_DIR=student_MAX $PY build/jepa_lookup.py models/jepa_main/jepa.pt >> $W/jepa_lookup.log 2>&1" || exit 1
rm -f $W/v5_f1.done; step f2 "final evaluation incl. JEPA models" sh build/final_eval.sh || exit 1
say "pipeline 7 done"
