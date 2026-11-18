#!/bin/sh
# v4 long-running pipeline. Every step is skipped when its output exists or resumes where it stopped,
# so after a container/worker restart just run it again:  nohup sh build/pipeline.sh >> build/work/pipeline.log 2>&1 &
cd "$(dirname "$0")/.."
PY="./.venv/bin/python -I"; export HF_HUB_DISABLE_PROGRESS_BARS=1 THREADS=4
W=build/work; say() { echo "[$(date '+%H:%M')] $*"; }

say "1 OPUS back-translation (variant B)"; $PY build/bt_v2.py opus >> $W/bt_opus_v2.log 2>&1 || exit 1

if [ ! -f build/glossary_auto.json ] && [ ! -f $W/glossary_auto.rejected ]; then
  say "2 grow the product glossary (OPUS / M2M100, round trip)"
  $PY build/expand_glossary.py > $W/glossary.log 2>&1 || exit 1
  $PY build/build_rules.py >> $W/glossary.log
  T=$($PY test/run_model.py cases.json | head -1); B=$($PY test/run_model.py blind.json | head -1)
  say "   with auto glossary: tuned $T, blind $B"
  if [ "$T" != "PASS 70/70" ] || [ "${B#PASS }" \< "42/48" ]; then
    say "   regression: auto glossary rejected"; mv build/glossary_auto.json $W/glossary_auto.rejected; $PY build/build_rules.py >> $W/glossary.log
  fi
  NEURAL_DIR=none $PY test/eval_gold.py >> $W/glossary.log 2>&1; tail -1 $W/glossary.log
fi

[ -f $W/bt_nllb_v2.enough ] || { say "3 NLLB back-translation (variant A)"; $PY build/bt_v2.py nllb >> $W/bt_nllb_v2.log 2>&1 || exit 1; }

for V in A B C; do [ -f $W/pairs_${V}_train.jsonl ] || { say "4 training pairs $V"; $PY build/student_data2.py $V >> $W/data_v2.log 2>&1 || exit 1; }; done

for V in B A C; do
  M=models/student_$V; MIN=240; [ $V = C ] && MIN=300
  if [ ! -f $M/student.pt ]; then
    if ls $M/ckpt/step_*.pt >/dev/null 2>&1 || [ -f $M/ckpt/latest.pt ]; then
      DONE=$($PY -c "import json;print(int(sum(r['minutes'] for r in json.load(open('$M/train_log.json'))['runs'])))" 2>/dev/null || echo 0)
      LEFT=$((MIN - DONE)); [ $LEFT -lt 5 ] && LEFT=5
      say "5 train $V: resume for $LEFT min"; $PY build/train_student2.py --variant $V --minutes $LEFT --total-min $MIN --resume --ckpt-min 30 >> $W/train_$V.log 2>&1 || exit 1
    else
      say "5 train $V: $MIN min"; $PY build/train_student2.py --variant $V --minutes $MIN --total-min $MIN --ckpt-min 30 >> $W/train_$V.log 2>&1 || exit 1
    fi
  fi
  [ -f $W/analysis_student_$V.json ] || { say "6 weight analysis $V"; DEV=pairs_${V}_dev.jsonl $PY build/analyze_student.py $M 500 > $W/analysis_$V.log 2>&1; }
  [ -f $W/edits_student_$V.json ] || { say "6 weight edits $V"; $PY build/weight_edits.py $M --dev pairs_${V}_dev.jsonl --train pairs_${V}_train.jsonl --finetune-min 15 --threads 4 > $W/edits_$V.log 2>&1; }
  say "   $V: $(grep DECISION $W/edits_$V.log)"
done
say "pipeline done"
