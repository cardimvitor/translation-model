#!/bin/sh
# v6: short-label data, then fine-tune the pretrained translator (restart-safe). Finished steps are skipped.
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; W=build/work
TOTAL=${TOTAL:-510}
if [ ! -f $W/v6_q.done ]; then
  echo "[$(date -u +%H:%M)] v6 short-label data (ESCI queries, OPUS-MT)"
  Q_CAP=30000 Q_LANGS=3 $PY build/bt_queries.py >> $W/bt_queries.log 2>&1 || exit 1
  $PY build/q_pairs.py >> $W/bt_queries.log 2>&1 || exit 1
  touch $W/v6_q.done
fi
if [ ! -f $W/v6_train.done ]; then
  echo "[$(date -u +%H:%M)] v6 fine-tune (total $TOTAL min)"
  $PY build/ft_marian.py train --minutes $TOTAL --total-min $TOTAL --resume --extra pairs_Q_train.jsonl >> $W/ft_marian.log 2>&1 || exit 1
  touch $W/v6_train.done
fi
if [ ! -f $W/v6_eval.done ]; then
  echo "[$(date -u +%H:%M)] v6 eval"
  $PY build/ft_marian.py eval >> $W/ft_marian_eval.log 2>&1 || exit 1
  touch $W/v6_eval.done
fi
echo "[$(date -u +%H:%M)] pipeline 8 done"
