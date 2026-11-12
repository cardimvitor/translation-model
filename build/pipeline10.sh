#!/bin/sh
# v7: language experts on top of v6 (restart-safe).
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; W=build/work; TOTAL=${TOTAL:-240}
if [ ! -f $W/v7_train.done ]; then
  echo "[$(date -u +%H:%M)] v7 experts (total $TOTAL min)"
  $PY build/ft_moe.py train --minutes $TOTAL --total-min $TOTAL --resume >> $W/ft_moe.log 2>&1 || exit 1
  touch $W/v7_train.done
fi
if [ ! -f $W/v7_eval.done ]; then
  $PY build/ft_moe.py eval >> $W/ft_moe_eval.log 2>&1 || exit 1; touch $W/v7_eval.done
fi
echo "[$(date -u +%H:%M)] pipeline 10 done"
