#!/bin/sh
# v8 (restart-safe): Hy short labels -> pick decoder layers -> fine-tune -> eval. Nothing else may use the CPU.
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; W=build/work; TOTAL=${TOTAL:-420}
if [ ! -f $W/v8_hy.done ]; then
  echo "[$(date -u +%H:%M)] Hy-MT2 short labels"
  $PY build/hy_queries.py >> $W/hy_queries.log 2>&1 || exit 1; touch $W/v8_hy.done
fi
if [ ! -f $W/v8_pick.done ]; then
  echo "[$(date -u +%H:%M)] which two decoder layers to keep (score right after cutting, no training)"
  : > $W/v8_pick.txt
  for K in 0,5 0,1 0,3 2,5 4,5; do
    $PY build/ft_v8.py prep --tag v8_pick --dec-keep $K > $W/v8_prep_pick.log 2>&1 || exit 1
    $PY build/ft_v8.py train --tag v8_pick --minutes 0.01 --ckpt-min 999 2>/dev/null | grep "^dev" | head -1 | sed "s/^/$K /" >> $W/v8_pick.txt
  done
  rm -rf models/v8_pick; touch $W/v8_pick.done
fi
if [ ! -f $W/v8_prep.done ]; then
  K=$($PY -c "
import re
best=max(open('$W/v8_pick.txt').read().strip().split('\n'), key=lambda l: sum(map(float, re.findall(r': ([\d.]+)', l)[:2])))
print(best.split()[0])")
  echo "[$(date -u +%H:%M)] decoder layers kept: $K"
  $PY build/ft_v8.py prep --dec-keep $K > $W/v8_prep.log 2>&1 || exit 1; touch $W/v8_prep.done
fi
if [ ! -f $W/v8_train.done ]; then
  echo "[$(date -u +%H:%M)] v8 fine-tune (total $TOTAL min)"
  $PY build/ft_v8.py train --minutes $TOTAL --total-min $TOTAL --resume >> $W/ft_v8.log 2>&1 || exit 1; touch $W/v8_train.done
fi
if [ ! -f $W/v8_eval.done ]; then
  $PY build/ft_v8.py eval >> $W/ft_v8_eval.log 2>&1 || exit 1; touch $W/v8_eval.done
fi
echo "[$(date -u +%H:%M)] pipeline 11 done"
