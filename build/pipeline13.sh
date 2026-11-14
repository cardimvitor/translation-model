#!/bin/sh
# v9 data (restart-safe): every ESCI query into 7 OPUS-MT languages, then more Hy-MT2 short labels.
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; W=build/work
if [ ! -f $W/v9_q.done ]; then
  echo "[$(date -u +%H:%M)] OPUS-MT short labels: all queries x 7 languages"
  Q_CAP=60000 Q_LANGS=7 $PY build/bt_queries.py >> $W/bt_queries9.log 2>&1 || exit 1; touch $W/v9_q.done
fi
if [ ! -f $W/v9_hy.done ]; then
  echo "[$(date -u +%H:%M)] Hy-MT2 short labels, second batch"
  $PY build/hy_queries.py >> $W/hy_queries9.log 2>&1 || exit 1; touch $W/v9_hy.done
fi
echo "[$(date -u +%H:%M)] pipeline 13 done"
