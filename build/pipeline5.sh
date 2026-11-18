#!/bin/sh
# v5: learn more from both teachers. Restart-safe: every step resumes or is skipped when done.
#   setsid nohup sh build/pipeline5.sh >> build/work/pipeline5.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY="./.venv/bin/python -I"; export HF_HUB_DISABLE_PROGRESS_BARS=1 THREADS=4
W=build/work; say() { echo "[$(date '+%H:%M')] $*"; }
step() { [ -f $W/v5_$1.done ] && return 0; say "$1: $2"; shift 2; "$@" && touch $W/v5_$1.done; }

step s1 "OPUS-MT back-translation of all 81k ESCI titles" sh -c "N_ESCI=81011 NEW_TITLES_ONLY=1 $PY build/bt_v2.py opus >> $W/bt_opus_v2.log 2>&1" || exit 1
step s2 "OPUS-MT targets for NLLB-made sources" sh -c "$PY build/kd_v5.py opus >> $W/kd_v5.log 2>&1" || exit 1
step s3 "NLLB targets in NLLB-best languages" sh -c "$PY build/kd_v5.py nllb >> $W/kd_v5.log 2>&1" || exit 1
[ -f build/pipeline5_train.sh ] && sh build/pipeline5_train.sh
