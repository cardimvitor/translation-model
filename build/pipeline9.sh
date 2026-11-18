#!/bin/sh
# Base models on the real-name test (resumable). Hy-MT2 runs alone: nothing else may use the CPU.
cd "$(dirname "$0")/.."; PY="./.venv/bin/python"; W=build/work
for T in opus nllb hy18; do [ -f $W/teachers/abo_$T.json ] || $PY build/eval_teachers.py $T >> $W/eval_teachers.log 2>&1 || echo "FAILED $T" >> $W/eval_teachers.log; done
[ -f models/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf ] || $PY -c "from huggingface_hub import hf_hub_download; hf_hub_download('tencent/Hy-MT2-7B-GGUF','Hy-MT2-7B-Q4_K_M.gguf',local_dir='models/hy-mt2')" >> $W/eval_teachers.log 2>&1
$PY build/eval_teachers.py hy7 >> $W/eval_teachers.log 2>&1 || echo "FAILED hy7" >> $W/eval_teachers.log
echo "pipeline 9 done" >> $W/eval_teachers.log
