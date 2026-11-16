#!/bin/sh
# Gives a fast model the v14 rules: beam width, decision rule, English double check (v14 values), word lists + word check,
# reliable mark and one-word guard, each tuned for this model's own confidence on data outside the tests.
#   sh build/v14fast_rules.sh student_V14F1
cd "$(dirname "$0")/.."; PY=./.venv/bin/python; M=$1; W=build/work
$PY - "$M" <<'PYEOF'
import json, sys
m = sys.argv[1]; p = f"models/{m}/vocab.json"; v = json.load(open(p, encoding="utf-8")); r = json.load(open("models/student_V14/vocab.json", encoding="utf-8"))["policy"]
v.setdefault("policy", {}).update({k: r[k] for k in ("en_sure", "en_tau", "en_sim", "w_max_words")}); json.dump(v, open(p, "w", encoding="utf-8"), ensure_ascii=False); print(m, v["policy"])
PYEOF
MAX_BEAMS=2 NEURAL_DIR=$M $PY build/tune_beams.py > $W/beams_$M.log 2>&1
NEURAL_DIR=$M $PY build/tune_routing3.py > $W/tune_$M.log 2>&1
$PY build/en_words.py $M 3 > $W/enwords_$M.log 2>&1
$PY build/tune_wordcheck.py $M > $W/wordcheck_$M.log 2>&1
$PY build/tune_onetau.py $M > $W/onetau_$M.log 2>&1
$PY build/tune_reliable.py $M 0.97 > $W/reliable_$M.log 2>&1
$PY -c "import json;print(json.load(open('models/$M/vocab.json'))['policy'])"
