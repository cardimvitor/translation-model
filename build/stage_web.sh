#!/bin/sh
# Stage the phone test page with the same paths it is published with.
set -e
cd "$(dirname "$0")/.."
D=web/dist; rm -rf $D; mkdir -p $D/data/translit $D/data/gazetteer $D/runtime $D/ort
cp web/index.html $D/
cp data/*.json $D/data/; cp data/translit/*.json $D/data/translit/; cp data/gazetteer/*.json $D/data/gazetteer/
for f in encheck normalizer bpe spm student accurate; do python3 build/ascii_js.py runtime/$f.js $D/runtime/$f.js; done
printf '<meta charset="utf-8">\n' | cat - web/index.html > $D/local_test.html   # the published page gets its charset from the host skeleton
echo "{" > $D/models_parts.tmp
for V in V6:student_V6 MAX:student_MAX C:student_C; do   # published as student_V6 / _MAX / _C
  N=student_${V%%:*}; S=models/${V#*:}; mkdir -p $D/models/$N
  for m in encoder decoder; do
    F=$S/${m}_int8.onnx
    if [ $(stat -c %s $F) -gt 14000000 ]; then      # the page host takes at most 15 MB per file: publish in pieces
      split -b 14000000 -d -a 2 $F $D/models/$N/${m}_int8.onnx.piece
      i=0; for P in $D/models/$N/${m}_int8.onnx.piece*; do mv $P $D/models/$N/${m}_int8.onnx.part$i.wasm; i=$((i+1)); done
      echo "\"$N/${m}_int8.onnx\": $i," >> $D/models_parts.tmp
    else cp $F $D/models/$N/${m}_int8.onnx.wasm; fi
  done
  cp $S/vocab.json $D/models/$N/; [ -f $S/spm_src.json ] && cp $S/spm_src.json $D/models/$N/
done
echo "\"_\": 0}" >> $D/models_parts.tmp; mv $D/models_parts.tmp $D/models/parts.json
cp node_modules/onnxruntime-web/dist/ort.wasm.min.js node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm $D/ort/
du -sh $D; find $D -type f -size +15M
