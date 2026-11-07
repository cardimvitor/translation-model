#!/bin/sh
# Stage the phone test page with the same paths it is published with.
set -e
cd "$(dirname "$0")/.."
D=web/dist; rm -rf $D; mkdir -p $D/data/translit $D/data/gazetteer $D/runtime $D/ort
cp web/index.html $D/
cp data/*.json $D/data/; cp data/translit/*.json $D/data/translit/; cp data/gazetteer/*.json $D/data/gazetteer/
for f in encheck normalizer student accurate; do python3 build/ascii_js.py runtime/$f.js $D/runtime/$f.js; done
printf '<meta charset="utf-8">\n' | cat - web/index.html > $D/local_test.html   # the published page gets its charset from the host skeleton
for V in A:student_A B:student_B C:student_C; do   # published as student_A / _B / _C
  N=student_${V%%:*}; S=models/${V#*:}; mkdir -p $D/models/$N
  for m in encoder decoder; do cp $S/${m}_int8.onnx $D/models/$N/${m}_int8.onnx.wasm; done; cp $S/vocab.json $D/models/$N/
done
cp node_modules/onnxruntime-web/dist/ort.wasm.min.js node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm $D/ort/
du -sh $D; find $D -type f -size +15M
