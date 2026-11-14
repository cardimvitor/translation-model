# Adding new things after training

Three ways, from fastest to most thorough. Each one tests itself and tells you PASS or FAIL.
All commands run from the repository root. Python needs `torch`, `transformers`, `sentencepiece`, `onnxruntime`, `onnx` (and `llama-cpp-python` only for way 3).

## Way 1. Add words or phrases: instant, no training

Use this for product words, brand terms or fixed phrases you want translated one exact way.

1. Write a text file, one term per line, three columns separated by a TAB: language code, foreign text, English.

   ```
   de	Wasserkocher	kettle
   pt	garrafa térmica	thermos bottle
   ```

   Language codes: zh es pt ar ru ja ko de fr hi tr id vi it pl th.
2. Run `python build/add_terms.py my_terms.tsv`.
3. It re-runs the 70 address/label cases and the test labels with the rules-only model. If a score drops, it undoes the change and says so. Otherwise the terms are live for every model.
4. For the browser build, run `sh build/stage_web.sh`.

Good to know: a label whose every word is in the word list is answered by the rules and never reaches the neural model, so this way always wins for those labels.

## Way 2. Add translated pairs: post-training

Use this when you have real examples (your own catalogue, corrections of wrong outputs, a new product category).

1. Write a file with one JSON object per line:

   ```
   {"src": "무선 청소기", "tgt": "cordless vacuum cleaner", "lang": "ko"}
   ```

   A few hundred pairs already help for a narrow topic; a few thousand for a broad one.
2. Put the old training data in place so the model does not forget: `mkdir -p build/work && for f in training_data/pairs_MAX_*.gz training_data/pairs_Q_train*.gz training_data/pairs_HQ_train*.gz; do gunzip -c $f > build/work/$(basename ${f%.gz}); done`
3. Rebuild the parent's full-precision weights once: `cat weights/student_V11_fp32/student.pt.part* > models/student_V11/student.pt`
4. Run: `sh build/post_train.sh student_V11 student_V12 120 my_pairs.jsonl`
   (parent model, name of the new model, minutes of training, your file or files).
5. The script does six steps on its own: start from the parent, fine-tune on your pairs mixed with the old data, export 8-bit, choose the beam width and retune the decision rule, run every test on parent and child, print the verdict.
6. **PASS**: `models/student_V12` is ready; use it with `NEURAL_DIR=student_V12` (Python) or `"student_V12"` (JavaScript). **FAIL**: some score fell by more than 0.5; keep the parent, then train longer or check the new pairs.
7. If the machine restarts, run the same command again: it continues where it stopped.

How long: 60 minutes for a few thousand pairs, 180–300 for tens of thousands (4 CPU cores).
The first line of output tells you the share of text the model cannot read at all; if it is above about 1%, the new data uses characters outside the model's vocabulary and this way will not teach them.

## Way 3. Add English-only product names

Use this when you have an English list (a new catalogue section) and no translations.

1. One English name per line in `names.txt`.
2. Download the writer model once: `python -c "from huggingface_hub import hf_hub_download; hf_hub_download('tencent/Hy-MT2-1.8B-GGUF','Hy-MT2-1.8B-Q4_K_M.gguf',local_dir='models/hy-mt2')"`
3. `python build/make_pairs.py names.txt --langs ko,th,de --out my_pairs.jsonl` (about 0.7 s per name and language).
4. Look at a sample of `my_pairs.jsonl`; delete lines that are wrong.
5. Continue with Way 2, step 2.

## Adding a whole new language

Not covered by one command. The rules need place names and transliteration for it, the language detector must know it, and the neural model needs pairs plus (for v8/v9-style models) a new expert. Ask for it as a project.

## What each way changes

| Way | Changes | Time | Risk |
|---|---|---|---|
| 1 words | `data/glossary.json` | seconds | lowest: undone automatically if a test drops |
| 2 pairs | new folder `models/<new>` (the parent is untouched) | 1–5 hours | gated: FAIL leaves the parent in use |
| 3 names | makes the pairs file for way 2 | 0.7 s per name and language | check the sample by eye |
