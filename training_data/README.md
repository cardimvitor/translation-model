# Training data (compressed)

Everything the models were trained on, as produced by the scripts in `build/`. Unpack with `gunzip` into `build/work/` to re-run training or evaluation.

- `pairs_<MODEL>_{train,dev}.jsonl.gz`: final training / held-out pairs per model (`kind`, `lang`, `src`, `tgt`). `pairs_MAX_*` plus `pairs_Q_train` is what v6 was fine-tuned on.
- `bt_*.jsonl.gz`: English titles translated by a teacher and back (OPUS-MT, NLLB, Hy-MT2), before filtering.
- `kd_v5.jsonl.gz`: teachers' English answers for training sources.
- `abo_{pairs,train,test}.jsonl.gz`: real multilingual product names from Amazon Berkeley Objects; `abo_test` is the held-out real-name test.
- `titles_*.json.gz`, `abo_titles.json.gz`: English product titles.

Licenses: Amazon ESCI (Apache-2.0), Amazon Berkeley Objects (CC BY 4.0), translations by OPUS-MT (CC BY 4.0), M2M100 (MIT), Hy-MT2 (Apache-2.0) and NLLB (CC BY-NC 4.0: files with `nllb` in the name and the A / C / C2 / ALL3 pairs are non-commercial). See `LICENSES.md`.
