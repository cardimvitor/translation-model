# Training data (compressed)

Everything the models were trained on, as produced by the scripts in `build/`. Unpack with `gunzip` into `build/work/` to re-run training or evaluation.

- `pairs_<MODEL>_{train,dev}.jsonl.gz`: final training / held-out pairs per model (`kind`, `lang`, `src`, `tgt`). `pairs_MAX_*` plus `pairs_Q_train` is what v6 was fine-tuned on.
- `bt_*.jsonl.gz`: English titles translated by a teacher and back (OPUS-MT, NLLB, Hy-MT2), before filtering.
- `kd_v5.jsonl.gz`: teachers' English answers for training sources.
- `abo_{pairs,train,test}.jsonl.gz`: real multilingual product names from Amazon Berkeley Objects; `abo_test` is the held-out real-name test.
- `titles_*.json.gz`, `abo_titles.json.gz`: English product titles.

Licenses: Amazon ESCI (Apache-2.0), Amazon Berkeley Objects (CC BY 4.0), translations by OPUS-MT (CC BY 4.0), M2M100 (MIT), Hy-MT2 (Apache-2.0) and NLLB (CC BY-NC 4.0: files with `nllb` in the name and the A / C / C2 / ALL3 pairs are non-commercial). See `LICENSES.md`.


## v11

- `v11_judged.jsonl.gz`: 8,331 short labels v10 missed, each with the old training answer, v10's answer and Hy-MT2-7B's answer (`build/hy7_judge.py judge`).
- `v11_real.jsonl.gz`: 1,800 real Spanish and Japanese search queries (Amazon ESCI) with Hy-MT2-7B's translation.
- `pairs_NEW_v11.jsonl.gz`: the new training pairs made from those two files plus 9,600 mixed-language labels (`build/v11_pairs.py`).
- `drop_pairs.jsonl.gz`: 5,976 old pairs the 7B judged wrong; put it in `build/work/` and `build/ft_v8.py` leaves them out of the replayed data.

## v12

- `pairs_NEW_v12.jsonl.gz`: 44,000 mixed-language pairs for all 16 languages (`build/v12_pairs.py`). Training on them made ordinary labels worse, so the run was rejected and v12 does not use them; kept as a record and as tuning data for the word check.
