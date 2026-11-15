# Label normalizer: shipping and product labels → English, offline

Turns e-commerce and cargo labels (cities, countries, streets, addresses, product names) written in 16 languages into English, on a phone or a server, with no network calls.

Languages: zh es pt ar ru ja ko de fr hi tr id vi it pl th. You send a list of `[operation, text]` items, where the operation is `translate` (the default) or `transliterate`. You don't need to send a country or a field type.

| Model | What it is | Size | Speed |
|---|---|---|---|
| **English check** | Says whether a text is already English. | 0.1 MB | ~11 µs per call |
| **Fast** | Place-name list (GeoNames + CLDR), product word list (~1,900 English words and phrases, 700–1,500 entries per language), per-script transliteration, address rules. No neural network. | 21 MB of data, loaded per script on demand | ~1 ms per item |
| **Accurate** | Fast, plus a neural translator for labels Fast doesn't fully understand. Places, addresses and English labels stay with Fast. | +13–106 MB model, +11 MB onnxruntime-web | ~10–75 ms per item on one server thread |

Accurate comes in several variants. They differ in which teachers wrote the training data:

| Variant | Teachers and data | License of the model weights |
|---|---|---|
| **`student_V12` (default, final)** | `student_V11`'s weights, unchanged, plus a word-level English check: a label judged English that contains a word outside the model's English word list is given to the neural model, so "led ampul" or "office sandalye" is translated. A training run on 44k mixed-language pairs was tested and rejected (it lost accuracy) | **commercial use OK** (attribution required) |
| `student_V11` (parent of v12) | `student_V10` post-trained on data checked by Hy-MT2-7B (6k wrong pairs removed, 6k hard labels corrected), 1.6k real Spanish and Japanese queries and 9.6k mixed-language labels; adds the English double check and a `reliable` flag on every answer | **commercial use OK** (attribution required) |
| `student_V10` (parent of v11) | `student_V9` post-trained with 15k head terms written by Hy-MT2-1.8B in all 16 languages; decodes short labels with a 2-candidate beam search | **commercial use OK** (attribution required) |
| `student_V9` (parent of v10) | `student_V8` post-trained with 200k more short labels (every ESCI query in 7 OPUS-MT languages, more Korean and Thai from Hy-MT2-1.8B), made with `build/post_train.sh` | **commercial use OK** (attribution required) |
| `student_V8` ("v8 Complete", parent of v9) | v6's encoder + a 2-layer decoder + English-only output layer + one 1.6 MB encoder expert per language; all v6 data plus 13k short labels written by Hy-MT2-1.8B for ko/th/vi/ja/zh | **commercial use OK** (attribution required) |
| **`student_V8F` (final: "v8 Fast")** | cut from v8 Complete (3 encoder layers, 1 decoder layer, no experts); for quick validation | **commercial use OK** (attribution required) |
| `student_V6` (v6) | pretrained OPUS-MT mul-en (77M parameters) fine-tuned on the MAX data plus 58k short labels from Amazon ESCI search queries | **commercial use OK** (attribution required) |
| `student_V7` (v7) | v6 frozen, plus one 3 MB expert per language (mixture of experts); the language detector picks the expert, so only that language's extra parameters are loaded and used | **commercial use OK** (attribution required) |
| `student_MAX` (v5) | OPUS-MT + Tencent Hy-MT2-1.8B answers + 35k real multilingual product names (Amazon Berkeley Objects) | **commercial use OK** (attribution required) |
| `student_B2` (v5) | OPUS-MT, with its own English answers as extra targets | **commercial use OK** (attribution required) |
| `student_C2` (v5) | OPUS-MT + NLLB | **non-commercial** |
| `student_B` (v4) | OPUS-MT, plus M2M100-418M for Korean and Thai | **commercial use OK** (attribution required) |
| `student_A` (v4) | Meta NLLB-200-distilled-600M | **non-commercial** |
| `student_C` (v4) | NLLB + OPUS-MT | **non-commercial** |

`models/` also holds the experimental models compared in the report (`student_H`, `student_ALL3`, `student_HY`, `student_JEPA`, `student_BASE`, `student_v3`).

v6 is 65M parameters after trimming the vocabulary (43 + 63 MB int8); it uses SentencePiece, implemented in `runtime/spm.py` and `runtime/spm.js`. v5 models are 22M parameters (14 + 8 MB int8); v4 models are 13 MB. Choose one with `NEURAL_DIR=student_B2` (Python) or the `dir` argument of `Student.create` (JS).

## Results

Every number below comes from data not used for training. Exact = output equals an accepted English answer; chrF = character-level similarity, 0–100.

- **Labels:** 320 product labels hand-written for this project (20 products × 16 languages).
- **Real names:** 993 real marketplace product names from Amazon Berkeley Objects, held out before training.

| Accurate with | Labels: exact | Labels: chrF | Real names: chrF | Commercial use |
|---|---|---|---|---|
| Fast only | 20.0% | 42.6 | 39.1 | yes |
| v4 B | 24.1% | 46.5 | 37.4 | yes |
| v4 A | 28.1% | 49.6 | 37.7 | no |
| v4 C | **30.6%** | **51.0** | 38.4 | no |
| v5 B2 | 28.1% | 49.8 | 38.4 | yes |
| v5 C2 | 29.1% | 51.0 | 38.6 | no |
| v5 MAX | 28.4% | 49.1 | 45.7 | yes |
| **v6** | 65.6% | 79.6 | 66.9 | yes |
| v7 (v6 + language experts) | 65.9% | 80.8 | 67.2 | yes |
| **v12** | **76.2%** | **87.8** | 67.5 | yes |
| v11 | 75.9% | 87.6 | 67.6 | yes |
| v10 | 72.8% | 86.8 | 67.3 | yes |
| v9 | 72.5% | 86.0 | 67.2 | yes |
| v8 Complete | 68.8% | 83.1 | **67.7** | yes |
| **v8 Fast** | 65.0% | 79.7 | 62.2 | yes |

Speed on one CPU thread (median, model alone, short label / product name): v8 Complete 7.5 / 34 ms, v8 Fast 4.7 / 20 ms, v6 16.8 / 80 ms. Sizes at 8-bit: 57 MB, 42 MB, 106 MB. Details in `reports/v8/`.

Some test items also appear, by coincidence, as source text in the training data (60 of the 320 labels, 103 of the 993 real names; common labels such as "phone case" are in catalogues too). On the strictly unseen items v6 scores 60.8% / 76.3 on labels and 66.8 chrF on real names; v5 MAX scores 18.1% / 41.3 and 45.7.

**Base models on the same tests** (run alone, chrF on labels / real names): Hy-MT2-7B 95.1 / 57.5 (real names on a 330-name sample, where v6 scores 66.3), Hy-MT2-1.8B 79.1 / 51.9, OPUS-MT per-language models 76.2 / 47.1, NLLB-600M 70.9 / 48.7, OPUS-MT mul-en before fine-tuning 34.2 / 38.4. v6 alone scores 79.6 / 68.0. Raw outputs are in `reports/analysis/teachers/`.

Every model passes 70/70 tuned cases and 43–44/48 unseen ("blind") cases.

v6 runs in about 15–75 ms per label on one server core and 0.1–0.5 s in a single-thread browser; v5 MAX takes about 10–65 ms. The settings that decide when v6's answer replaces Fast's ship inside its `vocab.json` (`policy`) and were tuned on `test/dev_labels.json` (hand-written, separate from the test labels) and held-out titles.

v5 also changed when the neural model is used: if Fast understood the label (at least one known word) or it looks like an address, Fast's answer stays. Otherwise the neural model translates the whole label, and its answer is used if it is English and keeps the words Fast was sure of. The rule was tuned on held-out training data, not on the test sets.

The full comparison (12 models, the Hy-MT2-7B / 1.8B, OPUS-MT and NLLB teachers on the same labels, weight edits, and a JEPA experiment that did not beat plain training) is in `web/report/index.html`, generated by `build/make_report.py`. Raw results are in `reports/v5/`.

**Python vs JavaScript:** identical output on 3,018 of 3,018 Fast inputs (eager and lazy loading) and 12,072 of 12,072 English-check inputs. The JS SentencePiece tokenizer matches Python on every test input, and Python matches the real sentencepiece library on 60,000 of 60,000 training sources. For Accurate, 315–318 of 318 are identical. The rare difference is a near-tie in the neural decoder between onnxruntime-web and onnxruntime, on labels where both answers are wrong.

How the weights were tested and changed after training is in [reports/WEIGHTS.md](reports/WEIGHTS.md).

## Adding new things later

Three tested ways, each with an automatic pass/fail check: add words instantly (`build/add_terms.py`), post-train on new pairs (`build/post_train.sh`), or turn an English-only list into pairs (`build/make_pairs.py`). Step by step in [docs/ADDING.md](docs/ADDING.md). `student_V9`, `student_V10` and `student_V11` were all made with the post-training tool.

Against Hy-MT2 (each model alone, 8-bit as shipped): v11 scores 88.0 chrF on labels and 68.5 on real names at 57 MB; Hy-MT2-1.8B (1.1 GB) scores 79.1 and 51.9; Hy-MT2-7B (4.6 GB) scores 95.1 and 57.5. v11 is not better than the 7B on short labels overall.

### Labels with one word in another language (v12)

`test/mixed_tr.json` holds 64 hand-written English + Turkish labels (never trained or tuned on). Exact / chrF: v12 75.0% / 89.5, v11 68.8% / 84.6, v10 60.9% / 80.5, Hy-MT2-1.8B 59.4% / 82.8, Hy-MT2-7B 84.4% / 93.9. The word check is tuned by `build/tune_wordcheck.py` on data outside the tests (`policy.w_tau`, `w_tau_known`); the word lists are built by `build/en_words.py` and stored in the model's `vocab.json`. About 5% of labels that really are English now get a neural second opinion; labels in other languages take the same time as before.

### The `reliable` flag (v11, v12)

Every result of the v11 system carries `reliable: true/false`. True means a known place, clearly English text, every word known to the rules, or a neural answer with confidence of at least `policy.rel_tau` (0.82, set on tuning labels only). On the test labels 68% of answers are marked reliable; those score 87.7% exact / 95.5 chrF (Hy-MT2-7B on the same labels: 91.3% / 95.3), the rest 50.5% / 70.6. The flag does not remove errors: about one reliable answer in eight still differs from the reference. Recalibrate with `python build/tune_reliable.py <model> [target]`.

v11 also translates labels that mix English with another language (`test/mixed_labels.json`: 83.4 chrF, v10 80.8) and re-checks labels the English check is unsure about (`policy.en_sure`, `en_tau`, `en_sim`).

## Use it

### JavaScript (browser or Node)

```js
const kit = ENNormalizerKit.createNormalizer({
  baseUrl: "/en-normalizer/",      // where data/ and models/ are served
  model: "accurate",                // or "fast"
  variant: "student_B",             // student_A | student_B | student_C
  ort: window.ort,                  // onnxruntime-web 1.20.1, needed for "accurate"
  preload: "none",                  // load per-script data only when a batch needs it
});
const out = await kit.normalize([["translate", "Кожаная сумка женская"], "東京都新宿区西新宿2-8-1", ["transliterate", "ул. Ленина 5"]]);
// [{ output: "women's leather bag", method: "dictionary", language: [["ru", 0.99], ...], script: "Cyrillic", confidence: 0.9, ... }, ...]
```

The TypeScript types are in `runtime/index.d.ts`. Errors are `NormalizerError` with code `BAD_INPUT`, `LOAD_FAILED`, `NOT_LOADED` or `TIMEOUT`.

### Python

```sh
pip install -e .              # data/ and models/ are read from the checkout
pip install -e ".[accurate]"  # adds onnxruntime + numpy for the Accurate model
echo '[["translate","Москва"],["transliterate","東京"]]' > in.json
en-normalize in.json --model accurate --variant student_B
```

```python
import en_normalizer as N
N.normalize([["translate", "bolso de cuero negro para mujer"]], model="accurate", variant="student_B")
```

### Phone test page

`web/index.html`. Run `sh build/stage_web.sh`, then serve `web/dist/`; it has a model selector for A, B and C.

## Tests

```sh
python test/gate.py                     # fails if any tuned, blind or gold score drops below test/gate_baseline.json
python test/make_parity_inputs.py && npm test                      # Python = JS for Fast, lazy loading, English check
NEURAL_DIR=student_B python test/make_parity_accurate.py && NEURAL_DIR=student_B node test/parity_accurate.js
python test/eval_gold.py                # gold set: Fast, student alone, Accurate (NEURAL_DIR picks the variant)
```

GitHub Actions runs the gate and both parity checks on every push (`.github/workflows/ci.yml`).

## Rebuild everything

The scripts in `build/` regenerate every file in `data/` and `models/`. They need Python 3.12, plus torch, transformers, sentencepiece and onnxruntime.

`build/pipeline.sh` (v4) and `build/pipeline5.sh` → `pipeline6.sh` → `pipeline7.sh` (v5 and the JEPA experiment) and `pipeline8.sh` (v7 language experts: `pipeline10.sh`, `ft_moe.py`, `export_moe.py`; v6: `bt_queries.py`, `ft_marian.py`, then `export_marian.py`, `quant_marian.py`, `tune_routing3.py`) run these steps, and resume after a restart:

1. Back-translation with each teacher.
2. Growing the product word list. The automatic additions are kept only if no test case gets worse.
3. Training pairs for each variant.
4. Training (checkpoint every 5 min, cosine schedule over the whole run, average of the last 3 checkpoints).
5. Weight analysis and weight edits.
6. Final evaluation (`build/final_eval.sh`) and the report (`build/make_report.py`).

v5 adds teacher answers (`build/kd_v5.py`, `build/hy_data.py`), Amazon Berkeley Objects names (`build/fetch_abo.py`, `build/abo_prep.py`, `build/bt_abo.py`), mined product phrases (`build/expand_glossary2.py`), the decision-rule tuning (`build/tune_routing.py`) and JEPA (`build/jepa.py`, `build/jepa_lookup.py`). Hy-MT2 runs through llama-cpp-python.

The training data is 39k English product titles: Amazon ESCI (Apache-2.0), plus the v3 titles from Shopify/product-catalogue (Apache-2.0) and Qdrant/hm_ecommerce_products (CC BY 4.0). Each title is translated into 3 of the 16 languages by the variant's teacher. For B, an OPUS-MT translation is kept only if translating it back to English keeps at least half of the title's words.

The training set also includes:
- place names from the gazetteer, with 5% of cities held out;
- the product word list;
- accent-free copies of 25% of the Latin-script pairs.

## Licenses

See [LICENSES.md](LICENSES.md). In short:
- `student_V12`, `student_V11`, `student_V10`, `student_V9`, `student_V8`, `student_V8F`, `student_V6`, `student_V7`, `student_MAX`, `student_B2` and `student_B` can be used commercially, with attribution.
- `student_A`, `student_C` and `student_C2` are non-commercial (NLLB-derived).
- City names need GeoNames attribution.
- The code (plus the hand-written rules, word list and test sets) is MIT; see [LICENSE](LICENSE).

## What else is saved here

- `training_data/`: every training and held-out pair, the teacher translations and answers, and the real-name test set (gzip).
- `weights/student_V6_fp32/`: v6's full-precision weights in three parts (see its README).
- `reports/`: the PDF report, per-model analysis and weight-edit files, base-model outputs, training logs.
- `models/`: all 13 models at 8-bit.
