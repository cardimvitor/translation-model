# Label normalizer: shipping and product labels → English, offline

Turns e-commerce and cargo labels (cities, countries, streets, addresses, product names) written in 16 languages into English, on a phone or a server, with no network calls.

Languages: zh es pt ar ru ja ko de fr hi tr id vi it pl th. You send a list of `[operation, text]` items, where the operation is `translate` (the default) or `transliterate`. You don't need to send a country or a field type.

| Model | What it is | Size | Speed |
|---|---|---|---|
| **English check** | Says whether a text is already English. | 0.1 MB | ~11 µs per call |
| **Fast** | Place-name list (GeoNames + CLDR), product word list (~980 words), per-script transliteration, address rules. No neural network. | 21 MB of data, loaded per script on demand | ~1 ms per item |
| **Accurate** | Fast, plus a small neural translator for words Fast doesn't know. The neural answer is accepted only if it keeps every word Fast already got right. | +13 MB model, +11 MB onnxruntime-web | ~11–17 ms per item on one thread |

Accurate comes in three variants. They differ only in which teacher translated the training data:

| Variant | Teacher | License of the model weights |
|---|---|---|
| `student_A` | Meta NLLB-200-distilled-600M | **non-commercial** (NLLB is CC BY-NC 4.0) |
| `student_B` (default) | Helsinki-NLP OPUS-MT, plus M2M100-418M for Korean and Thai | **commercial use OK** (OPUS-MT CC BY 4.0, M2M100 MIT; attribution required) |
| `student_C` | both of the above | **non-commercial** |

## Results

Every number below comes from data not used for training.

**Gold set:** 320 product labels hand-written for this project (20 products × 16 languages), never used for training. Exact = output equals an accepted English answer; chrF = character-level similarity, 0–100.

| | Exact | chrF |
|---|---|---|
| v3 Accurate (previous release) | 14.7% | 37.4 |
| Fast | 19.7% | 42.5 |
| Accurate, student B | 20.9% | 43.2 |
| Accurate, student A | 24.1% | 44.9 |
| Accurate, student C | **24.7%** | **45.4** |

**Hand-written test cases:**

| | Tuned cases | Unseen ("blind") cases |
|---|---|---|
| Fast | 70/70 | 43/48 |
| Accurate A | 70/70 | 45/48 |
| Accurate B | 70/70 | 43/48 |
| Accurate C | 70/70 | 44/48 |

**Held-out back-translated product titles** (student alone, chrF):

| v3 student | B | A | C |
|---|---|---|---|
| 16.8 | 34.1 | 40.1 | 40.1 |

The v3 figure was measured on B's held-out set.

**Python vs JavaScript:** identical output on 3,018 of 3,018 Fast inputs (eager and lazy loading) and 12,072 of 12,072 English-check inputs. For Accurate, 317–318 of 318 are identical. The rare difference is a near-tie in the neural decoder between onnxruntime-web and onnxruntime; in those cases JS falls back to the Fast answer.

How the weights were tested and changed after training is in [reports/WEIGHTS.md](reports/WEIGHTS.md).

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

`build/pipeline.sh` runs these steps, and resumes after a restart:

1. Back-translation with each teacher.
2. Growing the product word list. The automatic additions are kept only if no test case gets worse.
3. Training pairs for each variant.
4. Training (checkpoint every 5 min, cosine schedule over the whole run, average of the last 3 checkpoints).
5. Weight analysis and weight edits.

The training data is 39k English product titles: Amazon ESCI (Apache-2.0), plus the v3 titles from Shopify/product-catalogue (Apache-2.0) and Qdrant/hm_ecommerce_products (CC BY 4.0). Each title is translated into 3 of the 16 languages by the variant's teacher. For B, an OPUS-MT translation is kept only if translating it back to English keeps at least half of the title's words.

The training set also includes:
- place names from the gazetteer, with 5% of cities held out;
- the product word list;
- accent-free copies of 25% of the Latin-script pairs.

## Licenses

See [LICENSES.md](LICENSES.md). In short:
- Variant B can be used commercially, with attribution.
- Variants A and C are non-commercial.
- City names need GeoNames attribution.
- The code (plus the hand-written rules, word list and test sets) is MIT; see [LICENSE](LICENSE).
