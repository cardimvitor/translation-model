"""Write dist/readmes.json (README text of the three zips)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = "Chinese, Spanish, Portuguese, Arabic, Russian, Japanese, Korean, German, French, Hindi, Turkish, Indonesian, Vietnamese, Italian, Polish, Thai"
check = f"""# English check (model 3)

Answers one question very fast: is this text already English?
Size: `data/encheck.json` 0.1 MB. Speed: about 0.01-0.02 ms per call in JavaScript.

## Use
JavaScript (browser or Node):
```js
await ENCheck.init(name => fetch("data/" + name).then(r => r.json()));   // Node: require("./runtime/encheck.js")
ENCheck.isEnglish("leather bag");             // {{english: true,  score, reason}}
ENCheck.isEnglish("bolsa de couro");          // {{english: false, ...}}
ENCheck.isEnglish("Hotel Roma", "relaxed");   // names and brands count as neutral
```
Python: `from encheck import is_english` (needs `runtime/normalizer.py` next to it for text helpers).

## Two modes
- `strict` (default): is the input already English, so translation can be skipped? Unknown names are refused, so it says "not English" when unsure.
- `relaxed`: does a translated output look like English? Names and brands count as neutral.

## How it works
- Any letter outside the English alphabet (accents, Cyrillic, CJK, ...) means not English.
- Otherwise each word is scored by 3,000 character n-grams (English vs 9 other Latin-script languages) and checked against 6,000 common English words. A small logistic model combines both.

## Results (held-out text)
| Mode | Length | English recognized | Foreign rejected |
|---|---|---|---|
| strict | 1 word | 87.2% | 93.9% |
| strict | 2-3 words | 96.8% | 98.1% |
| strict | 5 words | 99.4% | 99.4% |
| strict | 10 words | 100% | 100% |
| relaxed | place names | 90.7% | - |
| relaxed | Fast-model outputs | 94.0% | - |

Python and JavaScript give identical answers on 12,072 test inputs.

"""
fast = f"""# Fast model (model 1)

Translates or transliterates labels (countries, cities, streets, addresses, product names) into English, and identifies the source language.
Languages: {LANGS}. Unknown scripts fall back to generic romanization.
Size: about 28 MB of JSON data (most of it is the 341,000-name place gazetteer). Speed: about 0.2 ms per item on a laptop and about 1 ms on a phone.

## Use
Input is a list of `[operation, text]` pairs, where operation is `"translate"` (default) or `"transliterate"`.

JavaScript:
```js
await ENNormalizer.init(name => fetch("data/" + name).then(r => r.json()), ENCheck);
ENNormalizer.normalize([["translate", "Москва"], ["transliterate", "ул. Ленина 5"], ["translate", "Кожаная сумка женская"]]);
```
Python: `import normalizer; normalizer.normalize([("translate", "東京都新宿区西新宿2-8-1")])`

Each result has:
- `output`: the English text;
- `language`: the top 3 `[code, probability]`;
- `script`;
- `method`: gazetteer, dictionary, transliteration or already-english;
- `confidence`;
- `coverage`: the share of words found in the dictionaries.

Pass `ascii_out=True` (Python) or `true` (JS) to get plain ASCII output.

## How it works
1. Whole-text lookup in the place gazetteer: 341,000 country and city names in all languages (CLDR and GeoNames), with English exonyms.
2. English check: text that is already English is returned unchanged.
3. Language ID: character n-grams; the writing script decides when it is unique.
4. Per part (split on commas): segmentation (Chinese, Japanese, Thai), address words (street, district, province, ...) with English word order, a 91-term product glossary, and per-script transliteration:
   - Chinese: pinyin;
   - Japanese: Hepburn;
   - Korean: Revised Romanization;
   - Russian: BGN/PCGN;
   - Hindi: Devanagari with schwa deletion;
   - Arabic, Thai: their own tables;
   - anything else: generic romanization.

## Results
| Test set | Score |
|---|---|
| Tuned cases (70, used while building) | 69/70 |
| Blind cases (48, written before testing, never tuned on) | 41/48 (addresses 8/8, cities 13/14, countries 5/5, products 12/17) |
| 400 unseen e-commerce product titles | exact 8.5%, chrF 43.6 (product vocabulary is the weak point; see the Accurate model) |
| Python vs JavaScript parity | 3,018/3,018 identical |

Known limits:
- Arabic transliteration has no short vowels ("Fhd" for Fahd).
- Product words outside the glossary are only transliterated.
- Some exonyms are missing (Pattaya for transliteration, New Delhi).

"""
acc = f"""# Accurate model (model 2)

The Fast model plus a small neural translator (a "student" model) for product text. It runs fully on the phone.
- Model: character-level Transformer with 7.1M parameters (4 encoder and 3 decoder layers, width 256), int8 ONNX.
- Size: 7.2 MB for the model plus 11 MB for onnxruntime-web, on top of the Fast data.
- Speed: about 13 ms per product in Python and about 20-80 ms on a phone (wasm, 1 thread). Places and addresses take the Fast path, so they cost the same as in Fast.

## Use
JavaScript (load `ort/ort.wasm.min.js` first):
```js
ort.env.wasm.wasmPaths = new URL("ort/", location.href).href; ort.env.wasm.numThreads = 1;
await ENCheck.init(loadJson); await ENNormalizer.init(loadJson, ENCheck);
const S = await Student.create(ort, n => fetch("models/" + n).then(r => r.arrayBuffer()), n => fetch("models/" + n).then(r => r.json()), ENNormalizer._.lower);
Accurate.init(ENNormalizer, S, Student);
await Accurate.normalize([["translate", "schwarze Lederjacke"], ["translate", "ساعة ذهبية"]]);
```
Python: `import accurate; accurate.normalize([...])` (needs `onnxruntime` and `numpy`).

## Routing
1. Known place (gazetteer), transliterate operation, already English, or address-like text (digits or address words): use the Fast answer. Neural models wrongly translate street names.
2. Every word known to Fast, or no product word found: use the Fast answer.
3. Otherwise, the student translates the part. Its answer is kept only when all of these hold:
   - its confidence is ≥ 0.5;
   - it is not degenerate;
   - the English check (relaxed) passes;
   - it keeps every word Fast found in the dictionary.

   Otherwise the Fast answer is used, and `method` says `(neural rejected)`.

## How the student was trained (here, on 4 CPU cores)
- English product titles came from two public sets:
  - 830 Shopify catalogue titles;
  - 1,100 H&M product descriptions.

  To these, 1,100 glossary combinations were added.
- Each title was translated into 8 random languages (of the 16) by Meta NLLB-200 (600M), giving 24,240 pairs. Using this back-translation means the English side is always clean.
- Filters kept 21,249 pairs: script, not-already-English, length ratio and confidence.
- Training pairs also include glossary pairs and 22,679 place-name pairs.
- Training ran for 100 minutes (11,800 steps). 5% of the titles and 5% of the cities were held out for testing.

## Results
| Test | Fast | Student alone | Accurate |
|---|---|---|---|
| 400 unseen product titles, exact match | 8.5% | 24.6% | 18.8% |
| 400 unseen product titles, chrF | 43.6 | 54.0 | 51.8 |
| Hand-written products, tuned (21) | 21/21 | 8/21 | 21/21 |
| Hand-written products, blind (17) | 12/17 | 2/17 | 13/17 |
| All tuned cases (70) | 69/70 | - | 69/70 |
| All blind cases (48) | 41/48 | - | 42/48 |

The student-alone figures on the first two rows come from a separate held-out sample of 402 titles.

Accurate improves product titles in every one of the 16 languages:
- Japanese: chrF 30 → 53;
- Hindi: 48 → 64;
- Korean: 42 → 57;
- Spanish: 60 → 68.

Places and addresses keep the Fast results.

Parity:
- The JavaScript (onnxruntime-web) and Python pipelines give identical outputs on 318/318 inputs.
- The student alone matches on 312/318. The 6 differences are low-confidence garbage that the gate rejects.

## Honest limits
- The student has seen only about 2,700 distinct English titles, so its product vocabulary is small. It sometimes swaps the item noun with high confidence (for example "jacket" becomes "watch"). The routing gate catches this only when Fast knew the word.
- More training data is the main lever: more titles and more back-translation hours. A larger student (d=384) would also help.
- The student alone is not good at place names (chrF 36 on unseen cities); the gazetteer and transliteration handle those.

"""
json.dump({"check": check, "fast": fast, "accurate": acc}, open(os.path.join(ROOT, "dist", "readmes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
