# Licenses of the data and models

| Part | Source | License |
|---|---|---|
| Country names | Unicode CLDR (via Babel) | Unicode License v3 |
| City names (`data/gazetteer/`) | GeoNames (allCountries, admin1CodesASCII, countryInfo, alternateNamesV2 from download.geonames.org); the Arabic and Hindi word readings in `data/rules.json` are also derived from GeoNames | CC BY 4.0, **attribution required** |
| Chinese readings | pypinyin | MIT |
| Japanese readings (`data/translit/ja.json`) | SudachiDict-core | Apache-2.0 |
| Thai word list | PyThaiNLP | Apache-2.0 |
| Fallback transliteration | any-ascii | ISC |
| Language-ID training text | NLTK data (UDHR, Europarl) | free use |
| English product titles (training only, not shipped) | Amazon ESCI (tasksource/esci); Shopify/product-catalogue; Qdrant/hm_ecommerce_products | Apache-2.0; Apache-2.0; CC BY 4.0 |
| Multilingual product names (training and held-out test, not shipped) | Amazon Berkeley Objects (ABO) | CC BY 4.0, **attribution required** |
| Teacher answers for H / ALL3 / HY / MAX (training only) | Tencent Hy-MT2-1.8B (GGUF via llama.cpp) | Apache-2.0 |
| `models/student_A` | trained on translations by Meta NLLB-200-distilled-600M | **CC BY-NC 4.0 derived: non-commercial** |
| `models/student_B` | trained on translations by Helsinki-NLP OPUS-MT (CC BY 4.0) and facebook/m2m100_418M (MIT) | commercial use OK, with attribution to OPUS-MT / Helsinki-NLP |
| `models/student_C` | both teachers | **non-commercial** (it includes NLLB-derived training data) |
| `models/student_B2` | trained on OPUS-MT translations and OPUS-MT answers | commercial use OK, with attribution to OPUS-MT / Helsinki-NLP |
| `models/student_C2` | OPUS-MT + NLLB | **non-commercial** (NLLB-derived) |
| `models/student_MAX` | OPUS-MT translations and answers; Tencent Hy-MT2-1.8B answers (Apache-2.0); Amazon Berkeley Objects product names (CC BY 4.0) | commercial use OK, with attribution to OPUS-MT / Helsinki-NLP and Amazon Berkeley Objects; Hy-MT2 Apache-2.0 notice |
| `models/student_V6` | fine-tuned from Helsinki-NLP/opus-mt-mul-en (CC BY 4.0) on the `student_MAX` data plus Amazon ESCI search queries (Apache-2.0) translated by OPUS-MT | commercial use OK, with attribution to OPUS-MT / Helsinki-NLP and Amazon Berkeley Objects; Hy-MT2 Apache-2.0 notice |
| `models/student_V7` | `student_V6` plus per-language experts trained on the same data | same as `student_V6` |
| `models/student_V10`, `models/student_V9`, `models/student_V8`, `models/student_V8F` | `student_V6` weights, fine-tuned on the same data plus short labels written by Tencent Hy-MT2-1.8B (Apache-2.0) | same as `student_V6` |
| `models/student_V14F1`, `models/student_V14F2` | `student_V8F` and `student_V14` weights (v2 retrained on the same data as v14) | same as `student_V6` |
| `models/student_V14` | the same weights as `student_V12` | same as `student_V6` |
| `models/student_V13` | the same weights as `student_V12` | same as `student_V6` |
| `models/student_V12` | the same weights as `student_V11`; its English and foreign word lists are counted from the training data listed above | same as `student_V6` |
| `models/student_V11` | `student_V10` weights, fine-tuned on the same data with corrections and real-query translations written by Tencent Hy-MT2-7B (Apache-2.0) | same as `student_V6` |
| Product word list, rules, test sets | written for this project | MIT (see LICENSE) |
| Code | this repository | MIT (see LICENSE) |

Whether a model trained on a teacher's outputs inherits the teacher's license is not settled law. The table follows the cautious reading. For commercial use, ship `student_V6`, `student_MAX`, `student_B2` or `student_B` and get legal advice.
