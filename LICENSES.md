# Licenses of the data and models

| Part | Source | License |
|---|---|---|
| Country names | Unicode CLDR (via Babel) | Unicode License v3 |
| City names (`data/gazetteer/`) | GeoNames (via geonamescache) | CC BY 4.0, **attribution required** |
| Chinese readings | pypinyin | MIT |
| Japanese readings (`data/translit/ja.json`) | SudachiDict-core | Apache-2.0 |
| Thai word list | PyThaiNLP | Apache-2.0 |
| Fallback transliteration | any-ascii | ISC |
| Language-ID training text | NLTK data (UDHR, Europarl) | free use |
| English product titles (training only, not shipped) | Amazon ESCI (tasksource/esci); Shopify/product-catalogue; Qdrant/hm_ecommerce_products | Apache-2.0; Apache-2.0; CC BY 4.0 |
| `models/student_A` | trained on translations by Meta NLLB-200-distilled-600M | **CC BY-NC 4.0 derived: non-commercial** |
| `models/student_B` | trained on translations by Helsinki-NLP OPUS-MT (CC BY 4.0) and facebook/m2m100_418M (MIT) | commercial use OK, with attribution to OPUS-MT / Helsinki-NLP |
| `models/student_C` | both teachers | **non-commercial** (it includes NLLB-derived training data) |
| Product word list, rules, test sets | written for this project | owner's choice |
| Code | this repository | **not set yet: the owner should add a LICENSE file** |

Whether a model trained on a teacher's outputs inherits the teacher's license is not settled law. The table follows the cautious reading. For commercial use, ship `student_B` and get legal advice.
