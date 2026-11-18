# Testing and changing the trained weights

After training, each student went through the same steps (`build/analyze_student.py`, `build/weight_edits.py`, `build/train_student2.py`). An edit was kept only if it raised the score on data not used to find it.

Two scores are used below:
- **Product chrF:** held-out back-translated titles.
- **Places chrF:** held-out city names.

All scores are 0–100.

## 1. Checkpoint averaging (last 3 checkpoints, 30 min apart)

Each variant has two rows: the last checkpoint, and the average of the last 3.

| Variant | Weights | Product chrF | Places chrF | Used |
|---|---|---|---|---|
| B | last | 33.63 | 35.64 | |
| B | average of 3 | 34.10 | 34.33 | yes |
| A | last | 39.90 | 33.74 | |
| A | average of 3 | 40.06 | 35.08 | yes |
| C | last | 39.94 | 37.05 | |
| C | average of 3 | 40.06 | 37.09 | yes |

## 2. Switching parts of the network off (ablation), then a greedy search

The ablation zeroed each part one at a time: the 36 attention heads, then each feed-forward block and each whole layer, 66 in all. It was measured on 500 held-out items.

Candidates whose removal helped were then combined greedily on a *separate* confirmation slice (900 titles and 300 places). An edit stayed only if the combined score (product + 0.3 × places) rose by more than 0.2.

The final edited model also had to stay within 0.3 chrF on the 320-item hand-written gold set (student alone).

| Variant | Kept edit | Product / places, before → after | Gold exact / chrF, before → after | Decision |
|---|---|---|---|---|
| B | zero `enc0.self.head3`, `enc4.self.head3` | 33.70 / 35.90 → 33.78 / 37.37 | 9.7% / 28.86 → 8.8% / 29.16 | **edited weights used** |
| A | zero `enc3.self.head2` | 38.70 / 37.84 → 38.99 / 37.83 | 20.6% / 40.07 → 19.1% / 38.63 | rejected: gold fell by 1.4 |
| C | zero `dec0.self.head0` | 40.74 / 35.12 → 40.86 / 35.48 | 18.4% / 40.29 → 20.0% / 40.56 | **edited weights used** |

Most single-head gains in the 500-item ablation did not survive the confirmation slice (B: 2 of 21 candidates kept; A: 1 of 12; C: 1 of 6). They were noise.

## 3. Short fine-tune with the pruned weights held at zero (15 min)

| Variant | Product / places | Gold exact / chrF | Used |
|---|---|---|---|
| B | 33.57 / 36.88 | 10.0% / 29.87 | no: tied with edit-only under the decision rule |
| A | 38.29 / 38.00 | 16.9% / 37.98 | no |
| C | 40.55 / 35.60 | 19.7% / 40.37 | no |

## 4. 8-bit (int8) vs full-precision (fp32) weights (gold set)

| Model | Student alone | Inside Accurate | Speed |
|---|---|---|---|
| A, int8 | 20.3% / 39.9 | 23.1% / 44.4 | 9.9 ms |
| A, fp32 | 20.6% / 40.1 | 23.1% / 44.3 | 20.8 ms |
| C, int8 | 20.6% / 40.7 | 23.4% / 44.7 | 9.3 ms |
| C, fp32 | 20.0% / 40.6 | 23.4% / 44.4 | 20.6 ms |

int8 loses nothing and is twice as fast, so all shipped models are int8 (13 MB each).

## Not done

- **Mixing weights across A, B and C ("soups"):** each variant has its own character vocabulary, so the weights can't be averaged directly. B also must not take in NLLB-derived weights, or it would lose its commercial license.
- **Output-layer biasing:** not tried.

## Caveat

The gold set was used as a pass/fail gate in step 2, though never for choosing *which* edit to try. That is mild selection on the gold set; the tuned and blind case sets were not used at all.

Raw logs: `reports/analysis_student_*.json`, `reports/edits_student_*.json`.
