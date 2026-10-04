## Localization scores

| Layer | gate: forget | gate: selective |
| --- | ---: | ---: |
| 0 | -0.0277 | +0.0391 |
| 1 | +0.0260 | +0.0701 |
| 2 | -0.0428 | +0.0220 |
| 3 | -0.0507 | +0.0040 |
| 4 | +0.0248 | +0.0847 |
| 5 | -0.0082 | +0.1040 |
| 6 | +0.0728 | +0.0005 |
| 7 | +0.1663 | +0.0120 |
| 8 | +0.0286 | -0.0162 |
| 9 | +0.0433 | -0.0254 |
| 10 | +0.1766 | +0.0072 |
| 11 | +0.1016 | +0.0055 |
| 12 | -0.0245 | -0.0163 |
| 13 | +0.0398 | -0.0258 |
| 14 | +0.0993 | +0.0341 |
| 15 | -0.5531 | -0.0048 |


Is the selected layer separable from the runner-up?

| Method | Score | Best layer | Runner-up | Gap | Gap in standard errors | Layers within 1 s.e. | Layer chosen by each half | Questions needed for a 2 s.e. gap |
| --- | --- | :---: | :---: | ---: | ---: | --- | :---: | ---: |
| gate | $F_\ell$ | 10 | 7 | +0.0103 | **0.32** | 7, 10 | 10 vs 10 | 5068 |
| gate | $R_\ell$ | 10 | 7 | +0.0151 | **0.40** | 7, 10 | 10 vs 10 | 3122 |
| gate | $S_\ell$ | 5 | 4 | +0.0193 | **0.28** | 0, 1, 4, 5 | 1 vs 5 | 6321 |

## Main comparison: gate (alpha = 1, k = 2)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 100.00 | 0.00 | 100.00 | 0.00 | -0.267 |
| Top-k localization | 7,10 | 28.91 | +71.09 [+65.62, +76.56] | 28.12 | +71.88 [+66.80, +77.34] | -1.530 |
| WMDP-vs-Retain | 4,5 | 23.05 | +76.95 [+71.88, +82.03] | 21.09 | +78.91 [+73.83, +83.98] | -2.130 |
| Bottom-k | 3,15 | 46.09 | +53.91 [+48.05, +59.77] | 43.75 | +56.25 [+50.39, +62.50] | -1.947 |
| Random (mean of 5) | varies | 44.38 | +55.62 [+53.12, +58.13] | 39.14 | +60.86 [+58.13, +63.52] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | +15.47 [+9.69, +21.17] | +11.02 [+5.63, +16.72] |
| WMDP-vs-Retain | +21.33 [+16.17, +26.33] | +18.05 [+12.89, +23.36] |
| Bottom-k | -1.72 [-7.27, +4.06] | -4.61 [-10.78, +1.87] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | -5.86 [-12.89, +1.17] |
| retain | -7.03 [-14.45, +0.78] |

Individual random selections: 6,14 -> WMDP 42.19 / retain 42.58; 10,12 -> WMDP 66.41 / retain 46.48; 1,3 -> WMDP 25.78 / retain 23.83; 2,14 -> WMDP 39.06 / retain 41.41; 4,15 -> WMDP 48.44 / retain 41.41

## Strength sweep (test split accuracy %)

**gate**

| Condition | a=0.0 | a=0.5 | a=1.0 |
| --- | ---: | ---: | ---: |
| Top-k localization WMDP | 100.00 | 83.98 | 28.91 |
| Top-k localization retain | 100.00 | 85.94 | 28.12 |
| WMDP-vs-Retain WMDP | 100.00 | 82.42 | 23.05 |
| WMDP-vs-Retain retain | 100.00 | 83.98 | 21.09 |
| Bottom-k WMDP | 100.00 | 90.62 | 46.09 |
| Bottom-k retain | 100.00 | 90.23 | 43.75 |
| Random (mean of 5) WMDP | 100.00 | 91.09 | 44.38 |
| Random (mean of 5) retain | 100.00 | 90.00 | 39.14 |

## General-biology control (64 MMLU high-school biology questions)

| Condition | Accuracy % | Drop pp [95% CI] |
| --- | ---: | ---: |
| Baseline | 100.00 | 0.00 |
| gate Top-k localization | 32.81 | +67.19 [+56.25, +78.12] |
| gate WMDP-vs-Retain | 18.75 | +81.25 [+70.31, +90.62] |
| gate Bottom-k | 31.25 | +68.75 [+56.25, +79.69] |
| gate Random (mean of 5) | 33.44 | +66.56 [+60.94, +71.88] |

## Prompt-format robustness (247 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 100.00 | 22.31 | +77.69 [+70.25, +85.12] | gate |
| harness | retain | 100.00 | 23.02 | +76.98 [+69.05, +84.13] | gate |
| chat_prefix | WMDP | 95.04 | 23.14 | +71.90 [+63.64, +80.17] | gate |
| chat_prefix | retain | 89.68 | 28.57 | +61.11 [+51.59, +70.63] | gate |
| chat_plain | WMDP | 42.15 | 22.31 | +19.83 [+12.40, +28.10] | gate |
| chat_plain | retain | 57.94 | 23.81 | +34.13 [+25.40, +42.86] | gate |

## Predicted answer letters on the WMDP test split

| Condition | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| gate | 215 | 27 | 14 | 0 |
| baseline | 56 | 70 | 66 | 64 |

