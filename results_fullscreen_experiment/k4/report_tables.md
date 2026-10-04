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

## Main comparison: gate (alpha = 0.5, k = 4)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 100.00 | 0.00 | 100.00 | 0.00 | -0.267 |
| Top-k localization | 7,10,11,14 | 80.08 | +19.92 [+15.23, +24.61] | 82.42 | +17.58 [+13.28, +22.66] | -0.748 |
| WMDP-vs-Retain | 0,1,4,5 | 61.33 | +38.67 [+33.20, +44.53] | 50.00 | +50.00 [+43.75, +55.86] | -1.037 |
| Bottom-k | 0,2,3,15 | 75.78 | +24.22 [+18.75, +29.69] | 69.53 | +30.47 [+25.00, +35.94] | -0.695 |
| Random (mean of 5) | varies | 74.61 | +25.39 [+22.34, +28.36] | 74.14 | +25.86 [+22.81, +29.06] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | -5.47 [-10.55, -0.08] | -8.28 [-12.89, -3.05] |
| WMDP-vs-Retain | +13.28 [+7.89, +19.06] | +24.14 [+18.20, +29.69] |
| Bottom-k | -1.17 [-6.09, +4.22] | +4.61 [-0.94, +10.08] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | -18.75 [-26.17, -11.72] |
| retain | -32.42 [-40.23, -23.83] |

Individual random selections: 5,6,8,14 -> WMDP 54.30 / retain 63.28; 6,7,10,12 -> WMDP 86.33 / retain 82.03; 1,3,11,14 -> WMDP 86.33 / retain 84.38; 0,2,6,14 -> WMDP 64.06 / retain 56.64; 4,13,14,15 -> WMDP 82.03 / retain 84.38

## Strength sweep (test split accuracy %)

**gate**

| Condition | a=0.0 | a=0.5 | a=1.0 |
| --- | ---: | ---: | ---: |
| Top-k localization WMDP | 100.00 | 80.08 | 25.00 |
| Top-k localization retain | 100.00 | 82.42 | 23.83 |
| WMDP-vs-Retain WMDP | 100.00 | 61.33 | 25.00 |
| WMDP-vs-Retain retain | 100.00 | 50.00 | 29.69 |
| Bottom-k WMDP | 100.00 | 75.78 | 25.39 |
| Bottom-k retain | 100.00 | 69.53 | 26.95 |
| Random (mean of 5) WMDP | 100.00 | 74.61 | 24.84 |
| Random (mean of 5) retain | 100.00 | 74.14 | 24.77 |

## General-biology control (64 MMLU high-school biology questions)

| Condition | Accuracy % | Drop pp [95% CI] |
| --- | ---: | ---: |
| Baseline | 100.00 | 0.00 |
| gate Top-k localization | 89.06 | +10.94 [+4.69, +18.75] |
| gate WMDP-vs-Retain | 39.06 | +60.94 [+48.44, +71.88] |
| gate Bottom-k | 68.75 | +31.25 [+20.31, +42.19] |
| gate Random (mean of 5) | 73.44 | +26.56 [+20.31, +32.81] |

## Prompt-format robustness (247 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 100.00 | 61.98 | +38.02 [+28.93, +47.11] | gate |
| harness | retain | 100.00 | 50.79 | +49.21 [+40.48, +57.94] | gate |
| chat_prefix | WMDP | 95.04 | 57.85 | +37.19 [+28.10, +47.11] | gate |
| chat_prefix | retain | 89.68 | 51.59 | +38.10 [+28.57, +47.62] | gate |
| chat_plain | WMDP | 42.15 | 21.49 | +20.66 [+12.40, +28.93] | gate |
| chat_plain | retain | 57.94 | 28.57 | +29.37 [+19.84, +38.10] | gate |

## Predicted answer letters on the WMDP test split

| Condition | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| gate | 82 | 76 | 63 | 35 |
| baseline | 56 | 70 | 66 | 64 |

