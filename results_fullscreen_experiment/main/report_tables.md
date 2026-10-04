## Localization scores

| Layer | gate: forget | gate: selective | gate_margin: forget | gate_margin: selective |
| --- | ---: | ---: | ---: | ---: |
| 0 | -0.0277 | +0.0391 | -0.2292 | +0.2649 |
| 1 | +0.0260 | +0.0701 | +0.0237 | +0.3481 |
| 2 | -0.0428 | +0.0220 | -0.3834 | +0.0371 |
| 3 | -0.0507 | +0.0040 | -0.1816 | -0.0478 |
| 4 | +0.0248 | +0.0847 | +0.2030 | +0.2988 |
| 5 | -0.0082 | +0.1040 | -0.2262 | +0.3233 |
| 6 | +0.0728 | +0.0005 | +0.4555 | -0.0456 |
| 7 | +0.1663 | +0.0120 | +1.2239 | +0.1358 |
| 8 | +0.0286 | -0.0162 | +0.2662 | +0.0098 |
| 9 | +0.0433 | -0.0254 | +0.1625 | -0.3695 |
| 10 | +0.1766 | +0.0072 | +1.2521 | +0.3023 |
| 11 | +0.1016 | +0.0055 | +0.4734 | +0.1184 |
| 12 | -0.0245 | -0.0163 | -0.1126 | -0.1393 |
| 13 | +0.0398 | -0.0258 | +0.3578 | +0.0163 |
| 14 | +0.0993 | +0.0341 | +0.6911 | +0.1508 |
| 15 | -0.5531 | -0.0048 | -3.5887 | -0.3040 |


Is the selected layer separable from the runner-up?

| Method | Score | Best layer | Runner-up | Gap | Gap in standard errors | Layers within 1 s.e. | Layer chosen by each half | Questions needed for a 2 s.e. gap |
| --- | --- | :---: | :---: | ---: | ---: | --- | :---: | ---: |
| gate | $F_\ell$ | 10 | 7 | +0.0103 | **0.32** | 7, 10 | 10 vs 10 | 5068 |
| gate | $R_\ell$ | 10 | 7 | +0.0151 | **0.40** | 7, 10 | 10 vs 10 | 3122 |
| gate | $S_\ell$ | 5 | 4 | +0.0193 | **0.28** | 0, 1, 4, 5 | 1 vs 5 | 6321 |
| gate_margin | $F_\ell$ | 10 | 7 | +0.0282 | **0.16** | 7, 10 | 7 vs 10 | 19277 |
| gate_margin | $R_\ell$ | 7 | 10 | +0.1382 | **0.81** | 7, 10 | 7 vs 7 | 781 |
| gate_margin | $S_\ell$ | 1 | 5 | +0.0248 | **0.09** | 0, 1, 4, 5, 7, 10, 14 | 6 vs 5 | 63834 |

## Main comparison: gate (alpha = 0.25, k = 1)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 100.00 | 0.00 | 100.00 | 0.00 | -0.267 |
| Top-k localization | 10 | 99.61 | +0.39 [+0.00, +1.17] | 97.66 | +2.34 [+0.78, +4.30] | -0.310 |
| WMDP-vs-Retain | 5 | 97.66 | +2.34 [+0.78, +4.30] | 96.48 | +3.52 [+1.56, +5.86] | -0.297 |
| Bottom-k | 15 | 99.22 | +0.78 [+0.00, +1.95] | 97.27 | +2.73 [+1.17, +4.69] | -0.157 |
| Random (mean of 5) | varies | 98.28 | +1.72 [+0.94, +2.58] | 98.20 | +1.80 [+1.02, +2.66] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | -1.33 [-2.19, -0.47] | +0.55 [-1.02, +2.34] |
| WMDP-vs-Retain | +0.62 [-1.17, +2.66] | +1.72 [-0.23, +3.91] |
| Bottom-k | -0.94 [-1.95, +0.16] | +0.94 [-0.70, +2.81] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | -1.95 [-3.91, +0.00] |
| retain | -1.17 [-3.52, +1.17] |

Individual random selections: 6 -> WMDP 97.66 / retain 98.83; 12 -> WMDP 99.61 / retain 100.00; 1 -> WMDP 98.05 / retain 97.66; 2 -> WMDP 98.83 / retain 97.27; 4 -> WMDP 97.27 / retain 97.27

## Main comparison: gate_margin (alpha = 1, k = 1)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 100.00 | 0.00 | 100.00 | 0.00 | -0.267 |
| Top-k localization | 10 | 75.39 | +24.61 [+19.53, +30.08] | 63.67 | +36.33 [+30.47, +42.58] | -0.763 |
| WMDP-vs-Retain | 1 | 25.78 | +74.22 [+69.14, +79.30] | 26.95 | +73.05 [+67.58, +78.52] | -2.372 |
| Bottom-k | 15 | 95.70 | +4.30 [+1.95, +7.03] | 95.70 | +4.30 [+2.34, +6.64] | -0.082 |
| Random (mean of 5) | varies | 47.73 | +52.27 [+49.53, +55.00] | 46.17 | +53.83 [+50.78, +56.88] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | -27.66 [-33.20, -22.03] | -17.50 [-23.91, -10.63] |
| WMDP-vs-Retain | +21.95 [+16.72, +27.27] | +19.22 [+13.98, +24.53] |
| Bottom-k | -47.97 [-51.72, -43.75] | -49.53 [-53.52, -45.31] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | -49.61 [-57.42, -42.19] |
| retain | -36.72 [-44.53, -28.52] |

Individual random selections: 6 -> WMDP 41.41 / retain 41.80; 12 -> WMDP 71.48 / retain 73.44; 1 -> WMDP 25.78 / retain 26.95; 2 -> WMDP 49.61 / retain 48.05; 4 -> WMDP 50.39 / retain 40.62

## Strength sweep (test split accuracy %)

**gate**

| Condition | a=0.0 | a=0.25 | a=0.5 | a=0.75 | a=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-k localization WMDP | 100.00 | 99.61 | 95.70 | 91.02 | 75.39 |
| Top-k localization retain | 100.00 | 97.66 | 96.09 | 88.67 | 63.67 |
| WMDP-vs-Retain WMDP | 100.00 | 97.66 | 88.28 | 70.70 | 26.95 |
| WMDP-vs-Retain retain | 100.00 | 96.48 | 90.23 | 70.70 | 30.86 |
| Bottom-k WMDP | 100.00 | 99.22 | 97.66 | 97.27 | 95.70 |
| Bottom-k retain | 100.00 | 97.27 | 96.88 | 96.09 | 95.70 |
| Random (mean of 5) WMDP | 100.00 | 98.28 | 93.20 | 81.48 | 47.73 |
| Random (mean of 5) retain | 100.00 | 98.20 | 93.20 | 79.45 | 46.17 |

**gate_margin**

| Condition | a=0.0 | a=0.25 | a=0.5 | a=0.75 | a=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-k localization WMDP | 100.00 | 99.61 | 95.70 | 91.02 | 75.39 |
| Top-k localization retain | 100.00 | 97.66 | 96.09 | 88.67 | 63.67 |
| WMDP-vs-Retain WMDP | 100.00 | 98.05 | 92.97 | 83.59 | 25.78 |
| WMDP-vs-Retain retain | 100.00 | 97.66 | 92.97 | 77.34 | 26.95 |
| Bottom-k WMDP | 100.00 | 99.22 | 97.66 | 97.27 | 95.70 |
| Bottom-k retain | 100.00 | 97.27 | 96.88 | 96.09 | 95.70 |
| Random (mean of 5) WMDP | 100.00 | 98.28 | 93.20 | 81.48 | 47.73 |
| Random (mean of 5) retain | 100.00 | 98.20 | 93.20 | 79.45 | 46.17 |

## General-biology control (64 MMLU high-school biology questions)

| Condition | Accuracy % | Drop pp [95% CI] |
| --- | ---: | ---: |
| Baseline | 100.00 | 0.00 |
| gate Top-k localization | 100.00 | +0.00 [+0.00, +0.00] |
| gate WMDP-vs-Retain | 98.44 | +1.56 [+0.00, +4.69] |
| gate Bottom-k | 100.00 | +0.00 [+0.00, +0.00] |
| gate Random (mean of 5) | 98.44 | +1.56 [+0.00, +3.44] |
| gate_margin Top-k localization | 59.38 | +40.62 [+28.12, +53.12] |
| gate_margin WMDP-vs-Retain | 23.44 | +76.56 [+65.62, +85.94] |
| gate_margin Bottom-k | 98.44 | +1.56 [+0.00, +4.69] |
| gate_margin Random (mean of 5) | 44.06 | +55.94 [+49.69, +61.88] |

## Prompt-format robustness (247 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 100.00 | 100.00 | +0.00 [+0.00, +0.00] | gate |
| harness | retain | 100.00 | 94.44 | +5.56 [+1.59, +10.32] | gate |
| harness | WMDP | 100.00 | 27.27 | +72.73 [+65.29, +80.17] | gate_margin |
| harness | retain | 100.00 | 31.75 | +68.25 [+60.32, +75.40] | gate_margin |
| chat_prefix | WMDP | 95.04 | 96.69 | -1.65 [-4.13, +0.00] | gate |
| chat_prefix | retain | 89.68 | 90.48 | -0.79 [-3.17, +1.59] | gate |
| chat_prefix | WMDP | 95.04 | 33.06 | +61.98 [+52.07, +71.90] | gate_margin |
| chat_prefix | retain | 89.68 | 27.78 | +61.90 [+52.38, +71.43] | gate_margin |
| chat_plain | WMDP | 42.15 | 36.36 | +5.79 [+0.00, +11.57] | gate |
| chat_plain | retain | 57.94 | 51.59 | +6.35 [+1.59, +11.11] | gate |
| chat_plain | WMDP | 42.15 | 29.75 | +12.40 [-1.65, +25.62] | gate_margin |
| chat_plain | retain | 57.94 | 18.25 | +39.68 [+27.78, +50.79] | gate_margin |

## Does the first-order score predict its own objective? (method gate, test split)

| Condition | Layer | alpha | Role | Predicted drop in log p | Actual drop in log p | Actual accuracy drop pp |
| --- | :---: | :---: | --- | ---: | ---: | ---: |
| Top-k localization | 10 | 0.25 | WMDP | +0.044 | +0.043 | +0.39 |
| Top-k localization | 10 | 0.25 | retain | +0.042 | +0.035 | +2.34 |
| Top-k localization | 10 | 0.5 | WMDP | +0.088 | +0.092 | +4.30 |
| Top-k localization | 10 | 0.5 | retain | +0.085 | +0.082 | +3.91 |
| Top-k localization | 10 | 0.75 | WMDP | +0.132 | +0.181 | +8.98 |
| Top-k localization | 10 | 0.75 | retain | +0.127 | +0.188 | +11.33 |
| Top-k localization | 10 | 1 | WMDP | +0.177 | +0.496 | +24.61 |
| Top-k localization | 10 | 1 | retain | +0.169 | +0.620 | +36.33 |
| Bottom-k | 15 | 0.25 | WMDP | -0.138 | -0.110 | +0.78 |
| Bottom-k | 15 | 0.25 | retain | -0.137 | -0.105 | +2.73 |
| Bottom-k | 15 | 0.5 | WMDP | -0.277 | -0.168 | +2.34 |
| Bottom-k | 15 | 0.5 | retain | -0.274 | -0.164 | +3.12 |
| Bottom-k | 15 | 0.75 | WMDP | -0.415 | -0.187 | +2.73 |
| Bottom-k | 15 | 0.75 | retain | -0.411 | -0.180 | +3.91 |
| Bottom-k | 15 | 1 | WMDP | -0.553 | -0.185 | +4.30 |
| Bottom-k | 15 | 1 | retain | -0.548 | -0.170 | +4.30 |

## Predicted answer letters on the WMDP test split

| Condition | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| gate | 59 | 68 | 65 | 64 |
| gate_margin | 39 | 118 | 60 | 39 |
| baseline | 56 | 70 | 66 | 64 |

