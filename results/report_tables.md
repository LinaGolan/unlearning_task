## Localization scores

| Layer | gate: forget | gate: selective | gate_margin: forget | gate_margin: selective |
| --- | ---: | ---: | ---: | ---: |
| 0 | +0.0993 | +0.0782 | -0.2524 | -0.0811 |
| 1 | -0.0227 | -0.0314 | +0.0391 | +0.1058 |
| 2 | +0.0304 | +0.0851 | +0.1305 | +0.5000 |
| 3 | +0.0468 | +0.1403 | -0.1400 | +0.0969 |
| 4 | +0.0501 | +0.1052 | -0.2832 | -0.3056 |
| 5 | -0.0048 | +0.0002 | +0.0087 | +0.3225 |
| 6 | -0.1729 | -0.1691 | -0.0310 | -0.2770 |
| 7 | +0.0585 | +0.0612 | +0.5012 | -0.0103 |
| 8 | +0.0293 | -0.0484 | +0.0886 | -0.2268 |
| 9 | -0.1030 | -0.0934 | -0.0584 | -0.2274 |
| 10 | -0.0601 | -0.0995 | +0.4473 | -0.0156 |
| 11 | -0.0321 | -0.0261 | +0.0747 | -0.0449 |
| 12 | -0.0068 | +0.0133 | -0.0635 | -0.0047 |
| 13 | +0.0322 | -0.0036 | +0.1632 | -0.0846 |
| 14 | -0.0755 | -0.0309 | +0.2132 | +0.1062 |
| 15 | +0.3085 | +0.2511 | -0.9703 | +0.0980 |


Is the selected layer separable from the runner-up?

| Method | Score | Best layer | Runner-up | Gap | Gap in standard errors | Layers within 1 s.e. | Layer chosen by each half | Questions needed for a 2 s.e. gap |
| --- | --- | :---: | :---: | ---: | ---: | --- | :---: | ---: |
| gate | $F_\ell$ | 15 | 0 | +0.2092 | **1.46** | 15 | 4 vs 15 | 240 |
| gate | $R_\ell$ | 8 | 15 | +0.0204 | **0.20** | 0, 6, 8, 10, 13, 15 | 15 vs 8 | 13208 |
| gate | $S_\ell$ | 15 | 3 | +0.1108 | **0.63** | 0, 2, 3, 4, 15 | 4 vs 15 | 1297 |
| gate_margin | $F_\ell$ | 7 | 10 | +0.0539 | **0.31** | 7, 10 | 10 vs 7 | 5168 |
| gate_margin | $R_\ell$ | 7 | 10 | +0.0486 | **0.29** | 7, 10 | 10 vs 7 | 6059 |
| gate_margin | $S_\ell$ | 2 | 5 | +0.1775 | **0.69** | 2, 5, 15 | 5 vs 15 | 1061 |

## Main comparison: gate (alpha = 0.25, k = 1)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 60.55 | 0.00 | 53.91 | 0.00 | -1.066 |
| Top-k localization | 15 | 58.59 | +1.95 [+0.00, +4.30] | 53.52 | +0.39 [-2.34, +3.12] | -1.210 |
| WMDP-vs-Retain | 15 | 58.59 | +1.95 [+0.00, +4.30] | 53.52 | +0.39 [-2.34, +3.12] | -1.210 |
| Bottom-k | 6 | 57.03 | +3.52 [+0.00, +6.64] | 51.95 | +1.95 [-1.56, +5.47] | -1.053 |
| Random (mean of 5) | varies | 58.67 | +1.88 [-0.00, +3.75] | 52.66 | +1.25 [-0.70, +3.12] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | +0.08 [-2.34, +2.66] | -0.86 [-3.59, +1.88] |
| WMDP-vs-Retain | +0.08 [-2.34, +2.66] | -0.86 [-3.59, +1.88] |
| Bottom-k | +1.64 [-0.78, +3.98] | +0.70 [-2.03, +3.44] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | +0.00 [+0.00, +0.00] |
| retain | +0.00 [+0.00, +0.00] |

Individual random selections: 6 -> WMDP 57.03 / retain 51.95; 12 -> WMDP 60.55 / retain 54.69; 1 -> WMDP 58.98 / retain 52.34; 2 -> WMDP 58.98 / retain 53.52; 4 -> WMDP 57.81 / retain 50.78

## Main comparison: gate_margin (alpha = 0.75, k = 1)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 60.55 | 0.00 | 53.91 | 0.00 | -1.066 |
| Top-k localization | 7 | 49.61 | +10.94 [+4.69, +17.58] | 45.70 | +8.20 [+1.95, +14.84] | -1.183 |
| WMDP-vs-Retain | 2 | 55.47 | +5.08 [-0.78, +10.94] | 48.83 | +5.08 [-0.78, +10.94] | -1.115 |
| Bottom-k | 15 | 57.81 | +2.73 [+0.00, +5.47] | 53.52 | +0.39 [-3.12, +3.91] | -1.923 |
| Random (mean of 5) | varies | 52.73 | +7.81 [+3.91, +11.80] | 45.47 | +8.44 [+4.45, +12.27] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | +3.12 [-2.42, +8.59] | -0.23 [-5.62, +5.63] |
| WMDP-vs-Retain | -2.73 [-6.56, +1.17] | -3.36 [-7.11, +0.55] |
| Bottom-k | -5.08 [-9.45, -0.62] | -8.05 [-12.50, -3.52] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | +5.86 [-0.78, +12.50] |
| retain | +3.12 [-3.91, +10.55] |

Individual random selections: 6 -> WMDP 48.83 / retain 41.41; 12 -> WMDP 58.98 / retain 51.17; 1 -> WMDP 52.34 / retain 43.75; 2 -> WMDP 55.47 / retain 48.83; 4 -> WMDP 48.05 / retain 42.19

## Strength sweep (test split accuracy %)

**gate**

| Condition | a=0.0 | a=0.25 | a=0.5 | a=0.75 | a=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-k localization WMDP | 60.55 | 58.59 | 58.20 | 57.81 | 57.42 |
| Top-k localization retain | 53.91 | 53.52 | 53.52 | 53.52 | 52.73 |
| WMDP-vs-Retain WMDP | 60.55 | 58.59 | 58.20 | 57.81 | 57.42 |
| WMDP-vs-Retain retain | 53.91 | 53.52 | 53.52 | 53.52 | 52.73 |
| Bottom-k WMDP | 60.55 | 57.03 | 54.69 | 48.83 | 33.59 |
| Bottom-k retain | 53.91 | 51.95 | 51.56 | 41.41 | 30.08 |
| Random (mean of 5) WMDP | 60.55 | 58.67 | 56.41 | 52.73 | 37.27 |
| Random (mean of 5) retain | 53.91 | 52.66 | 52.58 | 45.47 | 31.56 |

**gate_margin**

| Condition | a=0.0 | a=0.25 | a=0.5 | a=0.75 | a=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-k localization WMDP | 60.55 | 58.98 | 52.73 | 49.61 | 40.62 |
| Top-k localization retain | 53.91 | 54.30 | 50.39 | 45.70 | 35.94 |
| WMDP-vs-Retain WMDP | 60.55 | 58.98 | 58.98 | 55.47 | 39.45 |
| WMDP-vs-Retain retain | 53.91 | 53.52 | 51.95 | 48.83 | 31.25 |
| Bottom-k WMDP | 60.55 | 58.59 | 58.20 | 57.81 | 57.42 |
| Bottom-k retain | 53.91 | 53.52 | 53.52 | 53.52 | 52.73 |
| Random (mean of 5) WMDP | 60.55 | 58.67 | 56.41 | 52.73 | 37.27 |
| Random (mean of 5) retain | 53.91 | 52.66 | 52.58 | 45.47 | 31.56 |

## General-biology control (64 MMLU high-school biology questions)

| Condition | Accuracy % | Drop pp [95% CI] |
| --- | ---: | ---: |
| Baseline | 51.56 | 0.00 |
| gate Top-k localization | 50.00 | +1.56 [+0.00, +4.69] |
| gate WMDP-vs-Retain | 50.00 | +1.56 [+0.00, +4.69] |
| gate Bottom-k | 48.44 | +3.12 [-3.12, +9.38] |
| gate Random (mean of 5) | 49.38 | +2.19 [-0.94, +5.94] |
| gate_margin Top-k localization | 43.75 | +7.81 [-4.69, +20.31] |
| gate_margin WMDP-vs-Retain | 43.75 | +7.81 [-4.69, +20.31] |
| gate_margin Bottom-k | 50.00 | +1.56 [+0.00, +4.69] |
| gate_margin Random (mean of 5) | 40.94 | +10.63 [+2.81, +19.06] |

## Prompt-format robustness (242 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 59.35 | 56.10 | +3.25 [+0.81, +6.50] | gate |
| harness | retain | 55.46 | 56.30 | -0.84 [-5.04, +3.36] | gate |
| harness | WMDP | 59.35 | 54.47 | +4.88 [-3.25, +13.01] | gate_margin |
| harness | retain | 55.46 | 49.58 | +5.88 [-1.68, +14.29] | gate_margin |
| chat_prefix | WMDP | 54.47 | 56.10 | -1.63 [-6.50, +3.25] | gate |
| chat_prefix | retain | 57.14 | 57.14 | +0.00 [-4.20, +4.20] | gate |
| chat_prefix | WMDP | 54.47 | 44.72 | +9.76 [+1.63, +18.70] | gate_margin |
| chat_prefix | retain | 57.14 | 44.54 | +12.61 [+3.36, +21.01] | gate_margin |
| chat_plain | WMDP | 34.96 | 36.59 | -1.63 [-4.88, +1.63] | gate |
| chat_plain | retain | 32.77 | 34.45 | -1.68 [-4.20, +0.00] | gate |
| chat_plain | WMDP | 34.96 | 29.27 | +5.69 [+0.00, +12.20] | gate_margin |
| chat_plain | retain | 32.77 | 31.09 | +1.68 [-5.88, +9.24] | gate_margin |

## Does the first-order score predict its own objective? (method gate, test split)

| Condition | Layer | alpha | Role | Predicted drop in log p | Actual drop in log p | Actual accuracy drop pp |
| --- | :---: | :---: | --- | ---: | ---: | ---: |
| Top-k localization | 15 | 0.25 | WMDP | +0.077 | +0.144 | +1.95 |
| Top-k localization | 15 | 0.25 | retain | +0.014 | +0.085 | +0.39 |
| Top-k localization | 15 | 0.5 | WMDP | +0.154 | +0.442 | +2.34 |
| Top-k localization | 15 | 0.5 | retain | +0.029 | +0.294 | +0.39 |
| Top-k localization | 15 | 0.75 | WMDP | +0.231 | +0.856 | +2.73 |
| Top-k localization | 15 | 0.75 | retain | +0.043 | +0.611 | +0.39 |
| Top-k localization | 15 | 1 | WMDP | +0.308 | +1.142 | +3.12 |
| Top-k localization | 15 | 1 | retain | +0.057 | +0.848 | +1.17 |
| Bottom-k | 6 | 0.25 | WMDP | -0.043 | -0.013 | +3.52 |
| Bottom-k | 6 | 0.25 | retain | -0.001 | +0.020 | +1.95 |
| Bottom-k | 6 | 0.5 | WMDP | -0.086 | +0.013 | +5.86 |
| Bottom-k | 6 | 0.5 | retain | -0.002 | +0.075 | +2.34 |
| Bottom-k | 6 | 0.75 | WMDP | -0.130 | +0.097 | +11.72 |
| Bottom-k | 6 | 0.75 | retain | -0.003 | +0.181 | +12.50 |
| Bottom-k | 6 | 1 | WMDP | -0.173 | +0.244 | +26.95 |
| Bottom-k | 6 | 1 | retain | -0.004 | +0.311 | +23.83 |

## Predicted answer letters on the WMDP test split

| Condition | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| gate | 39 | 93 | 58 | 66 |
| gate_margin | 79 | 73 | 47 | 57 |
| baseline | 44 | 88 | 60 | 64 |

