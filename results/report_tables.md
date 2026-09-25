## Localization scores

| Layer | gate: forget | gate: selective | gate_margin: forget | gate_margin: selective | direction: contribution | direction: separability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.0993 | +0.0782 | -0.2524 | -0.0811 | +1.5152 | +0.9102 |
| 1 | -0.0227 | -0.0314 | +0.0391 | +0.1058 | +1.2542 | +0.9492 |
| 2 | +0.0304 | +0.0851 | +0.1305 | +0.5000 | +1.0467 | +0.9531 |
| 3 | +0.0468 | +0.1403 | -0.1400 | +0.0969 | +0.9157 | +0.9844 |
| 4 | +0.0501 | +0.1052 | -0.2832 | -0.3056 | +0.9666 | +0.9609 |
| 5 | -0.0048 | +0.0002 | +0.0087 | +0.3225 | +0.8219 | +0.9492 |
| 6 | -0.1729 | -0.1691 | -0.0310 | -0.2770 | +0.8891 | +0.9258 |
| 7 | +0.0585 | +0.0612 | +0.5012 | -0.0103 | +0.8227 | +0.8945 |
| 8 | +0.0293 | -0.0484 | +0.0886 | -0.2268 | +1.0113 | +0.8750 |
| 9 | -0.1030 | -0.0934 | -0.0584 | -0.2274 | +0.7329 | +0.8750 |
| 10 | -0.0601 | -0.0995 | +0.4473 | -0.0156 | +0.7772 | +0.8125 |
| 11 | -0.0321 | -0.0261 | +0.0747 | -0.0449 | +0.6797 | +0.8516 |
| 12 | -0.0068 | +0.0133 | -0.0635 | -0.0047 | +0.6452 | +0.8398 |
| 13 | +0.0322 | -0.0036 | +0.1632 | -0.0846 | +0.5668 | +0.8164 |
| 14 | -0.0755 | -0.0309 | +0.2132 | +0.1062 | +0.6843 | +0.8398 |
| 15 | +0.3085 | +0.2511 | -0.9703 | +0.0980 | +1.7027 | +0.9023 |

Split-half Spearman, gate: forget +0.19, retain -0.23, selective -0.14
Split-half Spearman, gate_margin: forget +0.61, retain +0.84, selective +0.17
Split-half Spearman, direction: contribution +0.99, separability +0.86

## Main comparison: gate (alpha = 0.25, k = 1)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 60.55 | 0.00 | 53.91 | 0.00 | -1.066 |
| Top-k localization | 15 | 58.59 | +1.95 [+0.00, +4.30] | 53.52 | +0.39 [-2.34, +3.12] | -1.210 |
| WMDP-vs-Retain | 15 | 58.59 | +1.95 [+0.00, +4.30] | 53.52 | +0.39 [-2.34, +3.12] | -1.210 |
| Bottom-k | 6 | 57.03 | +3.52 [+0.00, +6.64] | 51.95 | +1.95 [-1.56, +5.47] | -1.053 |
| Random (mean of 5) | varies | 58.67 | +1.88 [-0.00, +3.75] | 52.66 | +1.25 [-0.70, +3.13] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | +0.08 [-2.34, +2.66] | -0.86 [-3.59, +1.87] |
| WMDP-vs-Retain | +0.08 [-2.34, +2.66] | -0.86 [-3.59, +1.87] |
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
| Top-k localization | +3.13 [-2.42, +8.59] | -0.23 [-5.63, +5.62] |
| WMDP-vs-Retain | -2.73 [-6.56, +1.17] | -3.36 [-7.11, +0.55] |
| Bottom-k | -5.08 [-9.45, -0.63] | -8.05 [-12.50, -3.52] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | +5.86 [-0.78, +12.50] |
| retain | +3.12 [-3.91, +10.55] |

Individual random selections: 6 -> WMDP 48.83 / retain 41.41; 12 -> WMDP 58.98 / retain 51.17; 1 -> WMDP 52.34 / retain 43.75; 2 -> WMDP 55.47 / retain 48.83; 4 -> WMDP 48.05 / retain 42.19

## Main comparison: direction (alpha = 1, k = 1)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 60.55 | 0.00 | 53.91 | 0.00 | -1.066 |
| Top-k localization | 15 | 57.42 | +3.12 [+0.39, +5.86] | 55.08 | -1.17 [-3.12, +0.78] | -1.079 |
| WMDP-vs-Retain | 3 | 58.59 | +1.95 [-1.17, +5.08] | 52.73 | +1.17 [-1.17, +3.91] | -1.054 |
| Bottom-k | 13 | 57.81 | +2.73 [+0.00, +5.86] | 53.52 | +0.39 [-2.73, +3.52] | -1.090 |
| Random (mean of 5) | varies | 58.28 | +2.27 [+0.47, +4.14] | 54.45 | -0.55 [-2.11, +1.02] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | +0.86 [-1.02, +2.97] | -0.62 [-2.66, +1.41] |
| WMDP-vs-Retain | -0.31 [-2.89, +2.27] | +1.72 [-0.55, +3.98] |
| Bottom-k | +0.47 [-1.64, +2.73] | +0.94 [-1.56, +3.59] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | +1.17 [-1.95, +4.30] |
| retain | -2.34 [-5.08, +0.39] |

Individual random selections: 6 -> WMDP 58.98 / retain 55.47; 12 -> WMDP 58.59 / retain 54.69; 1 -> WMDP 60.94 / retain 54.69; 2 -> WMDP 57.42 / retain 53.91; 4 -> WMDP 55.47 / retain 53.52

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

**direction**

| Condition | a=0.0 | a=0.25 | a=0.5 | a=0.75 | a=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-k localization WMDP | 60.55 | 60.94 | 59.38 | 58.59 | 57.42 |
| Top-k localization retain | 53.91 | 54.30 | 54.69 | 55.08 | 55.08 |
| WMDP-vs-Retain WMDP | 60.55 | 60.16 | 59.77 | 59.77 | 58.59 |
| WMDP-vs-Retain retain | 53.91 | 53.12 | 53.12 | 52.34 | 52.73 |
| Bottom-k WMDP | 60.55 | 60.94 | 59.77 | 58.20 | 57.81 |
| Bottom-k retain | 53.91 | 53.52 | 52.73 | 53.91 | 53.52 |
| Random (mean of 5) WMDP | 60.55 | 60.39 | 59.77 | 58.44 | 58.28 |
| Random (mean of 5) retain | 53.91 | 54.06 | 54.30 | 54.37 | 54.45 |

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
| direction Top-k localization | 50.00 | +1.56 [+0.00, +4.69] |
| direction WMDP-vs-Retain | 53.12 | -1.56 [-4.69, +0.00] |
| direction Bottom-k | 51.56 | +0.00 [-4.69, +4.69] |
| direction Random (mean of 5) | 48.75 | +2.81 [+0.31, +6.25] |

## Prompt-format robustness (242 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 59.35 | 56.10 | +3.25 [+0.81, +6.50] | gate |
| harness | retain | 55.46 | 56.30 | -0.84 [-5.04, +3.36] | gate |
| harness | WMDP | 59.35 | 54.47 | +4.88 [-3.25, +13.01] | gate_margin |
| harness | retain | 55.46 | 49.58 | +5.88 [-1.68, +14.29] | gate_margin |
| harness | WMDP | 59.35 | 56.10 | +3.25 [-2.44, +8.94] | direction |
| harness | retain | 55.46 | 54.62 | +0.84 [-3.36, +5.04] | direction |
| chat_prefix | WMDP | 54.47 | 56.10 | -1.63 [-6.50, +3.25] | gate |
| chat_prefix | retain | 57.14 | 57.14 | +0.00 [-4.20, +4.20] | gate |
| chat_prefix | WMDP | 54.47 | 44.72 | +9.76 [+1.63, +18.70] | gate_margin |
| chat_prefix | retain | 57.14 | 44.54 | +12.61 [+3.36, +21.01] | gate_margin |
| chat_prefix | WMDP | 54.47 | 55.28 | -0.81 [-5.69, +4.07] | direction |
| chat_prefix | retain | 57.14 | 57.14 | +0.00 [-5.04, +5.04] | direction |
| chat_plain | WMDP | 34.96 | 36.59 | -1.63 [-4.88, +1.63] | gate |
| chat_plain | retain | 32.77 | 34.45 | -1.68 [-4.20, +0.00] | gate |
| chat_plain | WMDP | 34.96 | 29.27 | +5.69 [+0.00, +12.20] | gate_margin |
| chat_plain | retain | 32.77 | 31.09 | +1.68 [-5.88, +9.24] | gate_margin |
| chat_plain | WMDP | 34.96 | 29.27 | +5.69 [+1.63, +10.57] | direction |
| chat_plain | retain | 32.77 | 31.93 | +0.84 [-2.52, +5.04] | direction |

## Which change mattered: target layer x intervention type

| Layer chosen by | Intervention | Layer | alpha | WMDP % | Delta WMDP pp | Retain % | Delta Retain pp |
| --- | --- | :---: | :---: | ---: | ---: | ---: | ---: |
| gate | block_scale | 15 | 0.75 | 57.81 | +2.73 | 53.52 | +0.39 |
| gate | direction_ablate | 15 | 1 | 57.42 | +3.12 | 55.08 | -1.17 |
| gate_margin | block_scale | 2 | 0.75 | 55.47 | +5.08 | 48.83 | +5.08 |
| gate_margin | direction_ablate | 2 | 1 | 57.42 | +3.12 | 53.91 | +0.00 |
| direction | block_scale | 3 | 0.75 | 51.56 | +8.98 | 47.66 | +6.25 |
| direction | direction_ablate | 3 | 1 | 58.59 | +1.95 | 52.73 | +1.17 |

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
| WMDP-vs-Retain | 15 | 0.25 | WMDP | +0.077 | +0.144 | +1.95 |
| WMDP-vs-Retain | 15 | 0.25 | retain | +0.014 | +0.085 | +0.39 |
| WMDP-vs-Retain | 15 | 0.5 | WMDP | +0.154 | +0.442 | +2.34 |
| WMDP-vs-Retain | 15 | 0.5 | retain | +0.029 | +0.294 | +0.39 |
| WMDP-vs-Retain | 15 | 0.75 | WMDP | +0.231 | +0.856 | +2.73 |
| WMDP-vs-Retain | 15 | 0.75 | retain | +0.043 | +0.611 | +0.39 |
| WMDP-vs-Retain | 15 | 1 | WMDP | +0.308 | +1.142 | +3.12 |
| WMDP-vs-Retain | 15 | 1 | retain | +0.057 | +0.848 | +1.17 |
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
| direction | 48 | 80 | 59 | 69 |
| baseline | 44 | 88 | 60 | 64 |

