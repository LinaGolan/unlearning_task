## Localization scores

| Layer | gate: forget | gate: selective | direction: contribution | direction: separability |
| --- | ---: | ---: | ---: | ---: |
| 0 | +0.0993 | +0.0782 | +1.5152 | +0.9102 |
| 1 | -0.0227 | -0.0314 | +1.2542 | +0.9492 |
| 2 | +0.0304 | +0.0851 | +1.0467 | +0.9531 |
| 3 | +0.0468 | +0.1403 | +0.9157 | +0.9844 |
| 4 | +0.0501 | +0.1052 | +0.9666 | +0.9609 |
| 5 | -0.0048 | +0.0002 | +0.8219 | +0.9492 |
| 6 | -0.1729 | -0.1691 | +0.8891 | +0.9258 |
| 7 | +0.0585 | +0.0612 | +0.8227 | +0.8945 |
| 8 | +0.0293 | -0.0484 | +1.0113 | +0.8750 |
| 9 | -0.1030 | -0.0934 | +0.7329 | +0.8750 |
| 10 | -0.0601 | -0.0995 | +0.7772 | +0.8125 |
| 11 | -0.0321 | -0.0261 | +0.6797 | +0.8516 |
| 12 | -0.0068 | +0.0133 | +0.6452 | +0.8398 |
| 13 | +0.0322 | -0.0036 | +0.5668 | +0.8164 |
| 14 | -0.0755 | -0.0309 | +0.6843 | +0.8398 |
| 15 | +0.3085 | +0.2511 | +1.7027 | +0.9023 |

Split-half Spearman, gate: forget +0.19, retain -0.23, selective -0.14
Split-half Spearman, direction: contribution +0.99, separability +0.86

## Main comparison: gate (alpha = 1, k = 4)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 60.55 | 0.00 | 53.91 | 0.00 | -1.066 |
| Top-k localization | 0,4,7,15 | 28.52 | +32.03 [+23.83, +39.84] | 25.78 | +28.12 [+20.31, +35.94] | -1.901 |
| WMDP-vs-Retain | 2,3,4,15 | 26.56 | +33.98 [+26.17, +42.19] | 25.78 | +28.12 [+20.31, +35.94] | -2.515 |
| Bottom-k | 6,9,10,14 | 21.88 | +38.67 [+30.47, +46.88] | 31.64 | +22.27 [+14.06, +30.08] | -2.070 |
| Random (mean of 5) | varies | 24.77 | +35.78 [+28.44, +42.97] | 22.34 | +31.56 [+24.37, +38.44] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | -3.75 [-10.70, +3.28] | -3.44 [-10.16, +3.83] |
| WMDP-vs-Retain | -1.80 [-6.87, +3.44] | -3.44 [-8.36, +1.33] |
| Bottom-k | +2.89 [-3.05, +8.36] | -9.30 [-16.17, -2.73] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | -1.95 [-10.55, +6.64] |
| retain | +0.00 [-7.81, +7.81] |

Individual random selections: 5,6,8,14 -> WMDP 24.22 / retain 22.66; 6,7,10,12 -> WMDP 23.83 / retain 17.97; 1,3,11,14 -> WMDP 23.83 / retain 23.83; 0,2,6,14 -> WMDP 24.22 / retain 21.88; 4,13,14,15 -> WMDP 27.73 / retain 25.39

## Main comparison: direction (alpha = 1, k = 4)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 60.55 | 0.00 | 53.91 | 0.00 | -1.066 |
| Top-k localization | 0,1,2,15 | 56.25 | +4.30 [+0.39, +8.20] | 55.08 | -1.17 [-4.30, +1.95] | -1.090 |
| WMDP-vs-Retain | 1,2,3,4 | 55.08 | +5.47 [+1.56, +9.38] | 55.86 | -1.95 [-5.47, +1.17] | -1.076 |
| Bottom-k | 11,12,13,14 | 58.59 | +1.95 [-0.78, +4.69] | 53.52 | +0.39 [-2.34, +3.12] | -1.088 |
| Random (mean of 5) | varies | 56.80 | +3.75 [+0.86, +6.80] | 54.30 | -0.39 [-2.81, +1.88] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | +0.55 [-1.95, +3.12] | -0.78 [-3.52, +2.03] |
| WMDP-vs-Retain | +1.72 [-1.56, +4.84] | -1.56 [-4.22, +1.02] |
| Bottom-k | -1.80 [-3.98, +0.31] | +0.78 [-1.72, +3.44] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | -1.17 [-4.69, +2.34] |
| retain | +0.78 [-2.73, +4.69] |

Individual random selections: 5,6,8,14 -> WMDP 57.42 / retain 54.69; 6,7,10,12 -> WMDP 56.64 / retain 54.69; 1,3,11,14 -> WMDP 58.59 / retain 54.69; 0,2,6,14 -> WMDP 56.25 / retain 54.30; 4,13,14,15 -> WMDP 55.08 / retain 53.12

## Strength sweep (test split accuracy %)

**gate**

| Condition | a=0.0 | a=0.5 | a=1.0 |
| --- | ---: | ---: | ---: |
| Top-k localization WMDP | 60.55 | 44.53 | 28.52 |
| Top-k localization retain | 53.91 | 35.55 | 25.78 |
| WMDP-vs-Retain WMDP | 60.55 | 51.56 | 26.56 |
| WMDP-vs-Retain retain | 53.91 | 46.09 | 25.78 |
| Bottom-k WMDP | 60.55 | 50.39 | 21.88 |
| Bottom-k retain | 53.91 | 48.05 | 31.64 |
| Random (mean of 5) WMDP | 60.55 | 47.73 | 24.77 |
| Random (mean of 5) retain | 53.91 | 43.83 | 22.34 |

**direction**

| Condition | a=0.0 | a=0.5 | a=1.0 |
| --- | ---: | ---: | ---: |
| Top-k localization WMDP | 60.55 | 58.98 | 56.25 |
| Top-k localization retain | 53.91 | 55.08 | 55.08 |
| WMDP-vs-Retain WMDP | 60.55 | 57.03 | 55.08 |
| WMDP-vs-Retain retain | 53.91 | 54.30 | 55.86 |
| Bottom-k WMDP | 60.55 | 58.20 | 58.59 |
| Bottom-k retain | 53.91 | 53.91 | 53.52 |
| Random (mean of 5) WMDP | 60.55 | 57.50 | 56.80 |
| Random (mean of 5) retain | 53.91 | 54.45 | 54.30 |

## General-biology control (64 MMLU high-school biology questions)

| Condition | Accuracy % | Drop pp [95% CI] |
| --- | ---: | ---: |
| Baseline | 51.56 | 0.00 |
| gate Top-k localization | 20.31 | +31.25 [+14.06, +48.44] |
| gate WMDP-vs-Retain | 23.44 | +28.12 [+10.94, +45.31] |
| gate Bottom-k | 21.88 | +29.69 [+14.06, +43.75] |
| gate Random (mean of 5) | 20.31 | +31.25 [+15.31, +45.94] |
| direction Top-k localization | 46.88 | +4.69 [-1.56, +12.50] |
| direction WMDP-vs-Retain | 48.44 | +3.12 [-3.12, +10.94] |
| direction Bottom-k | 50.00 | +1.56 [+0.00, +4.69] |
| direction Random (mean of 5) | 50.62 | +0.94 [-1.25, +3.75] |

## Prompt-format robustness (242 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 59.35 | 26.02 | +33.33 [+21.14, +44.72] | gate |
| harness | retain | 55.46 | 27.73 | +27.73 [+15.13, +39.50] | gate |
| harness | WMDP | 59.35 | 53.66 | +5.69 [-0.81, +12.20] | direction |
| harness | retain | 55.46 | 57.14 | -1.68 [-6.72, +3.36] | direction |
| chat_prefix | WMDP | 54.47 | 24.39 | +30.08 [+18.70, +42.28] | gate |
| chat_prefix | retain | 57.14 | 30.25 | +26.89 [+14.29, +38.66] | gate |
| chat_prefix | WMDP | 54.47 | 49.59 | +4.88 [-1.63, +11.38] | direction |
| chat_prefix | retain | 57.14 | 59.66 | -2.52 [-9.24, +4.20] | direction |
| chat_plain | WMDP | 34.96 | 21.95 | +13.01 [+3.25, +22.76] | gate |
| chat_plain | retain | 32.77 | 20.17 | +12.61 [+3.36, +21.85] | gate |
| chat_plain | WMDP | 34.96 | 30.08 | +4.88 [+1.63, +8.94] | direction |
| chat_plain | retain | 32.77 | 36.97 | -4.20 [-8.40, +0.00] | direction |

## Which change mattered: target layer x intervention type

| Layer chosen by | Intervention | Layer | alpha | WMDP % | Delta WMDP pp | Retain % | Delta Retain pp |
| --- | --- | :---: | :---: | ---: | ---: | ---: | ---: |
| gate | block_scale | 2,3,4,15 | 1 | 26.56 | +33.98 | 25.78 | +28.12 |
| gate | direction_ablate | 2,3,4,15 | 1 | 53.12 | +7.42 | 54.69 | -0.78 |
| direction | block_scale | 1,2,3,4 | 1 | 23.83 | +36.72 | 25.39 | +28.52 |
| direction | direction_ablate | 1,2,3,4 | 1 | 55.08 | +5.47 | 55.86 | -1.95 |

## Does the first-order score predict its own objective? (method gate, test split)

| Condition | Layer | alpha | Role | Predicted drop in log p | Actual drop in log p | Actual accuracy drop pp |
| --- | :---: | :---: | --- | ---: | ---: | ---: |
| Top-k localization | 0 | 0.5 | WMDP | +0.050 | +0.351 | +16.02 |
| Top-k localization | 0 | 0.5 | retain | +0.011 | +0.444 | +18.36 |
| Top-k localization | 0 | 1 | WMDP | +0.099 | +0.834 | +32.03 |
| Top-k localization | 0 | 1 | retain | +0.021 | +0.864 | +28.12 |
| WMDP-vs-Retain | 2 | 0.5 | WMDP | +0.015 | +0.559 | +8.98 |
| WMDP-vs-Retain | 2 | 0.5 | retain | -0.027 | +0.492 | +7.81 |
| WMDP-vs-Retain | 2 | 1 | WMDP | +0.030 | +1.449 | +33.98 |
| WMDP-vs-Retain | 2 | 1 | retain | -0.055 | +1.423 | +28.12 |
| Bottom-k | 6 | 0.5 | WMDP | -0.086 | +0.062 | +10.16 |
| Bottom-k | 6 | 0.5 | retain | -0.002 | +0.139 | +5.86 |
| Bottom-k | 6 | 1 | WMDP | -0.173 | +1.003 | +38.67 |
| Bottom-k | 6 | 1 | retain | -0.004 | +0.831 | +22.27 |

## Predicted answer letters on the WMDP test split

| Condition | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| gate | 142 | 73 | 40 | 1 |
| direction | 53 | 72 | 55 | 76 |
| baseline | 44 | 88 | 60 | 64 |

