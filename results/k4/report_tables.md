## Localization scores

| Layer | gate: forget | gate: selective |
| --- | ---: | ---: |
| 0 | +0.0993 | +0.0782 |
| 1 | -0.0227 | -0.0314 |
| 2 | +0.0304 | +0.0851 |
| 3 | +0.0468 | +0.1403 |
| 4 | +0.0501 | +0.1052 |
| 5 | -0.0048 | +0.0002 |
| 6 | -0.1729 | -0.1691 |
| 7 | +0.0585 | +0.0612 |
| 8 | +0.0293 | -0.0484 |
| 9 | -0.1030 | -0.0934 |
| 10 | -0.0601 | -0.0995 |
| 11 | -0.0321 | -0.0261 |
| 12 | -0.0068 | +0.0133 |
| 13 | +0.0322 | -0.0036 |
| 14 | -0.0755 | -0.0309 |
| 15 | +0.3085 | +0.2511 |

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

## General-biology control (64 MMLU high-school biology questions)

| Condition | Accuracy % | Drop pp [95% CI] |
| --- | ---: | ---: |
| Baseline | 51.56 | 0.00 |
| gate Top-k localization | 20.31 | +31.25 [+14.06, +48.44] |
| gate WMDP-vs-Retain | 23.44 | +28.12 [+10.94, +45.31] |
| gate Bottom-k | 21.88 | +29.69 [+14.06, +43.75] |
| gate Random (mean of 5) | 20.31 | +31.25 [+15.31, +45.94] |

## Prompt-format robustness (242 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 59.35 | 26.02 | +33.33 [+21.14, +44.72] | gate |
| harness | retain | 55.46 | 27.73 | +27.73 [+15.13, +39.50] | gate |
| chat_prefix | WMDP | 54.47 | 24.39 | +30.08 [+18.70, +42.28] | gate |
| chat_prefix | retain | 57.14 | 30.25 | +26.89 [+14.29, +38.66] | gate |
| chat_plain | WMDP | 34.96 | 21.95 | +13.01 [+3.25, +22.76] | gate |
| chat_plain | retain | 32.77 | 20.17 | +12.61 [+3.36, +21.85] | gate |

## Predicted answer letters on the WMDP test split

| Condition | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| gate | 142 | 73 | 40 | 1 |
| baseline | 44 | 88 | 60 | 64 |

