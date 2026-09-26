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

## Main comparison: gate (alpha = 0.5, k = 2)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 60.55 | 0.00 | 53.91 | 0.00 | -1.066 |
| Top-k localization | 0,15 | 51.17 | +9.38 [+3.91, +15.23] | 41.80 | +12.11 [+6.64, +17.58] | -1.620 |
| WMDP-vs-Retain | 3,15 | 54.69 | +5.86 [+1.95, +10.16] | 53.91 | +0.00 [-5.08, +5.08] | -1.455 |
| Bottom-k | 6,9 | 53.91 | +6.64 [+1.95, +11.33] | 44.53 | +9.38 [+2.73, +15.62] | -1.083 |
| Random (mean of 5) | varies | 54.61 | +5.94 [+2.73, +9.14] | 51.33 | +2.58 [-0.78, +5.86] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | +3.44 [-1.64, +8.83] | +9.53 [+4.45, +14.53] |
| WMDP-vs-Retain | -0.08 [-3.28, +3.36] | -2.58 [-6.25, +1.09] |
| Bottom-k | +0.70 [-2.89, +4.30] | +6.80 [+1.56, +11.88] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | +3.52 [-2.34, +9.38] |
| retain | +12.11 [+5.86, +18.36] |

Individual random selections: 6,14 -> WMDP 54.30 / retain 50.78; 10,12 -> WMDP 56.64 / retain 52.34; 1,3 -> WMDP 53.52 / retain 49.61; 2,14 -> WMDP 57.81 / retain 52.34; 4,15 -> WMDP 50.78 / retain 51.56

## Strength sweep (test split accuracy %)

**gate**

| Condition | a=0.0 | a=0.5 | a=1.0 |
| --- | ---: | ---: | ---: |
| Top-k localization WMDP | 60.55 | 51.17 | 25.78 |
| Top-k localization retain | 53.91 | 41.80 | 26.17 |
| WMDP-vs-Retain WMDP | 60.55 | 54.69 | 37.50 |
| WMDP-vs-Retain retain | 53.91 | 53.91 | 35.55 |
| Bottom-k WMDP | 60.55 | 53.91 | 30.86 |
| Bottom-k retain | 53.91 | 44.53 | 24.22 |
| Random (mean of 5) WMDP | 60.55 | 54.61 | 35.31 |
| Random (mean of 5) retain | 53.91 | 51.33 | 29.30 |

## General-biology control (64 MMLU high-school biology questions)

| Condition | Accuracy % | Drop pp [95% CI] |
| --- | ---: | ---: |
| Baseline | 51.56 | 0.00 |
| gate Top-k localization | 48.44 | +3.12 [-7.81, +14.06] |
| gate WMDP-vs-Retain | 48.44 | +3.12 [+0.00, +7.81] |
| gate Bottom-k | 39.06 | +12.50 [+1.56, +25.00] |
| gate Random (mean of 5) | 47.50 | +4.06 [-0.94, +9.69] |

## Prompt-format robustness (242 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 59.35 | 55.28 | +4.07 [-0.81, +9.76] | gate |
| harness | retain | 55.46 | 54.62 | +0.84 [-6.72, +8.40] | gate |
| chat_prefix | WMDP | 54.47 | 52.03 | +2.44 [-4.07, +8.94] | gate |
| chat_prefix | retain | 57.14 | 52.94 | +4.20 [-4.20, +12.61] | gate |
| chat_plain | WMDP | 34.96 | 35.77 | -0.81 [-5.69, +4.07] | gate |
| chat_plain | retain | 32.77 | 34.45 | -1.68 [-7.56, +3.36] | gate |

## Predicted answer letters on the WMDP test split

| Condition | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| gate | 64 | 83 | 51 | 58 |
| baseline | 44 | 88 | 60 | 64 |

