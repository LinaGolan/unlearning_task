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

## Main comparison: direction (alpha = 0.5, k = 2)

| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % | Delta Retain pp [95% CI] | WMDP mean log p |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | - | 60.55 | 0.00 | 53.91 | 0.00 | -1.066 |
| Top-k localization | 0,15 | 59.77 | +0.78 [-1.56, +3.12] | 53.91 | +0.00 [-1.56, +1.56] | -1.069 |
| WMDP-vs-Retain | 3,4 | 57.42 | +3.12 [+0.39, +6.25] | 52.73 | +1.17 [-0.78, +3.52] | -1.062 |
| Bottom-k | 12,13 | 58.59 | +1.95 [-0.39, +4.69] | 54.30 | -0.39 [-2.73, +1.95] | -1.076 |
| Random (mean of 5) | varies | 58.67 | +1.87 [+0.08, +3.91] | 54.53 | -0.63 [-2.03, +0.70] | - |

Extra drop over the random-layer mean (positive = more damage than random):

| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |
| --- | ---: | ---: |
| Top-k localization | -1.09 [-2.81, +0.47] | +0.63 [-0.63, +1.88] |
| WMDP-vs-Retain | +1.25 [-0.94, +3.36] | +1.80 [+0.16, +3.59] |
| Bottom-k | +0.08 [-1.33, +1.48] | +0.23 [-1.64, +2.19] |

RQ3, paired directly: how much *more* damage the WMDP-only ranking does than the forget-vs-retain ranking (positive = WMDP-only is more damaging):

| Role | Extra damage from the WMDP-only ranking pp [95% CI] |
| --- | ---: |
| WMDP | -2.34 [-5.47, +0.78] |
| retain | -1.17 [-3.91, +1.17] |

Individual random selections: 6,14 -> WMDP 58.98 / retain 55.47; 10,12 -> WMDP 58.98 / retain 53.52; 1,3 -> WMDP 60.16 / retain 54.30; 2,14 -> WMDP 58.98 / retain 54.69; 4,15 -> WMDP 56.25 / retain 54.69

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

**direction**

| Condition | a=0.0 | a=0.5 | a=1.0 |
| --- | ---: | ---: | ---: |
| Top-k localization WMDP | 60.55 | 59.77 | 57.42 |
| Top-k localization retain | 53.91 | 53.91 | 52.34 |
| WMDP-vs-Retain WMDP | 60.55 | 57.42 | 55.47 |
| WMDP-vs-Retain retain | 53.91 | 52.73 | 53.91 |
| Bottom-k WMDP | 60.55 | 58.59 | 57.42 |
| Bottom-k retain | 53.91 | 54.30 | 53.91 |
| Random (mean of 5) WMDP | 60.55 | 58.67 | 56.72 |
| Random (mean of 5) retain | 53.91 | 54.53 | 54.30 |

## General-biology control (64 MMLU high-school biology questions)

| Condition | Accuracy % | Drop pp [95% CI] |
| --- | ---: | ---: |
| Baseline | 51.56 | 0.00 |
| gate Top-k localization | 48.44 | +3.12 [-7.81, +14.06] |
| gate WMDP-vs-Retain | 48.44 | +3.12 [+0.00, +7.81] |
| gate Bottom-k | 39.06 | +12.50 [+1.56, +25.00] |
| gate Random (mean of 5) | 47.50 | +4.06 [-0.94, +9.69] |
| direction Top-k localization | 51.56 | +0.00 [+0.00, +0.00] |
| direction WMDP-vs-Retain | 50.00 | +1.56 [-3.12, +7.81] |
| direction Bottom-k | 50.00 | +1.56 [+0.00, +4.69] |
| direction Random (mean of 5) | 50.00 | +1.56 [+0.00, +3.75] |

## Prompt-format robustness (242 question subset)

| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |
| --- | --- | ---: | ---: | ---: | --- |
| harness | WMDP | 59.35 | 55.28 | +4.07 [-0.81, +9.76] | gate |
| harness | retain | 55.46 | 54.62 | +0.84 [-6.72, +8.40] | gate |
| harness | WMDP | 59.35 | 56.10 | +3.25 [-1.63, +8.13] | direction |
| harness | retain | 55.46 | 56.30 | -0.84 [-4.20, +1.68] | direction |
| chat_prefix | WMDP | 54.47 | 52.03 | +2.44 [-4.07, +8.94] | gate |
| chat_prefix | retain | 57.14 | 52.94 | +4.20 [-4.20, +12.61] | gate |
| chat_prefix | WMDP | 54.47 | 51.22 | +3.25 [+0.00, +7.32] | direction |
| chat_prefix | retain | 57.14 | 59.66 | -2.52 [-7.56, +1.68] | direction |
| chat_plain | WMDP | 34.96 | 35.77 | -0.81 [-5.69, +4.07] | gate |
| chat_plain | retain | 32.77 | 34.45 | -1.68 [-7.56, +3.36] | gate |
| chat_plain | WMDP | 34.96 | 31.71 | +3.25 [+0.81, +7.32] | direction |
| chat_plain | retain | 32.77 | 31.93 | +0.84 [-1.68, +4.20] | direction |

## Which change mattered: target layer x intervention type

| Layer chosen by | Intervention | Layer | alpha | WMDP % | Delta WMDP pp | Retain % | Delta Retain pp |
| --- | --- | :---: | :---: | ---: | ---: | ---: | ---: |
| gate | block_scale | 3,15 | 0.5 | 54.69 | +5.86 | 53.91 | +0.00 |
| gate | direction_ablate | 3,15 | 0.5 | 58.98 | +1.56 | 53.91 | +0.00 |
| direction | block_scale | 3,4 | 0.5 | 48.83 | +11.72 | 50.78 | +3.12 |
| direction | direction_ablate | 3,4 | 0.5 | 57.42 | +3.12 | 52.73 | +1.17 |

## Does the first-order score predict its own objective? (method gate, test split)

| Condition | Layer | alpha | Role | Predicted drop in log p | Actual drop in log p | Actual accuracy drop pp |
| --- | :---: | :---: | --- | ---: | ---: | ---: |
| Top-k localization | 0 | 0.5 | WMDP | +0.050 | +0.554 | +9.38 |
| Top-k localization | 0 | 0.5 | retain | +0.011 | +0.543 | +12.11 |
| Top-k localization | 0 | 1 | WMDP | +0.099 | +0.816 | +34.77 |
| Top-k localization | 0 | 1 | retain | +0.021 | +0.834 | +27.73 |
| WMDP-vs-Retain | 3 | 0.5 | WMDP | +0.023 | +0.389 | +5.86 |
| WMDP-vs-Retain | 3 | 0.5 | retain | -0.047 | +0.282 | +0.00 |
| WMDP-vs-Retain | 3 | 1 | WMDP | +0.047 | +1.421 | +23.05 |
| WMDP-vs-Retain | 3 | 1 | retain | -0.093 | +1.151 | +18.36 |
| Bottom-k | 6 | 0.5 | WMDP | -0.086 | +0.016 | +6.64 |
| Bottom-k | 6 | 0.5 | retain | -0.002 | +0.135 | +9.38 |
| Bottom-k | 6 | 1 | WMDP | -0.173 | +0.345 | +29.69 |
| Bottom-k | 6 | 1 | retain | -0.004 | +0.409 | +29.69 |

## Predicted answer letters on the WMDP test split

| Condition | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| gate | 64 | 83 | 51 | 58 |
| direction | 45 | 83 | 57 | 71 |
| baseline | 44 | 88 | 60 | 64 |

