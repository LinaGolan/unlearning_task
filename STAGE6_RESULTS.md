# Stage 6 results: controls do not establish selective knowledge removal

Verified September 21, 2026. All **2,880 new control predictions** and **2,304 reused primary-prompt
predictions** passed offline verification. The fixed layer pairs and development-selected strength
**0.75** were preserved. The planned GPU experiments are complete; the final research report remains.

The controls do not overturn the Stage 5 result. Top selective still has no clear advantage over the
fixed random-pair mean. There are signs of broader disruption and changed answer-letter behavior,
but the evidence does not identify a unique cause or prove knowledge erasure.

## General biology

The unchanged model answered 30/64 questions correctly (**46.88%**). Top selective answered 25/64
(**39.06%**), an observed drop of **7.81 percentage points**. The paired 95% interval for that drop is
**-4.69 to 18.75 points**. It includes zero: this small sample is compatible with broader biology damage,
but does not establish a reliable drop for the selective pair.

| Method | Correct / 64 | Accuracy | Drop from biology baseline (pp) | Paired 95% interval for drop (pp) |
| --- | ---: | ---: | ---: | --- |
| Unchanged baseline | 30 | 46.88% | 0.00 | 0.00 to 0.00 |
| Top forget [15, 0] | 16 | 25.00% | 21.88 | 6.25 to 39.06 |
| Top selective [15, 7] | 25 | 39.06% | 7.81 | -4.69 to 18.75 |
| Bottom forget [6, 9] | 21 | 32.81% | 14.06 | -1.56 to 29.69 |
| Mean of five fixed random pairs | — | 44.38% | 2.50 | -6.25 to 10.94 |

Top forget caused a substantial observed drop on general biology as well as on WMDP and retain
questions. This supports a broader-disruption interpretation of that strategy. It does not demonstrate
where knowledge is stored. Biology and WMDP differ in difficulty and question style, so their drop
magnitudes alone are not a controlled comparison of domain specificity.

The selective pair's extra biology drop relative to the random mean was **5.31 points**, with a paired
95% interval of **-6.26 to 16.26 points**. There is no clear difference on this sample.

## One alternative instruction

These comparisons use the **same 128 forget and 128 retain questions** under both wordings. They are
a fixed subset of the Stage 5 final test, which had 256 per role. Their original-prompt accuracies and
drops can therefore differ from the full Stage 5 table. Every intervention is compared with the
unchanged model using the same wording; questions and answer choices are unchanged.

| Role and instruction | Baseline accuracy | Selective accuracy | Selective drop (pp) |
| --- | ---: | ---: | ---: |
| Forget, original | 53.13% (68/128) | 43.75% (56/128) | 9.38 |
| Forget, alternative | 53.91% (69/128) | 42.19% (54/128) | 11.72 |
| Retain, original | 53.13% (68/128) | 44.53% (57/128) | 8.59 |
| Retain, alternative | 50.78% (65/128) | 43.75% (56/128) | 7.03 |

Changing the wording changed the selective intervention's drop by:

- Forget: **+2.34 points**, paired 95% interval **-3.13 to 8.59**.
- Retain: **-1.56 points**, paired 95% interval **-7.81 to 4.69**.

Both intervals include zero. We did not detect a clear change in the intervention's effect under this
one alternative instruction. This is not an equivalence test and does not establish robustness to arbitrary
prompts. The selective method changed its predicted letter on **13/128 forget** and **15/128 retain**
questions between wordings, so individual predictions were not identical.

Under alternative wording, top selective's extra drop relative to the random-pair mean was:

- Forget: **+2.03 points**, paired 95% interval **-7.50 to 11.25**.
- Retain: **+2.50 points**, paired 95% interval **-5.47 to 10.31**.

Again, no clear advantage over random selection is established. The changed baseline and smaller
sample must not be mistaken for a new or improved operating point. Strength 0.75 remains fixed.

## Answer-letter shifts

On the 128 matched forget questions under the original instruction:

| Predicted letter | Unchanged baseline | Top selective |
| --- | ---: | ---: |
| A | 44 | 3 |
| B | 25 | 55 |
| C | 33 | 25 |
| D | 26 | 45 |

The same qualitative shift appears under the alternative instruction: A changes from 46 predictions
to 6, while B changes from 30 to 53. Similar shifts occur on retain and biology questions. The true
answer labels are unchanged. The selective intervention changes the model's answer behavior more
broadly than the number of correct answers alone reveals.

These are descriptive observations. We did not permute the answer positions, so we cannot conclude
that a position/letter bias causes the accuracy losses, or separate that mechanism from disrupted
reasoning or knowledge access. The model has not collapsed to always predicting a single letter.

## Scope of the conclusion

Across the completed stages: layer gradients predicted small local intervention effects well, but the
retain/selectivity rankings were unstable; selective performance on development did not carry over to
the full final test. These controls provide no clear reason to replace that negative finding with a claim
of successful selective forgetting. A careful report should distinguish causal intervention effects on
answers from permanent knowledge deletion or proof of domain-specific storage.

All intervals use 2,000 paired bootstrap resamples within subject. Prompt-effect differences pair both
wordings and all methods on the same resampled IDs. The intervals are conditional on the fixed model,
layers, strength, subjects, random pairs, and instruction pair; there is no multiplicity correction.
Random-pair variation is recorded separately from question uncertainty.

## Verified evidence

- Download: `stage6-results.zip`.
- SHA-256: `c269c7981e59ff2dcea0a41d329385c3cdd03f790af63551f14fb8681c9ef26a`.
- Run fingerprint: `345dabb2312384309bd5a05339c1f2808cfbabb8bfe5f1a04fa5878c02b5d7df`.
- Archive and verification: `outputs/stage6/verified_2026-09-21/`.
- Raw records, original tables, and figures: `outputs/stage6/verified_2026-09-21/outputs/stage6/full/`.

Verification checked source/model/configuration identity; the unchanged Stage 5 decision and reference
records; deterministic subject-balanced question selection; answers, prompt text, and token hashes;
all record checksums and derived logits/scores/correctness; complete conditions and cleanup; and
recomputed paired uncertainty, prompt comparisons, baseline diagnostics, and answer-letter counts.
All three downloaded figures were visually inspected. Original reports remain unchanged.

The experiment finished in one segment, approximately **6.9 minutes** excluding setup and model loading.
No model was loaded or GPU evaluation repeated during local review. Token hashes were checked,
but the tokenizer was not independently rerun.

Recheck offline:

```text
python scripts/verify_stage6_results.py "PATH_TO_RESULTS_ZIP" --output outputs/stage6/verified_2026-09-21
```

Next: consolidate the methods, complete results, limitations, requirement mapping, and discussion points
into the Stage 7 final report. No additional GPU experiment is required by the current plan.
