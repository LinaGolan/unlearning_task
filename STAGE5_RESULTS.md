# Stage 5 results: development gains did not carry over reliably

Verified September 21, 2026. Stage 5 is technically complete. The experiment does **not** establish
successful selective forgetting or a clear advantage of the selective pair over the fixed random-pair mean.

## The main result

Strength **0.75** was selected using development data only. On those questions, top selective reduced
forget accuracy by **6.25 percentage points** and retain accuracy by **0.78 points**, satisfying the
predeclared observed-development limit of five points. Strength 1 was rejected for excessive retain loss.

The same strength on the reserved final test reduced forget accuracy by **9.38 points**, but retain
accuracy also fell by **11.72 points**. The retain-preservation result did not carry over to new questions.
The five-point rule was a development selection criterion, not a guaranteed bound on test performance.

Each test role contains 256 questions. Every intervention below uses strength 0.75 and the unchanged
Stage 4 layer pairs. All numbers are percentages or percentage-point drops from the test baseline.

| Method | Forget accuracy | Retain accuracy | Forget drop | Retain drop |
| --- | ---: | ---: | ---: | ---: |
| Unchanged baseline | 53.91% | 54.69% | 0.00 | 0.00 |
| Top forget [15, 0] | 25.00% | 28.52% | 28.91 | 26.17 |
| Top selective [15, 7] | 44.53% | 42.97% | 9.38 | 11.72 |
| Bottom forget [6, 9] | 37.50% | 34.77% | 16.41 | 19.92 |
| Mean of five fixed random pairs | 45.47% | 45.86% | 8.44 | 8.83 |

The selective pair caused less damage than top forget to both roles. That alone does not establish
selectivity: its retain drop was larger than its forget drop, and it had no clear advantage over random
selection. Top forget strongly disrupted both kinds of question. Accuracy near 25% does not demonstrate
erasure of the underlying knowledge.

## Uncertainty

The paired bootstrap used 2,000 resamples of question IDs within subject, sharing sampled IDs across
conditions. These intervals are conditional on this model, prompt, selected layers, and fixed random pairs.

| Final-test quantity | Estimate (pp) | Paired 95% interval (pp) |
| --- | ---: | ---: |
| Selective forget drop from baseline | 9.38 | 3.13 to 16.02 |
| Selective retain drop from baseline | 11.72 | 5.08 to 18.37 |
| Selective extra forget drop versus random mean | 0.94 | -4.69 to 6.48 |
| Selective extra retain drop versus random mean | 2.89 | -2.89 to 8.91 |

Positive extra drop means more performance damage. Both comparisons against random include zero;
the experiment does not resolve a reliable difference on either role. The intervals are not corrected
for multiple comparisons. Variation across random pairs is reported separately in the saved analysis.

## What remains to investigate

The Stage 4 retain and selectivity rankings were unstable across data halves. The development/test
difference is consistent with noisy selection, but this experiment alone does not establish its cause.

Answer-letter behavior also changed. On the 256 test forget questions, the baseline predicted A/B/C/D
68/63/71/54 times. Top selective predicted 4/116/49/87; top forget predicted 60/34/153/9.
These shifts motivate the planned answer-letter and alternative-prompt checks. They are not yet evidence
of a particular mechanism or of knowledge removal.

Proceed with the already planned Stage 6 controls: 64 general-biology questions, alternative wording on
a fixed 128-question subset per main role, and answer-letter analysis. Keep strength **0.75**, the
layer pairs, and the selection rule unchanged. Do not choose a replacement from the final-test curves.
A negative main result still leaves meaningful control questions and a defensible mentor discussion.

## Verified evidence and recovery

- Original download: `stage5-results (1).zip`.
- SHA-256: `127714b2c78cdef1330a8b273ef732c3164535512e5dd6020873b8801c421bae`.
- Preserved archive: `outputs/stage5/verified_2026-09-21/original_results.zip`.
- Verification report: `outputs/stage5/verified_2026-09-21/verification.json`.
- Unpacked reports and figures: `outputs/stage5/verified_2026-09-21/outputs/stage5/{development,test}/`.
- Raw records are retained in the preserved ZIP, without duplicating thousands of individual files locally.

Verified all **8,192 new development + 16,896 new test predictions**, plus the **256 imported development
baseline records**. Checks covered record hashes, derived scores and correctness, prepared question IDs
and answers, prompt text and token hashes, model/configuration/source identities, fixed layer pairs,
the development-only decision, and its unchanged reuse in test. Recomputed all analysis statistics,
paired bootstrap intervals, and the main tables. The two final-test figures were visually inspected.
The tokenizer and model were not rerun.

Development finished in one segment. Test resumed after 11,003 predictions and added the remaining
5,893, with the same run fingerprint and passing final cleanup. The earlier interrupted segment retains
the status `running`; the later segment is complete and all expected unique records are present.
Recorded experiment segment time totals about **60 minutes**, excluding setup and model loading.

Recheck the archive without a GPU:

```text
python scripts/verify_stage5_results.py "PATH_TO_RESULTS_ZIP" --output outputs/stage5/verified_2026-09-21
```

The original Colab reports are preserved unchanged. This review adds interpretation without modifying
the decision, predictions, experiment source, or prior results.
