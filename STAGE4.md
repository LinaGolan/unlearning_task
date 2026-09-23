# Stage 4: Localization and Control A

**Status: complete. All 1,312 research-model records and the derived results were verified on September 20, 2026.**
Stages 1-3 are complete. This stage covers assignment Sections 4 (layer-wise localization),
5 (forget-versus-retain localization), part of 9 (causal validation), and the localization outputs in 11.
It prepares the fixed layer selections needed by Section 7. It does not perform the main comparison sweep or final-test evaluation.
The local validation includes interrupted versus uninterrupted execution, frozen selections, record tampering,
split separation, signed rankings, ties, safe result recovery, and synthetic figure rendering.
Synthetic checks are not research findings.

## Reviewed research results

The source archive is `stage4-results (1).zip`. Its original bytes and exact contents are preserved under
`outputs/stage4/verified_2026-09-20/`, together with an [offline verification record](outputs/stage4/verified_2026-09-20/verification.json).
Verification checked the source bundle and executed code hashes, run identity, exact sample IDs, baseline identity,
all 1,312 record checksums and derived scores, frozen selections, recomputed stability/Control A statistics,
the layer-score table, and the saved figures. No GPU computation was repeated.

All planned work completed: 256 localization gradients, 32 control gradients, and 1,024 interventions.
The recorded execution segment took 213.5 seconds (about 3.6 minutes), excluding setup/model loading.
Peak allocated memory during gradient measurements was 6.46 GiB. Cleanup checks passed.

The fixed selections are **zero-based layer indices**:

| Strategy | Layers |
| --- | --- |
| Highest forget score | 15, 0 |
| Highest selectivity score | 15, 7 |
| Lowest forget score | 6, 9 |
| Random seed 101 | 6, 14 |
| Random seed 202 | 10, 12 |
| Random seed 303 | 1, 3 |
| Random seed 404 | 2, 14 |
| Random seed 505 | 4, 15 |

Layer 15 has the highest full-localization forget score (0.3440) and selectivity (0.3373).
Layer 7 has a smaller positive forget score (0.1212) and relatively small retain score (0.0129),
giving selectivity 0.1084. These scores motivate testing the pair; they do not demonstrate selective forgetting.
Negative scores were retained: the bottom-forget layers have negative means, so weakening them is locally predicted
to improve the forget answer score. This is a signed-sensitivity control, not a claim that those layers are unimportant.

### Ranking stability is a material limitation

| Score | Half-to-half Spearman | Top-two overlap |
| --- | ---: | ---: |
| Forget | 0.588 | 1 of 2 |
| Retain | 0.115 | 0 of 2 |
| Selectivity | 0.476 | 0 of 2 |

The selective top pair was `[0, 7]` in the first half and `[15, 1]` in the second half.
Thus the full-data pair `[15, 7]` should not be described as a stable discovery of where the target knowledge is stored.
The retain and selectivity rankings triggered the prespecified stability flags.
We keep the full-data selections and report this uncertainty, as planned; no formula, sample, or layer pair is retuned.

### Control A supports small changes, with weaker predictions for larger changes

| Dataset | Strength | Pooled Spearman | Mean absolute score-drop error | Spearman of layer means |
| --- | ---: | ---: | ---: | ---: |
| Forget | 0.05 | 0.993 | 0.00270 | 0.991 |
| Retain | 0.05 | 0.995 | 0.00266 | 0.985 |
| Forget | 0.5 | 0.724 | 0.29680 | 0.894 |
| Retain | 0.5 | 0.656 | 0.26841 | 0.771 |

The small-strength predicted and measured changes agree closely. At strength 0.5, both ordering agreement and value
accuracy worsen. The higher absolute errors partly reflect the larger intervention scale; the lower correlations and
more scattered plots also show that simple linear extrapolation becomes less reliable.
Correlations are descriptive, not accuracy percentages or evidence that the method beats random layer selection.

**Decision:** proceed to Stage 5 with the frozen pairs and the full profile. Test actual forget/retain accuracy changes
against random controls, keeping ranking instability and nonlinear effects in the interpretation.
No common operating strength has been chosen and no final-test questions have been evaluated.
The selection fingerprint is `fa59d6ef9b703ee7edff5fb5e39a18d34d2f214573f9673d263c712553fc7586`.

## Run in Colab

1. Upload `notebooks/04_stage4_localization.ipynb` to Colab and select a **T4 GPU**.
2. Choose **Run all** and upload `dist/unlearning-stage4.zip` when asked.
3. Keep `RESTORE_RESULTS = False` for the first run. Allow access to the private `HF_TOKEN` secret or use the hidden prompt.
4. Download `stage4-results.zip` and share it for review, including if incomplete or a check fails.

The bundle includes the verified prepared data, answer-prefix baseline, and successful Stage 3 evidence.
No earlier stage needs rerunning. The notebook uses a separate `/content/unlearning_stage4` folder.
Stage 3 timings suggest approximately 5-10 minutes of model work for this stage, plus setup and model download;
the actual duration can differ with prompt lengths and Colab performance. No local environment changes are needed.

The program records **1,312 measurements** with the 16-layer research model:

| Work | Questions | Measurements |
| --- | --- | ---: |
| Localization | 128 forget + 128 retain, from the localization split | 256 forward/backward passes |
| Control A derivatives | 16 forget + 16 retain, from development | 32 forward/backward passes |
| Control A interventions | Those same 32 development questions, each layer separately, strengths 0.05 and 0.5 | 1,024 forward passes |

All 16 layer derivatives are obtained in one forward/backward pass per question.
Small synthetic preflight checks and two baseline replays are additional to these counts.
The saved Stage 2 baseline supplies the unchanged score for Control A; every control gradient also checks that its identity-gate scores match that baseline.

## What the scores mean

Let `M` be the natural log probability of the correct answer, normalized over A-D.
For each layer, calculate `s = dM/dg` at all layer gates equal to one. Positive `s` predicts a drop in
the correct-answer score when that layer is weakened. Negative `s` predicts an improvement for small suppression.
Keep all signed values and all localization questions, including initially incorrect answers.

For each layer:

```text
forget_score = mean derivative on forget localization questions
retain_score = mean derivative on retain localization questions
selectivity = forget_score - retain_score
```

The retained subject groups have equal counts, so the retain mean gives each subject equal weight.
The score measures sensitivity to this intervention, not a direct amount of knowledge stored in a layer.
Different dataset difficulty and answer confidence can affect the comparison.

## Fixed layer selections

Only after all 256 localization records are present, save:

- Two layers with the largest forget score.
- Two layers with the largest selectivity score.
- Two layers with the smallest forget score.
- Five distinct random pairs, using seeds 101, 202, 303, 404, and 505.

Rank ties are resolved by the lower layer index. Layer indices start at zero. Random pairs are generated
without looking at scores; a duplicate random pair is resampled with the same seeded generator.
Random pairs may overlap the ranked pairs, and ranked strategies may select the same layers.
Do not exclude overlaps to make the controls appear more different.

`selections.json` is fixed before Control A outcomes are available. Its fingerprint links the selections
to the exact localization records, model, prompt format, and run. It cannot be silently replaced by a later report.
No intervention strength for the main comparison is selected in this stage.

## Ranking stability

Split localization examples into two deterministic halves, separately within each role and subject.
Sort by a seeded SHA-256 key, then alternate examples between halves. Each half contains 64 forget and 64 retain questions;
each retain subject contributes eight questions per half.

Calculate forget, retain, and selectivity scores independently for each half. Report:

- Spearman rank correlation across the 16 layers, using average ranks for ties.
- The top two layers from each half, and how many they share.
- The two sets of mean scores in a scatter plot.

A Spearman correlation below 0.5, or an undefined correlation, is flagged for discussion. This is a descriptive
review threshold fixed before the run, not a significance test or a condition for changing the formula.
An undefined correlation means at least one set of scores is constant. Report it as `null`, not as zero.
Selections still use all localization questions. Two halves provide a limited stability check, not proof of generalization.

## Control A: Check the prediction against an intervention

Select 16 development questions from each role with a seeded hash rule, independently of answers, correctness, and measured scores.
The retain subset is balanced: two questions from each of its eight subjects. These questions are disjoint from localization.

For each selected question, layer, and strength:

```text
predicted_drop = alpha * gradient_for_that_question_and_layer
measured_drop = unchanged_correct_answer_score - changed_correct_answer_score
```

Compare the predictions at strengths **0.05** and **0.5** separately for forget and retain questions.
Save every paired observation, then report mean absolute error, root mean squared error, mean signed error,
Pearson correlation, Spearman correlation, and sign agreement. Values here are score changes in natural-log units,
not changes in accuracy percentage points. Pairs with either magnitude at or below 0.000001 are excluded from sign agreement only;
the report records their count. Correlations with constant values are undefined and recorded as `null`.

Show both all question/layer pairs and the mean effect per layer. Also report the mean of the per-question layer-rank correlations.
Pairs from the same question are dependent, so pooled correlations are descriptive; no p-values or claims of independent samples are made.
The figures show means and observations without confidence intervals. The paired bootstrap for main accuracy comparisons remains a later-stage task.

Small-strength agreement tests whether the local gradient is useful beyond the tiny numerical checks from Stage 3.
If predictions degrade at strength 0.5, discuss the limitation of a local approximation.
Poor scientific agreement does not invalidate technically correct measurements and does not change the frozen layer selections.

## Outputs and recovery

Results are written under `outputs/stage4/full/`:

| File | Purpose |
| --- | --- |
| `run.json` | Immutable model/data/prompt/code/environment identity, exact sample metadata, and matched control baselines |
| `records/*.json` | Atomic, checksummed gradient or intervention records, including prompts/tokens and per-answer scores |
| `summary.json` | Completion counts, selections, stability, control metrics, latest attempt status, and interpretation limits |
| `localization_summary.json`, `layer_scores.csv` | Full layer scores, ranks, and half assignments |
| `selections.json` | Fixed pairs and their provenance fingerprint |
| `question_layer_scores.csv` | Derivatives per question and layer, with the split explicitly recorded |
| `control_a_pairs.csv`, `control_a_layer_means.csv`, `control_a_summary.json` | Matched predictions and observed effects, with descriptive statistics |
| `figures/*.png`, `figures/*.pdf` | Layer-score plot, ranking-stability plot, and Control A figure |
| `segments/*.json` | Timing, progress, cleanup checks, and sanitized failure details per attempt |

Use `MAX_NEW = 256` for shorter segments. This limits additional records, not questions: a gradient record covers all layers,
while an intervention record covers one question/layer/strength. Download a backup after each segment.
Rerun the run cell to continue. Previously saved measurements are verified and reused; completed localization selections stay fixed.

After losing a runtime, upload the source bundle again, then set `RESTORE_RESULTS = True` in the optional restore cell and upload
the latest Stage 4 result ZIP. Result recovery restores only result artifacts, never source code from the results ZIP.
Conflicting existing files are rejected. The code, settings, and recorded runtime environment must match to resume the same run;
review mismatches instead of deleting records or loosening verification.

A plotting problem does not require repeating GPU measurements. Rebuild the report using `stage4-report`.
The real model is loaded only by `localize`; the report command runs without a GPU or model weights.
The notebook skips model measurements when the saved record counts already indicate completion and verifies/rebuilds the report instead.

## Completion and next decision

Stage 4 is complete only when all planned records, selections, tables, and figures exist and their evidence has been reviewed.
`complete` is technical completion, not proof that localization works. `partial` means more records remain.
`review_required` indicates a measurement, cleanup, or reporting issue to inspect.
Keep unstable rankings, weak control agreement, and negative scores in the final explanation.

After review, Stage 5 will compare the fixed selection strategies across strengths on development questions,
select a common operating strength using the planned retain-accuracy constraint, and then evaluate final-test questions.
Alternative-prompt robustness remains Stage 6. No Stage 5 or Stage 6 work starts automatically in this notebook.

## Developer commands

```powershell
python -m unittest discover -s tests -v
python scripts/build_stage4_bundle.py
```

The builder requires the verified Stage 2 and Stage 3 snapshots already saved in this project. It performs no model run or download,
and leaves earlier source bundles intact. In Colab, after installation and private token setup:

```bash
python -m unlearning localize --output outputs/stage4/full
python -m unlearning localize --output outputs/stage4/full --max-new 256
python -m unlearning stage4-report --output outputs/stage4/full
```

The first two are alternatives for full versus segmented execution. The report command verifies checksums, metadata,
derived answer predictions, matched control baselines, and predicted/actual drop calculations before creating outputs.
The plotting dependency is pinned separately in `requirements-stage4-colab.txt` so prior-stage dependency files remain intact.
Its [Matplotlib release](https://pypi.org/project/matplotlib/3.10.8/) supplies Python 3.13 wheels; exact plotting package versions are recorded with each report.
