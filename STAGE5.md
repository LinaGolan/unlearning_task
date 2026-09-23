# Stage 5: Fixed-layer comparisons and strength sweep

Status: **complete and verified on September 21, 2026**. All 25,088 new predictions were checked.
Development selected strength 0.75. On the final test, top selective reduced forget accuracy by 9.38
percentage points and retain accuracy by 11.72 points; preservation did not carry over from development,
and the comparison with random selection is inconclusive. See [STAGE5_RESULTS.md](STAGE5_RESULTS.md)
for the results, uncertainty, and evidence. Stage 6 controls remain to be done.

## What this stage answers

Does the localization score choose more useful layers than random selection? How does the result change
as we weaken those layers more strongly? This covers assignment sections **7 (random controls),
8 (strength ablation), and the main comparisons in 11 (required results)**. Saved code, predictions,
and reproduction instructions also support section 13. Stage 6 will address the remaining causal
controls and prompt robustness; this stage does not finish the assignment.

Forget accuracy is the fraction of WMDP-Bio questions answered correctly. Retain accuracy measures
the separate, balanced eight-subject MMLU sample. A drop is baseline accuracy minus changed-model
accuracy. A positive drop means worse performance; a negative drop means improvement.

## Frozen design

The model, revision, answer-prefix prompt, float32 evaluation, sample IDs, and intervention mechanism
are unchanged. No weights are trained. Both layers in a pair are weakened together at every token position.
Strength zero leaves the model unchanged; strength one removes both selected blocks' residual updates.

| Method | Zero-based decoder layers |
| --- | --- |
| Top forget | 15, 0 |
| Top selective | 15, 7 |
| Bottom forget | 6, 9 |
| Random seed 101 | 6, 14 |
| Random seed 202 | 10, 12 |
| Random seed 303 | 1, 3 |
| Random seed 404 | 2, 14 |
| Random seed 505 | 4, 15 |

These pairs come from the verified Stage 4 localization records. They stay fixed across all strengths
and splits. Bottom forget means the lowest **signed** scores, not necessarily irrelevant layers.
The weak half-to-half stability of retain and selectivity rankings remains a limitation to report.

All methods use strengths 0, 0.25, 0.5, 0.75, and 1. Strength zero is a shared baseline.

| Phase | Questions | New model evaluations |
| --- | --- | --- |
| Development | 128 forget + 128 retain | 8,192; reuse the 256 verified Stage 2 baseline predictions |
| Final test | 256 forget + 256 retain | 16,896, including a new 512-question unchanged baseline |
| Total | 768 distinct questions | 25,088, plus short synthetic evaluator and development replay checks |

## Strength selection and separation from test data

After **every development condition** is complete and model cleanup passes:

1. Consider nonzero strengths for **top selective** only.
2. Require a positive forget-accuracy drop and at most **5 percentage points** of retain loss.
3. Choose the largest forget drop; use the weaker strength when tied.
4. Write an immutable `operating_point.json`, including all candidates, evidence hashes, and the rule.
5. Use this one strength for every method in the main comparison table.

If none qualifies, report `no_suitable_operating_point`. Use 0.5 as an explicitly labeled **display-only
fallback** for the table, and still show the preplanned curves. This is not a successful operating point.
The five-point limit applies to observed development accuracy; it is not a guarantee about test performance.

The test runner verifies the complete decision before preparing or scoring any test question.
All prepared split files are checked for integrity during setup; their test contents do not choose the decision.
Test curves are descriptive. Do not pick a different strength, pair, or formula after seeing them.

## Uncertainty and outputs

The report uses 2,000 paired bootstrap resamples with fixed seed 20260919, separately for forget and retain.
It resamples question IDs within each subject, preserving subject balance, and uses the **same sampled IDs
across methods**. Reported intervals are 95% percentile intervals. They are conditional on the fixed model,
layers, strength, subjects, and random pairs; they do not include uncertainty from selecting those layers.
Development intervals are descriptive after selection. There is no multiple-comparison correction.

Random-pair mean accuracy has a question-bootstrap interval. The range and standard deviation across
the five fixed pairs are reported separately; the plot's shaded random band is that range, not a confidence interval.
Paired contrasts compare each main method with the mean of the fixed random pairs, and top selective
with top forget. Positive `extra_accuracy_drop` means more damage to the specified role.

Each phase writes `outputs/stage5/development/` or `outputs/stage5/test/`:

- `run.json`: immutable configuration, input prompts/tokens, frozen layer pairs, and source hashes.
- `records/*.json`: each new prediction with logits, answer probabilities, correctness, timing, and checksum.
- `segments/*.json`: progress, failure category if any, GPU memory, and model/hook cleanup check.
- `summary.json`: completion status, common-strength table, and limitations.
- `analysis.json`: all accuracies, paired intervals, random variation, and contrasts.
- `main_comparison.csv`, `accuracy.csv`, `paired_contrasts.csv`, `random_pair_variation.csv`.
- `figures/accuracy_curves.*` and `figures/accuracy_tradeoff.*`, in PNG and PDF.
- Development only: `operating_point.json`.

`main_comparison.csv` uses percentages and percentage points; the detailed analysis uses fractions.
Exact correct-answer counts and question counts are saved. Prompt/token text is saved once per question
in the run manifest instead of repeated for every condition.

## Run in Colab

1. Upload `notebooks/05_stage5_sweep.ipynb` to Colab and select a **T4 GPU**.
2. Choose **Run all** and upload `dist/unlearning-stage5.zip` when asked.
3. Allow access to `HF_TOKEN`. Leave `RESTORE_RESULTS = False`, `MAX_NEW = None` for a first full run.
4. Keep the development backup that downloads before test evaluation.
5. Download the final `stage5-results.zip` and share it for review.

This stage is longer than Stage 4. The exact time depends on Colab; each prediction is checkpointed.
For smaller segments set `MAX_NEW = 2048`, download a backup, and repeat the development or test run cell.
Test evaluation waits until development is complete. No rerun of earlier research stages is required.
The requirements file is reused from Stage 4; installation occurs only inside Colab, never locally.

If an older notebook reports `No module named 'unlearning'` in `run_phase`, the running kernel may
not have picked up the editable installation used by the subprocess commands. Run this in a new cell,
then rerun the failed development cell and continue below it; no setup rerun or runtime restart is needed:

```python
import sys
import importlib
from pathlib import Path

source_dir = Path('/content/unlearning_stage5/src')
assert (source_dir / 'unlearning' / '__init__.py').is_file(), 'Stage 5 source files are missing.'
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))
importlib.invalidate_caches()
from unlearning.sweep_report import sweep_report
print('Project imports work. Rerun the development cell.')
```

The current notebook includes this import-path setup automatically and checks it before the experiment.

If the runtime disconnects, open a fresh runtime with the same bundle, set `RESTORE_RESULTS = True`,
and upload the latest Stage 5 results ZIP. Restore checks paths and conflicts and imports only results,
not executable source. Saved records and their derived values are checked before resuming.
Resume requires matching code, settings, and recorded runtime versions. A mismatch must be reviewed,
not worked around by modifying fingerprints. Valid completed records are never replaced.

`complete` means all planned measurements and requested report work finished, not that the hypothesis won.
`partial` means more predictions remain. `review_required` means a technical failure needs attention;
download a backup and retain the error. A plotting error preserves predictions, tables, and the frozen decision.

Within the prepared Colab environment, offline reports can be regenerated with:

```text
python -m unlearning stage5-report --output outputs/stage5/development
python -m unlearning stage5-report --output outputs/stage5/test
```

## Interpretation checkpoint

Review both accuracies and paired differences at the common strength. A larger forget drop is useful only
in light of retain damage and uncertainty. Check whether selective selection improves the trade-off
relative to top forget and the random mean, and whether effects are consistent across strengths.
Do not treat a tiny advantage, a wide interval, or a low final accuracy as proof that knowledge was erased.
Preserve negative findings. A technically valid negative result can proceed to Stage 6 and be a clear,
useful outcome for the mentor discussion.
