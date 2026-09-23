# Assignment implementation and evidence

This repository contains the reproducible experiment, analysis, and verified numeric evidence. Research completion does **not** mean selective unlearning succeeded.

| Assignment section | Implementation or evidence |
| --- | --- |
| 1 Objective | Compare localized intervention with random selection and general damage |
| 2 Model and Dataset | `configs/experiment.json`; pinned model and dataset revisions; prepared-data manifest |
| 3 Baseline Evaluation | `src/unlearning/baseline.py`; per-question predictions; development prompt diagnostic |
| 4 Layer-wise Localization | `src/unlearning/localization.py`; scores for all 16 layers in `report/evidence/layer_scores.csv` |
| 5 Forget versus Retain | Signed selectivity score S = F − R; localization analysis and figures |
| 6 Causal Activation Intervention | `src/unlearning/interventions.py`; adjustable gates; real-model implementation checks |
| 7 Localization versus Random Controls | Three ranked pairs and five random pairs at the same k = 2 and alpha = 0.75 |
| 8 Strength Ablation | Five strengths; accuracy curves and forget-versus-retain tradeoff; no test retuning |
| 9 Causal Validation | Gradient predictions versus measured effects; biology control; answer-letter diagnostics |
| 10 Robustness | Matched questions under original and alternative instructions, each with its own baseline |
| 11 Required Results | Aggregate tables and figures in `report/evidence/` and `report/figures/` |
| 12 Research Questions | Experimental analyses and limitations in `STAGE4.md`, `STAGE5_RESULTS.md`, and `STAGE6_RESULTS.md` |
| 13 Code Deliverable | Source, tests, configs, notebooks, reproduction instructions, and release packages |

## Verified source locations in the working project

- Stage 2 reviewed baseline: `outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill/`.
- Stage 3 correctness: `outputs/stage3/verified_2026-09-20/outputs/stage3/20260920T160702Z-b27cdc/`.
- Stage 4: `outputs/stage4/verified_2026-09-20/outputs/stage4/full/`.
- Stage 5 reports: `outputs/stage5/verified_2026-09-21/outputs/stage5/`; raw prediction records are retained in `outputs/stage5/verified_2026-09-21/original_results.zip`.
- Stage 6: `outputs/stage6/verified_2026-09-21/outputs/stage6/full/`, plus its original archive and parent `verification.json`.

The source delivery ZIP includes aggregate numeric evidence and its SHA256 manifest in `report/evidence/`. Raw results are supplied separately as the evidence ZIP, to avoid duplicating large archives inside the source package. Verification JSONs retain historical local paths as provenance; these are not runtime dependencies. `REPRODUCE.md` gives portable paths for a fresh execution.

## Interpretation limits

- Selective forgetting and selective superiority over random layers were **not demonstrated**.
- One instruction variation is a narrow robustness check, not a universal guarantee.
- Activation suppression is temporary; weights are unchanged.
- The next answer-choice permutation experiment is proposed, not performed.
- All planned GPU experiments are complete.
