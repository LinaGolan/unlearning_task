# Stage 3: Intervention correctness and feasibility

**Status: complete. All 47 local tests passed, and the real-model Colab results were verified on September 20, 2026.**
Assignment coverage: Section 6 (causal activation intervention), and the implementation foundation for Section 4 (localization).
This stage does not produce an unlearning result or choose layers. Stages 1 and 2 are complete.

## Verified Colab result

The downloaded `stage3-results (1).zip` and `stage3-results.zip` contain identical evidence.
Run `20260920T160702Z-b27cdc` used the pinned Llama-3.2-1B-Instruct model and reviewed answer-prefix format on a Tesla T4.
The archive, executed source, configuration, run fingerprint, baseline identity, and four development sample IDs were checked locally.
The numerical derivative comparisons, memory calculation, and runtime forecast were independently recalculated from saved records.
No model rerun was needed for this review.

| Result | Verified value |
| --- | --- |
| Baseline replay, zero strength, disabled, and removed intervention | All passed; maximum A-D log-probability difference was exactly 0 |
| Full strength | All 16 layers passed their inputs forward unchanged at every position in the synthetic check |
| Gate derivatives | Finite values for all 16 layers on all four development examples |
| Numerical derivative comparisons | All 12 passed; largest absolute difference approximately 0.000914 |
| Cleanup and frozen-weight guards | Passed |
| Peak allocated GPU memory | 7.18 GiB out of 14.56 GiB total; 50.7% remained beyond tensor allocation |
| Mean measured gate forward/backward time | 0.469 seconds per checked example |
| Estimated full experiment compute | 1.89 GPU hours, excluding downloads and queueing; still a provisional estimate |

**Decision: proceed with the full profile.** Stage 4 is next: localization and Control A.
This validates the mechanism and feasibility; it does not yet establish selective forgetting.
No layers have been selected and no final-test questions have been evaluated.

The original download, exact archived contents, and [offline verification record](outputs/stage3/verified_2026-09-20/verification.json)
are preserved under `outputs/stage3/verified_2026-09-20/`.
The Stage 3 run fingerprint is `99d7e5c6dc10ff220640f974afa2e3bc3442c5acb4c1788c03e9d5a6ba7e9448`.
The reviewed baseline fingerprint is `577c3b0b89590d82eead1a0d87613142da189e6add20f975f920826d991813f8`.
Endpoint and cleanup assertions come from the verified executed code; the offline review does not repeat the GPU computation.

## Run it

1. Upload `notebooks/03_stage3_intervention.ipynb` to Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run the cells in order, or use **Run all**. Upload `dist/unlearning-stage3.zip` when asked.
4. Allow access to the private Colab secret `HF_TOKEN`, or use the hidden token prompt.
5. Download `stage3-results.zip` and share it for review, including if a check fails.

The one ZIP contains the current code, prepared data, and verified answer-prefix baseline records.
You do not need to rerun Stage 1, Stage 2, or the diagnostic notebook. A fresh Colab runtime will still need to download the model weights.
The notebook uses `/content/unlearning_stage3`, with baseline inputs under `inputs/` and a new output folder for each attempt.
No local environment changes, GPU runs, or dependency installations are needed on your computer.

## What the intervention does

A decoder layer receives an internal representation `h_in` and normally returns `h_out`.
Its contribution is the difference `r = h_out - h_in`. We reduce that contribution:

```text
h_changed = h_in + (1 - alpha) * r
```

At strength `alpha = 0`, behavior is unchanged. At `alpha = 1`, the layer passes its input forward unchanged.
At `alpha = 0.5`, it keeps half its contribution. We apply the same strength at every token position in each selected layer.
Unselected layers are left alone. Layer indices start at zero, so the 16-layer model uses indices 0 through 15.
This affects the whole decoder block, including attention and its feed-forward computation.

The layer still runs before its output is replaced; full suppression is not a speed optimization.
There is no retraining, saved weight update, or claim of permanent knowledge erasure.
The implementation supports arbitrary layer lists, strength control, disabling, and automatic cleanup after errors.
It supports tensor and tensor-first tuple decoder outputs, covering the local and Colab interfaces.
Use it for full-prompt evaluation with `use_cache=False`, not cached generation or concurrent calls on one model.

For example, inside Python:

```python
from unlearning.interventions import LayerIntervention

# Example indices only; these have NOT been selected by localization.
with LayerIntervention(evaluator.model, layers=[2, 8], alpha=0.5):
    predictions = evaluator.score(examples)
# Normal evaluation is restored here, including after an exception.
```

## Gradient meaning and checks

For localization later, define a gate `g = 1 - alpha` and score `M` as the correct answer's log probability,
normalized over A, B, C, and D. We compute `s_l = dM/dg_l` at all gates equal to 1.
Weights stay frozen; the gradients are with respect to gates, not model parameters.
A positive score predicts that slightly reducing that layer's contribution will lower the correct-answer score.
Negative scores are retained. A small derivative alone does not predict what happens at large suppression strengths.

The differentiable implementation uses `h_out + (g - 1) * (h_out - h_in)`. It is the same formula,
but at `g = 1` the forward value equals `h_out` exactly while the derivative still exists.
A shortcut that simply returns the original output for a differentiable gate equal to 1 would lose this derivative.

The Colab check uses four development examples: the median-length and longest saved prompt from each main dataset,
with ties resolved by example ID. Selection does not depend on correctness, answer label, or measured gradients.
It verifies the baseline's saved prompts, tokens, scores, model settings, and ability gate before continuing.
Harmless synthetic questions warm up the evaluator and check full-strength outputs.

| Check | Expected result |
| --- | --- |
| Replay the four saved baseline examples | Same prompt/tokens and A-D scores within the original tolerances; same prediction |
| Strength zero at all layers | Baseline scores reproduced |
| Strength one at all layers | Each block's output exactly equals its input at every position |
| Disabled and removed intervention | Baseline restored; no remaining hooks |
| Differentiable gates at one | Baseline scores reproduced; finite derivatives for every layer |
| Numerical derivatives | Agreement on the first, middle, and last layer for each median example, using two step sizes |
| Model state | Frozen flags, parameter/storage identities, mutation counters, hook IDs unchanged; no weight gradients |
| Actual gate-gradient run | Peak GPU allocation and elapsed time saved for all four examples |

Finite differences use `(M(g+epsilon) - M(g-epsilon)) / (2*epsilon)` with steps 0.01 and 0.005.
The only values above one are the small probes needed for this derivative check; experiment strengths remain in [0, 1].
Both steps must agree within `0.003 + 0.08 * abs(analytic_gradient)`. A gradient magnitude above 0.006
also requires matching signs; near-zero signs are too sensitive to numerical error.
These float32 tolerances are fixed in `configs/intervention_check.json` before the real-model run.
A failed comparison requires diagnosis, not silently relaxing thresholds.
Local tests additionally compare float64 derivatives with tight numerical differences and compare every tiny-model weight value.
The real-model mutation guard avoids copying a billion parameters; it is a practical guard, not a full weight checksum.

## Checkpoints and next decision

The program saves `run.json`, `summary.json`, `gate_probe.json`, and `finite_differences.json` as it progresses.
Failures retain completed checks and a sanitized failure report. A short interrupted Stage 3 attempt can be rerun;
the notebook saves a separate attempt rather than mixing records. The CLI requires a new empty output directory for each attempt.
This is not the resumable large localization run yet.
The result archive includes code, configuration, exact package versions, and the input bundle's checksums.
The verified Stage 2 baseline fingerprint is recorded with every attempt.

**Passed:** correctness checks succeed, at least 10% of GPU memory remains beyond peak tensor allocation,
and the estimated experiment fits the agreed six GPU hours in a full or reduced profile.
Allocated tensor memory is not all device memory. Timing excludes loading/queueing and assumes later prompts are comparable;
the estimate is a feasibility check, not a runtime guarantee. Review the recommended profile before Stage 4.

**Needs attention:** inspect the failing check, memory use, or forecast. Resolve technical failures before localization.
Do not change the model or answer format automatically. Do not claim Stage 3 complete before real-model evidence is reviewed.

**Next stage:** compute layer scores over the localization split, compare forget and retain scores, and perform Control A.
The numerical checks here test implementation. Control A later tests whether the approximation predicts actual effects.
Alternative-prompt robustness remains in Stage 6, using fixed layer selections and matched baselines.

## Developer commands

```powershell
python -m unittest discover -s tests -v
python scripts/build_stage3_bundle.py
```

The bundle builder verifies `data/prepared/full` and the archived reviewed baseline under
`outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill`. It performs no model runs or downloads
and does not rebuild the older Stage 1/2 bundles. Those inputs must exist to rebuild the archive locally.

In Colab, after package installation and private token setup, the core command is:

```bash
python -m unlearning intervention-check \
  --settings configs/baseline_prefill.json \
  --data inputs/data/prepared/full \
  --baseline inputs/outputs/stage2/prefill \
  --output outputs/stage3/check
```

Implementation API references: [PyTorch forward hooks](https://docs.pytorch.org/docs/2.4/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
and [gate derivatives with autograd.grad](https://docs.pytorch.org/docs/2.4/generated/torch.autograd.grad.html).
