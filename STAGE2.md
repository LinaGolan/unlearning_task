# Stage 2: Development baseline

**Status: complete.** The revised baseline and diagnostic archives were verified on September 20, 2026.
All 256 baseline predictions and 432 diagnostic records passed integrity checks and reproduced their summaries.

## Reviewed result and decision

| Development dataset | Original prompt | Explicit answer prefix | 95% interval with prefix |
| --- | --- | --- | --- |
| Forget | 36/128 (28.1%) | 78/128 (60.9%) | 52.3%-69.0% |
| Retain | 42/128 (32.8%) | 69/128 (53.9%) | 45.3%-62.3% |

Keep the recommended 1B model. Use `configs/baseline_prefill.json` for subsequent stages:
start the assistant response with `The correct answer is`, then score ` A`, ` B`, ` C`, ` D`.
Both accuracy thresholds and both lower-interval-above-chance checks passed. The original weak baseline
is preserved and must be discussed as a prompt-sensitivity finding, not omitted.
The same model, question IDs, correct answers, and unmodified choices were verified across both baselines.
No final-test predictions have been made.

Verified evidence: `outputs/stage2/verified_2026-09-20/verification.json`, with both original downloads
and extracted records beside it. See [DIAGNOSTIC.md](DIAGNOSTIC.md) for the investigation.
The latest full-profile estimate is about 1.91 GPU hours; it is provisional until Stage 3 checks actual
layer-gate gradients and memory use. No GPU rerun is needed to close Stage 2.

Stage 3 is next: implement the layer multiplier, confirm disabled/zero-strength behavior matches the
reviewed baseline, confirm full-strength passthrough, verify cleanup and numerical derivatives, and
measure actual gradient runtime/memory. Do not rank layers or start a large sweep before these checks pass.

Assignment coverage: Section 3 (Baseline Evaluation), baseline evidence for Section 11,
and reproducibility work for Section 13. The final-test baseline will be collected in Stage 5
after model, sample size, layer selection, and operating-strength choices are fixed.

## What this stage measures

Evaluate the unchanged `meta-llama/Llama-3.2-1B-Instruct` on the saved development splits:
128 WMDP-Bio questions and 128 retain questions, balanced across eight MMLU subjects.
No final-test question is sent to the model. Data verification still checks the integrity
and separation of all saved splits.

Each question gets four labeled choices, A-D. The model reads the entire prompt; we select
the next-token scores at its last real input position. The highest-scoring label is the
prediction. Exact ties use the first label, a recorded deterministic convention.

We apply a log-softmax to the four label scores. The correct answer's resulting log probability
is the continuous score used by the later localization method. It is conditional on choosing
one of A-D, not a calibrated measure of open-ended knowledge. We also record the total probability
mass assigned to A-D across the vocabulary, to help diagnose poor answer formatting.

The reviewed settings are fixed in `configs/baseline_prefill.json`; `configs/baseline.json` preserves the original prompt. The model's own chat template is used,
with an assistant generation header and fixed date `19 Sep 2026`. The prompt is tokenized without
adding special tokens a second time. Every question saves its exact prompt and input token IDs;
the run manifest saves the template. Tokenization of the prompt plus each label must match
the prompt tokens plus that single label token. No question is truncated or silently dropped.

This follows the model's chat interface rather than claiming exact reproduction of any published
WMDP or MMLU benchmark score. See the official [chat-template documentation](https://huggingface.co/docs/transformers/chat_templating)
and [Llama interface](https://huggingface.co/docs/transformers/model_doc/llama).

## Run in Colab

1. Upload `notebooks/02_stage2_baseline.ipynb` into Colab and select a T4 GPU.
2. Run the cells in order. Upload `dist/unlearning-stage2.zip` when asked.
3. If the runtime has no prepared data, upload your successful Stage 1 results ZIP.
   If resuming a lost session, upload your latest Stage 2 results ZIP instead.
4. Allow this notebook to read the Colab secret `HF_TOKEN`. The run cell reloads the token,
   checks access, and launches the baseline with the same token. Downloads also receive it explicitly.
5. Leave `MAX_NEW = None` to finish all questions, or use `64` for shorter segments.
6. Download `stage2-results.zip` before leaving the runtime, even after a partial run or failure.
   Share it for review before progressing to Stage 3.

The notebook can reuse prepared data from the current session. It does not require repeating
Stage 1 or downloading the public datasets again. It may need to download model weights in a new runtime.
No packages are installed or changed on the local computer.

Equivalent commands, from the project folder in a configured GPU environment:

```console
python -m unlearning baseline --settings configs/baseline_prefill.json --data data/prepared/full --output outputs/stage2/prefill
# Optional segment; rerunning continues with the next unsaved question:
python -m unlearning baseline --settings configs/baseline_prefill.json --data data/prepared/full --output outputs/stage2/prefill --max-new 64
# Offline verification and summary; no GPU or model download:
python -m unlearning baseline-report --output outputs/stage2/prefill
```

## Evidence and interruption recovery

The revised run uses `outputs/stage2/prefill/`; the original remains in `outputs/stage2/full/`:

| File | Purpose |
| --- | --- |
| `run.json` | Model/data versions, prompt and evaluator settings, library/hardware details, source fingerprint, expected development IDs |
| `predictions/*.json` | One atomic checkpoint per question, including A-D logits/probabilities, correct answer, prediction, prompt, timing, and checksum |
| `predictions.jsonl` | Consolidated copy rebuilt from the verified checkpoints |
| `evaluator_checks.json` | Single/batch consistency and independent teacher-forced likelihood check |
| `summary.json` | Progress; after completion, accuracy, counts, intervals, subject breakdowns, answer-letter distributions, feasibility checks |
| `accuracy.csv` | Two development-baseline rows with counts and 95% accuracy intervals |
| `segments/*.json` | Peak allocated GPU memory for each completed evaluation segment |
| `runtime_probe.json` | Forward/backward timing and peak memory on median-length and longest development prompts per domain |

Resume uses the atomic per-question files, not the possibly incomplete consolidated export.
Already saved answers are verified and skipped. A truncated temporary write is ignored and its question
can be recomputed. A corrupted finished file is rejected. Code, data, environment, or setting changes
require a new output directory so incompatible predictions cannot be mixed. Do not run two writers
against the same output directory at once.

Colab storage is temporary: resuming after losing a runtime requires a downloaded results ZIP.
The exported ZIP contains the source used for the run as well as the settings and results.

## Decision rules

Apply the pre-agreed ability checks only after all development predictions exist:

- Forget accuracy at least 35%.
- Retain accuracy at least 40%.
- The lower endpoint of each two-sided 95% Wilson accuracy interval strictly above 25%.

Wilson intervals are approximate sampling intervals and do not account for all subject clustering
or dataset-selection effects. The subject breakdown and answer-letter counts help interpret the overall result.
A failed ability check asks for evaluator review before considering the one planned 3B fallback.
The code does not switch models, resample questions, or change thresholds automatically.

The runtime estimate counts the planned baseline/intervention conditions, localization and Control A
backward passes, and the biology and wording controls. It uses measured forward time after synthetic
warm-up, an input-gradient forward/backward proxy, and an explicit 1.5 overhead factor. It compares
full and reduced profiles with the six-GPU-hour budget. Model downloads and waiting for a GPU are excluded.

The proxy is not a layer-localization result. Model weights stay fixed, and no importance scores are saved.
Its median/longest-prompt sample is intentionally conservative and small. Actual layer-gate memory/timing,
and the final profile choice, must be checked in Stage 3. If the probe fails, baseline predictions are
preserved and runtime feasibility is labeled as requiring review. Do not count an estimate as a completed experiment.

## Local validation

All 36 tests passed locally during implementation and diagnostic review, using Python 3.8, PyTorch 2.4.1, and Transformers 4.46.3.
The test suite uses invented questions and a random tiny Llama, with no network or real-model download.
It checks numerical answer scoring, right padding, token boundaries, prompt independence from the correct
answer, no truncation, Wilson intervals, threshold rules, frozen weights during the gradient probe,
resuming without duplicated predictions, and rejecting changed settings or corrupted results.

Run `python -m unittest discover -s tests -v` in the existing local environment. Numerical tests require
PyTorch and Transformers; they are skipped if those packages are absent, and always run in the configured Colab environment.
Local tests do not establish research accuracy; the real T4 evidence reviewed above supplies that measurement.
