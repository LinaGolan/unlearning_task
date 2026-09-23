# Layer Localization and Selective Unlearning

A staged research project by **LinaGolan**. The primary model is
`meta-llama/Llama-3.2-1B-Instruct`. See [PLAN.md](PLAN.md) for the full experiment and assignment mapping.

**Current scope: experiments, analysis, and reproducible source complete.** The reviewed answer-prefix baseline achieved 60.9% forget and 53.9% retain development accuracy.
The downloaded prediction and diagnostic records were verified on September 20, 2026.
**Stage 3 passed all real-model checks**, and its downloaded evidence was verified on September 20, 2026.
Peak allocated GPU memory was 7.18 GiB; the provisional full-experiment estimate is 1.89 GPU hours.
The experiment used the full profile. **Stage 4's 1,312 records were verified on September 20, 2026.**
The top-forget pair is `[15, 0]`, the top-selective pair `[15, 7]`, and the bottom-forget pair `[6, 9]` (zero-based).
Small-intervention predictions agree closely with measurements; stronger interventions are less predictable.
Retain and selectivity rankings are noisy across data halves. These limitations are documented in [STAGE4.md](STAGE4.md).
See [STAGE1.md](STAGE1.md) for setup evidence and [STAGE2.md](STAGE2.md) for baseline instructions.
**Stage 5 results were verified on September 21, 2026: all 25,088 new predictions passed review.**
At the development-selected strength 0.75, top selective reduced final-test forget accuracy by 9.38 points
and retain accuracy by 11.72 points. Retain preservation did not carry over from development, and there is
no clear advantage over the random-pair mean. See [STAGE5_RESULTS.md](STAGE5_RESULTS.md).

**Stage 6 was verified on September 21, 2026:** all 2,880 new control predictions and 2,304 primary
references passed review. Biology showed an uncertain selective drop; one alternative instruction did
not clearly change the intervention effect, and answer-letter shifts persisted. These controls do not
establish selective forgetting. See [STAGE6_RESULTS.md](STAGE6_RESULTS.md). The planned GPU runs are complete.

## Final deliverables

Start with the [Stage 5 results](STAGE5_RESULTS.md) and [Stage 6 controls](STAGE6_RESULTS.md).
The result is negative: the tested method did not establish selective forgetting. All planned experiments and their analysis are complete.

- [Reproduction guide](REPRODUCE.md): fresh-run commands, environment details, and offline processing.
- [Assignment coverage](ASSIGNMENT_COVERAGE.md): implementation and experimental evidence.
- [Source ZIP](https://github.com/LinaGolan/unlearning_task/releases/download/v1.0.0/unlearning-source.zip): source, tests, configs, notebooks, documentation, figures, and aggregate evidence.
- [Verified evidence ZIP](https://github.com/LinaGolan/unlearning_task/releases/download/v1.0.0/unlearning-verified-evidence.zip): prepared samples and original prediction evidence.

The ZIPs are local generated artifacts ignored by Git. Run `python scripts/package_delivery.py` in the completed workspace to rebuild them.
No new Colab run is needed. The instructions below document the completed experimental stages.

## Run Stage 6

1. Upload `notebooks/06_stage6_robustness.ipynb` to Colab and select a T4 GPU.
2. Choose **Run all**, upload `dist/unlearning-stage6.zip`, and allow `HF_TOKEN` access.
3. Keep the default first-run settings and download `stage6-results.zip` in step 5.

The bundle includes verified prior results. There are 2,880 new predictions, with per-question checkpoints.
The notebook includes the kernel import fix; no manual repair cell is needed.

## Run Stage 5

1. Upload `notebooks/05_stage5_sweep.ipynb` to Colab and select a T4 GPU.
2. Choose **Run all**, upload `dist/unlearning-stage5.zip`, and allow `HF_TOKEN` access.
3. Leave `RESTORE_RESULTS = False` and `MAX_NEW = None` for the first full run.
4. Download `stage5-results.zip` for review. A development backup also downloads before the test phase.

This stage performs 25,088 new evaluations and saves each prediction for recovery.
It includes paired uncertainty estimates and clearly reports if no strength meets the development retain limit.
See [STAGE5.md](STAGE5.md) for the frozen rules, assignment mapping, outputs, and resume instructions.

## Reproduce Stage 4

1. Upload `notebooks/04_stage4_localization.ipynb` to Colab and select a T4 GPU.
2. Choose **Run all** and upload `dist/unlearning-stage4.zip`. The bundle includes the verified data and prior-stage evidence.
3. Allow access to `HF_TOKEN`; leave the optional restore setting off for the first run.
4. Download `stage4-results.zip` for review. `MAX_NEW = 256` optionally runs shorter resumable segments.

This stage measures layer scores, ranking stability, and Control A. It fixes layer pairs using localization data only.
See [STAGE4.md](STAGE4.md) for the protocol, outputs, and recovery instructions.

## Reproduce Stage 3

1. Upload `notebooks/03_stage3_intervention.ipynb` to Colab and select a T4 GPU.
2. Run the cells and upload `dist/unlearning-stage3.zip` when asked. It includes the verified data and baseline inputs.
3. Allow access to your private `HF_TOKEN` secret or use the hidden prompt.
4. Download `stage3-results.zip` for review, even if a check needs attention.

No Stage 2 rerun is needed. This short check validates layer suppression, gradients, cleanup, and actual GPU cost.
See [STAGE3.md](STAGE3.md) for the mechanism, checks, and decision rules. It does not start localization or score final-test questions.

## Reproduce Stage 2 in Colab

1. Upload `notebooks/02_stage2_baseline.ipynb` into Colab and select a T4 GPU.
2. Run the cells in order; upload `dist/unlearning-stage2.zip` when asked.
3. If starting in a fresh runtime, upload your successful Stage 1 results ZIP to restore the verified data.
4. Enable this notebook's access to the private Colab secret `HF_TOKEN`.
5. Run the baseline, then download and share `stage2-results.zip` for review.

The evaluator only scores development questions. It saves predictions individually and supports resuming
with the same settings; `MAX_NEW = 64` optionally limits each run to 64 new questions.
Read [STAGE2.md](STAGE2.md) for scoring, checkpoints, interpretation, and the runtime estimate.
The reviewed prompt uses `--settings configs/baseline_prefill.json --output outputs/stage2/prefill`;
the original notebook also has the separate diagnostic and confirmation cells described in [DIAGNOSTIC.md](DIAGNOSTIC.md).

## Reproduce Stage 1 in Colab

Use your computer to read and edit the project. Use a free Colab GPU for the recommended model.

1. Open [Google Colab](https://colab.research.google.com/) and choose **File > Upload notebook**.
2. Upload `notebooks/01_stage1_setup.ipynb` from this project.
3. Choose **Runtime > Change runtime type > GPU**.
4. Run the notebook cells in order. The first code cell asks for `dist/unlearning-stage1.zip`.
5. For model access, use a private Hugging Face read token in the Colab secret `HF_TOKEN`,
   or the hidden prompt in the notebook. Your account must have approval for the
   [model repository](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
6. Download `stage1-results.zip` using the final cell, even if the model check fails.

The notebook provides short explanations at each step. It does not start the main experiments.
No GitHub upload, paid service, Google Drive mount, or local environment changes are required.

### If the first model check failed

Open `notebooks/00_check_model_access.ipynb` in Colab and run its one code cell. It needs no source ZIP or GPU.
It distinguishes a missing or invalid token from a model-access denial and downloads a safe diagnostic report.
Having a Hugging Face account, having model approval, and having an authorized token are separate requirements.

The setup notebook now uses Python-version-specific dependency pins and stops immediately if installation fails.
The earlier PyArrow 17 pin has no prebuilt Python 3.13 wheel. The Python 3.13 pins now match the package versions
recorded in Colab. Full pretrained-model loading subsequently passed after refreshing the token in the main
notebook. Dependency pins alone do not resolve an access error.

## What Stage 1 produces

| Output | Meaning |
| --- | --- |
| `data/prepared/full/manifest.json` | Source versions, settings, file checksums, counts, and duplicate exclusions |
| `data/prepared/full/{forget,retain}/{localization,development,test}.jsonl` | Separate questions for selecting layers, development decisions, and final evaluation |
| `data/prepared/full/biology_control/test.jsonl` | General-biology control questions |
| `outputs/stage1/environment.json` | Python, package, and GPU information |
| `outputs/stage1/tiny_smoke.json` | A forward-pass check with a tiny randomly initialized model |
| `outputs/stage1/model_smoke.json` | The real Llama model-loading and single-question check, or its failure reason |
| `outputs/stage1/model_access.json` | Separate token validation and access to the pinned model configuration, without weights |

A successful smoke check is not an accuracy result. A randomly initialized tiny model is not a substitute for Llama.
Stage 1 is complete only after the real model loads and its forward-pass check succeeds on the GPU.

## Reproducible data preparation

The configuration is `configs/experiment.json`. Model and dataset revisions are pinned to exact Hugging Face commits.
Data comes from public Parquet files at those commits; no remote dataset code runs.

The full profile uses **128 localization, 128 development, and 256 test questions per main dataset**,
plus 64 biology controls. Retain questions are balanced across eight subjects.

The reduced profile uses **64, 64, and 128 questions** per main dataset. Each reduced split is a subset
of its corresponding full split. A switch in profile never moves an old localization example into the test set.
Even a reduced preparation requires enough source examples to reserve the full split boundaries.

Questions are ranked with a seeded SHA-256 rule rather than depending on incoming row order.
Duplicate question text is removed before selection, including duplicates across roles and subjects.
This detects text duplicates, not all paraphrases. We retain the first occurrence in a stable order,
with forget questions taking priority over retain questions, and controls last.

The manifest records exact file checksums and the source row for every selected question. Verification checks
counts, answer labels, subject balance, provenance, and absence of shared question text across all splits.
Existing output directories are verified and reused only when their configuration and sources agree;
they are not silently overwritten. For new settings, choose a new output directory.

## Commands

The preferred ML environment is Colab with Python 3.10 or newer. The standard-library data tests also run on Python 3.8+.
Use a separate environment if installing locally; do not modify an existing research environment.

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python -m unlearning doctor
python -m unlearning prepare --profile full --output data/prepared/full
python -m unlearning verify --data data/prepared/full
python -m unlearning prepare --profile reduced --output data/prepared/reduced
python -m unlearning smoke --tiny --output outputs/stage1/tiny_smoke.json
python -m unlearning access-check
# GPU and authorized Hugging Face access required:
python -m unlearning smoke
```

Only PyArrow is needed to download and prepare data. Standard-library tests need no installation:

```bash
python -m unittest discover -s tests -v
```

Rebuild the source ZIP and notebook after changing project files:

```bash
python scripts/build_colab_bundle.py
```

The bundle uses an explicit source-file selection. Credentials, `.git`, cached downloads, model weights,
and result directories are excluded. The repository does not contain prepared dataset copies; they can be recreated.

## If a check fails

- **No GPU:** select a GPU in Colab. The recommended-model command refuses to download weights on CPU.
- **Model access:** run the access-check notebook or command. `invalid_token` means the token was rejected;
  `model_access_denied` means to check model approval and gated-model read permissions. Use the same account
  that owns the token. Error reports retain HTTP status codes but never include raw server messages or credentials.
- **Installation:** retain the error and runtime versions. Do not change package versions silently between experiments.
- **Memory:** keep the failure report. Batch size is one for the Stage 1 check; do not change the research model without reviewing the plan.
- **Interrupted preparation:** remove or rename only the incomplete prepared output directory after inspecting it, then rerun. Cached source files can be reused.
- **Changed or damaged data:** verification fails rather than accepting the changed files. Use a fresh output directory and preserve the earlier evidence.

## Completion

The stage documentation records the completed experiments, including the Stage 4 instability, Stage 5 negative result,
and Stage 6 uncertainties. No further GPU runs are required by the current plan.

Sources: [Llama model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct),
[WMDP](https://huggingface.co/datasets/cais/wmdp), [MMLU](https://huggingface.co/datasets/cais/mmlu),
and [Colab resource limits](https://research.google.com/colaboratory/faq.html).
