# Layer localization for selective unlearning

Does mechanistic localization pick better intervention targets than random layer
selection? This repository localizes the decoder layers that carry a model's
WMDP-Bio behaviour, ablates them at inference time, and measures what that costs
on WMDP-Bio versus a general-knowledge retain set.

* **Model** `meta-llama/Llama-3.2-1B-Instruct`, 16 decoder layers, weights frozen throughout.
* **Forget set** `cais/wmdp`, config `wmdp-bio`.
* **Retain set** `cais/mmlu`, eight high-school subjects (US history, world history, geography, government and politics, microeconomics, psychology, physics, computer science).
* **Extra control** `cais/mmlu` `high_school_biology`, to separate "WMDP knowledge" from "biology knowledge".
* **Report** [`Research_report_fullscreen.html`](Research_report_fullscreen.html), a self-contained HTML file with embedded figures.

Every model and dataset revision is pinned; splits come from SHA-256 rankings, so
the saved predictions reproduce the same selection and splits.

Questions must be answered correctly in their original order and in at least
18 of all 24 answer orderings. The remaining 23 orderings are scored only for
originally correct questions. Of 1,273 WMDP and 1,848 retain questions, 567 and
754 qualify; deterministic selection keeps 512 per set plus 64 biology controls.
Each main set has 128 localization, 128 development and 256 test questions.
The experiment uses original questions, not duplicated permutations. Baseline
accuracy is 100% on the selected test sets by construction; this is not an
improvement in the model on unfiltered data.

## Environment

Use Python 3.10+ with the dependencies in `requirements.txt`. Regenerating tables and figures from the included
predictions needs no model download or Hugging Face token. Running inference
requires access to the configured Llama model.

First, clone the repository and enter its directory:

```text
git clone https://github.com/LinaGolan/unlearning_task.git
cd unlearning_task
```

Create and activate an environment on Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies and the local package (same commands on either platform):

```text
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` selects an installation stack by Python version: newer
packages on Python 3.13+, and an older stack below that. It does not recreate the
saved run environment exactly. Model and dataset revisions are pinned in the
experiment configuration; dependency and hardware changes may affect a new model run.

## Regenerate the submitted results

These commands run offline after dependency installation, using the included
per-question predictions. No model inference is performed:

```text
python -m unlearning report
python -m unlearning report --methods gate --out results_fullscreen_experiment/k2 --figures results_fullscreen_experiment/k2/figures
python -m unlearning report --methods gate --out results_fullscreen_experiment/k4 --figures results_fullscreen_experiment/k4/figures
```

They produce CSV tables, detailed Markdown tables, analysis JSON and plots.
The final HTML report is maintained separately and is not overwritten. Its
cross-budget table combines the main, `k2` and `k4` analyses.

## Run a fresh experiment

Set `HF_TOKEN` to a Hugging Face token with access to Llama-3.2-1B-Instruct:
`export HF_TOKEN="your_token"` in Linux/macOS, or
`$env:HF_TOKEN = "your_token"` in PowerShell. Do not save the token in a file.

Use fresh output directories so the submitted predictions are preserved:

```text
python -m unlearning run --out results_fresh/main
python -m unlearning report --out results_fresh/main --figures results_fresh/main/figures
python -m unlearning run --methods gate --k 2 --strengths 0,0.5,1.0 --out results_fresh/k2
python -m unlearning report --methods gate --out results_fresh/k2 --figures results_fresh/k2/figures
python -m unlearning run --methods gate --k 4 --strengths 0,0.5,1.0 --out results_fresh/k4
python -m unlearning report --methods gate --out results_fresh/k4 --figures results_fresh/k4/figures
```

Inference selects an available execution device automatically; `--device cpu`
can be used explicitly. The machine must have enough memory for the model and
its activations. Execution time and fresh predictions can vary by environment.

`--methods` picks any subset of `gate,gate_margin`, `--k` sets the layer budget and
`--strengths` the strength grid. `report` takes `--methods` too, to restrict the tables and figures.

## Colab option

[Open the notebook in Colab](https://colab.research.google.com/github/LinaGolan/unlearning_task/blob/main/notebooks/full_screening.ipynb).
It clones this repository, regenerates the saved analysis, and optionally runs
fresh inference with an enabled `HF_TOKEN` secret and a GPU. Fresh predictions
are saved to Google Drive so completed conditions survive runtime disconnects.

## Reproduce screening

The submitted screening predictions, question statistics and selected splits are
in `results_fullscreen/`. `data_fullscreen/` preserves the full source pool and
compatible baseline predictions reused during this run; their provenance and
checksums are retained. No earlier experiment files are needed.

To screen all source questions afresh without reusing predictions:

```text
python -m unlearning.full_screening prepare --source results_fresh/source
python -m unlearning.full_screening run --source results_fresh/source --out results_fresh/screening --device cuda
python -m unlearning run --config results_fresh/screening/selected_config.json --data results_fresh/screening/selected_data --out results_fresh/rescreened_main
```

Fresh inference can change which questions pass. Use the committed selected data
(the CLI default) to repeat interventions on the submitted subset.

## Saved evidence

| Path | Contents |
| --- | --- |
| `results_fullscreen/selected_data/` | disjoint original questions and checksummed manifest |
| `results_fullscreen/selected_config.json` | pinned configuration for the selected splits |
| `results_fullscreen/question_statistics.jsonl` | original correctness and permutation counts for every source question |
| `results_fullscreen_experiment/main/` | both methods, localization, development sweeps, test results and controls |
| `results_fullscreen_experiment/k2/`, `k4/` | gate experiments with two and four selected layers |

Each experiment directory includes `setup.json` (environment and mechanism
checks), `localization.json`, per-question `predictions/`, `run.json`,
`analysis.json`, CSV tables, `report_tables.md` and `figures/`.

Runs resume completed conditions and cached localization. Use a fresh output
directory when changing the model, data, prompt, seed or layer budget. Screening
saves each completed batch. An interrupted experiment condition is rerun.

## Method summary

**Localization score.** Gate decoder block `l`'s residual contribution `r_l = h_out − h_in` by `g_l`,
so `h_out(g_l) = h_in + g_l·r_l`, and differentiate an objective `M(x)` at `g = 1`:
`s_l(x) = ∂M(x)/∂g_l`. Averaging over the localization questions gives `F_l` (forget), `R_l` (retain)
and the forget-vs-retain score `S_l = F_l − R_l`. All 16 gates come from one backward pass.

Two variants differ only in `M`:

| Method | Objective `M(x)` |
| --- | --- |
| `gate` | the correct letter's log probability, normalized over A–D |
| `gate_margin` | the decision margin `z_correct − max_{i≠correct} z_i` |

The margin measures how far the correct answer's score is above the strongest
incorrect answer. Comparing `gate` and `gate_margin` tests whether changing the
localization objective improves the forgetting–retention tradeoff.

**Intervention.** A forward hook scales the selected block's residual contribution,
`h' = h_in + (1 − α)·r_l`, so `α = 0` is the unmodified model and `α = 1` skips the block — a
generalized zero-ablation with `α` interpolating. It is a hook, so `α = 0` is bit-identical to the
original and removing it restores the model exactly. No weights change, which makes this reversible
suppression, not permanent knowledge erasure.

## Experimental discipline

The localization split chooses layers, and the development split chooses the
reported strength `α`. The test split evaluates the full predefined strength
grid to show how the effects change with intervention strength; test results
are not used to select the reported operating point. At that operating point,
the four conditions (top-k localized, top-k selective, random, bottom-k) share
the same `k` and `α` within each method.

## Repository layout

```
configs/experiment.json      pinned source configuration for screening
src/unlearning/data.py       download, deduplicate, split, checksum
src/unlearning/evaluate.py   prompt formats and next-token A-D scoring
src/unlearning/intervene.py  the intervention and the gate gradients
src/unlearning/localize.py   the gradient localization scores and stability checks
src/unlearning/experiment.py the driver: checks, localization, sweep, controls
src/unlearning/report.py     bootstrap intervals, tables, figures
src/unlearning/full_screening.py  correct-first screening and subset selection
notebooks/full_screening.ipynb     Colab runner
```
