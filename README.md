# Layer localization for selective unlearning

By **LinaGolan**.

Does mechanistic localization pick better intervention targets than random layer
selection? This repository localizes the decoder layers that carry a model's
WMDP-Bio behaviour, ablates them at inference time, and measures what that costs
on WMDP-Bio versus a general-knowledge retain set.

* **Model** `meta-llama/Llama-3.2-1B-Instruct` at revision `9213176726f574b556790deb65791e0c5aa438b6`, float32, 16 decoder layers, weights frozen throughout.
* **Forget set** `cais/wmdp`, config `wmdp-bio`.
* **Retain set** `cais/mmlu`, eight high-school subjects (US history, world history, geography, government and politics, microeconomics, psychology, physics, computer science).
* **Extra control** `cais/mmlu` `high_school_biology`, to separate "WMDP knowledge" from "biology knowledge".
* **Report** [`Research_report.html`](Research_report.html), a self-contained HTML file with embedded figures.

Every model and dataset revision is pinned; splits come from SHA-256 rankings, so
any machine reproduces the same questions.

## Environment

Use Python 3.10+ with the dependencies in `requirements.txt`. Regenerating tables and figures from the included
predictions needs no model download or Hugging Face token. Running inference
requires access to the configured Llama model.

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
python -m unlearning report --methods gate --out results/k2 --figures results/k2/figures
python -m unlearning report --methods gate --out results/k4 --figures results/k4/figures
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
python -m unlearning prepare
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

[Open the notebook in Colab](https://colab.research.google.com/github/LinaGolan/unlearning_task/blob/main/notebooks/experiment.ipynb).
It can clone this repository or accept a ZIP upload, run the same commands,
display the results and download an output archive. For saved-results analysis,
follow its setup instructions and skip to section 7. To run fresh inference,
follow section 6 and use a fresh output directory.

## What each step produces

| Path | Contents |
| --- | --- |
| `data/` | `forget/`, `retain/`, `biology/` splits as JSONL plus `manifest.json` with per-file checksums |
| `results/setup.json` | environment, the exact scored prompt for each format, and the mechanism checks |
| `results/localization.json` | per-layer scores for each objective, with split-half stability |
| `results/predictions/*.jsonl` | per-question predictions, one file per (split, method, selection, strength) |
| `results/run.json` | selected layers, the development-chosen strength, timings |
| `results/analysis.json` | accuracies, paired bootstrap intervals, controls |
| `results/table_*.csv` | the assignment's required comparison table, one per method |
| `results/report_tables.md` | generated numerical tables, including the Appendix A gradient diagnostic |
| `report/figures/*.png` | localization, strength-curve and tradeoff figures |
| `results/k2/`, `results/k4/` | the same artefacts for the `k = 2` and `k = 4` layer-budget arms |

`run` is resumable: a condition whose `results/predictions/*.jsonl` file already
exists is not re-scored, so an interrupted run continues where it stopped. Cached
localization scores are reused, and any method missing from the cache (after a
`--methods` change) is computed and merged. Because prediction filenames do not
record the layer budget, a run refuses to reuse an output directory built with a
different `k`, model, dataset, prompt or seed — use a fresh `--out` for those.

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

The margin is the quantity whose sign decides whether the prediction is right, which matters because
accuracy changes only when the argmax flips. `gate` alone satisfies the assignment; `gate_margin` tests
whether the first variant's failure is the objective rather than the idea.

**Intervention.** A forward hook scales the selected block's residual contribution,
`h' = h_in + (1 − α)·r_l`, so `α = 0` is the unmodified model and `α = 1` skips the block — a
generalized zero-ablation with `α` interpolating. It is a hook, so `α = 0` is bit-identical to the
original and removing it restores the model exactly. No weights change, which makes this reversible
suppression, not permanent knowledge erasure.

## Experimental discipline

The localization split chooses layers, the development split chooses the
strength `α`, and the test split is scored only once with both frozen. The
assignment's four conditions (top-k localized, top-k selective, random, bottom-k)
share the same `k` and the same `α` inside each method.

## Repository layout

```
configs/experiment.json      every pinned revision, split size, strength and seed
src/unlearning/data.py       download, deduplicate, split, checksum
src/unlearning/evaluate.py   prompt formats and next-token A-D scoring
src/unlearning/intervene.py  the intervention and the gate gradients
src/unlearning/localize.py   the gradient localization scores and stability checks
src/unlearning/experiment.py the driver: checks, localization, sweep, controls
src/unlearning/report.py     bootstrap intervals, tables, figures
notebooks/experiment.ipynb   Colab runner
```

The submission includes the two gradient methods, their main results and the
`k = 2`/`k = 4` results. Development predictions, individual random controls and
extra biology/prompt controls are retained so the analyses can be regenerated.
Historical setup metadata is preserved, including the legacy
`direction_alpha0_identity` check; no direction-ablation experiment is part of
the current submission. Download caches, regenerated datasets, scratch files and
the assignment handout are excluded from Git.
