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
* **Report** [`report/Research_report.md`](report/Research_report.md).

Every model and dataset revision is pinned; splits come from SHA-256 rankings, so
any machine reproduces the same questions.

## Environment

Python 3.9+ and one GPU with ≥8 GB. The saved runs used a single NVIDIA
A100-SXM4-80GB with Python 3.12.11; the main grid took about one hour.
A CPU-only machine works but is far slower.

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=...          # Llama-3.2 is a gated repository
```

`requirements.txt` provides a Python 3.13 stack with torch 2.11.0 and
transformers 5.16.1, and an older stack for earlier Python versions. The exact
saved run environment is recorded in `results/run.json`; the requirements on
Python 3.12 select the older stack, not that run's environment.

## Commands

```bash
export PYTHONPATH=src
python -m unlearning prepare             # download + split the questions
python -m unlearning run                 # the experiment (single GPU)
python -m unlearning report              # tables, intervals and figures
```

`notebooks/experiment.ipynb` runs these steps on Colab and zips the results.

The layer budget and the strength grid can be overridden without editing the
config, which is how the supplementary `k = 2` and `k = 4` arms were produced:

```bash
python -m unlearning run --k 2 --strengths 0,0.5,1.0 --out results/k2
python -m unlearning report --out results/k2 --figures results/k2/figures
```

`--methods` picks any subset of `gate,gate_margin,direction`, `--k` sets the layer budget and
`--strengths` the strength grid. `report` takes `--methods` too, to restrict the tables and figures.

## What each step produces

| Path | Contents |
| --- | --- |
| `data/` | `forget/`, `retain/`, `biology/` splits as JSONL plus `manifest.json` with per-file checksums |
| `results/setup.json` | environment, the exact scored prompt for each format, and the mechanism checks |
| `results/localization.json` | per-unit scores for every active method, with split-half stability |
| `results/directions.json` | the unit direction and retain-mean projection ablated by `direction` |
| `results/predictions/*.jsonl` | per-question predictions, one file per (split, method, selection, strength) |
| `results/run.json` | selected layers, the development-chosen strength, timings |
| `results/analysis.json` | accuracies, paired bootstrap intervals, controls |
| `results/table_*.csv` | the assignment's required comparison table, one per method |
| `results/report_tables.md` | every table the report quotes, generated rather than transcribed |
| `report/figures/*.png` | localization, strength-curve and tradeoff figures |
| `results/k2/`, `results/k4/` | the same artefacts for the `k = 2` and `k = 4` layer-budget arms |

`run` is resumable: a condition whose `results/predictions/*.jsonl` file already
exists is not re-scored, so an interrupted run continues where it stopped.

## Method summary

Three localization/intervention pairs are compared on the same 128 forget and 128 retain
localization questions. Each supplies a **WMDP-only** score and a **forget-vs-retain** score, so the
assignment's four conditions (top-k localized, top-k selective, random, bottom-k) are defined
identically for all of them.

| Method | WMDP-only score | Forget-vs-retain score | Intervention |
| --- | --- | --- | --- |
| `gate` | `F_l` on correct-answer log probability | `S_l = F_l − R_l` | scale the block's residual contribution |
| `gate_margin` | `F_l` on the decision margin | `S_l` on the margin | same |
| `direction` | `C_l` (relative residual update) | `A_l` (probe accuracy) | ablate one residual direction |

**Gradient attribution (`gate`, `gate_margin`).** Gate block `l`'s residual contribution
`r_l = h_out − h_in` by `g_l`, so `h_out(g_l) = h_in + g_l·r_l`, and differentiate an objective
`M(x)` at `g = 1`: `s_l(x) = ∂M(x)/∂g_l`. Averaging gives `F_l` (forget), `R_l` (retain) and
`S_l = F_l − R_l`. The two variants differ only in `M`: `gate` uses the correct letter's log
probability normalized over A–D, while `gate_margin` uses the **decision margin**
`z_correct − max_{i≠correct} z_i`, which is the quantity whose sign decides whether the prediction
is right. The intervention re-scales the whole block: `h' = h_in + (1 − α)·r_l`.

**Activation statistics (`direction`).** From each block's output at the final prompt token, `C_l` is
the mean relative residual update on forget prompts,
`d_l = mean_forget h_out,l − mean_retain h_out,l` with `u_l = d_l/‖d_l‖`, and `A_l` is the held-out
split-half accuracy of thresholding `h·u_l`. The intervention removes only that one direction,
moving its component to the retain mean: `h' = h_out − α·((h_out·u_l) − m_l)·u_l`.

Both interventions are forward hooks, so `α = 0` is bit-identical to the unmodified model and
removing the hook restores it exactly. No weights change, which makes this reversible suppression,
not permanent knowledge erasure.

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
src/unlearning/intervene.py  the two interventions and the gate gradients
src/unlearning/localize.py   the localization scores (gradient, activation)
src/unlearning/experiment.py the driver: checks, localization, sweep, controls
src/unlearning/report.py     bootstrap intervals, tables, figures
notebooks/experiment.ipynb   Colab runner
```
