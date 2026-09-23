# Research Plan: Layer Localization and Selective Unlearning

**Target meeting:** Thursday, September 24, 2026.

**Primary model:** `meta-llama/Llama-3.2-1B-Instruct`, as recommended in the assignment.

**Status:** Stages 1–6 and their analyses are complete. The explicit-answer-prefix baseline passed on the T4: 60.9% forget and 53.9% retain development accuracy. Its 256 predictions and 432 diagnostic records, plus the Stage 3 intervention checks, were verified on September 20, 2026. Stage 3 passed all correctness and feasibility checks; we will use the full profile. Stage 4 localization and Control A are complete: all 1,312 records were verified. The fixed top-forget pair is [15, 0], the top-selective pair [15, 7], and the bottom-forget pair [6, 9] (zero-based). Retain and selectivity rankings were unstable across halves; small-intervention predictions were accurate, with weaker agreement at strength 0.5. Stage 5 was verified on September 21, 2026: all 25,088 new predictions passed review. Development selected strength 0.75; the selective pair then reduced final-test forget accuracy by 9.38 points and retain accuracy by 11.72 points. Retain preservation did not carry over, and the random comparison is inconclusive. The unchanged settings were used in Stage 6, verified September 21: all 2,880 new control predictions and 2,304 primary references passed review. Biology showed an uncertain selective drop, the one alternative instruction did not clearly change the intervention effect, and answer-letter shifts persisted. Planned GPU runs are complete; source and verified evidence packages accompany the repository. See [STAGE5_RESULTS.md](STAGE5_RESULTS.md) and [STAGE6_RESULTS.md](STAGE6_RESULTS.md). See [STAGE2.md](STAGE2.md), [DIAGNOSTIC.md](DIAGNOSTIC.md), [STAGE3.md](STAGE3.md), and [STAGE4.md](STAGE4.md).

## 1. Goal and scope

**Assignment coverage:** Section 1 (Objective). The research questions are answered in the final report described in plan section 9.

The main research question is:

> Can an internal model signal identify better layers for selective intervention than random layer selection?

We want to reduce performance on WMDP-Bio while causing as little damage as possible to general performance.

The target for Thursday is a complete experiment on a manageable sample, reproducible code, results and plots, and a short research report. We will work step by step and review the reasoning behind each part. The code, README, and report will be in English.

This is a reasonable target with 3-4 hours of active work per day, provided that we obtain a free GPU and pass the early feasibility checks. Your computer will be used for development and analysis. The main experiments will run in Google Colab. Free GPU access is not guaranteed, so runs must save progress and support restarting.

We will focus on one model, one main localization method, and one intervention method. Success means a fair experiment and clear conclusions. It does not require our method to beat random selection.

## 2. Model and datasets

**Assignment coverage:** Section 2 (Model and Dataset). The general-biology dataset also supports Section 9 (Causal Validation). The separate localization, development, and test splits are additional experimental safeguards.

### Model choice and access

Start with `meta-llama/Llama-3.2-1B-Instruct`. This follows the assignment's recommendation directly.

Check access to the model before building the main experiments. The official Hugging Face repository requires accepting the model's access conditions. Account login and any required acceptance must be handled by the user. Do not put access tokens in code, notebooks, saved outputs, or Git.

First measure baseline performance. Do not assume that the model knows enough about WMDP-Bio simply because it is the recommended model.

If baseline performance is too low, check the evaluator first. If the evaluator is correct, try `meta-llama/Llama-3.2-3B-Instruct` once as a larger fallback, subject to access and available GPU memory. Document this change and its reason. Do not run an open-ended search across models.

### Dataset roles

| Dataset | Purpose |
| --- | --- |
| WMDP-Bio | Measure performance on the target domain |
| A balanced sample from eight MMLU subjects | Measure preservation of general performance |
| A small MMLU general-biology sample | Check whether the intervention harms biology more broadly |

Use these eight MMLU retain subjects:

- `high_school_us_history`
- `high_school_world_history`
- `high_school_geography`
- `high_school_government_and_politics`
- `high_school_microeconomics`
- `high_school_psychology`
- `high_school_physics`
- `high_school_computer_science`

Sample equal numbers of questions from each retain subject. This gives a reasonable range of tasks, but it does not measure every general capability.

Use `high_school_biology` for the separate biology control.

### Separate data for separate decisions

For both WMDP-Bio and the main retain dataset, create three groups with no overlap:

| Group | Questions per dataset | Purpose |
| --- | ---: | --- |
| Localization | 128 | Calculate scores and select layers |
| Development | 128 | Check feasibility and select an operating strength |
| Final test | 256 | Measure results after decisions are fixed |

The retain groups must remain balanced across the eight subjects. Save the question IDs and random seed. Reduced profiles use subsets within these same full-profile split boundaries, so changing profile cannot move localization questions into the test split. Filter duplicate question text across source pools before sampling and record the exclusions.

These are our own experimental splits, not official WMDP splits. Do not choose layers, change the score formula, or select the operating strength using final-test results.

## 3. Baseline evaluation

**Assignment coverage:** Section 3 (Baseline Evaluation), including accuracy, example counts, and per-example predictions. These measurements also supply the baseline row required by Section 11 (Required Results).

Present each question with four answer choices. Compare the model's scores for A, B, C, and D, and select the highest-scoring answer. Do not ask the model to generate long explanations.

Verify that the tokenizer represents the answer labels correctly and that scores come from the correct position after the prompt. Use a consistent prompt format appropriate for the instruction-tuned model, and save the exact template.

**Reviewed prompt decision (September 20):** Use `configs/baseline_prefill.json`: append `The correct answer is` to the assistant header and score the single-token suffixes ` A`, ` B`, ` C`, ` D`. A fixed development-only diagnostic showed that this reduces the original strong A preference; the subsequent full development run passed the ability checks. Keep this scoring format fixed across baselines, localization, and interventions. Preserve the original results and report prompt sensitivity. The diagnostic does not replace the planned final-test wording-robustness check.

For every question, save:

- Question ID and dataset split.
- Correct answer and model prediction.
- Scores for all four answer choices.
- Model support for the correct answer.
- Experiment settings.

Report accuracy and the number of evaluated questions for both datasets. Saving a continuous correct-answer score lets us detect changes even when the predicted answer does not change.

## 4. Localization method

**Assignment coverage:** Sections 4 (Layer-wise Localization) and 5 (Forget-vs-Retain Localization), plus the localization tables and plots required by Section 11 (Required Results).

### Main idea

Ask: **If we slightly weaken a layer's contribution, how much should the model's support for the correct answer fall?**

Let a layer receive an internal representation \(h_{\ell,\mathrm{in}}\) and produce \(h_{\ell,\mathrm{out}}\). Define its contribution as:

\[
r_\ell = h_{\ell,\mathrm{out}} - h_{\ell,\mathrm{in}}.
\]

Introduce a multiplier \(g_\ell\):

\[
h'_{\ell,\mathrm{out}} = h_{\ell,\mathrm{in}} + g_\ell r_\ell.
\]

When every multiplier is 1, the model behaves normally.

Let \(M(x)\) be the log probability of the correct answer, normalized across the four answer choices. Define the layer score for question \(x\) as:

\[
s_\ell(x) = \left.\frac{\partial M(x)}{\partial g_\ell}\right|_{\mathbf{g}=\mathbf{1}}.
\]

In plain language, this derivative measures how sensitive the correct-answer score is to a small change in the layer's contribution. A large positive score predicts that weakening the layer will reduce support for the correct answer.

Calculate scores for all layers using a forward pass and a backward pass. Keep model weights fixed: this is measurement, not training.

The main advantage is a clear connection between the measurement and the intervention. The main limitation is that a local gradient may not predict a large intervention. Effects from several layers may also interact rather than simply add together.

This is a simple gradient-based sensitivity method. It is related to attribution methods in interpretability research, but we will not describe it as a full implementation of AtP*.

### Forget and retain scores

This subsection directly addresses assignment Section 5. The preceding method definition addresses Section 4.

Average the score separately over localization questions from each dataset:

\[
L_F(\ell) = \operatorname{mean}_{x\in F}s_\ell(x),
\qquad
L_R(\ell) = \operatorname{mean}_{x\in R}s_\ell(x).
\]

Define selectivity as:

\[
S(\ell) = L_F(\ell) - L_R(\ell).
\]

A high selectivity score means that the layer supports correct answers more strongly on forget questions than on retain questions. Use a difference rather than a ratio, which could become unstable near a zero denominator.

Keep negative scores. They may indicate that weakening a layer would improve the correct-answer score. Use all localization questions, not only questions the model initially answers correctly.

Produce a table and plot for every layer, showing the forget score, retain score, and selectivity score. Discuss differences in dataset difficulty as a possible limitation of this comparison.

## 5. Intervention and required comparisons

**Assignment coverage:** Sections 6 (Implement a Causal Activation Intervention), 7 (Localization vs. Random Controls), 8 (Intervention-Strength Ablation), and the comparison table and curves in Section 11 (Required Results).

### Intervention definition

**Assignment section:** 6.

Reduce the layer's contribution while preserving its input:

\[
h'_{\ell,\mathrm{out}} = h_{\ell,\mathrm{in}} + (1-\alpha)r_\ell.
\]

- At strength 0, the layer behaves normally.
- At strength 0.5, half of the layer's contribution is retained.
- At strength 1, the layer passes its input forward without its contribution.

Apply the intervention at all relevant token positions in the selected layers. This is a defined alternative to zeroing the entire layer output, as permitted by the assignment.

Support arbitrary layer lists and intervention strengths, and allow interventions to be enabled and disabled without changing model weights.

### Layer-selection strategies

**Assignment sections:** 7 for the four selection strategies; 8 for the strength sweep.

Use two selected layers in the main experiment:

1. The two layers with the highest forget score.
2. The two layers with the highest selectivity score.
3. Two randomly selected layers, repeated with five fixed, distinct random selections.
4. The two layers with the lowest forget score.

Use strengths **0, 0.25, 0.5, 0.75, and 1** for every selection. Each method keeps its selected layers fixed across strengths. All methods receive the same number of layers and the same intervention strength.

Show all curves and variation between random selections. Five random selections are a useful control, but do not justify strong claims about every possible random selection.

### Metrics and the main results table

**Assignment sections:** 7, 8, and 11. These results provide evidence for research questions RQ1-RQ3 in Section 12.

For each condition, report forget accuracy, retain accuracy, and their changes from baseline:

\[
\Delta_F = \mathrm{Acc}_{\mathrm{base},F} - \mathrm{Acc}_{\mathrm{intervention},F},
\]

\[
\Delta_R = \mathrm{Acc}_{\mathrm{base},R} - \mathrm{Acc}_{\mathrm{intervention},R}.
\]

The main question is whether a method produces a larger forget drop for a similar amount of retain damage. Plot this trade-off as well as accuracy against intervention strength.

Choose one common strength for the main comparison table using development data only:

1. Consider strengths where the selective method loses no more than five percentage points of retain accuracy.
2. Choose the strength with the largest forget accuracy drop.
3. If tied, choose the weaker intervention.

If no nonzero strength reduces forget accuracy within the retain limit, report that no suitable operating point was found. Still show all curves and use strength 0.5 for the comparison table.

The five-point limit is a practical rule chosen before the experiment, not a universal definition of selectivity.

Use paired bootstrap intervals: resample question IDs and compare methods on the same sampled questions. Report uncertainty across questions separately from variation across random layer selections. Avoid treating tiny differences as clear improvements.

## 6. Causal validation and robustness

**Assignment coverage:** Sections 9 (Causal Validation) and 10 (Robustness), the control results in Section 11 (Required Results), and evidence for RQ4-RQ6 in Section 12 (Research Questions).

### Control A: Does the score predict an actual intervention?

**Assignment section:** 9. This also examines a limitation of the localization method required by Section 4.

Use 32 fixed development questions, 16 from each main dataset. For every layer separately, apply strengths 0.05 and 0.5.

Compare the predicted correct-answer-score drop, approximately \(\alpha s_\ell(x)\), with the measured drop. Examine agreement in values and rankings.

This tests whether the score predicts small interventions and whether it becomes less reliable for stronger interventions. Poor agreement can be a valid research finding, not necessarily a software error.

### Control B: Does general biology also suffer?

**Assignment section:** 9, by testing an alternative explanation for the apparent selectivity.

Use 64 general-biology questions. Evaluate the baseline and all layer-selection conditions at the common strength chosen during development.

If general biology also suffers, narrow the conclusion: the intervention may affect biology broadly rather than WMDP specifically. Differences in difficulty and question style still limit this comparison.

### Control C: Does the result survive different wording?

**Assignment section:** 10. The answer-letter analysis also supports Section 9 by checking for an intervention artifact.

Use a fixed subset of 128 final-test questions from each main dataset. Change the instruction wording but keep the questions and answer choices unchanged. Do not select new layers.

Evaluate the baseline and interventions under the new wording. Compare each intervention with the baseline from that same wording.

Also inspect predicted answer-letter distributions. This can reveal whether the intervention makes the model choose one letter almost every time.

### Limits on causal claims

An intervention can establish that changing a component affects behavior. It does not by itself prove that domain knowledge is stored in that component.

Discuss general degradation, disrupted reasoning, prompt sensitivity, distributed knowledge, and intervention artifacts. Describe the result as an inference-time intervention effect, not permanent knowledge deletion.

## 7. Checkpoints and fallback rules

**Assignment coverage:** Supporting safeguards for Sections 2-10 and the limitations discussed in Section 12. These checkpoints, thresholds, and fallback rules are our project-management choices, not extra requirements stated by the mentor.

| Checkpoint | What we check | Action if there is a problem |
| --- | --- | --- |
| Model access and GPU | Can we access the recommended model, obtain a GPU, and load it? | Complete the access step and try another free-GPU time window. If still blocked by Sunday evening, target working code and feasibility findings for the meeting. Do not automatically switch to paid compute. |
| Runtime and memory | Time normal evaluation and gradient computation on a small sample | Reduce batch size and use short run segments. If the projected full experiment exceeds about six GPU hours, choose the reduced profile before final testing: 64 localization, 64 development, and 128 test questions per main dataset. |
| Baseline ability | Is there enough performance to study a reduction? | Use practical thresholds of at least 35% forget accuracy and 40% retain accuracy on development data, with the lower bound of each 95% accuracy interval above 25%. Check the evaluator first, then try the 3B Llama fallback once if needed. |
| Implementation | Are gradients and interventions correct? | Stop and fix failures before large runs. Do not treat a software error as a finding. |
| Ranking stability | Do two halves of the localization data give similar rankings? | Report stability. If low, describe the ranking as noisy and limit conclusions. Do not search for a new formula just to obtain a better result. |
| Research outcome | Does localization beat random selection? | Preserve negative results. Analyze whether the limitation concerns the gradient approximation, selectivity, or measurement. |

The baseline thresholds are feasibility rules, not proof of meaningful domain knowledge. If the fallback model also performs poorly, stop expanding the model search. Present the working evaluator, measured limitation, and a focused next experiment.

**Technical failures and feasibility limits can justify a change. A result that does not support the hypothesis is still a result.**

## 8. Implementation and verification

**Assignment coverage:** Section 13 (Deliverables), especially reproducible code, experiment scripts, and result processing. The checks support the correctness of the experiments in Sections 3-10.

### How we will implement the project

**We will implement one stage at a time, not build the entire project in one pass.** The full plan defines the destination; the stages below define the order of work.

For each stage, implement the smallest working version, run its checks, and explain what changed, what the results mean, and any remaining problems. Review the result together before moving to the next stage. Do not start a large dependent experiment while an earlier correctness or feasibility check is unresolved.

| Stage | What we build or run | Completion check | Assignment sections |
| --- | --- | --- | --- |
| 1. Setup and data | Model access, Colab setup, configuration, saved question splits, and a tiny end-to-end run | Model loads, questions and labels are correct, splits do not overlap, and outputs can be saved | 2 and the setup/reproducibility parts of 13 |
| 2. Baseline | Answer scoring, per-question predictions, baseline accuracy, and runtime measurements on development data | Evaluator checks pass; apply the baseline and runtime rules in plan section 7, make a provisional model/profile decision and confirm actual gradient cost in Stage 3 | 3; baseline evidence for 11 |
| 3. Intervention mechanism | Layer multipliers, strength control, and enabling/disabling interventions | Strength zero matches baseline; full strength passes the layer input; cleanup and numerical gradient checks pass | 6; implementation foundations for 4 |
| 4. Localization | Forget and retain scores, selectivity ranking, per-layer plots, ranking stability, and Control A | Scores and plots are saved; technical checks pass; approximation quality is documented and layer selections are fixed | 4, 5, part of 9, and localization outputs for 11 |
| 5. Main comparisons | Four selection strategies, random repetitions, and the strength sweep | Select the common strength on development data, freeze the settings, then save final-test predictions and comparison results | 7, 8, and the main comparison outputs for 11 |
| 6. Remaining controls | General-biology control, alternative prompt evaluation, and answer-letter analysis | Save results with matched baselines and explain what they do and do not establish | Remaining work for 9, plus 10 and control outputs for 11 |
| 7. Analysis and delivery | Final tables and plots, uncertainty estimates, report, README, reproduction instructions, and meeting notes | Every assignment requirement in plan section 10 has an output or an explicitly stated limitation; all six research questions are answered | 11, 12, and 13 |

**Stages 1, 2, and 3 are complete.** Stage 3 real-model evidence was verified on September 20, 2026: all correctness checks and all 12 numerical derivative comparisons passed. Peak allocated GPU memory was 7.18 GiB on the T4; the provisional full-experiment estimate is 1.89 GPU hours. **The full profile is selected for the next stage.** See [STAGE3.md](STAGE3.md) for the evidence. Stage 4 is also complete: all 1,312 records, layer selections, statistics, and figures were reviewed on September 20, 2026. The fixed top-forget pair is [15, 0], the top-selective pair [15, 7], and the bottom-forget pair [6, 9]. Selectivity stability was limited (half-to-half Spearman 0.476, zero top-pair overlap); retain stability was weaker still (0.115). Control A agreed closely at strength 0.05, with weaker agreement at 0.5. Preserve these findings and the frozen selections. See [STAGE4.md](STAGE4.md) for detailed evidence. Stage 5 is complete: all 25,088 new predictions and the paired statistics were verified on September 21, 2026. Strength 0.75 was frozen on development data. Final-test selective drops were 9.38 points on forget and 11.72 points on retain, with no clear advantage over the random-pair mean. Preserve this negative result and the frozen settings. See [STAGE5_RESULTS.md](STAGE5_RESULTS.md). Stage 6 is complete and verified on September 21: all 2,880 new predictions and 2,304 reused references passed review. Biology showed a 7.81-point selective drop with an interval including zero; matched prompt-effect differences also included zero, and answer-letter shifts persisted. These controls do not overturn the negative main result. See [STAGE6_RESULTS.md](STAGE6_RESULTS.md). The planned GPU experiments are complete; source and verified evidence packages accompany the repository. A working stage is progress, not completion of the whole assignment. Negative scientific findings do not block the next stage if the implementation is correct; technical failures must be resolved or reported as a feasibility limit.

### Code structure and saved outputs

Build a small Python package with separate components for data preparation, evaluation, localization, interventions, experiment execution, and analysis.

A Colab notebook will run these components and explain each step. The core experiment must not depend on manually running notebook cells in a fragile order.

Provide command-line stages: `prepare`, `baseline`, `localize`, `sweep`, `controls`, and `report`.

A configuration file will specify model and dataset versions, splits, random seeds, selected layers, and strengths. Record the final configuration with each run.

Save model, dataset, and library versions; per-question predictions; scores; summary tables; plots; and progress information. Support resuming interrupted experiments without repeating completed work.

Export and download results after each run segment. Do not rely on Colab temporary storage surviving disconnection. Do not modify the existing Python environment on the local computer for the main experiment.

### Tests before large runs

- No overlap between localization, development, and test questions.
- Correct mapping between answer labels, correct answers, and model scores.
- Strength zero produces the same output as normal evaluation.
- Removing the intervention restores baseline behavior, without leftover hooks.
- Full intervention passes the selected layer's input forward.
- A small numerical check confirms gradient sign and value.
- Single-question and batched evaluation agree within numerical tolerance.
- Resuming does not duplicate questions or mix configurations.
- Localization reads only the localization split.

Develop these checks with a tiny model and synthetic questions. Then perform a short end-to-end check using the research model.

## 9. Schedule and deliverables

**Assignment coverage:** Sections 11 (Required Results), 12 (Research Questions), and 13 (Deliverables). The dates are our working schedule, not a deadline explicitly stated in the assignment document.

| Time | Expected output | Concepts to review |
| --- | --- | --- |
| Saturday-Sunday | Model access, runtime setup, splits, evaluator, baseline | Forward passes, logits, tokenization, accuracy |
| Monday | Tested intervention and layer scores | Residual connections, gradients, localization formula |
| Tuesday | Development results and final comparisons | Fair controls, randomness, separation of development and testing |
| Wednesday | Controls, plots, report, meeting preparation | Evidence, limitations, next experiments |
| Thursday | Final review | Explaining the decisions and findings independently |

Aim to stop adding experiments by Wednesday noon, leaving time for analysis and understanding.

Deliver:

- A clean code repository and a Colab notebook.
- An English README with setup instructions, dependencies, model and dataset details, commands, and expected outputs.
- An English report of about 4-6 pages.
- A short meeting preparation note explaining the reasoning, findings, limitations, and likely questions.

The report must include the required comparison table, layer localization plot, forget-versus-retain plot, intervention-strength curves, and control results. Explicitly answer all six assignment questions:

1. Does localization identify interventions with a larger WMDP effect than random selection?
2. Is the effect selective, with limited retain damage?
3. Does forget-versus-retain localization improve on forget-only localization?
4. What evidence supports causal relevance rather than correlation?
5. What are the main limitations of the localization method?
6. What experiment should come next to distinguish knowledge localization from general disruption?

Do not include retraining, a broad method search, or a user interface in this version.

The central explanation for the meeting is:

> We used an internal signal to predict useful intervention targets, tested those predictions, and measured whether the resulting damage was more selective than random or general disruption.

## 10. Assignment coverage checklist

**Assignment coverage:** All 13 sections. This is a tracking checklist, not an additional experiment. All planned experiments and reporting are complete; the reproducible code is versioned in Git. See [ASSIGNMENT_COVERAGE.md](ASSIGNMENT_COVERAGE.md) for checked evidence and [README.md](README.md) for repository contents. The experiment did not establish selective forgetting; a negative result does not make its implementation incomplete.

The assignment section numbers below refer to the original document, `Mechanistic Interpretability & Machine Unlearning - EX2.docx`. Plan section numbers refer to this file.

| Assignment section | Where it is addressed in this plan | Evidence required for completion |
| --- | --- | --- |
| 1. Objective | 1; interpretation and report in 6 and 9 | A clear research question and a conclusion about localization versus random selection and selective effects |
| 2. Model and Dataset | 2; feasibility checks in 7 | Model details, forget/retain dataset details, saved sample IDs, and reproducible setup |
| 3. Baseline Evaluation | 3; verification in 8 | Forget and retain accuracy, evaluated example counts, and saved per-example predictions |
| 4. Layer-wise Localization | 4; method check in 6 | Mathematical score definition, rationale, limitations, and a score table and plot for every layer |
| 5. Forget-vs-Retain Localization | 4 | The same signal measured on both datasets, an explicit selectivity formula, and layer scores and plots |
| 6. Implement a Causal Activation Intervention | 5; tests in 8 | A defined intervention supporting arbitrary layer selection, adjustable strength, and enabled/disabled evaluation |
| 7. Localization vs. Random Controls | 5 | Top-localized, top-selective, random, and bottom-ranked comparisons with the same layer count and strength; both accuracies and baseline drops |
| 8. Intervention-Strength Ablation | 5 | Results across the planned strengths, accuracy curves, and a comparison of forget reduction against retain damage |
| 9. Causal Validation | Controls A and B and causal interpretation in 6 | Additional experiments or analyses addressing alternative explanations, with appropriately limited causal claims |
| 10. Robustness | Control C in 6 | Original and alternative-prompt results using fixed layer selections and a baseline for each prompt format |
| 11. Required Results | 3-6; assembled in 9 | Required comparison table, localization plots, strength curves, and at least one robustness/control result |
| 12. Research Questions | Evidence in 4-7; explicit written answers in 9 | Answers to RQ1-RQ6, including limitations and a proposed next experiment beyond the controls already performed |
| 13. Deliverables | 8 and 9 | Clean reproducible repository, README, experiment and analysis code, and research report |

An item is complete only when its evidence exists and has been checked. If time or compute prevents completion, label that item incomplete and explain what remains; do not count a planned experiment as a result.

## References

These sources support the model, dataset, method, and setup decisions above. They are background references, not a separate assignment requirement.

- [Assignment's recommended model: Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Larger fallback: Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [WMDP dataset](https://huggingface.co/datasets/cais/wmdp)
- [MMLU dataset](https://huggingface.co/datasets/cais/mmlu)
- [AtP*: background on gradient approximations and their limitations](https://arxiv.org/abs/2403.00745)
- [Colab FAQ and resource limits](https://research.google.com/colaboratory/faq.html)
