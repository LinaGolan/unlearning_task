# Stage 2: Answer-letter bias investigation

The original baseline completed 256 development questions, but failed the planned ability checks:

| Dataset | Correct | Accuracy | 95% Wilson interval | A predictions |
| --- | ---: | ---: | --- | ---: |
| Forget | 36/128 | 28.1% | 21.1%-36.5% | 111/128 |
| Retain | 42/128 | 32.8% | 25.3%-41.3% | 97/128 |

The original evidence is preserved under `outputs/stage2/colab_original_2026-09-19/`.
All data checks and 256 prediction checksums were verified. No top-score ties occurred.
The saved prompts have the expected chat headers, question/choice text, and assistant generation header.
The saved code matches the local evaluator used for that run. Numerical consistency tests passed on the T4.
Those checks establish how the scores were computed, not that the model interpreted the task as intended.

## A separate portability fix

The offline verifier originally compared derived probabilities with exact floating-point equality.
Windows and Colab can differ in the last binary digits of logarithms/exponentials: one observed difference
was about 2e-16. The verifier now allows 1e-12 absolute/relative tolerance for recomputed numeric fields.
Checksums, labels, Boolean correctness, and input metadata remain strictly checked. This fix changes no
predictions and does not explain the A bias. A regression test distinguishes harmless rounding from
changed scores or predictions.

## Fixed diagnostic protocol

Before running new predictions, fix 16 forget development questions and two from each of the eight retain
subjects by a seeded hash. Selection never uses model predictions or correct-answer labels. Add four
invented, harmless sanity questions. For each question run four cyclic choice rotations, remapping the
correct answer and the model's prediction back to the original choice content.

There are three fixed conditions, totaling 432 scored prompts:

1. **Original:** reproduce the saved prompt and scoring convention.
2. **Assistant prefix:** keep the same user question but start the assistant reply with
   `The correct answer is`, then score the one-token suffixes ` A`, ` B`, ` C`, ` D`.
   This tests whether the initial answer-letter position is ambiguous. It is a diagnostic condition,
   not a silent replacement for the baseline.
3. **Reversed labels:** preserve the label-content mapping but display D, C, B, A from top to bottom.
   This helps distinguish preferring the label A from preferring the first displayed answer.

For all conditions, verify the answer-token boundary in the complete prompt. Save exact prompts,
token IDs, label scores, the unconstrained most-likely next token, and per-question checkpoints.
Replay all 32 original development prompts against the saved scores, and compare four real questions
against a direct all-position model forward pass without the optimized last-logit path or explicit
position IDs. On the four harmless invented questions only, also inspect up to 12 greedy continuation
tokens to see whether the model starts a sentence or supplies a letter. No free-text generation is
performed on WMDP questions.

Report accuracy both in the original order and averaged over rotations, label frequencies, and
whether the chosen **content** stays the same after reordering. The four rotations of one question
are correlated; do not treat them as four independent observations or apply the full development
ability thresholds to this small diagnostic sample.

The final test split stays reserved. The diagnostic does not automatically select the highest-scoring
prompt, switch models, average away bias for the final experiment, or change the original predictions.

## Running and reproducing

In the already configured Stage 2 Colab runtime, the additional cell in
`scripts/stage2_diagnostic_colab_cell.py` installs only the diagnostic module, runs the checks, and
downloads `stage2-diagnostic-results.zip`. An equivalent cell is saved in
`notebooks/02b_stage2_diagnostic.ipynb`; copy it into the existing Stage 2 notebook rather than
starting an empty runtime. With the current project installed, the equivalent command is:

```console
python -m unlearning.diagnostics
```

Results are separate in `outputs/stage2_diagnostic/`. Keep both the original Stage 2 ZIP and the
diagnostic ZIP. Subsequent decisions must explain whether the evidence points to numerical scoring,
answer format, label preference, display position, or limited model ability. If a prompt revision is
justified, save it as a new condition and rerun the full development baseline; do not overwrite the
original run or choose based on final-test accuracy.

The local suite has 35 passing tests, including an end-to-end tiny-Llama diagnostic and checks for
answer remapping, outcome-independent sampling, content consistency, and preservation of the original run.

## Background

Option-ID and order sensitivity are known evaluation issues, but prior work does not establish the
cause in this particular run. See [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/abs/2309.03882)
and [Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions](https://arxiv.org/abs/2308.11483).
The answer-prefix comparison uses the continuation framing described in the official
[chat-template documentation](https://huggingface.co/docs/transformers/chat_templating).

## Verified outcome (September 20, 2026)

Both latest downloads were preserved and verified in `outputs/stage2/verified_2026-09-20/`.
All 432 diagnostic records passed checksums; their summaries were independently recomputed.
All 32 original-prompt replays matched the saved scores exactly. Four direct native-forward comparisons
passed the specified tolerance. No scoring-index error was found in these checks.

When A was displayed last, the original-format model still chose it on 57/64 forget and 55/64 retain
rotated evaluations. This supports a label preference rather than a simple first-displayed-option preference.
The explicit assistant prefix reduced A selections to 16/64 and 22/64 respectively. It did not eliminate
all order sensitivity: chosen content stayed identical across all four rotations for only 7/16 forget
and 5/16 retain questions. Keep that limitation in the report.

The full-development confirmation preserved the original model and 256 question IDs and changed only
the answer-prefix scoring format. It achieved 78/128 forget and 69/128 retain correct answers, compared
with 36/128 and 42/128 originally. A counts fell from 111 to 24 on forget and from 97 to 37 on retain.
Both pre-agreed ability checks passed. Adopt the explicit-prefix configuration for subsequent stages;
no larger-model fallback is needed. This is a development-selected format, not evidence of prompt
invariance or an unlearning effect. The planned final-test robustness check remains necessary.

The full detailed records are now backed up locally; the earlier need to recover missing downloads
is resolved. The latest runtime estimate is 1.91 GPU hours for the full profile, subject to Stage 3 validation.
