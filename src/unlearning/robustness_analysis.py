"""Matched-prompt effects, paired uncertainty, and descriptive answer-letter checks."""

import statistics

from .evaluation import LABELS
from .metrics import wilson_interval
from .robustness import control_conditions, planned_jobs
from .sweep_analysis import paired_bootstrap


def analyze_controls(context, records):
    import numpy as np
    specs = control_conditions(context)
    expected = {(j["group"], j["id"], j["method"]) for j in planned_jobs(context)} | {
        ("primary", e["id"], s["method"]) for e in context["primary_examples"] for s in specs}
    by_key = {(r["group"], r["id"], r["method"]): r for r in records}
    if set(by_key) != expected or len(records) != len(expected):
        raise ValueError("Analysis requires complete controls and matched primary references.")
    groups = [("biology", "biology_control"), ("primary", "forget"), ("primary", "retain"),
              ("alternative", "forget"), ("alternative", "retain")]
    random_indices = [i for i, s in enumerate(specs) if s["method"].startswith("random_")]
    protocol = context["protocol"]
    rows, letters, random_rows, contrasts, baselines = [], [], [], [], []
    matrices, grouped_examples = {}, {}
    def interval(v):
        return np.percentile(v, [2.5, 97.5]).tolist()
    for group_index, (group, role) in enumerate(groups):
        source = context["primary_examples"] if group == "primary" else [e for e in context["examples"] if e["group"] == group]
        examples = sorted([e for e in source if e["role"] == role], key=lambda e: e["id"])
        matrix = np.array([[float(by_key[(group, e["id"], s["method"])]["correct"]) for s in specs] for e in examples])
        matrices[(group, role)], grouped_examples[(group, role)] = matrix, examples
        sampled = paired_bootstrap(matrix, [e["subject"] for e in examples], protocol["bootstrap_repetitions"], protocol["bootstrap_seed"] + group_index)
        actual = matrix.mean(axis=0)
        for i, spec in enumerate(specs):
            predictions = [by_key[(group, e["id"], spec["method"])] for e in examples]
            row = {"group": group, "role": role, "method": spec["method"], "alpha": spec["alpha"],
                "count": len(examples), "correct": int(matrix[:, i].sum()), "accuracy": float(actual[i]),
                "accuracy_ci95": interval(sampled[:, i]), "baseline_accuracy": float(actual[0]),
                "accuracy_drop": float(actual[0] - actual[i]), "drop_ci95": interval(sampled[:, 0] - sampled[:, i]),
                "mean_correct_answer_log_probability": statistics.mean(r["correct_answer_log_probability"] for r in predictions)}
            rows.append(row)
            counts = {label: sum(r["prediction"] == label for r in predictions) for label in LABELS}
            correct_counts = {label: sum(r["correct_answer"] == label for r in predictions) for label in LABELS}
            base_predictions = [by_key[(group, e["id"], "baseline")]["prediction"] for e in examples]
            letters.append({"group": group, "role": role, "method": spec["method"], "count": len(examples),
                "predicted_letters": counts, "correct_letters": correct_counts,
                "dominant_letter": max(LABELS, key=lambda label: counts[label]),
                "dominant_letter_fraction": max(counts.values()) / len(examples),
                "prediction_change_from_matched_baseline": sum(r["prediction"] != b for r, b in zip(predictions, base_predictions)) / len(examples),
                "letter_total_variation_from_matched_baseline": .5 * sum(abs(counts[label]/len(examples) - base_predictions.count(label)/len(examples)) for label in LABELS)})
        accuracy_by_pair = actual[random_indices]
        random_boot = sampled[:, random_indices].mean(axis=1)
        random_mean = float(accuracy_by_pair.mean())
        random_rows.append({"group": group, "role": role, "method": "random_mean", "alpha": context["decision"]["alpha"],
            "pair_count": len(random_indices), "mean_accuracy": random_mean,
            "accuracy_ci95_across_questions": interval(random_boot), "mean_drop": float(actual[0] - random_mean),
            "drop_ci95_across_questions": interval(sampled[:, 0] - random_boot),
            "accuracy_min_across_pairs": float(accuracy_by_pair.min()), "accuracy_max_across_pairs": float(accuracy_by_pair.max()),
            "accuracy_sd_across_pairs": float(accuracy_by_pair.std(ddof=1)) if len(random_indices) > 1 else None})
        for method in ("top_forget", "top_selective", "bottom_forget"):
            i = next(i for i, s in enumerate(specs) if s["method"] == method)
            contrasts.append({"group": group, "role": role, "method": method, "reference": "mean_of_fixed_random_pairs",
                "extra_accuracy_drop": float(random_mean - actual[i]), "paired_ci95": interval(random_boot - sampled[:, i])})
        count = int(matrix[:, 0].sum())
        wilson = wilson_interval(count, len(examples))
        baselines.append({"group": group, "role": role, "count": len(examples), "correct": count,
            "accuracy": float(actual[0]), "wilson_ci95": wilson, "lower_bound_above_chance": wilson[0] > .25,
            "note": "Descriptive baseline diagnostic only. A weak control baseline limits inference; do not retune the model or intervention."})
    prompt_rows = []
    for role_index, role in enumerate(("forget", "retain")):
        first, second = grouped_examples[("primary", role)], grouped_examples[("alternative", role)]
        if [e["id"] for e in first] != [e["id"] for e in second]:
            raise ValueError("Prompt comparison requires exactly the same question IDs.")
        primary, alternative = matrices[("primary", role)], matrices[("alternative", role)]
        # Joint columns pair question IDs across BOTH methods and prompt wordings.
        joint = np.concatenate([primary, alternative], axis=1)
        boot = paired_bootstrap(joint, [e["subject"] for e in first], protocol["bootstrap_repetitions"], protocol["bootstrap_seed"] + 5 + role_index)
        width = len(specs)
        before, after = boot[:, :width], boot[:, width:]
        for method, indices in [(s["method"], [i]) for i, s in enumerate(specs)] + [("random_mean", random_indices)]:
            primary_drop = float(primary[:, 0].mean() - primary[:, indices].mean())
            alternate_drop = float(alternative[:, 0].mean() - alternative[:, indices].mean())
            delta = (after[:, 0] - after[:, indices].mean(axis=1)) - (before[:, 0] - before[:, indices].mean(axis=1))
            change = after[:, indices].mean(axis=1) - before[:, indices].mean(axis=1)
            flips = statistics.mean(by_key[("primary", e["id"], specs[i]["method"])]["prediction"] !=
                                    by_key[("alternative", e["id"], specs[i]["method"])]["prediction"] for e in first for i in indices)
            prompt_rows.append({"role": role, "method": method, "count": len(first),
                "primary_accuracy_drop": primary_drop, "alternative_accuracy_drop": alternate_drop,
                "change_in_accuracy_drop": alternate_drop - primary_drop, "change_in_drop_ci95": interval(delta),
                "accuracy_change_from_wording": float(alternative[:, indices].mean() - primary[:, indices].mean()),
                "accuracy_change_ci95": interval(change), "prediction_change_from_wording": flips})
    return {"conditions": rows, "random_summary": random_rows, "paired_contrasts": contrasts,
        "matched_prompt_comparison": prompt_rows, "answer_letters": letters, "baseline_diagnostics": baselines,
        "uncertainty": {"method": "95% percentile paired bootstrap within subject; joint resampling across prompt wordings",
            "repetitions": protocol["bootstrap_repetitions"], "seed": protocol["bootstrap_seed"],
            "note": "Conditional on the fixed model, selected layers, strength, subjects, random pairs, and one alternative instruction. No multiplicity correction. Random-pair variation is separate from question uncertainty."},
        "limitations": ["Controls cannot establish permanent knowledge erasure or where knowledge is stored.",
            "Biology and WMDP differ in difficulty and question style; cross-dataset drop differences alone are not causal specificity evidence.",
            "One alternative instruction tests limited wording sensitivity, not arbitrary prompt robustness.",
            "Answer-letter shifts are descriptive diagnostics, not proof of an answer-position mechanism.",
            "Keep the Stage 5 negative finding and frozen settings; these controls do not select a replacement."]}
