"""Development-only operating-point choice and paired accuracy uncertainty."""

import statistics

from .data import digest


def conditions(context):
    return [{"method": "baseline", "alpha": 0.0, "layers": []}] + [
        {"method": s["name"], "alpha": alpha, "layers": s["layers"]}
        for s in context["selections"] for alpha in context["config"]["intervention"]["strengths"] if alpha != 0]


def condition_key(method, alpha):
    return method + ":" + str(float(alpha))


def choose_operating_point(context, records):
    """Use integer correct-answer counts to avoid rounding near the 5pp limit."""
    if context["split"] != "development" or any(r["split"] != "development" for r in records):
        raise ValueError("Operating strength may only be selected from development data.")
    expected = {(e["id"], c["method"], c["alpha"]) for e in context["examples"] for c in conditions(context)}
    actual = {(r["id"], r["method"], r["alpha"]) for r in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("The full development sweep must finish before strength selection.")
    counts = {role: sum(e["role"] == role for e in context["examples"]) for role in ("forget", "retain")}
    def correct(role, method, alpha):
        return sum(r["correct"] for r in records if r["role"] == role and r["method"] == method and r["alpha"] == alpha)
    base = {role: correct(role, "baseline", 0) for role in counts}
    rows = []
    for alpha in context["config"]["intervention"]["strengths"]:
        if alpha == 0:
            continue
        drop = {role: base[role] - correct(role, "top_selective", alpha) for role in counts}
        within = 100 * drop["retain"] <= context["protocol"]["retain_drop_limit_pp"] * counts["retain"]
        rows.append({"alpha": alpha, "forget_correct_drop": drop["forget"], "retain_correct_drop": drop["retain"],
                     "forget_drop_pp": 100 * drop["forget"] / counts["forget"],
                     "retain_drop_pp": 100 * drop["retain"] / counts["retain"],
                     "retain_limit_met": within, "beneficial_forget_drop": drop["forget"] > 0,
                     "eligible": within and drop["forget"] > 0})
    eligible = [r for r in rows if r["eligible"]]
    chosen = sorted(eligible, key=lambda r: (-r["forget_correct_drop"], r["alpha"]))[0]["alpha"] if eligible else context["protocol"]["fallback_display_strength"]
    return {"status": "selected" if eligible else "no_suitable_operating_point", "alpha": chosen,
            "display_only_fallback": not bool(eligible), "source_split": "development",
            "method_used_for_selection": "top_selective", "retain_drop_limit_pp": context["protocol"]["retain_drop_limit_pp"],
            "tie_rule": "weaker_strength", "counts": counts, "candidates": rows,
            "note": "Common strength for all methods. No suitable point means alpha=0.5 is for display only; it does not satisfy the success criterion."}


def paired_bootstrap(values, subjects, repetitions, seed):
    """Rows=questions, columns=conditions. Each bootstrap uses identical IDs across columns."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) != len(subjects) or not len(values):
        raise ValueError("Expected aligned question-by-condition values and subjects.")
    generator = np.random.RandomState(seed)
    groups = [np.array([i for i, subject in enumerate(subjects) if subject == name]) for name in sorted(set(subjects))]
    output = np.empty((repetitions, values.shape[1]), dtype=float)
    for start in range(0, repetitions, 128):
        size = min(128, repetitions - start)
        sampled = np.concatenate([generator.choice(group, (size, len(group)), replace=True) for group in groups], axis=1)
        output[start:start + size] = values[sampled].mean(axis=1)
    return output


def summarize_sweep(context, records, alpha):
    import numpy as np
    specs = conditions(context)
    rows, random_rows, contrasts = [], [], []
    by_key = {(r["id"], r["method"], r["alpha"]): r for r in records}
    expected = len(context["examples"]) * len(specs)
    if len(records) != expected or len(by_key) != expected:
        raise ValueError("Accuracy summaries require every planned question and condition.")
    random_names = [s["name"] for s in context["selections"] if s["name"].startswith("random_")]
    for role in ("forget", "retain"):
        examples = sorted((e for e in context["examples"] if e["role"] == role), key=lambda e: e["id"])
        matrix = np.array([[float(by_key[(e["id"], c["method"], c["alpha"])]["correct"]) for c in specs] for e in examples])
        sampled = paired_bootstrap(matrix, [e["subject"] for e in examples], context["protocol"]["bootstrap_repetitions"],
                                   context["protocol"]["bootstrap_seed"] + (0 if role == "forget" else 1))
        actual = matrix.mean(axis=0)
        def interval(vector):
            return np.percentile(vector, [2.5, 97.5]).tolist()
        for i, spec in enumerate(specs):
            predictions = [by_key[(e["id"], spec["method"], spec["alpha"])] for e in examples]
            rows.append({"role": role, "method": spec["method"], "alpha": spec["alpha"], "layers": spec["layers"],
                "count": len(examples), "correct": int(matrix[:, i].sum()), "accuracy": float(actual[i]),
                "accuracy_ci95": interval(sampled[:, i]), "baseline_accuracy": float(actual[0]),
                "accuracy_drop": float(actual[0] - actual[i]), "drop_ci95": interval(sampled[:, 0] - sampled[:, i]),
                "mean_correct_answer_log_probability": statistics.mean(r["correct_answer_log_probability"] for r in predictions),
                "predicted_letters": {letter: sum(r["prediction"] == letter for r in predictions) for letter in "ABCD"}})
        for strength in context["config"]["intervention"]["strengths"]:
            indices = [0] * len(random_names) if strength == 0 else [i for i, s in enumerate(specs) if s["method"] in random_names and s["alpha"] == strength]
            seed_accuracies = [float(actual[i]) for i in indices]
            average_boot = sampled[:, indices].mean(axis=1)
            random_rows.append({"role": role, "alpha": strength, "pair_count": len(indices),
                "mean_accuracy": statistics.mean(seed_accuracies), "accuracy_ci95_across_questions": interval(average_boot),
                "mean_drop": float(actual[0] - statistics.mean(seed_accuracies)),
                "drop_ci95_across_questions": interval(sampled[:, 0] - average_boot),
                "accuracy_min_across_pairs": min(seed_accuracies), "accuracy_max_across_pairs": max(seed_accuracies),
                "accuracy_sd_across_pairs": statistics.stdev(seed_accuracies) if len(indices) > 1 else None})
        random_indices = [i for i, s in enumerate(specs) if s["method"] in random_names and s["alpha"] == alpha]
        random_mean = sampled[:, random_indices].mean(axis=1)
        fixed_indices = {s["method"]: i for i, s in enumerate(specs) if s["alpha"] == alpha}
        for name in ("top_forget", "top_selective", "bottom_forget"):
            index = fixed_indices[name]
            # D_method - D_random = A_random - A_method. Positive: more damage for this role.
            difference = float(actual[random_indices].mean() - actual[index])
            contrasts.append({"role": role, "method": name, "reference": "mean_of_fixed_random_pairs", "alpha": alpha,
                "extra_accuracy_drop": difference, "paired_ci95": interval(random_mean - sampled[:, index])})
        selective, forget = fixed_indices["top_selective"], fixed_indices["top_forget"]
        contrasts.append({"role": role, "method": "top_selective", "reference": "top_forget", "alpha": alpha,
            "extra_accuracy_drop": float(actual[forget] - actual[selective]),
            "paired_ci95": interval(sampled[:, forget] - sampled[:, selective])})
    return {"conditions": rows, "random_summary": random_rows, "common_strength_contrasts": contrasts,
        "uncertainty": {"method": "95% percentile paired bootstrap, stratified within subject",
            "repetitions": context["protocol"]["bootstrap_repetitions"], "seed": context["protocol"]["bootstrap_seed"],
            "note": "Same resampled question IDs across conditions. Intervals are conditional on the fixed model, selected layers, strength, subjects, and five random pairs. Random-pair range/SD is separate from question uncertainty. No multiplicity correction or causal knowledge-erasure claim."}}
