"""Deterministic layer summaries and descriptive Control A statistics."""

import itertools
import math
import random
import statistics

from .data import digest


def ranks(values):
    """Ascending average ranks; tied scores share a rank for Spearman."""
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def correlation(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left, mean_right = statistics.mean(left), statistics.mean(right)
    x = [v - mean_left for v in left]
    y = [v - mean_right for v in right]
    denominator = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(x, y)) / denominator)) if denominator else None


def spearman(left, right):
    return correlation(ranks(left), ranks(right))


def ordered_layers(values, descending=True):
    return sorted(range(len(values)), key=lambda i: (-values[i] if descending else values[i], i))


def assign_halves(examples, seed):
    assignments = {}
    for group in sorted({(e["role"], e["subject"]) for e in examples}):
        rows = sorted((e for e in examples if (e["role"], e["subject"]) == group),
                      key=lambda e: (digest([seed, "localization_halves", e["id"]]), e["id"]))
        if len(rows) < 2 or len(rows) % 2:
            raise ValueError("Each subject needs an even number of localization examples for balanced halves.")
        assignments.update({row["id"]: i % 2 for i, row in enumerate(rows)})
    return assignments


def layer_means(records, layer_count):
    values = {}
    for role in ("forget", "retain"):
        rows = [r for r in records if r["role"] == role]
        if not rows:
            raise ValueError("Both roles are required for a layer summary.")
        values[role] = [statistics.mean(r["gradients"][i] for r in rows) for i in range(layer_count)]
    values["selectivity"] = [f - r for f, r in zip(values["forget"], values["retain"])]
    return values


def random_layer_sets(layer_count, k, seeds):
    # Match the planned fixed random seeds; retry deterministic collisions only.
    if math.comb(layer_count, k) < len(seeds):
        raise ValueError("Not enough distinct layer sets for the requested random controls.")
    seen, selections = set(), []
    for seed in seeds:
        generator = random.Random(seed)
        while True:
            layers = tuple(sorted(generator.sample(list(range(layer_count)), k)))
            if layers not in seen:
                break
        seen.add(layers)
        selections.append({"name": "random_{}".format(seed), "seed": seed, "layers": list(layers)})
    return selections


def localization_summary(records, examples, config, protocol, layer_count):
    if len(records) != len(examples) or {r["id"] for r in records} != {e["id"] for e in examples}:
        raise ValueError("Only a complete localization split may define layer selections.")
    if any(r["split"] != "localization" for r in records):
        raise ValueError("Layer selection may only use localization records.")
    k = config["intervention"]["k"]
    if not 0 < k <= layer_count:
        raise ValueError("Invalid number of selected layers.")
    means = layer_means(records, layer_count)
    orders = {name: ordered_layers(values) for name, values in means.items()}
    rows = [{"layer": i, **{name + "_score": means[name][i] for name in means},
             **{name + "_rank": orders[name].index(i) + 1 for name in means}} for i in range(layer_count)]
    selections = [
        {"name": "top_forget", "layers": orders["forget"][:k]},
        {"name": "top_selective", "layers": orders["selectivity"][:k]},
        {"name": "bottom_forget", "layers": ordered_layers(means["forget"], descending=False)[:k]},
    ] + random_layer_sets(layer_count, k, config["intervention"]["random_seeds"])
    halves = assign_halves(examples, config["seed"])
    half_means = [layer_means([r for r in records if halves[r["id"]] == half], layer_count) for half in (0, 1)]
    stability = {}
    for name in means:
        first, second = [part[name] for part in half_means]
        top_first, top_second = ordered_layers(first)[:k], ordered_layers(second)[:k]
        rho = spearman(first, second)
        stability[name] = {"spearman": rho, "top_k_first_half": top_first, "top_k_second_half": top_second,
                           "top_k_overlap": len(set(top_first) & set(top_second)),
                           "review_noise": rho is None or rho < protocol["stability_review_spearman_below"],
                           "first_half_scores": first, "second_half_scores": second}
    return {"layer_scores": rows, "selections": selections, "stability": stability,
            "half_assignments": halves,
            "counts": {role: sum(r["role"] == role for r in records) for role in ("forget", "retain")},
            "ties": {name: len(set(values)) != len(values) for name, values in means.items()},
            "note": "Signed means across all localization questions, including initially incorrect answers. Ties use lower layer index. Stability warnings describe uncertainty and do not change selections."}


def agreement(predicted, actual, zero_tolerance):
    if not predicted or len(predicted) != len(actual):
        raise ValueError("Paired nonempty score drops are required.")
    errors = [a - p for p, a in zip(predicted, actual)]
    nonzero = [(p, a) for p, a in zip(predicted, actual) if abs(p) > zero_tolerance and abs(a) > zero_tolerance]
    return {"count": len(errors), "pearson": correlation(predicted, actual), "spearman": spearman(predicted, actual),
            "mae": statistics.mean(abs(e) for e in errors), "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
            "mean_error_actual_minus_predicted": statistics.mean(errors),
            "sign_agreement": statistics.mean(p * a > 0 for p, a in nonzero) if nonzero else None,
            "sign_comparison_count": len(nonzero), "near_zero_pairs_excluded_from_sign": len(errors) - len(nonzero)}


def control_summary(records, layer_count, protocol):
    summaries, mean_rows = [], []
    for role, alpha in itertools.product(("forget", "retain"), protocol["control_strengths"]):
        group = [r for r in records if r["role"] == role and r["alpha"] == alpha]
        if len(group) != protocol["control_questions_per_role"] * layer_count:
            raise ValueError("Control A requires every planned question/layer/strength pair.")
        predicted, actual = [r["predicted_drop"] for r in group], [r["actual_drop"] for r in group]
        layer_rows = []
        for layer in range(layer_count):
            subset = [r for r in group if r["layer"] == layer]
            if len(subset) != protocol["control_questions_per_role"]:
                raise ValueError("A Control A layer has missing examples.")
            layer_rows.append({"role": role, "alpha": alpha, "layer": layer,
                "predicted_mean_drop": statistics.mean(r["predicted_drop"] for r in subset),
                "actual_mean_drop": statistics.mean(r["actual_drop"] for r in subset)})
        per_question = []
        for identity in sorted({r["id"] for r in group}):
            subset = sorted((r for r in group if r["id"] == identity), key=lambda r: r["layer"])
            rho = spearman([r["predicted_drop"] for r in subset], [r["actual_drop"] for r in subset])
            per_question.append({"id": identity, "spearman": rho})
        valid = [r["spearman"] for r in per_question if r["spearman"] is not None]
        summaries.append({"role": role, "alpha": alpha,
            "pooled_pairs": agreement(predicted, actual, protocol["sign_zero_tolerance"]),
            "layer_means": agreement([r["predicted_mean_drop"] for r in layer_rows],
                                     [r["actual_mean_drop"] for r in layer_rows], protocol["sign_zero_tolerance"]),
            "question_rank_correlations": per_question,
            "mean_question_spearman": statistics.mean(valid) if valid else None,
            "defined_question_correlations": len(valid),
            "note": "Descriptive correlations only: layers from the same question are dependent. Undefined correlations (constant scores) are null, not zero."})
        mean_rows.extend(layer_rows)
    return {"comparisons": summaries, "layer_means": mean_rows,
            "interpretation": "Compare small and large suppression. Weak agreement is a scientific finding; it does not automatically change the localization formula or chosen layers."}
