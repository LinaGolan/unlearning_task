"""Accuracy uncertainty and explicit, provisional compute-budget estimates."""

import math
import statistics

from .evaluation import LABELS


def wilson_interval(correct, total):
    """Two-sided 95% Wilson score interval for a binomial accuracy."""
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("Invalid accuracy counts.")
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def accuracy_summary(records):
    if not records:
        raise ValueError("Cannot report accuracy with no predictions.")
    correct = sum(row["correct"] for row in records)
    return {"count": len(records), "correct": correct, "accuracy": correct / len(records),
            "accuracy_interval_95": wilson_interval(correct, len(records)),
            "interval_method": "Wilson (approximate; ignores subject clustering)",
            "mean_correct_answer_log_probability": statistics.mean(row["correct_answer_log_probability"] for row in records),
            "mean_answer_probability_mass": statistics.mean(row["answer_probability_mass"] for row in records),
            "predicted_letters": {label: sum(row["prediction"] == label for row in records) for label in LABELS},
            "correct_letters": {label: sum(row["correct_answer"] == label for row in records) for label in LABELS}}


def ability_gate(by_role, settings):
    checks = {}
    for role in ("forget", "retain"):
        row = by_role[role]
        checks[role] = {"accuracy_passed": row["accuracy"] >= settings[role + "_accuracy_min"],
                        "lower_interval_above_chance": row["accuracy_interval_95"][0] > settings["chance_accuracy"]}
    passed = all(all(row.values()) for row in checks.values())
    return {"status": "passed" if passed else "review_required", "checks": checks,
            "next_action": "Proceed to Stage 3 correctness checks." if passed else
            "Review evaluator and development results first; only then consider the planned 3B fallback once. No automatic model switch."}


def runtime_forecast(config, records, gradient_probe, settings, layer_count):
    forwards = statistics.mean(row["seconds_per_example"] for row in records)
    gradients = statistics.mean(row["forward_backward_seconds"] for row in gradient_probe["measurements"])
    # 3 fixed selections + 5 random pairs. Alpha=0 is shared with the baseline.
    selections = 3 + len(config["intervention"]["random_seeds"])
    nonzero = sum(x != 0 for x in config["intervention"]["strengths"])
    profiles = {}
    for name, sizes in config["profiles"].items():
        counts = {
            "main_baselines_and_sweeps": (1 + selections * nonzero) * 2 * (sizes["development"] + sizes["test"]),
            "control_a_forwards": 32 * layer_count * 2,
            "biology_control": config["biology_control_size"] * (1 + selections),
            "alternate_prompt_control": 2 * min(128, sizes["test"]) * (1 + selections),
        }
        gradient_count = 2 * sizes["localization"] + 32
        hours = (sum(counts.values()) * forwards + gradient_count * gradients) * settings["forecast_overhead_factor"] / 3600
        profiles[name] = {"estimated_gpu_hours": hours, "forward_counts": counts,
                          "forward_backward_count": gradient_count,
                          "within_budget": hours <= settings["gpu_hour_budget"]}
    recommendation = "full" if profiles["full"]["within_budget"] else (
        "reduced" if profiles["reduced"]["within_budget"] else "review_compute_budget")
    return {"status": "provisional", "recommended_profile": recommendation, "profiles": profiles,
            "mean_forward_seconds": forwards, "mean_forward_backward_proxy_seconds": gradients,
            "overhead_factor": settings["forecast_overhead_factor"],
            "layer_count": layer_count,
            "note": "Includes the planned evaluation conditions, not downloads or queueing. Assumes development lengths represent later data. Recheck actual layer-gate timing in Stage 3 before freezing the profile."}
