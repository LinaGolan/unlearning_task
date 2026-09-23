"""Offline verification, immutable development decision, and Stage 5 reports."""

import importlib.metadata
import json
from pathlib import Path

from .data import digest, write_json
from .localization_report import csv_rows
from .sweep_analysis import conditions, summarize_sweep


def comparison_rows(analysis, alpha):
    rows = []
    names = [r["method"] for r in analysis["conditions"] if r["role"] == "forget" and (r["alpha"] == alpha or r["method"] == "baseline")]
    for name in names + ["random_mean"]:
        row = {"method": name, "alpha": 0 if name == "baseline" else alpha}
        for role in ("forget", "retain"):
            if name == "random_mean":
                item = next(r for r in analysis["random_summary"] if r["role"] == role and r["alpha"] == alpha)
                accuracy, drop, ci = item["mean_accuracy"], item["mean_drop"], item["drop_ci95_across_questions"]
            else:
                item = next(r for r in analysis["conditions"] if r["role"] == role and r["method"] == name and r["alpha"] == row["alpha"])
                accuracy, drop, ci = item["accuracy"], item["accuracy_drop"], item["drop_ci95"]
            row.update({role + "_accuracy_percent": 100 * accuracy, role + "_drop_pp": 100 * drop,
                        role + "_drop_ci95_low_pp": 100 * ci[0], role + "_drop_ci95_high_pp": 100 * ci[1]})
        rows.append(row)
    return rows


def make_figures(output, context, analysis, decision):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = Path(output) / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    created = []
    title = ("SYNTHETIC TEST DATA | " if context["is_test_fixture"] else "") + context["split"].capitalize()
    methods = [("top_forget", "Top forget", "#c2410c"), ("top_selective", "Top selective", "#0f766e"),
               ("bottom_forget", "Bottom forget", "#7c3aed")]
    def series(role, method):
        return sorted([r for r in analysis["conditions"] if r["role"] == role and r["method"] in (method, "baseline")], key=lambda r: r["alpha"])
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            relative = "figures/" + name + "." + extension
            fig.savefig(Path(output) / relative, dpi=150, bbox_inches="tight")
            created.append(relative)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, role in zip(axes, ("forget", "retain")):
        for method, label, color in methods:
            rows = series(role, method)
            axis.plot([r["alpha"] for r in rows], [100*r["accuracy"] for r in rows], "o-", color=color, label=label)
        random = sorted([r for r in analysis["random_summary"] if r["role"] == role], key=lambda r: r["alpha"])
        x = [r["alpha"] for r in random]
        axis.plot(x, [100*r["mean_accuracy"] for r in random], "o--", color="#475569", label="Random-pair mean")
        axis.fill_between(x, [100*r["accuracy_min_across_pairs"] for r in random],
                          [100*r["accuracy_max_across_pairs"] for r in random], color="#94a3b8", alpha=.25, label="Range across random pairs")
        axis.axvline(decision["alpha"], linestyle=":", color="#64748b")
        axis.set(title=role.capitalize(), xlabel="Intervention strength", ylabel="Accuracy (%)", ylim=(-2, 102), xticks=x)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    qualifier = "display-only fallback" if decision["display_only_fallback"] else "development-selected strength"
    fig.suptitle(title + " | accuracy curves; dotted line = " + qualifier)
    save(fig, "accuracy_curves")
    fig, axis = plt.subplots(figsize=(8, 6))
    for method, label, color in methods:
        forget, retain = series("forget", method), series("retain", method)
        axis.plot([100*r["accuracy_drop"] for r in retain], [100*r["accuracy_drop"] for r in forget], "o-", color=color, label=label)
        for f, r in zip(forget, retain):
            axis.annotate(str(f["alpha"]), (100*r["accuracy_drop"], 100*f["accuracy_drop"]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=color)
            if f["alpha"] == decision["alpha"]:
                axis.scatter([100*r["accuracy_drop"]], [100*f["accuracy_drop"]], s=150, facecolors="none", edgecolors=color, linewidths=2)
    random = {(r["role"], r["alpha"]): r for r in analysis["random_summary"]}
    strengths = context["config"]["intervention"]["strengths"]
    axis.plot([100*random[("retain", a)]["mean_drop"] for a in strengths],
              [100*random[("forget", a)]["mean_drop"] for a in strengths], "o--", color="#475569", label="Random-pair mean")
    axis.axvline(5, linestyle=":", color="#64748b", label="5 pp development selection limit")
    axis.axhline(0, color="#cbd5e1", linewidth=.8)
    axis.axvline(0, color="#cbd5e1", linewidth=.8)
    axis.set(xlabel="Retain accuracy drop (percentage points)", ylabel="Forget accuracy drop (percentage points)",
             title=title + " | trade-off\nRings mark the " + qualifier)
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    save(fig, "accuracy_tradeoff")
    return {"status": "complete", "files": created}


def sweep_report(output, make_plots=True):
    from .sweep import read_sweep_records, decision_payload
    output = Path(output)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    context = run["context"]
    records = read_sweep_records(output, run)
    segments = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "segments").glob("*.json"))]
    if any(s["run_fingerprint"] != run["fingerprint"] for s in segments):
        raise ValueError("Sweep segment identity mismatch.")
    last = segments[-1] if segments else {}
    expected = len(context["examples"]) * len(conditions(context))
    complete = len(records) == expected
    clean = last.get("status") == "complete" and last.get("cleanup_passed") is True
    failed = bool(last) and not clean
    summary = {"stage": 5, "split": context["split"], "run_fingerprint": run["fingerprint"],
        "status": "complete" if complete and clean else "review_required" if failed or complete else "partial",
        "completed_records": len(records), "expected_records": expected,
        "imported_baseline_records": len(context["imported_baselines"]),
        "new_predictions_saved": len(records) - len(context["imported_baselines"]),
        "is_research_result": not context["is_test_fixture"], "last_attempt": last,
        "final_test_evaluated": context["split"] == "test" and bool(records),
        "selection_fingerprint": context["selection_fingerprint"],
        "next_action": "Complete and review development before test." if context["split"] == "development" else "Review final-test evidence before Stage 6 robustness checks.",
        "interpretation": "Completion describes execution, not scientific success. Accuracy loss alone does not demonstrate knowledge erasure."}
    if not complete or not clean:
        write_json(output / "summary.json", summary)
        return summary
    if context["split"] == "development":
        decision = decision_payload(run, records)
        path = output / "operating_point.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != decision:
            raise ValueError("Refusing to overwrite a frozen development decision.")
        if not path.exists():
            write_json(path, decision)
    else:
        decision = context["frozen_decision"]
        content = {k: v for k, v in decision.items() if k != "decision_fingerprint"}
        if digest(content) != decision["decision_fingerprint"] or decision["source_split"] != "development":
            raise ValueError("Invalid frozen development decision in test context.")
    analysis = summarize_sweep(context, records, decision["alpha"])
    analysis.update(run_fingerprint=run["fingerprint"], split=context["split"], operating_point=decision,
                    is_research_result=not context["is_test_fixture"], numpy_version=importlib.metadata.version("numpy"))
    write_json(output / "analysis.json", analysis)
    csv_rows(output / "accuracy.csv", analysis["conditions"])
    csv_rows(output / "random_pair_variation.csv", analysis["random_summary"])
    csv_rows(output / "paired_contrasts.csv", analysis["common_strength_contrasts"])
    comparison = comparison_rows(analysis, decision["alpha"])
    csv_rows(output / "main_comparison.csv", comparison)
    summary.update(operating_point=decision, main_comparison=comparison, uncertainty=analysis["uncertainty"],
                   figures={"status": "deferred", "files": []})
    # Save numerical evidence before plotting; a plotting error must not lose the decision or data.
    write_json(output / "summary.json", summary)
    if make_plots:
        try:
            summary["figures"] = make_figures(output, context, analysis, decision)
            summary["figures"]["matplotlib_version"] = importlib.metadata.version("matplotlib")
        except Exception as exc:
            summary["status"] = "review_required"
            summary["figures"] = {"status": "failed", "exception_type": type(exc).__name__,
                                  "note": "Raw predictions and tables are saved. Retry stage5-report to rebuild figures."}
    write_json(output / "summary.json", summary)
    return summary
