"""Offline Stage 6 reports. Keep matched baselines and uncertainty explicit."""

import importlib.metadata
import json
from pathlib import Path

from .data import write_json
from .localization_report import csv_rows
from .robustness import planned_jobs, read_control_records
from .robustness_analysis import analyze_controls


def make_figures(output, context, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    prefix = "SYNTHETIC TEST DATA | " if context["is_test_fixture"] else ""
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            relative = "figures/" + name + "." + extension
            fig.savefig(Path(output) / relative, dpi=150, bbox_inches="tight")
            files.append(relative)
        plt.close(fig)
    def label(name):
        return name.replace("random_", "Random ").replace("_", " ").capitalize()
    def plot_values(axis, rows, x, color, name):
        values = np.array([100 * r["accuracy_drop"] for r in rows])
        # Percentile intervals may not surround the estimate, so draw endpoints directly.
        axis.bar(x, values, width=.35, color=color, label=name)
        for position, r in zip(x, rows):
            low, high = [100 * v for v in r["drop_ci95"]]
            axis.plot([position, position], [low, high], color="#334155", linewidth=1)
            axis.plot([position-.04, position+.04], [low, low], color="#334155", linewidth=1)
            axis.plot([position-.04, position+.04], [high, high], color="#334155", linewidth=1)
    biology = [r for r in analysis["conditions"] if r["group"] == "biology"]
    fig, axis = plt.subplots(figsize=(11, 5))
    plot_values(axis, biology, np.arange(len(biology)), "#0f766e", "Drop from matched biology baseline")
    axis.set_xticks(np.arange(len(biology)))
    axis.set_xticklabels([label(r["method"]) for r in biology], rotation=30, ha="right")
    base = next(r for r in analysis["baseline_diagnostics"] if r["group"] == "biology")
    axis.set(title=prefix + "General biology | baseline {:.1f}%, n={}\nFixed strength {}; paired 95% intervals".format(
        100 * base["accuracy"], base["count"], context["decision"]["alpha"]), ylabel="Accuracy drop (percentage points)")
    axis.axhline(0, color="#64748b", linewidth=.8)
    axis.grid(axis="y", alpha=.2)
    save(fig, "biology_control")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ("top_forget", "top_selective", "bottom_forget", "random_mean")
    for axis, role in zip(axes, ("forget", "retain")):
        for group, offset, color, legend in (("primary", -.19, "#64748b", "Original instruction"),
                                            ("alternative", .19, "#0f766e", "Alternative instruction")):
            rows = []
            for name in names:
                if name == "random_mean":
                    r = next(r for r in analysis["random_summary"] if r["group"] == group and r["role"] == role)
                    rows.append({"accuracy_drop": r["mean_drop"], "drop_ci95": r["drop_ci95_across_questions"]})
                else:
                    rows.append(next(r for r in analysis["conditions"] if r["group"] == group and r["role"] == role and r["method"] == name))
            plot_values(axis, rows, np.arange(len(names)) + offset, color, legend)
        axis.set_xticks(np.arange(len(names)))
        axis.set_xticklabels([label(n) for n in names], rotation=20, ha="right")
        axis.set(title=role.capitalize(), ylabel="Accuracy drop from same-wording baseline (pp)")
        axis.axhline(0, color="#64748b", linewidth=.8)
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle(prefix + "Prompt control | same questions, fixed layers and strength\nPaired 95% intervals; random mean is across the fixed pairs")
    save(fig, "prompt_control")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    groups = [("biology", "biology_control"), ("primary", "forget"), ("primary", "retain"),
              ("alternative", "forget"), ("alternative", "retain")]
    colors = ("#0f766e", "#60a5fa", "#f59e0b", "#a78bfa")
    for axis, (group, role) in zip(axes.flat, groups):
        rows = [r for r in analysis["answer_letters"] if r["group"] == group and r["role"] == role]
        bottom = np.zeros(len(rows))
        for letter, color in zip("ABCD", colors):
            values = np.array([100*r["predicted_letters"][letter]/r["count"] for r in rows])
            axis.bar(np.arange(len(rows)), values, bottom=bottom, color=color, label=letter)
            bottom += values
        axis.set_xticks(np.arange(len(rows)))
        axis.set_xticklabels([label(r["method"]) for r in rows], rotation=55, ha="right", fontsize=8)
        axis.set(title=group.capitalize() + " | " + role.replace("_", " "), ylabel="Predicted answer letters (%)", ylim=(0, 100))
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, title="Answer letter", loc="center", fontsize=12)
    fig.suptitle(prefix + "Answer-letter distributions | descriptive, not proof of a mechanism")
    save(fig, "answer_letters")
    return {"status": "complete", "files": files, "matplotlib_version": importlib.metadata.version("matplotlib")}


def robustness_report(output, make_plots=True):
    output = Path(output)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    c = run["context"]
    records = read_control_records(output, run)
    imported = len(c["primary_records"])
    expected = len(planned_jobs(c))
    completed = len(records) - imported
    segments = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "segments").glob("*.json"))]
    if any(s["run_fingerprint"] != run["fingerprint"] for s in segments):
        raise ValueError("Control segment fingerprint mismatch.")
    last = segments[-1] if segments else {}
    clean = last.get("cleanup_passed") is True and last.get("status") == "complete"
    failed = bool(last) and (not last.get("cleanup_passed") or last.get("status") == "failed")
    status = "complete" if completed == expected and clean else "review_required" if failed or completed == expected else "partial"
    summary = {"stage": 6, "status": status, "run_fingerprint": run["fingerprint"],
        "new_predictions_saved": completed, "expected_new_predictions": expected, "imported_primary_predictions": imported,
        "is_research_result": not c["is_test_fixture"], "decision": c["decision"], "last_attempt": last,
        "next_action": "Review the controls, then prepare the final report. Keep Stage 5 findings and frozen settings.",
        "counts": {group: {"saved": sum(r["group"] == group for r in records),
                            "expected": sum(j["group"] == group for j in planned_jobs(c))} for group in ("biology", "alternative")},
        "note": "Completion describes execution, not successful selective forgetting. Primary references use the same question subset as the alternative prompt."}
    write_json(output / "summary.json", summary)
    if status != "complete":
        return summary
    analysis = analyze_controls(c, records)
    analysis.update(run_fingerprint=run["fingerprint"], is_research_result=not c["is_test_fixture"],
                    decision=c["decision"], numpy_version=importlib.metadata.version("numpy"))
    write_json(output / "analysis.json", analysis)
    for name, key in (("accuracy", "conditions"), ("random_pair_variation", "random_summary"),
                      ("matched_prompt_comparison", "matched_prompt_comparison"), ("answer_letters", "answer_letters"),
                      ("baseline_diagnostics", "baseline_diagnostics"), ("paired_contrasts", "paired_contrasts")):
        csv_rows(output / (name + ".csv"), analysis[key])
    summary.update(baseline_diagnostics=analysis["baseline_diagnostics"],
        selective_results=[r for r in analysis["conditions"] if r["method"] == "top_selective"],
        selective_prompt_comparison=[r for r in analysis["matched_prompt_comparison"] if r["method"] == "top_selective"],
        limitations=analysis["limitations"], figures={"status": "deferred", "files": []})
    write_json(output / "summary.json", summary)
    if make_plots:
        try:
            summary["figures"] = make_figures(output, c, analysis)
        except Exception as exc:
            summary["status"] = "review_required"
            summary["figures"] = {"status": "failed", "exception_type": type(exc).__name__,
                                  "note": "Predictions and tables are preserved; retry stage6-report to rebuild figures."}
    write_json(output / "summary.json", summary)
    return summary
