"""Offline Stage 4 verification, immutable selections, tables, and static figures."""

import csv
import importlib.metadata
import json
import math
from pathlib import Path
import statistics

from .data import digest, write_json
from .localization_analysis import localization_summary, control_summary


def csv_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_selections(output, run, records, result):
    selection = {"schema_version": 1, "run_fingerprint": run["fingerprint"], "profile": run["context"]["profile"],
        "model": run["context"]["config"]["model"], "settings": run["context"]["settings"],
        "localization_record_hashes": [r["record_hash"] for r in records],
        "source_split": "localization", "k": run["context"]["config"]["intervention"]["k"],
        "selections": result["selections"], "tie_rule": run["context"]["protocol"]["tie_rule"],
        "note": "Fixed from complete localization data only. Control A, development accuracy, and final-test outcomes do not select layers. No operating strength has been chosen."}
    selection["selection_fingerprint"] = digest(selection)
    path = Path(output) / "selections.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != selection:
        raise ValueError("Saved layer selections differ. Do not overwrite a frozen selection.")
    if not path.exists():
        write_json(path, selection)
    return selection


def make_figures(output, result, control, intervention_records, test_fixture):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    plots = output / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    prefix = "SYNTHETIC TEST DATA - " if test_fixture else ""
    created = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(plots / (name + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(plots / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
        created.extend(["figures/" + name + ".png", "figures/" + name + ".pdf"])
    if result is not None:
        layers = [r["layer"] for r in result["layer_scores"]]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for name, color in (("forget", "#c2410c"), ("retain", "#1d4ed8")):
            axes[0].plot(layers, [r[name + "_score"] for r in result["layer_scores"]], marker="o", label=name.capitalize(), color=color)
        axes[0].set_ylabel("Mean derivative dM/dg")
        axes[0].legend()
        axes[0].set_title(prefix + "Layer sensitivity on localization questions")
        axes[1].bar(layers, [r["selectivity_score"] for r in result["layer_scores"]], color="#0f766e")
        axes[1].set_ylabel("Forget minus retain")
        axes[1].set_xlabel("Decoder layer (zero-based)")
        axes[1].set_xticks(layers)
        for axis in axes:
            axis.axhline(0, color="#475569", linewidth=.8)
            axis.grid(axis="y", alpha=.2)
        save(fig, "layer_scores")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(prefix + "Ranking stability across balanced halves")
        for axis, name in zip(axes, ("forget", "retain", "selectivity")):
            stability = result["stability"][name]
            x, y = stability["first_half_scores"], stability["second_half_scores"]
            axis.scatter(x, y, color="#0f766e")
            limits = [min(x + y + [0]), max(x + y + [0])]
            if limits[0] == limits[1]:
                limits = [-1, 1]
            axis.plot(limits, limits, "--", color="#64748b", linewidth=1)
            rho = stability["spearman"]
            axis.set_title(name.capitalize() + " | Spearman " + ("undefined" if rho is None else "{:.2f}".format(rho)))
            axis.set_xlabel("First-half mean score")
            axis.set_ylabel("Second-half mean score")
        save(fig, "ranking_stability")
    if control is not None:
        strengths = sorted({r["alpha"] for r in intervention_records})
        fig, axes = plt.subplots(2, len(strengths), figsize=(11, 9), squeeze=False)
        fig.suptitle(prefix + "Control A: predicted versus measured score drops")
        for i, role in enumerate(("forget", "retain")):
            for j, alpha in enumerate(strengths):
                axis = axes[i][j]
                rows = [r for r in intervention_records if r["role"] == role and r["alpha"] == alpha]
                means = [r for r in control["layer_means"] if r["role"] == role and r["alpha"] == alpha]
                x, y = [r["predicted_drop"] for r in rows], [r["actual_drop"] for r in rows]
                axis.scatter(x, y, alpha=.3, s=18, color="#64748b", label="Question/layer pairs")
                axis.scatter([r["predicted_mean_drop"] for r in means], [r["actual_mean_drop"] for r in means],
                             marker="D", s=32, color="#c2410c", label="Layer means")
                limits = [min(x + y + [0]), max(x + y + [0])]
                if limits[0] == limits[1]:
                    limits = [-1, 1]
                axis.plot(limits, limits, "--", color="#0f766e", linewidth=1, label="Exact agreement")
                axis.set_title("{} | strength {}".format(role.capitalize(), alpha))
                axis.set_xlabel("Predicted drop: alpha * dM/dg")
                axis.set_ylabel("Measured drop: M(baseline) - M(changed)")
                axis.axhline(0, color="#cbd5e1", linewidth=.6)
                axis.axvline(0, color="#cbd5e1", linewidth=.6)
                axis.legend(fontsize=8)
        save(fig, "control_a")
    return {"status": "complete", "files": created,
            "packages": {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")},
            "note": "Score drops are in natural-log-probability units, not accuracy percentage points. Figures contain no confidence intervals."}


def stage4_report(output, make_plots=True):
    from .localization import planned_jobs, read_records
    output = Path(output)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    context = run["context"]
    records = read_records(output, run)
    jobs = planned_jobs(context)
    kinds = ("localization", "control_gradient", "control_intervention")
    counts = {kind: {"saved": sum(r["kind"] == kind for r in records), "expected": sum(j["kind"] == kind for j in jobs)} for kind in kinds}
    complete = len(records) == len(jobs)
    report = {"stage": 4, "status": "complete" if complete else "partial", "run_fingerprint": run["fingerprint"],
        "profile": context["profile"], "is_research_result": not context["is_test_fixture"],
        "final_test_evaluated": False, "operating_strength_selected": False,
        "completed_records": len(records), "expected_records": len(jobs), "counts": counts,
        "layer_selections_fixed": False, "control_a_complete": False}
    segments = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "segments").glob("*.json"))]
    if any(s["run_fingerprint"] != run["fingerprint"] for s in segments):
        raise ValueError("A Stage 4 timing segment belongs to another run.")
    if segments:
        report["last_attempt"] = segments[-1]
        report["total_attempt_seconds"] = sum(s["elapsed_seconds"] for s in segments)
        if segments[-1]["status"] == "failed":
            report["status"] = "review_required"
    gradients = [r for r in records if r["kind"] != "control_intervention"]
    if gradients:
        peaks = [r["peak_gpu_allocated_bytes"] for r in gradients if r["peak_gpu_allocated_bytes"] is not None]
        report["measured_gradients"] = {"count": len(gradients), "mean_seconds": statistics.mean(r["forward_backward_seconds"] for r in gradients),
                                        "peak_gpu_allocated_bytes": max(peaks) if peaks else None}
    result, control = None, None
    localization = [r for r in records if r["kind"] == "localization"]
    effects = [r for r in records if r["kind"] == "control_intervention"]
    if counts["localization"]["saved"] == counts["localization"]["expected"]:
        result = localization_summary(localization, context["localization_examples"], context["config"], context["protocol"], context["layer_count"])
        frozen = write_selections(output, run, localization, result)
        report.update(layer_selections_fixed=True, selection_fingerprint=frozen["selection_fingerprint"],
                      selections=frozen["selections"], stability=result["stability"])
        write_json(output / "localization_summary.json", result)
        csv_rows(output / "layer_scores.csv", result["layer_scores"])
    if complete:
        control = control_summary(effects, context["layer_count"], context["protocol"])
        write_json(output / "control_a_summary.json", control)
        csv_rows(output / "control_a_layer_means.csv", control["layer_means"])
        report["control_a_complete"] = True
        report["control_a"] = [{k: row[k] for k in ("role", "alpha", "pooled_pairs", "layer_means", "mean_question_spearman")} for row in control["comparisons"]]
    if effects:
        csv_rows(output / "control_a_pairs.csv", [{k: row[k] for k in ("id", "role", "subject", "layer", "alpha", "baseline_score", "correct_answer_log_probability", "predicted_drop", "actual_drop", "prediction", "correct")} for row in effects])
    if gradients:
        csv_rows(output / "question_layer_scores.csv", [{"id": r["id"], "role": r["role"], "split": r["split"], "subject": r["subject"],
            "layer": i, "gradient": g, "baseline_prediction": r["prediction"], "baseline_correct": r["correct"]}
            for r in gradients for i, g in enumerate(r["gradients"])])
    report["figures"] = {"status": "not_requested" if not make_plots else "waiting_for_localization"}
    if make_plots and result is not None:
        try:
            report["figures"] = make_figures(output, result, control, effects, context["is_test_fixture"])
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            report["figures"] = {"status": "review_required", "exception_type": type(exc).__name__,
                                 "note": "Measurements are saved. Fix plotting and rerun stage4-report without repeating GPU work."}
            report["status"] = "review_required"
    report["next_action"] = ("Review rankings, stability, and Control A before Stage 5. No strength has been selected."
                             if complete and report["status"] == "complete" else
                             "Download a backup. Resume the same settings for partial work; inspect any failure before continuing.")
    report["interpretation"] = "Technical completion is not proof of selective forgetting. Low stability or weak Control A agreement must be reported; they do not trigger a search for a better-looking score formula."
    write_json(output / "summary.json", report)
    return report
