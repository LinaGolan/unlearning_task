"""Turn saved per-question predictions into the tables, intervals and figures.

Runs on CPU from results/ alone; no model is loaded. Uncertainty uses a paired
bootstrap over question IDs, so every condition is resampled on the same draws.
"""

import json
import random
import statistics
from pathlib import Path

from .data import write_json
from .evaluate import accuracy
from .experiment import METHODS, condition_name

ROLES = ("forget", "retain")
LABELS_FOR = {"top_localized": "Top-k localization", "top_selective": "WMDP-vs-Retain",
              "bottom_localized": "Bottom-k", "random_mean": "Random (mean of 5)"}
# Symbols used for the localization scores in the report.
SYMBOLS = {"forget": "F", "selective": "S", "retain": "R"}


def load_predictions(out_dir, name):
    path = Path(out_dir) / "predictions" / (name + ".jsonl")
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def by_id(records, role=None):
    return {r["id"]: r["correct"] for r in records if role is None or r["role"] == role}


def paired_bootstrap(base, variants, repetitions, seed, level):
    """Percentile intervals for each variant's accuracy drop against a shared baseline."""
    ids = sorted(base)
    rng = random.Random(seed)
    draws = [[rng.randrange(len(ids)) for _ in ids] for _ in range(repetitions)]
    out = {}
    for name, variant in variants.items():
        if set(variant) != set(base):
            raise ValueError("Paired bootstrap needs identical question sets: " + name)
        deltas = []
        for draw in draws:
            hit_base = sum(base[ids[i]] for i in draw)
            hit_variant = sum(variant[ids[i]] for i in draw)
            deltas.append(100 * (hit_base - hit_variant) / len(draw))
        deltas.sort()
        low = deltas[int((1 - level) / 2 * repetitions)]
        high = deltas[min(repetitions - 1, int((1 + level) / 2 * repetitions))]
        point = 100 * (sum(base.values()) - sum(variant.values())) / len(ids)
        out[name] = {"drop_pp": point, "interval_95": [low, high]}
    return out


def contrast_bootstrap(base, reference, variant, repetitions, seed, level):
    """How much more accuracy `variant` loses than `reference`, both against `base`.

    Positive means the variant damaged accuracy more than the reference did. Each
    resample draws one set of question IDs and applies it to both conditions, so
    the shared baseline cancels and only the paired difference is resampled.
    """
    ids = sorted(base)
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        draw = [rng.randrange(len(ids)) for _ in ids]
        hit_reference = sum(reference[ids[i]] for i in draw)
        hit_variant = sum(variant[ids[i]] for i in draw)
        values.append(100 * (hit_reference - hit_variant) / len(draw))
    values.sort()
    point = 100 * (sum(reference.values()) - sum(variant.values())) / len(ids)
    return {"extra_drop_pp": point,
            "interval_95": [values[int((1 - level) / 2 * repetitions)],
                            values[min(repetitions - 1, int((1 + level) / 2 * repetitions))]]}


def mean_records(groups):
    """Average correctness across random seeds, keeping one row per question."""
    ids = sorted(groups[0])
    return {i: statistics.mean(g[i] for g in groups) for i in ids}


def condition_rows(out_dir, split, method, selections, alpha, seeds):
    """Accuracy per role for the four task conditions plus the averaged random control.

    At alpha = 0 every condition is the unmodified model, which the driver stores
    once as the split baseline rather than duplicating per condition.
    """
    unmodified = load_predictions(out_dir, split + "__baseline") if alpha == 0 else None
    rows, raw = {}, {}
    for selection in list(LABELS_FOR)[:3]:
        records = load_predictions(out_dir, "{}__{}".format(split, condition_name(method, selection, alpha)))
        if records is None:
            records = unmodified
        if records is not None:
            raw[selection] = records
    randoms = [load_predictions(out_dir, "{}__{}".format(split, condition_name(method, "random_{}".format(s), alpha)))
               or unmodified for s in seeds]
    randoms = [r for r in randoms if r is not None]
    for name, records in raw.items():
        rows[name] = {role: accuracy([r for r in records if r["role"] == role]) for role in ROLES}
    if randoms:
        rows["random_mean"] = {role: {"accuracy": statistics.mean(
            accuracy([r for r in group if r["role"] == role])["accuracy"] for group in randoms),
            "count": len([r for r in randoms[0] if r["role"] == role])} for role in ROLES}
        rows["random_each"] = [{role: accuracy([r for r in group if r["role"] == role])["accuracy"]
                                for role in ROLES} for group in randoms]
    return rows, raw, randoms


def analyse(out_dir, only=None):
    """`only` restricts the analysis to a subset of the methods the run evaluated."""
    out_dir = Path(out_dir)
    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    localization = json.loads((out_dir / "localization.json").read_text(encoding="utf-8"))
    config = json.loads((out_dir / "setup.json").read_text(encoding="utf-8"))["config"]
    boot = config["bootstrap"]
    seeds = config["intervention"]["random_seeds"]
    methods = [m for m in METHODS if m in run["selections"] and (only is None or m in only)]
    if not methods:
        raise ValueError("None of the requested methods were evaluated in " + str(out_dir))

    analysis = {"k": config["intervention"]["k"], "methods": methods,
                "operating_point": {m: run["operating_point"][m]["alpha"] for m in methods},
                "selections": run["selections"], "localization": {}, "main": {}, "sweep": {},
                "controls": {}, "questions": run["questions"]}

    for method in methods:
        scores = localization[method]["scores"]
        analysis["localization"][method] = {
            "scores": scores,
            "standard_errors": localization[method].get("standard_errors", {}),
            "selection": localization[method].get("selection", {})}

    baseline = load_predictions(out_dir, "test__baseline")
    analysis["baseline"] = {role: accuracy([r for r in baseline if r["role"] == role]) for role in ROLES}
    analysis["baseline_development"] = {role: accuracy([r for r in load_predictions(out_dir, "development__baseline")
                                                       if r["role"] == role]) for role in ROLES}

    for method in methods:
        alpha = run["operating_point"][method]["alpha"]
        rows, raw, randoms = condition_rows(out_dir, "test", method, run["selections"][method], alpha, seeds)
        intervals, extra = {}, {}
        for role in ROLES:
            base = by_id(baseline, role)
            variants = {name: by_id(records, role) for name, records in raw.items()}
            if randoms:
                variants["random_mean"] = mean_records([by_id(g, role) for g in randoms])
            intervals[role] = paired_bootstrap(base, variants, boot["repetitions"], boot["seed"], boot["level"])
            if randoms:
                reference = mean_records([by_id(g, role) for g in randoms])
                extra[role] = {name: contrast_bootstrap(base, reference, by_id(records, role),
                                                        boot["repetitions"], boot["seed"], boot["level"])
                               for name, records in raw.items()}
        ranking = {}
        if "top_localized" in raw and "top_selective" in raw:
            for role in ROLES:
                ranking[role] = contrast_bootstrap(
                    by_id(baseline, role), by_id(raw["top_selective"], role),
                    by_id(raw["top_localized"], role),
                    boot["repetitions"], boot["seed"], boot["level"])
        analysis["main"][method] = {"alpha": alpha, "rows": rows, "intervals": intervals,
                                   "extra_over_random": extra, "selective_vs_localized": ranking,
                                   "development_rule": run["operating_point"][method]["rule"]}

        curve = {}
        for strength in run["strengths"]:
            rows_a, _, randoms_a = condition_rows(out_dir, "test", method, run["selections"][method], strength, seeds)
            curve[str(strength)] = {name: {role: rows_a[name][role]["accuracy"] for role in ROLES}
                                    for name in rows_a if name != "random_each"}
        analysis["sweep"][method] = curve

    biology_base = load_predictions(out_dir, "biology__baseline")
    biology = {"baseline": accuracy(biology_base)}
    for method in methods:
        alpha = run["operating_point"][method]["alpha"]
        variants, randoms = {}, []
        for selection in list(LABELS_FOR)[:3]:
            records = load_predictions(out_dir, "biology__" + condition_name(method, selection, alpha))
            if records is not None:
                variants[selection] = by_id(records)
                biology[method + "/" + selection] = accuracy(records)
        for seed in seeds:
            records = load_predictions(out_dir, "biology__" + condition_name(method, "random_{}".format(seed), alpha))
            if records is not None:
                randoms.append(by_id(records))
        if randoms:
            variants["random_mean"] = mean_records(randoms)
            biology[method + "/random_mean"] = {"accuracy": statistics.mean(
                statistics.mean(r.values()) for r in randoms), "count": len(biology_base)}
        biology[method + "/intervals"] = paired_bootstrap(by_id(biology_base), variants,
                                                          boot["repetitions"], boot["seed"], boot["level"])
    analysis["controls"]["biology"] = biology

    prompts = {}
    for style in [config["prompt"]] + config["alternative_prompts"]:
        base = load_predictions(out_dir, "prompt_{}__baseline".format(style))
        if base is None:
            continue
        entry = {"baseline": {role: accuracy([r for r in base if r["role"] == role]) for role in ROLES}}
        for method in methods:
            alpha = run["operating_point"][method]["alpha"]
            records = load_predictions(out_dir, "prompt_{}__{}".format(
                style, condition_name(method, "top_selective", alpha)))
            if records is None:
                continue
            entry[method] = {role: dict(accuracy([r for r in records if r["role"] == role]),
                                        **paired_bootstrap(by_id(base, role),
                                                           {"drop": by_id(records, role)},
                                                           boot["repetitions"], boot["seed"], boot["level"])["drop"])
                             for role in ROLES}
        prompts[style] = entry
    analysis["controls"]["prompts"] = prompts

    letters = {}
    for method in methods:
        alpha = run["operating_point"][method]["alpha"]
        records = load_predictions(out_dir, "test__" + condition_name(method, "top_selective", alpha))
        if records:
            letters[method] = accuracy([r for r in records if r["role"] == "forget"])["predicted_letters"]
    letters["baseline"] = accuracy([r for r in baseline if r["role"] == "forget"])["predicted_letters"]
    analysis["controls"]["answer_letters"] = letters

    # Reproduce the single-layer diagnostic in the report's Appendix A.
    # The gradient is with respect to g, while the intervention sets g = 1 - alpha;
    # a positive alpha * score therefore predicts a drop in correct-answer log p.
    first_order = []
    if "gate" in methods:
        for selection in ("top_localized", "bottom_localized"):
            selected = run["selections"]["gate"][selection]
            if len(selected) != 1:
                continue
            layer = selected[0]
            for alpha in run["strengths"]:
                if alpha == 0:
                    continue
                records = load_predictions(
                    out_dir, "test__" + condition_name("gate", selection, alpha))
                if records is None:
                    continue
                for role in ROLES:
                    measured = accuracy([r for r in records if r["role"] == role])
                    base = analysis["baseline"][role]
                    first_order.append({
                        "selection": selection, "layer": layer, "alpha": alpha, "role": role,
                        "predicted_log_probability_drop":
                            alpha * localization["gate"]["scores"][role][layer],
                        "actual_log_probability_drop":
                            base["mean_correct_log_probability"] - measured["mean_correct_log_probability"],
                        "actual_accuracy_drop_pp": 100 * (base["accuracy"] - measured["accuracy"]),
                    })
    if first_order:
        analysis["controls"]["first_order_check"] = first_order

    write_json(out_dir / "analysis.json", analysis)
    return analysis


def tables(analysis, out_dir):
    """Write the assignment's required comparison table as CSV, one file per method."""
    written = []
    for method, block in analysis["main"].items():
        lines = ["method,layers,alpha,wmdp_accuracy,delta_wmdp_pp,retain_accuracy,delta_retain_pp"]
        base = analysis["baseline"]
        lines.append("Baseline,-,0,{:.2f},0.00,{:.2f},0.00".format(
            100 * base["forget"]["accuracy"], 100 * base["retain"]["accuracy"]))
        for name, label in LABELS_FOR.items():
            row = block["rows"].get(name)
            if not row:
                continue
            layers = analysis["selections"][method].get(name)
            lines.append("{},{},{:g},{:.2f},{:.2f},{:.2f},{:.2f}".format(
                label, _units(layers).replace(",", "|"), block["alpha"],
                100 * row["forget"]["accuracy"],
                100 * (base["forget"]["accuracy"] - row["forget"]["accuracy"]),
                100 * row["retain"]["accuracy"],
                100 * (base["retain"]["accuracy"] - row["retain"]["accuracy"])))
        path = Path(out_dir) / "table_{}.csv".format(method)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(str(path))
    return written


def figures(analysis, figure_dir):
    """Three figures. Every panel carries exactly one y-axis: scores that live on
    different scales get their own panel rather than a shared twin axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = analysis.get("methods") or list(analysis["localization"])
    colours = {"top_localized": "#b91c1c", "top_selective": "#6d28d9",
               "bottom_localized": "#0f766e", "random_mean": "#64748b"}
    markers = dict(zip(methods, ["o", "s", "^", "D", "v", "P"]))
    made = []

    # 1: per-layer localization scores. One row per method, one column per score,
    #    so no panel mixes two scales.
    fig, axes = plt.subplots(len(methods), 2, figsize=(9.6, 3.3 * len(methods)), squeeze=False)
    for row, method in enumerate(methods):
        spec = METHODS[method]
        scores = analysis["localization"][method]["scores"]
        for column, (key, colour) in enumerate(((spec["localized"], "#b45309"),
                                               (spec["selective"], "#6d28d9"))):
            axis = axes[row][column]
            values = scores[key]
            layers = list(range(len(values)))
            axis.axhline(0, color="#94a3b8", lw=.8)
            axis.bar(layers, values, color=colour, width=.68)
            best = max(layers, key=lambda i: values[i])
            axis.annotate("layer {}".format(best), (best, values[best]), fontsize=7,
                          color=colour, ha="center",
                          va="bottom" if values[best] >= 0 else "top",
                          xytext=(0, 4 if values[best] >= 0 else -10), textcoords="offset points")
            axis.margins(y=.18)          # headroom so the annotation is not clipped
            axis.set_xticks(layers[::2])
            axis.set_xlabel("decoder layer")
            axis.set_ylabel("${}_\\ell$".format(SYMBOLS[key]))
            axis.set_title("{} — {}".format(method, "WMDP-only" if column == 0 else "forget-vs-retain"),
                           fontsize=9.5)
            axis.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(figure_dir / "localization.png", dpi=170)
    plt.close(fig)
    made.append("localization.png")

    # 2: strength curves. Single y-axis per panel (accuracy, %), line style separates the sets.
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 3.9),
                             sharey=True, squeeze=False)
    for axis, method in zip(axes[0], methods):
        curve = analysis["sweep"][method]
        strengths = sorted(curve, key=float)
        for name, colour in colours.items():
            for role, style in (("forget", "-"), ("retain", "--")):
                values = [100 * curve[st][name][role] for st in strengths if name in curve[st]]
                if len(values) == len(strengths):
                    axis.plot([float(st) for st in strengths], values, style, color=colour,
                              marker="o", ms=3.5,
                              label=LABELS_FOR[name] if role == "forget" else "_nolegend_")
        axis.axvline(analysis["main"][method]["alpha"], color="#0f172a", ls=":", lw=1)
        axis.set_title("{} (dotted: selected $\\alpha$)".format(method), fontsize=9.5)
        axis.set_xlabel("strength $\\alpha$")
        axis.grid(alpha=.25)
    axes[0][0].set_ylabel("accuracy (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    keys = [Line2D([], [], color="#334155", ls="-"), Line2D([], [], color="#334155", ls="--")]
    fig.legend(handles + keys, labels + ["WMDP (solid)", "retain (dashed)"],
               loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(figure_dir / "strength.png", dpi=170)
    plt.close(fig)
    made.append("strength.png")

    # 3: forgetting against retention. One pair of axes, both in percentage points.
    fig, axis = plt.subplots(figsize=(6.0, 4.6))
    base = analysis["baseline"]
    for method in methods:
        for name, colour in colours.items():
            xs, ys = [], []
            for st in sorted(analysis["sweep"][method], key=float):
                row = analysis["sweep"][method][st].get(name)
                if row is None:
                    continue
                xs.append(100 * (base["retain"]["accuracy"] - row["retain"]))
                ys.append(100 * (base["forget"]["accuracy"] - row["forget"]))
            axis.plot(xs, ys, markers[method], color=colour, ms=5.5, alpha=.85,
                      label="{} / {}".format(method, LABELS_FOR[name]))
    limit = max(axis.get_xlim()[1], axis.get_ylim()[1])
    axis.plot([0, limit], [0, limit], color="#94a3b8", ls=":", lw=.9, label="_nolegend_")
    axis.annotate("equal damage", (limit * .70, limit * .70), fontsize=6.5, color="#64748b",
                  rotation=38, ha="center", va="bottom")
    axis.annotate("better: more forgetting\nper unit of retain damage", (limit * .04, limit * .95),
                  fontsize=6.5, color="#64748b", va="top")
    axis.set_xlabel("retain accuracy drop (pp)")
    axis.set_ylabel("WMDP accuracy drop (pp)")
    axis.set_title("Forgetting vs retention (every tested strength)", fontsize=10)
    axis.grid(alpha=.25)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.savefig(figure_dir / "tradeoff.png", dpi=170)
    plt.close(fig)
    made.append("tradeoff.png")
    return made


def _units(selection):
    """Render a layer selection, or "varies" for the averaged random control."""
    return ",".join(map(str, selection)) if selection else "varies"


def _pct(value):
    return "{:.2f}".format(100 * value)


def _ci(entry):
    return "{:+.2f} [{:+.2f}, {:+.2f}]".format(entry["drop_pp"], *entry["interval_95"])


def markdown(analysis, out_dir):
    """Emit every table the report quotes, so no number is transcribed by hand."""
    lines, base = [], analysis["baseline"]
    methods = analysis.get("methods") or list(analysis["localization"])

    lines += ["## Localization scores", "",
              "| Layer | " + " | ".join("{}: {} | {}: {}".format(
                  m, METHODS[m]["localized"], m, METHODS[m]["selective"]) for m in methods) + " |",
              "| --- |" + " ---: |" * (2 * len(methods))]
    columns = []
    for m in methods:
        scores = analysis["localization"][m]["scores"]
        columns.append(scores[METHODS[m]["localized"]])
        columns.append(scores[METHODS[m]["selective"]])
    for layer in range(len(columns[0])):
        lines.append("| {} | ".format(layer) + " | ".join("{:+.4f}".format(c[layer]) for c in columns) + " |")
    lines += [""]
    # Whether the argmax is meaningful at all: how far the best layer is from the runner-up,
    # measured in standard errors of that gap.
    if any(analysis["localization"][m].get("selection") for m in methods):
        lines += ["", "Is the selected layer separable from the runner-up?", "",
                  "| Method | Score | Best layer | Runner-up | Gap | Gap in standard errors "
                  "| Layers within 1 s.e. | Layer chosen by each half | Questions needed for a 2 s.e. gap |",
                  "| --- | --- | :---: | :---: | ---: | ---: | --- | :---: | ---: |"]
        for m in methods:
            for key, entry in (analysis["localization"][m].get("selection") or {}).items():
                halves = entry.get("layer_chosen_by_each_half")
                lines.append("| {} | ${}_\\ell$ | {} | {} | {:+.4f} | **{:.2f}** | {} | {} | {} |".format(
                    m, SYMBOLS[key], entry["best_layer"], entry["runner_up"], entry["gap"],
                    entry["gap_in_standard_errors"],
                    ", ".join(map(str, entry["layers_within_one_standard_error"])),
                    "{} vs {}".format(*halves) if halves else "-",
                    entry.get("questions_for_two_standard_errors") or "-"))
        lines += [""]

    for method in methods:
        block = analysis["main"][method]
        lines += ["## Main comparison: {} (alpha = {:g}, k = {})".format(
                      method, block["alpha"], analysis.get("k", "?")), "",
                  "| Method | Layer | WMDP % | Delta WMDP pp [95% CI] | Retain % "
                  "| Delta Retain pp [95% CI] | WMDP mean log p |",
                  "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
                  "| Baseline | - | {} | 0.00 | {} | 0.00 | {:.3f} |".format(
                      _pct(base["forget"]["accuracy"]), _pct(base["retain"]["accuracy"]),
                      base["forget"]["mean_correct_log_probability"])]
        for name, label in LABELS_FOR.items():
            row = block["rows"].get(name)
            if not row:
                continue
            logp = row["forget"].get("mean_correct_log_probability")
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                label, _units(analysis["selections"][method].get(name)),
                _pct(row["forget"]["accuracy"]), _ci(block["intervals"]["forget"][name]),
                _pct(row["retain"]["accuracy"]), _ci(block["intervals"]["retain"][name]),
                "{:.3f}".format(logp) if logp is not None else "-"))
        lines.append("")
        if block["extra_over_random"]:
            lines += ["Extra drop over the random-layer mean (positive = more damage than random):", "",
                      "| Method | Extra WMDP pp [95% CI] | Extra Retain pp [95% CI] |",
                      "| --- | ---: | ---: |"]
            for name in block["extra_over_random"]["forget"]:
                forget = block["extra_over_random"]["forget"][name]
                retain = block["extra_over_random"]["retain"][name]
                lines.append("| {} | {:+.2f} [{:+.2f}, {:+.2f}] | {:+.2f} [{:+.2f}, {:+.2f}] |".format(
                    LABELS_FOR[name], forget["extra_drop_pp"], *forget["interval_95"],
                    *([retain["extra_drop_pp"]] + list(retain["interval_95"]))))
            lines.append("")
        ranking = block.get("selective_vs_localized") or {}
        if ranking:
            lines += ["RQ3, paired directly: how much *more* damage the WMDP-only ranking does than "
                      "the forget-vs-retain ranking (positive = WMDP-only is more damaging):", "",
                      "| Role | Extra damage from the WMDP-only ranking pp [95% CI] |", "| --- | ---: |"]
            for role in ROLES:
                lines.append("| {} | {:+.2f} [{:+.2f}, {:+.2f}] |".format(
                    "WMDP" if role == "forget" else "retain",
                    ranking[role]["extra_drop_pp"], *ranking[role]["interval_95"]))
            lines.append("")
        if "random_each" in block["rows"]:
            names = [n for n in analysis["selections"][method] if n.startswith("random_")]
            parts = ["{} -> WMDP {} / retain {}".format(
                _units(analysis["selections"][method][name]),
                _pct(row["forget"]), _pct(row["retain"]))
                for name, row in zip(names, block["rows"]["random_each"])]
            lines += ["Individual random selections: " + "; ".join(parts), ""]

    lines += ["## Strength sweep (test split accuracy %)", ""]
    for method in methods:
        curve = analysis["sweep"][method]
        strengths = sorted(curve, key=float)
        lines += ["**{}**".format(method), "",
                  "| Condition | " + " | ".join("a={}".format(st) for st in strengths) + " |",
                  "| --- |" + " ---: |" * len(strengths)]
        for name, label in LABELS_FOR.items():
            for role, tag in (("forget", "WMDP"), ("retain", "retain")):
                values = [curve[st][name][role] for st in strengths if name in curve[st]]
                if len(values) == len(strengths):
                    lines.append("| {} {} | ".format(label, tag) + " | ".join(_pct(v) for v in values) + " |")
        lines.append("")

    biology = analysis["controls"]["biology"]
    lines += ["## General-biology control ({} MMLU high-school biology questions)".format(
        biology["baseline"]["count"]), "",
        "| Condition | Accuracy % | Drop pp [95% CI] |", "| --- | ---: | ---: |",
        "| Baseline | {} | 0.00 |".format(_pct(biology["baseline"]["accuracy"]))]
    for method in methods:
        intervals = biology.get(method + "/intervals", {})
        for name, label in LABELS_FOR.items():
            key = method + "/" + name
            if key in biology:
                lines.append("| {} {} | {} | {} |".format(
                    method, label, _pct(biology[key]["accuracy"]),
                    _ci(intervals[name]) if name in intervals else "-"))
    lines.append("")

    prompts = analysis["controls"]["prompts"]
    if prompts:
        lines += ["## Prompt-format robustness ({} question subset)".format(
            analysis["questions"]["prompt_subset"]), "",
            "| Format | Role | Baseline % | Intervened % | Drop pp [95% CI] | Method |",
            "| --- | --- | ---: | ---: | ---: | --- |"]
        for style, entry in prompts.items():
            for method in methods:
                if method not in entry:
                    continue
                for role in ROLES:
                    lines.append("| {} | {} | {} | {} | {} | {} |".format(
                        style, "WMDP" if role == "forget" else "retain",
                        _pct(entry["baseline"][role]["accuracy"]), _pct(entry[method][role]["accuracy"]),
                        _ci(entry[method][role]), method))
        lines.append("")

    check = analysis["controls"].get("first_order_check") or []
    if check:
        lines += ["## Does the first-order score predict its own objective? (method gate, test split)", "",
                  "| Condition | Layer | alpha | Role | Predicted drop in log p | Actual drop in log p "
                  "| Actual accuracy drop pp |",
                  "| --- | :---: | :---: | --- | ---: | ---: | ---: |"]
        for row in check:
            lines.append("| {} | {} | {:g} | {} | {:+.3f} | {:+.3f} | {:+.2f} |".format(
                LABELS_FOR[row["selection"]], row["layer"], row["alpha"],
                "WMDP" if row["role"] == "forget" else "retain",
                row["predicted_log_probability_drop"], row["actual_log_probability_drop"],
                row["actual_accuracy_drop_pp"]))
        lines.append("")

    letters = analysis["controls"]["answer_letters"]
    lines += ["## Predicted answer letters on the WMDP test split", "",
              "| Condition | A | B | C | D |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, counts in letters.items():
        lines.append("| {} | {} |".format(name, " | ".join(str(counts[l]) for l in "ABCD")))
    lines.append("")

    path = Path(out_dir) / "report_tables.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
