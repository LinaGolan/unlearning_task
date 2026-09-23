"""Bounded Stage 3 validation on reviewed development examples, never test data."""

import json
import math
from pathlib import Path

from .baseline import development_examples, initialize_run, load_evaluator, read_predictions
from .data import digest, file_digest, write_json
from .evaluation import LABELS, evaluator_checks
from .interventions import LayerIntervention, decoder_layers, gate_gradients, hidden_output
from .metrics import ability_gate, accuracy_summary, runtime_forecast
from .runtime import environment_info


def read_check_settings(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value["schema_version"] != 1 or value["sample_rule"] != "median_and_longest_development_prompt_per_role" or value["finite_difference_rule"] != "median_prompt_per_role_first_middle_last_layers":
        raise ValueError("Unsupported Stage 3 check protocol.")
    steps = value["finite_difference_steps"]
    if not isinstance(steps, list) or len(steps) != 2 or len(set(steps)) != 2:
        raise ValueError("Two distinct finite-difference steps are required.")
    for number in steps + [value["gradient_atol"], value["gradient_rtol"]]:
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError("Numerical tolerances and steps must be positive and finite.")
    if max(steps) > .05 or not 0 < value["minimum_gpu_headroom_fraction"] < 1:
        raise ValueError("Invalid finite-difference step or memory headroom.")
    return value


def verified_inputs(config, settings, data_dir, baseline_dir):
    """Read-only verification, including the complete reviewed ability gate."""
    manifest, examples = development_examples(data_dir, config)
    run = json.loads((Path(baseline_dir) / "run.json").read_text(encoding="utf-8"))
    context = run["context"]
    if context["config"] != config or context["settings"] != settings or settings["prompt_version"] != "mcq_prefill_v1":
        raise ValueError("Stage 3 requires the matching reviewed answer-prefix baseline.")
    if context["data_request_hash"] != manifest["request_hash"] or context["profile"] != manifest["profile"]:
        raise ValueError("The baseline uses different prepared data.")
    records = read_predictions(baseline_dir, run)
    by_id = {row["id"]: row for row in records}
    if set(by_id) != {row["id"] for row in examples}:
        raise ValueError("The development baseline is incomplete or uses different questions.")
    for example in examples:
        record = by_id[example["id"]]
        if any(record[key] != example[key] for key in ("content_hash", "role", "subject", "split")) or record["correct_answer"] != LABELS[example["answer"]]:
            raise ValueError("Prepared questions differ from saved baseline metadata.")
    gate = ability_gate({role: accuracy_summary([r for r in records if r["role"] == role])
                         for role in ("forget", "retain")}, settings)
    if gate["status"] != "passed":
        raise ValueError("The baseline ability gate needs review before Stage 3.")
    return manifest, examples, run, records


def select_check_examples(examples, records):
    lengths = {r["id"]: r["input_tokens"] for r in records}
    selected = []
    for role in ("forget", "retain"):
        rows = sorted((e for e in examples if e["role"] == role), key=lambda e: (lengths[e["id"]], e["id"]))
        for kind, example in (("median", rows[len(rows) // 2]), ("longest", rows[-1])):
            selected.append(dict(example, check_kind=kind))
    return selected


def assert_scores_match(actual, expected, settings, description):
    left, right = actual["answer_log_probabilities"], expected["answer_log_probabilities"]
    errors = [abs(left[label] - right[label]) for label in LABELS]
    if any(not math.isclose(left[label], right[label], abs_tol=settings["score_atol"], rel_tol=settings["score_rtol"]) for label in LABELS):
        raise ValueError(description + ": A-D log probabilities disagree.")
    # A prediction flip is also evidence to review, even if close to a tie.
    if actual["prediction"] != expected["prediction"]:
        raise ValueError(description + ": predicted answer changed.")
    return max(errors)


def model_guard(model):
    """Lightweight real-model guard; tiny-model tests compare every weight value."""
    return {"parameters": [(name, id(p), p.data_ptr(), p._version, p.requires_grad)
                           for name, p in model.named_parameters()],
            "hooks": [(name, tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
                      for name, m in model.named_modules()]}


def full_strength_check(evaluator, example):
    """Observe outputs AFTER intervention at every block and token position."""
    import torch
    seen, handles = [], []
    def observer(index):
        def observe(module, args, kwargs, output):
            h_in = args[0] if args else kwargs["hidden_states"]
            seen.append({"layer": index, "all_positions_equal": torch.equal(h_in, hidden_output(output)),
                         "shape": list(h_in.shape)})
        return observe
    with LayerIntervention(evaluator.model, range(len(decoder_layers(evaluator.model))), alpha=1):
        try:
            for index, block in enumerate(decoder_layers(evaluator.model)):
                handles.append(block.register_forward_hook(observer(index), with_kwargs=True))
            evaluator.score([example])
        finally:
            for handle in handles:
                handle.remove()
    if len(seen) != len(decoder_layers(evaluator.model)) or not all(r["all_positions_equal"] for r in seen):
        raise ValueError("Full-strength layer outputs did not equal their inputs at every position.")
    return {"status": "passed", "layers": seen}


def finite_difference_check(evaluator, example, gradient, check_settings):
    import torch
    count = len(decoder_layers(evaluator.model))
    entries = []
    for layer in sorted({0, count // 2, count - 1}):
        analytic = gradient["gradients"][layer]
        for step in check_settings["finite_difference_steps"]:
            values = []
            for offset in (step, -step):
                gates = torch.ones(count, dtype=next(evaluator.model.parameters()).dtype, device=evaluator.device)
                gates[layer] += offset
                with LayerIntervention(evaluator.model, gates=gates):
                    values.append(evaluator.score([example])[0]["correct_answer_log_probability"])
            numeric = (values[0] - values[1]) / (2 * step)
            tolerance = check_settings["gradient_atol"] + check_settings["gradient_rtol"] * abs(analytic)
            # Sign near zero is not numerically meaningful; flag other reversals.
            meaningful = abs(analytic) > 2 * check_settings["gradient_atol"]
            sign_ok = not meaningful or analytic * numeric > 0
            entries.append({"layer": layer, "step": step, "analytic_dM_dg": analytic,
                            "numeric_dM_dg": numeric, "absolute_error": abs(analytic - numeric),
                            "allowed_error": tolerance, "sign_meaningful": meaningful,
                            "sign_passed": sign_ok,
                            "passed": abs(analytic - numeric) <= tolerance and sign_ok})
    return {"id": example["id"], "status": "passed" if all(e["passed"] for e in entries) else "review_required",
            "comparisons": entries}


def run_intervention_checks(config, settings, checks, data_dir, baseline_dir, output, evaluator=None):
    """Run four-example correctness/feasibility checks. Does not select layers."""
    import torch
    manifest, examples, baseline_run, records = verified_inputs(config, settings, data_dir, baseline_dir)
    sample = select_check_examples(examples, records)
    root = Path(__file__).parent
    context = {"stage": 3, "schema_version": 1, "config": config, "settings": settings,
               "checks": checks, "profile": manifest["profile"],
               "baseline_fingerprint": baseline_run["fingerprint"],
               "environment": environment_info(),
               "source_hashes": {p.name: file_digest(p) for p in sorted(root.glob("*.py"))},
               "examples": [{k: r[k] for k in ("id", "role", "split", "content_hash", "check_kind")} for r in sample]}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Each Stage 3 check is a new short attempt. Use a new empty output directory.")
    run = initialize_run(output, context)
    report = {"stage": 3, "status": "running", "run_fingerprint": run["fingerprint"],
              "baseline_fingerprint": baseline_run["fingerprint"], "is_research_result": False,
              "final_test_evaluated": False, "layers_selected": False, "checks": {}}
    def save_check(name, value):
        report["checks"][name] = value
        write_json(output / "summary.json", report)
        print("Stage 3: {} - {}".format(name, value["status"]), flush=True)
    write_json(output / "summary.json", report)
    before = None
    phase = "model_loading"
    try:
        evaluator = evaluator if evaluator is not None else load_evaluator(config, settings)
        model = evaluator.model
        before = model_guard(model)
        phase = "model_and_tokenizer_identity"
        old_context = baseline_run["context"]
        count = len(decoder_layers(model))
        if (count != old_context["layer_count"] or str(next(model.parameters()).dtype) != old_context["dtype"]
                or evaluator.tokenizer.get_chat_template() != old_context["chat_template"]
                or dict(zip(LABELS, evaluator.label_ids)) != old_context["label_token_ids"]
                or evaluator.settings != settings):
            raise ValueError("Model layout, precision, tokenizer, or settings changed since baseline.")
        phase = "evaluator"
        save_check("evaluator", evaluator_checks(evaluator))
        saved = {r["id"]: r for r in records}
        fresh, replay_errors, zero_errors = {}, [], []
        phase = "baseline_replay_and_zero_strength"
        for example in sample:
            original = saved[example["id"]]
            current = evaluator.score([example])[0]
            if current["prompt_hash"] != original["prompt_hash"] or current["input_ids"] != original["input_ids"]:
                raise ValueError("The reviewed baseline prompt or input tokens changed.")
            replay_errors.append(assert_scores_match(current, original, settings, "Baseline replay"))
            fresh[example["id"]] = current
            with LayerIntervention(model, range(count), alpha=0):
                zero = evaluator.score([example])[0]
            zero_errors.append(assert_scores_match(zero, current, settings, "Alpha zero"))
        save_check("baseline_replay", {"status": "passed", "examples": len(sample), "max_log_probability_error": max(replay_errors)})
        save_check("zero_strength", {"status": "passed", "max_log_probability_error": max(zero_errors)})
        harmless = {"id": "check-stage3", "question": "How many days are in a week?",
                    "choices": ["five", "six", "seven", "eight"], "answer": 2}
        phase = "full_strength"
        save_check("full_strength", full_strength_check(evaluator, harmless))
        phase = "disabled"
        with LayerIntervention(model, [0, count - 1] if count > 1 else [0], alpha=.5, enabled=False):
            disabled = evaluator.score([sample[0]])[0]
        error = assert_scores_match(disabled, fresh[sample[0]["id"]], settings, "Disabled intervention")
        save_check("disabled", {"status": "passed", "max_log_probability_error": error})

        # First harmless backward warms kernels; measurements below are all real gate derivatives.
        phase = "gate_gradient_warmup"
        gate_gradients(evaluator, harmless)
        measurements, differences = [], []
        for example in sample:
            phase = "gate_gradient_" + example["role"] + "_" + example["check_kind"]
            measured = gate_gradients(evaluator, example)
            actual = dict(answer_log_probabilities=dict(zip(LABELS, measured["answer_log_probabilities"])),
                          prediction=LABELS[max(range(4), key=lambda i: measured["answer_log_probabilities"][i])])
            assert_scores_match(actual, fresh[example["id"]], settings, "Differentiable identity gates")
            measured.update(role=example["role"], check_kind=example["check_kind"], split="development")
            measurements.append(measured)
            write_json(output / "gate_probe.json", {"measurements": measurements})
            if example["check_kind"] == "median":
                phase = "finite_difference_" + example["role"]
                differences.append(finite_difference_check(evaluator, example, measured, checks))
                write_json(output / "finite_differences.json", differences)
            print("Saved gate check {}/{}.".format(len(measurements), len(sample)), flush=True)
        passed = all(r["status"] == "passed" for r in differences)
        save_check("gradients", {"status": "passed" if passed else "review_required",
                                 "example_count": len(measurements), "layers_per_example": count,
                                 "finite_difference_comparisons": sum(len(r["comparisons"]) for r in differences),
                                 "note": "Numerical derivatives only; no layer ranking or selection."})
        phase = "cleanup_and_frozen_weights"
        restored = evaluator.score([sample[0]])[0]
        error = assert_scores_match(restored, fresh[sample[0]["id"]], settings, "Removed intervention")
        if model_guard(model) != before or any(p.grad is not None for p in model.parameters()):
            raise ValueError("Model parameter state or registered hooks changed during checks.")
        save_check("cleanup_and_frozen_weights", {"status": "passed", "max_log_probability_error": error,
                   "guard": "parameter identities, storage addresses, mutation counters, frozen flags, no weight gradients, and hook IDs"})
        phase = "compute_feasibility"
        forecast = runtime_forecast(config, records, {"measurements": measurements}, settings, count)
        forecast["mean_actual_gate_forward_backward_seconds"] = forecast.pop("mean_forward_backward_proxy_seconds")
        forecast["note"] = "Uses measured gate derivatives on median/longest development prompts and Stage 2 forward times. Excludes downloads/queueing; future prompt lengths and interference remain uncertain."
        report["runtime_forecast"] = forecast
        peaks = [m["peak_gpu_allocated_bytes"] for m in measurements if m["peak_gpu_allocated_bytes"] is not None]
        if evaluator.device.type == "cuda":
            total = torch.cuda.get_device_properties(evaluator.device).total_memory
            peak = max(peaks)
            headroom = (total - peak) / total
            feasible = headroom >= checks["minimum_gpu_headroom_fraction"] and forecast["recommended_profile"] != "review_compute_budget"
            save_check("compute_feasibility", {"status": "passed" if feasible else "review_required",
                       "peak_gpu_allocated_bytes": peak, "total_gpu_bytes": total,
                       "allocated_memory_headroom_fraction": headroom,
                       "recommended_profile": forecast["recommended_profile"],
                       "note": "Allocated tensor memory is not all device memory; this is a feasibility estimate."})
        else:
            save_check("compute_feasibility", {"status": "not_measured", "note": "CPU test fixture; no research GPU evidence."})
        report["status"] = "passed" if all(c["status"] == "passed" for c in report["checks"].values()) else "review_required"
        report["next_action"] = ("Review this evidence and confirm the profile before Stage 4 localization."
                                 if report["status"] == "passed" else "Review the flagged check before localization; do not change tolerances after seeing results without documenting why.")
    except Exception as exc:
        # No raw loader exception: credentials and remote payloads must not enter artifacts.
        from .access import failure_details
        report["status"] = "review_required"
        report["failure"] = failure_details(exc, phase)
        if phase != "model_loading":
            report["failure"].update(error_kind="out_of_memory" if isinstance(exc, torch.cuda.OutOfMemoryError) else "check_failed",
                                     error="Stage 3 stopped during {}. Inspect saved checks and share this report.".format(phase))
        report["next_action"] = "Download this result ZIP for review. No localization or final-test evaluation was started."
        if before is not None:
            report["cleanup_after_failure"] = model_guard(evaluator.model) == before
        write_json(output / "failure.json", report["failure"])
    write_json(output / "summary.json", report)
    return report
