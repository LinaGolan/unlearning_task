"""Resumable Stage 4 localization and the prespecified single-layer Control A."""

import json
import math
from pathlib import Path
import time
import uuid

from .baseline import equivalent_derived, initialize_run, load_evaluator
from .data import digest, file_digest, verify_prepared, write_json
from .evaluation import LABELS, evaluator_checks, summarize_logits
from .intervention_checks import assert_scores_match, model_guard, verified_inputs
from .interventions import LayerIntervention, decoder_layers, gate_gradients
from .runtime import environment_info


def read_localization_settings(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value["schema_version"] != 1 or value["control_questions_per_role"] != 16
            or value["control_strengths"] != [.05, .5]
            or value["control_sample_rule"] != "seeded_hash_balanced_by_subject"
            or value["half_split_rule"] != "seeded_hash_alternating_within_subject"
            or value["tie_rule"] != "lower_layer_index_first"):
        raise ValueError("Stage 4 settings differ from the declared sampling/Control A protocol.")
    if not -1 <= value["stability_review_spearman_below"] <= 1 or not 0 < value["sign_zero_tolerance"] < .001:
        raise ValueError("Invalid descriptive stability/sign thresholds.")
    return value


def localization_examples(data_dir, config):
    manifest = verify_prepared(data_dir)
    if manifest["config"] != config:
        raise ValueError("Prepared data differs from the experiment configuration.")
    pools = []
    for role in ("forget", "retain"):
        path = Path(data_dir) / role / "localization.jsonl"
        pools.append([dict(json.loads(line), role=role, split="localization")
                      for line in path.read_text(encoding="utf-8").splitlines()])
    return manifest, [row for pair in zip(*pools) for row in pair]


def select_control_examples(examples, config, protocol):
    selected = {}
    for role in ("forget", "retain"):
        subjects = config["datasets"][role]["configs"]
        if protocol["control_questions_per_role"] % len(subjects):
            raise ValueError("Control A must be balanced within each role's subjects.")
        per_subject = protocol["control_questions_per_role"] // len(subjects)
        selected[role] = []
        for subject in subjects:
            pool = sorted((e for e in examples if e["role"] == role and e["subject"] == subject),
                          key=lambda e: (digest([config["seed"], "control_a", e["id"]]), e["id"]))
            if len(pool) < per_subject or any(e["split"] != "development" for e in pool):
                raise ValueError("Control A requires enough development examples per subject.")
            selected[role].extend(pool[:per_subject])
        selected[role].sort(key=lambda e: (digest([config["seed"], "control_a_order", e["id"]]), e["id"]))
    return [row for pair in zip(selected["forget"], selected["retain"]) for row in pair]


def verified_stage3(stage3_dir, config, settings, baseline_run):
    root = Path(stage3_dir)
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    context = run["context"]
    if digest(context) != run["fingerprint"] or summary["run_fingerprint"] != run["fingerprint"]:
        raise ValueError("Stage 3 run identity is invalid.")
    if context["config"] != config or context["settings"] != settings or context["baseline_fingerprint"] != baseline_run["fingerprint"]:
        raise ValueError("Stage 3 used different model/data/prompt settings or a different baseline.")
    expected = {"baseline_replay", "cleanup_and_frozen_weights", "compute_feasibility", "disabled",
                "evaluator", "full_strength", "gradients", "zero_strength"}
    if (summary["status"] != "passed" or set(summary["checks"]) != expected
            or any(c["status"] != "passed" for c in summary["checks"].values())
            or summary["final_test_evaluated"] or summary["layers_selected"]):
        raise ValueError("Stage 3 is incomplete or requires review before localization.")
    if context["profile"] != "full" or summary["runtime_forecast"]["recommended_profile"] != "full":
        raise ValueError("Review the profile choice; this Stage 4 run implements the approved full profile.")
    # CLI/report additions are allowed; the measured mechanism and evaluator must be identical.
    for name in ("interventions.py", "evaluation.py", "data.py", "baseline.py"):
        if context["source_hashes"][name] != file_digest(Path(__file__).parent / name):
            raise ValueError("The validated mechanism/evaluator changed since Stage 3: " + name)
    return run


def stage4_inputs(config, settings, protocol, data_dir, baseline_dir, stage3_dir):
    manifest, development, baseline_run, records = verified_inputs(config, settings, data_dir, baseline_dir)
    if manifest["profile"] != "full":
        raise ValueError("Use the full profile approved after Stage 3.")
    stage3_run = verified_stage3(stage3_dir, config, settings, baseline_run)
    _, localization = localization_examples(data_dir, config)
    control = select_control_examples(development, config, protocol)
    if {e["id"] for e in localization} & {e["id"] for e in control}:
        raise ValueError("Localization and Control A must be disjoint.")
    return manifest, localization, control, baseline_run, records, stage3_run


def planned_jobs(context):
    jobs = [{"kind": "localization", "id": e["id"]} for e in context["localization_examples"]]
    jobs += [{"kind": "control_gradient", "id": e["id"]} for e in context["control_examples"]]
    jobs += [{"kind": "control_intervention", "id": e["id"], "layer": layer, "alpha": alpha}
             for e in context["control_examples"] for layer in range(context["layer_count"])
             for alpha in context["protocol"]["control_strengths"]]
    return jobs


def record_path(output, job):
    return Path(output) / "records" / (digest(job) + ".json")


def save_record(output, run, job, record):
    row = dict(record, **job, run_fingerprint=run["fingerprint"])
    row["record_hash"] = digest(row)
    path = record_path(output, job)
    if path.exists():
        raise ValueError("Refusing to overwrite a completed Stage 4 measurement.")
    write_json(path, row)
    return row


def read_records(output, run):
    if digest(run["context"]) != run["fingerprint"]:
        raise ValueError("Stage 4 run fingerprint mismatch.")
    context = run["context"]
    jobs = planned_jobs(context)
    expected = {digest(job): job for job in jobs}
    examples = {e["id"]: e for e in context["localization_examples"] + context["control_examples"]}
    records = {}
    for path in sorted((Path(output) / "records").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        checksum = row.pop("record_hash")
        if digest(row) != checksum or row["run_fingerprint"] != run["fingerprint"]:
            raise ValueError("Stage 4 record checksum/run identity mismatch.")
        if path.stem not in expected:
            raise ValueError("Unexpected Stage 4 record or condition.")
        job = expected[path.stem]
        if any(row[key] != value for key, value in job.items()):
            raise ValueError("Record condition differs from its filename.")
        metadata = examples[row["id"]]
        for key in ("content_hash", "role", "split", "subject", "correct_answer", "prompt_hash", "input_tokens", "input_ids_hash"):
            if row[key] != metadata[key]:
                raise ValueError("Stage 4 metadata mismatch: " + key)
        if digest(row["prompt"]) != row["prompt_hash"] or digest(row["input_ids"]) != row["input_ids_hash"] or len(row["input_ids"]) != row["input_tokens"]:
            raise ValueError("Saved prompt/tokens changed.")
        if job["kind"] != "control_intervention":
            gradients = row["gradients"]
            log_probs = row["answer_log_probabilities"]
            if len(gradients) != context["layer_count"] or not all(math.isfinite(g) for g in gradients):
                raise ValueError("Invalid layer derivatives.")
            if len(log_probs) != 4 or not all(math.isfinite(p) for p in log_probs) or not math.isclose(sum(math.exp(p) for p in log_probs), 1, abs_tol=1e-5):
                raise ValueError("Invalid normalized answer scores.")
            prediction = LABELS[max(range(4), key=lambda i: log_probs[i])]
            if (row["prediction"] != prediction or row["correct"] != (prediction == row["correct_answer"])
                    or row["correct_answer_log_probability"] != log_probs[LABELS.index(row["correct_answer"])]) :
                raise ValueError("Gradient record answer scores are inconsistent.")
            seconds = row["forward_backward_seconds"]
        else:
            derived = summarize_logits([row["answer_logits"][letter] for letter in LABELS], LABELS.index(row["correct_answer"]))
            if any(not equivalent_derived(row[key], value) for key, value in derived.items()):
                raise ValueError("Intervention prediction is inconsistent with its logits.")
            seconds = row["seconds_per_example"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid measurement duration.")
        row["record_hash"] = checksum
        records[path.stem] = row
    baseline = {e["id"]: e for e in context["control_baselines"]}
    for row in records.values():
        if row["kind"] == "control_gradient":
            comparable = dict(row, answer_log_probabilities=dict(zip(LABELS, row["answer_log_probabilities"])))
            assert_scores_match(comparable, baseline[row["id"]], context["settings"], "Saved Control A baseline replay")
        if row["kind"] == "control_intervention":
            parent = records.get(digest({"kind": "control_gradient", "id": row["id"]}))
            if parent is None:
                raise ValueError("Control A effect has no matching saved gradient.")
            base_score = baseline[row["id"]]["correct_answer_log_probability"]
            values = {"baseline_score": base_score, "predicted_drop": row["alpha"] * parent["gradients"][row["layer"]],
                      "actual_drop": base_score - row["correct_answer_log_probability"]}
            if any(not equivalent_derived(row[key], value) for key, value in values.items()):
                raise ValueError("Control A score-drop calculation is inconsistent.")
    return [records[digest(job)] for job in jobs if digest(job) in records]


def run_localization(config, settings, protocol, data_dir, baseline_dir, stage3_dir,
                     output, max_new=None, evaluator=None, make_plots=True):
    if max_new is not None and (type(max_new) is not int or max_new < 1):
        raise ValueError("max-new must be a positive number of additional records.")
    manifest, localization, control, baseline_run, baseline_records, stage3_run = stage4_inputs(
        config, settings, protocol, data_dir, baseline_dir, stage3_dir)
    test_fixture = evaluator is not None
    evaluator = evaluator if evaluator is not None else load_evaluator(config, settings)
    model = evaluator.model
    count = len(decoder_layers(model))
    old = baseline_run["context"]
    if (count != old["layer_count"] or str(next(model.parameters()).dtype) != old["dtype"]
            or evaluator.tokenizer.get_chat_template() != old["chat_template"]
            or dict(zip(LABELS, evaluator.label_ids)) != old["label_token_ids"] or evaluator.settings != settings):
        raise ValueError("Model layout, precision, tokenizer, or settings differ from the reviewed baseline.")
    prepared = {e["id"]: evaluator.prepare(e) for e in localization + control}
    baseline_by_id = {r["id"]: r for r in baseline_records}
    for example in control:
        item, original = prepared[example["id"]], baseline_by_id[example["id"]]
        if item["input_ids"] != original["input_ids"] or item["prompt_hash"] != original["prompt_hash"]:
            raise ValueError("Control A prompt or tokens changed since Stage 2.")
    def metadata(example):
        item = prepared[example["id"]]
        return {**{k: example[k] for k in ("id", "content_hash", "role", "split", "subject")},
                "correct_answer": LABELS[example["answer"]], "prompt_hash": item["prompt_hash"],
                "input_tokens": len(item["input_ids"]), "input_ids_hash": digest(item["input_ids"])}
    root = Path(__file__).parent
    context = {"stage": 4, "schema_version": 1, "config": config, "settings": settings, "protocol": protocol,
        "profile": manifest["profile"], "data_request_hash": manifest["request_hash"],
        "baseline_fingerprint": baseline_run["fingerprint"], "stage3_fingerprint": stage3_run["fingerprint"],
        "is_test_fixture": test_fixture, "environment": environment_info(), "layer_count": count,
        "chat_template": old["chat_template"], "label_token_ids": old["label_token_ids"], "dtype": old["dtype"],
        "source_hashes": {name: file_digest(root / name) for name in ("localization.py", "localization_analysis.py", "localization_report.py",
            "interventions.py", "intervention_checks.py", "evaluation.py", "baseline.py", "data.py", "metrics.py", "runtime.py")},
        "localization_examples": [metadata(e) for e in localization], "control_examples": [metadata(e) for e in control],
        "control_baselines": [{k: baseline_by_id[e["id"]][k] for k in ("id", "prediction", "answer_log_probabilities", "correct_answer_log_probability")}
                              for e in control]}
    output = Path(output)
    run = initialize_run(output, context)
    records = read_records(output, run)
    done = {digest({k: r[k] for k in (("kind", "id", "layer", "alpha") if r["kind"] == "control_intervention" else ("kind", "id"))}) for r in records}
    pending = [job for job in planned_jobs(context) if digest(job) not in done]
    if max_new is not None:
        pending = pending[:max_new]
    from .localization_report import stage4_report
    if not pending:
        return stage4_report(output, make_plots=make_plots)
    # Save partial progress before preflight; never replace completed per-question files.
    stage4_report(output, make_plots=False)
    examples = {e["id"]: e for e in localization + control}
    metadata_by_id = {e["id"]: e for e in context["localization_examples"] + context["control_examples"]}
    control_gradients = {r["id"]: r for r in records if r["kind"] == "control_gradient"}
    guard = model_guard(model)
    started = time.time()
    segment = {"run_fingerprint": run["fingerprint"], "started_unix": started, "completed_before": len(records), "new_records": 0, "status": "running"}
    phase = "evaluator_preflight"
    try:
        checks = evaluator_checks(evaluator)
        write_json(output / "evaluator_checks.json", dict(checks, run_fingerprint=run["fingerprint"]))
        # Replaying one question per role catches changes before collecting new localization scores.
        for role in ("forget", "retain"):
            example = next(e for e in control if e["role"] == role)
            assert_scores_match(evaluator.score([example])[0], baseline_by_id[example["id"]], settings, "Stage 4 preflight")
        for job in pending:
            phase = job["kind"]
            example = examples[job["id"]]
            common = {**metadata_by_id[example["id"]], **prepared[example["id"]]}
            common.pop("id")  # The condition key supplies the ID.
            if job["kind"] != "control_intervention":
                measured = gate_gradients(evaluator, example)
                measured.pop("id")
                prediction = LABELS[max(range(4), key=lambda i: measured["answer_log_probabilities"][i])]
                row = dict(common, **measured, prediction=prediction, correct=prediction == common["correct_answer"])
                if job["kind"] == "control_gradient":
                    comparable = dict(row, answer_log_probabilities=dict(zip(LABELS, row["answer_log_probabilities"])))
                    assert_scores_match(comparable, baseline_by_id[example["id"]], settings, "Control A gradient baseline")
            else:
                with LayerIntervention(model, [job["layer"]], alpha=job["alpha"]):
                    scores = evaluator.score([example])[0]
                base = baseline_by_id[example["id"]]["correct_answer_log_probability"]
                row = dict(common)
                row.update(scores, baseline_score=base,
                           predicted_drop=job["alpha"] * control_gradients[example["id"]]["gradients"][job["layer"]],
                           actual_drop=base - scores["correct_answer_log_probability"])
            saved = save_record(output, run, job, row)
            if job["kind"] == "control_gradient":
                control_gradients[example["id"]] = saved
            segment["new_records"] += 1
            if segment["new_records"] % 32 == 0 or segment["new_records"] == len(pending):
                print("Stage 4: saved {}/{} records ({}).".format(len(records) + segment["new_records"], len(planned_jobs(context)), phase), flush=True)
            if job["kind"] == "localization" and len(records) + segment["new_records"] == len(localization):
                stage4_report(output, make_plots=False)  # Freeze selections before Control A outcomes exist.
        if model_guard(model) != guard or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Model state or hooks changed during Stage 4.")
        segment["status"] = "complete"
    except Exception as exc:
        from .access import failure_details
        segment["status"] = "failed"
        segment["failure"] = failure_details(exc, phase)
        segment["failure"].update(error_kind="stage4_measurement_failed", error="Stage 4 stopped during {}. Completed records are preserved; review the saved failure before resuming.".format(phase))
    finally:
        segment["elapsed_seconds"] = time.time() - started
        segment["cleanup_passed"] = model_guard(model) == guard and all(p.grad is None for p in model.parameters())
        if not segment["cleanup_passed"]:
            segment["status"] = "failed"
            segment["failure"] = {"phase": "cleanup", "error": "Model state or hooks changed. Review before resuming."}
        write_json(output / "segments" / ("{:.6f}-{}.json".format(started, uuid.uuid4().hex[:8])), segment)
    return stage4_report(output, make_plots=make_plots)
