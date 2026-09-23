"""Resumable fixed-strength biology and prompt controls; no selection or training."""

import json
from pathlib import Path
import time
import uuid

from .baseline import initialize_run, load_evaluator
from .data import digest, file_digest, write_json
from .evaluation import LABELS, MCQEvaluator, evaluator_checks
from .intervention_checks import assert_scores_match, model_guard
from .interventions import LayerIntervention, decoder_layers
from .robustness_inputs import robustness_inputs
from .runtime import environment_info
from .sweep import validate_scores


def control_conditions(context):
    return [{"method": "baseline", "alpha": 0.0, "layers": []}] + [
        {"method": s["name"], "alpha": context["decision"]["alpha"], "layers": s["layers"]} for s in context["selections"]]


def planned_jobs(context):
    return [{"group": e["group"], "id": e["id"], "method": c["method"], "alpha": c["alpha"]}
            for group in ("biology", "alternative") for c in control_conditions(context)
            for e in context["examples"] if e["group"] == group]


def read_control_records(output, run):
    c = run["context"]
    if digest(c) != run["fingerprint"]:
        raise ValueError("Stage 6 run fingerprint differs.")
    decision = {k: v for k, v in c["decision"].items() if k != "decision_fingerprint"}
    if digest(decision) != c["decision"]["decision_fingerprint"] or decision["source_split"] != "development":
        raise ValueError("Invalid frozen strength decision.")
    metadata = {(e["group"], e["id"]): e for e in c["examples"]}
    primary = {e["id"]: e for e in c["primary_examples"]}
    if len(metadata) != len(c["examples"]) or len(primary) != len(c["primary_examples"]):
        raise ValueError("Duplicated Stage 6 question metadata.")
    for e in list(metadata.values()) + list(primary.values()):
        if (digest(e["prompt"]) != e["prompt_hash"] or digest(e["input_ids"]) != e["input_ids_hash"]
                or len(e["input_ids"]) != e["input_tokens"]):
            raise ValueError("Stage 6 prompt or token checksum mismatch.")
    expected_refs = {(e["id"], s["method"], s["alpha"]) for e in primary.values() for s in control_conditions(c)}
    seen = set()
    records = []
    for original in c["primary_records"]:
        r = dict(original)
        checksum = r.pop("record_hash")
        key = (r["id"], r["method"], r["alpha"])
        if digest(r) != checksum or r["run_fingerprint"] != c["stage5_test_fingerprint"] or key not in expected_refs or key in seen:
            raise ValueError("Invalid imported primary-prompt prediction.")
        validate_scores(r, primary[r["id"]])
        seen.add(key)
        records.append(dict(original, group="primary"))
    if seen != expected_refs:
        raise ValueError("Incomplete primary-prompt reference.")
    expected = {digest(j): j for j in planned_jobs(c)}
    saved = {}
    for path in sorted((Path(output) / "records").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        checksum = r.pop("record_hash")
        if digest(r) != checksum or r["run_fingerprint"] != run["fingerprint"] or path.stem not in expected:
            raise ValueError("Stage 6 record checksum, run, or filename mismatch.")
        if any(r[k] != v for k, v in expected[path.stem].items()):
            raise ValueError("Stage 6 record condition mismatch.")
        validate_scores(r, metadata[(r["group"], r["id"])])
        r["record_hash"] = checksum
        saved[path.stem] = r
    return records + [saved[digest(j)] for j in planned_jobs(c) if digest(j) in saved]


def run_robustness(config, settings, protocol, data_dir, baseline_dir, stage4_dir, stage5_archive,
                   output, max_new=None, evaluator=None, make_plots=True):
    if max_new is not None and (type(max_new) is not int or max_new <= 0):
        raise ValueError("max_new must be a positive integer.")
    prior = robustness_inputs(config, settings, protocol, data_dir, baseline_dir, stage4_dir, stage5_archive)
    fixture = evaluator is not None
    if prior["runs"]["test"]["context"]["is_test_fixture"] != fixture:
        raise ValueError("Synthetic inputs and research runs cannot be mixed.")
    evaluator = evaluator if evaluator is not None else load_evaluator(config, settings)
    model = evaluator.model
    old = prior["baseline_run"]["context"]
    if (len(decoder_layers(model)) != old["layer_count"] or evaluator.settings != settings
            or str(next(model.parameters()).dtype) != old["dtype"]
            or evaluator.tokenizer.get_chat_template() != old["chat_template"]
            or dict(zip(LABELS, evaluator.label_ids)) != old["label_token_ids"]):
        raise ValueError("Stage 6 evaluator differs from the reviewed baseline.")
    alternative_settings = dict(settings, instruction=protocol["alternative_instruction"])
    alternative = MCQEvaluator(model, evaluator.tokenizer, alternative_settings)
    evaluators = {"biology": evaluator, "alternative": alternative}
    examples = [dict(e, group="biology") for e in prior["biology_examples"]] + [dict(e, group="alternative") for e in prior["prompt_examples"]]
    metadata = []
    for e in examples:
        prepared = evaluators[e["group"]].prepare(e)
        metadata.append({**{k: e[k] for k in ("id", "role", "split", "subject", "content_hash", "group")},
            **prepared, "correct_answer": LABELS[e["answer"]], "input_ids_hash": digest(prepared["input_ids"]),
            "input_tokens": len(prepared["input_ids"])})
    context = {"stage": 6, "schema_version": 1, "config": config, "settings": settings,
        "alternative_settings": alternative_settings, "protocol": protocol, "environment": environment_info(),
        "is_test_fixture": fixture, "decision": prior["decision"], "stage5_archive_sha256": prior["archive_sha256"],
        "stage5_test_fingerprint": prior["runs"]["test"]["fingerprint"],
        "selections": prior["runs"]["test"]["context"]["selections"], "examples": metadata,
        "primary_examples": prior["primary_examples"], "primary_records": prior["primary_records"],
        "source_hashes": {name: file_digest(Path(__file__).parent / name) for name in
            ("robustness.py", "robustness_inputs.py", "robustness_analysis.py", "robustness_report.py",
             "interventions.py", "evaluation.py", "baseline.py", "data.py", "metrics.py", "runtime.py", "sweep_analysis.py")}}
    output = Path(output)
    run = initialize_run(output, context)
    records = read_control_records(output, run)
    saved = {(r["group"], r["id"], r["method"], r["alpha"]) for r in records}
    planned = planned_jobs(context)
    pending = [j for j in planned if (j["group"], j["id"], j["method"], j["alpha"]) not in saved]
    if max_new is not None:
        pending = pending[:max_new]
    from .robustness_report import robustness_report
    if not pending:
        return robustness_report(output, make_plots=make_plots)
    robustness_report(output, make_plots=False)
    by_id = {(e["group"], e["id"]): e for e in examples}
    meta = {(e["group"], e["id"]): e for e in metadata}
    specs = {c["method"]: c for c in control_conditions(context)}
    guard = model_guard(model)
    started = time.time()
    segment = {"run_fingerprint": run["fingerprint"], "started_unix": started, "new_records": 0,
               "completed_before": len(records) - len(context["primary_records"]), "status": "running"}
    phase = "preflight"
    interrupted = False
    try:
        checks = {"primary": evaluator_checks(evaluator), "alternative": evaluator_checks(alternative)}
        write_json(output / "evaluator_checks.json", dict(checks, run_fingerprint=run["fingerprint"]))
        base = {r["id"]: r for r in prior["baseline_records"]}
        # Reuse development questions for replay; do not select using new control outcomes.
        for role in ("forget", "retain"):
            e = next(e for e in prior["development_examples"] if e["role"] == role)
            assert_scores_match(evaluator.score([e])[0], base[e["id"]], settings, "Stage 6 development replay")
        primary_baselines = {r["id"]: r for r in prior["primary_records"] if r["method"] == "baseline"}
        for role in ("forget", "retain"):
            e = next(e for e in prior["prompt_examples"] if e["role"] == role)
            assert_scores_match(evaluator.score([e])[0], primary_baselines[e["id"]], settings, "Stage 6 imported test baseline replay")
        import torch
        if evaluator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(evaluator.device)
        for job in pending:
            phase = job["group"] + ":" + job["method"]
            key = (job["group"], job["id"])
            spec = specs[job["method"]]
            with LayerIntervention(model, spec["layers"], alpha=spec["alpha"]):
                score = evaluators[job["group"]].score([by_id[key]])[0]
            if score["prompt"] != meta[key]["prompt"] or score["input_ids"] != meta[key]["input_ids"]:
                raise ValueError("Control prompt changed during execution.")
            score.pop("prompt")
            score.pop("input_ids")
            row = {**score, **{k: meta[key][k] for k in ("role", "split", "subject", "content_hash")},
                   **job, "run_fingerprint": run["fingerprint"]}
            validate_scores(row, meta[key])
            row["record_hash"] = digest(row)
            path = output / "records" / (digest(job) + ".json")
            if path.exists():
                raise ValueError("Refusing to replace a saved control prediction.")
            write_json(path, row)
            segment["new_records"] += 1
            if segment["new_records"] % 128 == 0 or segment["new_records"] == len(pending):
                print("Stage 6: saved {}/{} new predictions ({}).".format(
                    segment["completed_before"] + segment["new_records"], len(planned), phase), flush=True)
        segment["status"] = "complete"
        segment["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(evaluator.device) if evaluator.device.type == "cuda" else None
    except KeyboardInterrupt:
        interrupted = True
        segment["status"] = "interrupted"
        segment["note"] = "Stopped by the user; verified saved predictions can resume."
    except Exception as exc:
        segment["status"] = "failed"
        segment["failure"] = {"phase": phase, "exception_type": type(exc).__name__,
                              "note": "Saved predictions are preserved. Review this error before resuming."}
    finally:
        segment["cleanup_passed"] = model_guard(model) == guard and all(p.grad is None for p in model.parameters())
        if not segment["cleanup_passed"]:
            segment["status"] = "failed"
            segment["failure"] = {"phase": "cleanup", "note": "Model parameters, gradients, or hooks changed."}
        segment["elapsed_seconds"] = time.time() - started
        write_json(output / "segments" / ("{:.6f}-{}.json".format(started, uuid.uuid4().hex[:8])), segment)
    return robustness_report(output, make_plots=make_plots and not interrupted)
