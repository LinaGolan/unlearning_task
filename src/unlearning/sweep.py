"""Frozen-layer sweeps with a development decision required before test scoring."""

import json
import math
from pathlib import Path
import time
import uuid

from .baseline import equivalent_derived, initialize_run, load_evaluator
from .data import digest, file_digest, write_json
from .evaluation import LABELS, evaluator_checks, summarize_logits
from .intervention_checks import assert_scores_match, model_guard, verified_inputs
from .interventions import LayerIntervention, decoder_layers
from .runtime import environment_info
from .sweep_analysis import conditions, choose_operating_point


def read_sweep_settings(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "retain_drop_limit_pp": 5, "fallback_display_strength": .5,
                "confidence_level": .95, "bootstrap_rule": "paired_question_ids_stratified_by_subject", "tie_rule": "weaker_strength"}
    if any(value[k] != v for k, v in expected.items()):
        raise ValueError("Sweep decision/uncertainty rules differ from the declared protocol.")
    if type(value["bootstrap_repetitions"]) is not int or value["bootstrap_repetitions"] < 100:
        raise ValueError("At least 100 bootstrap repetitions are required (research default: 2000).")
    if type(value["bootstrap_seed"]) is not int or not 0 <= value["bootstrap_seed"] < 2**32 - 1:
        raise ValueError("Invalid bootstrap seed.")
    return value


def verified_stage4(stage4_dir, config, settings, baseline_run):
    from .localization import read_records as read_localization_records, planned_jobs as localization_jobs
    from .localization_analysis import localization_summary
    folder = Path(stage4_dir)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    context = run["context"]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((folder / "selections.json").read_text(encoding="utf-8"))
    checksum = selection.pop("selection_fingerprint")
    if (digest(selection) != checksum or checksum != summary["selection_fingerprint"]
            or selection["run_fingerprint"] != run["fingerprint"] or summary["run_fingerprint"] != run["fingerprint"]
            or selection["model"] != config["model"] or selection["settings"] != settings):
        raise ValueError("Stage 4 selection identity mismatch.")
    if context["config"] != config or context["settings"] != settings or context["baseline_fingerprint"] != baseline_run["fingerprint"]:
        raise ValueError("Stage 4 differs from the reviewed model/data/prompt baseline.")
    if (summary["status"] != "complete" or not summary["control_a_complete"] or not summary["last_attempt"]["cleanup_passed"]
            or summary["last_attempt"]["status"] != "complete" or summary["final_test_evaluated"]):
        raise ValueError("Stage 4 must be complete and reviewed before the sweep.")
    for name in ("interventions.py", "evaluation.py", "data.py", "baseline.py"):
        if context["source_hashes"][name] != file_digest(Path(__file__).parent / name):
            raise ValueError("Validated model/evaluator code changed: " + name)
    records = read_localization_records(folder, run)
    if len(records) != len(localization_jobs(context)):
        raise ValueError("Stage 4 measurement records are incomplete.")
    localization = [r for r in records if r["kind"] == "localization"]
    result = localization_summary(localization, context["localization_examples"], config, context["protocol"], context["layer_count"])
    if selection["selections"] != result["selections"] or selection["localization_record_hashes"] != [r["record_hash"] for r in localization]:
        raise ValueError("Frozen layer pairs do not match the localization evidence.")
    selection["selection_fingerprint"] = checksum
    return run, selection


def sweep_inputs(config, settings, data_dir, baseline_dir, stage4_dir):
    manifest, development, baseline, records = verified_inputs(config, settings, data_dir, baseline_dir)
    if manifest["profile"] != "full":
        raise ValueError("Stage 5 uses the previously approved full profile.")
    stage4, selection = verified_stage4(stage4_dir, config, settings, baseline)
    return manifest, development, baseline, records, stage4, selection


def load_test_examples(data_dir):
    """Only call after verifying the complete frozen development decision."""
    pools = []
    for role in ("forget", "retain"):
        pools.append([dict(json.loads(line), role=role, split="test") for line in
            (Path(data_dir) / role / "test.jsonl").read_text(encoding="utf-8").splitlines()])
    return [r for pair in zip(*pools) for r in pair]


def jobs(context):
    specs = conditions(context)
    if context["split"] == "development":
        specs = specs[1:]  # The unchanged development predictions already exist.
    return [{"id": e["id"], "method": c["method"], "alpha": c["alpha"]} for c in specs for e in context["examples"]]


def path_for(output, job):
    return Path(output) / "records" / (digest(job) + ".json")


def validate_scores(row, metadata):
    for key in ("id", "role", "split", "subject", "content_hash", "correct_answer", "prompt_hash", "input_tokens"):
        if row[key] != metadata[key]:
            raise ValueError("Sweep prediction metadata mismatch: " + key)
    derived = summarize_logits([row["answer_logits"][k] for k in LABELS], LABELS.index(row["correct_answer"]))
    if any(not equivalent_derived(row[k], v) for k, v in derived.items()):
        raise ValueError("Sweep prediction disagrees with its logits.")
    if not math.isfinite(row["seconds_per_example"]) or row["seconds_per_example"] <= 0 or not 0 <= row["answer_probability_mass"] <= 1.00001:
        raise ValueError("Invalid sweep timing or answer mass.")


def read_sweep_records(output, run):
    context = run["context"]
    if digest(context) != run["fingerprint"]:
        raise ValueError("Sweep run fingerprint mismatch.")
    metadata = {e["id"]: e for e in context["examples"]}
    for e in metadata.values():
        if digest(e["prompt"]) != e["prompt_hash"] or digest(e["input_ids"]) != e["input_ids_hash"] or len(e["input_ids"]) != e["input_tokens"]:
            raise ValueError("Sweep input prompt/tokens changed.")
    imported = []
    if context["split"] == "development":
        for original in context["imported_baselines"]:
            row = dict(original)
            checksum = row.pop("record_hash")
            if digest(row) != checksum or row["run_fingerprint"] != context["baseline_fingerprint"]:
                raise ValueError("Imported baseline checksum mismatch.")
            validate_scores(row, metadata[row["id"]])
            if row["input_ids"] != metadata[row["id"]]["input_ids"] or row["prompt"] != metadata[row["id"]]["prompt"]:
                raise ValueError("Imported baseline prompt differs from the sweep.")
            imported.append(dict(original, method="baseline", alpha=0.0))
        if len(imported) != len(metadata) or {r["id"] for r in imported} != set(metadata):
            raise ValueError("Imported development baseline is incomplete.")
    expected = {digest(job): job for job in jobs(context)}
    saved = {}
    for path in sorted((Path(output) / "records").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        checksum = row.pop("record_hash")
        if digest(row) != checksum or row["run_fingerprint"] != run["fingerprint"] or path.stem not in expected:
            raise ValueError("Sweep record checksum, run identity, or filename mismatch.")
        if any(row[k] != v for k, v in expected[path.stem].items()):
            raise ValueError("Sweep condition differs from its filename.")
        validate_scores(row, metadata[row["id"]])
        row["record_hash"] = checksum
        saved[path.stem] = row
    return imported + [saved[digest(job)] for job in jobs(context) if digest(job) in saved]


def decision_payload(run, records):
    context = run["context"]
    result = choose_operating_point(context, records)
    result.update(schema_version=1, development_run_fingerprint=run["fingerprint"],
                  selection_fingerprint=context["selection_fingerprint"],
                  development_records_digest=digest([r["record_hash"] for r in records]),
                  model=context["config"]["model"], settings=context["settings"], protocol=context["protocol"])
    result["decision_fingerprint"] = digest(result)
    return result


def verified_decision(development_dir):
    folder = Path(development_dir)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    records = read_sweep_records(folder, run)
    expected = decision_payload(run, records)  # Also rejects partial or test runs.
    actual = json.loads((folder / "operating_point.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("The frozen operating strength differs from the complete development evidence.")
    segments = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder / "segments").glob("*.json"))]
    if not segments or segments[-1]["status"] != "complete" or not segments[-1]["cleanup_passed"]:
        raise ValueError("Development cleanup must pass before final-test scoring.")
    if any(s["run_fingerprint"] != run["fingerprint"] for s in segments):
        raise ValueError("Development segment identity mismatch.")
    return run, actual


def run_sweep(config, settings, protocol, data_dir, baseline_dir, stage4_dir, output,
              split="development", development_dir=None, max_new=None, evaluator=None, make_plots=True):
    if split not in ("development", "test") or (max_new is not None and (type(max_new) is not int or max_new <= 0)):
        raise ValueError("Invalid sweep split or segment size.")
    decision = None
    if split == "test":
        if development_dir is None:
            raise ValueError("The completed development run is required before test evaluation.")
        development_run, decision = verified_decision(development_dir)
    manifest, development, baseline, base_records, stage4, selection = sweep_inputs(config, settings, data_dir, baseline_dir, stage4_dir)
    if decision is not None:
        parent = development_run["context"]
        if (parent["config"] != config or parent["settings"] != settings or parent["protocol"] != protocol
                or parent["selection_fingerprint"] != selection["selection_fingerprint"]):
            raise ValueError("Final-test settings differ from the frozen development run.")
    examples = development if split == "development" else load_test_examples(data_dir)
    is_test_fixture = evaluator is not None
    if stage4["context"]["is_test_fixture"] != is_test_fixture or (decision is not None and parent["is_test_fixture"] != is_test_fixture):
        raise ValueError("Synthetic test evidence and real research runs cannot be mixed.")
    evaluator = evaluator if evaluator is not None else load_evaluator(config, settings)
    model = evaluator.model
    old = baseline["context"]
    if (len(decoder_layers(model)) != old["layer_count"] or evaluator.settings != settings
            or str(next(model.parameters()).dtype) != old["dtype"]
            or evaluator.tokenizer.get_chat_template() != old["chat_template"]
            or dict(zip(LABELS, evaluator.label_ids)) != old["label_token_ids"]):
        raise ValueError("Sweep evaluator differs from the reviewed baseline.")
    prepared = {e["id"]: evaluator.prepare(e) for e in examples}
    metadata = [{**{k: e[k] for k in ("id", "role", "subject", "content_hash", "split")},
                 **prepared[e["id"]], "correct_answer": LABELS[e["answer"]],
                 "input_ids_hash": digest(prepared[e["id"]]["input_ids"]),
                 "input_tokens": len(prepared[e["id"]]["input_ids"])} for e in examples]
    source = Path(__file__).parent
    context = {"stage": 5, "schema_version": 1, "split": split, "profile": manifest["profile"],
        "config": config, "settings": settings, "protocol": protocol, "environment": environment_info(),
        "is_test_fixture": is_test_fixture, "selection_fingerprint": selection["selection_fingerprint"],
        "selections": selection["selections"], "stage4_fingerprint": stage4["fingerprint"],
        "baseline_fingerprint": baseline["fingerprint"], "examples": metadata,
        "imported_baselines": base_records if split == "development" else [], "frozen_decision": decision,
        "source_hashes": {name: file_digest(source / name) for name in ("sweep.py", "sweep_analysis.py", "sweep_report.py",
            "interventions.py", "evaluation.py", "baseline.py", "data.py", "metrics.py", "runtime.py")}}
    output = Path(output)
    run = initialize_run(output, context)
    records = read_sweep_records(output, run)
    done = {(r["id"], r["method"], r["alpha"]) for r in records}
    pending = [j for j in jobs(context) if (j["id"], j["method"], j["alpha"]) not in done]
    if max_new is not None:
        pending = pending[:max_new]
    from .sweep_report import sweep_report
    if not pending:
        return sweep_report(output, make_plots=make_plots)
    sweep_report(output, make_plots=False)
    example_map = {e["id"]: e for e in examples}
    meta_map = {e["id"]: e for e in metadata}
    specs = {(c["method"], c["alpha"]): c for c in conditions(context)}
    guard = model_guard(model)
    started = time.time()
    segment = {"run_fingerprint": run["fingerprint"], "started_unix": started, "new_records": 0,
               "completed_before": len(records), "status": "running"}
    phase = "preflight"
    try:
        checks = evaluator_checks(evaluator)
        write_json(output / "evaluator_checks.json", dict(checks, run_fingerprint=run["fingerprint"]))
        bases = {r["id"]: r for r in base_records}
        # Use development replays even when executing the final-test phase.
        for role in ("forget", "retain"):
            e = next(e for e in development if e["role"] == role)
            assert_scores_match(evaluator.score([e])[0], bases[e["id"]], settings, "Sweep preflight baseline")
        import torch
        if evaluator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(evaluator.device)
        for job in pending:
            phase = job["method"] + "_" + str(job["alpha"])
            spec = specs[(job["method"], job["alpha"])]
            example = example_map[job["id"]]
            with LayerIntervention(model, spec["layers"], alpha=spec["alpha"]):
                score = evaluator.score([example])[0]
            meta = meta_map[example["id"]]
            if score["input_ids"] != meta["input_ids"] or score["prompt_hash"] != meta["prompt_hash"]:
                raise ValueError("Prompt changed during the sweep.")
            # Keep prompt/tokens once per question in run.json rather than 33 times.
            score.pop("prompt")
            score.pop("input_ids")
            row = {**score, **{k: meta[k] for k in ("role", "subject", "split", "content_hash")},
                   **job, "run_fingerprint": run["fingerprint"]}
            validate_scores(row, meta)
            row["record_hash"] = digest(row)
            path = path_for(output, job)
            if path.exists():
                raise ValueError("Refusing to replace a saved sweep prediction.")
            write_json(path, row)
            segment["new_records"] += 1
            if segment["new_records"] % 256 == 0 or segment["new_records"] == len(pending):
                print("Stage 5 {}: saved {}/{} predictions ({}).".format(split, len(records) + segment["new_records"],
                    len(examples) * len(conditions(context)), phase), flush=True)
        segment["status"] = "complete"
        segment["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(evaluator.device) if evaluator.device.type == "cuda" else None
    except Exception as exc:
        segment["status"] = "failed"
        segment["failure"] = {"phase": phase, "exception_type": type(exc).__name__,
                              "note": "Measurements stopped; saved predictions are preserved. Review this failure before resuming."}
    finally:
        segment["cleanup_passed"] = model_guard(model) == guard and all(p.grad is None for p in model.parameters())
        if not segment["cleanup_passed"]:
            segment["status"] = "failed"
            segment["failure"] = {"phase": "cleanup", "note": "Model state or hooks changed; review before continuing."}
        segment["elapsed_seconds"] = time.time() - started
        write_json(output / "segments" / ("{:.6f}-{}.json".format(started, uuid.uuid4().hex[:8])), segment)
    return sweep_report(output, make_plots=make_plots)
