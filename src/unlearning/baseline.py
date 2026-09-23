"""Development-only baseline, atomic checkpoints, and offline result review."""

import csv
import json
import math
import os
from pathlib import Path
import statistics

from .access import failure_details
from .data import digest, file_digest, verify_prepared, write_json
from .evaluation import (LABELS, MCQEvaluator, evaluator_checks, gradient_timing_probe,
                         summarize_logits)
from .metrics import accuracy_summary, ability_gate, runtime_forecast
from .runtime import environment_info


def read_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if settings["schema_version"] != 1 or settings["prompt_version"] not in ("mcq_v1", "mcq_prefill_v1"):
        raise ValueError("Unsupported baseline settings.")
    expected_prefixes = ("The correct answer is", " ") if settings["prompt_version"] == "mcq_prefill_v1" else ("", "")
    if (settings.get("assistant_prefix", ""), settings.get("answer_token_prefix", "")) != expected_prefixes:
        raise ValueError("Prompt prefix fields do not match the recorded prompt version.")
    for key in ("batch_size", "max_input_tokens"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError("{} must be a positive integer.".format(key))
    if settings["confidence_level"] != 0.95:
        raise ValueError("Only the prespecified 95% interval is implemented.")
    for key in ("instruction", "date_string"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError("A fixed {} is required.".format(key))
    for key in ("score_atol", "score_rtol", "gpu_hour_budget", "forecast_overhead_factor"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("{} must be positive and finite.".format(key))
    for key in ("forget_accuracy_min", "retain_accuracy_min", "chance_accuracy"):
        if not 0 <= settings[key] <= 1:
            raise ValueError("Invalid accuracy threshold.")
    return settings


def development_examples(data_dir, config):
    manifest = verify_prepared(data_dir)
    if manifest["config"] != config:
        raise ValueError("Prepared-data configuration differs from this experiment.")
    by_role = {}
    for role in ("forget", "retain"):
        path = Path(data_dir) / role / "development.jsonl"
        by_role[role] = [dict(json.loads(line), role=role, split="development")
                         for line in path.read_text(encoding="utf-8").splitlines()]
    # Interleave domains so an interrupted segment covers both.
    tasks = [row for pair in zip(by_role["forget"], by_role["retain"]) for row in pair]
    return manifest, tasks


def source_hash():
    root = Path(__file__).parent
    return digest({name: file_digest(root / name) for name in (
        "baseline.py", "evaluation.py", "metrics.py", "data.py", "runtime.py", "access.py")})


def initialize_run(output, context):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run.json"
    run = {"fingerprint": digest(context), "context": context}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != run:
            raise ValueError("Existing run has different data, code, model, settings, or environment. Use a new output directory.")
    else:
        if any(output.iterdir()):
            raise ValueError("Output is nonempty without a run manifest. Use a new directory.")
        write_json(path, run)
    return run


def checkpoint_path(output, example_id):
    return Path(output) / "predictions" / (digest(example_id) + ".json")


def equivalent_derived(left, right):
    """Allow last-bit libm differences across OS/Python, not changed predictions."""
    if isinstance(right, dict):
        return isinstance(left, dict) and left.keys() == right.keys() and all(
            equivalent_derived(left[key], value) for key, value in right.items())
    if isinstance(right, float):
        return type(left) in (float, int) and math.isfinite(left) and math.isclose(
            left, right, rel_tol=1e-12, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def save_prediction(output, run, example, scores):
    row = dict(scores, id=example["id"], content_hash=example["content_hash"],
               role=example["role"], split="development", subject=example["subject"],
               run_fingerprint=run["fingerprint"])
    row["record_hash"] = digest(row)
    path = checkpoint_path(output, row["id"])
    if path.exists():
        raise ValueError("Refusing to replace an existing prediction.")
    write_json(path, row)


def read_predictions(output, run):
    if run["fingerprint"] != digest(run["context"]):
        raise ValueError("Run manifest fingerprint mismatch.")
    expected = {row["id"]: row for row in run["context"]["examples"]}
    results = {}
    for path in sorted((Path(output) / "predictions").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        checksum = row.pop("record_hash")
        if checksum != digest(row) or row["run_fingerprint"] != run["fingerprint"]:
            raise ValueError("Prediction checksum or run fingerprint mismatch.")
        identity = row["id"]
        if identity not in expected or identity in results or path != checkpoint_path(output, identity):
            raise ValueError("Unknown or duplicated prediction.")
        metadata = expected[identity]
        for key in ("content_hash", "role", "subject", "correct_answer", "prompt_hash", "input_tokens"):
            if row[key] != metadata[key]:
                raise ValueError("Prediction metadata mismatch: " + key)
        if row["split"] != "development" or digest(row["prompt"]) != row["prompt_hash"]:
            raise ValueError("Prediction prompt or split mismatch.")
        if digest(row["input_ids"]) != metadata["input_ids_hash"] or len(row["input_ids"]) != row["input_tokens"]:
            raise ValueError("Saved input tokens changed.")
        recomputed = summarize_logits([row["answer_logits"][label] for label in LABELS], LABELS.index(row["correct_answer"]))
        if any(not equivalent_derived(row[key], value) for key, value in recomputed.items()):
            raise ValueError("Saved prediction is inconsistent with its logits.")
        if not math.isfinite(row["seconds_per_example"]) or row["seconds_per_example"] <= 0:
            raise ValueError("Invalid timing measurement.")
        if not 0 <= row["answer_probability_mass"] <= 1.00001:
            raise ValueError("Invalid answer probability mass.")
        row["record_hash"] = checksum
        results[identity] = row
    return [results[row["id"]] for row in run["context"]["examples"] if row["id"] in results]


def baseline_report(output):
    output = Path(output)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    records = read_predictions(output, run)
    context = run["context"]
    complete = len(records) == len(context["examples"])
    report = {"status": "complete" if complete else "partial", "stage": 2,
              "split": "development", "profile": context["profile"],
              "run_fingerprint": run["fingerprint"], "evaluated": len(records),
              "expected": len(context["examples"]), "is_research_result": True,
              "final_test_evaluated": False}
    if complete:
        segments = [json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted((output / "segments").glob("*.json"))]
        if any(row["run_fingerprint"] != run["fingerprint"] for row in segments):
            raise ValueError("Timing segment belongs to another run.")
        peaks = [row["peak_gpu_allocated_bytes"] for row in segments
                 if row["peak_gpu_allocated_bytes"] is not None]
        lengths = [row["input_tokens"] for row in records]
        report["measured_runtime"] = {
            "mean_seconds_per_example": statistics.mean(row["seconds_per_example"] for row in records),
            "peak_inference_gpu_allocated_bytes": max(peaks) if peaks else None,
            "input_tokens_min": min(lengths), "input_tokens_mean": statistics.mean(lengths),
            "input_tokens_max": max(lengths),
            "note": "Forward timing includes prompt preparation and excludes model loading and synthetic warm-up. Segment peaks may be incomplete after a hard runtime interruption."}
        by_role = {role: accuracy_summary([r for r in records if r["role"] == role])
                   for role in ("forget", "retain")}
        report["datasets"] = by_role
        report["retain_subjects"] = {subject: accuracy_summary([r for r in records if r["subject"] == subject])
                                     for subject in context["config"]["datasets"]["retain"]["configs"]}
        report["ability_gate"] = ability_gate(by_role, context["settings"])
        probe_path = output / "runtime_probe.json"
        if probe_path.exists():
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            if probe["run_fingerprint"] != run["fingerprint"]:
                raise ValueError("Runtime probe belongs to another run.")
            if probe["status"] == "passed":
                report["runtime_forecast"] = runtime_forecast(context["config"], records, probe, context["settings"], context["layer_count"])
            else:
                report["runtime_forecast"] = {"status": "review_required", "note": probe["note"]}
        else:
            report["runtime_forecast"] = {"status": "pending", "note": "Rerun baseline to collect the runtime probe."}
        report["next_action"] = ("Review the development results and provisional runtime forecast before Stage 3."
                                 if report["ability_gate"]["status"] == "passed" else report["ability_gate"]["next_action"])
        with (output / "accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", "split", "count", "correct", "accuracy", "wilson_95_low", "wilson_95_high"])
            for role, result in by_role.items():
                writer.writerow([role, "development", result["count"], result["correct"], result["accuracy"], *result["accuracy_interval_95"]])
    else:
        report["next_action"] = "Resume the same command. Accuracy gates are applied only after the entire development split is complete."
    write_json(output / "summary.json", report)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return report


def load_evaluator(config, settings):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline requires the Colab GPU. No research-model download was attempted.")
    if config["model"]["dtype"] != "float32":
        raise ValueError("Stage 2 currently requires float32, as specified in the plan.")
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set HF_TOKEN privately in this notebook before running the baseline.")
    shared = {"revision": config["model"]["revision"], "token": token}
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], **shared)
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["id"], **shared, torch_dtype=torch.float32,
            device_map={"": "cuda:0"}, use_safetensors=True,
            attn_implementation=settings["attention_implementation"])
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("GPU memory exhausted while loading. Close other model workloads and retry.") from None
    except Exception as exc:
        raise RuntimeError(failure_details(exc, "baseline_model_loading")["error"]) from None
    return MCQEvaluator(model, tokenizer, settings)


def run_baseline(config, settings, data_dir, output, max_new=None, evaluator=None):
    if max_new is not None and (type(max_new) is not int or max_new <= 0):
        raise ValueError("max-new must be a positive integer.")
    manifest, examples = development_examples(data_dir, config)
    evaluator = evaluator if evaluator is not None else load_evaluator(config, settings)
    prepared = [evaluator.prepare(row) for row in examples]
    metadata = [dict(id=row["id"], content_hash=row["content_hash"], role=row["role"],
                     subject=row["subject"], correct_answer=LABELS[row["answer"]],
                     prompt_hash=item["prompt_hash"], input_tokens=len(item["input_ids"]),
                     input_ids_hash=digest(item["input_ids"])) for row, item in zip(examples, prepared)]
    context = {"schema_version": 1, "config": config, "settings": settings,
               "data_request_hash": manifest["request_hash"], "profile": manifest["profile"],
               "source_hash": source_hash(), "environment": environment_info(),
               "device": str(evaluator.device), "dtype": str(next(evaluator.model.parameters()).dtype),
               "layer_count": len(evaluator.model.model.layers),
               "chat_template": evaluator.tokenizer.get_chat_template(),
               "label_token_ids": dict(zip(LABELS, evaluator.label_ids)),
               "scoring": "last real input position; log-softmax across A-D; first label wins exact ties",
               "examples": metadata}
    output = Path(output)
    run = initialize_run(output, context)
    records = read_predictions(output, run)
    checks = evaluator_checks(evaluator)  # Also warms up the model on harmless questions.
    write_json(output / "evaluator_checks.json", dict(checks, run_fingerprint=run["fingerprint"]))
    done = {row["id"] for row in records}
    pending = [row for row in examples if row["id"] not in done]
    if max_new is not None:
        pending = pending[:max_new]
    import torch
    if evaluator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluator.device)
    try:
        for start in range(0, len(pending), settings["batch_size"]):
            batch = pending[start:start + settings["batch_size"]]
            for example, scores in zip(batch, evaluator.score(batch)):
                save_prediction(output, run, example, scores)
                done.add(example["id"])
            if len(done) % 16 == 0 or start + len(batch) == len(pending):
                print("Saved {}/{} development predictions.".format(len(done), len(examples)), flush=True)
        if pending:
            # Keep each segment's peak rather than replacing it on a resumed run.
            write_json(output / "segments" / ("{:06d}.json".format(len(done))), {
                "run_fingerprint": run["fingerprint"], "new_predictions": len(pending),
                "completed_predictions": len(done),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(evaluator.device)
                if evaluator.device.type == "cuda" else None})
        # Probe representative median and longest prompts per domain, never test questions.
        probe_path = output / "runtime_probe.json"
        if len(done) == len(examples) and not probe_path.exists():
            lengths = {row["id"]: len(item["input_ids"]) for row, item in zip(examples, prepared)}
            sample = []
            for role in ("forget", "retain"):
                rows = sorted((row for row in examples if row["role"] == role), key=lambda row: lengths[row["id"]])
                sample.extend([rows[len(rows) // 2], rows[-1]])
            try:
                probe = gradient_timing_probe(evaluator, sample)
            except RuntimeError:
                # Keep completed accuracy results even if the more expensive probe fails.
                probe = {"status": "review_required", "note": "The input-gradient timing probe failed (possibly GPU memory). Baseline predictions are preserved. Review before Stage 3."}
            write_json(probe_path, dict(probe, run_fingerprint=run["fingerprint"]))
    finally:
        # Atomic per-question files are authoritative even if summary export is interrupted.
        report = baseline_report(output)
    return report
