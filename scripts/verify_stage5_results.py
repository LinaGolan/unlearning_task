"""Verify a downloaded Stage 5 ZIP offline without loading model weights.

Keep the original archive intact; unpack reports/figures for convenient review.
Raw records remain in the preserved ZIP to avoid thousands of duplicate files.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from unlearning.baseline import read_settings
from unlearning.data import digest, file_digest, read_config, write_json
from unlearning.sweep import (decision_payload, jobs, load_test_examples, read_sweep_settings,
                              sweep_inputs, validate_scores)
from unlearning.sweep_analysis import conditions, summarize_sweep
from unlearning.sweep_report import comparison_rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, location="root"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Keys differ: " + location)
        for key in expected:
            compare(actual[key], expected[key], location + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "Length differs: " + location)
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, location + "[{}]".format(index))
    elif isinstance(expected, float):
        require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), "Number differs: " + location)
    else:
        require(actual == expected, "Value differs: " + location)


def verify(archive_path, output):
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    config = read_config(ROOT / "configs/experiment.json")
    settings = read_settings(ROOT / "configs/baseline_prefill.json")
    protocol = read_sweep_settings(ROOT / "configs/sweep.json")
    data = ROOT / "data/prepared/full"
    baseline_dir = ROOT / "outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill"
    stage4_dir = ROOT / "outputs/stage4/verified_2026-09-20/outputs/stage4/full"
    _, development, baseline, base_records, stage4, selection = sweep_inputs(config, settings, data, baseline_dir, stage4_dir)
    base_map = {r["id"]: r for r in base_records}
    def user_text(example):
        return settings["instruction"] + "\n\n" + example["question"] + "\n" + "\n".join(
            "{}. {}".format(label, choice) for label, choice in zip("ABCD", example["choices"]))
    # Derive the fixed chat wrapper from the already-verified development baseline.
    wrapper = base_map[development[0]["id"]]["prompt"].split(user_text(development[0]))
    require(len(wrapper) == 2, "Cannot identify the reviewed prompt wrapper.")
    report = {"status": "verified", "archive": str(archive_path), "archive_sha256": file_digest(archive_path),
              "model_loaded": False, "phases": {}, "selection_fingerprint": selection["selection_fingerprint"]}
    decision = None
    development_ids = {e["id"] for e in development}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate archive paths.")
        for name in names:
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name, "Unsafe archive path.")
        def read(name):
            return json.loads(archive.read(name))
        for phase in ("development", "test"):
            prefix = "outputs/stage5/" + phase + "/"
            run = read(prefix + "run.json")
            context = run["context"]
            require(digest(context) == run["fingerprint"], "Run fingerprint differs.")
            require(context["stage"] == 5 and context["split"] == phase and context["profile"] == "full" and not context["is_test_fixture"], "Wrong run type.")
            for key, expected in (("config", config), ("settings", settings), ("protocol", protocol),
                    ("baseline_fingerprint", baseline["fingerprint"]), ("stage4_fingerprint", stage4["fingerprint"]),
                    ("selection_fingerprint", selection["selection_fingerprint"]), ("selections", selection["selections"])):
                compare(context[key], expected, phase + "." + key)
            for name, checksum in context["source_hashes"].items():
                require(file_digest(ROOT / "src/unlearning" / name) == checksum, "Local source differs: " + name)
                require(hashlib.sha256(archive.read("src/unlearning/" + name)).hexdigest() == checksum, "Archived source differs: " + name)
            if phase == "test":
                compare(context["frozen_decision"], decision, "frozen_test_decision")
            examples = development if phase == "development" else load_test_examples(data)
            if phase == "test":
                require(not development_ids.intersection(e["id"] for e in examples), "Development/test overlap.")
            require(len(context["examples"]) == len(examples), "Question count differs.")
            metadata = {e["id"]: e for e in context["examples"]}
            require(set(metadata) == {e["id"] for e in examples}, "Question IDs differ from prepared data.")
            for e in examples:
                m = metadata[e["id"]]
                for key in ("role", "split", "subject", "content_hash"):
                    compare(m[key], e[key], "question." + key)
                require(m["correct_answer"] == "ABCD"[e["answer"]], "Answer key differs.")
                require(m["prompt"] == wrapper[0] + user_text(e) + wrapper[1], "Question prompt differs from the reviewed format.")
                require(digest(m["prompt"]) == m["prompt_hash"] and digest(m["input_ids"]) == m["input_ids_hash"], "Input checksum differs.")
                require(len(m["input_ids"]) == m["input_tokens"] and 0 < m["input_tokens"] <= settings["max_input_tokens"], "Invalid input length.")
            records = []
            if phase == "development":
                compare(context["imported_baselines"], base_records, "imported_baselines")
                records = [dict(r, method="baseline", alpha=0.0) for r in base_records]
            else:
                require(context["imported_baselines"] == [], "Unexpected test baseline import.")
            planned = jobs(context)
            expected_paths = {prefix + "records/" + digest(j) + ".json" for j in planned}
            require(expected_paths == {n for n in names if n.startswith(prefix + "records/")}, "Missing or unexpected raw prediction.")
            for job in planned:
                row = read(prefix + "records/" + digest(job) + ".json")
                checksum = row.pop("record_hash")
                require(digest(row) == checksum and row["run_fingerprint"] == run["fingerprint"], "Prediction checksum or run mismatch.")
                for key in job:
                    compare(row[key], job[key], "job." + key)
                validate_scores(row, metadata[row["id"]])
                row["record_hash"] = checksum
                records.append(row)
            segments = [read(n) for n in sorted(names) if n.startswith(prefix + "segments/") and n.endswith(".json")]
            require(segments and segments[-1]["status"] == "complete" and segments[-1]["cleanup_passed"], "Last segment incomplete or unclean.")
            require(all(s["run_fingerprint"] == run["fingerprint"] for s in segments), "Segment identity differs.")
            if phase == "development":
                decision = decision_payload(run, records)
                compare(read(prefix + "operating_point.json"), decision, "operating_point")
                development_end = max(s["started_unix"] + s["elapsed_seconds"] for s in segments)
            else:
                require(min(s["started_unix"] for s in segments) > development_end, "Test execution preceded development completion.")
            derived = summarize_sweep(context, records, decision["alpha"])
            saved_analysis = read(prefix + "analysis.json")
            for key in derived:
                compare(saved_analysis[key], derived[key], phase + ".analysis." + key)
            compare(saved_analysis["operating_point"], decision, "analysis.decision")
            summary = read(prefix + "summary.json")
            require(summary["status"] == "complete" and summary["is_research_result"], "Summary is not complete research evidence.")
            require(summary["run_fingerprint"] == run["fingerprint"] == saved_analysis["run_fingerprint"], "Report run identity differs.")
            require(summary["completed_records"] == summary["expected_records"] == len(records), "Summary count differs.")
            require(summary["new_predictions_saved"] == len(planned), "New prediction count differs.")
            compare(summary["main_comparison"], comparison_rows(derived, decision["alpha"]), "main_comparison")
            compare(summary["operating_point"], decision, "summary.decision")
            report["phases"][phase] = {"run_fingerprint": run["fingerprint"], "verified_records": len(records),
                "new_predictions": len(planned), "counts_by_role": {role: sum(e["role"] == role for e in examples) for role in ("forget", "retain")},
                "main_comparison": summary["main_comparison"], "common_strength_contrasts": derived["common_strength_contrasts"],
                "segments": segments, "environment": context["environment"]}
            print("Verified {}: {} predictions, logits, question metadata, decision, and paired statistics.".format(phase, len(records)), flush=True)
        report["decision"] = decision
        report["new_predictions_verified"] = sum(v["new_predictions"] for v in report["phases"].values())
        report["notes"] = ["No model weights loaded or GPU work repeated.",
            "All raw records remain in original_results.zip; reports and figures are also unpacked.",
            "Input token hashes verified; tokenizer was not independently rerun.",
            "Bootstrap tables recomputed locally and matched within 1e-11 tolerance.",
            "Source files in the archive are evidence only and were not executed."]
        output.mkdir(parents=True, exist_ok=True)
        original = output / "original_results.zip"
        require(not original.exists() or file_digest(original) == report["archive_sha256"], "Different archive already stored at destination.")
        if not original.exists():
            shutil.copyfile(archive_path, original)
        for name in names:
            if name.startswith("outputs/stage5/") and "/records/" not in name and PurePosixPath(name).suffix in (".json", ".csv", ".png", ".pdf"):
                target = (output / name).resolve()
                require(output in target.parents, "Invalid extraction target.")
                content = archive.read(name)
                require(not target.exists() or target.read_bytes() == content, "Existing report differs: " + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(content)
        write_json(output / "verification.json", report)
    print(json.dumps({"status": "verified", "new_predictions": report["new_predictions_verified"],
                      "alpha": decision["alpha"], "archive_sha256": report["archive_sha256"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.archive, args.output)
