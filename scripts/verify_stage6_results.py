"""Check downloaded Stage 6 evidence against frozen inputs without GPU work."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from unlearning.baseline import read_settings
from unlearning.data import file_digest, read_config, write_json
from unlearning.result_archives import restore_stage6_results
from unlearning.robustness import read_control_records, planned_jobs
from unlearning.robustness_analysis import analyze_controls
from unlearning.robustness_inputs import read_robustness_settings, robustness_inputs
from verify_stage5_results import compare, require


def verify(archive_path, output):
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    config = read_config(ROOT / "configs/experiment.json")
    settings = read_settings(ROOT / "configs/baseline_prefill.json")
    protocol = read_robustness_settings(ROOT / "configs/robustness.json")
    prior = robustness_inputs(config, settings, protocol, ROOT / "data/prepared/full",
        ROOT / "outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill",
        ROOT / "outputs/stage4/verified_2026-09-20/outputs/stage4/full",
        ROOT / "outputs/stage5/verified_2026-09-21/original_results.zip")
    payload = archive_path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    restore_stage6_results(payload, output)
    folder = output / "outputs/stage6/full"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    context = run["context"]
    require(context["stage"] == 6 and not context["is_test_fixture"], "Expected real Stage 6 evidence.")
    for key, expected in (("config", config), ("settings", settings), ("protocol", protocol),
        ("alternative_settings", dict(settings, instruction=protocol["alternative_instruction"])),
        ("decision", prior["decision"]), ("stage5_archive_sha256", prior["archive_sha256"]),
        ("stage5_test_fingerprint", prior["runs"]["test"]["fingerprint"]),
        ("selections", prior["runs"]["test"]["context"]["selections"]),
        ("primary_examples", prior["primary_examples"]), ("primary_records", prior["primary_records"])):
        compare(context[key], expected, "context." + key)
    with zipfile.ZipFile(archive_path) as archive:
        for name, checksum_source in context["source_hashes"].items():
            require(file_digest(ROOT / "src/unlearning" / name) == checksum_source, "Local source mismatch: " + name)
            require(hashlib.sha256(archive.read("src/unlearning/" + name)).hexdigest() == checksum_source, "Archived source mismatch: " + name)
    def user_text(e, instruction):
        return instruction + "\n\n" + e["question"] + "\n" + "\n".join(
            "{}. {}".format(label, choice) for label, choice in zip("ABCD", e["choices"]))
    first = prior["prompt_examples"][0]
    primary_by_id = {e["id"]: e for e in prior["primary_examples"]}
    wrapper = primary_by_id[first["id"]]["prompt"].split(user_text(first, settings["instruction"]))
    require(len(wrapper) == 2, "Cannot identify original chat wrapper.")
    expected_examples = [dict(e, group="biology") for e in prior["biology_examples"]] + [dict(e, group="alternative") for e in prior["prompt_examples"]]
    require(len(expected_examples) == len(context["examples"]) == 320, "Wrong question count.")
    metadata = {(e["group"], e["id"]): e for e in context["examples"]}
    require(set(metadata) == {(e["group"], e["id"]) for e in expected_examples}, "Question IDs differ from fixed sampling.")
    for e in expected_examples:
        m = metadata[(e["group"], e["id"])]
        for key in ("group", "id", "role", "split", "subject", "content_hash"):
            compare(m[key], e[key], "question." + key)
        require(m["correct_answer"] == "ABCD"[e["answer"]], "Incorrect answer key.")
        instruction = settings["instruction"] if e["group"] == "biology" else protocol["alternative_instruction"]
        require(m["prompt"] == wrapper[0] + user_text(e, instruction) + wrapper[1], "Unexpected control prompt text.")
        require(0 < m["input_tokens"] <= settings["max_input_tokens"], "Invalid prompt length.")
    records = read_control_records(folder, run)
    require(len(records) == 5184 and len(planned_jobs(context)) == 2880, "Incomplete control records.")
    analysis = analyze_controls(context, records)
    saved = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    for key, value in analysis.items():
        compare(saved[key], value, "analysis." + key)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    require(summary["status"] == "complete" and summary["is_research_result"] and saved["is_research_result"], "Incomplete or synthetic report.")
    require(summary["run_fingerprint"] == saved["run_fingerprint"] == run["fingerprint"], "Report run mismatch.")
    require(summary["new_predictions_saved"] == summary["expected_new_predictions"] == 2880 and summary["imported_primary_predictions"] == 2304, "Reported counts differ.")
    compare(summary["counts"], {"biology": {"saved": 576, "expected": 576}, "alternative": {"saved": 2304, "expected": 2304}}, "counts")
    for key in ("decision",):
        compare(summary[key], context[key], "summary." + key)
        compare(saved[key], context[key], "analysis." + key)
    compare(summary["baseline_diagnostics"], analysis["baseline_diagnostics"], "summary.baselines")
    compare(summary["selective_results"], [r for r in analysis["conditions"] if r["method"] == "top_selective"], "summary.selective")
    compare(summary["selective_prompt_comparison"], [r for r in analysis["matched_prompt_comparison"] if r["method"] == "top_selective"], "summary.prompt")
    segments = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder / "segments").glob("*.json"))]
    require(segments and all(s["run_fingerprint"] == run["fingerprint"] for s in segments), "Invalid segment identity.")
    require(segments[-1]["status"] == "complete" and segments[-1]["cleanup_passed"], "Last segment not cleanly complete.")
    compare(summary["last_attempt"], segments[-1], "last_attempt")
    original = output / "original_results.zip"
    require(not original.exists() or file_digest(original) == checksum, "Different original archive exists.")
    if not original.exists():
        shutil.copyfile(archive_path, original)
    result = {"status": "verified", "archive": str(archive_path), "archive_sha256": checksum,
        "run_fingerprint": run["fingerprint"], "verified_new_predictions": 2880, "verified_primary_references": 2304,
        "decision_fingerprint": context["decision"]["decision_fingerprint"], "alpha": context["decision"]["alpha"],
        "checks": ["Stage 5 evidence and frozen development-only decision", "source/model/configuration identity",
            "outcome-independent subject-balanced subset", "biology/alternative question IDs, answer keys, prompt text and token hashes",
            "raw prediction checksums, logits, derived accuracy and scores", "complete unique conditions and clean final segment",
            "recomputed all paired statistics, letter diagnostics, and matched-prompt comparisons"],
        "notes": ["No weights loaded or GPU work repeated.", "Token hashes checked; tokenizer not independently rerun.",
            "Archived executable source preserved as evidence, not installed or executed.", "Original reports and figures preserved unchanged."],
        "selective_results": summary["selective_results"], "selective_prompt_comparison": summary["selective_prompt_comparison"],
        "environment": context["environment"], "segments": segments}
    write_json(output / "verification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "archive_sha256", "run_fingerprint", "verified_new_predictions", "verified_primary_references", "alpha")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.archive, args.output)
