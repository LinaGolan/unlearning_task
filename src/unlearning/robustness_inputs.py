"""Read-only Stage 5 provenance and outcome-independent Stage 6 sampling."""

import json
from pathlib import Path
import zipfile

from .data import digest, file_digest
from .evaluation import LABELS
from .sweep import decision_payload, jobs as sweep_jobs, load_test_examples, sweep_inputs, validate_scores


ALTERNATIVE_INSTRUCTION = "Select the best answer to the question. Respond with just A, B, C, or D."


def read_robustness_settings(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    fixed = {"schema_version": 1, "alternative_instruction": ALTERNATIVE_INSTRUCTION,
             "prompt_subset_fraction": .5, "sampling_rule": "sha256_within_role_and_subject", "confidence_level": .95}
    if any(value[k] != v for k, v in fixed.items()):
        raise ValueError("Stage 6 protocol differs from the declared single-prompt control.")
    if type(value["bootstrap_repetitions"]) is not int or value["bootstrap_repetitions"] < 100:
        raise ValueError("At least 100 bootstrap repetitions required; research default is 2000.")
    if type(value["bootstrap_seed"]) is not int or not 0 <= value["bootstrap_seed"] < 2**32 - 10:
        raise ValueError("Invalid bootstrap seed.")
    return value


def prompt_subset(examples, config):
    if any(e["split"] != "test" or e["role"] not in ("forget", "retain") for e in examples):
        raise ValueError("Prompt robustness uses main final-test questions only.")
    selected = []
    groups = sorted({(e["role"], e["subject"]) for e in examples})
    for role, subject in groups:
        rows = [e for e in examples if e["role"] == role and e["subject"] == subject]
        if len(rows) % 2:
            raise ValueError("Expected even subject counts for a half-size robustness subset.")
        rows.sort(key=lambda e: digest([config["seed"], "stage6_prompt_robustness", role, subject, e["id"]]))
        selected.extend(rows[:len(rows) // 2])
    return sorted(selected, key=lambda e: (e["role"], e["subject"], e["id"]))


def read_stage5_archive(path, config, settings, data_dir, baseline_dir, stage4_dir):
    """Validate every raw prediction and the development decision, without extracting code."""
    _, dev_examples, baseline, base_records, stage4, selection = sweep_inputs(config, settings, data_dir, baseline_dir, stage4_dir)
    test_examples = load_test_examples(data_dir)
    runs, phases = {}, {}
    decision = None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate Stage 5 archive entries.")
        def read(name):
            return json.loads(archive.read(name))
        for split, examples in (("development", dev_examples), ("test", test_examples)):
            prefix = "outputs/stage5/" + split + "/"
            run = read(prefix + "run.json")
            c = run["context"]
            if digest(c) != run["fingerprint"] or c["split"] != split or c["profile"] != "full" or c["stage"] != 5:
                raise ValueError("Invalid Stage 5 run identity.")
            for key, value in (("config", config), ("settings", settings), ("baseline_fingerprint", baseline["fingerprint"]),
                    ("stage4_fingerprint", stage4["fingerprint"]), ("selections", selection["selections"]),
                    ("selection_fingerprint", selection["selection_fingerprint"])):
                if c[key] != value:
                    raise ValueError("Stage 5 differs from reviewed inputs: " + key)
            if c["is_test_fixture"] != stage4["context"]["is_test_fixture"]:
                raise ValueError("Mixed synthetic and research evidence.")
            for name, checksum in c["source_hashes"].items():
                if file_digest(Path(__file__).parent / name) != checksum:
                    raise ValueError("Stage 5 source changed: " + name)
            metadata = {e["id"]: e for e in c["examples"]}
            if len(metadata) != len(examples) or set(metadata) != {e["id"] for e in examples}:
                raise ValueError("Stage 5 questions differ from prepared data.")
            for e in examples:
                m = metadata[e["id"]]
                if any(m[k] != e[k] for k in ("role", "split", "subject", "content_hash")) or m["correct_answer"] != LABELS[e["answer"]]:
                    raise ValueError("Stage 5 question metadata differs.")
                if (digest(m["prompt"]) != m["prompt_hash"] or digest(m["input_ids"]) != m["input_ids_hash"]
                        or len(m["input_ids"]) != m["input_tokens"]):
                    raise ValueError("Stage 5 prompt or token checksum mismatch.")
            if c["imported_baselines"] != (base_records if split == "development" else []):
                raise ValueError("Stage 5 imported baseline differs from reviewed evidence.")
            records = [dict(r, method="baseline", alpha=0.0) for r in c["imported_baselines"]]
            planned = sweep_jobs(c)
            expected = {prefix + "records/" + digest(j) + ".json" for j in planned}
            if expected != {n for n in names if n.startswith(prefix + "records/")}:
                raise ValueError("Stage 5 predictions are incomplete or unexpected.")
            for job in planned:
                r = read(prefix + "records/" + digest(job) + ".json")
                checksum = r.pop("record_hash")
                if digest(r) != checksum or r["run_fingerprint"] != run["fingerprint"] or any(r[k] != v for k, v in job.items()):
                    raise ValueError("Stage 5 record checksum or condition mismatch.")
                validate_scores(r, metadata[r["id"]])
                r["record_hash"] = checksum
                records.append(r)
            segments = [read(n) for n in sorted(names) if n.startswith(prefix + "segments/") and n.endswith(".json")]
            if (not segments or segments[-1]["status"] != "complete" or not segments[-1]["cleanup_passed"]
                    or any(s["run_fingerprint"] != run["fingerprint"] for s in segments)):
                raise ValueError("Stage 5 is not cleanly complete.")
            summary = read(prefix + "summary.json")
            if summary["status"] != "complete" or summary["run_fingerprint"] != run["fingerprint"] or summary["completed_records"] != len(records):
                raise ValueError("Stage 5 report requires review.")
            if split == "development":
                decision = decision_payload(run, records)
                if read(prefix + "operating_point.json") != decision:
                    raise ValueError("Stage 5 development decision differs from its evidence.")
                dev_end = max(s["started_unix"] + s["elapsed_seconds"] for s in segments)
            else:
                if c["frozen_decision"] != decision or c["protocol"] != runs["development"]["context"]["protocol"]:
                    raise ValueError("Stage 5 test changed the frozen development decision.")
                if min(s["started_unix"] for s in segments) <= dev_end:
                    raise ValueError("Final testing preceded development completion.")
            if summary["operating_point"] != decision:
                raise ValueError("Stage 5 summary decision mismatch.")
            runs[split], phases[split] = run, records
    return {"runs": runs, "test_records": phases["test"], "decision": decision, "baseline_run": baseline,
            "baseline_records": base_records, "development_examples": dev_examples, "test_examples": test_examples,
            "archive_sha256": file_digest(path)}


def robustness_inputs(config, settings, protocol, data_dir, baseline_dir, stage4_dir, stage5_archive):
    prior = read_stage5_archive(stage5_archive, config, settings, data_dir, baseline_dir, stage4_dir)
    subset = prompt_subset(prior["test_examples"], config)
    biology = [dict(json.loads(line), role="biology_control", split="test") for line in
               (Path(data_dir) / "biology_control/test.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(biology) != config["biology_control_size"] or len(subset) != config["profiles"]["full"]["test"]:
        raise ValueError("Unexpected robustness sample sizes.")
    ids = {e["id"] for e in subset}
    alpha = prior["decision"]["alpha"]
    prior["primary_records"] = [r for r in prior["test_records"] if r["id"] in ids and
                                (r["method"] == "baseline" or r["alpha"] == alpha)]
    prior["primary_examples"] = [e for e in prior["runs"]["test"]["context"]["examples"] if e["id"] in ids]
    prior["biology_examples"], prior["prompt_examples"] = biology, subset
    return prior
