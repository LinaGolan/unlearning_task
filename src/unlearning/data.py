"""Deterministic, disjoint splits from version-pinned public Parquet files.

Core splitting and verification use only the Python standard library.
Reading downloaded Parquet files requires pyarrow; no dataset scripts execute.
"""

import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

SPLITS = ("localization", "development", "test")
ROLES = ("forget", "retain", "biology_control")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema.")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer.")
    for source in [config["model"]] + [config["datasets"][role] for role in ROLES]:
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("Pin every model and dataset revision to a full commit SHA.")
    for role in ROLES:
        subjects = config["datasets"][role]["configs"]
        if not subjects or len(subjects) != len(set(subjects)):
            raise ValueError("Dataset subjects must be nonempty and unique.")
    for role in ("forget", "retain"):
        n_subjects = len(config["datasets"][role]["configs"])
        for profile, sizes in config["profiles"].items():
            for split in SPLITS:
                n = sizes[split]
                if type(n) is not int or n <= 0 or n % n_subjects:
                    raise ValueError("Split sizes must be positive and balanced across subjects.")
                if n > config["profiles"]["full"][split]:
                    raise ValueError("A reduced profile cannot exceed the full profile.")
    if type(config["biology_control_size"]) is not int or config["biology_control_size"] <= 0:
        raise ValueError("biology_control_size must be a positive integer.")
    return config


def normalized_question(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def make_example(raw, source, subject, row_index):
    question, choices, answer = raw["question"], list(raw["choices"]), raw["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Found an empty or invalid question.")
    if len(choices) != 4 or any(not isinstance(x, str) or not x.strip() for x in choices):
        raise ValueError("Every example must contain four nonempty choices.")
    if type(answer) is not int or answer not in range(4):
        raise ValueError("Answer must be an integer from 0 through 3.")
    return {
        "id": "{}:{}:test:{}".format(source["repo"], subject, row_index),
        "question": question,
        "choices": choices,
        "answer": answer,
        "subject": subject,
        "source_repo": source["repo"],
        "source_revision": source["revision"],
        "source_split": "test",
        "source_row": row_index,
        "question_hash": digest(normalized_question(question)),
        "content_hash": digest([question, choices, answer]),
    }


def load_sources(config, cache_dir):
    """Download only the ten small, configured public data files; never model weights."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Data preparation requires pyarrow. Use the Colab setup cell.") from exc
    pools, provenance = {}, []
    for role in ROLES:
        source = config["datasets"][role]
        pools[role] = []
        for subject in source["configs"]:
            remote_path = subject + "/test-00000-of-00001.parquet"
            cache = Path(cache_dir) / source["repo"].replace("/", "--") / source["revision"] / remote_path
            if not cache.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                url = "https://huggingface.co/datasets/{}/resolve/{}/{}".format(
                    source["repo"], source["revision"], remote_path
                )
                request = urllib.request.Request(url, headers={"User-Agent": "layer-unlearning-stage1"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    content = response.read()
                temporary = cache.with_suffix(".download")
                temporary.write_bytes(content)
                temporary.replace(cache)
            rows = pq.read_table(cache).to_pylist()
            pools[role].extend(make_example(row, source, subject, i) for i, row in enumerate(rows))
            provenance.append({
                "role": role, "repo": source["repo"], "revision": source["revision"],
                "file": remote_path, "sha256": file_digest(cache), "source_rows": len(rows),
            })
    return pools, provenance


def build_splits(config, pools, profile):
    """Deduplicate by question text and allocate full splits before taking prefixes.

    A reduced run uses a subset of each full split, so switching profiles cannot
    turn an old localization question into a final-test question.
    """
    sizes = config["profiles"][profile]
    full = config["profiles"]["full"]
    seed = config["seed"]
    seen_questions, seen_ids, dropped = set(), set(), {}
    clean, outputs = {}, {}
    for role in ROLES:
        clean[role], dropped[role] = [], 0
        for row in sorted(pools[role], key=lambda x: x["id"]):
            if row["id"] in seen_ids:
                raise ValueError("Duplicate source example ID: " + row["id"])
            seen_ids.add(row["id"])
            if row["question_hash"] in seen_questions:
                dropped[role] += 1
                continue
            seen_questions.add(row["question_hash"])
            clean[role].append(row)
    for role in ("forget", "retain"):
        subjects = config["datasets"][role]["configs"]
        for split in SPLITS:
            outputs[role + "/" + split] = []
        for subject in subjects:
            rows = [row for row in clean[role] if row["subject"] == subject]
            rows.sort(key=lambda row: (digest([seed, role, row["id"]]), row["id"]))
            required = sum(full.values()) // len(subjects)
            if len(rows) < required:
                raise ValueError("Insufficient unique questions in {}: need {}, have {}.".format(
                    subject, required, len(rows)))
            offset = 0
            for split in SPLITS:
                count = sizes[split] // len(subjects)
                outputs[role + "/" + split].extend(rows[offset:offset + count])
                offset += full[split] // len(subjects)
        for split in SPLITS:
            outputs[role + "/" + split].sort(key=lambda row: digest([seed, split, row["id"]]))
    control = sorted(clean["biology_control"], key=lambda row: digest([seed, "biology", row["id"]]))
    if len(control) < config["biology_control_size"]:
        raise ValueError("Insufficient unique general-biology control questions.")
    outputs["biology_control/test"] = control[:config["biology_control_size"]]
    return outputs, dropped


def save_prepared(config, pools, provenance, profile, output_dir):
    output_dir = Path(output_dir)
    outputs, dropped = build_splits(config, pools, profile)
    request_hash = digest([config, profile])
    if output_dir.exists() and any(output_dir.iterdir()):
        manifest = verify_prepared(output_dir)
        if manifest["request_hash"] != request_hash:
            raise ValueError("Output belongs to different settings. Choose a new output directory.")
        if manifest["sources"] != provenance:
            raise ValueError("Source data changed. Do not overwrite an existing experiment.")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in outputs.items():
        relative = name + ".jsonl"
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                        encoding="utf-8")
        files[relative] = {
            "count": len(rows), "sha256": file_digest(path),
            "subjects": dict(sorted(Counter(row["subject"] for row in rows).items())),
        }
    manifest = {
        "schema_version": 1, "profile": profile, "request_hash": request_hash,
        "config": config, "sources": provenance, "files": files,
        "excluded_duplicate_questions": dropped,
        "split_rule": "sha256 ranking; allocate full split boundaries, then take per-subject prefixes",
        "note": "Research test data is reserved; no model predictions are produced during preparation.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return verify_prepared(output_dir)


def verify_prepared(output_dir):
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config, profile = manifest["config"], manifest["profile"]
    if manifest["request_hash"] != digest([config, profile]):
        raise ValueError("Configuration fingerprint mismatch.")
    expected = {role + "/" + split + ".jsonl" for role in ("forget", "retain") for split in SPLITS}
    expected.add("biology_control/test.jsonl")
    if set(manifest["files"]) != expected:
        raise ValueError("Prepared dataset is missing an expected split or contains unexpected splits.")
    ids, questions = set(), set()
    for relative, info in manifest["files"].items():
        path = root / relative
        if file_digest(path) != info["sha256"]:
            raise ValueError("Checksum mismatch: " + relative)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        role, split_file = relative.split("/")
        split = split_file[:-len(".jsonl")]
        expected_count = config["biology_control_size"] if role == "biology_control" else config["profiles"][profile][split]
        if len(rows) != expected_count or len(rows) != info["count"]:
            raise ValueError("Wrong question count: " + relative)
        source = config["datasets"][role]
        subjects = Counter()
        for row in rows:
            rebuilt = make_example(row, source, row["subject"], row["source_row"])
            if rebuilt != row or row["subject"] not in source["configs"]:
                raise ValueError("Invalid example or provenance: " + relative)
            if row["id"] in ids or row["question_hash"] in questions:
                raise ValueError("Duplicate question or split overlap: " + relative)
            ids.add(row["id"])
            questions.add(row["question_hash"])
            subjects[row["subject"]] += 1
        if dict(subjects) != info["subjects"]:
            raise ValueError("Subject count mismatch: " + relative)
        if role != "biology_control" and dict(subjects) != {
            subject: expected_count // len(source["configs"]) for subject in source["configs"]
        }:
            raise ValueError("Unbalanced subjects: " + relative)
    return manifest
