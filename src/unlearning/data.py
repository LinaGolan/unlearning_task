"""Deterministic, disjoint question splits from revision-pinned public Parquet files.

Only the ten small configured data files are downloaded; no dataset scripts run.
Splits are derived from SHA-256 rankings, so any machine reproduces them exactly.
"""

import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

ROLES = ("forget", "retain", "biology")
SPLITS = ("localization", "development", "test")


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("Unsupported configuration schema.")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer.")
    for source in [config["model"]] + [config["datasets"][role] for role in ROLES]:
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("Pin every model and dataset revision to a full commit SHA.")
    for role in ("forget", "retain"):
        n = len(config["datasets"][role]["subjects"])
        for split in SPLITS:
            if config["splits"][split] % n:
                raise ValueError("Split sizes must divide evenly across the subjects of " + role)
    if config["intervention"]["k"] < 1:
        raise ValueError("k must be at least one layer.")
    return config


def _make_example(raw, repo, revision, subject, row):
    question, choices, answer = raw["question"], list(raw["choices"]), raw["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Empty question.")
    if len(choices) != 4 or any(not isinstance(c, str) or not c.strip() for c in choices):
        raise ValueError("Every example needs four non-empty choices.")
    if type(answer) is not int or answer not in range(4):
        raise ValueError("Answer must be 0-3.")
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    return {"id": "{}:{}:test:{}".format(repo, subject, row), "question": question,
            "choices": choices, "answer": answer, "subject": subject,
            "source_repo": repo, "source_revision": revision, "source_row": row,
            "question_hash": digest(normalized), "content_hash": digest([question, choices, answer])}


def load_sources(config, cache_dir):
    """Download the configured public Parquet files once and cache them by revision."""
    import pyarrow.parquet as pq
    pools, provenance = {}, []
    for role in ROLES:
        source = config["datasets"][role]
        pools[role] = []
        for subject in source["subjects"]:
            remote = subject + "/test-00000-of-00001.parquet"
            cache = Path(cache_dir) / source["repo"].replace("/", "--") / source["revision"] / remote
            if not cache.exists():
                url = "https://huggingface.co/datasets/{}/resolve/{}/{}".format(
                    source["repo"], source["revision"], remote)
                cache.parent.mkdir(parents=True, exist_ok=True)
                request = urllib.request.Request(url, headers={"User-Agent": "layer-unlearning"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = response.read()
                temp = cache.with_suffix(".download")
                temp.write_bytes(payload)
                temp.replace(cache)
            rows = pq.read_table(cache).to_pylist()
            pools[role] += [_make_example(row, source["repo"], source["revision"], subject, i)
                            for i, row in enumerate(rows)]
            provenance.append({"role": role, "repo": source["repo"], "revision": source["revision"],
                               "file": remote, "sha256": file_digest(cache), "source_rows": len(rows)})
    return pools, provenance


def build_splits(config, pools):
    """Deduplicate by question text, then cut disjoint per-subject splits by hash rank."""
    seed, sizes = config["seed"], config["splits"]
    seen, clean, dropped = set(), {}, {}
    for role in ROLES:
        clean[role], dropped[role] = [], 0
        for row in sorted(pools[role], key=lambda x: x["id"]):
            if row["question_hash"] in seen:
                dropped[role] += 1
                continue
            seen.add(row["question_hash"])
            clean[role].append(row)
    out = {}
    for role in ("forget", "retain"):
        subjects = list(config["datasets"][role]["subjects"])
        for split in SPLITS:
            out[role + "/" + split] = []
        for subject in subjects:
            rows = sorted((r for r in clean[role] if r["subject"] == subject),
                          key=lambda r: (digest([seed, role, r["id"]]), r["id"]))
            need = sum(sizes.values()) // len(subjects)
            if len(rows) < need:
                raise ValueError("Need {} unique questions in {}, have {}.".format(need, subject, len(rows)))
            offset = 0
            for split in SPLITS:
                count = sizes[split] // len(subjects)
                out[role + "/" + split] += rows[offset:offset + count]
                offset += count
        for split in SPLITS:
            out[role + "/" + split].sort(key=lambda r: digest([seed, split, r["id"]]))
    biology = sorted(clean["biology"], key=lambda r: digest([seed, "biology", r["id"]]))
    out["biology/test"] = biology[:config["biology_size"]]
    return out, dropped


def prepare(config, data_dir, cache_dir):
    """Write the split JSONL files plus a manifest, or verify an existing identical one."""
    data_dir = Path(data_dir)
    pools, provenance = load_sources(config, cache_dir)
    splits, dropped = build_splits(config, pools)
    request_hash = digest([config["seed"], config["datasets"], config["splits"], config["biology_size"]])
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["request_hash"] != request_hash:
            raise ValueError("data/ was built from different settings. Use a fresh directory.")
        verify(data_dir)
        return manifest
    files = {}
    for name, rows in splits.items():
        path = data_dir / (name + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                        encoding="utf-8")
        files[name + ".jsonl"] = {"count": len(rows), "sha256": file_digest(path),
                                  "subjects": dict(sorted(Counter(r["subject"] for r in rows).items()))}
    manifest = {"schema_version": 2, "request_hash": request_hash, "seed": config["seed"],
                "datasets": config["datasets"], "splits": config["splits"],
                "biology_size": config["biology_size"], "sources": provenance, "files": files,
                "excluded_duplicate_questions": dropped,
                "rule": "deduplicate by question text; per-subject SHA-256 rank cuts disjoint splits"}
    write_json(manifest_path, manifest)
    verify(data_dir)
    return manifest


def verify(data_dir):
    """Re-check every checksum and that no question appears in two splits."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    seen = set()
    for name, info in sorted(manifest["files"].items()):
        path = data_dir / name
        if file_digest(path) != info["sha256"]:
            raise ValueError("Checksum mismatch: " + name)
        rows = read_jsonl(path)
        if len(rows) != info["count"]:
            raise ValueError("Wrong question count: " + name)
        for row in rows:
            if row["question_hash"] in seen:
                raise ValueError("Question appears in two splits: " + row["id"])
            seen.add(row["question_hash"])
    return manifest


def load_split(data_dir, role, split):
    return [dict(row, role=role, split=split)
            for row in read_jsonl(Path(data_dir) / role / (split + ".jsonl"))]
