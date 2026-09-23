"""Restore only Stage 4 result artifacts from a user-provided backup ZIP."""

import io
from pathlib import Path, PurePosixPath
import zipfile


def restore_stage4_results(payload, project):
    return _restore_results(payload, project, 4, ("full",))


def restore_stage5_results(payload, project):
    return _restore_results(payload, project, 5, ("development", "test"))


def restore_stage6_results(payload, project):
    return _restore_results(payload, project, 6, ("full",))


def _restore_results(payload, project, stage, phases):
    project = Path(project).resolve()
    outputs = [(project / ("outputs/stage{}/".format(stage) + phase)).resolve() for phase in phases]
    prefixes = tuple("outputs/stage{}/{}/".format(stage, phase) for phase in phases)
    pending = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate paths in result archive.")
        for item in archive.infolist():
            name = item.filename
            if item.is_dir() or not name.startswith(prefixes):
                continue  # Source code/configuration in a results ZIP is evidence, not an update.
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError("Unsafe result archive path.")
            target = (project / name).resolve()
            if not any(output in target.parents for output in outputs) or target.suffix not in (".json", ".jsonl", ".csv", ".png", ".pdf"):
                raise ValueError("Unexpected Stage {} result file.".format(stage))
            content = archive.read(item)
            if target.exists() and target.read_bytes() != content:
                raise ValueError("Conflicting existing results. Restore into a fresh Colab runtime.")
            pending.append((target, content))
        if not any(target == outputs[0] / "run.json" for target, _ in pending):
            raise ValueError("The ZIP contains no Stage {} {} run manifest.".format(stage, phases[0]))
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
    return {"restored_files": len(pending), "output": str(outputs[0]),
            "note": "Run stage{}-report to verify record hashes and derived values before resuming.".format(stage)}
