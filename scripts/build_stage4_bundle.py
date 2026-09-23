"""Build the next Colab stage with verified Stage 2/3 evidence and prepared data."""

import hashlib
import json
from pathlib import Path
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from unlearning.baseline import read_settings
from unlearning.data import read_config
from unlearning.localization import read_localization_settings, stage4_inputs

BASELINE = ROOT / "outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill"
STAGE3 = ROOT / "outputs/stage3/verified_2026-09-20/outputs/stage3/20260920T160702Z-b27cdc"


def build_notebook():
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": "stage4-{:02d}".format(len(cells)), "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """
        # Stage 4: Find promising layers and test the predictions

        We have checked the baseline and the intervention mechanism. This stage asks:
        **Which layers support forget answers more than retain answers, and do their scores predict actual effects?**

        1. Calculate a score for every layer on **128 forget + 128 retain localization questions**.
        2. Compare rankings from two balanced halves of those questions.
        3. Fix the top-forget, top-selective, bottom-forget, and five random layer pairs.
        4. Run **Control A** on **16 forget + 16 retain development questions**:
           weaken each layer separately at strengths **0.05 and 0.5**, then compare predicted and measured effects.

        All weights stay frozen. Final-test questions are reserved for later.
        Poor ranking stability or poor predictions are findings to discuss, not a reason to change the formula.

        **To run:** select a **T4 GPU**, choose **Run all**, upload **unlearning-stage4.zip**, and allow `HF_TOKEN` access.
        The ZIP includes the verified data and Stage 2/3 evidence. No earlier stage needs rerunning.
        Download **stage4-results.zip** at the end, including if interrupted or incomplete.

        The measured Stage 3 timings suggest roughly **5-10 minutes of model work**, plus setup and model download.
        Actual time depends on Colab and prompt lengths. Progress is saved after every measurement.
    """)
    add("markdown", """
        ## 1. Upload the Stage 4 bundle

        Select `dist/unlearning-stage4.zip` from the project folder on your computer.
    """)
    add("code", """
        import hashlib
        import io
        import json
        import os
        from pathlib import Path, PurePosixPath
        import subprocess
        import sys
        import zipfile
        from google.colab import files

        uploaded = files.upload()
        if len(uploaded) != 1 or 'unlearning-stage4.zip' not in uploaded:
            raise ValueError('Select unlearning-stage4.zip from the project dist folder.')
        project = Path('/content/unlearning_stage4')
        project.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(uploaded['unlearning-stage4.zip'])) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive entries.')
            bundle = json.loads(archive.read('bundle_manifest.json'))
            if set(names) != set(bundle['files']) | {'bundle_manifest.json'}:
                raise ValueError('Archive contents do not match the bundle manifest.')
            for name in names:
                relative = PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or '\\\\' in name:
                    raise ValueError('Unexpected archive path.')
                target = (project / name).resolve()
                if project.resolve() not in target.parents:
                    raise ValueError('Archive path is outside the project.')
                content = archive.read(name)
                if name != 'bundle_manifest.json' and hashlib.sha256(content).hexdigest() != bundle['files'][name]:
                    raise ValueError('Archive checksum mismatch: ' + name)
                if name.startswith('inputs/') and target.exists() and target.read_bytes() != content:
                    raise ValueError('Existing inputs differ. Use a fresh Colab runtime.')
            for name in names:
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        del uploaded
        os.chdir(project)
        print('Stage 4 code and verified inputs ready.')
    """)
    add("markdown", """
        ## 2. Install and check the code and inputs

        The tests use synthetic data and tiny random models. They check calculation, data separation,
        and resuming after an interruption. These test results are not research findings.
    """)
    add("code", """
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--only-binary=:all:',
                        '-r', 'requirements-stage4-colab.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
        subprocess.run([sys.executable, '-m', 'unlearning', 'doctor',
                        '--output', 'outputs/stage4_environment.json'], check=True)
        environment = json.loads(Path('outputs/stage4_environment.json').read_text())
        if not environment['cuda_available']:
            raise RuntimeError('Select a T4 GPU runtime before continuing.')
        subprocess.run([sys.executable, '-c',
            "from unlearning.baseline import read_settings; from unlearning.data import read_config; "
            "from unlearning.localization import read_localization_settings, stage4_inputs; "
            "stage4_inputs(read_config('configs/experiment.json'), read_settings('configs/baseline_prefill.json'), "
            "read_localization_settings('configs/localization.json'), 'inputs/data/prepared/full', "
            "'inputs/outputs/stage2/prefill', 'inputs/stage3'); print('Verified Stage 4 inputs.')"], check=True)
    """)
    add("markdown", """
        ## 3. Optional: restore an interrupted Stage 4 run

        On your first run, leave this cell unchanged. It does not ask for another upload.
        To recover after losing a runtime, change `RESTORE_RESULTS` to `True` and upload your latest
        `stage4-results.zip`. Existing results must be identical; use a fresh runtime if they conflict.
        Records are checked before any GPU work resumes. Resume requires matching code, settings, and runtime versions.
    """)
    add("code", """
        RESTORE_RESULTS = False
        result_dir = Path('outputs/stage4/full')
        if RESTORE_RESULTS:
            from unlearning.result_archives import restore_stage4_results
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError('Select exactly one saved Stage 4 results ZIP.')
            print(restore_stage4_results(next(iter(uploaded.values())), project))
            del uploaded
            # A prior failed measurement still needs review, but valid partial records can resume.
            from unlearning.localization_report import stage4_report
            restored = stage4_report(result_dir, make_plots=False)
            print('Verified saved records:', restored['completed_records'], '/', restored['expected_records'])
            if restored['status'] == 'review_required':
                print('The previous attempt needs attention. Read its saved failure before resuming.')
    """)
    add("markdown", """
        ## 4. Run localization and Control A

        `MAX_NEW = None` runs all remaining work: **256 localization gradients, 32 control gradients,
        and 1,024 single-layer interventions**. It saves 1,312 records in total.
        Set `MAX_NEW = 256` for shorter segments, then download a backup and rerun this cell to continue.

        The layer pairs are fixed after all localization questions are complete, before the control outcomes.
        A positive score predicts that weakening a layer lowers the correct-answer score.
        Selectivity is the forget mean minus the retain mean. Negative scores are retained.
    """)
    add("code", """
        import getpass
        from google.colab import userdata
        MAX_NEW = None

        saved_summary = json.loads((result_dir / 'summary.json').read_text()) if (result_dir / 'summary.json').exists() else {}
        if saved_summary.get('completed_records') == saved_summary.get('expected_records') and saved_summary.get('expected_records'):
            # Regenerate tables/figures only; completed model measurements are not repeated.
            completed = subprocess.run([sys.executable, '-m', 'unlearning', 'stage4-report', '--output', str(result_dir)])
        else:
            try:
                os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN').strip()
            except Exception:
                os.environ['HF_TOKEN'] = getpass.getpass('Hugging Face read token (hidden): ').strip()
            if not os.environ['HF_TOKEN']:
                raise RuntimeError('No token supplied.')
            args = [sys.executable, '-m', 'unlearning', 'localize', '--output', str(result_dir)]
            if MAX_NEW is not None:
                args.extend(['--max-new', str(MAX_NEW)])
            completed = subprocess.run(args)
        if completed.returncode:
            print('The command needs attention. Read the message above and download the saved evidence below.')
        else:
            print('Segment finished. Review the summary and download a backup below.')
    """)
    add("markdown", """
        ## 5. Read the results

        **complete** means all measurements were saved and the report was produced. It does not mean the hypothesis succeeded.
        **partial** means more work remains; rerun the previous cell with the same settings.
        **review_required** means a technical failure or report issue needs attention.

        The figures show layer means, agreement between two data halves, and predicted versus measured score drops.
        In Control A, points near the diagonal indicate accurate predictions. Compare the small and larger strength.
        These are changes in the correct-answer score, not changes in accuracy percentage points.
    """)
    add("code", """
        from IPython.display import display, Image
        summary_path = result_dir / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            print('Stage 4:', summary['status'])
            print('Saved:', summary['completed_records'], '/', summary['expected_records'])
            for selection in summary.get('selections', []):
                print(selection['name'], ': layers', selection['layers'])
            for name, stability in summary.get('stability', {}).items():
                print(name, 'half-to-half Spearman:', stability['spearman'], '| top-pair overlap:', stability['top_k_overlap'])
            if summary.get('last_attempt', {}).get('failure'):
                print(json.dumps(summary['last_attempt']['failure'], indent=2))
            print(summary['next_action'])
            for path in sorted((result_dir / 'figures').glob('*.png')):
                display(Image(filename=str(path), width=900))
        else:
            print('No run report was written. Keep the error above and download the setup evidence below.')
    """)
    add("markdown", """
        ## 6. Download a backup

        Share `stage4-results.zip` for review. Download it after each segment and before disconnecting.
        The archive contains the per-question records, frozen selections (when ready), figures, source, settings, and prior-stage evidence.
        Stage 5 does not start automatically.
    """)
    add("code", """
        archive_path = Path('/content/stage4-results.zip')
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(result_dir.rglob('*')):
                if path.is_file() and not path.name.endswith('.tmp'):
                    archive.write(path, path.as_posix())
            for name in ('outputs/stage4_environment.json', 'bundle_manifest.json', 'STAGE4.md',
                         'requirements-colab.txt', 'requirements-stage4-colab.txt'):
                if Path(name).exists():
                    archive.write(name)
            for directory, pattern in (('configs', '*.json'), ('src/unlearning', '*.py'), ('inputs/stage3', '*.json')):
                for path in sorted(Path(directory).glob(pattern)):
                    archive.write(path, path.as_posix())
        files.download(str(archive_path))
    """)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "04_stage4_localization.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/04_stage4_localization.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    config = read_config(ROOT / "configs/experiment.json")
    settings = read_settings(ROOT / "configs/baseline_prefill.json")
    protocol = read_localization_settings(ROOT / "configs/localization.json")
    _, localization, control, baseline_run, _, stage3_run = stage4_inputs(config, settings, protocol,
        ROOT / "data/prepared/full", BASELINE, STAGE3)
    notebook = build_notebook()
    entries = {}
    for name in ("pyproject.toml", "requirements-colab.txt", "requirements-stage4-colab.txt", "README.md", "PLAN.md",
                 "STAGE1.md", "STAGE2.md", "STAGE3.md", "STAGE4.md", "DIAGNOSTIC.md"):
        entries[name] = (ROOT / name).read_bytes()
    for directory, suffix in (("src", ".py"), ("tests", ".py"), ("configs", ".json")):
        for path in sorted((ROOT / directory).rglob("*" + suffix)):
            entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    entries[notebook.relative_to(ROOT).as_posix()] = notebook.read_bytes()
    entries["scripts/build_stage4_bundle.py"] = Path(__file__).read_bytes()
    for path in sorted((ROOT / "data/prepared/full").rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            entries["inputs/" + path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for path in sorted(BASELINE.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv"):
            entries["inputs/outputs/stage2/prefill/" + path.relative_to(BASELINE).as_posix()] = path.read_bytes()
    for path in sorted(STAGE3.glob("*.json")):
        entries["inputs/stage3/" + path.name] = path.read_bytes()
    manifest = {"stage": 4, "schema_version": 1, "profile": "full",
        "baseline_fingerprint": baseline_run["fingerprint"], "stage3_fingerprint": stage3_run["fingerprint"],
        "localization_questions": len(localization), "control_questions": len(control),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(entries.items())}}
    entries["bundle_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output = ROOT / "dist/unlearning-stage4.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    print(json.dumps({"notebook": str(notebook), "bundle": str(output), "bytes": output.stat().st_size,
                      "localization_questions": len(localization), "control_questions": len(control)}, indent=2))


if __name__ == "__main__":
    main()
