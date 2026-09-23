"""Build a self-contained Stage 5 notebook using verified Stage 2 and 4 evidence."""

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
from unlearning.sweep import read_sweep_settings, sweep_inputs

BASELINE = ROOT / "outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill"
STAGE4 = ROOT / "outputs/stage4/verified_2026-09-20/outputs/stage4/full"


def build_notebook():
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": "stage5-{:02d}".format(len(cells)), "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """
        # Stage 5: Compare the fixed layer pairs

        We now measure how much each intervention reduces **forget accuracy** and **retain accuracy**.
        Forget accuracy means the fraction of WMDP-Bio questions answered correctly.
        We want forget accuracy to fall while retain accuracy stays close to its baseline.

        The layer pairs were fixed in Stage 4. This notebook:

        1. Tries strengths 0.25, 0.5, 0.75, and 1 on development questions for all eight fixed pairs.
        2. Chooses one common strength using **only the top-selective development results** and saves that decision.
        3. Evaluates all fixed curves on the reserved final-test questions, with no further tuning.
        4. Produces a comparison table, plots, and paired uncertainty estimates.

        The rule chooses the largest forget-accuracy drop with at most **5 percentage points** of retain loss.
        Ties use the weaker strength. If no strength qualifies, the report says so; 0.5 is only a display fallback.

        **Run:** select a **T4 GPU**, choose **Run all**, upload **unlearning-stage5.zip**, and allow `HF_TOKEN` access.
        Leave `RESTORE_RESULTS = False` and `MAX_NEW = None` for a first full run.
        The bundle includes the verified inputs; earlier stages do not need rerunning.

        This stage performs **25,088 new question evaluations**, so allow substantially more time than Stage 4.
        Runtime depends on Colab; progress is saved after every prediction. Download a backup before leaving.
        A development backup downloads before test scoring, and **stage5-results.zip** downloads at the end.
        General-biology and alternative-prompt robustness checks come in Stage 6.
    """)
    add("markdown", "## 1. Upload the Stage 5 bundle\nSelect `dist/unlearning-stage5.zip` from your computer.")
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
        if len(uploaded) != 1 or 'unlearning-stage5.zip' not in uploaded:
            raise ValueError('Select unlearning-stage5.zip from the project dist folder.')
        project = Path('/content/unlearning_stage5')
        project.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(uploaded['unlearning-stage5.zip'])) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive entries.')
            bundle = json.loads(archive.read('bundle_manifest.json'))
            if set(names) != set(bundle['files']) | {'bundle_manifest.json'}:
                raise ValueError('Archive contents differ from the manifest.')
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
                if target.exists() and target.read_bytes() != content:
                    raise ValueError('Existing bundle differs. Use a fresh Colab runtime.')
            for name in names:
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        del uploaded
        os.chdir(project)
        print('Stage 5 bundle and verified prior evidence ready.')
    """)
    add("markdown", """
        ## 2. Check the environment and inputs

        These software tests use tiny random models and synthetic questions. They are not research results.
        No model weights are downloaded until the experiment starts.
    """)
    add("code", """
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--only-binary=:all:',
                        '-r', 'requirements-stage4-colab.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
        # Editable installs affect new Python processes; this already-running kernel
        # also needs the source directory on its own import path.
        import importlib
        source_dir = str((project / 'src').resolve())
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        importlib.invalidate_caches()
        from unlearning.sweep_report import sweep_report
        print('Project imports work in the notebook kernel.')
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
        subprocess.run([sys.executable, '-m', 'unlearning', 'doctor', '--output', 'outputs/stage5_environment.json'], check=True)
        environment = json.loads(Path('outputs/stage5_environment.json').read_text())
        if not environment['cuda_available']:
            raise RuntimeError('Select a T4 GPU runtime before continuing.')
        subprocess.run([sys.executable, '-c',
            "from unlearning.baseline import read_settings; from unlearning.data import read_config; "
            "from unlearning.sweep import sweep_inputs; "
            "sweep_inputs(read_config('configs/experiment.json'), read_settings('configs/baseline_prefill.json'), "
            "'inputs/data/prepared/full', 'inputs/outputs/stage2/prefill', 'inputs/stage4'); "
            "print('Verified Stage 5 inputs and frozen layer pairs.')"], check=True)
    """)
    add("markdown", """
        ## 3. Optional: restore a saved run

        Leave this unchanged on your first run. If Colab disconnected, set `RESTORE_RESULTS = True`
        and upload your most recent Stage 5 results ZIP (development-only or full).
        Saved predictions are checked and reused. Resume needs the same bundle, settings, and runtime versions.
    """)
    add("code", """
        RESTORE_RESULTS = False
        if RESTORE_RESULTS:
            from unlearning.result_archives import restore_stage5_results
            from unlearning.sweep_report import sweep_report
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError('Select one saved Stage 5 results ZIP.')
            print(restore_stage5_results(next(iter(uploaded.values())), project))
            del uploaded
            for phase in ('development', 'test'):
                folder = Path('outputs/stage5') / phase
                if (folder / 'run.json').exists():
                    report = sweep_report(folder, make_plots=False)
                    print(phase, report['status'], report['completed_records'], '/', report['expected_records'])
                    if report['last_attempt'].get('failure'):
                        print('Review previous failure:', report['last_attempt']['failure'])
    """)
    add("markdown", """
        ## 4. Settings, backup helper, and access

        `MAX_NEW = None` runs all remaining predictions in each phase. For shorter sessions, set it to `2048`.
        Then download a backup and rerun the development cell until complete, followed by the test cell.
        A completed phase regenerates its report without loading the model.

        The backup helper can be called at any time after stopping a long-running cell:
        `download_backup()` saves everything completed so far. Temporary files are excluded.
    """)
    add("code", """
        MAX_NEW = None
        import getpass
        from google.colab import userdata

        def download_backup(filename='stage5-results.zip'):
            archive_path = Path('/content') / filename
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
                for directory in ('outputs/stage5', 'configs', 'src/unlearning', 'inputs/stage4'):
                    for path in sorted(Path(directory).rglob('*')):
                        if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.png', '.pdf', '.py'):
                            archive.write(path, path.as_posix())
                for name in ('outputs/stage5_environment.json', 'bundle_manifest.json', 'STAGE5.md',
                             'requirements-colab.txt', 'requirements-stage4-colab.txt'):
                    if Path(name).exists():
                        archive.write(name)
            files.download(str(archive_path))
            print('Backup:', filename, '| bytes:', archive_path.stat().st_size)

        def run_phase(phase):
            folder = Path('outputs/stage5') / phase
            # Verify raw files before deciding to skip GPU execution.
            from unlearning.sweep_report import sweep_report
            saved = sweep_report(folder, make_plots=False) if (folder / 'run.json').exists() else {}
            if saved.get('completed_records') == saved.get('expected_records') and saved.get('expected_records'):
                args = [sys.executable, '-m', 'unlearning', 'stage5-report', '--output', str(folder)]
            else:
                if not os.environ.get('HF_TOKEN'):
                    try:
                        os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN').strip()
                    except Exception:
                        os.environ['HF_TOKEN'] = getpass.getpass('Hugging Face read token (hidden): ').strip()
                if not os.environ.get('HF_TOKEN'):
                    raise RuntimeError('No token supplied.')
                args = [sys.executable, '-m', 'unlearning', 'sweep', '--split', phase]
                if MAX_NEW is not None:
                    args += ['--max-new', str(MAX_NEW)]
            completed = subprocess.run(args)
            if completed.returncode:
                print('This phase needs attention. Download a backup and keep the error above.')
            return json.loads((folder / 'summary.json').read_text()) if (folder / 'summary.json').exists() else {}
    """)
    add("markdown", """
        ## 5. Development sweep and frozen decision

        This phase adds **8,192 predictions** on 128 forget and 128 retain questions.
        The 256 unchanged-model predictions are reused from Stage 2.
        The saved operating point records every candidate and why it did or did not qualify.
        `complete` means execution finished; it does not mean selective forgetting succeeded.
    """)
    add("code", """
        development_report = run_phase('development')
        print('Development:', development_report.get('status', 'no report'))
        decision = development_report.get('operating_point')
        if decision:
            print('Decision:', decision['status'], '| common strength:', decision['alpha'])
            print('Display-only fallback:', decision['display_only_fallback'])
            for row in decision['candidates']:
                print(row)
        download_backup('stage5-development-results.zip')
    """)
    add("markdown", """
        ## 6. Final-test sweep

        This starts only after the complete development decision passes verification.
        It adds **16,896 predictions**: an unchanged baseline plus all fixed interventions,
        on 256 forget and 256 retain questions.
        The test curves help describe the experiment; they must not choose a new strength or layer pair.
    """)
    add("code", """
        from unlearning.sweep import verified_decision
        try:
            development_run, frozen_decision = verified_decision('outputs/stage5/development')
        except (ValueError, OSError, KeyError) as exc:
            print('Test not started. Finish or review development first:', str(exc))
        else:
            if development_report.get('status') != 'complete':
                print('Test not started. Review the development report issue first.')
            else:
                test_report = run_phase('test')
                print('Final test:', test_report.get('status', 'no report'))
    """)
    add("markdown", """
        ## 7. Read the comparison and download the results

        Accuracy is the percent answered correctly. A **drop** is baseline accuracy minus intervention accuracy:
        positive means performance fell, negative means it improved.
        The table uses the same development-selected strength for every method.
        If no strength qualified, the table is explicitly a display-only comparison at 0.5.

        The plots show all strengths. The shaded random band is the range across the five fixed random pairs;
        it is **not** a confidence interval. Paired 95% intervals are in the saved tables.
        They describe question-sampling uncertainty conditional on this experiment, not every possible layer selection.
        Noisy Stage 4 rankings remain a limitation even if a comparison looks favorable.

        Share the final **stage5-results.zip** for review. Stage 6 does not start automatically.
    """)
    add("code", """
        from IPython.display import display, Image
        for phase in ('development', 'test'):
            folder = Path('outputs/stage5') / phase
            if not (folder / 'summary.json').exists():
                continue
            report = json.loads((folder / 'summary.json').read_text())
            print(phase.upper(), report['status'], report['completed_records'], '/', report['expected_records'])
            if report.get('operating_point'):
                print('Common strength:', report['operating_point']['alpha'], '|', report['operating_point']['status'])
                print('Method                 Forget acc   Retain acc   Forget drop   Retain drop')
                for row in report['main_comparison']:
                    print('{:<22} {:>8.2f}%   {:>8.2f}%   {:>8.2f} pp   {:>8.2f} pp'.format(
                        row['method'], row['forget_accuracy_percent'], row['retain_accuracy_percent'],
                        row['forget_drop_pp'], row['retain_drop_pp']))
            if report.get('last_attempt', {}).get('failure'):
                print(report['last_attempt']['failure'])
            for path in sorted((folder / 'figures').glob('*.png')):
                display(Image(filename=str(path), width=900))
        download_backup()
    """)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "05_stage5_sweep.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/05_stage5_sweep.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    config = read_config(ROOT / "configs/experiment.json")
    settings = read_settings(ROOT / "configs/baseline_prefill.json")
    read_sweep_settings(ROOT / "configs/sweep.json")
    _, development, baseline, _, stage4, selection = sweep_inputs(config, settings,
        ROOT / "data/prepared/full", BASELINE, STAGE4)
    notebook = build_notebook()
    entries = {}
    for name in ("pyproject.toml", "requirements-colab.txt", "requirements-stage4-colab.txt", "README.md", "PLAN.md",
                 "STAGE1.md", "STAGE2.md", "STAGE3.md", "STAGE4.md", "STAGE5.md", "DIAGNOSTIC.md"):
        entries[name] = (ROOT / name).read_bytes()
    for directory, suffix in (("src", ".py"), ("tests", ".py"), ("configs", ".json")):
        for path in sorted((ROOT / directory).rglob("*" + suffix)):
            entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    entries[notebook.relative_to(ROOT).as_posix()] = notebook.read_bytes()
    entries["scripts/build_stage5_bundle.py"] = Path(__file__).read_bytes()
    for directory, destination, suffixes in ((ROOT / "data/prepared/full", "inputs/data/prepared/full", (".json", ".jsonl")),
            (BASELINE, "inputs/outputs/stage2/prefill", (".json", ".jsonl", ".csv")),
            (STAGE4, "inputs/stage4", (".json", ".csv", ".png", ".pdf"))):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                entries[destination + "/" + path.relative_to(directory).as_posix()] = path.read_bytes()
    manifest = {"stage": 5, "schema_version": 1, "profile": "full", "baseline_fingerprint": baseline["fingerprint"],
        "stage4_fingerprint": stage4["fingerprint"], "selection_fingerprint": selection["selection_fingerprint"],
        "development_questions": len(development), "test_questions": 2 * config["profiles"]["full"]["test"],
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(entries.items())}}
    entries["bundle_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output = ROOT / "dist/unlearning-stage5.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    print(json.dumps({"notebook": str(notebook), "bundle": str(output), "bytes": output.stat().st_size,
                      "stage4_verified": stage4["fingerprint"], "selections": selection["selections"]}, indent=2))


if __name__ == "__main__":
    main()
