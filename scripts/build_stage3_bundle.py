"""Package Stage 3 with verified existing inputs; no model/data downloads or runs."""

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
from unlearning.intervention_checks import verified_inputs

VERIFIED_BASELINE = ROOT / "outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill"


def build_notebook():
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": "stage3-{:02d}".format(len(cells)), "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
        # Stage 3: Check the layer intervention

        We already measured the unchanged model: **60.9% forget accuracy and 53.9% retain accuracy**.
        Now we check the mechanism we will use to weaken selected layers.

        - Strength **0** should leave the model unchanged.
        - Strength **1** should pass a selected layer's input through unchanged.
        - Small numerical changes should agree with the calculated gradients.
        - Removing the intervention should restore normal behavior.

        This notebook checks **four existing development questions** plus harmless synthetic questions.
        It does not rank layers, rerun the full baseline, or evaluate the final test set.
        The ZIP already includes our verified inputs. You do not need the old result ZIPs.

        1. Select **Runtime > Change runtime type > T4 GPU**.
        2. Run the cells in order (or **Run all**).
        3. Upload **unlearning-stage3.zip** when asked.
        4. Allow access to the Colab secret **HF_TOKEN**, or enter it in the hidden prompt.
        5. Download **stage3-results.zip** and share it for review, even if a check needs attention.

        Setup and model download may take several minutes. The checks are a short run; they do not start Stage 4.
    """)
    add("markdown", """
        ## 1. Upload the code and verified inputs

        Select `dist/unlearning-stage3.zip` from your project folder.
        We use a separate Colab folder for this stage. The saved Stage 2 results are read-only inputs.
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
        if len(uploaded) != 1 or 'unlearning-stage3.zip' not in uploaded:
            raise ValueError('Select the new unlearning-stage3.zip from the project dist folder.')
        project = Path('/content/unlearning_stage3')
        project.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(uploaded['unlearning-stage3.zip'])) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive entries.')
            bundle = json.loads(archive.read('bundle_manifest.json'))
            if set(names) != set(bundle['files']) | {'bundle_manifest.json'}:
                raise ValueError('Archive contents do not match the bundle manifest.')
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\\\' in name:
                    raise ValueError('Unexpected archive path.')
                target = (project / name).resolve()
                if project.resolve() not in target.parents:
                    raise ValueError('Archive path is outside the project.')
                content = archive.read(name)
                if name != 'bundle_manifest.json' and hashlib.sha256(content).hexdigest() != bundle['files'][name]:
                    raise ValueError('Archive checksum mismatch: ' + name)
                if name.startswith('inputs/') and target.exists() and target.read_bytes() != content:
                    raise ValueError('Existing inputs differ. Start a fresh Colab runtime.')
            # Validate everything before writing anything.
            for name in names:
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        del uploaded
        os.chdir(project)
        print('Stage 3 code and verified inputs ready.')
        print('Reviewed baseline:', bundle['baseline_fingerprint'])
    """)
    add("markdown", """
        ## 2. Install and run local correctness checks in Colab

        These tests use a tiny random model. They check the code, not research accuracy.
        This step also checks that your runtime has a GPU and the packaged inputs match the reviewed baseline.
    """)
    add("code", """
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--only-binary=:all:',
                        '-r', 'requirements-colab.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
        subprocess.run([sys.executable, '-m', 'unlearning', 'doctor',
                        '--output', 'outputs/stage3_environment.json'], check=True)
        environment = json.loads(Path('outputs/stage3_environment.json').read_text())
        if not environment['cuda_available']:
            raise RuntimeError('Select a T4 GPU runtime before continuing.')
        subprocess.run([sys.executable, '-c',
            "from unlearning.baseline import read_settings; "
            "from unlearning.data import read_config; "
            "from unlearning.intervention_checks import verified_inputs; "
            "verified_inputs(read_config('configs/experiment.json'), read_settings('configs/baseline_prefill.json'), "
            "'inputs/data/prepared/full', 'inputs/outputs/stage2/prefill'); print('Verified Stage 3 inputs.')"], check=True)
    """)
    add("markdown", """
        ## 3. Run the real-model checks

        The model and answer format stay the same as the reviewed baseline. All model weights stay frozen.
        The gradient tells us how the correct answer's score responds to a small change in one layer's contribution.
        We compare it with actually making a small change, in both directions.

        This cell records a new attempt each time. Download its results before disconnecting.
        If a check fails, review it before changing the settings or starting localization.
    """)
    add("code", """
        import getpass
        from datetime import datetime, timezone
        import uuid
        from google.colab import userdata

        try:
            os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN').strip()
        except Exception:
            os.environ['HF_TOKEN'] = getpass.getpass('Hugging Face read token (hidden): ').strip()
        if not os.environ['HF_TOKEN']:
            raise RuntimeError('No token supplied.')
        attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
        result_dir = Path('outputs/stage3') / attempt
        completed = subprocess.run([sys.executable, '-m', 'unlearning', 'intervention-check',
            '--settings', 'configs/baseline_prefill.json',
            '--data', 'inputs/data/prepared/full', '--baseline', 'inputs/outputs/stage2/prefill',
            '--output', str(result_dir)])
        if not (result_dir / 'summary.json').exists():
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / 'summary.json').write_text(json.dumps({
                'stage': 3, 'status': 'review_required', 'exit_code': completed.returncode,
                'note': 'The command stopped before writing its report. Read the error above.',
                'final_test_evaluated': False, 'is_research_result': False}), encoding='utf-8')
        print('Checks finished. Read the summary and download the results below.')
    """)
    add("markdown", """
        ## 4. Read the result and download it

        **passed** means the technical checks passed and the measured compute looks feasible.
        **review_required** means we need to inspect a check or the compute estimate before continuing.
        Neither status says that selective unlearning works; that is what later experiments will test.

        After downloading, share `stage3-results.zip`. You can then disconnect the Colab runtime.
    """)
    add("code", """
        summary = json.loads((result_dir / 'summary.json').read_text())
        print('Stage 3:', summary['status'])
        for name, check in summary.get('checks', {}).items():
            print('  ' + name + ': ' + check['status'])
        if 'failure' in summary:
            print(json.dumps(summary['failure'], indent=2))
        forecast = summary.get('runtime_forecast', {})
        if forecast:
            print('Suggested profile:', forecast['recommended_profile'])
            for name, value in forecast['profiles'].items():
                print(name + ': about ' + str(round(value['estimated_gpu_hours'], 2)) + ' GPU hours (estimate)')
        print(summary.get('next_action', summary.get('note', 'Share this result for review.')))

        archive_path = Path('/content/stage3-results.zip')
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(result_dir.rglob('*')):
                if path.is_file():
                    archive.write(path, path.as_posix())
            archive.write('outputs/stage3_environment.json')
            archive.write('bundle_manifest.json')
            for path in sorted(Path('configs').glob('*.json')):
                archive.write(path, path.as_posix())
            for path in sorted(Path('src/unlearning').glob('*.py')):
                archive.write(path, path.as_posix())
            archive.write('STAGE3.md')
        files.download(str(archive_path))
    """)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "03_stage3_intervention.ipynb"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/03_stage3_intervention.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    config = read_config(ROOT / "configs/experiment.json")
    settings = read_settings(ROOT / "configs/baseline_prefill.json")
    _, _, run, records = verified_inputs(config, settings, ROOT / "data/prepared/full", VERIFIED_BASELINE)
    notebook = build_notebook()
    entries = {}
    for name in ("pyproject.toml", "requirements-colab.txt", "README.md", "PLAN.md", "STAGE1.md", "STAGE2.md", "STAGE3.md", "DIAGNOSTIC.md"):
        entries[name] = (ROOT / name).read_bytes()
    for directory, suffix in (("src", ".py"), ("tests", ".py"), ("configs", ".json")):
        for path in sorted((ROOT / directory).rglob("*" + suffix)):
            entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    entries[notebook.relative_to(ROOT).as_posix()] = notebook.read_bytes()
    entries["scripts/build_stage3_bundle.py"] = Path(__file__).read_bytes()
    for path in sorted((ROOT / "data/prepared/full").rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            entries["inputs/" + path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for path in sorted(VERIFIED_BASELINE.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv"):
            entries["inputs/outputs/stage2/prefill/" + path.relative_to(VERIFIED_BASELINE).as_posix()] = path.read_bytes()
    manifest = {"stage": 3, "schema_version": 1, "baseline_fingerprint": run["fingerprint"],
                "baseline_records": len(records), "note": "Verified existing inputs, no new model run.",
                "files": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(entries.items())}}
    entries["bundle_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output = ROOT / "dist/unlearning-stage3.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    print(json.dumps({"notebook": str(notebook), "bundle": str(output), "bytes": output.stat().st_size,
                      "verified_baseline_records": len(records), "baseline_fingerprint": run["fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
