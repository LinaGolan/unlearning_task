"""Build the Stage 2 Colab entry point; the experiment lives in the Python package."""

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def build_stage2_notebook():
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "id": "stage2-{:02d}".format(len(cells)),
                "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
        # Stage 2: Measure the unchanged model

        We will score **128 forget + 128 retain development questions** with the same Llama model.
        We will save each answer, accuracy with uncertainty, and runtime measurements.
        No layers are changed. The final test questions stay reserved for later.

        1. Select **Runtime > Change runtime type > T4 GPU**.
        2. Upload `unlearning-stage2.zip` when asked below.
        3. Have your successful Stage 1 results ZIP ready. A fresh runtime will ask for it.
        4. Enable this notebook's access to your private Colab secret **HF_TOKEN**.
        5. Run the cells in order, then download and share `stage2-results.zip`.

        If a runtime disconnects, upload the latest Stage 2 results ZIP instead of Stage 1 in step 3.
        Saved answers are reused when the code, settings, data, and environment match.
        A new runtime may need to download the model again. Do not change the model based on partial results.
    """)
    add("markdown", """
        ## 1. Upload the Stage 2 source bundle

        Choose the new `dist/unlearning-stage2.zip` from the project folder on your computer.
        This updates the experiment code. It does not erase existing prepared data or result files.
    """)
    add("code", """
        import io
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import zipfile
        from google.colab import files

        uploaded = files.upload()
        name = 'unlearning-stage2.zip'
        if name not in uploaded:
            raise ValueError('Please select unlearning-stage2.zip from the project dist folder.')
        project = Path('/content/unlearning_task')
        project.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(uploaded[name])) as archive:
            for entry in archive.infolist():
                target = (project / entry.filename).resolve()
                if project.resolve() not in target.parents:
                    raise ValueError('Unexpected archive path.')
            archive.extractall(project)
        del uploaded
        os.chdir(project)
        print('Stage 2 code ready.')
    """)
    add("markdown", """
        ## 2. Install and check the evaluator

        The tests use invented questions and a tiny random model. They check answer labels,
        padding, answer likelihoods, safe resuming, and unchanged model weights.
        These test results are not research accuracy measurements.
    """)
    add("code", """
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--only-binary=:all:',
                        '-r', 'requirements-colab.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)

        def run(*args, required=True):
            result = subprocess.run([sys.executable, '-m', 'unlearning', *args])
            if required and result.returncode:
                raise RuntimeError('The command failed. Read the message above; saved predictions are preserved.')
            return result.returncode

        run('doctor', '--output', 'outputs/stage2_environment.json')
        environment = json.loads(Path('outputs/stage2_environment.json').read_text())
        if not environment['cuda_available']:
            raise RuntimeError('Select a GPU runtime before continuing.')
    """)
    add("markdown", """
        ## 3. Restore the verified data or resume a saved run

        On a fresh runtime, select your successful `stage1-results (3).zip` (its name can vary).
        To resume interrupted work, select the latest `stage2-results.zip` instead.
        If this runtime already has your data, the cell reuses it.

        Set `RESTORE_RESULTS = True` if you need to restore saved results into a runtime that already has data.
        Existing files are reused only when identical; conflicting files are never silently replaced.
    """)
    add("code", """
        RESTORE_RESULTS = False
        if RESTORE_RESULTS or not Path('data/prepared/full/manifest.json').exists():
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError('Select exactly one Stage 1 or Stage 2 results ZIP.')
            payload = next(iter(uploaded.values()))
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for entry in archive.infolist():
                    if entry.is_dir() or not entry.filename.endswith(('.json', '.jsonl')):
                        continue
                    if not entry.filename.startswith(('data/prepared/full/', 'outputs/stage2/full/')):
                        continue
                    target = (project / entry.filename).resolve()
                    if project.resolve() not in target.parents:
                        raise ValueError('Unexpected archive path.')
                    content = archive.read(entry)
                    if target.exists() and target.read_bytes() != content:
                        raise ValueError('Conflicting saved file: ' + entry.filename + '. Use a fresh runtime.')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
            del uploaded, payload
        run('verify', '--data', 'data/prepared/full')
    """)
    add("markdown", """
        ## 4. Run the development baseline

        This cell reloads **HF_TOKEN in this notebook**, checks access, and runs the baseline
        in a process that inherits that same token. It never prints the token.

        Each question gets a prompt with A-D options. We read the scores for the next answer letter
        and choose the largest. We also save the correct answer's probability among these four choices.
        This probability is not the probability of producing a correct free-text explanation.

        The default processes all remaining development questions. To use short segments, set
        `MAX_NEW = 64`; download results after each segment, then rerun this cell to continue.
        Already-saved questions are not evaluated again. Model loading and numerical checks repeat.

        Near the end, four development prompts measure the cost of a forward and backward pass.
        We do not train the model. This is a runtime estimate; actual layer-gradient checks come in Stage 3.
    """)
    add("code", """
        import getpass
        from google.colab import userdata

        MAX_NEW = None  # Use 64 for shorter segments; None runs all remaining questions.
        try:
            os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN').strip()
        except Exception:
            os.environ['HF_TOKEN'] = getpass.getpass('Hugging Face read token (hidden): ').strip()
        if not os.environ['HF_TOKEN']:
            raise RuntimeError('No token supplied.')
        run('access-check', '--output', 'outputs/stage2_access.json')
        args = ['baseline', '--data', 'data/prepared/full', '--output', 'outputs/stage2/full']
        if MAX_NEW is not None:
            args.extend(['--max-new', str(MAX_NEW)])
        exit_code = run(*args, required=False)
        if exit_code:
            print('The run stopped. Download the evidence below so we can review it.')
        elif json.loads(Path('outputs/stage2/full/summary.json').read_text())['status'] == 'partial':
            print('Segment saved. Download a backup, then rerun this cell to continue.')
        else:
            print('Development baseline saved. Review it below; do not change models or start interventions yet.')
    """)
    add("markdown", """
        ## 5. Read the result

        **Accuracy** is the fraction answered correctly. A **95% interval** describes uncertainty from
        the limited sample; it does not cover every possible difference in subjects or prompt wording.

        We agreed to require at least **35% forget accuracy**, **40% retain accuracy**, and a lower
        interval bound above **25%** for each. A failed rule prompts a review; it is not a reason to
        quietly change the sample or claim an unlearning result.

        The runtime forecast compares full and reduced profiles against six GPU hours. It is provisional:
        Stage 3 must check the actual intervention/gradient implementation before we fix the experiment size.
    """)
    add("code", """
        if Path('outputs/stage2/full/run.json').exists():
            run('baseline-report', '--output', 'outputs/stage2/full')
            summary = json.loads(Path('outputs/stage2/full/summary.json').read_text())
            print('Progress:', summary['evaluated'], '/', summary['expected'])
            for role, result in summary.get('datasets', {}).items():
                low, high = result['accuracy_interval_95']
                print(f"{role}: {result['accuracy']:.1%} ({result['correct']}/{result['count']}), 95% interval {low:.1%}-{high:.1%}")
            if 'ability_gate' in summary:
                print('Baseline check:', summary['ability_gate']['status'])
                print('Compute recommendation:', summary.get('runtime_forecast', {}).get('recommended_profile', 'review required'))
            print(summary['next_action'])
        else:
            print('No baseline run was created. Review the setup/access error and download the available evidence.')
    """)
    add("markdown", """
        ## 6. Download the evidence

        Run this even if the baseline stopped early. Keep the ZIP before leaving Colab.
        It contains data, settings, code, individual predictions and checks; no token, model weights, or caches.
        Share the ZIP for review. **Stage 2 is complete only after the real baseline evidence is reviewed.**
    """)
    add("code", """
        evidence = Path('/content/stage2-results.zip')
        with zipfile.ZipFile(evidence, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for folder in (Path('data/prepared/full'), Path('outputs/stage2'), Path('configs'), Path('src/unlearning')):
                if folder.exists():
                    for path in sorted(folder.rglob('*')):
                        if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.py'):
                            archive.write(path, path.as_posix())
            for name in ('outputs/stage2_access.json', 'outputs/stage2_environment.json', 'requirements-colab.txt'):
                path = Path(name)
                if path.exists():
                    archive.write(path, path.as_posix())
        files.download(str(evidence))
    """)
    notebook = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks" / "02_stage2_baseline.ipynb"
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build_stage2_notebook())
