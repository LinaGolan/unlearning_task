"""Build a portable notebook and a source-only ZIP without publishing anything."""

import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []

    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "id": "stage1-{:02d}".format(len(cells)),
                "source": textwrap.dedent(text).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
        # Stage 1: Setup, data, and the first model check

        This notebook prepares the experiment. It does **not** run the research baseline,
        select layers, or apply interventions.

        1. In Colab, choose **Runtime > Change runtime type > GPU**.
        2. Run the cells below in order. You will upload `unlearning-stage1.zip` from the project `dist` folder.
        3. Have a Hugging Face read token ready in Colab Secrets under `HF_TOKEN`, or use the hidden prompt.
           Never paste the token into a code cell or chat.
        4. Your account must have access to
           [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
        5. Download the results in the last cell, even if the model check fails.

        Expected time: setup and the first model download may take several minutes.
        A successful one-question check is evidence that the pipeline runs, not that the model performs well.
    """)
    add("markdown", """
        ## 1. Upload the project source

        Select `unlearning-stage1.zip`. This contains the code, configuration, tests, and instructions.
        It contains no credentials, model weights, or saved notebook outputs.
    """)
    add("code", """
        import io
        import os
        from pathlib import Path
        import zipfile
        from google.colab import files

        uploaded = files.upload()
        archive_name = 'unlearning-stage1.zip'
        if archive_name not in uploaded:
            raise ValueError('Please select unlearning-stage1.zip from the project dist folder.')
        project = Path('/content/unlearning_task')
        project.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(uploaded[archive_name])) as archive:
            for entry in archive.infolist():
                target = (project / entry.filename).resolve()
                if project.resolve() not in target.parents:
                    raise ValueError('Unexpected path inside the archive.')
            archive.extractall(project)
        del uploaded
        os.chdir(project)
        print('Project ready:', project)
    """)
    add("markdown", """
        ## 2. Install project dependencies

        Colab supplies the GPU-compatible PyTorch installation. We record its exact version in the results.
        Other direct dependencies are pinned, with separate versions for Python 3.13 and older runtimes.
        Installation errors stop this cell. This does not change your local computer.
        The experiment commands run in fresh Python processes, so they see the installed package versions.
    """)
    add("code", """
        import os
        import subprocess
        import sys
        os.chdir('/content/unlearning_task')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        '--only-binary=:all:', '-r', 'requirements-colab.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
    """)
    add("markdown", """
        ## 3. Check the code and runtime

        The data tests use invented questions and need no downloads. The tiny model has random weights;
        it checks that our forward-pass code runs. It is not the research model.
    """)
    add("code", """
        import json
        from pathlib import Path
        import subprocess
        import sys

        def run(*args, required=True):
            result = subprocess.run([sys.executable, *args])
            if required and result.returncode:
                raise RuntimeError('The command failed. Read the error above before continuing.')
            return result.returncode

        run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
        run('-m', 'unlearning', 'doctor')
        run('-m', 'unlearning', 'smoke', '--tiny', '--output', 'outputs/stage1/tiny_smoke.json')
    """)
    add("markdown", """
        ## 4. Prepare and verify the real data

        Download ten small public data files. The model is not involved in this step.
        The full profile reserves 128 localization, 128 development, and 256 final-test questions per main dataset,
        plus 64 general-biology control questions. Duplicate question text is removed before sampling.

        A **manifest** is a record of the selected data, settings, source versions, and file checksums.
        Keep it with the experiment. The final-test questions are reserved for later evaluation;
        do not inspect their answers to choose a method.
    """)
    add("code", """
        run('-m', 'unlearning', 'prepare', '--profile', 'full', '--output', 'data/prepared/full')
        run('-m', 'unlearning', 'verify', '--data', 'data/prepared/full')
        manifest = json.loads(Path('data/prepared/full/manifest.json').read_text())
        print('Data fingerprint:', manifest['request_hash'])
    """)
    add("markdown", """
        ## 5. Provide model access privately

        Having a Hugging Face account is separate from having access to this model.
        Check the [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) while logged in.
        If needed, request access there first. A token alone does not bypass that requirement.

        This cell tries the Colab secret `HF_TOKEN`; if it is absent, it uses a hidden input prompt.
        It does not print the token or save it in the notebook, Git, or result files.
        It then checks the token and downloads one small model configuration file to test access.
        Rerunning this cell reloads the secret, so a corrected token replaces a previously invalid one.
    """)
    add("code", """
        import getpass
        import os
        from google.colab import userdata

        try:
            os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN').strip()
        except Exception:
            os.environ['HF_TOKEN'] = getpass.getpass('Hugging Face read token (hidden): ').strip()
        if not os.environ.get('HF_TOKEN'):
            raise RuntimeError('No token supplied. Add a Colab secret or rerun this cell.')
        run('-m', 'unlearning', 'access-check', required=False)
        access_report = json.loads(Path('outputs/stage1/model_access.json').read_text())
        if access_report['status'] != 'passed':
            print('Fix the reported access issue before running the model cell. You can still download the reports below.')
    """)
    add("markdown", """
        ## 6. Load the recommended model and run one harmless question

        This downloads the pinned Llama-3.2-1B-Instruct weights and uses the GPU.
        It checks the four answer-label tokens and runs one forward pass on a simple arithmetic question.
        We use float32 initially to keep later gradient checks straightforward.

        If this fails, keep the failure report and download the results below. Do not start the main experiments.
    """)
    add("code", """
        access_report = json.loads(Path('outputs/stage1/model_access.json').read_text())
        if access_report['status'] != 'passed':
            raise RuntimeError('Model access has not passed. Follow the message from the previous cell first.')
        model_check_exit = run('-m', 'unlearning', 'smoke', required=False)
        if model_check_exit:
            failure = json.loads(Path('outputs/stage1/model_smoke.json').read_text())
            print(failure.get('error', 'Read model_smoke.json for details.'))
            print('Stage 1 is not complete: the recommended-model check needs attention.')
        else:
            print('Stage 1 checks passed. Stop here so we can review before Stage 2.')
    """)
    add("markdown", """
        ## 7. Download the evidence

        Download this ZIP before leaving Colab. It includes the prepared data and manifest, runtime information,
        and model-check reports. It excludes credentials, caches, and model weights.
        Bring `model_smoke.json` back for review; no screenshots of credentials are needed.
    """)
    add("code", """
        from pathlib import Path
        import zipfile
        from google.colab import files

        evidence = Path('/content/stage1-results.zip')
        with zipfile.ZipFile(evidence, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for folder in (Path('data/prepared'), Path('outputs/stage1')):
                if folder.exists():
                    for path in sorted(folder.rglob('*')):
                        if path.is_file() and path.suffix in ('.json', '.jsonl'):
                            archive.write(path, path.as_posix())
            archive.write('configs/experiment.json', 'configs/experiment.json')
        files.download(str(evidence))
    """)
    notebook = {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks" / "01_stage1_setup.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return path


def build_recovery_notebook():
    """A standalone diagnosis for existing users; no ZIP, GPU, or model weights required."""
    settings = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))["model"]
    instructions = (
        "# Diagnose Llama model access\n\n"
        "This is a small recovery notebook for the Stage 1 tokenizer-download failure. "
        "It requires no project ZIP or GPU and does not download model weights.\n\n"
        "Run the one code cell below. It uses the private Colab secret `HF_TOKEN`, or asks for a token with a hidden prompt. "
        "Do not paste a token into notebook source or chat.\n\n"
        "The check first verifies the token, then requests one small file from the pinned Llama model. "
        "It prints only safe diagnostic fields and downloads `model_access.json`.\n\n"
        "If the token is invalid, replace it in Colab Secrets. If model access is denied, sign in to "
        "[the Llama model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) with the account that owns the token, "
        "check approval, and check the token's read permission for the gated model. "
        "If the check passes, return to the Stage 1 notebook and rerun its private-token and model-check cells.\n"
    )
    code = (ROOT / "src/unlearning/access.py").read_text(encoding="utf-8") + "\n" + textwrap.dedent("""
        import getpass
        import json
        from pathlib import Path
        from google.colab import userdata, files

        try:
            private_token = userdata.get('HF_TOKEN').strip()
        except Exception:
            private_token = getpass.getpass('Hugging Face read token (hidden): ').strip()
    """)
    code += "\nsettings = " + repr(settings) + "\n"
    code += textwrap.dedent("""
        report = diagnose_model_access(settings, token=private_token)
        del private_token
        output = Path('/content/model_access.json')
        output.write_text(json.dumps(report, indent=2) + '\\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
        files.download(str(output))
    """)
    notebook = {"cells": [
        {"cell_type": "markdown", "id": "access-intro", "metadata": {}, "source": instructions},
        {"cell_type": "code", "id": "access-check", "metadata": {}, "execution_count": None,
         "outputs": [], "source": code}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks" / "00_check_model_access.ipynb"
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    from build_stage2_notebook import build_stage2_notebook
    notebook = build_notebook()
    recovery_notebook = build_recovery_notebook()
    stage2_notebook = build_stage2_notebook()
    output = ROOT / "dist" / "unlearning-stage1.zip"
    output.parent.mkdir(exist_ok=True)
    paths = [ROOT / name for name in (
        "pyproject.toml", "requirements-colab.txt", "README.md", "PLAN.md", "STAGE1.md", "STAGE2.md", ".gitignore")]
    for folder in ("src", "tests", "configs", "notebooks", "scripts"):
        paths.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(notebook)
    print(recovery_notebook)
    print(output)
    stage2_output = ROOT / "dist" / "unlearning-stage2.zip"
    stage2_output.write_bytes(output.read_bytes())
    print(stage2_notebook)
    print(stage2_output)


if __name__ == "__main__":
    main()
