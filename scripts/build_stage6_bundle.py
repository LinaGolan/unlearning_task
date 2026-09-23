"""Package the fixed Stage 6 controls with verified prior-stage evidence."""

import hashlib
import json
from pathlib import Path
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from unlearning.baseline import read_settings
from unlearning.data import file_digest, read_config
from unlearning.robustness_inputs import read_robustness_settings, robustness_inputs

BASELINE = ROOT / "outputs/stage2/verified_2026-09-20/prefill/outputs/stage2/prefill"
STAGE4 = ROOT / "outputs/stage4/verified_2026-09-20/outputs/stage4/full"
STAGE5 = ROOT / "outputs/stage5/verified_2026-09-21/original_results.zip"


def build_notebook():
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": "stage6-{:02d}".format(len(cells)), "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """
        # Stage 6: General biology and prompt robustness

        Stage 5 did not establish selective forgetting. The selected intervention harmed retain performance
        on the final test and showed no clear advantage over the random-pair average.
        These controls help explain that finding. They do not select new layers or a new strength.

        **Fixed:** the same model, eight layer pairs, and development-selected strength **0.75**.

        - **General biology:** 64 separate MMLU biology questions, with an unchanged baseline and all eight pairs.
        - **Alternative instruction:** a fixed subset of 128 forget + 128 retain test questions, again with a baseline and all pairs.
        - **Matched comparison:** reuse Stage 5 predictions on those exact questions under the original instruction.
        - **Answer-letter checks:** inspect whether predictions shift toward particular letters.

        Each intervention is compared with the baseline under the **same instruction**.
        There are **2,880 new predictions**, much fewer than Stage 5. Allow roughly **10-20 minutes of model work**,
        plus setup and model loading; this estimate uses Stage 5 timings and may vary on Colab.

        **Run:** choose a **T4 GPU**, select **Run all**, upload **unlearning-stage6.zip**, and allow `HF_TOKEN` access.
        The bundle includes the verified Stage 5 archive. You do not need to upload earlier results separately.
        Leave the optional restore off for a first run. Download **stage6-results.zip** at the end.

        The notebook sets up imports in the active kernel automatically. No separate import-fix cell is needed.
    """)
    add("markdown", "## 1. Upload the Stage 6 bundle\nChoose `dist/unlearning-stage6.zip` from your computer.")
    # Reuse the checked upload/integrity cell, without modifying the previous notebook.
    previous = json.loads((ROOT / "notebooks/05_stage5_sweep.ipynb").read_text(encoding="utf-8"))
    upload = next(c["source"] for c in previous["cells"] if c["cell_type"] == "code" and "bundle_manifest.json" in c["source"] and "files.upload()" in c["source"])
    add("code", upload.replace("stage5", "stage6").replace("Stage 5", "Stage 6"))
    add("markdown", """
        ## 2. Check the environment and verified inputs

        The software tests use tiny random CPU models and synthetic questions, not research evidence.
        Input verification checks the existing Stage 5 records and decision without rerunning that experiment.
    """)
    add("code", """
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--only-binary=:all:',
                        '-r', 'requirements-stage4-colab.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
        import importlib
        source_dir = str((project / 'src').resolve())
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        importlib.invalidate_caches()
        from unlearning.robustness_report import robustness_report
        print('Project imports work in the notebook kernel.')
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
        subprocess.run([sys.executable, '-m', 'unlearning', 'doctor', '--output', 'outputs/stage6_environment.json'], check=True)
        environment = json.loads(Path('outputs/stage6_environment.json').read_text())
        if not environment['cuda_available']:
            raise RuntimeError('Select a T4 GPU runtime before continuing.')
        from unlearning.baseline import read_settings
        from unlearning.data import read_config
        from unlearning.robustness_inputs import read_robustness_settings, robustness_inputs
        prior = robustness_inputs(read_config('configs/experiment.json'), read_settings('configs/baseline_prefill.json'),
            read_robustness_settings('configs/robustness.json'), 'inputs/data/prepared/full',
            'inputs/outputs/stage2/prefill', 'inputs/stage4', 'inputs/stage5-results.zip')
        print('Verified fixed strength:', prior['decision']['alpha'])
        print('Biology questions:', len(prior['biology_examples']), '| Prompt-control questions:', len(prior['prompt_examples']))
        del prior
    """)
    add("markdown", """
        ## 3. Optional: restore an interrupted Stage 6 run

        Leave `RESTORE_RESULTS = False` on your first run. After a disconnection, use the same bundle,
        run steps 1-2, change this to `True`, and upload your latest **stage6-results.zip**.
        Then continue with steps 4-5. Saved predictions will be verified and reused.
        A code, settings, or runtime-version mismatch must be reviewed rather than bypassed.
    """)
    add("code", """
        RESTORE_RESULTS = False
        result_dir = Path('outputs/stage6/full')
        if RESTORE_RESULTS:
            from unlearning.result_archives import restore_stage6_results
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError('Select one saved Stage 6 results ZIP.')
            print(restore_stage6_results(next(iter(uploaded.values())), project))
            del uploaded
            report = robustness_report(result_dir, make_plots=False)
            print(report['status'], report['new_predictions_saved'], '/', report['expected_new_predictions'])
            if report['last_attempt'].get('failure'):
                print('Review the prior failure:', report['last_attempt']['failure'])
    """)
    add("markdown", """
        ## 4. Run the controls

        `MAX_NEW = None` completes all remaining work. Use `MAX_NEW = 512` for shorter segments.
        The run saves each prediction. Biology runs first (576 predictions), followed by the alternative
        instruction (2,304 predictions). The 2,304 matching original-instruction predictions are reused.

        No optimization, training, or selection takes place. The model is loaded once and shared by the two evaluators.
        A completed run only regenerates its report. If you interrupt a run, call `download_backup()`
        in a separate cell before disconnecting, then resume from that backup later.
    """)
    add("code", """
        MAX_NEW = None
        import getpass
        from google.colab import userdata

        def download_backup():
            archive_path = Path('/content/stage6-results.zip')
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
                for directory in ('outputs/stage6', 'configs', 'src/unlearning'):
                    for path in sorted(Path(directory).rglob('*')):
                        if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.png', '.pdf', '.py'):
                            archive.write(path, path.as_posix())
                for name in ('outputs/stage6_environment.json', 'bundle_manifest.json', 'STAGE6.md', 'STAGE5_RESULTS.md',
                             'requirements-colab.txt', 'requirements-stage4-colab.txt'):
                    if Path(name).exists():
                        archive.write(name)
            files.download(str(archive_path))
            print('Download stage6-results.zip before disconnecting.')

        saved = robustness_report(result_dir, make_plots=False) if (result_dir / 'run.json').exists() else {}
        if saved.get('new_predictions_saved') == saved.get('expected_new_predictions') and saved.get('expected_new_predictions'):
            args = [sys.executable, '-m', 'unlearning', 'stage6-report', '--output', str(result_dir)]
        else:
            if not os.environ.get('HF_TOKEN'):
                try:
                    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN').strip()
                except Exception:
                    os.environ['HF_TOKEN'] = getpass.getpass('Hugging Face read token (hidden): ').strip()
            if not os.environ.get('HF_TOKEN'):
                raise RuntimeError('No token supplied.')
            args = [sys.executable, '-m', 'unlearning', 'robustness', '--output', str(result_dir)]
            if MAX_NEW is not None:
                args += ['--max-new', str(MAX_NEW)]
        completed = subprocess.run(args)
        if completed.returncode:
            print('The run needs attention. Keep the error and download a backup in step 5.')
        else:
            print('Segment finished. Review the summary and download the results in step 5.')
    """)
    add("markdown", """
        ## 5. Read the results and download a backup

        `complete` means the controls finished, not that selective forgetting succeeded.
        `partial` means more predictions remain; rerun step 4 with the same settings.
        `review_required` means a technical issue needs attention. Download the saved evidence even then.

        **Biology:** a drop may indicate broader biology damage. Different dataset difficulty limits comparisons with WMDP.
        **Prompt control:** compare intervention drops relative to each instruction's own baseline, on the same question IDs.
        **Letters:** a strong letter preference may help explain performance changes, but is not proof of a mechanism.
        The saved tables include paired intervals, individual random pairs, and joint comparisons across both instructions.

        Download **stage6-results.zip** and share it for review. The final report comes after reviewing these results.
    """)
    add("code", """
        from IPython.display import display, Image
        if (result_dir / 'summary.json').exists():
            summary = json.loads((result_dir / 'summary.json').read_text())
            print('Stage 6:', summary['status'], summary['new_predictions_saved'], '/', summary['expected_new_predictions'])
            print('Frozen strength:', summary['decision']['alpha'])
            for row in summary.get('selective_results', []):
                print('{} / {}: selective accuracy {:.2f}%, drop {:.2f} percentage points'.format(
                    row['group'], row['role'], 100*row['accuracy'], 100*row['accuracy_drop']))
            for row in summary.get('selective_prompt_comparison', []):
                low, high = [100*x for x in row['change_in_drop_ci95']]
                print('{}: change in selective effect across wording {:.2f} pp; paired 95% interval [{:.2f}, {:.2f}]'.format(
                    row['role'], 100*row['change_in_accuracy_drop'], low, high))
            if summary.get('last_attempt', {}).get('failure'):
                print(summary['last_attempt']['failure'])
            for path in sorted((result_dir / 'figures').glob('*.png')):
                display(Image(filename=str(path), width=950))
        else:
            print('No run report yet. Keep the error above for review.')
        download_backup()
    """)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "06_stage6_robustness.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/06_stage6_robustness.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    verification = json.loads((STAGE5.parent / "verification.json").read_text(encoding="utf-8"))
    if file_digest(STAGE5) != verification["archive_sha256"] or verification["status"] != "verified":
        raise ValueError("The Stage 5 archive differs from reviewed evidence.")
    prior = robustness_inputs(read_config(ROOT / "configs/experiment.json"), read_settings(ROOT / "configs/baseline_prefill.json"),
        read_robustness_settings(ROOT / "configs/robustness.json"), ROOT / "data/prepared/full", BASELINE, STAGE4, STAGE5)
    notebook = build_notebook()
    entries = {}
    for name in ("pyproject.toml", "requirements-colab.txt", "requirements-stage4-colab.txt", "README.md", "PLAN.md",
                 "STAGE1.md", "STAGE2.md", "STAGE3.md", "STAGE4.md", "STAGE5.md", "STAGE5_RESULTS.md", "STAGE6.md", "DIAGNOSTIC.md"):
        entries[name] = (ROOT / name).read_bytes()
    for directory, suffix in (("src", ".py"), ("tests", ".py"), ("configs", ".json")):
        for path in sorted((ROOT / directory).rglob("*" + suffix)):
            entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for path in (notebook, ROOT / "notebooks/05_stage5_sweep.ipynb", Path(__file__).resolve()):
        entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for directory, destination, suffixes in ((ROOT / "data/prepared/full", "inputs/data/prepared/full", (".json", ".jsonl")),
            (BASELINE, "inputs/outputs/stage2/prefill", (".json", ".jsonl", ".csv")),
            (STAGE4, "inputs/stage4", (".json",))):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                entries[destination + "/" + path.relative_to(directory).as_posix()] = path.read_bytes()
    entries["inputs/stage5-results.zip"] = STAGE5.read_bytes()
    manifest = {"stage": 6, "schema_version": 1, "profile": "full", "stage5_archive_sha256": prior["archive_sha256"],
        "stage5_test_fingerprint": prior["runs"]["test"]["fingerprint"], "decision_fingerprint": prior["decision"]["decision_fingerprint"],
        "biology_questions": len(prior["biology_examples"]), "prompt_questions": len(prior["prompt_examples"]),
        "new_predictions": (len(prior["biology_examples"]) + len(prior["prompt_examples"])) * (1 + len(prior["runs"]["test"]["context"]["selections"])),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(entries.items())}}
    entries["bundle_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output = ROOT / "dist/unlearning-stage6.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name.endswith(".zip") else zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    print(json.dumps({"notebook": str(notebook), "bundle": str(output), "bytes": output.stat().st_size,
                      "new_predictions": manifest["new_predictions"], "fixed_alpha": prior["decision"]["alpha"]}, indent=2))


if __name__ == "__main__":
    main()
