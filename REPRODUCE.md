# Reproducing the experiment

The completed experiment used Colab with a Tesla T4. Do not rerun it just to read the results. Figures and aggregate evidence are in `report/figures/` and `report/evidence/`. See `ASSIGNMENT_COVERAGE.md` for the mapping to the assignment and original evidence locations.

## Environment and access

Use a **new Colab GPU runtime**, not the existing local Python environment. Accept the model's access terms with your Hugging Face account and set a private `HF_TOKEN` secret. Never put a token in a notebook, source file, or ZIP.

Recorded execution: Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, tokenizers 0.23.1, huggingface-hub 1.29.0, accelerate 1.14.0, safetensors 0.8.0, pyarrow 23.0.1. The requirements files pin the observed Python 3.13 stack and preserve a separate older-Python stack. Torch uses the runtime's CUDA-compatible installation. Hardware and numerical-library changes may affect results; fingerprints prevent silently mixing incompatible resumed runs.

From the repository root, in the new runtime:

```bash
python -m pip install -r requirements-stage4-colab.txt
python -m pip install -e . --no-deps
python -m unlearning doctor --output outputs/stage1/environment.json
python -m unlearning access-check
python -m unlearning smoke
```

Use the existing notebook secret-loading cell to put the token into the runtime environment. The install commands above are instructions for a fresh runtime.

## Portable fresh execution

These commands explicitly set input paths; they do not require the author's dated verified folders or notebook bundles. Model and data revisions, sample seed, subjects, k, and strength grid are fixed in `configs/experiment.json`. Keep source and configurations unchanged through a resumed run.

```bash
python -m unlearning prepare --profile full --output data/prepared/full
python -m unlearning verify --data data/prepared/full
python -m unlearning baseline --settings configs/baseline_prefill.json --data data/prepared/full --output outputs/stage2/prefill
python -m unlearning intervention-check --data data/prepared/full --baseline outputs/stage2/prefill --output outputs/stage3/check
python -m unlearning localize --data data/prepared/full --baseline outputs/stage2/prefill --stage3 outputs/stage3/check --output outputs/stage4/full
python -m unlearning sweep --split development --data data/prepared/full --baseline outputs/stage2/prefill --stage4 outputs/stage4/full --output outputs/stage5/development
python -m unlearning sweep --split test --data data/prepared/full --baseline outputs/stage2/prefill --stage4 outputs/stage4/full --development outputs/stage5/development --output outputs/stage5/test
```

Inspect each stage's summary before continuing. Stage 3 must pass before localization. Development must finish before test; do not bypass its operating-point decision. The reviewed answer-prefix evaluator is used directly here. The historical original-prompt diagnostic is documented in `DIAGNOSTIC.md`; it is not a prerequisite to reproduce the final protocol.

Create the Stage 5 archive from the repository root using a Python cell:

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

with ZipFile('stage5-results.zip', 'w', compression=ZIP_DEFLATED) as archive:
    for phase in ('development', 'test'):
        for path in sorted((Path('outputs/stage5') / phase).rglob('*')):
            if path.is_file():
                archive.write(path, path.as_posix())
```

Then run the frozen controls:

```bash
python -m unlearning robustness --data data/prepared/full --baseline outputs/stage2/prefill --stage4 outputs/stage4/full --stage5-archive stage5-results.zip --output outputs/stage6/full
```

The original run yielded 1,312 Stage 4 records, 25,088 new Stage 5 predictions, and 2,880 new Stage 6 predictions. Stage 6 reuses 2,304 primary-prompt references. Save and download progress archives before a Colab runtime disconnects. `--max-new` limits an execution segment where supported; rerun the same command to continue. Do not edit a run's metadata to force resumption after changing the source, environment, or configuration.

## Offline reports and software checks

To inspect the delivered run, unpack the source ZIP into a new folder, then unpack the evidence ZIP into that same folder. The large Stage 5 and 6 results remain inside `archives/`. Restore only their result files with this Python cell from the new folder; archived code is not executed:

```python
from pathlib import Path
from zipfile import ZipFile

root = Path.cwd().resolve()
for stage in (5, 6):
    with ZipFile(root / f'archives/stage{stage}-results.zip') as archive:
        for name in archive.namelist():
            if not name.startswith(f'outputs/stage{stage}/') or name.endswith('/'):
                continue
            target = (root / name).resolve()
            if root not in target.parents:
                raise ValueError('Unexpected archive path')
            content = archive.read(name)
            if target.exists() and target.read_bytes() != content:
                raise ValueError('Existing result differs; use a clean folder')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
```

These commands process existing raw records without loading model weights:

```bash
python -m unlearning baseline-report --output outputs/stage2/prefill
python -m unlearning stage4-report --output outputs/stage4/full
python -m unlearning stage5-report --output outputs/stage5/development
python -m unlearning stage5-report --output outputs/stage5/test
python -m unlearning stage6-report --output outputs/stage6/full
python -m unittest discover -s tests
```

The existing suite passed 63 tests on the local checked environment. Stage verifier scripts perform additional archive and provenance checks; some intentionally depend on the original project's verified prior-stage inputs. They are audit helpers, not required steps for the portable fresh execution above. Do not execute code found in result archives to inspect results.

## Figures and delivery

With `matplotlib` available, the source delivery package includes enough aggregate evidence to rebuild the figures without a GPU:

```bash
python scripts/plot_final_report.py
```

The plotting script uses the numeric tables in `report/evidence/`. Their SHA256 hashes are recorded alongside the tables.

The source ZIP includes code, tests, configs, notebooks, figures, aggregate evidence, and documentation. The separate evidence ZIP preserves original results and prepared-data metadata needed to inspect this particular run. Neither package includes model weights, private credentials, caches, or a Python environment. Dataset and model access remain subject to their respective upstream access terms. Download both packages from [release v1.0.0](https://github.com/LinaGolan/unlearning_task/releases/tag/v1.0.0).
