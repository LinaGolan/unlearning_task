# Stage 1 progress

## Current status

Stage 1 is complete. The successful Colab evidence was reviewed on September 19, 2026.
The pinned recommended model loaded on a Tesla T4 and completed its forward-pass check;
the returned data also passed local verification.

Assignment coverage: Section 2 (model and datasets) and the setup/reproducibility parts of Section 13.
Stage 2 implementation is ready; its real baseline run is pending. See [STAGE2.md](STAGE2.md).

## Verified locally

- The model and dataset revisions are pinned to exact commits.
- All ten real public dataset files were downloaded and parsed successfully.
- The full profile contains 128 localization, 128 development, and 256 test questions for each main dataset.
- There are 64 separate general-biology control questions.
- Counts, checksums, provenance, subject balance, and disjoint splits passed verification.
- Duplicate filtering excluded 18 retain-source questions and one biology-control source question before sampling.
  No forget-source questions were excluded. These counts describe source-pool filtering, not removed predictions.
- Twelve standard-library tests passed, covering deterministic selection, disjoint splits, reduced/full consistency,
  changed settings, invalid answers, insufficient data, and tampered files.
- The reduced profile was also prepared and verified on the real data: 64 localization, 64 development,
  and 128 test questions per main dataset, plus the same 64 biology-control questions.
- A tiny randomly initialized Llama model completed a CPU forward pass with finite answer scores.
  This confirms the basic runtime path, not the pretrained model's behavior.
- The Colab notebook passed notebook-format and code-syntax checks. The source archive excludes
  result directories, cached data, model weights, and credentials.

## Verified successful Colab run

Evidence: `outputs/stage1/colab_verified_2026-09-19/`, including the original archive
and its extracted reports. Source: the user-provided `stage1-results (3).zip`.
Archive SHA-256: `dd7a6561aabd022b6d37796ef6bdeccd802bbff167be7015967d09f1f0b4da5c`.

- Authentication and access to the pinned model configuration passed.
- Llama-3.2-1B-Instruct loaded in float32: 1,235,814,400 parameters and 16 layers.
- The harmless question received the correct answer B; its probability among A-D was 60.4%.
- The four answer labels were distinct single tokens (IDs 32, 33, 34, 35).
- The 76-token forward pass took 0.780 seconds; peak PyTorch GPU allocation was
  4,992,524,800 bytes (4.65 GiB), on a T4 with 14.56 GiB total memory.
- This is one short forward pass, not a warmed-up throughput measurement or a guarantee
  that longer inputs and later gradient computations will fit. Benchmark those in the next stages.
- All 1,088 returned questions passed checksums, provenance, subject balance, and split-separation
  verification. The archive configuration matches the current project configuration.
- The tiny-model check passed. Python and library versions are saved in the reports.

## Next stage

Stage 2 implements the multiple-choice evaluator and a Colab notebook for unchanged-model performance
and runtime on representative inputs. Its real Colab run remains pending. No baseline or research intervention has run yet.
The smoke-test prompt contains an automatically inserted date; Stage 2 should fix and record the
prompt date so reruns on different days use identical text.

## Evidence from the first Colab run

The downloaded `stage1-results.zip` reports a Tesla T4 GPU with about 15 GiB of device memory,
Python 3.13.15, and a passing tiny-model check. The actual tokenizer failure was wrapped as an `OSError`;
the original report did not retain the HTTP status. A later failed report recorded HTTP 401;
the successful run above supersedes those failures after the token was refreshed in the main notebook.

The runtime used newer dependencies than the original requirements file. The old PyArrow 17 release has
no Python 3.13 wheel, and the original installation cell did not stop when pip failed. The installer now
uses Python-version-specific pins and checks the process exit status. The new access diagnostic distinguishes
token authentication from permission to download the model and avoids exposing private error details.

No WMDP accuracy, retain accuracy, localization scores, or intervention results are available yet.

## What to understand at this stage

- **Model revision:** the exact saved version of the model, so later runs use the same weights.
- **Localization split:** questions used later to choose layers.
- **Development split:** questions used for setup decisions and choosing intervention strength.
- **Test split:** reserved questions used only after those choices are fixed.
- **Manifest:** the experiment's data record, including settings, versions, counts, and file checksums.
- **Smoke check:** a small execution check that proves the pipeline runs; it does not measure model quality.
