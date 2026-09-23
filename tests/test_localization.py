"""Stage 4 data boundaries, signed rankings, paired controls, and safe resuming."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unlearning.baseline import read_settings, run_baseline
from unlearning.data import digest, file_digest, save_prepared, write_json
from unlearning.intervention_checks import development_examples
from unlearning.interventions import gate_gradients
from unlearning.localization import (localization_examples, read_localization_settings, read_records,
    run_localization, select_control_examples, verified_stage3)
from unlearning.localization_analysis import (agreement, assign_halves, correlation, localization_summary,
    ordered_layers, random_layer_sets, ranks, spearman)
from unlearning.localization_report import stage4_report
from unlearning.result_archives import restore_stage4_results
from test_data import fixtures, ROOT
from test_evaluation import HAS_TORCH
from test_interventions import prefill_evaluator


def fixture_stage3(path, config, settings, baseline_run):
    """Synthetic prior-stage metadata for unit tests only, never research evidence."""
    names = ("interventions.py", "evaluation.py", "data.py", "baseline.py")
    context = {"config": config, "settings": settings, "baseline_fingerprint": baseline_run["fingerprint"],
               "profile": "full", "source_hashes": {name: file_digest(ROOT / "src/unlearning" / name) for name in names}}
    run = {"context": context, "fingerprint": digest(context)}
    summary = {"run_fingerprint": run["fingerprint"], "status": "passed", "final_test_evaluated": False,
        "layers_selected": False, "runtime_forecast": {"recommended_profile": "full"},
        "checks": {name: {"status": "passed"} for name in ("baseline_replay", "cleanup_and_frozen_weights",
             "compute_feasibility", "disabled", "evaluator", "full_strength", "gradients", "zero_strength")}}
    write_json(path / "run.json", run)
    write_json(path / "summary.json", summary)


class AnalysisTests(unittest.TestCase):
    def test_average_tied_ranks_constant_correlations_and_signed_drops(self):
        self.assertEqual(ranks([4, 1, 1, 2]), [4, 1.5, 1.5, 3])
        self.assertEqual(spearman([1, 2, 3], [6, 5, 4]), -1)
        self.assertIsNone(correlation([2, 2], [1, 4]))
        self.assertIsNone(spearman([1], [3]))
        result = agreement([-.5, 0, .5], [-.5, 0, .5], 1e-6)
        self.assertEqual(result["mae"], 0)
        self.assertEqual(result["pearson"], 1)
        self.assertEqual(result["sign_agreement"], 1)
        self.assertEqual(result["near_zero_pairs_excluded_from_sign"], 1)
        self.assertEqual(agreement([1, -1], [-1, 1], 1e-6)["sign_agreement"], 0)

    def test_signed_means_selectivity_ties_and_all_questions(self):
        config, _ = fixtures()
        protocol = read_localization_settings(ROOT / "configs/localization.json")
        records = []
        for role, values in (("forget", [4, 3, 2, -2]), ("retain", [4, 0, -1, -1])):
            for i in range(8):
                records.append({"id": role + str(i), "role": role, "split": "localization", "subject": role,
                                "gradients": values[:], "correct": i % 2 == 0})
        result = localization_summary(records, records, config, protocol, 4)
        sets = {s["name"]: s["layers"] for s in result["selections"]}
        self.assertEqual(sets["top_forget"], [0, 1])
        self.assertEqual(sets["top_selective"], [1, 2])
        self.assertEqual(sets["bottom_forget"], [3, 2])
        self.assertEqual([r["selectivity_score"] for r in result["layer_scores"]], [0, 3, 3, -1])
        self.assertEqual(result["counts"], {"forget": 8, "retain": 8})
        self.assertTrue(result["ties"]["selectivity"])
        self.assertEqual(result["stability"]["selectivity"]["spearman"], 1)
        self.assertEqual(ordered_layers([-3, -2, -2]), [1, 2, 0])
        with self.assertRaisesRegex(ValueError, "complete localization"):
            localization_summary(records[:-1], records, config, protocol, 4)
        contaminated = [dict(r, split="development") for r in records]
        with self.assertRaisesRegex(ValueError, "only use localization"):
            localization_summary(contaminated, contaminated, config, protocol, 4)

    def test_five_random_pairs_are_unique_and_reproducible(self):
        first = random_layer_sets(16, 2, [101, 202, 303, 404, 505])
        self.assertEqual(first, random_layer_sets(16, 2, [101, 202, 303, 404, 505]))
        self.assertEqual(len({tuple(r["layers"]) for r in first}), 5)
        self.assertTrue(all(len(set(r["layers"])) == 2 for r in first))
        with self.assertRaisesRegex(ValueError, "Not enough"):
            random_layer_sets(2, 2, [1, 2])

    def test_subject_balanced_sampling_halves_and_split_boundaries(self):
        config, pools = fixtures()
        protocol = read_localization_settings(ROOT / "configs/localization.json")
        with tempfile.TemporaryDirectory() as tmp:
            save_prepared(config, pools, [], "full", tmp)
            _, local = localization_examples(tmp, config)
            _, development = development_examples(tmp, config)
            sample = select_control_examples(development, config, protocol)
            changed = select_control_examples([dict(e, answer=0, correct=False) for e in reversed(development)], config, protocol)
            self.assertEqual([e["id"] for e in sample], [e["id"] for e in changed])
            self.assertEqual(len(sample), 32)
            self.assertFalse({e["id"] for e in sample} & {e["id"] for e in local})
            self.assertEqual({e["split"] for e in local}, {"localization"})
            for subject in config["datasets"]["retain"]["configs"]:
                self.assertEqual(sum(e["subject"] == subject for e in sample), 2)
            halves = assign_halves(local, config["seed"])
            self.assertEqual(halves, assign_halves(list(reversed(local)), config["seed"]))
            for subject in config["datasets"]["retain"]["configs"]:
                self.assertEqual([sum(e["subject"] == subject and halves[e["id"]] == h for e in local) for h in (0, 1)], [1, 1])
            with self.assertRaisesRegex(ValueError, "development"):
                select_control_examples([dict(e, split="test") for e in development], config, protocol)

    def test_changed_protocol_and_failed_stage3_are_rejected(self):
        config, _ = fixtures()
        settings = read_settings(ROOT / "configs/baseline_prefill.json")
        baseline = {"fingerprint": "synthetic-baseline"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture_stage3(path, config, settings, baseline)
            self.assertTrue(verified_stage3(path, config, settings, baseline))
            summary = json.loads((path / "summary.json").read_text())
            summary["checks"]["gradients"]["status"] = "review_required"
            write_json(path / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verified_stage3(path, config, settings, baseline)
            protocol = read_localization_settings(ROOT / "configs/localization.json")
            protocol["control_strengths"] = [.1, .9]
            write_json(path / "protocol.json", protocol)
            with self.assertRaisesRegex(ValueError, "declared"):
                read_localization_settings(path / "protocol.json")


class ArchiveTests(unittest.TestCase):
    @staticmethod
    def archive(entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, value in entries.items():
                archive.writestr(name, value)
        return buffer.getvalue()

    def test_restore_preserves_identical_results_and_ignores_source_updates(self):
        payload = self.archive({"outputs/stage4/full/run.json": "{}", "outputs/stage4/full/records/one.json": "{}",
                                "src/unlearning/localization.py": "should never be installed from results"})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(restore_stage4_results(payload, tmp)["restored_files"], 2)
            self.assertEqual(restore_stage4_results(payload, tmp)["restored_files"], 2)
            self.assertFalse((Path(tmp) / "src").exists())

    def test_restore_rejects_traversal_and_conflicts_before_writing_any_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            escaped = self.archive({"outputs/stage4/full/run.json": "{}", "outputs/stage4/full/../../escape.json": "{}"})
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                restore_stage4_results(escaped, tmp)
            self.assertFalse((Path(tmp) / "outputs").exists())
            original = self.archive({"outputs/stage4/full/run.json": "{}"})
            restore_stage4_results(original, tmp)
            conflicting = self.archive({"outputs/stage4/full/new.json": "{}", "outputs/stage4/full/run.json": '{"different":true}'})
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                restore_stage4_results(conflicting, tmp)
            self.assertFalse((Path(tmp) / "outputs/stage4/full/new.json").exists())


@unittest.skipUnless(HAS_TORCH, "Requires existing numerical dependencies; no downloads")
class RunnerTests(unittest.TestCase):
    def test_failure_resume_frozen_selections_raw_controls_and_offline_tamper_checks(self):
        evaluator = prefill_evaluator()
        evaluator.settings.update(forget_accuracy_min=0, retain_accuracy_min=0, chance_accuracy=0)
        config, pools = fixtures()
        config["intervention"].update(k=1, random_seeds=[101, 202])
        protocol = read_localization_settings(ROOT / "configs/localization.json")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data, baseline, stage3, output = [base / name for name in ("data", "baseline", "stage3", "resumed")]
            save_prepared(config, pools, [], "full", data)
            run_baseline(config, evaluator.settings, data, baseline, evaluator=evaluator)
            baseline_run = json.loads((baseline / "run.json").read_text())
            fixture_stage3(stage3, config, evaluator.settings, baseline_run)
            baseline_bytes = {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
            calls = []
            def interrupted(ev, example):
                calls.append(example)
                if len(calls) == 6:
                    raise RuntimeError("synthetic interruption, private detail")
                return gate_gradients(ev, example)
            args = (config, evaluator.settings, protocol, data, baseline, stage3)
            with patch("unlearning.localization.gate_gradients", side_effect=interrupted):
                partial = run_localization(*args, output, evaluator=evaluator, make_plots=False)
            self.assertEqual(partial["status"], "review_required")
            self.assertEqual(partial["completed_records"], 5)
            self.assertNotIn("private detail", json.dumps(partial))
            self.assertFalse((output / "selections.json").exists())
            old = {p.name: p.read_bytes() for p in (output / "records").glob("*.json")}
            with patch("unlearning.localization.gate_gradients", wraps=gate_gradients) as measured:
                localized = run_localization(*args, output, max_new=27, evaluator=evaluator, make_plots=False)
            self.assertEqual(measured.call_count, 27)
            self.assertTrue(all(call.args[1]["split"] == "localization" for call in measured.call_args_list))
            self.assertEqual(localized["status"], "partial")
            self.assertTrue(localized["layer_selections_fixed"])
            self.assertFalse(localized["control_a_complete"])
            frozen = (output / "selections.json").read_bytes()
            with patch("unlearning.localization.gate_gradients", wraps=gate_gradients) as measured, patch.object(evaluator, "score", wraps=evaluator.score) as scored:
                completed = run_localization(*args, output, evaluator=evaluator, make_plots=False)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["counts"], {"localization": {"saved": 32, "expected": 32},
                "control_gradient": {"saved": 32, "expected": 32}, "control_intervention": {"saved": 128, "expected": 128}})
            self.assertTrue(all(call.args[1]["split"] == "development" for call in measured.call_args_list))
            scored_real = [e for call in scored.call_args_list for e in call.args[0] if not e["id"].startswith("check-")]
            self.assertEqual({e["split"] for e in scored_real}, {"development"})
            self.assertFalse(completed["final_test_evaluated"])
            self.assertFalse(completed["is_research_result"])
            self.assertEqual((output / "selections.json").read_bytes(), frozen)
            for name, content in old.items():
                self.assertEqual((output / "records" / name).read_bytes(), content)
            self.assertEqual(baseline_bytes, {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob("*") if p.is_file()})
            full = base / "uninterrupted"
            run_localization(*args, full, evaluator=evaluator, make_plots=False)
            def scientific_values(folder):
                run = json.loads((folder / "run.json").read_text())
                return [{k: r[k] for k in ("kind", "id", "prediction", "correct_answer_log_probability", "gradients", "actual_drop", "predicted_drop", "layer", "alpha") if k in r}
                        for r in read_records(folder, run)]
            self.assertEqual(scientific_values(output), scientific_values(full))
            self.assertEqual(stage4_report(output, make_plots=False)["selections"], completed["selections"])
            # A rehashed but mathematically wrong drop must not pass offline review.
            paths = list((output / "records").glob("*.json"))
            target = next(p for p in paths if json.loads(p.read_text())["kind"] == "control_intervention")
            record = json.loads(target.read_text())
            record["actual_drop"] += 1
            record.pop("record_hash")
            record["record_hash"] = digest(record)
            write_json(target, record)
            with self.assertRaisesRegex(ValueError, "score-drop"):
                stage4_report(output, make_plots=False)
            # Frozen selections cannot silently change, even if all records are intact.
            selection = json.loads((full / "selections.json").read_text())
            selection["selections"][0]["layers"] = [99]
            write_json(full / "selections.json", selection)
            with self.assertRaisesRegex(ValueError, "frozen selection"):
                stage4_report(full, make_plots=False)


if __name__ == "__main__":
    unittest.main()
