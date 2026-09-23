"""Matched wording, outcome-independent selection, recovery, and immutable controls."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unlearning.baseline import run_baseline
from unlearning.data import digest, save_prepared, write_json
from unlearning.localization import read_localization_settings, run_localization
from unlearning.robustness import run_robustness, control_conditions, read_control_records
from unlearning.robustness_inputs import read_robustness_settings, prompt_subset, robustness_inputs
from unlearning.robustness_analysis import analyze_controls
from unlearning.robustness_report import robustness_report
from unlearning.sweep import run_sweep, read_sweep_settings, load_test_examples
from unlearning.result_archives import restore_stage6_results
from test_data import ROOT, fixtures
from test_evaluation import HAS_TORCH
from test_interventions import prefill_evaluator
from test_localization import fixture_stage3
import test_localization


class RobustnessAnalysisTests(unittest.TestCase):
    def test_sampling_is_balanced_and_ignores_outcomes(self):
        config, pools = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            save_prepared(config, pools, [], "full", tmp)
            rows = load_test_examples(tmp)
            sample = prompt_subset(rows, config)
            changed = prompt_subset([dict(e, answer=0, correct=True, prediction="A") for e in reversed(rows)], config)
            self.assertEqual([e["id"] for e in sample], [e["id"] for e in changed])
            self.assertEqual(len(sample), 16)
            for subject in config["datasets"]["retain"]["configs"]:
                self.assertEqual(sum(e["subject"] == subject for e in sample), 1)
            with self.assertRaisesRegex(ValueError, "final-test"):
                prompt_subset([dict(e, split="development") for e in rows], config)

    def test_joint_prompt_pairing_uses_matched_baselines(self):
        protocol = read_robustness_settings(ROOT / "configs/robustness.json")
        protocol["bootstrap_repetitions"] = 100
        selections = [{"name": n, "layers": [0]} for n in ("top_forget", "top_selective", "bottom_forget", "random_101", "random_202")]
        primary = [{"id": role + str(i), "role": role, "subject": role} for role in ("forget", "retain") for i in range(4)]
        c = {"protocol": protocol, "decision": {"alpha": .75}, "selections": selections, "primary_examples": primary,
             "examples": [dict(e, group="alternative") for e in primary] +
                         [{"id": "bio"+str(i), "role": "biology_control", "subject": "bio", "group": "biology"} for i in range(4)]}
        records = []
        for e in [dict(e, group="primary") for e in primary] + c["examples"]:
            for spec in control_conditions(c):
                # Wording changes EVERY condition by the same amount: intervention drop must stay zero.
                correct = e["group"] != "alternative"
                records.append(dict(e, **spec, correct=correct, prediction="A" if correct else "B", correct_answer="A", correct_answer_log_probability=-.5))
        a = analyze_controls(c, records)
        for r in a["matched_prompt_comparison"]:
            self.assertEqual(r["accuracy_change_from_wording"], -1)
            self.assertEqual(r["change_in_accuracy_drop"], 0)
            self.assertEqual(r["change_in_drop_ci95"], [0, 0])
            self.assertEqual(r["prediction_change_from_wording"], 1)
        for r in a["conditions"]:
            self.assertEqual(r["accuracy_drop"], 0)
        with self.assertRaisesRegex(ValueError, "complete controls"):
            analyze_controls(c, records[:-1])

    def test_stage6_restore_does_not_install_source(self):
        payload = test_localization.ArchiveTests.archive({"outputs/stage6/full/run.json": "{}", "src/unlearning/robustness.py": "ignored"})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(restore_stage6_results(payload, tmp)["restored_files"], 1)
            self.assertFalse((Path(tmp) / "src").exists())


@unittest.skipUnless(HAS_TORCH, "Uses existing tiny CPU model only")
class RobustnessRunnerTests(unittest.TestCase):
    def test_end_to_end_archive_provenance_interrupt_resume_and_prompt_isolation(self):
        evaluator = prefill_evaluator()
        evaluator.settings.update(forget_accuracy_min=0, retain_accuracy_min=0, chance_accuracy=0)
        config, pools = fixtures()
        config["intervention"].update(k=1, random_seeds=[101, 202])
        sweep_protocol = read_sweep_settings(ROOT / "configs/sweep.json")
        sweep_protocol["bootstrap_repetitions"] = 100
        protocol = read_robustness_settings(ROOT / "configs/robustness.json")
        protocol["bootstrap_repetitions"] = 100
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, baseline, stage3, stage4, development, test, output = [root / n for n in
                ("data", "baseline", "stage3", "stage4", "development", "test", "controls")]
            save_prepared(config, pools, [], "full", data)
            run_baseline(config, evaluator.settings, data, baseline, evaluator=evaluator)
            fixture_stage3(stage3, config, evaluator.settings, json.loads((baseline / "run.json").read_text()))
            run_localization(config, evaluator.settings, read_localization_settings(ROOT / "configs/localization.json"),
                             data, baseline, stage3, stage4, evaluator=evaluator, make_plots=False)
            run_sweep(config, evaluator.settings, sweep_protocol, data, baseline, stage4, development, evaluator=evaluator, make_plots=False)
            run_sweep(config, evaluator.settings, sweep_protocol, data, baseline, stage4, test, split="test",
                      development_dir=development, evaluator=evaluator, make_plots=False)
            archive_path = root / "stage5.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for phase, folder in (("development", development), ("test", test)):
                    for path in folder.rglob("*.json"):
                        archive.write(path, "outputs/stage5/" + phase + "/" + path.relative_to(folder).as_posix())
            archive_bytes = archive_path.read_bytes()
            args = (config, evaluator.settings, protocol, data, baseline, stage4, archive_path)
            inputs = robustness_inputs(*args)
            self.assertEqual(len(inputs["prompt_examples"]), 16)
            self.assertEqual(len(inputs["primary_records"]), 96)
            first = run_robustness(*args, output, max_new=3, evaluator=evaluator, make_plots=False)
            self.assertEqual(first["status"], "partial")
            self.assertEqual(first["new_predictions_saved"], 3)
            self.assertEqual(first["expected_new_predictions"], 144)
            saved = {p.name: p.read_bytes() for p in (output / "records").glob("*.json")}
            # User cancellation is recorded as interrupted rather than a stale running segment.
            with patch("unlearning.robustness.evaluator_checks", side_effect=KeyboardInterrupt):
                stopped = run_robustness(*args, output, evaluator=evaluator, make_plots=False)
            self.assertEqual(stopped["status"], "partial")
            self.assertEqual(stopped["last_attempt"]["status"], "interrupted")
            self.assertTrue(stopped["last_attempt"]["cleanup_passed"])
            from unlearning.evaluation import MCQEvaluator
            original = MCQEvaluator.score
            observed = []
            def track(ev, examples):
                observed.extend((e.get("group"), e["id"], ev.settings["instruction"]) for e in examples)
                return original(ev, examples)
            with patch.object(MCQEvaluator, "score", track):
                complete = run_robustness(*args, output, evaluator=evaluator, make_plots=False)
            self.assertEqual(complete["status"], "complete")
            self.assertFalse(complete["is_research_result"])
            self.assertEqual(complete["counts"], {"biology": {"saved": 48, "expected": 48}, "alternative": {"saved": 96, "expected": 96}})
            for group, _, instruction in observed:
                if group == "alternative":
                    self.assertEqual(instruction, protocol["alternative_instruction"])
                elif group == "biology":
                    self.assertEqual(instruction, evaluator.settings["instruction"])
            self.assertEqual(archive_path.read_bytes(), archive_bytes)
            for name, content in saved.items():
                self.assertEqual((output / "records" / name).read_bytes(), content)
            with patch.object(MCQEvaluator, "score", side_effect=AssertionError("No repeated predictions")):
                again = run_robustness(*args, output, evaluator=evaluator, make_plots=False)
            self.assertEqual(again["selective_results"], complete["selective_results"])
            target = next((output / "records").glob("*.json"))
            row = json.loads(target.read_text())
            row["correct"] = not row["correct"]
            row.pop("record_hash")
            row["record_hash"] = digest(row)
            write_json(target, row)
            with self.assertRaisesRegex(ValueError, "logits"):
                robustness_report(output, make_plots=False)
            # A changed frozen test decision must fail BEFORE a model is loaded.
            corrupt = root / "corrupt.zip"
            with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(corrupt, "w") as dest:
                for item in source.infolist():
                    content = source.read(item.filename)
                    if item.filename == "outputs/stage5/test/run.json":
                        run = json.loads(content)
                        run["context"]["frozen_decision"]["alpha"] = .123
                        run["fingerprint"] = digest(run["context"])
                        content = json.dumps(run).encode()
                    dest.writestr(item.filename, content)
            with patch("unlearning.robustness.load_evaluator") as load:
                with self.assertRaises(ValueError):
                    run_robustness(config, evaluator.settings, protocol, data, baseline, stage4, corrupt, root / "bad", make_plots=False)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
