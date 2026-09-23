"""Stage 5 selection boundaries, paired uncertainty, test isolation, and recovery."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unlearning.baseline import run_baseline
from unlearning.data import digest, save_prepared, write_json
from unlearning.localization import read_localization_settings, run_localization
from unlearning.sweep import read_sweep_settings, run_sweep, verified_decision, read_sweep_records
from unlearning.sweep_analysis import choose_operating_point, conditions, paired_bootstrap, summarize_sweep
from unlearning.sweep_report import sweep_report, make_figures
from unlearning.result_archives import restore_stage5_results
from test_data import ROOT, fixtures
from test_evaluation import HAS_TORCH
from test_interventions import prefill_evaluator
from test_localization import fixture_stage3
import test_localization


def synthetic_sweep():
    config, _ = fixtures()
    context = {"config": config, "protocol": read_sweep_settings(ROOT / "configs/sweep.json"),
               "split": "development", "is_test_fixture": True,
               "selections": [{"name": name, "layers": [i % 2]} for i, name in enumerate(
                   ("top_forget", "top_selective", "bottom_forget", "random_101", "random_202"))],
               "examples": [{"id": role + str(i), "role": role, "subject": role, "split": "development"}
                            for role in ("forget", "retain") for i in range(20)]}
    context["protocol"]["bootstrap_repetitions"] = 100
    records = [dict(e, method=c["method"], alpha=c["alpha"], correct=True,
                    prediction="A", correct_answer_log_probability=-.5)
               for c in conditions(context) for e in context["examples"]]
    return context, records


class SweepAnalysisTests(unittest.TestCase):
    def test_exact_limit_ties_retain_improvements_and_fallback(self):
        context, records = synthetic_sweep()
        self.assertTrue(choose_operating_point(context, records)["display_only_fallback"])
        # 1/20 = exactly 5pp retain loss is allowed; 2/20 is not.
        for row in records:
            i = int(row["id"].replace(row["role"], ""))
            if row["method"] == "top_selective":
                if row["role"] == "forget":
                    row["correct"] = i >= (3 if row["alpha"] in (.25, .5) else 5)
                else:
                    row["correct"] = i >= (1 if row["alpha"] in (.25, .5) else 2)
        choice = choose_operating_point(context, records)
        self.assertEqual(choice["alpha"], .25)
        self.assertFalse(choice["display_only_fallback"])
        self.assertEqual([r["eligible"] for r in choice["candidates"]], [True, True, False, False])
        # Baseline worse than the intervention is a negative retain drop, not an error.
        for row in records:
            if row["method"] == "baseline" and row["role"] == "retain":
                row["correct"] = False
        self.assertEqual(choose_operating_point(context, records)["alpha"], .75)
        with self.assertRaisesRegex(ValueError, "full development"):
            choose_operating_point(context, records[:-1])
        with self.assertRaisesRegex(ValueError, "only.*development"):
            choose_operating_point(dict(context, split="test"), records)

    def test_paired_bootstrap_preserves_identity_subject_weights_and_contrasts(self):
        import numpy as np
        values = [[0, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0]]
        sampled = paired_bootstrap(values, ["one", "one", "two", "two"], 100, 42)
        self.assertTrue(np.array_equal(sampled[:, 0], sampled[:, 1]))
        self.assertTrue(np.all(sampled[:, 0] >= .5))
        self.assertTrue(np.all(sampled[:, 0] + sampled[:, 2] == 1))
        context, records = synthetic_sweep()
        for row in records:
            if row["method"] == "top_selective" and row["role"] == "forget":
                row["correct"] = False
        result = summarize_sweep(context, records, .5)
        selective = next(r for r in result["common_strength_contrasts"] if r["role"] == "forget" and r["method"] == "top_selective")
        self.assertEqual(selective["extra_accuracy_drop"], 1)
        self.assertEqual(selective["paired_ci95"], [1, 1])
        for row in result["conditions"]:
            if row["method"] == "baseline":
                self.assertEqual(row["drop_ci95"], [0, 0])
        self.assertEqual(result["random_summary"][0]["accuracy_sd_across_pairs"], 0)

    def test_restore_both_phases_without_installing_source(self):
        archive = test_localization.ArchiveTests.archive({"outputs/stage5/development/run.json": "{}",
            "outputs/stage5/test/run.json": "{}", "src/unlearning/sweep.py": "do not install"})
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(restore_stage5_results(archive, tmp)["restored_files"], 2)
            self.assertFalse((Path(tmp) / "src").exists())
            bad = test_localization.ArchiveTests.archive({"outputs/stage5/development/run.json": "{}", "outputs/stage5/test/../../escape.json": "bad"})
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                restore_stage5_results(bad, tmp)


@unittest.skipUnless(HAS_TORCH, "Uses existing tiny CPU model; no downloads")
class SweepRunnerTests(unittest.TestCase):
    def test_resume_freeze_before_test_and_tamper_detection(self):
        evaluator = prefill_evaluator()
        evaluator.settings.update(forget_accuracy_min=0, retain_accuracy_min=0, chance_accuracy=0)
        config, pools = fixtures()
        config["intervention"].update(k=1, random_seeds=[101, 202])
        protocol = read_sweep_settings(ROOT / "configs/sweep.json")
        protocol["bootstrap_repetitions"] = 100
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, baseline, stage3, stage4, development, test = [root / n for n in ("data", "baseline", "stage3", "stage4", "development", "test")]
            save_prepared(config, pools, [], "full", data)
            run_baseline(config, evaluator.settings, data, baseline, evaluator=evaluator)
            fixture_stage3(stage3, config, evaluator.settings, json.loads((baseline / "run.json").read_text()))
            run_localization(config, evaluator.settings, read_localization_settings(ROOT / "configs/localization.json"),
                             data, baseline, stage3, stage4, evaluator=evaluator, make_plots=False)
            baseline_bytes = {p.name: p.read_bytes() for p in (baseline / "predictions").glob("*.json")}
            args = (config, evaluator.settings, protocol, data, baseline, stage4)
            with patch.object(evaluator, "score", wraps=evaluator.score) as scored:
                partial = run_sweep(*args, development, max_new=5, evaluator=evaluator, make_plots=False)
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(partial["new_predictions_saved"], 5)
            real_scored = [e for call in scored.call_args_list for e in call.args[0] if not e["id"].startswith("check-")]
            self.assertEqual({e["split"] for e in real_scored}, {"development"})
            self.assertFalse((development / "operating_point.json").exists())
            with patch("unlearning.sweep.load_test_examples") as forbidden:
                with self.assertRaisesRegex(ValueError, "full development"):
                    run_sweep(*args, test, split="test", development_dir=development, evaluator=evaluator, make_plots=False)
                forbidden.assert_not_called()
            saved = {p.name: p.read_bytes() for p in (development / "records").glob("*.json")}
            # Simulate a scoring failure after the 2 baseline replays.
            original = evaluator.score
            seen = []
            def interrupted(examples):
                seen.append(examples)
                if len(seen) == 5:
                    raise RuntimeError("synthetic interruption: private text")
                return original(examples)
            with patch.object(evaluator, "score", side_effect=interrupted):
                failed = run_sweep(*args, development, max_new=8, evaluator=evaluator, make_plots=False)
            self.assertEqual(failed["status"], "review_required")
            self.assertNotIn("private text", json.dumps(failed))
            completed = run_sweep(*args, development, evaluator=evaluator, make_plots=False)
            self.assertEqual(completed["status"], "complete")
            self.assertFalse(completed["is_research_result"])
            self.assertEqual(completed["new_predictions_saved"], 640)
            for name, content in saved.items():
                self.assertEqual((development / "records" / name).read_bytes(), content)
            _, decision = verified_decision(development)
            frozen = (development / "operating_point.json").read_bytes()
            test_result = run_sweep(*args, test, split="test", development_dir=development, evaluator=evaluator, make_plots=False)
            self.assertEqual(test_result["status"], "complete")
            self.assertEqual(test_result["new_predictions_saved"], 672)
            self.assertEqual(test_result["operating_point"], decision)
            self.assertEqual((development / "operating_point.json").read_bytes(), frozen)
            self.assertEqual(baseline_bytes, {p.name: p.read_bytes() for p in (baseline / "predictions").glob("*.json")})
            # A completed run repeats no forward passes and keeps its decision.
            with patch.object(evaluator, "score", side_effect=AssertionError("No re-scoring")):
                again = run_sweep(*args, development, evaluator=evaluator, make_plots=False)
            self.assertEqual(again["main_comparison"], completed["main_comparison"])
            # Recomputed checksums cannot hide an incorrect derived accuracy.
            target = next((test / "records").glob("*.json"))
            row = json.loads(target.read_text())
            row["correct"] = not row["correct"]
            row.pop("record_hash")
            row["record_hash"] = digest(row)
            write_json(target, row)
            with self.assertRaisesRegex(ValueError, "logits"):
                sweep_report(test, make_plots=False)
            decision["alpha"] = .123
            write_json(development / "operating_point.json", decision)
            with patch("unlearning.sweep.load_test_examples") as forbidden:
                with self.assertRaisesRegex(ValueError, "frozen operating"):
                    run_sweep(*args, test, split="test", development_dir=development, evaluator=evaluator, make_plots=False)
                forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
