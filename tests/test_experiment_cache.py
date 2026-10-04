import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from unlearning.experiment import evaluate_condition


class ConditionCacheTests(unittest.TestCase):
    def test_cli_defaults_to_verified_screened_splits(self):
        from unlearning.__main__ import main
        with patch('unlearning.experiment.run') as run:
            self.assertEqual(main(['run']), 0)
        self.assertEqual(run.call_args.args[:3], (
            'results_fullscreen/selected_config.json',
            'results_fullscreen/selected_data', 'results_fullscreen_experiment/main'))
        self.assertTrue(run.call_args.kwargs['prepared_data'])

    def test_identical_condition_reuses_predictions_without_scoring(self):
        class Evaluator:
            def score(self, examples):
                raise AssertionError("Equivalent condition should not be rescored")

        records = [{"id": "question-1", "prediction": "B"}]
        with tempfile.TemporaryDirectory() as folder:
            output = evaluate_condition(Evaluator(), [{"id": "question-1"}], folder,
                                        "test__gate_margin__top_localized__a0.5",
                                        "block_scale", 0.5, reuse_records=records, layers=[10])
            self.assertEqual(output, records)
            path = Path(folder) / "predictions/test__gate_margin__top_localized__a0.5.jsonl"
            self.assertIn('"question-1"', path.read_text(encoding="utf-8"))
