import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unlearning.data import digest, read_jsonl
from unlearning.full_screening import _export, _score, prepare
from unlearning.permutations import expand
from test_permutations import example, prediction


class ScreeningTests(unittest.TestCase):
    def test_fresh_preparation_needs_no_previous_experiment(self):
        row = example()
        pools = {'forget': [row], 'retain': [row], 'biology': []}
        config = {'model': {}, 'prompt': 'harness', 'seed': 1}
        with tempfile.TemporaryDirectory() as folder:
            with patch('unlearning.full_screening.read_config', return_value=config), \
                    patch('unlearning.full_screening.load_sources', return_value=(pools, [])):
                rows = prepare('config.json', 'cache', folder)
            self.assertEqual([r['id'] for r in rows], [row['id']])
            self.assertEqual(read_jsonl(Path(folder) / 'reused_predictions.jsonl'), [])
            self.assertEqual(read_jsonl(Path(folder) / 'questions.jsonl'), rows)

    def test_correct_first_reuses_predictions_and_caps_selected_splits(self):
        rows = []
        for role, n in (('forget', 33), ('retain', 33), ('biology', 1)):
            for i in range(n):
                row = dict(example(), id=f'{role}-{i}', role=role,
                           subject='biology' if role != 'retain' else 'history',
                           question=f'Which colour in {role} {i}?')
                row['question_hash'] = digest(row['question'])
                row['content_hash'] = digest([row['question'], row['choices'], row['answer']])
                rows.append(row)
        cached = rows[0]
        reused = {cached['id']: dict(prediction(cached), content_hash=cached['content_hash'])}

        class FakeEvaluator:
            config = {'batch_size': 4}
            def __init__(self):
                self.calls = []
            def score(self, batch):
                self.calls.extend(row['id'] for row in batch)
                return [prediction(row, row['id'] != 'forget-32') for row in batch]

        config = {'seed': 1, 'datasets': {}, 'prompt': 'harness'}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            evaluator = FakeEvaluator()
            originals = _score(rows, out / 'originals.jsonl', reused, evaluator)
            self.assertNotIn(cached['id'], evaluator.calls)
            correct = [row for row in rows if originals[row['id']]['correct']]
            variants = [v for row in correct for v in expand(row)[1:]]
            scored = _score(variants, out / 'variants.jsonl', {}, evaluator)
            summary = _export(config, rows, originals, scored, out)
            self.assertEqual(summary['eligible']['forget'], 32)
            self.assertEqual(summary['selected'], {'forget': 32, 'retain': 32, 'biology': 1})
            self.assertEqual(summary['splits'], {'localization': 8, 'development': 8, 'test': 16})
            self.assertEqual(summary['permutation_predictions'], 23 * 66)
            self.assertNotIn('forget-32::perm:01', scored)
