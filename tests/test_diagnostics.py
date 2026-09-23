"""Answer remapping, diagnostic sampling, and portable numeric verification."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unlearning.baseline import equivalent_derived, read_settings, run_baseline
from unlearning.data import save_prepared
from unlearning.diagnostics import (condition_summary, diagnostic_prompt, rotate_example,
                                    run_diagnostic, select_examples)
from test_data import fixtures, ROOT
from test_evaluation import CharacterTokenizer, HAS_TORCH, tiny_evaluator


class DiagnosticTokenizer(CharacterTokenizer):
    def encode(self, text, add_special_tokens=False):
        ids = []
        index = 0
        while index < len(text):
            if text[index] == ' ' and index + 1 < len(text) and text[index + 1] in 'ABCD':
                ids.append(200 + 'ABCD'.index(text[index + 1]))
                index += 2
            else:
                ids.append(ord(text[index]) + 2)
                index += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return ''.join((' ' + 'ABCD'[i - 200]) if 200 <= i < 204 else chr(max(0, i - 2)) for i in ids)


class DiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, 'Requires the existing local numerical-test dependencies')
    def test_prefilled_baseline_matches_the_tested_diagnostic_condition(self):
        from unlearning.diagnostics import DiagnosticEvaluator
        from unlearning.evaluation import MCQEvaluator, evaluator_checks
        base = tiny_evaluator()
        tokenizer = DiagnosticTokenizer()
        settings = read_settings(ROOT / 'configs/baseline_prefill.json')
        revised = MCQEvaluator(base.model, tokenizer, settings)
        original = DiagnosticEvaluator(base.model, tokenizer, base.settings)
        example = {'id': 'prefix-equivalence', 'question': 'Which number is even?',
                   'choices': ['3', '4', '5', '7'], 'answer': 1}
        before = original.score_condition(example, 'assistant_prefix')
        after = revised.score([example])[0]
        for key in ('prompt', 'input_ids', 'answer_logits', 'prediction'):
            self.assertEqual(before[key], after[key])
        self.assertEqual(evaluator_checks(revised)['status'], 'passed')

    def test_cross_platform_last_bit_rounding_is_allowed_but_wrong_predictions_are_not(self):
        self.assertTrue(equivalent_derived({'A': 0.8460885424443747}, {'A': 0.8460885424443745}))
        self.assertFalse(equivalent_derived({'A': .84}, {'A': .85}))
        self.assertFalse(equivalent_derived('A', 'B'))
        self.assertFalse(equivalent_derived(1, True))
        self.assertFalse(equivalent_derived(float('nan'), 0.5))

    def test_rotation_preserves_correct_content_and_visits_every_position(self):
        row = {'choices': ['one', 'two', 'three', 'four'], 'answer': 2}
        positions = []
        for shift in range(4):
            rotated, order = rotate_example(row, shift)
            self.assertEqual(rotated['choices'][rotated['answer']], 'three')
            self.assertEqual(order[rotated['answer']], 2)
            positions.append(rotated['answer'])
        self.assertEqual(set(positions), {0, 1, 2, 3})
        self.assertEqual(row['choices'], ['one', 'two', 'three', 'four'])

    def test_sampling_does_not_use_correctness_or_answer_label(self):
        examples = [{'id': str(i), 'role': 'forget', 'subject': 'bio', 'answer': i % 4} for i in range(30)]
        examples += [{'id': 'retain-' + str(i), 'role': 'retain', 'subject': str(i % 8), 'answer': i % 4} for i in range(40)]
        first = [row['id'] for row in select_examples(examples, 42)]
        changed = [dict(row, answer=0, prediction='A', correct=False) for row in reversed(examples)]
        self.assertEqual(first, [row['id'] for row in select_examples(changed, 42)])
        self.assertEqual(len(first), 32)

    def test_reversing_display_keeps_labels_and_prefill_checks_boundary(self):
        evaluator_settings = {'instruction': 'Choose one.', 'date_string': '19 Sep 2026', 'max_input_tokens': 2048}
        row = {'question': 'Test', 'choices': ['one', 'two', 'three', 'four']}
        tokenizer = DiagnosticTokenizer()
        normal = diagnostic_prompt(tokenizer, row, evaluator_settings, 'original')
        reversed_row = diagnostic_prompt(tokenizer, row, evaluator_settings, 'reversed_labels')
        self.assertIn('D. four\nC. three\nB. two\nA. one', reversed_row['prompt'])
        prefixed = diagnostic_prompt(tokenizer, row, evaluator_settings, 'assistant_prefix')
        self.assertEqual(prefixed['prompt'], normal['prompt'] + 'The correct answer is')
        self.assertEqual(prefixed['candidate_suffixes'], [' A', ' B', ' C', ' D'])
        self.assertEqual(prefixed['label_ids'], [200, 201, 202, 203])

    def test_summary_distinguishes_following_content_from_always_answering_a(self):
        rows = []
        for variant in ('original', 'assistant_prefix'):
            for shift in range(4):
                rotated, order = rotate_example({'choices': list('wxyz'), 'answer': 2}, shift)
                prediction = 0 if variant == 'original' else rotated['answer']
                rows.append({'id': 'one', 'role': 'forget', 'variant': variant, 'shift': shift,
                    'prediction': 'ABCD'[prediction], 'correct': prediction == rotated['answer'],
                    'original_choice_prediction': order[prediction], 'answer_probability_mass': .9})
        summary = condition_summary(rows)
        self.assertEqual(summary['original']['forget']['rotation_averaged_accuracy'], .25)
        self.assertEqual(summary['original']['forget']['content_consistency_across_rotations'], 0)
        self.assertEqual(summary['assistant_prefix']['forget']['content_consistency_across_rotations'], 1)
        self.assertEqual(summary['assistant_prefix']['forget']['rotation_averaged_accuracy'], 1)

    @unittest.skipUnless(HAS_TORCH, 'Requires the existing local numerical-test dependencies')
    def test_end_to_end_diagnostic_replays_baseline_without_modifying_it(self):
        config, pools = fixtures()
        evaluator = tiny_evaluator()
        evaluator.tokenizer = DiagnosticTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_prepared(config, pools, [], 'full', root / 'data')
            run_baseline(config, evaluator.settings, root / 'data', root / 'baseline', evaluator=evaluator)
            original = {str(p.relative_to(root / 'baseline')): p.read_bytes()
                        for p in (root / 'baseline').rglob('*') if p.is_file()}
            report = run_diagnostic(root / 'data', root / 'baseline', root / 'diagnostic', evaluator=evaluator)
            self.assertTrue(report['numerical_checks_passed'])
            self.assertFalse(report['final_test_evaluated'])
            self.assertEqual(len(list((root / 'diagnostic' / 'predictions').glob('*.json'))), 432)
            self.assertEqual(len(report['harmless_continuations']), 8)
            for name, content in original.items():
                self.assertEqual((root / 'baseline' / name).read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
