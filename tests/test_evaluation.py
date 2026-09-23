"""Numerical evaluator checks with a random tiny Llama; no model downloads."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unlearning.baseline import (baseline_report, read_settings, run_baseline)
from unlearning.data import digest, save_prepared
from unlearning.evaluation import (MCQEvaluator, evaluator_checks, gradient_timing_probe,
                                   label_token_ids, next_token_logits, padded_inputs,
                                   prepare_prompt, render_prompt, summarize_logits)
from unlearning.metrics import ability_gate, wilson_interval
from test_data import fixtures, ROOT

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


class CharacterTokenizer:
    """Synthetic tokenizer with exact letter boundaries and a date-aware template."""
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("The template already supplies special tokens.")
        return [ord(char) + 2 for char in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, date_string):
        assert tokenize is False and add_generation_prompt is True
        return "<system date=" + date_string + ">\n" + messages[0]["content"] + "\n<assistant>\n"

    def get_chat_template(self):
        return "synthetic-character-template-v1"


def tiny_evaluator():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(53)
    torch.set_num_threads(1)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=2048))
    return MCQEvaluator(model, CharacterTokenizer(), read_settings(ROOT / "configs/baseline.json"))


class ScoringTests(unittest.TestCase):
    def test_stable_normalization_correct_letter_and_ties(self):
        row = summarize_logits([10000, 10002, 10000, 10000], 1)
        self.assertEqual(row["prediction"], "B")
        self.assertTrue(row["correct"])
        self.assertAlmostEqual(sum(row["answer_probabilities"].values()), 1)
        self.assertAlmostEqual(row["correct_answer_probability"], 0.7112345942)
        self.assertEqual(summarize_logits([0, 0, 0, 0], 3)["prediction"], "A")
        with self.assertRaises(ValueError):
            summarize_logits([float('nan'), 0, 0, 0], 0)

    def test_prompt_is_fixed_and_independent_of_correct_answer(self):
        settings = read_settings(ROOT / "configs/baseline.json")
        row = {"id": "synthetic", "question": "A question", "choices": list("1234"), "answer": 0}
        tokenizer = CharacterTokenizer()
        prompt = render_prompt(tokenizer, row, settings)
        row["answer"] = 3
        self.assertEqual(prompt, render_prompt(tokenizer, row, settings))
        self.assertIn("19 Sep 2026", prompt)
        self.assertTrue(prompt.endswith("<assistant>\n"))
        settings["max_input_tokens"] = 10
        with self.assertRaisesRegex(ValueError, "No truncation"):
            prepare_prompt(tokenizer, row, settings, label_token_ids(tokenizer))

    def test_contextual_token_boundary_mismatch_fails(self):
        tokenizer = CharacterTokenizer()
        original = tokenizer.encode
        tokenizer.encode = lambda text, **kwargs: original(text, **kwargs) + ([9] if len(text) > 1 and text.endswith("A") else [])
        row = {"id": "synthetic", "question": "Test", "choices": list("1234")}
        with self.assertRaisesRegex(ValueError, "boundary"):
            prepare_prompt(tokenizer, row, read_settings(ROOT / "configs/baseline.json"), label_token_ids(tokenizer))

    def test_wilson_interval_and_predeclared_thresholds(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038315304)
        self.assertAlmostEqual(high, 0.5961684696)
        self.assertAlmostEqual(wilson_interval(0, 100)[0], 0)
        settings = read_settings(ROOT / "configs/baseline.json")
        results = {role: {"accuracy": .5, "accuracy_interval_95": [.4, .6]} for role in ("forget", "retain")}
        self.assertEqual(ability_gate(results, settings)["status"], "passed")
        results["forget"]["accuracy_interval_95"][0] = .25
        self.assertEqual(ability_gate(results, settings)["status"], "review_required")
        results["forget"]["accuracy_interval_95"][0] = .4
        results["retain"]["accuracy"] = .39
        self.assertEqual(ability_gate(results, settings)["status"], "review_required")


@unittest.skipUnless(HAS_TORCH, "PyTorch and Transformers are required for numerical checks")
class NumericalTests(unittest.TestCase):
    def test_tiny_llama_batch_and_teacher_forced_likelihood(self):
        self.assertEqual(evaluator_checks(tiny_evaluator())["status"], "passed")

    def test_padding_uses_last_real_token_not_last_column(self):
        import torch
        from types import SimpleNamespace
        class PositionModel:
            def forward(self, **kwargs):
                pass

            def __call__(self, **kwargs):
                logits = torch.arange(2 * 3 * 4).reshape(2, 3, 4).float()
                return SimpleNamespace(logits=logits)
        inputs = padded_inputs([{"input_ids": [4]}, {"input_ids": [5, 6, 7]}], 0, "cpu")
        scores = next_token_logits(PositionModel(), inputs)
        self.assertEqual(scores.tolist(), [[0, 1, 2, 3], [20, 21, 22, 23]])

    def test_gradient_probe_does_not_train_or_change_weights(self):
        import torch
        evaluator = tiny_evaluator()
        before = [parameter.detach().clone() for parameter in evaluator.model.parameters()]
        row = {"id": "gradient-check", "question": "Test", "choices": list("1234"), "answer": 0}
        self.assertEqual(gradient_timing_probe(evaluator, [row])["status"], "passed")
        for old, parameter in zip(before, evaluator.model.parameters()):
            self.assertTrue(torch.equal(old, parameter))
            self.assertIsNone(parameter.grad)
            self.assertFalse(parameter.requires_grad)

    def test_interruption_resume_matches_uninterrupted_predictions(self):
        config, pools = fixtures()
        evaluator = tiny_evaluator()
        with tempfile.TemporaryDirectory() as tmp:
            data, output = Path(tmp) / "data", Path(tmp) / "resumed"
            save_prepared(config, pools, [], "full", data)
            partial = run_baseline(config, evaluator.settings, data, output, max_new=5, evaluator=evaluator)
            self.assertEqual(partial["status"], "partial")
            self.assertNotIn("ability_gate", partial)
            first_files = {p.name: p.read_bytes() for p in (output / "predictions").glob("*.json")}
            with patch.object(evaluator, "score", wraps=evaluator.score) as score:
                complete = run_baseline(config, evaluator.settings, data, output, evaluator=evaluator)
                # The resumed path only scores unsaved real questions (plus synthetic checks).
                ids = [row["id"] for call in score.call_args_list for row in call.args[0] if not row["id"].startswith("check-")]
                self.assertEqual(len(ids), 27)
            self.assertEqual(complete["evaluated"], 32)
            self.assertEqual(complete["status"], "complete")
            for name, content in first_files.items():
                self.assertEqual((output / "predictions" / name).read_bytes(), content)
            full_output = Path(tmp) / "uninterrupted"
            run_baseline(config, evaluator.settings, data, full_output, evaluator=evaluator)
            def values(folder):
                return [(row["id"], row["prediction"], row["answer_logits"]) for row in
                        map(json.loads, (folder / "predictions.jsonl").read_text().splitlines())]
            self.assertEqual(values(output), values(full_output))
            self.assertTrue(all(json.loads(line)["split"] == "development" for line in
                                (output / "predictions.jsonl").read_text().splitlines()))
            self.assertFalse(complete["final_test_evaluated"])
            self.assertEqual(complete["runtime_forecast"]["status"], "provisional")

    def test_changed_settings_and_tampered_results_are_rejected(self):
        config, pools = fixtures()
        evaluator = tiny_evaluator()
        with tempfile.TemporaryDirectory() as tmp:
            data, output = Path(tmp) / "data", Path(tmp) / "out"
            save_prepared(config, pools, [], "full", data)
            run_baseline(config, evaluator.settings, data, output, max_new=1, evaluator=evaluator)
            changed = copy.deepcopy(evaluator.settings)
            changed["date_string"] = "20 Sep 2026"
            with self.assertRaisesRegex(ValueError, "different"):
                run_baseline(config, changed, data, output, evaluator=evaluator)
            path = next((output / "predictions").glob("*.json"))
            row = json.loads(path.read_text())
            row["prediction"] = "invalid"
            # Even with a rehashed file, saved predictions must agree with logits.
            row.pop("record_hash")
            row["record_hash"] = digest(row)
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                baseline_report(output)

    def test_forward_failure_preserves_completed_questions_and_partial_report(self):
        config, pools = fixtures()
        evaluator = tiny_evaluator()
        original = evaluator.score
        real_calls = []
        def fail_on_second_real_batch(rows):
            if not rows[0]["id"].startswith("check-"):
                real_calls.append(rows[0]["id"])
                if len(real_calls) == 2:
                    raise RuntimeError("Simulated runtime interruption")
            return original(rows)
        with tempfile.TemporaryDirectory() as tmp:
            data, output = Path(tmp) / "data", Path(tmp) / "out"
            save_prepared(config, pools, [], "full", data)
            with patch.object(evaluator, "score", side_effect=fail_on_second_real_batch):
                with self.assertRaisesRegex(RuntimeError, "Simulated"):
                    run_baseline(config, evaluator.settings, data, output, evaluator=evaluator)
            report = baseline_report(output)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["evaluated"], 1)
            self.assertEqual(len(list((output / "predictions").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
