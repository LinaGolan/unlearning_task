"""Independent residual/derivative checks, hook lifetime, and Stage 3 integration."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unlearning.baseline import read_settings, run_baseline
from unlearning.data import save_prepared
from unlearning.interventions import LayerIntervention, gate_gradients, hidden_output
from unlearning.intervention_checks import (finite_difference_check, full_strength_check, model_guard,
    read_check_settings, run_intervention_checks, select_check_examples, verified_inputs)
from test_data import fixtures, ROOT
from test_diagnostics import DiagnosticTokenizer
from test_evaluation import HAS_TORCH, tiny_evaluator


def toy_model(tuple_output=False):
    import torch
    class Block(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64), requires_grad=False)
            self.extra = object()

        def forward(self, hidden_states):
            value = hidden_states + self.scale * hidden_states.square()
            return (value, self.extra) if tuple_output else value

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block(.2), Block(-.3), Block(.1)])

        def forward(self, x):
            for block in self.model.layers:
                x = hidden_output(block(hidden_states=x))
            return x
    return Model().eval()


def prefill_evaluator():
    from unlearning.evaluation import MCQEvaluator
    base = tiny_evaluator()
    return MCQEvaluator(base.model, DiagnosticTokenizer(), read_settings(ROOT / "configs/baseline_prefill.json"))


@unittest.skipUnless(HAS_TORCH, "Requires existing PyTorch and Transformers; no downloads")
class InterventionTests(unittest.TestCase):
    def test_selected_layers_and_all_positions_match_independent_formula(self):
        import torch
        for tuples in (False, True):
            model = toy_model(tuples)
            x = torch.linspace(-.8, .8, 24, dtype=torch.float64).reshape(2, 4, 3)
            for alpha in (0, .25, .5, .75, 1):
                expected = x.clone()
                for layer, scale in enumerate((.2, -.3, .1)):
                    expected = expected + (1 - alpha if layer in (0, 2) else 1) * scale * expected.square()
                with LayerIntervention(model, [2, 0], alpha):
                    actual = model(x)
                torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)

    def test_endpoint_identity_and_tuple_extras_are_preserved(self):
        import torch
        model = toy_model(True)
        x = torch.randn(2, 5, 3, dtype=torch.float64)
        block = model.model.layers[1]
        with LayerIntervention(model, [1], 1):
            result = block(x)
            self.assertIs(result[0], x)
            self.assertIs(result[1], block.extra)
        baseline = model(x)
        with LayerIntervention(model, range(3), 0):
            self.assertTrue(torch.equal(model(x), baseline))
        with LayerIntervention(model, range(3), 1):
            self.assertTrue(torch.equal(model(x), x))

    def test_float64_gradients_match_central_differences_at_identity(self):
        import torch
        model = toy_model()
        x = torch.tensor([[[.3, -.5], [.7, .1]]], dtype=torch.float64)
        gates = torch.ones(3, dtype=torch.float64, requires_grad=True)
        baseline = model(x)
        with LayerIntervention(model, gates=gates):
            result = model(x)
            self.assertTrue(torch.equal(result, baseline))
            grad, = torch.autograd.grad(result.square().sum(), gates)
        for index in range(3):
            values = []
            for offset in (1e-5, -1e-5):
                perturbed = torch.ones(3, dtype=torch.float64)
                perturbed[index] += offset
                with LayerIntervention(model, gates=perturbed):
                    values.append(model(x).square().sum().item())
            self.assertAlmostEqual(grad[index].item(), (values[0] - values[1]) / 2e-5, places=9)
        self.assertTrue((grad.abs() > 0).all())

    def test_failure_cleanup_preserves_foreign_hooks_and_every_weight(self):
        import torch
        model = toy_model()
        foreign = model.model.layers[0].register_forward_hook(lambda *args: None)
        before = model_guard(model)
        values = [p.clone() for p in model.parameters()]
        x = torch.ones(1, 3, 2, dtype=torch.float64)
        baseline = model(x)
        try:
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with LayerIntervention(model, [0, 2], .5):
                    model(x)
                    raise RuntimeError("deliberate")
            self.assertEqual(before, model_guard(model))
            self.assertTrue(torch.equal(model(x), baseline))
            for expected, parameter in zip(values, model.parameters()):
                self.assertTrue(torch.equal(expected, parameter))
                self.assertIsNone(parameter.grad)
        finally:
            foreign.remove()

    def test_failed_hook_registration_and_nested_context_cleanup(self):
        model = toy_model()
        before = model_guard(model)
        with patch.object(model.model.layers[1], "register_forward_hook", side_effect=RuntimeError("register failed")):
            with self.assertRaisesRegex(RuntimeError, "register failed"):
                with LayerIntervention(model, [0, 1], .5):
                    pass
        self.assertEqual(before, model_guard(model))
        with LayerIntervention(model, [0], .5):
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                with LayerIntervention(model, [1], .5):
                    pass
        self.assertEqual(before, model_guard(model))

    def test_invalid_layers_strengths_gates_and_unfrozen_models_fail(self):
        import torch
        model = toy_model()
        for layers in ([-1], [3], [0, 0], [True], [1.5]):
            with self.assertRaises(ValueError):
                LayerIntervention(model, layers, .5)
        for alpha in (-.01, 1.01, float("nan"), True):
            with self.assertRaises(ValueError):
                LayerIntervention(model, [0], alpha)
        for gates in (torch.ones(2, dtype=torch.float64), torch.ones(3), torch.tensor([1, float("nan"), 1], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                LayerIntervention(model, gates=gates)
        model.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            with LayerIntervention(model, [0]):
                pass

    def test_disabled_and_empty_selection_leave_model_unchanged(self):
        import torch
        model = toy_model()
        x = torch.ones(2, 3, 2, dtype=torch.float64)
        expected = model(x)
        for layers, enabled in (([0, 1], False), ([], True)):
            with LayerIntervention(model, layers, 1, enabled=enabled):
                self.assertTrue(torch.equal(model(x), expected))

    def test_tiny_llama_gate_derivatives_endpoints_and_weights(self):
        import torch
        evaluator = prefill_evaluator()
        example = {"id": "check-grad", "question": "How many?", "choices": list("1234"), "answer": 2}
        values = [p.clone() for p in evaluator.model.parameters()]
        base = evaluator.score([example])[0]
        # enable_grad inside our helper also supports callers using no_grad.
        with torch.no_grad():
            measured = gate_gradients(evaluator, example)
        self.assertAlmostEqual(measured["correct_answer_log_probability"], base["correct_answer_log_probability"], places=5)
        self.assertTrue(all(abs(g) > 1e-6 for g in measured["gradients"]))
        checks = read_check_settings(ROOT / "configs/intervention_check.json")
        self.assertEqual(finite_difference_check(evaluator, example, measured, checks)["status"], "passed")
        self.assertEqual(full_strength_check(evaluator, example)["status"], "passed")
        for before, parameter in zip(values, evaluator.model.parameters()):
            self.assertTrue(torch.equal(before, parameter))
            self.assertIsNone(parameter.grad)
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError, "inference_mode"):
                gate_gradients(evaluator, example)

    def test_wrong_gradient_sign_is_rejected(self):
        evaluator = prefill_evaluator()
        example = {"id": "check-bad-grad", "question": "A number?", "choices": list("1234"), "answer": 0}
        measured = gate_gradients(evaluator, example)
        measured["gradients"] = [100 if g < 0 else -100 for g in measured["gradients"]]
        result = finite_difference_check(evaluator, example, measured, read_check_settings(ROOT / "configs/intervention_check.json"))
        self.assertEqual(result["status"], "review_required")
        self.assertTrue(any(not row["sign_passed"] for row in result["comparisons"]))

    def test_runner_reuses_baseline_and_never_scores_test_or_localization(self):
        evaluator = prefill_evaluator()
        # Random synthetic model: research accuracy thresholds do not apply.
        evaluator.settings.update(forget_accuracy_min=0, retain_accuracy_min=0, chance_accuracy=0)
        config, pools = fixtures()
        checks = read_check_settings(ROOT / "configs/intervention_check.json")
        with tempfile.TemporaryDirectory() as tmp:
            data, baseline, output = [Path(tmp) / name for name in ("data", "baseline", "check")]
            save_prepared(config, pools, [], "full", data)
            run_baseline(config, evaluator.settings, data, baseline, evaluator=evaluator)
            baseline_bytes = {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
            with patch.object(evaluator, "score", wraps=evaluator.score) as score:
                result = run_intervention_checks(config, evaluator.settings, checks, data, baseline, output, evaluator=evaluator)
            self.assertEqual(result["checks"]["gradients"]["status"], "passed")
            self.assertEqual(result["checks"]["cleanup_and_frozen_weights"]["status"], "passed")
            self.assertEqual(result["checks"]["compute_feasibility"]["status"], "not_measured")
            self.assertEqual(result["status"], "review_required")  # CPU cannot complete the GPU gate.
            self.assertFalse(result["final_test_evaluated"])
            real = [e for call in score.call_args_list for e in call.args[0] if not e["id"].startswith("check-")]
            self.assertTrue(real)
            self.assertEqual({e["split"] for e in real}, {"development"})
            self.assertLessEqual(len({e["id"] for e in real}), 4)
            self.assertEqual(baseline_bytes, {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob("*") if p.is_file()})
            with self.assertRaisesRegex(ValueError, "new empty output"):
                run_intervention_checks(config, evaluator.settings, checks, data, baseline, output, evaluator=evaluator)
            with patch("unlearning.intervention_checks.full_strength_check", side_effect=RuntimeError("private string must not be exported")):
                failure = run_intervention_checks(config, evaluator.settings, checks, data, baseline, Path(tmp) / "failure", evaluator=evaluator)
            self.assertEqual(failure["status"], "review_required")
            self.assertEqual(failure["failure"]["phase"], "full_strength")
            self.assertTrue(failure["cleanup_after_failure"])
            self.assertNotIn("private string", json.dumps(failure))
            changed = dict(evaluator.settings, instruction="different")
            with self.assertRaisesRegex(ValueError, "matching reviewed"):
                verified_inputs(config, changed, data, baseline)


class SamplingTests(unittest.TestCase):
    def test_length_selection_is_deterministic_and_independent_of_answers(self):
        examples = [{"id": role + str(i), "role": role, "answer": i % 4} for role in ("forget", "retain") for i in range(8)]
        records = [dict(e, input_tokens=10 + int(e["id"][-1])) for e in examples]
        first = select_check_examples(examples, records)
        changed = select_check_examples([dict(e, answer=0) for e in reversed(examples)], list(reversed(records)))
        self.assertEqual([e["id"] for e in first], [e["id"] for e in changed])
        self.assertEqual([e["id"][-1] for e in first], ["4", "7", "4", "7"])


if __name__ == "__main__":
    unittest.main()
