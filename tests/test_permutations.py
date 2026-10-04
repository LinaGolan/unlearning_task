import tempfile
import unittest
from pathlib import Path

from unlearning.data import digest, read_jsonl
from unlearning.permutations import expand, write_jsonl
from unlearning.full_screening import _read_saved, _score


def example(answer=2):
    choices = ["red", "blue", "green", "yellow"]
    return {"id": "example", "question": "Which colour?", "choices": choices,
            "answer": answer, "role": "forget", "subject": "biology", "split": "test",
            "question_hash": digest("Which colour?"),
            "content_hash": digest(["Which colour?", choices, answer])}


def prediction(row, correct=True):
    answer = row["answer"]
    return {"id": row["id"], "role": row["role"], "subject": row["subject"],
            "style": "harness", "correct_answer": "ABCD"[answer], "correct": correct,
            "prediction": "ABCD"[answer if correct else (answer + 1) % 4]}


class PermutationTests(unittest.TestCase):
    def test_all_answers_preserved_and_balanced_across_positions(self):
        for answer in range(4):
            original = example(answer)
            variants = expand(original)
            self.assertEqual(len({r["id"] for r in variants}), 24)
            self.assertEqual(len({tuple(r["choice_order"]) for r in variants}), 24)
            self.assertEqual(variants[0]["choices"], original["choices"])
            self.assertEqual(variants[0]["answer"], answer)
            self.assertEqual(original["id"], "example")
            for row in variants:
                self.assertEqual(row["choices"][row["answer"]], original["choices"][answer])
                self.assertEqual(row["parent_id"], original["id"])
                self.assertEqual(row["split"], "test")
            self.assertEqual([sum(r["answer"] == i for r in variants) for i in range(4)], [6] * 4)

    def test_stale_or_inconsistent_predictions_rejected(self):
        row = example()
        valid = dict(prediction(row), content_hash=row["content_hash"])
        for field, value in (("id", "different"), ("style", "chat_plain"),
                             ("correct_answer", "A"), ("correct", False)):
            with self.assertRaises(ValueError):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "saved.jsonl"
                    write_jsonl(path, [dict(valid, **{field: value})])
                    _read_saved(path, {row["id"]: row})

    def test_resume_scores_only_missing_variants(self):
        variants = expand(example())

        class FakeEvaluator:
            style = "harness"
            config = {"batch_size": 8}

            def __init__(self):
                self.calls = []

            def score(self, rows):
                self.calls.extend(r["id"] for r in rows)
                return [prediction(r, r["permutation_index"] != 23) for r in rows]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            evaluator = FakeEvaluator()
            first = _score(variants[:8], path, {}, evaluator)
            self.assertEqual(len(first), 8)
            evaluator.calls.clear()
            records = _score(variants, path, {}, evaluator)
            self.assertEqual(evaluator.calls, [r["id"] for r in variants[8:]])
            self.assertEqual(sum(row["correct"] for row in records.values()), 23)
            evaluator.calls.clear()
            self.assertEqual(list(_score(variants, path, {}, evaluator).values()), read_jsonl(path))
            self.assertEqual(evaluator.calls, [])

if __name__ == "__main__":
    unittest.main()
