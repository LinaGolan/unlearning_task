"""Scientific safeguards: no GPU, model download, or external datasets required."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlearning.data import (build_splits, make_example, read_config,
                             save_prepared, verify_prepared)

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    config = read_config(ROOT / "configs/experiment.json")
    config["profiles"] = {
        "full": {"localization": 16, "development": 16, "test": 16},
        "reduced": {"localization": 8, "development": 8, "test": 8},
    }
    config["biology_control_size"] = 8
    pools = {}
    for role, source in config["datasets"].items():
        pools[role] = []
        for subject in source["configs"]:
            for i in range(64):
                row = {"question": "{} sample {}".format(subject, i),
                       "choices": ["one", "two", "three", "four"], "answer": i % 4}
                pools[role].append(make_example(row, source, subject, i))
    return config, pools


class SplitTests(unittest.TestCase):
    def test_disjoint_and_balanced(self):
        config, pools = fixtures()
        splits, _ = build_splits(config, pools, "full")
        hashes = [row["question_hash"] for rows in splits.values() for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        for name, rows in splits.items():
            self.assertEqual(len(rows), 8 if name.startswith("biology") else 16)
            if name.startswith("retain"):
                for subject in config["datasets"]["retain"]["configs"]:
                    self.assertEqual(sum(row["subject"] == subject for row in rows), 2)

    def test_stable_under_source_reordering(self):
        config, pools = fixtures()
        first, _ = build_splits(config, pools, "full")
        reversed_pools = {role: list(reversed(rows)) for role, rows in pools.items()}
        second, _ = build_splits(config, reversed_pools, "full")
        self.assertEqual(first, second)

    def test_reduced_profile_never_moves_questions_between_splits(self):
        config, pools = fixtures()
        full, _ = build_splits(config, pools, "full")
        reduced, _ = build_splits(config, pools, "reduced")
        for name in full:
            self.assertLessEqual({r["id"] for r in reduced[name]}, {r["id"] for r in full[name]})

    def test_seed_changes_selection(self):
        config, pools = fixtures()
        first, _ = build_splits(config, pools, "full")
        config["seed"] += 1
        second, _ = build_splits(config, pools, "full")
        self.assertNotEqual(first, second)

    def test_cross_dataset_duplicate_text_is_excluded(self):
        config, pools = fixtures()
        row = pools["retain"][0]
        row["question"] = pools["forget"][0]["question"].upper() + "  "
        row["question_hash"] = pools["forget"][0]["question_hash"]
        _, dropped = build_splits(config, pools, "full")
        self.assertEqual(dropped["retain"], 1)

    def test_insufficient_pool_fails_instead_of_sampling_with_replacement(self):
        config, pools = fixtures()
        pools["forget"] = pools["forget"][:10]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            build_splits(config, pools, "full")

    def test_invalid_answer_rejected(self):
        config, _ = fixtures()
        for invalid in (-1, 4, True, "A"):
            with self.assertRaisesRegex(ValueError, "Answer"):
                make_example({"question": "test", "choices": ["a", "b", "c", "d"], "answer": invalid},
                             config["datasets"]["forget"], "wmdp-bio", 0)

    def test_saved_files_verified_and_rerun_preserves_manifest(self):
        config, pools = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            first = save_prepared(config, pools, [], "full", directory)
            self.assertEqual(first, verify_prepared(directory))
            self.assertEqual(first, save_prepared(config, pools, [], "full", directory))

    def test_tampered_data_detected(self):
        config, pools = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            save_prepared(config, pools, [], "full", directory)
            path = Path(directory) / "forget/test.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                verify_prepared(directory)

    def test_changed_settings_cannot_overwrite_existing_data(self):
        config, pools = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            save_prepared(config, pools, [], "full", directory)
            with self.assertRaisesRegex(ValueError, "different settings"):
                save_prepared(config, pools, [], "reduced", directory)

    def test_manifest_fingerprint_tampering_detected(self):
        config, pools = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            save_prepared(config, pools, [], "full", directory)
            path = Path(directory) / "manifest.json"
            report = json.loads(path.read_text())
            report["config"]["seed"] += 1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                verify_prepared(directory)

    def test_invalid_unbalanced_config_rejected(self):
        config, _ = fixtures()
        config["profiles"]["full"]["test"] = 17
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "balanced"):
                read_config(path)


if __name__ == "__main__":
    unittest.main()
