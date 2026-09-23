"""Fixed, development-only checks for option-position and answer-format bias.

This does not select a new prompt, change the baseline, or tune on final-test data.
"""

import json
from pathlib import Path
import statistics

from .baseline import development_examples, load_evaluator, read_predictions
from .data import digest, file_digest, write_json
from .evaluation import LABELS, MCQEvaluator, next_token_logits, padded_inputs, summarize_logits
from .runtime import environment_info

VARIANTS = ("original", "assistant_prefix", "reversed_labels")
SANITY = [
    {"id": "sanity-even", "question": "Which number is even?", "choices": ["3", "4", "5", "7"], "answer": 1},
    {"id": "sanity-count", "question": "A box contains two red balls and one blue ball. How many balls are there in total?",
     "choices": ["one", "two", "three", "four"], "answer": 2},
    {"id": "sanity-capital", "question": "What is the capital of France?",
     "choices": ["Rome", "Berlin", "Madrid", "Paris"], "answer": 3},
    {"id": "sanity-color", "question": "On a clear sunny day, what color does the sky usually appear?",
     "choices": ["blue", "purple", "green", "orange"], "answer": 0},
]


def select_examples(examples, seed):
    """16 forget + two per retain subject; never select by prediction or correctness."""
    selected = []
    subjects = sorted({row["subject"] for row in examples if row["role"] == "retain"})
    for role, subject, count in [("forget", None, 16)] + [("retain", name, 2) for name in subjects]:
        pool = [row for row in examples if row["role"] == role and (subject is None or row["subject"] == subject)]
        if len(pool) < count:
            raise ValueError("Insufficient development questions for the fixed diagnostic sample.")
        pool.sort(key=lambda row: digest([seed, "bias-diagnostic-v1", row["id"]]))
        selected.extend(pool[:count])
    return selected


def rotate_example(example, shift):
    """Map each displayed A-D slot back to its original choice index."""
    order = [(i + shift) % 4 for i in range(4)]
    return dict(example, choices=[example["choices"][i] for i in order],
                answer=order.index(example["answer"])), order


def diagnostic_prompt(tokenizer, example, settings, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown diagnostic condition.")
    pairs = list(zip(LABELS, example["choices"]))
    if variant == "reversed_labels":
        pairs.reverse()  # Same label-content mapping, but D is physically first.
    text = settings["instruction"] + "\n\n" + example["question"] + "\n"
    text += "\n".join("{}. {}".format(label, choice) for label, choice in pairs)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, date_string=settings["date_string"])
    candidates = list(LABELS)
    if variant == "assistant_prefix":
        prompt += "The correct answer is"
        candidates = [" " + label for label in LABELS]
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > settings["max_input_tokens"]:
        raise ValueError("Diagnostic prompt is too long; no truncation permitted.")
    label_ids = []
    for suffix in candidates:
        whole = tokenizer.encode(prompt + suffix, add_special_tokens=False)
        if whole[:-1] != ids or len(whole) != len(ids) + 1:
            raise ValueError("Diagnostic answer suffix is not one token at the actual prompt boundary.")
        label_ids.append(whole[-1])
    if len(set(label_ids)) != 4:
        raise ValueError("Diagnostic answer tokens are not distinct.")
    return {"prompt": prompt, "prompt_hash": digest(prompt), "input_ids": ids,
            "candidate_suffixes": candidates, "label_ids": label_ids}


class DiagnosticEvaluator(MCQEvaluator):
    def score_condition(self, example, variant):
        import torch
        prepared = diagnostic_prompt(self.tokenizer, example, self.settings, variant)
        inputs = padded_inputs([prepared], self.pad_id, self.device)
        with torch.inference_mode():
            full = next_token_logits(self.model, inputs)[0]
            selected = full[prepared["label_ids"]]
            mass = float((selected.logsumexp(-1) - full.logsumexp(-1)).exp())
            top = int(full.argmax())
        return dict(summarize_logits(selected.cpu().tolist(), example["answer"]),
                    **prepared, answer_probability_mass=mass,
                    unconstrained_next_token_id=top,
                    unconstrained_next_token=self.tokenizer.decode([top]))

    def direct_reference(self, example):
        """No optimized last-logit path or explicit position IDs: all-position reference."""
        import torch
        prepared = diagnostic_prompt(self.tokenizer, example, self.settings, "original")
        ids = torch.tensor([prepared["input_ids"]], device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=ids, use_cache=False, return_dict=True).logits[0, -1].float()
        return summarize_logits(logits[prepared["label_ids"]].cpu().tolist(), example["answer"])

    def harmless_continuation(self, example, variant):
        """At most 12 greedy tokens on invented harmless questions only."""
        import torch
        if not example["id"].startswith("sanity-"):
            raise ValueError("Free-text continuation is restricted to the harmless sanity questions.")
        prepared = diagnostic_prompt(self.tokenizer, example, self.settings, variant)
        ids = list(prepared["input_ids"])
        generated = []
        stops = getattr(self.model.generation_config, "eos_token_id", None)
        stops = set(stops if isinstance(stops, list) else [stops])
        with torch.inference_mode():
            for _ in range(12):
                inputs = padded_inputs([{"input_ids": ids}], self.pad_id, self.device)
                token = int(next_token_logits(self.model, inputs)[0].argmax())
                generated.append(token)
                if token in stops:
                    break
                ids.append(token)
        return {"id": example["id"], "variant": variant, "generated_token_ids": generated,
                "continuation": self.tokenizer.decode(generated, skip_special_tokens=True),
                "note": "Greedy continuation, capped at 12 tokens; may be incomplete. Not the baseline scoring rule."}


def condition_summary(rows):
    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for role in ("forget", "retain", "sanity"):
            group = [r for r in rows if r["variant"] == variant and r["role"] == role]
            ids = sorted({r["id"] for r in group})
            if not group:
                continue
            stable = sum(len({r["original_choice_prediction"] for r in group if r["id"] == identity}) == 1 for identity in ids)
            same_label = sum(len({r["prediction"] for r in group if r["id"] == identity}) == 1 for identity in ids)
            unrotated = [r for r in group if r["shift"] == 0]
            summary[variant][role] = {
                "unique_questions": len(ids), "evaluations": len(group),
                "unrotated_accuracy": sum(r["correct"] for r in unrotated) / len(unrotated),
                "rotation_averaged_accuracy": sum(r["correct"] for r in group) / len(group),
                "content_consistency_across_rotations": stable / len(ids),
                "same_label_across_rotations": same_label / len(ids),
                "predicted_labels": {label: sum(r["prediction"] == label for r in group) for label in LABELS},
                "mean_answer_probability_mass": statistics.mean(r["answer_probability_mass"] for r in group),
                "note": "Four orderings of each question are correlated, not four independent samples."}
    return summary


def run_diagnostic(data_dir="data/prepared/full", baseline_dir="outputs/stage2/full",
                   output="outputs/stage2_diagnostic", evaluator=None):
    baseline_dir, output = Path(baseline_dir), Path(output)
    original_run = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
    original_rows = {row["id"]: row for row in read_predictions(baseline_dir, original_run)}
    context = original_run["context"]
    if len(original_rows) != len(context["examples"]):
        raise ValueError("Complete original baseline evidence is required.")
    config, settings = context["config"], context["settings"]
    manifest, development = development_examples(data_dir, config)
    if manifest["request_hash"] != context["data_request_hash"]:
        raise ValueError("Diagnostic data differs from the original baseline.")
    selected = select_examples(development, config["seed"])
    selected += [dict(row, role="sanity", subject="invented", content_hash=digest(row)) for row in SANITY]
    base = evaluator if evaluator is not None else load_evaluator(config, settings)
    probe = DiagnosticEvaluator(base.model, base.tokenizer, settings)
    protocol = {"schema_version": 1, "baseline_fingerprint": original_run["fingerprint"],
                "config": config, "settings": settings, "variants": list(VARIANTS),
                "shifts": [0, 1, 2, 3], "ids": [row["id"] for row in selected],
                "environment": environment_info(), "source_hash": file_digest(Path(__file__)),
                "sample_rule": "hash-selected, 16 forget and 2 per retain subject; independent of original predictions",
                "purpose": "Diagnosis only. No automatic prompt selection, model switch, or final-test evaluation."}
    fingerprint = digest(protocol)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Diagnostic protocol changed; use a new output directory.")
    elif any(output.iterdir()):
        raise ValueError("Diagnostic output has files without a protocol.")
    else:
        write_json(protocol_path, protocol)
    rows, reference_checks, continuations = [], [], []
    expected_count = len(selected) * len(VARIANTS) * 4
    for example in selected:
        for variant in VARIANTS:
            for shift in range(4):
                transformed, order = rotate_example(example, shift)
                path = output / "predictions" / (digest([example["id"], variant, shift]) + ".json")
                if path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    checksum = row.pop("record_hash")
                    if checksum != digest(row) or row["protocol_hash"] != fingerprint:
                        raise ValueError("Diagnostic checkpoint mismatch.")
                    row["record_hash"] = checksum
                else:
                    result = probe.score_condition(transformed, variant)
                    row = dict(result, id=example["id"], role=example["role"], subject=example["subject"],
                               variant=variant, shift=shift, choice_order=order, protocol_hash=fingerprint,
                               original_choice_prediction=order[LABELS.index(result["prediction"])])
                    row["record_hash"] = digest(row)
                    write_json(path, row)
                rows.append(row)
                if len(rows) % 48 == 0:
                    print("Diagnostic {}/{} checks saved.".format(len(rows), expected_count), flush=True)
                if variant == "original" and shift == 0 and example["role"] != "sanity":
                    old = original_rows[example["id"]]
                    if row["input_ids"] != old["input_ids"] or row["prompt"] != old["prompt"]:
                        raise ValueError("Original prompt replay does not match the saved baseline.")
                    error = max(abs(row["answer_log_probabilities"][l] - old["answer_log_probabilities"][l]) for l in LABELS)
                    reference_checks.append({"id": example["id"], "kind": "saved_baseline_replay",
                                             "max_log_probability_error": error,
                                             "passed": error <= settings["score_atol"]})
        # Compare against an all-position native forward on two real questions per role.
        prior = sum(r["kind"] == "native_forward" and r["role"] == example["role"] for r in reference_checks)
        if example["role"] != "sanity" and prior < 2:
            reference = probe.direct_reference(example)
            result = next(r for r in rows if r["id"] == example["id"] and r["variant"] == "original" and r["shift"] == 0)
            error = max(abs(result["answer_log_probabilities"][l] - reference["answer_log_probabilities"][l]) for l in LABELS)
            reference_checks.append({"id": example["id"], "kind": "native_forward", "role": example["role"],
                                     "max_log_probability_error": error, "passed": error <= settings["score_atol"]})
        if example["role"] == "sanity":
            for variant in ("original", "assistant_prefix"):
                continuations.append(probe.harmless_continuation(example, variant))
    summary = {"status": "complete", "protocol_hash": fingerprint, "final_test_evaluated": False,
               "conditions": condition_summary(rows), "reference_checks": reference_checks,
               "numerical_checks_passed": all(row["passed"] for row in reference_checks),
               "harmless_continuations": continuations,
               "next_action": "Review replay, label-versus-position behavior, and prompt sensitivity. Do not automatically pick the highest accuracy or replace the baseline."}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_diagnostic(), indent=2))
