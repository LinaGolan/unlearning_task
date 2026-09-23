"""Shared next-token MCQ scoring. No generation, training, or answer-dependent prompt."""

import inspect
import math
import time

from .data import digest

LABELS = "ABCD"


def render_prompt(tokenizer, example, settings):
    # The correct answer is deliberately never included in the messages.
    text = settings["instruction"] + "\n\n" + example["question"] + "\n"
    text += "\n".join("{}. {}".format(label, choice)
                      for label, choice in zip(LABELS, example["choices"]))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, date_string=settings["date_string"])
    return prompt + settings.get("assistant_prefix", "")


def label_token_ids(tokenizer, answer_token_prefix=""):
    ids = [tokenizer.encode(answer_token_prefix + label, add_special_tokens=False) for label in LABELS]
    if any(len(item) != 1 for item in ids) or len({item[0] for item in ids}) != 4:
        raise ValueError("A-D must each be one distinct token; review the scoring rule.")
    return [item[0] for item in ids]


def prepare_prompt(tokenizer, example, settings, label_ids):
    prompt = render_prompt(tokenizer, example, settings)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > settings["max_input_tokens"]:
        raise ValueError("Prompt length {} exceeds the allowed range for {}. No truncation was applied."
                         .format(len(ids), example["id"]))
    # Verify the answer-token boundary in context, rather than assuming isolated
    # tokenization of 'A' always matches tokenization after the assistant header.
    for label, label_id in zip(LABELS, label_ids):
        suffix = settings.get("answer_token_prefix", "") + label
        if tokenizer.encode(prompt + suffix, add_special_tokens=False) != ids + [label_id]:
            raise ValueError("Answer-token boundary changed after the prompt; review the evaluator.")
    return {"prompt": prompt, "input_ids": ids, "prompt_hash": digest(prompt)}


def padded_inputs(prepared, pad_id, device):
    import torch
    width = max(len(row["input_ids"]) for row in prepared)
    ids = torch.full((len(prepared), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(prepared):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], device=device)
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp_min(0)}


def next_token_logits(model, inputs):
    """Select the last real token, including when batches contain right padding."""
    import torch
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    # Avoid allocating sequence_length x vocab_size logits for the default B=1.
    # Transformers 4 and 5 name this optimization differently.
    keep_last = False
    if inputs["attention_mask"].shape[0] == 1:
        parameters = inspect.signature(model.forward).parameters
        for key in ("logits_to_keep", "num_logits_to_keep"):
            if key in parameters:
                kwargs[key] = 1
                keep_last = True
                break
    logits = model(**kwargs).logits
    if keep_last:
        result = logits[:, -1, :]
    else:
        last = inputs["attention_mask"].sum(-1) - 1
        result = logits[torch.arange(logits.shape[0], device=logits.device), last]
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite next-token logits; do not continue the experiment.")
    return result.float()


def summarize_logits(logits, answer):
    """Stable A-D normalization, usable without PyTorch for result verification."""
    if len(logits) != 4 or any(not math.isfinite(x) for x in logits):
        raise ValueError("Expected four finite logits.")
    if type(answer) is not int or answer not in range(4):
        raise ValueError("Answer must be 0 through 3.")
    maximum = max(logits)
    shifted = [x - maximum for x in logits]
    log_total = math.log(sum(math.exp(x) for x in shifted))
    log_probs = [x - log_total for x in shifted]
    prediction = max(range(4), key=lambda i: logits[i])  # Ties: first letter.
    ordered = sorted(logits, reverse=True)
    return {"answer_logits": dict(zip(LABELS, logits)),
            "answer_probabilities": dict(zip(LABELS, [math.exp(x) for x in log_probs])),
            "answer_log_probabilities": dict(zip(LABELS, log_probs)),
            "correct_answer": LABELS[answer], "prediction": LABELS[prediction],
            "correct": prediction == answer,
            "correct_answer_log_probability": log_probs[answer],
            "correct_answer_probability": math.exp(log_probs[answer]),
            "top_two_logit_margin": ordered[0] - ordered[1]}


class MCQEvaluator:
    def __init__(self, model, tokenizer, settings):
        self.model, self.tokenizer, self.settings = model, tokenizer, settings
        self.device = next(model.parameters()).device
        self.label_ids = label_token_ids(tokenizer, settings.get("answer_token_prefix", ""))
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer has no usable padding token.")
        model.eval()
        model.requires_grad_(False)

    def synchronize(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def prepare(self, example):
        return prepare_prompt(self.tokenizer, example, self.settings, self.label_ids)

    def score(self, examples):
        import torch
        self.synchronize()
        start = time.perf_counter()
        prepared = [self.prepare(row) for row in examples]
        inputs = padded_inputs(prepared, self.pad_id, self.device)
        with torch.inference_mode():
            full_logits = next_token_logits(self.model, inputs)
            scores = full_logits[:, self.label_ids]
            mass = (scores.logsumexp(-1) - full_logits.logsumexp(-1)).exp().cpu().tolist()
            values = scores.cpu().tolist()
        self.synchronize()
        seconds = (time.perf_counter() - start) / len(examples)
        return [dict(summarize_logits(value, example["answer"]),
                     **item, input_tokens=len(item["input_ids"]),
                     answer_probability_mass=prob_mass, seconds_per_example=seconds)
                for example, item, value, prob_mass in zip(examples, prepared, values, mass)]


def evaluator_checks(evaluator):
    """Check batching and an independent teacher-forced next-token likelihood."""
    import torch
    examples = [
        {"id": "check-short", "question": "Which number is even?",
         "choices": ["3", "4", "5", "7"], "answer": 1},
        {"id": "check-long", "question": "A box contains two red balls and one blue ball. How many balls are in the box in total?",
         "choices": ["one", "two", "three", "four"], "answer": 2},
    ]
    singles = [evaluator.score([row])[0] for row in examples]
    batched = evaluator.score(examples)
    atol, rtol = evaluator.settings["score_atol"], evaluator.settings["score_rtol"]
    errors = []
    for single, batch in zip(singles, batched):
        left = torch.tensor(list(single["answer_log_probabilities"].values()))
        right = torch.tensor(list(batch["answer_log_probabilities"].values()))
        errors.append(float((left - right).abs().max()))
        if not torch.allclose(left, right, atol=atol, rtol=rtol):
            raise ValueError("Single and padded-batch scoring disagree; stop before evaluating data.")
        if single["prediction"] != batch["prediction"] and single["top_two_logit_margin"] > 2 * (atol + rtol * float(left.abs().max())):
            raise ValueError("Batching changed a prediction beyond numerical tolerance.")
    # Append each candidate and mask every target except that last token.
    # The model's own causal loss supplies an independent position/label check.
    prepared = evaluator.prepare(examples[0])
    losses = []
    with torch.inference_mode():
        for token in evaluator.label_ids:
            ids = torch.tensor([prepared["input_ids"] + [token]], device=evaluator.device)
            labels = torch.full_like(ids, -100)
            labels[0, -1] = token
            loss = evaluator.model(input_ids=ids, labels=labels, use_cache=False).loss
            losses.append(-float(loss))
    reference = torch.log_softmax(torch.tensor(losses), dim=-1)
    actual = torch.tensor(list(singles[0]["answer_log_probabilities"].values()))
    if not torch.allclose(reference, actual, atol=atol, rtol=rtol):
        raise ValueError("Next-token scores disagree with teacher-forced likelihoods.")
    return {"status": "passed", "batch_max_log_probability_error": max(errors),
            "teacher_forced_max_log_probability_error": float((reference - actual).abs().max()),
            "label_token_ids": dict(zip(LABELS, evaluator.label_ids)),
            "synthetic_predictions": [row["prediction"] for row in singles],
            "note": "Numerical implementation checks; synthetic answer accuracy is not a pass criterion."}


def gradient_timing_probe(evaluator, examples):
    """Input-gradient timing proxy; actual layer-gate validation belongs to Stage 3."""
    import torch
    measurements = []
    for example in examples:
        prepared = evaluator.prepare(example)
        inputs = padded_inputs([prepared], evaluator.pad_id, evaluator.device)
        evaluator.synchronize()
        if evaluator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(evaluator.device)
        start = time.perf_counter()
        embeddings = evaluator.model.get_input_embeddings()(inputs.pop("input_ids")).detach()
        embeddings.requires_grad_(True)
        logits = next_token_logits(evaluator.model, dict(inputs, inputs_embeds=embeddings))
        score = torch.log_softmax(logits[0, evaluator.label_ids], dim=-1)[example["answer"]]
        gradient, = torch.autograd.grad(score, embeddings)
        if not torch.isfinite(gradient).all():
            raise ValueError("Non-finite input gradients in timing probe.")
        evaluator.synchronize()
        measurements.append({"id": example["id"], "input_tokens": len(prepared["input_ids"]),
                             "forward_backward_seconds": time.perf_counter() - start,
                             "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(evaluator.device)
                             if evaluator.device.type == "cuda" else None})
        del embeddings, logits, score, gradient
    return {"status": "passed", "kind": "input_gradient_timing_proxy", "measurements": measurements,
            "note": "Weights are frozen. This measures no layer importance; Stage 3 must time and verify the actual gates."}
