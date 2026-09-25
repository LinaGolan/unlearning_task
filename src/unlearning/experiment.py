"""End-to-end experiment driver. Every stage writes JSON and is resumable.

Order: prepare data, check the mechanism, localize on the localization split,
select layers, sweep strengths on development, freeze the operating point, then
evaluate the frozen choices on the held-out test split and run the controls.
"""

import json
import platform
import time
from pathlib import Path

import torch

from . import data as datamod
from .data import digest, load_split, read_config, write_json
from .evaluate import LABELS, Evaluator, accuracy, label_token_ids, render
from .intervene import Intervention, decoder_layers, gate_gradient
from .localize import activation_scores, gradient_scores, random_layers, select

# Each method pairs a localization score with the intervention it naturally implies.
# localized/selective name the WMDP-only and forget-vs-retain score keys.
METHODS = {
    "gate": {"intervention": "block_scale", "unit": "layer",
             "localized": "forget", "selective": "selective"},
    "gate_margin": {"intervention": "block_scale", "unit": "layer",
                    "localized": "forget", "selective": "selective"},
    "direction": {"intervention": "direction_ablate", "unit": "layer",
                  "localized": "contribution", "selective": "separability"},
}


def log(message):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), message), flush=True)


def load_model(config, device=None):
    """Load the pinned model. transformers 5 renamed `torch_dtype` to `dtype`, so try
    both and then verify the weights really came back in the requested precision —
    a silently ignored kwarg would change every number in the experiment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = config["model"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    wanted = getattr(torch, spec["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    common = {"revision": spec["revision"], "attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(spec["id"], dtype=wanted, **common)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(spec["id"], torch_dtype=wanted, **common)
    model = model.to(device)
    actual = next(model.parameters()).dtype
    if actual != wanted:
        raise ValueError("Model loaded as {} but {} was requested; the precision kwarg was "
                         "ignored by this transformers version.".format(actual, wanted))
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def environment(model):
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
            "layers": len(decoder_layers(model)),
            "hidden_size": model.config.hidden_size,
            "parameters": sum(p.numel() for p in model.parameters())}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    import transformers
    info["transformers"] = transformers.__version__
    return info


def mechanism_checks(evaluator, examples):
    """Prove the interventions are identity at alpha=0, reversible, and differentiated correctly."""
    model = evaluator.model
    sample = examples[:4]
    base = evaluator.score(sample)
    checks = {}
    before = len(list(model.modules())), sum(len(b._forward_hooks) for b in decoder_layers(model))
    with Intervention(model, "block_scale", layers=(0, 1), alpha=0.0):
        checks["block_scale_alpha0_identity"] = evaluator.score(sample) == base
    with Intervention(model, "block_scale", layers=(0,), alpha=1.0):
        suppressed = evaluator.score(sample)
    checks["block_scale_alpha1_changes_scores"] = suppressed != base
    directions = {0: (torch.zeros(model.config.hidden_size), 0.0)}
    with Intervention(model, "direction_ablate", layers=(0,), alpha=0.0, directions=directions):
        checks["direction_alpha0_identity"] = evaluator.score(sample) == base
    checks["hooks_removed"] = (len(list(model.modules())),
                               sum(len(b._forward_hooks) for b in decoder_layers(model))) == before
    checks["weights_frozen"] = not any(p.requires_grad for p in model.parameters())

    # Central finite differences on three layers confirm the analytic gate gradient.
    layers = len(decoder_layers(model))
    probes, step = [], 0.01
    for example in sample[:2]:
        _, analytic = gate_gradient(evaluator, example)
        inputs, last = evaluator._batch([example])
        for layer in (0, layers // 2, layers - 1):
            values = []
            for delta in (step, -step):
                gates = torch.ones(layers, device=evaluator.device,
                                   dtype=next(model.parameters()).dtype)
                gates[layer] += delta
                with Intervention(model, gates=gates), torch.inference_mode():
                    logits = evaluator._forward(inputs, last)
                values.append(float(torch.log_softmax(logits[0, evaluator.label_ids], -1)[example["answer"]]))
            numeric = (values[0] - values[1]) / (2 * step)
            probes.append({"id": example["id"], "layer": layer, "analytic": analytic[layer],
                           "finite_difference": numeric, "absolute_error": abs(numeric - analytic[layer])})
    checks["gate_gradient_matches_finite_difference"] = max(p["absolute_error"] for p in probes) < 5e-3
    if not all(v for k, v in checks.items()):
        raise RuntimeError("Mechanism checks failed: " + json.dumps(checks))
    return {"checks": checks, "finite_difference_probes": probes,
            "max_absolute_gradient_error": max(p["absolute_error"] for p in probes)}


def prompt_examples(evaluator, examples):
    """Record one rendered prompt per style so the exact scored text is auditable."""
    out = {}
    for style in ("harness", "chat_prefix", "chat_plain"):
        text, special = render(evaluator.tokenizer, examples[0],
                              evaluator.topics[examples[0]["subject"]], style)
        out[style] = {"prompt": text, "add_special_tokens": special,
                      "scored_tokens": dict(zip(LABELS, label_token_ids(evaluator.tokenizer, style)))}
    return out


def condition_name(method, selection, alpha):
    return "{}__{}__a{:g}".format(method, selection, alpha)


def evaluate_condition(evaluator, examples, out_dir, name, kind=None, alpha=0.0, **kwargs):
    """Score `examples`, caching per-question records under results/predictions/."""
    path = Path(out_dir) / "predictions" / (name + ".jsonl")
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if kind is None or alpha == 0.0:
        records = evaluator.score(examples)
    else:
        with Intervention(evaluator.model, kind, alpha=alpha, **kwargs):
            records = evaluator.score(examples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    return records


def tensor_directions(saved, device):
    return {int(layer): (torch.tensor(info["unit"], device=device), info["retain_mean_projection"])
            for layer, info in saved.items()}


def selections_for(method, scores, config, universe):
    """The four task conditions: top localized, top selective, bottom localized, random."""
    spec = METHODS[method]
    budget = config["intervention"]["k"]
    chosen = {"top_localized": select(scores[spec["localized"]], budget, largest=True),
              "top_selective": select(scores[spec["selective"]], budget, largest=True),
              "bottom_localized": select(scores[spec["localized"]], budget, largest=False)}
    for seed in config["intervention"]["random_seeds"]:
        chosen["random_{}".format(seed)] = random_layers(universe, budget, seed)
    return chosen


def intervention_kwargs(method, chosen, directions):
    """Translate a selection into the keyword arguments Intervention expects."""
    if METHODS[method]["intervention"] == "direction_ablate":
        return {"layers": chosen, "directions": directions}
    return {"layers": chosen}


def drop_table(baseline, records_by_condition):
    """Accuracy and percentage-point drop per role for each condition."""
    table = {}
    for name, records in records_by_condition.items():
        row = {}
        for role in ("forget", "retain"):
            subset = [r for r in records if r["role"] == role]
            if not subset:
                continue
            base = [r for r in baseline if r["role"] == role]
            summary = accuracy(subset)
            row[role] = dict(summary, drop_pp=100 * (accuracy(base)["accuracy"] - summary["accuracy"]))
        table[name] = row
    return table


def choose_strength(table, strengths, budget_pp):
    """Largest WMDP drop whose retain drop stays inside the budget; ties take weaker alpha."""
    eligible = []
    for alpha in strengths:
        if alpha == 0:
            continue
        row = table.get(alpha)
        if row and row["forget"]["drop_pp"] > 0 and row["retain"]["drop_pp"] <= budget_pp:
            eligible.append((row["forget"]["drop_pp"], -alpha, alpha))
    if eligible:
        return max(eligible)[2], "largest WMDP drop with retain drop <= {} pp".format(budget_pp)
    fallback = max(((table[a]["forget"]["drop_pp"] - table[a]["retain"]["drop_pp"], -a, a)
                    for a in strengths if a and a in table), default=None)
    if fallback is None:
        raise ValueError("No development strengths were evaluated.")
    return fallback[2], "no strength met the retain budget; largest WMDP-minus-retain margin instead"


def run(config_path, data_dir, out_dir, cache_dir, device=None, k=None, strengths=None,
        methods=None):
    """`k`, `strengths` and `methods` override the config, so sweeping a dimension
    needs no new config file."""
    config = read_config(config_path)
    if k is not None:
        config["intervention"]["k"] = k
    if strengths is not None:
        config["intervention"]["strengths"] = strengths
    active = [m for m in METHODS if m in (methods or config["intervention"]["methods"])]
    if not active:
        raise ValueError("No known methods selected; choose from " + ", ".join(METHODS))
    config["intervention"]["methods"] = active
    out_dir, started = Path(out_dir), time.time()
    log("preparing data")
    manifest = datamod.prepare(config, data_dir, cache_dir)
    write_json(out_dir / "data_manifest.json", manifest)

    log("loading model")
    model, tokenizer = load_model(config, device)
    evaluator = Evaluator(model, tokenizer, config)
    layers = len(decoder_layers(model))
    splits = {(role, split): load_split(data_dir, role, split)
              for role in ("forget", "retain") for split in ("localization", "development", "test")}
    splits[("biology", "test")] = load_split(data_dir, "biology", "test")
    localization = [row for pair in zip(splits[("forget", "localization")],
                                        splits[("retain", "localization")]) for row in pair]

    log("mechanism checks")
    setup = {"environment": environment(model), "config": config,
             "prompt_styles": prompt_examples(evaluator, localization),
             "mechanism": mechanism_checks(evaluator, localization)}
    write_json(out_dir / "setup.json", setup)
    log("gate gradient max finite-difference error {:.2e}".format(setup["mechanism"]["max_absolute_gradient_error"]))

    path = out_dir / "localization.json"
    if path.exists():
        localization_result = json.loads(path.read_text(encoding="utf-8"))
        directions_saved = json.loads((out_dir / "directions.json").read_text(encoding="utf-8"))
    else:
        localization_result = {"questions": {role: sum(e["role"] == role for e in localization)
                                            for role in ("forget", "retain")}}
        directions_saved = {}
        for method in active:
            if method in ("gate", "gate_margin"):
                objective = "margin" if method == "gate_margin" else "logprob"
                log("localization: gradient attribution ({}) over {} questions".format(
                    objective, len(localization)))
                scores, detail = gradient_scores(evaluator, localization, config["seed"], log, objective)
                localization_result["per_question_gradients_" + objective] = detail["per_question"]
            else:
                log("localization: activation statistics")
                scores, detail = activation_scores(evaluator, localization, config["seed"])
                directions_saved = detail.pop("directions")
            localization_result[method] = {
                "scores": scores, "split_half_spearman": detail.get("split_half_spearman", {}),
                "split_half_scores": detail.get("split_half_scores", [])}
        write_json(path, localization_result)
        write_json(out_dir / "directions.json", directions_saved)
    directions = tensor_directions(directions_saved, evaluator.device)

    selections = {method: selections_for(method, localization_result[method]["scores"], config, layers)
                  for method in active}
    kwargs_for = {method: {name: intervention_kwargs(method, chosen, directions)
                           for name, chosen in selections[method].items()}
                  for method in active}
    log("selected layers: " + json.dumps({m: {k: list(v) for k, v in sel.items()}
                                          for m, sel in selections.items()}))

    strengths = config["intervention"]["strengths"]
    sweeps, baselines = {}, {}
    for split in ("development", "test"):
        questions = splits[("forget", split)] + splits[("retain", split)]
        log("baseline on {} ({} questions)".format(split, len(questions)))
        baselines[split] = evaluate_condition(evaluator, questions, out_dir, split + "__baseline")
        sweeps[split] = {}
        for method in active:
            kind = METHODS[method]["intervention"]
            # Development only needs the condition the strength rule reads; the full
            # grid of conditions is reserved for the held-out test split.
            wanted = ["top_selective"] if split == "development" else list(selections[method])
            for selection in wanted:
                for alpha in strengths:
                    if alpha == 0:
                        sweeps[split][(method, selection, alpha)] = baselines[split]
                        continue
                    name = split + "__" + condition_name(method, selection, alpha)
                    records = evaluate_condition(evaluator, questions, out_dir, name, kind,
                                                 alpha, **kwargs_for[method][selection])
                    sweeps[split][(method, selection, alpha)] = records
            log("{}: {} sweep complete".format(split, method))

    operating = {}
    for method in active:
        table = {alpha: drop_table(baselines["development"],
                                   {"x": sweeps["development"][(method, "top_selective", alpha)]})["x"]
                 for alpha in strengths}
        alpha, rule = choose_strength(table, strengths, config["retain_budget_pp"])
        operating[method] = {"alpha": alpha, "rule": rule,
                             "development_table": {str(a): table[a] for a in table}}
        log("{}: development-selected alpha = {} ({})".format(method, alpha, rule))

    log("controls: general biology, alternative prompts, cross-check")
    controls = {"biology": {}, "prompts": {}, "cross": {}}
    biology = splits[("biology", "test")]
    controls["biology"]["baseline"] = evaluate_condition(evaluator, biology, out_dir, "biology__baseline")
    for method in active:
        kind, alpha = METHODS[method]["intervention"], operating[method]["alpha"]
        for selection in selections[method]:
            name = "biology__" + condition_name(method, selection, alpha)
            controls["biology"][method + "/" + selection] = evaluate_condition(
                evaluator, biology, out_dir, name, kind, alpha, **kwargs_for[method][selection])

    subset = [e for e in splits[("forget", "test")] + splits[("retain", "test")]
              if int(digest([config["seed"], "prompt_subset", e["id"]])[:8], 16) % 2 == 0]
    for style in [config["prompt"]] + config["alternative_prompts"]:
        other = Evaluator(model, tokenizer, config, style)
        controls["prompts"][style] = {"baseline": evaluate_condition(
            other, subset, out_dir, "prompt_{}__baseline".format(style))}
        for method in active:
            kind, alpha = METHODS[method]["intervention"], operating[method]["alpha"]
            name = "prompt_{}__{}".format(style, condition_name(method, "top_selective", alpha))
            controls["prompts"][style][method] = evaluate_condition(
                other, subset, out_dir, name, kind, alpha, **kwargs_for[method]["top_selective"])
        log("alternative prompt {} complete".format(style))

    test_questions = splits[("forget", "test")] + splits[("retain", "test")]
    for target in active:
        for donor in active:
            kind, alpha = METHODS[donor]["intervention"], operating[donor]["alpha"]
            chosen = selections[target]["top_selective"]
            name = "cross__target{}__{}__a{:g}".format(target, kind, alpha)
            kwargs = intervention_kwargs(donor, chosen, directions)
            controls["cross"]["units={}/intervention={}".format(target, kind)] = {
                "alpha": alpha,
                "records": evaluate_condition(evaluator, test_questions, out_dir, name,
                                              kind, alpha, **kwargs)}

    summary = {"schema_version": 2, "elapsed_seconds": time.time() - started,
               "methods": active, "layer_count": layers,
               "selections": {m: {k: list(v) for k, v in s.items()} for m, s in selections.items()},
               "operating_point": operating, "strengths": strengths,
               "questions": {"localization": len(localization),
                             "development": len(splits[("forget", "development")]) * 2,
                             "test": len(splits[("forget", "test")]) * 2,
                             "biology": len(biology), "prompt_subset": len(subset)},
               "prediction_files": sorted(p.name for p in (out_dir / "predictions").glob("*.jsonl")),
               "environment": setup["environment"]}
    write_json(out_dir / "run.json", summary)
    log("run complete in {:.1f} minutes".format(summary["elapsed_seconds"] / 60))
    return summary
