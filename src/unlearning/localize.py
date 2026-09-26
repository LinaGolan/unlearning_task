"""Layer localization by gradient attribution over the localization questions.

For decoder block l write its residual contribution r_l = h_out - h_in and gate it
by g_l, so h_out(g_l) = h_in + g_l * r_l. For an objective M(x),

    s_l(x) = dM(x)/dg_l at g = 1,  F_l = mean_forget s_l,  R_l = mean_retain s_l,
    S_l = F_l - R_l.

F_l says how strongly the answer relies on layer l; S_l asks for reliance specific
to WMDP rather than shared with the retain set. It is a first-order estimate around
the unmodified model, so it need not predict strong ablation.

Each mean is reported with its standard error, because choosing a layer is an argmax
over 16 noisy estimates and the useful question is whether the best layer is separable
from the runner-up at all.
"""

import math
import random
import statistics

from .data import digest
from .intervene import decoder_layers, gate_gradient


def gradient_scores(evaluator, examples, seed, progress=None, objective="logprob"):
    """Mean signed gate gradients per layer, with the standard error of each mean.

    The standard errors matter: selecting a layer means taking an argmax over 16 noisy
    estimates, so whether the best layer is actually distinguishable from the runner-up
    is a property of the data, not of the score. `selection` records that separation.
    """
    per_question = {}
    for index, example in enumerate(examples):
        score, gradients = gate_gradient(evaluator, example, objective)
        per_question[example["id"]] = {"role": example["role"], "subject": example["subject"],
                                       "correct_log_probability": score, "gradients": gradients}
        if progress and (index + 1) % 32 == 0:
            progress("gradients ({}) {}/{}".format(objective, index + 1, len(examples)))
    layers = len(decoder_layers(evaluator.model))
    rows = list(per_question.values())

    def summarise(role):
        chosen = [r["gradients"] for r in rows if r["role"] == role]
        means = [statistics.mean(g[l] for g in chosen) for l in range(layers)]
        errors = [statistics.stdev(g[l] for g in chosen) / math.sqrt(len(chosen))
                  for l in range(layers)]
        return means, errors, len(chosen)

    def pick_on(subset):
        """Which layer each score would select from a subset of the questions."""
        out = {}
        for role in ("forget", "retain"):
            chosen = [r["gradients"] for r in subset if r["role"] == role]
            out[role] = [statistics.mean(g[l] for g in chosen) for l in range(layers)]
        out["selective"] = [f - r for f, r in zip(out["forget"], out["retain"])]
        return {name: max(range(layers), key=lambda i: values[i]) for name, values in out.items()}

    # Split the questions in two by hash and record which layer each half would choose.
    keyed = sorted(per_question.items(), key=lambda kv: digest([seed, "half", kv[0]]))
    halves = [pick_on([v for _, v in keyed[::2]]), pick_on([v for _, v in keyed[1::2]])]

    forget, forget_se, n_forget = summarise("forget")
    retain, retain_se, n_retain = summarise("retain")
    scores = {"forget": forget, "retain": retain,
              "selective": [f - r for f, r in zip(forget, retain)]}
    # S_l is a difference of two independent means, so its errors add in quadrature.
    errors = {"forget": forget_se, "retain": retain_se,
              "selective": [math.hypot(a, b) for a, b in zip(forget_se, retain_se)]}

    n_used = min(n_forget, n_retain)
    selection = {}
    for name, values in scores.items():
        order = sorted(range(layers), key=lambda i: -values[i])
        best, runner_up = order[0], order[1]
        gap = values[best] - values[runner_up]
        spread = math.hypot(errors[name][best], errors[name][runner_up])
        selection[name] = {"best_layer": best, "runner_up": runner_up, "gap": gap,
                           "gap_standard_error": spread,
                           "gap_in_standard_errors": gap / spread if spread else float("inf"),
                           "layers_within_one_standard_error": [
                               l for l in range(layers)
                               if values[l] > values[best] - math.hypot(errors[name][best],
                                                                       errors[name][l])],
                           "layer_chosen_by_each_half": [h[name] for h in halves],
                           # s.e. shrinks as 1/sqrt(n), so this is how many questions per role
                           # would be needed before the top two layers differ by 2 s.e.
                           "questions_for_two_standard_errors":
                               round(n_used * (2 * spread / gap) ** 2) if gap > 0 else None}
    return scores, {"standard_errors": errors, "selection": selection,
                    "questions": {"forget": n_forget, "retain": n_retain},
                    "per_question": per_question}


def select(values, k, largest=True):
    """The k highest (or lowest) scoring layers; ties resolve to the lower index."""
    order = sorted(range(len(values)), key=(lambda i: (-values[i], i)) if largest
                   else (lambda i: (values[i], i)))
    return tuple(sorted(order[:k]))


def random_layers(n_layers, k, seed):
    return tuple(sorted(random.Random(seed).sample(range(n_layers), k)))
