"""Two layer-localization scores over the same 128 forget and 128 retain questions.

Gradient attribution (method G)
  For decoder block l write its residual contribution r_l = h_out - h_in and gate
  it by g_l, so h_out(g_l) = h_in + g_l * r_l. With M(x) = log p(correct letter | x)
  normalized over A-D,
      s_l(x) = dM(x)/dg_l at g = 1,  F_l = mean_forget s_l,  R_l = mean_retain s_l,
      S_l = F_l - R_l.
  F_l says how strongly the answer relies on layer l; S_l asks for reliance that is
  specific to WMDP rather than shared with the retain set. It is a first-order
  estimate around the unmodified model, so it need not predict strong ablation.

Activation statistics (method D)
  Using each block's input h_in,l and output h_out,l at the final prompt token,
      C_l = mean_forget ||r_l|| / ||h_in,l||                        (WMDP-only)
      d_l = mean_forget h_out,l - mean_retain h_out,l,  u_l = d_l / ||d_l||
      A_l = held-out split-half accuracy of thresholding h . u_l    (WMDP-vs-retain)
  C_l measures how much layer l moves the stream on WMDP prompts; A_l measures how
  linearly separable the two domains are in its output. u_l also supplies the
  direction that the rank-1 intervention removes.
"""

import random
import statistics

import torch

from .data import digest
from .intervene import decoder_layers, gate_gradient

def spearman(left, right):
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for position in range(i, j + 1):
                ranks[order[position]] = shared
            i = j + 1
        return ranks
    a, b = rank(left), rank(right)
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    top = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    scale = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return top / scale if scale else 0.0


def _halves(examples, seed):
    keyed = sorted(examples, key=lambda e: digest([seed, "half", e["id"]]))
    return keyed[::2], keyed[1::2]


def gradient_scores(evaluator, examples, seed, progress=None, objective="logprob"):
    """Mean signed gate gradients per layer, plus a split-half stability check."""
    per_question = {}
    for index, example in enumerate(examples):
        score, gradients = gate_gradient(evaluator, example, objective)
        per_question[example["id"]] = {"role": example["role"], "subject": example["subject"],
                                       "correct_log_probability": score, "gradients": gradients}
        if progress and (index + 1) % 32 == 0:
            progress("gradients ({}) {}/{}".format(objective, index + 1, len(examples)))
    layers = len(decoder_layers(evaluator.model))

    def means(rows, role):
        chosen = [r["gradients"] for r in rows if r["role"] == role]
        return [statistics.mean(g[l] for g in chosen) for l in range(layers)]

    rows = list(per_question.values())
    forget, retain = means(rows, "forget"), means(rows, "retain")
    scores = {"forget": forget, "retain": retain,
              "selective": [f - r for f, r in zip(forget, retain)]}
    first, second = _halves(examples, seed)
    halves = []
    for half in (first, second):
        subset = [per_question[e["id"]] for e in half]
        f, r = means(subset, "forget"), means(subset, "retain")
        halves.append({"forget": f, "retain": r, "selective": [a - b for a, b in zip(f, r)]})
    stability = {name: spearman(halves[0][name], halves[1][name]) for name in scores}
    return scores, {"split_half_spearman": stability,
                    "split_half_scores": halves,
                    "per_question": per_question}


def activation_scores(evaluator, examples, seed):
    """Per-layer contribution and domain separability, plus the ablation directions."""
    before, after = evaluator.block_activations(examples)   # each [n, layers, d_model]
    is_forget = torch.tensor([e["role"] == "forget" for e in examples])
    layers = before.shape[1]

    def contribution_of(h_in, h_out, flags):
        return float(((h_out - h_in).norm(dim=-1) / h_in.norm(dim=-1))[flags].mean())

    contribution, separability, directions = [], [], {}
    for layer in range(layers):
        contribution.append(contribution_of(before[:, layer], after[:, layer], is_forget))
        direction = after[is_forget, layer].mean(0) - after[~is_forget, layer].mean(0)
        unit = direction / direction.norm()
        projection = after[:, layer] @ unit
        directions[layer] = {"unit": unit.tolist(),
                             "retain_mean_projection": float(projection[~is_forget].mean()),
                             "forget_mean_projection": float(projection[is_forget].mean()),
                             "direction_norm": float(direction.norm())}
        separability.append(_split_half_probe(after[:, layer], is_forget, examples, seed))
    scores = {"contribution": contribution, "separability": separability}

    index = {e["id"]: i for i, e in enumerate(examples)}
    halves = []
    for half in _halves(examples, seed):
        rows = torch.tensor([index[e["id"]] for e in half])
        flags = is_forget[rows]
        halves.append({"contribution": [contribution_of(before[rows, l], after[rows, l], flags)
                                        for l in range(layers)],
                       "separability": [_split_half_probe(after[rows, l], flags, half, seed)
                                        for l in range(layers)]})
    stability = {name: spearman(halves[0][name], halves[1][name]) for name in scores}
    return scores, {"split_half_spearman": stability, "split_half_scores": halves,
                    "directions": directions}


def _split_half_probe(activations, is_forget, examples, seed):
    """Difference-in-means classifier fitted on one half, scored on the other."""
    first, second = _halves(examples, seed)
    index = {e["id"]: i for i, e in enumerate(examples)}
    correct = total = 0
    for train, test in ((first, second), (second, first)):
        rows = torch.tensor([index[e["id"]] for e in train])
        held = torch.tensor([index[e["id"]] for e in test])
        flags = is_forget[rows]
        if not flags.any() or flags.all():
            continue
        unit = activations[rows][flags].mean(0) - activations[rows][~flags].mean(0)
        unit = unit / unit.norm()
        fitted = activations[rows] @ unit
        threshold = (fitted[flags].mean() + fitted[~flags].mean()) / 2
        predicted = (activations[held] @ unit) > threshold
        correct += int((predicted == is_forget[held]).sum())
        total += len(held)
    return correct / total if total else 0.0


def select(values, k, largest=True):
    """The k highest (or lowest) scoring layers; ties resolve to the lower index."""
    order = sorted(range(len(values)), key=(lambda i: (-values[i], i)) if largest
                   else (lambda i: (values[i], i)))
    return tuple(sorted(order[:k]))


def random_layers(n_layers, k, seed):
    return tuple(sorted(random.Random(seed).sample(range(n_layers), k)))
