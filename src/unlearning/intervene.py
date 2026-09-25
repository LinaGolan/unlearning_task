"""Two reversible activation interventions on selected decoder layers.

Both are forward hooks on whole decoder blocks, so model weights never change and
removing the hook restores the original model exactly. Layer indices are 0-based.

block_scale      h' = h_in + (1 - a) * (h_out - h_in)      (whole decoder block)
direction_ablate h' = h_out - a * ((h_out . u) - m) * u      (one residual direction)
"""

import torch

METHODS = ("block_scale", "direction_ablate")


def decoder_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not len(layers):
        raise ValueError("Expected a Llama-style model.model.layers decoder.")
    return layers


def _hidden(output):
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor):
        raise ValueError("Unsupported decoder block output.")
    return hidden


class Intervention:
    """Context manager applying one method to `layers` at strength `alpha`.

    `gates`, used only for differentiating the localization score, replaces alpha
    with one differentiable scale per layer: h' = h_out + (g - 1) * (h_out - h_in),
    which is exactly the identity at g = 1 while keeping a usable derivative.
    `directions` maps a layer index to (unit_direction, retain_mean_projection)
    and is required by direction_ablate.
    """

    def __init__(self, model, method="block_scale", layers=(), alpha=0.0,
                 directions=None, gates=None):
        self.model, self.method = model, method
        self.blocks = decoder_layers(model)
        self.layers = tuple(layers)
        self.alpha, self.directions, self.gates = float(alpha), directions or {}, gates
        self.handles = []
        if method not in METHODS:
            raise ValueError("Unknown intervention method: " + str(method))
        if any(type(i) is not int or not 0 <= i < len(self.blocks) for i in self.layers):
            raise ValueError("Layer indices must be 0-based integers inside the decoder.")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Duplicate layer indices are not allowed.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Strength alpha must lie in [0, 1].")
        if gates is not None:
            if self.layers or self.alpha or method != "block_scale":
                raise ValueError("Use either layers/alpha or a gate vector, not both.")
            if gates.shape != (len(self.blocks),):
                raise ValueError("Supply one gate per decoder layer.")
            self.layers = tuple(range(len(self.blocks)))
        if method == "direction_ablate" and any(i not in self.directions for i in self.layers):
            raise ValueError("direction_ablate needs a direction for every selected layer.")

    def _hook(self, index):
        def apply(module, args, kwargs, output):
            h_in = args[0] if args else kwargs.get("hidden_states")
            h_out = _hidden(output)
            if h_in is None or h_in.shape != h_out.shape:
                raise ValueError("Decoder block input and output shapes disagree.")
            if self.gates is not None:
                changed = h_out + (self.gates[index] - 1) * (h_out - h_in)
            elif self.alpha == 0.0:
                return output
            elif self.method == "block_scale":
                changed = h_in if self.alpha == 1.0 else h_out - self.alpha * (h_out - h_in)
            else:
                unit, mean = self.directions[index]
                unit = unit.to(h_out.dtype).to(h_out.device)
                projection = (h_out * unit).sum(-1, keepdim=True) - mean
                changed = h_out - self.alpha * projection * unit
            return (changed,) + output[1:] if isinstance(output, tuple) else changed
        return apply

    def __enter__(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("Use an eval-mode model with frozen weights.")
        try:
            for index in self.layers:
                self.handles.append(
                    self.blocks[index].register_forward_hook(self._hook(index), with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def gate_gradient(evaluator, example, objective="logprob"):
    """d objective / d g_l at g = 1, for every layer at once.

    objective="logprob" differentiates the correct letter's log probability,
    normalized over A-D. objective="margin" differentiates the decision margin
    z_correct - max_(i != correct) z_i, which is the quantity that decides whether
    the prediction is right, so its gradient speaks directly to accuracy.
    A positive value predicts that weakening layer l by alpha lowers the objective
    by roughly alpha * value.
    """
    inputs, last = evaluator._batch([example])
    model = evaluator.model
    gates = torch.ones(len(decoder_layers(model)), device=evaluator.device,
                       dtype=next(model.parameters()).dtype, requires_grad=True)
    with torch.enable_grad():
        with Intervention(model, gates=gates):
            logits = evaluator._forward(inputs, last)
            answers = logits[0, evaluator.label_ids]
            target = example["answer"]
            if objective == "margin":
                rivals = torch.cat([answers[:target], answers[target + 1:]])
                score = answers[target] - rivals.max()
            elif objective == "logprob":
                score = torch.log_softmax(answers, dim=-1)[target]
            else:
                raise ValueError("Unknown gradient objective: " + str(objective))
        gradient, = torch.autograd.grad(score, gates)
    if not torch.isfinite(gradient).all():
        raise ValueError("Non-finite gate gradient for " + example["id"])
    return float(score.detach()), gradient.detach().float().cpu().tolist()
