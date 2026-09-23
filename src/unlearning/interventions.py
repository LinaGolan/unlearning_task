"""Temporary decoder-block residual gates; no model parameter updates."""

import math
import weakref

_ACTIVE = weakref.WeakKeyDictionary()


def decoder_layers(model):
    try:
        layers = model.model.layers
    except AttributeError:
        raise ValueError("Expected a Llama-style model.model.layers decoder.") from None
    if not len(layers):
        raise ValueError("The decoder has no layers.")
    return layers


def hidden_output(output):
    """Transformers 4 returns a tuple; newer Llama blocks can return a tensor."""
    import torch
    hidden = output[0] if isinstance(output, tuple) and output else output
    if not isinstance(hidden, torch.Tensor):
        raise ValueError("Unsupported decoder output; expected a tensor or tensor-first tuple.")
    return hidden


class LayerIntervention:
    """Apply h' = h_in + (1-alpha)*(h_out-h_in) at every token position.

    Layer indices are zero-based. For differentiation, supply a floating gate
    vector instead: h' = h_out + (g-1)*(h_out-h_in). This equal expression keeps
    g=1 exactly unchanged while retaining its derivative. Gate values slightly
    above 1 are allowed ONLY through this low-level API for central differences.
    Do not share one model between concurrent contexts or use cached generation.
    """

    def __init__(self, model, layers=(), alpha=0.0, enabled=True, gates=None):
        import torch
        self.model = model
        self.blocks = decoder_layers(model)
        self.layers = tuple(layers)
        if any(type(index) is not int or not 0 <= index < len(self.blocks) for index in self.layers):
            raise ValueError("Layer indices must be zero-based integers inside the decoder.")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Duplicate layer indices are not allowed.")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Intervention strength alpha must be finite and between 0 and 1.")
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean.")
        if gates is not None:
            if self.layers or alpha != 0:
                raise ValueError("Use either selected layers/alpha or a gate vector, not both.")
            parameter = next(model.parameters())
            if not isinstance(gates, torch.Tensor) or gates.shape != (len(self.blocks),):
                raise ValueError("Supply one gate per decoder layer.")
            if gates.dtype != parameter.dtype or gates.device != parameter.device:
                raise ValueError("Gates must match the model's floating dtype and device.")
            if not gates.is_floating_point() or not torch.isfinite(gates).all():
                raise ValueError("Gates must be finite floating-point values.")
            self.layers = tuple(range(len(self.blocks)))
        self.alpha, self.enabled, self.gates = float(alpha), enabled, gates
        self.handles = []
        self.entered = False

    def _hook(self, index):
        def apply(module, args, kwargs, output):
            h_in = args[0] if args else kwargs.get("hidden_states")
            h_out = hidden_output(output)
            if h_in is None or h_in.shape != h_out.shape:
                raise ValueError("Decoder input/output shapes do not match.")
            if self.gates is not None:
                changed = h_out + (self.gates[index] - 1) * (h_out - h_in)
            elif self.alpha == 0:
                return output
            elif self.alpha == 1:
                changed = h_in
            else:
                changed = h_out - self.alpha * (h_out - h_in)
            return (changed,) + output[1:] if isinstance(output, tuple) else changed
        return apply

    def __enter__(self):
        if self.entered or self.model in _ACTIVE:
            raise RuntimeError("Nested interventions on the same model are not supported.")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("Use an evaluation-mode model with all model weights frozen.")
        self.entered = True
        _ACTIVE[self.model] = self
        try:
            if self.enabled:
                for index in self.layers:
                    self.handles.append(self.blocks[index].register_forward_hook(self._hook(index), with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if _ACTIVE.get(self.model) is self:
            del _ACTIVE[self.model]
        self.entered = False
        return False


def gate_gradients(evaluator, example):
    """Differentiate correct A-D log probability at g=1, without training.

    Returns detached CPU numbers so callers cannot retain a model-sized graph.
    Positive s_l predicts a positive score drop alpha*s_l when suppressing l.
    """
    import time
    import torch
    from .evaluation import next_token_logits, padded_inputs

    if torch.is_inference_mode_enabled():
        raise ValueError("Gate derivatives cannot run inside torch.inference_mode().")
    prepared = evaluator.prepare(example)
    inputs = padded_inputs([prepared], evaluator.pad_id, evaluator.device)
    evaluator.synchronize()
    if evaluator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluator.device)
    start = time.perf_counter()
    with torch.enable_grad():
        gates = torch.ones(len(decoder_layers(evaluator.model)), device=evaluator.device,
                           dtype=next(evaluator.model.parameters()).dtype, requires_grad=True)
        with LayerIntervention(evaluator.model, gates=gates):
            logits = next_token_logits(evaluator.model, inputs)[0, evaluator.label_ids]
            log_probs = torch.log_softmax(logits, dim=-1)
            score = log_probs[example["answer"]]
            gradients, = torch.autograd.grad(score, gates)
    evaluator.synchronize()
    elapsed = time.perf_counter() - start
    if not torch.isfinite(gradients).all():
        raise ValueError("Non-finite layer-gate derivative.")
    return {"id": example["id"], "input_tokens": len(prepared["input_ids"]),
            "correct_answer_log_probability": float(score.detach()),
            "answer_log_probabilities": log_probs.detach().cpu().tolist(),
            "gradients": gradients.detach().cpu().tolist(),
            "forward_backward_seconds": elapsed,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(evaluator.device)
            if evaluator.device.type == "cuda" else None}
