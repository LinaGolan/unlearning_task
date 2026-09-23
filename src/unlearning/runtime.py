"""Runtime evidence and one harmless forward-pass check, not a baseline study."""

import importlib.metadata
import platform
import sys
import time

from .data import write_json
from .access import failure_details


def environment_info():
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "accelerate", "safetensors", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": sys.version, "platform": platform.platform(), "packages": packages}
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory}
    except ImportError:
        report["cuda_available"] = False
    return report


def run_smoke(config, output, tiny=False):
    report = {
        "status": "started", "test_kind": "tiny_random_model" if tiny else "recommended_model",
        "environment": environment_info(), "is_research_result": False,
    }
    write_json(output, report)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

        if tiny:
            torch.manual_seed(config["seed"])
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=64,
            ))
            inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
            label_ids = [5, 6, 7, 8]
            report["model"] = {"id": "random-tiny-llama", "pretrained": False}
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("No GPU found. Select a GPU runtime in Colab, then retry. No model weights were loaded.")
            settings = config["model"]
            report["model"] = settings.copy()
            write_json(output, report)
            try:
                tokenizer = AutoTokenizer.from_pretrained(settings["id"], revision=settings["revision"])
            except Exception as exc:
                report.update(failure_details(exc, "tokenizer_download"))
                raise RuntimeError(report["error"]) from None
            question = (
                "Choose the correct answer. Reply with only A, B, C, or D.\n\n"
                "Which number is even?\nA. 3\nB. 4\nC. 5\nD. 7"
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            encoded_labels = [tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"]
            if any(len(ids) != 1 for ids in encoded_labels) or len({ids[0] for ids in encoded_labels}) != 4:
                raise RuntimeError("Answer labels are not four distinct single tokens; stop and review answer scoring.")
            label_ids = [ids[0] for ids in encoded_labels]
            report["answer_token_ids"] = dict(zip("ABCD", label_ids))
            report["prompt"] = prompt
            report["expected_answer"] = "B"
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    settings["id"], revision=settings["revision"], torch_dtype=torch.float32,
                    device_map={"": "cuda:0"}, use_safetensors=True,
                )
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    report.update(error_kind="gpu_out_of_memory", phase="model_loading",
                                  error="The GPU ran out of memory. Close other model workloads in this runtime and retry.")
                else:
                    report.update(failure_details(exc, "model_loading"))
                raise RuntimeError(report["error"]) from None
            inputs = {name: value.to("cuda:0") for name, value in inputs.items()}
            torch.cuda.reset_peak_memory_stats()
        model.eval()
        model.requires_grad_(False)
        if not tiny:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Forward pass produced non-finite logits.")
            answer_scores = logits[label_ids]
            probabilities = torch.softmax(answer_scores, dim=-1)
        if not tiny:
            torch.cuda.synchronize()
        report.update({
            "status": "passed", "forward_seconds": time.perf_counter() - start,
            "input_tokens": inputs["input_ids"].shape[-1],
            "layer_count": len(model.model.layers), "parameter_count": sum(p.numel() for p in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "answer_probabilities": dict(zip("ABCD", probabilities.cpu().tolist())),
            "prediction": "ABCD"[int(answer_scores.argmax().item())],
            "note": "One pipeline check. This is not a WMDP/retain baseline or evidence of model quality.",
        })
        if not tiny:
            report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc) if isinstance(exc, RuntimeError) else "Runtime check failed. Review the installation and rerun."
        write_json(output, report)
        raise RuntimeError(report["error"]) from None
    write_json(output, report)
    return report
