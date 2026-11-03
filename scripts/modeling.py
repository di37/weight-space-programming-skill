"""PyTorch helpers that work for any Hugging Face decoder with `mlp.down_proj` layers (Qwen, Llama, Gemma, Mistral, ...).

load_hf picks the right Auto class for text-only or multimodal checkpoints; decoder_layers finds the decoder's layer
list; OutputOnlyPresencePenalty gives transformers vLLM's presence-penalty semantics.
"""
import json
from pathlib import Path


def load_hf(model_dir, device):
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer  # noqa: PLC0415

    arch = json.loads((Path(model_dir) / "config.json").read_text()).get("architectures", [""])[0]
    cls = AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM
    model = cls.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
    return model, AutoTokenizer.from_pretrained(model_dir)


def decoder_layers(model):
    """(prefix, layers): the ModuleList whose blocks have an MLP with a down_proj, e.g. model.language_model.layers."""
    import torch  # noqa: PLC0415

    found = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, torch.nn.ModuleList) and len(mod)
             and hasattr(mod[0], "mlp") and hasattr(mod[0].mlp, "down_proj")]
    if len(found) != 1:
        raise ValueError(f"expected one decoder layer list with mlp.down_proj, found {[n for n, _ in found]}")
    return found[0]


class OutputOnlyPresencePenalty:
    """Subtract `penalty` from every token already in the generated part (positions >= prompt_width)."""

    def __init__(self, penalty, prompt_width):
        self.penalty, self.prompt_width = penalty, prompt_width

    def __call__(self, input_ids, scores):
        import torch  # noqa: PLC0415

        generated = input_ids[:, self.prompt_width:]
        if generated.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
        return scores - self.penalty * seen.to(scores.dtype)
