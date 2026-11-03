"""Images for vision-language models: loading, the user turn's content, and the model inputs each engine needs.

A row, job or pair with an "image" field (a file path) is an image job. Every engine loads images the same way (RGB,
longest side at most MAX_SIDE pixels), so an image gives the same tokens everywhere. The prompt is rendered with the
model's own chat template from the content [image, text], which holds one image placeholder; the model's processor
then expands that placeholder into the image's tokens.
"""
MAX_SIDE = 448


def load_image(path, max_side=MAX_SIDE):
    from PIL import Image  # noqa: PLC0415

    try:
        img = Image.open(path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read image {path}: {exc}") from exc
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
    return img


def user_content(text, image):
    return [{"type": "image"}, {"type": "text", "text": text}] if image else text


def render_text(tokenizer, user_text, thinking=True, image=None):
    """The generation prompt as text. Templates without a thinking switch ignore enable_thinking."""
    return tokenizer.apply_chat_template([{"role": "user", "content": user_content(user_text, image)}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def load_processor(model_dir):
    from transformers import AutoProcessor  # noqa: PLC0415

    processor = AutoProcessor.from_pretrained(model_dir)
    processor.tokenizer.padding_side = "left"
    return processor


def expanded_inputs(processor, prompt_text, image_path):
    """Processor outputs (batch of one) for a rendered prompt and its image, image tokens expanded."""
    return processor(text=[prompt_text], images=[load_image(image_path)], return_tensors="pt")


def image_forward_kwargs(processor, prompt_text, image_path, extra_ids, device):
    """Full-model forward arguments for prompt + image followed by extra_ids (a continuation). Per-token inputs the
    processor returns (input_ids, attention_mask, token type ids such as Qwen's mm_token_type_ids) are extended
    over the continuation: its ids, mask 1, type 0 (text)."""
    import torch  # noqa: PLC0415

    enc = expanded_inputs(processor, prompt_text, image_path)
    width = enc["input_ids"].shape[1]
    out = {}
    for key, value in enc.items():
        if hasattr(value, "dim") and value.dim() == 2 and tuple(value.shape) == (1, width):
            fill = extra_ids if key == "input_ids" else [int(key == "attention_mask")] * len(extra_ids)
            value = torch.cat([value, torch.tensor([fill], dtype=value.dtype)], 1)
        out[key] = value.to(device)
    return out
