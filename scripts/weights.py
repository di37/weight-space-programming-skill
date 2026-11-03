"""Read and write a Hugging Face checkpoint's safetensors by tensor name, for single-file and sharded models alike.

Works on the raw files, so every tensor not named stays byte-identical (norms, vision tower, MTP heads included).
Loading a model through an inference engine and saving it would not guarantee that.
"""
import hashlib
import json
import os
import re
from pathlib import Path

CHUNK = 1 << 24
DOWN_PROJ = re.compile(r"layers\.(\d+)\.mlp\.down_proj\.weight$")
OUT_OF_SCOPE = re.compile(r"(embed_tokens|lm_head|(^|\.)norm)\.weight$")


def shard_map(model_dir):
    """{tensor name: shard file name}."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    from safetensors import safe_open  # noqa: PLC0415

    names = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            names.update({k: path.name for k in f.keys()})
    return names


def read_tensor(model_dir, name):
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(Path(model_dir) / shard_map(model_dir)[name], framework="pt") as f:
        return f.get_tensor(name)


def down_proj_name(model_dir, layer):
    matches = [n for n in shard_map(model_dir) if (m := DOWN_PROJ.search(n)) and int(m.group(1)) == layer]
    language = [n for n in matches if "visual" not in n and "vision" not in n]
    if len(language) != 1:
        raise ValueError(f"layer {layer}: expected one decoder down_proj tensor, found {matches}")
    return language[0]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def write_tensors(model_dir, replacements):
    """Replace tensors in place (same names, shapes, dtypes), shard by shard, each via a temporary file and rename."""
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    bad = sorted(n for n in replacements if OUT_OF_SCOPE.search(n))
    if bad:
        raise ValueError(f"out-of-scope tensors (embeddings, output head, norms): {bad}")
    shards = shard_map(model_dir)
    for shard in sorted({shards[n] for n in replacements}):
        path = Path(model_dir) / shard
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {"format": "pt"}
        tensors = load_file(path)
        for name, value in replacements.items():
            if shards[name] != shard:
                continue
            if value.shape != tensors[name].shape or value.dtype != tensors[name].dtype:
                raise ValueError(f"{name}: edit {tuple(value.shape)} {value.dtype} vs model "
                                 f"{tuple(tensors[name].shape)} {tensors[name].dtype}")
            tensors[name] = value.contiguous()
        tmp = path.with_suffix(".tmp")
        save_file(tensors, tmp, metadata=metadata)
        os.replace(tmp, path)
