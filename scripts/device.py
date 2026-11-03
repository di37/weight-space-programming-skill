"""Detect the GPU backend and size generation batches for whatever machine this runs on (nothing is hard-coded).

    python scripts/device.py --model <model dir> [--max-tokens 32768] [--jobs 1]

Prints the PyTorch device (`cuda` covers NVIDIA CUDA and AMD ROCm builds of PyTorch; `mps` is Apple Silicon), which
generation engines are installed, the GPU memory available, the model's KV-cache bytes per token, and a batch size.
The batch keeps a full batch of KV caches at max tokens under MEMORY_SHARE of the memory, divided between --jobs
concurrent generation jobs. PEAK_FACTOR covers the observed peaks: KV caches are copied as sequences join and leave,
so peak use ran about 4x the KV size.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

MEMORY_SHARE = 0.8
PEAK_FACTOR = 4
MAX_BATCH = 64
BYTES_PER_VALUE = 2  # bf16 KV cache
CPU_FALLBACK_BYTES = 16 << 30  # when the OS does not report physical memory (e.g. Windows without psutil)


def torch_device():
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def installed(module):
    return importlib.util.find_spec(module) is not None


def gpu_memory_bytes(device):
    """Memory a GPU job may use: free CUDA/ROCm memory, Metal's recommended working set, or total RAM on CPU."""
    import torch  # noqa: PLC0415

    if device == "cuda":
        free, _total = torch.cuda.mem_get_info()
        return free
    if device == "mps":
        return torch.mps.recommended_max_memory()
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return CPU_FALLBACK_BYTES


def text_config(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return cfg.get("text_config", cfg)


def kv_bytes_per_token(model_dir):
    """K and V for every layer that keeps a KV cache (linear-attention layers keep a fixed-size state instead)."""
    cfg = text_config(model_dir)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    types = cfg.get("layer_types")
    cached_layers = sum(t != "linear_attention" for t in types) if types else cfg["num_hidden_layers"]
    return 2 * kv_heads * head_dim * BYTES_PER_VALUE * cached_layers


def recommended_batch(model_dir, max_tokens, jobs=1, memory=None):
    memory = memory if memory is not None else gpu_memory_bytes(torch_device())
    per_sequence = kv_bytes_per_token(model_dir) * max_tokens * PEAK_FACTOR
    return max(1, min(MAX_BATCH, int(memory * MEMORY_SHARE / jobs // per_sequence)))


def engines(device):
    """Generation engines usable here, fastest first."""
    found = []
    if device == "cuda" and installed("vllm"):
        found.append("vllm")
    if device == "mps" and installed("mlx_lm"):
        found.append("mlx")
    if installed("transformers"):
        found.append("hf")
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--jobs", type=int, default=1, help="generation jobs that will share the GPU")
    args = p.parse_args()
    device = torch_device()
    memory = gpu_memory_bytes(device)
    print(json.dumps({"device": device, "engines": engines(device), "memory_gb": round(memory / 2**30, 1),
                      "kv_kib_per_token": kv_bytes_per_token(args.model) / 1024,
                      "batch": recommended_batch(args.model, args.max_tokens, args.jobs, memory)}, indent=1))


if __name__ == "__main__":
    main()
