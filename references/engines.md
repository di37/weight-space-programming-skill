# Engines, installation, sizing and pitfalls (any OS, any GPU)

## Contents
1. Which engine on which machine
2. Installation
3. Sizing batches and memory
4. Engine notes and pitfalls
5. Scripts and options
6. Smoke test and what has been verified

## 1. Which engine on which machine

Generation (by far the slowest part) needs a fast engine. Probing, editing, diffing and grading are plain PyTorch and
Python and run everywhere.

| Machine | Generation engine | Probe / edit / diff |
|---|---|---|
| Linux with an NVIDIA GPU (CUDA) | **vLLM** (fastest; continuous batching) | PyTorch CUDA |
| Linux with an AMD GPU (ROCm) | **vLLM** ROCm build, else transformers on PyTorch ROCm | PyTorch ROCm (reports itself as `cuda`) |
| Windows with an NVIDIA GPU | **vLLM inside WSL2** (vLLM is Linux-only), else transformers on native PyTorch CUDA | PyTorch CUDA (native or WSL2) |
| macOS on Apple Silicon | **mlx-lm**; **mlx-vlm** for runs with images | PyTorch MPS |
| Anything else / CPU only | transformers (very slow; tests only) | PyTorch CPU |

`scripts/run_eval.py --engine auto` picks the first usable one in that order (`scripts/device.py` lists what it found).
Use **the same engine for base and copy** within one comparison.

## 2. Installation

Python 3.10 or newer. A virtual environment per project keeps the system Python clean.

```bash
python -m pip install torch transformers safetensors numpy pandas pyarrow pytest huggingface_hub
```

Then the engine for your machine:

- NVIDIA / AMD on Linux or WSL2: `python -m pip install vllm` (for AMD, use the vLLM ROCm build or its Docker image).
- Apple Silicon: `python -m pip install mlx-lm`, plus `mlx-vlm` for image tasks.
- Image tasks on any machine: `pillow` (and `torchvision` if your transformers version asks for it for the fast image
  processor).
- transformers on NVIDIA with a hybrid linear-attention model (Qwen3.5, Qwen3-Next): also install
  `flash-linear-attention` and `causal-conv1d`, or generation falls back to a much slower PyTorch path.

Install PyTorch for your accelerator as its install page says (CUDA wheel, ROCm wheel, or the default on macOS).
The model needs a transformers version that knows its architecture (Qwen3.5 needs transformers 5.x).

Maths grading uses math-verify. Keep it out of the main environment, for example with uv:

```bash
uv run --no-project --with math-verify==0.9.0 --with latex2sympy2_extended==1.11.0 --with antlr4-python3-runtime==4.13.2 python scripts/grade.py --task math <files>
```

Data: `hf download TIGER-Lab/MMLU-Pro data/test-00000-of-00001.parquet --repo-type dataset --revision
b189ec765aa7ed75c8acfea42df31fdae71f97be --local-dir data/MMLU-Pro` and `hf download HuggingFaceH4/MATH-500
--repo-type dataset --local-dir data/MATH-500`.

## 3. Sizing batches and memory

Run `python scripts/device.py --model <model dir> --max-tokens <cap> --jobs <generation jobs at once>`. It reads the
free GPU memory (CUDA/ROCm), Metal's recommended working set (Apple) or RAM (CPU), and the model's KV-cache size per
token from `config.json`. `run_eval.py --batch auto` uses the same sizing.

- **KV cache per token per sequence** = 2 (K and V) × KV heads × head dim × 2 bytes × layers that keep a cache.
  Linear-attention layers keep a small fixed state instead. For Qwen3.5-0.8B that is 12 KiB, about 0.4 GB per
  sequence at 32,768 tokens.
- **Peaks are about 4× the KV size**, because caches are copied as sequences join and leave a batch (seen with
  mlx-lm). The sizing divides by that.
- **An out-of-memory crash kills the process and every unfinished answer**, and the unfinished ones are exactly the
  long, looping answers. Finished answers are already on disk, and the run resumes from them.
- **At most two generation jobs at once.** They share one GPU, so two jobs do not double throughput, but they let the
  base held-out run overlap the probing. Give each `--jobs 2`. Never start the second while the first is on its
  longest answers.
- **vLLM** manages memory itself (`gpu_memory_utilization=0.85` in `backends.py`); `--batch` caps its concurrent
  sequences. It writes results every 4 × batch prompts, so a crash loses at most one chunk.
- **Time is dominated by the base model's loops**, which run every looping answer to the cap. A looping answer
  advances at roughly 10–30 tokens per second per sequence on consumer GPUs with mlx-lm or transformers. Datacenter
  GPUs with vLLM are an order of magnitude faster: an independent re-run generated 2,000 MATH-500 answers at 32k in
  51 minutes on one H200.

## 4. Engine notes and pitfalls

- **Presence penalty semantics.** The authors used vLLM, which penalises only generated tokens. mlx-lm's own
  `make_presence_penalty` also counts the prompt (verified: a processor saw the 13 prompt tokens at its first step)
  and defaults to a 20-token window. `backends.py` uses an output-only penalty for mlx-lm and transformers, so all
  three engines match vLLM. (Results produced before 3 Nov 2026 with mlx-lm's own processor penalised the prompt too;
  paired comparisons within one run are still fair.)
- **transformers with hybrid linear attention:** left padding turns the padded rows into NaN (verified on Qwen3.5 in
  bf16 and fp32). `backends.py` therefore batches only prompts of identical length for such models, which is slow.
  Prefer vLLM or mlx-lm for generation; transformers is fine for probing, which reads one text at a time.
- **transformers generation is static-batch:** a batch runs until its longest answer finishes, so one looping answer
  holds up the rest. Keep batches small with it.
- **Never save weights from an engine.** mlx-lm's loader rewrites norm weights (+1) and drops MTP tensors, and other
  engines fuse or convert tensors. `weights.py` reads and writes the raw safetensors files, so untouched tensors stay
  byte-identical.
- **Random numbers** depend on the engine, the batch size and the batch composition. Base and copy samples are
  therefore not seed-paired: pair by question within one comparison, and never compare scores measured on different
  questions.
- **Long teacher-forced reads:** `probe.py` asks the decoder for hidden states only and computes logits for the
  windows it scores, in chunks of 256 positions. Full logits for a 16k-token text would take about 8 GB. Models with
  final-logit soft-capping (Gemma 2 and 3) are handled.
- **Stopping jobs:** use the job runner's own stop (Claude Code: TaskStop). `pkill -f <text>` also kills any watcher
  whose command line contains that text.
- **pytest:** if an auto-loaded third-party plugin breaks collection, run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- **Images.** Every engine loads an image the same way (`vision.py`: RGB, longest side at most 448 pixels, about 170
  image tokens for Qwen3.5), so an image gives the same tokens everywhere.
  - mlx-vlm generates image answers one at a time (about 0.3–1.5 s per caption on Apple Silicon). Its logits
    processors also receive prompt tokens, so `backends.py` treats whatever the first call holds as the prompt.
  - transformers: the processor pads on the right by default, which garbles batched generation. `vision.py` sets left
    padding, and `backends.py` batches only prompts with the same expanded length, so it never pads image prompts.
  - Qwen3.5's full-model forward with an image also needs `mm_token_type_ids` (for its multimodal positions). Probing
    rebuilds every per-token input the processor returns, extended over the continuation.
  - vLLM gets the rendered prompt text and the image as `multi_modal_data`, and expands the image itself.

## 5. Scripts and options

All scripts live in `scripts/` and import each other. Run them as `python <skill dir>/scripts/<name>.py ...`.

| Script | What it does | Main options |
|---|---|---|
| `device.py` | Device, usable engines, memory, KV per token, recommended batch | `--model`, `--max-tokens`, `--jobs` |
| `copy_model.py` | Make the working copy, or restore its weights to base; checks sha256 | `--base`, `--copy`, `--restore` |
| `make_splits.py` | Seeded, stratified, disjoint practice and held-out splits, plus `test_ids.txt` | `--task mmlu\|math`, `--source`, `--out`, `--groups`, `--practice`, `--heldout`, `--test`, `--seed` |
| `make_image_data.py` | Image presets: reads only the rows needed from the Hub, saves resized images, splits by image | `--preset caption\|relevance\|pope`, `--out`, `--practice`, `--heldout`, `--test`, `--seed`, `--max-side`, `--source` |
| `make_pairs.py` | Matched pairs from a graded run: failing answer against the gold answer, same prompt, image and thinking | `--task`, `--data`, `--runs`, `--graded`, `--out`, `--label`, `--max` |
| `run_eval.py` | Resumable, supervised generation from a model folder: a crashed run restarts with half the batch (up to 4 times) and resumes; rows with an `image` run as image jobs | `--task mmlu\|math\|caption\|relevance\|pope\|jsonl`, `--model`, `--label`, `--data`, `--out`, `--ids-file`, `--profile thinking\|non-thinking\|greedy`, `--temperature/--top-p/--top-k/--presence-penalty`, `--max-tokens`, `--batch auto\|n`, `--jobs`, `--engine` |
| `grade.py` | Strict and lenient scores, cut, thinking closed, verbatim loops, median tokens and answer words, flag rate, per group, yes-rate and F1 for yes/no; paired fixed and broken counts with the sign test (for correctness, per-answer scores and the flag); scores every file on the common questions | `--task`, `--grader mc\|math\|cider\|yesno\|exact\|contains\|regex\|none\|FILE.py:FUNC`, `--pattern`, `--flag`, files (first = reference), `--out` |
| `loops.py` | Loop onset and matched windows | |
| `probe.py` | `collect` (loops, automatic), `collect-pairs` (matched pairs), `collect-decisions` (wrong against right short answers), `build` (direction, columns, edit, random control), `screen` (log-prob changes); images included | see its docstring |
| `apply_edit.py` | Write an edit into the copy; refuses if the copy is already edited or names a tensor out of scope | `--base`, `--copy`, `--edit` |
| `diff_edit.py` | Which tensors differ, columns, rank-1 share, α, cosines between layers | `--base`, `--edited`, `--out` |
| `weights.py`, `backends.py`, `modeling.py`, `tasks.py`, `vision.py`, `cider.py` | Library modules: raw checkpoint I/O (single or sharded), engines, PyTorch helpers, benchmarks, image inputs, CIDEr-D | |
| `tests/test_core.py`, `tests/test_vision.py` | Unit tests, no model, GPU or network needed | `python -m pytest scripts/tests -q` |

## 6. Smoke test and what has been verified

On a new machine, check the setup in under five minutes before any long run:

```bash
python scripts/device.py --model <base dir>
python -m pytest scripts/tests -q
head -2 <splits>/test_ids.txt > two.txt
python scripts/run_eval.py --task mmlu --model <base dir> --label smoke --data <splits>/heldout.jsonl --ids-file two.txt --out smoke.jsonl --max-tokens 64 --batch 2
```

Verified end to end: the mlx-lm and transformers engines, thinking on and off, a custom JSONL task with the
`contains` grader and a flag, the crash supervisor (a generation process killed mid-run restarted with half the batch
and finished with no answer lost or duplicated), PyTorch probing (its log-probs match mlx-lm's to about 0.001), both
contrast routes (loops and matched pairs), edit building (the portable build chose 255 of the original 256 columns),
applying, restoring and diffing an edit, the splits (byte-identical to the original run's) and grading. Image tasks
(4 Nov 2026): the three presets built from the Hub, generation with mlx-vlm and with transformers (batched by equal
length), CIDEr-D and yes/no grading, pairs and decisions probing with images, screening, and an edit written into a
copy that mlx-vlm then loaded and ran, with the vision tower untouched. **Not yet run: the vLLM engine (text or
images) and the CUDA and ROCm devices.** They follow those libraries' documented APIs; run the smoke test first and fix `backends.py` if the
installed vLLM version differs.
