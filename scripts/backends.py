"""Generation engines behind one interface, all with the authors' sampling protocol and vLLM's presence-penalty semantics.

Engines, fastest first: vLLM (Linux or WSL with an NVIDIA or AMD GPU), mlx-lm (Apple Silicon), Hugging Face
transformers (any PyTorch device; static batches, so slower when some answers loop to the cap). Every engine
penalises only tokens already present in the generated text (vLLM semantics; mlx-lm's own processor also counts the
prompt). Each engine calls on_done(job, token_ids, finish_reason) as answers finish, with finish_reason "stop" or
"length".

Image jobs (a job with an "image" path; vision.py): vLLM takes the rendered prompt and the image as multimodal data;
on Apple, mlx-vlm generates them one at a time (image answers are short); transformers runs the model's processor
and batches prompts of equal expanded length.

Sampling profiles: `thinking` is the authors' protocol (Qwen3.5's thinking-mode settings); `non-thinking` is the model
card's non-thinking settings; `greedy` is deterministic decoding with thinking off. Use the model card's settings for
other models (override with --temperature/--top-p/--top-k/--presence-penalty in run_eval.py).
"""
import time

from vision import load_image, load_processor, render_text

PROFILES = {
    "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "thinking": True},
    "non-thinking": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "thinking": False},
    "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0, "thinking": False},
}
EXTRA_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<|eot_id|>")
HEARTBEAT_S = 120


def stop_token_ids(tokenizer):
    vocab = tokenizer.get_vocab()
    ids = {tokenizer.eos_token_id} | {vocab[t] for t in EXTRA_STOP_TOKENS if t in vocab}
    return sorted(i for i in ids if i is not None)


def render(tokenizer, user_text, thinking=True, image=None):
    """The generation prompt as token ids (an image prompt holds one unexpanded placeholder)."""
    return tokenizer.encode(render_text(tokenizer, user_text, thinking, image), add_special_tokens=False)


class FirstCallIsPromptPenalty:
    """Presence penalty on generated tokens for mlx-vlm, whose logits processors also receive prompt tokens: all of
    them on the first call, or only the last one when the prompt was prefilled in chunks. Whatever the first call
    holds is the prompt."""

    def __init__(self, penalty):
        self.penalty, self.skip = penalty, None

    def __call__(self, tokens, logits):
        if self.skip is None:
            self.skip = tokens.size
            return logits
        generated = tokens[self.skip:]
        if generated.size:
            logits[:, generated] -= self.penalty
        return logits


class VLLMEngine:
    def __init__(self, model_dir, max_tokens, batch, seed, sampling):
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        self.llm = LLM(model=str(model_dir), dtype="bfloat16", seed=seed, max_num_seqs=batch,
                       max_model_len=max_tokens + 4096, gpu_memory_utilization=0.85)
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(n=1, max_tokens=max_tokens, seed=seed, temperature=sampling["temperature"],
                                     top_p=sampling["top_p"], top_k=sampling["top_k"] or -1,
                                     presence_penalty=sampling["presence_penalty"])
        self.chunk = batch * 4

    @staticmethod
    def request(job):
        if job.get("image"):
            return {"prompt": job["prompt_text"], "multi_modal_data": {"image": load_image(job["image"])}}
        return {"prompt_token_ids": job["prompt_ids"]}

    def generate(self, jobs, on_done):
        for start in range(0, len(jobs), self.chunk):
            part = jobs[start:start + self.chunk]
            outs = self.llm.generate([self.request(j) for j in part], self.params)
            for job, out in zip(part, outs):
                on_done(job, list(out.outputs[0].token_ids), out.outputs[0].finish_reason)


class MLXEngine:
    def __init__(self, model_dir, max_tokens, batch, seed, sampling):
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm.utils import load_model, load_tokenizer  # noqa: PLC0415

        mx.random.seed(seed)
        mx.set_cache_limit(1 << 30)
        self.model, _ = load_model(model_dir)
        self.tokenizer = load_tokenizer(model_dir)
        self.max_tokens, self.batch, self.sampling = max_tokens, batch, sampling

    @staticmethod
    def presence_penalty(penalty, prompt_len):
        def processor(tokens, logits):
            generated = tokens[prompt_len:]
            if generated.size:
                logits[:, generated] -= penalty
            return logits
        return processor

    def generate(self, jobs, on_done):
        from mlx_lm.generate import BatchGenerator  # noqa: PLC0415
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        stops = stop_token_ids(self.tokenizer)
        sp = self.sampling
        sampler = make_sampler(temp=sp["temperature"], top_p=sp["top_p"], top_k=sp["top_k"])
        gen = BatchGenerator(self.model, stop_tokens=[[i] for i in stops], sampler=sampler,
                             completion_batch_size=self.batch, prefill_batch_size=min(self.batch, 8))
        procs = [[self.presence_penalty(sp["presence_penalty"], len(j["prompt_ids"]))] if sp["presence_penalty"]
                 else [] for j in jobs]
        uids = gen.insert([j["prompt_ids"] for j in jobs], [self.max_tokens] * len(jobs), logits_processors=procs)
        by_uid, tokens = dict(zip(uids, jobs)), {u: [] for u in uids}
        last = time.time()
        while responses := gen.next_generated():
            for r in responses:
                if r.token not in stops:
                    tokens[r.uid].append(r.token)
                if r.finish_reason is not None:
                    on_done(by_uid[r.uid], tokens.pop(r.uid), r.finish_reason)
            if time.time() - last > HEARTBEAT_S:
                last = time.time()
                print(f"  {len(tokens)} unfinished, longest {max(map(len, tokens.values()), default=0)} tokens",
                      flush=True)
        gen.close()


class MLXVisionEngine:
    """mlx-vlm on Apple Silicon, for runs with images (text-only jobs work too), one answer at a time."""

    def __init__(self, model_dir, max_tokens, batch, seed, sampling):
        import mlx.core as mx  # noqa: PLC0415
        from mlx_vlm import load  # noqa: PLC0415

        mx.random.seed(seed)
        mx.set_cache_limit(1 << 30)
        self.model, self.processor = load(str(model_dir))
        self.tokenizer = self.processor.tokenizer
        self.max_tokens, self.sampling = max_tokens, sampling

    def generate(self, jobs, on_done):
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415
        from mlx_vlm.generate import generate_step  # noqa: PLC0415
        from mlx_vlm.utils import prepare_inputs  # noqa: PLC0415

        stops, sp = set(stop_token_ids(self.tokenizer)), self.sampling
        sampler = make_sampler(temp=sp["temperature"], top_p=sp["top_p"], top_k=sp["top_k"])
        image_token = getattr(self.model.config, "image_token_index", None)
        for job in jobs:
            images = [load_image(job["image"])] if job.get("image") else None
            inputs = prepare_inputs(self.processor, images=images, prompts=[job["prompt_text"]],
                                    image_token_index=image_token)
            extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
            procs = [FirstCallIsPromptPenalty(sp["presence_penalty"])] if sp["presence_penalty"] else []
            tokens, reason = [], "length"
            for token, _ in generate_step(inputs["input_ids"], self.model, inputs.get("pixel_values"),
                                          inputs.get("attention_mask"), max_tokens=self.max_tokens, sampler=sampler,
                                          logits_processors=procs, **extra):
                if token in stops:
                    reason = "stop"
                    break
                tokens.append(token)
            on_done(job, tokens, reason)


class HFEngine:
    def __init__(self, model_dir, max_tokens, batch, seed, sampling):
        import torch  # noqa: PLC0415

        from device import torch_device  # noqa: PLC0415
        from modeling import load_hf  # noqa: PLC0415

        from device import text_config  # noqa: PLC0415

        torch.manual_seed(seed)
        self.model, self.tokenizer = load_hf(model_dir, torch_device())
        self.tokenizer.padding_side = "left"
        self.model_dir, self.processor = model_dir, None
        self.max_tokens, self.batch, self.sampling = max_tokens, batch, sampling
        # Left padding turns padded rows into NaN in transformers' PyTorch linear-attention layers (checked on
        # Qwen3.5), so hybrid models only batch prompts of identical length.
        self.pad_safe = "linear_attention" not in (text_config(model_dir).get("layer_types") or [])

    def batches(self, jobs, lengths):
        """Text jobs batch freely on pad-safe models; image jobs, and every job of a hybrid model, batch only with
        jobs of the same (expanded) prompt length, so they need no padding."""
        if self.pad_safe and not any(j.get("image") for j in jobs):
            return [jobs[i:i + self.batch] for i in range(0, len(jobs), self.batch)]
        by_len = {}
        for j, n in zip(jobs, lengths):
            by_len.setdefault((bool(j.get("image")), n), []).append(j)
        return [b[i:i + self.batch] for b in by_len.values() for i in range(0, len(b), self.batch)]

    def encode(self, job):
        """Model inputs for one job, batch of one: token ids alone, or the processor's outputs for an image job."""
        import torch  # noqa: PLC0415

        if not job.get("image"):
            return {"input_ids": torch.tensor([job["prompt_ids"]])}
        if self.processor is None:
            self.processor = load_processor(self.model_dir)
        return dict(self.processor(text=[job["prompt_text"]], images=[load_image(job["image"])], return_tensors="pt"))

    def inputs(self, part, encoded, pad):
        """One batch's inputs: equal-length rows are concatenated (image tensors too); text rows are left-padded."""
        import torch  # noqa: PLC0415

        encs = [encoded[id(j)] for j in part]
        width = max(e["input_ids"].shape[1] for e in encs)
        if all(e["input_ids"].shape[1] == width for e in encs):
            batch = {k: torch.cat([e[k] for e in encs]) for k in encs[0]}
            batch.setdefault("attention_mask", torch.ones_like(batch["input_ids"]))
        else:
            rows = [e["input_ids"][0].tolist() for e in encs]
            batch = {"input_ids": torch.tensor([[pad] * (width - len(r)) + r for r in rows]),
                     "attention_mask": torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])}
        return {k: v.to(self.model.device) for k, v in batch.items()}, width

    def generate(self, jobs, on_done):
        import torch  # noqa: PLC0415
        from transformers import LogitsProcessorList  # noqa: PLC0415

        from modeling import OutputOnlyPresencePenalty  # noqa: PLC0415

        stops, pad = stop_token_ids(self.tokenizer), self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        encoded = {id(j): self.encode(j) for j in jobs}
        for part in self.batches(jobs, [encoded[id(j)]["input_ids"].shape[1] for j in jobs]):
            inputs, width = self.inputs(part, encoded, pad)
            sp = self.sampling
            procs = [OutputOnlyPresencePenalty(sp["presence_penalty"], width)] if sp["presence_penalty"] else []
            decoding = ({"do_sample": True, "temperature": sp["temperature"], "top_p": sp["top_p"],
                         "top_k": sp["top_k"] or None} if sp["temperature"] > 0 else {"do_sample": False})
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_tokens, eos_token_id=stops,
                                          pad_token_id=pad, logits_processor=LogitsProcessorList(procs), **decoding)
            for job, row in zip(part, out[:, width:].tolist()):
                end = next((k for k, t in enumerate(row) if t in stops), None)
                on_done(job, row if end is None else row[:end], "length" if end is None else "stop")


ENGINES = {"vllm": VLLMEngine, "mlx": MLXEngine, "hf": HFEngine}
VISION_ENGINES = {**ENGINES, "mlx": MLXVisionEngine}


def make_engine(name, model_dir, max_tokens, batch, seed=0, sampling=None, vision=False):
    from device import engines, torch_device  # noqa: PLC0415

    usable = engines(torch_device(), vision)
    chosen = usable[0] if name == "auto" and usable else name
    table = VISION_ENGINES if vision else ENGINES
    if chosen not in table:
        raise SystemExit(f"no generation engine available (found {usable}); install vllm, mlx-lm "
                         f"{'and mlx-vlm ' if vision else ''}or transformers")
    sampling = sampling or PROFILES["thinking"]
    print(f"engine: {chosen}{' (images)' if vision else ''} (usable here: {usable}), batch {batch}, "
          f"sampling {sampling}", flush=True)
    return table[chosen](model_dir, max_tokens, batch, seed, sampling)
