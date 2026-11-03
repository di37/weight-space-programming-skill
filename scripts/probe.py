"""Read a model's internals on contrasting texts, build direction edits, and screen them.
PyTorch only, so it runs on CUDA, ROCm, Apple MPS or CPU. Forward passes only: no gradients, no optimiser.

Two ways to get the contrast (the behaviour to remove = "unwanted", its counterpart = "wanted"):

    # automatic: answers that loop to the cap; wanted = just before the loop, unwanted = inside it
    python scripts/probe.py collect --task mmlu --model <dir> --data <split.jsonl> --runs <answers.jsonl> --out <probe dir>
    # any behaviour: your matched pairs {"id", "prompt", "prefix"?, "image"?, "wanted", "unwanted"} as JSON lines
    python scripts/probe.py collect-pairs --model <dir> --pairs pairs.jsonl [--runs <answers.jsonl>] --out <probe dir>
    # short decisions (yes/no, a letter, a caption): the model's own wrong answers against its right ones
    python scripts/probe.py collect-decisions --task pope --model <dir> --data <split.jsonl> --runs <answers.jsonl>
        --graded <summary.json from grade.py> --out <probe dir>

    python scripts/probe.py build --probes <probe dir> --model <dir> --layers 15 19 --k 128 --alpha 6 --out edits/E.safetensors
    python scripts/probe.py build ... --control random --out edits/E_random.safetensors
    python scripts/probe.py screen --probes <probe dir> --model <dir> --edits edits/*.safetensors

collect: answers that hit the cap and settle into a verbatim loop (loops.loop_onset) are teacher-forced (prompt +
answer); the wanted window is the 1,024 tokens just before the loop, the unwanted window 1,024 tokens inside it.
collect-pairs: each pair is two texts, the rendered prompt + prefix + one continuation; each window covers that
continuation's tokens. Pairs should differ only in the behaviour (same prompt, same prefix, similar length).
collect-decisions: each graded answer is one text, prompt + the model's own answer, labelled unwanted if wrong (or, for
a scored grader, in the lower half by score) and wanted if right (the upper half). Its states are read at the
positions that produce the answer's tokens (one before each), since with a shared prompt the state that decides a
one-token answer is the same for both answers of a pair. All three record, per layer, the mean residual-stream output
and the mean MLP intermediate activation (the input of down_proj) over each window, and the mean log-prob of the
window's tokens. Items split into build (2/3) and screen (1/3) halves by id; with --runs, up to 16 finished answers add
a clean window for side effects: [512, 1536) of a long answer, or the whole of a short one. Rows, pairs and runs with
an `image` are read with the image (vision.py).
build: u_L = normalised (mean unwanted - mean wanted) residual after layer L over the build half (for pairs and loops, the
mean of each item's difference); neuron score = (mean unwanted - mean wanted activation) x (u_L . down_proj column); W' = W - alpha u (u^T W) on the top-k columns,
computed from --model's weights in float32 and saved in its dtype under its tensor names. --control random swaps u for
a random unit vector orthogonal to it, scaled to the same Frobenius change. For a second round on an edited copy, give
the copy as --model to collect and build, and the copy as --base to apply_edit.py.
screen: change in mean log-prob with each edit applied: unwanted should fall; wanted and clean should stay. Edits are
applied in memory here only because these are teacher-forced reads; generation always loads the copy from disk.
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from grade import answer_part
from loops import loop_onset, matched_windows
from tasks import TASKS, load_rows, spec, with_image_path
from vision import expanded_inputs, image_forward_kwargs, load_processor, render_text
from weights import DOWN_PROJ, down_proj_name, read_tensor

CLEAN_WINDOW = (512, 1536)
MAX_CLEAN = 16
LOGPROB_CHUNK = 256
WINDOWS = ("unwanted", "wanted", "clean")
KINDS = ("unwanted", "wanted")


class Prompter:
    """Token ids of rendered prompts; with an image, the processor expands the placeholder into the image's tokens."""

    def __init__(self, tokenizer, model_dir=None, processor=None):
        self.tokenizer, self.model_dir, self._processor = tokenizer, model_dir, processor

    @property
    def processor(self):
        if self._processor is None:
            self._processor = load_processor(self.model_dir)
        return self._processor

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def head(self, user_text, thinking, image=None):
        """(rendered prompt text, its token ids, image tokens expanded)."""
        text = render_text(self.tokenizer, user_text, thinking, image)
        if not image:
            return text, self.encode(text)
        return text, expanded_inputs(self.processor, text, image)["input_ids"][0].tolist()

    def text(self, user_text, thinking, image, continuation_ids, windows, state_windows=None):
        """A probe text: prompt + continuation, its windows, and what an image forward needs to rebuild it."""
        prompt_text, head = self.head(user_text, thinking, image)
        text = {"ids": head + continuation_ids, "windows": {k: (len(head) + s, len(head) + e) for k, (s, e) in
                                                            windows.items()}}
        if state_windows:
            text["state_windows"] = {k: (len(head) + s, len(head) + e) for k, (s, e) in state_windows.items()}
        if image:
            text.update(image=image, prompt_text=prompt_text, head=len(head))
        return text


class WindowReader:
    """Hooks that average each layer's output and each down_proj input over the given windows during a forward."""

    def __init__(self, layers, windows):
        self.windows, self.resid, self.act, self.handles = windows, {}, {}, []
        for i, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(self._output_hook(i)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._input_hook(i)))

    def _means(self, x):
        return {n: x[0, s:e].float().mean(0).cpu() for n, (s, e) in self.windows.items()}

    def _output_hook(self, i):
        def hook(_module, _args, out):
            self.resid[i] = self._means(out[0] if isinstance(out, tuple) else out)
        return hook

    def _input_hook(self, i):
        def hook(_module, args):
            self.act[i] = self._means(args[0])
        return hook

    def stacked(self, name, which):
        source = self.resid if which == "resid" else self.act
        return torch.stack([source[i][name] for i in sorted(source)]).numpy()

    def remove(self):
        for h in self.handles:
            h.remove()


def window_logprobs(model, hidden, ids, windows):
    head = model.get_output_embeddings()
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    target = torch.tensor(ids, device=hidden.device)
    out = {}
    for name, (s, e) in windows.items():
        total = 0.0
        for a in range(s, e, LOGPROB_CHUNK):
            b = min(a + LOGPROB_CHUNK, e)
            logits = head(hidden[0, a - 1:b - 1]).float()
            if cap:
                logits = torch.tanh(logits / cap) * cap
            total += torch.log_softmax(logits, -1).gather(1, target[a:b, None]).sum().item()
        out[name] = total / (e - s)
    return out


def model_call(model, text, prompter):
    """The module to run and its inputs: the text decoder for plain text, the full model (vision tower too) with an
    image."""
    if not text.get("image"):
        return model.get_decoder(), {"input_ids": torch.tensor([text["ids"]], device=model.device)}
    kwargs = image_forward_kwargs(prompter.processor, text["prompt_text"], text["image"], text["ids"][text["head"]:],
                                  model.device)
    if kwargs["input_ids"][0].tolist() != text["ids"]:
        raise SystemExit(f"image tokens of {text['image']} differ from when it was collected; use the same processor")
    return model.model, kwargs


def forward(model, layers, text, prompter=None, keep_states=True):
    windows = text["windows"]
    state_windows = text.get("state_windows", windows)
    reader = WindowReader(layers, state_windows) if keep_states else None
    try:
        module, kwargs = model_call(model, text, prompter)
        with torch.no_grad():
            hidden = module(**kwargs).last_hidden_state
            logprob = window_logprobs(model, hidden, text["ids"], windows)
        states = {(n, w): reader.stacked(n, w) for n in state_windows for w in ("resid", "act")} if reader else {}
    finally:
        if reader:
            reader.remove()
    return logprob, states


def half(item_id):
    return "build" if zlib.crc32(str(item_id).encode()) % 3 else "screen"


def text_item(item_id, kind, texts, split_key=None):
    """split_key: what decides the build/screen half (default the id), so samples of one question stay together."""
    return {"id": str(item_id), "half": kind or half(split_key or item_id), "texts": texts}


def task_rows(task, data):
    return {str(r[TASKS[task]["id"]]): r for r in load_rows(data, task=task)}


def read_runs(runs):
    return [json.loads(line) for line in Path(runs).read_text().splitlines() if line]


def row_prompt(task, row):
    return TASKS[task]["prompt"](row), row.get(spec(task, "image")) if spec(task, "image") else None


def clean_items(prompter, task, rows, recs):
    items = []
    for rec in recs:
        if rec["finish_reason"] != "stop" or len(items) >= MAX_CLEAN:
            continue
        answer_ids = prompter.encode(rec["text"])
        window = CLEAN_WINDOW if len(answer_ids) > CLEAN_WINDOW[1] else (0, len(answer_ids))
        if window[1] > window[0]:
            prompt, image = row_prompt(task, rows[rec["id"]])
            text = prompter.text(prompt, rec.get("thinking", True), image, answer_ids[:window[1]], {"clean": window})
            items.append(text_item(rec["id"], "clean", [text]))
    return items


def loop_items(prompter, task, data, runs):
    rows, recs = task_rows(task, data), read_runs(runs)
    items = []
    for rec in recs:
        if rec["finish_reason"] != "length":
            continue
        answer_ids = prompter.encode(rec["text"])
        onset = loop_onset(answer_ids)
        wins = matched_windows(0, onset, len(answer_ids)) if onset is not None else None
        if wins:
            prompt, image = row_prompt(task, rows[rec["id"]])
            text = prompter.text(prompt, rec.get("thinking", True), image, answer_ids[:wins[1][1]],
                                 {"wanted": wins[0], "unwanted": wins[1]})
            items.append(text_item(rec["id"], None, [text]))
    return items + clean_items(prompter, task, rows, recs)


def pair_items(prompter, pairs_path, thinking):
    items, base_dir = [], Path(pairs_path).parent
    for line in Path(pairs_path).read_text().splitlines():
        if not line.strip():
            continue
        row = with_image_path(json.loads(line), "image", base_dir)
        prefix = prompter.encode(row.get("prefix", ""))
        texts = []
        for kind in KINDS[::-1]:
            branch = prompter.encode(row[kind])
            if not branch:
                raise SystemExit(f"pair {row['id']}: empty {kind} continuation")
            texts.append(prompter.text(row["prompt"], thinking, row.get("image"), prefix + branch,
                                       {kind: (len(prefix), len(prefix) + len(branch))}))
        items.append(text_item(row["id"], None, texts))
    return items


def decision_labels(graded):
    """{(id, sample): "unwanted" | "wanted"} over answered items: wrong or right; for a scored grader, the lower and
    upper halves by rank (the middle item of an odd count is left out), so tied scores cannot empty a class."""
    marks = [g for g in graded if g.get("answered")]
    scored = [g for g in marks if g.get("score") is not None]
    if scored:
        ranked = sorted(scored, key=lambda g: (g["score"], str(g["id"]), g["sample"]))
        half_n = len(ranked) // 2
        return {**{(g["id"], g["sample"]): "unwanted" for g in ranked[:half_n]},
                **{(g["id"], g["sample"]): "wanted" for g in ranked[len(ranked) - half_n:]}}
    return {(g["id"], g["sample"]): "wanted" if g["correct"] else "unwanted" for g in marks
            if g.get("correct") is not None}


def split_answer(rec):
    """(text before the answer, the answer) with the answer's leading whitespace kept in the prefix, as generated."""
    answer = answer_part(rec)
    if not answer or not answer.strip():
        return None, None
    answer = answer.lstrip()
    return rec["text"][:len(rec["text"]) - len(answer)], answer


def decision_items(prompter, task, data, recs, graded):
    """One text per graded answer: wrong (or below-median) answers unwanted, right (above-median) ones wanted."""
    rows, labels = task_rows(task, data), decision_labels(graded)
    items = []
    for rec in recs:
        kind = labels.get((rec["id"], rec["sample"]))
        before, answer = split_answer(rec)
        if kind is None or answer is None or rec["id"] not in rows:
            continue
        prefix, answer_ids = prompter.encode(before), prompter.encode(answer)
        s, e = len(prefix), len(prefix) + len(answer_ids)
        prompt, image = row_prompt(task, rows[rec["id"]])
        text = prompter.text(prompt, rec.get("thinking", True), image, prefix + answer_ids, {kind: (s, e)},
                             state_windows={kind: (s - 1, e - 1)})
        items.append(text_item(f"{rec['id']}#{rec['sample']}" if rec["sample"] else rec["id"], None, [text],
                               split_key=rec["id"]))
    return items


def load_model(model_dir):
    from device import torch_device  # noqa: PLC0415
    from modeling import decoder_layers, load_hf  # noqa: PLC0415

    model, tokenizer = load_hf(model_dir, torch_device())
    return model, Prompter(tokenizer, model_dir), decoder_layers(model)[1]


def read_items(model, layers, items, out_dir, prompter=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, it in enumerate(items):
        it["logprob"] = {}
        for text in it["texts"]:
            logprob, states = forward(model, layers, text, prompter)
            it["logprob"].update(logprob)
            arrays.update({f"{i}_{n}_{w}": v for (n, w), v in states.items()})
        print(f"[{i + 1}/{len(items)}] {it['id']} {it['half']} {it['logprob']}", flush=True)
    np.savez(out / "states.npz", **arrays)
    (out / "items.json").write_text(json.dumps(items))
    counts = {h: sum(it["half"] == h for it in items) for h in ("build", "screen", "clean")}
    kinds = {k: sum(k in t["windows"] for it in items if it["half"] == "build" for t in it["texts"]) for k in KINDS}
    print(f"{counts}, build windows {kinds} -> {out}")
    if counts["build"] < 4 or counts["screen"] < 2 or min(kinds.values()) < 4:
        print("WARNING: few build or screen items, or under 4 of one kind; collect more loops, pairs or answers "
              "(12+ of each recommended)")


def cmd_collect(args):
    model, prompter, layers = load_model(args.model)
    items = loop_items(prompter, args.task, args.data, args.runs)[:args.max_items]
    read_items(model, layers, items, args.out, prompter)


def cmd_collect_pairs(args):
    model, prompter, layers = load_model(args.model)
    items = pair_items(prompter, args.pairs, args.thinking == "on")
    if args.runs:
        if not args.data:
            raise SystemExit("--runs needs --data (the split it was generated from) and --task")
        items += clean_items(prompter, args.task, task_rows(args.task, args.data), read_runs(args.runs))
    read_items(model, layers, items, args.out, prompter)


def cmd_collect_decisions(args):
    graded = json.loads(Path(args.graded).read_text())["graded"]
    label = args.label or Path(args.runs).stem
    if label not in graded:
        raise SystemExit(f"{args.graded} has no graded run '{label}' (it has {sorted(graded)}); pass --label")
    model, prompter, layers = load_model(args.model)
    items = decision_items(prompter, args.task, args.data, read_runs(args.runs), graded[label])[:args.max_items]
    read_items(model, layers, items, args.out, prompter)


def class_mean(states, build, kind, which, layer):
    values = [states[key][layer] for i in build if (key := f"{i}_{kind}_{which}") in states]
    if not values:
        raise SystemExit(f"no {kind} windows in the build half of the probes")
    return np.mean(values, axis=0)


def direction_and_scores(items, states, layer, w_down):
    """u: the normalised difference between the unwanted and wanted class means of the residual (with paired items,
    the mean of each pair's difference); scores: the activation difference times each column's write along u."""
    build = [i for i, it in enumerate(items) if it["half"] == "build"]
    if not build:
        raise SystemExit("no build-half items in the probes")
    diff = torch.tensor(class_mean(states, build, "unwanted", "resid", layer)
                        - class_mean(states, build, "wanted", "resid", layer), dtype=w_down.dtype)
    u = diff / diff.norm()
    act_diff = torch.tensor(class_mean(states, build, "unwanted", "act", layer)
                            - class_mean(states, build, "wanted", "act", layer), dtype=w_down.dtype)
    return u, act_diff * (u @ w_down)


def edit_layer(w_down, u, cols, alpha, control="none", seed=0):
    w = w_down.float()
    sel = w[:, cols]
    delta = -alpha * torch.outer(u, u @ sel)
    if control == "random":
        r = torch.randn(u.shape, generator=torch.Generator().manual_seed(seed))
        r = r - (r @ u) * u
        r = r / r.norm()
        rand = -torch.outer(r, r @ sel)
        delta = rand * (delta.norm() / rand.norm())
    edited = w.clone()
    edited[:, cols] = sel + delta
    return edited.to(w_down.dtype)


def cmd_build(args):
    from safetensors.torch import save_file  # noqa: PLC0415

    items = json.loads((Path(args.probes) / "items.json").read_text())
    states = dict(np.load(Path(args.probes) / "states.npz"))
    edits, report = {}, {}
    for layer in args.layers:
        name = down_proj_name(args.model, layer)
        w = read_tensor(args.model, name)
        u, scores = direction_and_scores(items, states, layer, w.float())
        cols = torch.argsort(scores, descending=True)[:args.k]
        edits[name] = edit_layer(w, u, cols, args.alpha, args.control, args.seed + layer)
        report[name] = {"columns": sorted(cols.tolist()), "top_score": scores[cols[0]].item()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(edits, args.out)
    meta = {"layers": args.layers, "k": args.k, "alpha": args.alpha, "control": args.control, "from": str(args.model),
            "tensors": report}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({**meta, "tensors": {k: round(v["top_score"], 4) for k, v in report.items()}}))


def screen_edit(model, layers, items, path, prompter=None):
    from safetensors.torch import load_file  # noqa: PLC0415

    params = {int(DOWN_PROJ.search(n).group(1)): v for n, v in load_file(path).items()}
    originals = {i: layers[i].mlp.down_proj.weight.data.clone() for i in params}
    try:
        for i, v in params.items():
            layers[i].mlp.down_proj.weight.data = v.to(originals[i].device, originals[i].dtype)
        deltas = {name: [] for name in WINDOWS}
        for it in items:
            for text in it["texts"]:
                logprob, _ = forward(model, layers, text, prompter, keep_states=False)
                for name, value in logprob.items():
                    deltas[name].append(value - it["logprob"][name])
    finally:
        for i, w in originals.items():
            layers[i].mlp.down_proj.weight.data = w
    return {k: round(sum(v) / len(v), 4) if v else None for k, v in deltas.items()}


def cmd_screen(args):
    items = json.loads((Path(args.probes) / "items.json").read_text())
    chosen = [it for it in items if it["half"] in ("screen", "clean")]
    model, prompter, layers = load_model(args.model)
    results = {}
    for path in args.edits:
        results[Path(path).stem] = screen_edit(model, layers, chosen, path, prompter)
        print(f"{Path(path).stem}: {results[Path(path).stem]}", flush=True)
    print(json.dumps(results, indent=1))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--task", choices=sorted(TASKS), required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--data", required=True)
    c.add_argument("--runs", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-items", type=int, help="read only the first n texts (a quick check of the setup)")
    cp = sub.add_parser("collect-pairs")
    cp.add_argument("--model", required=True)
    cp.add_argument("--pairs", required=True)
    cp.add_argument("--thinking", choices=("on", "off"), default="on", help="render prompts with thinking on or off")
    cp.add_argument("--task", choices=sorted(TASKS), default="jsonl", help="for --runs clean windows")
    cp.add_argument("--data", help="the split --runs was generated from (for clean windows)")
    cp.add_argument("--runs", help="finished answers to use as clean side-effect windows")
    cp.add_argument("--out", required=True)
    d = sub.add_parser("collect-decisions")
    d.add_argument("--task", choices=sorted(TASKS), required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--data", required=True, help="the split --runs was generated from")
    d.add_argument("--runs", required=True)
    d.add_argument("--graded", required=True, help="grade.py --out file that graded --runs")
    d.add_argument("--label", help="the run's name in --graded (default: the --runs file name)")
    d.add_argument("--max-items", type=int)
    d.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--probes", required=True)
    b.add_argument("--model", required=True, help="the model the edit is computed from (base, or the copy in round 2)")
    b.add_argument("--layers", type=int, nargs="+", required=True)
    b.add_argument("--k", type=int, default=128)
    b.add_argument("--alpha", type=float, default=6.0)
    b.add_argument("--control", choices=("none", "random"), default="none")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--probes", required=True)
    s.add_argument("--model", required=True, help="the same model as build's --model")
    s.add_argument("--edits", nargs="+", required=True)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    {"collect": cmd_collect, "collect-pairs": cmd_collect_pairs, "collect-decisions": cmd_collect_decisions,
     "build": cmd_build, "screen": cmd_screen}[a.cmd](a)
