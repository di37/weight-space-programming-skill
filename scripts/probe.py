"""Read a model's internals on contrasting texts, build direction edits, and screen them.
PyTorch only, so it runs on CUDA, ROCm, Apple MPS or CPU. Forward passes only: no gradients, no optimiser.

Two ways to get the contrast (the behaviour to remove = "unwanted", its counterpart = "wanted"):

    # automatic: answers that loop to the cap; wanted = just before the loop, unwanted = inside it
    python scripts/probe.py collect --task mmlu --model <dir> --data <split.jsonl> --runs <answers.jsonl> --out <probe dir>
    # any behaviour: your matched pairs {"id", "prompt", "prefix" (optional), "wanted", "unwanted"} as JSON lines
    python scripts/probe.py collect-pairs --model <dir> --pairs pairs.jsonl [--runs <answers.jsonl>] --out <probe dir>

    python scripts/probe.py build --probes <probe dir> --model <dir> --layers 15 19 --k 128 --alpha 6 --out edits/E.safetensors
    python scripts/probe.py build ... --control random --out edits/E_random.safetensors
    python scripts/probe.py screen --probes <probe dir> --model <dir> --edits edits/*.safetensors

collect: answers that hit the cap and settle into a verbatim loop (loops.loop_onset) are teacher-forced (prompt +
answer); the wanted window is the 1,024 tokens just before the loop, the unwanted window 1,024 tokens inside it.
collect-pairs: each pair is two texts, the rendered prompt + prefix + one continuation; each window covers that
continuation's tokens. Pairs should differ only in the behaviour (same prompt, same prefix, similar length). Both
record, per layer, the mean residual-stream output and the mean MLP intermediate activation (the input of down_proj)
over each window, and the mean log-prob of the window's tokens. Items split into build (2/3) and screen (1/3) halves
by id; with --runs, up to 16 finished answers add a clean window [512, 1536) for side effects.
build: u_L = normalised mean(unwanted - wanted) residual after layer L over the build half; neuron score =
mean(unwanted - wanted activation) x (u_L . down_proj column); W' = W - alpha u (u^T W) on the top-k columns,
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

from backends import render
from loops import loop_onset, matched_windows
from tasks import TASKS, load_rows
from weights import DOWN_PROJ, down_proj_name, read_tensor

CLEAN_WINDOW = (512, 1536)
MAX_CLEAN = 16
LOGPROB_CHUNK = 256
WINDOWS = ("unwanted", "wanted", "clean")


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


def forward(model, layers, ids, windows, keep_states=True):
    reader = WindowReader(layers, windows) if keep_states else None
    try:
        with torch.no_grad():
            hidden = model.get_decoder()(input_ids=torch.tensor([ids], device=model.device)).last_hidden_state
            logprob = window_logprobs(model, hidden, ids, windows)
        states = {(n, w): reader.stacked(n, w) for n in windows for w in ("resid", "act")} if reader else {}
    finally:
        if reader:
            reader.remove()
    return logprob, states


def half(item_id):
    return "build" if zlib.crc32(str(item_id).encode()) % 3 else "screen"


def text_item(item_id, kind, texts):
    return {"id": str(item_id), "half": kind or half(item_id), "texts": texts}


def clean_items(tokenizer, task, rows, recs):
    items = []
    for rec in recs:
        if rec["finish_reason"] != "stop" or len(items) >= MAX_CLEAN:
            continue
        prompt_ids = render(tokenizer, TASKS[task]["prompt"](rows[rec["id"]]), rec.get("thinking", True))
        answer_ids = tokenizer.encode(rec["text"], add_special_tokens=False)
        if len(answer_ids) > CLEAN_WINDOW[1]:
            s, e = (len(prompt_ids) + c for c in CLEAN_WINDOW)
            items.append(text_item(rec["id"], "clean", [{"ids": (prompt_ids + answer_ids)[:e],
                                                         "windows": {"clean": (s, e)}}]))
    return items


def loop_items(tokenizer, task, data, runs):
    rows = {str(r[TASKS[task]["id"]]): r for r in load_rows(data)}
    recs = [json.loads(line) for line in Path(runs).read_text().splitlines() if line]
    items = []
    for rec in recs:
        if rec["finish_reason"] != "length":
            continue
        prompt_ids = render(tokenizer, TASKS[task]["prompt"](rows[rec["id"]]), rec.get("thinking", True))
        answer_ids = tokenizer.encode(rec["text"], add_special_tokens=False)
        onset = loop_onset(answer_ids)
        wins = matched_windows(len(prompt_ids), onset, len(answer_ids)) if onset is not None else None
        if wins:
            items.append(text_item(rec["id"], None, [{"ids": (prompt_ids + answer_ids)[:wins[1][1]],
                                                      "windows": {"wanted": wins[0], "unwanted": wins[1]}}]))
    return items + clean_items(tokenizer, task, rows, recs)


def pair_items(tokenizer, pairs_path, thinking):
    items = []
    for row in (json.loads(line) for line in Path(pairs_path).read_text().splitlines() if line.strip()):
        head = render(tokenizer, row["prompt"], thinking) + tokenizer.encode(row.get("prefix", ""),
                                                                             add_special_tokens=False)
        texts = []
        for kind in ("wanted", "unwanted"):
            branch = tokenizer.encode(row[kind], add_special_tokens=False)
            if not branch:
                raise SystemExit(f"pair {row['id']}: empty {kind} continuation")
            texts.append({"ids": head + branch, "windows": {kind: (len(head), len(head) + len(branch))}})
        items.append(text_item(row["id"], None, texts))
    return items


def load_model(model_dir):
    from device import torch_device  # noqa: PLC0415
    from modeling import decoder_layers, load_hf  # noqa: PLC0415

    model, tokenizer = load_hf(model_dir, torch_device())
    return model, tokenizer, decoder_layers(model)[1]


def read_items(model, layers, items, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, it in enumerate(items):
        it["logprob"] = {}
        for text in it["texts"]:
            logprob, states = forward(model, layers, text["ids"], text["windows"])
            it["logprob"].update(logprob)
            arrays.update({f"{i}_{n}_{w}": v for (n, w), v in states.items()})
        print(f"[{i + 1}/{len(items)}] {it['id']} {it['half']} {it['logprob']}", flush=True)
    np.savez(out / "states.npz", **arrays)
    (out / "items.json").write_text(json.dumps(items))
    counts = {h: sum(it["half"] == h for it in items) for h in ("build", "screen", "clean")}
    print(f"{counts} -> {out}")
    if counts["build"] < 4 or counts["screen"] < 2:
        print("WARNING: few build or screen items; collect more loops or write more pairs (12+ recommended)")


def cmd_collect(args):
    model, tokenizer, layers = load_model(args.model)
    read_items(model, layers, loop_items(tokenizer, args.task, args.data, args.runs)[:args.max_items], args.out)


def cmd_collect_pairs(args):
    model, tokenizer, layers = load_model(args.model)
    items = pair_items(tokenizer, args.pairs, args.thinking == "on")
    if args.runs:
        rows = {str(r[TASKS[args.task]["id"]]): r for r in load_rows(args.data)}
        recs = [json.loads(line) for line in Path(args.runs).read_text().splitlines() if line]
        items += clean_items(tokenizer, args.task, rows, recs)
    read_items(model, layers, items, args.out)


def direction_and_scores(items, states, layer, w_down):
    build = [i for i, it in enumerate(items) if it["half"] == "build"]
    if not build:
        raise SystemExit("no build-half items in the probes")
    diff = torch.tensor(np.mean([states[f"{i}_unwanted_resid"][layer] - states[f"{i}_wanted_resid"][layer]
                                 for i in build], axis=0))
    u = diff / diff.norm()
    act_diff = torch.tensor(np.mean([states[f"{i}_unwanted_act"][layer] - states[f"{i}_wanted_act"][layer]
                                     for i in build], axis=0))
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


def screen_edit(model, layers, items, path):
    from safetensors.torch import load_file  # noqa: PLC0415

    params = {int(DOWN_PROJ.search(n).group(1)): v for n, v in load_file(path).items()}
    originals = {i: layers[i].mlp.down_proj.weight.data.clone() for i in params}
    try:
        for i, v in params.items():
            layers[i].mlp.down_proj.weight.data = v.to(originals[i].device, originals[i].dtype)
        deltas = {name: [] for name in WINDOWS}
        for it in items:
            for text in it["texts"]:
                logprob, _ = forward(model, layers, text["ids"], text["windows"], keep_states=False)
                for name, value in logprob.items():
                    deltas[name].append(value - it["logprob"][name])
    finally:
        for i, w in originals.items():
            layers[i].mlp.down_proj.weight.data = w
    return {k: round(sum(v) / len(v), 4) if v else None for k, v in deltas.items()}


def cmd_screen(args):
    items = json.loads((Path(args.probes) / "items.json").read_text())
    chosen = [it for it in items if it["half"] in ("screen", "clean")]
    model, _, layers = load_model(args.model)
    results = {}
    for path in args.edits:
        results[Path(path).stem] = screen_edit(model, layers, chosen, path)
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
    {"collect": cmd_collect, "collect-pairs": cmd_collect_pairs, "build": cmd_build, "screen": cmd_screen}[a.cmd](a)
