"""Generate answers for a benchmark split from a model directory on disk (base or edited copy), resumably.

    python scripts/run_eval.py --task mmlu|math|caption|relevance|pope|jsonl --model <model dir> --label base
        --data <split.jsonl> --out <answers.jsonl> [--ids-file ids.txt] [--profile thinking|non-thinking|greedy]
        [--max-tokens N] [--batch auto] [--engine auto|vllm|mlx|hf] [--jobs 1]

--task jsonl takes your own file of {"id", "prompt", optional "answer", "group", "image"} rows. --profile picks the
sampling settings (backends.PROFILES); the default is the task's (tasks.py: thinking, the authors' protocol, for text
tasks; non-thinking for the image presets), as is --max-tokens (32,768 for text, 512 for image presets).
--temperature, --top-p, --top-k and --presence-penalty override single values, e.g. to match another model's card.
Rows with an image run as image jobs (vision.py). Each finished answer is appended to --out at once (id, sample,
label, group, answer, image, profile, thinking, finish_reason, n_tokens, seconds, text), and answers already in --out
are skipped, so a stopped or crashed run resumes. Never evaluate an in-memory edit: point --model at the edited copy's
folder. Use the same profile and engine for base and copy.

The run is supervised: generation happens in a child process, and if it dies (out of GPU memory, or anything else)
it is restarted with half the batch, up to RETRIES times, resuming from the answers already written.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from backends import PROFILES, make_engine
from device import recommended_batch
from tasks import TASKS, job_meta, load_rows, spec
from vision import render_text

RETRIES = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ids-file", type=Path)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--max-tokens", type=int, help="default: the task's (tasks.py)")
    p.add_argument("--batch", default="auto", help="'auto' sizes it from free GPU memory (device.py)")
    p.add_argument("--jobs", type=int, default=1, help="generation jobs sharing the GPU, for --batch auto")
    p.add_argument("--engine", default="auto", choices=("auto", "vllm", "mlx", "hf"))
    p.add_argument("--profile", choices=sorted(PROFILES), help="default: the task's (tasks.py)")
    for flag in ("temperature", "top-p", "top-k", "presence-penalty"):
        p.add_argument(f"--{flag}", type=float, help="override the profile's value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.profile = args.profile or spec(args.task, "profile")
    args.max_tokens = args.max_tokens or spec(args.task, "max_tokens")
    return args


def sampling_from(args):
    sampling = dict(PROFILES[args.profile])
    for key in ("temperature", "top_p", "top_k", "presence_penalty"):
        if getattr(args, key) is not None:
            sampling[key] = int(getattr(args, key)) if key == "top_k" else getattr(args, key)
    return sampling


def done_keys(path):
    if not path.exists():
        return set()
    return {(r["id"], r["sample"]) for r in map(json.loads, path.read_text().splitlines()) if r}


def without_batch(argv):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "--batch":
            skip = True
        elif not a.startswith("--batch="):
            out.append(a)
    return out


def supervise(args):
    batch = recommended_batch(args.model, args.max_tokens, args.jobs) if args.batch == "auto" else int(args.batch)
    for attempt in range(1, RETRIES + 1):
        cmd = [sys.executable, __file__, *without_batch(sys.argv[1:]), "--batch", str(batch), "--child"]
        code = subprocess.run(cmd).returncode
        if code == 0:
            return
        if batch == 1 and attempt > 1:
            break
        batch = max(1, batch // 2)
        print(f"generation stopped with exit code {code}; resuming with batch {batch} "
              f"(attempt {attempt + 1} of {RETRIES})", flush=True)
    raise SystemExit("generation kept failing; see the log above (out of memory even at batch 1, or a bug)")


def main():
    args = parse_args()
    if not args.child:
        supervise(args)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_keys(args.out)
    rows = load_rows(args.data, args.ids_file, args.task)
    pending = [(row, s) for row in rows for s in range(args.samples)
               if (job_meta(args.task, row, s)["id"], s) not in skip]
    if not pending:
        print(f"nothing to do: all answers are already in {args.out}")
        return
    batch = int(args.batch)
    sampling = sampling_from(args)
    metas = [job_meta(args.task, row, s) for row, s in pending]
    vision = any(m["image"] for m in metas)
    engine = make_engine(args.engine, args.model, args.max_tokens, batch, args.seed, sampling, vision)
    prompt = TASKS[args.task]["prompt"]
    jobs = []
    for (row, _), meta in zip(pending, metas):
        text = render_text(engine.tokenizer, prompt(row), sampling["thinking"], meta["image"])
        jobs.append({**meta, "label": args.label, "profile": args.profile, "thinking": sampling["thinking"],
                     "prompt_text": text, "prompt_ids": engine.tokenizer.encode(text, add_special_tokens=False)})
    started, finished = time.time(), [0]
    with args.out.open("a") as out:
        def on_done(job, token_ids, finish_reason):
            rec = {k: v for k, v in job.items() if k not in ("prompt_ids", "prompt_text")}
            rec.update(finish_reason=finish_reason, n_tokens=len(token_ids), seconds=round(time.time() - started, 1),
                       text=engine.tokenizer.decode(token_ids))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            finished[0] += 1
            print(f"[{finished[0]}/{len(jobs)}] {rec['id']} s{rec['sample']} {finish_reason} {len(token_ids)} tok "
                  f"at {rec['seconds']}s", flush=True)

        print(f"{args.label}: {len(jobs)} answers to generate ({len(skip)} already done) -> {args.out}", flush=True)
        engine.generate(jobs, on_done)


if __name__ == "__main__":
    main()
