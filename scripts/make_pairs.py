"""Make matched pairs from a graded run: the model's own failing answer as `unwanted`, the gold answer as `wanted`.

    python scripts/make_pairs.py --task caption --data <split.jsonl> --runs <answers.jsonl> --graded <summary.json>
        --out pairs.jsonl [--label base] [--max 64]

A failing answer is a wrong one or, for a scored grader (cider), one in the lower half of the run's scores. Each pair keeps
the prompt, the image and the thinking (as the shared prefix) of that answer, so the two continuations branch at the
answer. Wanted text: for captions, the reference that agrees most with the other references (consensus_reference);
for yes/no tasks, the right word in the model's own style ("No." against "Yes."); otherwise the row's answer. Then run
`probe.py collect-pairs --thinking on|off` with the run's thinking setting. Same-prompt pairs cannot separate a
one-token decision (both branch from the same state): for yes/no or letter answers, use `probe.py collect-decisions`.
"""
import argparse
import json
from pathlib import Path

from grade import normalise
from probe import decision_labels, read_runs, row_prompt, split_answer, task_rows
from tasks import spec

ONE_TOKEN_WORDS = 2


def words(text):
    return set(normalise(text).split())


def consensus_reference(refs):
    """The reference sharing the most words with the others (ties: the first)."""
    def agreement(i):
        own = words(refs[i])
        return sum(len(own & words(r)) / max(len(own | words(r)), 1) for j, r in enumerate(refs) if j != i)
    return refs[max(range(len(refs)), key=lambda i: (agreement(i), -i))]


def styled_yes_no(gold, answer):
    word = gold.capitalize() if answer[:1].isupper() else gold
    return word + ("." if answer.rstrip().endswith(".") else "")


def wanted_text(task, row, answer):
    gold = row.get(spec(task, "answer"))
    if isinstance(gold, list):
        return consensus_reference(gold)
    if isinstance(gold, str) and normalise(gold) in ("yes", "no"):
        return styled_yes_no(normalise(gold), answer.strip())
    return gold if isinstance(gold, str) else None


def make_pairs(task, data, recs, graded, limit=None):
    rows, labels = task_rows(task, data), decision_labels(graded)
    pairs = []
    for rec in recs:
        before, answer = split_answer(rec)
        if labels.get((rec["id"], rec["sample"])) != "unwanted" or answer is None or rec["id"] not in rows:
            continue
        row = rows[rec["id"]]
        wanted = wanted_text(task, row, answer)
        if not wanted or normalise(wanted) == normalise(answer):
            continue
        prompt, image = row_prompt(task, row)
        pair = {"id": f"{rec['id']}#{rec['sample']}" if rec["sample"] else rec["id"], "prompt": prompt,
                "prefix": before, "wanted": wanted, "unwanted": answer.rstrip()}
        pairs.append({**pair, "image": image} if image else pair)
    return pairs[:limit]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--graded", required=True, help="grade.py --out file that graded --runs")
    p.add_argument("--label", help="the run's name in --graded (default: the --runs file name)")
    p.add_argument("--max", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    graded = json.loads(Path(args.graded).read_text())["graded"]
    label = args.label or Path(args.runs).stem
    if label not in graded:
        raise SystemExit(f"{args.graded} has no graded run '{label}' (it has {sorted(graded)}); pass --label")
    pairs = make_pairs(args.task, args.data, read_runs(args.runs), graded[label], args.max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pairs))
    print(f"{len(pairs)} pairs -> {args.out}")
    if pairs and max(len(x["unwanted"].split()) for x in pairs) <= ONE_TOKEN_WORDS:
        print("WARNING: one-word answers; same-prompt pairs cannot separate the decision. Use probe.py "
              "collect-decisions instead.")
    if len(pairs) < 12:
        print("WARNING: under 12 pairs; run more practice questions first")


if __name__ == "__main__":
    main()
