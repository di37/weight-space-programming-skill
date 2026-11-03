"""Make disjoint, seeded practice and held-out splits, plus a small fixed held-out test set.

    python scripts/make_splits.py --task mmlu --source <MMLU-Pro test parquet> --out data/mmlu_science
        [--groups physics chemistry biology engineering] [--practice 32] [--heldout 40] [--test 30] [--seed 0]
    python scripts/make_splits.py --task math --source <MATH-500 test.jsonl> --out data/math500 --practice 25 --heldout 75 --test 30

Per group (MMLU-Pro category or MATH subject): shuffle with the seed, take --practice then --heldout rows. Writes
practice.jsonl and heldout.jsonl, and test_ids.txt: --test held-out ids spread as evenly as possible over the groups in
the order given (30 over 4 groups = 8, 8, 7, 7), taken in file order, chosen before any answer is seen.
"""
import argparse
import json
import random
from pathlib import Path

from tasks import TASKS

MMLU_SCIENCE = ("physics", "chemistry", "biology", "engineering")


def read_source(task, path):
    if task == "math":
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    import pandas as pd  # noqa: PLC0415

    df = pd.read_parquet(path)
    return [{"question_id": str(r.question_id), "category": r.category, "question": r.question,
             "options": [str(o) for o in r.options], "answer": r.answer, "src": r.src} for r in df.itertuples()]


def quotas(total, groups):
    base, extra = divmod(total, len(groups))
    return {g: base + (i < extra) for i, g in enumerate(groups)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--groups", nargs="+")
    p.add_argument("--practice", type=int, default=32, help="per group")
    p.add_argument("--heldout", type=int, default=40, help="per group")
    p.add_argument("--test", type=int, default=30, help="held-out questions in test_ids.txt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    spec, rows = TASKS[args.task], read_source(args.task, args.source)
    groups = args.groups or (list(MMLU_SCIENCE) if args.task == "mmlu" else sorted({r[spec["group"]] for r in rows}))
    rng, splits = random.Random(args.seed), {"practice": [], "heldout": []}
    for g in groups:
        members = [r for r in rows if r[spec["group"]] == g]
        rng.shuffle(members)
        splits["practice"] += members[:args.practice]
        splits["heldout"] += members[args.practice:args.practice + args.heldout]
    ids = [{str(r[spec["id"]]) for r in s} for s in splits.values()]
    assert not ids[0] & ids[1], "practice and held-out overlap"
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    want, test = quotas(args.test, groups), []
    for g in groups:
        test += [str(r[spec["id"]]) for r in splits["heldout"] if r[spec["group"]] == g][:want[g]]
    (args.out / "test_ids.txt").write_text("\n".join(test) + "\n")
    print(json.dumps({name: len(part) for name, part in splits.items()} | {"test": len(test), "quotas": want}))


if __name__ == "__main__":
    main()
