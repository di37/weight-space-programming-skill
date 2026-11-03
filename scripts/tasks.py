"""Benchmarks: data loading, the user-turn prompt, the default grader and the fields kept with each answer.

Built in: mmlu (MMLU-Pro rows: question_id, category, question, options, answer letter; official zero-shot chain-of-
thought instruction), math (MATH-500 rows: unique_id, subject, problem, answer; Qwen's boxed-answer instruction).
Anything else: jsonl, your own JSON-lines file with `id` and `prompt` (the full user turn), and optionally `answer`
(for graders that need one) and `group` (a category for per-group scores).
"""
import json
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
MMLU_INSTRUCTION = ("The following are multiple choice questions (with answers) about {subject}. Think step by step "
                    "and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.")
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def mmlu_prompt(row):
    options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(row["options"]))
    return (MMLU_INSTRUCTION.format(subject=row["category"]) + "\n\nQuestion:\n" + row["question"] + "\nOptions:\n"
            + options + "\nAnswer: Let's think step by step.")


def math_prompt(row):
    return row["problem"] + "\n" + MATH_INSTRUCTION


TASKS = {
    "mmlu": {"id": "question_id", "group": "category", "prompt": mmlu_prompt, "grader": "mc"},
    "math": {"id": "unique_id", "group": "subject", "prompt": math_prompt, "grader": "math"},
    "jsonl": {"id": "id", "group": "group", "prompt": lambda row: row["prompt"], "grader": "none"},
}


def row_id(task, row):
    return str(row[TASKS[task]["id"]])


def load_rows(path, ids_file=None, task=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if ids_file:
        wanted = set(Path(ids_file).read_text().split())
        key = TASKS[task]["id"] if task else None
        rows = [r for r in rows if str(r[key] if key else r.get("question_id", r.get("unique_id", r.get("id"))))
                in wanted]
    return rows


def job_meta(task, row, sample):
    return {"id": row_id(task, row), "sample": sample, "group": row.get(TASKS[task]["group"], "all"),
            "answer": row.get("answer")}
