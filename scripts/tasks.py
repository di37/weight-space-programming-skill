"""Benchmarks: data loading, the user-turn prompt, the default grader and the fields kept with each answer.

Built in, text: mmlu (MMLU-Pro rows: question_id, category, question, options, answer letter; official zero-shot
chain-of-thought instruction), math (MATH-500 rows: unique_id, subject, problem, answer; Qwen's boxed-answer
instruction). Built in, image (rows from make_image_data.py, each with an `image` file path): caption (COCO captions,
`references`; lmms-eval's caption instruction; CIDEr-D), relevance (an image and a `text`: does the text describe the
image, answer yes/no), pope (POPE object questions, answer yes/no; lmms-eval's instruction). Anything else: jsonl,
your own JSON-lines file with `id` and `prompt` (the full user turn), and optionally `answer` (for graders that need
one), `group` (a category for per-group scores) and `image` (a file path, which makes the row an image job).
Relative image paths are read relative to the data file. Image presets default to the non-thinking profile and 512
new tokens; text tasks to the authors' thinking protocol and 32,768.
"""
import json
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
MMLU_INSTRUCTION = ("The following are multiple choice questions (with answers) about {subject}. Think step by step "
                    "and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.")
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
CAPTION_INSTRUCTION = "Provide a one-sentence caption for the provided image."
RELEVANCE_INSTRUCTION = ("Is the following text relevant to the image, that is, does it describe what the image shows? "
                         "Answer yes or no.\nText: \"{text}\"")
POPE_INSTRUCTION = "{question}\nAnswer the question using a single word or phrase."
DEFAULTS = {"answer": "answer", "image": None, "profile": "thinking", "max_tokens": 32768}
IMAGE_TASK = {"image": "image", "profile": "non-thinking", "max_tokens": 512}


def mmlu_prompt(row):
    options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(row["options"]))
    return (MMLU_INSTRUCTION.format(subject=row["category"]) + "\n\nQuestion:\n" + row["question"] + "\nOptions:\n"
            + options + "\nAnswer: Let's think step by step.")


def math_prompt(row):
    return row["problem"] + "\n" + MATH_INSTRUCTION


TASKS = {
    "mmlu": {"id": "question_id", "group": "category", "prompt": mmlu_prompt, "grader": "mc"},
    "math": {"id": "unique_id", "group": "subject", "prompt": math_prompt, "grader": "math"},
    "caption": {"id": "id", "group": "group", "prompt": lambda row: CAPTION_INSTRUCTION, "grader": "cider",
                "answer": "references", **IMAGE_TASK},
    "relevance": {"id": "id", "group": "answer", "prompt": lambda row: RELEVANCE_INSTRUCTION.format(text=row["text"]),
                  "grader": "yesno", **IMAGE_TASK},
    "pope": {"id": "id", "group": "answer", "prompt": lambda row: POPE_INSTRUCTION.format(question=row["question"]),
             "grader": "yesno", **IMAGE_TASK},
    "jsonl": {"id": "id", "group": "group", "prompt": lambda row: row["prompt"], "grader": "none", "image": "image"},
}


def spec(task, key):
    return TASKS[task].get(key, DEFAULTS.get(key))


def row_id(task, row):
    return str(row[TASKS[task]["id"]])


def with_image_path(row, field, data_dir):
    """A copy of the row with its image path made absolute (relative paths are relative to the data file)."""
    if not field or not row.get(field):
        return row
    path = Path(row[field])
    return {**row, field: str(path if path.is_absolute() else (data_dir / path).resolve())}


def load_rows(path, ids_file=None, task=None):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if ids_file:
        wanted = set(Path(ids_file).read_text().split())
        key = TASKS[task]["id"] if task else None
        rows = [r for r in rows if str(r[key] if key else r.get("question_id", r.get("unique_id", r.get("id"))))
                in wanted]
    field = spec(task, "image") if task else "image"
    return [with_image_path(r, field, path.parent) for r in rows]


def job_meta(task, row, sample):
    field = spec(task, "image")
    return {"id": row_id(task, row), "sample": sample, "group": row.get(TASKS[task]["group"], "all"),
            "answer": row.get(spec(task, "answer")), "image": row.get(field) if field else None}
