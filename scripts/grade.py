"""Grade answer files from run_eval.py and compare runs on the same questions.

    python scripts/grade.py --task mmlu base.jsonl copy.jsonl [--out summary.json]
    python scripts/grade.py --task jsonl --grader contains base.jsonl copy.jsonl
    python scripts/grade.py --task jsonl --grader my_grader.py:grade --flag "(?i)great question" base.jsonl copy.jsonl
    python scripts/grade.py --task caption base.jsonl copy.jsonl

Graders (default per task: mmlu -> mc, math -> math, caption -> cider, relevance and pope -> yesno, jsonl -> none):
  mc        MMLU-Pro reading: 'answer is (X)', else 'Answer: X', else the last standalone letter A-J
  math      the last \\boxed{...}, checked with math-verify (install it as references/engines.md says)
  cider     a score, not right/wrong: CIDEr-D (cider.py) of the caption (first line, labels and quotes stripped)
            against the row's reference list, IDF from the first file's references
  yesno     the first standalone yes/no in the answer equals the row's `answer`; adds yes-rate, precision, recall, F1
  exact     normalised equality of the --pattern capture (default: the last non-empty line) with the row's `answer`
  contains  the normalised `answer` appears in the answer text
  regex     the answer text matches --pattern (no `answer` needed)
  none      no correctness: behaviour metrics only
  FILE.py:FUNCTION  your own: FUNCTION(row, answer_text) -> bool, row being the JSON line written by run_eval.py
The answer text is what follows the last '</think>' when the model thought, else the whole text. Strict: an answer
whose thinking never closed has none. Lenient: read the unclosed text too. Behaviour metrics, always: cut at the token
limit, thinking closed, verbatim loop (loops.py on words), median tokens; with --flag REGEX, the share of answer texts
that match (e.g. a phrase or habit you want gone), and median answer words. The first file is the reference for
pairing; with several files, every run is scored on the questions all of them answered. For a scored grader (cider),
pairing counts per-answer scores: "fixed" = scored higher than the reference run, "broke" = scored lower.
"""
import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

from cider import CiderD
from loops import loop_onset
from tasks import TASKS

ANSWER_IS = re.compile(r"answer is \(?([A-J])\)?")
ANSWER_COLON = re.compile(r".*[aA]nswer:\s*\(?([A-J])\)?")
LAST_LETTER = re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL)
YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
CAPTION_LABEL = re.compile(r"^(caption|description)\s*:\s*", re.IGNORECASE)
SHORT_ANSWER_CHARS = 200
VERIFY_TIMEOUT_S = 5
LOOP_SPAN_WORDS = 256


def answer_part(row, strict=True):
    text = row["text"]
    if not row.get("thinking", True):
        return text
    if "</think>" in text:
        return text.split("</think>")[-1]
    return None if strict else text


def normalise(s):
    return re.sub(r"\s+", " ", str(s)).strip().strip(".").lower()


def extract_letter(text):
    if text is None:
        return None
    for pattern in (ANSWER_IS, ANSWER_COLON, LAST_LETTER):
        m = pattern.search(text)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return None


def box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    stripped = rest.lstrip()
    if stripped.startswith("{"):
        depth = 0
        for k, ch in enumerate(stripped):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                return stripped[1:k].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text):
    if text is None:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    return None if idx < 0 else box_at(text, idx)


def math_equal(pred, gold):
    import math_verify  # noqa: PLC0415

    if pred is None or gold is None:
        return False
    g = math_verify.parse("\\boxed{" + gold + "}", parsing_timeout=VERIFY_TIMEOUT_S)
    p = math_verify.parse("\\boxed{" + pred + "}", parsing_timeout=VERIFY_TIMEOUT_S)
    try:
        return bool(g and p and math_verify.verify(g, p, timeout_seconds=VERIFY_TIMEOUT_S))
    except Exception as exc:  # noqa: BLE001 - odd LaTeX: count as wrong, but say so
        print(f"  math-verify failed on {pred!r} vs {gold!r}: {type(exc).__name__}")
        return False


def final_line(text):
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else None


def caption_text(text):
    """The caption: the first non-empty line, without a 'Caption:' label, markdown emphasis or quotes."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    line = CAPTION_LABEL.sub("", lines[0]).strip("*_ ").strip("\"'\u201c\u201d ").strip()
    return line or None


def yes_no(text):
    m = YES_NO.search(text) if text else None
    return m.group(1).lower() if m else None


def cider_grader(rows):
    """CIDEr-D with IDF from the references of rows (one set per question id)."""
    by_id = {r["id"]: r["answer"] for r in rows if r.get("answer")}
    scorer = CiderD(list(by_id.values()))

    def grade_caption(row, t):
        pred = caption_text(t)
        return pred, scorer.score(pred, row["answer"]) if pred is not None else 0.0
    grade_caption.scored = True
    return grade_caption


def without_pred(grader):
    """Mark a grader that judges the whole text without extracting a prediction, so any verdict counts as an answer.
    For the others (mc, math, yesno, exact, cider), an answer counts only when a prediction was extracted."""
    grader.pred_free = True
    return grader


def make_grader(name, pattern, rows=None):
    """(row, answer_text) -> (pred, correct or None), or (pred, score) for a grader marked `scored`. rows: every
    answer being graded, for graders with corpus statistics (cider)."""
    rx = re.compile(pattern, re.DOTALL) if pattern else None
    if name == "cider":
        return cider_grader(rows or [])
    if name == "yesno":
        return lambda row, t: (p := yes_no(t), p is not None and p == normalise(row["answer"]))
    if name == "mc":
        return lambda row, t: (p := extract_letter(t), p == row["answer"])
    if name == "math":
        return lambda row, t: (p := last_boxed(t), math_equal(p, row["answer"]))
    if name == "exact":
        def exact(row, t):
            m = rx.search(t) if (rx and t) else None
            pred = (m.group(1) if m and m.groups() else m.group(0) if m else None) if rx else final_line(t)
            return pred, pred is not None and normalise(pred) == normalise(row["answer"])
        return exact
    if name == "contains":
        return without_pred(lambda row, t: (None, t is not None and normalise(row["answer"]) in normalise(t)))
    if name == "regex":
        return without_pred(lambda row, t: (None, bool(t and rx.search(t))))
    if name == "none":
        return without_pred(lambda row, t: (None, None))
    path, _, func = name.partition(":")
    spec = importlib.util.spec_from_file_location("user_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    user = getattr(module, func or "grade")
    return without_pred(lambda row, t: (None, bool(user(row, t)) if t is not None else False))


def grade(row, grader, flag=None):
    strict, lenient = answer_part(row), answer_part(row, strict=False)
    pred, value = grader(row, strict)
    _, value_lenient = grader(row, lenient)
    scored = getattr(grader, "scored", False)
    words = row["text"].split()
    gold = row.get("answer")
    return {"id": row["id"], "sample": row["sample"], "group": row.get("group", "all"), "pred": pred,
            "gold": gold if isinstance(gold, str) and len(gold) <= SHORT_ANSWER_CHARS else None,
            "correct": None if scored else value, "correct_lenient": None if scored else value_lenient,
            "score": value if scored else None, "score_lenient": value_lenient if scored else None,
            "cut": row["finish_reason"] == "length", "answer_words": len(strict.split()) if strict else 0,
            "answered": strict is not None and (pred is not None or (getattr(grader, "pred_free", False)
                                                                      and value is not None)),
            "closed_thinking": (not row.get("thinking", True)) or "</think>" in row["text"],
            "looped": loop_onset(words, span=LOOP_SPAN_WORDS) is not None,
            "flagged": bool(flag and lenient and re.search(flag, lenient)), "n_tokens": row["n_tokens"]}


def pct(xs):
    return round(100 * sum(xs) / max(len(xs), 1), 1)


def yes_no_metrics(graded):
    """Yes-rate and precision/recall/F1 with "yes" as the positive class (POPE's convention)."""
    tp = sum(g["pred"] == "yes" and g["gold"] == "yes" for g in graded)
    said_yes, gold_yes = sum(g["pred"] == "yes" for g in graded), sum(g["gold"] == "yes" for g in graded)
    precision, recall = tp / max(said_yes, 1), tp / max(gold_yes, 1)
    return {"yes_rate": pct([g["pred"] == "yes" for g in graded]), "precision": round(100 * precision, 1),
            "recall": round(100 * recall, 1),
            "f1": round(100 * 2 * precision * recall / (precision + recall), 1) if precision + recall else 0.0}


def summarize(graded, flag=None):
    n = len(graded)
    out = {"answers": n, "cut_at_token_limit": pct([g["cut"] for g in graded]),
           "closed_thinking": pct([g["closed_thinking"] for g in graded]),
           "looped": pct([g["looped"] for g in graded]),
           "median_tokens": sorted(g["n_tokens"] for g in graded)[n // 2] if n else None,
           "median_answer_words": sorted(g["answer_words"] for g in graded)[n // 2] if n else None}
    if flag:
        out["flagged"] = pct([g["flagged"] for g in graded])
    if n and graded[0]["score"] is not None:
        scores = [g["score"] for g in graded]
        mean = sum(scores) / n
        sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / max(n - 1, 1))
        groups = sorted({g["group"] for g in graded})
        out.update(score=round(mean, 3), score_se=round(sd / math.sqrt(n), 3),
                   score_lenient=round(sum(g["score_lenient"] for g in graded) / n, 3),
                   no_answer=pct([not g["answered"] for g in graded]),
                   by_group={grp: round(sum(g["score"] for g in graded if g["group"] == grp)
                                        / sum(g["group"] == grp for g in graded), 3) for grp in groups})
    if n and graded[0]["correct"] is not None:
        acc = sum(g["correct"] for g in graded) / n
        groups = sorted({g["group"] for g in graded})
        out.update(accuracy=round(100 * acc, 1), accuracy_se=round(100 * math.sqrt(acc * (1 - acc) / n), 1),
                   accuracy_lenient=pct([g["correct_lenient"] for g in graded]),
                   no_answer=pct([not g["answered"] for g in graded]),
                   by_group={grp: pct([g["correct"] for g in graded if g["group"] == grp]) for grp in groups})
        if all(g["gold"] in ("yes", "no") for g in graded):
            out.update(yes_no_metrics(graded))
    return out


def sign_test(wins, losses):
    """Two-sided exact sign test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def paired_scores(before, after):
    """Per-answer scores in both runs: fixed = higher in the second run, broke = lower."""
    b = {(g["id"], g["sample"]): g["score"] for g in before}
    a = {(g["id"], g["sample"]): g["score"] for g in after}
    common = sorted(b.keys() & a.keys())
    fixed, broke = sum(a[k] > b[k] for k in common), sum(a[k] < b[k] for k in common)
    change = sum(a[k] - b[k] for k in common) / max(len(common), 1)
    return {"metric": "score", "paired_answers": len(common), "fixed": fixed, "broke": broke,
            "same": len(common) - fixed - broke, "mean_change": round(change, 3), "sign_test_p": sign_test(fixed, broke)}


def paired(before, after, key="correct"):
    """Pairs present in both runs: how many the second run fixed and broke (key: 'correct', 'score', or 'flagged'
    inverted)."""
    if key == "score":
        return paired_scores(before, after)
    good = (lambda g: g[key]) if key == "correct" else (lambda g: not g[key])
    b = {(g["id"], g["sample"]): good(g) for g in before}
    a = {(g["id"], g["sample"]): good(g) for g in after}
    common = sorted(b.keys() & a.keys())
    fixed, broke = sum(a[k] and not b[k] for k in common), sum(b[k] and not a[k] for k in common)
    return {"metric": key, "paired_answers": len(common), "fixed": fixed, "broke": broke,
            "both_good": sum(a[k] and b[k] for k in common), "sign_test_p": sign_test(fixed, broke)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--grader", help="mc, math, exact, contains, regex, none, or FILE.py:FUNCTION")
    p.add_argument("--pattern", help="regex for --grader exact (capture group 1) or regex")
    p.add_argument("--flag", help="regex for a behaviour to count in the answer text")
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    runs = {f.stem: [json.loads(line) for line in f.read_text().splitlines() if line] for f in args.files}
    if len(runs) < len(args.files):
        raise SystemExit("answer files need different names (each file's name labels its run)")
    grader = make_grader(args.grader or TASKS[args.task]["grader"], args.pattern, rows=next(iter(runs.values())))
    graded = {label: [grade(r, grader, args.flag) for r in rows] for label, rows in runs.items()}
    if len(graded) > 1:
        common = set.intersection(*({(g["id"], g["sample"]) for g in rows} for rows in graded.values()))
        graded = {k: [g for g in rows if (g["id"], g["sample"]) in common] for k, rows in graded.items()}
    summary = {label: summarize(rows, args.flag) for label, rows in graded.items()}
    first, *rest = list(graded)
    for label in rest:
        if graded[first] and graded[first][0]["score"] is not None:
            summary[f"{label} vs {first}"] = paired(graded[first], graded[label], key="score")
        elif graded[first] and graded[first][0]["correct"] is not None:
            summary[f"{label} vs {first}"] = paired(graded[first], graded[label])
        if args.flag:
            summary[f"{label} vs {first} (flag)"] = paired(graded[first], graded[label], key="flagged")
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "graded": graded}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
