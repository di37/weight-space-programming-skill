# Evaluation: splits, protocol, MMLU-Pro, paired statistics, report

## Contents
1. Splits
2. The sampling protocol
3. Grading maths answers
4. MMLU-Pro (and other multiple-choice) targets
5. Paired statistics
6. Behaviour audit
7. Report template

## 1. Splits

Fix every split before the first edit, and write each to disk with a hash so you can show later that it never changed.

```python
import hashlib, json, random
from collections import defaultdict

def make_splits(items, seed, sizes):
    """items: [{"id", "category", ...}]; sizes: {"heldout": 0.5, "diagnostic": 0.1, "screen": 0.15, "confirm": 0.25}
    as fractions of each category, assigned in that order; the last split also takes any rounding remainder.
    Returns {split: [ids]}."""
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it["id"])
    rng = random.Random(seed)
    splits = defaultdict(list)
    names = list(sizes)
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        rng.shuffle(ids)
        start = 0
        for name in names:
            end = len(ids) if name == names[-1] else start + round(sizes[name] * len(ids))
            splits[name] += ids[start:end]
            start = end
    return dict(splits)

def save_splits(splits, path):
    text = json.dumps(splits, indent=1, sort_keys=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()   # write this into the work notes
```

Guidance:

- **Held-out** is used once, at the end, for base and edited.
- **Diagnostic** is for the baseline reading, the failure categories and the matched contexts.
- **Screen** is for choosing among candidates, and must be disjoint from the texts the columns were chosen on.
- **Confirm** is for paired confirmation of the frozen edit. It is not used for any choice.
- **Size by the effect.** One accuracy has an SE of about 50/√n points. A paired difference is tighter whenever base
  and edited agree on most items. Run time is the real constraint: a base model that loops writes about 32k tokens per
  answer (see `engines.md`).
- **When the benchmark has a train split**, take practice from it and evaluate on test, as the authors did with GSM8K
  and MATH. When it has none, as with MMLU-Pro, stratify the test pool by category as above.

## 2. The sampling protocol

This is the authors' `think35` profile for Qwen3.5-0.8B, from `exp8_math/code/evaluate.py` and `profiles.py`. Use it
unchanged for base and edited, and for practice and held-out alike.

| Setting | Value |
|---|---|
| Chat template | `apply_chat_template([{"role": "user", "content": ...}], add_generation_prompt=True, enable_thinking=True)`. The rendered prompt ends `<\|im_start\|>assistant\n<think>\n`. |
| User turn (maths) | problem + `"\n"` + `"Please reason step by step, and put your final answer within \boxed{}."` |
| temperature / top_p / top_k | 0.6 / 0.95 / 20 |
| presence penalty | 1.5 on every token generated so far (vLLM semantics: output tokens only) |
| max new tokens | 32,768, reasoning and answer together |
| stop | `<\|im_end\|>`, `<\|endoftext\|>` |
| samples per problem | authors: MATH-500 4, GSM8K 2, AMC23 16, MMLU-Pro 1, GPQA 4, IFEval 2 |

The Qwen3.5-0.8B model card warns that this model is prone to thinking loops. The protocol keeps thinking on
deliberately, because the loops are the target. The authors' first profile (`nothink35`: temperature 0.7, top_p 0.8,
8,192 tokens, thinking off) is a different benchmark. Do not mix the two.

## 3. Grading maths answers

`scripts/grade.py --task math` re-implements `exp8_math/code/mathqa.py`:

1. Take the text after the last `</think>`.
2. The answer is the content of the last `\boxed{...}` (or `\fbox{...}`), brace matched. `\boxed 5` is read up to the
   next `$` or the end of the line.
3. No closed box counts as wrong.
4. `correct = math_verify.verify(parse("\\boxed{gold}"), parse("\\boxed{pred}"))`.

To compare per problem with the authors, their MATH-500 results are in their logs:
`exp8_math/eval/{unedited,p50c}/parts/math500_*/math500_per_problem.json`.

## 4. MMLU-Pro (and other multiple-choice) targets

The authors evaluated MMLU-Pro with "a fixed random 1,000 of 12,032 [questions], the official prompt and answer
extraction, 1 sample, the reasoning cut at the last `</think>`". Their multiple-choice module (`mcqa`) is not in the
published logs, so their exact prompt is unknown. Pick a prompt, write it down, and use it identically everywhere.

- **Data.** TIGER-Lab/MMLU-Pro on Hugging Face. Each item has `question_id`, `question`, `options` (up to 10 options,
  A–J; some items have fewer), `answer` (a letter), `answer_index` and `category`. Science targets: `physics`,
  `chemistry`, `biology`, `engineering`. The `validation` split is only the few-shot examples, so practice and held-out
  both come from `test`.
- **Prompt.** The official instruction is: *Think step by step and then finish your answer with "the answer is (X)"
  where X is the correct letter choice.* A natural zero-shot user turn for a thinking model looks like this:

  ```
  The following are multiple choice questions (with answers) about {category}. Think step by step and then finish
  your answer with "the answer is (X)" where X is the correct letter choice.

  Question:
  {question}
  Options:
  A. {option 0}
  B. {option 1}
  ...
  Answer: Let's think step by step.
  ```

  This is exactly what `scripts/tasks.py` sends.

  The official script is 5-shot CoT. With thinking on and a 32k budget, zero-shot is the cleaner comparison. Say which
  you used.
- **Extraction (official order).** Read the text after the last `</think>`. Try `answer is \(?([A-J])\)?` first, then
  `.*[aA]nswer:\s*([A-J])`, then the last standalone `\b[A-J]\b`. An answer cut before `</think>` has no answer and
  counts as wrong. Some versions of the official script guess a random letter when extraction fails. Do not do that,
  because it adds noise to exactly the answers the edit is meant to rescue. Report the no-answer share instead.
- **Chance level is 10%** with 10 options. A base model that loops on most items will sit near it.
- **Reference points.** The authors' p50c maths edit moved MMLU-Pro (all subjects) only from 40.3 to 45.5, against
  MATH-500's 7.9 → 48.1. Expect a science gain to be bounded by how often base answers in those categories are cut
  or unanswered. Measure that first.
- **Scripts.** `scripts/make_splits.py --task mmlu` (practice, held-out and a fixed test-id list),
  `scripts/run_eval.py --task mmlu` (generation, resumable JSON lines) and `scripts/grade.py --task mmlu`.

## 5. Paired statistics

Pair by item. When there are several samples per item, average them per item first, because samples of one item are
correlated and treating them as independent understates the SE.

```python
import numpy as np
from scipy.stats import binomtest

def paired_report(base, edited):
    """base, edited: {item_id: mean correctness over that item's samples}, over the same items."""
    ids = sorted(base.keys() & edited.keys())
    b = np.array([base[i] for i in ids])
    e = np.array([edited[i] for i in ids])
    d = e - b
    gain = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    rescues, losses = int((d > 0).sum()), int((d < 0).sum())
    exact_p = binomtest(rescues, rescues + losses, 0.5).pvalue if rescues + losses else 1.0
    return {"items": len(ids), "base": b.mean(), "edited": e.mean(), "gain": gain, "se": se,
            "ci95": (gain - 1.96 * se, gain + 1.96 * se), "rescues": rescues, "losses": losses,
            "exact_sign_test_p": exact_p}
```

- With one sample per item, the sign test on rescues against losses is McNemar's exact test.
- Report each category separately as well as pooled. Subgroups are noisy, so give their n.
- Pair only within one comparison. Never compare an edited score with a base score measured on different problems or
  in a different run of a different size: batch composition changes the random numbers.
- `scripts/grade.py` reports fixed and broken counts with the exact sign test. With several samples per item, add the
  paired SE above.

## 6. Behaviour audit

Compute each of these for base and edited, on the confirmation and held-out sets:

| Measure | Why it matters |
|---|---|
| share cut at the token limit | the mechanism the edit targets |
| share with no extractable answer | rescues should come from here, and from the cut share |
| share that closed thinking (`</think>`) | the edit should make thinking finish, not skip it |
| median tokens of correct answers | should stay long (p50c: 9,172 edited against 5,310 base); a collapse means the edit forced early stops |
| tail repetition (`method.md` section 9) | should fall sharply on cut answers |
| whitespace-only tails, other new degeneration | the edit's own failure modes (p50c: 0 → 51 of 1,152) |
| accuracy on answers base finished | should not fall: the edit should rescue, not trade |
| each loss, read in full | regressions show what the edit broke |

## 7. Report template

```markdown
# <Model> weight edit for <target>: results

## Edit
- Tensors: <names>. Columns per tensor: <k> of <m>. α = <α>. Weights changed: <count> (<share>% of <total>).
- Relative Frobenius change per tensor: <...>. sha256 of each edited tensor: <...>.
- Direction: <n> matched pairs from <split>, residual after layer <l>, last <T> tokens.
- Rules kept: no gradients or training; no edits to embeddings, output head or final norm; no whole-tensor scaling.

## Selection (disclosed)
| Candidate | Teacher-forced repetitive / productive Δ (nats/token) | Screening paired gain (SE, n) | Chosen? |
|---|---|---|---|
| ... | ... | ... | ... |
| random control (norm-matched) | ... | ... | no |

## Held-out result (seen once, after freezing)
| | Base | Edited | Paired Δ (SE, 95% CI) | Rescues / losses | p |
|---|---|---|---|---|---|
| pooled | | | | | |
| per category ... | | | | | |

## Behaviour
| | Base | Edited |
|---|---|---|
| cut at token limit | | |
| no answer | | |
| closed thinking | | |
| median tokens (correct) | | |
| tail repetition (cut answers) | | |

## Caveats
- The edit removes <failure mode>. It lets the model finish answers it could already work out; it does not add
  knowledge. Gains are bounded by how much of base's failure was this mode.
- <regressions and new degeneration>, <deviations from the authors' protocol>, <sample sizes>.
```
