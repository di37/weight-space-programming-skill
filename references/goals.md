# Goals beyond maths and science: recipes

The method needs three things for any goal:

1. **A benchmark.** Prompts the model answers, and a way to score or count the answers.
2. **A contrast.** Examples of the unwanted behaviour next to the wanted one, as similar as possible otherwise.
3. **A behaviour the model already "knows" how to avoid.** The edit switches a habit off; it does not teach facts or
   skills. If the model cannot do the task even when the habit is absent, expect no gain (the Gemma null result).

## Contents
1. Choosing the contrast
2. Writing matched pairs
3. Benchmarks and graders for your own goal
4. Recipes
5. Out of scope

## 1. Choosing the contrast

| The failure looks like | Contrast source | Command |
|---|---|---|
| Answers repeat themselves until the token limit (thinking or answer loops) | **automatic**: loop onset in the model's own answers | `probe.py collect --runs <base answers>` |
| Anything else you can write down (a phrase, a habit, a format, a language, a tone) | **matched pairs** you write | `probe.py collect-pairs --pairs pairs.jsonl` |

Loops are found automatically because the model produces them itself; every other behaviour needs pairs.

## 2. Writing matched pairs

One JSON line per pair: `{"id", "prompt", "prefix" (optional), "wanted", "unwanted"}`.

- **Same prompt, same prefix, one difference.** Everything except the behaviour should match: topic, facts, length
  within about 20%, tone otherwise. If the unwanted text is also longer or more confident, the direction picks those
  up as well.
- **Realistic continuations.** Write what the model itself would plausibly say, in its style. The best source is the
  model's own outputs: take a base answer that shows the behaviour as `unwanted`, and the same answer with the
  behaviour fixed as `wanted`.
- **How many.** At least 12 pairs (8 build and 4 screen after the automatic split). 20–40 is better. The authors found
  their direction from 4 hand-written pairs and checked it on 6 fresh ones, but those pairs were very tightly matched.
- **The prefix** holds shared text before the branch point, for example the first sentence of the answer, so the
  windows cover only where the two differ.
- **Thinking.** Use `--thinking on` if the behaviour happens inside the thinking, else `off`. It must match the
  profile you evaluate with.
- **Clean windows.** Pass `--runs` and `--data` with finished base answers, so screening also measures the side
  effect on normal text.

## 3. Benchmarks and graders for your own goal

Write prompts as `{"id", "prompt", "answer" (optional), "group" (optional)}` JSON lines, then use `--task jsonl`.
Split them yourself into practice and held-out files, by a fixed rule chosen before any answer is seen: about 60/40,
stratified by group.

| You can check answers by | Grader |
|---|---|
| A short exact answer | `--grader exact` (last line), or `--grader exact --pattern "answer: (.*)"` |
| The right answer appearing anywhere | `--grader contains` |
| A required format or phrase | `--grader regex --pattern "..."` |
| Running code, calling a checker, anything else | `--grader my_grader.py:grade`, with `def grade(row, text) -> bool` |
| No single right answer (style, tone) | `--grader none` plus `--flag "<regex for the unwanted behaviour>"` |

`--flag` counts how often the unwanted behaviour appears. With two runs, `grade.py` pairs it too: "fixed" means the
behaviour disappeared from an answer. For goals without a right answer, the flag rate is the headline number.
Always also read some answers: a flag regex can be gamed by rephrasing.

## 4. Recipes

| Goal | Benchmark | Contrast | Score |
|---|---|---|---|
| Maths (built in) | MATH-500 (`--task math`) | automatic loops | `--task math` (math-verify) |
| Science (built in) | MMLU-Pro physics, chemistry, biology, engineering (`--task mmlu`) | automatic loops | `--task mmlu` |
| Other MMLU-Pro subjects | `make_splits.py --task mmlu --groups law history ...` | automatic loops | `--task mmlu` |
| Coding | your problems with tests, `--task jsonl` | automatic loops if it loops, else pairs (e.g. wrong output format) | `--grader my_tests.py:grade` running the tests in a sandbox |
| Following format instructions | prompts with a required format | pairs: right format vs typical wrong one | `--grader regex --pattern <format>` |
| Answering in the asked language | prompts in language X | pairs: reply in X vs reply in English | `--grader my_lang.py:grade` (a language check) or `--flag` |
| Sycophancy (agreeing with a wrong user suggestion) | questions with a wrong suggested answer | pairs: correct answer vs agreeing with the user | `--grader contains` (correct answer) and `--flag` for agreement phrases |
| Verbosity or filler | open questions | pairs: concise vs padded with filler | `--grader none --flag "<filler phrases>"`, and median tokens |
| Missing final answer | task prompts | pairs: ends with the required final line vs trails off | `--grader regex --pattern <final line>` |

The authors' sycophancy experiment (their Experiment 7) found the behaviour by comparing attention patterns, not by
an edit. Treat sycophancy edits as exploratory.

## 5. Out of scope

Do not use this skill to remove a model's safety behaviour: making it comply with harmful requests, stripping
refusals, or bypassing its safeguards. The rules in SKILL.md also exclude edits that only work by forcing a global
change (sharper or flatter outputs, always closing the thinking early).
