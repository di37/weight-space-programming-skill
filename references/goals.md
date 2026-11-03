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
5. Image tasks
6. Out of scope

## 1. Choosing the contrast

| The failure looks like | Contrast source | Command |
|---|---|---|
| Answers repeat themselves until the token limit (thinking or answer loops) | **automatic**: loop onset in the model's own answers | `probe.py collect --runs <base answers>` |
| Anything else you can write down (a phrase, a habit, a format, a language, a tone) | **matched pairs** you write | `probe.py collect-pairs --pairs pairs.jsonl` |
| Short answers that are wrong: yes/no, a letter, a caption (any task with a grader) | **automatic**: the model's own wrong answers against its right ones | `probe.py collect-decisions --runs <base answers> --graded <summary.json>` |
| Answers that differ from a gold text in style or content (captions, formats) | **automatic pairs**: the model's failing answer against the gold answer, same prompt and image | `make_pairs.py`, then `collect-pairs` |

Loops and decisions come from the model's own answers; every other behaviour needs pairs. A pair whose branches are
one word long ("Yes" against "No") carries no decision, since both start from the same state: use decisions.

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
| Image captions (built in) | COCO (`make_image_data.py --preset caption`) | decisions, or pairs from `make_pairs.py` | `--task caption` (CIDEr-D) |
| Is a text or query relevant to an image (built in) | COCO captions with hard negatives (`--preset relevance`) | decisions | `--task relevance` (yes/no, F1) |
| Object hallucination (built in) | POPE adversarial (`--preset pope`) | decisions | `--task pope` (yes/no, F1) |
| Your own image questions | `--task jsonl` rows with `image` | decisions, or pairs with `image` | `--grader yesno`, `exact`, `contains`, or your own |

The authors' sycophancy experiment (their Experiment 7) found the behaviour by comparing attention patterns, not by
an edit. Treat sycophancy edits as exploratory.

## 5. Image tasks

The model must read images itself (a vision-language checkpoint such as Qwen3.5, Qwen3-VL or Gemma 3). The edit
still changes only the language model's `down_proj` columns; the vision tower stays byte-identical.

- **Data.** `make_image_data.py` builds the presets. For your own, write `{"id", "image", "prompt", "answer"?,
  "group"?}` rows with `--task jsonl`; image paths are relative to the JSON file. Split by image, so no image appears
  in both practice and held-out.
- **Profile.** Image presets run with thinking off (512 new tokens), as is usual for captioning and yes/no
  questions. Turn thinking on with `--profile thinking --max-tokens 32768` only when the task needs reasoning (charts,
  counting); then loops become possible and `collect` applies too.
- **Scores.** Captions get CIDEr-D, a score per image, so pairing counts images that scored higher or lower. Yes/no
  tasks also report the yes-rate, precision, recall and F1, with per-class accuracy in `by_group`.
- **What an edit can do here.** It removes one habit: long, generic or hallucinated captions, or a bias towards "yes"
  or "no". It cannot teach the model to see something it does not. On yes/no tasks, a direction from wrong against
  right answers is often mostly the answer bias, so check the yes-rate and F1: a gain in accuracy with a flat F1 is a
  threshold shift (the smoke test in SKILL.md found exactly this).
- **Pairs for captions.** `make_pairs.py` pairs each below-median caption with the reference that agrees most with
  the other references. That contrast mixes style (length, wording) with content (what is described), so expect a
  noisier direction than for loops.
- **Clean windows.** Short answers from another task make good side-effect windows: for captions, pass a POPE run
  with `--task pope --data ... --runs ...`.

## 6. Out of scope

Do not use this skill to remove a model's safety behaviour: making it comply with harmful requests, stripping
refusals, or bypassing its safeguards. The rules in SKILL.md also exclude edits that only work by forcing a global
change (sharper or flatter outputs, always closing the thinking early).
