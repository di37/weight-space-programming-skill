# Weight-space programming skill

A [Claude Code](https://claude.com/claude-code) skill that makes a small open LLM better at a goal by **editing its
weights directly, with no training**. It follows Experiment 8 of
["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang,
Khattab, MIT CSAIL, 2026). There, a rank-1 edit to two MLP layers of Qwen3.5-0.8B removed its thinking loops and
raised MATH-500 from 7.9% to 48.1%.

The skill:

1. Finds the behaviour to remove, either automatically (answers that loop to the token limit) or from matched example
   pairs you write (a wanted and an unwanted continuation of the same prompt).
2. Reads the model's internals with forward passes only, and computes the direction separating unwanted from wanted
   text, plus the MLP `down_proj` columns that write it.
3. Writes a rank-1 edit, `W' = W − α·u(uᵀW)`, on those columns, into a **copy of the model on disk**.
4. Screens the candidates against a size-matched random control. Each edit must pass a measured practice check, and a
   failed check escalates through fixed steps.
5. Confirms the result with a paired test on held-out questions, and gives a clear verdict.

There are ready presets for **maths** (MATH-500) and **science** (MMLU-Pro physics, chemistry, biology,
engineering). Any other goal works from your own prompts, a grader, and matched pairs: output format, answer
language, verbosity, sycophancy, and so on. See `references/goals.md`.

## Install

Clone it into your Claude Code skills folder: user-wide (`~/.claude/skills/`) or per project (`.claude/skills/`).

```bash
git clone https://github.com/di37/weight-space-programming-skill ~/.claude/skills/weight-space-programming
```

Python packages (a virtual environment is recommended):

```bash
python -m pip install torch transformers safetensors numpy pandas pyarrow pytest huggingface_hub
```

Then add one generation engine for your machine:

| Machine | Engine |
|---|---|
| Linux, or Windows via WSL2, with an NVIDIA or AMD GPU | `vllm` |
| macOS on Apple Silicon | `mlx-lm` |
| Anything else | transformers (already installed; slow) |

`references/engines.md` covers each setup, batch sizing and a five-minute smoke test.

## Use

In Claude Code, ask for what you want. For example:

- "Make Qwen3.5-0.8B better at science without training."
- "Edit the weights so the model stops looping on maths problems."
- "Make this model answer in the language it's asked in. Here are some prompts."

Claude loads the skill and follows its fast path:

1. check the machine;
2. copy the model;
3. make practice and held-out splits;
4. run the baseline;
5. find the direction;
6. screen the candidate edits;
7. write the best one into the copy;
8. gate it on practice questions;
9. test it on held-out questions;
10. report.

The scripts in `scripts/` also run on their own; each has usage notes in its docstring.

## What to expect

No method can promise that an edit helps: some models have no single habit to switch off. The skill guarantees the
process instead:

- crashed runs restart with a smaller batch, and no finished answer is lost;
- no edit reaches the held-out test without passing a practice check;
- a failed check escalates to the next candidate, then to a second round, then to a stop that restores the copy;
- every run ends with a verdict and numbers.

## Results

All runs used Qwen3.5-0.8B with thinking on. Sampling was the paper's protocol: temperature 0.6, top_p 0.95,
top_k 20, presence penalty 1.5, and at most 32,768 new tokens (16,384 on practice runs). Every comparison is paired:
base and edited model answered the same questions.

### 1. The published maths edit, verified and reproduced

The authors' edited checkpoint differs from the base model in exactly 2 of 488 tensors (`layers.15` and
`layers.19` `mlp.down_proj`). Each change is rank 1 on 64 columns: 99.999% of the change lies in one direction, and
the removed fraction is 6.000. So the edit does more than subtract the loop direction: it flips it, so a column's
component c becomes −5c. Both tensors match the sha256 hashes in the authors' manifest.

| MATH-500 check | Base | Edited |
|---|---|---|
| 5 level-3 problems (fixed rule: the first five in file order) | 2/5, 3 of them looping to the cap | **3/5**, 2 at the cap |
| Paired | | 1 fixed, 0 broken |

On these 5, both models matched the authors' published per-problem outcomes on almost every problem. The sample is
far too small to measure the size of the gain.

### 2. A new edit for science (MMLU-Pro), made from scratch with the skill

**Data.** Questions came from MMLU-Pro physics, chemistry, biology and engineering, in seeded, disjoint splits:
128 practice questions and a fixed 30-question held-out test (8 physics, 8 chemistry, 7 biology, 7 engineering).

**Baseline diagnosis** (practice questions, 87 answered):

| | Base |
|---|---|
| Accuracy | 51.7% (physics 37.5, chemistry 44.8, biology 76.9) |
| Answers that hit the 16k cap | 29.9% (12 physics, 14 chemistry, 0 biology) |
| Accuracy on answers that finished | 74% (45 of 61) |
| Capped answers that were verbatim loops | 23 of 26 (onset between 2.7k and 14.8k tokens) |

When the model finishes, it is mostly right; the loops are what cost it the points.

**The loop direction.** It came from 17 loops: the 1,024 tokens just before each loop starts against 1,024 tokens
inside it. From layer 10 onwards the direction is 45–64% of the residual stream's norm, and individual loops align
with it (cosine 0.80–0.85 in layers 11–22). Inside a loop the model is nearly certain: −0.005 nats per token, against
−0.3 just before it.

**The candidate edits.** Every candidate makes the same kind of change, in three steps:

1. In each chosen layer, take the neurons whose output pushes hardest towards looping.
2. Flip that push along the loop direction. α sets how hard: α = 6 turns a push of +1 into −5, α = 3 into −2, and
   α = 1 only removes it.
3. Leave everything else in the model untouched.

The candidates differ only in which layers, how many neurons and how strong. The letters are just names:

| Name | Layers | Neurons edited per layer (of 3,584) | α | Weights changed |
|---|---|---|---|---|
| A | 15, 19 | 64 | 6 | 131,072 (0.016%), the same shape as the authors' maths edit |
| B | 15, 19 | 64 | 3 | 131,072 |
| C | 17, 19 | 64 | 6 | 131,072 |
| D | 11, 15, 19 | 64 | 6 | 196,608 |
| **E** | 15, 19 | **128** | 6 | 262,144 (0.03%) |
| **F** | 15, 19 | **256** | 6 | 524,288 (0.07%) |
| G | 15, 19 | all 3,584 | 1 (removal only) | 7,340,032 |
| H | 11, 15, 17, 19, 21 | 128 | 6 | 655,360 |
| I | 19 | 128 | 6 | 131,072 |
| R (control) | 15, 19 | 64 | a random direction, with the same size of change as A | 131,072 |

**E and F** are the two that matter below. **E** edits 128 neurons in each of layers 15 and 19. It was the best of the
first screen, so it was checked in generation first. **F** is the same edit with twice as many neurons (256 per layer):
a stronger dose of the same fix. F is the edit tested on the held-out questions, and the one the edited model holds
now.

**Screening.** This is the change in mean log-prob per token with each edit applied. It was measured on 6 held-back
loops, on their productive counterparts (the reasoning just before each loop) and on 16 normal answers. On loops, more
negative is better; elsewhere, near 0 is better.

| Name | Loops | Productive | Normal |
|---|---|---|---|
| A | −0.059 | −0.029 | −0.023 |
| B | −0.009 | −0.013 | −0.008 |
| C | −0.024 | −0.016 | −0.013 |
| D | −0.075 | −0.032 | −0.026 |
| **E** | −0.103 | −0.031 | −0.026 |
| **F** | **−0.139** | −0.031 | −0.028 |
| G | −0.002 | −0.005 | −0.002 |
| H | −0.102 | −0.022 | −0.027 |
| I | −0.039 | −0.016 | −0.015 |
| R (control) | −0.0001 | +0.0002 | −0.003 |

The random control R does nothing, so the effect is specific to the loop direction. G, which only removes the
direction, also does nothing: the edit has to flip it. F hits loops hardest with the same side effects as E.

**Practice check of E** (128 neurons per layer; 32 questions: the 26 where base hit the cap, plus 6 it got right):

| | Base | E |
|---|---|---|
| Accuracy | 18.8% | 25.0% |
| Hit the cap | 81.2% | 71.9% |
| Thinking closed | 18.8% | 31.2% |
| Base loops that now finished | | 3 of 26 |
| Paired | | 2 fixed, 0 broken |

E delayed the loops (median onset 6.9k → 8.9k tokens) but broke few of them. It would fail the skill's gate today.
So the next step was F, the stronger version.

**Held-out test of F** (256 neurons per layer; 30 questions never used to build or choose the edit; 32k cap):

| | Base | F |
|---|---|---|
| **Accuracy** | **40.0%** (12/30) | **50.0%** (15/30) |
| Physics / chemistry / biology / engineering | 25.0 / 37.5 / 71.4 / 28.6 | 37.5 / 25.0 / 85.7 / 57.1 |
| Lenient accuracy (also reads unclosed thinking) | 50.0% | 53.3% |
| Hit the 32k cap | 36.7% | 36.7% |
| Thinking closed | 63.3% | 86.7% |
| Median length | 18,002 tokens | 19,114 tokens |
| Paired | | **4 fixed, 1 broken** (sign test p = 0.38) |

F changes 524,288 weights, 0.07% of the model: 256 columns in each of two `down_proj` tensors, 13.3% relative change
per tensor.

**What this shows.** On the same 30 unseen questions the edited model got 3 more right. It fixed 4 questions where
base failed and broke 1. That is consistent with a real gain, but at 30 questions it is not statistically
significant.

The behaviour changed in a clear way: the edited model left its thinking far more often (86.7% against 63.3%). But
the loop often came back after it: 6 of the 7 capped answers that had closed their thinking then repeated themselves
in the final answer. So the edit moved the loop rather than removing it. The skill now handles exactly this case with
a second-round edit built from the edited model's own new loops.

### 3. Matched pairs, a mechanics check

With 9 toy pairs ("What a fantastic question, you clearly have a brilliant mind! …" against the same answer without
it), an edit shaped like A (layers 15 + 19, 64 neurons, α = 6) but built from the flattery direction made the unwanted opener 1.00 nats per token less likely on
held-back pairs, while the plain answer became slightly more likely (+0.05). The random control moved by +0.02. This
only shows the route works mechanically; it has not been evaluated in generation.

### Caveats

- The samples are small: 5 maths problems and 30 held-out science questions, with one answer per question.
- The practice check is partly in-sample: 17 of its 26 looping questions also built the direction. Only the held-out
  test is clean.
- The science runs above were generated before the skill's presence penalty was switched to vLLM's semantics. They
  also penalised prompt tokens, for base and edited alike, so the paired comparisons are fair, but the absolute
  numbers may differ slightly from the paper's protocol.
- For comparison, the authors' own maths edit took MMLU-Pro (all subjects, 1,000 questions) from 40.3% to 45.5% and
  cut its loop rate from 28.9% to 0.5%. The next experiment worth running is that edit against these same 30
  questions, and the second-round edit described above.

## Tests

```bash
python -m pytest scripts/tests -q
```

## Credits

- The method and the published edit: Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, "Sparks of
  Weight-Space Programming" (2026). Their logs and code are at
  [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).
- Model: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0).
- Benchmarks: [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) and
  [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

The skill was written with Claude Code. It is not affiliated with the paper's authors.

## Licence

MIT. See `LICENSE`.
