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

Results so far, on Qwen3.5-0.8B and MMLU-Pro science: a held-out paired test of 30 questions went from 40% to 50%
(4 fixed, 1 broken). That isn't statistically significant at this size, and the edit partly moved the loops into the
final answer instead of removing them. The full numbers are in `SKILL.md`.

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
