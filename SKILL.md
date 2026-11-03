---
name: weight-space-programming
description: Make a small open LLM or vision-language model (e.g. Qwen3.5-0.8B) better at a goal by editing its weights directly, with no training ("weight-space programming"). Presets for maths (MATH-500), science (MMLU-Pro), image captions (COCO), image-text relevance and object questions (POPE); any other benchmark or behaviour works from your own prompts (with images if needed) and matched pairs. Finds the direction separating the unwanted behaviour from the wanted one, edits the MLP down_proj columns that write it in a copy of the model on disk, screens against a random control, gates each edit on a practice check, escalates when one fails, and confirms with a paired held-out test. Linux, Windows (WSL2) and macOS; NVIDIA, AMD or Apple GPUs. Use when the user says "make this model better at X without training", "edit the weights to fix X", "stop the model doing X", "improve its captions", "weight-space programming" or "WSP". Not for fine-tuning, LoRA, RL, prompting, API-only models, or removing safety behaviour.
---

# Weight-space programming

Improve a small open model at a goal by editing its weights directly, with no optimiser. This follows Experiment 8 of
"Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2 Nov 2026): Qwen3.5-0.8B's thinking
loops were traced to one residual direction written by the MLPs of layers 15 and 19, and a rank-1 edit to 64
`down_proj` columns per layer raised MATH-500 from 7.9% to 48.1%.

**Follow the fast path below in order.** Every script is bundled in `scripts/` and tested. Do not redesign the
pipeline or write new tooling unless a step fails. Image goals (vision-language models such as Qwen3.5, which reads
images natively) use the same steps; each step says what differs for them.

**What the skill guarantees, and what it cannot.** No method can promise that an edit helps: some models have no
single habit to switch off. The skill guarantees the process instead:
- no lost work: crashes restart with a smaller batch and resume;
- no edit reaches the held-out test without passing a measured practice check;
- a failed check escalates through fixed steps instead of stopping;
- every run ends with a clear verdict and numbers, and the copy is restored if nothing works.

References, for detail only:
- `references/goals.md`: other goals; how to write matched pairs; graders; recipes.
- `references/engines.md`: which engine on which OS and GPU, installation, sizing, pitfalls, smoke test.
- `references/evaluation.md`: splits, sampling protocol, MMLU-Pro specifics, paired statistics, report template.
- `references/method.md`: the maths of each step and the authors' exact recipe.

## Fast path

Set four placeholders once (PowerShell: `$env:SKILL = "..."`, then `$env:SKILL`):

```bash
SKILL=<this skill's folder>  BASE=<base model folder>  COPY=<base model folder>-<goal>  WORK=<results folder>
```

Run every generation job in the background (Claude Code: the Bash tool's `run_in_background`, not a shell `&`) and
wait for its completion notice instead of polling. Generation dominates the time, so tell the user the budget up
front: 1.5–3 hours on a consumer GPU for the text presets, much less with vLLM on a datacenter GPU. Image presets
answer in a few dozen tokens, so a whole pass takes 15–30 minutes.

**0. Ask only what you need** (one question). Ask for the goal. If it is maths, science, image captions, whether a
text or query is relevant to an image, or object questions about images, use the preset: no more questions. Otherwise
also ask for, or help write:
- prompts that show the failure, as `{"id", "prompt", "answer"?, "group"?, "image"?}` JSON lines (`image`: a file
  path, relative to the JSON file);
- how to score them (`references/goals.md` section 3);
- for non-loop behaviours, 12+ matched pairs (section 2).

Confirm "edit weights, no training".

**1. Check the machine** (2 min): `python $SKILL/scripts/device.py --model $BASE --jobs 2`. On a new machine, also run
the smoke test in `references/engines.md` section 6.

**2. Copy the model** (1 min). Every edit goes into this copy's files; every evaluation loads the copy.

```bash
python $SKILL/scripts/copy_model.py --base $BASE --copy $COPY
```

**3. Splits** (seconds).
- **Science preset:** `python $SKILL/scripts/make_splits.py --task mmlu --source <MMLU-Pro test parquet> --out $WORK/splits`.
- **Maths preset:** the same with `--task math --source <MATH-500 test.jsonl> --practice 25 --heldout 75`.
- **Image presets:** `python $SKILL/scripts/make_image_data.py --preset caption --out $WORK/splits` (or `relevance`,
  or `pope`). It reads only the rows it needs from the Hugging Face Hub and saves the images beside the splits.
- **Own goal:** split your prompt file into `practice.jsonl` and `heldout.jsonl` by a fixed rule, plus a
  `test_ids.txt` of about 30 held-out ids, all chosen before any answer is seen.

Below, `T` is the task (`mmlu`, `math`, `caption`, `relevance`, `pope` or `jsonl`) and `G` is the grader options
(none for the presets; see `references/goals.md` otherwise).

**4. Start two base runs at once:**

```bash
python $SKILL/scripts/run_eval.py --task $T --model $BASE --label base --data $WORK/splits/practice.jsonl --out $WORK/practice/base.jsonl --max-tokens 16384 --jobs 2
```

```bash
python $SKILL/scripts/run_eval.py --task $T --model $BASE --label base --data $WORK/splits/heldout.jsonl --ids-file $WORK/splits/test_ids.txt --out $WORK/test/base.jsonl --jobs 2
```

Use the model card's sampling for other models (`--profile`, or the override flags; `references/engines.md`). The
held-out run does not depend on the edit. **Image presets:** leave out `--max-tokens` and `--jobs`; they default to
512 tokens and the non-thinking profile, and run one after the other in minutes.

**5. Get the contrast.**
- **Loops (presets, or any goal where answers hit the cap):** loops appear only when an answer hits the cap. Count
  them with `grep -c '"length"' $WORK/practice/base.jsonl`; at 15 or more, stop the practice run.
- **Other behaviours:** you need the pairs, and the practice run only needs to finish to give the baseline.
- **Short answers (yes/no, a letter, a caption):** the practice run only needs to finish; the contrast is its own
  wrong (or below-median) answers against its right ones, read in step 6.

Then grade:

```bash
python $SKILL/scripts/grade.py --task $T $G $WORK/practice/base.jsonl --out $WORK/practice/summary.json
```

**Go/no-go.** Estimate the room for gain: the share of answers showing the behaviour, times the score on answers that
don't. If it is under about 10%, tell the user to expect little before going on.

**6. Read the direction, build candidates, screen** (minutes, forward passes only). Loops:

```bash
python $SKILL/scripts/probe.py collect --task $T --model $BASE --data $WORK/splits/practice.jsonl --runs $WORK/practice/base.jsonl --out $WORK/probes
```

Short answers, including every image preset:

```bash
python $SKILL/scripts/probe.py collect-decisions --task $T --model $BASE --data $WORK/splits/practice.jsonl --runs $WORK/practice/base.jsonl --graded $WORK/practice/summary.json --out $WORK/probes
```

Captions can also use pairs made from the run (the model's weaker captions against the references); try them when
the decisions route fails the screen:

```bash
python $SKILL/scripts/make_pairs.py --task $T --data $WORK/splits/practice.jsonl --runs $WORK/practice/base.jsonl --graded $WORK/practice/summary.json --out $WORK/pairs.jsonl
```

Pairs (yours or from `make_pairs.py`; add `--runs`, `--data` and `--task` for clean side-effect windows):

```bash
python $SKILL/scripts/probe.py collect-pairs --model $BASE --pairs pairs.jsonl --thinking on --out $WORK/probes
```

Then build the first-round candidates, the control, and screen:

```bash
for spec in "15 19:128:6" "15 19:256:6" "15 19:64:6" "11 15 17 19 21:128:6"; do IFS=: read L K A <<< "$spec"; python $SKILL/scripts/probe.py build --probes $WORK/probes --model $BASE --layers $L --k $K --alpha $A --out "$WORK/edits/L${L// /-}_k${K}_a${A}.safetensors"; done
```

```bash
python $SKILL/scripts/probe.py build --probes $WORK/probes --model $BASE --layers 15 19 --k 128 --alpha 6 --control random --out $WORK/edits/random.safetensors
```

```bash
python $SKILL/scripts/probe.py screen --probes $WORK/probes --model $BASE --edits $WORK/edits/*.safetensors
```

On Windows PowerShell, run each `build` line on its own. The layers above suit a 24-layer model. For others, use the
layers at about 60% and 80% of the depth, a five-layer spread, and the deeper one alone.

Rank the candidates by the drop on `unwanted`, keeping only those whose `wanted` and `clean` drops are under a third
of it. The random control must stay near 0; if it moves as much as the best candidate, the direction is not specific,
so go to the stop step.

**7. Write the best candidate into the copy and verify** (1 min). Never evaluate an in-memory edit.

```bash
python $SKILL/scripts/apply_edit.py --base $BASE --copy $COPY --edit $WORK/edits/<candidate>.safetensors
```

```bash
python $SKILL/scripts/diff_edit.py --base $BASE --edited $COPY --out $WORK/edit_report.json
```

The diff must show only the intended `down_proj` tensors, `changed_columns` = k, a rank-1 share of about 1 and a
median α of about the chosen α.

**8. Practice check: the gate** (15–25 min, alongside the base held-out run). Run the copy on every practice question
base failed (cut, wrong or flagged), plus 6 it handled well:

```bash
python -c "import json; g=json.load(open('$WORK/practice/summary.json'))['graded']['base']; bad=[x['id'] for x in g if x['cut'] or x['correct'] is False or x['flagged']]; ok=[x['id'] for x in g if x['id'] not in bad]; open('$WORK/check_ids.txt','w').write('\n'.join(bad+ok[::max(1,len(ok)//6)][:6])+'\n')"
```

```bash
python $SKILL/scripts/run_eval.py --task $T --model $COPY --label copy --data $WORK/splits/practice.jsonl --ids-file $WORK/check_ids.txt --out $WORK/practice/copy.jsonl --max-tokens 16384 --jobs 2
```

```bash
python $SKILL/scripts/grade.py --task $T $G $WORK/practice/base.jsonl $WORK/practice/copy.jsonl
```

**The gate passes** only if all of these hold:
- fixed ≥ max(3, 20% of the base failures checked),
- broke ≤ fixed ÷ 4,
- for loop goals, the cut share falls by at least a quarter;
- for yes/no goals, the F1 rises as well as the accuracy. An edit that only moves the yes-rate trades one class for
  the other.

For captions, "fixed" and "broke" count images whose CIDEr-D rose or fell (`grade.py` pairs per-image scores).

Pass: go to step 10. Fail: step 9.

**9. Escalate in this order; each rung is one more practice check.**
1. **The next-ranked candidate.** Restore the copy (`copy_model.py --base $BASE --copy $COPY --restore`), apply it,
   and repeat steps 7–8. Do this for at most two more candidates.
2. **Round 2, when the behaviour moved rather than vanished.** For example, the thinking now closes but the loop
   restarts after `</think>`, or a new unwanted pattern appears in the copy's answers. Treat the edited copy as the
   new base:
   - make `COPY2` with `copy_model.py --base $COPY --copy $COPY2`;
   - run `probe.py collect` (or `collect-pairs`) with `--model $COPY --runs $WORK/practice/copy.jsonl`;
   - build and screen with `--model $COPY`;
   - apply with `--base $COPY --copy $COPY2`;
   - check `COPY2` as in step 8.
3. **Stop.** No reliable gain after three checks (or after the time budget the user set). Restore the copy, and report
   the verdict "no reliable gain" with every candidate's numbers. For Qwen3.5-0.8B, offer the published maths edit as
   a comparison (it cut MMLU-Pro's cut share from 28.9% to 0.5% in the authors' logs; see the end of this file). Use
   it only if the user agrees.

Never use held-out results to choose; only practice checks decide.

**10. Held-out test from the copy** (30–60 min). Start it when the base held-out run is past its longest answers:

```bash
python $SKILL/scripts/run_eval.py --task $T --model $COPY --label copy --data $WORK/splits/heldout.jsonl --ids-file $WORK/splits/test_ids.txt --out $WORK/test/copy.jsonl --jobs 2
```

```bash
python $SKILL/scripts/grade.py --task $T $G $WORK/test/base.jsonl $WORK/test/copy.jsonl --out $WORK/test/summary.json
```

Give the verdict from the paired result:
- **"Clear gain"** when fixed > broke and sign-test p < 0.05.
- **"Consistent with a gain, not proven"** when fixed > broke but p ≥ 0.05; recommend a larger test.
- **"No gain"** when fixed ≤ broke; recommend restoring the copy.

**11. Report** with the template in `references/evaluation.md`:
- the held-out score before and after, and the paired fixed/broke counts;
- the behaviour rates (cut, flagged, thinking closed);
- every candidate with its screen and check numbers, and the random control;
- the weights changed;
- the caveat: switching a habit off is not new ability.

### Mistakes that cost hours before (do not repeat)

- **Too large a batch, or two long jobs at their peaks.** An out-of-memory crash kills the in-flight answers, which are
  the loops. Use `--batch auto --jobs 2`, and never start a second 32k job while the first is on its longest answers.
  The supervisor restarts crashed runs with half the batch.
- **Waiting for whole runs.** About 15 loops are enough.
- **In-memory edits for evaluation.** Only `probe.py screen` applies edits in memory, and only for teacher-forced reads.
- **Trusting the screen alone.** A screen drop of −0.10 on loops broke only 3 of 26 loops in generation, so always
  run the gate.
- **Grading a partial run, or picking test questions by which finished first.**
- **Stopping jobs with `pkill -f <text>`,** which also kills watchers whose command line contains that text.
- **Same-prompt pairs for one-word answers.** Both branches start from the same state, so the pair carries no
  decision. Use `collect-decisions`.
- **Reading a yes/no gain from accuracy alone.** Check the yes-rate and per-class accuracy (`by_group`): a smoke test
  edit raised POPE accuracy only by saying "no" more often.

## Rules (as in the post)

Keep these and say in the report that you did.

- **No training.** No backward passes, autograd, optimisers, or iterative fitting of weights to a loss. Nudging α or
  weights until a metric improves counts as fitting. Picking among the fixed candidate list above does not.
- **Forward passes for measurement are allowed:** generation, teacher-forced log-probs, hidden states and MLP
  activations, and statistics of those or of the weights (means, differences, SVD).
- **Out of scope:**
  - the embeddings, the (tied) output head and the final norm (`weights.py` refuses them);
  - multiplying a whole tensor by a constant;
  - global sharpening or flattening;
  - forcing the thinking to close early regardless of the problem;
  - removing safety behaviour: refusals, safeguards, harmful-request compliance.
- **Only weight values change.** Keep names, shapes, dtype, config, tokenizer, chat template and decoding settings.
- **Split discipline.** Disjoint practice and held-out sets, fixed before any edit. Look at held-out results once,
  after the edit is frozen.
- **Disclosed selection.** The report lists every candidate tried and the metric that chose the winner.

## Why each step is the way it is

- **Contrast:** the direction is the mean difference, at each layer's output, between unwanted and wanted text that is
  otherwise matched.
  - For loops, `loops.py` takes the 1,024 tokens just before the loop onset and 1,024 tokens inside it: same question,
    same answer, nearly the same position. The onset is the first point where 90% of the next 512 tokens repeat an
    earlier 16-gram.
  - For other behaviours, the pairs give the match.
  - For short answers, the classes are the model's own wrong and right answers, so they differ in input as well as
    in behaviour; enough items average the inputs out. The states are read one position before each answer token,
    where the answer is decided.
- **Writer columns:** score_j = (mean unwanted − mean wanted activation of neuron j) × (u · W_down[:, j]); take the top
  k. The static write u·W alone is not enough: only 21–22 of the published 64 columns per layer are in its top 64.
- **The edit:** W'[:, S] = W[:, S] − α u (uᵀ W[:, S]), computed in float32 from the model being edited. α = 1 removes
  the component, which did nothing for loops; α = 6 flips it to −5c, so those neurons write against the behaviour.
- **Screening:** the change in mean log-prob with the edit applied. Unwanted text should become less likely; wanted
  and clean text should stay. A norm-matched random direction on the same columns is the control.
- **The gate:** screen numbers predict generation only loosely (see the results below), so a short generation check on
  practice questions decides before the held-out test.
- **Saving:** `weights.py` rewrites only the touched safetensors shards from the raw files. Inference engines rewrite
  or drop tensors on load, so never save weights from one.
- **Protocol:** thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5 on generated tokens only,
  32,768 new tokens (`--profile thinking`). All engines match vLLM's semantics.
- **Pairing:** engines and batch sizes change the random numbers, so pair by question within one comparison. Never
  compare scores measured on different questions or with different engines.

## Results so far (Qwen3.5-0.8B)

- **Published maths edit, verified:** only `layers.15/19.mlp.down_proj` change, rank 1 on 64 columns each, α = 6.000,
  and the hashes match the authors' manifest.
- **MMLU-Pro science, 3 Nov 2026** (direction from 17 real loops):
  - Base practice: 51.7% on 87 answers, 29.9% cut at 16k (physics and chemistry loop; biology never).
  - Screen (unwanted / wanted / clean, nats per token). k is the number of neurons edited per layer, and the names
    match the README:

    | Name | Edit | Unwanted | Wanted | Clean |
    |---|---|---|---|---|
    | A | layers 15+19, k 64, α 6 | −0.059 | −0.029 | −0.023 |
    | E | layers 15+19, k 128, α 6 | −0.103 | −0.031 | −0.026 |
    | F | layers 15+19, k 256, α 6 (E with twice the neurons) | −0.139 | −0.031 | −0.028 |
    | H | 5 layers (11, 15, 17, 19, 21), k 128, α 6 | −0.102 | −0.022 | −0.027 |
    | I | layer 19 alone, k 128, α 6 | −0.039 | −0.016 | −0.015 |
    | G | layers 15+19, all 3,584 neurons, α 1 (removal only) | −0.002 | −0.005 | −0.002 |
    | R | random direction, same size of change as A (control) | −0.0001 | +0.0002 | −0.003 |

  - E failed the practice check: 3 of 26 loops finished, 2 fixed, 0 broken. Under the gate above it would have
    escalated to F. F was taken straight to the held-out test.
  - **Held-out test (30 questions, paired): base 40.0% (12/30) against F 50.0% (15/30); 4 fixed, 1 broken, p = 0.38,
    so "consistent with a gain, not proven".**
  - F moved the loop instead of removing it: thinking closed in 86.7% of answers against 63.3%, but 6 of the 7 capped
    copy answers that had closed their thinking then looped in the final answer. That is exactly the case for round 2
    of step 9.
- **Matched pairs, mechanics check:** on 9 toy pairs ("flattering opener" against a direct answer), an L15+19 k64 α6
  edit lowered the unwanted continuation by 1.00 nats per token, with the wanted one +0.05 and the random control +0.02.
  This only shows the mechanics; it has not been evaluated in generation.

- **Image tasks, smoke test, 4 Nov 2026** (Qwen3.5-0.8B, non-thinking, mlx-vlm; small splits, so mechanics only):
  - Base practice: POPE adversarial 84.4% of 90 (yes-rate 47.8%); COCO captions CIDEr-D 0.577 on 60, median 16 words.
  - Screen: no candidate passed for captions (pairs or decisions; the random control moved as much). For POPE
    decisions, the best (L15+19, k 256) dropped wrong answers by 0.68 and right ones by 0.30 nats, against 0.20 and
    0.09 for the random control.
  - POPE gate on 20 practice questions: 30% → 50% (4 fixed, 0 broke), but the yes-rate fell from 45% to 25%. Held-out
    30: 80.0% → 83.3% (3 fixed, 2 broke, p = 1.0), F1 80 → 80, "no" right 80% → 100%, "yes" 80% → 67%: a threshold
    shift, not better discrimination. The F1 rule in the gate now catches this.

## Reproducing the published edit

Download `exp8_math/runs/p50c/edit/` (`edit.npz`, `manifest.json`) and `exp8_math/code/{apply_edit.py,st_numpy.py}`
from github.com/RikiyaT/lm_understand-logs into their own folder, and run them in isolation (`python -I`) against a
copy of `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`. The script checks every sha256.
Then check the result with `diff_edit.py` and evaluate it like any candidate (step 8, then step 10).
