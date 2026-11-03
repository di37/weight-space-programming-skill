# Method: the maths of each step, and the authors' recipe

Sources: the post (rikiyat.github.io/blog/weight-space-programming) and the logs in
github.com/RikiyaT/lm_understand-logs: `exp8_math/README.md`, `runs/p50c/work_notes/{experiment_history.md,
final_report.md}`, `runs/p50c/tool_calls.jsonl` (the code Astra actually ran), and `gemma_followup/README.md`.

## Contents
1. Notation
2. The authors' p50c run, step by step
3. Writing matched contexts
4. The direction u
5. Choosing layers
6. Writer-column scores
7. The edit, and what α does
8. Screening: branch log-probs, premature close, random control
9. Tail repetition
10. What did not work, and the null result

## 1. Notation

Qwen3.5-0.8B (text part) has 24 layers, width d = 1,024 and MLP width m = 3,584. Layers 3, 7, 11, 15, 19 and 23 are
full-attention layers; the other 18 are Gated DeltaNet linear-attention layers. The input embedding and the output
layer are tied.

| Symbol | Meaning |
|---|---|
| `W_down^l` | `model.language_model.layers.{l}.mlp.down_proj.weight`, stored as [d, m] = [1024, 3584]. Column j is what MLP neuron j writes into the residual stream. |
| `a^l` | the input to `down_proj` at layer l: `silu(gate_proj(x)) * up_proj(x)`, with `x = post_attention_layernorm(h)`. |
| `h_k` | the residual stream after k layers. `h_0` is the embeddings and `h_{l+1}` is the output of layer l. This is the authors' "hidden state index". |
| "rep", "prod" | the repetitive and productive member of a matched pair. |
| T | the number of final tokens averaged over (the authors used 24–36). |

## 2. The authors' p50c run, step by step

Budget: 50 forward passes. Practice problems came from GSM8K train and MATH train. Every generation used the
evaluation protocol.

| Stage | What was done | Result |
|---|---|---|
| Diagnose (pass 1) | Base model on 96 practice problems | 11/96 correct; 85 cut at 32,768 tokens; cut tails about 95% duplicate 8-word sequences; no finished wrong answers |
| Direction (passes 2–3) | Teacher-forced read of 4 hand-written matched pairs. The first try went through the chat template and picked up an empty think block; it was redone with raw text. | A repetition direction at every depth; cosine about 0.7 with the generation contrast (mean activations of failing vs correct answers) |
| One neuron (pass 4) | Zero column 11 of L19 `down_proj`, the top-scoring writer | +6.25 pp, SE 4.35 (n = 32): weak |
| One layer (pass 5) | Rank-1 removal, α = 1, L19 only | +6.25 pp, SE 4.35: weak |
| Two layers, α = 1 | L15 + L19 | Fresh 128 problems: 8.6% → 32.8%, +24.2 pp, SE 4.0, 32 rescues / 1 loss |
| Dose | α = 2 (fresh 128 problems × 2); α = 3 (exploratory 128 × 2) | +34.8 pp, SE 3.5; +39.8 pp, SE 3.8. Whitespace-tail degeneration appears at α = 3 (12/256). |
| Alternatives | Halve attention heads L15h7 and L19h5 (high induction scores) | Less selective in teacher-forced reads; dropped |
| Control | Norm-matched orthogonal random direction | +1.6 pp, SE 3.5; repetitive log-prob +0.010; dropped |
| Dense dose (teacher-forced, 6 fresh pairs) | full α = 1 / 2 / 4 | Repetitive −0.024 / −0.049 / −0.104 nats/token; productive +0.002 / +0.003 / +0.005 |
| Layer isolation (teacher-forced) | L15 alone at α = 4; L19 alone at α = 4 | Repetitive −0.081 and −0.034: both sites contribute |
| Sparsify | Top-64 writer columns per layer, α = 4, at a 16k cap | +26.6 pp, the same as the full direction at α = 4 (+26.6), for about half the weight change |
| Final candidate | Top-64, α = 6 | Teacher-forced: repetitive −0.103, productive −0.009. Generation at 32k: 7/64 → 35/64; cut 89% → 8% |
| Freeze, then confirm | Hashes recorded; 1,152 fresh problems in locked batches of 64 | 10.6% → 60.4%, +49.8 pp, SE 1.5, 579 rescues / 5 losses; median tokens of correct answers 9,172 |
| Held-out (after the run) | MATH-500, 4 samples per problem | 7.9% → 48.1%; cut 93.0% → 33.1% |

The top-64 columns carried about 91% (L15) and 98% (L19) of the repetitive-minus-productive writing contrast on the
transfer probe, while accounting for only 50% and 40% of the full direction's weight change.

## 3. Writing matched contexts

Each pair should be identical up to a branch point and differ only in behaviour.

- **Prompt.** Use the exact evaluation prompt as a raw string, the rendered generation prompt ending in `<think>\n`.
- **Shared opening.** One to three sentences of correct reasoning, the same in both texts.
- **Productive branch.** New, correct steps: a sub-result, a check, the next quantity. No repetition.
- **Repetitive branch.** A short doubt-and-restate sentence repeated 3–7 times. Copy the shape of the base model's real
  loops: read some cut tails first.
- **Lengths** within about 10% of each other, so the last-T-token windows compare like with like.
- **No `</think>` and no final answer** in either text.
- **Bookkeeping.** For each text, record the token index where the branch starts and the total length. Teacher-forced
  comparisons then use exactly those tokens.

An illustration of the shape, for a science target:

```
prompt:      <rendered generation prompt for: "A block slides from rest down a frictionless incline of height 5 m.
             What is its speed at the bottom? (g = 9.8 m/s^2)" + instruction>
shared:      Energy is conserved, so m g h = (1/2) m v^2 and v = sqrt(2 g h).
productive:  Then 2 g h = 2 * 9.8 * 5 = 98 and sqrt(98) is about 9.90. The mass cancels, so it is not needed.
             The units are sqrt(m/s^2 * m) = m/s, so v is about 9.9 m/s.
repetitive:  But does the mass matter? No, the mass cancels. But does the mass matter? No, the mass cancels.
             But does the mass matter? No, the mass cancels. But does the mass matter? No, the mass cancels.
```

The authors used 4 pairs for the direction and the column selection, then 6 fresh pairs (plus 2 premature-close texts)
for transfer and screening. Never pick columns on the texts you screen with.

## 4. The direction u

```python
import numpy as np

def tail_mean(x, t):
    """x: [tokens, dim] for one text, already cut to its true length (no padding)."""
    return x[-t:].mean(0)

def failure_direction(resid, pairs, layer, t=32):
    """resid[i][k]: residual after k layers for text i; pairs: [(productive_i, repetitive_i), ...].
    Returns the unit vector pointing from productive towards repetitive at the output of `layer`."""
    diffs = [tail_mean(resid[r][layer + 1], t) - tail_mean(resid[p][layer + 1], t) for p, r in pairs]
    d = np.mean(diffs, axis=0)
    return d / np.linalg.norm(d)
```

Astra used the last 36 tokens for the direction and the last 24 for the column scores. Check that the sign of u agrees
on held-back pairs. The contrast's norm grows with depth (0.11 at `h_0`, 1.5 at `h_16`, 2.2 at `h_20`). The last
residual `h_24` is dominated by next-token prediction: its cosine with the generation contrast fell to 0.40.

## 5. Choosing layers

For each layer, split the growth of the contrast along u into an MLP part and an attention part:

```python
def mlp_write_along_u(w_down, u, act_rep, act_prod, t=32):
    """How much layer l's MLP writes along u, repetitive minus productive (one pair; average over pairs)."""
    return float(u @ (w_down @ (act_rep[-t:].mean(0) - act_prod[-t:].mean(0))))
```

Block growth is `u · (Δh_{l+1} − Δh_l)`, where Δh is the rep − prod difference of tail means. The attention part is
that growth minus the MLP part. Take the layers whose MLPs write the most along u, then confirm each one causally (edit
that layer alone and look for a dose response). Correlation with correct answers is not enough: correct answers come
from easier and shorter problems.

## 6. Writer-column scores

```python
def writer_scores(w_down, u, act, pairs, t=24):
    """w_down: [d, m]; act[i]: [tokens, m] down_proj input for text i. High score = the column writes along u, and
    more so in repetitive contexts than in productive ones."""
    write = u @ w_down                                        # [m] static write of each column along u
    act_diff = np.mean([act[r][-t:].mean(0) - act[p][-t:].mean(0) for p, r in pairs], axis=0)
    return write * act_diff

def top_columns(scores, k=64):
    return np.argsort(-scores)[:k]
```

This is the formula in Astra's code, `proj * (means[1::2] - means[::2]).mean(0)`, with productive texts at even
indices. Check transfer on fresh pairs with `scipy.stats.spearmanr` between the two score vectors and the overlap of
their top-k sets. Also report the share of the positive score mass the top k captures.

## 7. The edit, and what α does

```python
def rank1_edit(w0, u, cols, alpha):
    """W'[:, S] = W[:, S] - alpha * u (u^T W[:, S]). Returns a new float64 array; w0 is not modified."""
    w0 = np.asarray(w0, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    delta = np.zeros_like(w0)
    delta[:, cols] = -alpha * np.outer(u, u @ w0[:, cols])
    return w0 + delta
```

- For an edited column, `u·w'_j = (1 − α) u·w_j`. The part orthogonal to u is unchanged.
- α = 1 projects the component out. α = 2 reflects it. α = 6 makes it −5 times the original, so the columns push
  against the loop whenever they fire.
- ‖ΔW‖_F = α · ‖uᵀ W[:, S]‖. Report the relative change ‖ΔW‖_F / ‖W‖_F per tensor. It was 10.6% (L15) and 8.1% (L19)
  for the published edit, and 3.5% / 3.4% for the first full-direction α = 1 edit.
- Rebuild every candidate from the original weights. Cast to bf16 only when you save. Under 1% of the touched entries
  round back to their original bf16 value (130,188 of 131,072 changed in p50c).
- The written formula is `W = W0 - 6*outer(u, mask64*(u @ W0))`. It is the same as the function above with `cols` set
  to the 64 selected columns.

## 8. Screening

**Branch log-prob change.** Run the same token ids through base and candidate. Token-level log-probs are offset by
one: `lp[k] = log p(token k+1 | tokens ≤ k)`.

```python
def branch_delta(lp_edit, lp_base, start, length):
    """Mean change in log-prob over the branch tokens start..length-1 of one text."""
    return float((lp_edit - lp_base)[start - 1:length - 1].mean())
```

Average over the repetitive texts and over the productive texts separately. Report the last-24-token change too, since
the loop strengthens as it repeats. A good candidate drops repetitive tokens by about 0.1 nats/token and leaves
productive tokens within about 0.01.

**Premature close.** Take the prompt plus a short, unfinished correct opening, then `\n</think>\n\n`. Read the
log-prob of the `</think>` token. It should stay very unlikely: the authors saw about −19.8 for base and −19.4 for the
final edit. A rise means the edit is cutting thinking short instead of removing loops.

**Norm-matched random control.**

```python
def random_control(w0, u, cols, alpha, seed=0):
    """A direction orthogonal to u, applied to the same columns, with the same Frobenius change as the real edit."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(u.shape)
    r -= (r @ u) * u
    r /= np.linalg.norm(r)
    w0 = np.asarray(w0, dtype=np.float64)
    beta = alpha * np.linalg.norm(u @ w0[:, cols]) / np.linalg.norm(r @ w0[:, cols])
    return rank1_edit(w0, r, cols, beta)
```

Run it through the same teacher-forced probe, and once through a paired generation. If the control helps as much as
the real edit, the gain comes from perturbation, not from the mechanism.

**Paired generation screen.** Use 64–128 screening problems at the full token budget. A lower cut share does not by
itself mean higher accuracy: judge candidates by accuracy, and use the cut share as the explanation.

## 9. Tail repetition

The authors' exact definition is not published. This is a reasonable stand-in, not yet tested. Check that it gives
values near theirs: about 0.96 for base answers cut at the limit and about 0.03 for finished answers.

```python
def tail_dup_share(text, tail_words=400, n=8):
    """Share of the n-word sequences in the last `tail_words` words that already occurred earlier in the text."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    start = max(0, len(grams) - tail_words)
    seen = set(grams[:start])
    hits = 0
    for g in grams[start:]:
        hits += g in seen
        seen.add(g)
    return hits / max(1, len(grams) - start)
```

Report it separately for cut and finished answers. Also count whitespace-only tails: the edit's alternative
degeneration rose from 0 to 51 of 1,152 answers in p50c.

## 10. What did not work, and the null result

- **p10c (10 passes).** The direction was removed from 30 matrices (attention and MLP) in layers 8–22 at α = 1.5. That
  is broad, and small in each tensor. MATH-500 reached only 12.3%, with 87% still cut.
- **p25c (25 passes).** Head 7 of layer 15 was switched off (256 output columns zeroed). It was plausible from attention
  patterns, but MATH-500 reached 14.1%, with 86% still cut. Heads that look like copy heads are not necessarily what
  drives the loop.
- **p50c (50 passes)** reached 48.1%. The difference came from dense teacher-forced dose tests, doses large enough to
  matter, sparsifying to the columns that actually write the direction, and confirmation on fresh problems.
- **Gemma 3 270M.** Its edits scaled single neurons or heads (×0.5, ×1.1, ×1.5, ×2). The largest move on MATH-500 was
  6.8 → 7.7 and none was established. With no failure mechanism to remove, small targeted edits do nothing measurable.
  Submit the best candidate anyway, and report honestly that the gain is not established.
- Edited models can fail in new ways. Read the regressions. In p50c's first loss, base had boxed the right answer before
  being cut, and the edited model finished with a wrong one.
