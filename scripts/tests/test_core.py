"""Unit tests for the portable scripts. No model or GPU needed.

    python -m pytest scripts/tests -q
"""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device import recommended_batch  # noqa: E402
from grade import extract_letter, grade, last_boxed, make_grader, paired, sign_test  # noqa: E402
from loops import GAP_BEFORE, SETTLE_AFTER, loop_onset, matched_windows  # noqa: E402
from make_splits import quotas  # noqa: E402
from modeling import OutputOnlyPresencePenalty  # noqa: E402
from probe import direction_and_scores, edit_layer, pair_items  # noqa: E402
from run_eval import sampling_from, without_batch  # noqa: E402

ROWS, COLS = 64, 96


def varied(n, seed=0):
    rng = random.Random(seed)
    return [rng.randrange(50_000) for _ in range(n)]


def unit(v):
    return v / v.norm()


def row(text, answer="C", finish="stop", thinking=True):
    return {"id": "1", "sample": 0, "group": "g", "answer": answer, "finish_reason": finish, "n_tokens": 9,
            "text": text, "thinking": thinking}


class CharTokenizer:
    """A stand-in tokenizer: one token per character, and a fixed chat wrapper."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "<u>" + messages[0]["content"] + "<a>" + ("<t>" if enable_thinking else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_loop_onset_and_windows():
    assert loop_onset(varied(4000)) is None
    tokens = varied(3000) + varied(40, seed=1) * 100
    assert 3000 <= loop_onset(tokens) <= 3000 + 40 + 16
    prod, rep = matched_windows(answer_start=100, onset=2000, answer_len=5000, window=1000)
    assert prod == (100 + 2000 - GAP_BEFORE - 1000, 100 + 2000 - GAP_BEFORE)
    assert rep == (100 + 2000 + SETTLE_AFTER, 100 + 2000 + SETTLE_AFTER + 1000)
    assert matched_windows(answer_start=0, onset=500, answer_len=5000, window=1000) is None


def test_builtin_answer_reading():
    assert extract_letter("so the answer is (C).") == "C"
    assert extract_letter("Final Answer: B") == "B"
    g = grade(row("<think>... the answer is (C) ... loop", finish="length"), make_grader("mc", None))
    assert g["pred"] is None and not g["correct"] and g["correct_lenient"] and g["cut"]
    assert last_boxed("\\boxed{1} then \\boxed{\\frac{3}{4}}") == "\\frac{3}{4}"
    assert last_boxed("\\boxed{\\frac{3}{") is None


def test_generic_graders_flag_and_thinking_off():
    no_think = row("Paris is the capital.\nParis.", answer="paris", thinking=False)
    assert grade(no_think, make_grader("exact", None))["correct"]
    assert grade(no_think, make_grader("contains", None))["correct"]
    assert grade(no_think, make_grader("exact", r"capital of (\w+)"))["correct"] is False
    assert grade(no_think, make_grader("regex", r"(?i)paris"))["correct"]
    g = grade(no_think, make_grader("none", None), flag=r"capital")
    assert g["correct"] is None and g["flagged"] and g["closed_thinking"]


def test_custom_grader_file(tmp_path):
    path = tmp_path / "my_grader.py"
    path.write_text("def grade(row, text):\n    return text.strip().endswith('42')\n")
    g = make_grader(f"{path}:grade", None)
    assert grade(row("<think>x</think> it is 42"), g)["correct"]
    assert not grade(row("<think>x</think> it is 41"), g)["correct"]


def test_paired_and_sign_test():
    before = [{"id": str(i), "sample": 0, "correct": c, "flagged": f} for i, (c, f) in
              enumerate([(0, 1), (0, 1), (1, 0), (1, 0)])]
    after = [{"id": str(i), "sample": 0, "correct": c, "flagged": f} for i, (c, f) in
             enumerate([(1, 0), (0, 1), (1, 0), (0, 1)])]
    p = paired(before, after)
    assert (p["fixed"], p["broke"], p["both_good"]) == (1, 1, 1)
    assert paired(before, after, key="flagged")["fixed"] == 1
    assert sign_test(10, 0) == 0.002 and sign_test(5, 5) == 1.0


def test_pair_windows_cover_exactly_the_continuations(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"id": "p1", "prompt": "Q", "prefix": "Sure. ", "wanted": "No.", "unwanted": "Yes!"}))
    (item,) = pair_items(CharTokenizer(), pairs, thinking=False)
    for text, kind, branch in zip(item["texts"], ("wanted", "unwanted"), ("No.", "Yes!")):
        s, e = text["windows"][kind]
        assert "".join(map(chr, text["ids"][s:e])) == branch
        assert "".join(map(chr, text["ids"][:s])) == "<u>Q<a>Sure. "


def test_edit_flips_alpha_times_component_on_chosen_columns_only():
    w = torch.randn(ROWS, COLS, generator=torch.Generator().manual_seed(0))
    u = unit(torch.randn(ROWS, generator=torch.Generator().manual_seed(1)))
    cols = torch.tensor([3, 10, 50])
    w2 = edit_layer(w, u, cols, alpha=6.0)
    assert torch.allclose((u @ w2)[cols], -5.0 * (u @ w)[cols], atol=1e-4)
    untouched = [i for i in range(COLS) if i not in (3, 10, 50)]
    assert torch.equal(w2[:, untouched], w[:, untouched])


def test_random_control_matches_size_and_misses_direction():
    w = torch.randn(ROWS, COLS, generator=torch.Generator().manual_seed(2))
    u = unit(torch.randn(ROWS, generator=torch.Generator().manual_seed(3)))
    cols = torch.tensor([1, 2, 3, 4])
    real, rand = edit_layer(w, u, cols, 6.0) - w, edit_layer(w, u, cols, 6.0, "random") - w
    assert abs(real.norm().item() - rand.norm().item()) < 1e-3
    assert (u @ rand).abs().max().item() < 1e-4


def test_scores_pick_the_neuron_that_writes_the_unwanted_direction():
    u_true = unit(torch.tensor([1.0] + [0.0] * (ROWS - 1)))
    w = torch.randn(ROWS, COLS, generator=torch.Generator().manual_seed(4)) * 0.01
    w[0, 7] += 1.0
    act = torch.zeros(COLS)
    act[7] = 2.0
    states = {"0_wanted_resid": torch.zeros(2, ROWS).numpy(), "0_unwanted_resid": torch.stack([u_true] * 2).numpy(),
              "0_wanted_act": torch.zeros(2, COLS).numpy(), "0_unwanted_act": torch.stack([act] * 2).numpy()}
    u, scores = direction_and_scores([{"half": "build"}], states, 1, w)
    assert torch.allclose(u, u_true) and scores.argmax().item() == 7


def test_presence_penalty_ignores_prompt_tokens():
    out = OutputOnlyPresencePenalty(1.5, prompt_width=2)(torch.tensor([[5, 6, 7, 6]]), torch.zeros(1, 10))
    assert out[0, 7] == -1.5 and out[0, 6] == -1.5 and out[0, 5] == 0.0


def test_batch_sizing_quotas_profiles_and_supervisor_args(tmp_path):
    cfg = {"hidden_size": 1024, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
           "num_hidden_layers": 4, "layer_types": ["linear_attention"] * 3 + ["full_attention"]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    per_token = 2 * 2 * 256 * 2 * 1
    assert recommended_batch(tmp_path, 1000, memory=per_token * 1000 * 4 * 10 / 0.8) == 10
    assert recommended_batch(tmp_path, 1000, jobs=2, memory=per_token * 1000 * 4 * 10 / 0.8) == 5
    assert quotas(30, ["a", "b", "c", "d"]) == {"a": 8, "b": 8, "c": 7, "d": 7}
    args = SimpleNamespace(profile="non-thinking", temperature=None, top_p=0.9, top_k=None, presence_penalty=0.0)
    s = sampling_from(args)
    assert s["thinking"] is False and s["top_p"] == 0.9 and s["presence_penalty"] == 0.0 and s["temperature"] == 0.7
    assert without_batch(["--task", "x", "--batch", "8", "--batch=4", "--out", "y"]) == ["--task", "x", "--out", "y"]
