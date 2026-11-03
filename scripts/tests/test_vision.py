"""Unit tests for image tasks: CIDEr-D, yes/no reading, image prompts and probe windows, pairs, data building.
No model, GPU or network needed.

    python -m pytest scripts/tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import FirstCallIsPromptPenalty  # noqa: E402
from cider import CiderD, tokens  # noqa: E402
from grade import caption_text, grade, make_grader, paired, summarize, yes_no  # noqa: E402
from make_image_data import relevance_rows, split_by_image  # noqa: E402
from make_pairs import consensus_reference, make_pairs  # noqa: E402
from probe import (Prompter, decision_items, decision_labels, direction_and_scores, pair_items,  # noqa: E402
                   split_answer, text_item)
from tasks import TASKS, job_meta, load_rows  # noqa: E402
from vision import user_content  # noqa: E402

IMAGE_TOKENS = 5
PAD = "<img>"


class FakeTokenizer:
    """One token per character; an image turn renders a single placeholder, as Qwen's template does."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        content = messages[0]["content"]
        text = content if isinstance(content, str) else PAD + content[1]["text"]
        return "<u>" + text + "<a>" + ("<t>" if enable_thinking else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class FakeProcessor:
    """Expands the placeholder into IMAGE_TOKENS copies of token 1, like a vision processor."""

    def __call__(self, text, images, return_tensors):
        import torch  # noqa: PLC0415

        head, tail = text[0].split(PAD)
        ids = [ord(c) for c in head] + [1] * IMAGE_TOKENS + [ord(c) for c in tail]
        return {"input_ids": torch.tensor([ids])}


def tiny_image(path):
    from PIL import Image  # noqa: PLC0415

    Image.new("RGB", (8, 6), (200, 30, 30)).save(path)
    return path


def run_row(rid, text, answer, group="g", thinking=False, image=None):
    return {"id": rid, "sample": 0, "group": group, "answer": answer, "finish_reason": "stop", "n_tokens": 5,
            "text": text, "thinking": thinking, "image": image}


def test_cider_matches_the_official_scale_and_edge_cases():
    refs = [["a b c d e"] * 3, ["f g h i j"] * 3]
    scorer = CiderD(refs)
    assert abs(scorer.score("a b c d e", refs[0]) - 10.0) < 1e-9
    assert scorer.score("f g h i j", refs[0]) == 0.0
    assert scorer.score("", refs[0]) == 0.0
    assert CiderD([["a b c"]]).score("a b c", ["a b c"]) == 0.0  # one image: every idf is 0, as in the official code
    assert tokens("A Dog's toy, on THE grass.") == ["a", "dog's", "toy", "on", "the", "grass"]


def test_cider_rewards_closer_captions_and_penalises_length_in_bigrams():
    refs = [["a black dog runs on the grass", "a dog running across a green lawn", "black dog playing outside"],
            ["two people ride bikes down a city street", "cyclists on a busy road", "people biking in traffic"],
            ["a plate of pasta with tomato sauce", "spaghetti on a white plate", "a bowl of noodles and sauce"]]
    scorer = CiderD(refs)
    close, loose = scorer.score("a black dog running on the grass", refs[0]), scorer.score("an animal outdoors", refs[0])
    assert close > loose > -1e-12
    padded = scorer.score("a black dog running on the grass " + "near some trees by a wall " * 3, refs[0])
    assert padded < close


def test_caption_and_yes_no_reading():
    assert caption_text('Caption: "A dog on grass."\n\nMore text') == "A dog on grass."
    assert caption_text("**A red bus.**") == "A red bus."
    assert caption_text("\n  \n") is None and caption_text(None) is None
    assert yes_no("Yes, there is a cat.") == "yes" and yes_no("no.") == "no"
    assert yes_no("The answer is: No") == "no" and yes_no("Maybe") is None
    assert yes_no("Nothing matches; yes.") == "yes"


def test_cider_grader_scores_pairs_and_summarises():
    refs = [["a black dog on grass", "a dog in a field"], ["a red bus on a street", "a bus in the city"]]
    base = [run_row("1", "A cat.", refs[0]), run_row("2", "A red bus on a street.", refs[1])]
    copy = [run_row("1", "A black dog on the grass.", refs[0]), run_row("2", "A red bus on a street.", refs[1])]
    grader = make_grader("cider", None, rows=base)
    gb, gc = [grade(r, grader) for r in base], [grade(r, grader) for r in copy]
    assert gb[0]["correct"] is None and gc[0]["score"] > gb[0]["score"] and gc[1]["score"] == gb[1]["score"]
    s = summarize(gb)
    assert "score" in s and "accuracy" not in s and s["median_answer_words"] >= 2
    p = paired(gb, gc, key="score")
    assert (p["fixed"], p["broke"], p["same"]) == (1, 0, 1) and p["mean_change"] > 0


def test_yes_no_grader_reports_bias_and_f1():
    rows = [run_row("1", "Yes", "yes"), run_row("2", "Yes.", "no"), run_row("3", "No", "no"), run_row("4", "No", "yes")]
    graded = [grade(r, make_grader("yesno", None)) for r in rows]
    assert [g["correct"] for g in graded] == [True, False, True, False]
    s = summarize(graded)
    assert s["accuracy"] == 50.0 and s["yes_rate"] == 50.0 and s["precision"] == 50.0 and s["recall"] == 50.0


def test_image_tasks_resolve_images_and_keep_references(tmp_path):
    (tmp_path / "images").mkdir()
    data = tmp_path / "practice.jsonl"
    data.write_text(json.dumps({"id": "7", "image": "images/7.jpg", "references": ["a cat"], "group": "coco"}) + "\n")
    (row,) = load_rows(data, task="caption")
    assert row["image"] == str((tmp_path / "images/7.jpg").resolve())
    meta = job_meta("caption", row, 0)
    assert meta["answer"] == ["a cat"] and meta["image"] == row["image"]
    assert TASKS["caption"]["profile"] == "non-thinking" and "caption" in TASKS["caption"]["prompt"](row)
    assert "Text: \"a dog\"" in TASKS["relevance"]["prompt"]({"text": "a dog"})
    assert TASKS["pope"]["prompt"]({"question": "Is there a cat?"}).startswith("Is there a cat?")
    assert job_meta("mmlu", {"question_id": 1, "category": "x", "answer": "A"}, 0)["image"] is None
    assert user_content("hi", None) == "hi" and user_content("hi", "x.jpg")[0] == {"type": "image"}


def test_image_pair_windows_skip_the_expanded_image_tokens(tmp_path):
    tiny_image(tmp_path / "x.jpg")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"id": "p1", "prompt": "Describe.", "image": "x.jpg", "wanted": "A cat.",
                                 "unwanted": "A big orange cat sitting."}))
    (item,) = pair_items(Prompter(FakeTokenizer(), processor=FakeProcessor()), pairs, thinking=False)
    for text, kind, branch in zip(item["texts"], ("wanted", "unwanted"), ("A cat.", "A big orange cat sitting.")):
        s, e = text["windows"][kind]
        assert "".join(map(chr, text["ids"][s:e])) == branch
        assert text["ids"].count(1) == IMAGE_TOKENS and text["image"] == str(tmp_path / "x.jpg") and text["head"] == s


def test_decision_items_read_the_states_that_produce_the_answer(tmp_path):
    tiny_image(tmp_path / "x.jpg")
    data = tmp_path / "split.jsonl"
    data.write_text("".join(json.dumps({"id": i, "question": "Is there a cat?", "answer": a, "image": "x.jpg"}) + "\n"
                            for i, a in (("1", "yes"), ("2", "no"))))
    runs = [run_row("1", "Yes", "yes", image="x.jpg"), run_row("2", "Yes", "no", image="x.jpg")]
    graded = [{"id": "1", "sample": 0, "correct": True, "score": None, "answered": True},
              {"id": "2", "sample": 0, "correct": False, "score": None, "answered": True}]
    items = decision_items(Prompter(FakeTokenizer(), processor=FakeProcessor()), "pope", data, runs, graded)
    kinds = {it["id"]: next(iter(it["texts"][0]["windows"])) for it in items}
    assert kinds == {"1": "wanted", "2": "unwanted"}
    text = items[1]["texts"][0]
    (s, e), (ss, se) = text["windows"]["unwanted"], text["state_windows"]["unwanted"]
    assert "".join(map(chr, text["ids"][s:e])) == "Yes" and (ss, se) == (s - 1, e - 1)


def test_direction_from_unpaired_items_is_the_difference_of_class_means():
    states = {"0_unwanted_resid": np.array([[2.0, 0.0]]), "1_unwanted_resid": np.array([[4.0, 0.0]]),
              "2_wanted_resid": np.array([[0.0, 1.0]]), "0_unwanted_act": np.array([[1.0, 0.0, 0.0]]),
              "1_unwanted_act": np.array([[1.0, 0.0, 0.0]]), "2_wanted_act": np.array([[0.0, 0.0, 0.0]])}
    import torch  # noqa: PLC0415

    w = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    u, scores = direction_and_scores([{"half": "build"}] * 3, states, 0, w)
    expected = torch.tensor([3.0, -1.0]) / torch.tensor([3.0, -1.0]).norm()
    assert torch.allclose(u, expected) and scores.argmax().item() == 0


def test_make_pairs_uses_gold_text_for_failures_only(tmp_path):
    data = tmp_path / "split.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in (
        {"id": "1", "image": "a.jpg", "references": ["a dog on grass", "a dog in a park", "a cat"]},
        {"id": "2", "image": "b.jpg", "references": ["a red bus", "a bus on a street"]})))
    runs = [run_row("1", "A colourful scene with many things.", None, image="a.jpg"),
            run_row("2", "A red bus.", None, image="b.jpg")]
    graded = [{"id": "1", "sample": 0, "correct": None, "score": 0.1, "answered": True},
              {"id": "2", "sample": 0, "correct": None, "score": 2.0, "answered": True}]
    pairs = make_pairs("caption", data, runs, graded)
    assert [p["id"] for p in pairs] == ["1"] and pairs[0]["image"].endswith("a.jpg")
    assert pairs[0]["wanted"] == "a dog on grass" and pairs[0]["unwanted"] == "A colourful scene with many things."
    assert consensus_reference(["a dog on grass", "a dog in a park", "a cat"]) == "a dog on grass"
    pope = tmp_path / "pope.jsonl"
    pope.write_text(json.dumps({"id": "1", "image": "a.jpg", "question": "Is there a dog?", "answer": "no"}) + "\n")
    yn = make_pairs("pope", pope, [run_row("1", "Yes.", "no")], [{"id": "1", "sample": 0, "correct": False,
                                                                  "score": None, "answered": True}])
    assert yn[0]["wanted"] == "No." and yn[0]["unwanted"] == "Yes."


def test_relevance_rows_are_balanced_and_negatives_come_from_other_images():
    images = [{"id": str(i), "image": f"images/{i}.jpg", "references": [f"a {w} on a table", f"the {w} sits there"]}
              for i, w in enumerate(["cup", "book", "phone", "lamp", "shoe", "plate", "pen", "clock"])]
    rows = relevance_rows(images, seed=0)
    assert rows == relevance_rows(images, seed=0)
    assert sorted(r["answer"] for r in rows) == ["no"] * 4 + ["yes"] * 4
    own = {img["id"]: img["references"] for img in images}
    for r in rows:
        assert (r["text"] in own[r["id"]]) == (r["answer"] == "yes") and r["group"] == r["answer"]


def test_split_by_image_keeps_each_image_in_one_split():
    rows = [{"id": f"{img}-{q}", "image_key": img} for img in "abcdefgh" for q in range(3)]
    practice, heldout = split_by_image(rows, 6, 9, seed=1)
    assert len(practice) >= 6 and len(heldout) >= 9
    assert not {r["image_key"] for r in practice} & {r["image_key"] for r in heldout}


def test_review_fixes_unparsed_answers_ties_whitespace_and_split_keys(tmp_path):
    g = grade(run_row("1", "I can't tell.", "yes"), make_grader("yesno", None))
    assert g["correct"] is False and g["answered"] is False
    assert grade(run_row("1", "It is blue.", "blue"), make_grader("contains", None))["answered"]
    tied = [{"id": str(i), "sample": 0, "score": 0.0, "answered": True} for i in range(5)]
    labels = decision_labels(tied + [{"id": "9", "sample": 0, "score": 1.0, "answered": True}])
    assert sorted(labels.values()).count("unwanted") == 3 and labels[("9", 0)] == "wanted"
    assert split_answer(run_row("1", "<think>x</think>\n\nNo.", "no", thinking=True)) == ("<think>x</think>\n\n", "No.")
    a = text_item("q#1", None, [], split_key="q")
    assert a["half"] == text_item("q", None, [])["half"]


def test_mlx_vision_penalty_treats_the_first_call_as_prompt():
    pen = FirstCallIsPromptPenalty(1.5)
    first = pen(np.array([4, 5, 6]), np.zeros((1, 10)))
    assert not first.any()
    later = pen(np.array([4, 5, 6, 7, 7]), np.zeros((1, 10)))
    assert later[0, 7] == -1.5 and later[0, 4] == 0.0 and later.sum() == -1.5
