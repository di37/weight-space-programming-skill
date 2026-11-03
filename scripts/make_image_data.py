"""Make image benchmark splits: disjoint, seeded practice and held-out sets, a fixed test list, and the image files.

    python scripts/make_image_data.py --preset caption|relevance|pope --out $WORK/splits
        [--practice 100] [--heldout 200] [--test 100] [--seed 0] [--max-side 448] [--source <local parquet>]

Presets (read from the Hugging Face Hub row group by row group, so only the rows needed are downloaded, about 17 MB
per 100 images; --source reads a downloaded copy of the same parquet file instead):
  caption    COCO 2017 val captions (lmms-lab/COCO-Caption2017): one row per image, its 5 `references`.
  relevance  the same COCO images, one row each with a `text` and `answer` yes/no, half and half. "yes": one of the
             image's own captions. "no": a hard negative, the caption of another image that shares the most content
             words with this image's captions, skipping near-duplicates of them. A few negatives may still fit the
             image; both runs see the same ones, so paired comparisons are unaffected.
  pope       POPE adversarial (lmms-lab/POPE): "Is there a <object> in the image?", answer yes/no, 6 per image, the
             negatives being objects that often co-occur with the image's objects.
Splits are by image, so no image is in both. Images are saved under <out>/images/ as JPEG with the longest side at
most --max-side, and rows refer to them by relative path. Writes practice.jsonl, heldout.jsonl and test_ids.txt
(--test held-out ids spread evenly over the groups, taken in file order, chosen before any answer is seen).
"""
import argparse
import io
import json
import random
import re
from pathlib import Path

from make_splits import quotas
from vision import MAX_SIDE

SOURCES = {"coco": "datasets/lmms-lab/COCO-Caption2017/data/val-00000-of-00002.parquet",
           "pope": "datasets/lmms-lab/POPE/Full/adversarial-00000-of-00001.parquet"}
OVERSAMPLE = 1.25
NEAR_DUPLICATE = 0.5
NEGATIVE_CANDIDATES = 200
STOP = set("a an the of on in at to with and or is are was be being by for from as it its this that there their "
           "his her some two one three into near next up down out over under while very".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def hardest_negative(image, pool, rng):
    """The caption of another image closest in content words to this image's captions, minus near-duplicates."""
    own_refs = [content_words(r) for r in image["references"]]
    own = set().union(*own_refs)
    others = [x for x in pool if x["id"] != image["id"]]
    sample = rng.sample(others, min(NEGATIVE_CANDIDATES, len(others)))
    candidates = [(jaccard(content_words(c), own), c) for x in sample for c in x["references"]
                  if all(jaccard(content_words(c), r) <= NEAR_DUPLICATE for r in own_refs)]
    if not candidates:
        raise SystemExit(f"no negative caption for image {image['id']}; read more images")
    return max(candidates, key=lambda sc: sc[0])[1]


def relevance_rows(images, seed, pool=None):
    rng = random.Random(seed)
    labels = ["yes", "no"] * (len(images) // 2) + ["yes"] * (len(images) % 2)
    rng.shuffle(labels)
    rows = []
    for image, label in zip(images, labels):
        text = rng.choice(image["references"]) if label == "yes" else hardest_negative(image, pool or images, rng)
        rows.append({"id": image["id"], "image": image["image"], "text": text, "answer": label, "group": label})
    return rows


def split_by_image(rows, practice, heldout, seed):
    """Whole images to practice until it has `practice` rows, then to held-out until it has `heldout`."""
    keys = sorted({r["image_key"] for r in rows})
    random.Random(seed).shuffle(keys)
    by_key = {}
    for r in rows:
        by_key.setdefault(r["image_key"], []).append(r)
    splits, target = ([], []), (practice, heldout)
    part = 0
    for key in keys:
        if len(splits[part]) >= target[part]:
            part += 1
            if part == 2:
                break
        splits[part].extend(by_key[key])
    if len(splits[1]) < heldout:
        raise SystemExit(f"only {len(rows)} rows read: not enough for {practice} practice + {heldout} held-out")
    return splits


def row_groups(source):
    """Yield the source parquet's row groups as lists of dicts, fetching each only when needed."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    try:
        if Path(source).exists():
            handle = open(source, "rb")  # noqa: SIM115 - closed below
        else:
            from huggingface_hub import HfFileSystem  # noqa: PLC0415

            handle = HfFileSystem().open(source, "rb")
        with handle:
            f = pq.ParquetFile(handle)
            for g in range(f.num_row_groups):
                yield f.read_row_group(g).to_pylist()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {source} ({type(exc).__name__}: {exc}). Check the network, or download "
                         f"the parquet file from the Hugging Face Hub and pass --source") from exc


def save_image(data, path, max_side):
    from PIL import Image  # noqa: PLC0415

    if path.exists():
        return
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
    img.save(path, quality=92)


def read_coco(source, images_needed, out, max_side):
    rows = []
    for group in row_groups(source):
        for r in group:
            image_id = Path(r["question_id"]).stem
            save_image(r["image"]["bytes"], out / "images" / f"{image_id}.jpg", max_side)
            rows.append({"id": image_id, "image_key": image_id, "image": f"images/{image_id}.jpg",
                         "references": list(r["answer"]), "group": "coco"})
        if len(rows) >= images_needed:
            break
    return rows


def read_pope(source, questions_needed, out, max_side):
    rows = []
    for group in row_groups(source):
        for r in group:
            key = r["image_source"]
            save_image(r["image"]["bytes"], out / "images" / f"{key}.jpg", max_side)
            rows.append({"id": str(r["question_id"]), "image_key": key, "image": f"images/{key}.jpg",
                         "question": r["question"], "answer": r["answer"].strip().lower(),
                         "group": r["answer"].strip().lower(), "category": r["category"]})
        if len(rows) >= questions_needed:
            break
    return rows


def build(args):
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    needed = int((args.practice + args.heldout) * OVERSAMPLE) + 1
    if args.preset == "pope":
        pool = read_pope(args.source or SOURCES["pope"], needed, args.out, args.max_side)
        return split_by_image(pool, args.practice, args.heldout, args.seed), ["yes", "no"]
    pool = read_coco(args.source or SOURCES["coco"], needed, args.out, args.max_side)
    practice, heldout = split_by_image(pool, args.practice, args.heldout, args.seed)
    if args.preset == "caption":
        return (practice, heldout), ["coco"]
    return (relevance_rows(practice, args.seed, pool), relevance_rows(heldout, args.seed + 1, pool)), ["yes", "no"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", choices=("caption", "relevance", "pope"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--practice", type=int, default=100, help="rows (pope: questions)")
    p.add_argument("--heldout", type=int, default=200)
    p.add_argument("--test", type=int, default=100, help="held-out ids in test_ids.txt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-side", type=int, default=MAX_SIDE)
    p.add_argument("--source", help="a local copy of the preset's parquet file (default: read from the Hub)")
    args = p.parse_args()
    (practice, heldout), groups = build(args)
    clean = [[{k: v for k, v in r.items() if k != "image_key"} for r in part] for part in (practice, heldout)]
    for name, part in zip(("practice", "heldout"), clean):
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    want, test = quotas(args.test, groups), []
    for g in groups:
        test += [r["id"] for r in clean[1] if r["group"] == g][:want[g]]
    (args.out / "test_ids.txt").write_text("\n".join(test) + "\n")
    print(json.dumps({"practice": len(clean[0]), "heldout": len(clean[1]), "test": len(test),
                      "images": len(list((args.out / "images").glob("*.jpg"))), "out": str(args.out)}))


if __name__ == "__main__":
    main()
