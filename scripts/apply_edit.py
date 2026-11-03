"""Write an edit (safetensors of replacement tensors) into the working copy of the model on disk.

    python scripts/apply_edit.py --base <base dir> --copy <copy dir> --edit edits/<edit>.safetensors

Refuses unless every safetensors file of the copy is still bit-identical to the base, so edits never stack. Each
touched shard is rewritten through a temporary file and a rename, so an interrupted write leaves the copy intact.
To switch to another edit, first restore the copy's files from the base (a plain file copy, see SKILL.md).
"""
import argparse
from pathlib import Path

from weights import file_sha256, shard_map, write_tensors


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--copy", type=Path, required=True)
    p.add_argument("--edit", type=Path, required=True)
    args = p.parse_args()
    if args.base.resolve() == args.copy.resolve():
        raise SystemExit("refusing to edit the base model; make a copy first")
    from safetensors.torch import load_file  # noqa: PLC0415

    shards = sorted(set(shard_map(args.base).values()))
    changed = [s for s in shards if file_sha256(args.copy / s) != file_sha256(args.base / s)]
    if changed:
        raise SystemExit(f"the copy already differs from base in {changed}; restore those files from base first")
    edits = load_file(args.edit)
    unknown = sorted(set(edits) - set(shard_map(args.copy)))
    if unknown:
        raise SystemExit(f"the edit names tensors the model does not have: {unknown}")
    write_tensors(args.copy, edits)
    for shard in sorted({shard_map(args.copy)[n] for n in edits}):
        print(f"{shard}: {len([n for n in edits if shard_map(args.copy)[n] == shard])} tensors replaced, "
              f"sha256 {file_sha256(args.copy / shard)}")


if __name__ == "__main__":
    main()
