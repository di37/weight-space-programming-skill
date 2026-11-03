"""Make the working copy of a model, or restore it to the base, on any OS.

    python scripts/copy_model.py --base <base dir> --copy <copy dir>            # new copy
    python scripts/copy_model.py --base <base dir> --copy <copy dir> --restore  # undo an edit (weights only)

The copy gets every file of the base except download caches. On filesystems with copy-on-write clones (APFS, Btrfs,
XFS) the OS may make this instant; elsewhere it is a full copy of the weights. Afterwards every safetensors file of
the copy is checked against the base by sha256.
"""
import argparse
import shutil
from pathlib import Path

from weights import file_sha256

SKIP = shutil.ignore_patterns(".cache", ".git", "*.tmp")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--copy", type=Path, required=True)
    p.add_argument("--restore", action="store_true", help="copy the base's safetensors files over the copy's")
    args = p.parse_args()
    if args.base.resolve() == args.copy.resolve():
        raise SystemExit("base and copy are the same folder")
    shards = sorted(f.name for f in args.base.glob("*.safetensors"))
    if args.restore:
        for name in shards:
            shutil.copy2(args.base / name, args.copy / name)
    elif args.copy.exists():
        raise SystemExit(f"{args.copy} already exists; use --restore to reset its weights")
    else:
        shutil.copytree(args.base, args.copy, ignore=SKIP)
    for name in shards:
        same = file_sha256(args.base / name) == file_sha256(args.copy / name)
        print(f"{name}: {'identical to base' if same else 'DIFFERS FROM BASE'}")


if __name__ == "__main__":
    main()
