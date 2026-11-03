"""Compare an edited model directory with its base: which tensors changed, and whether each change is the intended edit.

    python scripts/diff_edit.py --base <base dir> --edited <copy dir> [--out report.json]

For every tensor whose bits differ: changed entries, columns and rows, relative Frobenius change. For 2-D changes it
also takes the SVD of the change over the changed columns: the share of its energy in the first direction u (about 1
for a rank-1 edit), and how much of each column's u-component was removed (about alpha). Finally the cosine between
the directions of different layers.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from weights import shard_map


def changed_tensors(base, edited):
    names_b, names_e = shard_map(base), shard_map(edited)
    if set(names_b) != set(names_e):
        raise SystemExit(f"tensor names differ: {sorted(set(names_b) ^ set(names_e))[:10]}")
    changed = []
    for shard in sorted(set(names_b.values())):
        with safe_open(Path(base) / shard, framework="pt") as fb, safe_open(Path(edited) / shard, framework="pt") as fe:
            for name in fb.keys():
                a, b = fb.get_tensor(name), fe.get_tensor(name)
                if not torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)):
                    changed.append((name, a, b))
    return len(names_b), changed


def describe(name, w0, w1):
    delta = w1.double() - w0.double()
    mask = delta != 0
    info = {"name": name, "shape": list(w0.shape), "changed_entries": int(mask.sum()),
            "relative_change": round(float(delta.norm() / w0.double().norm()), 5)}
    if w0.dim() != 2:
        return info, None
    cols = mask.any(0).nonzero().squeeze(1)
    d = delta[:, cols]
    u_mat, s, _ = torch.linalg.svd(d, full_matrices=False)
    u = u_mat[:, 0]
    if (u @ w0.double()[:, cols]).mean() < 0:
        u = -u
    removed = -(u @ d) / (u @ w0.double()[:, cols])
    info.update(changed_columns=len(cols), changed_rows=int(mask.any(1).sum()),
                rank1_energy_share=round(float(s[0] ** 2 / (s ** 2).sum()), 6),
                alpha_min_median_max=[round(float(x), 4) for x in (removed.min(), removed.median(), removed.max())])
    return info, u


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--edited", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    total, changed = changed_tensors(args.base, args.edited)
    infos, dirs = [], {}
    for name, w0, w1 in changed:
        info, u = describe(name, w0, w1)
        infos.append(info)
        if u is not None:
            dirs[name] = u
    names = list(dirs)
    cosines = {f"{a} | {b}": round(float(dirs[a] @ dirs[b]), 4) for i, a in enumerate(names) for b in names[i + 1:]}
    report = {"total_tensors": total, "changed": [i["name"] for i in infos], "tensors": infos,
              "direction_cosines": cosines}
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
