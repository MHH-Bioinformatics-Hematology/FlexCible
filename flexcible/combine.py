"""Build a single cool whose upper triangle is one matrix and lower another.

HiGlass draws a non-symmetric cool as a full square, so putting measured data
above the diagonal and predicted below it gives the side-by-side comparison in
one view. The same mechanism serves reference-vs-edited prediction.

When the two sides come from genomes of different length (an edit changed the
coordinates), the shorter side is padded so the diagonal still lines up and
the inserted or deleted stretch shows as a gap in the unedited side.

CLI:
  python -m flexcible.combine --upper measured.cool --lower predicted.cool \
      --chrom 2 --out combined.cool
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def combine_matrices(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Upper triangle from `upper`, lower from `lower`, padded to one size."""
    n = max(upper.shape[0], lower.shape[0])
    out = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, 1)
    il = np.tril_indices(n, -1)
    up = np.zeros((n, n)); up[: upper.shape[0], : upper.shape[1]] = upper
    lo = np.zeros((n, n)); lo[: lower.shape[0], : lower.shape[1]] = lower
    out[iu] = up[iu]
    out[il] = lo[il]
    d = np.arange(n)
    out[d, d] = 0.5 * (up[d, d] + lo[d, d])
    return out


def write_combined(upper_cool: str, lower_cool: str, chrom: str, out_path: str,
                   binsize: int | None = None) -> str:
    import cooler
    import pandas as pd

    from .genome import match_name
    from .matrix import fetch_chrom_matrix

    cu, cl = cooler.Cooler(upper_cool), cooler.Cooler(lower_cool)
    binsize = binsize or cu.binsize
    a = fetch_chrom_matrix(upper_cool, match_name(chrom, list(cu.chromnames)))
    b = fetch_chrom_matrix(lower_cool, match_name(chrom, list(cl.chromnames)))
    m = combine_matrices(a.astype(np.float64), b.astype(np.float64))
    n = m.shape[0]
    name = chrom if chrom.startswith("chr") else f"chr{chrom}"
    bins = pd.DataFrame({"chrom": [name] * n,
                         "start": np.arange(n) * binsize,
                         "end": (np.arange(n) + 1) * binsize})
    i, j = np.nonzero(m)
    px = pd.DataFrame({"bin1_id": i, "bin2_id": j, "count": m[i, j]})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(out_path, bins=bins, pixels=px,
                         dtypes={"count": "float64"},
                         symmetric_upper=False, ordered=False,
                         triucheck=False, dupcheck=False, boundscheck=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Combine two matrices into one upper/lower cool.")
    ap.add_argument("--upper", required=True)
    ap.add_argument("--lower", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(write_combined(args.upper, args.lower, args.chrom, args.out))


if __name__ == "__main__":
    main()
