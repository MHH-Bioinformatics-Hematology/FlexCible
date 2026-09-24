"""Per-chromosome expected(d) and per-diagonal log-offset(d) of a balanced cool,
exactly as scripts/oe_proper.py computes them, saved for re-adding the distance
decay to proper-O/E targets and predictions (log-obs space evaluation).

    python scripts/expected_profile.py --input <mcool> --binsize 2000 \
        --band 511 --output expected_gm12878_2kb.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oe_proper import open_at, stream_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--binsize", type=int, default=2000)
    ap.add_argument("--band", type=int, default=511)
    ap.add_argument("--chroms", nargs="*", default=None)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    clr = open_at(a.input, a.binsize)
    chroms = a.chroms or [c for c in clr.chromnames
                          if c.replace("chr", "") in [str(i) for i in range(1, 23)]]
    w_all = clr.bins()[:]["weight"].values
    out = {}
    for ch in chroms:
        o = clr.offset(ch); n = clr.extent(ch)[1] - o
        mp = np.isfinite(w_all[o:o + n])
        exp, off = stream_stats(clr, ch, a.band, w_all, mp)
        out[f"{ch}:exp"], out[f"{ch}:offset"] = exp, off
        print(ch, "exp[1]", exp[1], "exp[band]", exp[-1], flush=True)
        np.savez(a.output, **out)


if __name__ == "__main__":
    main()
