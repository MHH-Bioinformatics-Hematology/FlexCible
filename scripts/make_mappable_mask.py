"""Write a per-bin mappability mask beside each normalised matrix.

Why: window.max_zero_fraction is inert on Akita-space targets. It counts
exactly-zero pixels, but in that space an empty-but-mappable pixel is -2 (real
signal: fewer contacts than expected), and Gaussian smoothing leaves almost
nothing at exactly 0.0. Measured on chr20, it drops 0 of 124 windows -- so
unmappable pericentromeric regions reach training unfiltered, and silently.

The source coolers already know which bins are unusable: balancing leaves a NaN
weight wherever it could not converge, which is exactly the unmappable and
repeat-collapsed sequence. This dumps that per chromosome so precompute can
drop windows on a signal that means something.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cooler
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from akita_normalize import open_at  # noqa: E402


def build(src: str, binsize: int) -> dict[str, np.ndarray]:
    clr = open_at(src, binsize)
    if "weight" not in clr.bins().columns:
        raise SystemExit(f"{src}: no balancing weights, cannot judge mappability")
    bins = clr.bins()[:]
    out = {}
    for ch in clr.chromnames:
        lo, hi = clr.extent(ch)
        w = bins["weight"].values[lo:hi]
        out[ch] = np.isfinite(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the raw/balanced cool or mcool")
    ap.add_argument("--out", required=True, help="<normalised>.mappable.npz")
    ap.add_argument("--binsize", type=int, default=2000)
    a = ap.parse_args()

    m = build(a.source, a.binsize)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **{k: v for k, v in m.items()})
    tot = sum(v.size for v in m.values())
    ok = sum(int(v.sum()) for v in m.values())
    print(f"{a.out}: {ok:,}/{tot:,} bins mappable ({100*ok/tot:.1f}%)")
    worst = sorted(((100 * v.mean(), k) for k, v in m.items()))[:3]
    print("  least mappable: " + ", ".join(f"{k} {p:.0f}%" for p, k in worst))


if __name__ == "__main__":
    main()
