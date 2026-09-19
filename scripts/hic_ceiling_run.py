"""Replicate agreement for the Galaxy-built matrices, at every resolution.

Two kinds of comparison:

  pairs  -- one biological replicate against another (HCT116, MOLM-13). This
            is the ceiling as usually quoted, and it is limited by the
            SHALLOWER member of the pair.

  depth  -- GM13977's 16 replicates pooled into DISJOINT subsets of size
            1, 2, 4 and 8 and compared pool against pool. Disjointness is the
            whole point: an overlapping pool shares reads with its partner and
            gets credit for reproducing its own sampling noise, which is
            exactly the artefact that makes a binomial split look reassuring.
            If agreement at 1 kb climbs with pool depth, the collapse at fine
            bins is sparse sampling; if it plateaus well below the coarse-bin
            value, the fine-scale signal is genuinely irreproducible.

The two sides of each depth comparison are assigned by alternating replicates
ordered by depth, so the pools are as close to equal depth as the data allow
rather than one side being systematically deeper.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "hic_matrices"
CEIL = Path(__file__).parent / "replicate_ceiling.py"
RESOLUTIONS = [1000, 2000, 5000, 10000, 25000, 40000]

# GM13977 replicate -> ENA run, with the pair counts reported by ENA.
GM = {
    1: ("SRR27906244", 77_538_055), 2: ("SRR27906243", 73_845_149),
    3: ("SRR27906242", 73_018_506), 4: ("SRR27906241", 120_276_532),
    5: ("SRR27906240", 79_145_055), 6: ("SRR27906239", 102_289_834),
    7: ("SRR27906238", 123_770_217), 8: ("SRR27906237", 45_523_018),
    9: ("SRR27906311", 85_589_064), 10: ("SRR27906310", 82_815_653),
    11: ("SRR27906309", 85_782_900), 12: ("SRR27906308", 47_010_860),
    13: ("SRR27906307", 85_907_692), 14: ("SRR27906306", 35_356_528),
    15: ("SRR27906305", 84_416_778), 16: ("SRR27906304", 81_132_106),
}


def depth_pools():
    """Disjoint, depth-balanced A/B pools at k = 1, 2, 4, 8 replicates."""
    order = sorted(GM, key=lambda r: -GM[r][1])   # deepest first
    plans = {}
    for k in (1, 2, 4, 8):
        take = order[: 2 * k]
        a = take[0::2]
        b = take[1::2]
        assert not set(a) & set(b), "pools must not share a replicate"
        plans[k] = (sorted(a), sorted(b))
    return plans


def run(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="label=fileA,fileB (paths relative to runs/hic_matrices)")
    ap.add_argument("--resolutions", type=int, nargs="*", default=RESOLUTIONS)
    ap.add_argument("--n-regions", type=int, default=40)
    ap.add_argument("--print-plan", action="store_true")
    a = ap.parse_args()

    if a.print_plan:
        for k, (pa, pb) in depth_pools().items():
            da = sum(GM[r][1] for r in pa)
            db = sum(GM[r][1] for r in pb)
            print(f"k={k:2d}  A={pa} ({da/1e6:6.1f}M pairs)")
            print(f"      B={pb} ({db/1e6:6.1f}M pairs)")
        return

    for res in a.resolutions:
        specs = []
        for spec in a.pairs:
            label, files = spec.split("=", 1)
            fa, fb = (OUT / f for f in files.split(","))
            if not (fa.exists() and fb.exists()):
                print(f"  skip {label} @ {res}: missing matrix")
                continue
            specs.append(f"{label}={fa},{fb}")
        if not specs:
            continue
        run([sys.executable, CEIL, "--pairs", *specs, "--res", str(res),
             "--n-regions", str(a.n_regions),
             "--out", ROOT / "runs" / f"gx_ceiling_{res}.csv"])


if __name__ == "__main__":
    main()
