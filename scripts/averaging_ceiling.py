"""How far a perfect 2 kb prediction, block-averaged onto a coarser grid, can
agree with that grid's own O/E target: the cell's measured 2 kb O/E map is
averaged with crosscell_predict.coarse_score onto 5, 10 and 25 kb and scored
against the measured 5, 10 and 25 kb O/E maps on the 413 Akita test regions.
O/E at each resolution is computed from that resolution's counts
(oe_proper.py: expected, per-diagonal centring and smoothing per grid), so
the ceiling is below 1.

    python scripts/averaging_ceiling.py --cells gm12878 k562 --out <csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multicell_expanded as mx                                   # noqa: E402
import score_like_hicgan as sc                                    # noqa: E402
from crosscell_predict import coarse_score                        # noqa: E402
from hicmultiome.genome import match_name                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=["gm12878", "k562"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    regions = pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    rows = []
    for cell in a.cells:
        fine = cooler.Cooler(str(mx.target(cell, 2000)))
        coarse = {r: cooler.Cooler(str(mx.target(cell, r))) for r in (5000, 10000, 25000)}
        for _, r in regions.iterrows():
            nm = match_name(r.chrom, list(fine.chromnames))
            if nm is None:
                continue
            T = np.asarray(fine.matrix(balance=False).fetch(f"{nm}:{r.start}-{r.end}"), float)
            M = np.where(T == 0, np.nan, T)
            rec = {"cell": cell, "region": f"{r.chrom}:{r.start}-{r.end}"}
            for res, clr in coarse.items():
                v = coarse_score(M, int(r.start), int(r.end), fine.binsize, res, clr, r.chrom)
                if v is not None:
                    rec[f"measured_2kb_averaged_{res // 1000}kb"] = v
            rows.append(rec)
        print(cell, pd.DataFrame([x for x in rows if x["cell"] == cell]).median(numeric_only=True)
              .round(4).to_dict(), flush=True)
    out = Path(a.out)
    tmp = out.with_suffix(".part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(out)


if __name__ == "__main__":
    main()
