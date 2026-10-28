"""HiCRep SCC between the measured maps of every pair of cell types.

Shows how much contact structure cell types share, the basis of the consensus
reference. Stratum-adjusted correlation (HiCRep definition, smoothing h 5, 1 Mb band) on
the O/E target maps crosscell_oe/<cell>_proper_<res>kb.cool, on the held-out
chromosomes 2, 3, 11, 14 and 17. Uses flexcible.evaluate.scc_streaming, which
works per chromosome, because the 4DN and ENCODE maps differ in their
chromosome sets and the hicrep package requires identical bin tables.

    ~/miniconda3/envs/hicdiff/bin/python scripts/pairwise_celltype_scc.py --res 25 10
Output: results/pairwise_celltype_scc/pairs_<res>kb.csv (cell_a, cell_b, chrom, scc)
"""
from __future__ import annotations

import argparse
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OE = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_oe")
OUT = Path(__file__).resolve().parents[1] / "results/pairwise_celltype_scc"
CELLS = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
         "gm23248", "mcf7", "cd14mono", "pc9", "k562", "hmec_deep", "huvec_deep"]
CHROMS = ["2", "3", "11", "14", "17"]


def one_pair(job: tuple[str, str, int]) -> list[tuple]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flexcible.evaluate import scc_streaming
    a, b, res = job
    pa, pb = OE / f"{a}_proper_{res}kb.cool", OE / f"{b}_proper_{res}kb.cool"
    band = 1_000_000 // (res * 1000)
    return [(a, b, c, float(scc_streaming(str(pa), str(pb), c, band, h=5)))
            for c in CHROMS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="+", type=int, default=[25, 10])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for res in args.res:
        jobs = [(a, b, res) for a, b in itertools.combinations(CELLS, 2)]
        out = OUT / f"pairs_{res}kb.csv"
        tmp = out.with_suffix(".tmp")
        with ProcessPoolExecutor(args.workers) as ex, open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell_a", "cell_b", "chrom", "scc"])
            for rows in ex.map(one_pair, jobs):
                w.writerows(rows)
        tmp.rename(out)
        print(f"wrote {out}: {len(jobs)} pairs x {len(CHROMS)} chromosomes")


if __name__ == "__main__":
    main()
