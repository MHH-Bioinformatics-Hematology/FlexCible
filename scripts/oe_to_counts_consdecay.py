"""Back-transform predicted O/E to counts with the CONSENSUS distance decay (no use of the target cell's Hi-C).

scripts/oe_to_counts.py takes expected(d) and offset(d) from the measured map of the cell that is predicted. This variant
takes them from the measured balanced maps of the cells in the reference consensus of that cell, exactly the cells whose
O/E maps the model sees as reference (the cell list in <consensus>.cool.json: leave-one-out consensus of 11 cells for
GM12878 and IMR-90, consensus12 for K562; the target is never among them):

  log expected_cons(d) = mean over cells c of log expected_c(d)      (geometric mean of the per-cell decays)
  offset_cons(d)       = mean over cells c of offset_c(d)

with expected_c(d) and offset_c(d) from oe_proper.stream_stats on the measured 10 kb map of cell c (chromosome-wide,
same definition as oe_to_counts.py). The back-transform is unchanged: count = exp(oe + offset_cons(d)) * expected_cons(d).

    python scripts/oe_to_counts_consdecay.py --oe pred.cool --consensus-json <consensus>.cool.json --output out.cool \
        --chroms chr2 [--stats-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe_to_counts as OC  # noqa: E402

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
PP = Path("/mnt/processed/wolffjoa/restart_of_prediction_project")
MEASURED = {
    "gm12878": PP / "originalData/4Dnucleosome/GM12878_4DNFIXP4QG5B.mcool",
    "h1": PP / "originalData/4Dnucleosome/H1_4DNFI6HDY7WZ.mcool",
    "k562": PP / "originalData/4Dnucleosome/K562_4DNFI18UHVRO.mcool",
    "imr90": Path("/mnt/processed/wolffjoa/hyperparameter_publication/imr90.mcool"),
    "hff": HM / "matrices/hff_microc.mcool",
    "hct116": HM / "matrices/hct116_untreated.mcool",
}
for _c in ["hepg2", "a549", "nhek", "gm23248", "mcf7", "cd14mono", "pc9"]:
    MEASURED[_c] = HM / "encode_hic" / f"{_c}_10kb.cool"      # same sources as scripts/cellspecific.py MEASURED


def cell_stats(cell, chrom, band, stats_dir):
    cache = Path(stats_dir) / f"{cell}_{chrom}_stats.npz" if stats_dir else None
    clr = OC.OP.open_at(str(MEASURED[cell]), 10000)
    return OC.chrom_stats(clr, chrom, band, cache)


def consensus_stats(cells, chrom, band, stats_dir=None):
    E, O = [], []
    for c in cells:
        e, o = cell_stats(c, chrom, band, stats_dir)
        E.append(e)
        O.append(o)
    E, O = np.array(E), np.array(O)
    with np.errstate(divide="ignore", invalid="ignore"):
        le = np.log(E)
    le_ok = np.where(np.isfinite(le), le, np.nan)
    return np.exp(np.nanmean(le_ok, axis=0)), np.nanmean(O, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oe", required=True)
    ap.add_argument("--consensus-json", required=True, help="<consensus>.cool.json: lists the cells of the reference")
    ap.add_argument("--output", required=True)
    ap.add_argument("--chroms", nargs="*", default=["chr2"])
    ap.add_argument("--max-dist", type=int, default=OC.OP.MAX_DIST)
    ap.add_argument("--stats-dir", default=None, help="folder for the per-cell npz stat caches")
    a = ap.parse_args()
    cells = json.load(open(a.consensus_json))["cells"]
    band = a.max_dist // 10000
    oe = cooler.Cooler(a.oe)
    bins = oe.bins()[:][["chrom", "start", "end"]]
    print("consensus decay from", cells, flush=True)

    def stream():
        for ch in a.chroms:
            exp, off = consensus_stats(cells, ch, band, a.stats_dir)
            df = OC.to_counts(oe, ch, band, exp, off)
            print(f"  {ch}: {len(df):,} px", flush=True)
            yield df

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(a.output, bins=bins, pixels=stream(), dtypes={"count": np.float32}, ordered=True)
    print("  ->", a.output, flush=True)


if __name__ == "__main__":
    main()
