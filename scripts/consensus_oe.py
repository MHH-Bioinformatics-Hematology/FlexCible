"""Consensus O/E map: the pixel-wise mean of several cells' proper O/E maps.

Input maps come from scripts/oe_proper.py at one resolution. They share the
hg38 bin grid and store every pixel of the 2 Mb band (identical pixel tables),
so each chromosome's pixel table is streamed in chunks from every map
(matched by chromosome name, since bin table order differs between maps),
checked to hold the same pixels and averaged value by value. Pixels a cell
cannot map carry that cell's smoothed value near 0 (the expected level), as
in the maps themselves. The
result serves as data.reference_cool (a reference map as model input) and as
the reference-map-alone baseline.

    python scripts/consensus_oe.py --res 25000 --cells imr90 hff h1 ... \
        --out crosscell_oe/consensus11_proper_25kb.cool

Written to <out>.part and renamed when complete; <out>.json records the cells.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cooler
import h5py
import numpy as np
import pandas as pd

OE = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_oe")
CHUNK = 20_000_000


def cell_cool(cell: str, res: int) -> Path:
    return OE / f"{cell}_proper_{res // 1000}kb.cool"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--grid", default=None,
                    help="cool whose bin table the output uses (default: the "
                         "GM12878 proper map at --res, the training target grid)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = [cell_cool(c, a.res) for c in a.cells]
    clrs = [cooler.Cooler(str(p)) for p in paths]
    grid = cooler.Cooler(a.grid or str(cell_cool("gm12878", a.res)))
    bins = grid.bins()[:][["chrom", "start", "end"]]
    offs = []
    for k in clrs:
        with h5py.File(k.filename, "r") as h:
            offs.append((h[k.root] if k.root != "/" else h)["indexes/bin1_offset"][:])
    # chromosomes are matched by name (bin tables differ in order, e.g. chrM
    # first in some ENCODE maps); every map must hold the same local pixels
    chroms = [c for c in grid.chromnames if all(c in k.chromnames for k in clrs)]
    print(f"{len(clrs)} maps at {a.res} bp: {', '.join(a.cells)}", flush=True)
    t0 = time.time()
    n_px = {}

    def chunks():
        for chrom in chroms:
            nb = int(grid.extent(chrom)[1] - grid.offset(chrom))
            rng = []
            for c, k, o in zip(a.cells, clrs, offs):
                fo = int(k.offset(chrom))
                if int(k.extent(chrom)[1]) - fo != nb:
                    raise SystemExit(f"{c} {chrom}: {int(k.extent(chrom)[1]) - fo} bins, grid {nb}")
                rng.append((fo, int(o[fo]), int(o[fo + nb])))
            npx = {r[2] - r[1] for r in rng}
            if len(npx) != 1:
                raise SystemExit(f"{chrom}: pixel counts differ {npx}")
            npx = npx.pop()
            if npx == 0:
                continue
            go = int(grid.offset(chrom))
            for s in range(0, npx, CHUNK):
                e = min(s + CHUNK, npx)
                b1 = b2 = acc = None
                for c, k, (fo, p0, _) in zip(a.cells, clrs, rng):
                    px = k.pixels()[p0 + s:p0 + e]
                    i, j = px["bin1_id"].values - fo, px["bin2_id"].values - fo
                    if b1 is None:
                        b1, b2 = i, j
                        acc = px["count"].values.astype(np.float64)
                        if (j >= nb).any() or (i < 0).any():
                            raise SystemExit(f"{c} {chrom}: trans pixels")
                    else:
                        if not (np.array_equal(i, b1) and np.array_equal(j, b2)):
                            raise SystemExit(f"{c} {chrom}: pixel table differs")
                        acc += px["count"].values
                acc /= len(clrs)
                yield pd.DataFrame({"bin1_id": b1 + go, "bin2_id": b2 + go,
                                    "count": acc.astype(np.float32)})
            n_px[chrom] = npx
            print(f"  {chrom}: {npx:,} pixels ({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = Path(a.out)
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)
    cooler.create_cooler(str(part), bins=bins, pixels=chunks(),
                         dtypes={"count": np.float32}, ordered=True)
    part.rename(out)
    meta = {"res": a.res, "cells": a.cells, "inputs": [str(p) for p in paths],
            "grid": grid.filename, "pixels_per_chrom": n_px,
            "method": "pixel-wise mean of oe_proper.py maps"}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(f"-> {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
