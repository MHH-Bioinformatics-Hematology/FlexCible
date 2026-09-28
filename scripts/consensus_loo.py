"""All leave-one-out consensus maps and the full consensus in one pass.

For N cells (proper O/E maps from oe_proper.py, identical pixel tables per
chromosome, as scripts/consensus_oe.py requires) this reads every map once and
writes consensus<N>_proper_<k>kb.cool (mean of all N) and, per cell,
consensus_loo_<cell>_proper_<k>kb.cool = (sum of all N - that cell) / (N - 1),
which equals the mean of the other N - 1 maps (up to float rounding) without
reading them N times; at 2 kb one read of 12 maps replaces 143. Values are
summed in float64 per chunk and staged as per-chunk .npy files under --tmp
(not /tmp; restartable, removed at the end), then each output is written as a
cooler (.part, renamed when complete) with a .json listing its cells.

    python scripts/consensus_loo.py --res 5000 --cells gm12878 imr90 ... \
        --tmp /mnt/processed/.../runs/multicell_reference/consensus_tmp
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cooler
import h5py
import numpy as np
import pandas as pd

OE = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_oe")
CHUNK = 20_000_000


def cool_of(cell, res):
    return OE / f"{cell}_proper_{res // 1000}kb.cool"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--full-name", default=None,
                    help="output name of the full consensus (default consensus<N>)")
    ap.add_argument("--tmp", required=True, help="staging folder for the per-chunk arrays")
    a = ap.parse_args()
    k = a.res // 1000
    N = len(a.cells)
    full = OE / f"{a.full_name or f'consensus{N}'}_proper_{k}kb.cool"
    loo = {c: OE / f"consensus_loo_{c}_proper_{k}kb.cool" for c in a.cells}
    todo = {"full": full, **loo}
    if all(p.exists() for p in todo.values()):
        print("all outputs exist")
        return
    clrs = [cooler.Cooler(str(cool_of(c, a.res))) for c in a.cells]
    grid = cooler.Cooler(str(cool_of("gm12878", a.res)))
    bins = grid.bins()[:][["chrom", "start", "end"]]
    offs = []
    for c in clrs:
        with h5py.File(c.filename, "r") as h:
            offs.append((h[c.root] if c.root != "/" else h)["indexes/bin1_offset"][:])
    chroms = [c for c in grid.chromnames if all(c in x.chromnames for x in clrs)]
    # pixel ranges per chromosome and map; total pixel count
    plan, total = [], 0
    for chrom in chroms:
        nb = int(grid.extent(chrom)[1] - grid.offset(chrom))
        rng = []
        for cell, c, o in zip(a.cells, clrs, offs):
            fo = int(c.offset(chrom))
            if int(c.extent(chrom)[1]) - fo != nb:
                raise SystemExit(f"{cell} {chrom}: bin count differs from the grid")
            rng.append((fo, int(o[fo]), int(o[fo + nb])))
        n = {r[2] - r[1] for r in rng}
        if len(n) != 1:
            raise SystemExit(f"{chrom}: pixel counts differ {n}")
        n = n.pop()
        plan.append((chrom, nb, rng, n, total))
        total += n
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    names = ["full"] + list(a.cells)
    chunks = []                        # (ids file, {name: values file}) in pixel order
    t0 = time.time()
    for chrom, nb, rng, n, base in plan:
        go = int(grid.offset(chrom))
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            tag = f"{chrom}_{s}"
            files = {nm: tmp / f"v_{nm}_{tag}.npy" for nm in names}
            fid = tmp / f"ids_{tag}.npy"
            chunks.append((fid, files))
            if fid.exists() and all(f.exists() for f in files.values()):
                continue
            b1 = b2 = None
            vals = np.empty((N, e - s))
            for ci, (cell, c, (fo, p0, _)) in enumerate(zip(a.cells, clrs, rng)):
                px = c.pixels()[p0 + s:p0 + e]
                i, j = px["bin1_id"].values - fo, px["bin2_id"].values - fo
                if b1 is None:
                    b1, b2 = i, j
                elif not (np.array_equal(i, b1) and np.array_equal(j, b2)):
                    raise SystemExit(f"{cell} {chrom}: pixel table differs")
                vals[ci] = px["count"].values
            acc = vals.sum(axis=0)
            np.save(files["full"], (acc / N).astype(np.float32))
            for ci, cell in enumerate(a.cells):
                np.save(files[cell], ((acc - vals[ci]) / (N - 1)).astype(np.float32))
            np.save(fid, np.stack([b1 + go, b2 + go]).astype(np.int64))
        print(f"  read {chrom}: {n:,} pixels ({(time.time() - t0) / 60:.1f} min)", flush=True)

    def pixels(nm):
        for fid, files in chunks:
            ij = np.load(fid)
            yield pd.DataFrame({"bin1_id": ij[0], "bin2_id": ij[1],
                                "count": np.load(files[nm])})

    for name, path in todo.items():
        if path.exists():
            continue
        ci = None if name == "full" else a.cells.index(name)
        cells = a.cells if ci is None else [c for c in a.cells if c != name]
        part = path.with_name(path.name + ".part")
        part.unlink(missing_ok=True)
        cooler.create_cooler(str(part), bins=bins, pixels=pixels(name),
                             dtypes={"count": np.float32}, ordered=True)
        part.rename(path)
        Path(str(path) + ".json").write_text(json.dumps({
            "res": a.res, "cells": cells, "inputs": [str(cool_of(c, a.res)) for c in cells],
            "grid": grid.filename,
            "method": ("pixel-wise mean of oe_proper.py maps" if ci is None else
                       f"(sum of {N} maps - {name}) / {N - 1}, one pass "
                       "(scripts/consensus_loo.py)")}, indent=1))
        print(f"-> {path} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    shutil.rmtree(tmp)
    print("CONSENSUS_LOO_DONE", flush=True)


if __name__ == "__main__":
    main()
