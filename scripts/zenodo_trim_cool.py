"""Deposit copies of the O/E cool maps: optional distance band cut and stronger HDF5 compression.

The source files are never modified. A copy is written with the pixels whose diagonal offset
(bin2_id - bin1_id) is at most --max-diag (default: keep all) and with byte-shuffle + gzip 9 in
larger chunks, which is what makes the copy smaller. Values are copied unchanged, in the dtype of
the source (float32 for the O/E maps, float64 for the Akita-grid consensus).
The sidecar <name>.cool.json (cells of a consensus) is copied next to the output.

The band cut is lossless for the model when --max-diag >= window size - 1 at that bin size
(2 kb: 512-bin window -> 511; the 25, 10 and 5 kb windows cover the stored 2 Mb band, so no cut).

    python scripts/zenodo_trim_cool.py --input X.cool --output Y.cool [--max-diag 511]
    python scripts/zenodo_trim_cool.py --check X.cool Y.cool [--max-diag 511]   # verify a copy

--check compares the two files: identical bins and chromosomes, and identical pixel values for
every pixel within the band (and, without a band, for every pixel).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cooler
import h5py
import numpy as np
import pandas as pd

H5OPTS = dict(compression="gzip", compression_opts=9, shuffle=True)   # chunk shape chosen by h5py per dataset
CHUNK = 50_000_000


def pixel_chunks(src: cooler.Cooler, max_diag: int | None):
    n = src.info["nnz"]
    with src.open("r") as h:
        p = h["pixels"]
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            b1 = p["bin1_id"][s:e]
            b2 = p["bin2_id"][s:e]
            v = p["count"][s:e]
            if max_diag is not None:
                m = (b2 - b1) <= max_diag
                b1, b2, v = b1[m], b2[m], v[m]
            yield pd.DataFrame({"bin1_id": b1, "bin2_id": b2, "count": v})


def repack_pixels(tmp: str, out: str):
    """Copy chroms, bins, indexes and attributes from tmp and rewrite the three pixel columns with
    large chunks (byte shuffle + gzip 9), which is where the size is; h5py's default chunks for the
    pixel columns are small and compress poorly."""
    with h5py.File(tmp, "r") as hi, h5py.File(out, "w") as ho:
        for k, v in hi.attrs.items():
            ho.attrs[k] = v
        for g in ("chroms", "bins", "indexes"):
            hi.copy(g, ho)
        pi = hi["pixels"]
        po = ho.create_group("pixels")
        n = pi["bin1_id"].shape[0]
        for col in ("bin1_id", "bin2_id", "count"):
            d = po.create_dataset(col, shape=(n,), dtype=pi[col].dtype, chunks=(max(1, min(500_000, n)),),
                                  compression="gzip", compression_opts=9, shuffle=True)
            for s in range(0, n, CHUNK):
                e = min(s + CHUNK, n)
                d[s:e] = pi[col][s:e]


def trim(inp: str, out: str, max_diag: int | None):
    src = cooler.Cooler(inp)
    bins = src.bins()[:]
    with src.open("r") as h:
        count_dtype = h["pixels"]["count"].dtype
    t0 = time.time()
    tmp = out + ".part"
    cooler.create_cooler(tmp, bins, pixel_chunks(src, max_diag), dtypes={"count": count_dtype},
                         ordered=True, symmetric_upper=True, h5opts=H5OPTS, assembly=src.info.get("genome-assembly"),
                         metadata=dict(src.info.get("metadata") or {}, zenodo_copy_of=Path(inp).name,
                                       max_diag=max_diag if max_diag is not None else "all"))
    repack_pixels(tmp, out + ".repack")
    Path(tmp).unlink()
    Path(out + ".repack").rename(out)
    side = Path(inp + ".json")
    if side.exists():
        shutil.copyfile(side, out + ".json")
    gi, go = Path(inp).stat().st_size / 1e9, Path(out).stat().st_size / 1e9
    print(f"{Path(inp).name}: {gi:.2f} GB -> {go:.2f} GB ({go / gi:.2f}), max_diag={max_diag}, {time.time() - t0:.0f} s")


def check(a: str, b: str, max_diag: int | None):
    ca, cb = cooler.Cooler(a), cooler.Cooler(b)
    assert ca.binsize == cb.binsize, "bin size differs"
    assert ca.bins()[:].equals(cb.bins()[:]), "bins differ"
    assert list(ca.chromnames) == list(cb.chromnames), "chromosomes differ"
    na, nb = ca.info["nnz"], cb.info["nnz"]
    with ca.open("r") as ha, cb.open("r") as hb:
        pa, pb = ha["pixels"], hb["pixels"]
        pos_b = 0
        for s in range(0, na, CHUNK):
            e = min(s + CHUNK, na)
            b1 = pa["bin1_id"][s:e]; b2 = pa["bin2_id"][s:e]; v = pa["count"][s:e]
            if max_diag is not None:
                m = (b2 - b1) <= max_diag
                b1, b2, v = b1[m], b2[m], v[m]
            k = len(b1)
            assert pos_b + k <= nb, "copy has fewer pixels than the band of the source"
            assert np.array_equal(pb["bin1_id"][pos_b:pos_b + k], b1), f"bin1 differs at {pos_b}"
            assert np.array_equal(pb["bin2_id"][pos_b:pos_b + k], b2), f"bin2 differs at {pos_b}"
            assert np.array_equal(pb["count"][pos_b:pos_b + k], v), f"values differ at {pos_b}"
            pos_b += k
        assert pos_b == nb, f"copy has {nb} pixels, band of the source has {pos_b}"
    print(f"check OK: {Path(b).name} holds exactly the {nb} source pixels with diag <= {max_diag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--max-diag", type=int, default=None, help="keep pixels with bin2 - bin1 <= this (default all)")
    ap.add_argument("--check", nargs=2, metavar=("SOURCE", "COPY"))
    a = ap.parse_args()
    if a.check:
        check(a.check[0], a.check[1], a.max_diag)
        return
    if not (a.input and a.output):
        sys.exit("--input and --output, or --check")
    if Path(a.output).exists():
        sys.exit(f"{a.output} exists; refusing to overwrite")
    trim(a.input, a.output, a.max_diag)


if __name__ == "__main__":
    main()
