"""Mean DNA methylation as a model input channel (registry section 27).

One ENCODE4 GRCh38 WGBS CpG methylation bed per cell (the preferred-default
replicate), binned to a 1 kb bigWig ``Meth.bigwig`` under
``<HM>/meth/<cell>/``, which runs pass as ``samples[].extra_bigwig_dirs``.

Bin value = 1 + coverage-weighted methylated fraction of the CpGs in the bin
(range 1 to 2). Bins with fewer than MIN_COV CpG reads get 1 + the
genome-wide mean of that cell, so a raw zero stays reserved for "no
methylation data" (the absent value of track dropout and of cells without
WGBS). Cells without GRCh38 WGBS on ENCODE (HCT116, MCF-7 and HFF have hg19
RRBS only, NHEK, PC-9 and HUVEC none) carry the channel as missing.

    python scripts/methylation.py --stage fetch build [--cells gm12878 k562]
"""
from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
MD = HM / "meth"
RAW = MD / "raw"
BIN = 1000
MIN_COV = 5
TRACK = "Meth"
# cell -> ENCODE file accession (methylation state at CpG, GRCh38, preferred default)
FILES = {
    "gm12878": "ENCFF570TIL",    # ENCSR890UQO
    "imr90": "ENCFF404JRI",      # ENCSR888FON
    "h1": "ENCFF434CNG",         # ENCSR617FKV
    "hepg2": "ENCFF453UDK",      # ENCSR881XOU
    "a549": "ENCFF948WVD",       # ENCSR481JIW
    "gm23248": "ENCFF170XYJ",    # ENCSR440MLE
    "cd14mono": "ENCFF689TNG",   # ENCSR017BUL
    "k562": "ENCFF660IHA",       # ENCSR765JPC (held out)
    "hmec": "ENCFF478RLX",       # ENCSR656TQD (held out; hmec and hmec_deep)
}
ALIAS = {"hmec_deep": "hmec"}
CHROM_SIZES_FROM = HM / "hg38_imr90" / "CTCF.bigwig"


def meth_dir(cell: str) -> Path | None:
    """The cell's methylation folder, None for a cell without WGBS."""
    c = ALIAS.get(cell, cell)
    return MD / c if c in FILES else None


def has_meth(cell: str) -> bool:
    d = meth_dir(cell)
    return d is not None and (d / f"{TRACK}.bigwig").exists()


def raw_path(cell: str) -> Path:
    return RAW / f"{FILES[cell]}.bed.gz"


def fetch(cell: str) -> None:
    out = raw_path(cell)
    if out.exists():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    acc = FILES[cell]
    url = f"https://www.encodeproject.org/files/{acc}/@@download/{acc}.bed.gz"
    part = out.with_name(out.name + ".part")
    for attempt in range(5):
        rc = subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", str(part), url]).returncode
        if rc == 0:
            part.rename(out)
            print(f"[fetch] {cell} {acc} {out.stat().st_size >> 20} MB", flush=True)
            return
        time.sleep(30 * (attempt + 1))
    raise SystemExit(f"[fetch] {cell}: curl failed for {url}")


def chrom_sizes() -> dict[str, int]:
    import pyBigWig
    bw = pyBigWig.open(str(CHROM_SIZES_FROM))
    try:
        return {c: n for c, n in bw.chroms().items() if "_" not in c and c != "chrM"}
    finally:
        bw.close()


def bin_counts(path: Path, sizes: dict[str, int]):
    """Per 1 kb bin: methylated and total CpG read counts."""
    import pandas as pd
    meth = {c: np.zeros(n // BIN + 1) for c, n in sizes.items()}
    cov = {c: np.zeros(n // BIN + 1) for c, n in sizes.items()}
    with gzip.open(path, "rt") as fh:
        for ch in pd.read_csv(fh, sep="\t", header=None, usecols=[0, 1, 9, 10],
                              names=["chrom", "start", "cov", "pct"],
                              dtype={"chrom": str, "start": np.int64, "cov": np.float64,
                                     "pct": np.float64}, chunksize=5_000_000):
            for c, g in ch.groupby("chrom", sort=False):
                if c not in meth:
                    continue
                b = (g["start"].values // BIN).clip(0, len(meth[c]) - 1)
                np.add.at(cov[c], b, g["cov"].values)
                np.add.at(meth[c], b, g["cov"].values * g["pct"].values / 100.0)
    return meth, cov


def bin_values(meth: dict, cov: dict) -> dict[str, np.ndarray]:
    """1 + methylated fraction; low-coverage bins get 1 + the genome mean."""
    tot_m = sum(m[c >= MIN_COV].sum() for m, c in zip(meth.values(), cov.values()))
    tot_c = sum(c[c >= MIN_COV].sum() for c in cov.values())
    mean = tot_m / max(tot_c, 1.0)
    out = {}
    for ch in meth:
        frac = np.full(len(meth[ch]), mean)
        ok = cov[ch] >= MIN_COV
        frac[ok] = meth[ch][ok] / cov[ch][ok]
        out[ch] = (1.0 + frac).astype(np.float32)
    return out


def build(cell: str) -> None:
    import pyBigWig
    d = MD / cell
    dst = d / f"{TRACK}.bigwig"
    if dst.exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    sizes = chrom_sizes()
    vals = bin_values(*bin_counts(raw_path(cell), sizes))
    tmp = d / f"{TRACK}.bigwig.part"
    bw = pyBigWig.open(str(tmp), "w")
    bw.addHeader(list(sizes.items()))
    for ch, n in sizes.items():
        v = vals[ch][: (n + BIN - 1) // BIN]
        # the last bin may run past the chromosome end: write it clipped
        full = n // BIN
        if full:
            bw.addEntries(ch, 0, values=v[:full].tolist(), span=BIN, step=BIN)
        if n % BIN:
            bw.addEntries([ch], [full * BIN], ends=[n], values=[float(v[full])])
    bw.close()
    tmp.rename(dst)
    cov_frac = np.mean(np.concatenate([vals[c] for c in vals]))
    print(f"[build] {cell}: {dst} (mean bin value {cov_frac:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", nargs="+", choices=["fetch", "build"], required=True)
    ap.add_argument("--cells", nargs="*", default=list(FILES))
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    cells = [ALIAS.get(c, c) for c in a.cells]
    if "fetch" in a.stage:
        with ProcessPoolExecutor(a.jobs) as ex:
            list(ex.map(fetch, cells))
    if "build" in a.stage:
        with ProcessPoolExecutor(a.jobs) as ex:
            list(ex.map(build, cells))


if __name__ == "__main__":
    sys.exit(main())
