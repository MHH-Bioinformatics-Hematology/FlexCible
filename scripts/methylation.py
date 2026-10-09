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

Version 2 (registry section 35): the hg19 ENCODE RRBS of HCT116, MCF-7 and HFF
(newborn foreskin fibroblast, a different sample of the tissue HFFc6 comes
from) are lifted to hg38 CpG by CpG with the UCSC hg19ToHg38 chain
(``hg19_counts``), both replicates summed, and binned like the WGBS cells;
``has_meth(cell, v2=True)`` then counts 10 of the 12 training cells (NHEK and
PC-9 have no ENCODE methylation data). The version 1 cell set stays as the
runs of section 27 saw it.

    python scripts/methylation.py --stage fetch build [--cells gm12878 k562]
"""
from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("HICMULTI_DATA", str(Path.home() / "data/hic-multiome")))   # local-home data tree
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
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
# version 2: hg19 RRBS (ENCODE bedMethyl, replicates 1 and 2 summed), lifted to hg38
HG19 = {
    "hct116": ["ENCFF001TMM", "ENCFF001TMN"],   # ENCSR000DFS
    "mcf7": ["ENCFF001TOZ", "ENCFF001TPA"],     # ENCSR939RXT
    "hff": ["ENCFF001TPS", "ENCFF001TPT"],      # ENCSR000DFG, foreskin fibroblast (newborn)
}
CHAIN19 = DATA / "chain/hg19ToHg38.over.chain.gz"
ALIAS = {"hmec_deep": "hmec"}
CHROM_SIZES_FROM = HM / "hg38_imr90" / "CTCF.bigwig"


def meth_dir(cell: str, v2: bool = False) -> Path | None:
    """The cell's methylation folder, None for a cell without methylation data
    (v2: the lifted hg19 RRBS cells count too)."""
    c = ALIAS.get(cell, cell)
    return MD / c if (c in FILES or (v2 and c in HG19)) else None


def has_meth(cell: str, v2: bool = False) -> bool:
    d = meth_dir(cell, v2)
    return d is not None and (d / f"{TRACK}.bigwig").exists()


def raw_path(cell: str) -> Path:
    return RAW / f"{FILES[cell]}.bed.gz"


def fetch(cell: str) -> None:
    if cell in HG19:
        return fetch19(cell)
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


def fetch19(cell: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for acc in HG19[cell]:
        out = RAW / f"{acc}.bed.gz"
        if out.exists():
            continue
        part = out.with_name(out.name + ".part")
        url = f"https://www.encodeproject.org/files/{acc}/@@download/{acc}.bed.gz"
        rc = subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", str(part), url]).returncode
        if rc:
            raise SystemExit(f"[fetch] {cell}: curl failed for {url}")
        part.rename(out)
        print(f"[fetch] {cell} {acc} {out.stat().st_size >> 10} kB", flush=True)


def chain_arrays(chain: Path = CHAIN19):
    """Per source chromosome numpy arrays of the chain blocks: starts, ends,
    target chromosome, target start, strand."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hicmultiome.liftover import LiftOver
    lo = LiftOver(chain)
    out = {}
    for c, v in lo.blocks.items():
        a = np.array([b[0] for b in v]), np.array([b[1] for b in v])
        out[c] = (a[0], a[1], np.array([b[2] for b in v], dtype=object),
                  np.array([b[3] for b in v]), np.array([b[4] for b in v]))
    return out


def lift_positions(arr: dict, chrom: str, pos: np.ndarray):
    """Vectorised CpG position lift: (target chromosome array, target position,
    ok mask); positions on unmapped sequence have ok False."""
    if chrom not in arr:
        return None, None, np.zeros(len(pos), bool)
    s, e, dst, dst0, strand = arr[chrom]
    i = np.searchsorted(s, pos, side="right") - 1
    ok = (i >= 0) & (pos < e[np.clip(i, 0, None)])
    i = np.clip(i, 0, None)
    off = pos - s[i]
    newpos = np.where(strand[i] == 1, dst0[i] + off, dst0[i] + (e[i] - s[i]) - off - 1)
    return dst[i], newpos, ok


def hg19_counts(accs: list[str], sizes: dict[str, int]):
    """Like bin_counts, for hg19 bedMethyl files lifted to hg38 (replicates summed)."""
    import pandas as pd
    arr = chain_arrays()
    meth = {c: np.zeros(n // BIN + 1) for c, n in sizes.items()}
    cov = {c: np.zeros(n // BIN + 1) for c, n in sizes.items()}
    n_in = n_lift = 0
    for acc in accs:
        with gzip.open(RAW / f"{acc}.bed.gz", "rt") as fh:
            skip = 1 if fh.readline().startswith("track") else 0      # UCSC track line
        with gzip.open(RAW / f"{acc}.bed.gz", "rt") as fh:
            df = pd.read_csv(fh, sep="\t", header=None, usecols=[0, 1, 9, 10], skiprows=skip,
                             names=["chrom", "start", "cov", "pct"],
                             dtype={"chrom": str, "start": np.int64, "cov": np.float64,
                                    "pct": np.float64})
        for c, g in df.groupby("chrom", sort=False):
            pos = g["start"].values.astype(np.int64)
            n_in += len(pos)
            dchr, npos, ok = lift_positions(arr, c, pos)
            if not ok.any():
                continue
            dchr, npos = dchr[ok], npos[ok]
            cv, pc = g["cov"].values[ok], g["pct"].values[ok]
            n_lift += int(ok.sum())
            for d in np.unique(dchr):
                if d not in meth:
                    continue
                m = dchr == d
                b = (npos[m] // BIN).clip(0, len(meth[d]) - 1)
                np.add.at(cov[d], b, cv[m])
                np.add.at(meth[d], b, cv[m] * pc[m] / 100.0)
    print(f"[lift] {accs}: {n_in} CpG records, {n_lift} lifted ({n_lift / max(n_in, 1):.3f})",
          flush=True)
    return meth, cov


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
    if cell in HG19:
        vals = bin_values(*hg19_counts(HG19[cell], sizes))
    else:
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
    ap.add_argument("--cells", nargs="*", default=list(FILES) + list(HG19))
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
