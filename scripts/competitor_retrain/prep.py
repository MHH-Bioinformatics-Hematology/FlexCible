"""Preparation of targets, quantile maps and Epiphany inputs (run on the remote computer).

    python prep.py --stage targets qmaps epi --cells ALL --workers 16
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

EPI_TRACKS = ["DNase", "H3K27ac", "H3K4me3", "H3K27me3", "CTCF"]      # channel order of data_load()
CHROMS = C.HELD + C.VALID + C.TRAIN


def epi_file(cell, chrom):
    return C.local_or_share("epi_inputs", cell, chrom)


def genome_mean(path):
    import pyBigWig
    h = pyBigWig.open(str(path)).header()
    return h["sumData"] / h["nBasesCovered"]


def epi_track_path(cell, t):
    return C.dnase_path(cell) if t == "DNase" else C.mark_path(cell, t)


def job_cofeat(cell, chrom):
    """(2, size) float16: C.Origami feature channels of the whole chromosome (quantile-mapped CTCF,
    log(1 + mapped DNase)), read by the training and prediction scripts instead of the bigWigs."""
    import os
    import pyBigWig
    f = C.cofeat_file(cell, chrom)
    if f.exists():
        return cell, chrom
    size = C.chrom_size(chrom)
    qm = C.co_qmaps(cell)
    ctcf, acc = C.co_features(pyBigWig.open(str(C.mark_path(cell, "CTCF"))),
                              pyBigWig.open(str(C.dnase_path(cell))), qm, chrom, 0, size)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(f) + ".tmp.npy", np.stack([ctcf, acc]).astype(np.float16))
    os.replace(str(f) + ".tmp.npy", f)
    return cell, chrom


def job_targets(cell, chrom):
    C.make_band(cell, chrom)
    return cell, chrom


def job_qmap(cell):
    C.co_qmaps(cell)
    return cell


def job_epi(cell, chrom):
    """(5, n100) float32: 100 bp means of the five tracks, each divided by its genome mean (so
    that the genome mean is 1, the scale of the RPGC-like tracks of the released model)."""
    import pyBigWig
    f = epi_file(cell, chrom)
    if f.exists():
        return cell, chrom
    size = C.chrom_size(chrom)
    n = size // 100
    A = np.zeros((len(EPI_TRACKS), n), np.float32)
    for k, t in enumerate(EPI_TRACKS):
        p = epi_track_path(cell, t)
        bw = pyBigWig.open(str(p))
        v = np.zeros(n, np.float64)
        step = 10_000_000                     # 1 bp values in chunks, 100 bp means over covered bases
        for s0 in range(0, n * 100, step):
            e = min(s0 + step, n * 100)
            r = np.array(bw.values(chrom, s0, e), np.float32).reshape(-1, 100)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v[s0 // 100:e // 100] = np.nan_to_num(np.nanmean(r, axis=1))
        A[k] = v / genome_mean(p)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(f) + ".tmp.npy", A)
    import os
    os.replace(str(f) + ".tmp.npy", f)
    return cell, chrom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True, choices=["targets", "qmaps", "epi", "cofeats"])
    ap.add_argument("--cells", nargs="*", default=C.ALL_CELLS)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        for st in a.stage:
            t0 = time.time()
            if st == "targets":
                fs = [ex.submit(job_targets, c, ch) for c in a.cells for ch in CHROMS]
            elif st == "qmaps":
                fs = [ex.submit(job_qmap, c) for c in a.cells]
            elif st == "cofeats":
                fs = [ex.submit(job_cofeat, c, ch) for c in a.cells for ch in CHROMS]
            else:
                fs = [ex.submit(job_epi, c, ch) for c in a.cells for ch in CHROMS]
            for f in fs:
                f.result()
            print(f"{st}: {len(fs)} jobs, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
