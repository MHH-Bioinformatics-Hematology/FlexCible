"""Training data of the Akita retraining: sequence codes, targets, windows.

Stages
  seq      hg38 chromosomes 1-22 as uint8 codes (A0 C1 G2 T3 N4) in data/seq
  bands    our proper O/E maps (crosscell_oe/<cell>_proper_2kb.cool, 2000 bp
           grid) of the 12 training cells resampled onto Akita's 2048 bp grid,
           whole chromosomes as bands (bin, distance 0..447, cell), float16,
           NaN = unmappable. Resampling: bilinear in (bin, bin) space at the
           2048 bp bin centres, a pixel is kept when its four neighbours are
           measured (hic_measures.resample_band, the convention of the
           competitor blocks of registry sections 18, 37 and 58).
  windows  window tables of both splits (common.SPLITS), written to
           data/windows_<split>.tsv

Window rules. Windows of 1 048 576 bp start on the 2048 bp grid (targets =
the 448 central bins, 65 536 bp from the window start) and lie inside one
chromosome. Training windows: stride 229 376 bp (a quarter of the target
span), only chromosomes of the split's train list, wholly inside the merged
Akita training regions and not overlapping the Akita validation/test stretches
(the data.train_regions_bed and data.holdout_bed of our chrho models, so both
learn from the same genomic windows); validation windows: stride 458 752 bp on
the validation chromosomes, no region filter. Every window needs at most 20 %
N and at least 30 % finite target pixels (mean over cells).

    python scripts/akita_retrain/build_data.py --stage seq bands windows --jobs 4
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import common as C  # noqa: E402


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def seq_one(chrom: str) -> str:
    out = C.seq_dir() / f"{chrom}.npy"
    if out.exists():
        return chrom
    from pyfaidx import Fasta
    fa = Fasta(str(C.FASTA), as_raw=True, sequence_always_upper=True)
    s = np.frombuffer(str(fa[chrom][:]).encode("ascii"), dtype=np.uint8)
    lut = np.full(256, 4, np.uint8)
    for k, ch in enumerate("ACGT"):
        lut[ord(ch)] = k
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out) + ".tmp.npy", lut[s])
    Path(str(out) + ".tmp.npy").rename(out)
    return chrom


def bands_one(chrom: str) -> str:
    import cooler
    import hic_measures as hm
    out = C.band_dir() / f"{chrom}.npy"
    if out.exists():
        return chrom
    size = None
    mm = None
    for k, cell in enumerate(C.CELLS):
        path = C.OE / f"{cell}_proper_{C.SRC_RES // 1000}kb.cool"
        if size is None:
            size = int(cooler.Cooler(str(path)).chromsizes[chrom])
            n = int(np.ceil(size / C.BIN))
            C.band_dir().mkdir(parents=True, exist_ok=True)
            tmp = C.band_dir() / f"{chrom}.tmp.npy"
            mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                           shape=(n, C.NB, len(C.CELLS)))
        B = hm.load_band(path, chrom, n, C.NB - 1, C.BIN)    # bilinear at the bin centres when SRC_RES != BIN
        if B is None:
            raise SystemExit(f"{chrom} missing in {path}")
        mm[:, :, k] = B.astype(np.float16)
        del B
    mm.flush()
    del mm
    tmp.rename(out)
    return chrom


def read_bed(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, sep="\t", header=None, comment="#", usecols=[0, 1, 2],
                    names=["c", "s", "e"])
    rows = []
    for c, g in d.sort_values(["c", "s"]).groupby("c"):
        cs = ce = None
        for s, e in zip(g.s, g.e):
            if cs is None:
                cs, ce = s, e
            elif s <= ce:
                ce = max(ce, e)
            else:
                rows.append((c, cs, ce))
                cs, ce = s, e
        rows.append((c, cs, ce))
    return pd.DataFrame(rows, columns=["c", "s", "e"])


def windows():
    train_reg = read_bed(C.TRAIN_REGIONS_BED)
    hold = read_bed(C.HOLDOUT_BED)
    fai = pd.read_csv(str(C.FASTA) + ".fai", sep="\t", header=None, usecols=[0, 1],
                      names=["c", "n"])
    size = dict(zip(fai.c, fai.n))
    r, c = C.iu()
    cache = {}

    def quality(chrom, starts):
        if chrom not in cache:
            cache[chrom] = (np.load(C.seq_dir() / f"{chrom}.npy", mmap_mode="r"),
                            np.load(C.band_dir() / f"{chrom}.npy", mmap_mode="r"))
        seq, band = cache[chrom]
        nf, ff = [], []
        for s in starts:
            nf.append(float((seq[s:s + C.SEQ_LEN] == 4).mean()))
            b0 = (s + C.CROP) // C.BIN
            v = band[b0 + r, c - r]          # (NV, ncell) float16
            ff.append(float(np.isfinite(v).mean()))
        return np.array(nf), np.array(ff)

    for split, sp in C.SPLITS.items():
        rows = []
        for role, chroms in (("train", sp["train"]), ("val", sp["val"])):
            stride = C.STRIDE_TRAIN if role == "train" else C.STRIDE_EVAL
            for ch in chroms:
                starts = np.arange(0, size[ch] - C.SEQ_LEN + 1, stride)
                if role == "train":
                    t = train_reg[train_reg.c == ch]
                    h = hold[hold.c == ch]
                    keep = []
                    for s in starts:
                        e = s + C.SEQ_LEN
                        inside = ((t.s <= s) & (t.e >= e)).any()
                        over = ((h.s < e) & (h.e > s)).any()
                        keep.append(bool(inside and not over))
                    starts = starts[np.array(keep, bool)]
                if len(starts) == 0:
                    continue
                nf, ff = quality(ch, starts)
                ok = (nf <= 0.2) & (ff >= 0.3)
                for s, a, b, o in zip(starts, nf, ff, ok):
                    rows.append((split, role, ch, int(s), round(a, 4), round(b, 4), bool(o)))
        d = pd.DataFrame(rows, columns=["split", "role", "chrom", "start", "n_frac",
                                        "finite_frac", "keep"])
        d.to_csv(C.windows_file(split), sep="\t", index=False)
        k = d[d.keep]
        log(f"split {split}: " + ", ".join(
            f"{ro} {int((k.role == ro).sum())}/{int((d.role == ro).sum())} windows"
            for ro in ("train", "val")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True, choices=["seq", "bands", "windows"])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    for st in a.stage:
        if st in ("seq", "bands"):
            fn = seq_one if st == "seq" else bands_one
            with ProcessPoolExecutor(a.jobs) as ex:
                for ch in ex.map(fn, C.AUTOSOMES[::-1]):    # largest last: start small ones
                    log(f"{st} {ch} done")
        else:
            windows()


if __name__ == "__main__":
    main()
