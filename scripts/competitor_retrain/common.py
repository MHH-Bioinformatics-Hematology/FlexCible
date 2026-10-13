"""Shared definitions of the competitor retraining (registry section 73).

C.Origami and Epiphany retrained on our cell lines and our chromosome split, each with its own
architecture and its own native input set:

  C.Origami  hg38 sequence (one-hot, 5 channels) + CTCF + chromatin accessibility, 2 097 152 bp
             window, 256 x 256 output (8192 bp pixels), input of every cell mapped onto the
             distributions of the released IMR-90 tracks (quantile map; the "q" variant of
             registry section 72), accessibility = ENCODE DNase-seq (ATAC bigWigs exist for 7 of
             our 15 cells only; DNase for all)
  Epiphany   DNase, H3K27ac, H3K4me3, H3K27me3, CTCF at 100 bp (means of the bigWig), 1.4 Mb
             convolutional window per 10 kb row, 200 rows per step, 100 diagonals (990 kb reach)

Targets are our proper O/E maps (scripts/oe_proper.py: expected over mappable pairs, log O/E,
per-diagonal centring, clip +-2, Gaussian sigma 2 bins) at 10 kb, the grid the scoring of
registry section 58 samples every competitor on. Epiphany is trained on the 10 kb pixels as
they are; C.Origami on the exact area average of the 10 kb pixels over its 8192 bp bins.

Split (identical for both): training chromosomes 1, 4-10, 12, 13, 15, 16, 19, 20, 22 (all but
2, 3, 11, 14, 17, X and the two validation chromosomes), validation chromosomes 18 and 21,
test chromosomes 2, 3, 11, 14, 17 (the chromosomes held out of every chrho model).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
DATA = Path(os.environ.get("HICMULTI_DATA", str(Path.home() / "data/hic-multiome")))
OE = HM / "crosscell_oe"
RT = HM / "competitor_retrain"
HERE = Path(__file__).resolve().parent

HELD = ["chr2", "chr3", "chr11", "chr14", "chr17"]
VALID = ["chr18", "chr21"]
TRAIN = [f"chr{i}" for i in (1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 19, 20, 22)]

# training cells: every cell of the 12-cell family whose native inputs of both competitors exist
# (GM23248 has no CTCF track); held-out cell types: never seen by any model of ours
TRAIN_CELLS = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "mcf7",
               "cd14mono", "pc9"]
TEST_CELLS = ["k562", "hmec_deep", "huvec_deep"]
ALL_CELLS = TRAIN_CELLS + TEST_CELLS
# hic_measures cell name -> (bigWig directory, DNase file stem)
_TRACK_CELL = {"hmec_deep": "hmec", "huvec_deep": "huvec"}


def track_dir(cell: str) -> Path:
    c = _TRACK_CELL.get(cell, cell)
    return DATA / "hg38_gm12878" if c == "gm12878" else HM / f"hg38_{c}"


def dnase_path(cell: str) -> Path:
    return RT / "dnase" / f"{_TRACK_CELL.get(cell, cell)}.bigWig"


def mark_path(cell: str, mark: str) -> Path:
    return track_dir(cell) / f"{mark}.bigwig"


def target_cool(cell: str, res: int = 10000) -> Path:
    return OE / f"{cell}_proper_{res // 1000}kb.cool"


CHROM_SIZES = {}


def chrom_size(chrom: str) -> int:
    if not CHROM_SIZES:
        import cooler
        c = cooler.Cooler(str(target_cool("imr90")))
        CHROM_SIZES.update({k: int(v) for k, v in c.chromsizes.items()})
    return CHROM_SIZES[chrom]


# --------------------------------------------------------------------------- #
# sequence (codes a0 t1 c2 g3 n4), cached once per chromosome on the share
# --------------------------------------------------------------------------- #
SEQ_CACHE = [HM / "runs/aml_competitors_loops/seq", RT / "seq"]
CO_ROOT = Path.home() / "src/C.Origami/corigami_data/data/hg38"
FASTA_GZ = RT / "fasta"


def hg38_seq(chrom: str) -> np.ndarray:
    for d in SEQ_CACHE:
        f = d / f"{chrom}.npy"
        if f.exists():
            return np.load(f, mmap_mode="r")
    import gzip
    src = next((p for p in (FASTA_GZ / f"{chrom}.fa.gz", CO_ROOT / "dna_sequence" / f"{chrom}.fa.gz")
                if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"no sequence for {chrom}")
    raw = gzip.open(src, "rb").read()
    raw = raw[raw.find(b"\n") + 1:].replace(b"\n", b"").lower()
    lut = np.full(256, 4, np.uint8)
    for k, ch in enumerate(b"atcg"):
        lut[ch] = k
    out = lut[np.frombuffer(raw, np.uint8)]
    f = SEQ_CACHE[-1] / f"{chrom}.npy"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f.with_suffix(".tmp.npy"), out)
    f.with_suffix(".tmp.npy").rename(f)
    return np.load(f, mmap_mode="r")


# --------------------------------------------------------------------------- #
# targets: banded 10 kb O/E arrays
# --------------------------------------------------------------------------- #
BAND = 212          # diagonals 0..211 of the 10 kb O/E (the cool holds 200)


def band_file(cell: str, chrom: str) -> Path:
    return RT / "targets" / f"{cell}_{chrom}.npy"


def make_band(cell: str, chrom: str) -> Path:
    """(n, BAND) float16 array: O/E(i, i + d) of the 10 kb target, NaN = missing (a stored 0 is
    missing, as in hic_measures.load_band)."""
    import cooler
    f = band_file(cell, chrom)
    if f.exists():
        return f
    clr = cooler.Cooler(str(target_cool(cell)))
    n = int(np.ceil(clr.chromsizes[chrom] / clr.binsize))
    o = clr.offset(chrom)
    px = clr.matrix(balance=False, as_pixels=True).fetch(chrom)
    i = px.bin1_id.values - o
    d = px.bin2_id.values - px.bin1_id.values
    v = px["count"].values.astype(np.float32)
    ok = (d >= 0) & (d < BAND) & (v != 0) & np.isfinite(v)
    B = np.full((n, BAND), np.nan, np.float32)
    B[i[ok], d[ok]] = v[ok]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(f) + ".tmp.npy", B.astype(np.float16))
    os.replace(str(f) + ".tmp.npy", f)
    return f


def load_band(cell: str, chrom: str) -> np.ndarray:
    return np.load(make_band(cell, chrom), mmap_mode="r")


# --------------------------------------------------------------------------- #
# C.Origami input conventions
# --------------------------------------------------------------------------- #
WIN = 2_097_152
PIX = 8192
NPIX = 256
CO_FEATS = RT / "reference"      # copies of the released IMR-90 ctcf_log2fc.bw and atac.bw
REF_CTCF = CO_FEATS / "ctcf_log2fc.bw"
REF_ATAC = CO_FEATS / "atac.bw"
QMAP = RT / "qmap"


def quantile_map(src: str, ref: str, cache: Path):
    """(x, y) of the monotone map from the 1 bp value distribution of bigWig `src` (600 random
    25 kb segments of chr1-22, the same segments for source and reference) onto that of `ref`;
    applied with np.interp (registry section 72)."""
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    import pyBigWig
    rg = np.random.default_rng(0)
    sizes = pyBigWig.open(str(ref)).chroms()
    chroms = [f"chr{i}" for i in range(1, 23)]
    w = np.array([sizes[c] for c in chroms], float)
    segs = [(chroms[k], int(rg.integers(0, sizes[chroms[k]] - 25000)))
            for k in rg.choice(len(chroms), 600, p=w / w.sum())]

    def sample(f):
        bw = pyBigWig.open(str(f))
        v = np.concatenate([np.array(bw.values(c, s0, s0 + 25000), dtype=float) for c, s0 in segs])
        return v[np.isfinite(v)]

    q = np.concatenate([np.linspace(0, 99, 100), np.linspace(99, 99.99, 100), [100]]) / 100
    x = np.quantile(sample(src), q)
    y = np.quantile(sample(ref), q)
    x, ix = np.unique(x, return_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache) + ".tmp.npz", x=x, y=y[ix])
    os.replace(str(cache) + ".tmp.npz", cache)
    return x, y[ix]


def co_qmaps(cell: str):
    """Quantile maps of the CTCF and the accessibility track of a cell onto the IMR-90 references."""
    return (quantile_map(mark_path(cell, "CTCF"), REF_CTCF, QMAP / f"{cell}_ctcf.npz"),
            quantile_map(dnase_path(cell), REF_ATAC, QMAP / f"{cell}_dnase.npz"))


LOCAL = Path(os.environ["COMPRT_LOCAL"]) if os.environ.get("COMPRT_LOCAL") else None


def local_or_share(kind: str, cell: str, chrom: str) -> Path:
    """Large per-chromosome input arrays: training and validation chromosomes on the local disk of
    the training machine ($COMPRT_LOCAL, read at every step), test chromosomes (read once per
    prediction) and everything when COMPRT_LOCAL is unset on the share."""
    base = LOCAL if (LOCAL is not None and chrom not in HELD) else RT
    return base / kind / f"{cell}_{chrom}.npy"


def cofeat_file(cell: str, chrom: str) -> Path:
    return local_or_share("co_feats", cell, chrom)


def bw_values(bw, chrom: str, s: int, e: int) -> np.ndarray:
    """bigWig values of [s, e) at 1 bp; positions outside the chromosome and NaN = 0."""
    size = bw.chroms()[chrom]
    out = np.zeros(e - s, np.float32)
    a, b = max(s, 0), min(e, size)
    if b > a:
        out[a - s:b - s] = np.nan_to_num(np.array(bw.values(chrom, a, b), np.float32))
    return out


def co_features(bw_ctcf, bw_acc, qm, chrom: str, s: int, e: int):
    """C.Origami feature channels of [s, e): quantile-mapped CTCF (as in ctcf_log2fc.bw) and
    log(1 + mapped accessibility) (norm 'log' of the original training module)."""
    (cx, cy), (ax, ay) = qm
    c = np.interp(bw_values(bw_ctcf, chrom, s, e), cx, cy).astype(np.float32)
    a = np.log(np.maximum(np.interp(bw_values(bw_acc, chrom, s, e), ax, ay), 0) + 1).astype(np.float32)
    return c, a


def area_weights(s: int, nsrc: int, src_res: int = 10000, out_res: int = PIX, nout: int = NPIX
                 ) -> tuple[np.ndarray, int]:
    """Matrix W (nout, nsrc') of overlap fractions of the output bins [s + k * out_res, ...) with
    the source bins of size src_res, and the index of the first source bin."""
    b0 = s // src_res
    b1 = -(-(s + nout * out_res) // src_res)
    W = np.zeros((nout, b1 - b0), np.float32)
    for k in range(nout):
        lo, hi = s + k * out_res, s + (k + 1) * out_res
        for j in range(lo // src_res, -(-hi // src_res)):
            ov = min(hi, (j + 1) * src_res) - max(lo, j * src_res)
            if ov > 0:
                W[k, j - b0] = ov / out_res
    return W, b0


def co_target(band: np.ndarray, s: int):
    """(256, 256) O/E of the 8192 bp bins starting at bp s (area average of the 10 kb O/E over the
    valid source pixels) and the (256, 256) validity mask (>= 50% of the area measured). `band`:
    (n, BAND) array of load_band."""
    n = band.shape[0]
    W, b0 = area_weights(s, n)
    m = W.shape[1]
    b1 = b0 + m
    X = np.full((m, m), np.nan, np.float32)
    for d in range(min(BAND, m)):
        r = np.arange(max(0, -b0), min(m - d, n - b0))      # source rows b0 + r in [0, n)
        if len(r) == 0:
            continue
        v = np.asarray(band[b0 + r, d], np.float32)
        X[r, r + d] = v
        X[r + d, r] = v
    valid = np.isfinite(X).astype(np.float32)
    num = W @ np.nan_to_num(X) @ W.T
    den = W @ valid @ W.T
    ok = den >= 0.5
    out = np.where(ok, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return out, ok


# --------------------------------------------------------------------------- #
# writing band arrays as cools (the format the scoring reads: competitor_oe/<name>_<chrom>.cool)
# --------------------------------------------------------------------------- #
COMP_OE = HM / "runs/benchmark/competitor_oe"
PRE_COUNTS = RT / "pre_counts"


def band_to_cool(M: np.ndarray, chrom: str, res: int, path: Path, dtype=np.float32) -> Path:
    """M: (n, w) array, M[i, d] = value of the pixel (i, i + d), NaN = absent; pixels with a value
    are written (a stored 0 is read as missing by the scoring, as in oe_proper outputs)."""
    import cooler
    import pandas as pd
    n, w = M.shape
    size = chrom_size(chrom)
    starts = np.arange(0, size, res)
    bins = pd.DataFrame({"chrom": chrom, "start": starts, "end": np.minimum(starts + res, size)})
    assert len(bins) >= n - 1
    ii, dd = np.nonzero(np.isfinite(M))
    j = ii + dd
    keep = j < len(bins)
    px = pd.DataFrame({"bin1_id": ii[keep].astype(np.int64), "bin2_id": j[keep].astype(np.int64),
                       "count": M[ii[keep], dd[keep]].astype(dtype)})
    px = px.sort_values(["bin1_id", "bin2_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    cooler.create_cooler(str(tmp), bins=bins, pixels=px, dtypes={"count": dtype}, ordered=True, symmetric_upper=True)
    os.replace(tmp, path)
    return path
