"""Multi-measure evaluation of predicted Hi-C maps (registry section 31).

One prediction is compared with one measured map on whole chromosomes by many
measures, so that no single number decides which model is best:

  count-space measures of the hyperparameter-scoring work (J. Wolff,
  "A novel metric for hyperparameter optimization in Hi-C matrix
  prediction", ~/src/hyperparameterScoringHiC, Zenodo 10.5281/zenodo.17296001),
  computed with the tools and settings used there:
    ENT3C          ENT3C run_all per pair (env ent3c), SUB_M_SIZE_FIX 500,
                   PHI_MAX 1e3, phi 1, no balancing weights (NormM 0)
    HiCRep         python hicrep (env hicrep), h 5, dBPMax 1 Mb, no
                   down-sampling
    GenomeDISCO, HiC-Spector, QuASAR-Rep
                   3DChromatin_ReplicateQC run_all (env chromatin_replicateQC)
                   with the parameter file of that work
    Pearson        Pearson AUC of Hi-cGAN's computePearsonCorrelation (per
                   distance Pearson of the counts over 1 Mb, area under the
                   curve on a [0, 1] distance axis), reimplemented exactly
    TAD Score MSE, TAD Fraction, TAD Fraction Exact Match
                   hicFindTADs of HiCExplorer 3.7.6 with the settings of
                   Hi-cGAN's optimizer (min depth 3 bins, max depth 10 bins,
                   step 1 bin, min boundary distance 100 kb, FDR,
                   threshold 0.5), on the prediction and on the target
  structure measures on the log O/E maps:
    Pearson on all band pixels, median Pearson over 1 Mb tiles (the
    registry's per-region metric), per-distance Pearson and its AUC, HiCRep
    SCC, insulation-profile Pearson, boundary recall and precision (hicFindTADs
    boundaries of the prediction against those of the target, 1 bin
    tolerance), loops against the hicDetectLoops 3.7.6 calls of the measured
    map (rank-matched donut recall, APA ratio, anchor AUROC; at 10 kb, outside
    the trainstretch set, also hicDetectLoops 3.7.6 on the prediction:
    recall, precision, F1),
    compartments (local E1 of 2 Mb windows against the target's, compartment
    contact preference per bin against the target's, sign agreement with the
    measured genome-wide E1, saddle strength).

Count space. Our maps are proper log O/E (scripts/oe_proper.py: balanced
counts, expected over mappable pairs, log, per-diagonal centring, clip,
Gaussian smoothing). Every map of a block (prediction, consensus, other cells,
replicate, the target itself) is converted back with the target's own
balanced expected and per-diagonal offset and scaled to the raw contact total
of the measured chromosome band: counts = exp(v + offset(d)) * E(d) * s. All
maps of a block share one curve, as in akita_counts_scoring.py; the measured
map itself (balanced counts at the same scale) scored against the converted
target ("measured map, balanced counts") is the cost of the round trip
through the clipped, smoothed O/E target.

Whole chromosomes. Predictions are tiled over whole chromosomes by
hicmultiome.predict (half-window stride, Hann-weighted averaging, test-time
reverse averaging). Every map is limited to the model window's band (2.0 Mb at
25 kb, 1.92 Mb at 10 kb), so a chromosome-wide E1 of a prediction can only be
built from contacts within the band (the compartment measures above).

Evaluation sets (blocks):
  heldout     K562, HMEC deep, HUVEC deep (no model saw these cells): whole
              chr2, chr12, chr18
  trainchrom  the 12 training cells on chr2, chr18: the chromosome-held-out
              residual reference model (cascade fold models cascfold_odd/even,
              the all12refres setup trained on the odd or on the even
              chromosomes; its out-of-fold maps are the level priors in
              crosscell_oe/cascade) against the baselines and the
              competitors; all12refres, trained on windows of these
              chromosomes (about 75% of each chromosome), is the in-training
              counterpart (flag chrom_in_training)
  trainstretch the 12 training cells on chr2, chr18, pixels restricted to the
              Akita valid+test stretches (data.holdout_bed, never overlapped
              by a training window of any cell; 23 to 27% of the chromosome,
              stretches of 1 to 11 Mb): the clean evaluation of the main
              models on training cells
ENT3C and the 3DChromatin measures run on a subset of the blocks (slow_tools:
all held-out-cell blocks at 25 kb, chr18 otherwise; minutes per map).
Competitors (predictions reused, never rerun; benchmark_blocks.sample_competitor
samples their O/E maps at our bin centres, hg19 maps through the liftover):
Hi-cGAN (GM12878, even chromosomes held out; GM12878 model on K562/HMEC/HUVEC;
IMR-90 models at 10 kb on chr2), C.Origami (IMR-90, chr2 test chromosome,
10 kb), HiC-Reg (GM12878 25 kb, within-chromosome CV maps, 1 Mb reach: own
block with a 1 Mb band) and Epiphany (GM12878 10 kb, chr17 held out, 990 kb
reach: own block). Akita exists only as 1 Mb, 2 kb Akita-space predictions,
so it has no whole-chromosome map here.

Combined score: the trained regressors of the hyperparameter-scoring work
(Revision_1/regression_model_results_revision_no_gpu, joblib, env
hyperparameter_gpu) applied to the nine count-space measures at 25 kb, with a
domain check against their training rows (stage combine).

Stages:
  predict   whole-chromosome predictions (CPU, --threads, lock files)
  expected  balanced expected, per-diagonal offset and raw scale per
            cell/res/chromosome
  score     every block: native measures, count cools, external tools
  combine   regressors (runs itself under the hyperparameter_gpu python)
  report    tables (results/hic_measures/*.csv), rankings, figures, STATUS.md

    python scripts/hic_measures.py --stage predict --threads 8
    python scripts/hic_measures.py --stage expected score --jobs 6
    python scripts/hic_measures.py --stage combine report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = HM / "crosscell_oe"
MR = HM / "runs/multicell_reference"
OUT = HM / "runs/hic_measures"
TABLES = REPO / "results/hic_measures"
ENVS = Path.home() / "miniconda3/envs"
PY = str(ENVS / "hicdiff/bin/python")
HX376 = ENVS / "__hicexplorer@3.7.6/bin"
REPQC = ENVS / "chromatin_replicateQC/bin/3DChromatin_ReplicateQC"
ENT3C = ENVS / "ent3c/bin/ENT3C"
HICREP_PY = ENVS / "hicrep/bin/python"
HYPER_PY = ENVS / "hyperparameter_gpu/bin/python"
HP = Path("/mnt/processed/wolffjoa/hyperparameter_publication")
REPQC_PARAMS = HP / "Revision_1/3dchromatin/parameters.txt"
REGRESSORS = HP / "Revision_1/regression_model_results_revision_no_gpu"
REGRESSOR_ROWS = HP / "Revision_1/merged_df.csv"
HOLDOUT_BED = Path.home() / "data/hic-multiome/akita_holdout_valid_test.bed"
REGIONS413 = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization"
                  "/akita_validation/sequences_test_list.tsv")

# the nine measures of the hyperparameter-scoring work, in its column order
NINE = ["ENT3C", "QuASAR-Rep", "HiC-Spector", "GenomeDISCO", "Pearson",
        "TAD Score MSE", "TAD Fraction", "TAD Fraction Exact Match", "HiCRep"]
LOWER_BETTER = {"TAD Score MSE"}
# TAD Fraction: 1 is ideal (as many boundaries as the target)

TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
           "gm23248", "mcf7", "cd14mono", "pc9"]
HELDOUT = ["k562", "hmec_deep", "huvec_deep"]
BIGWIG = {"k562": HM / "hg38_k562", "hmec_deep": HM / "hg38_hmec", "huvec_deep": HM / "hg38_huvec"}
WINDOW = {25000: 80, 10000: 192}
RESOLUTIONS = [25000, 10000]
CHROMS = {"heldout": ["chr2", "chr12", "chr18"], "trainchrom": ["chr2", "chr18"],
          "trainstretch": ["chr2", "chr18"]}

# models (run folder <name>_<k>kb in runs/multicell_reference); a model enters
# when its checkpoint exists at that resolution
MODELS = ["all12refres", "all12honest", "all12refin", "all12refres_drop", "all12honest_drop",
          "all12refres_refdrop", "all12refres_drop_hard", "all12refres_drop_varw",
          "all12refin_drop", "all12refin_drop_ctcfac", "all12deg_refres_drop", "all12multires"]
REFERENCE_FREE = {"all12honest", "all12honest_drop"}
CASCFOLD = "cascfold"          # chromosome-held-out all12refres setup (out-of-fold priors)

# best single other measured map per cell (registry section 15 tables)
BEST_OTHER = {
    25000: {"gm12878": "k562", "imr90": "gm23248", "hct116": "a549", "hff": "gm23248",
            "h1": "hct116", "hepg2": "a549", "a549": "hct116", "nhek": "hmec_deep",
            "gm23248": "hff", "mcf7": "nhek", "cd14mono": "gm12878", "pc9": "a549",
            "k562": "hct116", "hmec_deep": "nhek", "huvec_deep": "imr90"},
    10000: {"gm12878": "k562", "imr90": "gm23248", "hct116": "a549", "hff": "gm23248",
            "h1": "hct116", "hepg2": "hct116", "a549": "hct116", "nhek": "hmec_deep",
            "gm23248": "hff", "mcf7": "h1", "cd14mono": "gm12878", "pc9": "a549",
            "k562": "hct116", "hmec_deep": "nhek", "huvec_deep": "imr90"},
}
# an independent replicate map of the same cell (proper O/E), the ceiling
REPLICATE = {"gm12878": "intact_gm12878_ENCSR965OVH_r75_ice",
             "k562": "intact_k562_ENCSR676BEP_r5_ice",
             "imr90": "imr90_rep1_ice", "hmec_deep": "hmec_rep1_ice",
             "huvec_deep": "huvec_rep1_ice"}
# competitors per (set, cell, res): benchmark_competitors.COMPETITORS names
COMPETITORS = {
    ("heldout", "k562", 25000): ["hicgan_gm12878_on_k562_25kb"],
    ("heldout", "k562", 10000): ["hicgan_gm12878_on_k562_10kb"],
    ("heldout", "hmec_deep", 25000): ["hicgan_gm12878_on_hmec_25kb"],
    ("heldout", "hmec_deep", 10000): ["hicgan_gm12878_on_hmec_10kb"],
    ("heldout", "huvec_deep", 25000): ["hicgan_gm12878_on_huvec_25kb"],
    ("heldout", "huvec_deep", 10000): ["hicgan_gm12878_on_huvec_10kb"],
    ("trainchrom", "gm12878", 25000): ["hicgan_gm12878_25kb"],
    ("trainchrom", "gm12878", 10000): ["hicgan_gm12878_10kb"],
    ("trainchrom", "imr90", 10000): ["corigami_imr90_8kb", "hicgan_imr90_on_imr90_10kb",
                                     "hicgan_gm12878_on_imr90_10kb"],
}
# competitor blocks with a shorter band than our window
EXTRA_BLOCKS = [
    # (set, cell, res, chrom, band bins, competitors)
    ("hicreg", "gm12878", 25000, "chr2", 40, ["hicreg_gm12878_25kb"]),
    ("hicreg", "gm12878", 25000, "chr18", 40, ["hicreg_gm12878_25kb"]),
    ("epiphany", "gm12878", 10000, "chr17", 99, ["epiphany_gm12878_10kb"]),
]


def slow_tools(b: dict) -> bool:
    """ENT3C and the 3DChromatin measures (GenomeDISCO, HiC-Spector,
    QuASAR-Rep) cost 1 to 5 min per map on chr2 at 25 kb (3DChromatin works
    through its pairs serially) and several times that at 10 kb. They run on
    every held-out-cell block at 25 kb, on chr18 for the training-cell blocks
    at 25 kb, and on chr18 of the held-out cells at 10 kb."""
    if b["res"] == 25000:
        return b["set"] == "heldout" or b["chrom"] == "chr18"
    return b["set"] == "heldout" and b["chrom"] == "chr18"

COMP_LABEL = {"hicgan": "Hi-cGAN", "corigami": "C.Origami", "hicreg": "HiC-Reg",
              "epiphany": "Epiphany"}


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "hic_measures.log", "a") as fh:
        fh.write(line + "\n")


def claim(p: Path) -> bool:
    """Atomic lock next to an output, so several workers share one queue."""
    lock = Path(str(p) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release(p: Path) -> None:
    Path(str(p) + ".lock").unlink(missing_ok=True)


# ===================================================================== #
# pure functions on banded maps
#
# A chromosome map is stored as a band B[i, d] = value of pixel (i, i + d),
# d = 0..D, NaN = not measured / not predicted.
# ===================================================================== #
def band_from_pixels(i, j, v, n: int, D: int) -> np.ndarray:
    i = np.asarray(i); j = np.asarray(j)
    a, b = np.minimum(i, j), np.maximum(i, j)
    d = b - a
    k = (d <= D) & (a >= 0) & (b < n)
    B = np.full((n, D + 1), np.nan, np.float64)
    B[a[k], d[k]] = np.asarray(v, float)[k]
    return B


def band_to_dense(B: np.ndarray, s: int, L: int, fill=np.nan) -> np.ndarray:
    """Dense symmetric L x L block starting at bin s (pixels beyond the band = fill)."""
    n, Dp = B.shape
    M = np.full((L, L), fill, np.float64)
    for a in range(L):
        i = s + a
        if i >= n:
            break
        m = min(Dp, L - a, n - i)
        M[a, a:a + m] = B[i, :m]
        M[a:a + m, a] = B[i, :m]
    return M


def diag_values(B: np.ndarray, d: int) -> np.ndarray:
    n = B.shape[0]
    return B[:n - d, d]


def common_mask(maps: list[np.ndarray]) -> np.ndarray:
    m = np.ones(maps[0].shape, bool)
    for M in maps:
        m &= np.isfinite(M)
    return m


def masked(B: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, B, np.nan)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan")
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def per_distance_pearson(T: np.ndarray, P: np.ndarray, dmin: int = 0) -> np.ndarray:
    out = np.full(T.shape[1], np.nan)
    for d in range(dmin, T.shape[1]):
        out[d] = _corr(diag_values(T, d), diag_values(P, d))
    return out


def auc01(r: np.ndarray) -> float:
    """Area under a per-distance curve on a [0, 1] distance axis over the
    finite entries (Hi-cGAN's getCorrelation + sklearn.metrics.auc)."""
    idx = np.flatnonzero(np.isfinite(r))
    if idx.size < 2:
        return float("nan")
    x = idx / (len(r) - 1)
    return float(np.trapezoid(r[idx], x) if hasattr(np, "trapezoid") else np.trapz(r[idx], x))


def hicgan_pearson_auc(Tc: np.ndarray, Pc: np.ndarray, res: int) -> float:
    """Pearson AUC of Hi-cGAN's computePearsonCorrelation on count maps:
    distances 0 .. round(1 Mb / res) - 1, missing pixels as 0 counts, per
    distance Pearson (NaN dropped), AUC with the distance axis divided by the
    largest distance."""
    K = int(np.round(1_000_000 / res))
    T = np.nan_to_num(Tc[:, :K])
    P = np.nan_to_num(Pc[:, :K])
    r = np.full(K, np.nan)
    for d in range(K):
        a, b = diag_values(T, d), diag_values(P, d)
        if a.std() > 0 and b.std() > 0:
            r[d] = np.corrcoef(a, b)[0, 1]
    idx = np.flatnonzero(np.isfinite(r))
    if idx.size < 2:
        return float("nan")
    x = idx / (K - 1)
    f = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(f(r[idx], x))


def band_pearson(T: np.ndarray, P: np.ndarray, dmin: int = 2) -> float:
    return _corr(T[:, dmin:].ravel(), P[:, dmin:].ravel())


def tile_pearsons(T: np.ndarray, P: np.ndarray, tile: int, dmin: int = 2,
                  min_pixels: int = 100) -> np.ndarray:
    """Per-tile Pearson over the upper-triangle pixels with both bins in a
    tile of `tile` bins (the registry's per-region metric, 1 Mb tiles)."""
    n, Dp = T.shape
    out = []
    for s in range(0, n - tile + 1, tile):
        a, b = [], []
        for d in range(dmin, min(Dp, tile)):
            a.append(T[s:s + tile - d, d]); b.append(P[s:s + tile - d, d])
        a, b = np.concatenate(a), np.concatenate(b)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= min_pixels and a[ok].std() > 0 and b[ok].std() > 0:
            out.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.asarray(out)


def band_scc(T: np.ndarray, P: np.ndarray, dmin: int = 1) -> float:
    """HiCRep stratum-adjusted correlation on the O/E band (no extra smoothing,
    as benchmark_metrics.scc)."""
    from scipy.stats import rankdata
    num = den = 0.0
    for d in range(dmin, T.shape[1]):
        a, b = diag_values(T, d), diag_values(P, d)
        ok = np.isfinite(a) & np.isfinite(b)
        m = int(ok.sum())
        if m < 3:
            continue
        a, b = a[ok], b[ok]
        if a.std() == 0 or b.std() == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        w = m * np.sqrt(np.var(rankdata(a) / m) * np.var(rankdata(b) / m))
        num += w * r
        den += w
    return num / den if den > 0 else float("nan")


def insulation_band(B: np.ndarray, w: int) -> np.ndarray:
    """Diamond insulation: mean log O/E of the w x w square between
    [i - w, i) and (i, i + w]; NaN when fewer than half the pixels exist.
    Needs D >= 2w."""
    n, Dp = B.shape
    if Dp - 1 < 2 * w:
        raise ValueError("band too narrow for the insulation window")
    s = np.zeros(n); c = np.zeros(n)
    for a in range(1, w + 1):            # row offset i - a
        for b in range(1, w + 1):        # column offset i + b
            d = a + b
            rows = np.arange(a, n - b)  # centre bins i with both ends inside
            v = B[rows - a, d]
            ok = np.isfinite(v)
            s[rows[ok]] += v[ok]
            c[rows[ok]] += 1
    prof = np.where(c >= 0.5 * w * w, s / np.maximum(c, 1), np.nan)
    return prof


def match_points(ref: np.ndarray, pred: np.ndarray, tol: int) -> tuple[int, int]:
    """(reference positions matched, predicted positions matched) within tol."""
    ref = np.asarray(ref, int); pred = np.asarray(pred, int)
    if ref.size == 0 or pred.size == 0:
        return 0, 0
    ps = np.sort(pred); rs = np.sort(ref)

    def hits(x, ys):
        k = np.searchsorted(ys, x)
        lo = np.abs(x - ys[np.clip(k - 1, 0, len(ys) - 1)])
        hi = np.abs(x - ys[np.clip(k, 0, len(ys) - 1)])
        return int((np.minimum(lo, hi) <= tol).sum())
    return hits(rs, ps), hits(ps, rs)


def match_loops(ref: np.ndarray, pred: np.ndarray, tol: int) -> tuple[int, int]:
    """(reference loops matched, predicted loops matched): both anchors within tol."""
    if len(ref) == 0 or len(pred) == 0:
        return 0, 0
    ref = np.asarray(ref, int); pred = np.asarray(pred, int)
    hr = sum(bool(np.any((np.abs(pred[:, 0] - a) <= tol) & (np.abs(pred[:, 1] - b) <= tol)))
             for a, b in ref)
    hp = sum(bool(np.any((np.abs(ref[:, 0] - a) <= tol) & (np.abs(ref[:, 1] - b) <= tol)))
             for a, b in pred)
    return int(hr), int(hp)


def to_counts(B: np.ndarray, expected: np.ndarray, offset: np.ndarray, scale: float
              ) -> np.ndarray:
    """Proper log O/E band -> count band: exp(v + offset(d)) * E(d) * scale.
    Inverts oe_proper.py apart from its clipping and smoothing."""
    D = B.shape[1] - 1
    with np.errstate(over="ignore", invalid="ignore"):
        C = np.exp(B + offset[None, :D + 1]) * expected[None, :D + 1] * scale
    return C


def to_oe(C: np.ndarray, expected: np.ndarray, offset: np.ndarray, scale: float) -> np.ndarray:
    D = C.shape[1] - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(C / (expected[None, :D + 1] * scale)) - offset[None, :D + 1]


def compartment_labels(e1_bins: np.ndarray) -> np.ndarray:
    """+1 / -1 / 0 (unknown) per bin from the measured E1."""
    lab = np.sign(np.nan_to_num(e1_bins))
    return lab


def contact_preference(B: np.ndarray, lab: np.ndarray, dmin: int) -> np.ndarray:
    """Per bin: mean log O/E with A partners minus with B partners (partners
    dmin..D bins away; labels from the measured E1)."""
    n, Dp = B.shape
    sA = np.zeros(n); cA = np.zeros(n); sB = np.zeros(n); cB = np.zeros(n)
    for d in range(dmin, Dp):
        v = B[:n - d, d]
        ok = np.isfinite(v)
        i = np.arange(n - d)
        for x, y in ((i, i + d), (i + d, i)):
            la = lab[y]
            a = ok & (la > 0); b = ok & (la < 0)
            np.add.at(sA, x[a], v[a]); np.add.at(cA, x[a], 1)
            np.add.at(sB, x[b], v[b]); np.add.at(cB, x[b], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cp = sA / cA - sB / cB
    cp[(cA < 3) | (cB < 3)] = np.nan
    return cp


def saddle_strength(B: np.ndarray, e1_bins: np.ndarray, dmin: int, q: float = 0.2
                    ) -> float:
    """(AA + BB) / (AB + BA) of mean O/E (exp of log O/E) over band pixels
    dmin..D, A / B = top / bottom q of the measured E1."""
    e = e1_bins
    ok = np.isfinite(e)
    if ok.sum() < 20:
        return float("nan")
    hi, lo = np.quantile(e[ok], 1 - q), np.quantile(e[ok], q)
    grp = np.where(e >= hi, 1, np.where(e <= lo, -1, 0))
    grp[~ok] = 0
    n, Dp = B.shape
    s = {"AA": 0.0, "BB": 0.0, "AB": 0.0}; c = {"AA": 0, "BB": 0, "AB": 0}
    for d in range(dmin, Dp):
        v = np.exp(B[:n - d, d])
        g1, g2 = grp[:n - d], grp[d:]
        f = np.isfinite(v)
        for key, m in (("AA", (g1 == 1) & (g2 == 1)), ("BB", (g1 == -1) & (g2 == -1)),
                       ("AB", (g1 * g2) == -1)):
            m &= f
            s[key] += v[m].sum(); c[key] += int(m.sum())
    if min(c.values()) < 10:
        return float("nan")
    aa, bb, ab = s["AA"] / c["AA"], s["BB"] / c["BB"], s["AB"] / c["AB"]
    return float((aa + bb) / (2 * ab))


def local_e1(B: np.ndarray, gc: np.ndarray, W: int, step: int) -> np.ndarray:
    """E1 of the row correlation matrix of each W-bin window (the whole window
    is inside the band), phased by GC per window, z-scored, averaged over the
    overlapping windows. A chromosome-wide E1 cannot be built from a 2 Mb band;
    this is the compartment signal the band carries."""
    n = B.shape[0]
    acc = np.zeros(n); cnt = np.zeros(n)
    for s in range(0, max(1, n - W + 1), step):
        M = band_to_dense(B, s, min(W, n - s))
        live = np.isfinite(M).mean(1) > 0.7
        if live.sum() < W // 2:
            continue
        X = np.where(np.isfinite(M), M, 0.0)[live][:, live]
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.nan_to_num(np.corrcoef(X))
        v = np.linalg.eigh(C)[1][:, -1]
        g = gc[s:s + M.shape[0]][live]
        okg = np.isfinite(g)
        if okg.sum() > 5 and v[okg].std() > 0 and g[okg].std() > 0:
            v = v * np.sign(np.corrcoef(v[okg], g[okg])[0, 1])
        if v.std() == 0:
            continue
        v = (v - v.mean()) / v.std()
        idx = np.arange(s, s + M.shape[0])[live]
        acc[idx] += v; cnt[idx] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def coarse_track(x: np.ndarray, f: int) -> np.ndarray:
    nb = len(x) // f
    if nb == 0:
        return np.zeros(0)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(x[:nb * f].reshape(nb, f), 1)


def parse_subset(name: str) -> list[str]:
    """Measure list of a regressor file name 'best_model_<algo>_<m1 m2 ...>_degree_<k>.pkl'
    (measure names contain spaces; the longest name is matched first)."""
    body = name[len("best_model_"):] if name.startswith("best_model_") else name
    body = body.split("_", 1)[1]
    body = re.sub(r"_degree_\d+\.pkl$", "", body)
    out, s = [], body.strip()
    names = sorted(NINE, key=len, reverse=True)
    while s:
        for m in names:
            if s == m or s.startswith(m + " "):
                out.append(m)
                s = s[len(m):].strip()
                break
        else:
            raise ValueError(f"cannot parse measure subset of {name!r} at {s!r}")
    return out


# ===================================================================== #
# fast combined score on windows (checkpoint selection, train.select_metric
# = "hic_measures"; API fixed: combined_score(maps, res) -> float)
# ===================================================================== #
COMBINED_GROUPS = {
    # structure level: window measures (hicmultiome.structure, plus the
    # pooled per-distance Pearson AUC and the boundary-count agreement)
    "pixels": ("pearson", "scc", "pearson_auc"),
    "tads": ("insulation_corr", "boundary_f1", "tad_fraction_agreement"),
    "loops": ("loop_enrich_corr", "loop_apa_agreement"),
    "compartments": ("compartment_corr", "eigenvector_abs_corr"),
}


def combined_parts(maps, res: int) -> dict:
    """Every term of combined_score: per-window medians of the
    hicmultiome.structure measures and two pooled terms."""
    from hicmultiome.structure import boundaries, insulation, structure_metrics
    rows, nt, npb = [], 0, 0
    pd_x: dict[int, list] = {}
    pd_y: dict[int, list] = {}
    w = max(3, int(round(250_000 / res)))
    for P, T in maps:
        P = np.asarray(P, np.float64); T = np.asarray(T, np.float64)
        m = structure_metrics(P, T, res)
        r = m.get("loop_apa_ratio", np.nan)
        m["loop_apa_agreement"] = min(r, 1 / r) if np.isfinite(r) and r > 0 else 0.0
        rows.append(m)
        it, ip = insulation(T, w), insulation(P, w)
        st, sp_ = np.nanstd(it), np.nanstd(ip)
        if np.isfinite(st) and st > 0:
            nt += len(boundaries(it, 0.5 * st))
            npb += len(boundaries(ip, 0.5 * sp_)) if np.isfinite(sp_) and sp_ > 0 else 0
        n = T.shape[0]
        for d in range(0, n // 2):
            a, b = np.diagonal(T, d), np.diagonal(P, d)
            ok = np.isfinite(a) & np.isfinite(b)
            pd_x.setdefault(d, []).append(a[ok]); pd_y.setdefault(d, []).append(b[ok])
    out = {}
    for k in ("pearson", "scc", "insulation_corr", "boundary_f1", "loop_enrich_corr",
              "loop_apa_agreement", "compartment_corr", "eigenvector_abs_corr"):
        v = np.array([r.get(k, np.nan) for r in rows], float)
        out[k] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
    rd = np.full(len(pd_x), np.nan)
    for d in range(len(pd_x)):
        rd[d] = _corr(np.concatenate(pd_x[d]), np.concatenate(pd_y[d]))
    out["pearson_auc"] = auc01(rd)
    out["tad_fraction_agreement"] = (min(npb / nt, nt / npb) if nt > 0 and npb > 0 else 0.0)
    return out


def combined_score(maps, res: int) -> float:
    """Combined validation score of predicted windows, higher = better.

    maps: list of (prediction, target) square windows in log O/E (any affine
    encoding of it); res: bin size in bp. The score is the mean of four
    structure levels, each the mean of its terms (COMBINED_GROUPS): pixels
    (median window Pearson, median stratified correlation, Pearson AUC of the
    per-distance Pearson pooled over windows), TADs (median insulation-profile
    correlation, median boundary F1, agreement min(r, 1/r) of the pooled
    boundary counts r = predicted / target, the window analogue of TAD
    Fraction), loops (median donut-enrichment correlation, median APA
    agreement) and compartments (median correlation of the windows' row
    correlation matrices, median |correlation| of their first
    eigenvectors). Missing terms are left out of their level.

    This is the window-level proxy of the multi-measure evaluation: the nine
    count-space measures and their trained regressors (stage combine) need
    whole-chromosome maps and external tools (minutes per map), too slow for
    every epoch; the levels here are the fast measures of the same
    structures (TADs, loops, compartments within 2 Mb) at equal weight."""
    parts = combined_parts(maps, res)
    lv = []
    for keys in COMBINED_GROUPS.values():
        v = [parts[k] for k in keys if np.isfinite(parts.get(k, np.nan))]
        if v:
            lv.append(float(np.mean(v)))
    return float(np.mean(lv)) if lv else float("nan")


# ===================================================================== #
# sources
# ===================================================================== #
def run_dir(model: str, res: int) -> Path:
    return MR / f"{model}_{res // 1000}kb"


def model_available(model: str, res: int) -> bool:
    r = run_dir(model, res)
    return (r / "checkpoint-best" / "model.pt").exists() and (
        MR / f"{model}_{res // 1000}kb.yaml").exists()


def target_cool(cell: str, res: int) -> Path:
    return OE / f"{cell}_proper_{res // 1000}kb.cool"


def consensus_cool(cell: str, res: int) -> Path:
    k = res // 1000
    return OE / (f"consensus_loo_{cell}_proper_{k}kb.cool" if cell in TRAIN12
                 else f"consensus12_proper_{k}kb.cool")


def pred_cool(model: str, res: int, cell: str) -> Path:
    return OUT / "pred" / f"{res // 1000}kb" / model / f"{cell}.cool"


def cascfold_cool(cell: str, res: int) -> Path:
    return OE / "cascade" / f"prior_{cell}_{res // 1000}kb.cool"


def block_sources(set_: str, cell: str, res: int, extra_comps: list[str] | None = None
                  ) -> list[dict]:
    """Ordered sources of a block: name, kind, path, flags."""
    out = []
    for m in MODELS:
        # trainchrom: the main models saw these chromosomes (their clean
        # training-cell evaluation is trainstretch); all12refres stays as the
        # in-training counterpart of the chromosome-held-out cascade fold model
        if set_ == "trainchrom" and m != "all12refres":
            continue
        if model_available(m, res):
            fl = "clean"
            if set_ == "trainchrom":
                fl = "chrom_in_training"
            if m == "all12deg_refres_drop" and cell == "hct116" and set_ != "trainstretch":
                fl = "chrom_in_training"
            out.append(dict(name=m, kind="model", path=pred_cool(m, res, cell), flag=fl))
    if cell in TRAIN12 and cascfold_cool(cell, res).exists():
        out.append(dict(name=CASCFOLD, kind="cool", path=cascfold_cool(cell, res),
                        flag="clean"))
    out.append(dict(name="consensus", kind="cool", path=consensus_cool(cell, res),
                    flag="baseline"))
    bo = BEST_OTHER[res].get(cell)
    if bo:
        out.append(dict(name="best_other_map", kind="cool", path=target_cool(bo, res),
                        flag=f"baseline:{bo}"))
    rep = REPLICATE.get(cell)
    if rep:
        p = OE / f"{rep}_{res // 1000}kb.cool"
        if p.exists():
            out.append(dict(name="replicate", kind="cool", path=p, flag=f"ceiling:{rep}"))
    comps = extra_comps if extra_comps is not None else COMPETITORS.get((set_, cell, res), [])
    for c in comps:
        out.append(dict(name=c, kind="competitor", path=None, flag="competitor"))
    return out


def blocks() -> list[dict]:
    out = []
    for res in RESOLUTIONS:
        D = WINDOW[res] - 1
        for cell in HELDOUT:
            for ch in CHROMS["heldout"]:
                out.append(dict(set="heldout", cell=cell, res=res, chrom=ch, D=D))
        for set_ in ("trainchrom", "trainstretch"):
            for cell in TRAIN12:
                for ch in CHROMS[set_]:
                    out.append(dict(set=set_, cell=cell, res=res, chrom=ch, D=D))
    for set_, cell, res, ch, D, comps in EXTRA_BLOCKS:
        out.append(dict(set=set_, cell=cell, res=res, chrom=ch, D=D, comps=comps))
    for b in out:
        b["id"] = f"{b['set']}_{b['cell']}_{b['res'] // 1000}kb_{b['chrom']}"
    return out


def block_source_list(b: dict) -> list[dict]:
    if b["set"] in ("hicreg", "epiphany"):
        src = block_sources("trainstretch", b["cell"], b["res"], b["comps"])
        for s in src:
            if s["kind"] == "model":
                s["flag"] = "chrom_in_training"
        return src
    return block_sources(b["set"], b["cell"], b["res"])


# ===================================================================== #
# stage: predict
# ===================================================================== #
def pred_chroms(cell: str) -> list[str]:
    ch = set()
    if cell in HELDOUT:
        ch |= set(CHROMS["heldout"])
    else:
        ch |= set(CHROMS["trainchrom"]) | set(CHROMS["trainstretch"])
    for set_, c, res, chrom, D, comps in EXTRA_BLOCKS:
        if c == cell:
            ch.add(chrom)
    return sorted(ch, key=lambda c: int(c[3:]) if c[3:].isdigit() else 99)


def predict_one(model: str, res: int, cell: str, threads: int, gpu: str = "") -> None:
    out = pred_cool(model, res, cell)
    if out.exists() or not claim(out):
        return
    try:
        cfg = MR / f"{model}_{res // 1000}kb.yaml"
        ck = run_dir(model, res) / "checkpoint-best"
        cmd = [PY, "-m", "hicmultiome.predict", "--config", str(cfg), "--checkpoint", str(ck),
               "--chroms", ",".join(pred_chroms(cell)), "--tta", "--out", str(out) + ".tmp"]
        if cell in HELDOUT:
            cmd += ["--sample", "gm12878", "--bigwig-dir", str(BIGWIG[cell]), "--cell", cell]
        else:
            cmd += ["--sample", cell]
        if model not in REFERENCE_FREE:
            cmd += ["--reference", str(consensus_cool(cell, res))]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        lf = out.with_suffix(".log")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(lf, "w") as fh:
            fh.write(" ".join(cmd) + "\n"); fh.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=env)
        if rc == 0 and Path(str(out) + ".tmp").exists():
            os.replace(str(out) + ".tmp", out)
            log(f"[predict] {model} {res // 1000}kb {cell} ok")
        else:
            log(f"[predict] FAILED {model} {res // 1000}kb {cell} rc={rc} ({lf})")
    finally:
        release(out)


def sample_cells_of(model: str, res: int) -> set[str]:
    import yaml
    cfg = yaml.safe_load((MR / f"{model}_{res // 1000}kb.yaml").read_text())
    return {s["name"] for s in cfg["data"]["samples"]}


def stage_predict(threads: int, jobs: int, gpu: str = "") -> None:
    """CPU workers (gpu "") or one GPU worker per call (gpu "0" or "1"; the
    model forward on the GPU, under 1 GB). Several calls share the queue."""
    todo = []
    for res in RESOLUTIONS:
        for m in MODELS:
            if not model_available(m, res):
                log(f"[predict] {m} {res // 1000}kb: no checkpoint, skipped")
                continue
            names = sample_cells_of(m, res)
            for cell in HELDOUT + TRAIN12:
                if cell in TRAIN12 and cell not in names:
                    continue
                if not pred_cool(m, res, cell).exists():
                    todo.append((m, res, cell))
    log(f"[predict] {len(todo)} predictions")
    per = max(1, threads // jobs)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in [ex.submit(predict_one, m, r, c, per, gpu) for m, r, c in todo]:
            f.result()


# ===================================================================== #
# stage: expected (count conversion curves)
# ===================================================================== #
def expected_file(cell: str, res: int, chrom: str) -> Path:
    return OUT / "expected" / f"{cell}_{res // 1000}kb_{chrom}.npz"


def measured_uri(cell: str, res: int) -> str:
    import cellspecific as cs
    return cs.measured_uri(cell, res)


def compute_expected(cell: str, res: int, chrom: str, D: int = 200) -> Path:
    """Balanced expected E(d) and per-diagonal log offset of the measured map
    (exactly the curves oe_proper.py used for the target), and the raw/balanced
    scale of the band."""
    f = expected_file(cell, res, chrom)
    if f.exists():
        return f
    import cooler
    import oe_proper as op
    clr = cooler.Cooler(measured_uri(cell, res))
    name = chrom if chrom in clr.chromnames else chrom.replace("chr", "")
    o = clr.offset(name); n = clr.extent(name)[1] - o
    w_all = clr.bins()[:]["weight"].values
    mp = np.isfinite(w_all[o:o + n])
    # the target was built with band = 2 Mb / res; the curves of oe_proper
    # depend on the band only through the per-diagonal values, so any D works
    exp_b, offset = op.stream_stats(clr, name, D, w_all, mp)
    px = clr.matrix(balance=False, as_pixels=True).fetch(name)
    i = px.bin1_id.values - o; j = px.bin2_id.values - o
    k = (j - i <= D) & (j >= i)
    i, j, v = i[k], j[k], px["count"].values[k].astype(float)
    w = w_all[o:o + n]
    good = np.isfinite(w[i]) & np.isfinite(w[j])
    raw_total = float(v[good].sum())
    bal_total = float((v[good] * w[i[good]] * w[j[good]]).sum())
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, expected=exp_b, offset=offset, scale=raw_total / bal_total, n=n,
             mappable=mp, weight=w, raw_i=i, raw_j=j, raw_v=v, uri=measured_uri(cell, res))
    return f


def stage_expected(jobs: int) -> None:
    todo = sorted({(b["cell"], b["res"], b["chrom"]) for b in blocks()})
    log(f"[expected] {len(todo)} curves")

    def one(t):
        f = expected_file(*t)
        if f.exists() or not claim(f):
            return
        try:
            compute_expected(*t)
            log(f"[expected] {t} ok")
        except Exception as e:                       # noqa: BLE001
            log(f"[expected] FAILED {t}: {e!r}")
        finally:
            release(f)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(one, todo))


# ===================================================================== #
# stage: score
# ===================================================================== #
def load_band(path: Path, chrom: str, n: int, D: int, res: int | None = None
              ) -> np.ndarray | None:
    import cooler
    clr = cooler.Cooler(str(path))
    name = next((c for c in clr.chromnames if c == chrom or c == chrom.replace("chr", "")), None)
    if name is None or clr.binsize is None:
        return None
    res = res or clr.binsize
    px = clr.matrix(balance=False, as_pixels=True).fetch(name)
    o = clr.offset(name)
    if clr.binsize == res:
        return band_from_pixels(px.bin1_id.values - o, px.bin2_id.values - o,
                                px["count"].values, n, D)
    # another bin grid (the single-lane replicate maps at 25 kb are stored at
    # 24 kb): bilinear sampling at our bin centres
    rr = clr.binsize
    nr = int(np.ceil(clr.chromsizes[name] / rr))
    Dr = int(np.ceil((D + 2) * res / rr)) + 2
    R = band_from_pixels(px.bin1_id.values - o, px.bin2_id.values - o,
                         px["count"].values, nr, Dr)
    R[R == 0] = np.nan
    return resample_band(R, rr, n, D, res)


def resample_band(R: np.ndarray, rr: int, n: int, D: int, res: int) -> np.ndarray:
    """Band R on an rr grid -> band on a res grid, bilinear in (bin, bin)
    space; a pixel is kept only when all four neighbours are measured."""
    from scipy.ndimage import map_coordinates
    nr = R.shape[0]
    M = np.full((n, D + 1), np.nan)
    i = np.arange(n)
    for d in range(D + 1):
        ui = (i + 0.5) * res / rr - 0.5
        uj = (i + d + 0.5) * res / rr - 0.5
        row, dist = np.minimum(ui, uj), np.abs(uj - ui)
        ok = (i + d < n) & (row >= 0) & (row <= nr - 1) & (dist <= R.shape[1] - 1)
        if not ok.any():
            continue
        # bilinear over the four (row, column) neighbours of (ui, uj)
        a0, b0 = np.floor(ui[ok]).astype(int), np.floor(uj[ok]).astype(int)
        fa, fb = ui[ok] - a0, uj[ok] - b0
        acc = np.zeros(ok.sum()); good = np.ones(ok.sum(), bool)
        for da, wa in ((0, 1 - fa), (1, fa)):
            for db, wb in ((0, 1 - fb), (1, fb)):
                ra, rb = np.clip(a0 + da, 0, nr - 1), np.clip(b0 + db, 0, nr - 1)
                lo, dd = np.minimum(ra, rb), np.abs(rb - ra)
                inside = dd < R.shape[1]
                v = np.full(len(lo), np.nan)
                v[inside] = R[lo[inside], dd[inside]]
                w = wa * wb
                good &= np.isfinite(v) | (w == 0)
                acc += np.where(w > 0, np.nan_to_num(v) * w, 0.0)
        out = np.where(good, acc, np.nan)
        M[i[ok], d] = out
    return M


def competitor_band(comp: str, chrom: str, n: int, D: int, res: int) -> np.ndarray | None:
    """Whole-chromosome band of a competitor's O/E map sampled at our bin
    centres, assembled from overlapping blocks of benchmark_blocks.sample_competitor."""
    import benchmark_blocks as bb
    import benchmark_competitors as bc
    spec = bc.COMPETITORS[comp]
    if spec.get("chroms") and chrom not in spec["chroms"]:
        return None
    reach = min(D, int(spec["reach"] // res))
    B = np.full((n, D + 1), np.nan)
    S = 2 * (D + 1)
    got = False
    for s in range(0, n, S):
        L = min(S + D + 1, n - s)
        if L < 4:
            continue
        M = bb.sample_competitor(spec, comp, chrom, s, L, res)
        if M is None:
            continue
        got = True
        for a in range(min(S, L)):
            m = min(reach + 1, L - a)
            B[s + a, :m] = M[a, a:a + m]
    if not got:
        return None
    B[:, reach + 1:] = np.nan
    return B


def holdout_stretch_mask(chrom: str, n: int, D: int, res: int) -> np.ndarray:
    st = pd.read_csv(HOLDOUT_BED, sep="\t", comment="#", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"])
    st = st[st.chrom == chrom]
    lab = np.full(n, -1)
    for k, (s, e) in enumerate(zip(st.start.values, st.end.values)):
        a, b = int(np.ceil(s / res)), int(e // res)       # bins wholly inside
        lab[a:b] = k
    M = np.zeros((n, D + 1), bool)
    for d in range(D + 1):
        i = np.arange(n - d)
        M[i, d] = (lab[i] >= 0) & (lab[i] == lab[i + d])
    return M


def gc_bins(chrom: str, n: int, res: int) -> np.ndarray:
    gc = pd.read_csv(HM / "runs/cellspecific/e1/gc_100kb.tsv", sep="\t")
    g = gc[gc.chrom == chrom].sort_values("start")
    f = 100000 // res if 100000 % res == 0 else None
    centres = (np.arange(n) + 0.5) * res
    idx = (centres // 100000).astype(int)
    v = np.full(n, np.nan)
    gv = g.GC.values
    ok = idx < len(gv)
    v[ok] = gv[idx[ok]]
    return v


def e1_bins(cell: str, chrom: str, n: int, res: int) -> np.ndarray:
    import cellspecific as cs
    e = pd.read_csv(cs.e1_file(cell), sep="\t")
    e = e[e.chrom == chrom].sort_values("start")
    centres = (np.arange(n) + 0.5) * res
    idx = (centres // 100000).astype(int)
    v = np.full(n, np.nan)
    ev = e.E1.values
    ok = idx < len(ev)
    v[ok] = ev[idx[ok]]
    return v


def ref_loops(cell: str, chrom: str, res: int, n: int, D: int) -> np.ndarray:
    import cellspecific as cs
    lp = cs.read_loops(cell)
    lp = lp[lp.chrom == chrom]
    a = ((lp.s1.values + 5000) // res).astype(int)
    b = ((lp.s2.values + 5000) // res).astype(int)
    k = (b - a >= 2) & (b - a <= D) & (b < n)
    if not k.any():
        return np.zeros((0, 2), int)
    return np.unique(np.stack([a[k], b[k]], 1), axis=0)


def loop_params(res: int):
    import benchmark_blocks as bb
    return bb.params(res)["loop"]


def rank_loop_metrics(T: np.ndarray, P: np.ndarray, ref: np.ndarray, res: int) -> dict:
    """benchmark_metrics.loop_counts on overlapping dense blocks of the band;
    each reference loop is scored in the block where its first anchor lies in
    the block's core."""
    import benchmark_metrics as bm
    prm = loop_params(res)
    n, Dp = T.shape
    S = 4 * Dp
    items = []
    for s in range(0, n, S):
        core = (ref[:, 0] >= s) & (ref[:, 0] < s + S)
        if not core.any():
            continue
        L = min(S + Dp, n - s)
        r = ref[core] - s
        r = r[r[:, 1] < L]
        if len(r) == 0:
            continue
        Td = band_to_dense(T, s, L); Pd = band_to_dense(P, s, L)
        items.append(bm.loop_counts(Td, Pd, r, prm, seed=s))
    if not items:
        return {}
    return bm.pool_loops(items, prm)


def write_count_cool(C: np.ndarray, chrom: str, res: int, path: Path, chromsize: int) -> None:
    import cooler
    n, Dp = C.shape
    bins = pd.DataFrame({"chrom": chrom, "start": np.arange(n) * res,
                         "end": np.minimum((np.arange(n) + 1) * res, chromsize)})
    ii, dd = np.nonzero(np.isfinite(C) & (np.rint(np.nan_to_num(C)) > 0))
    v = np.rint(C[ii, dd]).astype(np.int32)
    px = pd.DataFrame({"bin1_id": ii, "bin2_id": ii + dd, "count": v})
    px = px[px.bin2_id < n].sort_values(["bin1_id", "bin2_id"])
    tmp = Path(str(path) + ".tmp.cool")
    path.parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(str(tmp), bins=bins, pixels=px, ordered=True,
                         dtypes={"count": np.int32})
    os.replace(tmp, path)


def write_pixels_gz(C: np.ndarray, chrom: str, res: int, path: Path) -> None:
    """3DChromatin_ReplicateQC edge format: chr1 start1 chr2 start2 count."""
    import gzip
    n, Dp = C.shape
    ii, dd = np.nonzero(np.isfinite(C) & (np.rint(np.nan_to_num(C)) > 0))
    jj = ii + dd
    k = jj < n
    ii, jj, dd = ii[k], jj[k], dd[k]
    v = np.rint(C[ii, dd]).astype(np.int64)
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt") as fh:
        for a, b, c in zip(ii * res, jj * res, v):
            fh.write(f"{chrom}\t{a}\t{chrom}\t{b}\t{c}\n")
    os.replace(tmp, path)


def sh(cmd, logf: Path, env=None, cwd=None) -> int:
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n"); fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               env=env, cwd=cwd)


ONE_THREAD = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
                  MPLBACKEND="Agg")


def run_findtads(cool: Path, outdir: Path, res: int, chrom: str) -> Path | None:
    pref = outdir / "tads"
    if (outdir / "tads_boundaries.bed").exists():
        return pref
    outdir.mkdir(parents=True, exist_ok=True)
    rc = sh([HX376 / "hicFindTADs", "--matrix", cool, "--minDepth", 3 * res,
             "--maxDepth", 10 * res, "--step", res, "--numberOfProcessors", 1,
             "--outPrefix", pref, "--minBoundaryDistance", 100000,
             "--correctForMultipleTesting", "fdr", "--thresholdComparisons", 0.5,
             "--chromosomes", chrom], outdir / "hicFindTADs.log", ONE_THREAD)
    return pref if rc == 0 and (outdir / "tads_boundaries.bed").exists() else None


def tad_measures(pred: Path | None, truth: Path | None, res: int) -> dict:
    """TAD Score MSE, TAD Fraction, TAD Fraction Exact Match (as the
    hyperparameter work: score bedgraphs aligned by position; boundary counts;
    exact coordinate matches over the target's boundaries) and boundary
    recall / precision within 1 bin."""
    out = {m: np.nan for m in ("TAD Score MSE", "TAD Fraction", "TAD Fraction Exact Match",
                               "bnd_recall", "bnd_precision", "n_bnd_pred", "n_bnd_target")}
    if pred is None or truth is None:
        return out
    cols = ["chromosome", "start", "end", "score"]
    sp = pd.read_csv(str(pred) + "_score.bedgraph", sep="\t", names=cols, comment="#")
    st = pd.read_csv(str(truth) + "_score.bedgraph", sep="\t", names=cols, comment="#")
    m = sp.merge(st, on=["chromosome", "start", "end"], suffixes=("_p", "_t"))
    out["TAD Score MSE"] = float(((m.score_p - m.score_t) ** 2).mean()) if len(m) else np.nan
    bc = ["chromosome", "start", "end", "name", "score", "strand"]

    def rd(p):
        f = str(p) + "_boundaries.bed"
        if os.path.getsize(f) == 0:
            return pd.DataFrame(columns=bc)
        return pd.read_csv(f, sep="\t", names=bc, comment="#")
    bp, bt = rd(pred), rd(truth)
    out["n_bnd_pred"], out["n_bnd_target"] = len(bp), len(bt)
    if len(bt):
        out["TAD Fraction"] = len(bp) / len(bt)
        ex = bp[["chromosome", "start", "end"]].merge(bt[["chromosome", "start", "end"]])
        out["TAD Fraction Exact Match"] = len(ex) / len(bt)
        hr, hp = match_points(bt.start.values // res, bp.start.values // res, 1)
        out["bnd_recall"] = hr / len(bt)
        out["bnd_precision"] = hp / len(bp) if len(bp) else np.nan
    return out


def run_detectloops(cool: Path, out: Path, chrom: str) -> Path | None:
    if out.exists():
        return out
    rc = sh([HX376 / "hicDetectLoops", "-m", cool, "-o", str(out) + ".tmp",
             "--maxLoopDistance", 2000000, "-pit", 0, "-t", 1, "-tpc", 1,
             "--chromosomes", chrom], Path(str(out) + ".log"), ONE_THREAD)
    if rc == 0 and Path(str(out) + ".tmp").exists():
        os.replace(str(out) + ".tmp", out)
        return out
    return None


def called_loops(f: Path | None, res: int) -> np.ndarray | None:
    if f is None:
        return None
    if os.path.getsize(f) == 0:
        return np.zeros((0, 2), int)
    d = pd.read_csv(f, sep="\t", header=None, usecols=[1, 4])
    return np.stack([(d[1].values + res // 2) // res, (d[4].values + res // 2) // res], 1)


HICREP_CODE = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
import numpy as np
from hicrep.utils import readMcool
from hicrep import hicrepSCC
job = json.load(open(sys.argv[1]))
out = {}
truth, _ = readMcool(job["truth"], -1)
for name, path in job["preds"].items():
    try:
        c, _ = readMcool(path, -1)
        s = hicrepSCC(c, truth, 5, 1000000, False, [job["chrom"]])
        out[name] = float(np.mean(s))
    except Exception as e:
        out[name] = None
json.dump(out, open(sys.argv[2], "w"))
"""


def run_hicrep(truth: Path, preds: dict[str, Path], chrom: str, work: Path) -> dict:
    res_f = work / "hicrep.json"
    if res_f.exists():
        return json.loads(res_f.read_text())
    job = work / "hicrep_job.json"
    job.write_text(json.dumps({"truth": str(truth), "chrom": chrom,
                               "preds": {k: str(v) for k, v in preds.items()}}))
    code = work / "hicrep_run.py"
    code.write_text(HICREP_CODE)
    rc = sh([HICREP_PY, code, job, str(res_f) + ".tmp"], work / "hicrep.log", ONE_THREAD)
    if rc == 0:
        os.replace(str(res_f) + ".tmp", res_f)
        return json.loads(res_f.read_text())
    return {}


def run_ent3c(truth: Path, pred: Path, name: str, chrom: str, res: int, work: Path
              ) -> float:
    """One ENT3C run per pair (the common-empty-bin exclusion of ENT3C runs
    over the files of one run, so pairs are run separately, as in the
    hyperparameter work)."""
    od = work / "ent3c" / name
    sim = od / "PAIR_ENT3C_similarity.csv"
    if not sim.exists():
        od.mkdir(parents=True, exist_ok=True)
        cfg = {"PHI_MAX": 1e3, "CHRSPLIT": None, "SUB_M_SIZE_FIX": 500, "ChrNr": chrom,
               "Resolution": str(res), "phi": 1, "NormM": 0, "WEIGHTS_NAME": "weight",
               "DATA_PATH": str(pred.parent) + "/",
               "FILES": [pred.name, name, os.path.relpath(truth, pred.parent), "TARGET"],
               "OUT_DIR": str(od), "OUT_PREFIX": "PAIR"}
        (od / "config.json").write_text(json.dumps(cfg, indent=1))
        env = {k: v for k, v in ONE_THREAD.items() if k != "DISPLAY"}
        env["MPLBACKEND"] = "Agg"            # ENT3C plots; an X backend blocks
        sh(["bash", "-c", f"yes y | {ENT3C} run_all --config={od / 'config.json'}"],
           od / "ent3c.log", env)
        hits = list(od.rglob("PAIR_ENT3C_similarity.csv"))
        if hits and hits[0] != sim:
            shutil.copyfile(hits[0], sim)
    if not sim.exists():
        return float("nan")
    d = pd.read_csv(sim, sep="\t")
    return float(d.Q.iloc[0]) if len(d) else float("nan")


def run_repqc(truth_px: Path, preds_px: dict[str, Path], chrom: str, res: int, n: int,
              work: Path, tag: str = "") -> pd.DataFrame:
    """GenomeDISCO, HiC-Spector, QuASAR-Rep of every prediction against the
    target, one 3DChromatin_ReplicateQC run per chunk of predictions (the tool
    works through its pairs one after the other, so chunks run in parallel)."""
    od = work / "repqc" / tag if tag else work / "repqc"
    score = od / "out" / "scores" / f"reproducibility.{chrom}.txt"
    alt = od / "out" / "scores" / "reproducibility.genomewide.txt"
    if not (score.exists() or alt.exists()):
        import gzip
        od.mkdir(parents=True, exist_ok=True)
        nodes = od / "bins.bed.gz"
        with gzip.open(nodes, "wt") as fh:
            for i in range(n):
                fh.write(f"{chrom}\t{i * res}\t{(i + 1) * res}\t{i * res}\n")
        samples = od / "samples.txt"
        names = {k: re.sub(r"[^A-Za-z0-9]", "x", k) for k in preds_px}
        with open(samples, "w") as fh:
            fh.write(f"TARGET\t{truth_px}\n")
            for k, p in preds_px.items():
                fh.write(f"{names[k]}\t{p}\n")
        pairs = od / "pairs.txt"
        with open(pairs, "w") as fh:
            for k in preds_px:
                fh.write(f"TARGET\t{names[k]}\n")
        (od / "names.json").write_text(json.dumps(names))
        if (od / "out").exists():
            shutil.rmtree(od / "out")
        sh([REPQC, "run_all", "--metadata_samples", samples, "--metadata_pairs", pairs,
            "--bins", nodes, "--outdir", od / "out", "--parameters_file", REPQC_PARAMS,
            "--methods", "GenomeDISCO,HiC-Spector,QuASAR-Rep", "--concise_analysis"],
           od / "repqc.log", ONE_THREAD, cwd=od)
    f = score if score.exists() else alt if alt.exists() else None
    if f is None:
        return pd.DataFrame()
    d = pd.read_csv(f, sep="\t")
    d.columns = [c.lstrip("#") for c in d.columns]
    names = json.loads((od / "names.json").read_text())
    inv = {v: k for k, v in names.items()}
    d["source"] = d.Sample2.map(inv)
    return d


def score_block(b: dict, jobs: int) -> Path:
    """All measures of every source of a block -> blocks/<id>/measures.csv."""
    bd = OUT / "blocks" / b["id"]
    outf = bd / "measures.csv"
    if outf.exists():
        return outf
    t0 = time.time()
    cell, res, chrom, D = b["cell"], b["res"], b["chrom"], b["D"]
    ef = expected_file(cell, res, chrom)
    if not ef.exists():
        compute_expected(cell, res, chrom)
    E = np.load(ef)
    n = int(E["n"])
    T = load_band(target_cool(cell, res), chrom, n, D)
    if T is None:
        log(f"[score] {b['id']}: no target")
        return outf
    T[T == 0] = np.nan
    src = block_source_list(b)
    maps = {}
    for s in src:
        if s["kind"] == "competitor":
            M = competitor_band(s["name"], chrom, n, D, res)
        elif s["path"] is None or not Path(s["path"]).exists():
            M = None
        else:
            M = load_band(Path(s["path"]), chrom, n, D, res)
            if M is not None and s["kind"] == "cool":
                M[M == 0] = np.nan
        if M is None:
            continue
        cov = np.isfinite(M[np.isfinite(T)]).mean()
        if cov < 0.5:
            log(f"[score] {b['id']}: {s['name']} covers {cov:.2f} of the target, left out")
            continue
        maps[s["name"]] = M
    src = [s for s in src if s["name"] in maps]
    # common pixels: finite in the target and every source; trainstretch
    # additionally restricts to pixels inside one held-out stretch
    mask = common_mask([T] + list(maps.values()))
    if b["set"] == "trainstretch":
        mask &= holdout_stretch_mask(chrom, n, D, res)
    if mask.sum() < 1000:
        log(f"[score] {b['id']}: {mask.sum()} common pixels, skipped")
        return outf
    Tm = masked(T, mask)
    Mm = {k: masked(v, mask) for k, v in maps.items()}
    W = WINDOW[res]
    ins_w = {25000: 3, 10000: 5}[res]
    tile = 1_000_000 // res
    e1 = e1_bins(cell, chrom, n, res)
    lab = compartment_labels(e1)
    gc = gc_bins(chrom, n, res)
    dmin_cp = max(2, 200_000 // res)
    cp_T = contact_preference(Tm, lab, dmin_cp)
    sad_T = saddle_strength(Tm, e1, dmin_cp)
    le_T = local_e1(Tm, gc, min(W, D + 1), max(1, W // 8))
    f100 = 100_000 // res
    le_T100 = coarse_track(le_T, f100)
    e1_100 = coarse_track(e1, f100)
    cp_T100 = coarse_track(cp_T, f100)
    refL = ref_loops(cell, chrom, res, n, D)
    if b["set"] == "trainstretch":
        lab_ok = np.zeros(n, bool)
        lab_ok[np.flatnonzero(mask[:, 0])] = True
        refL = refL[lab_ok[refL[:, 0]] & lab_ok[refL[:, 1]]] if len(refL) else refL
        if len(refL):
            keep = mask[refL[:, 0], np.minimum(refL[:, 1] - refL[:, 0], D)]
            refL = refL[keep]
    rows = []

    def native(name: str, P: np.ndarray) -> dict:
        r = dict(source=name)
        r["oe_pearson"] = band_pearson(Tm, P)
        tp = tile_pearsons(Tm, P, tile)
        r["oe_pearson_tile_median"] = float(np.median(tp)) if tp.size else np.nan
        r["n_tiles"] = int(tp.size)
        pdp = per_distance_pearson(Tm, P, 0)
        r["oe_pearson_auc"] = auc01(pdp[2:])
        for kb in (50, 100, 250, 500, 1000):
            d = kb * 1000 // res
            if d <= D:
                r[f"oe_r_{kb}kb"] = pdp[d]
        r["oe_scc"] = band_scc(Tm, P)
        it, ip = insulation_band(Tm, ins_w), insulation_band(P, ins_w)
        r["insulation_r"] = _corr(it, ip)
        lm = rank_loop_metrics(Tm, P, refL, res) if len(refL) else {}
        r["loop_recall_rank"] = lm.get("loop_recall", np.nan)
        r["apa_ratio"] = lm.get("apa_ratio", np.nan)
        r["anchor_auroc"] = lm.get("anchor_auroc", np.nan)
        r["n_ref_loops"] = int(len(refL))
        cp = contact_preference(P, lab, dmin_cp)
        cp100 = coarse_track(cp, f100)
        r["comp_pref_r"] = _corr(cp_T100, cp100)
        ok = np.isfinite(cp100) & np.isfinite(e1_100) & (e1_100 != 0)
        r["comp_sign_agree"] = (float(np.mean(np.sign(cp100[ok]) == np.sign(e1_100[ok])))
                                if ok.sum() > 10 else np.nan)
        r["comp_pref_e1_r"] = _corr(e1_100, cp100)
        s = saddle_strength(P, e1, dmin_cp)
        r["saddle_strength"] = s
        r["saddle_ratio"] = s / sad_T if np.isfinite(sad_T) and sad_T else np.nan
        # the window-level combined score (checkpoint selection) on the
        # chromosome cut into non-overlapping model windows
        wins = []
        for s0 in range(0, n - W + 1, W):
            Tw = band_to_dense(Tm, s0, W)
            if np.isfinite(Tw).mean() > 0.3:
                wins.append((band_to_dense(P, s0, W), Tw))
        r["window_combined"] = combined_score(wins, res) if wins else np.nan
        le = coarse_track(local_e1(P, gc, min(W, D + 1), max(1, W // 8)), f100)
        r["local_e1_r"] = _corr(le_T100, le)
        okl = np.isfinite(le) & np.isfinite(le_T100)
        r["local_e1_sign_agree"] = (float(np.mean(np.sign(le[okl]) == np.sign(le_T100[okl])))
                                    if okl.sum() > 10 else np.nan)
        r["local_e1_vs_measured_e1_r"] = _corr(e1_100, le)
        return r

    for s in src:
        r = native(s["name"], Mm[s["name"]])
        r.update(kind=s["kind"], flag=s["flag"])
        rows.append(r)
    # the target's own compartment/insulation summaries (ceilings of the band)
    tgt = dict(source="target", kind="target", flag="target",
               saddle_strength=sad_T, comp_pref_e1_r=_corr(e1_100, cp_T100),
               local_e1_vs_measured_e1_r=_corr(e1_100, le_T100))
    okt = np.isfinite(cp_T100) & np.isfinite(e1_100) & (e1_100 != 0)
    tgt["comp_sign_agree"] = (float(np.mean(np.sign(cp_T100[okt]) == np.sign(e1_100[okt])))
                              if okt.sum() > 10 else np.nan)
    rows.append(tgt)
    nat = pd.DataFrame(rows)
    log(f"[score] {b['id']}: native measures for {len(src)} sources "
        f"({time.time() - t0:.0f} s)")

    # ---------------- count space ----------------
    exp_b, off, scale = E["expected"], E["offset"], float(E["scale"])
    cd = bd / "counts"
    cd.mkdir(parents=True, exist_ok=True)
    import cooler
    chromsize = int(cooler.Cooler(str(target_cool(cell, res))).chromsizes[chrom])
    Tc = to_counts(Tm, exp_b, off, scale)
    count_maps = {"target": Tc}
    for s in src:
        count_maps[s["name"]] = to_counts(Mm[s["name"]], exp_b, off, scale)
    # the measured map itself (balanced counts at the raw scale) on the same
    # pixels: what the round trip through the smoothed O/E target costs
    w = E["weight"]
    ri, rj = E["raw_i"], E["raw_j"]
    bal = E["raw_v"] * w[ri] * w[rj] * scale
    raw = band_from_pixels(ri, rj, bal, n, D)
    count_maps["measured_balanced"] = np.where(mask, np.nan_to_num(raw), np.nan)
    for k, C in count_maps.items():
        p = cd / f"{k}.cool"
        if not p.exists():
            write_count_cool(C, chrom, res, p, chromsize)
        g = cd / f"{k}_pixels.txt.gz"
        if not g.exists():
            write_pixels_gz(C, chrom, res, g)
    preds = [k for k in count_maps if k != "target"]
    cnt = {k: {"source": k} for k in preds}
    for k in preds:
        cnt[k]["Pearson"] = hicgan_pearson_auc(Tc, count_maps[k], res)

    # external tools, in parallel (one thread each)
    def tads(k):
        return k, run_findtads(cd / f"{k}.cool", bd / "tads" / k, res, chrom)

    def loops(k):
        # hicDetectLoops on every map at 10 kb, except the masked
        # trainstretch blocks (rank-matched loop measures only there)
        if res != 10000 or b["set"] == "trainstretch":
            return k, None
        return k, run_detectloops(cd / f"{k}.cool", bd / "loops" / f"{k}.bedgraph", chrom)

    def ent(k):
        return k, run_ent3c(cd / "target.cool", cd / f"{k}.cool", k, chrom, res, bd)

    slow = slow_tools(b)
    tasks = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        nch = max(1, min(jobs, len(preds))) if slow else 0
        rq_f = [ex.submit(run_repqc, cd / "target_pixels.txt.gz",
                          {k: cd / f"{k}_pixels.txt.gz" for k in preds[c::nch]},
                          chrom, res, n, bd, f"part{c}") for c in range(nch)]
        tasks.append(("hicrep", ex.submit(run_hicrep, cd / "target.cool",
                                          {k: cd / f"{k}.cool" for k in preds}, chrom, bd)))
        tad_f = [ex.submit(tads, k) for k in ["target"] + preds]
        loop_f = [ex.submit(loops, k) for k in ["target"] + preds]
        ent_f = [ex.submit(ent, k) for k in preds] if slow else []
        tad_res = dict(f.result() for f in tad_f)
        loop_res = dict(f.result() for f in loop_f)
        ent_res = dict(f.result() for f in ent_f)
        rq = (pd.concat([f.result() for f in rq_f], ignore_index=True) if rq_f
              else pd.DataFrame())
        hr = tasks[0][1].result()
    for k in preds:
        cnt[k].update(tad_measures(tad_res.get(k), tad_res.get("target"), res))
        cnt[k]["ENT3C"] = ent_res.get(k, np.nan)
        cnt[k]["HiCRep"] = hr.get(k, np.nan) if hr.get(k) is not None else np.nan
        if len(rq) and k in set(rq.source):
            row = rq[rq.source == k].iloc[0]
            for m in ("GenomeDISCO", "HiC-Spector", "QuASAR-Rep"):
                cnt[k][m] = float(row[m]) if m in row else np.nan
        if res == 10000:
            pl = called_loops(loop_res.get(k), res)
            if pl is not None and len(refL):
                pl = pl[(pl[:, 1] - pl[:, 0] >= 2) & (pl[:, 1] - pl[:, 0] <= D)]
                if b["set"] == "trainstretch" and len(pl):
                    pl = pl[mask[pl[:, 0], pl[:, 1] - pl[:, 0]]]
                hr_, hp_ = match_loops(refL, pl, 1)
                rc = hr_ / len(refL)
                pr = hp_ / len(pl) if len(pl) else np.nan
                cnt[k]["loop_recall_call"] = rc
                cnt[k]["loop_precision_call"] = pr
                cnt[k]["loop_f1_call"] = (2 * rc * pr / (rc + pr)
                                          if np.isfinite(pr) and rc + pr > 0 else 0.0)
                cnt[k]["n_loops_called"] = int(len(pl))
    cdf = pd.DataFrame(list(cnt.values()))
    df = nat.merge(cdf, on="source", how="outer")
    df.loc[df.source == "measured_balanced", ["kind", "flag"]] = ["measured", "roundtrip_ceiling"]
    for k, v in (("set", b["set"]), ("cell", cell), ("res", res), ("chrom", chrom),
                 ("band_bins", D), ("block", b["id"]), ("n_pixels", int(mask.sum()))):
        df[k] = v
    df.to_csv(outf, index=False)
    log(f"[score] {b['id']}: done ({time.time() - t0:.0f} s)")
    return outf


def block_ready(b: dict) -> bool:
    """Every model prediction of the block exists (a block is scored once,
    with all of its sources)."""
    return all(Path(s["path"]).exists() for s in block_source_list(b) if s["kind"] == "model")


def stage_score(jobs: int, only: list[str] | None = None, wait: bool = False) -> None:
    todo = [b for b in blocks() if not (OUT / "blocks" / b["id"] / "measures.csv").exists()]
    if only:
        todo = [b for b in todo if any(re.search(o, b["id"]) for o in only)]
    # held-out cells first, then the clean training-cell sets
    order = {"heldout": 0, "trainchrom": 1, "hicreg": 2, "epiphany": 2, "trainstretch": 3}
    todo.sort(key=lambda b: (order[b["set"]], -b["res"], b["cell"], b["chrom"]))
    log(f"[score] {len(todo)} blocks to score")
    while todo:
        ready = [b for b in todo if block_ready(b)]
        if not ready:
            if not wait:
                log(f"[score] {len(todo)} blocks wait for predictions")
                return
            time.sleep(60)
            continue
        b = ready[0]
        todo.remove(b)
        f = OUT / "blocks" / b["id"] / "measures.csv"
        if f.exists() or not claim(f):
            continue
        try:
            score_block(b, jobs)
        except Exception as e:                       # noqa: BLE001
            import traceback
            log(f"[score] FAILED {b['id']}: {e!r}\n{traceback.format_exc()}")
        finally:
            release(f)


def stage_ceilings() -> None:
    """Rank-matched loop recall of the target against itself per block (the
    ceiling of that measure: the target's own strongest donut peaks against
    the hicDetectLoops calls of the measured map)."""
    rows = []
    for b in blocks():
        if not (OUT / "blocks" / b["id"] / "measures.csv").exists():
            continue
        cell, res, chrom, D = b["cell"], b["res"], b["chrom"], b["D"]
        E = np.load(expected_file(cell, res, chrom))
        n = int(E["n"])
        T = load_band(target_cool(cell, res), chrom, n, D)
        T[T == 0] = np.nan
        if b["set"] == "trainstretch":
            T = masked(T, holdout_stretch_mask(chrom, n, D, res))
        ref = ref_loops(cell, chrom, res, n, D)
        if b["set"] == "trainstretch" and len(ref):
            m = holdout_stretch_mask(chrom, n, D, res)
            ref = ref[m[ref[:, 0], np.minimum(ref[:, 1] - ref[:, 0], D)]]
        lm = rank_loop_metrics(T, T, ref, res) if len(ref) else {}
        rows.append(dict(block=b["id"], set=b["set"], cell=cell, res=res, chrom=chrom,
                         n_ref_loops=len(ref), loop_recall_rank_target=lm.get("loop_recall")))
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "loop_recall_ceiling.csv", index=False)
    log(f"[ceilings] {len(rows)} blocks")


def collect() -> pd.DataFrame:
    fs = sorted((OUT / "blocks").glob("*/measures.csv"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


# ===================================================================== #
# stage: combine (regressors of the hyperparameter-scoring work)
# ===================================================================== #
# (label, file) -- the lowest-error pipelines of the original selection per
# regressor and pipelines with the measure subsets that the nested
# cross-validation of the revision selected most often
REGRESSOR_FILES = [
    ("TabPFN_8m", "best_model_TabPFN_ENT3C QuASAR-Rep HiC-Spector GenomeDISCO TAD Score MSE "
                  "TAD Fraction TAD Fraction Exact Match HiCRep_degree_2.pkl"),
    ("TabPFN_6m", "best_model_TabPFN_QuASAR-Rep HiC-Spector Pearson TAD Score MSE TAD Fraction "
                  "HiCRep_degree_6.pkl"),
    ("GB_7m", "best_model_gradientBoosting_QuASAR-Rep HiC-Spector GenomeDISCO Pearson "
              "TAD Score MSE TAD Fraction TAD Fraction Exact Match_degree_1.pkl"),
    ("GB_4m", "best_model_gradientBoosting_GenomeDISCO TAD Score MSE TAD Fraction "
              "HiCRep_degree_3.pkl"),
    ("RF_4m", "best_model_random forest_GenomeDISCO TAD Score MSE TAD Fraction "
              "HiCRep_degree_1.pkl"),
]


def stage_combine() -> None:
    """Runs under the hyperparameter_gpu python (joblib pipelines, TabPFN on
    the GPU it was saved for); re-executes itself there when needed."""
    if Path(sys.executable).resolve() != HYPER_PY.resolve():
        env = dict(os.environ, OMP_NUM_THREADS="8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        rc = subprocess.call([str(HYPER_PY), __file__, "--stage", "combine"], env=env, cwd=REPO)
        if rc:
            log(f"[combine] FAILED rc={rc}")
        return
    import warnings
    warnings.filterwarnings("ignore")
    import joblib
    d = collect()
    if d.empty:
        return
    train = pd.read_csv(REGRESSOR_ROWS)
    rows = d[d.source != "target"].copy()
    lo, hi = train[NINE].min(), train[NINE].max()
    q99 = train[NINE].quantile(0.99); q01 = train[NINE].quantile(0.01)
    for m in NINE:
        rows[f"in_domain_{m}"] = rows[m].between(lo[m], hi[m])
    rows["n_measures_in_domain"] = rows[[f"in_domain_{m}" for m in NINE]].sum(1)
    for label, f in REGRESSOR_FILES:
        p = REGRESSORS / f
        if not p.exists():
            log(f"[combine] missing {p.name}")
            continue
        cols = parse_subset(f)
        model = joblib.load(p)
        X = rows[cols].astype(float)
        ok = X.notna().all(1).values
        pred = np.full(len(rows), np.nan)
        if ok.any():
            pred[ok] = model.predict(X.values[ok])
        rows[f"combined_{label}"] = pred
        rows[f"domain_{label}"] = rows[[f"in_domain_{m}" for m in cols]].all(1)
        tr = model.predict(train[cols].values)
        log(f"[combine] {label}: {len(cols)} measures, training rows r(ELO) "
            f"{np.corrcoef(tr, train.ELO)[0, 1]:.3f}, prediction range on training rows "
            f"{tr.min():.1f}..{tr.max():.1f}; here {np.nanmin(pred):.1f}..{np.nanmax(pred):.1f}")
    keep = ["block", "set", "cell", "res", "chrom", "source", "kind", "flag"]
    out = rows[keep + [c for c in rows.columns if c.startswith(("combined_", "domain_",
                                                                "in_domain_", "n_measures"))]]
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "combined_scores.csv", index=False)
    dom = pd.DataFrame({"measure": NINE, "train_min": lo.values, "train_q01": q01.values,
                        "train_median": train[NINE].median().values,
                        "train_q99": q99.values, "train_max": hi.values,
                        "here_median_25kb": [rows[rows.res == 25000][m].median() for m in NINE],
                        "here_min_25kb": [rows[rows.res == 25000][m].min() for m in NINE],
                        "here_max_25kb": [rows[rows.res == 25000][m].max() for m in NINE],
                        "n_scored_25kb": [int(rows[rows.res == 25000][m].notna().sum())
                                          for m in NINE],
                        "share_in_range_25kb": [rows[(rows.res == 25000) & rows[m].notna()]
                                                [f"in_domain_{m}"].mean() for m in NINE]})
    dom.to_csv(TABLES / "regressor_domain.csv", index=False)
    log(f"[combine] wrote {TABLES / 'combined_scores.csv'}")


# ===================================================================== #
# stage: report
# ===================================================================== #
MEASURE_COLS = NINE + [
    "oe_pearson", "oe_pearson_tile_median", "oe_pearson_auc", "oe_scc", "insulation_r",
    "bnd_recall", "bnd_precision", "loop_recall_rank", "apa_ratio", "anchor_auroc",
    "loop_recall_call", "loop_precision_call", "loop_f1_call", "comp_pref_r",
    "comp_sign_agree", "saddle_ratio", "local_e1_r", "local_e1_sign_agree", "window_combined",
    "oe_r_50kb", "oe_r_100kb", "oe_r_250kb", "oe_r_500kb", "oe_r_1000kb"]
SOURCE_LABEL = {CASCFOLD: "all12refres setup, chromosome held out (cascade fold)",
                "hicgan_gm12878_model_on_cell": "Hi-cGAN, GM12878 model on the cell's ChIP",
                "consensus": "consensus alone", "best_other_map": "best other measured map",
                "replicate": "replicate map (ceiling)",
                "measured_balanced": "measured map, balanced counts (round-trip ceiling)"}


def source_label(s: str) -> str:
    if s in SOURCE_LABEL:
        return SOURCE_LABEL[s]
    for k, v in COMP_LABEL.items():
        if s.startswith(k):
            return f"{v} ({s})"
    return s


def registry_pearson(res: int) -> pd.DataFrame:
    """Median Pearson over the 413 Akita test regions per model and cell (the
    registry metric; crosscell/predict_<cell>.csv of each run), and the same
    for the consensus and the best other map from the baseline columns."""
    rows = []
    for m in MODELS:
        for cell in HELDOUT + TRAIN12:
            f = run_dir(m, res) / "crosscell" / f"predict_{cell}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            rows.append(dict(source=m, cell=cell, res=res,
                             pearson413=float(d["model"].median())))
            if m == "all12refres":
                cc = [c for c in d.columns if c.startswith("base_consensus")]
                if cc:
                    rows.append(dict(source="consensus", cell=cell, res=res,
                                     pearson413=float(d[cc[0]].median())))
                bo = BEST_OTHER[res].get(cell)
                if bo and f"base_{bo}" in d.columns:
                    rows.append(dict(source="best_other_map", cell=cell, res=res,
                                     pearson413=float(d[f"base_{bo}"].median())))
    return pd.DataFrame(rows)


# multi-measure rank score: per block the rank of each model among the models
# of the block per measure (0 worst .. 1 best), averaged within a structure
# level, then over the four levels (equal weight per level)
MMRS_LEVELS = {
    "maps": ["oe_pearson_tile_median", "oe_scc", "HiCRep", "GenomeDISCO", "HiC-Spector",
             "QuASAR-Rep", "ENT3C", "Pearson"],
    "tads": ["insulation_r", "bnd_recall", "bnd_precision", "TAD Score MSE", "TAD Fraction",
             "TAD Fraction Exact Match"],
    "loops": ["loop_recall_rank", "apa_agreement", "anchor_auroc", "loop_f1_call"],
    "compartments": ["comp_pref_r", "comp_sign_agree", "saddle_agreement", "local_e1_r"],
}


def oriented(d: pd.DataFrame, m: str) -> pd.Series:
    """Measure turned into higher = better."""
    if m == "TAD Score MSE":
        return -d[m]
    if m == "TAD Fraction":
        return -np.abs(d[m] - 1)
    if m == "apa_agreement":
        r = d["apa_ratio"].astype(float)
        with np.errstate(divide="ignore"):
            v = np.where(r > 0, np.minimum(r, 1 / r), 0.0)
        return pd.Series(np.where(r.notna(), v, np.nan), index=d.index)
    if m == "saddle_agreement":
        r = d["saddle_ratio"].astype(float)
        with np.errstate(divide="ignore"):
            v = np.where(r > 0, np.minimum(r, 1 / r), 0.0)
        return pd.Series(np.where(r.notna(), v, np.nan), index=d.index)
    return d[m]


def model_sources(set_: str) -> list[str]:
    return MODELS + ([CASCFOLD] if set_ == "trainchrom" else [])


def add_rank_score(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["mmrs"] = np.nan
    for lv in MMRS_LEVELS:
        d[f"mmrs_{lv}"] = np.nan
    for blk, g in d.groupby("block"):
        g = g[g.source.isin(model_sources(g.set.iloc[0]))]
        if len(g) < 3:
            continue
        lvs = []
        for lv, ms in MMRS_LEVELS.items():
            rk = []
            for m in ms:
                if m not in g.columns and m not in ("apa_agreement", "saddle_agreement"):
                    continue
                v = pd.Series(np.asarray(oriented(g, m), float), index=g.index)
                if v.notna().sum() < 3:
                    continue
                rk.append((v.rank(method="average") - 1) / (v.notna().sum() - 1))
            if rk:
                s = pd.concat(rk, axis=1).mean(1)
                d.loc[g.index, f"mmrs_{lv}"] = s
                lvs.append(s)
        if lvs:
            d.loc[g.index, "mmrs"] = pd.concat(lvs, axis=1).mean(1)
    return d


def stage_report() -> None:
    d = collect()
    if d.empty:
        log("[report] nothing scored yet")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    cs = TABLES / "combined_scores.csv"
    if cs.exists():
        c = pd.read_csv(cs)
        keyc = ["block", "source"]
        d = d.merge(c[keyc + [x for x in c.columns if x.startswith(("combined_", "domain_",
                                                                    "n_measures"))]],
                    on=keyc, how="left")
    d = add_rank_score(d)
    # one row label for the Hi-cGAN GM12878 model on the three held-out cells
    d["competitor"] = np.where(d.kind == "competitor", d.source, "")
    d["source"] = d.source.str.replace(r"^hicgan_gm12878_on_(k562|hmec|huvec)_\d+kb$",
                                       "hicgan_gm12878_model_on_cell", regex=True)
    d.to_csv(TABLES / "measures_per_chromosome.csv", index=False)
    comb = [x for x in d.columns if x.startswith("combined_")]
    cols = [c for c in MEASURE_COLS + ["mmrs"] + [f"mmrs_{lv}" for lv in MMRS_LEVELS] + comb
            if c in d.columns]
    per_cell = (d[d.source != "target"].groupby(["set", "res", "cell", "source"])[cols]
                .mean().reset_index())
    reg = pd.concat([registry_pearson(r) for r in RESOLUTIONS], ignore_index=True)
    per_cell = per_cell.merge(reg, on=["source", "cell", "res"], how="left")
    cols = cols + ["pearson413"]
    per_cell.to_csv(TABLES / "measures_per_cell.csv", index=False)
    g = per_cell.groupby(["set", "res", "source"])
    summ = g[cols].median().reset_index()
    summ["n_cells"] = g.size().values
    summ["label"] = summ.source.map(source_label)
    summ.to_csv(TABLES / "summary.csv", index=False)
    rk = ranking_tables(per_cell, cols)
    rk.to_csv(TABLES / "ranking.csv", index=False)
    figures(per_cell, summ, cols, rk)
    write_status(d, summ, rk)
    log(f"[report] tables in {TABLES}")


def ranking_tables(per_cell: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per set and resolution, over the models of the set (median over
    cells): Kendall tau and Spearman rho of every measure's model ranking
    against the ranking by the registry median Pearson (413 regions) and by
    the median 1 Mb tile Pearson of this evaluation; best model by each."""
    from scipy.stats import kendalltau, spearmanr
    out = []
    for (set_, res), g in per_cell.groupby(["set", "res"]):
        models = [m for m in model_sources(set_) if m in set(g.source)]
        if len(models) < 3:
            continue
        gm = g[g.source.isin(models)]
        med = gm.groupby("source")[cols].median().reindex(models)
        for basis in ("pearson413", "oe_pearson_tile_median"):
            base = med[basis]
            for m in cols:
                if m == basis:
                    continue
                key = {"apa_ratio": "apa_agreement", "saddle_ratio": "saddle_agreement"}.get(m, m)
                v = pd.Series(np.asarray(oriented(med, key), float), index=med.index)
                ok = v.notna() & base.notna()
                if ok.sum() < 3:
                    continue
                vv, bb = v[ok], base[ok]
                out.append(dict(set=set_, res=res, basis=basis, measure=m,
                                n_models=int(ok.sum()),
                                kendall=kendalltau(bb, vv).statistic,
                                spearman=spearmanr(bb, vv).statistic,
                                n_ties=int(len(vv) - vv.nunique()),
                                best_by_measure=vv.idxmax(), best_by_basis=bb.idxmax(),
                                ranking=" > ".join(vv.sort_values(ascending=False).index)))
    return pd.DataFrame(out)


def figures(per_cell, summ, cols, rk) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [c for c in ["pearson413", "oe_pearson_tile_median", "oe_scc", "insulation_r",
                        "bnd_recall", "loop_recall_rank", "apa_ratio", "loop_f1_call",
                        "comp_pref_r", "comp_sign_agree", "saddle_ratio", "local_e1_r"] + NINE
            + ["window_combined", "mmrs"] + [c for c in cols if c.startswith("combined_")]
            if c in summ.columns]
    for (set_, res), g in summ.groupby(["set", "res"]):
        g = g.set_index("source")
        order = [m for m in model_sources(set_) + ["consensus", "best_other_map", "replicate",
                                                     "measured_balanced"] if m in g.index]
        order += [s for s in g.index if s not in order]
        M = g.loc[order, show].astype(float)
        O = pd.DataFrame({c: np.asarray(oriented(M, c) if c in ("TAD Score MSE", "TAD Fraction")
                                         else M[c], float) for c in show}, index=order)
        Z = (O - O.mean()) / O.std().replace(0, np.nan)
        fig, ax = plt.subplots(figsize=(0.55 * len(show) + 4.5, 0.36 * len(order) + 2.2))
        ax.imshow(Z.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        for i in range(len(order)):
            for j in range(len(show)):
                v = M.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5)
        ax.set_xticks(range(len(show)), show, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(order)), [source_label(s) for s in order], fontsize=7)
        ax.set_title(f"{set_}, {res // 1000} kb: median over cells; colour = z-score per "
                     f"column, red = better", fontsize=8)
        fig.tight_layout()
        fig.savefig(TABLES / f"heatmap_{set_}_{res // 1000}kb.png", dpi=140)
        plt.close(fig)
    # agreement of every measure's model ranking with the median-Pearson ranking
    r = rk[rk.basis == "pearson413"]
    if r.empty:
        return
    keys = sorted(r[["set", "res"]].drop_duplicates().itertuples(index=False, name=None))
    ms = [m for m in show if m != "pearson413" and m in set(r.measure)]
    fig, ax = plt.subplots(figsize=(0.45 * len(ms) + 3, 0.45 * len(keys) + 2))
    K = np.full((len(keys), len(ms)), np.nan)
    for i, (s, rr) in enumerate(keys):
        for j, m in enumerate(ms):
            v = r[(r.set == s) & (r.res == rr) & (r.measure == m)].kendall
            if len(v):
                K[i, j] = v.iloc[0]
    ax.imshow(K, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if np.isfinite(K[i, j]):
                ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(ms)), ms, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(keys)), [f"{s}, {rr // 1000} kb" for s, rr in keys], fontsize=7)
    ax.set_title("Kendall tau: model ranking by each measure vs by median Pearson "
                 "(413 regions)", fontsize=8)
    fig.tight_layout()
    fig.savefig(TABLES / "ranking_agreement.png", dpi=140)
    plt.close(fig)


REG_A = [("pearson413", "Pearson 413 regions"), ("oe_pearson_tile_median", "Pearson 1 Mb tiles"),
         ("oe_scc", "SCC"), ("insulation_r", "insulation r"), ("bnd_recall", "boundary recall"),
         ("bnd_precision", "boundary precision"), ("loop_recall_rank", "loop recall"),
         ("apa_ratio", "APA ratio"), ("loop_f1_call", "loop F1 (called)"),
         ("comp_pref_r", "compartment preference r"), ("comp_sign_agree", "A/B sign agreement"),
         ("saddle_ratio", "saddle ratio"), ("local_e1_r", "local E1 r"),
         ("window_combined", "window combined"), ("mmrs", "rank score")]
REG_B = [("ENT3C", "ENT3C"), ("HiCRep", "HiCRep"), ("GenomeDISCO", "GenomeDISCO"),
         ("HiC-Spector", "HiC-Spector"), ("QuASAR-Rep", "QuASAR-Rep"), ("Pearson", "Pearson AUC"),
         ("TAD Score MSE", "TAD score MSE"), ("TAD Fraction", "TAD fraction"),
         ("TAD Fraction Exact Match", "TAD FEM"), ("combined_TabPFN_8m", "TabPFN 8m"),
         ("combined_TabPFN_6m", "TabPFN 6m"), ("combined_GB_7m", "GB 7m"),
         ("combined_GB_4m", "GB 4m"), ("combined_RF_4m", "RF 4m")]
REG_SET = {"heldout": "held-out cells K562, HMEC deep, HUVEC deep; chr2, chr12, chr18",
           "trainstretch": "12 training cells, Akita valid+test stretches of chr2 and chr18",
           "trainchrom": "12 training cells, whole chr2 and chr18",
           "hicreg": "GM12878, whole chr2 and chr18, 1 Mb band (HiC-Reg reach)",
           "epiphany": "GM12878, whole chr17, 990 kb band (Epiphany reach)"}


def reg_label(s: str, set_: str, res: int) -> str:
    lab = {CASCFOLD: "cascade fold model (all12refres setup, chromosome held out)",
           "consensus": "consensus alone", "best_other_map": "best other measured map",
           "replicate": "replicate map", "measured_balanced": "measured map, balanced counts",
           "hicgan_gm12878_model_on_cell": "Hi-cGAN, GM12878 model",
           "hicgan_gm12878_25kb": "Hi-cGAN (GM12878, held-out chromosome)",
           "hicgan_gm12878_10kb": "Hi-cGAN (GM12878, held-out chromosome)",
           "hicgan_gm12878_on_imr90_10kb": "Hi-cGAN, GM12878 model on IMR-90 (chr2)",
           "hicgan_imr90_on_imr90_10kb": "Hi-cGAN, IMR-90 model (chr2)",
           "corigami_imr90_8kb": "C.Origami (IMR-90, chr2)",
           "hicreg_gm12878_25kb": "HiC-Reg (within-chromosome CV)",
           "epiphany_gm12878_10kb": "Epiphany (held-out chr17)"}.get(s, s)
    if s == "all12refres" and set_ in ("trainchrom", "hicreg", "epiphany"):
        lab += " (chromosome in training)"
    elif s in MODELS and set_ in ("hicreg", "epiphany"):
        lab += " (chromosome in training)"
    return lab


def stage_tables() -> None:
    """Markdown tables of summary.csv for the registry
    (runs/hic_measures/registry_tables.md)."""
    summ = pd.read_csv(TABLES / "summary.csv")
    out = []
    for set_ in ("heldout", "trainstretch", "trainchrom", "hicreg", "epiphany"):
        for res in RESOLUTIONS:
            g = summ[(summ.set == set_) & (summ.res == res)].set_index("source")
            if g.empty:
                continue
            order = [m for m in model_sources(set_) + ["consensus", "best_other_map",
                                                         "replicate", "measured_balanced"]
                     if m in g.index]
            order += sorted(s for s in g.index if s not in order)
            for cols, what in ((REG_A, "log O/E structure measures"),
                               (REG_B, "count-space measures and regressor scores")):
                cols = [(c, h) for c, h in cols if c in g.columns and g[c].notna().any()]
                out.append(f"{REG_SET[set_]}, {res // 1000} kb, {what} (median over "
                           f"{int(g.n_cells.max())} cells of the per-cell mean over "
                           "chromosomes):")
                out.append("")
                out.append("| source | " + " | ".join(h for _, h in cols) + " |")
                out.append("|---" * (len(cols) + 1) + "|")
                for src in order:
                    r = g.loc[src]
                    vals = " | ".join("" if pd.isna(r[c]) else
                                      (f"{r[c]:.2f}" if c.startswith("combined_") else f"{r[c]:.3f}")
                                      for c, _ in cols)
                    out.append(f"| {reg_label(src, set_, res)} | {vals} |")
                out.append("")
    (OUT / "registry_tables.md").write_text("\n".join(out) + "\n")
    log(f"[tables] {OUT / 'registry_tables.md'}")


def write_status(d: pd.DataFrame, summ: pd.DataFrame, rk: pd.DataFrame) -> None:
    nb = d.block.nunique()
    tot = len(blocks())
    lines = ["# hic_measures status", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}. Blocks scored: {nb} of {tot}.", ""]
    for (set_, res), g in summ.groupby(["set", "res"]):
        lines.append(f"## {set_}, {res // 1000} kb (median over cells)")
        lines.append("")
        show = [c for c in ["pearson413", "oe_pearson_tile_median", "oe_scc", "insulation_r",
                            "bnd_recall", "loop_recall_rank", "comp_pref_r", "HiCRep",
                            "GenomeDISCO", "TAD Fraction", "ENT3C", "window_combined", "mmrs"]
                + [c for c in g.columns if c.startswith("combined_")] if c in g.columns]
        lines.append("| source | n | " + " | ".join(show) + " |")
        lines.append("|---" * (len(show) + 2) + "|")
        for _, r in g.iterrows():
            vals = " | ".join("" if pd.isna(r[c]) else f"{r[c]:.3f}" for c in show)
            lines.append(f"| {r['label']} | {r['n_cells']} | {vals} |")
        lines.append("")
    (OUT / "STATUS.md").write_text("\n".join(lines) + "\n")


# ===================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["predict", "expected", "score", "ceilings", "combine", "report",
                             "tables"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None, help="regexes of block ids")
    ap.add_argument("--gpu", default="", help="predict: GPU index for this worker (default CPU)")
    ap.add_argument("--wait", action="store_true",
                    help="score: wait for missing predictions instead of stopping")
    a = ap.parse_args()
    for st in a.stage:
        if st == "predict":
            stage_predict(a.threads, 1 if a.gpu else max(1, min(a.jobs, a.threads)), a.gpu)
        elif st == "expected":
            stage_expected(min(a.jobs, 4))
        elif st == "score":
            stage_score(a.jobs, a.only, a.wait)
        elif st == "ceilings":
            stage_ceilings()
        elif st == "combine":
            stage_combine()
        elif st == "report":
            stage_report()
        elif st == "tables":
            stage_tables()


if __name__ == "__main__":
    main()
