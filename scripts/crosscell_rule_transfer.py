"""Are the mark-to-structure rules of AML-specific loops and stripes the same in other cell types? (registry section 65)

Premise: the way chromatin marks and methylation determine loops, stripes and TADs does not depend on the
cell type, so the rule "the cell's pattern of H3K27ac / H3K27me3 / CTCF (and methylation) relative to a
reference leads to a gained loop or stripe" must hold in the 12 training cell lines, the held-out cells and
the AML patients alike. This script builds the cell-specific loops and stripes of every cell with the same
operational definitions as the AML objects (scripts/aml_loop_classifier.py, aml_loop_detection.py), derives
cell-agnostic features (cell minus leave-one-out consensus, multi-scale, per-cell standardised), tests every
rule within each cell and transfers classifiers / regressors between cells and AML patients.

Cells (15): the 12 training cells of scripts/multicell_expanded.py (gm12878, imr90, hct116, hff, h1, hepg2,
a549, nhek, gm23248, mcf7, cd14mono, pc9) with the leave-one-out consensus of the other 11
(crosscell_oe/consensus_loo_<cell>_proper_10kb.cool) and the held-out k562, hmec (deep map), huvec (deep map)
with the consensus of the 12 (consensus12). Cancer lines: k562, hct116, mcf7, a549, hepg2, pc9.

Definitions (10 kb, log O/E maps from scripts/oe_proper.py as for the AML patients)
  enrichment   log2 donut enrichment (flexcible.bioscore.local_enrichment, inner 1, outer 4) at the local
               maxima of the cell map (> 0.4, distance MIN_BINS to 150 bins), consensus value = 3x3 maximum
               around the pixel; gain = cell enrichment minus consensus enrichment (as aml_loop_classifier.hic_one).
  candidates   pairs of peak bins (top 8 percent of H3K27ac or CTCF, in the cell or in the consensus track).
  specific     enrichment >= E_SPEC and gain >= G_LOOP;  shared = enrichment >= 0.6 in the cell and in the
               consensus (not within 2 bins of a specific loop); matched 1:1 on distance, CTCF motif orientation
               class and the anchors' mean H3K27ac (A_full) or on the anchors' consensus CTCF and H3K27ac (A_ref).
  stripes      rectangle mean of log O/E (4 anchor bins x 40 extent bins, extent 50 to 90 bins to the right or
               left; scripts/aml_loop_classifier.shift_scores), anchor containing a peak bin; specific = cell
               score >= S_MIN and gain >= G_STRIPE; shared = both scores >= S_MIN; matched on direction,
               orientation class and anchor H3K27ac.
  types        loops PE / PS / other and stripes enh / rep as in aml_loop_classifier.pair_feats (TSS in one anchor
               and partner H3K27ac / H3K27me3 above the 70th percentile of the pooled training scale).

Tracks of every cell and AML patient are quantile-normalised at 10 kb to the average training-cell
distribution; AML references are the HSPC CD34+ track (and the mean of the other AML donors).

Stages
    python scripts/crosscell_rule_transfer.py --stage tracks hic objects features rules transfer figures
Run with the hicdiff conda env (cooler, pyBigWig, scikit-learn), CPU only, nice 10, <= 8 threads.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import aml_loop_classifier as alc  # noqa: E402  (helpers: windows, banded, shift_scores, matching, models, metrics)

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = HM / "crosscell_oe"
DATA = Path.home() / "data/hic-multiome"
WORK = HM / "crosscell_rules"
OUT = REPO / "results/crosscell_rules"
AML_CACHE = alc.CACHE                      # 10 kb track caches of the AML patients and HSPC
AML_METH = alc.METH
CLF = REPO / "results/aml_loop_classifier_v2/features.csv.gz"
DET = REPO / "results/aml_loop_detection"

RES = 10_000
CHROMS = alc.CHROMS
MARKS = ["H3K27ac", "H3K27me3", "CTCF"]
SCALES = [0, 2, 10]                        # half-widths in bins around the anchor: 10 kb, +-20 kb, +-100 kb
SCALE_NAME = {0: "10kb", 2: "20kb", 10: "100kb"}
MIN_BINS, MAX_BINS = alc.MIN_BINS, alc.MAX_BINS
PEAK_Q = 0.92
E_MIN = 0.6                                # shared loop: enrichment in the cell and the consensus
E_SPEC, G_LOOP = 0.5, 0.30                 # cell-specific loop: enrichment and gain over the consensus
RA, EB, OFF = 4, 40, 50                    # stripe geometry: anchor bins, extent bins, extent start offset
S_MIN, G_STRIPE = 0.55, 0.20               # stripe score of a stripe, gain of a cell-specific stripe
SEED = 0
CAP_POS = 2500                             # specific objects per cell and kind kept for matching / modelling

TRAIN = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248", "mcf7", "cd14mono", "pc9"]
HELD = ["k562", "hmec", "huvec"]
CELLS = TRAIN + HELD
CANCER = {"k562", "hct116", "mcf7", "a549", "hepg2", "pc9"}
COOL = {c: OE / f"{c}_proper_10kb.cool" for c in CELLS}
COOL["hmec"] = OE / "hmec_deep_proper_10kb.cool"
COOL["huvec"] = OE / "huvec_deep_proper_10kb.cool"
CONS = {c: OE / (f"consensus_loo_{c}_proper_10kb.cool" if c in TRAIN else "consensus12_proper_10kb.cool") for c in CELLS}
CHIP = {c: HM / f"hg38_{c}" for c in CELLS}
CHIP["gm12878"] = DATA / "hg38_gm12878"
METHDIR = {c: HM / "meth" / c / "Meth.bigwig" for c in CELLS}
AML_DONORS = ["018", "027", "168", "270", "472", "546", "629", "773", "1360", "1916"]
AML_CTCF = ["018", "027", "168", "270", "472", "546", "629", "773"]
DONOR_GROUP = {"168": "168+496", "496": "168+496", "260": "260+424", "424": "260+424"}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sizes() -> dict:
    return alc.chrom_sizes()


def nbins(c: str, sz: dict) -> int:
    return sz[c] // RES + 1


def load_arr(path: Path, key: str | None = None) -> dict:
    z = np.load(path)
    return {c: z[c] for c in CHROMS} if key is None else {c: z[f"{c}|{key}"] for c in CHROMS}


# ================================================================================ tracks
def raw_track_job(job):
    """10 kb bin mean of a ChIP bigWig (or methylation fraction = value - 1) of one cell."""
    import pyBigWig
    cell, mark = job
    dest = WORK / f"raw_{cell}_{mark}.npz"
    if dest.exists():
        return job, True
    f = METHDIR[cell] if mark == "Meth" else CHIP[cell] / f"{mark}.bigwig"
    if not f.exists():
        return job, False
    sz = sizes()
    bw = pyBigWig.open(str(f))
    out = {}
    for c in CHROMS:
        n, nb = nbins(c, sz), sz[c] // RES
        a = np.full(n, np.nan, np.float32)
        a[:nb] = np.array(bw.stats(c, 0, nb * RES, type="mean", nBins=nb), dtype=float)
        out[c] = a - 1.0 if mark == "Meth" else a
    bw.close()
    np.savez_compressed(dest, **out)
    return job, True


PROBS = np.r_[np.linspace(0, 0.99, 991), np.linspace(0.99, 1, 1001)[1:]]


def have(cell: str, mark: str) -> bool:
    return (WORK / f"raw_{cell}_{mark}.npz").exists()


def quantiles_of(arrs: dict) -> np.ndarray:
    v = np.concatenate([a[np.isfinite(a)] for a in arrs.values()])
    return np.quantile(v, PROBS)


def qn_apply(arrs: dict, target: np.ndarray) -> dict:
    qs = quantiles_of(arrs)
    out = {}
    for c, a in arrs.items():
        p = np.interp(a, qs, PROBS)
        out[c] = np.where(np.isfinite(a), np.interp(p, PROBS, target), np.nan).astype(np.float32)
    return out


def aml_raw(src: str, mark: str) -> dict | None:
    f = AML_CACHE / f"trk_{src}_{mark}.npz"
    return load_arr(f, "mean") if f.exists() else None


def aml_meth10(src: str) -> dict | None:
    """10 kb mean methylation fraction of an AML sample (cache of 1 kb bins, value 1 + fraction) or a
    healthy donor (bigWig)."""
    dest = WORK / f"aml_meth10_{src}.npz"
    if dest.exists():
        return load_arr(dest)
    sz = sizes()
    f1 = AML_CACHE / f"trk_{src}_Meth.npz"
    out = {}
    if f1.exists():
        z = np.load(f1)
        for c in CHROMS:
            kb = z[c].astype(float)
            n, nb = nbins(c, sz), len(kb) // 10
            a = np.full(n, np.nan, np.float32)
            a[:nb] = np.nanmean(kb[:nb * 10].reshape(nb, 10), axis=1) - 1.0
            out[c] = a
    else:
        f = AML_METH / f"xu_{src}" / "Meth.bigwig"
        if not f.exists():
            return None
        import pyBigWig
        bw = pyBigWig.open(str(f))
        for c in CHROMS:
            n, nb = nbins(c, sz), sz[c] // RES
            a = np.full(n, np.nan, np.float32)
            a[:nb] = np.array(bw.stats(c, 0, nb * RES, type="mean", nBins=nb), dtype=float) - 1.0
            out[c] = a
        bw.close()
    np.savez_compressed(dest, **out)
    return out


def stage_tracks(workers: int) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = [(c, m) for c in CELLS for m in MARKS + ["Meth"]]
    with ProcessPoolExecutor(workers) as ex:
        for (c, m), ok in ex.map(raw_track_job, jobs):
            if not ok:
                log(f"[tracks] {c} {m}: no file")
    # quantile-normalisation target: mean quantile function of the training cells
    tgt = {}
    for m in MARKS:
        qs = [quantiles_of(load_arr(WORK / f"raw_{c}_{m}.npz")) for c in TRAIN if have(c, m)]
        tgt[m] = np.mean(qs, axis=0)
    np.savez(WORK / "qn_target.npz", **tgt)
    for m in MARKS:
        for c in CELLS:
            if have(c, m):
                np.savez_compressed(WORK / f"qn_{c}_{m}.npz", **qn_apply(load_arr(WORK / f"raw_{c}_{m}.npz"), tgt[m]))
        for p in AML_DONORS + ["hspc"]:
            a = aml_raw(p, m)
            if a is not None:
                np.savez_compressed(WORK / f"qn_aml{p}_{m}.npz", **qn_apply(a, tgt[m]))
    # leave-one-out reference tracks (mean of the consensus cells)
    for c in CELLS:
        others = [x for x in TRAIN if x != c]
        for m in MARKS:
            acc = None
            others_m = [x for x in others if have(x, m)]
            for x in others_m:
                a = load_arr(WORK / f"qn_{x}_{m}.npz")
                acc = {k: v.astype(np.float64) for k, v in a.items()} if acc is None else \
                    {k: acc[k] + a[k] for k in acc}
            np.savez_compressed(WORK / f"ref_{c}_{m}.npz", **{k: (v / len(others_m)).astype(np.float32) for k, v in acc.items()})
        # methylation reference: mean of the other cells that have a usable methylation track
        ms = [x for x in others if meth_ok(x)]
        acc = None
        for x in ms:
            a = load_arr(WORK / f"raw_{x}_Meth.npz")
            acc = {k: v.astype(np.float64) for k, v in a.items()} if acc is None else {k: acc[k] + a[k] for k in acc}
        np.savez_compressed(WORK / f"ref_{c}_Meth.npz", **{k: (v / len(ms)).astype(np.float32) for k, v in acc.items()})
    log("[tracks] done")


_METH_OK: dict = {}


def meth_ok(cell: str) -> bool:
    """A methylation track is usable when it varies along the genome (flat tracks are imputed constants)."""
    if cell not in _METH_OK:
        f = WORK / f"raw_{cell}_Meth.npz"
        if not f.exists() or not METHDIR[cell].exists():
            _METH_OK[cell] = False
        else:
            a = load_arr(f)["chr1"]
            _METH_OK[cell] = bool(np.nanstd(a) > 0.06)
    return _METH_OK[cell]


def target_q() -> dict:
    z = np.load(WORK / "qn_target.npz")
    return {m: z[m] for m in MARKS}


def peak_bins(cell: str) -> dict:
    """Peak bins: top (1 - PEAK_Q) of H3K27ac or CTCF in the cell or in its consensus track."""
    tq = target_q()
    out = {c: None for c in CHROMS}
    for m in ("H3K27ac", "CTCF"):
        thr = tq[m][np.searchsorted(PROBS, PEAK_Q)]
        own = load_arr(WORK / f"qn_{cell}_{m}.npz") if have(cell, m) else None
        ref = load_arr(WORK / f"ref_{cell}_{m}.npz")
        thr_ref = np.quantile(np.concatenate([a[np.isfinite(a)] for a in ref.values()]), PEAK_Q)
        for c in CHROMS:
            p = (ref[c] >= thr_ref)
            if own is not None:
                p = p | (own[c] >= thr)
            out[c] = p if out[c] is None else out[c] | p
    return out


# ================================================================================ Hi-C
def hic_cell(cell: str) -> str:
    """Loop candidates (local maxima of the donut enrichment with both anchors on peak bins) and stripe
    scores of one cell and of its consensus."""
    import cooler
    from flexcible.bioscore import BioParams, local_enrichment
    from scipy.ndimage import maximum_filter
    dest_l, dest_s = WORK / f"loops_{cell}.csv.gz", WORK / f"stripes_{cell}.csv.gz"
    if dest_l.exists() and dest_s.exists():
        return cell
    prm = BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4)
    cp, ch = cooler.Cooler(str(COOL[cell])), cooler.Cooler(str(CONS[cell]))
    peaks = peak_bins(cell)
    sz = sizes()
    loops, stripes, valid = [], [], {}
    for chrom in CHROMS:
        n = nbins(chrom, sz)
        pk = peaks[chrom]
        v = np.zeros(n)
        for s, e in alc._windows(n):
            Mp, Mh = alc._fetch(cp, chrom, s, e), alc._fetch(ch, chrom, s, e)
            Lp = np.log2(local_enrichment(np.exp(Mp), prm))
            Lh = np.log2(local_enrichment(np.exp(Mh), prm))
            mp = maximum_filter(np.nan_to_num(Lp, nan=-9), size=3)
            mh = maximum_filter(np.nan_to_num(Lh, nan=-9), size=3)
            r0 = alc.MARGIN if s > 0 else 0
            own = np.arange(r0, alc.MARGIN + alc.STRIDE)
            isum = (np.nan_to_num(Lp, nan=-9) == mp) & (Lp > 0.4)
            ii, jj = np.nonzero(np.triu(isum, MIN_BINS))
            keep = (jj - ii <= MAX_BINS) & (ii >= r0) & (ii < alc.MARGIN + alc.STRIDE) & (s + jj < n)
            ii, jj = ii[keep], jj[keep]
            pm = pk[s + ii] & pk[s + jj]
            ii, jj = ii[pm], jj[pm]
            loops.append(pd.DataFrame({"chrom": chrom, "i": s + ii, "j": s + jj, "e_cell": Lp[ii, jj],
                                       "e_cons": mh[ii, jj]}))
            ok = np.isfinite(Mp)
            band = np.zeros(alc.WIN)
            for d in (2, 10, 40, 100):
                band += np.pad(np.diagonal(ok, d).astype(float), (0, d))
            idx = s + own
            m = idx < n
            v[idx[m]] = band[own][m] / 4.0
        valid[chrom] = v >= 0.5
        # stripes: right (extent OFF.. bins to the right) and left geometries
        Dp, Dh = alc.banded(cp, chrom), alc.banded(ch, chrom)
        for dname, o in (("R", OFF), ("L", -OFF - EB)):
            sp = alc.shift_scores(Dp, RA, EB, o)
            sh = alc.shift_scores(Dh, RA, EB, o)
            anc = np.zeros(n, bool)
            for k in range(RA):
                anc[:n - k] |= pk[k:]
            vv = valid[chrom]
            lo_b = np.minimum(np.arange(n), np.arange(n) + o)
            hi_b = np.maximum(np.arange(n) + RA, np.arange(n) + o + EB)
            ok = anc & np.isfinite(sp) & np.isfinite(sh) & (lo_b >= 0) & (hi_b < n)
            ok &= vv & vv[np.clip(np.arange(n) + o, 0, n - 1)]
            # local maximum of the cell score along the chromosome (+-3 bins) among candidates
            spm = np.where(ok, sp, -9.0)
            mx = maximum_filter(spm, size=7)
            sel = np.nonzero(ok & (sp >= mx) & (sp >= 0.3))[0]
            stripes.append(pd.DataFrame({"chrom": chrom, "a": sel, "dir": dname, "s_cell": sp[sel], "s_cons": sh[sel]}))
        log(f"[hic] {cell} {chrom}")
    pd.concat(loops, ignore_index=True).to_csv(dest_l, index=False)
    pd.concat(stripes, ignore_index=True).to_csv(dest_s, index=False)
    np.savez_compressed(WORK / f"valid_{cell}.npz", **valid)
    return cell


def stage_hic(workers: int) -> None:
    with ProcessPoolExecutor(workers) as ex:
        for c in ex.map(hic_cell, CELLS):
            log(f"[hic] {c} done")


# ================================================================================ objects
_CTX: dict = {}


def motif_gene():
    if "m" not in _CTX:
        sz = sizes()
        _CTX["m"] = alc.load_motif_bins(sz)
        _CTX["g"] = alc.load_genes(sz)
    return _CTX["m"], _CTX["g"]


def bin_mean(arr: dict, chrom: str, lo, hi):
    a = arr[chrom]
    return np.array([np.nanmean(a[l:h]) if np.isfinite(a[l:h]).any() else np.nan for l, h in zip(lo, hi)])


def loop_attrs(cell_tr: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Matching variables of loops: bin distance, orientation class, mean cell H3K27ac and mean
    consensus CTCF / H3K27ac of the two anchors."""
    mo, _ = motif_gene()
    n = len(df)
    cls = np.empty(n, object)
    ac, hc, hk = (np.full(n, np.nan) for _ in range(3))
    for c, g in df.groupby("chrom"):
        ix = np.where(df.chrom.values == c)[0]
        i, j = g.i.values, g.j.values
        m = mo[c]
        for k, a, b, x, y in zip(ix, m["strand"][i], m["strand"][j], m["score"][i], m["score"][j]):
            cls[k] = alc.orient_class(a, b, x, y)
        if cell_tr["own"].get("H3K27ac") is not None:
            t = cell_tr["own"]["H3K27ac"][c]
            ac[ix] = 0.5 * (t[i] + t[j])
        hc[ix] = 0.5 * (cell_tr["ref"]["CTCF"][c][i] + cell_tr["ref"]["CTCF"][c][j])
        hk[ix] = 0.5 * (cell_tr["ref"]["H3K27ac"][c][i] + cell_tr["ref"]["H3K27ac"][c][j])
    return pd.DataFrame({"d": (df.j - df.i).values, "cls": cls, "ac": ac, "hc": hc, "hk": hk})


def stripe_attrs(cell_tr: dict, df: pd.DataFrame) -> pd.DataFrame:
    mo, _ = motif_gene()
    n = len(df)
    cls = np.empty(n, object)
    ac, hc, hk = (np.full(n, np.nan) for _ in range(3))
    for c, g in df.groupby("chrom"):
        ix = np.where(df.chrom.values == c)[0]
        m = mo[c]
        for k, a, d in zip(ix, g.a.values, g["dir"].values):
            sc = m["score"][a:a + RA]
            kb = int(np.argmax(sc))
            dirn = 1 if d == "R" else -1
            o = "none" if sc[kb] <= 0 else ("conv" if m["strand"][a + kb] == dirn else "div")
            cls[k] = f"{d}_{o}"
        a0 = g.a.values
        if cell_tr["own"].get("H3K27ac") is not None:
            ac[ix] = bin_mean(cell_tr["own"]["H3K27ac"], c, a0, a0 + RA)
        hc[ix] = bin_mean(cell_tr["ref"]["CTCF"], c, a0, a0 + RA)
        hk[ix] = bin_mean(cell_tr["ref"]["H3K27ac"], c, a0, a0 + RA)
    return pd.DataFrame({"d": np.zeros(n), "cls": cls, "ac": ac, "hc": hc, "hk": hk})


def cell_tracks(cell: str) -> dict:
    own = {m: (load_arr(WORK / f"qn_{cell}_{m}.npz") if have(cell, m) else None) for m in MARKS}
    ref = {m: load_arr(WORK / f"ref_{cell}_{m}.npz") for m in MARKS}
    return {"own": own, "ref": ref}


def objects_cell(cell: str) -> tuple:
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(SEED + CELLS.index(cell))
    tr = cell_tracks(cell)
    valid = np.load(WORK / f"valid_{cell}.npz")
    rows, qual = [], []
    # ---------------------------------------------------------------- loops
    L = pd.read_csv(WORK / f"loops_{cell}.csv.gz")
    L["gain"] = L.e_cell - L.e_cons
    L = L[[bool(valid[c][i] and valid[c][j]) for c, i, j in zip(L.chrom, L.i, L.j)]].reset_index(drop=True)
    pos = L[(L.e_cell >= E_SPEC) & (L.gain >= G_LOOP)].reset_index(drop=True)
    pos_all_l = pos
    if len(pos) > CAP_POS:
        pos = pos.sample(CAP_POS, random_state=SEED).sort_values(["chrom", "i", "j"]).reset_index(drop=True)
    pool = L[(L.e_cell >= E_MIN) & (L.e_cons >= E_MIN)]
    keep = np.ones(len(pool), bool)
    for c, g in pos.groupby("chrom"):
        ix = np.where(pool.chrom.values == c)[0]
        if len(ix):
            dd, _ = cKDTree(g[["i", "j"]].values).query(pool.iloc[ix][["i", "j"]].values, p=np.inf)
            keep[ix[dd <= 2]] = False
    pool = pool[keep].reset_index(drop=True)
    if len(pos) and len(pool):
        pa, qa = loop_attrs(tr, pos), loop_attrs(tr, pool)
        for r in pos.itertuples():
            rows.append(dict(cell=cell, kind="loop", set="pos", pair_id=f"{cell}:L{r.Index}", y=1, chrom=r.chrom,
                             a_lo=r.i, a_hi=r.i + 1, b_lo=r.j, b_hi=r.j + 1, e_cell=r.e_cell, e_cons=r.e_cons, gain=r.gain))
        for name, covs in (("A_full", ["ac"]), ("A_ref", ["hc", "hk"]), ("A_do", [])):
            if name == "A_full" and tr["own"]["H3K27ac"] is None:
                continue
            m = alc.match_greedy(pa, qa, covs, rng)
            ok = m >= 0
            for k in np.nonzero(ok)[0]:
                q = pool.iloc[m[k]]
                rows.append(dict(cell=cell, kind="loop", set=name, pair_id=f"{cell}:L{k}", y=0, chrom=q.chrom,
                                 a_lo=q.i, a_hi=q.i + 1, b_lo=q.j, b_hi=q.j + 1, e_cell=q.e_cell, e_cons=q.e_cons, gain=q.gain))
            qual.append(dict(cell=cell, kind="loop", set=name, n_pos=len(pos), n_matched=int(ok.sum()), pool=len(pool),
                             smd_logdist=alc.smd(np.log(pa.d[ok]), np.log(qa.d.values[m[ok]])),
                             smd_ac=alc.smd(pa.ac[ok], qa.ac.values[m[ok]]),
                             smd_ref_ctcf=alc.smd(pa.hc[ok], qa.hc.values[m[ok]]),
                             smd_ref_ac=alc.smd(pa.hk[ok], qa.hk.values[m[ok]])))
    # ---------------------------------------------------------------- stripes
    S = pd.read_csv(WORK / f"stripes_{cell}.csv.gz")
    S["gain"] = S.s_cell - S.s_cons
    spos = S[(S.s_cell >= S_MIN) & (S.gain >= G_STRIPE)].reset_index(drop=True)
    spos_all = spos
    if len(spos) > CAP_POS:
        spos = spos.sample(CAP_POS, random_state=SEED).sort_values(["chrom", "dir", "a"]).reset_index(drop=True)
    spool = S[(S.s_cell >= S_MIN) & (S.s_cons >= S_MIN)]
    near = np.zeros(len(spool), bool)
    for (c, d), g in spos.groupby(["chrom", "dir"]):
        ix = np.where((spool.chrom.values == c) & (spool["dir"].values == d))[0]
        if len(ix):
            dd, _ = cKDTree(g[["a"]].values.astype(float)).query(spool.iloc[ix][["a"]].values.astype(float))
            near[ix[dd <= 8]] = True
    spool = spool[~near].reset_index(drop=True)
    if len(spos) and len(spool):
        pa, qa = stripe_attrs(tr, spos), stripe_attrs(tr, spool)
        geo = lambda a, d: ((a + OFF, a + OFF + EB) if d == "R" else (a - OFF - EB, a - OFF))  # noqa: E731
        for r in spos.itertuples():
            b0, b1 = geo(r.a, r.dir)
            rows.append(dict(cell=cell, kind="stripe", set="pos", pair_id=f"{cell}:S{r.Index}", y=1, chrom=r.chrom,
                             a_lo=r.a, a_hi=r.a + RA, b_lo=b0, b_hi=b1, e_cell=r.s_cell, e_cons=r.s_cons, gain=r.gain))
        for name, covs in (("A_full", ["ac"]), ("A_ref", ["hc", "hk"]), ("A_do", [])):
            if name == "A_full" and tr["own"]["H3K27ac"] is None:
                continue
            m = alc.match_greedy(pa, qa, covs, rng)
            ok = m >= 0
            for k in np.nonzero(ok)[0]:
                q = spool.iloc[m[k]]
                b0, b1 = geo(q.a, q.dir)
                rows.append(dict(cell=cell, kind="stripe", set=name, pair_id=f"{cell}:S{k}", y=0, chrom=q.chrom,
                                 a_lo=q.a, a_hi=q.a + RA, b_lo=b0, b_hi=b1, e_cell=q.s_cell, e_cons=q.s_cons, gain=q.gain))
            qual.append(dict(cell=cell, kind="stripe", set=name, n_pos=len(spos), n_matched=int(ok.sum()), pool=len(spool),
                             smd_logdist=0.0, smd_ac=alc.smd(pa.ac[ok], qa.ac.values[m[ok]]),
                             smd_ref_ctcf=alc.smd(pa.hc[ok], qa.hc.values[m[ok]]),
                             smd_ref_ac=alc.smd(pa.hk[ok], qa.hk.values[m[ok]])))
    geo = lambda a, d: ((a + OFF, a + OFF + EB) if d == "R" else (a - OFF - EB, a - OFF))  # noqa: E731
    for r in pos_all_l.itertuples():
        rows.append(dict(cell=cell, kind="loop", set="pos_all", pair_id=f"{cell}:LA{r.Index}", y=1, chrom=r.chrom,
                         a_lo=r.i, a_hi=r.i + 1, b_lo=r.j, b_hi=r.j + 1, e_cell=r.e_cell, e_cons=r.e_cons, gain=r.gain))
    for r in spos_all.itertuples():
        b0, b1 = geo(r.a, r.dir)
        rows.append(dict(cell=cell, kind="stripe", set="pos_all", pair_id=f"{cell}:SA{r.Index}", y=1, chrom=r.chrom,
                         a_lo=r.a, a_hi=r.a + RA, b_lo=b0, b_hi=b1, e_cell=r.s_cell, e_cons=r.s_cons, gain=r.gain))
    return cell, rows, qual


def stage_objects(args) -> None:
    rows, qual = [], []
    with ProcessPoolExecutor(min(args.workers, 6)) as ex:
        for cell, r, q in ex.map(objects_cell, CELLS):
            rows += r
            qual += q
            log(f"[objects] {cell}: {len(r)} rows")
    O = pd.DataFrame(rows)
    O.to_csv(WORK / "cell_objects.csv.gz", index=False)
    pd.DataFrame(qual).to_csv(OUT / "matching_quality_cells.csv", index=False)
    log(f"[objects] {O.shape}")


# ================================================================================ features
class Prefix:
    """Prefix sums of a per-chromosome array: mean over bins [lo, hi) in O(1)."""

    def __init__(self, arr: dict):
        self.S, self.C = {}, {}
        for c, a in arr.items():
            f = np.isfinite(a)
            self.S[c] = np.r_[0.0, np.cumsum(np.where(f, a, 0.0), dtype=np.float64)]
            self.C[c] = np.r_[0, np.cumsum(f)]
        self.n = {c: len(a) for c, a in arr.items()}

    def mean(self, c: str, lo, hi, s: int = 0) -> np.ndarray:
        n = self.n[c]
        l, h = np.clip(lo - s, 0, n), np.clip(hi + s, 0, n)
        cnt = self.C[c][h] - self.C[c][l]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, (self.S[c][h] - self.S[c][l]) / np.maximum(cnt, 1), np.nan)

    def box_stats(self, s: int) -> tuple[float, float]:
        """Median and robust SD (1.4826 MAD) of the box mean of half-width s over the genome."""
        v = []
        for c in self.n:
            i = np.arange(self.n[c])
            v.append(self.mean(c, i, i + 1, s)[::3])
        v = np.concatenate(v)
        v = v[np.isfinite(v)]
        med = float(np.median(v))
        sd = float(1.4826 * np.median(np.abs(v - med)))
        return med, (sd if sd > 1e-6 else float(v.std() + 1e-9))


BLOCK_S = EB // 2                         # half-width of the extent block for standardisation


class TrackSet:
    """Own, reference and delta tracks of one sample (cell or AML donor) with their per-sample
    standardisation (median and robust SD of the delta box means)."""

    def __init__(self, own: dict, ref: dict, meth_own=None, meth_ref=None):
        self.marks = [m for m in MARKS if own.get(m) is not None]
        self.own = {m: Prefix(own[m]) for m in self.marks}
        self.ref = {m: Prefix(ref[m]) for m in self.marks}
        self.delta = {m: Prefix({c: own[m][c] - ref[m][c] for c in CHROMS}) for m in self.marks}
        self.std = {(m, s): self.delta[m].box_stats(s) for m in self.marks for s in SCALES + [BLOCK_S]}
        self.meth = None
        if meth_own is not None and meth_ref is not None:
            self.meth_own, self.meth_ref = Prefix(meth_own), Prefix(meth_ref)
            self.meth_delta = Prefix({c: meth_own[c] - meth_ref[c] for c in CHROMS})
            self.meth_std = self.meth_delta.box_stats(10)
            self.meth = True

    def stats_table(self, name: str) -> list:
        rows = [dict(sample=name, mark=m, scale=SCALE_NAME.get(s, f"block{s}"), median=v[0], sd=v[1])
                for (m, s), v in self.std.items()]
        if self.meth:
            rows.append(dict(sample=name, mark="Meth", scale="100kb", median=self.meth_std[0], sd=self.meth_std[1]))
        return rows


def q70() -> dict:
    tq = target_q()
    k = np.searchsorted(PROBS, 0.70)
    return {m: float(tq[m][k]) for m in ("H3K27ac", "H3K27me3")}


def features_for(O: pd.DataFrame, ts: TrackSet) -> pd.DataFrame:
    """Cell-agnostic feature matrix of the objects `O` of one sample. Loops: regions A and B are the two
    anchor bins, each widened by the scale; stripes: A is the anchor (widened by the scale), B the extent
    block (not widened). Every region quantity enters as min and max over A and B: o_ own, r_ reference
    (consensus level), d_ own minus reference, z_ d_ standardised per sample (median / robust SD of the
    delta box means of the sample, genome-wide)."""
    mo, ge = motif_gene()
    qt = q70()
    O = O.reset_index(drop=True)
    n = len(O)
    F: dict = {}
    nan = lambda: np.full(n, np.nan)  # noqa: E731
    stripe = (O.kind == "stripe").values
    typ = np.array(["other"] * n, dtype=object)
    for c, g in O.groupby("chrom"):
        ix = g.index.values
        alo, ahi, blo, bhi = (g[k].values.astype(int) for k in ("a_lo", "a_hi", "b_lo", "b_hi"))
        st = stripe[ix]
        for m in ts.marks:
            for s in SCALES:
                for tag, P in (("o", ts.own[m]), ("r", ts.ref[m]), ("d", ts.delta[m])):
                    a = P.mean(c, alo, ahi, s)
                    b = np.where(st, P.mean(c, blo, bhi, 0), P.mean(c, blo, bhi, s))
                    nm = f"{tag}_{m}_{SCALE_NAME[s]}"
                    F.setdefault(nm + "_min", nan())[ix] = np.fmin(a, b)
                    F.setdefault(nm + "_max", nan())[ix] = np.fmax(a, b)
                    if tag == "d":
                        med_a, sd_a = ts.std[(m, s)]
                        med_b = np.where(st, ts.std[(m, BLOCK_S)][0], med_a)
                        sd_b = np.where(st, ts.std[(m, BLOCK_S)][1], sd_a)
                        za, zb = (a - med_a) / sd_a, (b - med_b) / sd_b
                        F.setdefault(f"z_{m}_{SCALE_NAME[s]}_min", nan())[ix] = np.fmin(za, zb)
                        F.setdefault(f"z_{m}_{SCALE_NAME[s]}_max", nan())[ix] = np.fmax(za, zb)
                        F.setdefault(f"z_{m}_{SCALE_NAME[s]}_anc", nan())[ix] = za
        if ts.meth:
            for tag, P in (("o", ts.meth_own), ("r", ts.meth_ref), ("d", ts.meth_delta)):
                a = P.mean(c, alo, ahi, 10)
                b = np.where(st, P.mean(c, blo, bhi, 0), P.mean(c, blo, bhi, 10))
                F.setdefault(f"{tag}_Meth_100kb_min", nan())[ix] = np.fmin(a, b)
                F.setdefault(f"{tag}_Meth_100kb_max", nan())[ix] = np.fmax(a, b)
                if tag == "d":
                    za, zb = (a - ts.meth_std[0]) / ts.meth_std[1], (b - ts.meth_std[0]) / ts.meth_std[1]
                    F.setdefault("z_Meth_100kb_min", nan())[ix] = np.fmin(za, zb)
                    F.setdefault("z_Meth_100kb_max", nan())[ix] = np.fmax(za, zb)
        msc, mst = mo[c]["score"], mo[c]["strand"]
        tss = ge[c]["tss"]
        own_ac = ts.own.get("H3K27ac")
        ac_a = own_ac.mean(c, alo, ahi, 0) if own_ac is not None else np.full(len(ix), np.nan)
        ac_b = own_ac.mean(c, blo, bhi, 0) if own_ac is not None else np.full(len(ix), np.nan)
        me_a = ts.own["H3K27me3"].mean(c, alo, ahi, 0)
        me_b = ts.own["H3K27me3"].mean(c, blo, bhi, 0)
        for i, k in enumerate(ix):
            a0, a1, b0, b1 = alo[i], ahi[i], blo[i], bhi[i]
            sa = msc[a0:a1]
            ka = int(np.argmax(sa))
            scA = float(sa[ka])
            stA = float(mst[a0 + ka]) if scA > 0 else 0.0
            gA = float(tss[a0:a1].sum() > 0)
            if not st[i]:
                sb = msc[b0:b1]
                kb = int(np.argmax(sb))
                scB = float(sb[kb])
                stB = float(mst[b0 + kb]) if scB > 0 else 0.0
                oc = alc.orient_class(stA, stB, scA, scB)
                gB = float(tss[b0:b1].sum() > 0)
                dist = (b0 - a0) * RES
                if np.isfinite(ac_a[i]):
                    for prom, pac, pme in ((gA, ac_b[i], me_b[i]), (gB, ac_a[i], me_a[i])):
                        if prom:
                            if pme >= qt["H3K27me3"]:
                                typ[k] = "PS"
                                break
                            if pac >= qt["H3K27ac"]:
                                typ[k] = "PE"
            else:
                scB, gB = scA, gA
                dirn = 1 if b0 > a0 else -1
                oc = "none" if scA <= 0 else ("conv" if stA == dirn else "div")
                dist = abs(b0 - a0) * RES
                if np.isfinite(ac_a[i]):
                    typ[k] = "rep" if me_a[i] >= qt["H3K27me3"] else ("enh" if ac_a[i] >= qt["H3K27ac"] else "other")
            F.setdefault("motif_score_min", nan())[k] = min(scA, scB)
            F.setdefault("motif_score_max", nan())[k] = max(scA, scB)
            F.setdefault("gene_tss_min", nan())[k] = min(gA, gB)
            F.setdefault("gene_tss_max", nan())[k] = max(gA, gB)
            F.setdefault("gene_nprom", nan())[k] = gA + (0.0 if st[i] else gB)
            for o_ in alc.ORIENT:
                F.setdefault(f"motif_orient_{o_}", nan())[k] = float(oc == o_)
            F.setdefault("dist_log10", nan())[k] = np.log10(max(dist, 1))
    F["pair_type"] = typ
    return pd.concat([O, pd.DataFrame(F)], axis=1)


def cell_trackset(cell: str) -> TrackSet:
    own = {m: (load_arr(WORK / f"qn_{cell}_{m}.npz") if have(cell, m) else None) for m in MARKS}
    ref = {m: load_arr(WORK / f"ref_{cell}_{m}.npz") for m in MARKS}
    mo = mr = None
    if meth_ok(cell):
        mo = load_arr(WORK / f"raw_{cell}_Meth.npz")
        ms = [x for x in TRAIN if x != cell and meth_ok(x)]
        mr = {c: np.mean([load_arr(WORK / f"raw_{x}_Meth.npz")[c] for x in ms], axis=0) for c in CHROMS}
    return TrackSet(own, ref, mo, mr)


def aml_trackset(p: str, ref: str = "hspc") -> TrackSet:
    own = {m: (load_arr(WORK / f"qn_aml{p}_{m}.npz") if (WORK / f"qn_aml{p}_{m}.npz").exists() else None) for m in MARKS}
    if ref == "hspc":
        rf = {m: load_arr(WORK / f"qn_amlhspc_{m}.npz") for m in MARKS}
    else:
        rf = {}
        for m in MARKS:
            oth = [q for q in AML_DONORS if q != p and (WORK / f"qn_aml{q}_{m}.npz").exists()]
            arrs = [load_arr(WORK / f"qn_aml{q}_{m}.npz") for q in oth]
            rf[m] = {c: np.mean([a[c] for a in arrs], axis=0).astype(np.float32) for c in CHROMS}
    mo = aml_meth10(p) if p in AML_CTCF else None
    h = [aml_meth10(q) for q in ("103", "673")]
    mr = {c: np.nanmean([x[c] for x in h], axis=0) for c in CHROMS} if mo is not None else None
    return TrackSet({m: v for m, v in own.items() if v is not None} | {m: None for m, v in own.items() if v is None},
                    rf, mo, mr)


def aml_objects() -> pd.DataFrame:
    """The AML objects of section 62 (10 test donors): positives and matched shared controls, with
    gain over the HSPC consensus and recurrence class."""
    F = pd.read_csv(CLF, usecols=["patient", "kind", "set", "pair_id", "y", "chrom", "a_lo", "a_hi", "b_lo", "b_hi"],
                    dtype={"patient": str})
    G = pd.read_csv(DET / "object_scores.csv.gz", dtype={"patient": str},
                    usecols=["patient", "kind", "set", "pair_id", "gain|measured", "measured", "hspc", "cls", "fold"])
    O = F.merge(G, on=["patient", "kind", "set", "pair_id"], how="inner")
    O = O[O.set.isin(["pos", "A_full", "A_hspc"])].copy()
    O["set"] = O.set.replace({"A_hspc": "A_ref"})
    O = O.rename(columns={"gain|measured": "gain", "measured": "e_cell", "hspc": "e_cons"})
    O["cell"] = "aml" + O.patient
    return O.reset_index(drop=True)


def stage_features(args) -> None:
    O = pd.read_csv(WORK / "cell_objects.csv.gz")
    O = O[O.set.isin(["pos", "A_full", "A_ref"])]
    stats, parts = [], []
    for cell in CELLS:
        o = O[O.cell == cell]
        if o.empty:
            continue
        ts = cell_trackset(cell)
        stats += ts.stats_table(cell)
        f = features_for(o, ts)
        f["domain"], f["ref"] = "cell", "consensus"
        f["cancer"] = int(cell in CANCER)
        parts.append(f)
        log(f"[features] {cell}: {len(f)} rows, meth={bool(ts.meth)}, marks={ts.marks}")
    pd.concat(parts, ignore_index=True).to_csv(WORK / "cell_features.csv.gz", index=False)
    A = aml_objects()
    A.to_csv(WORK / "aml_objects.csv.gz", index=False)
    for ref in ("hspc", "otheraml"):
        parts = []
        for p in AML_DONORS:
            o = A[A.patient == p]
            ts = aml_trackset(p, ref)
            if ref == "hspc":
                stats += ts.stats_table("aml" + p)
            f = features_for(o, ts)
            f["domain"], f["ref"], f["cancer"] = "aml", ref, 1
            parts.append(f)
            log(f"[features] aml{p} ref={ref}: {len(f)} rows")
        pd.concat(parts, ignore_index=True).to_csv(WORK / f"aml_features_{ref}.csv.gz", index=False)
    pd.DataFrame(stats).to_csv(OUT / "delta_standardisation.csv", index=False)


# ================================================================================ rules (within group)
KEY_RULES = [
    # name, kind, class, feature, metric
    ("stripe all: dH3K27ac +-100kb (min A,B) vs gain", "stripe", "all", "z_H3K27ac_100kb_min", "rho"),
    ("stripe enh: dH3K27ac +-100kb (min A,B) vs gain", "stripe", "enh", "z_H3K27ac_100kb_min", "rho"),
    ("stripe enh: dH3K27ac +-100kb (min A,B) vs shared", "stripe", "enh", "z_H3K27ac_100kb_min", "auroc"),
    ("stripe all: dH3K27ac +-100kb (min A,B) vs shared", "stripe", "all", "z_H3K27ac_100kb_min", "auroc"),
    ("stripe rep: dH3K27me3 anchor vs shared", "stripe", "rep", "z_H3K27me3_10kb_anc", "auroc"),
    ("stripe rep: dH3K27me3 anchor vs gain", "stripe", "rep", "z_H3K27me3_10kb_anc", "rho"),
    ("loop PE: dH3K27ac max(A,B) vs shared", "loop", "PE", "z_H3K27ac_10kb_max", "auroc"),
    ("loop PE: dH3K27ac min(A,B) vs shared", "loop", "PE", "z_H3K27ac_10kb_min", "auroc"),
    ("loop PE: dH3K27ac min(A,B) vs gain", "loop", "PE", "z_H3K27ac_10kb_min", "rho"),
    ("loop PS: dH3K27me3 max(A,B) vs shared", "loop", "PS", "z_H3K27me3_10kb_max", "auroc"),
    ("loop PS: dH3K27me3 max(A,B) vs gain", "loop", "PS", "z_H3K27me3_10kb_max", "rho"),
    ("loop all: dH3K27ac +-100kb (min) vs gain", "loop", "all", "z_H3K27ac_100kb_min", "rho"),
    ("loop all: dH3K27ac max(A,B) vs shared", "loop", "all", "z_H3K27ac_10kb_max", "auroc"),
    ("loop all: dCTCF max(A,B) vs shared", "loop", "all", "z_CTCF_10kb_max", "auroc"),
    ("stripe all: dCTCF anchor vs shared", "stripe", "all", "z_CTCF_10kb_anc", "auroc"),
    ("loop all: dMeth +-100kb (min) vs shared", "loop", "all", "z_Meth_100kb_min", "auroc"),
    ("loop all: dMeth +-100kb (min) vs gain", "loop", "all", "z_Meth_100kb_min", "rho"),
    ("stripe all: dMeth +-100kb (min) vs shared", "stripe", "all", "z_Meth_100kb_min", "auroc"),
    ("loop all: consensus H3K27ac +-100kb (max) vs gain", "loop", "all", "r_H3K27ac_100kb_max", "rho"),
    ("stripe all: consensus H3K27ac +-100kb (min) vs gain", "stripe", "all", "r_H3K27ac_100kb_min", "rho"),
]
CLASSES = {"loop": ["all", "PE", "PS", "other"], "stripe": ["all", "enh", "rep", "other"]}


def load_all(ref: str = "hspc") -> pd.DataFrame:
    """Cells and AML donors in one frame; controls inherit the pair type and the recurrence class of the
    positive they are matched to."""
    C = pd.read_csv(WORK / "cell_features.csv.gz")
    A = pd.read_csv(WORK / f"aml_features_{ref}.csv.gz", dtype={"patient": str})
    C["group"], A["group"] = C.cell, A.cell
    C["cls"] = "cell"
    D = pd.concat([C, A], ignore_index=True)
    key = ["group", "kind", "pair_id"]
    P = D[D.set == "pos"].drop_duplicates(key).set_index(key)
    idx = pd.MultiIndex.from_frame(D[key])
    D["type_own"] = D.pair_type
    D["type_inh"] = P.pair_type.reindex(idx).values
    V = pd.read_csv(CLF, usecols=["patient", "kind", "set", "pair_id", "pair_type"], dtype={"patient": str})
    V["set"] = V.set.replace({"A_hspc": "A_ref"})
    V["group"] = "aml" + V.patient
    V = V.drop_duplicates(["group", "kind", "set", "pair_id"]).rename(columns={"pair_type": "type_v2"})
    D = D.merge(V[["group", "kind", "set", "pair_id", "type_v2"]], on=["group", "kind", "set", "pair_id"], how="left")
    D["cls_pos"] = P.cls.reindex(idx).values
    D["mate_gain"] = P.gain.reindex(idx).values
    return D


def auroc_fast(x_pos: np.ndarray, x_ctl: np.ndarray) -> float:
    from scipy.stats import rankdata
    x_pos, x_ctl = x_pos[np.isfinite(x_pos)], x_ctl[np.isfinite(x_ctl)]
    if len(x_pos) < 15 or len(x_ctl) < 15:
        return np.nan
    r = rankdata(np.r_[x_pos, x_ctl])
    return float((r[:len(x_pos)].sum() - len(x_pos) * (len(x_pos) + 1) / 2) / (len(x_pos) * len(x_ctl)))


def spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or np.std(x[m]) == 0:
        return np.nan
    return float(spearmanr(x[m], y[m])[0])


def rule_feature_list(D: pd.DataFrame) -> list:
    return [c for c in D.columns if c[:2] in ("z_", "r_", "o_")]


def paired(d: pd.DataFrame, ctl_set: str, cl: str, tcol: str = "type_own"):
    """Specific objects of class `cl` (own-mark type), the subset of them that has a matched control, and the
    controls of the same own-mark type (registry section 64: classes are defined from the sample's own marks
    in positives and controls alike)."""
    pos, ctl_all = d[d.set == "pos"], d[d.set == ctl_set]
    ctl = ctl_all
    if cl != "all":
        pos, ctl = pos[pos[tcol] == cl], ctl_all[ctl_all[tcol] == cl]
    pm = pos[pos.pair_id.isin(ctl_all.pair_id)]
    return pos, pm, ctl


def eval_group(Dg: pd.DataFrame, feats: list, ctl_set: str, tcol: str = "type_own") -> list:
    """Per feature and class: Spearman with the gain over all specific objects of the class and AUROC of the
    specific objects against their matched controls (pairs with a match only)."""
    rows = []
    for kind in ("loop", "stripe"):
        d = Dg[Dg.kind == kind]
        for cl in CLASSES[kind]:
            pos, pm, cm = paired(d, ctl_set, cl, tcol)
            if len(pos) < 15:
                continue
            for f in feats:
                if f not in d.columns or not np.isfinite(pos[f].values).any():
                    continue
                rows.append(dict(kind=kind, cls=cl, feature=f, n_pos=len(pos), n_pairs=len(pm), n_ctl=len(cm),
                                 rho=spearman_fast(pos[f].values, pos.gain.values),
                                 auroc=auroc_fast(pm[f].values, cm[f].values)))
    return rows


def boot_rule(Dg: pd.DataFrame, rule: tuple, ctl_set: str, nb: int, seed: int = 0, tcol: str = "type_own") -> tuple:
    """Point estimate and bootstrap SE of one key rule in one group (loci resampled; a control moves with its
    matched positive)."""
    _, kind, cl, f, metric = rule
    d = Dg[Dg.kind == kind]
    if f not in d.columns:
        return np.nan, np.nan
    pos, pm, cm = paired(d, ctl_set, cl, tcol)
    if len(pos) < 15 or len(cm) < 15 or not np.isfinite(pos[f].values).any():
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    if metric == "rho":
        x, y = pos[f].values, pos.gain.values
        est = spearman_fast(x, y)
        vals = [spearman_fast(x[ii], y[ii]) for ii in (rng.integers(0, len(x), len(x)) for _ in range(nb))]
    else:
        x, y = pm[f].values, cm[f].values
        est = auroc_fast(x, y)
        vals = [auroc_fast(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))]) for _ in range(nb)]
    return est, float(np.nanstd(vals))


def boot_ci(v: np.ndarray, nb: int = 4000, seed: int = 0) -> tuple:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v)).mean() for _ in range(nb)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stage_rules(args) -> None:
    allrows, keyrows = [], []
    for ref in ("hspc", "otheraml"):
        D = load_all(ref)
        if ref == "otheraml":
            D = D[D.domain == "aml"]
        feats = rule_feature_list(D)
        groups = [(g, Dg) for g, Dg in D.groupby("group")]
        if ref == "hspc":
            groups.append(("AML_pooled", D[D.domain == "aml"]))
        for g, Dg in groups:
            dom = "amlpool" if g == "AML_pooled" else Dg.domain.iloc[0]
            for tmode, tcol in (("own", "type_own"), ("inherit", "type_inh"), ("v2inherit", "type_v2")):
                if tmode == "v2inherit" and dom == "cell":
                    continue
                for ctl in ("A_ref", "A_full"):
                    if g != "AML_pooled":
                        for r in eval_group(Dg, feats, ctl, tcol):
                            allrows.append(dict(group=g, domain=dom, ref=ref, cancer=int(Dg.cancer.iloc[0]), controls=ctl,
                                                types=tmode, **r))
                    for rule in KEY_RULES:
                        est, se = boot_rule(Dg, rule, ctl, 200, tcol=tcol)
                        keyrows.append(dict(group=g, domain=dom, ref=ref, cancer=int(Dg.cancer.iloc[0]), controls=ctl,
                                            types=tmode, rule=rule[0], kind=rule[1], cls=rule[2], feature=rule[3],
                                            metric=rule[4], est=est, se=se))
        log(f"[rules] ref={ref} done")
    pd.DataFrame(allrows).to_csv(OUT / "rules_all_features.csv.gz", index=False)
    K = pd.DataFrame(keyrows)
    K["effect"] = np.where(K.metric == "auroc", K.est - 0.5, K.est)
    K.to_csv(OUT / "rules_key.csv", index=False)
    for tmode in ("own", "inherit", "v2inherit"):
        for ctl in ("A_ref", "A_full"):
            k = K[(K.controls == ctl) & (K.types == tmode)]
            summarise_rules(k[k.domain != "amlpool"], OUT / f"rules_consistency_{ctl}_{tmode}.csv", k[k.domain == "amlpool"])


def summarise_rules(K: pd.DataFrame, dest: Path, pooled: pd.DataFrame) -> None:
    from scipy.stats import mannwhitneyu
    rows = []
    for rule, g in K.groupby("rule", sort=False):
        row = dict(rule=rule, kind=g.kind.iloc[0], cls=g.cls.iloc[0], metric=g.metric.iloc[0], feature=g.feature.iloc[0])
        sets = {"cells_all": g[(g.domain == "cell")], "cells_normal": g[(g.domain == "cell") & (g.cancer == 0)],
                "cells_cancer": g[(g.domain == "cell") & (g.cancer == 1)],
                "aml_hspc": g[(g.domain == "aml") & (g.ref == "hspc")],
                "aml_otheraml": g[(g.domain == "aml") & (g.ref == "otheraml")]}
        for nm, s in sets.items():
            e = s.est.values
            m, lo, hi = boot_ci(e)
            row.update({f"{nm}_n": int(np.isfinite(e).sum()), f"{nm}_mean": m, f"{nm}_lo": lo, f"{nm}_hi": hi,
                        f"{nm}_sd": float(np.nanstd(e)) if np.isfinite(e).sum() > 1 else np.nan})
            ref0 = 0.5 if row["metric"] == "auroc" else 0.0
            row[f"{nm}_frac_pos"] = float(np.mean(e[np.isfinite(e)] > ref0)) if np.isfinite(e).any() else np.nan
        # heterogeneity of the cells: tau2 = var(est) - mean(se^2), I2
        s = sets["cells_all"]
        ok = np.isfinite(s.est.values) & np.isfinite(s.se.values)
        if ok.sum() > 3:
            v, se2 = np.var(s.est.values[ok], ddof=1), np.mean(s.se.values[ok] ** 2)
            row["cells_I2"] = float(max(0.0, (v - se2) / v)) if v > 0 else np.nan
        a, b = sets["cells_normal"].est.dropna().values, sets["cells_cancer"].est.dropna().values
        if len(a) > 2 and len(b) > 2:
            rng = np.random.default_rng(1)
            dd = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(4000)]
            row.update(cancer_minus_normal=float(b.mean() - a.mean()), cmn_lo=float(np.percentile(dd, 2.5)),
                       cmn_hi=float(np.percentile(dd, 97.5)), cmn_p_mwu=float(mannwhitneyu(a, b).pvalue))
        a2, b2 = sets["cells_all"].est.dropna().values, sets["aml_hspc"].est.dropna().values
        if len(a2) > 2 and len(b2) > 2:
            rng = np.random.default_rng(2)
            dd = [rng.choice(b2, len(b2)).mean() - rng.choice(a2, len(a2)).mean() for _ in range(4000)]
            row.update(aml_minus_cells=float(b2.mean() - a2.mean()), amc_lo=float(np.percentile(dd, 2.5)),
                       amc_hi=float(np.percentile(dd, 97.5)))
        pp = pooled[pooled.rule == rule]
        row["aml_pooled_est"] = float(pp.est.iloc[0]) if len(pp) else np.nan
        row["aml_pooled_se"] = float(pp.se.iloc[0]) if len(pp) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(dest, index=False)


# ================================================================================ transfer
FEATSETS_M = {
    "M2": ["H3K27ac", "H3K27me3"],
    "M3": ["H3K27ac", "H3K27me3", "CTCF"],
}
LOCUS = ["dist_log10", "motif_orient_conv", "motif_orient_div", "motif_orient_tandem", "motif_orient_none",
         "motif_score_min", "motif_score_max", "gene_tss_min", "gene_tss_max", "gene_nprom"]
RULE_LOCUS = ["dist_log10", "motif_orient_conv", "motif_orient_div", "motif_orient_tandem", "motif_orient_none"]


def featset(name: str, marks: str = "M2") -> list:
    ms = FEATSETS_M[marks]
    delta = [f"z_{m}_{SCALE_NAME[s]}_{k}" for m in ms for s in SCALES for k in ("min", "max")]
    ref = [f"r_{m}_{SCALE_NAME[s]}_{k}" for m in ms for s in SCALES for k in ("min", "max")]
    return {"ALL": delta + ref + LOCUS, "RULE": delta + RULE_LOCUS, "STATE": ref + LOCUS, "LOCUS": LOCUS,
            "DELTA": delta}[name]


def key_matrix(d: pd.DataFrame) -> np.ndarray:
    ks = alc.row_keys(d.assign(kind=d.kind.values))
    K = np.full((len(d), 4), -1, np.int64)
    for i, k in enumerate(ks):
        k = sorted(k)[:4]
        K[i, :len(k)] = k
    return K


def overlap_mask(Ktrain: np.ndarray, Ktest: np.ndarray, pad: int = 2) -> np.ndarray:
    """True for training rows with an anchor bin within `pad` bins of an anchor bin of a test row."""
    t = np.unique(Ktest[Ktest >= 0])
    exp = np.unique((t[:, None] + np.arange(-pad, pad + 1)[None, :]).ravel())
    return np.isin(Ktrain, exp).any(1)


class Pool:
    """All objects of one kind and control set: matched positives and controls of every group, with the
    locus keys and the within-group gain percentile."""

    def __init__(self, D: pd.DataFrame, kind: str, ctl: str):
        d = D[(D.kind == kind) & D.set.isin(["pos", ctl])].copy()
        bad = d[(d.set == "pos") & ~np.isfinite(d.gain)]
        if len(bad):
            bk = set(zip(bad.group, bad.pair_id))
            d = d[[(g, p) not in bk for g, p in zip(d.group, d.pair_id)]]
        have = d[d.set == ctl].groupby("group").pair_id.apply(set).to_dict()
        keep = (d.set == ctl) | ((d.set == "pos") & np.array([pid in have.get(g, ()) for g, pid in zip(d.group, d.pair_id)]))
        d = d[keep].reset_index(drop=True)
        d["gr"] = np.nan                                          # percentile of the gain within the group (positives)
        ip = d.index[d.set == "pos"]
        d.loc[ip, "gr"] = d.loc[ip].groupby("group").gain.rank(pct=True)
        self.d, self.kind, self.ctl = d, kind, ctl
        self.K = key_matrix(d)
        self.idx = {g: np.where(d.group.values == g)[0] for g in d.group.unique()}

    def rows(self, groups) -> np.ndarray:
        return np.concatenate([self.idx[g] for g in groups if g in self.idx]) if len(groups) else np.array([], int)


def make_reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                         l2_regularization=2.0, random_state=0)


def fit_predict(P: Pool, train_groups: list, test_group: str, feats: list, weights: dict | None = None,
                importance: bool = False) -> dict:
    d = P.d
    te = P.idx.get(test_group, np.array([], int))
    tr = P.rows(train_groups)
    tr = tr[~overlap_mask(P.K[tr], P.K[te])]
    from threadpoolctl import threadpool_limits
    X = d[feats].values.astype(float)
    y = (d.set.values == "pos").astype(int)
    w = None
    if weights:
        w = np.array([weights.get(g, 1.0) for g in d.group.values[tr]])
    with threadpool_limits(4):
        clf = alc.make_model("gb")
        clf.fit(X[tr], y[tr], sample_weight=w)
        trp = tr[y[tr] == 1]
        reg = make_reg()
        reg.fit(X[trp], d.gr.values[trp], sample_weight=None if w is None else np.array([weights.get(g, 1.0) for g in d.group.values[trp]]))
    out = {"te": te, "score": clf.predict_proba(X[te])[:, 1], "pred_gain": reg.predict(X[te]), "n_train": len(tr)}
    if importance:
        blocks = {"delta": [i for i, f in enumerate(feats) if f.startswith("z_")],
                  "reference": [i for i, f in enumerate(feats) if f.startswith("r_")],
                  "locus": [i for i, f in enumerate(feats) if not f.startswith(("z_", "r_"))]}
        base = auroc_fast(out["score"][y[te] == 1], out["score"][y[te] == 0])
        rng = np.random.default_rng(0)
        for b, cols in blocks.items():
            if not cols:
                continue
            drops = []
            for _ in range(3):
                Xp = X[te].copy()
                perm = rng.permutation(len(te))
                Xp[:, cols] = Xp[perm][:, cols]
                with threadpool_limits(4):
                    sp = clf.predict_proba(Xp)[:, 1]
                drops.append(base - auroc_fast(sp[y[te] == 1], sp[y[te] == 0]))
            out[f"imp_{b}"] = float(np.mean(drops))
    return out


def score_group(P: Pool, res: dict, nb: int = 200, seed: int = 0) -> dict:
    d = P.d
    te = res["te"]
    pos = (d.set.values[te] == "pos")
    sc, pg = res["score"], res["pred_gain"]
    r = dict(n_pos=int(pos.sum()), n_ctl=int((~pos).sum()))
    if pos.sum() < 15 or (~pos).sum() < 15:
        return r
    rng = np.random.default_rng(seed)
    r["auroc"] = auroc_fast(sc[pos], sc[~pos])
    gr = d.gr.values[te][pos]
    r["rho"] = spearman_fast(pg[pos], gr)
    a = [auroc_fast(sc[pos][rng.integers(0, pos.sum(), pos.sum())], sc[~pos][rng.integers(0, (~pos).sum(), (~pos).sum())]) for _ in range(nb)]
    b = [spearman_fast(pg[pos][ii], gr[ii]) for ii in (rng.integers(0, pos.sum(), pos.sum()) for _ in range(nb))]
    r["auroc_se"], r["rho_se"] = float(np.nanstd(a)), float(np.nanstd(b))
    ty = d.pair_type.values[te]
    for t in (("PE", "PS", "other") if P.kind == "loop" else ("enh", "rep", "other")):
        m = ty == t
        if (pos & m).sum() >= 15 and (~pos & m).sum() >= 15:
            r[f"auroc_{t}"] = auroc_fast(sc[pos & m], sc[~pos & m])
    if "cls" in d.columns and d.group.values[te[0]].startswith("aml"):
        pc = d.cls_pos.values[te]
        m = pos & (pc == "private")
        # controls are matched to a positive: keep the controls of private positives
        pid = set(d.pair_id.values[te][m])
        mc = (~pos) & np.array([x in pid for x in d.pair_id.values[te]])
        if m.sum() >= 15 and mc.sum() >= 15:
            r["auroc_private"] = auroc_fast(sc[m], sc[mc])
            r["rho_private"] = spearman_fast(pg[m], gr[np.where(pos)[0].searchsorted(np.where(m)[0])])
    return r


def stage_transfer(args) -> None:
    D = load_all("hspc")
    D = D[D.set.isin(["pos", "A_full", "A_ref"])]
    cells_all = [c for c in CELLS]
    donors = ["aml" + p for p in AML_DONORS]
    rows, imps = [], []
    t0 = time.time()
    for kind in ("loop", "stripe"):
        for ctl in ("A_full", "A_ref"):
            P = Pool(D, kind, ctl)
            have_g = set(P.idx)
            cells = [c for c in cells_all if c in have_g]
            dons = [g for g in donors if g in have_g]
            log(f"[transfer] {kind}/{ctl}: {len(cells)} cells, {len(dons)} donors, {len(P.d)} rows")
            fsets = [("ALL", "M2"), ("RULE", "M2"), ("STATE", "M2"), ("LOCUS", "M2"), ("ALL", "M3")]
            if ctl == "A_ref":
                fsets = [("ALL", "M2"), ("RULE", "M2"), ("STATE", "M2")]
            for fs, mk in fsets:
                feats = featset(fs, mk)
                if mk == "M3":
                    cells_m = [c for c in cells if c != "gm23248"]
                    dons_m = [g for g in dons if g[3:] in AML_CTCF]
                else:
                    cells_m, dons_m = cells, dons
                tag = f"{fs}-{mk}"

                def rec(exp, train_desc, test, res, **kw):
                    rows.append(dict(experiment=exp, kind=kind, controls=ctl, featset=tag, train=train_desc, test=test,
                                     test_domain="aml" if test.startswith("aml") else "cell",
                                     n_train=res["n_train"], **score_group(P, res), **kw))
                    if "imp_delta" in res:
                        imps.append(dict(experiment=exp, kind=kind, controls=ctl, featset=tag, test=test,
                                         **{k: v for k, v in res.items() if k.startswith("imp_")}))
                # (a) leave-one-cell-out, non-AML cells only
                for c in cells_m:
                    res = fit_predict(P, [x for x in cells_m if x != c], c, feats, importance=(fs == "ALL" and mk == "M2"))
                    rec("a_cells_to_cell", "other cells", c, res)
                # (b) all cells -> each AML donor
                for g in dons_m:
                    res = fit_predict(P, cells_m, g, feats)
                    rec("b_cells_to_aml", "all cells", g, res)
                # (c) all AML donors -> each cell
                for c in cells_m:
                    res = fit_predict(P, dons_m, c, feats)
                    rec("c_aml_to_cell", "all AML donors", c, res)
                if fs in ("ALL", "RULE") and mk == "M2":
                    # (d) held-out donor: AML only vs AML + cells (unweighted and 50:50 weighted)
                    n_cell_rows = len(P.rows(cells_m))
                    for g in dons_m:
                        oth = [x for x in dons_m if x != g]
                        rec("d_aml_only", "other AML donors", g, fit_predict(P, oth, g, feats))
                        rec("d_aml_plus_cells", "other AML donors + all cells", g, fit_predict(P, oth + cells_m, g, feats))
                        n_aml_rows = len(P.rows(oth))
                        wts = {x: 1.0 for x in cells_m}
                        wts.update({x: n_cell_rows / max(n_aml_rows, 1) for x in oth})
                        rec("d_aml_plus_cells_w50", "other AML donors (weight 50:50) + all cells", g,
                            fit_predict(P, oth + cells_m, g, feats, weights=wts))
                        rec("d_cells_only", "all cells", g, fit_predict(P, cells_m, g, feats))
                    # (e) held-out cell: other cells only vs other cells + AML
                    for c in cells_m:
                        oc = [x for x in cells_m if x != c]
                        rec("e_cells_plus_aml", "other cells + all AML donors", c, fit_predict(P, oc + dons_m, c, feats))
            log(f"[transfer] {kind}/{ctl} done ({time.time() - t0:.0f} s)")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "transfer_results.csv", index=False)
    pd.DataFrame(imps).to_csv(OUT / "transfer_block_importance.csv", index=False)
    summarise_transfer(R)
    stage_signs(args)


def summarise_transfer(R: pd.DataFrame) -> None:
    out = []
    for (exp, kind, ctl, fs), g in R.groupby(["experiment", "kind", "controls", "featset"]):
        for dom in ("cell", "aml"):
            h = g[g.test_domain == dom]
            if h.empty:
                continue
            row = dict(experiment=exp, kind=kind, controls=ctl, featset=fs, test_domain=dom, n_groups=len(h),
                       train=h.train.iloc[0])
            for met in ("auroc", "rho", "auroc_private", "auroc_PE", "auroc_PS", "auroc_enh", "auroc_rep"):
                if met not in h:
                    continue
                m, lo, hi = boot_ci(h[met].values)
                row.update({f"{met}_mean": m, f"{met}_lo": lo, f"{met}_hi": hi, f"{met}_n": int(h[met].notna().sum())})
            out.append(row)
    pd.DataFrame(out).to_csv(OUT / "transfer_summary.csv", index=False)
    # paired differences d: AML only vs AML + cells (donor bootstrap)
    rows = []
    for (kind, ctl, fs), g in R[R.experiment.str.startswith("d_")].groupby(["kind", "controls", "featset"]):
        base = g[g.experiment == "d_aml_only"].set_index("test")
        for exp in ("d_aml_plus_cells", "d_aml_plus_cells_w50", "d_cells_only"):
            o = g[g.experiment == exp].set_index("test")
            for met in ("auroc", "rho", "auroc_private", "rho_private"):
                if met not in o or met not in base:
                    continue
                dd = (o[met] - base[met]).dropna()
                if len(dd) < 3:
                    continue
                m, lo, hi = boot_ci(dd.values, seed=3)
                rows.append(dict(kind=kind, controls=ctl, featset=fs, comparison=f"{exp} minus d_aml_only", metric=met,
                                 n_donors=len(dd), diff=m, lo=lo, hi=hi, donors_better=int((dd > 0).sum())))
    for (kind, ctl, fs), g in R[R.experiment.isin(["a_cells_to_cell", "e_cells_plus_aml"])].groupby(["kind", "controls", "featset"]):
        base = g[g.experiment == "a_cells_to_cell"].set_index("test")
        o = g[g.experiment == "e_cells_plus_aml"].set_index("test")
        for met in ("auroc", "rho"):
            if met not in o or len(o) < 3:
                continue
            dd = (o[met] - base[met]).dropna()
            if len(dd) < 3:
                continue
            m, lo, hi = boot_ci(dd.values, seed=4)
            rows.append(dict(kind=kind, controls=ctl, featset=fs, comparison="e_cells_plus_aml minus a_cells_to_cell", metric=met,
                             n_donors=len(dd), diff=m, lo=lo, hi=hi, donors_better=int((dd > 0).sum())))
    pd.DataFrame(rows).to_csv(OUT / "transfer_paired_differences.csv", index=False)


def stage_signs(args) -> None:
    """Per group logistic regression on the standardised delta features (+ distance): coefficient sign and
    size consistency across cells and AML donors."""
    D = load_all("hspc")
    D = D[D.set.isin(["pos", "A_full"])]
    feats = [f"z_{m}_{SCALE_NAME[s]}_{k}" for m in FEATSETS_M["M2"] for s in SCALES for k in ("min", "max")] + ["dist_log10"]
    rows = []
    groups = list(CELLS) + ["aml" + p for p in AML_DONORS] + ["AML_pooled"]
    for kind in ("loop", "stripe"):
        for g in groups:
            d = D[(D.kind == kind) & ((D.domain == "aml") if g == "AML_pooled" else (D.group == g))]
            pos = d[d.set == "pos"]
            ctl = d[d.set == "A_full"]
            pos = pos[pos.pair_id.isin(ctl.pair_id)]
            if len(pos) < 60 or len(ctl) < 60:
                continue
            dd = pd.concat([pos, ctl])
            X, y = dd[feats].values.astype(float), (dd.set.values == "pos").astype(int)
            fl = feats if kind == "loop" else [f for f in feats if f != "dist_log10"]
            ix = [feats.index(f) for f in fl]
            m = alc.make_model("lr")
            m.fit(X[:, ix], y)
            coef = m[-1].coef_[0]
            for f, c in zip(fl, coef):
                rows.append(dict(kind=kind, group=g, domain="aml" if g.startswith("aml") or g == "AML_pooled" else "cell",
                                 cancer=int(g in CANCER or g.startswith("aml")), feature=f, coef=float(c), n_pos=len(pos)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "lr_coefficients.csv", index=False)
    out = []
    for (kind, f), g in S[S.group != "AML_pooled"].groupby(["kind", "feature"]):
        c, a = g[g.domain == "cell"], g[g.domain == "aml"]
        row = dict(kind=kind, feature=f, cells_mean=c.coef.mean(), cells_sd=c.coef.std(), cells_frac_pos=(c.coef > 0).mean(),
                   cells_n=len(c), aml_mean=a.coef.mean(), aml_sd=a.coef.std(), aml_frac_pos=(a.coef > 0).mean(), aml_n=len(a),
                   sign_agree_cells_aml=bool(np.sign(c.coef.mean()) == np.sign(a.coef.mean())))
        ca, no = c[c.cancer == 1].coef.values, c[c.cancer == 0].coef.values
        row.update(cancer_mean=float(ca.mean()) if len(ca) else np.nan, normal_mean=float(no.mean()) if len(no) else np.nan)
        out.append(row)
    pd.DataFrame(out).to_csv(OUT / "lr_sign_consistency.csv", index=False)


# ================================================================================ counts
def stage_counts(args) -> None:
    O = pd.read_csv(WORK / "cell_objects.csv.gz")
    rows = []
    for cell in CELLS:
        o = O[O.cell == cell]
        ts = cell_trackset(cell)
        allp = o[o.set == "pos_all"]
        f = features_for(allp, ts)
        for kind in ("loop", "stripe"):
            fk = f[f.kind == kind]
            ok = o[o.kind == kind]
            cls = ["PE", "PS", "other"] if kind == "loop" else ["enh", "rep", "other"]
            row = dict(group=cell, domain="cell", cancer=int(cell in CANCER), kind=kind, n_specific_total=len(fk),
                       **{f"total_{c}": int((fk.pair_type == c).sum()) for c in cls},
                       n_pos_used=int((ok.set == "pos").sum()), n_matched_A_full=int((ok.set == "A_full").sum()),
                       n_matched_A_ref=int((ok.set == "A_ref").sum()))
            row["n_objects_A_full"] = row["n_matched_A_full"] * 2
            rows.append(row)
    A = pd.read_csv(WORK / "aml_objects.csv.gz", dtype={"patient": str})
    Af = pd.read_csv(WORK / "aml_features_hspc.csv.gz", dtype={"patient": str})
    for p in AML_DONORS:
        for kind in ("loop", "stripe"):
            a, af = A[(A.patient == p) & (A.kind == kind)], Af[(Af.patient == p) & (Af.kind == kind)]
            pos = af[af.set == "pos"]
            cls = ["PE", "PS", "other"] if kind == "loop" else ["enh", "rep", "other"]
            row = dict(group="aml" + p, domain="aml", cancer=1, kind=kind, n_specific_total=len(pos),
                       **{f"total_{c}": int((pos.pair_type == c).sum()) for c in cls},
                       n_pos_used=len(pos), n_matched_A_full=int((a.set == "A_full").sum()),
                       n_matched_A_ref=int((a.set == "A_ref").sum()), n_private=int((pos.cls == "private").sum()),
                       fold=int(a.fold.iloc[0]))
            row["n_objects_A_full"] = row["n_matched_A_full"] * 2
            rows.append(row)
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "object_counts.csv", index=False)
    # totals and per-fold comparison with AML
    tot = []
    for kind in ("loop", "stripe"):
        for dom in ("cell", "aml"):
            c = C[(C.kind == kind) & (C.domain == dom)]
            tot.append(dict(kind=kind, domain=dom, n_groups=len(c), specific_total=int(c.n_specific_total.sum()),
                            pos_used=int(c.n_pos_used.sum()), matched_pairs_A_full=int(c.n_matched_A_full.sum()),
                            matched_pairs_A_ref=int(c.n_matched_A_ref.sum()), objects_A_full=int(c.n_objects_A_full.sum()),
                            **{f"total_{k}": int(c[f"total_{k}"].sum()) for k in (["PE", "PS", "other"] if kind == "loop" else ["enh", "rep", "other"])},
                            mean_objects_A_full_per_group=float(c.n_objects_A_full.mean())))
    fold = C[(C.domain == "aml")].groupby(["kind", "fold"]).agg(pos=("n_pos_used", "sum"), pairs=("n_matched_A_full", "sum")).reset_index()
    fold["objects_A_full"] = fold.pairs * 2
    pd.DataFrame(tot).to_csv(OUT / "object_counts_totals.csv", index=False)
    fold.to_csv(OUT / "object_counts_aml_per_fold.csv", index=False)
    log(pd.DataFrame(tot).to_string())
    log(fold.to_string())


# ================================================================================ figures
def stage_figures(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 150})
    K = pd.read_csv(OUT / "rules_key.csv")
    normal = [c for c in CELLS if c not in CANCER]
    cancer = [c for c in CELLS if c in CANCER]
    aml = ["aml" + p for p in AML_DONORS]
    cols = normal + cancer + aml + ["AML_pooled"]
    rules = [r[0] for r in KEY_RULES]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), sharey=True)
    for ax, tmode, title in ((axes[0], "own", "classes from the own marks of positives and controls"),
                             (axes[1], "inherit", "controls take the class of their matched positive (registry 64)")):
        k = K[(K.types == tmode) & (K.controls == "A_ref") & (K.ref == "hspc")]
        W = k.pivot_table(index="rule", columns="group", values="est", sort=False).reindex(index=rules, columns=cols)
        E = W.copy()
        for r in rules:
            if K[K.rule == r].metric.iloc[0] == "auroc":
                E.loc[r] = W.loc[r] - 0.5
        im = ax.imshow(E.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.3, 0.3))
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                v = W.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "-."), ha="center", va="center", fontsize=5.2,
                            color="white" if abs(E.values[i, j]) > 0.2 else "black")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace("aml", "") if c.startswith("aml") else c for c in cols], rotation=90)
        ax.axvline(len(normal) - 0.5, color="k", lw=0.8)
        ax.axvline(len(normal) + len(cancer) - 0.5, color="k", lw=1.6)
        ax.axvline(len(normal) + len(cancer) + len(aml) - 0.5, color="k", lw=0.8)
        ax.set_title(f"Controls matched on reference state (A_ref): {title}", fontsize=8)
        ax.text(len(normal) / 2 - 0.5, -1.6, "normal cells", ha="center", fontsize=8)
        ax.text(len(normal) + len(cancer) / 2 - 0.5, -1.6, "cancer cells", ha="center", fontsize=8)
        ax.text(len(normal) + len(cancer) + len(aml) / 2 - 0.5, -1.6, "AML donors (ref HSPC)", ha="center", fontsize=8)
    axes[0].set_yticks(range(len(rules)))
    axes[0].set_yticklabels([r + (" [AUROC]" if K[K.rule == r].metric.iloc[0] == "auroc" else " [rho]") for r in rules])
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="effect: Spearman rho, or AUROC - 0.5")
    fig.suptitle("Mark-to-structure rules per cell: delta of the cell over its leave-one-out consensus (AML: patient over HSPC); "
                 "numbers are rho or AUROC", fontsize=9)
    fig.savefig(OUT / "fig_effect_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    R = pd.read_csv(OUT / "transfer_results.csv")
    regimes = [("a_cells_to_cell", "cells (N-1) to cell"), ("c_aml_to_cell", "AML donors to cell"),
               ("e_cells_plus_aml", "cells (N-1) + AML to cell"), ("b_cells_to_aml", "all cells to AML donor"),
               ("d_aml_only", "other AML donors to donor"), ("d_aml_plus_cells", "other donors + cells to donor"),
               ("d_aml_plus_cells_w50", "other donors + cells (50:50) to donor")]
    tests = normal + cancer + aml
    for ctl in ("A_full", "A_ref"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 6.2))
        for r_, kind in enumerate(("loop", "stripe")):
            for c_, (met, mt) in enumerate((("auroc", "AUROC (specific vs matched shared)"), ("rho", "Spearman: predicted vs measured gain"))):
                ax = axes[r_, c_]
                g = R[(R.kind == kind) & (R.controls == ctl) & (R.featset == "ALL-M2")]
                M = np.full((len(regimes), len(tests)), np.nan)
                for i, (e, _) in enumerate(regimes):
                    h = g[g.experiment == e].set_index("test")[met]
                    for j, t in enumerate(tests):
                        if t in h.index:
                            M[i, j] = h[t]
                cen = 0.5 if met == "auroc" else 0.0
                lim = 0.25 if met == "auroc" else 0.5
                im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(cen, cen - lim, cen + lim))
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        if np.isfinite(M[i, j]):
                            ax.text(j, i, f"{M[i, j]:.2f}".replace("0.", ".").replace("-.", "-."), ha="center", va="center", fontsize=5.2,
                                    color="white" if abs(M[i, j] - cen) > 0.65 * lim else "black")
                ax.set_xticks(range(len(tests)))
                ax.set_xticklabels([t.replace("aml", "") for t in tests], rotation=90)
                ax.set_yticks(range(len(regimes)))
                ax.set_yticklabels([x[1] for x in regimes])
                ax.axvline(len(normal) - 0.5, color="k", lw=0.8)
                ax.axvline(len(normal) + len(cancer) - 0.5, color="k", lw=1.6)
                ax.set_title(f"{kind}s, {mt}", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        fig.suptitle(f"Transfer between cell lines and AML donors (boosted trees on delta, consensus-state and locus features; "
                     f"controls {ctl}; locus-disjoint, group-held-out); columns: normal cells | cancer cells | AML donors", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(OUT / ("fig_transfer_matrix.png" if ctl == "A_full" else "fig_transfer_matrix_Aref.png"), bbox_inches="tight")
        plt.close(fig)
    log("[figures] done")


# ================================================================================ main
STAGES = ["tracks", "hic", "objects", "features", "counts", "rules", "transfer", "figures"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    g = globals()
    for st in args.stage:
        log(f"=== stage {st}")
        fn = g.get(f"stage_{st}")
        if fn is None:
            log(f"stage {st}: not implemented"); continue
        if st in ("tracks", "hic"):
            fn(args.workers)
        else:
            fn(args)


if __name__ == "__main__":
    main()
