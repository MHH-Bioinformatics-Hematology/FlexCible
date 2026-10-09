"""Similarity-weighted consensus against the plain leave-one-out consensus.

Question: does weighting the other cells' proper O/E maps by how similar their
ChIP (or DNA methylation) profiles are to the target cell's predict the target's
Hi-C better than the plain unweighted mean (the consensus of registry section
13)? CPU only, no model.

Targets: the 5 held-out cells (k562, hmec, huvec, hmec_deep, huvec_deep; the
references are the 12 training cells) and, as a second set, each of the 12
training cells with strict leave-one-out (the 11 other training cells). The
similarity S(T, c) uses only the inputs (ChIP bigWigs, Meth bigWig) of T and of
the references, never any Hi-C of T.

Similarity variants (13 ChIP tracks as the model reads them, log1p of the
binned mean signal; Meth = the section 27 methylation track):
  chip_gw        Pearson over all bins of the genome, mean over the tracks both
                 cells have
  chip_gw_sp     the same with Spearman
  ctcfk27_gw     CTCF and H3K27ac only (genome-wide Pearson)
  chip_loc       the Pearson computed inside each evaluation window (weights
                 vary by locus), mean over tracks
  ctcfk27_loc    CTCF and H3K27ac only, inside each window
  meth_gw / meth_loc   Meth track only; a cell without Meth (target or
                 reference) has no defined similarity and takes the mean of the
                 defined similarities of its row (a target without Meth gets
                 equal weights)
  comb_gw / comb_loc   mean of the available ChIP and Meth similarities

Weighting: softmax(S / tau) over the references, tau in a grid including
infinity (= plain mean), and top-k (k in 1, 2, 3, 5 and all = plain mean; k = 1
is the limit tau -> 0). tau and k are selected per variant on 300 random
training-chromosome windows (the "fit" set of consensus_variance_diag.py, which
never overlaps the test windows or the 413 Akita regions) by the median of the
per-window Pearson over the training cells as targets, leave-the-target-out (a
training target's own Hi-C never enters its own selection). The held-out cells
and the evaluation windows never select anything.

Evaluation as in consensus_variance_diag.py: median per-region Pearson on the
413 Akita test regions ("akita413") and on the disjoint test windows of the
mc_all12ref cache ("testwin"), upper triangle with k >= 2, target and plain
consensus nonzero. Reference rows labelled ORACLE use the target's own Hi-C
(best single other cell, non-negative least squares weights in-sample and
2-fold cross-fitted over regions, softmax of the true Hi-C similarity with the
best tau) and are not achievable.

    python scripts/consensus_weighted.py --res 25000 10000

Outputs: results/consensus_weighted/*.csv and consensus_weighted_<k>kb.png.
Binned tracks are cached under --cache-dir (outside the repository).
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls
from scipy.stats import rankdata, spearmanr

import consensus_variance_diag as cv
from consensus_variance_diag import (HM, TEST5, TRAIN12, Windows, cell_cool, pearson_by_group)

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = Path(__file__).resolve().parent.parent / "results" / "consensus_weighted"
CHIP = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
        "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
CORE = ["CTCF", "H3K27ac"]
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
SIM_CELLS = TRAIN12 + ["k562", "hmec", "huvec"]            # cells with input tracks
SRC = {"hmec_deep": "hmec", "huvec_deep": "huvec"}          # same inputs as the shallow map
CI = {c: i for i, c in enumerate(SIM_CELLS)}
TAUS = [np.inf, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02, 0.01, 0.005]
KS = ["all", 5, 3, 2, 1]
SIMS = ["chip_gw", "chip_gw_sp", "ctcfk27_gw", "chip_loc", "ctcfk27_loc",
        "meth_gw", "meth_loc", "comb_gw", "comb_loc"]
METH_ONLY = {"meth_gw", "meth_loc"}
SEED_NOISE = 0.013
TIE = 1e-4                                          # |delta| below this counts as a tie, not a win
GROUPS = {"heldout5": TEST5, "train12": TRAIN12}


def ci(c):
    return CI[SRC.get(c, c)]


# ----------------------------------------------------------------- input tracks

def track_path(cell, track):
    if track == "Meth":
        p = HM / "meth" / cell / "Meth.bigwig"
    else:
        p = HM / f"hg38_{cell}" / f"{track}.bigwig"
        if not p.exists():                      # gm12878 tracks live outside the project data folder
            p = Path.home() / "data/hic-multiome" / f"hg38_{cell}" / f"{track}.bigwig"
    return p if p.exists() else None


def read_binned(args):
    cell, track, res = args
    import pyBigWig
    p = track_path(cell, track)
    if p is None:
        return cell, track, None
    bw = pyBigWig.open(str(p))
    out = {}
    for ch in CHROMS:
        n = bw.chroms()[ch] // res
        v = bw.stats(ch, 0, n * res, type="mean", nBins=n)
        out[ch] = np.nan_to_num(np.array([np.nan if x is None else x for x in v], np.float32), nan=0.0)
    bw.close()
    return cell, track, out


def load_tracks(res, cache_dir, jobs):
    """dict (cell, track) -> {chrom: float32 bins}; ChIP log1p-transformed, Meth raw."""
    f = Path(cache_dir) / f"tracks_{res // 1000}kb.npz"
    if f.exists():
        z = np.load(f)
        T = {}
        for k in z.files:
            c, t, ch = k.split("|")
            T.setdefault((c, t), {})[ch] = z[k]
        return T
    todo = [(c, t, res) for c in SIM_CELLS for t in CHIP + ["Meth"]]
    T = {}
    with ProcessPoolExecutor(jobs) as ex:
        for c, t, o in ex.map(read_binned, todo):
            if o is not None:
                if t != "Meth":
                    o = {ch: np.log1p(np.clip(a, 0, None)) for ch, a in o.items()}
                T[(c, t)] = o
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(f, **{f"{c}|{t}|{ch}": a for (c, t), o in T.items() for ch, a in o.items()})
    return T


def zrows(A):
    """Row z-scores (NaN rows stay NaN); zero-variance rows become NaN."""
    mu = np.nanmean(A, 1, keepdims=True)
    sd = np.nanstd(A, 1, keepdims=True)
    Z = (A - mu) / np.where(sd > 1e-9, sd, np.nan)
    return Z


def corr_rows(A, rank=False):
    """Pairwise Pearson (or Spearman) between the rows of A (NaN rows -> NaN)."""
    A = np.array(A, np.float64)
    bad = np.isnan(A).all(1)
    if rank:
        A[~bad] = np.apply_along_axis(rankdata, 1, A[~bad])
    Z = zrows(A)
    ok = np.isfinite(Z).all(1)
    C = np.full((len(A), len(A)), np.nan)
    Zo = Z[ok]
    C[np.ix_(ok, ok)] = Zo @ Zo.T / Zo.shape[1]
    return C


def track_matrix(T, track, chroms, b0=None, n=None):
    """(15, bins) matrix of one track over the whole genome or one window; NaN rows if absent."""
    rows = []
    for c in SIM_CELLS:
        o = T.get((c, track))
        if o is None:
            rows.append(None)
        elif b0 is None:
            rows.append(np.concatenate([o[ch] for ch in chroms]))
        else:
            a = o[chroms][b0:b0 + n]
            rows.append(np.pad(a, (0, n - len(a))))
    L = max(len(r) for r in rows if r is not None)
    return np.stack([np.full(L, np.nan, np.float32) if r is None else r for r in rows])


def genome_sims(T):
    """(15, 15) similarity matrices, genome-wide."""
    per, per_sp = {}, {}
    for t in CHIP + ["Meth"]:
        M = track_matrix(T, t, CHROMS)
        per[t] = corr_rows(M)
        if t != "Meth":
            per_sp[t] = corr_rows(M, rank=True)
    chip = np.nanmean([per[t] for t in CHIP], 0)
    S = {"chip_gw": chip,
         "chip_gw_sp": np.nanmean([per_sp[t] for t in CHIP], 0),
         "ctcfk27_gw": np.nanmean([per[t] for t in CORE], 0),
         "meth_gw": per["Meth"],
         "comb_gw": np.nanmean([chip, per["Meth"]], 0)}
    return S, per


def window_geometry(W):
    """First bin and number of bins of every window of a Windows object."""
    n = pd.Series(W.diag).groupby(W.win).max().reindex(range(len(W.wins))).fillna(-1).values.astype(int) + 1
    b0 = np.array([s // W.res for _, s, _ in W.wins])
    return b0, n


def local_sims(T, W):
    """(nwin, 15, 15) similarity per window and variant."""
    b0, nb = window_geometry(W)
    G = len(W.wins)
    out = {k: np.full((G, 15, 15), np.nan) for k in ("chip_loc", "ctcfk27_loc", "meth_loc", "comb_loc")}
    for w, (ch, _, _) in enumerate(W.wins):
        if nb[w] < 6:
            continue
        per = {t: corr_rows(track_matrix(T, t, ch, int(b0[w]), int(nb[w]))) for t in CHIP + ["Meth"]}
        chip = np.nanmean([per[t] for t in CHIP], 0)
        out["chip_loc"][w] = chip
        out["ctcfk27_loc"][w] = np.nanmean([per[t] for t in CORE], 0)
        out["meth_loc"][w] = per["Meth"]
        out["comb_loc"][w] = np.nanmean([chip, per["Meth"]], 0)
    return out


# ----------------------------------------------------------------- weights

def fill_neutral(s):
    """Replace NaN similarities by the mean of the defined ones of the row; rows
    with none stay all zero (equal weights)."""
    s = np.array(s, np.float64)
    m = np.nanmean(s, 1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.where(np.isfinite(s), s, m)


def weights(s, kind, par):
    """s: (rows, R) filled similarities -> (rows, R) weights."""
    R = s.shape[1]
    if kind == "softmax":
        if np.isinf(par):
            return np.full(s.shape, 1.0 / R)
        z = np.exp((s - s.max(1, keepdims=True)) / par)
        return z / z.sum(1, keepdims=True)
    k = R if par == "all" else int(par)
    w = np.zeros_like(s)
    idx = np.argsort(-s, 1, kind="stable")[:, :k]
    np.put_along_axis(w, idx, 1.0 / k, 1)
    return w


def sim_rows(Sim, var, T, refs):
    """(1 or nwin, R) raw similarity of target T to the refs for one variant."""
    S = Sim[var]
    t = ci(T)
    r = [ci(c) for c in refs]
    return S[t, r][None, :] if S.ndim == 2 else S[:, t, r]


def predict(Vref, win, w):
    if w.shape[0] == 1:
        return (w[0].astype(np.float32) @ Vref)
    p = np.zeros(Vref.shape[1], np.float32)
    for r in range(Vref.shape[0]):
        p += w[:, r].astype(np.float32)[win] * Vref[r]
    return p


def region_pearson(pred, y, win, valid, G):
    r = pearson_by_group(pred, y, win, valid)
    return np.pad(r, (0, G - len(r)), constant_values=np.nan)


class Set:
    """Pixel vectors of one window set, with its window geometry and similarities."""

    def __init__(self, W, Sg, Sl):
        self.W, self.G = W, len(W.wins)
        self.win, self.diag = W.win, W.diag
        self.Sim = dict(Sg) | dict(Sl)

    def refs_for(self, T):
        return [c for c in TRAIN12 if c != T]

    def stack(self, refs):
        return np.stack([self.W.V[c] for c in refs]).astype(np.float32)


def cands():
    return [("softmax", t) for t in TAUS] + [("topk", k) for k in KS]


def n_defined(s):
    return float(np.isfinite(s).sum(1).mean())


def informative(var, T):
    """False for a variant that cannot differ from the plain mean for T (no Meth)."""
    return not (var in METH_ONLY and not track_path(SRC.get(T, T), "Meth"))


# ----------------------------------------------------------------- selection on fit windows

def fit_scores(F, log):
    """Per variant, candidate and training cell u: per-window Pearson on the fit windows."""
    sc = {}
    for u in TRAIN12:
        refs = F.refs_for(u)
        V = F.stack(refs)
        y = F.W.V[u]
        plain = V.mean(0)
        valid = (y != 0) & (plain != 0) & np.isfinite(y)
        for var in SIMS:
            s = fill_neutral(sim_rows(F.Sim, var, u, refs))
            for kind, par in cands():
                w = weights(s, kind, par)
                r = region_pearson(predict(V, F.win, w), y, F.win, valid, F.G)
                sc.setdefault(var, {}).setdefault((kind, par), {})[u] = r
        log(f"  fit scores {u}")
    return sc


def select(sc, var, T, kind):
    """Candidate of `kind` with the best pooled median Pearson over training cells
    other than T (and, for Meth-only variants, with Meth)."""
    us = [u for u in TRAIN12 if u != T and (var not in METH_ONLY or track_path(u, "Meth"))]
    best, bv, curve = None, -np.inf, {}
    for (k_, par), d in sc[var].items():
        if k_ != kind:
            continue
        v = float(np.nanmedian(np.concatenate([d[u] for u in us])))
        curve[par] = v
        if v > bv + 1e-12:
            best, bv = par, v
    return best, curve


# ----------------------------------------------------------------- oracles

def nnls_w(V, y, valid):
    A = V[:, valid].astype(np.float64)
    Gm = A @ A.T
    b = A @ y[valid].astype(np.float64)
    L = cholesky(Gm + 1e-9 * np.trace(Gm) / len(Gm) * np.eye(len(Gm)), lower=True)
    w, _ = nnls(L.T, solve_triangular(L, b, lower=True))
    return w


def oracle_nnls(V, y, valid, win, G, crossfit):
    if not crossfit:
        w = nnls_w(V, y, valid)
        return region_pearson(w.astype(np.float32) @ V, y, win, valid, G)
    r = np.full(G, np.nan)
    for fold in (0, 1):
        tr = valid & ((win % 2) == fold)
        w = nnls_w(V, y, tr)
        rr = region_pearson(w.astype(np.float32) @ V, y, win, valid & ((win % 2) != fold), G)
        sel = (np.arange(G) % 2) != fold
        r[sel] = rr[sel]
    return r


# ----------------------------------------------------------------- bootstrap

def boot_delta(Rv, Rp, n_boot, seed=3):
    """Rv, Rp: (cells, G) region Pearson of a variant and the plain mean.
    Returns per-cell and median-over-cells deltas (median-of-regions difference)
    with percentile bootstrap intervals over regions (shared resample)."""
    rng = np.random.default_rng(seed)
    G = Rv.shape[1]
    ok = np.isfinite(Rv) & np.isfinite(Rp)
    Rv = np.where(ok, Rv, np.nan); Rp = np.where(ok, Rp, np.nan)
    d0 = np.nanmedian(Rv, 1) - np.nanmedian(Rp, 1)
    m0 = np.median(d0)
    bc, bm = [], []
    for _ in range(n_boot):
        i = rng.integers(0, G, G)
        d = np.nanmedian(Rv[:, i], 1) - np.nanmedian(Rp[:, i], 1)
        bc.append(d); bm.append(np.median(d))
    bc = np.array(bc)
    return (d0, np.percentile(bc, 2.5, 0), np.percentile(bc, 97.5, 0),
            m0, np.percentile(bm, 2.5), np.percentile(bm, 97.5))


# ----------------------------------------------------------------- evaluation of one set

def evaluate_set(name, E, F_sc, res, n_boot, log):
    """All variants for all targets on one window set."""
    k = res // 1000
    G = E.G
    grid, rows, simrows, rankrows = [], [], [], []
    regions = {}                                     # (group, cell) -> {variant: region Pearson}
    for group, cells in GROUPS.items():
        for T in cells:
            refs = E.refs_for(T)
            V = E.stack(refs)
            y = E.W.V[T]
            plain = V.mean(0)
            valid = (y != 0) & (plain != 0) & np.isfinite(y)
            rp = region_pearson(plain, y, E.win, valid, G)
            reg = {"plain": rp}
            # Hi-C similarity of T with every reference (reference only)
            rc = np.stack([region_pearson(V[i], y, E.win, valid & (V[i] != 0), G) for i in range(len(refs))])
            h_med = np.nanmedian(rc, 1)
            reg["best_single (ORACLE)"] = rc[int(np.nanargmax(h_med))]
            for i, c in enumerate(refs):
                simrows.append({"res_kb": k, "set": name, "group": group, "cell": T, "ref": c,
                                "hic_corr_median_region": h_med[i]})
            reg["oracle_nnls_insample (ORACLE)"] = oracle_nnls(V, y, valid, E.win, G, False)
            reg["oracle_nnls_crossfit (ORACLE)"] = oracle_nnls(V, y, valid, E.win, G, True)
            # oracle similarity: softmax of the true Hi-C similarity, best tau in-sample
            sh = np.array(h_med)[None, :]
            best, bv = None, -np.inf
            for t in TAUS:
                r = region_pearson(predict(V, E.win, weights(fill_neutral(sh), "softmax", t)), y, E.win, valid, G)
                grid.append({"res_kb": k, "set": name, "group": group, "cell": T, "variant": "hicsim (ORACLE)",
                             "kind": "softmax", "par": t, "median_pearson": np.nanmedian(r)})
                if np.nanmedian(r) > bv:
                    best, bv, rb = t, np.nanmedian(r), r
            reg["oracle_hicsim_softmax (ORACLE)"] = rb
            sel_info = {"hicsim (ORACLE)": (best, "")}
            for var in SIMS:
                s_raw = sim_rows(E.Sim, var, T, refs)
                s = fill_neutral(s_raw)
                nd = n_defined(s_raw)
                sel_soft, _ = select(F_sc, var, T, "softmax")
                sel_topk, _ = select(F_sc, var, T, "topk")
                cache = {}
                for kind, par in cands():
                    w = weights(s, kind, par)
                    r = region_pearson(predict(V, E.win, w), y, E.win, valid, G)
                    cache[(kind, par)] = r
                    grid.append({"res_kb": k, "set": name, "group": group, "cell": T, "variant": var,
                                 "kind": kind, "par": par, "median_pearson": np.nanmedian(r),
                                 "delta_vs_plain": np.nanmedian(r) - np.nanmedian(rp)})
                reg[f"{var}|softmax_sel"] = cache[("softmax", sel_soft)]
                reg[f"{var}|topk_sel"] = cache[("topk", sel_topk)]
                reg[f"{var}|top1"] = cache[("topk", 1)]
                sel_info[var] = (sel_soft, sel_topk, nd)
                # rank agreement between S and the Hi-C similarity (step 4)
                gw = E.Sim[var].ndim == 2
                s_mean = np.nanmean(s_raw, 0) if not gw else s_raw[0]
                okm = np.isfinite(s_mean)
                rho = spearmanr(s_mean[okm], h_med[okm])[0] if okm.sum() >= 4 else np.nan
                rho_w = np.nan
                if not gw:
                    rr = []
                    for w_ in range(G):
                        a, b = s_raw[w_], rc[:, w_]
                        m = np.isfinite(a) & np.isfinite(b)
                        if m.sum() >= 4 and np.ptp(a[m]) > 0 and np.ptp(b[m]) > 0:
                            rr.append(spearmanr(a[m], b[m])[0])
                    rho_w = float(np.median(rr)) if rr else np.nan
                rankrows.append({"res_kb": k, "set": name, "group": group, "cell": T, "variant": var,
                                 "n_refs_with_S": int(okm.sum()), "n_refs": len(refs),
                                 "spearman_S_vs_hic": rho, "median_window_spearman": rho_w})
                simrows.extend({"res_kb": k, "set": name, "group": group, "cell": T, "ref": c,
                                f"S_{var}": s_mean[i]} for i, c in enumerate(refs))
            for var, v in sel_info.items():
                rows.append({"res_kb": k, "set": name, "group": group, "cell": T, "variant": var,
                             "selected_tau": v[0], "selected_k": v[1] if len(v) > 2 else "",
                             "refs_with_S": v[2] if len(v) > 2 else len(refs)})
            regions[(group, T)] = reg
            log(f"  [{k} kb {name}] {T}: plain {np.nanmedian(rp):.4f}")
    # per cell and summary tables
    names = list(next(iter(regions.values())).keys())
    per, summ = [], []
    for group, cells in GROUPS.items():
        Rp = np.stack([regions[(group, T)]["plain"] for T in cells])
        for nm in names:
            if nm == "plain":
                continue
            Rv = np.stack([regions[(group, T)][nm] for T in cells])
            d0, lo, hi, m0, mlo, mhi = boot_delta(Rv, Rp, n_boot)
            med_v = np.nanmedian(Rv, 1); med_p = np.nanmedian(Rp, 1)
            var0 = nm.split("|")[0]
            infm = np.array([informative(var0, T) if var0 in SIMS else True for T in cells])
            for i, T in enumerate(cells):
                per.append({"res_kb": k, "set": name, "group": group, "cell": T, "variant": nm,
                            "median_pearson": med_v[i], "plain": med_p[i], "delta": d0[i],
                            "delta_ci_lo": lo[i], "delta_ci_hi": hi[i],
                            "frac_regions_above_plain": float(np.nanmean((Rv[i] > Rp[i])[np.isfinite(Rv[i] - Rp[i])])),
                            "informative": bool(infm[i])})
            summ.append({"res_kb": k, "set": name, "group": group, "variant": nm, "n_cells": len(cells),
                         "n_informative": int(infm.sum()),
                         "median_pearson_over_cells": float(np.median(med_v)),
                         "plain_median_over_cells": float(np.median(med_p)),
                         "median_delta": float(np.median(d0)), "mean_delta": float(np.mean(d0)),
                         "median_delta_ci_lo": mlo, "median_delta_ci_hi": mhi,
                         "cells_won": int((d0 > TIE).sum()), "cells_lost": int((d0 < -TIE).sum()),
                         "cells_tied": int((np.abs(d0) <= TIE).sum()),
                         "diff_of_medians": float(np.median(med_v) - np.median(med_p))})
        summ.append({"res_kb": k, "set": name, "group": group, "variant": "plain", "n_cells": len(cells),
                     "n_informative": len(cells),
                     "median_pearson_over_cells": float(np.median(np.nanmedian(Rp, 1))),
                     "plain_median_over_cells": float(np.median(np.nanmedian(Rp, 1))), "median_delta": 0.0})
    return grid, rows, simrows, rankrows, per, summ, regions


# ----------------------------------------------------------------- figure

def figure(res, summ, per, rank, fitcurves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = res // 1000
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
    blue, grey, orange = "#08519c", "#8a8a8a", "#d95f02"
    vs = [f"{v}|softmax_sel" for v in SIMS]
    lab = [v for v in SIMS]
    for a, g, title in ((ax[0, 0], "heldout5", "5 held-out cells"), (ax[0, 1], "train12", "12 training cells, leave-one-out")):
        d = per[(per.set == "akita413") & (per.group == g)]
        for j, v in enumerate(vs):
            x = d[d.variant == v]
            a.scatter(np.full(len(x), j) + np.random.default_rng(j).uniform(-.12, .12, len(x)), x.delta, s=16, color=blue, alpha=.8, zorder=3)
            a.plot([j - .25, j + .25], [x.delta.median()] * 2, color="k", lw=1.6, zorder=4)
        m = d[d.variant == "oracle_hicsim_softmax (ORACLE)"].delta.median()
        a.axhline(m, color=orange, lw=1.2, ls="--", label="ORACLE softmax of the true Hi-C similarity (median, not achievable)")
        a.axhline(0, color="k", lw=.8)
        a.axhspan(-SEED_NOISE, SEED_NOISE, color=grey, alpha=.12, label="seed noise 0.013")
        a.axhline(0.005, color=blue, lw=.8, ls=":")
        a.set_xticks(range(len(vs))); a.set_xticklabels(lab, rotation=40, ha="right", fontsize=8)
        a.set_ylabel("median Pearson minus plain mean (413 regions)")
        a.set_title(f"{title}: softmax weights, tau fitted on training cells", fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
    ax[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    a = ax[1, 0]
    r = rank[(rank.set == "akita413")]
    for j, v in enumerate(SIMS):
        for g, c in (("heldout5", blue), ("train12", grey)):
            x = r[(r.variant == v) & (r.group == g)].spearman_S_vs_hic.dropna()
            a.scatter(np.full(len(x), j) + (.12 if g == "train12" else -.12), x, s=14, color=c, alpha=.8)
    a.axhline(0, color="k", lw=.8)
    a.set_xticks(range(len(SIMS))); a.set_xticklabels(SIMS, rotation=40, ha="right", fontsize=8)
    a.set_ylabel("Spearman, S vs Hi-C similarity of the references")
    a.set_title("rank agreement (blue held-out, grey training)", fontsize=10)
    a.spines[["top", "right"]].set_visible(False)
    a = ax[1, 1]
    for v, c in zip(("chip_gw", "ctcfk27_gw", "meth_gw", "comb_gw", "chip_loc"),
                    ("#08519c", "#6baed6", "#d95f02", "#1b9e77", "#7570b3")):
        cur = fitcurves.get(v, {})
        xs = [t if np.isfinite(t) else 3.0 for t in TAUS]
        a.plot(xs, [cur.get(t, np.nan) for t in TAUS], "-o", ms=3, color=c, label=v)
    a.set_xscale("log"); a.invert_xaxis()
    a.set_xticks([3.0, 1, .3, .1, .03, .01]); a.set_xticklabels(["inf", "1", "0.3", "0.1", "0.03", "0.01"])
    a.minorticks_off()
    a.set_xlabel("tau (inf = plain mean)"); a.set_ylabel("median window Pearson, 12 training cells")
    a.set_title("tau selection on the fit windows", fontsize=10)
    a.legend(fontsize=8, frameon=False); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{k} kb: similarity-weighted consensus against the plain mean", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, nargs="+", default=[25000, 10000])
    ap.add_argument("--sets", nargs="+", default=["akita413", "testwin"])
    ap.add_argument("--cache-dir", default=str(HM / "cache_consensus_weighted"))
    ap.add_argument("--hic-cache-dir", default=str(HM / "cache_consensus_variance"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--figure-only", action="store_true", help="redraw the figures from the CSVs in --out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)

    if a.figure_only:
        for res in a.res:
            k = res // 1000
            fs = pd.read_csv(out / f"fit_selection_{k}kb.csv")
            curves = {v: {(np.inf if p == "inf" else float(p)): x for p, x in zip(d.par, d.median_window_pearson_pooled)}
                      for v, d in fs[fs.kind == "softmax"].groupby("variant")}
            figure(res, None, pd.read_csv(out / f"per_cell_{k}kb.csv"), pd.read_csv(out / f"rank_agreement_{k}kb.csv"),
                   curves, out / f"consensus_weighted_{k}kb.png")
        return
    for res in a.res:
        k = res // 1000
        maps = {c: cell_cool(c, res) for c in TRAIN12 + TEST5}
        T = load_tracks(res, a.cache_dir, a.jobs)
        have = {c: [t for t in CHIP + ["Meth"] if (c, t) in T] for c in SIM_CELLS}
        log(f"[{k} kb] tracks per cell: " + ", ".join(f"{c}={len(v)}" for c, v in have.items()))
        Sg, per_track = genome_sims(T)
        sets = {}
        for kind in ["fit"] + a.sets:
            W = Windows(res, kind, maps, a.hic_cache_dir, a.jobs)
            sets[kind] = Set(W, Sg, local_sims(T, W))
            log(f"[{k} kb] {kind}: {len(W.wins)} windows, {len(W.diag)} pixels")
        fit = sets["fit"]
        # fit windows never overlap the evaluation windows
        for kind in a.sets:
            n_ov = sum(1 for (c, s, e) in fit.W.wins for (c2, s2, e2) in sets[kind].W.wins
                       if c == c2 and s < e2 and s2 < e)
            log(f"[{k} kb] fit windows overlapping {kind}: {n_ov}")
            if n_ov:
                raise SystemExit("fit windows overlap the evaluation windows")
        sc = fit_scores(fit, log)
        fitcurves = {v: select(sc, v, "k562", "softmax")[1] for v in SIMS}      # all 12 training cells
        pd.DataFrame([{"res_kb": k, "variant": v, "kind": kd, "par": p, "median_window_pearson_pooled": x}
                      for v in SIMS for kd in ("softmax", "topk")
                      for p, x in select(sc, v, "k562", kd)[1].items()]).to_csv(out / f"fit_selection_{k}kb.csv", index=False)
        G_, R_, S_, K_, P_, M_ = [], [], [], [], [], []
        for kind in a.sets:
            g, r, s, rk, p, m, _ = evaluate_set(kind, sets[kind], sc, res, a.n_boot, log)
            G_ += g; R_ += r; S_ += s; K_ += rk; P_ += p; M_ += m
        pd.DataFrame(G_).to_csv(out / f"grid_by_cell_{k}kb.csv", index=False)
        pd.DataFrame(R_).to_csv(out / f"selected_params_{k}kb.csv", index=False)
        pd.DataFrame(S_).groupby(["res_kb", "set", "group", "cell", "ref"], as_index=False).first() \
            .to_csv(out / f"similarity_{k}kb.csv", index=False)
        rank = pd.DataFrame(K_); rank.to_csv(out / f"rank_agreement_{k}kb.csv", index=False)
        per = pd.DataFrame(P_); per.to_csv(out / f"per_cell_{k}kb.csv", index=False)
        summ = pd.DataFrame(M_); summ.to_csv(out / f"summary_{k}kb.csv", index=False)
        pt = [{"res_kb": k, "cell_a": c, "track": t, "pearson_vs": cc, "r": per_track[t][ci(c), ci(cc)]}
              for t in per_track for c in ("k562", "hmec", "huvec") for cc in TRAIN12]
        pd.DataFrame(pt).to_csv(out / f"similarity_by_track_{k}kb.csv", index=False)
        figure(res, summ, per, rank, fitcurves, out / f"consensus_weighted_{k}kb.png")
        log(f"[{k} kb] done")


if __name__ == "__main__":
    main()
