"""Does the across-cell variability of Hi-C (other cells only) predict where the
leave-one-out consensus is wrong for a target cell?

Per-cell proper O/E maps (scripts/oe_proper.py, crosscell_oe/<cell>_proper_<k>kb.cool)
are read in three window sets: the 413 Akita test regions ("akita413", the
convention of the reference_summary tables), the disjoint test windows of the
mc_all12ref cache ("testwin") and 300 random training windows ("fit", used only
to fit the shrinkage parameters and the loop-like thresholds). For a target cell
T the other cells are the 12 training cells for the 5 held-out cells
(k562, hmec, huvec, hmec_deep, huvec_deep; = consensus12) and the 11 other
training cells for a training cell (= consensus_loo_<T>). No held-out cell ever
enters a mean, a standard deviation or a threshold used for itself or anyone.

Per pixel (upper triangle, diagonal k >= 2, target and consensus nonzero):
  mean_others, std_others (ddof 1), presence (share of other cells above their
  own 90th percentile of that diagonal, taken on the fit windows), range
  (max - min), residual = measured_T - mean_others.

    python scripts/consensus_variance_diag.py --res 25000 10000 [5000]

Outputs: results/consensus_variance/*.csv and consensus_variance_<k>kb.png.
Pixel vectors are cached under --cache-dir (outside the repository).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = HM / "crosscell_oe"
CACHE = HM / "cache_expanded"
REGIONS = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/"
               "akita_normalization/akita_validation/sequences_test_list.tsv")
OUT = Path(__file__).resolve().parent.parent / "results" / "consensus_variance"

TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
           "gm23248", "mcf7", "cd14mono", "pc9"]
TEST5 = ["k562", "hmec", "huvec", "hmec_deep", "huvec_deep"]
TEST3 = ["k562", "hmec_deep", "huvec_deep"]
REP_CELLS = ["gm12878", "imr90", "k562", "hmec", "huvec"]       # rep1/rep2 single lane
REP_IDENT = {"hmec_deep": "hmec", "huvec_deep": "huvec"}
DEEP_PAIRS = {"k562_676BEP": ("intact_k562_ENCSR676BEP_r5", "intact_k562_ENCSR676BEP_r7"),
              "gm12878_653JLK": ("intact_gm12878_ENCSR653JLK_r7", "intact_gm12878_ENCSR653JLK_r9"),
              "gm12878_946AKI": ("intact_gm12878_ENCSR946AKI_r8", "intact_gm12878_ENCSR946AKI_r12"),
              "gm12878_965OVH": ("intact_gm12878_ENCSR965OVH_r75", "intact_gm12878_ENCSR965OVH_r76")}
BANDS_BP = [(50e3, 100e3), (100e3, 250e3), (250e3, 500e3), (500e3, 1e6), (1e6, 2.1e6)]
LAMBDAS = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8]
FEAT_EDGES = [0, .6, .8, 1., 1.2, 1.5, 2., np.inf]            # s / mean s of the diagonal
PRES_EDGES = [-1e-9, 1e-9, .1, .2, .35, .5, 1.0 + 1e-9]
DIAG_EDGES_BP = [0, 100e3, 200e3, 400e3, 800e3, 1.2e6, 1e9]


def cell_cool(name: str, res: int) -> Path:
    k = res // 1000
    if name in ("hmec", "huvec"):
        return OE / f"{name}_rep1_ice_{k}kb.cool"
    return OE / f"{name}_proper_{k}kb.cool"


def rep_cool(stem: str, res: int) -> Path:
    return OE / f"{stem}_ice_{res // 1000}kb.cool"


# ----------------------------------------------------------------- windows

def windows(res: int, kind: str, n_fit: int = 300, seed: int = 0):
    if kind == "akita413":
        r = pd.read_csv(REGIONS, sep="\t", header=None, usecols=[0, 1, 2],
                        names=["chrom", "start", "end"])
        return [(c, int(s), int(e)) for c, s, e in r.itertuples(index=False)]
    m = json.loads((CACHE / f"mc_all12ref_{res // 1000}kb" / "manifest.json").read_text())
    W = int(m["window_size"])
    split = "test" if kind == "testwin" else "train"
    out = []
    for sh in m["shards"]:
        if sh["sample"] == "gm12878" and sh["split"] == split:
            for st in np.load(sh["path"])["start"].astype(np.int64):
                out.append((sh["chrom"], int(st) * res, int(st + W) * res))
    if kind == "fit":
        rng = np.random.default_rng(seed)
        out = [out[i] for i in sorted(rng.choice(len(out), min(n_fit, len(out)), replace=False))]
    return out


def read_vectors(args):
    path, wins = args
    clr = cooler.Cooler(str(path))
    mat = clr.matrix(balance=False)
    names = set(clr.chromnames)
    vec, dg, wid = [], [], []
    for w, (c, s, e) in enumerate(wins):
        try:
            M = np.asarray(mat.fetch(f"{c if c in names else c.replace('chr', '')}:{s}-{e}"), float)
        except Exception:
            M = None
        if M is None or M.shape[0] != M.shape[1]:
            vec.append(None); dg.append(None); continue
        iu = np.triu_indices(M.shape[0], 2)
        vec.append(np.nan_to_num(M[iu]).astype(np.float32))
        dg.append((iu[1] - iu[0]).astype(np.int16))
    return vec, dg


class Windows:
    """Flat pixel vectors of every map in one window set (cached)."""

    def __init__(self, res, kind, names_paths, cache_dir, jobs):
        self.res, self.kind = res, kind
        wins = windows(res, kind.split("_rep")[0])
        self.wins = wins
        f = Path(cache_dir) / f"{kind}_{res // 1000}kb.npz"
        have = dict(np.load(f)) if f.exists() else {}
        todo = [(n, p) for n, p in names_paths.items() if f"v_{n}" not in have]
        if todo:
            with ProcessPoolExecutor(jobs) as ex:
                res_all = list(ex.map(read_vectors, [(p, wins) for _, p in todo]))
            for (n, _), (vec, dg) in zip(todo, res_all):
                if "diag" not in have:
                    ok = [i for i, v in enumerate(vec) if v is not None]
                    have["ok"] = np.array(ok)
                    have["diag"] = np.concatenate([dg[i] for i in ok])
                    have["win"] = np.concatenate([np.full(len(dg[i]), i, np.int32) for i in ok])
                ok = have["ok"]
                if any(vec[i] is None for i in ok):
                    raise SystemExit(f"{n}: window missing in {kind}")
                have[f"v_{n}"] = np.concatenate([vec[i] for i in ok])
                if len(have[f"v_{n}"]) != len(have["diag"]):
                    raise SystemExit(f"{n}: pixel count differs in {kind}")
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            np.savez(f, **have)
        self.diag = have["diag"].astype(np.int32)
        self.win = have["win"]
        self.V = {k[2:]: have[k] for k in have if k.startswith("v_")}


# ----------------------------------------------------------------- statistics

def pearson_by_group(x, y, g, mask, min_n=100):
    """Per group Pearson of x,y over mask (n > min_n, nonzero std); NaN otherwise."""
    g = g[mask]; x = x[mask].astype(np.float64); y = y[mask].astype(np.float64)
    G = int(g.max()) + 1 if len(g) else 0
    n = np.bincount(g, minlength=G).astype(float)
    sx, sy = np.bincount(g, x, G), np.bincount(g, y, G)
    sxx, syy, sxy = np.bincount(g, x * x, G), np.bincount(g, y * y, G), np.bincount(g, x * y, G)
    with np.errstate(all="ignore"):
        cov = sxy - sx * sy / n
        vx, vy = sxx - sx * sx / n, syy - sy * sy / n
        r = cov / np.sqrt(vx * vy)
    r[(n <= min_n) | (vx <= 1e-12) | (vy <= 1e-12)] = np.nan
    return r


def diag_rank(x, diag):
    """Percentile rank of x within each diagonal (average ties)."""
    return pd.Series(x).groupby(diag).rank(pct=True, method="average").values


def corr_ci(a, b, win, n_boot=300, seed=1):
    """Pearson r of a,b with a window bootstrap 95% CI (sufficient statistics per window)."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    G = int(win.max()) + 1
    S = np.stack([np.bincount(win, w, G) for w in (np.ones_like(a), a, b, a * a, b * b, a * b)], 1)

    def r_of(T):
        n, sa, sb, saa, sbb, sab = T
        return (sab - sa * sb / n) / np.sqrt((saa - sa ** 2 / n) * (sbb - sb ** 2 / n))
    r = r_of(S.sum(0))
    rng = np.random.default_rng(seed)
    bs = [r_of(S[rng.integers(0, G, G)].sum(0)) for _ in range(n_boot)]
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def r2_per_diag(y, x, diag, extra=None):
    """Pooled and median-per-diagonal R2 of y on x (OLS with intercept, per diagonal)."""
    sse = sst = 0.0
    r2s = []
    order = np.argsort(diag, kind="stable")
    d = diag[order]; ys = y[order].astype(np.float64)
    X = x[order].astype(np.float64)
    if X.ndim == 1:
        X = X[:, None]
    cuts = np.flatnonzero(np.diff(d)) + 1
    for yy, xx in zip(np.split(ys, cuts), np.split(X, cuts)):
        if len(yy) < 200:
            continue
        A = np.c_[np.ones(len(yy)), xx]
        beta = np.linalg.lstsq(A, yy, rcond=None)[0]
        e = ((yy - A @ beta) ** 2).sum(); t = ((yy - yy.mean()) ** 2).sum()
        sse += e; sst += t; r2s.append(1 - e / t if t > 0 else np.nan)
    return (1 - sse / sst if sst > 0 else np.nan), float(np.nanmedian(r2s)) if r2s else np.nan


def band_of(diag, res):
    bp = diag.astype(np.float64) * res
    b = np.full(len(diag), -1, np.int8)
    for i, (lo, hi) in enumerate(BANDS_BP):
        b[(bp >= lo) & (bp < hi)] = i
    return b


class Stats:
    """mean/std/presence/range of the other cells for one target in one window set."""

    def __init__(self, W: Windows, T: str, thr, train_cells=TRAIN12):
        self.T = T
        self.others = [c for c in train_cells if c != T]
        S = np.stack([W.V[c] for c in self.others])
        self.m = S.mean(0).astype(np.float32)
        self.s = S.std(0, ddof=1).astype(np.float32)
        th = np.stack([thr[c][np.minimum(W.diag, thr[c].shape[0] - 1)] for c in self.others])
        self.pres = (S > th).mean(0).astype(np.float32)
        self.rng = (S.max(0) - S.min(0)).astype(np.float32)
        self.y = W.V[T]
        self.valid = (self.y != 0) & (self.m != 0) & np.isfinite(self.y)
        self.res = (self.y - self.m).astype(np.float32)
        self.diag, self.win = W.diag, W.win
        # mean s of each diagonal (this target's own others, evaluation set)
        nd = int(W.diag.max()) + 1
        v = self.valid
        sd = np.bincount(W.diag[v], self.s[v], nd) / np.maximum(np.bincount(W.diag[v], minlength=nd), 1)
        self.sbar = np.maximum(sd, 1e-6)
        self.f = (self.s / self.sbar[W.diag]).astype(np.float32)


def thresholds(Wfit: Windows, cells, q=0.9):
    nd = int(Wfit.diag.max()) + 1
    thr = {}
    for c in cells:
        x = Wfit.V[c]; nz = x != 0
        qs = pd.Series(x[nz]).groupby(Wfit.diag[nz]).quantile(q)
        cnt = pd.Series(x[nz]).groupby(Wfit.diag[nz]).size()
        t = np.full(nd, np.inf)
        for d_, v_ in qs.items():
            if cnt[d_] > 50:
                t[d_] = v_
        t[np.isinf(t)] = np.nanmax(t[np.isfinite(t)])
        thr[c] = t.astype(np.float32)
    return thr


# ----------------------------------------------------------------- shrinkage

class Shrink:
    """Parameters fit on training cells' fit windows; leave-one-target-out sums."""

    def __init__(self, res, fit_stats: dict):
        self.res = res
        self.fit = fit_stats                       # training cell -> Stats on fit windows
        nd = max(int(s.diag.max()) for s in fit_stats.values()) + 1
        self.nd = nd
        self.sy = {}; self.ny = {}
        self.pear = {}; self.sums = {}
        for u, st in fit_stats.items():
            v = st.valid
            self.sy[u] = np.bincount(st.diag[v], st.y[v], nd)
            self.ny[u] = np.bincount(st.diag[v], minlength=nd).astype(float)
        self._mu_cache = {}

    def mu(self, T):
        """Per-diagonal mean of training-cell target values, never T's own."""
        if T not in self._mu_cache:
            us = [u for u in self.fit if u != T]
            self._mu_cache[T] = sum(self.sy[u] for u in us) / np.maximum(sum(self.ny[u] for u in us), 1)
        return self._mu_cache[T]

    def pred_lambda(self, st: Stats, mu, lam):
        w = 1.0 / (1.0 + lam * st.f ** 2)
        mud = mu[st.diag]
        return mud + w * (st.m - mud)

    def _lambda_scores(self):
        if "lam" not in self.pear:
            self.pear["lam"] = {}
            for u, st in self.fit.items():
                mu = self.mu(u)
                self.pear["lam"][u] = {lam: pearson_by_group(self.pred_lambda(st, mu, lam), st.y, st.win, st.valid)
                                       for lam in LAMBDAS}
        return self.pear["lam"]

    def best_lambda(self, T):
        sc = self._lambda_scores()
        us = [u for u in self.fit if u != T]
        med = {lam: np.nanmedian(np.concatenate([sc[u][lam] for u in us])) for lam in LAMBDAS}
        return max(med, key=med.get), med

    def _bin_of(self, st, feature):
        if feature == "s":
            return np.digitize(st.f, FEAT_EDGES[1:-1])
        if feature == "pres":
            return np.digitize(st.pres, PRES_EDGES[1:-1])
        if feature == "diag":
            return np.digitize(st.diag.astype(float) * self.res, DIAG_EDGES_BP[1:-1])
        raise ValueError(feature)

    def _slope_sums(self, feature):
        key = f"slope_{feature}"
        if key not in self.sums:
            self.sums[key] = {}
            for u, st in self.fit.items():
                mu = self.mu(u)[st.diag]
                b = self._bin_of(st, feature); v = st.valid
                x = (st.m - mu)[v].astype(np.float64); y = (st.y - mu)[v].astype(np.float64)
                nb = 8
                self.sums[key][u] = (np.bincount(b[v], x * y, nb), np.bincount(b[v], x * x, nb))
        return self.sums[key]

    def slopes(self, T, feature):
        sm = self._slope_sums(feature)
        us = [u for u in self.fit if u != T]
        xy = sum(sm[u][0] for u in us); xx = sum(sm[u][1] for u in us)
        return xy / np.maximum(xx, 1e-12)

    def pred_slope(self, st, mu, beta, feature):
        mud = mu[st.diag]
        return mud + beta[self._bin_of(st, feature)] * (st.m - mud)

    def const_w(self, T):
        sm = self._slope_sums("diag")
        us = [u for u in self.fit if u != T]
        return sum(sm[u][0].sum() for u in us) / sum(sm[u][1].sum() for u in us)


# ----------------------------------------------------------------- analysis

def corr_block(st: Stats, res, label):
    """All residual/variance associations for one target on one window set."""
    v = st.valid
    a = np.abs(st.res[v]); sg = st.res[v]; s = st.s[v]; p = st.pres[v]; rg = st.rng[v]
    d = st.diag[v]; w = st.win[v]
    mud = pd.Series(st.m[v]).groupby(d).transform("mean").values
    dev = st.m[v] - mud
    over = (st.m[v] - st.y[v]) * np.sign(dev)             # consensus overshoot away from the diagonal mean
    rank = {k: diag_rank(x, d) for k, x in (("abs", a), ("sg", sg), ("s", s), ("p", p),
                                            ("rg", rg), ("dev", dev), ("ov", over))}
    out = {"cell": st.T, "n_pixels": int(v.sum()), "n_windows": int(len(np.unique(w)))}
    sp = lambda x, y: float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    out["spearman_abs_vs_std_pixel"] = sp(a, s)
    r, lo, hi = corr_ci(rank["abs"], rank["s"], w)
    out["spearman_abs_vs_std_diagrank"], out["ci_lo"], out["ci_hi"] = float(r), float(lo), float(hi)
    for k, nm in (("p", "presence"), ("rg", "range")):
        out[f"spearman_abs_vs_{nm}_diagrank"] = float(np.corrcoef(rank["abs"], rank[k])[0, 1])
    out["spearman_signed_vs_std_pixel"] = sp(sg, s)
    for k, nm in (("s", "std"), ("p", "presence"), ("rg", "range"), ("dev", "mean_dev")):
        out[f"spearman_signed_vs_{nm}_diagrank"] = float(np.corrcoef(rank["sg"], rank[k])[0, 1])
    out["spearman_overshoot_vs_std_diagrank"] = float(np.corrcoef(rank["ov"], rank["s"])[0, 1])
    out["spearman_overshoot_vs_presence_diagrank"] = float(np.corrcoef(rank["ov"], rank["p"])[0, 1])
    # per-diagonal Spearman (mean over diagonals with >= 500 pixels)
    cnt = np.bincount(d)
    pd_r = []
    for k in np.flatnonzero(cnt >= 500):
        mk = d == k
        pd_r.append(np.corrcoef(rankdata(a[mk]), rankdata(s[mk]))[0, 1])
    out["mean_per_diagonal_spearman_abs_std"] = float(np.nanmean(pd_r))
    # (c) residual^2 on std^2 per diagonal
    r2p, r2m = r2_per_diag(a.astype(np.float64) ** 2, s.astype(np.float64) ** 2, d)
    out["R2_res2_on_std2_pooled"], out["R2_res2_on_std2_median_diag"] = r2p, r2m
    X = np.c_[s ** 2, p, rg ** 2, dev ** 2]
    out["R2_res2_on_4features_pooled"], _ = r2_per_diag(a.astype(np.float64) ** 2, X, d)
    # curves for the figure: |res| by std quantile bin within distance band
    band = band_of(d, res)
    rows = []
    for b in range(len(BANDS_BP)):
        mk = band == b
        if mk.sum() < 2000:
            continue
        q = np.quantile(s[mk], np.linspace(0, 1, 13))
        idx = np.clip(np.searchsorted(q, s[mk], side="right") - 1, 0, 11)
        for i in range(12):
            sel = idx == i
            rows.append({"cell": st.T, "band": b, "bin": i, "n": int(sel.sum()),
                         "std_mean": float(s[mk][sel].mean()), "abs_res_mean": float(a[mk][sel].mean()),
                         "res2_mean": float((a[mk][sel].astype(np.float64) ** 2).mean())})
    return out, rows, (rank["abs"], rank["s"], w, rank["sg"], rank["p"])


def pooled_from_cells(parts, tag):
    """Pooled Pearson of the within-diagonal ranks across cells, with window bootstrap CI."""
    off = 0; A = []; B = []; Wn = []
    for ra, rs, w in parts:
        A.append(ra); B.append(rs); Wn.append(w + off); off += int(w.max()) + 1
    r, lo, hi = corr_ci(np.concatenate(A), np.concatenate(B), np.concatenate(Wn))
    return {"group": tag, "spearman_abs_vs_std_diagrank_pooled": float(r), "ci_lo": float(lo), "ci_hi": float(hi)}


def evaluate(res, kind, W, thr, shr, targets_by_group, fout):
    rows_c, rows_curve, pooled, rows_shr, regs = [], [], [], [], []
    parts = {g: [] for g in targets_by_group}
    for g, cells in targets_by_group.items():
        for T in cells:
            st = Stats(W, T, thr)
            blk, curve, ranks = corr_block(st, res, g)
            blk.update({"res_kb": res // 1000, "set": kind, "group": g})
            rows_c.append(blk)
            for c in curve:
                c.update({"res_kb": res // 1000, "set": kind, "group": g}); rows_curve.append(c)
            parts[g].append(ranks[:3])
            # (d) shrinkage variants
            mu = shr.mu(T)
            lam, lam_med = shr.best_lambda(T)
            w_c = shr.const_w(T)
            variants = {"consensus": st.m,
                        f"const_w={w_c:.3f}": mu[st.diag] + w_c * (st.m - mu[st.diag]),
                        f"std_shrink_lambda={lam}": shr.pred_lambda(st, mu, lam),
                        "std_bins_slope": shr.pred_slope(st, mu, shr.slopes(T, "s"), "s"),
                        "presence_bins_slope": shr.pred_slope(st, mu, shr.slopes(T, "pres"), "pres"),
                        "distance_only_slope": shr.pred_slope(st, mu, shr.slopes(T, "diag"), "diag")}
            variants = {k: v for k, v in variants.items() if v is not None}
            base = None
            for name, pr in variants.items():
                r = pearson_by_group(pr, st.y, st.win, st.valid)
                if name == "consensus":
                    base = r
                short = name.split("=")[0]
                rows_shr.append({"res_kb": res // 1000, "set": kind, "group": g, "cell": T, "variant": short,
                                 "param": name.split("=")[1] if "=" in name else "",
                                 "n_regions": int(np.isfinite(r).sum()),
                                 "median_region_pearson": float(np.nanmedian(r)),
                                 "median_delta_vs_consensus": float(np.nanmedian(r - base)),
                                 "frac_regions_above_consensus": float(np.nanmean((r > base)[np.isfinite(r)])),
                                 "lambda_curve": json.dumps({str(k): round(float(v), 4) for k, v in lam_med.items()})
                                 if short.startswith("std_shrink") else ""})
            print(f"  [{res // 1000} kb {kind}] {T} ({g}) rho_diag={blk['spearman_abs_vs_std_diagrank']:.3f} "
                  f"R2={blk['R2_res2_on_std2_pooled']:.3f} cons={rows_shr[-len(variants)]['median_region_pearson']:.4f}",
                  flush=True)
    for g in targets_by_group:
        pooled.append(pooled_from_cells(parts[g], g) | {"res_kb": res // 1000, "set": kind})
    return rows_c, rows_curve, pooled, rows_shr


# ----------------------------------------------------------------- noise floor

def noise_analysis(res, kind, W, thr, Wrep):
    """Replicate noise: rep1-rep2 (single lane) and deep intact pairs."""
    rows = []
    sl = {}                                      # single-lane noise variance per pixel and cell
    for c in REP_CELLS:
        a, b = Wrep.V[f"{c}_rep1"], Wrep.V[f"{c}_rep2"]
        ok = (a != 0) & (b != 0)
        sl[c] = np.where(ok, (a - b) ** 2 / 2, np.nan).astype(np.float32)
    nz_sl = np.nanmean(np.stack(list(sl.values())), 0)
    deep = {}
    for nm, (r1, r2) in DEEP_PAIRS.items():
        a, b = Wrep.V[r1], Wrep.V[r2]
        deep[nm] = np.where((a != 0) & (b != 0), (a - b) ** 2 / 2, np.nan).astype(np.float32)
    nz_deep = np.nanmean(np.stack(list(deep.values())), 0)
    band = band_of(W.diag, res)
    for T in TRAIN12 + TEST5:
        st = Stats(W, T, thr)
        ident = REP_IDENT.get(T, T)
        v = st.valid & np.isfinite(nz_sl) & np.isfinite(nz_deep)
        d = st.diag[v]
        rs, rn = diag_rank(st.s[v], d), diag_rank(np.sqrt(nz_sl[v]), d)
        rd = diag_rank(np.sqrt(nz_deep[v]), d)
        rec = {"res_kb": res // 1000, "set": kind, "cell": T,
               "group": "heldout5" if T in TEST5 else "train12",
               "spearman_std_vs_singlelane_noise_diagrank": float(np.corrcoef(rs, rn)[0, 1]),
               "spearman_std_vs_deep_noise_diagrank": float(np.corrcoef(rs, rd)[0, 1]),
               "spearman_std_vs_singlelane_noise_pixel": float(np.corrcoef(rankdata(st.s[v]), rankdata(nz_sl[v]))[0, 1]),
               "frac_std2_singlelane_noise": float(np.mean(nz_sl[v]) / np.mean(st.s[v].astype(np.float64) ** 2)),
               "frac_std2_deep_noise": float(np.mean(nz_deep[v]) / np.mean(st.s[v].astype(np.float64) ** 2))}
        for b in range(len(BANDS_BP)):
            mb = v & (band == b)
            if mb.sum() > 2000:
                rec[f"band{b}_frac_std2_singlelane"] = float(np.mean(nz_sl[mb]) / np.mean(st.s[mb].astype(np.float64) ** 2))
                rec[f"band{b}_frac_std2_deep"] = float(np.mean(nz_deep[mb]) / np.mean(st.s[mb].astype(np.float64) ** 2))
        # the target's own replicate noise against its residual
        if ident in sl:
            vt = st.valid & np.isfinite(sl[ident])
            dt = st.diag[vt]
            ra, rr = diag_rank(np.abs(st.res[vt]), dt), diag_rank(np.sqrt(sl[ident][vt]), dt)
            rec["spearman_abs_res_vs_own_replicate_diff_diagrank"] = float(np.corrcoef(ra, rr)[0, 1])
            rec["frac_res2_own_singlelane_noise"] = float(np.mean(sl[ident][vt]) / np.mean(st.res[vt].astype(np.float64) ** 2))
            rs2 = diag_rank(st.s[vt], dt)
            # partial: does std still track |res| after removing the target's replicate noise rank
            from numpy.linalg import lstsq
            X = np.c_[np.ones(len(rr)), rr]
            ea = ra - X @ lstsq(X, ra, rcond=None)[0]
            es = rs2 - X @ lstsq(X, rs2, rcond=None)[0]
            rec["partial_spearman_abs_res_vs_std_given_own_noise"] = float(np.corrcoef(ea, es)[0, 1])
        rows.append(rec)
    # deep K562 pair as noise of the deep K562 target
    st = Stats(W, "k562", thr)
    a, b = Wrep.V[DEEP_PAIRS["k562_676BEP"][0]], Wrep.V[DEEP_PAIRS["k562_676BEP"][1]]
    vt = st.valid & (a != 0) & (b != 0)
    for r in rows:
        if r["cell"] == "k562":
            r["frac_res2_deep_pair_noise"] = float(np.mean((a[vt] - b[vt]) ** 2 / 2) / np.mean(st.res[vt].astype(np.float64) ** 2))
            r["spearman_abs_res_vs_deep_pair_diff_diagrank"] = float(np.corrcoef(
                diag_rank(np.abs(st.res[vt]), st.diag[vt]), diag_rank(np.abs(a[vt] - b[vt]), st.diag[vt]))[0, 1])
    return rows


# ----------------------------------------------------------------- figure

def figure(res, curve, outpng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ramp = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
    names = ["50-100 kb", "100-250 kb", "250-500 kb", "0.5-1 Mb", "1-2 Mb"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, g, title in ((axs[0], "heldout5", "5 held-out cells"), (axs[1], "train12", "12 training cells, leave-one-out")):
        d = curve[(curve.group == g)]
        for b in range(len(BANDS_BP)):
            x = d[d.band == b].groupby("bin").apply(lambda t: np.average(t.std_mean, weights=t.n), include_groups=False)
            y = d[d.band == b].groupby("bin").apply(lambda t: np.average(t.abs_res_mean, weights=t.n), include_groups=False)
            if len(x):
                ax.plot(x.values, y.values, "-o", ms=3, lw=1.6, color=ramp[b], label=names[b])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("std of the other cells (log O/E)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.25, lw=.5)
    axs[0].set_ylabel("mean |measured - consensus|")
    axs[1].legend(title="distance", fontsize=8, frameon=False)
    fig.suptitle(f"{res // 1000} kb: consensus error against across-cell variability (12 equal-count bins per band)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpng, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, nargs="+", default=[25000, 10000])
    ap.add_argument("--sets", nargs="+", default=["akita413", "testwin"])
    ap.add_argument("--cache-dir", default=str(HM / "cache_consensus_variance"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    groups = {"heldout5": TEST5, "heldout3": TEST3, "train12": TRAIN12}
    for res in a.res:
        k = res // 1000
        maps = {c: cell_cool(c, res) for c in TRAIN12 + TEST5}
        reps = {f"{c}_rep{i}": rep_cool(f"{c}_rep{i}", res) for c in REP_CELLS for i in (1, 2)}
        reps |= {n: rep_cool(n, res) for pair in DEEP_PAIRS.values() for n in pair}
        for n, p in {**maps, **reps}.items():
            if not p.exists():
                raise SystemExit(f"missing {p}")
        Wfit = Windows(res, "fit", maps, a.cache_dir, a.jobs)
        thr = thresholds(Wfit, TRAIN12)
        fit_stats = {u: Stats(Wfit, u, thr) for u in TRAIN12}
        shr = Shrink(res, fit_stats)
        allc, allcurve, allpool, allshr, allnoise = [], [], [], [], []
        for kind in a.sets:
            W = Windows(res, kind, maps, a.cache_dir, a.jobs)
            Wr = Windows(res, kind + "_rep", reps, a.cache_dir, a.jobs)
            # sanity: the stored leave-one-out consensus equals the recomputed mean
            if kind == "akita413":
                chk = cooler.Cooler(str(OE / f"consensus_loo_gm12878_proper_{k}kb.cool"))
                M = np.asarray(chk.matrix(balance=False).fetch(f"{W.wins[0][0]}:{W.wins[0][1]}-{W.wins[0][2]}"), float)
                n0 = M.shape[0]; iu = np.triu_indices(n0, 2)
                mine = np.stack([W.V[c][W.win == 0] for c in TRAIN12 if c != "gm12878"]).mean(0)
                print(f"[{k} kb] max |stored consensus_loo - recomputed mean| = {np.abs(M[iu] - mine).max():.2e}", flush=True)
            c, cu, po, sh = evaluate(res, kind, W, thr, shr, groups, None)
            allc += c; allcurve += cu; allpool += po; allshr += sh
            allnoise += noise_analysis(res, kind, W, thr, Wr)
        pd.DataFrame(allc).to_csv(out / f"corr_by_cell_{k}kb.csv", index=False)
        pd.DataFrame(allpool).to_csv(out / f"corr_pooled_{k}kb.csv", index=False)
        sh = pd.DataFrame(allshr)
        sh.to_csv(out / f"shrinkage_by_cell_{k}kb.csv", index=False)
        summ = (sh.groupby(["res_kb", "set", "group", "variant"])
                  .agg(n_cells=("cell", "size"), median_of_cell_medians=("median_region_pearson", "median"),
                       mean_delta_vs_consensus=("median_delta_vs_consensus", "mean"),
                       cells_above_consensus=("median_delta_vs_consensus", lambda x: int((x > 0).sum())))
                  .reset_index())
        summ.to_csv(out / f"shrinkage_summary_{k}kb.csv", index=False)
        pd.DataFrame(allnoise).to_csv(out / f"noise_floor_{k}kb.csv", index=False)
        curve = pd.DataFrame(allcurve)
        cur = curve[curve.set == a.sets[0]]
        cur.to_csv(out / f"binned_curve_{k}kb.csv", index=False)
        figure(res, cur, out / f"consensus_variance_{k}kb.png")
        print(f"[{k} kb] done", flush=True)


if __name__ == "__main__":
    main()
