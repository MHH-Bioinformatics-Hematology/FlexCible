"""Structure scores of our models against Akita and the other published predictors, one definition
on identical pixels (registry section 73).

Stages
  akita   413 Akita test regions, GM12878, 2048 bp grid, 448-bin crop (Akita's target space): Akita,
          Hi-cGAN (submitted), our ChIP-only model (akita_native_honest), our model with the
          consensus (akita_native_refres), the consensus of four measured cells (consensus4) and the
          HFF measured map. All predictions are read from disk (nothing is rerun). Every metric of a
          region uses the same pixels: upper triangle, >= 2 bins off the diagonal, finite in the
          target and in every method except Hi-cGAN (which has no prediction in 2 regions; where it
          exists it is finite on the whole mask).
  aml     AML object windows (10 kb, 192 bins) of 10 held-out donors: C.Origami, Epiphany, Akita
          (saved predictions of scripts/aml_competitors_loops.py) and our no-AML 12-cell models and
          consensus12, against the measured patient map, same metric definitions at 10 kb;
          plus the pooled table of eval_f10/11/12_patients.csv (our AML-inclusive models).
  chrho   whole held-out chromosomes (results/hic_measures_chrho and results/benchmark chrho tables):
          collected, nothing recomputed.
  report  markdown tables.

Metric definitions (all maps log O/E; higher = better)
  pearson, scc          Pearson over the mask pixels; HiCRep stratum-adjusted correlation (benchmark_metrics)
  ins_r_<w>             Pearson between the insulation profiles (mean log O/E in the square
                        [i-w, i) x (i, i+w], flexcible.structure.insulation) of prediction and
                        target; w in bins (2 kb grid 5, 12, 25, 50, 122 = 10, 25, 51, 102, 250 kb;
                        10 kb grid 5, 10, 25 = 50, 100, 250 kb)
  bnd_*                 boundaries = local minima of the 51 kb (2 kb grid) / 50 kb (10 kb grid)
                        insulation profile whose prominence over the maxima of the 51 / 50 kb
                        flanks is >= 0.5 SD of that profile (each map its own SD), minima closer than
                        tol + 1 bins merged. Reference = the boundaries of the measured target on
                        the same pixels. Recall, precision, F1 with tolerance 5 bins (10.2 kb) at
                        2 kb and 1 bin (10 kb) at 10 kb; strict = 2 bins (2 kb grid) / 0 bins.
                        Per-region F1 and pooled (micro) recall, precision, F1.
  bnd_strength_r        Pearson over bins between the boundary-strength profiles (prominence of the
                        insulation minimum over its flanks, defined for every bin) of prediction and
                        target
  moc                   measure of concordance (Zufferey et al. 2018) between the domain partitions
                        defined by the boundaries above, over the bins where both profiles exist
  hft_bnd_recall        pooled recall of the hicFindTADs boundaries of the measured count map
                        (10 kb, reference of results/benchmark) by the K strongest insulation minima
                        of the map, K = number of reference boundaries in the region, tolerance 5
                        bins (benchmark_metrics.boundary_recall; rank-matched, so recall = precision)
  loop_recall, apa_ratio, anchor_auroc
                        hicDetectLoops loops of the measured count map as reference; K strongest
                        donut enrichments of the map; APA ratio = (APA_pred - 1) / (APA_target - 1)
                        (benchmark_metrics.loop_counts / pool_loops; parameters of
                        benchmark_blocks.params(2048))
  loop_enrich_corr, compartment_corr, eigenvector_abs_corr, bnd_f1_struct, ins_r_250k
                        flexcible.structure.structure_metrics (the definition behind the
                        insulation_corr / boundary_f1 columns of the AML eval tables)
  stripe_r              Pearson over bins between the anchor profiles (mean log O/E of the pixels
                        20 to 200 kb from the bin, both sides) of prediction and target; a simple
                        stripe-related proxy, not a stripe caller

    python scripts/structure_scores_vs_akita.py --stage akita aml chrho report --threads 5
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
BENCH = HM / "runs/benchmark"
OUT = REPO / "results/structure_vs_akita"
RASTER = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization/"
              "akita_testset_rasters.npz")
AML_COMP = HM / "runs/aml_competitors_loops"
AML_REMOTE = HM / "runs/aml_train_remote"

AK_METHODS = ["akita", "hicgan", "akita_native_honest", "akita_native_refres", "consensus4", "map_hff"]
OURS = ["akita_native_honest", "akita_native_refres"]
LABEL = {"akita": "Akita (GM12878 head)", "hicgan": "Hi-cGAN (submitted)",
         "akita_native_honest": "ours, ChIP only", "akita_native_refres": "ours, with consensus",
         "consensus4": "consensus of 4 measured cells", "map_hff": "HFF measured map"}


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


# --------------------------------------------------------------------------- #
# metric definitions
# --------------------------------------------------------------------------- #
def _corr(a, b, min_n=10):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return np.nan
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def prominence_profile(prof, flank):
    """min(max of the left flank, max of the right flank) - value, NaN where a flank is empty."""
    n = prof.size
    out = np.full(n, np.nan)
    for b in range(n):
        v = prof[b]
        if not np.isfinite(v):
            continue
        lo, hi = prof[max(0, b - flank):b], prof[b + 1:b + 1 + flank]
        lo, hi = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
        if lo.size and hi.size:
            out[b] = min(lo.max(), hi.max()) - v
    return out


def call_boundaries(prof, flank, frac, tol):
    """Local minima with prominence >= frac * SD(prof); minima within tol bins merged (strongest kept)."""
    sd = np.nanstd(prof)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros(0, int), prominence_profile(prof, flank)
    prm = prominence_profile(prof, flank)
    cand = []
    for b in range(1, prof.size - 1):
        v = prof[b]
        if np.isfinite(v) and v <= prof[b - 1] and v <= prof[b + 1] and np.isfinite(prm[b]) \
                and prm[b] >= frac * sd:
            cand.append(b)
    cand.sort(key=lambda b: -prm[b])
    kept = []
    for b in cand:
        if all(abs(b - k) > tol for k in kept):
            kept.append(b)
    return np.array(sorted(kept), int), prm


def match_counts(ref, pred, tol):
    if ref.size == 0 or pred.size == 0:
        return 0, 0
    rec = int(sum(np.abs(pred - x).min() <= tol for x in ref))
    prec = int(sum(np.abs(ref - x).min() <= tol for x in pred))
    return rec, prec


def f1_of(rec_hits, n_ref, prec_hits, n_pred):
    if n_ref == 0 and n_pred == 0:
        return np.nan
    if n_ref == 0 or n_pred == 0:
        return 0.0
    r, p = rec_hits / n_ref, prec_hits / n_pred
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def moc(bt, bp, lo, hi):
    """Measure of concordance of the partitions of bins [lo, hi) cut at boundaries bt and bp."""
    if hi - lo < 4:
        return np.nan
    idx = np.arange(lo, hi)
    lt = np.searchsorted(np.sort(bt), idx, side="right")
    lp = np.searchsorted(np.sort(bp), idx, side="right")
    nt, npp = lt.max() + 1, lp.max() + 1
    if nt * npp == 1:
        return np.nan
    C = np.zeros((nt, npp))
    np.add.at(C, (lt, lp), 1)
    st, sp = C.sum(1), C.sum(0)
    tot = (C ** 2 / np.outer(np.maximum(st, 1), np.maximum(sp, 1))).sum()
    den = np.sqrt(nt * npp) - 1
    return float((tot - 1) / den) if den > 0 else np.nan


def anchor_profile(M, dmin, dmax):
    n = M.shape[0]
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    sel = (d >= dmin) & (d <= dmax) & np.isfinite(M)
    cnt = sel.sum(1)
    s = np.where(sel, M, 0.0).sum(1)
    return np.where(cnt >= 10, s / np.maximum(cnt, 1), np.nan)


GRID = {
    2048: dict(ins=[(5, "10k"), (12, "25k"), (25, "50k"), (50, "100k"), (122, "250k")],
               bw=25, flank=25, tol=5, strict=2, stripe=(10, 98)),
    10000: dict(ins=[(5, "50k"), (10, "100k"), (25, "250k")],
                bw=5, flank=5, tol=1, strict=0, stripe=(2, 20)),
}


def symmetrize(M, mask):
    U = np.where(mask, M, np.nan)
    return np.where(np.isnan(U), U.T, U)


def region_scores(T, P, mask, res, ref_bnd=None, ref_loops=None, seed=0, hft=True):
    """All metrics of map P against target T (both symmetric, NaN outside the shared mask)."""
    import benchmark_metrics as bm
    from flexcible import structure as st
    g = GRID[res]
    out = {}
    out["pearson"] = bm.pearson(P, T, mask)
    out["scc"] = bm.scc(P, T, mask)
    it_by = {}
    for w, lab in g["ins"]:
        ip, it = st.insulation(P, w), st.insulation(T, w)
        out[f"ins_r_{lab}"] = _corr(ip, it)
        it_by[w] = (ip, it)
    ip, it = it_by[g["bw"]]
    bt, st_t = call_boundaries(it, g["flank"], 0.5, g["tol"])
    bp, st_p = call_boundaries(ip, g["flank"], 0.5, g["tol"])
    for tag, tol in (("", g["tol"]), ("_strict", g["strict"])):
        r_hit, p_hit = match_counts(bt, bp, tol)
        out[f"bnd_f1{tag}"] = f1_of(r_hit, bt.size, p_hit, bp.size)
        if tag == "":
            out["bnd_rec_hits"], out["bnd_ref_n"] = r_hit, int(bt.size)
            out["bnd_prec_hits"], out["bnd_pred_n"] = p_hit, int(bp.size)
            out["bnd_recall"] = r_hit / bt.size if bt.size else np.nan
            out["bnd_precision"] = p_hit / bp.size if bp.size else np.nan
    out["bnd_strength_r"] = _corr(st_p, st_t)
    ok = np.isfinite(ip) & np.isfinite(it)
    if ok.sum() > 4:
        lo, hi = int(np.where(ok)[0].min()), int(np.where(ok)[0].max()) + 1
        out["moc"] = moc(bt, bp, lo, hi)
    else:
        out["moc"] = np.nan
    dmin, dmax = g["stripe"]
    out["stripe_r"] = _corr(anchor_profile(P, dmin, dmax), anchor_profile(T, dmin, dmax))
    try:
        sm = st.structure_metrics(P, T, res)
        out["loop_enrich_corr"] = sm["loop_enrich_corr"]
        out["compartment_corr"] = sm["compartment_corr"]
        out["eigenvector_abs_corr"] = sm["eigenvector_abs_corr"]
        out["bnd_f1_struct"] = sm["boundary_f1"]
        out["ins_r_struct"] = sm["insulation_corr"]
    except Exception:                                           # noqa: BLE001
        pass
    lc = None
    if hft:
        import benchmark_blocks as bb
        prm = bb.params(res)
        h, nb = bm.boundary_recall(np.nan_to_num(P), ref_bnd, prm["ins"], prm["btol"], prm["flank"])
        out["hft_hits"], out["hft_n"] = h, nb
        lc = bm.loop_counts(T, P, ref_loops, prm["loop"], seed=seed)
        out["loop_hits"], out["loop_n"] = lc["loop_hits"], lc["loop_n"]
    return out, lc


# --------------------------------------------------------------------------- #
# stage akita
# --------------------------------------------------------------------------- #
_JOBS: list = []


def _akita_worker(i):
    key, b0, T, maps, mask, bnd, loops = _JOBS[i]
    n = T.shape[0]
    Tm = symmetrize(T, mask)
    rows, lcs = [], {}
    for m, P in maps.items():
        if P is None:
            rows.append({"region": key, "method": m})
            continue
        sub = mask & np.isfinite(P)
        if sub.sum() != mask.sum():
            rows.append({"region": key, "method": m, "note": "partial"})
            continue
        Pm = symmetrize(P, mask)
        r, lc = region_scores(Tm, Pm, mask, 2048, bnd, loops, seed=i)
        r.update(region=key, method=m, n_pixels=int(mask.sum()))
        rows.append(r)
        lcs[m] = lc
    return rows, lcs


def stage_akita(threads, boot, seed, limit=0):
    import benchmark_blocks as bb
    import benchmark_competitors as bc
    from flexcible import bioscore as bs
    Z = np.load(RASTER, allow_pickle=True)
    side, off = int(Z["side"]), int(Z["offset"])
    crop = (512 - side) // 2
    regs = [str(r) for r in Z["regions"]]
    blkA = np.load(BENCH / "blocks/A_gm12878_2kb.npz")
    spaces = {n: np.load(BENCH / "pred/akita_space" / f"{n}.npz") for n in OURS}
    meta = {"cell": "gm12878", "res": 2048}
    n_part = 0
    for i, reg in enumerate(regs):
        if limit and i >= limit:
            break
        chrom, se = reg.split(":")
        s = int(se.split("-")[0])
        dense = lambda v: bs.dense_from_triu(v, side, off).astype(float)       # noqa: E731
        T = dense(Z["akita_target"][i, :, 1])
        T[T == 0] = np.nan
        maps = {"akita": dense(Z["akita_pred"][i, :, 1])}
        h = dense(Z["hicgan"][i])
        maps["hicgan"] = h if np.isfinite(h).any() else None
        for n in OURS:
            maps[n] = np.asarray(spaces[n][reg], float)[crop:crop + side, crop:crop + side]
        for k, f in (("consensus4", "akita_consensus4_2048bp.cool"), ("map_hff", "akita_hff_all_2048bp.cool")):
            if f"{reg}|{k}" in blkA.files:
                maps[k] = blkA[f"{reg}|{k}"].astype(float)
            else:
                B = bb.fetch_oe(bb._cool(bc.AK2 / f), chrom, s // 2048, s // 2048 + 512)
                maps[k] = None if B is None else np.where(B == 0, np.nan, B)[crop:crop + side, crop:crop + side]
        iu = np.triu(np.ones((side, side), bool), 2)
        mask = iu & np.isfinite(T)
        for k in AK_METHODS:
            if k != "hicgan" and maps[k] is not None:
                mask &= np.isfinite(maps[k])
        b0 = s // 2048 + crop
        bnd, loops = bb.ref_sets(meta, reg, b0, side)
        if mask.sum() < 500:
            continue
        _JOBS.append((reg, b0, T, maps, mask, bnd, loops))
        n_part += int(maps["hicgan"] is not None and (mask & ~np.isfinite(maps["hicgan"])).any())
    log(f"akita: {len(_JOBS)} of {len(regs)} regions assembled; Hi-cGAN partially missing in {n_part}")
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(threads) as pool:
        res = pool.map(_akita_worker, range(len(_JOBS)), chunksize=4)
    log(f"akita: scored in {time.time() - t0:.0f} s")
    rows = [r for rs, _ in res for r in rs]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "akita413_per_region.csv.gz", index=False, float_format="%.5f")
    lcs = [(_JOBS[i][0], l) for i, (_, l) in enumerate(res)]
    akita_tables(df, lcs, boot, seed)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
REGION_METRICS = ["pearson", "scc", "ins_r_10k", "ins_r_25k", "ins_r_50k", "ins_r_100k", "ins_r_250k",
                  "bnd_f1", "bnd_f1_strict", "bnd_recall", "bnd_precision", "bnd_strength_r", "moc",
                  "stripe_r", "loop_enrich_corr", "compartment_corr", "eigenvector_abs_corr",
                  "bnd_f1_struct"]


def akita_tables(df, lcs, boot, seed):
    import benchmark_blocks as bb
    from flexcible import bioscore as bs
    prm = bb.params(2048)["loop"]
    rng = np.random.default_rng(seed)
    regions = sorted(df.region.unique())
    R = len(regions)
    ridx = {r: i for i, r in enumerate(regions)}
    methods = [m for m in AK_METHODS if m in set(df.method)]
    # --- per-region distributions
    srows = []
    for m in methods:
        d = df[df.method == m]
        rec = {"method": m, "label": LABEL[m], "n_regions": int(d.pearson.notna().sum())}
        for k in REGION_METRICS:
            v = d[k].dropna().values
            if v.size:
                rec[f"{k}_median"] = float(np.median(v))
                rec[f"{k}_q25"], rec[f"{k}_q75"] = float(np.quantile(v, .25)), float(np.quantile(v, .75))
                rec[f"{k}_mean"] = float(v.mean())
                rec[f"{k}_n"] = int(v.size)
        srows.append(rec)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "akita413_per_region_summary.csv", index=False, float_format="%.4f")

    # --- pooled (count) metrics with region bootstrap
    cnt = {}
    for m in methods:
        d = df[df.method == m].set_index("region").reindex(regions)
        cnt[m] = {k: d[k].fillna(0).values.astype(float) for k in
                  ("bnd_rec_hits", "bnd_ref_n", "bnd_prec_hits", "bnd_pred_n", "hft_hits", "hft_n",
                   "loop_hits", "loop_n")}
        cnt[m]["present"] = d.pearson.notna().values
    lcd = {m: [None] * R for m in methods}
    for reg, l in lcs:
        for m, lc in l.items():
            lcd[m][ridx[reg]] = lc
    w = 2 * prm.apa_half + 1
    stk = {}
    for m in methods:
        t = np.zeros((R, w, w)); p = np.zeros((R, w, w)); na = np.zeros(R)
        anc, con = [None] * R, [None] * R
        for i, lc in enumerate(lcd[m]):
            if lc is not None and lc["apa_n"]:
                t[i], p[i], na[i] = lc["apa_t"], lc["apa_p"], lc["apa_n"]
            if lc is not None:
                anc[i], con[i] = lc["anchor"], lc["control"]
        stk[m] = (t, p, na, anc, con)

    def pooled(m, idx):
        c = cnt[m]
        pr = c["present"][idx]
        idx = idx[pr]
        o = {}
        rh, rn = c["bnd_rec_hits"][idx].sum(), c["bnd_ref_n"][idx].sum()
        ph, pn = c["bnd_prec_hits"][idx].sum(), c["bnd_pred_n"][idx].sum()
        o["bnd_recall_pooled"] = rh / rn if rn else np.nan
        o["bnd_precision_pooled"] = ph / pn if pn else np.nan
        r_, p_ = o["bnd_recall_pooled"], o["bnd_precision_pooled"]
        o["bnd_f1_pooled"] = 2 * r_ * p_ / (r_ + p_) if r_ + p_ > 0 else np.nan
        o["hft_bnd_recall"] = c["hft_hits"][idx].sum() / max(1, c["hft_n"][idx].sum())
        o["loop_recall"] = c["loop_hits"][idx].sum() / max(1, c["loop_n"][idx].sum())
        t, p, na, _, _ = stk[m]
        nat = na[idx].sum()
        if nat:
            at = bs.apa_score(t[idx].sum(0), nat, prm.apa_corner)
            ap = bs.apa_score(p[idx].sum(0), nat, prm.apa_corner)
            o["apa_ratio"] = (ap - 1) / (at - 1) if at > 1 else np.nan
        else:
            o["apa_ratio"] = np.nan
        return o

    full = np.arange(R)
    point = {m: pooled(m, full) for m in methods}
    draws = rng.integers(0, R, size=(boot, R))
    bs_stats = {m: pd.DataFrame([pooled(m, d) for d in draws]) for m in methods}
    # anchor AUROC (not bootstrapped: pooled over regions)
    for m in methods:
        anc = np.concatenate([a for a in stk[m][3] if a is not None and a.size])
        con = np.concatenate([c for c in stk[m][4] if c is not None and c.size])
        anc, con = anc[np.isfinite(anc)], con[np.isfinite(con)]
        point[m]["anchor_auroc"] = bs._auroc(anc, con) if anc.size > 5 and con.size > 5 else np.nan
    prow = []
    for m in methods:
        rec = {"method": m, "label": LABEL[m], "n_regions": int(cnt[m]["present"].sum()),
               "n_ref_boundaries_own": int(cnt[m]["bnd_ref_n"][cnt[m]["present"]].sum()),
               "n_ref_hft_boundaries": int(cnt[m]["hft_n"][cnt[m]["present"]].sum()),
               "n_ref_loops": int(cnt[m]["loop_n"][cnt[m]["present"]].sum())}
        for k, v in point[m].items():
            rec[k] = v
            if k in bs_stats[m]:
                rec[f"{k}_lo"], rec[f"{k}_hi"] = np.nanpercentile(bs_stats[m][k], [2.5, 97.5])
        prow.append(rec)
    P = pd.DataFrame(prow)
    P.to_csv(OUT / "akita413_pooled.csv", index=False, float_format="%.4f")

    # --- paired comparisons: ours (and others) against Akita and the other references
    from scipy.stats import wilcoxon
    pairs = []
    pooled_keys = ["bnd_f1_pooled", "bnd_recall_pooled", "bnd_precision_pooled", "hft_bnd_recall",
                   "loop_recall", "apa_ratio"]
    wide = {k: df.pivot(index="region", columns="method", values=k).reindex(regions)
            for k in REGION_METRICS + ["hft_hits", "hft_n", "loop_hits", "loop_n"]}
    for o in OURS:
        for other in [m for m in methods if m != o]:
            for k in REGION_METRICS:
                x, y = wide[k][o].values, wide[k][other].values
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 5:
                    continue
                xd, yd = x[ok], y[ok]
                d = xd - yd
                bm_ = np.array([np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(boot)])
                wins, ties = int((xd > yd).sum()), int((xd == yd).sum())
                nn = d.size - ties
                wf = np.array([(d[(j := rng.integers(0, d.size, d.size))] > 0).sum() /
                               max(1, (d[j] != 0).sum()) for _ in range(boot)])
                try:
                    pw = float(wilcoxon(xd, yd).pvalue)
                except ValueError:
                    pw = np.nan
                pairs.append({"ours": o, "other": other, "metric": k, "kind": "per-region", "n": int(d.size),
                              "median_ours": float(np.median(xd)), "median_other": float(np.median(yd)),
                              "median_diff": float(np.median(d)),
                              "diff_lo": float(np.percentile(bm_, 2.5)), "diff_hi": float(np.percentile(bm_, 97.5)),
                              "wins": wins, "losses": int((xd < yd).sum()), "ties": ties,
                              "win_frac": wins / nn if nn else np.nan,
                              "win_frac_lo": float(np.percentile(wf, 2.5)), "win_frac_hi": float(np.percentile(wf, 97.5)),
                              "wilcoxon_p": pw})
            for k in pooled_keys:
                diff = bs_stats[o][k].values - bs_stats[other][k].values
                rec = {"ours": o, "other": other, "metric": k, "kind": "pooled",
                       "n": int(min(cnt[o]["present"].sum(), cnt[other]["present"].sum())),
                       "median_ours": point[o][k], "median_other": point[other][k],
                       "median_diff": point[o][k] - point[other][k],
                       "diff_lo": float(np.nanpercentile(diff, 2.5)), "diff_hi": float(np.nanpercentile(diff, 97.5))}
                pairs.append(rec)
    # per-region win counts of the rank-matched recalls (regions with a reference)
    for o in OURS:
        for other in [m for m in methods if m != o]:
            for k, (hn, dn) in (("hft_bnd_recall", ("hft_hits", "hft_n")), ("loop_recall", ("loop_hits", "loop_n"))):
                n_ = wide[dn][o].values
                x = wide[hn][o].values / np.where(n_ > 0, n_, np.nan)
                y = wide[hn][other].values / np.where(n_ > 0, n_, np.nan)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 5:
                    continue
                d = x[ok] - y[ok]
                nn = int((d != 0).sum())
                pairs.append({"ours": o, "other": other, "metric": k + "_per_region", "kind": "per-region",
                              "n": int(ok.sum()), "median_ours": float(np.median(x[ok])),
                              "median_other": float(np.median(y[ok])), "median_diff": float(np.median(d)),
                              "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                              "win_frac": (d > 0).sum() / nn if nn else np.nan})
    Pr = pd.DataFrame(pairs)
    Pr.to_csv(OUT / "akita413_paired.csv", index=False, float_format="%.4g")
    # ours-with-consensus against ours-ChIP-only is part of the grid above
    log("akita: tables written")


# --------------------------------------------------------------------------- #
# stage aml
# --------------------------------------------------------------------------- #
AML_PATIENTS = ["018", "027", "168", "270", "472", "546", "629", "773"]       # C.Origami patients
AML_FOLD = {"018": 10, "027": 10, "270": 10, "168": 11, "546": 11, "629": 11, "1360": 11,
            "472": 12, "773": 12, "1916": 12}
AML_SRC = {"c12_refin": ("npz", "ours/c12_refin@patient__xu_{p}/marks.npz"),
           "c12_refres": ("npz", "ours/c12_refres@patient__xu_{p}/marks.npz"),
           "corigami_q": ("own", "corigami/q@patient__xu_{p}.npz"),
           "epiphany_mean": ("own", "epiphany/mean@patient__xu_{p}.npz"),
           "epiphany_zero": ("own", "epiphany/zero@patient__xu_{p}.npz"),
           "akita": ("own", "akita/akita.npz")}
AML_LABEL = {"c12_refin": "ours, 12 cells, reference input (no AML)", "c12_refres": "ours, 12 cells, residual on consensus12 (no AML)",
             "consensus12": "consensus12 (12 cell lines, measured)", "corigami_q": "C.Origami (CTCF, ATAC quantile-matched)",
             "epiphany_mean": "Epiphany (H3K4me3 = mean)", "epiphany_zero": "Epiphany (H3K4me3 = 0)",
             "akita": "Akita (GM12878 head)"}


def _aml_worker(task):
    p, win, T, maps, mask = task
    Tm = symmetrize(T, mask)
    rows = []
    for m, P in maps.items():
        r, _ = region_scores(Tm, symmetrize(P, mask), mask, 10000, hft=False)
        r.update(patient=p, window=win, method=m, n_pixels=int(mask.sum()))
        rows.append(r)
    return rows


def stage_aml(threads):
    import cooler
    import multicell_expanded as mx
    from flexcible.genome import match_name
    global _JOBS
    c12 = cooler.Cooler(str(mx.consensus12(10000)))
    all_rows = []
    for p in AML_PATIENTS:
        files = {m: AML_COMP / rel.format(p=p) for m, (_, rel) in AML_SRC.items()}
        if not all(f.exists() for f in files.values()):
            log(f"aml {p}: missing sources {[m for m, f in files.items() if not f.exists()]}")
            continue
        zs = {m: np.load(f) for m, f in files.items()}
        keys = set(zs["c12_refin"].files)
        for m in zs:
            keys &= set(zs[m].files)
        clr = cooler.Cooler(str(HM / f"aml/oe/xu_{p}_proper_10kb.cool"))
        tasks = []
        for win in sorted(keys):
            chrom, se = win.split(":")
            s, e = map(int, se.split("-"))
            try:
                T = np.asarray(clr.matrix(balance=False).fetch(f"{match_name(chrom, list(clr.chromnames))}:{s}-{e}"), float)
                C = np.asarray(c12.matrix(balance=False).fetch(f"{match_name(chrom, list(c12.chromnames))}:{s}-{e}"), float)
            except Exception:                                  # noqa: BLE001
                continue
            if T.shape != (192, 192) or C.shape != (192, 192):
                continue
            T[T == 0] = np.nan
            C[C == 0] = np.nan
            maps = {"consensus12": C}
            for m, (kind, _) in AML_SRC.items():
                A = zs[m][win].astype(float)
                if kind == "npz":
                    live = np.abs(np.nan_to_num(A)).sum(1) > 0
                    A = np.where(live[:, None] & live[None, :], A, np.nan)
                maps[m] = A
            mask = np.triu(np.ones((192, 192), bool), 2) & np.isfinite(T)
            for P in maps.values():
                mask &= np.isfinite(P)
            if mask.sum() < 500:
                continue
            tasks.append((p, win, T.astype(np.float32), {k: v.astype(np.float32) for k, v in maps.items()}, mask))
        log(f"aml {p}: {len(tasks)} of {len(keys)} windows with every method")
        _JOBS = tasks
        ctx = mp.get_context("fork")
        t0 = time.time()
        with ctx.Pool(threads) as pool:
            res = pool.map(_aml_worker_idx, range(len(tasks)), chunksize=16)
        all_rows += [r for rs in res for r in rs]
        log(f"aml {p}: scored in {time.time() - t0:.0f} s")
        _JOBS = []
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "aml_windows_per_window.csv.gz", index=False, float_format="%.5f")
    aml_tables(df)
    aml_eval_tables()


def _aml_worker_idx(i):
    return _aml_worker(_JOBS[i])


AML_METRICS = ["pearson", "scc", "ins_r_50k", "ins_r_100k", "ins_r_250k", "bnd_f1", "bnd_f1_strict",
               "bnd_strength_r", "moc", "stripe_r", "loop_enrich_corr", "compartment_corr",
               "eigenvector_abs_corr", "bnd_f1_struct"]


def aml_tables(df):
    # per-patient medians, then median / range over patients
    pm = df.groupby(["patient", "method"])[AML_METRICS].median().reset_index()
    nw = df.groupby(["patient", "method"]).size().rename("n_windows").reset_index()
    pm = pm.merge(nw, on=["patient", "method"])
    pm.to_csv(OUT / "aml_windows_per_patient.csv", index=False, float_format="%.4f")
    rows = []
    for m, g in pm.groupby("method"):
        rec = {"method": m, "label": AML_LABEL.get(m, m), "n_patients": int(g.patient.nunique()),
               "n_windows": int(g.n_windows.sum())}
        for k in AML_METRICS:
            rec[f"{k}_median"] = float(g[k].median())
            rec[f"{k}_min"], rec[f"{k}_max"] = float(g[k].min()), float(g[k].max())
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "aml_windows_summary.csv", index=False, float_format="%.4f")
    prow = []
    wide = {k: pm.pivot(index="patient", columns="method", values=k) for k in AML_METRICS}
    ww = {k: df.pivot_table(index=["patient", "window"], columns="method", values=k) for k in AML_METRICS}
    for o in ("c12_refin", "c12_refres"):
        for other in ("akita", "corigami_q", "epiphany_mean", "epiphany_zero", "consensus12"):
            for k in AML_METRICS:
                d = (wide[k][o] - wide[k][other]).dropna()
                wd = (ww[k][o] - ww[k][other]).dropna()
                prow.append({"ours": o, "other": other, "metric": k, "n_patients": int(d.size),
                             "patient_median_diff": float(d.median()), "patients_ours_ahead": int((d > 0).sum()),
                             "n_windows": int(wd.size), "window_median_diff": float(wd.median()),
                             "window_win_frac": float((wd > 0).sum() / max(1, (wd != 0).sum()))})
    pd.DataFrame(prow).to_csv(OUT / "aml_windows_paired.csv", index=False, float_format="%.4f")


def aml_eval_tables():
    import glob
    fs = sorted(glob.glob(str(AML_REMOTE / "fold1[012]/eval_f1[012]_patients.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[(d.group == "patient")]
    cols = ["pearson", "scc", "insulation_corr", "boundary_f1", "loop_enrich_corr", "compartment_corr",
            "eigenvector_abs_corr"]
    d.to_csv(OUT / "aml_eval_patients_rows.csv", index=False, float_format="%.4f")
    rows = []
    for (res, model, cond), g in d.groupby(["res_kb", "model", "condition"]):
        pat = g.groupby("cell")[cols].mean()
        rec = {"res_kb": res, "model": model, "condition": cond, "n_patients": int(pat.shape[0]),
               "n_regions_per_patient_median": float(g.n_regions.median())}
        for k in cols:
            rec[f"{k}_median"] = float(pat[k].median())
            rec[f"{k}_min"], rec[f"{k}_max"] = float(pat[k].min()), float(pat[k].max())
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "aml_eval_summary.csv", index=False, float_format="%.4f")


# --------------------------------------------------------------------------- #
# stage chrho
# --------------------------------------------------------------------------- #
CH_OURS = ["all12refin_drop_chrho", "all12refres_drop_chrho", "all12honest_drop_chrho",
           "all12one_fullrd_chrho", "all12one_fullrd_chrho_noref"]       # last two: release twin (registry 101)
CH_SKIP = {"all12one_chrho", "all12one_chrho_noref"}                      # in the per-block tables since 91, not in this table
CH_REF = ["consensus", "best_other_map", "cascfold", "replicate"]
CH_COLS = ["oe_pearson", "oe_scc", "insulation_r", "bnd_recall", "bnd_precision", "TAD Fraction",
           "TAD Score MSE", "loop_recall_rank", "apa_ratio", "loop_f1_call", "comp_pref_r"]


def stage_chrho():
    frames = []
    for f in ("chrho_per_block.csv", "chrho_25kb_per_block.csv"):
        d = pd.read_csv(REPO / "results/hic_measures_chrho" / f)
        frames.append(d)
    D = pd.concat(frames, ignore_index=True).drop_duplicates(["block", "source"])
    D = D[D.set.isin(["chrho_corigami", "chrho_epiphany", "chrho_hicgan"]) & ~D.source.isin(CH_SKIP)]
    D["bnd_f1"] = 2 * D.bnd_recall * D.bnd_precision / (D.bnd_recall + D.bnd_precision)
    cols = CH_COLS + ["bnd_f1"]
    g = D.groupby(["set", "cell", "res", "source"])
    M = g[cols].mean().reset_index()
    M["n_chromosomes"] = g.chrom.nunique().values
    M["chromosomes"] = g.chrom.apply(lambda s: ",".join(sorted(s.unique(), key=lambda c: int(c[3:])))).values
    M.to_csv(OUT / "chrho_summary_structure.csv", index=False, float_format="%.4f")
    # chromosome-wise ahead counts of ours against each competitor
    rows = []
    comps = D[D.kind == "competitor"] if "kind" in D else D
    for (st_, cell, res), g2 in D.groupby(["set", "cell", "res"]):
        comp = sorted(set(g2[g2.kind == "competitor"].source))
        for c in comp:
            for o in CH_OURS:
                a = g2[g2.source == o].set_index("chrom")
                b = g2[g2.source == c].set_index("chrom")
                ch = sorted(set(a.index) & set(b.index))
                if not ch:
                    continue
                rec = {"set": st_, "cell": cell, "res": res, "competitor": c, "ours": o,
                       "chromosomes": ",".join(ch)}
                for k in cols:
                    dd = (a.loc[ch, k] - b.loc[ch, k]).dropna()
                    if k == "TAD Score MSE":
                        dd = -dd                               # lower is better
                    rec[f"{k}_diff_mean"] = float(dd.mean()) if dd.size else np.nan
                    rec[f"{k}_ahead"] = f"{int((dd > 0).sum())}/{int(dd.size)}"
                rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "chrho_vs_competitor_structure.csv", index=False, float_format="%.4f")
    log("chrho: tables written")


# --------------------------------------------------------------------------- #
# stage report
# --------------------------------------------------------------------------- #
def f3(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


NOTES = """## Definitions

Akita table (GM12878, 413 Akita test regions, 2048 bp bins, 448-bin crop = 917 kb, Akita's target space of clipped, smoothed log O/E). One mask per region: upper triangle, at least 2 bins off the diagonal, finite in the target and in every method except Hi-cGAN. Hi-cGAN has no prediction (or a partial one) in 5 regions and is scored in 408.
- Pearson, SCC: Pearson over the mask pixels; HiCRep stratum-adjusted correlation.
- ins r w: Pearson over bins between the insulation profiles of prediction and target; insulation = mean log O/E in the square [i-w, i) x (i, i+w]; w = 5, 12, 25, 50, 122 bins = 10, 25, 51, 102, 250 kb.
- bnd F1, recall, precision: boundaries are local minima of the 51 kb insulation profile whose prominence over the maxima of the 51 kb flanks is at least 0.5 SD of that profile (each map its own SD); minima within the tolerance merged. Reference = boundaries of the measured target on the same pixels. Tolerance 5 bins (10 kb); strict 2 bins (4 kb). Per-region F1 (median) and pooled micro-averaged F1 over all regions.
- bnd strength r: Pearson over bins between the boundary-strength profiles (prominence of the insulation minimum over its flanks) of prediction and target.
- MoC: measure of concordance (Zufferey et al. 2018) between the domain partitions cut at the boundaries above.
- hFT bnd recall: pooled recall of the hicFindTADs boundaries of the measured count map (10 kb) by the K strongest insulation minima of the map, K = number of reference boundaries in the region, tolerance 5 bins; K-matched, so recall equals precision.
- loop recall, APA ratio: hicDetectLoops loops of the measured count map as reference; K strongest donut enrichments of the map (K-matched); APA ratio = (APA of prediction - 1) / (APA of target - 1) at the reference anchors. Pooled over regions with a reference loop.
- stripe r: Pearson over bins between the anchor profiles (mean log O/E of the pixels 20 to 200 kb from the bin) of prediction and target. A proxy; there is no stripe caller in the benchmark.
- compart. r, eigenvector |r|, loop enrichment r: flexcible/structure.py (correlation of the within-window correlation matrices; first eigenvector; donut enrichment). Within 917 kb these are local measures: the compartment values are a checkerboard proxy, the 2 kb donut (4 to 10 kb) is smaller than a loop.
- Paired columns: ours minus Akita per region; interval = 2.5 and 97.5 percentiles of 2000 region bootstrap resamples; ahead = regions with ours strictly higher; ties excluded from the fraction.
Chromosome table: values of results/hic_measures_chrho (scripts/hic_measures.py), mean over the held-out chromosomes listed, 10 or 25 kb, own band per block (C.Origami 2 Mb at 25 kb and 1.92 Mb at 10 kb, Epiphany 990 kb, Hi-cGAN 2 Mb). Boundaries are hicFindTADs boundaries of the prediction against those of the measured map (1 bin tolerance), the insulation square is 50 kb (10 kb) or 75 kb (25 kb), loop recall is K-matched against hicDetectLoops of the measured map. These definitions differ from the Akita table (Akita has no whole-chromosome prediction) and the two tables are not comparable in value.
AML windows: 10 kb, 192 bins, windows centred on AML-specific loops and stripes of 8 donors (C.Origami patients), saved predictions of scripts/aml_competitors_loops.py, measured patient map as target; same definitions as the Akita table at 10 kb (insulation squares 50, 100 and 250 kb; boundary tolerance 1 bin; no external reference calls).

## Caveats

- Akita is sequence only, one head per training cell type (GM12878 head here); it sees no cell-specific inputs and is restricted to its five cell types. Our ChIP-only model receives the cell's own ChIP-seq tracks; our with-consensus model additionally receives the Hi-C of other cells (consensus of four measured cells) as input or residual base. The consensus and the HFF map are measured maps of other cells, not predictions.
- Region windows are Akita's test regions, held out of all our trainings (section 18). Hi-cGAN values are the predictions of the submitted model rasterised into Akita's space (akita_testset_rasters.npz), as in section 18.
- Pooled recall and APA are over regions with a reference object; K-matched recall equals precision.
- Chromosome blocks contain 2 or 3 chromosomes: ahead counts are descriptive, there is no interval. Held-out chromosomes per competitor are those of benchmark_competitors.COMPETITORS (C.Origami chr2, chr3; Epiphany chr3, 11, 17; Hi-cGAN even chromosomes), the models of ours are the chromosome-held-out drop_chrho models of section 37.
- AML windows are selected around AML-specific objects, not random windows; there are 8 donors, and Akita is the same prediction for every donor.
"""


def _ci(q, nd=3):
    return f"{f3(q.median_diff, nd)} [{f3(q.diff_lo, nd)}, {f3(q.diff_hi, nd)}]"


def stage_report():
    S = pd.read_csv(OUT / "akita413_per_region_summary.csv").set_index("method")
    P = pd.read_csv(OUT / "akita413_pooled.csv").set_index("method")
    Pr = pd.read_csv(OUT / "akita413_paired.csv")
    order = [m for m in ["akita", "akita_native_honest", "akita_native_refres", "consensus4", "map_hff", "hicgan"]
             if m in S.index]
    md = []
    # ---- table 1: per-region medians and pooled counts
    md.append("### Structure scores vs Akita, GM12878, Akita test regions (median over regions; pooled values with region-bootstrap 95% interval)\n")
    cols = [("pearson", "Pearson"), ("scc", "SCC"), ("ins_r_10k", "ins r 10 kb"), ("ins_r_50k", "ins r 51 kb"),
            ("ins_r_100k", "ins r 102 kb"), ("ins_r_250k", "ins r 250 kb"), ("bnd_f1", "bnd F1"),
            ("bnd_strength_r", "bnd strength r"), ("moc", "MoC"), ("stripe_r", "stripe r"),
            ("compartment_corr", "compart. r")]
    md.append("| predictor | regions | " + " | ".join(c for _, c in cols) +
              " | pooled bnd F1 | hFT bnd recall | loop recall | APA ratio |")
    md.append("|---|---|" + "---|" * (len(cols) + 4))
    for m in order:
        r, p = S.loc[m], P.loc[m]
        cells = [f3(r[f"{k}_median"]) for k, _ in cols]
        ci = lambda k: f"{f3(p[k])} [{f3(p[k + '_lo'])}, {f3(p[k + '_hi'])}]"       # noqa: E731
        md.append(f"| {LABEL[m]} | {int(r.n_regions)} | " + " | ".join(cells) +
                  f" | {ci('bnd_f1_pooled')} | {ci('hft_bnd_recall')} | {ci('loop_recall')} | {ci('apa_ratio')} |")
    md.append("")
    # ---- table 1b: paired against Akita
    md.append("### Paired against Akita on the same regions and pixels (difference ours minus Akita, median over regions [region-bootstrap 95% interval]; regions ours is ahead / regions compared)\n")
    md.append("| metric | ours, ChIP only: difference | ahead (fraction [95%]) | ours, with consensus: difference | ahead (fraction [95%]) |")
    md.append("|---|---|---|---|---|")
    names = dict(cols)
    names.update({"bnd_f1_strict": "bnd F1 strict (4 kb)", "bnd_recall": "bnd recall", "bnd_precision": "bnd precision",
                  "loop_enrich_corr": "loop enrichment r", "eigenvector_abs_corr": "eigenvector |r|",
                  "ins_r_25k": "ins r 25 kb", "hft_bnd_recall_per_region": "hFT bnd recall (per region)",
                  "loop_recall_per_region": "loop recall (per region)"})
    for k in ["pearson", "scc", "ins_r_10k", "ins_r_25k", "ins_r_50k", "ins_r_100k", "ins_r_250k", "bnd_f1",
              "bnd_f1_strict", "bnd_recall", "bnd_precision", "bnd_strength_r", "moc", "stripe_r",
              "compartment_corr", "eigenvector_abs_corr", "loop_enrich_corr",
              "hft_bnd_recall_per_region", "loop_recall_per_region"]:
        row = [names.get(k, k)]
        for o in OURS:
            q = Pr[(Pr.ours == o) & (Pr.other == "akita") & (Pr.metric == k)]
            if q.empty:
                row += ["", ""]
                continue
            q = q.iloc[0]
            lo = "" if pd.isna(q.get("win_frac_lo")) else f" [{f3(q.win_frac_lo, 2)}, {f3(q.win_frac_hi, 2)}]"
            dif = f3(q.median_diff) if pd.isna(q.get("diff_lo")) else _ci(q)
            row += [dif, f"{int(q.wins)}/{int(q.n)} ({f3(q.win_frac, 2)}{lo})"]
        md.append("| " + " | ".join(row) + " |")
    for k, nm in (("bnd_f1_pooled", "pooled bnd F1"), ("hft_bnd_recall", "pooled hFT bnd recall"),
                  ("loop_recall", "pooled loop recall"), ("apa_ratio", "APA ratio")):
        row = [nm]
        for o in OURS:
            q = Pr[(Pr.ours == o) & (Pr.other == "akita") & (Pr.metric == k)]
            row += [_ci(q.iloc[0]) if len(q) else "", "pooled"]
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    # ---- table 2: chromosome level
    C = pd.read_csv(OUT / "chrho_summary_structure.csv")
    A = pd.read_csv(OUT / "chrho_vs_competitor_structure.csv")
    ccols = [("oe_pearson", "Pearson"), ("oe_scc", "SCC"), ("insulation_r", "ins r"), ("bnd_recall", "bnd recall"),
             ("bnd_precision", "bnd prec."), ("bnd_f1", "bnd F1"), ("TAD Fraction", "TAD fraction"),
             ("TAD Score MSE", "TAD score MSE"), ("loop_recall_rank", "loop recall"), ("apa_ratio", "APA ratio"),
             ("comp_pref_r", "compart. r")]
    short = {"all12refres_drop_chrho": "ours, consensus as residual (refres)", "all12refin_drop_chrho": "ours, consensus as input (refin)",
             "all12honest_drop_chrho": "ours, ChIP only (honest)",
             "all12one_fullrd_chrho": "release twin, consensus input", "all12one_fullrd_chrho_noref": "release twin, ChIP only",
             "consensus": "consensus of other cells",
             "measured_balanced": "measured map, balanced (ceiling)"}
    setname = {"chrho_corigami": "C.Origami", "chrho_epiphany": "Epiphany", "chrho_hicgan": "Hi-cGAN"}
    for res in (10000, 25000):
        md.append(f"### Whole held-out chromosomes at {res // 1000} kb (mean over chromosomes; hicFindTADs boundaries, 1 bin tolerance)\n")
        md.append("| block | predictor | chr | " + " | ".join(c for _, c in ccols) + " | chromosomes ahead of the competitor: ins r / bnd F1 / loop recall / compart. r |")
        md.append("|---|---|---|" + "---|" * (len(ccols) + 1))
        x = C[C.res == res]
        for (st_, cell), g in x.groupby(["set", "cell"]):
            comp = [u for u in g.source if u not in short and u not in ("best_other_map", "cascfold", "replicate")
                    and not u.startswith("hicgan_imr90") and not u.startswith("hicgan_gm12878_on_imr90")]
            rows = [u for u in ["all12refres_drop_chrho", "all12refin_drop_chrho", "all12honest_drop_chrho",
                                "all12one_fullrd_chrho", "all12one_fullrd_chrho_noref", "consensus"] + comp + ["measured_balanced"] if u in set(g.source)]
            for u in rows:
                r = g[g.source == u].iloc[0]
                lab = short.get(u, u + " (competitor)")
                ah = ""
                if u in short and u.startswith("all12") and comp:
                    a = A[(A.set == st_) & (A.cell == cell) & (A.res == res) & (A.competitor == comp[0]) & (A.ours == u)]
                    if len(a):
                        a = a.iloc[0]
                        ah = " / ".join(a[f"{k}_ahead"] for k in ("insulation_r", "bnd_f1", "loop_recall_rank", "comp_pref_r"))
                md.append(f"| {setname[st_]} {cell} | {lab} | {r.chromosomes} | " +
                          " | ".join(f3(r[k]) for k, _ in ccols) + f" | {ah} |")
        md.append("")
    # ---- table 3: AML windows
    W = pd.read_csv(OUT / "aml_windows_summary.csv").set_index("method")
    md.append("### AML patients, 10 kb object windows of 8 donors (median over the 8 donors of the per-donor window median; identical pixels in every window)\n")
    wc = [("pearson", "Pearson"), ("scc", "SCC"), ("ins_r_50k", "ins r 50 kb"), ("ins_r_250k", "ins r 250 kb"),
          ("bnd_f1", "bnd F1"), ("bnd_strength_r", "bnd strength r"), ("moc", "MoC"), ("stripe_r", "stripe r"),
          ("compartment_corr", "compart. r")]
    md.append("| predictor | windows | " + " | ".join(c for _, c in wc) + " |")
    md.append("|---|---|" + "---|" * len(wc))
    for m in ["c12_refres", "c12_refin", "consensus12", "akita", "corigami_q", "epiphany_mean", "epiphany_zero"]:
        if m in W.index:
            r = W.loc[m]
            md.append(f"| {AML_LABEL[m]} | {int(r.n_windows)} | " + " | ".join(f3(r[f"{k}_median"]) for k, _ in wc) + " |")
    md.append("")
    E = pd.read_csv(OUT / "aml_eval_summary.csv")
    md.append("### AML patients, our AML-inclusive and no-AML models against the measured patient map, 413 held-out regions, 10 donors (folds 10 to 12; median over donors [min, max])\n")
    md.append("| kb | model | marks | Pearson | SCC | insulation r | boundary F1 (structure.py) | loop enrichment r | compart. r |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in E[E.condition.isin(["marks"])].sort_values(["res_kb", "model"]).iterrows():
        md.append(f"| {int(r.res_kb)} | {r.model} | marks | {f3(r.pearson_median)} | {f3(r.scc_median)} | "
                  f"{f3(r.insulation_corr_median)} [{f3(r.insulation_corr_min)}, {f3(r.insulation_corr_max)}] | "
                  f"{f3(r.boundary_f1_median)} [{f3(r.boundary_f1_min)}, {f3(r.boundary_f1_max)}] | "
                  f"{f3(r.loop_enrich_corr_median)} | {f3(r.compartment_corr_median)} |")
    (OUT / "structure_scores_vs_akita.md").write_text("\n".join(md) + "\n\n" + NOTES)
    log("report: structure_scores_vs_akita.md written")


def cols_all():
    return REGION_METRICS + ["bnd_f1_pooled", "hft_bnd_recall", "loop_recall", "apa_ratio"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["akita", "aml", "chrho", "report"],
                    choices=["akita", "aml", "chrho", "report"])
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="akita stage: first N regions (smoke test)")
    ap.add_argument("--out", default="", help="output directory (default results/structure_vs_akita)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    os.nice(10)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    if "akita" in a.stage:
        stage_akita(min(a.threads, 6), a.boot, a.seed, a.limit)
    if "aml" in a.stage:
        stage_aml(min(a.threads, 6))
    if "chrho" in a.stage:
        stage_chrho()
    if "report" in a.stage:
        stage_report()


if __name__ == "__main__":
    main()
