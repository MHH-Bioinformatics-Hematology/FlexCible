"""Part 3: can a prediction be MORE right than a shallow measurement?

The claim to test is "where the experiment is unreliable, scoring a model
against it punishes the model for being right". Asserting that is worthless;
this makes it falsifiable.

Design
------
Take the deep measured GM12878 map (4DNFIXP4QG5B, 4.0e9 contacts, 2 kb bins)
over exactly the Akita test spans. For a read fraction f, split the raw counts
of every region into two DISJOINT parts:

    S ~ Binomial(C, f)          the "shallow experiment"
    R = C - S                   the reference it is judged against

Disjointness is the whole point. If the shallow map were an f-subsample of the
same reads that make the reference, the two would share their Poisson noise
and the measurement would be credited for reproducing its own noise -- the
comparison would be rigged in favour of the measurement before it started.
Splitting makes S and R independent given the underlying contact frequency,
exactly as two separate experiments would be.

Two further things are done to keep the shallow map from getting unearned
credit:

* S and R are balanced INDEPENDENTLY, each from its own marginals (a local
  ICE). Using the deep cooler's shared balancing weights for both would print
  the same per-bin bias pattern into both maps, and the measurement would then
  correlate with the reference partly through that shared artefact.
* every candidate has its own per-diagonal expected removed, and the pixel
  correlation is computed after subtracting each map's per-diagonal mean, so
  distance decay -- which dominates any correlation over a contact map -- and
  the differing expected conventions cannot contribute.

Then, against the same reference R: does the prediction correlate with R
better than S does, and does it recover R's insulation profile and boundaries
better than S does?

What would refute the claim
---------------------------
* the prediction never beats S at any tested f -> refuted outright;
* it only wins below a depth nobody would ever sequence -> technically true,
  useless, and to be reported as such;
* it wins on the pixel correlation but not on the feature metrics -> it is
  winning by being smooth, not by being right, and the claim does not stand.

Mappability
-----------
The second angle: bins whose balancing weight in the deep cooler is NaN (the
balancing could not converge -- unmappable or repeat-collapsed sequence) and
bins in the lowest deep-coverage decile are flagged BEFORE anything is scored.
Every correlation is reported separately inside and outside that flag, so
"the model disagrees with the measurement where the measurement is bad" is a
measurement rather than a hope.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cooler                                                       # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome import bioscore as bs                              # noqa: E402
from hicmultiome.utils import get_logger                            # noqa: E402

DEEP = ("/mnt/processed/wolffjoa/restart_of_prediction_project/originalData/"
        "4Dnucleosome/GM12878_4DNFIXP4QG5B.mcool::/resolutions/2000")
AKITA_RES, AKITA_SIDE, AKITA_CROP = 2048, 448, 32
log = get_logger("bioscore-depth")


# --------------------------------------------------------------------------- #
def block_sum(C: np.ndarray, g: int) -> np.ndarray:
    """Sum g x g blocks. Exact for counts -- unlike interpolation."""
    if g <= 1:
        return C
    n = C.shape[0] // g * g
    return C[:n, :n].reshape(n // g, g, n // g, g).sum(axis=(1, 3))


def ice(C: np.ndarray, usable: np.ndarray, n_iter: int = 40) -> np.ndarray:
    """Region-local iterative correction, from this map's own marginals.

    Each depth gets its own bias vector, which is what an independent
    experiment would have. Sharing the deep map's weights would leak the deep
    map's per-bin structure into the shallow one.
    """
    M = np.array(C, dtype=np.float64, copy=True)
    M[~usable, :] = 0.0
    M[:, ~usable] = 0.0
    for _ in range(n_iter):
        s = M.sum(axis=1)
        good = usable & (s > 0)
        if good.sum() < 10:
            break
        s = s / s[good].mean()
        s[~good] = 1.0
        s[s <= 0] = 1.0
        M /= s[:, None]
        M /= s[None, :]
    out = M.astype(np.float64)
    out[~usable, :] = np.nan
    out[:, ~usable] = np.nan
    return out


def obs_over_expected(bal: np.ndarray) -> np.ndarray:
    """observed / expected, expected taken per diagonal from this map itself."""
    n = bal.shape[0]
    out = np.full((n, n), np.nan)
    for d in range(n):
        i = np.arange(n - d)
        v = bal[i, i + d]
        ok = np.isfinite(v)
        if ok.sum() < 3:
            continue
        m = v[ok].mean()
        if not (m > 0):
            continue
        r = np.full(v.shape, np.nan)
        r[ok] = v[ok] / m
        out[i, i + d] = r
        out[i + d, i] = r
    return out


def destripe(M: np.ndarray) -> np.ndarray:
    """Subtract each diagonal's mean. Distance decay must not score."""
    n = M.shape[0]
    out = np.array(M, dtype=float, copy=True)
    for d in range(n):
        i = np.arange(n - d)
        v = out[i, i + d]
        ok = np.isfinite(v)
        if ok.sum() < 3:
            continue
        v = v - v[ok].mean()
        out[i, i + d] = v
        out[i + d, i] = v
    return out


def _smooth_nan(M: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN rather than spreading it."""
    from scipy.ndimage import gaussian_filter

    ok = np.isfinite(M).astype(float)
    num = gaussian_filter(np.where(ok > 0, M, 0.0), sigma, mode="nearest")
    den = gaussian_filter(ok, sigma, mode="nearest")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)
    out[ok == 0] = np.nan
    return out


def strat_pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    ok = mask & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 200:
        return float("nan")
    x, y = a[ok], b[ok]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="runs/bioscore/our_preds.npz")
    ap.add_argument("--deep", default=DEEP)
    ap.add_argument("--rasters-dir", default="runs/bioscore")
    ap.add_argument("--fractions", default="0.3,0.1,0.03,0.01,0.003,0.001")
    ap.add_argument("--bin-factor", type=int, default=4,
                    help="sum g x g raw-count blocks first; 4 -> 8 kb bins")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-dist-kb", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lowconf-mode", choices=("global", "per-region"),
                    default="global",
                    help="how the low-coverage decile is cut. 'global' is the "
                         "right one: it lets regions differ in how reliable "
                         "their measurement is. 'per-region' flags ~10%% of "
                         "every region by construction and is kept only so "
                         "the two definitions can be compared.")
    ap.add_argument("--out", default="runs/bioscore/depth")
    a = ap.parse_args()

    fracs = [float(x) for x in a.fractions.split(",")]
    P = np.load(a.preds, allow_pickle=True)
    regions = [str(r) for r in P["regions"]]
    ours_native = P["ours_native"]
    win_start = P["win_start"]
    our_res = int(P["resolution"])

    D = Path(a.rasters_dir)
    ak = np.load(D / "rast_akita.npy", mmap_mode="r")
    hg = np.load(D / "rast_hicgan.npy", mmap_mode="r")
    rreg = [str(r) for r in np.load(D / "rast_regions.npy", allow_pickle=True)]
    side, off = (int(x) for x in np.load(D / "rast_meta.npy"))
    rindex = {r: i for i, r in enumerate(rreg)}

    clr = cooler.Cooler(a.deep)
    total_contacts = float(clr.info.get("sum", np.nan))
    bins = clr.bins()
    g = a.bin_factor
    coarse_res = clr.binsize * g
    ins_win = max(4, int(round(50000 / coarse_res)))
    btol = max(1, int(round(16000 / coarse_res)))
    max_d = int(a.max_dist_kb * 1000 / coarse_res)
    log.info("deep %.3g contacts; scoring at %d bp bins (raw counts summed "
             "%dx%d); insulation window %d bins, boundary tol %d bins",
             total_contacts, coarse_res, g, g, ins_win, btol)

    idx = list(range(0, len(regions), a.stride))
    if a.limit:
        idx = idx[: a.limit]
    log.info("%d regions", len(idx))

    span_bp = AKITA_SIDE * AKITA_RES

    # --- a-priori bin confidence, on ONE global scale ---------------------- #
    # The threshold has to be global. Taking the lowest-coverage decile within
    # each region would flag ~10% of the bins of every region by construction,
    # so regions could not differ in how reliable their measurement is -- which
    # is precisely the variable Part 3c needs. Coverage comes from the cooler's
    # VC weight (1/VC is proportional to the bin's genome-wide marginal), and
    # a NaN balancing weight marks a bin the balancing could not converge on:
    # unmappable or repeat-collapsed sequence.
    def region_bins(k):
        chrom, sp = regions[k].split(":")
        st = int(sp.split("-")[0]) + AKITA_CROP * AKITA_RES
        b0 = int(np.ceil(st / clr.binsize))
        nbb = int(span_bp // clr.binsize) // g * g
        ext = clr.extent(chrom)
        sl = slice(ext[0] + b0, ext[0] + b0 + nbb)
        return b0, nbb, bins["weight"][sl].values, bins["VC"][sl].values

    covs = []
    for k in (idx if a.lowconf_mode == "global" else []):
        _, _, wv, vc = region_bins(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(np.isfinite(vc) & (vc > 0), 1.0 / vc, np.nan)
        covs.append(cv[np.isfinite(wv)])
    allcov = np.concatenate(covs) if covs else np.zeros(0)
    cov_thr = float(np.percentile(allcov, 10)) if allcov.size else None
    log.info("low-confidence mode %s; threshold %s (from %d mappable bins)",
             a.lowconf_mode, "n/a" if cov_thr is None else f"{cov_thr:.4g}",
             allcov.size)
    rows = []
    rng = np.random.default_rng(a.seed)
    bp = bs.BioParams(insulation_window=ins_win, boundary_tol=btol)

    for count, k in enumerate(idx):
        name = regions[k]
        chrom, sp = name.split(":")
        st_bp = int(sp.split("-")[0]) + AKITA_CROP * AKITA_RES
        b0 = int(np.ceil(st_bp / clr.binsize))
        nb = int(span_bp // clr.binsize) // g * g
        lo_bp, hi_bp = b0 * clr.binsize, (b0 + nb) * clr.binsize
        try:
            C = np.nan_to_num(np.asarray(clr.matrix(balance=False).fetch(
                f"{chrom}:{lo_bp}-{hi_bp}"), dtype=np.float64))
            ext = clr.extent(chrom)
            sl = slice(ext[0] + b0, ext[0] + b0 + nb)
            w = bins["weight"][sl].values
            vc = bins["VC"][sl].values
        except Exception as exc:                       # pragma: no cover
            log.warning("%s: %s", name, exc)
            continue
        if C.shape[0] != nb:
            continue

        # --- a-priori low-confidence bins, from the experiment alone -------- #
        mappable = np.isfinite(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = np.where(np.isfinite(vc) & (vc > 0), 1.0 / vc, 0.0)
        thr = (cov_thr if cov_thr is not None else
               (np.percentile(cov[mappable], 10) if mappable.any() else 0.0))
        low_fine = (~mappable) | (cov <= thr)

        nc = nb // g
        low = low_fine.reshape(nc, g).any(axis=1)
        Cc = block_sum(C, g)
        # `usable` = has any data at all. Low-confidence bins stay IN, because
        # the whole question is how the candidates behave there; the flag
        # partitions the pixels, it does not discard them.
        usable = Cc.sum(axis=1) > 0
        if usable.sum() < 30:
            continue

        d = np.abs(np.subtract.outer(np.arange(nc), np.arange(nc)))
        band = (d >= 2) & (d <= max_d) & usable[:, None] & usable[None, :]
        low_pair = low[:, None] | low[None, :]
        pair_all = band
        pair_hi = band & ~low_pair
        pair_lo = band & low_pair

        # --- predictions on the same coarse grid --------------------------- #
        def to_coarse(log_map, src_start, src_res):
            fine = bs.resample_square(np.asarray(log_map, float), src_start,
                                      src_res, lo_bp, nb, clr.binsize)
            return bs.coarsen(np.exp(np.clip(fine, -8, 8)), g)

        cands = {"ours": to_coarse(ours_native[k], int(win_start[k]) * our_res,
                                   our_res)}
        if name in rindex:
            j = rindex[name]
            for lab, arr in (("akita", ak), ("hicgan", hg)):
                cands[lab] = to_coarse(
                    bs.dense_from_triu(np.asarray(arr[j]), side, off),
                    st_bp, AKITA_RES)

        for f in fracs:
            S = rng.binomial(Cc.astype(np.int64), f).astype(np.float64)
            R = Cc - S
            ref_oe = obs_over_expected(ice(R, usable))
            sh_oe = obs_over_expected(ice(S, usable))
            ref_t = destripe(np.log1p(ref_oe))
            # Steelman for the measurement: nobody analyses a shallow map raw.
            # Smoothing it is the cheapest thing an analyst does, and without
            # this control the prediction would only be beating a strawman.
            sh_sm = {f"shallow_smooth{sig}": _smooth_nan(sh_oe, sig)
                     for sig in (1.0, 2.0, 3.0, 4.0, 6.0)}
            base = {"region": name, "fraction": f,
                    "deep_contacts": float(Cc.sum()),
                    "shallow_contacts": float(S.sum()),
                    "frac_low_conf_bins": float(low.mean())}
            for lab, M in ([("shallow", sh_oe)] + list(sh_sm.items())
                           + list(cands.items())):
                Moe = np.where(usable[:, None] & usable[None, :], M, np.nan)
                rec = dict(base, model=lab)
                Mt = destripe(np.log1p(Moe))
                rec["pearson"] = strat_pearson(Mt, ref_t, pair_all)
                rec["pearson_highconf"] = strat_pearson(Mt, ref_t, pair_hi)
                rec["pearson_lowconf"] = strat_pearson(Mt, ref_t, pair_lo)
                rm = bs.region_metrics(np.log(np.maximum(ref_oe, 1e-3)),
                                       np.log(np.maximum(Moe, 1e-3)),
                                       bp, seed_offset=k)
                rec["insulation_pearson"] = rm.insulation_pearson
                rec["b_tp_true"], rec["b_tp_pred"] = rm.b_tp_true, rm.b_tp_pred
                rec["b_n_true"], rec["b_n_pred"] = rm.b_n_true, rm.b_n_pred
                rows.append(rec)
        if count % 20 == 0:
            log.info("%d/%d %s", count, len(idx), name)

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)

    summ = []
    for (f, lab), gg in df.groupby(["fraction", "model"]):
        prec = gg.b_tp_pred.sum() / max(1, gg.b_n_pred.sum())
        rec = gg.b_tp_true.sum() / max(1, gg.b_n_true.sum())
        summ.append({
            "fraction": f, "model": lab, "n_regions": len(gg),
            "contacts_per_region": gg.shallow_contacts.mean()
            if lab.startswith("shallow") else np.nan,
            "genome_equivalent_contacts": (
                total_contacts * f if lab.startswith("shallow") else np.nan),
            "pearson": gg.pearson.mean(),
            "pearson_highconf": gg.pearson_highconf.mean(),
            "pearson_lowconf": gg.pearson_lowconf.mean(),
            "insulation_pearson": gg.insulation_pearson.mean(),
            "boundary_f1": 2 * prec * rec / (prec + rec) if prec + rec else np.nan,
        })
    S = pd.DataFrame(summ).sort_values(["fraction", "model"])
    S.to_csv(out.parent / (out.name + "_summary.csv"), index=False)
    print(S.round(4).to_string(index=False))
    (out.parent / (out.name + "_meta.json")).write_text(json.dumps({
        "deep": a.deep, "total_contacts": total_contacts,
        "coarse_res": int(coarse_res), "fractions": fracs,
        "lowconf_mode": a.lowconf_mode,
        "insulation_window": int(ins_win), "boundary_tol": int(btol),
        "max_dist_bins": int(max_d),
        "n_regions": len(idx), "seed": int(a.seed)}, indent=2))
    log.info("wrote %s", out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
