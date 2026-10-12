"""Input-information ceiling of the criteria2 loop test (registry section 71).

Question: how well does a flexible feature-based classifier separate AML-specific PRIVATE loops from
reference / HSPC-matched random pairs (control set A_refrand of criteria2, section 69) when it may use all
patient-level information that exists for the ten test donors? If it is far above the residual AUROC of the
Hi-C models (0.66 to 0.69) the model leaves information on the table, if it is near them the inputs are the limit.

Datasets (all with the criteria2 matching: scripts/aml_loop_detection.match_controls, calipers 0.15 log2 on the
AML reference and on the measured HSPC enrichment, 0.10 log10 on distance, exact CTCF orientation class,
greedy one-to-one within the donor, controls = random distance-matched pairs B_rand)
  priv    private AML-specific loops (present in no other donor) against A_refrand (the criteria2 set)
  allloop all AML-specific loops against matched B_rand
  enh     enhancer-like AML-specific stripes against matched shifted rectangles (B_rand)

Features (no feature uses the Hi-C of the target object), rows aligned to results/aml_loop_detection/
object_scores.csv.gz (the object table of the Hi-C models):
  a level    AML reference and HSPC consensus Hi-C enrichment of the candidate
  b seq      CTCF motif strength / orientation / hit count, distance, gene / TSS annotation, CpG islands
  x context  HSPC marks (all CD34+ experiments), healthy methylation (PBMC, CD14+), HSPC ATAC and RNA
  c own      patient H3K27ac, H3K27me3, CTCF (anchor A, B, min, max, sum, product, promoter / partner, mid;
             bin, +-20 kb, +-100 kb), CTCF at the strongest motif
  d diff     patient minus HSPC (also z-scored, percentile, healthy-donor and AML-mean references)
  e meth     patient WGBS (mean, minimum, LMR fraction, motif methylation) and patient minus healthy / AML mean
  f atac     patient ATAC (RPGC, quantile matched) and patient minus HSPC, multiscale, ATAC x H3K27ac
  g rna      anchor-gene expression of the patient (TSS in the anchor, +-20 kb) and patient minus HSPC
Features of the scripts aml_loop_mark_correlations.py (features.pkl), aml_loop_classifier.py (v2 table) and
aml_private_loop_info.py (new_features.csv.gz); the multiscale ATAC / RNA blocks are computed here from the
same track caches with the interval code of aml_loop_mark_correlations.py.

Models: histogram gradient boosting and L2 logistic regression (aml_loop_mark_correlations.hgb_model / lin_model),
leave-one-donor-out with locus-disjoint folds (training rows within 2 bins of a test anchor are dropped),
pooled AUROC with donor bootstrap (1000 resamples), macro AUROC over donors, per donor. Regression
(HGB / ridge on the measured residual among private loops) for the Spearman ceiling.

    nice -n 10 python scripts/aml_input_ceiling.py --stage data models report
CPU only; <= 6 processes with one thread each.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import aml_loop_classifier as alc  # noqa: E402
import aml_loop_detection as ald  # noqa: E402
import aml_loop_mark_correlations as alm  # noqa: E402

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
PRIM = HM / "aml/primary"
WORK = PRIM / "work/inputceiling"
PRIV = PRIM / "work/privinfo"
SCORES = HM / "runs/diff_marks/aml_loop_detection_diff_loopfocus_after_T1b/object_scores.csv.gz"
DET = REPO / "results/aml_loop_detection/object_scores.csv.gz"
OUT = REPO / "results/aml_input_ceiling"
KEY = ["patient", "kind", "set", "pair_id"]
HIC_MODELS = ["aml_refin", "aml_diff", "aml_diffhspc", "aml_loopfocus", "base_refin@aml"]
SEED = 0
NBOOT = 1000
JOBS = 6
CHROMS = alm.CHROMS
SCALES = alm.SCALES

GROUP_LABEL = {"a": "a level", "b": "b seq/orient/dist", "x": "x HSPC context", "c": "c own marks",
               "d": "d patient-HSPC marks", "e": "e methylation", "f": "f ATAC", "g": "g RNA"}
ORDER = ["a", "b", "x", "c", "d", "e", "f", "g"]
PATIENT_LEVEL = ["c", "d", "e", "f", "g"]


def log(m: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


# ===================================================================== data
def group_of(c: str) -> str:
    if c in ("reference", "hspc"):
        return "a"
    if c.startswith("clf:"):
        n = c[4:]
        if n.startswith(("motif_", "gene_", "dist_")):
            return "b"
        return {"hspc": "x", "own": "c", "diff": "d", "meth": "e"}[n.split("_")[0]]
    if c.startswith("cgi|"):
        return "b"
    if c.startswith("mot:"):
        t = c.split("|")[0][4:]
        if t in ("n", "best_score"):
            return "b"
        return "c" if t == "bound_ctcf" else "d" if t == "bound_dctcf" else "e"
    tag = c.split("|")[0]
    if tag.startswith(("ac:", "me3:", "ct:")):
        rep = tag.split(":")[1]
        return "c" if rep == "own" else "x" if rep in ("hspc", "hspcm") else "d"
    if tag.startswith("meth:"):
        return "x" if tag.split(":")[2] in ("H", "C") else "e"
    if tag.startswith("atac:"):
        return {"own": "f", "hspc": "x", "diff": "f"}[tag.split(":")[1]]
    if tag.startswith("rna:"):
        return {"own": "g", "hspc": "x", "diff": "g", "up": "g"}[tag.split(":")[1]]
    for pre, g in (("atacown", "f"), ("atacdiff", "f"), ("atachspc", "x"), ("acxatac_own", "f"), ("acxatac_diff", "f"),
                   ("acxatac_hspc", "x"), ("rnaown", "g"), ("rnadiff", "g"), ("rnahspc", "x")):
        if c.startswith(pre):
            return g
    raise ValueError(f"no group for feature {c}")


def multiscale(D: pd.DataFrame) -> pd.DataFrame:
    """ATAC and RNA blocks (tag|position|scale) from the track caches of aml_private_loop_info.py."""
    F: dict = {}
    hz = {k: np.load(PRIV / f"{k}_hspc.npz") for k in ("atac", "rna")}
    ha = {c: hz["atac"][c + "|mean"].astype(float) for c in CHROMS}
    hr = {c: hz["rna"][c + "|prom"].astype(float) for c in CHROMS}
    for p in sorted(D.patient.unique()):
        ip = np.where(D.patient.values == p)[0]
        tr = {}
        fa, fr = PRIV / f"atac_{p}.npz", PRIV / f"rna_{p}.npz"
        if fa.exists():
            z = np.load(fa)
            o = {c: z[c + "|mean"].astype(float) for c in CHROMS}
            tr["atac:own"], tr["atac:hspc"] = o, ha
            tr["atac:diff"] = {c: o[c] - ha[c] for c in CHROMS}
        if fr.exists():
            z = np.load(fr)
            o = {c: z[c + "|prom"].astype(float) for c in CHROMS}
            tr["rna:own"], tr["rna:hspc"] = o, hr
            tr["rna:diff"] = {c: o[c] - hr[c] for c in CHROMS}
            tr["rna:up"] = {c: z[c + "|up"].astype(float) for c in CHROMS}
        for tag, T in tr.items():
            if tag.endswith(":hspc"):
                continue                           # patient independent: filled once below
            for c in CHROMS:
                ic = ip[D.chrom.values[ip] == c]
                if len(ic):
                    alm.block(F, ic, D, alm.Iv(T[c]), "mean", tag, None, mid=True)
        log(f"[multiscale] {p}: {len(F)} columns")
    # HSPC blocks are the same for every patient: compute once for all rows
    for tag, T in (("atac:hspc", ha), ("rna:hspc", hr)):
        for c in CHROMS:
            ic = np.where(D.chrom.values == c)[0]
            if len(ic):
                alm.block(F, ic, D, alm.Iv(T[c]), "mean", tag, None, mid=True)
    return pd.DataFrame(F)


def stage_data() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(SCORES, dtype={"patient": str})
    Dd = pd.read_csv(DET, dtype={"patient": str}, usecols=KEY)
    assert (T[KEY].astype(str).values == Dd[KEY].astype(str).values).all(), "object tables differ"
    X = pd.read_pickle(alm.WORK / "features.pkl")
    assert len(X) == len(T)
    nf = pd.read_csv(PRIV / "new_features.csv.gz", dtype={"patient": str})
    keep = [c for c in nf.columns if c.startswith(("atac", "acxatac", "rna"))]
    nf = nf[KEY + keep]
    M = T[KEY].merge(nf, on=KEY, how="left")
    assert len(M) == len(T)
    F1 = pd.read_csv(ald.FEATS, usecols=KEY + ["_orient", "dist_log10"], dtype={"patient": str})
    T = T.merge(F1, on=KEY, how="left")
    assert len(T) == len(X)
    MS = multiscale(T)
    FX = pd.concat([X.reset_index(drop=True), M[keep].reset_index(drop=True), MS.reset_index(drop=True),
                    T[["reference", "hspc"]].reset_index(drop=True)], axis=1)
    FX = FX.loc[:, ~FX.columns.duplicated()]
    FX = FX.astype(np.float32)
    grp = pd.Series({c: group_of(c) for c in FX.columns})
    log("features per group: " + ", ".join(f"{g}={int((grp == g).sum())}" for g in ORDER))
    T.to_pickle(WORK / "objects.pkl")
    FX.to_pickle(WORK / "features_all.pkl")
    grp.to_csv(OUT / "feature_groups.csv", header=["group"])


def load_data():
    return pd.read_pickle(WORK / "objects.pkl"), pd.read_pickle(WORK / "features_all.pkl")


# ===================================================================== datasets and folds
def build_sets(T: pd.DataFrame) -> dict:
    """Matched (pos, ctl) rows (row labels of T) of the three datasets plus the rows to score."""
    out = {}
    L = T[T.kind == "loop"]
    ST = T[T.kind == "stripe"]
    mq = []
    for name, frame, P, pool in (
            ("priv", L, L[(L.set == "pos") & (L.cls == "private")], "B_rand"),
            ("allloop", L, L[L.set == "pos"], "B_rand"),
            ("enh", ST, ST[(ST.set == "pos") & (ST.pair_type == "enh")], "B_rand")):
        C = frame[frame.set == pool]
        m = ald.match_controls(P, C)
        out[name] = {"pos": m.pos.values, "ctl": m.ctl.values, "all_pos": P.index.values}
        mq.append({"dataset": name, "n_positives": len(P), "n_matched_pairs": len(m),
                   "n_donors": P.patient.nunique(),
                   "auroc_reference_matched": ald.auroc(-frame.loc[m.pos, "reference"].values, -frame.loc[m.ctl, "reference"].values),
                   "auroc_hspc_matched": ald.auroc(-frame.loc[m.pos, "hspc"].values, -frame.loc[m.ctl, "hspc"].values),
                   "auroc_distance_matched": ald.auroc(frame.loc[m.pos, "dist_log10"].values, frame.loc[m.ctl, "dist_log10"].values)})
        log(f"dataset {name}: {len(P)} positives, {len(m)} matched pairs")
    pd.DataFrame(mq).to_csv(OUT / "dataset_summary.csv", index=False, float_format="%.4f")
    return out


def locus_keys(T: pd.DataFrame, rows: np.ndarray) -> list:
    d = T.loc[rows, ["chrom", "a_lo", "a_hi", "b_lo", "b_hi", "kind"]]
    return alc.row_keys(d)


def make_folds(T: pd.DataFrame, train_rows: np.ndarray, score_rows: np.ndarray, pad: int = 2) -> list:
    """Per donor: (train positions, test positions) as row labels of T. Training rows within `pad` bins of a test
    anchor are dropped (locus-disjoint)."""
    kt = dict(zip(train_rows, locus_keys(T, train_rows)))
    ks = dict(zip(score_rows, locus_keys(T, score_rows)))
    donors = sorted(T.loc[score_rows, "patient"].unique())
    folds = []
    pt = T.loc[train_rows, "patient"].values
    ps = T.loc[score_rows, "patient"].values
    for d in donors:
        te = score_rows[ps == d]
        S = set()
        for r in te:
            for k in ks[r]:
                S.update(range(k - pad, k + pad + 1))
        tr = np.array([r for r, p in zip(train_rows, pt) if p != d and not (kt[r] & S)], int)
        folds.append((d, tr, te))
    return folds


# ===================================================================== feature sets
def feature_sets(grp: pd.Series) -> dict:
    g = {k: list(grp.index[grp == k]) for k in ORDER}
    sets = {}
    cum = []
    for k in ORDER:
        cum += g[k]
        sets["cum_" + k] = list(cum)
    allc = list(cum)
    sets["everything"] = allc
    sets["no_patient_features"] = sum((g[k] for k in ("a", "b", "x")), [])
    sets["patient_only_plus_level"] = g["a"] + sum((g[k] for k in PATIENT_LEVEL), [])
    xs = {"xmarks": [c for c in g["x"] if c.startswith(("ac:", "me3:", "ct:", "clf:"))],
          "xmeth": [c for c in g["x"] if c.startswith("meth:")],
          "xatacrna": [c for c in g["x"] if c.startswith(("atac", "rna", "acxatac"))]}
    for k, v in xs.items():
        sets["alone_" + k] = g["a"] + v
    for k in ORDER[1:]:
        sets["alone_" + k] = g["a"] + g[k]
        sets["drop_" + k] = [c for c in allc if c not in set(g[k])]
    return sets


def parallel_fit(task, models, X, y, jobs_spec, colsets, jobs=JOBS):
    """jobs_spec: list of (key, tr, te); returns {(model, set, key): (te, p)}."""
    from joblib import Parallel, delayed
    tasks = [(m, s, k, tr, te) for m in models for s in colsets for (k, tr, te) in jobs_spec]
    colidx = {s: np.array([X.columns.get_loc(c) for c in cols]) for s, cols in colsets.items()}
    A = X.values

    def one(m, s, tr, te):
        return alm.fit_fold(task, m, A[:, colidx[s]], y, tr, te)

    res = Parallel(n_jobs=jobs, max_nbytes="1M", verbose=0)(delayed(one)(m, s, tr, te) for (m, s, k, tr, te) in tasks)
    return {(m, s, k): r for (m, s, k, tr, te), r in zip(tasks, res)}


# ===================================================================== statistics
def boot_weights(nd: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((B + 1, nd))
    W[0] = 1
    for b in range(1, B + 1):
        W[b] = np.bincount(rng.integers(0, nd, nd), minlength=nd)
    return W


def ci(v):
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def partial_rank_corr(x, y, Z, di, nd, W):
    """Spearman of x and y given the ranks of the columns of Z: Pearson correlation of the rank residuals,
    donor bootstrap through per-donor sums (residualisation fixed at the full sample)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x, axis=0), rankdata(y)
    ZZ = np.column_stack([np.ones(len(y))] + [rankdata(Z[:, k]) for k in range(Z.shape[1])])
    res = lambda v: v - ZZ @ np.linalg.lstsq(ZZ, v, rcond=None)[0]
    ex = np.column_stack([res(rx[:, k]) for k in range(rx.shape[1])]) if rx.ndim == 2 else res(rx)[:, None]
    ey = res(ry)

    def dsum(A):
        o = np.zeros((nd,) + A.shape[1:])
        np.add.at(o, di, A)
        return o
    n = dsum(np.ones(len(y)))
    sx, sxx, sy, syy, sxy = dsum(ex), dsum(ex * ex), dsum(ey), dsum(ey * ey), dsum(ex * ey[:, None])
    N = (W @ n)[:, None]
    with np.errstate(all="ignore"):
        cov = W @ sxy - (W @ sx) * (W @ sy)[:, None] / N
        return cov / np.sqrt((W @ sxx - (W @ sx) ** 2 / N) * (W @ syy - (W @ sy) ** 2 / N[:, 0])[:, None])


def summarise(M: np.ndarray, names: list, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for k, nm in enumerate(names):
        lo, hi = ci(M[1:, k])
        r = {"variant": nm, "estimate": M[0, k], "lo": lo, "hi": hi}
        r.update(extra or {})
        rows.append(r)
    return pd.DataFrame(rows)


# ===================================================================== models
def stage_models(only: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    T, X = load_data()
    grp = pd.read_csv(OUT / "feature_groups.csv", index_col=0)["group"]
    sets = feature_sets(grp)
    DS = build_sets(T)
    models = ["hgb", "lr"]
    all_auc, all_pd, all_sp, all_oof, all_marks, all_inc = [], [], [], [], [], []
    for dname, d in DS.items():
        if only and dname not in only:
            continue
        log(f"=== dataset {dname}")
        pos, ctl = d["pos"], d["ctl"]
        train_rows = np.concatenate([pos, ctl])
        y = np.full(len(T), np.nan)
        y[pos], y[ctl] = 1.0, 0.0
        extra = np.setdiff1d(d["all_pos"], pos)             # unmatched private positives: scored for the Spearman
        score_rows = np.concatenate([train_rows, extra])
        folds = make_folds(T, train_rows, score_rows)
        spec = [(k, tr, te) for (k, tr, te) in folds]
        log("folds: " + ", ".join(f"{k}: train {len(tr)} test {len(te)}" for k, tr, te in folds))
        res = parallel_fit("clf", models, X, y, spec, sets)
        oof = {}
        for (m, s, k), (te, p) in res.items():
            oof.setdefault((m, s), np.full(len(T), np.nan))[te] = p
        # hi-c models on the same objects
        hic = {}
        for nm in HIC_MODELS:
            if nm in T:
                hic[nm] = (T[nm] - T["reference"]).values
                if nm + "|none" in T:
                    hic[nm + "|none"] = (T[nm + "|none"] - T["reference"]).values
        # ---- evaluation on matched pairs
        ev = train_rows
        ispos = y[ev] == 1
        donors = sorted(T.loc[ev, "patient"].unique())
        di = np.searchsorted(donors, T.loc[ev, "patient"].values)
        nd = len(donors)
        W = boot_weights(nd, NBOOT, SEED + 1)
        names, cols = [], []
        for (m, s), v in oof.items():
            names.append(f"{m}|{s}")
            cols.append(v[ev])
        for nm, v in hic.items():
            names.append("hic|" + nm)
            cols.append(v[ev])
        # permutation null of the full model (labels shuffled within donor)
        rng = np.random.default_rng(SEED + 7)
        yp = y.copy()
        for dn in donors:
            idx = ev[T.loc[ev, "patient"].values == dn]
            yp[idx] = rng.permutation(y[idx])
        resp = parallel_fit("clf", ["hgb"], X, yp, spec, {"everything": sets["everything"]})
        pv = np.full(len(T), np.nan)
        for (m, s, k), (te, p) in resp.items():
            pv[te] = p
        names.append("hgb|permuted_labels")
        cols.append(pv[ev])
        S = np.column_stack(cols)
        keepc = np.isfinite(S).all(0)
        if not keepc.all():
            log("variants with missing scores on the evaluation rows are left out: " + ", ".join(np.array(names)[~keepc]))
        S = S[:, keepc]
        names = list(np.array(names)[keepc])
        ok = np.ones(len(S), bool)
        A = alm.auroc_pooled_boot(S[ok], ispos[ok], di[ok], nd, W)
        out = summarise(A, names, {"dataset": dname, "n_pos": int(ispos[ok].sum()), "n_ctl": int((~ispos[ok]).sum())})
        macro = alm.auroc_per_donor(S[ok], ispos[ok], di[ok], nd, min_n=10)
        out["macro_auroc"] = np.nanmean(macro, axis=0)
        out["model"] = out.variant.str.split("|").str[0]
        out["set"] = out.variant.str.split("|", n=1).str[1]
        all_auc.append(out)
        for k, dn in enumerate(donors):
            for j, nm in enumerate(names):
                all_pd.append({"dataset": dname, "variant": nm, "donor": dn, "auroc": macro[k, j]})
        # ---- paired differences
        def delta(a, b, label, kind):
            ja, jb = names.index(a), names.index(b)
            v = A[:, ja] - A[:, jb]
            lo, hi = ci(v[1:])
            return {"dataset": dname, "contrast": label, "kind": kind, "a": a, "b": b, "delta": v[0], "lo": lo, "hi": hi}
        marks = []
        for m in models:
            marks.append(delta(f"{m}|everything", f"{m}|no_patient_features", f"{m}: all features minus without own-patient features", "marks_effect"))
            prev = None
            for k in ORDER:
                cur = f"{m}|cum_{k}"
                if prev:
                    marks.append(delta(cur, prev, f"{m}: + group {k}", "increment"))
                prev = cur
            for k in ORDER[1:]:
                marks.append(delta(f"{m}|everything", f"{m}|drop_{k}", f"{m}: all minus group {k}", "drop_one"))
            marks.append(delta(f"{m}|everything", "hic|aml_diff", f"{m} everything minus aml_diff", "vs_hic"))
            marks.append(delta(f"{m}|everything", "hic|aml_refin", f"{m} everything minus aml_refin", "vs_hic"))
        for nm in ("aml_refin", "aml_diff", "aml_diffhspc"):
            if "hic|" + nm + "|none" in names:
                marks.append(delta("hic|" + nm, "hic|" + nm + "|none", f"hic {nm}: with minus without own marks", "hic_marks_effect"))
        all_marks += marks
        # ---- Spearman with the measured residual among all private positives (priv dataset only)
        if dname == "priv":
            pp = d["all_pos"]
            yr = (T.loc[pp, "measured"] - T.loc[pp, "reference"]).values
            dpp = np.searchsorted(donors, T.loc[pp, "patient"].values)
            Pm, pn = [], []
            for (m, s), v in oof.items():
                Pm.append(v[pp])
                pn.append(f"{m}|{s}")
            for nm, v in hic.items():
                Pm.append(v[pp])
                pn.append("hic|" + nm)
            Pm = np.column_stack(Pm)
            kc = np.isfinite(Pm).all(0)
            Pm, pn = Pm[:, kc], list(np.array(pn)[kc])
            okp = np.isfinite(Pm).all(1) & np.isfinite(yr)
            Z = T.loc[pp, ["reference", "hspc"]].values
            Wp = W
            sp = alm.pooled_rank_corr(Pm[okp], yr[okp], dpp[okp], nd, Wp)
            spp = partial_rank_corr(Pm[okp], yr[okp], Z[okp], dpp[okp], nd, Wp)
            for kind, M in (("spearman_classifier_score_vs_residual", sp), ("partial_spearman_given_reference_hspc", spp)):
                s_ = summarise(M, pn, {"dataset": dname, "target": "measured minus reference", "stat": kind, "n": int(okp.sum())})
                all_sp.append(s_)
            # regression ceiling for the residual, private loops only
            log("regression on the measured residual")
            yreg = np.full(len(T), np.nan)
            yreg[pp] = yr
            kk = locus_keys(T, pp)
            rows_pp = pp
            rfolds = []
            ppat = T.loc[pp, "patient"].values
            for dn in donors:
                te = rows_pp[ppat == dn]
                Sx = set()
                for i in np.where(ppat == dn)[0]:
                    for q in kk[i]:
                        Sx.update(range(q - 2, q + 3))
                tr = np.array([r for i, r in enumerate(rows_pp) if ppat[i] != dn and not (kk[i] & Sx)], int)
                rfolds.append((dn, tr, te))
            rres = parallel_fit("reg", models, X, yreg, rfolds, {s: sets[s] for s in ["cum_" + k for k in ORDER] +
                                                                  ["no_patient_features", "patient_only_plus_level"]})
            roof = {}
            for (m, s, k), (te, p) in rres.items():
                roof.setdefault((m, s), np.full(len(T), np.nan))[te] = p
            Rm = np.column_stack([v[pp] for v in roof.values()])
            rn = [f"reg_{m}|{s}" for (m, s) in roof]
            okr = np.isfinite(Rm).all(1) & np.isfinite(yr)
            spr = alm.pooled_rank_corr(Rm[okr], yr[okr], dpp[okr], nd, Wp)
            sprp = partial_rank_corr(Rm[okr], yr[okr], Z[okr], dpp[okr], nd, Wp)
            for kind, M in (("spearman_regression_vs_residual", spr), ("partial_spearman_given_reference_hspc", sprp)):
                all_sp.append(summarise(M, rn, {"dataset": dname, "target": "measured minus reference", "stat": kind, "n": int(okr.sum())}))
            for (m, s), v in roof.items():
                oof[("reg_" + m, s)] = v
        # ---- keep oof scores
        o = T.loc[score_rows, KEY + ["chrom", "a_lo", "b_lo", "reference", "hspc", "measured"]].copy()
        o.insert(0, "dataset", dname)
        o["label"] = y[score_rows]
        for (m, s), v in oof.items():
            if s in ("everything", "no_patient_features", "cum_a", "cum_b", "cum_x", "cum_c", "cum_d", "cum_e", "cum_f", "cum_g", "patient_only_plus_level"):
                o[f"{m}|{s}"] = v[score_rows]
        for nm, v in hic.items():
            o["hic|" + nm] = v[score_rows]
        all_oof.append(o)
        pd.concat(all_auc).to_csv(OUT / "auroc_all_variants.csv", index=False, float_format="%.4f")
        pd.DataFrame(all_pd).to_csv(OUT / "auroc_per_donor.csv", index=False, float_format="%.4f")
        pd.DataFrame(all_marks).to_csv(OUT / "paired_contrasts.csv", index=False, float_format="%.4f")
        if all_sp:
            pd.concat(all_sp).to_csv(OUT / "spearman.csv", index=False, float_format="%.4f")
        pd.concat(all_oof).to_csv(OUT / "oof_scores.csv.gz", index=False, float_format="%.5f")
        log(f"dataset {dname} written")
    if not only or "priv" in only:
        stage_learning_curve(T, X, sets, DS["priv"])


def stage_learning_curve(T, X, sets, d) -> None:
    """AUROC of the full feature set against the number of training donors (the ceiling is data-limited when the
    curve still rises at 9 donors)."""
    log("learning curve")
    pos, ctl = d["pos"], d["ctl"]
    train_rows = np.concatenate([pos, ctl])
    y = np.full(len(T), np.nan)
    y[pos], y[ctl] = 1.0, 0.0
    folds = make_folds(T, train_rows, train_rows)
    rng = np.random.default_rng(SEED + 11)
    rows = []
    donors = [k for k, _, _ in folds]
    for kd in (1, 2, 3, 5, 7, 9):
        reps = 1 if kd == 9 else 4
        spec = []
        for r in range(reps):
            for (k, tr, te) in folds:
                pat = T.loc[tr, "patient"].values
                others = [x for x in donors if x != k]
                pick = rng.choice(others, kd, replace=False)
                spec.append((f"{k}#{r}", tr[np.isin(pat, pick)], te))
        for sname in ("everything", "no_patient_features"):
            res = parallel_fit("clf", ["hgb", "lr"], X, y, spec, {sname: sets[sname]})
            for r in range(reps):
                for m in ("hgb", "lr"):
                    v = np.full(len(T), np.nan)
                    for (mm, ss, k), (te, p) in res.items():
                        if mm == m and k.endswith(f"#{r}"):
                            v[te] = p
                    ev = train_rows
                    donor_arr = T.loc[ev, "patient"].values
                    ispos = y[ev] == 1
                    a = ald.auroc(v[ev][ispos], v[ev][~ispos])
                    macro = np.nanmean([ald.auroc(v[ev][(donor_arr == dn) & ispos], v[ev][(donor_arr == dn) & ~ispos]) for dn in donors])
                    rows.append({"n_train_donors": kd, "rep": r, "model": m, "set": sname, "auroc_pooled": a, "auroc_macro": macro})
        log(f"  {kd} training donors done")
    pd.DataFrame(rows).to_csv(OUT / "learning_curve.csv", index=False, float_format="%.4f")


# ===================================================================== report
def stage_report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = pd.read_csv(OUT / "auroc_all_variants.csv")
    Pd = pd.read_csv(OUT / "auroc_per_donor.csv", dtype={"donor": str})
    C = pd.read_csv(OUT / "paired_contrasts.csv")
    S = pd.read_csv(OUT / "spearman.csv")
    LC = pd.read_csv(OUT / "learning_curve.csv")
    # ---- main table
    rows = []
    lab = {"cum_a": "a level only (check)", "cum_b": "b + sequence / orientation / distance", "cum_x": "x + HSPC context",
           "cum_c": "c + own marks", "cum_d": "d + patient-minus-HSPC marks", "cum_e": "e + methylation",
           "cum_f": "f + ATAC", "cum_g": "g + RNA (h: everything)"}
    for ds in A.dataset.unique():
        a = A[A.dataset == ds].set_index("variant")
        for m in ("hgb", "lr"):
            for s, l in lab.items():
                r = a.loc[f"{m}|{s}"]
                rows.append({"dataset": ds, "kind": "classifier", "model": m, "feature_set": l, "auroc": r.estimate,
                             "lo": r.lo, "hi": r.hi, "macro_auroc": r.macro_auroc, "n_pos": r.n_pos})
            for s in ("no_patient_features", "patient_only_plus_level"):
                r = a.loc[f"{m}|{s}"]
                rows.append({"dataset": ds, "kind": "classifier", "model": m, "feature_set": s, "auroc": r.estimate,
                             "lo": r.lo, "hi": r.hi, "macro_auroc": r.macro_auroc, "n_pos": r.n_pos})
        r = a.loc["hgb|permuted_labels"]
        rows.append({"dataset": ds, "kind": "null", "model": "hgb", "feature_set": "everything, labels permuted within donor",
                     "auroc": r.estimate, "lo": r.lo, "hi": r.hi, "macro_auroc": r.macro_auroc, "n_pos": r.n_pos})
        for v in a.index:
            if v.startswith("hic|") and not v.endswith("|none"):
                r = a.loc[v]
                rows.append({"dataset": ds, "kind": "hic_model", "model": v[4:], "feature_set": "model residual",
                             "auroc": r.estimate, "lo": r.lo, "hi": r.hi, "macro_auroc": r.macro_auroc, "n_pos": r.n_pos})
    Tm = pd.DataFrame(rows)
    Tm.to_csv(OUT / "table_main.csv", index=False, float_format="%.4f")
    # ---- figure
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    col = {"hgb": "#1b6ca8", "lr": "#d9822b"}
    hic = {"aml_refin": "#555555", "aml_diff": "#b03a2e"}
    pr = A[A.dataset == "priv"].set_index("variant")
    x = np.arange(len(lab))
    a0 = ax[0, 0]
    for off, m in ((-0.12, "hgb"), (0.12, "lr")):
        e = np.array([pr.loc[f"{m}|{s}", "estimate"] for s in lab])
        lo = np.array([pr.loc[f"{m}|{s}", "lo"] for s in lab])
        hi = np.array([pr.loc[f"{m}|{s}", "hi"] for s in lab])
        a0.errorbar(x + off, e, yerr=[e - lo, hi - e], fmt="o-", color=col[m], label={"hgb": "gradient boosting", "lr": "logistic regression"}[m], capsize=2)
    for nm, c in hic.items():
        a0.axhline(pr.loc["hic|" + nm, "estimate"], color=c, ls="--", lw=1)
        a0.text(len(lab) - 0.5, pr.loc["hic|" + nm, "estimate"] + 0.004, nm, color=c, ha="right", fontsize=8)
    a0.axhline(0.5, color="k", lw=0.6)
    a0.set_xticks(x, [k.replace("cum_", "") for k in lab])
    a0.set_xlabel("groups added cumulatively (a level, b seq, x HSPC context,\nc own, d diff, e meth, f ATAC, g RNA)")
    a0.set_ylabel("AUROC private loops vs A_refrand")
    a0.set_title("A  Cumulative feature groups")
    a0.legend(frameon=False, loc="upper left")
    # B: alone and drop-one
    a1 = ax[0, 1]
    gs = ORDER[1:]
    y0 = np.arange(len(gs))
    for off, m in ((-0.17, "hgb"), (0.17, "lr")):
        alone = [pr.loc[f"{m}|alone_{g}", "estimate"] for g in gs]
        drop = [pr.loc[f"{m}|drop_{g}", "estimate"] for g in gs]
        a1.barh(y0 + off, np.array(alone) - 0.5, height=0.3, left=0.5, color=col[m], alpha=0.9)
        a1.plot(drop, y0 + off, "k|", ms=12, mew=2)
    a1.axvline(pr.loc["hgb|everything", "estimate"], color=col["hgb"], ls=":", lw=1)
    a1.axvline(pr.loc["lr|everything", "estimate"], color=col["lr"], ls=":", lw=1)
    a1.set_yticks(y0, [GROUP_LABEL[g] for g in gs])
    a1.set_xlim(0.48, 0.80)
    a1.set_xlabel("AUROC: bars = level + group alone,\nticks = everything minus group")
    a1.set_title("B  Single groups")
    # C: per donor
    a2 = ax[0, 2]
    pdn = Pd[Pd.dataset == "priv"]
    don = sorted(pdn.donor.unique())
    for off, v, c, l in ((-0.27, "hgb|everything", col["hgb"], "boosting, everything"), (-0.09, "lr|everything", col["lr"], "logistic, everything"),
                         (0.09, "hic|aml_diff", hic["aml_diff"], "aml_diff"), (0.27, "hic|aml_refin", hic["aml_refin"], "aml_refin")):
        vals = [pdn[(pdn.variant == v) & (pdn.donor == d)].auroc.values[0] for d in don]
        a2.bar(np.arange(len(don)) + off, np.array(vals) - 0.5, bottom=0.5, width=0.17, color=c, label=l)
    a2.set_xticks(np.arange(len(don)), don, rotation=45)
    a2.set_ylabel("AUROC within donor")
    a2.set_title("C  Per donor")
    a2.legend(frameon=False, fontsize=7, ncol=2)
    # D: learning curve
    a3 = ax[1, 0]
    for (m, s), g in LC.groupby(["model", "set"]):
        v = g.groupby("n_train_donors").auroc_pooled.agg(["mean", "std"])
        a3.errorbar(v.index, v["mean"], yerr=v["std"].fillna(0), marker="o", ms=4, capsize=2, color=col[m],
                    ls="-" if s == "everything" else "--", label=f"{m}, {'everything' if s == 'everything' else 'no patient features'}")
    a3.axhline(pr.loc["hic|aml_diff", "estimate"], color=hic["aml_diff"], ls=":", lw=1)
    a3.set_xlabel("training donors")
    a3.set_ylabel("pooled AUROC (private loops)")
    a3.set_title("D  Learning curve")
    a3.legend(frameon=False, fontsize=7)
    # E: datasets
    a4 = ax[1, 1]
    dss = [d for d in ("priv", "allloop", "enh") if d in A.dataset.unique()]
    for k, (v, c, l) in enumerate((("hgb|everything", col["hgb"], "boosting, everything"), ("lr|everything", col["lr"], "logistic, everything"),
                                    ("hic|aml_diff", hic["aml_diff"], "aml_diff"), ("hic|aml_refin", hic["aml_refin"], "aml_refin"))):
        e = [A[(A.dataset == d) & (A.variant == v)].iloc[0] for d in dss]
        est = np.array([r.estimate for r in e])
        a4.bar(np.arange(len(dss)) + (k - 1.5) * 0.2, est - 0.5, bottom=0.5, width=0.19, color=c, label=l,
               yerr=[est - np.array([r.lo for r in e]), np.array([r.hi for r in e]) - est], capsize=2)
    a4.set_xticks(np.arange(len(dss)), [{"priv": "private loops", "allloop": "all AML loops", "enh": "enhancer-like stripes"}[d] for d in dss])
    a4.set_ylabel("AUROC vs matched random controls")
    a4.set_title("E  Datasets")
    a4.legend(frameon=False, fontsize=7)
    # F: Spearman
    a5 = ax[1, 2]
    sp = S[(S.stat == "partial_spearman_given_reference_hspc")].set_index("variant")
    pl = S[(S.stat.isin(["spearman_classifier_score_vs_residual", "spearman_regression_vs_residual"]))].set_index("variant")
    items = [("reg_hgb|cum_g", "regression boosting"), ("reg_lr|cum_g", "regression ridge"), ("hgb|everything", "classifier boosting"),
             ("lr|everything", "classifier logistic"), ("hic|aml_diff", "aml_diff"), ("hic|aml_refin", "aml_refin")]
    yy = np.arange(len(items))
    for off, tab, c, l in ((-0.18, pl, "#7f7f7f", "Spearman"), (0.18, sp, "#2a8c4a", "partial (reference, HSPC)")):
        e = np.array([tab.loc[v, "estimate"] for v, _ in items])
        lo = np.array([tab.loc[v, "lo"] for v, _ in items])
        hi = np.array([tab.loc[v, "hi"] for v, _ in items])
        a5.errorbar(e, yy + off, xerr=[e - lo, hi - e], fmt="o", color=c, label=l, capsize=2)
    a5.axvline(0, color="k", lw=0.6)
    a5.set_yticks(yy, [l for _, l in items])
    a5.set_xlabel("Spearman with measured residual among private loops")
    a5.set_title("F  Residual ranking")
    a5.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_input_ceiling.png", dpi=150)
    log("report written")
    print(Tm[Tm.dataset == "priv"].round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["data", "models", "report"])
    ap.add_argument("--only", nargs="*", default=None, help="datasets of the models stage (priv allloop enh)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if "data" in a.stage:
        stage_data()
    if "models" in a.stage:
        stage_models(a.only)
    if "report" in a.stage:
        stage_report()


if __name__ == "__main__":
    main()
