"""Can CpG methylation inside a CTCF motif exclude motif hits that are not bound (registry section 48)? CPU only.

Sections 41 and 46 showed, as a height regression, that motif CpG methylation lowers CTCF height beyond
motif strength. This script asks the site-level question that matters when a model has no CTCF track:
is a motif hit bound or unbound, and does methylation identify the unbound ones?

Sites: the 72,626 CTCF motif hits (JASPAR MA0139.2, PWM >= 0.6 of the maximum) of
scripts/chip_seq_meth_relationships.py with the per-cell tables (CTCF height = log1p of the mean ChIP signal
within +-100 bp of the motif centre; motif and flank methylation from the CpG-level WGBS, >= 10 reads).
Patients (Xu et al. 2022): the tables of scripts/aml_ctcf_methylation.py (MACS2 CUT&Tag signal; WGBS bigWig
with no read counts).

Labels, per cell (no peak calls exist for the ENCODE cells in the repo)
  A  bound   = CTCF height above the 95th percentile of the cell's background windows (random 200 bp windows
               >= 500 bp from every motif hit, so the same +-100 bp mean as the sites);
     unbound = height at or below the median of the cell's 72,626 motif hits.
  B  bound   = top 20 percent of the cell's motif hits by height; unbound = at or below the median.
  Sites between the two thresholds are ambiguous and excluded from every main number.

Models (odd chromosomes of the 6 training cells with a CTCF track (GM23248 has none) -> even chromosomes of the scored cell):
  features  a strength + orientation | a_cpg a + number of motif CpGs | b a_cpg + covered + motif methylation
            (pooled mean, maximum over CpGs with >= 5 reads) | c b + flank methylation | d methylation alone
            (no strength, no orientation); *_nm = without the CpG maximum (patients have 10 bp bins only).
  models    logistic regression, histogram gradient boosting.
  schemes   seen_cell (the training cells with a CTCF track, other chromosomes), unseen_cell (leave-one-cell-out; K562 and HMEC
            are in no training set and are scored with the model of all 6 training cells).
Metrics AUROC and AUPRC on sites with a covered motif CpG (>= 10 reads) and on all labelled sites; chromosome
block bootstrap (even chromosomes resampled) for the metric and the paired difference to the strength-only model.

Stages: bg labels coverage models exclusion patients hic figures
    python scripts/ctcf_site_exclusion.py --stage bg labels coverage models exclusion patients hic figures
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import chip_seq_meth_relationships as csm                           # noqa: E402
import aml_ctcf_methylation as aml                                  # noqa: E402

OUT = REPO / "results/ctcf_site_exclusion"
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
WORK = HM / "runs/ctcf_site_exclusion"
S41 = csm.WORK / "sites"
HELD = csm.HELD_CELLS
# GM23248 has no CTCF track, so it is in neither the training set nor the scored cells here
CELLS = [c for c in csm.CELLS if (Path(csm.mt.cell_dir(c)) / "CTCF.bigwig").exists()]
TRAIN = [c for c in csm.TRAIN_CELLS if c in CELLS]
CHROMS, ODD, EVEN = csm.CHROMS, csm.ODD, csm.EVEN
L = 15
MIN_COV = 10                 # pooled reads over the motif CpGs (as section 41)
MAX_COV = 5                  # reads of a single CpG entering the per-site maximum
N_BG = 100_000
BG_EXCL = 500                # bp between a background window centre and any motif hit
STRONG = 0.75                # strong motif: PWM score >= 0.75 of the maximum
THRESH = [0.5, 0.7, 0.9]
SEED = 20261011
B_BOOT = 300
DEFS = ["A", "B"]
REGIONS = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization/"
               "akita_validation/sequences_test_list.tsv")
CROP = 65536
STR = ["frac", "strand"]
M_B = ["n_cpg", "covered", "meth_motif", "meth_max"]
SETS = {"a": STR, "a_cpg": STR + ["n_cpg"], "b": STR + M_B, "c": STR + M_B + ["meth_flank", "flank_ok"],
        "d": M_B + ["meth_flank", "flank_ok"]}
SETS["b_nm"] = [c for c in SETS["b"] if c != "meth_max"]
SETS["c_nm"] = [c for c in SETS["c"] if c != "meth_max"]
SETS["d_nm"] = [c for c in SETS["d"] if c != "meth_max"]
PATIENT_SETS = ["a", "a_cpg", "b_nm", "c_nm", "d_nm"]
BOOT_SETS = ["a", "a_cpg", "b", "c", "d"]
NAMES = {"gm12878": "GM12878", "imr90": "IMR-90", "h1": "H1", "hepg2": "HepG2", "a549": "A549",
         "gm23248": "GM23248", "cd14mono": "CD14+ monocyte", "k562": "K562 (held out)",
         "hmec": "HMEC (held out)"}


def log(msg):
    WORK.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(WORK / "ctcf_site_exclusion.log", "a") as fh:
        fh.write(line + "\n")


def num(ch):
    return int(ch[3:])


# --------------------------------------------------------------------------- frames
def per_site_max(cell, t):
    """Maximum methylated fraction over the CpGs wholly inside the motif (CpG with >= MAX_COV reads),
    NaN for sites without a covered motif CpG (pooled reads < MIN_COV)."""
    f = WORK / f"meth_max_{cell}.npy"
    if f.exists():
        return np.load(f)
    z = np.load(csm.WORK / "cpg" / f"{cell}.npz")
    mx = np.full(len(t), np.nan)
    ch = t.chrom.to_numpy()
    for c in CHROMS:
        ix = np.where(ch == c)[0]
        pos, cov, me = z[c + "_pos"], z[c + "_cov"], z[c + "_meth"]
        st = t.pos.to_numpy()[ix].astype(np.int64)
        a, b = np.searchsorted(pos, st), np.searchsorted(pos, st + L - 1)
        best = np.full(len(ix), -1.0)
        for k in range(int((b - a).max()) if len(ix) else 0):
            j = np.minimum(a + k, len(pos) - 1)
            ok = (a + k < b) & (cov[j] >= MAX_COV)
            fr = me[j] / np.maximum(cov[j], 1)
            best = np.where(ok & (fr > best), fr, best)
        mx[ix] = np.where(best >= 0, best, np.nan)
    mx = np.where(t.cov_motif.to_numpy() >= MIN_COV, mx, np.nan)
    np.save(f, mx)
    return mx


_BASE = None


def base():
    global _BASE
    if _BASE is None:
        _BASE = aml.base_table()
    return _BASE


def cell_frame(cell):
    t = pd.read_csv(S41 / f"table_{cell}.csv.gz")
    b = base()
    assert (t.pos.to_numpy() == b.pos.to_numpy()).all() and (t.chrom.to_numpy() == b.chrom.to_numpy()).all()
    covered = t.cov_motif.to_numpy() >= MIN_COV
    d = pd.DataFrame({"chrom": t.chrom, "pos": t.pos, "strand": t.strand.astype(float), "frac": t.frac,
                      "h": t.h_CTCF, "n_cpg": b.n_cpg_motif.to_numpy(), "covered": covered.astype(float),
                      "meth_motif": np.where(covered, t.meth_motif, np.nan),
                      "meth_max": per_site_max(cell, t),
                      "meth_flank": t.meth_flank, "flank_ok": t.meth_flank.notna().astype(float),
                      "even": t.chrom.map(num).to_numpy() % 2 == 0})
    return d


def patient_frame(p):
    t = aml.load_patient(p)
    cov = t.m_motif.notna().to_numpy()
    return pd.DataFrame({"chrom": t.chrom, "pos": t.pos, "strand": t.strand.astype(float), "frac": t.frac,
                         "h": t.h_CTCF, "n_cpg": t.n_cpg_motif.to_numpy(), "covered": cov.astype(float),
                         "meth_motif": t.m_motif, "meth_max": np.nan,
                         "meth_flank": t.m_flank, "flank_ok": t.m_flank.notna().astype(float),
                         "even": t.chrom.map(num).to_numpy() % 2 == 0})


def entities():
    """(name, kind, bigwig path) of every cell and patient."""
    out = [(c, "cell", Path(csm.mt.cell_dir(c)) / "CTCF.bigwig") for c in CELLS]
    cf = aml.cuttag_files()
    out += [(p, "patient", cf[(p, "CTCF")]) for p in aml.both_patients()]
    return out


def frame_of(name, kind):
    return cell_frame(name) if kind == "cell" else patient_frame(name)


# --------------------------------------------------------------------------- bg
def bg_job(args):
    import pyBigWig
    name, kind, path = args
    f = WORK / f"bg_{name}.npy"
    if f.exists():
        return name
    rng = np.random.default_rng(SEED + sum(map(ord, name)))
    b = base()
    sites = {c: np.sort(b.pos.to_numpy()[b.chrom.to_numpy() == c] + L // 2) for c in CHROMS}
    bw = pyBigWig.open(str(path))
    sizes = bw.chroms()
    chs = [c for c in CHROMS if c in sizes]
    w = np.array([sizes[c] for c in chs], float)
    pick = rng.choice(len(chs), 2 * N_BG, p=w / w.sum())
    vals = []
    for i, c in enumerate(chs):
        n = int((pick == i).sum())
        cen = rng.integers(500, sizes[c] - 500, n)
        s = sites[c]
        j = np.searchsorted(s, cen)
        dl = np.abs(cen - s[np.clip(j - 1, 0, len(s) - 1)])
        dr = np.abs(cen - s[np.clip(j, 0, len(s) - 1)])
        cen = cen[np.minimum(dl, dr) > BG_EXCL]
        for x in cen:
            try:
                v = bw.stats(c, int(x - 100), int(x + 100), type="mean")[0]
            except RuntimeError:
                v = None
            if v is not None:
                vals.append(np.log1p(max(v, 0.0)))
    v = np.array(vals)[:N_BG]
    np.save(f, v)
    log(f"bg {name}: {len(v)} windows, q50 {np.quantile(v, .5):.2f} q95 {np.quantile(v, .95):.2f}")
    return name


def stage_bg(workers):
    with ProcessPoolExecutor(min(workers, 4)) as ex:
        list(ex.map(bg_job, entities()))


# --------------------------------------------------------------------------- labels
def thresholds(name, h):
    bg = np.load(WORK / f"bg_{name}.npy")
    med, q80 = float(np.median(h)), float(np.quantile(h, 0.8))
    q95 = float(np.quantile(bg, 0.95))
    return {"bg_q50": float(np.quantile(bg, .5)), "bg_q95": q95, "bg_q99": float(np.quantile(bg, .99)),
            "n_bg": len(bg), "motif_median": med, "motif_q80": q80,
            "A_bound": q95, "A_unbound": min(med, q95), "B_bound": q80, "B_unbound": med}


def label(h, th, d):
    """1 bound, 0 unbound, NaN ambiguous."""
    y = np.full(len(h), np.nan)
    y[h <= th[f"{d}_unbound"]] = 0
    y[h > th[f"{d}_bound"]] = 1
    return y


def add_labels(t, name):
    th = thresholds(name, t.h.to_numpy())
    for d in DEFS:
        t["y" + d] = label(t.h.to_numpy(), th, d)
    return t, th


def stage_labels():
    rows = []
    for name, kind, _ in entities():
        t, th = add_labels(frame_of(name, kind), name)
        r = {"entity": name, "kind": kind, **th}
        for d in DEFS:
            y = t["y" + d]
            r.update({f"{d}_n_bound": int((y == 1).sum()), f"{d}_n_unbound": int((y == 0).sum()),
                      f"{d}_n_ambiguous": int(y.isna().sum())})
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "labels_thresholds.csv", index=False, float_format="%.4f")


def stage_coverage():
    rows = []
    for name, kind, _ in entities():
        t, _ = add_labels(frame_of(name, kind), name)
        strong = t.frac >= STRONG
        for scope, m in (("all_sites", np.ones(len(t), bool)), ("strong_sites", strong.to_numpy()),
                         ("even_chromosomes", t.even.to_numpy())):
            d = t[m]
            rows.append({"entity": name, "kind": kind, "scope": scope, "n_sites": len(d),
                         "share_with_motif_CpG": float((d.n_cpg > 0).mean()),
                         "share_covered": float((d.covered == 1).mean()),
                         "covered_given_CpG": float((d.covered[d.n_cpg > 0] == 1).mean()) if (d.n_cpg > 0).any() else np.nan,
                         "n_covered": int((d.covered == 1).sum()),
                         "A_share_bound_labelled": float((d.yA == 1).sum() / max(d.yA.notna().sum(), 1)),
                         "A_share_bound_covered": float((d.yA[d.covered == 1] == 1).sum() / max(d.yA[d.covered == 1].notna().sum(), 1)),
                         "A_share_bound_uncovered": float((d.yA[d.covered == 0] == 1).sum() / max(d.yA[d.covered == 0].notna().sum(), 1))})
    pd.DataFrame(rows).to_csv(OUT / "coverage.csv", index=False, float_format="%.4f")


# --------------------------------------------------------------------------- metrics
def auroc(y, p):
    from scipy.stats import rankdata
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def boot_ci(entries, base_key, rng, B=B_BOOT):
    """entries: {key: p}, shared y and chrom. Returns {key: (auc_lo, auc_hi, ap_lo, ap_hi, dauc_lo, dauc_hi, dap_lo, dap_hi)}."""
    y, ch, P = entries["_y"], entries["_ch"], entries["P"]
    chs = np.unique(ch)
    grp = {c: np.where(ch == c)[0] for c in chs}
    acc = {k: [] for k in P}
    for _ in range(B):
        idx = np.concatenate([grp[c] for c in rng.choice(chs, len(chs), replace=True)])
        yy = y[idx]
        if yy.sum() < 5 or (len(yy) - yy.sum()) < 5:
            continue
        for k, p in P.items():
            acc[k].append((auroc(yy, p[idx]), auprc(yy, p[idx])))
    out = {}
    for k in P:
        a = np.array(acc[k])
        ab = np.array(acc[base_key])
        if len(a) < 20:
            out[k] = (np.nan,) * 8
            continue
        d = a - ab
        q = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5]))
        out[k] = q(a[:, 0]) + q(a[:, 1]) + q(d[:, 0]) + q(d[:, 1])
    return out


def make_model(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=500))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=50, random_state=0)


def metric_rows(d, scheme, model, cell, kind, P, scope, y, ch, rng, ci, bsets=BOOT_SETS):
    """One row per feature set for a scored scope."""
    ents = {"_y": y.astype(int), "_ch": ch, "P": {s: P[s] for s in bsets if s in P}}
    cis = boot_ci(ents, "a", rng) if (ci and len(y) > 50 and 0 < y.sum() < len(y)) else {}
    rows = []
    for s, p in P.items():
        r = {"def": d, "scheme": scheme, "model": model, "entity": cell, "kind": kind, "set": s, "scope": scope,
             "n_sites": len(y), "n_bound": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan,
             "auroc": auroc(y.astype(int), p) if len(y) else np.nan,
             "auprc": auprc(y.astype(int), p) if len(y) else np.nan}
        ci_ = cis.get(s, (np.nan,) * 8)
        for k, v in zip(["auroc_lo", "auroc_hi", "auprc_lo", "auprc_hi", "d_auroc_lo", "d_auroc_hi",
                         "d_auprc_lo", "d_auprc_hi"], ci_):
            r[k] = v
        r["d_auroc"] = r["auroc"] - auroc(y.astype(int), P["a"]) if len(y) else np.nan
        r["d_auprc"] = r["auprc"] - auprc(y.astype(int), P["a"]) if len(y) else np.nan
        rows.append(r)
    return rows


def stage_models():
    rng = np.random.default_rng(SEED)
    F = {}
    for c in CELLS:
        F[c], _ = add_labels(cell_frame(c), c)
    PT = {}
    for p in aml.both_patients():
        PT[p], _ = add_labels(patient_frame(p), p)
    rows, prow = [], []
    for d in DEFS:
        yc = "y" + d
        lab = {c: F[c][F[c][yc].notna()] for c in CELLS}
        odd = {c: lab[c][~lab[c].even] for c in CELLS}
        even = {c: lab[c][lab[c].even] for c in CELLS}
        folds = {"all_train": TRAIN} | {c: [x for x in TRAIN if x != c] for c in TRAIN}
        for kind in ("lr", "gb"):
            pred = {}                                   # (fold, set) -> {cell: p on even labelled rows}
            ppred = {}
            for fold, trc in folds.items():
                tr = pd.concat([odd[c] for c in trc])
                ytr = tr[yc].to_numpy().astype(int)
                for s, cols in SETS.items():
                    m = make_model(kind).fit(tr[cols].to_numpy(float), ytr)
                    targets = [c for c in CELLS if (fold == "all_train") or c == fold]
                    for c in targets:
                        pred[(fold, s, c)] = m.predict_proba(even[c][cols].to_numpy(float))[:, 1]
                    if fold == "all_train" and s in PATIENT_SETS:
                        for p, t in PT.items():
                            e = t[t[yc].notna() & t.even]
                            ppred[(s, p)] = m.predict_proba(e[cols].to_numpy(float))[:, 1]
                log(f"models {d} {kind} fold {fold} done")
            for scheme, cells in (("seen_cell", TRAIN), ("unseen_cell", CELLS)):
                for c in cells:
                    fold = "all_train" if (scheme == "seen_cell" or c in HELD) else c
                    e = even[c]
                    P = {s: pred[(fold, s, c)] for s in SETS}
                    y = e[yc].to_numpy()
                    ch = e.chrom.to_numpy()
                    cov = e.covered.to_numpy() == 1
                    rows += metric_rows(d, scheme, kind, c, "cell", {s: p[cov] for s, p in P.items()},
                                        "covered_motif_CpG", y[cov], ch[cov], rng, True)
                    rows += metric_rows(d, scheme, kind, c, "cell", P, "all_labelled_sites", y, ch, rng,
                                        False)
            for p, t in PT.items():
                e = t[t[yc].notna() & t.even]
                P = {s: ppred[(s, p)] for s in PATIENT_SETS}
                y, ch, cov = e[yc].to_numpy(), e.chrom.to_numpy(), e.covered.to_numpy() == 1
                prow += metric_rows(d, "patient_transfer", kind, p, "patient",
                                    {s: v[cov] for s, v in P.items()}, "covered_motif_CpG", y[cov], ch[cov], rng, True,
                                    PATIENT_SETS)
                prow += metric_rows(d, "patient_transfer", kind, p, "patient", P, "all_labelled_sites", y, ch,
                                    rng, False)
    pd.DataFrame(rows).to_csv(OUT / "site_classification_metrics.csv", index=False, float_format="%.5f")
    pd.DataFrame(prow).to_csv(OUT / "patient_transfer_metrics.csv", index=False, float_format="%.5f")


# --------------------------------------------------------------------------- exclusion
BINS = np.linspace(0, 1, 11)


def rule_stats(t, yc, thr):
    """t: strong sites. Rule: covered and meth_motif > thr -> unbound."""
    lab = t[t[yc].notna()]
    cov = lab[lab.covered == 1]
    exc = cov[cov.meth_motif > thr]
    nb_c, nu_c = int((cov[yc] == 1).sum()), int((cov[yc] == 0).sum())
    nb_a, nu_a = int((lab[yc] == 1).sum()), int((lab[yc] == 0).sum())
    ub_e, b_e = int((exc[yc] == 0).sum()), int((exc[yc] == 1).sum())
    return {"n_strong_labelled": len(lab), "n_strong_covered": len(cov), "n_bound_covered": nb_c,
            "n_unbound_covered": nu_c, "n_unbound_all": nu_a, "n_bound_all": nb_a,
            "n_excluded": len(exc), "unbound_excluded": ub_e, "bound_excluded": b_e,
            "precision_unbound": ub_e / len(exc) if len(exc) else np.nan,
            "base_unbound_covered": nu_c / max(nb_c + nu_c, 1),
            "false_exclusion_rate": b_e / nb_c if nb_c else np.nan,
            "removed_share_unbound_covered": ub_e / nu_c if nu_c else np.nan,
            "removed_share_unbound_all": ub_e / nu_a if nu_a else np.nan}


def boot_rule(t, yc, thr, rng, B=500):
    chs = t.chrom.unique()
    grp = {c: t[t.chrom == c] for c in chs}
    v = []
    for _ in range(B):
        r = rule_stats(pd.concat([grp[c] for c in rng.choice(chs, len(chs), replace=True)]), yc, thr)
        v.append([r["precision_unbound"], r["false_exclusion_rate"], r["removed_share_unbound_all"]])
    return np.nanpercentile(np.array(v), [2.5, 97.5], axis=0)


def exclusion_for(name, kind, rng, out_bins, out_rules):
    t, _ = add_labels(frame_of(name, kind), name)
    s = t[t.frac >= STRONG]
    for d in DEFS:
        yc = "y" + d
        lab = s[(s.covered == 1) & s[yc].notna()]
        bi = np.clip(np.digitize(lab.meth_motif, BINS[1:-1]), 0, 9)
        for i in range(10):
            g = lab[bi == i]
            nb, nu = int((g[yc] == 1).sum()), int((g[yc] == 0).sum())
            out_bins.append({"entity": name, "kind": kind, "def": d, "bin": f"{BINS[i]:.1f}-{BINS[i + 1]:.1f}",
                             "bin_lo": BINS[i], "n_labelled": nb + nu, "n_bound": nb, "n_unbound": nu,
                             "fraction_bound": nb / (nb + nu) if nb + nu else np.nan})
        for thr in THRESH:
            r = rule_stats(s, yc, thr)
            lo, hi = boot_rule(s[s[yc].notna()], yc, thr, rng)
            r.update({"entity": name, "kind": kind, "def": d, "threshold": thr,
                      "held_out_cell": name in HELD,
                      "precision_lo": lo[0], "precision_hi": hi[0], "fer_lo": lo[1], "fer_hi": hi[1],
                      "removed_all_lo": lo[2], "removed_all_hi": hi[2]})
            out_rules.append(r)
    return len(s)


def stage_exclusion():
    rng = np.random.default_rng(SEED + 1)
    bins, rules = [], []
    for c in CELLS:
        n = exclusion_for(c, "cell", rng, bins, rules)
        log(f"exclusion {c}: {n} strong sites")
    pd.DataFrame(bins).to_csv(OUT / "exclusion_bound_fraction_by_methylation.csv", index=False, float_format="%.5f")
    pd.DataFrame(rules).to_csv(OUT / "exclusion_rules_cells.csv", index=False, float_format="%.5f")


def stage_patients():
    rng = np.random.default_rng(SEED + 2)
    bins, rules = [], []
    parts = []
    for p in aml.both_patients():
        exclusion_for(p, "patient", rng, bins, rules)
        t, _ = add_labels(patient_frame(p), p)
        t["patient"] = p
        parts.append(t[t.frac >= STRONG])
    pd.DataFrame(bins).to_csv(OUT / "exclusion_patients_bound_fraction_by_methylation.csv", index=False,
                              float_format="%.5f")
    R = pd.DataFrame(rules)
    pooled = []
    for d in DEFS:
        for thr in THRESH:
            g = R[(R["def"] == d) & (R.threshold == thr)]
            sm = g[["n_bound_covered", "n_unbound_covered", "n_unbound_all", "n_bound_all", "n_excluded",
                    "unbound_excluded", "bound_excluded"]].sum()
            pooled.append({"def": d, "threshold": thr, **sm.to_dict(), "n_patients": len(g),
                           "precision_unbound": sm.unbound_excluded / max(sm.n_excluded, 1),
                           "base_unbound_covered": sm.n_unbound_covered / max(sm.n_unbound_covered + sm.n_bound_covered, 1),
                           "false_exclusion_rate": sm.bound_excluded / max(sm.n_bound_covered, 1),
                           "removed_share_unbound_all": sm.unbound_excluded / max(sm.n_unbound_all, 1)})
    R.to_csv(OUT / "exclusion_rules_patients.csv", index=False, float_format="%.5f")
    pd.DataFrame(pooled).to_csv(OUT / "exclusion_rules_patients_pooled.csv", index=False, float_format="%.5f")


# --------------------------------------------------------------------------- hic
def stage_hic():
    reg = pd.read_csv(REGIONS, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    log(f"{len(reg)} test regions")
    rows = []
    ents = [(c, "cell") for c in CELLS] + [(p, "patient") for p in aml.both_patients()]
    for name, kind in ents:
        t, _ = add_labels(frame_of(name, kind), name)
        t = t.copy()
        for crop, tag in ((0, "full_window"), (CROP, "scored_central_window")):
            inreg = np.zeros(len(t), bool)
            nreg = {}
            for r in reg.itertuples():
                m = ((t.chrom == r.chrom) & (t.pos >= r.start + crop) & (t.pos < r.end - crop)).to_numpy()
                inreg |= m
                nreg[(r.chrom, r.start)] = m
            t["in_" + tag] = inreg
            for d in DEFS:
                yc = "y" + d
                s = t[(t.frac >= STRONG) & t["in_" + tag]]
                lab = s[s[yc].notna()]
                cov = lab[lab.covered == 1]
                for thr in THRESH:
                    exc_mask = (t.frac >= STRONG) & (t.covered == 1) & (t.meth_motif > thr) & t[yc].notna()
                    ex = t[exc_mask & t["in_" + tag]]
                    nreg_hit = sum(1 for m in nreg.values() if (m & exc_mask.to_numpy()).any())
                    rows.append({"entity": name, "kind": kind, "def": d, "window": tag, "threshold": thr,
                                 "strong_sites_in_regions": len(s), "strong_labelled_in_regions": len(lab),
                                 "strong_covered_labelled_in_regions": len(cov),
                                 "unbound_strong_in_regions": int((lab[yc] == 0).sum()),
                                 "bound_strong_in_regions": int((lab[yc] == 1).sum()),
                                 "excluded_in_regions": len(ex),
                                 "excluded_unbound_in_regions": int((ex[yc] == 0).sum()),
                                 "excluded_bound_in_regions": int((ex[yc] == 1).sum()),
                                 "regions_with_excluded_site": nreg_hit, "n_regions": len(reg),
                                 "excluded_all_genome": int(exc_mask.sum()),
                                 "excluded_share_of_unbound_strong_in_regions":
                                     int((ex[yc] == 0).sum()) / max(int((lab[yc] == 0).sum()), 1)})
    pd.DataFrame(rows).to_csv(OUT / "hic_regions_excluded.csv", index=False, float_format="%.5f")


# --------------------------------------------------------------------------- figures
def stage_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = pd.read_csv(OUT / "site_classification_metrics.csv")
    keep = ["def", "scheme", "model", "entity", "set", "n_sites", "n_bound", "prevalence", "auroc", "auroc_lo",
            "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "d_auroc", "d_auroc_lo", "d_auroc_hi", "d_auprc",
            "d_auprc_lo", "d_auprc_hi"]
    M[(M.scope == "covered_motif_CpG") & M["set"].isin(BOOT_SETS)][keep].to_csv(
        OUT / "summary_covered_sites.csv", index=False, float_format="%.4f")
    sets = ["a", "a_cpg", "b", "c", "d"]
    cols = {"a": "#7f7f7f", "a_cpg": "#aec7e8", "b": "#1f77b4", "c": "#08306b", "d": "#d62728"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for k, d in enumerate(DEFS):
        gg = M[(M["def"] == d) & (M.model == "gb") & (M.scope == "covered_motif_CpG") & (M.scheme == "unseen_cell")]
        for j, s in enumerate(sets):
            q = gg[gg["set"] == s].set_index("entity").reindex(CELLS)
            x = np.arange(len(CELLS)) + (j - 2) * 0.16
            ax[k].bar(x, q.auroc, 0.16, color=cols[s], label=s,
                      yerr=[q.auroc - q.auroc_lo, q.auroc_hi - q.auroc], error_kw={"lw": 0.8})
        ax[k].axhline(0.5, color="k", lw=0.6)
        ax[k].set_xticks(range(len(CELLS)), [NAMES[c].replace(" (held out)", "*") for c in CELLS], rotation=45, ha="right")
        ax[k].set_title(f"label definition {d}, gradient boosting, unseen cell, sites with motif CpG")
        ax[k].set_ylim(0.4, 1.0)
    ax[0].set_ylabel("AUROC (bound vs unbound), 95% chromosome bootstrap")
    ax[0].legend(ncol=5, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_auroc_by_feature_set.png", dpi=150)
    plt.close(fig)

    Bn = pd.read_csv(OUT / "exclusion_bound_fraction_by_methylation.csv")
    Bn = Bn[Bn["def"] == "A"]
    Pb = pd.read_csv(OUT / "exclusion_patients_bound_fraction_by_methylation.csv")
    Pb = Pb[Pb["def"] == "A"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for c in CELLS:
        q = Bn[Bn.entity == c]
        ax[0].plot(q.bin_lo + 0.05, q.fraction_bound, "-o" if c in HELD else "-", ms=3, label=NAMES[c])
    ax[0].set_title("cells, strong motifs, label A")
    ax[0].legend(fontsize=7)
    pool = Pb.groupby("bin_lo")[["n_bound", "n_unbound"]].sum()
    for p in Pb.entity.unique():
        q = Pb[Pb.entity == p]
        ax[1].plot(q.bin_lo + 0.05, q.fraction_bound, color="#bbbbbb", lw=0.8)
    ax[1].plot(pool.index + 0.05, pool.n_bound / (pool.n_bound + pool.n_unbound), "k-o", ms=3, label="pooled")
    ax[1].set_title("AML patients, strong motifs, label A (bin means, no read counts)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("motif CpG methylation")
    ax[0].set_ylabel("fraction bound among labelled sites")
    fig.tight_layout()
    fig.savefig(OUT / "fig_bound_fraction_by_methylation.png", dpi=150)
    plt.close(fig)

    R = pd.read_csv(OUT / "exclusion_rules_cells.csv")
    R = R[R["def"] == "A"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for j, (col, ttl) in enumerate([("precision_unbound", "precision: excluded sites that are unbound"),
                                    ("false_exclusion_rate", "bound covered sites wrongly excluded"),
                                    ("removed_share_unbound_all", "share of all unbound strong hits removed")]):
        for i, thr in enumerate(THRESH):
            q = R[R.threshold == thr].set_index("entity").reindex(CELLS)
            ax[j].bar(np.arange(len(CELLS)) + (i - 1) * 0.27, q[col], 0.27, label=f"t = {thr}")
        ax[j].set_xticks(range(len(CELLS)), [NAMES[c].replace(" (held out)", "*") for c in CELLS], rotation=45, ha="right")
        ax[j].set_title(ttl, fontsize=9)
        ax[j].set_ylim(0, 1)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exclusion_rule.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["bg", "labels", "coverage", "models", "exclusion", "patients", "hic", "figures"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if "bg" in a.stage:
        stage_bg(a.workers)
    if "labels" in a.stage:
        stage_labels()
    if "coverage" in a.stage:
        stage_coverage()
    if "models" in a.stage:
        stage_models()
    if "exclusion" in a.stage:
        stage_exclusion()
    if "patients" in a.stage:
        stage_patients()
    if "hic" in a.stage:
        stage_hic()
    if "figures" in a.stage:
        stage_figures()


if __name__ == "__main__":
    main()
