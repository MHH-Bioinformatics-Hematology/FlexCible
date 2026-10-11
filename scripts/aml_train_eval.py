"""Held-out patient evaluation of the AML-inclusive models (registry section 36).

Models (fold f, resolution r): the AML-trained all12aml_refin_drop /
all12aml_honest_drop and the models trained without AML data
(all12refin_drop, all12honest_drop), on the 413 Akita test regions (no
training window of any cell overlaps them) of the four test patients of the
fold and of U-937 WT. Reference-map models read the AML reference of the fold
(mean of the training patients); the non-AML refin model is also run with
consensus12. Two input conditions per sample: its own marks (CUT&Tag
H3K27ac, H3K27me3 and, for 9 patients, CTCF; U-937: its ChIP) and no ChIP at
all (sequence and motifs only), so the gain of the ChIP beyond the reference
is read from the pair.

Scores: Pearson, SCC and structure measures (crosscell_predict --structure);
the baselines to beat on the same pixels: the AML reference (mean of the
training patients), the mean of all eleven other patients (leave-one-patient-
out), the HSPC consensus and consensus12; the residual correlation
corr(P - C, T - C) with C the AML reference or consensus12; the same on the
patient-specific regions (scripts/aml_primary_explain.py stage learn: both
deviation scores in the patient's top 20%); and at 10 kb Xu et al.'s
AML-specific loops and stripes (Supplementary Tables 1 and 4) on windows
centred on them: log2 donut enrichment at the loop pixel, mean stripe band,
in the prediction, the measured map and the AML reference.

    python scripts/aml_train_eval.py --fold 0
"""
from __future__ import annotations

import argparse
import os
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
import aml_train as at                                              # noqa: E402
import multicell_expanded as mx                                     # noqa: E402

OUTD = mx.HM / "runs/aml_train_eval"
RES_OUT = REPO / "results/aml_train"
PY = sys.executable
RES = [25000, 10000]
GPUS = os.environ.get("HICMULTI_GPUS", "0,1").split(",")
VARIANT = ""          # "extra" (folds >= 10): only the Meth and the residual-reference models, tables eval_f<N>_extra_*
TAG = ""


def log(msg):
    OUTD.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUTD / "aml_train_eval.log", "a") as fh:
        fh.write(line + "\n")


def hspc_lopo(d: str, res: int) -> Path:
    """Mean of the three other HSPC donors (the baseline of a held-out donor)."""
    import aml_primary as ap
    return ap.build_aml_consensus(res, None, labs=[at.hspc_label(x) for x in at.HSPC_TRAIN + at.HSPC_TEST
                                                  if x != d],
                                  out=at.CONS_DIR / f"hspclopo_{d}_{res // 1000}kb.cool")


def samples(fold: int, res: int) -> list[dict]:
    out = []
    for p in at.test_patients(fold):
        d = at.INP / at.label(p)
        marks = sorted(b.stem for b in d.glob("*.bigwig"))
        out.append({"cell": at.guard(p), "patient": p, "group": "patient", "dir": d, "marks": marks,
                    "meth": at.patient_meth_dir(at.guard(p)),
                    "assay": "hic", "lab": "xu",
                    "target": at.OE_DIR / f"xu_{p}_proper_{res // 1000}kb.cool",
                    "lopo": (at.clean_lopo(p, res) if fold >= at.NEW_FOLD_MIN
                                 else at.CONS_DIR / f"amlcons_lopo_xu_{p}_{res // 1000}kb.cool")})
    for dn in at.HSPC_TEST:
        marks = sorted(b.stem for b in at.HSPC_MARKS.glob("*.bigwig"))
        out.append({"cell": at.hspc_guard(dn), "patient": f"HSPC-{dn}", "group": "hspc", "meth": None,
                    "dir": at.HSPC_MARKS, "marks": marks, "assay": "hic", "lab": "xu",
                    "target": at.OE_DIR / f"{at.hspc_label(dn)}_proper_{res // 1000}kb.cool",
                    "lopo": hspc_lopo(dn, res)})
    d = at.AML / "hg38_u937"
    out.append({"cell": "u937", "patient": "u937", "group": "u937", "meth": None, "dir": d,
                "marks": sorted(b.stem for b in d.glob("*.bigwig")), "assay": "microc", "lab": "",
                "target": at.OE_DIR / f"u937_wt_c1_proper_{res // 1000}kb.cool",
                "lopo": (at.clean_lopo(None, res) if fold >= at.NEW_FOLD_MIN
                         else at.CONS_DIR / f"amlcons_all_{res // 1000}kb.cool")})
    return out


# (label, run base, with the Meth track) of the section 66 models; the first is the primary run
DIFF_MODELS = [("aml_diff_loopfocus", "all12diff_loopfocus_refres_drop", False),
               ("aml_diff", "all12diff_refres_drop", False),
               ("aml_diffhspc", "all12diffhspc_refres_drop", False),
               ("aml_diff_loopfocusaml", "all12diff_loopfocusaml_refres_drop", False),
               ("aml_allobj", "all12allobj_refres_drop", False),
               ("aml_diffmeth_loopfocus", "all12diffmeth_loopfocus_refres_drop", True)]


def diff_group_of(s: dict) -> str:
    """Reference group of a sample for the diff-marks models: patients and U-937 read the AML
    reference set, HSPC donors the HSPC donors."""
    return "hspc" if s["group"] == "hspc" else "aml"


def models(fold: int, res: int) -> list[dict]:
    k = res // 1000
    sfx = f"_f{fold}" if fold else ""
    c12 = mx.consensus12(res)
    # ref "aml": the reference of the sample's own kind (AML reference for a
    # patient or U-937, HSPC training mean for a held-out HSPC donor)
    ms = [
        {"label": "aml_refin", "run": f"all12aml_refin_drop{sfx}_{k}kb", "ref": "aml", "lab": True},
        {"label": "aml_refin@hspc", "run": f"all12aml_refin_drop{sfx}_{k}kb", "ref": "hspc",
         "lab": True},
        {"label": "aml_honest", "run": f"all12aml_honest_drop{sfx}_{k}kb", "ref": None, "lab": True},
        {"label": "base_refin@aml", "run": f"all12refin_drop_{k}kb", "ref": "aml", "lab": False},
        {"label": "base_refin@c12", "run": f"all12refin_drop_{k}kb", "ref": c12, "lab": False},
        {"label": "base_honest", "run": f"all12honest_drop_{k}kb", "ref": None, "lab": False},
    ]
    extra = fold >= at.NEW_FOLD_MIN and VARIANT == "extra"
    if extra:                         # folds >= 10: only the variants, evaluated after the main models
        ms = []
    if fold == 0 or extra:            # the same recipe with the Meth channel (section 38)
        ms.append({"label": "aml_meth_refin", "run": f"all12aml_meth_refin_drop{sfx}_{k}kb",
                   "ref": "aml", "lab": True, "meth": True})
    if fold == 0 or extra:            # the reference as residual instead of an input plane
        ms.append({"label": "aml_refres", "run": f"all12aml_refres_drop{sfx}_{k}kb", "ref": "aml",
                   "lab": True})
    if extra:                         # the loop-centric recipe (registry section 63)
        ms.append({"label": "aml_loopfocus", "run": f"all12aml_loopfocus_refres_drop{sfx}_{k}kb",
                   "ref": "aml", "lab": True})
    if extra:                         # differential mark channels and all-cell objects (section 66)
        for lab, run, meth in DIFF_MODELS:
            ms.append({"label": lab, "run": f"{run}{sfx}_{k}kb", "ref": "aml", "lab": True,
                       "diff": True, **({"meth": True} if meth else {})})
    if fold == 0 and res == 25000:
        ms.append({"label": "aml_refin_nolab", "run": "all12aml_refin_drop_nolab_25kb",
                   "ref": "aml", "lab": False})
    return ms


def conds_of(m: dict) -> tuple:
    """Input conditions: the sample's own ChIP marks, none, and for a model
    with the Meth channel also the marks plus Meth and Meth alone."""
    return ("marks", "marks_meth", "meth_only", "none") if m.get("meth") else ("marks", "none")


def keep_sets_of(m: dict, s: dict) -> list[str]:
    ks = [f"marks={','.join(s['marks'])}"]
    if m.get("meth"):
        ks += [f"marks_meth={','.join(s['marks'] + ['Meth'])}", "meth_only=Meth"]
    return ks + ["none="]


def ref_of(m: dict, s: dict, fold: int, res: int):
    r = m["ref"]
    if r == "aml":
        return (at.hspc_ref_path(res, None) if s["group"] == "hspc"
                else at.ref_path(fold, res, None))
    if r == "hspc":
        return at.hspc_ref_path(res, None)
    return r


def ready(run: str) -> bool:
    """Training finished (stopped by patience); `best` exists from the first checkpoint on."""
    si = mx.stop_info(run, 10)
    return bool(si and si[2] == "patience") and (mx.REF_OUT / run / "best").exists()


def ckpt(run: str) -> Path:
    return mx.best_ckpt(mx.REF_OUT / run)


def common_others(s: dict, fold: int, res: int) -> list[str]:
    import aml_primary_explain as ex
    o = [f"aml_train_mean={at.ref_path(fold, res, None)}", f"consensus12={mx.consensus12(res)}",
         f"hspc_train_mean={at.hspc_ref_path(res, None)}"]
    # the 4-donor HSPC consensus contains a held-out donor: its leave-one-out
    # mean is the baseline there
    o.append(f"hspc={s['lopo'] if s['group'] == 'hspc' else ex.hspc_consensus_path(res)}")
    if s["lopo"].exists() and s["group"] != "hspc":
        o.append(f"aml_lopo_all={s['lopo']}")
    return o


def predict(fold: int, res: int, m: dict, s: dict, gpu: str) -> int:
    out = OUTD / f"f{fold}" / f"{res // 1000}kb" / m["label"] / s["cell"]
    if (out / "marks" / f"predict_{s['cell']}.csv").exists():
        return 0
    ck = ckpt(m["run"])
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml", "--checkpoint", ck,
           "--cell", s["cell"], "--bigwig-dir", s["dir"], "--target", s["target"], "--tta",
           "--assay", s["assay"], "--binsize", res, "--device", "cuda", "--structure",
           "--out-dir", out, "--save-pred", str(out / "{set}.npz"),
           "--keep-sets", *keep_sets_of(m, s),
           "--others", *common_others(s, fold, res)]
    if m.get("meth") and s["meth"] is not None:
        cmd += ["--extra-bigwig-dir", s["meth"]]
    if m["lab"]:
        cmd += ["--lab", s["lab"]]
    if m.get("diff"):
        cmd += ["--diff-group", diff_group_of(s)]
    ref = ref_of(m, s, fold, res)
    if ref is not None:
        cmd += ["--reference", ref]
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", CUDA_VISIBLE_DEVICES=gpu)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predict.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def stage_predict(fold: int):
    tasks = [(res, m, s) for res in RES for m in models(fold, res)
             for s in samples(fold, res)
             if ready(m["run"])]
    lanes = [tasks[i::len(GPUS)] for i in range(len(GPUS))]

    def lane(i):
        for res, m, s in lanes[i]:
            rc = predict(fold, res, m, s, GPUS[i])
            log(f"f{fold} {res // 1000}kb {m['label']} {s['cell']} rc={rc}")
    with ThreadPoolExecutor(len(GPUS)) as ex:
        list(ex.map(lane, range(len(GPUS))))


# --------------------------------------------------------------------------- #
def resid_corr(P, T, C):
    n = T.shape[0]
    iu = np.triu_indices(n, 2)
    p, t, c = P[iu], T[iu], C[iu]
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(c)
    if ok.sum() < 200:
        return np.nan
    a, b = p[ok] - c[ok], t[ok] - c[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def specific_regions(patient: str, res: int) -> set[str]:
    f = REPO / "results/aml_primary/explain_learn_regions.csv"
    if not f.exists() or patient == "u937":
        return set()
    d = pd.read_csv(f, usecols=["patient", "res_kb", "region"])
    return set(d[(d.patient == at.label(patient)) & (d.res_kb == res // 1000)].region)


def stage_score(fold: int):
    import cooler
    from aml_eval import fetch_region, pred_matrix
    import aml_primary_explain as ex
    rows, rrows = [], []
    for res in RES:
        base_refs = {"aml_train_mean": cooler.Cooler(str(at.ref_path(fold, res, None))),
                     "consensus12": cooler.Cooler(str(mx.consensus12(res))),
                     "hspc_train_mean": cooler.Cooler(str(at.hspc_ref_path(res, None)))}
        for s in samples(fold, res):
            refs = dict(base_refs)
            refs["hspc"] = cooler.Cooler(str(s["lopo"] if s["group"] == "hspc"
                                             else ex.hspc_consensus_path(res)))
            T = cooler.Cooler(str(s["target"]))
            spec = specific_regions(s["patient"], res)
            cache: dict = {}
            for m in models(fold, res):
                d = OUTD / f"f{fold}" / f"{res // 1000}kb" / m["label"] / s["cell"]
                for cond in conds_of(m):
                    f = d / cond / f"predict_{s['cell']}.csv"
                    if not f.exists():
                        continue
                    df = pd.read_csv(f)
                    rec = {"fold": fold, "res_kb": res // 1000, "model": m["label"],
                           "cell": s["cell"], "group": s["group"], "condition": cond,
                           "has_meth": s["meth"] is not None, "n_regions": int(df.model.notna().sum()),
                           "pearson": df.model.median()}
                    for c in ("struct_scc", "struct_insulation_corr", "struct_boundary_f1",
                              "struct_loop_enrich_corr", "struct_compartment_corr",
                              "struct_eigenvector_abs_corr"):
                        if c in df:
                            rec[c[7:]] = df[c].median()
                    for c in df.columns:
                        if c.startswith("base_"):
                            rec[c] = df[c].median()
                    sp = df[df.region.isin(spec)] if spec else df.iloc[0:0]
                    rec["n_specific"] = len(sp)
                    rec["pearson_specific"] = sp.model.median() if len(sp) else np.nan
                    for c in df.columns:
                        if c.startswith("base_") and len(sp):
                            rec[c + "_specific"] = sp[c].median()
                    # residual correlations from the saved predictions
                    z = np.load(d / f"{cond}.npz")
                    rr = {k: [] for k in refs}
                    rrs = {k: [] for k in refs}
                    for reg in z.files:
                        Tm = fetch_region(T, reg, cache)
                        if Tm is None:
                            continue
                        P = pred_matrix(z[reg])
                        if P.shape != Tm.shape:
                            continue
                        for k, c in refs.items():
                            C = fetch_region(c, reg, cache)
                            if C is None or C.shape != Tm.shape:
                                continue
                            v = resid_corr(P, Tm, C)
                            rr[k].append(v)
                            if reg in spec:
                                rrs[k].append(v)
                    for k in refs:
                        rec[f"resid_vs_{k}"] = np.nanmedian(rr[k]) if rr[k] else np.nan
                        rec[f"resid_vs_{k}_specific"] = np.nanmedian(rrs[k]) if rrs[k] else np.nan
                    rows.append(rec)
    R = pd.DataFrame(rows)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(RES_OUT / f"eval_f{fold}{TAG}_patients.csv", index=False, float_format="%.4f")
    log(f"scores: {len(R)} rows")
    return R


# --------------------------------------------------------------------------- #
# Xu et al. AML-specific loops and stripes (10 kb)
def stage_features(fold: int):
    import cooler
    from aml_eval import fetch_region, loop_params, pred_matrix
    import aml_primary_explain as ex
    from hicmultiome.bioscore import local_enrichment
    res = 10000
    W = ex.feature_windows()
    tp = {at.label(p) for p in at.test_patients(fold)}
    W = W[W.patient.isin(tp)]
    if W.empty:
        return pd.DataFrame()
    prm = loop_params(res)
    ref = cooler.Cooler(str(at.ref_path(fold, res, None)))
    c12 = cooler.Cooler(str(mx.consensus12(res)))
    rows = []
    # predictions on the feature windows
    jobs = []
    for p, g in W.groupby("patient"):
        pp = p.replace("xu_", "")
        regs = OUTD / f"f{fold}" / "features" / f"{p}.tsv"
        regs.parent.mkdir(parents=True, exist_ok=True)
        g[["chrom", "start", "end"]].drop_duplicates().to_csv(regs, sep="\t", header=False, index=False)
        s = next(x for x in samples(fold, res) if x["patient"] == pp and x["group"] == "patient")
        for m in models(fold, res):
            if ready(m["run"]):
                jobs.append((p, s, m, regs))

    def run(i_job):
        i, (p, s, m, regs) = i_job
        gpu = GPUS[i % len(GPUS)]
        out = OUTD / f"f{fold}" / "features" / f"{m['label']}__{p}"
        if (out / "none" / f"predict_{s['cell']}.csv").exists():
            return
        ck = ckpt(m["run"])
        cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml", "--checkpoint", ck,
               "--cell", s["cell"], "--bigwig-dir", s["dir"], "--target", s["target"], "--tta",
               "--assay", s["assay"], "--binsize", res, "--device", "cuda", "--regions", regs,
               "--out-dir", out, "--save-pred", str(out / "{set}.npz"),
               "--keep-sets", *keep_sets_of(m, s)]
        if m.get("meth") and s["meth"] is not None:
            cmd += ["--extra-bigwig-dir", s["meth"]]
        if m["lab"]:
            cmd += ["--lab", s["lab"]]
        if m.get("diff"):
            cmd += ["--diff-group", diff_group_of(s)]
        if ref_of(m, s, fold, res) is not None:
            cmd += ["--reference", ref_of(m, s, fold, res)]
        env = dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=gpu)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "predict.log", "a") as fh:
            subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                            cwd=REPO, env=env)
    with ThreadPoolExecutor(len(GPUS)) as exr:
        list(exr.map(run, enumerate(jobs)))
    for p, g in W.groupby("patient"):
        T = cooler.Cooler(str(at.OE_DIR / f"{p}_proper_10kb.cool"))
        preds = {}
        for m in models(fold, res):
            for cond in conds_of(m):
                f = OUTD / f"f{fold}" / "features" / f"{m['label']}__{p}" / f"{cond}.npz"
                if f.exists():
                    preds[(m["label"], cond)] = dict(np.load(f))
        for r in g.itertuples():
            win = f"{r.chrom}:{int(r.start)}-{int(r.end)}"
            cache: dict = {}
            mats = {"measured": fetch_region(T, win, cache),
                    "aml_reference": fetch_region(ref, win, cache),
                    "consensus12": fetch_region(c12, win, cache)}
            for k, pr in preds.items():
                if win in pr:
                    mats["|".join(k)] = pred_matrix(pr[win])
            if any(v is None for v in mats.values()):
                continue
            n = mats["measured"].shape[0]
            if any(v.shape != (n, n) for v in mats.values()):
                continue
            rec = {"patient": p, "kind": r.kind, "window": win}
            i = (int(r.a1) - int(r.start)) // res
            if r.kind == "loop":
                j = (int(r.a2) - int(r.start)) // res
                if not (0 <= i < n and 0 <= j < n):
                    continue
                for kk, M in mats.items():
                    e = local_enrichment(np.exp(np.nan_to_num(M)), prm)
                    rec[kk] = float(np.log2(e[min(i, j), max(i, j)]))
            else:
                j1 = (int(r.e1) - int(r.start)) // res
                j2 = (int(r.e2) - int(r.start)) // res
                i2 = (int(r.a2) - int(r.start)) // res
                if not (0 <= i < n and 0 <= j1 < n and 0 < j2 <= n):
                    continue
                for kk, M in mats.items():
                    band = M[slice(max(0, i), min(n, i2 + 1)), slice(max(0, j1), min(n, j2))]
                    rec[kk] = float(np.nanmean(band)) if np.isfinite(band).any() else np.nan
            rows.append(rec)
    F = pd.DataFrame(rows)
    F.to_csv(RES_OUT / f"eval_f{fold}{TAG}_features_windows.csv", index=False)
    out = []
    for kind, g in F.groupby("kind"):
        meas = g["measured"] - g["aml_reference"]
        rec = {"fold": fold, "kind": kind, "n": len(g), "patients": g.patient.nunique(),
               "measured_minus_aml_reference_median": meas.median()}
        for c in [c for c in g.columns if "|" in c]:
            d = g[c] - g["aml_reference"]
            ok = d.notna() & meas.notna()
            rec[f"{c}|minus_ref_median"] = d.median()
            rec[f"{c}|frac_above_ref"] = float((d > 0).mean())
            rec[f"{c}|corr_with_measured_minus_ref"] = (
                float(np.corrcoef(d[ok], meas[ok])[0, 1]) if ok.sum() > 10 else np.nan)
        out.append(rec)
    O = pd.DataFrame(out)
    O.to_csv(RES_OUT / f"eval_f{fold}{TAG}_features.csv", index=False, float_format="%.4f")
    return O


def summary(R: pd.DataFrame, fold: int):
    """Median over the test patients, over the held-out HSPC donors, and U-937."""
    cols = ["pearson", "scc", "insulation_corr", "loop_enrich_corr", "compartment_corr",
            "pearson_specific", "resid_vs_aml_train_mean", "resid_vs_consensus12",
            "resid_vs_hspc", "resid_vs_hspc_train_mean", "resid_vs_aml_train_mean_specific",
            "resid_vs_consensus12_specific", "resid_vs_hspc_specific",
            "base_aml_train_mean", "base_aml_lopo_all", "base_consensus12", "base_hspc",
            "base_hspc_train_mean", "base_aml_train_mean_specific", "base_hspc_specific"]
    cols = [c for c in cols if c in R.columns]
    parts = []
    R = R.copy()
    R.loc[(R.group == "patient") & R.has_meth, "group_wgbs"] = "patient_wgbs"
    for g, lab in (("patient", "test patients (median)"),
                   ("patient_wgbs", "test patients with WGBS (027, 270, 773; median)"),
                   ("hspc", "held-out HSPC donors (median)"), ("u937", "U-937 WT")):
        A = R[(R.group == g) | (R.group_wgbs == g)].groupby(["res_kb", "model", "condition"])[cols].median().reset_index()
        A["cells"] = lab
        parts.append(A)
    S = pd.concat(parts, ignore_index=True)
    S.to_csv(RES_OUT / f"eval_f{fold}{TAG}_summary.csv", index=False, float_format="%.4f")
    print(S.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variant", default="", choices=["", "extra"],
                    help="extra: evaluate only the Meth, residual-reference and loop-centric models of a fold >= 10")
    ap.add_argument("--stage", nargs="+", default=["predict", "score", "features"])
    a = ap.parse_args()
    global VARIANT, TAG
    VARIANT, TAG = a.variant, ("_" + a.variant if a.variant else "")
    if "predict" in a.stage:
        stage_predict(a.fold)
    if "score" in a.stage:
        summary(stage_score(a.fold), a.fold)
    if "features" in a.stage:
        O = stage_features(a.fold)
        if len(O):
            print(O.round(3).T.to_string())


if __name__ == "__main__":
    main()
