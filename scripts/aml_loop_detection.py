"""Does the model predict AML-specific loops and stripes in held-out patients? (registry section 62)

Loop-level analysis for the ten test donors of the rebuilt folds 10, 11, 12 (registry 52, 54, 61):
018, 027, 270 | 168, 546, 629, 1360 | 472, 773, 1916. Objects and matched controls are those of the
loop classifier (scripts/aml_loop_classifier.py, results/aml_loop_classifier/features.csv.gz, 10 kb):
positives = AML-specific loops (Xu et al. 2022 Suppl. Table 1, 60 kb to 1.5 Mb) and stripes (Suppl.
Table 4); controls = A_hspc, A_full, A_do (shared loops / stripes, matched 1:1 on distance,
orientation and marks) and B_rand (random pair at the same distance / shifted rectangle).

For each patient the fold's models predict a 1.92 Mb window (192 bins of 10 kb) centred on every
object (positives and controls):
  aml_refin|marks   AML-trained, own marks, the fold's AML reference (7 or 6 training patients)
  aml_refin|none    the same model without any ChIP input (sequence, motifs and reference only)
  aml_refin@hspc    the same model and marks with the HSPC reference instead of the AML reference
  base_refin@aml    the model trained without AML data, own marks, the AML reference
Further maps in the same windows: measured patient, AML reference alone ('reference baseline'),
measured HSPC consensus.

Scores per object: loops, log2 donut enrichment (hicmultiome.bioscore.local_enrichment, inner 1,
outer 4, 10 kb, as in the classifier labels) as the maximum over the 3x3 pixels around the
annotated pixel (the annotation and the local maximum can differ by one bin) and at the exact
pixel; stripes, the mean log O/E over the stripe rectangle (anchor rows x extent columns).
Gain = value minus the measured HSPC consensus value (the site figures); the model's own HSPC
counterpart is aml_refin@hspc.

Recurrence of an AML-specific loop: the number of other donors among the 25 AML samples of Suppl.
Table 1 (168/496 and 260/424 count once; the patient's own donor is excluded) with a loop within
one bin in both anchors (stripes: overlapping anchor and overlapping extent); recurrent >= 2,
shared 1, private 0. n_train counts the donors of the fold's training set (the reference) alone.

    python scripts/aml_loop_detection.py --stage windows predict score stats inc figure --gpu 0
    python scripts/aml_loop_detection.py --recipe loopfocus --stage predict score stats inc criteria --gpu 0
Run with the hicdiff conda env.

--recipe loopfocus (registry section 63) adds the loop-centric model all12aml_loopfocus_refres_drop_f<N>
(label aml_loopfocus; predictions of the other models are reused from the sections 62 run), reads the
objects and windows of results/aml_loop_detection and writes every table to
results/aml_loop_detection_loopfocus/; stage criteria compares it with aml_refin on the success criteria
(AUROC of private loops against the matched shared loops >= 0.70, amplitude >= 30 percent of the
measured gain).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OUTD = HM / "runs/aml_loop_detection"
RES_OUT = REPO / "results/aml_loop_detection"
FEATS = REPO / "results/aml_loop_classifier/features.csv.gz"
XT = REPO / "results/aml_primary/xu_tables"
RES = 10000
WIN = 192
FOLDS = {10: ["018", "027", "270"], 11: ["168", "546", "629", "1360"], 12: ["472", "773", "1916"]}
DONOR = {"168": "168+496", "496": "168+496", "260": "260+424", "424": "260+424"}
CONTROLS = ["A_hspc", "A_full", "A_do", "B_rand"]
# (label, model label of aml_train_eval.models, keep sets)
RUNS = [("aml_refin", ["marks", "none"]), ("aml_refin@hspc", ["marks"]), ("base_refin@aml", ["marks"])]
SEED = 0
NBOOT = 2000
EXTRA_LABELS = ("aml_loopfocus",)    # models of the extra variant of aml_train_eval
ITEMS = RES_OUT / "items.csv.gz"     # objects and windows (always the section 62 table)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def fold_of(p: str) -> int:
    return next(f for f, ps in FOLDS.items() if p in ps)


def donor(p: str) -> str:
    return DONOR.get(p, p)


# ===================================================================== windows and recurrence
def read_tab(name: str) -> pd.DataFrame:
    d = pd.read_csv(XT / name, dtype={"patient": str})
    d["patient"] = d.patient.str.lstrip("#")
    d["patient"] = d.patient.map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)
    return d


def recurrence(items: pd.DataFrame) -> pd.DataFrame:
    """n_other (donors among all AML samples with the object within one bin / overlapping) and
    n_train (training donors of the fold) for every object (positives and controls alike)."""
    L = read_tab("st1_aml_specific_loops.csv")
    L = L[L.chrom1 == L.chrom2]
    S = read_tab("st4_aml_specific_stripes.csv")
    S = S[S.chrom_anchor == S.chrom_extent]
    allp = sorted(set(L.patient) | set(S.patient))
    Ld = {p: g for p, g in L.groupby("patient")}
    Sd = {p: g for p, g in S.groupby("patient")}
    n_other = np.zeros(len(items), int)
    n_train = np.zeros(len(items), int)
    names = np.empty(len(items), object)
    for k, r in enumerate(items.itertuples()):
        f = r.fold
        train = {q for q in sum(FOLDS.values(), []) if fold_of(q) != f}
        seen, seen_tr, who = set(), set(), []
        for q in allp:
            if donor(q) == donor(r.patient):
                continue
            if r.kind == "loop":
                g = Ld.get(q)
                if g is None:
                    continue
                m = ((g.chrom1 == r.chrom) & ((g.start1 // RES - r.a_lo).abs() <= 1) &
                     ((g.start2 // RES - r.b_lo).abs() <= 1)).any()
            else:
                g = Sd.get(q)
                if g is None:
                    continue
                m = ((g.chrom_anchor == r.chrom) & (g.start_anchor // RES < r.a_hi) &
                     (-(-g.end_anchor // RES) > r.a_lo) & (g.start_extent // RES < r.b_hi) &
                     (-(-g.end_extent // RES) > r.b_lo)).any()
            if m:
                seen.add(donor(q))
                who.append(q)
                if q in train:
                    seen_tr.add(donor(q))
        n_other[k], n_train[k], names[k] = len(seen), len(seen_tr), ";".join(who)
    items["n_other"], items["n_train"], items["with_patients"] = n_other, n_train, names
    return items


def stage_windows() -> pd.DataFrame:
    import cooler
    F = pd.read_csv(FEATS, usecols=["patient", "kind", "set", "pair_id", "chrom", "a_lo", "a_hi",
                                    "b_lo", "b_hi", "pair_type", "lab_e_pat", "lab_e_hspc"],
                    dtype={"patient": str})
    F["patient"] = F.patient.map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)
    F = F[F.patient.isin(sum(FOLDS.values(), []))].copy()
    F["fold"] = F.patient.map(fold_of)
    lo = np.minimum(F.a_lo, F.b_lo)
    hi = np.maximum(F.a_hi, F.b_hi)
    F["span"] = hi - lo
    sizes = cooler.Cooler(str(HM / "aml/oe/xu_018_proper_10kb.cool")).chromsizes
    nb = F.chrom.map(lambda c: int(sizes[c]) // RES)
    ok = (F.span <= 150) & (nb >= WIN)
    log(f"objects {len(F)}; dropped (span > 150 bins): {int((~ok).sum())}")
    F = F[ok].copy()
    mid = (np.minimum(F.a_lo, F.b_lo) + np.maximum(F.a_hi, F.b_hi)) // 2
    nbs = F.chrom.map(lambda c: int(sizes[c]) // RES)
    F["win_bin"] = np.clip(mid - WIN // 2, 0, nbs - WIN).astype(int)
    F["win"] = F.chrom + ":" + (F.win_bin * RES).astype(str) + "-" + ((F.win_bin + WIN) * RES).astype(str)
    F = F.reset_index(drop=True)
    F.insert(0, "item", F.index)
    F = recurrence(F)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    F.to_csv(RES_OUT / "items.csv.gz", index=False)
    for p, g in F.groupby("patient"):
        d = OUTD / "windows"
        d.mkdir(parents=True, exist_ok=True)
        w = g[["chrom", "win_bin"]].drop_duplicates()
        pd.DataFrame({"c": w.chrom, "s": w.win_bin * RES, "e": (w.win_bin + WIN) * RES}).to_csv(
            d / f"xu_{p}.tsv", sep="\t", header=False, index=False)
    print(F.groupby(["kind", "set"]).size().unstack())
    print(F[F.set == "pos"].groupby(["kind", "patient"]).size().unstack(0).T)
    return F


# ===================================================================== prediction
def predict(gpu: str, fold: int, p: str, label: str, sets: list[str], threads: int) -> None:
    import aml_train as at
    import aml_train_eval as te
    s = next(x for x in te.samples(fold, RES) if x["patient"] == p and x["group"] == "patient")
    te.VARIANT = "extra" if label in EXTRA_LABELS else ""
    m = next(x for x in te.models(fold, RES) if x["label"] == label)
    te.VARIANT = ""
    out = OUTD / f"f{fold}" / f"{label}__xu_{p}"
    if all((out / f"{k}.npz").exists() for k in sets):
        return
    ck = te.ckpt(m["run"])
    keep = [f"marks={','.join(s['marks'])}"] + (["none="] if "none" in sets else [])
    cmd = [te.PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml", "--checkpoint", ck,
           "--cell", s["cell"], "--bigwig-dir", s["dir"], "--target", s["target"], "--tta",
           "--assay", s["assay"], "--binsize", RES, "--device", "cuda",
           "--regions", OUTD / "windows" / f"xu_{p}.tsv", "--out-dir", out,
           "--save-pred", str(out / "{set}.npz"), "--keep-sets", *keep]
    if m["lab"]:
        cmd += ["--lab", s["lab"]]
    ref = te.ref_of(m, s, fold, RES)
    if ref is not None:
        cmd += ["--reference", ref]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               CUDA_VISIBLE_DEVICES=gpu)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predict.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        rc = subprocess.call(["nice", "-n", "10"] + [str(c) for c in cmd], stdout=fh,
                             stderr=subprocess.STDOUT, cwd=REPO, env=env)
    log(f"f{fold} {label} {p} rc={rc}")


def stage_predict(gpu: str, threads: int, only: list[str] | None) -> None:
    for fold, ps in FOLDS.items():
        for p in ps:
            if only and p not in only:
                continue
            for label, sets in RUNS:
                predict(gpu, fold, p, label, sets, threads)


# ===================================================================== scoring
def enrich_maps(M: np.ndarray, prm) -> np.ndarray:
    from hicmultiome.bioscore import local_enrichment
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log2(local_enrichment(np.exp(M), prm))


def stage_score(only: list[str] | None = None) -> pd.DataFrame:
    import cooler
    from aml_eval import fetch_region, pred_matrix
    import aml_primary_explain as ex
    import aml_train as at
    from hicmultiome.bioscore import BioParams
    prm = BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4)
    items = pd.read_csv(ITEMS, dtype={"patient": str})
    hspc = cooler.Cooler(str(ex.hspc_consensus_path(RES)))
    rows = []
    for fold, ps in FOLDS.items():
        ref = cooler.Cooler(str(at.ref_path(fold, RES, None)))
        for p in ps:
            if only and p not in only:
                continue
            g = items[items.patient == p]
            T = cooler.Cooler(str(at.OE_DIR / f"xu_{p}_proper_10kb.cool"))
            preds = {}
            for label, sets in RUNS:
                for k in sets:
                    f = OUTD / f"f{fold}" / f"{label}__xu_{p}" / f"{k}.npz"
                    if f.exists():
                        nm = f"{label}|{k}" if (label in ("aml_refin", "aml_loopfocus") and k == "none") else label
                        preds[nm] = np.load(f)
            log(f"score {p}: {len(g)} objects, models {list(preds)}")
            cache: dict = {}
            for win, gw in g.groupby("win"):
                maps = {"measured": fetch_region(T, win, cache), "reference": fetch_region(ref, win, cache),
                        "hspc": fetch_region(hspc, win, cache)}
                for nm, z in preds.items():
                    if win in z.files:
                        maps[nm] = pred_matrix(z[win])
                if any(maps.get(k) is None or maps[k].shape != (WIN, WIN) for k in ("measured", "reference", "hspc")):
                    continue
                enr = {nm: enrich_maps(M, prm) for nm, M in maps.items() if M.shape == (WIN, WIN)}
                for r in gw.itertuples():
                    ws = r.win_bin
                    rec = {"item": r.item}
                    if r.kind == "loop":
                        i, j = r.a_lo - ws, r.b_lo - ws
                        if not (1 <= i < WIN - 1 and 1 <= j < WIN - 1):
                            continue
                        for nm, E in enr.items():
                            blk = E[i - 1:i + 2, j - 1:j + 2]
                            rec[nm] = float(np.nanmax(blk)) if np.isfinite(blk).any() else np.nan
                            rec[nm + "#px"] = float(E[i, j])
                    else:
                        a0, a1, b0, b1 = r.a_lo - ws, r.a_hi - ws, r.b_lo - ws, r.b_hi - ws
                        if min(a0, b0) < 0 or max(a1, b1) > WIN:
                            continue
                        for nm, M in maps.items():
                            band = M[a0:a1, b0:b1]
                            rec[nm] = float(np.nanmean(band)) if np.isfinite(band).any() else np.nan
                    rows.append(rec)
    R = pd.DataFrame(rows)
    D = items.merge(R, on="item", how="inner")
    D.to_csv(RES_OUT / ("object_scores.csv.gz" if not only else "object_scores_partial.csv.gz"),
             index=False, float_format="%.5f")
    log(f"scored {len(D)} of {len(items)} objects")
    return D


# ===================================================================== statistics
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_donor(df: pd.DataFrame, fn, rng, nboot=NBOOT):
    """Point estimate over all objects and the 95% interval over donors resampled with replacement."""
    est = fn(df)
    ds = df.patient.unique()
    groups = {d: df[df.patient == d] for d in ds}
    out = []
    for _ in range(nboot):
        pick = rng.choice(ds, len(ds))
        out.append(fn(pd.concat([groups[d] for d in pick], ignore_index=True)))
    out = np.array(out, float)
    out = out[np.isfinite(out)]
    lo, hi = (np.percentile(out, [2.5, 97.5]) if len(out) > 20 else (np.nan, np.nan))
    return est, float(lo), float(hi)


PRED = ["aml_refin", "aml_refin|none", "base_refin@aml", "reference"]
PRED_EXTRA = ["aml_loopfocus", "aml_loopfocus|none"]
CLASSES = ["all", "recurrent", "shared1", "private"]


def add_derived(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    D["cls"] = np.where(D.n_other >= 2, "recurrent", np.where(D.n_other == 1, "shared1", "private"))
    D["cls_train"] = np.where(D.n_train >= 2, "train2+", np.where(D.n_train == 1, "train1", "train0"))
    for nm in ["measured", "aml_refin", "aml_refin|none", "base_refin@aml", "reference", "aml_refin@hspc",
               *PRED_EXTRA]:
        if nm in D:
            D["gain|" + nm] = D[nm] - D["hspc"]
    D["gain_model_hspc|aml_refin"] = D["aml_refin"] - D["aml_refin@hspc"]
    D["gain_model_hspc|aml_refin|none"] = D["aml_refin|none"] - D["aml_refin@hspc"]
    return D


def sel(D: pd.DataFrame, kind: str, cls: str, which: str = "pos") -> pd.DataFrame:
    d = D[D.kind == kind]
    if which == "pos":
        d = d[d.set == "pos"]
    if cls != "all":
        d = d[d.cls == cls]
    return d


def stage_stats(D: pd.DataFrame) -> None:
    from scipy.stats import pearsonr, spearmanr
    rng = np.random.default_rng(SEED)
    D = add_derived(D)
    D.to_csv(RES_OUT / "object_scores.csv.gz", index=False, float_format="%.5f")
    P = D[D.set == "pos"]
    # ---- class sizes
    cs = (P.groupby(["kind", "patient", "pair_type", "cls"]).size().rename("n").reset_index())
    cs.to_csv(RES_OUT / "class_sizes.csv", index=False)
    # ---- (1)-(3) level and gain per patient / pooled, per class
    preds = [c for c in PRED + PRED_EXTRA if c in D]
    rows = []
    for kind in ("loop", "stripe"):
        for cls in CLASSES:
            for typ in ("all", "PE", "PS", "enh", "rep", "other"):
                d = sel(D, kind, cls)
                if typ != "all":
                    d = d[d.pair_type == typ]
                if len(d) < 5:
                    continue
                rec = {"kind": kind, "class": cls, "pair_type": typ, "n": len(d),
                       "n_patients": d.patient.nunique(), "measured_level": d.measured.median(),
                       "hspc_level": d.hspc.median(), "measured_gain": d["gain|measured"].median()}
                for nm in preds + ["aml_refin@hspc"]:
                    rec[f"{nm}_level"] = d[nm].median()
                    rec[f"{nm}_gain_vs_hspc"] = d["gain|" + nm].median()
                    ratio = d["gain|" + nm].median() / d["gain|measured"].median()
                    rec[f"{nm}_amplitude_ratio"] = ratio
                    ok = d[["gain|" + nm, "gain|measured"]].dropna()
                    rec[f"{nm}_spearman"] = spearmanr(ok.iloc[:, 0], ok.iloc[:, 1])[0] if len(ok) > 10 else np.nan
                rec["aml_refin_gain_over_modelhspc"] = d["gain_model_hspc|aml_refin"].median()
                rec["aml_refin_inc_over_reference"] = (d["aml_refin"] - d["reference"]).median()
                rec["aml_refin_inc_over_none"] = (d["aml_refin"] - d["aml_refin|none"]).median()
                rec["measured_inc_over_reference"] = (d["measured"] - d["reference"]).median()
                rows.append(rec)
    pd.DataFrame(rows).to_csv(RES_OUT / "levels_by_class.csv", index=False, float_format="%.4f")

    # ---- correlation of predicted and measured gain, per patient and pooled with donor bootstrap
    crows = []
    for kind in ("loop", "stripe"):
        for cls in CLASSES:
            d = sel(D, kind, cls)
            for nm in preds:
                for tgt in ("gain", "gain_over_ref"):
                    def f(x, nm=nm, tgt=tgt, m="spearman"):
                        y = x["gain|measured"] if tgt == "gain" else x["measured"] - x["reference"]
                        z = x["gain|" + nm] if tgt == "gain" else x[nm] - x["reference"]
                        ok = np.isfinite(y) & np.isfinite(z)
                        if ok.sum() < 10 or z[ok].std() == 0:
                            return np.nan
                        return spearmanr(z[ok], y[ok])[0] if m == "spearman" else pearsonr(z[ok], y[ok])[0]
                    if nm == "reference" and tgt == "gain_over_ref":
                        continue
                    s, slo, shi = boot_donor(d, f, rng, 500)
                    pr, plo, phi = boot_donor(d, lambda x: f(x, m="pearson"), rng, 500)
                    per = [f(g) for _, g in d.groupby("patient")]
                    perp = [f(g, m="pearson") for _, g in d.groupby("patient")]
                    crows.append({"kind": kind, "class": cls, "predictor": nm, "target": tgt, "n": len(d),
                                  "spearman": s, "spearman_lo": slo, "spearman_hi": shi,
                                  "pearson": pr, "pearson_lo": plo, "pearson_hi": phi,
                                  "median_patient_spearman": np.nanmedian(per),
                                  "median_patient_pearson": np.nanmedian(perp),
                                  "patients_spearman_pos": int(np.sum(np.array(per) > 0))})
    pd.DataFrame(crows).to_csv(RES_OUT / "gain_correlation.csv", index=False, float_format="%.4f")

    # ---- AUROC pos vs controls; fraction above the 90th percentile of the controls
    arows = []
    for kind in ("loop", "stripe"):
        for cs_ in CLASSES:
            for ctl in CONTROLS:
                for score in ["level", "gain"]:
                    for nm in ["measured"] + preds + ["aml_refin@hspc", "gain_model_hspc"]:
                        if nm == "gain_model_hspc":
                            if score == "level":
                                continue
                            col = "gain_model_hspc|aml_refin"
                        elif nm == "aml_refin@hspc" and score == "level":
                            col = nm
                        else:
                            col = nm if score == "level" else "gain|" + nm
                        if col not in D:
                            continue
                        pos = D[(D.kind == kind) & (D.set == "pos")]
                        neg = D[(D.kind == kind) & (D.set == ctl)]
                        if cs_ != "all":
                            pos = pos[pos.cls == cs_]
                        # matched pairs: the control inherits the positive's pair_id within a patient
                        def f(x, pos_n=None):
                            a, b = x[x.set == "pos"], x[x.set == ctl]
                            return auroc(a[col].values, b[col].values)

                        def f90(x):
                            a, b = x[x.set == "pos"], x[x.set == ctl]
                            bb = b[col].dropna()
                            if len(bb) < 10:
                                return np.nan
                            thr = np.percentile(bb, 90)
                            aa = a[col].dropna()
                            return float((aa > thr).mean()) if len(aa) else np.nan
                        x = pd.concat([pos, neg], ignore_index=True)
                        if len(pos) < 10 or len(neg) < 10:
                            continue
                        au, lo, hi = boot_donor(x, f, rng, 500)
                        f9, l9, h9 = boot_donor(x, f90, rng, 500)
                        per = [f(g) for _, g in x.groupby("patient")]
                        arows.append({"kind": kind, "class": cs_, "control": ctl, "score": score, "predictor": nm,
                                      "n_pos": len(pos), "n_ctl": len(neg), "auroc": au, "auroc_lo": lo,
                                      "auroc_hi": hi, "median_patient_auroc": np.nanmedian(per),
                                      "frac_above_ctl_p90": f9, "frac_lo": l9, "frac_hi": h9})
    pd.DataFrame(arows).to_csv(RES_OUT / "auroc.csv", index=False, float_format="%.4f")

    # ---- per patient summary
    prow = []
    for (kind, p), d in D[D.set == "pos"].groupby(["kind", "patient"]):
        rec = {"kind": kind, "patient": p, "fold": int(d.fold.iloc[0]), "n": len(d),
               "n_recurrent": int((d.cls == "recurrent").sum()), "n_private": int((d.cls == "private").sum()),
               "measured_gain": d["gain|measured"].median()}
        for nm in preds:
            ok = d[["gain|" + nm, "gain|measured"]].dropna()
            rec[f"{nm}_gain"] = d["gain|" + nm].median()
            rec[f"{nm}_spearman"] = spearmanr(ok.iloc[:, 0], ok.iloc[:, 1])[0] if len(ok) > 10 else np.nan
            rec[f"{nm}_pearson"] = pearsonr(ok.iloc[:, 0], ok.iloc[:, 1])[0] if len(ok) > 10 else np.nan
        for ctl in ["A_hspc"]:
            c = D[(D.kind == kind) & (D.set == ctl) & (D.patient == p)]
            for nm in ["measured"] + preds:
                rec[f"auroc_level_{nm}"] = auroc(d[nm].values, c[nm].values)
                rec[f"auroc_gain_{nm}"] = auroc(d["gain|" + nm].values, c["gain|" + nm].values)
        prow.append(rec)
    pd.DataFrame(prow).to_csv(RES_OUT / "per_patient.csv", index=False, float_format="%.4f")

    # ---- sensitivity: recurrence by the training reference (donors of the fold's training set)
    srows = []
    for kind in ("loop", "stripe"):
        for cls in ["train2+", "train1", "train0"]:
            d = D[(D.kind == kind) & (D.set == "pos") & (D.cls_train == cls)]
            if len(d) < 10:
                continue
            rec = {"kind": kind, "class_train": cls, "n": len(d), "measured_gain": d["gain|measured"].median()}
            for nm in preds:
                rec[f"{nm}_gain"] = d["gain|" + nm].median()
            rec["ref_minus_hspc_level"] = (d["reference"] - d["hspc"]).median()
            srows.append(rec)
    pd.DataFrame(srows).to_csv(RES_OUT / "levels_by_train_recurrence.csv", index=False, float_format="%.4f")
    log("stats written")



def stage_inc(D: pd.DataFrame) -> None:
    """Medians of the per-object differences with donor-bootstrap intervals, by class and pair type,
    and AUROC (gain, A_hspc) by pair type."""
    rng = np.random.default_rng(SEED + 1)
    D = add_derived(D) if "cls" not in D else D
    P = D[D.set == "pos"]
    defs = {"measured_gain": lambda d: d["gain|measured"],
            "model_gain": lambda d: d["gain|aml_refin"],
            "reference_gain": lambda d: d["gain|reference"],
            "none_gain": lambda d: d["gain|aml_refin|none"],
            "base_gain": lambda d: d["gain|base_refin@aml"],
            "model_minus_reference": lambda d: d["aml_refin"] - d["reference"],
            "model_minus_none": lambda d: d["aml_refin"] - d["aml_refin|none"],
            "model_minus_base": lambda d: d["aml_refin"] - d["base_refin@aml"],
            "measured_minus_reference": lambda d: d["measured"] - d["reference"],
            "model_gain_over_model_hspc": lambda d: d["aml_refin"] - d["aml_refin@hspc"]}
    if "aml_loopfocus" in D:
        defs.update({"loopfocus_gain": lambda d: d["gain|aml_loopfocus"],
                     "loopfocus_minus_reference": lambda d: d["aml_loopfocus"] - d["reference"],
                     "loopfocus_minus_refin": lambda d: d["aml_loopfocus"] - d["aml_refin"]})
    rows = []
    for kind in ("loop", "stripe"):
        types = ["all", "PE", "PS", "enh", "rep", "other"]
        for cls in CLASSES:
            for typ in types:
                d = P[P.kind == kind]
                d = d if cls == "all" else d[d.cls == cls]
                d = d if typ == "all" else d[d.pair_type == typ]
                if len(d) < 15 or d.patient.nunique() < 4:
                    continue
                rec = {"kind": kind, "class": cls, "pair_type": typ, "n": len(d), "n_patients": d.patient.nunique()}
                for nm, fn in defs.items():
                    est, lo, hi = boot_donor(d.assign(v=fn(d)), lambda x: x.v.median(), rng, 1000)
                    rec[nm], rec[nm + "_lo"], rec[nm + "_hi"] = est, lo, hi
                    per = d.assign(v=fn(d)).groupby("patient").v.median()
                    rec[nm + "_patients_pos"] = int((per > 0).sum())
                a_ = defs["measured_gain"](d).median()
                rec["amplitude_ratio_model"] = defs["model_gain"](d).median() / a_
                rec["amplitude_ratio_reference"] = defs["reference_gain"](d).median() / a_
                rows.append(rec)
    pd.DataFrame(rows).to_csv(RES_OUT / "increments.csv", index=False, float_format="%.4f")
    arows = []
    for kind in ("loop", "stripe"):
        for typ in ["PE", "PS", "enh", "rep", "other"]:
            for cls in ["all", "recurrent", "private"]:
                for nm in ["measured", "aml_refin", "aml_refin|none", "base_refin@aml", "reference"]:
                    col = "gain|" + nm if nm != "measured" else "gain|measured"
                    pos = D[(D.kind == kind) & (D.set == "pos") & (D.pair_type == typ)]
                    neg = D[(D.kind == kind) & (D.set == "A_hspc") & (D.pair_type == typ)]
                    if cls != "all":
                        pos = pos[pos.cls == cls]
                    if len(pos) < 15 or len(neg) < 15:
                        continue
                    x = pd.concat([pos, neg], ignore_index=True)
                    f = lambda y: auroc(y[y.set == "pos"][col].values, y[y.set == "A_hspc"][col].values)
                    au, lo, hi = boot_donor(x, f, rng, 300)
                    arows.append({"kind": kind, "pair_type": typ, "class": cls, "predictor": nm, "n_pos": len(pos),
                                  "n_ctl": len(neg), "auroc_gain_A_hspc": au, "lo": lo, "hi": hi})
    pd.DataFrame(arows).to_csv(RES_OUT / "auroc_by_pair_type.csv", index=False, float_format="%.4f")
    log("increments written")


def stage_criteria(D: pd.DataFrame) -> pd.DataFrame:
    """Success criteria of the loop-centric recipe (registry section 63) for the models present:
    AUROC of the gain over HSPC, AML-specific against matched shared loops (A_hspc), for private
    loops (target >= 0.70, reference model 0.59) and all loops, and the amplitude (median model
    gain / median measured gain, target >= 0.30; reference model 0.11 private) for loops and
    stripes. Donor-bootstrap intervals. The Pearson criterion (no loss against the aml_refin models,
    0.813 at 10 kb pooled) comes from aml_train_eval.py --fold N --variant extra."""
    rng = np.random.default_rng(SEED + 2)
    D = add_derived(D) if "cls" not in D else D
    rows = []
    for nm in [c for c in ["aml_refin", "aml_loopfocus", "aml_loopfocus|none", "reference"] if c in D]:
        col = "gain|" + nm
        for kind in ("loop", "stripe"):
            for cls in ("private", "all"):
                pos = D[(D.kind == kind) & (D.set == "pos")]
                neg = D[(D.kind == kind) & (D.set == "A_hspc")]
                if cls != "all":
                    pos = pos[pos.cls == cls]
                x = pd.concat([pos, neg], ignore_index=True)
                if len(pos) < 10 or len(neg) < 10:
                    continue
                au, lo, hi = boot_donor(x, lambda y: auroc(y[y.set == "pos"][col].values,
                                                           y[y.set == "A_hspc"][col].values), rng, 500)
                am, alo, ahi = boot_donor(pos, lambda y: y[col].median() / y["gain|measured"].median(), rng, 500)
                rows.append({"model": nm, "kind": kind, "class": cls, "n_pos": len(pos), "n_ctl": len(neg),
                             "auroc_gain_A_hspc": au, "auroc_lo": lo, "auroc_hi": hi,
                             "amplitude_ratio": am, "amp_lo": alo, "amp_hi": ahi,
                             "median_model_gain": pos[col].median(), "median_measured_gain": pos["gain|measured"].median(),
                             "pass_auroc_0.70": bool(au >= 0.70) if kind == "loop" and cls == "private" else np.nan,
                             "pass_amplitude_0.30": bool(am >= 0.30) if cls == "private" else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(RES_OUT / "criteria.csv", index=False, float_format="%.4f")
    print(R.round(3).to_string(index=False))
    return R


# ===================================================================== figure
def stage_figure(D: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = add_derived(D) if "cls" not in D else D
    A = pd.read_csv(RES_OUT / "auroc.csv")
    C = pd.read_csv(RES_OUT / "gain_correlation.csv")
    L = pd.read_csv(RES_OUT / "levels_by_class.csv")
    cols = {"measured": "#222222", "aml_refin": "#1f77b4", "aml_refin|none": "#7fb2d9",
            "base_refin@aml": "#2ca02c", "reference": "#d95f02"}
    lab = {"measured": "measured", "aml_refin": "model, own marks", "aml_refin|none": "model, no marks",
           "base_refin@aml": "model without AML training", "reference": "AML reference alone"}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    # (a) gain by class, loops
    a = ax[0, 0]
    cl = ["private", "shared1", "recurrent"]
    w = 0.15
    for k, nm in enumerate(["measured", "aml_refin", "aml_refin|none", "base_refin@aml", "reference"]):
        v = [L[(L.kind == "loop") & (L["class"] == c) & (L.pair_type == "all")][
            "measured_gain" if nm == "measured" else f"{nm}_gain_vs_hspc"].iloc[0] for c in cl]
        a.bar(np.arange(3) + (k - 2) * w, v, w, color=cols[nm], label=lab[nm])
    ns = [int(L[(L.kind == "loop") & (L["class"] == c) & (L.pair_type == "all")].n.iloc[0]) for c in cl]
    a.set_xticks(range(3), [f"{c}\n(n={n})" for c, n in zip(cl, ns)])
    a.axhline(0, color="k", lw=0.6)
    a.set_ylabel("median gain over measured HSPC (log2 enrichment)")
    a.set_title("a  Loops: gain over HSPC by recurrence")
    a.set_ylim(0, 0.55)
    a.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    # (b) same, stripes
    a = ax[0, 1]
    for k, nm in enumerate(["measured", "aml_refin", "aml_refin|none", "base_refin@aml", "reference"]):
        v = [L[(L.kind == "stripe") & (L["class"] == c) & (L.pair_type == "all")][
            "measured_gain" if nm == "measured" else f"{nm}_gain_vs_hspc"].iloc[0] for c in cl]
        a.bar(np.arange(3) + (k - 2) * w, v, w, color=cols[nm])
    ns = [int(L[(L.kind == "stripe") & (L["class"] == c) & (L.pair_type == "all")].n.iloc[0]) for c in cl]
    a.set_xticks(range(3), [f"{c}\n(n={n})" for c, n in zip(cl, ns)])
    a.axhline(0, color="k", lw=0.6)
    a.set_ylabel("median gain over measured HSPC (mean log O/E)")
    a.set_title("b  Stripes: gain over HSPC by recurrence")
    # (c) scatter predicted vs measured gain, loops
    a = ax[0, 2]
    d = D[(D.kind == "loop") & (D.set == "pos")]
    col = d.cls.map({"recurrent": "#b2182b", "shared1": "#999999", "private": "#2166ac"})
    a.scatter(d["gain|measured"], d["gain|aml_refin"], s=4, c=col, alpha=0.5, lw=0)
    a.axhline(0, color="k", lw=0.5)
    a.axvline(0, color="k", lw=0.5)
    r = C[(C.kind == "loop") & (C["class"] == "all") & (C.predictor == "aml_refin") & (C.target == "gain")].iloc[0]
    a.set_title(f"c  Loops, per object (red recurrent, blue private)\nSpearman {r.spearman:.2f} "
                f"[{r.spearman_lo:.2f}, {r.spearman_hi:.2f}]")
    a.set_xlabel("measured gain over HSPC")
    a.set_ylabel("predicted gain over HSPC (model, own marks)")
    # (d) AUROC of level and gain, A_hspc, per class
    for q, (kind, ttl) in enumerate((("loop", "d  Loops"), ("stripe", "e  Stripes"))):
        a = ax[1, q]
        preds = ["measured", "aml_refin", "aml_refin|none", "base_refin@aml", "reference"]
        x0 = 0
        ticks, tl = [], []
        for cls in ["all", "private", "recurrent"]:
            for score in ("level", "gain"):
                for k, nm in enumerate(preds):
                    s = A[(A.kind == kind) & (A["class"] == cls) & (A.control == "A_hspc") & (A.score == score) &
                          (A.predictor == nm)]
                    if s.empty:
                        continue
                    s = s.iloc[0]
                    a.bar(x0 + k * 0.15, s.auroc - 0.5, 0.14, bottom=0.5, color=cols[nm],
                          yerr=[[s.auroc - s.auroc_lo], [s.auroc_hi - s.auroc]], error_kw={"lw": 0.6})
                ticks.append(x0 + 0.3)
                tl.append(f"{cls}\n{score}")
                x0 += 1.0
        a.axhline(0.5, color="k", lw=0.6)
        a.set_xticks(ticks, tl, fontsize=7)
        a.set_ylabel("AUROC, AML-specific vs matched shared (A_hspc)")
        a.set_ylim(0.0, 1.0)
        a.set_title(f"{ttl}: separation from matched shared objects")
    # (f) fraction above control p90
    a = ax[1, 2]
    x0 = 0
    ticks, tl = [], []
    for kind in ("loop", "stripe"):
        for cls in ["all", "private", "recurrent"]:
            for k, nm in enumerate(preds):
                s = A[(A.kind == kind) & (A["class"] == cls) & (A.control == "A_hspc") & (A.score == "gain") &
                      (A.predictor == nm)]
                if s.empty:
                    continue
                s = s.iloc[0]
                a.bar(x0 + k * 0.15, s.frac_above_ctl_p90, 0.14, color=cols[nm],
                      yerr=[[s.frac_above_ctl_p90 - s.frac_lo], [s.frac_hi - s.frac_above_ctl_p90]],
                      error_kw={"lw": 0.6})
            ticks.append(x0 + 0.3)
            tl.append(f"{kind}\n{cls}")
            x0 += 1.0
    a.axhline(0.1, color="k", lw=0.6, ls=":")
    a.set_xticks(ticks, tl, fontsize=7)
    a.set_ylabel("fraction above 90th percentile of controls (gain)")
    a.set_title("f  AML-specific objects above matched-control p90")
    for q in ax.ravel():
        for sp in ("top", "right"):
            q.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES_OUT / "aml_loop_detection.png", dpi=150)
    log("figure written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["windows", "predict", "score", "stats", "inc", "figure"])
    ap.add_argument("--gpu", default="0", help="one local GPU index for inference")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="test patients to predict (default all ten)")
    ap.add_argument("--recipe", default="", choices=["", "loopfocus"],
                    help="loopfocus: add the loop-centric model, tables in results/aml_loop_detection_loopfocus")
    a = ap.parse_args()
    global RES_OUT
    if a.recipe == "loopfocus":
        RUNS.append(("aml_loopfocus", ["marks", "none"]))
        RES_OUT = REPO / "results/aml_loop_detection_loopfocus"
        RES_OUT.mkdir(parents=True, exist_ok=True)
        if "windows" in a.stage or "figure" in a.stage:
            raise SystemExit("--recipe loopfocus reuses the section 62 windows and has no figure stage")
    if "windows" in a.stage:
        stage_windows()
    if "predict" in a.stage:
        stage_predict(a.gpu, a.threads, a.only)
    D = None
    if "score" in a.stage:
        D = stage_score(a.only)
    if "stats" in a.stage:
        D = D if D is not None else pd.read_csv(RES_OUT / "object_scores.csv.gz", dtype={"patient": str})
        stage_stats(D)
    if "inc" in a.stage:
        stage_inc(pd.read_csv(RES_OUT / "object_scores.csv.gz", dtype={"patient": str}))
    if "figure" in a.stage:
        D = pd.read_csv(RES_OUT / "object_scores.csv.gz", dtype={"patient": str})
        stage_figure(D)
    if "criteria" in a.stage:
        stage_criteria(pd.read_csv(RES_OUT / "object_scores.csv.gz", dtype={"patient": str}))


if __name__ == "__main__":
    main()
