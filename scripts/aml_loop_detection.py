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
    python scripts/aml_loop_detection.py --recipe diff_loopfocus --stage predict score stats inc criteria --gpu 0
    python scripts/aml_loop_detection.py --recipe loopfocus2 --stage predict score --only 018 027 270 --gpu 0   # registry 70
    python scripts/aml_loop_detection.py --stage criteria2 --scores results/aml_loop_detection_loopfocus2/object_scores_partial.csv.gz --tag f10_loopfocus2
Run with the hicdiff conda env.

--recipe loopfocus (registry section 63) adds the loop-centric model all12aml_loopfocus_refres_drop_f<N>
(label aml_loopfocus; predictions of the other models are reused from the sections 62 run), reads the
objects and windows of results/aml_loop_detection and writes every table to
results/aml_loop_detection_loopfocus/; stage criteria compares it with aml_refin on the success criteria
(AUROC of private loops against the matched shared loops >= 0.70, amplitude >= 30 percent of the
measured gain).

Stage criteria2 (registry section 69, after the section 63 pilot) tests the shortcut that the gain over
the measured HSPC map allows (shared-loop controls are strong in HSPC and in the reference, so under-
predicting strong loops alone gives a high AUROC) and judges a run on confound-free criteria. It reads
any object_scores table of this script and needs only numpy / pandas / scipy:
    python scripts/aml_loop_detection.py --stage criteria2 --scores <object_scores.csv.gz> --tag <name>
Tables go to results/aml_loop_detection_criteria2/<tag>_*.csv (see stage_criteria2).
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
# models of the extra variant of aml_train_eval (section 63 loop-centric recipe; section 66 differential
# mark channels with all-cell objects, and its ablations)
DIFF_LABELS = ("aml_diff_loopfocus", "aml_diff", "aml_diffhspc", "aml_diff_loopfocusaml", "aml_allobj",
               "aml_diffmeth_loopfocus")
# section 70: controls matched on the reference / HSPC enrichment, shared loops weak, marks-dependence margin
LOOP2_LABELS = ("aml_loopfocus2",)
DIFF2_LABELS = ("aml_diff_loopfocus2", "aml_diff_loopfocus2aml", "aml_diffmeth_loopfocus2",
                "aml_diffhspc_loopfocus2")
METH_RESID_LABELS = ("aml_meth_refin", "aml_refres")     # AML+Meth and residual-reference models of the fold
SEED_LABELS = ("aml_refin_s1", "aml_refin_s2", "aml_diff_s1", "aml_diff_s2")   # seed replicates (registry 72)
EXTRA_LABELS = ("aml_loopfocus", *LOOP2_LABELS, *DIFF_LABELS, *DIFF2_LABELS, *METH_RESID_LABELS, *SEED_LABELS)
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
    te.VARIANT = "seeds" if label in SEED_LABELS else ("extra" if label in EXTRA_LABELS else "")
    m = next(x for x in te.models(fold, RES) if x["label"] == label)
    te.VARIANT = ""
    if label in EXTRA_LABELS and not te.ready(m["run"]):
        log(f"f{fold} {label} {p}: run {m['run']} not finished, skipped")
        return
    out = OUTD / f"f{fold}" / f"{label}__xu_{p}"
    if all((out / f"{k}.npz").exists() for k in sets):
        return
    try:
        ck = te.ckpt(m["run"])
        if not (Path(ck) / "config.yaml").exists():
            raise FileNotFoundError(f"{ck}/config.yaml")
    except (FileNotFoundError, OSError, ValueError, StopIteration) as e:    # a model that is not there is skipped, never a crash
        log(f"f{fold} {label} {p}: no checkpoint for {m['run']} ({e}), skipped")
        return
    mk = list(s["marks"]) + (["Meth"] if m.get("meth") and s["meth"] is not None else [])
    keep = [f"marks={','.join(mk)}"] + (["none="] if "none" in sets else [])
    cmd = [te.PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml", "--checkpoint", ck,
           "--cell", s["cell"], "--bigwig-dir", s["dir"], "--target", s["target"], "--tta",
           "--assay", s["assay"], "--binsize", RES, "--device", "cuda",
           "--regions", OUTD / "windows" / f"xu_{p}.tsv", "--out-dir", out,
           "--save-pred", str(out / "{set}.npz"), "--keep-sets", *keep]
    if m["lab"]:
        cmd += ["--lab", s["lab"]]
    if m.get("diff"):
        cmd += ["--diff-group", te.diff_group_of(s)]
    if m.get("meth") and s["meth"] is not None:
        cmd += ["--extra-bigwig-dir", s["meth"]]
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
                        nm = f"{label}|{k}" if (label in ("aml_refin", *EXTRA_LABELS) and k == "none") else label
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
PRED_EXTRA = [x for lab in EXTRA_LABELS for x in (lab, lab + "|none")]
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
    for nm in [c for c in ["aml_refin", *PRED_EXTRA, "reference"] if c in D]:
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
    # enhancer-like stripes (pair_type enh): Spearman of the model's gain over HSPC with the measured
    # gain; criterion of section 66: higher than that of the reference map alone (donor bootstrap)
    from scipy.stats import spearmanr
    enh = D[(D.kind == "stripe") & (D.set == "pos") & (D.pair_type == "enh")]
    if len(enh) >= 10:
        def sp(x, col):
            ok = np.isfinite(x[col]) & np.isfinite(x["gain|measured"])
            if ok.sum() < 10 or x[col][ok].std() == 0:
                return np.nan
            return spearmanr(x[col][ok], x["gain|measured"][ok])[0]
        base = None
        for nm in ["reference"] + [c for c in ["aml_refin", *PRED_EXTRA] if c in D]:
            col = "gain|" + nm
            r_, lo, hi = boot_donor(enh, lambda y, col=col: sp(y, col), rng, 500)
            if nm == "reference":
                base = r_
            rows.append({"model": nm, "kind": "stripe", "class": "enh_gain_spearman", "n_pos": len(enh),
                         "gain_spearman": r_, "gain_spearman_lo": lo, "gain_spearman_hi": hi,
                         "pass_enh_gain_above_reference": bool(r_ > base) if nm != "reference" and base is not None
                         and np.isfinite(r_) else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(RES_OUT / "criteria.csv", index=False, float_format="%.4f")
    print(R.round(3).to_string(index=False))
    return R


# ===================================================================== criteria2
CRIT2_OUT = REPO / "results/aml_loop_detection_criteria2"
CAL_REF, CAL_DIST = 0.15, 0.10        # calipers of the matching: log2 enrichment (reference, HSPC), log10 distance


def match_controls(P: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """One-to-one nearest-neighbour matching without replacement within each donor: positives P to
    candidate controls C on the AML reference enrichment, the measured HSPC enrichment (both within
    CAL_REF), log10 distance (within CAL_DIST) and CTCF orientation class (exact). Greedy by increasing
    scaled squared distance. Returns rows (pos, ctl) of integer row labels."""
    out = []
    for p, g in P.groupby("patient"):
        c = C[C.patient == p].drop_duplicates(["chrom", "a_lo", "b_lo"])
        if c.empty:
            continue
        dr = (g.reference.values[:, None] - c.reference.values[None]) / CAL_REF
        dh = (g.hspc.values[:, None] - c.hspc.values[None]) / CAL_REF
        dd = (g.dist_log10.values[:, None] - c.dist_log10.values[None]) / CAL_DIST
        ok = (np.abs(dr) <= 1) & (np.abs(dh) <= 1) & (np.abs(dd) <= 1) & (
            g._orient.astype(str).to_numpy()[:, None] == c._orient.astype(str).to_numpy()[None])
        cost = np.where(ok, dr ** 2 + dh ** 2 + dd ** 2, np.inf)
        used_p, used_c = set(), set()
        for flat in np.argsort(cost, axis=None):
            i, j = divmod(int(flat), cost.shape[1])
            if not np.isfinite(cost[i, j]):
                break
            if i in used_p or j in used_c:
                continue
            used_p.add(i)
            used_c.add(j)
            out.append((g.index[i], c.index[j]))
    return pd.DataFrame(out, columns=["pos", "ctl"])


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    s = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / s) if s > 0 else np.nan


def _resid_rank(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y)), X])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def _spearman(x, y, covs=None) -> float:
    from scipy.stats import rankdata
    ok = np.isfinite(x) & np.isfinite(y)
    if covs is not None:
        ok &= np.all(np.isfinite(covs), 1)
    if ok.sum() < 10:
        return np.nan
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    if covs is not None:
        Z = np.column_stack([rankdata(covs[ok, k]) for k in range(covs.shape[1])])
        rx, ry = _resid_rank(rx, Z), _resid_rank(ry, Z)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def stage_criteria2(D: pd.DataFrame, tag: str, nboot: int = 1000) -> None:
    """Confound-free criteria for the AML-specific loops (registry section 69).

    Why: the success criterion of section 63 scores the gain (model minus the MEASURED HSPC map) of
    AML-specific (private) loops against shared loops chosen for a strong HSPC signal. The HSPC value alone
    separates the two classes (AUROC about 0.99), so any model that tracks the HSPC / reference level less
    closely scores high without patient-specific information. criteria2 therefore
      1. builds controls matched on the reference and HSPC enrichment: A_ref (shared-loop candidates
         A_hspc, A_full, A_do pooled) and A_refrand (random distance-matched pairs B_rand, same matching;
         the only pool that overlaps the positives in reference / HSPC level), and reports the match quality;
      2. scores level, gain over HSPC and RESIDUAL (model minus the AML reference map, the quantity the
         residual model learns) with and without the patient's own marks (partner model '<m>|none');
      3. correlates the predicted with the measured residual among private loops (plain and partial on
         reference and HSPC ranks) and checks calibration at random pairs.
    Intervals: donor bootstrap (resampling donors) when >= 6 donors, otherwise resampling objects within
    donor (ci_mode column). Pass criteria (private loops, per model with a '|none' partner):
      C1 marks effect  AUROC(residual, A_refrand) with marks minus without >= 0.03 and CI lower bound >= 0
      C2 AUROC(residual, A_refrand) >= 0.70
      C3 Spearman(predicted residual, measured residual) among private loops >= 0.20
      C4 no inflation at random pairs: median(model) - median(measured) at B_rand <= 0.05 and the same for
         the 90th percentile <= 0.10 (log2 enrichment)
    The section 63 criterion (gain AUROC vs A_hspc >= 0.70) is reported as C0, not part of the verdict.
    """
    from scipy.stats import rankdata
    D = add_derived(D) if "cls" not in D else D
    F = pd.read_csv(FEATS, usecols=["patient", "kind", "set", "pair_id", "_orient", "dist_log10"],
                    dtype={"patient": str})
    F["patient"] = F.patient.map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)
    D = D.merge(F, on=["patient", "kind", "set", "pair_id"], how="left")
    CRIT2_OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 3)
    L = D[D.kind == "loop"].reset_index(drop=True)
    ST = D[D.kind == "stripe"].reset_index(drop=True)
    base = [m for m in ["aml_refin", "base_refin@aml", *EXTRA_LABELS] if m in L]
    models = [m for m in base] + [m + "|none" for m in base if m + "|none" in L] + ["reference"]
    donors = sorted(L.patient.unique())
    ci_mode = "donor" if len(donors) >= 6 else "object"
    log(f"criteria2 '{tag}': {len(donors)} donors, models {models}, CI by {ci_mode}")

    # ---- control sets: (positive row, control row) pairs; unmatched sets use all private positives
    pos = L[(L.set == "pos") & (L.cls == "private")]
    sets = {}
    for nm, ctl_sets in (("A_hspc", ["A_hspc"]), ("B_rand", ["B_rand"])):
        c = L[L.set.isin(ctl_sets)]
        sets[nm] = (pos.index.values, c.index.values)
    mq = []
    for nm, pool in (("A_ref", ["A_hspc", "A_full", "A_do"]), ("A_refrand", ["B_rand"])):
        C = L[L.set.isin(pool)]
        m = match_controls(pos, C)
        sets[nm] = (m.pos.values, m.ctl.values)
        pp, cc = L.loc[m.pos], L.loc[m.ctl]
        full = L[L.set == ("A_hspc" if nm == "A_ref" else "B_rand")]
        rec = {"tag": tag, "control_set": nm, "pool": "+".join(pool), "n_pos_private": len(pos),
               "n_matched": len(m), "frac_matched": len(m) / len(pos)}
        for v in ("reference", "hspc", "dist_log10", "measured"):
            rec[f"smd_before_{v}"] = _smd(pos[v].values, full[v].values)
            rec[f"smd_after_{v}"] = _smd(pp[v].values, cc[v].values) if len(m) > 5 else np.nan
            rec[f"mean_pos_{v}"] = pp[v].mean() if len(m) else np.nan
            rec[f"mean_ctl_{v}"] = cc[v].mean() if len(m) else np.nan
        for nm2, v in (("-hspc", -L.hspc), ("-reference", -L.reference)):
            rec[f"auroc_{nm2}_before"] = auroc(v[pos.index].values, v[full.index].values)
            rec[f"auroc_{nm2}_after"] = auroc(v[m.pos].values, v[m.ctl].values) if len(m) > 5 else np.nan
        mq.append(rec)
    pd.DataFrame(mq).to_csv(CRIT2_OUT / f"{tag}_match_quality.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(mq).round(3).T.to_string())

    # ---- score vectors
    score = {}
    for nm in models:
        score[(nm, "level")] = L[nm].values
        score[(nm, "gain")] = (L[nm] - L.hspc).values
        score[(nm, "resid")] = (L[nm] - L.reference).values
        if nm + "@hspc" in L:      # the model's own HSPC counterpart (run on HSPC reference, same marks)
            score[(nm, "gain_modelhspc")] = (L[nm] - L[nm + "@hspc"]).values
    for nm, v in (("-hspc", -L.hspc), ("-reference", -L.reference), ("-measured", -L.measured)):
        score[(nm, "level")] = v.values       # trivial baselines: the confound alone
    pos_idx = pos.index.values
    cov = L[["reference", "hspc"]].values
    meas_res = (L.measured - L.reference).values
    meas_gain = L["gain|measured"].values
    enh = ST[(ST.set == "pos") & (ST.pair_type == "enh")]

    def by_donor(idx, frame):
        pdn = frame.patient.values
        return {d: idx[pdn[idx] == d] for d in np.unique(pdn[idx])}

    pos_by = by_donor(pos_idx, L)
    set_by = {k: (by_donor(v[0], L), by_donor(v[1], L), v) for k, v in sets.items()}   # pos / ctl per donor
    enh_by = by_donor(enh.index.values, ST)
    Sdat = {"level": {}, "resid": {}, "gain": {}}
    for nm in models:
        Sdat["resid"][nm] = (ST[nm] - ST.reference).values
        Sdat["gain"][nm] = (ST[nm] - ST.hspc).values
    st_meas = {"resid": (ST.measured - ST.reference).values, "gain": ST["gain|measured"].values}

    def draw(mode: str):
        """Row indices per set and for the private positives under one resample."""
        if mode == "point":
            return {k: v[2] for k, v in set_by.items()}, pos_idx, enh.index.values
        if mode == "donor":
            pick = rng.choice(donors, len(donors))
            cat = lambda dct, ds: np.concatenate([dct[d] for d in ds if d in dct])
            sd = {k: (cat(v[0], pick), cat(v[1], pick)) for k, v in set_by.items()}
            return sd, cat(pos_by, pick), cat(enh_by, pick) if enh_by else np.array([], int)
        rs = lambda a: a[rng.integers(0, len(a), len(a))] if len(a) else a
        sd = {k: (np.concatenate([rs(v[0][d]) for d in v[0]]), np.concatenate([rs(v[1][d]) for d in v[1]]))
              for k, v in set_by.items()}
        return sd, np.concatenate([rs(a) for a in pos_by.values()]), \
            np.concatenate([rs(a) for a in enh_by.values()]) if enh_by else np.array([], int)

    def stats(mode: str) -> dict:
        sd, pi, ei = draw(mode)
        r = {}
        for (nm, sc), v in score.items():
            for k, (a, b) in sd.items():
                r[("auroc", nm, sc, k)] = auroc(v[a], v[b])
        for nm in models:
            for k in sets:
                for sc in ("level", "gain", "resid"):
                    if nm + "|none" in models and not nm.endswith("|none"):
                        r[("dmarks", nm, sc, k)] = r[("auroc", nm, sc, k)] - r[("auroc", nm + "|none", sc, k)]
            if nm != "reference":
                r[("sp_resid", nm, "private", "")] = _spearman(score[(nm, "resid")][pi], meas_res[pi])
                r[("sp_resid_partial", nm, "private", "")] = _spearman(score[(nm, "resid")][pi], meas_res[pi], cov[pi])
            r[("sp_gain", nm, "private", "")] = _spearman(score[(nm, "gain")][pi], meas_gain[pi])
            if len(ei) > 10 and nm in Sdat["gain"]:
                r[("sp_stripe_gain_enh", nm, "enh", "")] = _spearman(Sdat["gain"][nm][ei], st_meas["gain"][ei])
                r[("sp_stripe_resid_enh", nm, "enh", "")] = _spearman(Sdat["resid"][nm][ei], st_meas["resid"][ei])
        for nm in models:
            if ("sp_stripe_gain_enh", "reference", "enh", "") in r and nm != "reference" and \
                    ("sp_stripe_gain_enh", nm, "enh", "") in r:
                r[("d_stripe_gain_enh_vs_reference", nm, "enh", "")] = \
                    r[("sp_stripe_gain_enh", nm, "enh", "")] - r[("sp_stripe_gain_enh", "reference", "enh", "")]
        return r

    point = stats("point")
    boots = {"donor": [], "object": []}
    for mode in boots:
        for _ in range(nboot):
            boots[mode].append(stats(mode))
    rows = []
    for key, est in point.items():
        row = {"tag": tag, "stat": key[0], "model": key[1], "score": key[2], "control_set": key[3], "estimate": est}
        for mode in boots:
            v = np.array([b.get(key, np.nan) for b in boots[mode]], float)
            v = v[np.isfinite(v)]
            row[f"{mode}_lo"], row[f"{mode}_hi"] = (np.percentile(v, [2.5, 97.5]) if len(v) > 50 else (np.nan, np.nan))
        row["ci_mode"] = ci_mode
        row["lo"], row["hi"] = row[f"{ci_mode}_lo"], row[f"{ci_mode}_hi"]
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(CRIT2_OUT / f"{tag}_stats.csv", index=False, float_format="%.4f")

    # ---- per-donor AUROC with and without marks (paired)
    pdr = []
    for nm in base:
        if nm + "|none" not in L:
            continue
        for k, (pd_, cd_, _) in set_by.items():
            for sc in ("level", "gain", "resid"):
                for d in donors:
                    if d in pd_ and d in cd_:
                        a, b = pd_[d], cd_[d]
                        w, n0 = auroc(score[(nm, sc)][a], score[(nm, sc)][b]), auroc(score[(nm + "|none", sc)][a], score[(nm + "|none", sc)][b])
                        pdr.append({"tag": tag, "model": nm, "control_set": k, "score": sc, "donor": d,
                                    "n_pos": len(a), "n_ctl": len(b), "auroc_marks": w, "auroc_none": n0,
                                    "delta": w - n0})
    pd.DataFrame(pdr).to_csv(CRIT2_OUT / f"{tag}_per_donor_marks.csv", index=False, float_format="%.4f")

    # ---- calibration at random pairs and controls
    inf = []
    q = lambda x, p: float(np.nanpercentile(x, p))
    for s_ in ("pos", "A_hspc", "B_rand"):
        d = L[(L.set == s_) & ((L.cls == "private") if s_ == "pos" else True)]
        for nm in ["measured", *models]:
            v = d[nm].values
            rec = {"tag": tag, "set": s_, "model": nm, "n": len(d), "median_level": np.nanmedian(v),
                   "p90_level": q(v, 90), "median_minus_hspc": np.nanmedian(v - d.hspc.values),
                   "median_minus_reference": np.nanmedian(v - d.reference.values)}
            inf.append(rec)
    I = pd.DataFrame(inf)
    I.to_csv(CRIT2_OUT / f"{tag}_calibration.csv", index=False, float_format="%.4f")

    # ---- linear adjustment of the scores for reference / HSPC level (descriptive; the positives and
    # shared controls barely overlap in these covariates, so A_hspc values extrapolate)
    adj = []
    for k in ("A_hspc", "A_refrand", "B_rand"):
        a, b = sets[k][0], sets[k][1]
        idx = np.concatenate([np.unique(a), np.unique(b)])
        X = np.column_stack([L.reference, L.hspc, L.reference ** 2, L.hspc ** 2, L.reference * L.hspc])[idx]
        lab = np.r_[np.ones(len(np.unique(a))), np.zeros(len(np.unique(b)))]
        for (nm, sc), v in score.items():
            y = v[idx]
            ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
            if ok.sum() < 20:
                continue
            e = _resid_rank(rankdata(y[ok]), X[ok])
            adj.append({"tag": tag, "control_set": k, "model": nm, "score": sc,
                        "auroc_raw": auroc(y[ok][lab[ok] == 1], y[ok][lab[ok] == 0]),
                        "auroc_adjusted": auroc(e[lab[ok] == 1], e[lab[ok] == 0])})
    pd.DataFrame(adj).to_csv(CRIT2_OUT / f"{tag}_adjusted.csv", index=False, float_format="%.4f")

    # ---- criteria
    def get(stat, nm, sc, k=""):
        s_ = R[(R.stat == stat) & (R.model == nm) & (R.score == sc) & (R.control_set == k)]
        return s_.iloc[0] if len(s_) else None

    out = []
    for nm in base:
        rec = {"tag": tag, "model": nm, "n_donors": len(donors), "ci_mode": ci_mode,
               "n_pos_private": len(pos), "n_matched_A_refrand": len(sets["A_refrand"][0]),
               "n_matched_A_ref": len(sets["A_ref"][0])}
        g0 = get("auroc", nm, "gain", "A_hspc")
        rec["C0_gain_auroc_A_hspc"], rec["C0_pass"] = g0.estimate, bool(g0.estimate >= 0.70)
        a2 = get("auroc", nm, "resid", "A_refrand")
        rec["C2_resid_auroc_A_refrand"], rec["C2_lo"], rec["C2_hi"] = a2.estimate, a2.lo, a2.hi
        rec["C2_pass"] = bool(a2.estimate >= 0.70)
        a2b = get("auroc", nm, "resid", "A_ref")
        rec["info_resid_auroc_A_ref_shared"] = a2b.estimate if a2b is not None else np.nan
        rec["info_level_auroc_B_rand"] = get("auroc", nm, "level", "B_rand").estimate
        rec["info_level_auroc_B_rand_reference"] = get("auroc", "reference", "level", "B_rand").estimate
        if nm + "|none" in L:
            d1 = get("dmarks", nm, "resid", "A_refrand")
            rec["C1_marks_delta_resid_A_refrand"], rec["C1_lo"], rec["C1_hi"] = d1.estimate, d1.lo, d1.hi
            rec["C1_pass"] = bool(d1.estimate >= 0.03 and d1.lo >= 0)
            d0 = get("dmarks", nm, "gain", "A_hspc")
            rec["info_marks_delta_gain_A_hspc"], rec["info_marks_delta_gain_lo"], rec["info_marks_delta_gain_hi"] = d0.estimate, d0.lo, d0.hi
            d3 = get("dmarks", nm, "level", "A_hspc")
            rec["info_marks_delta_level_A_hspc"] = d3.estimate
            dr = get("dmarks", nm, "resid", "A_hspc")
            rec["info_marks_delta_resid_A_hspc"] = dr.estimate
        else:
            rec["C1_pass"] = np.nan
        sp = get("sp_resid", nm, "private")
        rec["C3_resid_spearman"], rec["C3_lo"], rec["C3_hi"] = sp.estimate, sp.lo, sp.hi
        rec["C3_pass"] = bool(sp.estimate >= 0.20)
        spp = get("sp_resid_partial", nm, "private")
        rec["info_resid_spearman_partial"], rec["info_partial_lo"], rec["info_partial_hi"] = spp.estimate, spp.lo, spp.hi
        ib = I[(I.set == "B_rand") & (I.model == nm)].iloc[0]
        im = I[(I.set == "B_rand") & (I.model == "measured")].iloc[0]
        rec["C4_median_excess_B_rand"] = ib.median_level - im.median_level
        rec["C4_p90_excess_B_rand"] = ib.p90_level - im.p90_level
        rec["C4_pass"] = bool(rec["C4_median_excess_B_rand"] <= 0.05 and rec["C4_p90_excess_B_rand"] <= 0.10)
        st = get("sp_stripe_gain_enh", nm, "enh")
        if st is not None:
            rec["info_stripe_enh_gain_spearman"] = st.estimate
            dd = get("d_stripe_gain_enh_vs_reference", nm, "enh")
            if dd is not None:
                rec["info_stripe_enh_minus_reference"], rec["info_stripe_lo"], rec["info_stripe_hi"] = dd.estimate, dd.lo, dd.hi
        flags = [rec.get(c) for c in ("C1_pass", "C2_pass", "C3_pass", "C4_pass")]
        rec["criteria2_pass"] = bool(all(f is True or f == True for f in flags)) if all(
            isinstance(f, (bool, np.bool_)) for f in flags) else np.nan
        out.append(rec)
    C = pd.DataFrame(out)
    C.to_csv(CRIT2_OUT / f"{tag}_criteria2.csv", index=False, float_format="%.4f")
    print(C.round(3).T.to_string())
    log("criteria2 written")


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
    ap.add_argument("--scores", default="", help="criteria2: object_scores table (default: RES_OUT/object_scores.csv.gz)")
    ap.add_argument("--tag", default="", help="criteria2: prefix of the output tables")
    ap.add_argument("--nboot", type=int, default=1000, help="criteria2: bootstrap replicates")
    ap.add_argument("--only", nargs="*", help="test patients to predict (default all ten)")
    ap.add_argument("--recipe", default="", choices=["", "loopfocus", "diff_loopfocus", "loopfocus2", "diff_loopfocus2", "meth_resid", "seeds"],
                    help="loopfocus: add the loop-centric model, tables in results/aml_loop_detection_loopfocus; "
                         "diff_loopfocus (registry section 66): add the differential-mark models "
                         f"{', '.join(DIFF_LABELS)} (those whose runs are finished) and the loop-centric "
                         "model, tables in results/aml_loop_detection_diff_loopfocus")
    a = ap.parse_args()
    global RES_OUT
    if a.recipe:
        RUNS.append(("aml_loopfocus", ["marks", "none"]))
        if a.recipe == "seeds":
            RUNS.extend((lab, ["marks", "none"]) for lab in ("aml_diff", *SEED_LABELS))
        if a.recipe == "meth_resid":
            RUNS.extend((lab, ["marks", "none"]) for lab in METH_RESID_LABELS)
        if a.recipe.endswith("loopfocus2"):
            RUNS.extend((lab, ["marks", "none"]) for lab in LOOP2_LABELS)
        if a.recipe == "diff_loopfocus":
            RUNS.extend((lab, ["marks", "none"]) for lab in DIFF_LABELS)
        if a.recipe == "diff_loopfocus2":
            RUNS.extend((lab, ["marks", "none"]) for lab in DIFF2_LABELS)
        RES_OUT = REPO / f"results/aml_loop_detection_{a.recipe}"
        RES_OUT.mkdir(parents=True, exist_ok=True)
        if "windows" in a.stage or "figure" in a.stage:
            raise SystemExit(f"--recipe {a.recipe} reuses the section 62 windows and has no figure stage")
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
    if "criteria2" in a.stage:
        src = Path(a.scores) if a.scores else RES_OUT / "object_scores.csv.gz"
        stage_criteria2(pd.read_csv(src, dtype={"patient": str}), a.tag or src.parent.name, a.nboot)


if __name__ == "__main__":
    main()
