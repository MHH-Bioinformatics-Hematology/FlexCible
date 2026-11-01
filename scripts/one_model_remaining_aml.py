"""AML-specific loops and stripes in the ten held-out patients on the release checkpoint all12one_fullrd
(registry section 98; protocol of sections 62 / 72, scripts/aml_competitors_loops.py).

Objects, 192-bin windows, the enrichment estimators, the detection rule (share of the AML-specific objects
above the 90th percentile of the same predictor at the B_rand pairs of the same donors), the A_refrand matched
controls, the donor bootstrap and the Spearman statistics are those of `aml_competitors_loops.py`, whose
functions are called unchanged. The release checkpoint is the 10 kb view `all12one_fullrd_10kb`, one
network that has seen no AML sample. Patient marks CTCF, H3K27ac, H3K27me3 (1360 and 1916 lack CTCF, absent),
all other marks and methylation absent, cohesin fraction 1. Settings (columns of the score table):

  rel_aml     reference input = the AML reference map of the fold's training patients (AML data used as
              input only; the setting of `base_refin@aml`, paper 48 % / 67 %)
  rel_c12     reference input = consensus12, the mean of the 12 training cell lines (no AML; the setting of
              the `c12_*` models, paper 40-42 % / 52-57 %)
  rel_noref   reference plane at 0 (`flexcible predict --no-reference`), the marks-only use
  honest_old  the dedicated no-reference model all12honest_drop_10kb of the paper (same windows), the baseline of
              rel_noref
  <name>@hspc the same setting on the HSPC CD34+ inputs (rel_c12, rel_noref, honest_old), for the gain over HSPC

    # remote GPU host (HICMULTI_HM = data root with the AML inputs; one patient at a time):
    python scripts/one_model_remaining_aml.py predict --patients 018 027 ...
    # local:
    python scripts/one_model_remaining_aml.py score
    python scripts/one_model_remaining_aml.py stats --nboot 500
    python scripts/one_model_remaining_aml.py windows      # structure measures of the object windows (Table S13)
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

HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
OUTD = HM / "runs/aml_one_model"
WINDOWS = HM / "runs/aml_loop_detection/windows"
MR = HM / "runs/multicell_reference"
VIEW = "all12one_fullrd_10kb"
OLD = "all12honest_drop_10kb"
PATIENTS = ["018", "027", "270", "168", "546", "629", "1360", "472", "773", "1916"]
FOLD = {"018": 10, "027": 10, "270": 10, "168": 11, "546": 11, "629": 11, "1360": 11,
        "472": 12, "773": 12, "1916": 12}
RES = 10000
RES_OUT = REPO / "results/one_model/remaining/aml"
OLD_SCORES = REPO / "results/aml_competitors_loops/scores.csv.gz"
NEW_COLS = ["rel_aml", "rel_c12", "rel_noref", "honest_old", "rel_c12@hspc", "rel_noref@hspc", "honest_old@hspc"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _run(cmd, out, env):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predict.log", "a") as fh:
        fh.write("[cmd] " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=env)


# ---------------------------------------------------------------- predict
def stage_predict(a):
    py = os.environ.get("HICMULTI_PY", sys.executable)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    c12 = HM / f"crosscell_oe/consensus12_proper_{RES // 1000}kb.cool"
    for p in a.patients:
        fold = FOLD[p]
        for group, cell, d, tgt in (
                ("patient", "amlxu" + p, HM / "aml/primary/inputs" / f"xu_{p}", HM / f"aml/oe/xu_{p}_proper_10kb.cool"),
                ("hspc", "amlxuhspc341", HM / "aml/primary/inputs/hspc_cd34", HM / "aml/oe/xu_HSPC-341_proper_10kb.cool")):
            marks = sorted(b.stem for b in d.glob("*.bigwig"))
            mk = ",".join(marks)
            jobs = []
            if group == "patient":
                aml_ref = HM / f"aml/primary/consensus/amltrain_f{fold}_all_10kb.cool"
                jobs.append(("rel_aml", MR / VIEW, aml_ref, [f"marks={mk}"], ["marks"]))
            jobs.append(("rel_c12", MR / VIEW, c12, [f"marks={mk}", f"marks_noref={mk}"], ["marks", "marks_noref"]))
            if a.old:
                jobs.append(("honest_old", MR / OLD, None, [f"marks={mk}"], ["marks"]))
            for lab, run, ref, ks, sets in jobs:
                out = OUTD / "ours" / f"{lab}@{group}__xu_{p}"
                if all((out / f"{s}.npz").exists() for s in sets):
                    continue
                ck = run / "checkpoint-best"
                cmd = [py, "scripts/crosscell_predict.py", "--config", ck / "config.yaml", "--checkpoint", ck,
                       "--cell", cell, "--bigwig-dir", d, "--target", tgt, "--tta", "--assay", "hic",
                       "--binsize", RES, "--device", a.device, "--regions", WINDOWS / f"xu_{p}.tsv",
                       "--out-dir", out, "--save-pred", str(out / "{set}.npz"), "--keep-sets", *ks,
                       "--set-batch", a.set_batch]
                if ref is not None:
                    cmd += ["--reference", ref]
                rc = _run(cmd, out, env)
                log(f"{lab} {group} {p} rc={rc}")


# ---------------------------------------------------------------- score
def sources(p):
    S = {}
    for lab, name in (("rel_aml", "rel_aml"), ("rel_c12", "rel_c12"), ("honest_old", "honest_old")):
        for grp, suf in (("patient", ""), ("hspc", "@hspc")):
            f = OUTD / "ours" / f"{lab}@{grp}__xu_{p}" / "marks.npz"
            if f.exists():
                S[name + suf] = ("npz", f)
    for grp, suf in (("patient", ""), ("hspc", "@hspc")):
        f = OUTD / "ours" / f"rel_c12@{grp}__xu_{p}" / "marks_noref.npz"
        if f.exists():
            S["rel_noref" + suf] = ("npz", f)
    return S


def stage_score(a):
    import aml_competitors_loops as acl
    acl.sources = sources
    acl.PARTS = OUTD / "scores"
    acl.KEEP = ["item"]                         # new columns only; joined to the old table afterwards
    for p in a.patients or PATIENTS:
        acl.stage_score([p])
    parts = [pd.read_csv(acl.PARTS / f"scores_{p}.csv.gz", dtype={"patient": str}) for p in PATIENTS
             if (acl.PARTS / f"scores_{p}.csv.gz").exists()]
    N = pd.concat(parts, ignore_index=True)
    N = N[["item"] + [c for c in N.columns if c in NEW_COLS or c.replace("#px", "") in NEW_COLS]]
    O = pd.read_csv(OLD_SCORES, dtype={"patient": str})
    M = O.merge(N, on="item", how="left")
    RES_OUT.mkdir(parents=True, exist_ok=True)
    M.to_csv(RES_OUT / "scores.csv.gz", index=False, float_format="%.5f")
    log(f"merged {len(M)} objects, new columns {[c for c in N.columns if c != 'item']}")


# ---------------------------------------------------------------- stats
def stage_stats(a):
    import aml_competitors_loops as acl
    acl.RES_OUT = RES_OUT
    acl.BLOCKS = {
        "reference": ["measured", "reference", "consensus12"],
        "no AML in training (paper models)": ["base_refin@aml", "c12_refin", "c12_refres", "c12_diffcells"],
        "release checkpoint all12one_fullrd (no AML in training)": ["rel_aml", "rel_c12", "rel_noref"],
        "dedicated no-reference model": ["honest_old"],
        "competitors (published models, no AML in training)": ["corigami_q", "corigami_raw", "epiphany_zero",
                                                               "epiphany_mean", "akita"],
        "trained with AML patients (not comparable to competitors)": ["aml_refin", "aml_diff", "aml_diffhspc"]}
    acl.LABEL.update({"rel_aml": "release, AML reference input", "rel_c12": "release, consensus12",
                      "rel_noref": "release, marks only (no reference)",
                      "honest_old": "dedicated no-reference model (all12honest_drop)"})
    acl.NBOOT_ = a.nboot
    for score in ("level", "gain"):
        acl.stage_stats(a.nboot, str(RES_OUT / "scores.csv.gz"), score)
        acl.stage_summary("" if score == "level" else "_gain")


# ---------------------------------------------------------------- structure measures around the AML objects (Table S13)
def stage_windows(a):
    """The 10 kb object windows of the eight patients with CTCF data, one pixel mask for all methods, nine structure
    measures (scripts/structure_scores_vs_akita.py stage aml, called unchanged) with the release rows added to the
    paper rows (c12_refin / c12_refres, C.Origami, Epiphany, Akita, consensus12)."""
    import structure_scores_vs_akita as sv
    out = RES_OUT / "windows"
    out.mkdir(parents=True, exist_ok=True)
    sv.OUT = out
    for nm, f in (("rel_c12", "rel_c12@patient__xu_{p}/marks.npz"), ("rel_noref", "rel_c12@patient__xu_{p}/marks_noref.npz"),
                  ("rel_aml", "rel_aml@patient__xu_{p}/marks.npz"), ("honest_old", "honest_old@patient__xu_{p}/marks.npz")):
        sv.AML_SRC[nm] = ("npz", str(OUTD / "ours" / f))
    sv.AML_LABEL.update({"rel_c12": "release, consensus12", "rel_noref": "release, marks only", "rel_aml": "release, AML reference",
                         "honest_old": "dedicated no-reference model"})

    def tables(df):
        pm = df.groupby(["patient", "method"])[sv.AML_METRICS].median().reset_index()
        nw = df.groupby(["patient", "method"]).size().rename("n_windows").reset_index()
        pm = pm.merge(nw, on=["patient", "method"])
        pm.to_csv(out / "aml_windows_per_patient.csv", index=False, float_format="%.4f")
        rows = []
        for m, g in pm.groupby("method"):
            rec = {"method": m, "label": sv.AML_LABEL.get(m, m), "n_patients": int(g.patient.nunique()),
                   "n_windows": int(g.n_windows.sum())}
            for k in sv.AML_METRICS:
                rec[f"{k}_median"] = float(g[k].median())
                rec[f"{k}_min"], rec[f"{k}_max"] = float(g[k].min()), float(g[k].max())
            rows.append(rec)
        pd.DataFrame(rows).to_csv(out / "aml_windows_summary.csv", index=False, float_format="%.4f")
        wide = {k: pm.pivot(index="patient", columns="method", values=k) for k in sv.AML_METRICS}
        ww = {k: df.pivot_table(index=["patient", "window"], columns="method", values=k) for k in sv.AML_METRICS}
        prow = []
        for o in ("c12_refin", "c12_refres", "rel_c12", "rel_noref"):
            for other in ("akita", "corigami_q", "epiphany_mean", "epiphany_zero", "consensus12", "c12_refres"):
                if other == o:
                    continue
                for k in sv.AML_METRICS:
                    d = (wide[k][o] - wide[k][other]).dropna()
                    wd = (ww[k][o] - ww[k][other]).dropna()
                    prow.append({"ours": o, "other": other, "metric": k, "n_patients": int(d.size),
                                 "patient_median_diff": float(d.median()), "patients_ours_ahead": int((d > 0).sum()),
                                 "n_windows": int(wd.size), "window_median_diff": float(wd.median()),
                                 "window_win_frac": float((wd > 0).sum() / max(1, (wd != 0).sum()))})
        pd.DataFrame(prow).to_csv(out / "aml_windows_paired.csv", index=False, float_format="%.4f")

    sv.aml_tables = tables
    sv.aml_eval_tables = lambda: None
    os.nice(10)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    sv.stage_aml(min(a.threads, 6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["predict", "score", "stats", "windows"])
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--set-batch", type=int, default=2)
    ap.add_argument("--old", action="store_true", help="also predict the dedicated no-reference model")
    ap.add_argument("--nboot", type=int, default=500)
    a = ap.parse_args()
    if a.stage == "predict":
        a.patients = a.patients or PATIENTS
        stage_predict(a)
    elif a.stage == "score":
        stage_score(a)
    elif a.stage == "windows":
        stage_windows(a)
    else:
        stage_stats(a)


if __name__ == "__main__":
    main()
