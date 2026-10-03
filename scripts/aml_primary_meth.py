"""Methylation in place of ChIP for the primary AML patients (Xu et al. 2022).

Xu et al. deposited whole-genome bisulfite sequencing of 20 AML patients
(GSE152099, hg38, bismark + methylpy; processed files only: CG methylation
level per 10 bp bin as bigWig, no coverage). Ten of them also have CUT&Tag
(103, 673, 629, 270, 773, 018, 546, 472, 168, 027); ten have WGBS only (566,
260, 1071, 1324, 424, 496, 504, 798, 838, 933); 1916 and 1360 have CUT&Tag
without WGBS. The 2 PBMC and further healthy WGBS of the paper are not
deposited.

stages
  build     Meth.bigwig per patient, built like scripts/methylation.py:
            1 kb bins, value 1 + methylated fraction, bins without enough data
            at 1 + the patient's genome mean, raw zero reserved for "no data".
            The ENCODE builder weights the CpGs of a bin by read coverage and
            needs 5 reads; the Xu bigWigs hold methylation per 10 bp bin
            without coverage, so a 1 kb bin is the unweighted mean of its
            covered 10 bp bins (each already a read-weighted fraction over
            the CpGs it holds) and needs MIN_BINS (3) of them.
  predict   all12refres_drop_meth_10kb with keep sets and two references
            (consensus12; leave-one-patient-out AML consensus), 413 regions.
  score     per patient, shared pixels: every keep set x reference, the
            reference alone, the other-patient mean, the model without the
            methylation channel (all12refres_drop_10kb).
  checks    local compartment signal and Xu's AML-specific loops and stripes
            with and without methylation.

    python scripts/aml_primary_meth.py --stage build predict score checks
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aml_primary as ap                                          # noqa: E402
import aml_primary_explain as ax                                  # noqa: E402
import methylation as me                                          # noqa: E402

BIN = me.BIN
MIN_BINS = 3
WG = ap.PRIM / "wgbs"
MD = ap.PRIM / "meth"
MODEL = "all12refres_drop_meth_10kb"
BASE = "all12refres_drop_10kb"
RES = 10000
PRED = ap.PRIM / "pred_meth"
log = ap.log

# (name, tracks kept); missing tracks of a patient drop out of every set
SETS = {
    "nometh": ["CTCF", "H3K27ac", "H3K27me3"],
    "meth": ["CTCF", "H3K27ac", "H3K27me3", "Meth"],
    "meth_for_k27me3": ["CTCF", "H3K27ac", "Meth"],
    "meth_for_ctcf": ["H3K27ac", "H3K27me3", "Meth"],
    "k27ac_meth": ["H3K27ac", "Meth"],
    "meth_only": ["Meth"],
}


def wgbs_files() -> dict[str, Path]:
    """patient label (xu_<id>) -> bigWig."""
    out = {}
    for f in sorted(WG.glob("*_allc_*.CGN.10bp.bw")):
        pid = f.name.split("_allc_")[1].split(".")[0]
        out["xu_" + ax.norm_id(pid)] = f
    return out


def both() -> list[str]:
    return [p for p in wgbs_files() if (ap.INP / p).exists()]


def wgbs_only() -> list[str]:
    return [p for p in wgbs_files() if not (ap.INP / p).exists()]


# ---------------------------------------------------------------- build
def build_one(item):
    import pyBigWig
    p, f = item
    dst = MD / p / "Meth.bigwig"
    if dst.exists():
        return p
    (MD / p).mkdir(parents=True, exist_ok=True)
    sizes = me.chrom_sizes()
    bw = pyBigWig.open(str(f))
    s = {c: np.zeros(n // BIN + 1) for c, n in sizes.items()}
    k = {c: np.zeros(n // BIN + 1) for c, n in sizes.items()}
    for c in sizes:
        if c not in bw.chroms():
            continue
        iv = bw.intervals(c)
        if not iv:
            continue
        a = np.asarray(iv, dtype=np.float64)
        b = (a[:, 0] // BIN).astype(np.int64).clip(0, len(s[c]) - 1)
        np.add.at(s[c], b, a[:, 2])
        np.add.at(k[c], b, 1.0)
    bw.close()
    ok_s = sum(s[c][k[c] >= MIN_BINS].sum() for c in s)
    ok_k = sum(k[c][k[c] >= MIN_BINS].sum() for c in k)
    mean = ok_s / max(ok_k, 1.0)
    out = bw_out = pyBigWig.open(str(dst) + ".part", "w")
    bw_out.addHeader(list(sizes.items()))
    cov = []
    for c, n in sizes.items():
        frac = np.full(len(s[c]), mean)
        ok = k[c] >= MIN_BINS
        frac[ok] = s[c][ok] / k[c][ok]
        v = (1.0 + frac).astype(np.float32)
        cov.append(ok[: n // BIN].mean())
        full = n // BIN
        if full:
            bw_out.addEntries(c, 0, values=v[:full].tolist(), span=BIN, step=BIN)
        if n % BIN:
            bw_out.addEntries([c], [full * BIN], ends=[n], values=[float(v[full])])
    bw_out.close()
    Path(str(dst) + ".part").rename(dst)
    log(f"[build] {p}: genome mean methylation {mean:.3f}, bins with data {np.mean(cov):.3f}")
    return p


def stage_build(workers: int) -> None:
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(build_one, list(wgbs_files().items())))


# ---------------------------------------------------------------- predict
def predict_one(p: str, ref_kind: str, device: str, threads: int) -> None:
    run = ap.PMODELS.get(MODEL) or None
    ck = ap.MR / MODEL / (ap.MR / MODEL / "best").read_text().strip()
    cfg = ck / "config.yaml"
    out = PRED / ref_kind / p
    tag = out / "done"
    if tag.exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    if ref_kind == "c12":
        ref, cell = ap.consensus12(RES), "aml_" + p
    else:
        ref, cell = ap.aml_consensus_path(RES, p), ap.guard_name(p)
    if (ap.INP / p).exists():
        bw = ap.INP / p
    else:
        bw = PRED / "empty"
        bw.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in bw.glob("*.bigwig")} | {"Meth"}
    sets = []
    for nm, tr in SETS.items():
        keep = [t for t in tr if t in have]
        sets.append(f"{nm}={','.join(keep)}")
    cmd = [ap.PY, "scripts/crosscell_predict.py", "--config", str(cfg), "--checkpoint", str(ck),
           "--cell", cell, "--bigwig-dir", str(bw), "--extra-bigwig-dir", str(MD / p),
           "--target", str(ap.map_path(p, RES)), "--tta", "--assay", "hic", "--binsize", str(RES),
           "--device", "cuda" if device != "cpu" else "cpu", "--reference", str(ref),
           "--out-dir", str(out / "csv"), "--keep-sets", *sets,
           "--save-pred", str(out / "{set}.npz")]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               CUDA_VISIBLE_DEVICES="" if device == "cpu" else device)
    with open(out / "run.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=ap.REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        log(f"[FAILED] {p} {ref_kind} rc={rc}")
        return
    tag.write_text("ok")
    log(f"[predict ok] {p} {ref_kind}")


def stage_predict(devices: list[str], threads: int) -> None:
    import queue
    import threading
    q = queue.Queue()
    for p in wgbs_files():
        for rk in ("c12", "aml"):
            q.put((p, rk))

    def worker(dev):
        while True:
            try:
                p, rk = q.get_nowait()
            except queue.Empty:
                return
            predict_one(p, rk, dev, threads)
    ts = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


# ---------------------------------------------------------------- score
def load_sets(p: str) -> dict[str, dict]:
    out = {}
    for rk in ("c12", "aml"):
        for nm in SETS:
            f = PRED / rk / p / f"{nm}.npz"
            if f.exists():
                out[f"{nm}|{rk}"] = dict(np.load(f))
    return out


def score_patient(p: str) -> list[dict]:
    import cooler
    import benchmark_metrics as bm
    from aml_eval import fetch_region, pred_matrix
    clrs = {q: cooler.Cooler(str(ap.map_path(q, RES))) for q in ap.aml_patients(RES)}
    c12 = cooler.Cooler(str(ap.consensus12(RES)))
    aml = cooler.Cooler(str(ap.aml_consensus_path(RES, p)))
    preds = load_sets(p)
    base = {}
    for tag, nm in (("", "base|c12"), (ap.AMLREF, "base|aml")):
        f = ap.pred_file(BASE, p + tag)
        if f.exists():
            base[nm] = dict(np.load(f))
    rows = []
    for r in ap.region_list().itertuples():
        cache: dict = {}
        T = fetch_region(clrs[p], r.region, cache)
        if T is None:
            continue
        mats = {}
        for k, pr in {**preds, **base}.items():
            if r.region in pr and pr[r.region].shape == T.shape:
                mats["model:" + k] = pred_matrix(pr[r.region])
        mats["map:consensus12"] = fetch_region(c12, r.region, cache)
        mats["map:aml_consensus"] = fetch_region(aml, r.region, cache)
        others = [fetch_region(clrs[q], r.region, cache) for q in clrs if q != p]
        others = [o for o in others if o is not None and o.shape == T.shape]
        if others:
            mats["map:other_patient_mean"] = np.nanmean(np.stack(others), 0)
        mats = {k: v for k, v in mats.items() if v is not None and v.shape == T.shape}
        mask = bm.triu_mask(T.shape[0], 2) & np.isfinite(T)
        for M in mats.values():
            mask &= np.isfinite(M)
        if mask.sum() < 100:
            continue
        for k, M in mats.items():
            rows.append({"patient": p, "method": k, "region": r.region,
                         "pearson": bm.pearson(M, T, mask), "scc": bm.scc(M, T, mask)})
    return rows


def stage_score(workers: int) -> None:
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(score_patient, list(wgbs_files())))
    L = pd.DataFrame([r for part in parts for r in part])
    L.to_csv(ap.WORK / "meth_scores_long.csv.gz", index=False)
    S = (L.groupby(["patient", "method"]).agg(pearson=("pearson", "median"), scc=("scc", "median"),
                                              n_regions=("pearson", "count")).reset_index())
    cat = {p: ("both" if p in both() else "wgbs_only") for p in wgbs_files()}
    S["group"] = S.patient.map(cat)
    S["has_ctcf"] = S.patient.map(lambda p: (ap.INP / p / "CTCF.bigwig").exists())
    ap.RES_DIR.mkdir(parents=True, exist_ok=True)
    S.to_csv(ap.RES_DIR / "meth_patients.csv", index=False)
    M = S.groupby(["group", "method"])[["pearson", "scc"]].median().reset_index()
    M.to_csv(ap.RES_DIR / "meth_summary.csv", index=False)
    log("[score] meth_patients.csv, meth_summary.csv")


# ---------------------------------------------------------------- checks
def first_pc(M: np.ndarray) -> np.ndarray | None:
    """First eigenvector of the correlation matrix of log O/E (upper and
    lower triangle, NaN as 0), unit norm; the local checkerboard of a window."""
    X = np.nan_to_num(M)
    X = X - X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    if (sd == 0).all():
        return None
    C = (X / np.where(sd == 0, 1, sd)) @ (X / np.where(sd == 0, 1, sd)).T / X.shape[1]
    w, v = np.linalg.eigh(C)
    return v[:, -1]


def stage_checks_compartment() -> None:
    """Local compartment signal. Per region the first eigenvector (PC1) of the
    correlation matrix of the measured map, consensus12, the HSPC consensus
    and each prediction; sign fixed so that it correlates positively with
    consensus12's PC1. Regions with the largest PC1 change of the patient
    against the HSPC consensus (top 20% per patient: 1 - |r|, i.e. the
    pattern differs) are the switch candidates; scored: corr(PC1_pred -
    PC1_HSPC, PC1_T - PC1_HSPC) there. The paper's switch calls (Fig. 1e) hold
    no coordinates, and chromosome-wide A/B compartments are not accessible
    from 1 Mb windows: this is a window-level proxy."""
    import cooler
    from aml_eval import fetch_region, pred_matrix
    hs = cooler.Cooler(str(ax.hspc_consensus_path(RES)))
    c12 = cooler.Cooler(str(ap.consensus12(RES)))
    rows = []
    for p in both():
        T = cooler.Cooler(str(ap.map_path(p, RES)))
        preds = load_sets(p)
        sel = [k for k in preds if k.split("|")[0] in ("nometh", "meth", "meth_for_k27me3",
                                                         "meth_for_ctcf", "meth_only")]
        for r in ap.region_list().itertuples():
            cache: dict = {}
            Mt, Mh, Mc = (fetch_region(c, r.region, cache) for c in (T, hs, c12))
            if Mt is None or Mh is None or Mc is None or Mt.shape != Mh.shape:
                continue
            pcs = {"T": first_pc(Mt), "H": first_pc(Mh), "C": first_pc(Mc)}
            if any(v is None for v in pcs.values()):
                continue
            ref = pcs["C"]
            sg = lambda v: v * (1 if np.dot(v, ref) >= 0 else -1)         # noqa: E731
            pcs = {k: sg(v) for k, v in pcs.items()}
            rec = {"patient": p, "region": r.region,
                   "pc_r_T_H": float(np.corrcoef(pcs["T"], pcs["H"])[0, 1]),
                   "pc_r_T_C": float(np.corrcoef(pcs["T"], pcs["C"])[0, 1])}
            for k in sel:
                if r.region in preds[k]:
                    P = pred_matrix(preds[k][r.region])
                    if P.shape != Mt.shape:
                        continue
                    pc = first_pc(P)
                    if pc is None:
                        continue
                    pc = sg(pc)
                    rec[f"r_T|{k}"] = float(np.corrcoef(pc, pcs["T"])[0, 1])
                    d_p, d_t = pc - pcs["H"], pcs["T"] - pcs["H"]
                    rec[f"dres|{k}"] = float(np.corrcoef(d_p, d_t)[0, 1])
            rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(ap.WORK / "meth_compartment_regions.csv.gz", index=False)
    out = []
    for p, g in D.groupby("patient"):
        sw = g[g.pc_r_T_H.rank(pct=True) <= 0.2]            # lowest agreement with HSPC
        rec = {"patient": p, "n_switch_regions": len(sw), "pc_r_T_H_switch": sw.pc_r_T_H.median()}
        for c in D.columns:
            if c.startswith(("r_T|", "dres|")):
                rec[c + "|switch"] = sw[c].median()
                rec[c + "|all"] = g[c].median()
        out.append(rec)
    pd.DataFrame(out).to_csv(ap.RES_DIR / "meth_compartment.csv", index=False)
    log("[checks] meth_compartment.csv")


FEAT_SETS = ["nometh", "meth"]


def stage_checks_features(devices: list[str]) -> None:
    """Xu's AML-specific loops and stripes: the meth model on the 10 kb
    feature windows with and without the patient's own methylation."""
    import queue
    import threading
    W = pd.read_csv(ap.WORK / "feature_windows.csv")
    pats = [p for p in W.patient.unique() if p in both()]
    out = PRED / "features"
    out.mkdir(parents=True, exist_ok=True)
    ck = ap.MR / MODEL / (ap.MR / MODEL / "best").read_text().strip()
    q = queue.Queue()
    for p in pats:
        q.put(p)

    def worker(dev):
        while True:
            try:
                p = q.get_nowait()
            except queue.Empty:
                return
            if (out / f"{p}.done").exists():
                continue
            g = W[W.patient == p]
            reg = out / f"{p}.tsv"
            g[["chrom", "start", "end"]].drop_duplicates().to_csv(reg, sep="\t", header=False,
                                                                   index=False)
            have = {f.stem for f in (ap.INP / p).glob("*.bigwig")} | {"Meth"}
            sets = [f"{nm}={','.join(t for t in SETS[nm] if t in have)}" for nm in FEAT_SETS]
            cmd = [ap.PY, "scripts/crosscell_predict.py", "--config", str(ck / "config.yaml"),
                   "--checkpoint", str(ck), "--cell", "aml_" + p, "--bigwig-dir", str(ap.INP / p),
                   "--extra-bigwig-dir", str(MD / p), "--target", str(ap.grid_cool(RES)), "--tta",
                   "--assay", "hic", "--binsize", str(RES), "--device", "cuda", "--regions", str(reg),
                   "--reference", str(ap.consensus12(RES)), "--out-dir", str(out / f"{p}_csv"),
                   "--keep-sets", *sets, "--save-pred", str(out / f"{p}__{{set}}.npz")]
            env = dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=dev)
            with open(out / f"{p}.log", "w") as fh:
                rc = subprocess.run(cmd, cwd=ap.REPO, env=env, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
            if rc == 0:
                (out / f"{p}.done").write_text("ok")
                log(f"[features predict] {p}")
    ts = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    import cooler
    from aml_eval import fetch_region, loop_params, pred_matrix
    from hicmultiome.bioscore import local_enrichment
    prm = loop_params(ax.FEAT_RES)
    HC = cooler.Cooler(str(ax.hspc_consensus_path(ax.FEAT_RES)))
    C12 = cooler.Cooler(str(ap.consensus12(ax.FEAT_RES)))
    rows = []
    for p in pats:
        T = cooler.Cooler(str(ap.map_path(p, ax.FEAT_RES)))
        preds = {nm: dict(np.load(out / f"{p}__{nm}.npz")) for nm in FEAT_SETS
                 if (out / f"{p}__{nm}.npz").exists()}
        for r in W[W.patient == p].itertuples():
            win = f"{r.chrom}:{int(r.start)}-{int(r.end)}"
            cache: dict = {}
            mats = {"measured": fetch_region(T, win, cache), "hspc": fetch_region(HC, win, cache),
                    "c12": fetch_region(C12, win, cache)}
            for nm, pr in preds.items():
                if win in pr:
                    mats[nm] = pred_matrix(pr[win])
            if any(v is None for v in mats.values()) or len(mats) < 5:
                continue
            n = mats["measured"].shape[0]
            if any(v.shape != (n, n) for v in mats.values()):
                continue
            rec = {"patient": p, "kind": r.kind}
            i = (int(r.a1) - int(r.start)) // ax.FEAT_RES
            if r.kind == "loop":
                j = (int(r.a2) - int(r.start)) // ax.FEAT_RES
                if not (0 <= i < n and 0 <= j < n):
                    continue
                for kk, M in mats.items():
                    e = local_enrichment(np.exp(np.nan_to_num(M)), prm)
                    rec[kk] = float(np.log2(e[min(i, j), max(i, j)]))
                am, bm_ = ax.anchor_marks(p, r.chrom, int(r.a1)), ax.anchor_marks(p, r.chrom, int(r.a2))
                rec["type"] = ("P-S-like" if am.get("H3K27me3") or bm_.get("H3K27me3") else
                               "P-E-like" if am.get("H3K27ac") and bm_.get("H3K27ac") else "other")
            else:
                j1 = (int(r.e1) - int(r.start)) // ax.FEAT_RES
                j2 = (int(r.e2) - int(r.start)) // ax.FEAT_RES
                i2 = (int(r.a2) - int(r.start)) // ax.FEAT_RES
                if not (0 <= i < n and 0 <= j1 < n and 0 < j2 <= n):
                    continue
                for kk, M in mats.items():
                    band = M[max(0, i):min(n, i2 + 1), max(0, j1):min(n, j2)]
                    rec[kk] = float(np.nanmean(band)) if np.isfinite(band).any() else np.nan
                rec["type"] = "stripe"
            rows.append(rec)
    F = pd.DataFrame(rows)
    F.to_csv(ap.WORK / "meth_features.csv.gz", index=False)
    res = []
    for (kind, typ), g in F.groupby(["kind", "type"]):
        rec = {"kind": kind, "type": typ, "n": len(g), "patients": g.patient.nunique(),
               "measured_minus_hspc": (g.measured - g.hspc).median()}
        for nm in FEAT_SETS:
            for ref in ("hspc", "c12"):
                d_m, d_p = g.measured - g[ref], g[nm] - g[ref]
                ok = d_m.notna() & d_p.notna()
                rec[f"{nm}_minus_{ref}"] = d_p.median()
                rec[f"{nm}_corr_with_measured_vs_{ref}"] = (
                    float(np.corrcoef(d_m[ok], d_p[ok])[0, 1]) if ok.sum() > 10 else np.nan)
        res.append(rec)
    pd.DataFrame(res).to_csv(ap.RES_DIR / "meth_features.csv", index=False)
    log("[checks] meth_features.csv")


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--stage", nargs="+", required=True,
                   choices=["build", "predict", "score", "checks"])
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--devices", nargs="+", default=["0", "0", "1", "1"])
    a.add_argument("--threads", type=int, default=2)
    a = a.parse_args()
    if "build" in a.stage:
        stage_build(a.workers)
    if "predict" in a.stage:
        stage_predict(a.devices, a.threads)
    if "score" in a.stage:
        stage_score(a.workers)
    if "checks" in a.stage:
        stage_checks_compartment()
        stage_checks_features(a.devices)


if __name__ == "__main__":
    main()
