"""Controls for the primary AML result (registry section 29c).

control 2  The same task on non-AML cells with only the patient marks. For the
           held-out K562, HMEC deep and HUVEC deep and the training cells
           GM12878, IMR-90, HepG2, A549 and H1 (reported separately),
           all12refres_drop at 10 and 25 kb predicts the 413 Akita test
           regions from CTCF + H3K27ac + H3K27me3 (and from H3K27ac +
           H3K27me3, the set of the patients without CTCF) after the quantile
           matching used for the patients, and from all of the cell's tracks.
           Reference: the cell's leave-one-out consensus (training cells) or
           consensus12 (held-out cells). Metrics as for the patients: the
           deviation score of the measured map T from the reference C
           (scripts/hard_split.py: 0.5 (1 - r) + 0.5 var(T - C)/var(T)),
           the hard-split coverage and masked-bin filters, the hardest 20%
           of the passing regions, and the residual correlation
           corr(P - C, T - C) there. There is no HSPC analogue for these
           cells, so only the consensus-based definition exists; the
           patients are compared on the same definition (vs consensus12).
control 3  Mark-to-structure at the AML-specific loop anchors: does the
           patient's own anchor H3K27ac / H3K27me3 separate loops with a high
           measured gain over the HSPC consensus (AUROC of the anchor signal
           against the measured gain), and does the predicted gain follow the
           anchor signal (AUROC of the predicted gain for the same labels)?

    python scripts/aml_primary_controls.py --stage inputs predict score loops
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aml_primary as ap                                          # noqa: E402
import multicell_expanded as mx                                   # noqa: E402

log = ap.log
HELD = ["k562", "hmec_deep", "huvec_deep"]
TRAINED = ["gm12878", "imr90", "hepg2", "a549", "h1"]
CELLS = HELD + TRAINED
MODELS = {10000: "all12refres_drop_10kb", 25000: "all12refres_drop_25kb"}
IN_NA = ap.PRIM / "inputs_nonaml"
PRED_NA = ap.PRIM / "pred_nonaml"
MARKS3 = ["CTCF", "H3K27ac", "H3K27me3"]
SETS = {"p3": ["CTCF", "H3K27ac", "H3K27me3"], "p2": ["H3K27ac", "H3K27me3"]}


def _qm(job):
    c, m = job
    dest = IN_NA / c / f"{m}.bigwig"
    src = Path(mx.CELLS[c][1]) / f"{m}.bigwig"
    if not dest.exists() and src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        ap.quantile_track(src, dest, m)
        log(f"[input] {c} {m}")


def stage_inputs() -> None:
    from concurrent.futures import ProcessPoolExecutor
    for m in MARKS3:
        ap.reference_quantiles(m)
    with ProcessPoolExecutor(6) as ex:
        list(ex.map(_qm, [(c, m) for c in CELLS for m in MARKS3]))


def ckpt(model: str) -> Path:
    run = ap.MR / model
    return run / (run / "best").read_text().strip()


def predict_cell(cell: str, res: int, device: str, threads: int) -> None:
    model = MODELS[res]
    ck = ckpt(model)
    ref = mx.reference_of(model, cell)
    for tag, bw, sets in (("qm", IN_NA / cell, SETS),
                          ("all", Path(mx.CELLS[cell][1]), {"all": mx.ALL13})):
        out = PRED_NA / model / cell / tag
        if (out / "done").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        have = {f.stem for f in Path(bw).glob("*.bigwig")}
        ks = [f"{nm}={','.join(t for t in tr if t in have)}" for nm, tr in sets.items()]
        cmd = [ap.PY, "scripts/crosscell_predict.py", "--config", str(ck / "config.yaml"),
               "--checkpoint", str(ck), "--cell", cell, "--bigwig-dir", str(bw),
               "--target", str(mx.target(cell, res)), "--tta", "--assay", mx.CELLS[cell][2],
               "--binsize", str(res), "--device", "cuda" if device != "cpu" else "cpu",
               "--reference", str(ref), "--out-dir", str(out / "csv"), "--keep-sets", *ks,
               "--save-pred", str(out / "{set}.npz")]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads),
                   CUDA_VISIBLE_DEVICES="" if device == "cpu" else device)
        with open(out / "run.log", "w") as fh:
            rc = subprocess.run(cmd, cwd=ap.REPO, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc:
            log(f"[FAILED] {cell} {res} {tag}")
            continue
        (out / "done").write_text("ok")
        log(f"[predict ok] {cell} {res} {tag}")


def stage_predict(devices: list[str], threads: int) -> None:
    import queue
    import threading
    q = queue.Queue()
    for c in CELLS:
        for res in MODELS:
            q.put((c, res))

    def worker(dev):
        while True:
            try:
                c, r = q.get_nowait()
            except queue.Empty:
                return
            predict_cell(c, r, dev, threads)
    ts = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def score_cell(cell: str, res: int) -> list[dict]:
    import cooler
    import hard_split as hs
    from aml_eval import pred_matrix
    model = MODELS[res]
    T_c = cooler.Cooler(str(mx.target(cell, res)))
    C_c = cooler.Cooler(str(hs.consensus_of(cell, res)))
    raw = cooler.Cooler(hs.raw_uri(cell, res))
    preds = {}
    for tag, nm in (("qm", "p3"), ("qm", "p2"), ("all", "all")):
        f = PRED_NA / model / cell / tag / f"{nm}.npz"
        if f.exists():
            preds[nm] = dict(np.load(f))
    rows = []
    for r in ap.region_list().itertuples():
        T = hs.fetch(T_c, r.chrom, r.start, r.end)
        C = hs.fetch(C_c, r.chrom, r.start, r.end)
        if C.shape != T.shape:
            continue
        RC = hs.fetch(raw, r.chrom, r.start, r.end)
        mb = hs.masked_bins(raw, r.chrom, r.start, r.end)
        if RC.shape != T.shape or len(mb) != T.shape[0]:
            continue
        st, valid = hs.region_stats(T, C, RC, mb, None)
        if "score" not in st:
            continue
        rec = {"cell": cell, "res_kb": res // 1000, "region": r.region, **st}
        for nm, pr in preds.items():
            if r.region in pr and pr[r.region].shape == T.shape:
                P = pred_matrix(pr[r.region])
                ok = valid & np.isfinite(P)
                rec[f"resid|{nm}"] = hs.corr((P - C)[ok], (T - C)[ok])
                rec[f"r|{nm}"] = hs.corr(P[ok], T[ok])
        rows.append(rec)
    return rows


def stage_score() -> None:
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(c, r) for c in CELLS for r in MODELS]
    with ProcessPoolExecutor(8) as ex:
        parts = [f.result() for f in [ex.submit(score_cell, *t) for t in tasks]]
    D = pd.DataFrame([r for p in parts for r in p])
    D.to_csv(ap.WORK / "controls_nonaml_regions.csv.gz", index=False)
    out = []
    for (c, k), g in D.groupby(["cell", "res_kb"]):
        g = g[(g.masked_frac <= 0.2)]
        g = g[g.coverage >= g.coverage.quantile(0.15)]
        hard = g[g.score.rank(pct=True, ascending=False) <= 0.2]
        for sel, gg in (("hardest_20pct", hard), ("all_passing", g)):
            rec = {"cell": c, "res_kb": k, "set": sel, "n_regions": len(gg),
                   "group": "held_out" if c in HELD else "training",
                   "score_median": gg.score.median()}
            for nm in ("p3", "p2", "all"):
                if f"resid|{nm}" in gg:
                    rec[f"resid_{nm}"] = gg[f"resid|{nm}"].median()
                    rec[f"r_{nm}"] = gg[f"r|{nm}"].median()
            out.append(rec)
    P = pd.DataFrame(out)
    P.to_csv(ap.RES_DIR / "control_nonaml_cells.csv", index=False)
    S = P.groupby(["group", "res_kb", "set"])[[c for c in P.columns if c.startswith(
        ("resid_", "r_", "score"))]].median().reset_index()
    S.to_csv(ap.RES_DIR / "control_nonaml_summary.csv", index=False)
    log("[score] control_nonaml_cells.csv")


# ---------------------------------------------------------------- control 3
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stage_loops() -> None:
    import pyBigWig
    F = pd.read_csv(ap.WORK / "explain_features.csv.gz")
    W = pd.read_csv(ap.WORK / "feature_windows.csv")
    F = F[F.kind == "loop"].copy()
    W = W[W.kind == "loop"].copy()
    W["window"] = W.chrom + ":" + W.start.astype(int).astype(str) + "-" + W.end.astype(int).astype(str)
    F = F.merge(W[["patient", "window", "chrom", "a1", "a2"]].drop_duplicates(["patient", "window", "a1"]),
                on=["patient", "window"], how="left")
    F = F.dropna(subset=["a1"])
    models = ["all12refres_drop_10kb", "all12honest_drop_10kb", "min_ctcf_h3k27ac_refres_10kb"]
    for mk in ("H3K27ac", "H3K27me3", "CTCF"):
        vals = []
        for r in F.itertuples():
            f = ap.INP / r.patient / f"{mk}.bigwig"
            if not f.exists():
                vals.append(np.nan)
                continue
            bw = pyBigWig.open(str(f))
            a = [bw.stats(r.chrom, int(x), int(x) + 10000, type="mean")[0] or 0.0 for x in (r.a1, r.a2)]
            bw.close()
            vals.append(float(np.mean(a)))
        F[f"anchor_{mk}"] = vals
    F["gain_meas"] = F["measured"] - F["hspc_consensus"]
    for m in models:
        if m in F:
            F[f"gain_{m}"] = F[m] - F["hspc_consensus"]
    rows = []
    for mk in ("H3K27ac", "H3K27me3", "CTCF"):
        col = f"anchor_{mk}"
        z = F.groupby("patient")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        F[f"z_{col}"] = z
        ok = F[col].notna()
        hi = F[ok & (z >= z[ok].quantile(2 / 3))]
        lo = F[ok & (z <= z[ok].quantile(1 / 3))]
        if len(hi) < 20 or len(lo) < 20:
            continue
        rows.append({"mark": mk, "n_high": len(hi), "n_low": len(lo),
                     "score": "measured gain over HSPC", "auroc": auroc(hi.gain_meas.values, lo.gain_meas.values)})
        for m in models:
            if f"gain_{m}" in F:
                rows.append({"mark": mk, "n_high": len(hi), "n_low": len(lo),
                             "score": f"predicted gain over HSPC ({m})",
                             "auroc": auroc(hi[f"gain_{m}"].values, lo[f"gain_{m}"].values)})
        # the reverse direction: does the anchor signal predict which loops gain
        g = F[ok]
        top = g[g.gain_meas >= g.gain_meas.quantile(2 / 3)]
        bot = g[g.gain_meas <= g.gain_meas.quantile(1 / 3)]
        rows.append({"mark": mk, "n_high": len(top), "n_low": len(bot),
                     "score": "anchor signal separates high from low measured gain",
                     "auroc": auroc(top[f"z_{col}"].values, bot[f"z_{col}"].values)})
        for m in models:
            if f"gain_{m}" in F:
                tp = g[g[f"gain_{m}"] >= g[f"gain_{m}"].quantile(2 / 3)]
                bt = g[g[f"gain_{m}"] <= g[f"gain_{m}"].quantile(1 / 3)]
                rows.append({"mark": mk, "n_high": len(tp), "n_low": len(bt),
                             "score": f"anchor signal separates high from low predicted gain ({m})",
                             "auroc": auroc(tp[f"z_{col}"].values, bt[f"z_{col}"].values)})
    pd.DataFrame(rows).to_csv(ap.RES_DIR / "control_loops_marks.csv", index=False)
    log("[loops] control_loops_marks.csv")


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--stage", nargs="+", required=True,
                   choices=["inputs", "predict", "score", "loops"])
    a.add_argument("--devices", nargs="+", default=["0", "0", "1", "1"])
    a.add_argument("--threads", type=int, default=2)
    a = a.parse_args()
    if "inputs" in a.stage:
        stage_inputs()
    if "predict" in a.stage:
        stage_predict(a.devices, a.threads)
    if "score" in a.stage:
        stage_score()
    if "loops" in a.stage:
        stage_loops()


if __name__ == "__main__":
    main()
