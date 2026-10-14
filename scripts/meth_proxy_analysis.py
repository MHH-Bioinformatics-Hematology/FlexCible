"""DNA methylation as a proxy for missing assays (registry section 78).

One analysis for the question: when histone marks, CTCF or cohesin are missing, how much
of the Hi-C accuracy that is lost can the methylation track (Meth) give back? Methylation
never extends a complete track set here; every number is a replacement fraction

    RF = (score(S + Meth) - score(S)) / (score(all 13 ChIP tracks) - score(S))

for a set S of tracks left after a group was removed, with Meth present or absent in the
SAME trained model (track dropout makes a removed group and a missing Meth an
in-distribution state). The group sets of registry section 35 (scripts/meth_replace.py)
are used unchanged, so the numbers of section 35 are the 10 kb consensus rows of this table.

Parts (stages):
  predict      inference of the cell-line Meth models with each group removed, with and
               without Meth, on the 413 Akita test regions: with the consensus (reference
               residual) all12refres_drop_meth_10kb, _v2_10kb, all12refres_drop_motifmeth_10 and
               25 kb; without it all12honest_drop_meth_10 and 25 kb (+ seed s1) and
               all12honest_drop_motifmeth_10 and 25 kb (+ s1 at 25 kb). Existing predictions
               of scripts/meth_replace.py are reused. --structure for every set.
  aml          inference of the AML test donors (aml_meth_refin of folds 10 to 12, 10 and 25
               kb): own marks H3K27ac, H3K27me3, CTCF; each removed and all three, with and
               without the patient's WGBS; Pearson, patient-specific windows, insulation.
  aml_loops    10 kb loop and stripe detection on the windows of scripts/aml_loop_detection.py
               for the same conditions (share of AML-specific loops above the 90th percentile
               of random pairs, residual AUROC against matched random pairs, marks effect).
  summary      tables in results/meth_proxy/ and the figure.

The noise threshold of a fraction is the honest two-seed spread of the model family
(reference_compare.csv: median over cells of the range of the median Pearson over the seeds
s0, s1, s2 of the same recipe without Meth, per resolution), not the 0.013 of section 8.

    python scripts/meth_proxy_analysis.py --stage predict --gpu 0
    python scripts/meth_proxy_analysis.py --stage aml aml_loops --gpu 0
    python scripts/meth_proxy_analysis.py --stage summary
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
import meth_replace as mr                                           # noqa: E402
import methylation as me                                            # noqa: E402
import multicell_expanded as mx                                     # noqa: E402

HM = mx.HM
OUT = HM / "runs/meth_proxy"
OLD = HM / "runs/meth_replace"
RES_OUT = REPO / "results/meth_proxy"
PY = os.environ.get("HICMULTI_PY", sys.executable)
NICE = ["nice", "-n", "10"]

# --------------------------------------------------------------------------- models
# name -> family (consensus = reference residual input, honest = no reference), resolution,
# motif_meth planes, methylation cell set version, seed
MODELS = {
    "all12honest_drop_meth_10kb": dict(family="honest", res=10, planes=False, v2=True, seed=0),
    "all12honest_drop_meth_s1_10kb": dict(family="honest", res=10, planes=False, v2=True, seed=1),
    "all12honest_drop_meth_25kb": dict(family="honest", res=25, planes=False, v2=True, seed=0),
    "all12honest_drop_meth_s1_25kb": dict(family="honest", res=25, planes=False, v2=True, seed=1),
    "all12honest_drop_motifmeth_10kb": dict(family="honest", res=10, planes=True, v2=True, seed=0),
    "all12honest_drop_motifmeth_25kb": dict(family="honest", res=25, planes=True, v2=True, seed=0),
    "all12honest_drop_motifmeth_s1_25kb": dict(family="honest", res=25, planes=True, v2=True, seed=1),
    "all12refres_drop_motifmeth_25kb": dict(family="consensus", res=25, planes=True, v2=False, seed=0),
    "all12refres_drop_motifmeth_10kb": dict(family="consensus", res=10, planes=True, v2=False, seed=0),
    # sections 35 predictions (runs/meth_replace) are reused for these two
    "all12refres_drop_meth_10kb": dict(family="consensus", res=10, planes=False, v2=False, seed=0),
    "all12refres_drop_meth_v2_10kb": dict(family="consensus", res=10, planes=False, v2=True, seed=0),
}
WGBS_TRAIN = ["gm12878", "imr90", "hepg2", "a549", "h1", "gm23248", "cd14mono"]
RRBS_TRAIN = ["hct116", "mcf7", "hff"]
HELD = ["k562", "hmec_deep"]
CELLS_S35 = ["gm12878", "imr90", "hepg2", "a549", "h1", "k562", "hmec_deep"]
CELLS_S35_V2 = CELLS_S35 + RRBS_TRAIN
SETS = mr.SETS                        # section 35 group sets, without Meth
SET_ORDER = mr.SET_ORDER
METRICS = {"pearson": "model", "scc": "struct_scc", "insulation": "struct_insulation_corr",
           "boundary_f1": "struct_boundary_f1", "loop_enrichment": "struct_loop_enrich_corr",
           "compartment": "struct_compartment_corr", "eigenvector": "struct_eigenvector_abs_corr"}
MAIN_METRICS = ["pearson", "insulation", "boundary_f1", "loop_enrichment", "compartment"]
SINGLE = ["minus_CTCF", "minus_cohesin", "minus_active", "minus_genebody", "minus_repressive"]
CUMUL = ["cum1", "cum2", "cum3", "ctcf_only", "ctcf_h3k27ac", "ctcf_h3k27ac_h3k27me3", "seq_only"]


def cells_of(model: str) -> list[str]:
    """Cells scored per model. Primary honest models: the seven WGBS training cells and the two held-out
    cells; seed replicates: the two held-out cells; motif-plane models: two training cells and the held-out cells; the consensus
    models of section 35 keep their own cell lists (RRBS cells only in the v2 model)."""
    spec = MODELS[model]
    if model.startswith("all12refres_drop_meth"):
        return CELLS_S35_V2 if spec["v2"] else CELLS_S35
    if spec["seed"] > 0:
        return list(HELD)
    if spec["planes"]:
        return ["gm12878", "hepg2"] + HELD
    return WGBS_TRAIN + HELD


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "meth_proxy.log", "a") as fh:
        fh.write(line + "\n")


def keep_sets() -> dict[str, list[str]]:
    out = dict(SETS)
    out.update({f"{k}+meth": v + [me.TRACK] for k, v in SETS.items()})
    return out


def pred_file(model: str, st: str, cell: str) -> Path | None:
    for base in (OUT, OLD):
        f = base / model / st / f"predict_{cell}.csv"
        if f.exists():
            return f
    return None


def consensus_floor(cell: str, res: int) -> Path:
    return mx.consensus_loo(mx.CELLS[cell][3], res) if cell in mx.TRAIN12 else mx.consensus12(res)


def gpu_free_mib() -> int:
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                                     "-i", "0"], text=True, timeout=30)
        return int(o.split()[0])
    except Exception:
        return 0


def wait_gpu(need: int = 900) -> None:
    """The GPU is shared with training and other inference: start a job only when `need` MiB are free
    (an inference process takes 0.7 to 1.5 GB); never take memory from another job."""
    n = 0
    while gpu_free_mib() < need:
        if n % 20 == 0:
            log(f"waiting for the GPU ({gpu_free_mib()} MiB free, need {need})")
        n += 1
        time.sleep(60)


def release(name: str) -> None:
    import claim_job as cj
    cj.release(cj.DEFAULT_DIR, name)


def run_claimed(name: str, fn, tries: int = 4) -> int:
    """Claim `name`, wait for GPU memory, run fn(); on failure release the claim and retry later."""
    if not claim(name):
        return 0
    rc = 1
    for k in range(tries):
        wait_gpu()
        rc = fn()
        if rc == 0:
            return 0
        log(f"{name}: attempt {k + 1} rc={rc}, retry in 120 s")
        time.sleep(120)
    release(name)
    return rc


def claim(name: str) -> bool:
    """Claim a job on the shared drive (scripts/claim_job.py); False when another host has it."""
    import claim_job as cj
    d = cj.DEFAULT_DIR
    c = cj.read_claim(d, name)
    if c and c.get("host") == cj.host() and os.environ.get("METH_PROXY_RESUME"):
        return True                       # own earlier claim (resumed job)
    return cj.claim(d, name, None, "meth_proxy_analysis predict") == 0


def predict(model: str, cell: str, gpu: str, threads: int = 3) -> int:
    spec = MODELS[model]
    sets = keep_sets()
    if all(pred_file(model, s, cell) for s in sets):
        return 0
    res = spec["res"] * 1000
    ck = mx.best_ckpt(mx.out_of(model) / model)
    out = OUT / model
    _, bw, assay, ident = mx.CELLS[cell]
    tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
           "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw,
           "--target", mx.target(cell, res), "--tta", "--assay", assay, "--binsize", res,
           "--device", "cuda", "--out-dir", out, "--structure",
           "--keep-sets", *[f"{s}={','.join(t)}" for s, t in sets.items()]]
    if spec["family"] == "consensus":
        ref = mx.reference_of(model, cell)
        cmd += ["--reference", ref, "--others", f"{tag}={ref}"]
    else:
        cmd += ["--others", f"{tag}={consensus_floor(cell, res)}"]
    if me.has_meth(cell, True):
        cmd += ["--extra-bigwig-dir", me.meth_dir(cell, True)]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES=gpu)
    out.mkdir(parents=True, exist_ok=True)

    def run() -> int:
        with open(out / f"predict_{cell}.log", "a") as fh:
            fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
            fh.flush()
            return subprocess.call(NICE + [str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                                   cwd=REPO, env=env)
    return run_claimed(f"meth_proxy_{model}_{cell}", run)


def stage_predict(gpu: str, lanes: int, only: list[str] | None) -> None:
    tasks = [(m, c) for m in MODELS if not only or m in only for c in cells_of(m)
             if not all(pred_file(m, s, c) for s in keep_sets())]
    # held-out cells first, then the primary models on training cells, the seed replicates last
    tasks.sort(key=lambda t: ("_s1_" in t[0], t[1] not in HELD, list(MODELS).index(t[0])))
    log(f"predict: {len(tasks)} model-cell tasks on gpu{gpu}, {lanes} lanes")
    L = [tasks[i::lanes] for i in range(lanes)]

    def lane(i):
        for m, c in L[i]:
            rc = predict(m, c, gpu, 6 // lanes)
            log(f"predict {m} {c} rc={rc}")
    with ThreadPoolExecutor(lanes) as ex:
        list(ex.map(lane, range(lanes)))


# =========================================================================== AML test donors
AML_PATIENTS = {10: ["018", "027", "270"], 11: ["168", "546", "629"], 12: ["472", "773"]}   # donors with WGBS
MARKS3 = ["CTCF", "H3K27ac", "H3K27me3"]
AML_OUT = OUT / "aml"
AML_LOOP_OUT = OUT / "aml_loops"
AML_SETS = ["full", "minus_H3K27ac", "minus_H3K27me3", "minus_CTCF", "minus_H3K27ac_H3K27me3",
            "minus_H3K27ac_CTCF", "minus_H3K27me3_CTCF", "minus_all3"]


def aml_sets(marks: list[str]) -> dict[str, list[str]]:
    """Keep sets of a donor's own marks (never more than the three of the AML ChIP panel)."""
    m = list(marks)
    def drop(*x):
        return [k for k in m if k not in x]
    out = {"full": m, "minus_H3K27ac": drop("H3K27ac"), "minus_H3K27me3": drop("H3K27me3"),
           "minus_CTCF": drop("CTCF"), "minus_H3K27ac_H3K27me3": drop("H3K27ac", "H3K27me3"),
           "minus_H3K27ac_CTCF": drop("H3K27ac", "CTCF"), "minus_H3K27me3_CTCF": drop("H3K27me3", "CTCF"),
           "minus_all3": []}
    out.update({f"{k}+meth": v + [me.TRACK] for k, v in list(out.items())})
    return out


def aml_run(fold: int, res: int) -> str:
    return f"all12aml_meth_refin_drop_f{fold}_{res // 1000}kb"


def aml_predict(fold: int, p: str, res: int, mode: str, gpu: str, threads: int = 4) -> int:
    """mode 'regions': 413 test regions with --structure (Pearson, insulation, specific windows);
    'windows': the 192-bin object windows of the loop detection (10 kb)."""
    import aml_train as at
    import aml_train_eval as te
    s = next(x for x in te.samples(fold, res) if x["patient"] == p and x["group"] == "patient")
    out = (AML_OUT if mode == "regions" else AML_LOOP_OUT) / f"f{fold}" / f"{res // 1000}kb" / s["cell"]
    sets = aml_sets(s["marks"])
    if (out / "minus_all3+meth.npz").exists():
        return 0
    ck = te.ckpt(aml_run(fold, res))
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml", "--checkpoint", ck,
           "--cell", s["cell"], "--bigwig-dir", s["dir"], "--target", s["target"], "--tta",
           "--assay", s["assay"], "--binsize", res, "--device", "cuda", "--out-dir", out,
           "--save-pred", str(out / "{set}.npz"), "--lab", s["lab"],
           "--reference", at.ref_path(fold, res, None), "--extra-bigwig-dir", s["meth"],
           "--keep-sets", *[f"{k}={','.join(v)}" for k, v in sets.items()]]
    if mode == "regions":
        cmd += ["--structure", "--others", *te.common_others(s, fold, res)]
    else:
        cmd += ["--regions", HM / "runs/aml_loop_detection/windows" / f"xu_{p}.tsv"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES=gpu)
    out.mkdir(parents=True, exist_ok=True)

    def run() -> int:
        with open(out / "predict.log", "a") as fh:
            fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
            fh.flush()
            return subprocess.call(NICE + [str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                                   cwd=REPO, env=env)
    return run_claimed(f"meth_proxy_aml_{mode}_f{fold}_{res // 1000}kb_{p}", run)


def stage_aml(a) -> None:
    for fold, ps in AML_PATIENTS.items():
        for res in (10000, 25000):
            for p in ps:
                rc = aml_predict(fold, p, res, "regions", a.gpu)
                log(f"aml regions f{fold} {res // 1000}kb {p} rc={rc}")


def stage_aml_loops(a) -> None:
    for fold, ps in AML_PATIENTS.items():
        for p in ps:
            rc = aml_predict(fold, p, 10000, "windows", a.gpu)
            log(f"aml windows f{fold} 10kb {p} rc={rc}")


# =========================================================================== tables
NB = 1000                                  # bootstrap replicates
SEC51_CONSENSUS_NOISE = 0.0062             # upper end of the two-seed spreads of consensus-input models (section 51)
GROUP_OF_SET = {"minus_CTCF": "CTCF", "minus_cohesin": "cohesin", "minus_active": "active",
                "minus_genebody": "gene body", "minus_repressive": "repressive"}
CELL_GROUP = {**{c: "training WGBS" for c in WGBS_TRAIN}, **{c: "training RRBS" for c in RRBS_TRAIN},
              **{c: "held-out" for c in HELD}}


def load_regions() -> pd.DataFrame:
    """All predictions as one long table of per-region scores (cached to runs/meth_proxy/regions.parquet)."""
    cols = {v: k for k, v in METRICS.items()}
    parts = []
    for model, spec in MODELS.items():
        for cell in cells_of(model):
            for st in keep_sets():
                f = pred_file(model, st, cell)
                if f is None:
                    continue
                d = pd.read_csv(f)
                keep = ["region"] + [c for c in cols if c in d]
                d = d[keep].rename(columns=cols)
                base = st.removesuffix("+meth")
                d.insert(0, "meth", st.endswith("+meth"))
                d.insert(0, "set", base)
                d.insert(0, "cell", cell)
                d.insert(0, "model", model)
                parts.append(d)
    return pd.concat(parts, ignore_index=True)


def cell_matrix(R: pd.DataFrame, model: str, cell: str, metric: str):
    """(regions, labels, array [n_sets_with_meth_flag, n_regions]) aligned on the regions common to all sets."""
    d = R[(R.model == model) & (R.cell == cell)]
    if d.empty or metric not in d:
        return None
    w = d.pivot_table(index="region", columns=["set", "meth"], values=metric, aggfunc="first")
    return w


def rf_cell(w: pd.DataFrame, st: str, rng_idx: np.ndarray | None = None):
    """gain and loss (medians over the regions, optionally one bootstrap draw of regions)."""
    try:
        a, b, f = w[(st, False)].to_numpy(), w[(st, True)].to_numpy(), w[("full", False)].to_numpy()
    except KeyError:
        return None
    if rng_idx is not None:
        a, b, f = a[rng_idx], b[rng_idx], f[rng_idx]
    return np.nanmedian(b) - np.nanmedian(a), np.nanmedian(f) - np.nanmedian(a)


def noise_of(R: pd.DataFrame) -> pd.DataFrame:
    """Seed noise of the median Pearson per family and resolution: median over cells of the absolute
    difference of the median Pearson of seeds 0 and 1 (full set, no Meth channel input), for the honest
    family; the consensus family has no replicate here and takes the upper end of section 51."""
    rows = []
    for res in (10, 25):
        for tag, a, b in (("honest meth", f"all12honest_drop_meth_{res}kb", f"all12honest_drop_meth_s1_{res}kb"),
                          ("honest motifmeth", f"all12honest_drop_motifmeth_{res}kb", f"all12honest_drop_motifmeth_s1_{res}kb")):
            if a not in MODELS or b not in MODELS:
                continue
            for st in SET_ORDER:
                dd = []
                for cell in cells_of(a):
                    x = R[(R.model == a) & (R.cell == cell) & (R.set == st) & (~R.meth)]
                    y = R[(R.model == b) & (R.cell == cell) & (R.set == st) & (~R.meth)]
                    if len(x) and len(y):
                        dd.append(abs(x.pearson.median() - y.pearson.median()))
                if dd:
                    rows.append({"res_kb": res, "family": tag, "set": st, "n_cells": len(dd),
                                 "median_abs_diff": float(np.median(dd)), "max_abs_diff": float(np.max(dd))})
    return pd.DataFrame(rows)


def boot_cell_gain_loss(w: pd.DataFrame, nb: int, rng: np.random.Generator, sets: list[str]):
    """For one cell and metric: point gain/loss and bootstrap draws (paired draw of regions across all
    columns) for every set; rows with a missing score in any column are dropped."""
    cols = list(w.columns)
    A = w.to_numpy(float)
    A = A[np.isfinite(A).all(axis=1)]
    n = len(A)
    if n < 20 or ("full", False) not in cols:
        return {}
    med = np.median(A, axis=0)
    idx = rng.integers(0, n, size=(nb, n))
    Mb = np.empty((nb, A.shape[1]))
    for b in range(nb):
        Mb[b] = np.median(A[idx[b]], axis=0)
    ci = {c: k for k, c in enumerate(cols)}
    f, fb = med[ci[("full", False)]], Mb[:, ci[("full", False)]]
    out = {}
    for st in sets:
        if (st, False) not in ci or (st, True) not in ci:
            continue
        a, b_ = med[ci[(st, False)]], med[ci[(st, True)]]
        ab, bb = Mb[:, ci[(st, False)]], Mb[:, ci[(st, True)]]
        out[st] = dict(gain=b_ - a, loss=f - a, gain_b=bb - ab, loss_b=fb - ab, without=a, withm=b_, full=f,
                       n_regions=n)
    return out


def noise_for(model: str, noise_tab: pd.DataFrame) -> float:
    spec = MODELS[model]
    if spec["family"] == "consensus":
        return SEC51_CONSENSUS_NOISE
    t = noise_tab[(noise_tab.res_kb == spec["res"]) & (noise_tab.family == "honest meth")]
    return float(t.median_abs_diff.median()) if len(t) else np.nan


def analyse(R: pd.DataFrame, models: dict, cells_of_model, group_of, groups: list[str], metrics: list[str],
            sets: list[str], noise_of_model, nb: int = 500, seed: int = 0):
    """Replacement fractions per cell and pooled per group. R: long table (model, cell, set, meth, region,
    <metrics>). Pooled: mean gain / mean loss over the cells, percentile bootstrap over cells (>= 3 cells)
    and regions; the fraction is given where the mean loss reaches the noise of the family and its
    bootstrap lower bound is above 0."""
    rng = np.random.default_rng(seed)
    cellrows, store = [], {}
    for model in models:
        nz = noise_of_model(model)
        for cell in cells_of_model(model):
            for metric in metrics:
                w = cell_matrix(R, model, cell, metric)
                if w is None:
                    continue
                for st, r in boot_cell_gain_loss(w, nb, rng, sets).items():
                    store[(model, cell, metric, st)] = r
                    cellrows.append({"model": model, "cell": cell, "cell_group": group_of(cell), "set": st,
                                     "metric": metric, "n_regions": r["n_regions"],
                                     "score_without_meth": r["without"], "score_with_meth": r["withm"],
                                     "score_full": r["full"], "gain": r["gain"],
                                     "gain_lo": np.percentile(r["gain_b"], 2.5),
                                     "gain_hi": np.percentile(r["gain_b"], 97.5), "loss": r["loss"],
                                     "loss_lo": np.percentile(r["loss_b"], 2.5),
                                     "loss_hi": np.percentile(r["loss_b"], 97.5), "noise": nz,
                                     "rf": r["gain"] / r["loss"] if (r["loss"] >= nz and st != "full") else np.nan})
    prow = []
    for model in models:
        nz = noise_of_model(model)
        for grp in groups:
            cells = [c for c in cells_of_model(model) if grp in group_of(c, many=True)]
            for metric in metrics:
                for st in sets:
                    rs = [store[(model, c, metric, st)] for c in cells if (model, c, metric, st) in store]
                    if not rs:
                        continue
                    G = np.array([r["gain_b"] for r in rs])
                    L = np.array([r["loss_b"] for r in rs])
                    k = len(rs)
                    g0, l0 = np.mean([r["gain"] for r in rs]), np.mean([r["loss"] for r in rs])
                    if k < 3:
                        gb, lb = G.mean(0), L.mean(0)
                    else:
                        pick = rng.integers(0, k, size=(G.shape[1], k))
                        cols = np.arange(G.shape[1])[:, None]
                        gb, lb = G[pick, cols].mean(1), L[pick, cols].mean(1)
                    defined = l0 >= nz and np.percentile(lb, 2.5) > 0 and st != "full"
                    rfb = gb / lb
                    prow.append({"model": model, "cell_group": grp, "n_cells": k, "set": st, "metric": metric,
                                 "mean_gain": g0, "gain_lo": np.percentile(gb, 2.5),
                                 "gain_hi": np.percentile(gb, 97.5), "mean_loss": l0,
                                 "loss_lo": np.percentile(lb, 2.5), "loss_hi": np.percentile(lb, 97.5),
                                 "noise": nz, "rf": g0 / l0 if defined else np.nan,
                                 "rf_lo": np.percentile(rfb, 2.5) if defined else np.nan,
                                 "rf_hi": np.percentile(rfb, 97.5) if defined else np.nan,
                                 "n_cells_rf_defined": int(sum(r["loss"] >= nz for r in rs))})
    return pd.DataFrame(cellrows), pd.DataFrame(prow)


def cell_groups(c: str, many: bool = False):
    g = CELL_GROUP[c]
    if not many:
        return g
    return {g, "training all"} if g.startswith("training") else {g}


def tag_models(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(1, "family", df.model.map(lambda m: MODELS[m]["family"]))
    df.insert(2, "res_kb", df.model.map(lambda m: MODELS[m]["res"]))
    df.insert(3, "planes", df.model.map(lambda m: MODELS[m]["planes"]))
    df.insert(4, "seed", df.model.map(lambda m: MODELS[m]["seed"]))
    return df


# =========================================================================== AML scoring
def aml_region_table() -> pd.DataFrame:
    """Per-region scores of the AML test donors: Pearson, structure measures, residual correlation with the
    AML reference (the patient-specific part), patient-specific windows flagged (explain_learn_regions)."""
    import cooler
    from aml_eval import fetch_region, pred_matrix
    import aml_train as at
    import aml_train_eval as te
    rows = []
    for fold, ps in AML_PATIENTS.items():
        for res in (10000, 25000):
            ref = cooler.Cooler(str(at.ref_path(fold, res, None)))
            for p in ps:
                s = next(x for x in te.samples(fold, res) if x["patient"] == p and x["group"] == "patient")
                out = AML_OUT / f"f{fold}" / f"{res // 1000}kb" / s["cell"]
                T = cooler.Cooler(str(s["target"]))
                spec = te.specific_regions(p, res)
                cache: dict = {}
                for st in aml_sets(s["marks"]):
                    f = out / st / f"predict_{s['cell']}.csv"
                    z = out / f"{st}.npz"
                    if not f.exists() or not z.exists():
                        continue
                    df = pd.read_csv(f)
                    zz = np.load(z)
                    rr = {}
                    for reg in zz.files:
                        Tm = fetch_region(T, reg, cache)
                        C = fetch_region(ref, reg, cache)
                        if Tm is None or C is None:
                            continue
                        P = pred_matrix(zz[reg])
                        if P.shape == Tm.shape == C.shape:
                            rr[reg] = te.resid_corr(P, Tm, C)
                    base = st.removesuffix("+meth")
                    d = pd.DataFrame({"region": df.region, "pearson": df.model,
                                      "insulation": df.struct_insulation_corr,
                                      "loop_enrichment": df.struct_loop_enrich_corr,
                                      "compartment": df.struct_compartment_corr,
                                      "boundary_f1": df.struct_boundary_f1,
                                      "resid_ref": df.region.map(rr)})
                    sp = d.region.isin(spec)
                    d["pearson_specific"] = np.where(sp, d.pearson, np.nan)
                    d["resid_ref_specific"] = np.where(sp, d.resid_ref, np.nan)
                    d["insulation_specific"] = np.where(sp, d.insulation, np.nan)
                    d.insert(0, "meth", st.endswith("+meth"))
                    d.insert(0, "set", base)
                    d.insert(0, "cell", p)
                    d.insert(0, "model", f"aml_meth_refin_{res // 1000}kb")
                    rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aml_noise(_model: str) -> float:
    return SEC51_CONSENSUS_NOISE


def aml_tables(nb: int = 500):
    A = aml_region_table()
    if A.empty:
        return None, None, None
    A.to_csv(RES_OUT / "aml_regions.csv.gz", index=False, float_format="%.5f")
    models = {m: {} for m in A.model.unique()}
    pats = lambda m: sorted(A[A.model == m].cell.unique())
    C, P = analyse(A, models, pats, lambda c, many=False: ({"AML donors"} if many else "AML donors"),
                   ["AML donors"], ["pearson", "pearson_specific", "resid_ref", "resid_ref_specific", "insulation",
                                     "insulation_specific", "loop_enrichment", "compartment", "boundary_f1"],
                   [x for x in AML_SETS], aml_noise, nb=nb)
    return A, C, P


def aml_object_scores() -> pd.DataFrame:
    """Enrichment of AML-specific loops / stripes and their controls in the predicted windows (10 kb) for
    every input condition; same objects, windows and scoring as scripts/aml_loop_detection.py."""
    import cooler
    from aml_eval import fetch_region, pred_matrix
    import aml_primary_explain as ex
    import aml_train as at
    import aml_train_eval as te
    import aml_loop_detection as ald
    from hicmultiome.bioscore import BioParams
    prm = BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4)
    items = pd.read_csv(ald.ITEMS, dtype={"patient": str})
    hspc = cooler.Cooler(str(ex.hspc_consensus_path(ald.RES)))
    rows = []
    for fold, ps in AML_PATIENTS.items():
        ref = cooler.Cooler(str(at.ref_path(fold, ald.RES, None)))
        for p in ps:
            s = next(x for x in te.samples(fold, ald.RES) if x["patient"] == p and x["group"] == "patient")
            out = AML_LOOP_OUT / f"f{fold}" / "10kb" / s["cell"]
            names = list(aml_sets(s["marks"]))
            zs = {n: np.load(out / f"{n}.npz") for n in names if (out / f"{n}.npz").exists()}
            if not zs:
                continue
            g = items[items.patient == p]
            T = cooler.Cooler(str(at.OE_DIR / f"xu_{p}_proper_10kb.cool"))
            cache: dict = {}
            for win, gw in g.groupby("win"):
                maps = {"measured": fetch_region(T, win, cache), "reference": fetch_region(ref, win, cache),
                        "hspc": fetch_region(hspc, win, cache)}
                if any(maps[k] is None or maps[k].shape != (ald.WIN, ald.WIN) for k in maps):
                    continue
                for n, z in zs.items():
                    if win in z.files:
                        maps[n] = pred_matrix(z[win])
                enr = {nm: ald.enrich_maps(M, prm) for nm, M in maps.items() if M.shape == (ald.WIN, ald.WIN)}
                for r in gw.itertuples():
                    ws = r.win_bin
                    rec = {"item": r.item}
                    if r.kind == "loop":
                        i, j = r.a_lo - ws, r.b_lo - ws
                        if not (1 <= i < ald.WIN - 1 and 1 <= j < ald.WIN - 1):
                            continue
                        for nm, E in enr.items():
                            blk = E[i - 1:i + 2, j - 1:j + 2]
                            rec[nm] = float(np.nanmax(blk)) if np.isfinite(blk).any() else np.nan
                    else:
                        a0, a1, b0, b1 = r.a_lo - ws, r.a_hi - ws, r.b_lo - ws, r.b_hi - ws
                        if min(a0, b0) < 0 or max(a1, b1) > ald.WIN:
                            continue
                        for nm, M in maps.items():
                            band = M[a0:a1, b0:b1]
                            rec[nm] = float(np.nanmean(band)) if np.isfinite(band).any() else np.nan
                    rows.append(rec)
    D = items.merge(pd.DataFrame(rows), on="item", how="inner")
    return D


def _auroc(pos, neg):
    from scipy.stats import rankdata
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def loop_stats(D: pd.DataFrame, nb: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per kind (loop, stripe), condition and measure: share of AML-specific objects above the 90th
    percentile of the random controls (B_rand) of the same condition, AUROC of the level and of the
    residual over the AML reference against the random controls; replacement fractions with donor
    bootstrap."""
    rng = np.random.default_rng(seed)
    names = [c for c in D.columns if c in aml_sets(MARKS3)]
    donors = sorted(D.patient.unique())
    out = []
    for kind in ("loop", "stripe"):
        K = D[D.kind == kind]
        by = {d: K[K.patient == d] for d in donors}

        def measures(frames, nm):
            x = pd.concat(frames, ignore_index=True)
            pos, ctl = x[x.set == "pos"], x[x.set == "B_rand"]
            lev_p, lev_c = pos[nm].to_numpy(float), ctl[nm].to_numpy(float)
            res_p, res_c = (pos[nm] - pos.reference).to_numpy(float), (ctl[nm] - ctl.reference).to_numpy(float)
            thr = np.nanpercentile(lev_c, 90) if np.isfinite(lev_c).sum() > 10 else np.nan
            return {"share_above_p90": float(np.nanmean(lev_p > thr)) if np.isfinite(thr) else np.nan,
                    "auroc_level": _auroc(lev_p, lev_c), "auroc_resid": _auroc(res_p, res_c),
                    "median_level": float(np.nanmedian(lev_p))}
        base = {nm: measures(list(by.values()), nm) for nm in names}
        boots = []
        for _ in range(nb):
            pick = rng.choice(donors, len(donors))
            frames = [by[d] for d in pick]
            boots.append({nm: measures(frames, nm) for nm in names})
        for meas in ("share_above_p90", "auroc_level", "auroc_resid", "median_level"):
            for st in AML_SETS:
                if st == "full" or st not in base or f"{st}+meth" not in base:
                    continue
                gain = base[f"{st}+meth"][meas] - base[st][meas]
                loss = base["full"][meas] - base[st][meas]
                gb = np.array([b[f"{st}+meth"][meas] - b[st][meas] for b in boots])
                lb = np.array([b["full"][meas] - b[st][meas] for b in boots])
                rfb = gb / lb
                out.append({"kind": kind, "measure": meas, "set": st,
                            "n_pos": int((K.set == "pos").sum()), "n_ctl": int((K.set == "B_rand").sum()),
                            "without_meth": base[st][meas], "with_meth": base[f"{st}+meth"][meas],
                            "full": base["full"][meas], "gain": gain,
                            "gain_lo": np.nanpercentile(gb, 2.5), "gain_hi": np.nanpercentile(gb, 97.5),
                            "loss": loss, "loss_lo": np.nanpercentile(lb, 2.5),
                            "loss_hi": np.nanpercentile(lb, 97.5),
                            "rf": gain / loss if loss > 0 and np.nanpercentile(lb, 2.5) > 0 else np.nan,
                            "rf_lo": np.nanpercentile(rfb, 2.5), "rf_hi": np.nanpercentile(rfb, 97.5)})
    return pd.DataFrame(out)


# =========================================================================== summary and figure
MARK_GROUP = {"CTCF": "CTCF", "RAD21": "cohesin", "SMC3": "cohesin", "H3K27ac": "active", "H3K4me1": "active",
              "H3K4me2": "active", "H3K4me3": "active", "H3K9ac": "active", "H3K36me3": "gene body",
              "H3K79me2": "gene body", "H4K20me1": "gene body", "H3K27me3": "repressive",
              "H3K9me3": "repressive"}
SET_LABEL = {"minus_CTCF": "minus CTCF", "minus_cohesin": "minus cohesin", "minus_active": "minus active",
             "minus_genebody": "minus gene body", "minus_repressive": "minus repressive",
             "cum1": "cum1 (= minus cohesin)", "cum2": "cum2 (+ repressive)", "cum3": "cum3 (+ gene body)",
             "ctcf_only": "CTCF only", "ctcf_h3k27ac": "CTCF + H3K27ac",
             "ctcf_h3k27ac_h3k27me3": "CTCF + H3K27ac + H3K27me3", "seq_only": "no ChIP (sequence, motifs)"}
REF_MODEL = {("consensus", 10): "all12refres_drop_meth_10kb", ("honest", 10): "all12honest_drop_meth_10kb",
             ("honest", 25): "all12honest_drop_meth_25kb"}


def signal_level() -> pd.DataFrame:
    """Part B of section 35 (gradient boosting, unseen cells) and the section 59 CNN on K562 and HMEC."""
    g = pd.read_csv(REPO / "results/methylation/meth_track_prediction.csv")
    u = g[g.scheme == "unseen_cell"].groupby(["mark", "features"])[["r2", "pearson"]].median().unstack()
    out = pd.DataFrame({"mark": u.index})
    out["gb_seq_pearson"] = u[("pearson", "seq")].values
    out["gb_seq_meth_pearson"] = u[("pearson", "seq+meth")].values
    out["gb_seq_r2"] = u[("r2", "seq")].values
    out["gb_seq_meth_r2"] = u[("r2", "seq+meth")].values
    out["gb_meth_gain_pearson"] = out.gb_seq_meth_pearson - out.gb_seq_pearson
    c = pd.read_csv(REPO / "results/seq_cnn/v2/comparison_by_mark.csv")
    c = c[c.set == "unseen_median2"].pivot(index="mark", columns="variant", values="pearson")
    out["cnn_seq_pearson"] = out.mark.map(c["seq"])
    out["cnn_seq_meth_aug_pearson"] = out.mark.map(c["seq_meth_aug"])
    out["cnn_meth_gain_pearson"] = out.cnn_seq_meth_aug_pearson - out.cnn_seq_pearson
    out["group"] = out.mark.map(MARK_GROUP)
    return out


def stage_summary(a) -> None:
    RES_OUT.mkdir(parents=True, exist_ok=True)
    R = load_regions()
    log(f"regions table {R.shape}")
    noise = noise_of(R)
    noise.to_csv(RES_OUT / "seed_noise.csv", index=False, float_format="%.5f")
    models = {m: MODELS[m] for m in MODELS if (R.model == m).any()}
    metrics = MAIN_METRICS + ["scc", "eigenvector"]
    C, P = analyse(R, models, cells_of, cell_groups, ["training WGBS", "training RRBS", "held-out", "training all"],
                   metrics, SET_ORDER, lambda m: noise_for(m, noise), nb=a.nboot)
    C, P = tag_models(C), tag_models(P)
    C.to_csv(RES_OUT / "replacement_by_cell.csv", index=False, float_format="%.5f")
    P.to_csv(RES_OUT / "replacement_pooled.csv", index=False, float_format="%.5f")
    sig = signal_level()
    sig.to_csv(RES_OUT / "signal_level.csv", index=False, float_format="%.4f")
    mark_summary(P, sig, None).to_csv(RES_OUT / "mark_summary.csv", index=False, float_format="%.4f")
    A, AC, AP = aml_tables(a.nboot)
    if AP is not None:
        AC.to_csv(RES_OUT / "aml_replacement_by_donor.csv", index=False, float_format="%.5f")
        AP.to_csv(RES_OUT / "aml_replacement_pooled.csv", index=False, float_format="%.5f")
        mark_summary(P, sig, AP).to_csv(RES_OUT / "mark_summary.csv", index=False, float_format="%.4f")
        D = aml_object_scores()
        if len(D):
            D.to_csv(RES_OUT / "aml_object_scores.csv.gz", index=False, float_format="%.5f")
            loop_stats(D, a.nboot).to_csv(RES_OUT / "aml_loops_replacement.csv", index=False, float_format="%.5f")
    figure(P, AP)
    log("summary written")


def mark_summary(P: pd.DataFrame, sig: pd.DataFrame, AP) -> pd.DataFrame:
    rows = []
    single = {g: s for s, g in GROUP_OF_SET.items()}
    aml_set = {"CTCF": "minus_CTCF", "H3K27ac": "minus_H3K27ac", "H3K27me3": "minus_H3K27me3"}
    for _, r in sig.iterrows():
        rec = r.to_dict()
        st = single[r.group]
        for (fam, res), model in REF_MODEL.items():
            for cg in ("held-out", "training WGBS"):
                q = P[(P.model == model) & (P.cell_group == cg) & (P.set == st) & (P.metric == "pearson")]
                if len(q):
                    q = q.iloc[0]
                    key = f"{fam}_{res}kb_{'heldout' if cg == 'held-out' else 'train'}"
                    rec[f"{key}_loss"] = q.mean_loss
                    rec[f"{key}_gain"] = q.mean_gain
                    rec[f"{key}_rf"] = q.rf
                    rec[f"{key}_rf_lo"] = q.rf_lo
                    rec[f"{key}_rf_hi"] = q.rf_hi
        if AP is not None and r.mark in aml_set:
            for res in (10, 25):
                q = AP[(AP.model == f"aml_meth_refin_{res}kb") & (AP.set == aml_set[r.mark]) &
                       (AP.metric == "pearson")]
                if len(q):
                    q = q.iloc[0]
                    rec[f"aml_{res}kb_loss"], rec[f"aml_{res}kb_gain"], rec[f"aml_{res}kb_rf"] = \
                        q.mean_loss, q.mean_gain, q.rf
        rows.append(rec)
    return pd.DataFrame(rows)


def figure(P: pd.DataFrame, AP) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["minus_CTCF", "minus_cohesin", "minus_active", "minus_genebody", "minus_repressive", "cum2", "cum3",
             "ctcf_h3k27ac_h3k27me3", "ctcf_h3k27ac", "ctcf_only", "seq_only"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    cols = {"consensus": "#4C72B0", "honest": "#DD8452"}

    def bars(a, cg, res, key, title):
        w = 0.38
        for k, fam in enumerate(("consensus", "honest")):
            model = REF_MODEL.get((fam, res))
            if model is None:
                continue
            q = P[(P.model == model) & (P.cell_group == cg) & (P.metric == "pearson")].set_index("set")
            x = np.arange(len(order)) + (k - 0.5) * w
            if key == "rf":
                y = [q.rf.get(s, np.nan) for s in order]
                lo = [q.rf_lo.get(s, np.nan) for s in order]
                hi = [q.rf_hi.get(s, np.nan) for s in order]
            else:
                y = [q.mean_gain.get(s, np.nan) for s in order]
                lo = [q.gain_lo.get(s, np.nan) for s in order]
                hi = [q.gain_hi.get(s, np.nan) for s in order]
            y, lo, hi = map(np.array, (y, lo, hi))
            a.bar(x, y, w, color=cols[fam], label=fam if fam == "honest" and False else f"{'with' if fam == 'consensus' else 'without'} consensus")
            a.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="black", lw=0.8)
        a.set_xticks(range(len(order)))
        a.set_xticklabels([SET_LABEL[s].replace(" (", "\n(") for s in order], rotation=60, ha="right", fontsize=7)
        a.set_title(title, fontsize=10)
        a.axhline(0, color="black", lw=0.5)
    bars(ax[0, 0], "held-out", 10, "rf", "Replacement fraction, held-out cells, 10 kb")
    bars(ax[0, 1], "training WGBS", 10, "rf", "Replacement fraction, training cells, 10 kb")
    bars(ax[0, 2], "held-out", 10, "gain", "Gain from Meth (Pearson), held-out cells, 10 kb")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_ylabel("RF")
    ax[0, 2].set_ylabel("Pearson")
    # panel D: gain vs loss, all pooled rows
    q = P[(P.metric == "pearson") & (P.cell_group.isin(["held-out", "training WGBS"])) & (P.set != "full")]
    for fam in ("consensus", "honest"):
        d = q[q.family == fam]
        ax[1, 0].scatter(d.mean_loss, d.mean_gain, s=14, color=cols[fam], alpha=0.7,
                         label=f"{'with' if fam == 'consensus' else 'without'} consensus")
    mx_ = max(q.mean_loss.max(), 0.05)
    ax[1, 0].plot([0, mx_], [0, mx_], color="grey", lw=0.8)
    ax[1, 0].set_xlabel("loss without the tracks (Pearson)")
    ax[1, 0].set_ylabel("gain from Meth (Pearson)")
    ax[1, 0].set_title("Gain against loss (identity = full replacement)", fontsize=10)
    ax[1, 0].legend(fontsize=8)
    # panel E: AML
    if AP is not None and len(AP):
        sets = ["minus_H3K27ac", "minus_H3K27me3", "minus_CTCF", "minus_H3K27ac_H3K27me3", "minus_all3"]
        for k, (res, c) in enumerate((("10", "#4C72B0"), ("25", "#55A868"))):
            q = AP[(AP.model == f"aml_meth_refin_{res}kb") & (AP.metric == "pearson")].set_index("set")
            x = np.arange(len(sets)) + (k - 0.5) * 0.38
            y = np.array([q.mean_gain.get(s, np.nan) for s in sets])
            lo = np.array([q.gain_lo.get(s, np.nan) for s in sets])
            hi = np.array([q.gain_hi.get(s, np.nan) for s in sets])
            ax[1, 1].bar(x, y, 0.38, color=c, label=f"{res} kb")
            ax[1, 1].errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="black", lw=0.8)
        ax[1, 1].set_xticks(range(len(sets)))
        ax[1, 1].set_xticklabels([s.replace("minus_", "minus ").replace("_", " + ") for s in sets], rotation=60,
                                 ha="right", fontsize=7)
        ax[1, 1].set_title("AML test donors: gain from the patient's WGBS (Pearson)", fontsize=10)
        ax[1, 1].legend(fontsize=8)
        ax[1, 1].axhline(0, color="black", lw=0.5)
    # panel F: signal-level predictability against Hi-C replacement
    ms = pd.read_csv(RES_OUT / "mark_summary.csv")
    if "honest_10kb_heldout_rf" in ms:
        d = ms.drop_duplicates("group")
        ax[1, 2].scatter(ms.gb_seq_meth_pearson, ms.honest_10kb_train_gain, s=30, color="#DD8452")
        for _, r in ms.iterrows():
            ax[1, 2].annotate(r.mark, (r.gb_seq_meth_pearson, r.honest_10kb_train_gain), fontsize=7)
        ax[1, 2].set_xlabel("signal-level Pearson from sequence + Meth (unseen cells)")
        ax[1, 2].set_ylabel("Hi-C gain from Meth when the group is removed (Pearson)")
        ax[1, 2].set_title("Signal-level predictability and Hi-C-level gain, without consensus", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES_OUT / "meth_proxy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    np.seterr(divide="ignore", invalid="ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["predict", "aml", "aml_loops", "summary"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--nboot", type=int, default=500)
    a = ap.parse_args()
    if "predict" in a.stage:
        if a.dry:
            for m in MODELS:
                t = [c for c in cells_of(m) if not all(pred_file(m, s, c) for s in keep_sets())]
                print(m, len(t), t)
        else:
            stage_predict(a.gpu, a.lanes, a.only)
    for st in ("aml", "aml_loops", "summary"):
        if st in a.stage:
            globals()[f"stage_{st}"](a)


if __name__ == "__main__":
    main()
