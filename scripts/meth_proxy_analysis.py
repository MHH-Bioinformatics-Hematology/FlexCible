"""DNA methylation as a proxy for missing assays (registry section 74).

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
    "all12refres_drop_meth_10kb": dict(family="consensus", res=10, planes=False, v2=False, seed=0),
    "all12refres_drop_meth_v2_10kb": dict(family="consensus", res=10, planes=False, v2=True, seed=0),
    "all12refres_drop_motifmeth_10kb": dict(family="consensus", res=10, planes=True, v2=False, seed=0),
    "all12refres_drop_motifmeth_25kb": dict(family="consensus", res=25, planes=True, v2=False, seed=0),
    "all12honest_drop_meth_10kb": dict(family="honest", res=10, planes=False, v2=True, seed=0),
    "all12honest_drop_meth_s1_10kb": dict(family="honest", res=10, planes=False, v2=True, seed=1),
    "all12honest_drop_meth_25kb": dict(family="honest", res=25, planes=False, v2=True, seed=0),
    "all12honest_drop_meth_s1_25kb": dict(family="honest", res=25, planes=False, v2=True, seed=1),
    "all12honest_drop_motifmeth_10kb": dict(family="honest", res=10, planes=True, v2=True, seed=0),
    "all12honest_drop_motifmeth_25kb": dict(family="honest", res=25, planes=True, v2=True, seed=0),
    "all12honest_drop_motifmeth_s1_25kb": dict(family="honest", res=25, planes=True, v2=True, seed=1),
}
WGBS_TRAIN = ["gm12878", "imr90", "hepg2", "a549", "h1", "gm23248", "cd14mono"]
RRBS_TRAIN = ["hct116", "mcf7", "hff"]
HELD = ["k562", "hmec_deep"]
SETS = mr.SETS                        # section 35 group sets, without Meth
SET_ORDER = mr.SET_ORDER
METRICS = {"pearson": "model", "scc": "struct_scc", "insulation": "struct_insulation_corr",
           "boundary_f1": "struct_boundary_f1", "loop_enrichment": "struct_loop_enrich_corr",
           "compartment": "struct_compartment_corr", "eigenvector": "struct_eigenvector_abs_corr"}
MAIN_METRICS = ["pearson", "insulation", "boundary_f1", "loop_enrichment", "compartment"]
SINGLE = ["minus_CTCF", "minus_cohesin", "minus_active", "minus_genebody", "minus_repressive"]
CUMUL = ["cum1", "cum2", "cum3", "ctcf_only", "ctcf_h3k27ac", "ctcf_h3k27ac_h3k27me3", "seq_only"]


def cells_of(model: str) -> list[str]:
    return WGBS_TRAIN + (RRBS_TRAIN if MODELS[model]["v2"] else []) + HELD


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
    with open(out / f"predict_{cell}.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call(NICE + [str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def stage_predict(gpu: str, lanes: int, only: list[str] | None) -> None:
    tasks = [(m, c) for m in MODELS if not only or m in only for c in cells_of(m)
             if not all(pred_file(m, s, c) for s in keep_sets())]
    log(f"predict: {len(tasks)} model-cell tasks on gpu{gpu}, {lanes} lanes")
    L = [tasks[i::lanes] for i in range(lanes)]

    def lane(i):
        for m, c in L[i]:
            rc = predict(m, c, gpu, 6 // lanes)
            log(f"predict {m} {c} rc={rc}")
    with ThreadPoolExecutor(lanes) as ex:
        list(ex.map(lane, range(lanes)))
