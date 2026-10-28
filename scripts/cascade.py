"""Coarse-to-fine cascade 25 -> 10 -> 5 -> 2 kb (multi-resolution section, B).

Each fine level is the 12-cell residual reference model (all12refres setup,
honest validation) with one more input plane: the prediction of the next
coarser level (model.prior; stored at the coarse resolution, each fine bin
takes the coarse bin holding its centre). Training on in-sample predictions
of the coarser model would leak the target, so every prior is out of fold,
2-fold by chromosome:

  fold models   level 25 kb: all12refres config trained on the odd
                chromosomes (fold "odd": chr1, 3, ..., 21) or on the even ones
                and chrX (fold "even"); levels 10 and 5 kb: the cascade config
                (with its own out-of-fold prior) trained the same way.
  priors        for every cell (12 training, K562, HMEC, HUVEC): the odd-fold
                model predicts the even chromosomes and chrX, the even-fold
                model the odd chromosomes (predict.py genome-wide, TTA);
                merged into crosscell_oe/cascade/prior_<cell>_<k>kb.cool.
                Training cells read their own leave-one-out consensus and
                normaliser; held-out cells consensus12. Validation and test
                windows of a fold model lie on its own chromosomes.
  final models  all12casc_10kb (prior 25 kb), all12casc_5kb (prior 10 kb) and
                all12casc_2kb (prior 5 kb) on all training chromosomes, then
                evaluated like every 12-cell run (multicell_expanded --stage
                eval, reference_summary.csv).

    python scripts/cascade.py --stage all
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multicell_expanded as mx                                      # noqa: E402

REPO, PY, TORCHRUN = mx.REPO, mx.PY, mx.TORCHRUN
OUT = mx.REF_OUT
PRIOR_DIR = mx.OE / "cascade"
ODD = [f"chr{i}" for i in range(1, 23, 2)]
EVEN = [f"chr{i}" for i in range(2, 23, 2)] + ["chrX"]
FOLDS = {"odd": ODD, "even": EVEN}
PREDICTS = {"odd": EVEN, "even": ODD}
COARSE_LEVELS = [25000, 10000, 5000]            # levels that feed a finer one
FINAL = [10000, 5000, 2000]
IDENT_CELLS = mx.TRAIN12 + ["k562", "hmec_deep", "huvec_deep"]   # one per identity
LOG = OUT / "cascade.log"


def sh(cmd, log: Path, env=None):
    print("[cmd]", " ".join(map(str, cmd)), flush=True)
    with open(log, "a") as fh:
        return subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=REPO).returncode


def note(msg):
    import time
    with open(OUT / "driver.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def done(log: Path) -> bool:
    t = log.read_text(errors="replace") if log.exists() else ""
    return "early stop at epoch" in t or "done." in t


def level_config(res: int) -> dict:
    """25 kb: the all12refres config; finer: the cascade config."""
    base = "all12refres" if res == 25000 else "all12casc"
    return mx.run_config(mx.run_name(base, res), 100, 10, 500)


def fold_name(res, fold):
    return f"cascfold_{fold}_{res // 1000}kb"


def precompute(res):
    cfg = level_config(res)
    cache = Path(cfg["train"]["cache_dir"])
    if (cache / "manifest.json").exists():
        return
    cp = OUT / f"cascade_{res // 1000}kb.precompute.yaml"
    yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
    if sh([PY, "-m", "flexcible.precompute", "--config", cp, "--workers", 8],
          OUT / f"cascade_{res // 1000}kb.precompute.log"):
        raise SystemExit(f"precompute failed at {res}")


def train_fold(res, fold):
    name = fold_name(res, fold)
    log = OUT / f"{name}.log"
    if done(log):
        return
    cfg = copy.deepcopy(level_config(res))
    cfg["train"].update({"chroms": FOLDS[fold], "output_dir": str(OUT / name)})
    cp = OUT / f"{name}.yaml"
    yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
    log.write_text("")
    note(f"train {name}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1")
    sh([TORCHRUN, "--standalone", "--nproc_per_node=2", "-m", "flexcible.train",
        "--config", cp], log, env)
    if not done(log):
        raise SystemExit(f"training failed: {name}")


def predict_fold(res, fold, cell, gpu):
    """Genome-wide prediction of one cell's chromosomes of the other fold."""
    name = fold_name(res, fold)
    run = OUT / name
    ck = run / (run / "best").read_text().strip()
    out = PRIOR_DIR / "folds" / f"{mx.CELLS[cell][3]}_{res // 1000}kb_{fold}.cool"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part.cool")
    part.unlink(missing_ok=True)
    cmd = [PY, "-m", "flexcible.predict", "--config", ck / "config.yaml", "--checkpoint", ck,
           "--chroms", ",".join(PREDICTS[fold]), "--out", part, "--tta"]
    if cell in mx.TRAIN12:
        cmd += ["--sample", cell]
    else:
        cmd += ["--bigwig-dir", mx.CELLS[cell][1], "--cell", cell,
                "--reference", mx.consensus12(res)]
        if res in mx.CASCADE_FROM:
            cmd += ["--prior", mx.cascade_prior(cell, mx.CASCADE_FROM[res])]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if sh(cmd, OUT / f"{name}.predict.log", env):
        raise SystemExit(f"prediction failed: {name} {cell}")
    part.rename(out)
    return out


def merge(res, cell):
    import cooler
    import pandas as pd

    out = mx.cascade_prior(cell, res)
    if out.exists():
        return
    parts = [cooler.Cooler(str(PRIOR_DIR / "folds" / f"{mx.CELLS[cell][3]}_{res // 1000}kb_{f}.cool"))
             for f in FOLDS]
    px = pd.concat([p.pixels()[:] for p in parts], ignore_index=True)
    px = px.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    if px.duplicated(["bin1_id", "bin2_id"]).any():
        raise SystemExit(f"fold predictions overlap for {cell} {res}")
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)
    cooler.create_cooler(str(part), bins=parts[0].bins()[:][["chrom", "start", "end"]],
                         pixels=px, dtypes={"count": np.float32}, ordered=True)
    part.rename(out)
    Path(str(out) + ".json").write_text(json.dumps({
        "cell": cell, "res": res, "cells": [],
        "out_of_fold": {"odd_model_trained_on": ODD, "predicts": EVEN,
                        "even_model_trained_on": EVEN, "predicts_odd": ODD},
        "models": [str(OUT / fold_name(res, f)) for f in FOLDS],
        "note": "prediction of a cascade level, no measured map of any cell; cells [] "
                "so the reference leakage guard does not apply"}, indent=1))


def priors(res):
    import threading

    todo = [(f, c) for c in IDENT_CELLS for f in FOLDS]
    lock = threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                if not todo:
                    return
                f, c = todo.pop(0)
            predict_fold(res, f, c, gpu)

    ths = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for c in IDENT_CELLS:
        merge(res, c)
    note(f"cascade priors {res // 1000} kb written")


def final(res):
    name = mx.run_name("all12casc", res)
    if sh([PY, "scripts/multicell_expanded.py", "--runs", "all12casc", "--res", res,
           "--stage", "precompute", "--workers", 8], OUT / "cascade_final.log"):
        raise SystemExit(f"precompute failed {name}")
    note(f"train {name}")
    if sh([PY, "scripts/multicell_expanded.py", "--runs", "all12casc", "--res", res,
           "--stage", "train", "--gpus", "0,1"], OUT / "cascade_final.log"):
        raise SystemExit(f"train failed {name}")


def evaluate(res):
    A = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "k562", "hmec_deep"]
    B = ["nhek", "gm23248", "mcf7", "cd14mono", "pc9", "hmec", "huvec", "huvec_deep"]
    ps = [subprocess.Popen([str(x) for x in [PY, "scripts/multicell_expanded.py", "--runs",
                                             "all12casc", "--res", res, "--stage", "eval",
                                             "--gpus", g, "--cells", *cells]],
                           cwd=REPO, stdout=open(OUT / "cascade_eval.log", "a"),
                           stderr=subprocess.STDOUT) for g, cells in (("0", A), ("1", B))]
    for p in ps:
        p.wait()
    sh([PY, "scripts/multicell_expanded.py", "--stage", "refsummary"], OUT / "cascade_eval.log")
    note(f"done all12casc @ {res}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["all", "evals", "precompute", "fold", "priors"],
                    default="all")
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--fold", choices=list(FOLDS), default=None)
    a = ap.parse_args()
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    # single steps, for scripts/gpu_scheduler.py
    if a.stage == "precompute":
        precompute(a.res)
        return
    if a.stage == "fold":
        train_fold(a.res, a.fold)
        return
    if a.stage == "priors":
        priors(a.res)
        return
    if a.stage == "evals":
        for r in FINAL:
            evaluate(r)
        return
    evals = []
    for res in COARSE_LEVELS:
        precompute(res)
        for f in FOLDS:
            train_fold(res, f)
        priors(res)
        fine = {25000: 10000, 10000: 5000, 5000: 2000}[res]
        precompute(fine)                      # the finer level's cache needs these priors
        final(fine)
        # evaluate the finished final model while the next level trains
        import threading
        t = threading.Thread(target=evaluate, args=(fine,))
        t.start()
        evals.append(t)
    for t in evals:
        t.join()
    print("CASCADE_DONE", flush=True)


if __name__ == "__main__":
    main()
