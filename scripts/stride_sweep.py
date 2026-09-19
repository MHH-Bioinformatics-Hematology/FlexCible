"""Sweep training-set size via window_subsample: accuracy vs windows vs time.

For each factor k, train from scratch (fixed wall-clock budget) on every k-th
cached window -- effective stride = window.stride * k -- then score the best
checkpoint on Akita's own 2 kb benchmark. Writes a CSV and two plots:
accuracy vs number of training windows, and training time vs windows. It
answers "how much training data does the model actually need" and how stride
trades accuracy for speed.

All local. Each training run uses torchrun over the given GPUs; scoring uses
one GPU. Run it when the GPUs are free:

    CUDA-side is chosen per step, so do not set CUDA_VISIBLE_DEVICES yourself.
    python scripts/stride_sweep.py --config configs/akita_2kb_seq.yaml \
        --factors 1,2,4,8,16 --minutes 45 --gpus 0,1
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _run(cmd, log_path, env):
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False)
    return Path(log_path).read_text(errors="replace")


def _parse_akita_median(text: str) -> float:
    m = re.search(r"ours\s*:\s*median\s+([0-9.]+)", text)
    return float(m.group(1)) if m else math.nan


def _parse_n_windows(text: str) -> float:
    m = re.search(r"window_subsample=\d+:\s*(\d+)\s*training windows", text)
    if not m:
        m = re.search(r"(\d+)\s*train windows", text)
    return float(m.group(1)) if m else math.nan


def _best_val(run_dir: Path) -> float:
    lc = run_dir / "learning_curve.csv"
    if not lc.exists():
        return math.nan
    import pandas as pd

    df = pd.read_csv(lc)
    vv = df.dropna(subset=["val_pearson"])
    return float(vv["val_pearson"].max()) if len(vv) else math.nan


def _plot(rows, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["n_windows"])
    x = [r["n_windows"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(x, [r["akita_median"] for r in rows], "o-", color="#2c7fb8",
             label="Akita 2 kb median")
    ax1.plot(x, [r["best_val_pearson"] for r in rows], "s--", color="#31a354",
             label="best val dist-Pearson")
    ax1.axhline(0.556, color="#d62728", lw=1, ls=":", label="Akita (recorded) 0.556")
    ax1.set_xlabel("training windows")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Accuracy vs training-set size")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(x, [r["train_min"] for r in rows], "o-", color="#756bb1")
    ax2.set_xlabel("training windows")
    ax2.set_ylabel("training time (min)")
    ax2.set_title("Training time vs training-set size")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--factors", default="1,2,4,8,16")
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--out-dir",
                    default="/mnt/processed/wolffjoa/hic-multiome/runs/stride_sweep")
    ap.add_argument("--tta", action="store_true", default=True)
    a = ap.parse_args()

    factors = [int(x) for x in a.factors.split(",") if x.strip()]
    gpus = [g for g in a.gpus.split(",") if g.strip()]
    ngpu = len(gpus)
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(a.config))
    base_stride = int(base["window"]["stride"])
    py = sys.executable
    torchrun = str(Path(py).parent / "torchrun")

    rows = []
    for k in factors:
        run_dir = outdir / f"k{k}"
        cfg = copy.deepcopy(base)
        cfg["train"].update({"window_subsample": k, "max_minutes": a.minutes,
                             "resume": False, "output_dir": str(run_dir)})
        cfg_path = outdir / f"k{k}.yaml"
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
        print(f"[sweep] k={k} eff_stride={base_stride * k}: training {a.minutes} min "
              f"on GPU(s) {gpus}", flush=True)
        t0 = time.time()
        cmd = ([torchrun, "--standalone", f"--nproc_per_node={ngpu}",
                "-m", "hicmultiome.train", "--config", str(cfg_path)] if ngpu > 1
               else [py, "-m", "hicmultiome.train", "--config", str(cfg_path)])
        tlog = _run(cmd, outdir / f"k{k}.train.log", env)
        train_min = (time.time() - t0) / 60.0

        senv = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0])
        scmd = [py, "scripts/score_like_hicgan.py", "--config", str(cfg_path),
                "--checkpoint", str(run_dir / "checkpoint-best"),
                "--out", str(run_dir / "akita.csv")]
        if a.tta:
            scmd.append("--tta")
        print(f"[sweep] k={k}: scoring on Akita benchmark", flush=True)
        slog = _run(scmd, outdir / f"k{k}.score.log", senv)

        row = {"k": k, "eff_stride": base_stride * k,
               "n_windows": _parse_n_windows(tlog), "train_min": round(train_min, 1),
               "akita_median": _parse_akita_median(slog),
               "best_val_pearson": _best_val(run_dir)}
        rows.append(row)
        print(f"[sweep] k={k}: {row}", flush=True)

    import csv as _csv
    with open(outdir / "sweep.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _plot(rows, outdir / "sweep.png")
    print(f"[sweep] wrote {outdir / 'sweep.csv'} and {outdir / 'sweep.png'}")
    print("\nk\teff_stride\twindows\ttrain_min\takita_median\tbest_val")
    for r in rows:
        print(f"{r['k']}\t{r['eff_stride']}\t{r['n_windows']:.0f}\t{r['train_min']}"
              f"\t{r['akita_median']:.3f}\t{r['best_val_pearson']:.3f}")


if __name__ == "__main__":
    main()
