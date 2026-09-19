"""Periodically score the current best checkpoint on Akita's 2 kb benchmark, so
a running training can be watched as a trend rather than a single end number.

Every --interval-min it scores checkpoint-best (even chromosomes, no TTA -- the
fast read), records (time, epoch, median) to trend.csv, and redraws trend.png
with the 0.410 GC/CpG-only baseline and the 0.556 Akita target marked. It stops
when the training process for the config is gone (one final score first).

    python scripts/score_trend.py --config configs/akita_2kb_seq.yaml \
        --run-dir /mnt/.../runs/akita_2kb_seq --interval-min 30 --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path


def _epoch_from_log(log_path: Path) -> int:
    if not log_path.exists():
        return -1
    last = -1
    for m in re.finditer(r"epoch (\d+) step", log_path.read_text(errors="replace")):
        last = int(m.group(1))
    return last


def _score(py, config, ckpt, out_csv, gpu) -> float:
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    r = subprocess.run(
        [py, "scripts/score_like_hicgan.py", "--config", config,
         "--checkpoint", str(ckpt), "--even-only", "--out", str(out_csv)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)
    m = re.search(r"ours\s*:\s*median\s+([0-9.]+)", r.stdout.decode(errors="replace"))
    return float(m.group(1)) if m else float("nan")


def _plot(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in rows]
    med = [r["akita_median"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, med, "o-", color="#2c7fb8", label="ours (seq encoder)")
    ax.axhline(0.410, color="#31a354", ls="--", lw=1, label="GC/CpG-only baseline 0.410")
    ax.axhline(0.556, color="#d62728", ls=":", lw=1, label="Akita 0.556")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Akita 2 kb median (even chroms)")
    ax.set_title("Sequence-encoder run: Akita score vs training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval-min", type=float, default=30)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--proc-match", default=None,
                    help="pgrep -f pattern for the training process; default: the config")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    train_log = run_dir.parent / (run_dir.name + ".train.log")
    trend_csv = run_dir / "trend.csv"
    trend_png = run_dir / "trend.png"
    match = a.proc_match or a.config
    py = sys.executable
    rows = []
    if trend_csv.exists():
        with open(trend_csv) as f:
            rows = [{"time": r["time"], "epoch": int(r["epoch"]),
                     "akita_median": float(r["akita_median"])}
                    for r in csv.DictReader(f)]

    def training_alive() -> bool:
        return subprocess.run(["pgrep", "-f", match],
                              stdout=subprocess.DEVNULL).returncode == 0

    while True:
        alive = training_alive()
        ckpt = run_dir / "checkpoint-best"
        if (ckpt / "model.pt").exists():
            med = _score(py, a.config, ckpt, run_dir / "akita_trend.csv", a.gpu)
            row = {"time": dt.datetime.now().isoformat(timespec="seconds"),
                   "epoch": _epoch_from_log(train_log), "akita_median": med}
            rows.append(row)
            with open(trend_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["time", "epoch", "akita_median"])
                w.writeheader()
                w.writerows(rows)
            try:
                _plot(rows, trend_png)
            except Exception:
                pass
            print(f"[trend] {row['time']} epoch {row['epoch']} akita_median {med:.3f}",
                  flush=True)
        if not alive:
            print("[trend] training process gone; final score done, stopping.",
                  flush=True)
            break
        time.sleep(a.interval_min * 60)


if __name__ == "__main__":
    main()
