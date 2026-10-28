"""Promoter capture Hi-C in training, GM12878 2 kb, full data.

Variants, each on top of the reference model of scripts/full_confirm.py
(all 13 ChIP tracks, CTCF/YY1/ZNF143/MAZ motifs, no encoder, no distance bias,
track_dropout 0, patience 10 epochs, 100 validation windows):

  A   capture as an auxiliary TARGET only: second output channel, bait-masked
      loss (capture.loss_weight), no capture input.
  B   capture as INPUT with modality dropout (capture.input_dropout of the
      training windows see the absent encoding) plus the auxiliary target on
      those blanked windows. Scored twice from one checkpoint: capture given
      and capture absent.

The reference checkpoints (full__all13__4motifs, seeds 0 and 1) are rescored
on the same windows by scripts/eval_capture_windows.py. Training waits until
--gpu-flag exists. Runs go to <out-dir>/<variant>__seed<k>; finished runs are
reused. Summary: <out-dir>/capture_summary.csv.

    python scripts/capture_experiment.py --variants A B --seeds 0 1
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNS = Path("/mnt/processed/wolffjoa/hic-multiome/runs")
BASE_RUNS = {0: RUNS / "motif_grid/full__all13__4motifs",
             1: RUNS / "motif_grid/full__all13__4motifs__seed1"}
ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
MOTIFS4 = ["MA0139.2", "MA0095.3", "MA0088.2", "MA1522.1"]


def capin_cache(src: Path, dst: Path) -> None:
    """A view of the capture cache whose bookkeeping says capture_input=true
    (the shards are shared; only manifest/channels differ). Checkpoints copy
    channels.json, and predict/eval read capture_input from it."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("norm_*.json"):
        shutil.copy2(f, dst / f.name)
    for name in ("manifest.json", "channels.json"):
        d = json.loads((src / name).read_text())
        d["capture_input"] = True
        (dst / name).write_text(json.dumps(d, indent=2))


def run_cfg(base: dict, variant: str, seed: int, out: Path, cache: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["sequence"]["encoder"] = "none"
    cfg["model"]["dist_bias"] = False
    cfg["capture"]["use_as_input"] = variant == "B"
    cfg["train"].update({
        "val_exclude_overlap": False,   # as run, before the honest default
        "seed": seed, "select_tracks": ALL13, "select_motifs": MOTIFS4,
        "chroms": None, "val_regions": 100, "patience_epochs": 10,
        "max_epochs": 100, "max_steps": 10_000_000, "eval_every": 10_000_000,
        "resume": False, "save_checkpoints": True, "track_dropout": 0.0,
        "output_dir": str(out), "cache_dir": cache,
    })
    return cfg


def finished(log: Path) -> bool:
    t = log.read_text(errors="replace") if log.exists() else ""
    return "early stop at epoch" in t or "done." in t


def evaluate(run: Path, cache: str, capture: str, out: Path, env: dict) -> dict:
    if not out.exists():
        subprocess.run([sys.executable, str(REPO / "scripts/eval_capture_windows.py"),
                        "--run", str(run), "--cache", cache, "--capture", capture,
                        "--out", str(out)], check=True, cwd=REPO, env=env)
    return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--config", default=str(REPO / "configs/full2k_capture.yaml"))
    ap.add_argument("--out-dir", default=str(RUNS / "capture"))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--gpu-flag", default=str(RUNS / "GPU_FREE_FOR_CAPTURE"))
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(a.config))
    cache = base["train"]["cache_dir"]
    cache_b = cache.rstrip("/") + "_capin"
    capin_cache(REPO / cache, REPO / cache_b)
    gpus = [g for g in a.gpus.split(",") if g]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
    torchrun = str(Path(sys.executable).parent / "torchrun")

    rows = []
    # reference model, rescored on the capture windows (no GPU needed to wait for)
    for seed, brun in BASE_RUNS.items():
        r = evaluate(brun, cache, "absent", out / f"base__seed{seed}.eval.json", env)
        rows.append({"model": "base (no capture)", "seed": seed, "capture_at_test": "no", **r})

    for seed in a.seeds:
        for v in a.variants:
            name = f"{v}__seed{seed}"
            run = out / name
            log = out / f"{name}.log"
            if not a.eval_only and not finished(log):
                while not Path(a.gpu_flag).exists():
                    print(f"[wait] {a.gpu_flag} not present, {time.strftime('%H:%M')}",
                          flush=True)
                    time.sleep(180)
                cfg = run_cfg(base, v, seed, run, cache_b if v == "B" else cache)
                cp = out / f"{name}.yaml"
                yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
                print(f"[run] {name}", flush=True)
                with open(log, "w") as fh:
                    subprocess.run([torchrun, "--standalone",
                                    f"--nproc_per_node={len(gpus)}", "-m",
                                    "flexcible.train", "--config", str(cp)],
                                   stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
            if not finished(log):
                print(f"[skip] {name} not finished", flush=True)
                continue
            modes = ["absent", "present"] if v == "B" else ["absent"]
            for m in modes:
                r = evaluate(run, cache, m, out / f"{name}.{m}.eval.json", env)
                label = {"A": "A: capture as auxiliary target",
                         "B": "B: capture input + aux target"}[v]
                rows.append({"model": label, "seed": seed,
                             "capture_at_test": "yes" if m == "present" else "no", **r})
            print(f"[done] {name}", flush=True)

    keys = ["model", "seed", "capture_at_test", "n", "oe_median_r", "bait_median_r",
            "nonbait_median_r", "capture_channel_bait_median_r", "bait_pixel_fraction",
            "run"]
    with open(out / "capture_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    main()
