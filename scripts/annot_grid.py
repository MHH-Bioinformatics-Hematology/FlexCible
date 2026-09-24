"""Follow-up arms to scripts/motif_grid.py, same protocol (proxy chromosomes,
100 held-out windows, patience early stopping, DDP on all GPUs, finished runs
reused on restart):

  annotation arm (cache/full2k_annot = full2k_motifs + 8 cell-invariant
  annotation tracks ANN_*: CpG islands, TSS, gene body, exon, LTR/LINE/SINE/
  Satellite coverage): ChIP {noChIP, noCohesin, all13} x 4motifs+annot,
  plus CTCF+cohesin ChIP only x {4motifs, 4motifs+annot}
  (histone ChIP replaced by annotation). The {noChIP, noCohesin, all13} x
  4motifs baselines are the motif-grid runs (identical rows in both caches).

  distance-bias arm (cache/full2k_motifs): all13 + CTCF motif with and without
  a learned per-distance output bias, both with checkpoints so they can be
  re-scored in log-obs space (scripts/eval_val_windows.py).

  honest-validation arm: {noChIP, noCohesin, histones} x {4motifs,
  4motifs+enc} with train.val_exclude_overlap (no training window shares a bin
  with a validation window), since the DNA encoder memorises position.

Waits for GRID_DONE in the motif grid log so the GPUs are never shared.

    python scripts/annot_grid.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motif_grid import ALL13, ALL_M, CHIP, CTCF_M, PROXY, best_val  # noqa: E402

ANN = ["ANN_CGI", "ANN_TSS", "ANN_GeneBody", "ANN_Exon", "ANN_LTR", "ANN_LINE",
       "ANN_SINE", "ANN_Satellite"]
CHIP2 = dict(CHIP, ctcfCohesin=["CTCF", "RAD21", "SMC3"])
A_CFG, M_CFG = "configs/full2k_annot.yaml", "configs/full2k_motifs.yaml"

# name, config, ChIP set, annot?, motifs, encoder, dist_bias, save_ckpt
RUNS = (
    [(f"{c}__4motifs+annot", A_CFG, c, True, ALL_M, False, False, False)
     for c in ["noChIP", "noCohesin", "all13"]]
    + [("ctcfCohesin__4motifs", A_CFG, "ctcfCohesin", False, ALL_M, False, False, False),
       ("ctcfCohesin__4motifs+annot", A_CFG, "ctcfCohesin", True, ALL_M, False, False, False)]
    + [("all13__ctcfMotif__ckpt", M_CFG, "all13", False, CTCF_M, False, False, True),
       ("all13__ctcfMotif+distBias", M_CFG, "all13", False, CTCF_M, False, True, True)]
)
# honest-validation arm: the 1 Mb DNA encoder memorises genomic position, and
# the default validation windows overlap training windows by half a window, so
# encoder runs are compared with and without the encoder with every training
# window overlapping a validation window removed (train.val_exclude_overlap)
HV = [(f"{c}__{s}__hv", M_CFG, c, False, ALL_M, enc, False, False)
      for c in ["noChIP", "noCohesin", "histones"]
      for s, enc in [("4motifs", False), ("4motifs+enc", True)]]
RUNS = RUNS + HV


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--val-regions", type=int, default=100)
    ap.add_argument("--chroms", default=",".join(PROXY))
    ap.add_argument("--out-dir", default="/mnt/processed/wolffjoa/hic-multiome/runs/motif_grid")
    ap.add_argument("--only", default="", help="comma list of run names")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    chroms = [c for c in a.chroms.split(",") if c]
    gpus = [g for g in a.gpus.split(",") if g]
    torchrun = str(Path(sys.executable).parent / "torchrun")
    only = {x for x in a.only.split(",") if x}

    grid_log = out / "grid.log"
    while "GRID_DONE" not in (grid_log.read_text(errors="replace") if grid_log.exists() else ""):
        print("[wait] motif grid", flush=True); time.sleep(120)

    rows = []
    for name, cfgp, chip, annot, motifs, enc, dbias, ckpt in RUNS:
        if only and name not in only:
            continue
        base = yaml.safe_load(open(cfgp))
        man = Path(base["train"]["cache_dir"]) / "manifest.json"
        while not man.exists():
            print(f"[wait] cache {man}", flush=True); time.sleep(120)
        log = out / f"{name}.log"
        t = log.read_text(errors="replace") if log.exists() else ""
        if not ("early stop at epoch" in t or "done." in t):
            tracks = CHIP2[chip] + (ANN if annot else [])
            cfg = copy.deepcopy(base)
            cfg["sequence"]["encoder"] = "dilated" if enc else "none"
            cfg["model"]["dist_bias"] = bool(dbias)
            cfg["train"]["val_exclude_overlap"] = name.endswith("__hv")
            cfg["train"].update({
                "select_tracks": tracks, "select_motifs": motifs,
                "chroms": chroms, "val_regions": a.val_regions,
                "patience_epochs": a.patience, "max_epochs": a.max_epochs,
                "max_steps": 10_000_000, "eval_every": 10_000_000, "resume": False,
                "save_checkpoints": True, "track_dropout": 0.0,
                "output_dir": str(out / name),
            })
            cp = out / f"{name}.yaml"
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
            print(f"[run] {name}: {len(tracks)} tracks, motifs={motifs}, encoder={enc}, "
                  f"dist_bias={dbias}", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
            with open(log, "w") as fh:
                subprocess.run([torchrun, "--standalone", f"--nproc_per_node={len(gpus)}",
                                "-m", "hicmultiome.train", "--config", str(cp)],
                               stdout=fh, stderr=subprocess.STDOUT, env=env)
            t = log.read_text(errors="replace")
        v = best_val(t)
        print(f"[done] {name}: val={v:.4f}", flush=True)
        rows.append({"name": name, "val": round(v, 4)})
        with open(out / "annot_grid.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "val"])
            w.writeheader(); w.writerows(rows)
    print("ANNOT_GRID_DONE", flush=True)


if __name__ == "__main__":
    main()
