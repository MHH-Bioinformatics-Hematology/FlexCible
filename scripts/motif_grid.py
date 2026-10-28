"""Sequence (motif / learned DNA) x ChIP ablation grid on proxy chromosomes.

Crosses which ChIP tracks the model sees with which sequence-derived channels it
sees, to test how far DNA-encoded binding sites replace the corresponding ChIP
(the backward search showed the CTCF motif makes CTCF ChIP near-redundant).
Every run uses the same cache (cache/full2k_motifs: 13 ChIP + gc/cpg + CTCF,
YY1, ZNF143, MAZ motif planes), the same proxy chromosomes, the same 100 held-out
validation windows and patience-based early stopping; runs go one at a time on
all GPUs (DDP) and finished runs are reused on restart.

    python scripts/motif_grid.py --config configs/full2k_motifs.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
COHESIN = {"RAD21", "SMC3"}
CHIP = {
    "noChIP": [],
    "histones": [t for t in ALL13 if t not in COHESIN | {"CTCF"}],
    "noCohesin": [t for t in ALL13 if t not in COHESIN],
    "all13": ALL13,
}
CTCF_M = ["MA0139.2"]
ALL_M = ["MA0139.2", "MA0095.3", "MA0088.2", "MA1522.1"]   # CTCF YY1 ZNF143 MAZ
SEQ = {                       # (motifs, learned DNA encoder)
    "gcOnly": ([], False),
    "ctcfMotif": (CTCF_M, False),
    "4motifs": (ALL_M, False),
    "4motifs+enc": (ALL_M, True),
}
PROXY = ["chr1", "chr4", "chr7", "chr10", "chr13", "chr16", "chr19"]


def best_val(text):
    v = [float(x) for x in re.findall(r"dist-pearson=(-?[0-9.]+)", text)]
    return max(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--val-regions", type=int, default=100)
    ap.add_argument("--chroms", default=",".join(PROXY))
    ap.add_argument("--out-dir", default="/mnt/processed/wolffjoa/hic-multiome/runs/motif_grid")
    a = ap.parse_args()

    base = yaml.safe_load(open(a.config))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    chroms = [c for c in a.chroms.split(",") if c]
    gpus = [g for g in a.gpus.split(",") if g]
    py = sys.executable
    torchrun = str(Path(py).parent / "torchrun")

    # wait for the cache
    man = Path(base["train"]["cache_dir"]) / "manifest.json"
    while not man.exists():
        print(f"[wait] cache {man}", flush=True); time.sleep(120)

    runs = [(c, s) for c in ["noChIP", "histones", "noCohesin", "all13"]
            for s in ["gcOnly", "ctcfMotif", "4motifs", "4motifs+enc"]]
    rows = []
    for chip, seq in runs:
        name = f"{chip}__{seq}"
        log = out / f"{name}.log"
        t = log.read_text(errors="replace") if log.exists() else ""
        if not ("early stop at epoch" in t or "done." in t):
            motifs, enc = SEQ[seq]
            cfg = copy.deepcopy(base)
            cfg["sequence"]["encoder"] = "dilated" if enc else "none"
            cfg["train"].update({
                "val_exclude_overlap": False,   # as run, before the honest default
                "select_tracks": CHIP[chip], "select_motifs": motifs,
                "chroms": chroms, "val_regions": a.val_regions,
                "patience_epochs": a.patience, "max_epochs": a.max_epochs,
                "max_steps": 10_000_000, "eval_every": 10_000_000, "resume": False,
                "save_checkpoints": True, "track_dropout": 0.0,
                "output_dir": str(out / name),
            })
            cp = out / f"{name}.yaml"
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
            print(f"[run] {name}: {len(CHIP[chip])} ChIP, motifs={motifs}, encoder={enc}", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
            with open(log, "w") as fh:
                subprocess.run([torchrun, "--standalone", f"--nproc_per_node={len(gpus)}",
                                "-m", "flexcible.train", "--config", str(cp)],
                               stdout=fh, stderr=subprocess.STDOUT, env=env)
            t = log.read_text(errors="replace")
        v = best_val(t)
        print(f"[done] {name}: val={v:.4f}", flush=True)
        rows.append({"chip": chip, "seq": seq, "n_chip": len(CHIP[chip]), "val": round(v, 4)})
        with open(out / "motif_grid.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["chip", "seq", "n_chip", "val"])
            w.writeheader(); w.writerows(rows)

    # table: rows = ChIP set, cols = sequence set
    print("\nval (median per-region Pearson, 100 held-out proxy windows)")
    seqs = list(SEQ)
    print("ChIP \\ seq".ljust(12) + "".join(s.rjust(13) for s in seqs))
    for chip in CHIP:
        vals = {r["seq"]: r["val"] for r in rows if r["chip"] == chip}
        print(chip.ljust(12) + "".join(f"{vals.get(s, float('nan')):13.4f}" for s in seqs))
    print("GRID_DONE")


if __name__ == "__main__":
    main()
