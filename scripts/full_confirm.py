"""Re-run selected proxy-grid configurations on FULL data (all training
chromosomes), same settings otherwise, with checkpoints, to check that the
proxy conclusions of scripts/motif_grid.py / scripts/annot_grid.py hold.

Run names follow the grids: <chip>__<seq>, chip in {noChIP, histones,
noCohesin, all13, ctcfCohesin}; seq a '+'-joined subset of
{gcOnly, ctcfMotif, 4motifs, annot, enc, distBias}. Outputs go to
<out-dir>/full__<name>. Finished runs are reused.

    python scripts/full_confirm.py all13__ctcfMotif noCohesin__4motifs+enc
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annot_grid import A_CFG, ANN, CHIP2, M_CFG  # noqa: E402
from motif_grid import ALL_M, CTCF_M, best_val  # noqa: E402


def spec(name):
    chip, seq = name.split("__")
    parts = set(seq.split("+"))
    motifs = (ALL_M if "4motifs" in parts else CTCF_M if "ctcfMotif" in parts else [])
    annot = "annot" in parts
    return dict(cfg=A_CFG if annot else M_CFG,
                tracks=CHIP2[chip] + (ANN if annot else []), motifs=motifs,
                enc="enc" in parts, dist_bias="distBias" in parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--val-regions", type=int, default=100)
    ap.add_argument("--out-dir", default="/mnt/processed/wolffjoa/hic-multiome/runs/motif_grid")
    ap.add_argument("--seed", type=int, default=None,
                    help="train.seed override (replicates); also changes the "
                         "validation windows, the test split stays the same")
    a = ap.parse_args()
    out = Path(a.out_dir)
    gpus = [g for g in a.gpus.split(",") if g]
    torchrun = str(Path(sys.executable).parent / "torchrun")
    for name in a.names:
        s = spec(name)
        run = f"full__{name}" + (f"__seed{a.seed}" if a.seed is not None else "")
        log = out / f"{run}.log"
        t = log.read_text(errors="replace") if log.exists() else ""
        if not ("early stop at epoch" in t or "done." in t):
            cfg = copy.deepcopy(yaml.safe_load(open(s["cfg"])))
            cfg["sequence"]["encoder"] = "dilated" if s["enc"] else "none"
            cfg["model"]["dist_bias"] = s["dist_bias"]
            if a.seed is not None:
                cfg["train"]["seed"] = a.seed
            cfg["train"].update({
                "select_tracks": s["tracks"], "select_motifs": s["motifs"],
                "chroms": None, "val_regions": a.val_regions,
                "patience_epochs": a.patience, "max_epochs": a.max_epochs,
                "max_steps": 10_000_000, "eval_every": 10_000_000, "resume": False,
                "save_checkpoints": True, "track_dropout": 0.0,
                "output_dir": str(out / run),
            })
            cp = out / f"{run}.yaml"
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
            print(f"[run] {run}: {s}", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
            with open(log, "w") as fh:
                subprocess.run([torchrun, "--standalone", f"--nproc_per_node={len(gpus)}",
                                "-m", "hicmultiome.train", "--config", str(cp)],
                               stdout=fh, stderr=subprocess.STDOUT, env=env)
            t = log.read_text(errors="replace")
        print(f"[done] {run}: val={best_val(t):.4f}", flush=True)
    print("FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
