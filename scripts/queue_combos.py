"""Queue a fixed list of track combinations, run AFTER the backward search.

Waits for any running backward_tracks.py to finish (so it never competes for the
GPUs), then trains each combo to convergence with the exact same Akita-style
validation harness (train.select_tracks + val_regions, patience early stopping,
2-GPU DDP, one candidate at a time). Reuses the akita_native cache and any
already-cached candidate log; writes runs/track_correlation/combos.csv.

    python scripts/queue_combos.py --config configs/akita_native.yaml \
        --cache cache/akita_native --val-regions 100 --patience 10 --gpus 0,1
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backward_tracks as bt   # _launch/_cached/eval_sets/_key

COMBOS = {
    "one_per_cluster_8": ["CTCF", "RAD21", "H3K27ac", "H3K27me3", "H3K36me3",
                          "H3K79me2", "H3K9me3", "H4K20me1"],
    "compact_5": ["CTCF", "RAD21", "H3K27ac", "H3K27me3", "H3K36me3"],
    "minimal_4": ["CTCF", "RAD21", "H3K27ac", "H3K27me3"],
    "swap_5": ["CTCF", "SMC3", "H3K4me1", "H3K27me3", "H3K36me3"],
}


def wait_for_backward():
    while True:
        r = subprocess.run(["pgrep", "-f", "backward_tracks.py"],
                           capture_output=True, text=True)
        pids = [p for p in r.stdout.split() if p.strip()]
        if not pids:
            return
        print(f"  backward search still running (pid {pids}); waiting...",
              flush=True)
        time.sleep(120)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--val-regions", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--out-dir",
                    default="/mnt/processed/wolffjoa/hic-multiome/runs/track_correlation")
    ap.add_argument("--cand-root",
                    default="/mnt/processed/wolffjoa/hic-multiome/runs/backward_tracks",
                    help="reuse candidate logs cached here (same cache/harness)")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()

    if not a.no_wait:
        print("[queue] waiting for backward search to finish before using GPUs",
              flush=True)
        wait_for_backward()
        print("[queue] backward search done; starting combos", flush=True)

    base = yaml.safe_load(open(a.config))
    gpus = [g for g in a.gpus.split(",") if g.strip()]
    cand_root = Path(a.cand_root)           # candidate logs shared with backward search
    cand_root.mkdir(parents=True, exist_ok=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    rows = []
    for name, tracks in COMBOS.items():
        print(f"\n[combo] {name}: {tracks}", flush=True)
        res = bt.eval_sets(base, a.cache, [tracks], a.val_regions, a.patience,
                           a.max_epochs, cand_root, gpus, py)
        val = res[frozenset(tracks)]
        rows.append({"combo": name, "n_tracks": len(tracks), "val": round(val, 4),
                     "tracks": "+".join(sorted(tracks))})
        with open(out / "combos.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["combo", "n_tracks", "val", "tracks"])
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: -r["val"]))
        print(f"[combo] {name}: val={val:.4f}", flush=True)

    print("\n[queue] done. combos ranked by val:")
    for r in sorted(rows, key=lambda r: -r["val"]):
        print(f"  {r['val']:.4f}  {r['combo']:20s} {r['tracks']}")
    print("wrote", out / "combos.csv")


if __name__ == "__main__":
    main()
