"""Greedy forward selection of ChIP tracks: find the best-performing combination.

Start from the single best track, then repeatedly add the track that most
improves the score, each candidate trained for a short fixed budget (default
100 steps) and ranked by its best validation per-distance Pearson. No encoder;
the sequence GC/CpG features and motif planes are kept as they were. Reuses the
full precomputed cache (train.select_tracks slices the rows) so no candidate
re-precomputes. Two candidates train at once, one per GPU.

Writes greedy.csv (round, added track, best-val, selected set) and greedy.png
(best-val vs number of tracks), and prints the winning combination.

    python scripts/greedy_tracks.py --config configs/akita_2kb_ddp.yaml \
        --cache cache/akita_full_2kb --steps 100 --gpus 0,1
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


def _best_val(log_text: str) -> float:
    vals = [float(m) for m in re.findall(r"dist-pearson=(-?[0-9.]+)", log_text)]
    return max(vals) if vals else float("nan")


def _cached_val(out_root, selected, track):
    """Reuse a candidate already trained in a previous (interrupted) run: if its
    log shows a completed run (early stop or done), return its best val, so the
    greedy is resumable and never recomputes finished candidates."""
    log = out_root / f"cand_{len(selected)}_{track}.log"
    if not log.exists():
        return None
    txt = log.read_text(errors="replace")
    if "early stop at epoch" in txt or "done." in txt:
        v = _best_val(txt)
        return v if v == v else None
    return None


def _run_candidate(base, cache, max_epochs, patience, out_root, selected, track, gpu, py):
    """Launch one 1-GPU training for selected+[track], with per-candidate early
    stopping (up to max_epochs, stop after `patience` epochs without val
    improvement). Returns (Popen, log_path)."""
    cfg = copy.deepcopy(base)
    cfg["train"].update({
        "select_tracks": selected + [track],
        "max_epochs": max_epochs, "patience_epochs": patience,
        "max_steps": 10_000_000, "eval_every": 10_000_000,   # epoch-end validate only
        "resume": False, "compile": True, "cache_dir": cache, "max_minutes": None,
        "save_checkpoints": False,
        "output_dir": str(out_root / ("__".join(selected + [track])[:80] or track)),
    })
    cfg["train"].pop("resume_from", None)
    cfg_path = out_root / f"cand_{len(selected)}_{track}.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    log_path = out_root / f"cand_{len(selected)}_{track}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    fh = open(log_path, "w")
    p = subprocess.Popen([py, "-m", "hicmultiome.train", "--config", str(cfg_path)],
                         stdout=fh, stderr=subprocess.STDOUT, env=env)
    p._fh = fh
    return p, log_path


def _eval_round(base, cache, max_epochs, patience, out_root, selected, candidates, gpus, py):
    """Train every candidate track (added to `selected`) across the GPUs as a
    rolling pool: the instant a GPU's candidate finishes, the next queued one is
    launched on it, so no GPU idles waiting for a slower partner. Returns
    {track: best_val}."""
    results = {}
    queue = []
    for track in candidates:                 # reuse finished candidates
        cv = _cached_val(out_root, selected, track)
        if cv is not None:
            results[track] = cv
            print(f"    +{track}: best_val={cv:.4f} (cached)", flush=True)
        else:
            queue.append(track)
    free = list(gpus)
    running = {}                       # gpu -> (track, Popen, log_path)
    while queue or running:
        while queue and free:
            gpu = free.pop(0)
            track = queue.pop(0)
            print(f"    train +{track} on GPU {gpu}", flush=True)
            p, log_path = _run_candidate(base, cache, max_epochs, patience,
                                         out_root, selected, track, gpu, py)
            running[gpu] = (track, p, log_path)
        time.sleep(5)
        for gpu, (track, p, log_path) in list(running.items()):
            if p.poll() is not None:
                p._fh.close()
                results[track] = _best_val(Path(log_path).read_text(errors="replace"))
                print(f"    +{track}: best_val={results[track]:.4f} (GPU {gpu} free)",
                      flush=True)
                del running[gpu]
                free.append(gpu)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5,
                    help="stop a candidate after this many epochs with no val gain")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--out-dir",
                    default="/mnt/processed/wolffjoa/hic-multiome/runs/greedy_tracks")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="stop when the best add improves best-val by <= tol")
    a = ap.parse_args()

    base = yaml.safe_load(open(a.config))
    base["sequence"]["encoder"] = "none"          # no encoder for the ablation
    all_tracks = [t["name"] for t in base["data"]["tracks"]]
    gpus = [g for g in a.gpus.split(",") if g.strip()]
    out_root = Path(a.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    selected, remaining, rounds = [], list(all_tracks), []
    best_so_far = float("-inf")
    while remaining:
        print(f"[round {len(selected)+1}] selected={selected}", flush=True)
        res = _eval_round(base, a.cache, a.max_epochs, a.patience, out_root, selected, remaining, gpus, py)
        track, val = max(res.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else -9))
        rounds.append({"round": len(selected) + 1, "added": track, "best_val": val,
                       "selected": "+".join(selected + [track]),
                       "all_candidates": ";".join(f"{k}:{v:.4f}" for k, v in
                                                  sorted(res.items(), key=lambda x: -x[1]))})
        print(f"[round {len(selected)+1}] winner: +{track} (best_val={val:.4f})", flush=True)
        if val - best_so_far <= a.tol:
            print(f"[stop] no improvement over {best_so_far:.4f} (tol {a.tol}); "
                  f"best combination = {selected}", flush=True)
            break
        best_so_far = val
        selected.append(track)
        remaining.remove(track)

    with open(out_root / "greedy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "added", "best_val", "selected",
                                          "all_candidates"])
        w.writeheader()
        w.writerows(rounds)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["round"] for r in rounds]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, [r["best_val"] for r in rounds], "o-")
        for r in rounds:
            plt.annotate(r["added"], (r["round"], r["best_val"]), fontsize=7,
                         rotation=30, ha="left")
        plt.xlabel("number of tracks (greedy order)")
        plt.ylabel(f"best val dist-Pearson (early-stop, max {a.max_epochs} ep)")
        plt.title("Greedy ChIP-track selection")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_root / "greedy.png", dpi=150)
    except Exception as e:
        print("plot failed:", e)

    best = max(rounds, key=lambda r: r["best_val"])
    print(f"\n[greedy] wrote {out_root/'greedy.csv'} and greedy.png")
    print(f"[greedy] BEST COMBINATION ({best['best_val']:.4f}): {best['selected']}")
    print("\nround\tadded\tbest_val\tselected")
    for r in rounds:
        print(f"{r['round']}\t{r['added']}\t{r['best_val']:.4f}\t{r['selected']}")


if __name__ == "__main__":
    main()
