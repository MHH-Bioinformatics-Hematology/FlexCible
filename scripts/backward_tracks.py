"""Backward elimination of ChIP tracks, validated the Akita way.

Start from all tracks (the known-best set). Each round, try removing each
remaining track, train the reduced set to convergence, and keep the removal that
costs least on the validation set; stop when the best removal drops the
validation score by more than --tol. The Akita TEST regions are never touched;
validation is the median per-region Pearson of log(obs/exp) over a fixed set of
--val-regions held-out windows (train.val_regions), and each candidate is early
stopped with patience --patience (up to --max-epochs). Reuses the 2048 bp cache
via train.select_tracks, so no candidate re-precomputes; two candidates train at
once, one per GPU; finished candidates are reused on restart.

    python scripts/backward_tracks.py --config configs/akita_native.yaml \
        --cache cache/akita_native --val-regions 100 --patience 10 --gpus 0,1
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _key(tracks, chroms=None):
    s = "|".join(sorted(tracks))
    if chroms:                       # fast-proxy runs keyed separately from full-data
        s += "::" + ",".join(sorted(chroms))
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _best_val(text):
    vals = [float(m) for m in re.findall(r"dist-pearson=(-?[0-9.]+)", text)]
    return max(vals) if vals else float("nan")


def _cached(out_root, tracks, chroms=None):
    log = out_root / f"cand_{_key(tracks, chroms)}.log"
    if log.exists():
        t = log.read_text(errors="replace")
        if "early stop at epoch" in t or "done." in t:
            v = _best_val(t)
            return v if v == v else None
    return None


def _launch(base, cache, tracks, val_regions, patience, max_epochs, out_root, gpus, py,
            chroms=None):
    cfg = copy.deepcopy(base)
    cfg["sequence"]["encoder"] = "none"
    cfg["train"].update({
        "val_exclude_overlap": False,   # as run, before the honest default
        "select_tracks": sorted(tracks), "val_regions": val_regions,
        "patience_epochs": patience, "max_epochs": max_epochs,
        "max_steps": 10_000_000, "eval_every": 10_000_000, "resume": False,
        "compile": True, "save_checkpoints": False, "cache_dir": cache,
        "max_minutes": None, "chroms": (sorted(chroms) if chroms else None),
        "output_dir": str(out_root / _key(tracks, chroms)),
    })
    cfg["train"].pop("resume_from", None)
    cfg_path = out_root / f"cand_{_key(tracks, chroms)}.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    log_path = out_root / f"cand_{_key(tracks, chroms)}.log"
    fh = open(log_path, "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
    torchrun = str(Path(py).parent / "torchrun")
    cmd = ([torchrun, "--standalone", f"--nproc_per_node={len(gpus)}",
            "-m", "hicmultiome.train", "--config", str(cfg_path)] if len(gpus) > 1
           else [py, "-m", "hicmultiome.train", "--config", str(cfg_path)])
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    p._fh = fh
    return p, log_path


def eval_sets(base, cache, subsets, val_regions, patience, max_epochs, out_root, gpus, py,
              chroms=None):
    """{frozenset(tracks): val} for each subset. Each candidate trains on ALL
    GPUs (DDP), one candidate at a time; finished candidates are reused."""
    results = {}
    for ts in subsets:
        cv = _cached(out_root, ts, chroms)
        if cv is not None:
            results[frozenset(ts)] = cv
            print(f"    {sorted(ts)}: {cv:.4f} (cached)", flush=True)
            continue
        print(f"    drop-eval {sorted(ts)} on GPUs {gpus} (DDP)", flush=True)
        p, log = _launch(base, cache, ts, val_regions, patience, max_epochs, out_root, gpus, py,
                         chroms=chroms)
        p.wait()
        p._fh.close()
        results[frozenset(ts)] = _best_val(Path(log).read_text(errors="replace"))
        print(f"    {sorted(ts)}: {results[frozenset(ts)]:.4f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--val-regions", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--tol", type=float, default=0.005,
                    help="stop when the best removal drops val by more than this "
                         "(ignored with --full-path)")
    ap.add_argument("--full-path", action="store_true",
                    help="never stop: remove the least-harmful track every round down "
                         "to a single track, recording the whole elimination path")
    ap.add_argument("--chroms", default=None,
                    help="restrict training to these chromosomes (comma list) for a fast "
                         "proxy search, e.g. chr2,chr3")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--out-dir",
                    default="/mnt/processed/wolffjoa/hic-multiome/runs/backward_tracks")
    a = ap.parse_args()

    base = yaml.safe_load(open(a.config))
    all_tracks = [t["name"] for t in base["data"]["tracks"]]
    gpus = [g for g in a.gpus.split(",") if g.strip()]
    chroms = [c for c in a.chroms.split(",") if c.strip()] if a.chroms else None
    out_root = Path(a.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    if chroms:
        print(f"[fast] restricting training to chromosomes: {chroms}", flush=True)

    def val_of(ts):
        return eval_sets(base, a.cache, [ts], a.val_regions, a.patience, a.max_epochs,
                         out_root, gpus, py, chroms=chroms)[frozenset(ts)]

    selected = list(all_tracks)
    print(f"[round 0] full set ({len(selected)} tracks)", flush=True)
    best = val_of(selected)
    rounds = [{"round": 0, "removed": "", "n_tracks": len(selected), "val": best,
               "kept": "+".join(sorted(selected)), "all_removals": ""}]
    rankings = []          # every candidate of every round: which drop, its val, its rank
    print(f"[round 0] all {len(selected)} tracks: val={best:.4f}", flush=True)

    def write_out():
        with open(out_root / "backward.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["round", "removed", "n_tracks", "val",
                                              "kept", "all_removals"])
            w.writeheader()
            w.writerows(rounds)
        with open(out_root / "backward_rankings.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["round", "rank", "removed", "val", "chosen"])
            w.writeheader()
            w.writerows(rankings)

    write_out()
    while len(selected) > 1:
        rnd = len(all_tracks) - len(selected) + 1
        print(f"[round {rnd}] removing one of {sorted(selected)}", flush=True)
        subsets = [[t for t in selected if t != d] for d in selected]
        res = eval_sets(base, a.cache, subsets, a.val_regions, a.patience, a.max_epochs,
                        out_root, gpus, py, chroms=chroms)
        # full ranking of this round's removals (best first) -- keeps 2nd, 3rd, ...
        ranked = sorted(
            [((set(selected) - set(list(fs))).pop(), val) for fs, val in res.items()],
            key=lambda tv: -(tv[1] if tv[1] == tv[1] else -9))
        removed, v = ranked[0]
        accept = a.full_path or (best - v <= a.tol)
        for rk, (t, vv) in enumerate(ranked, 1):
            rankings.append({"round": rnd, "rank": rk, "removed": t, "val": round(vv, 4),
                             "chosen": (t == removed and accept)})
        print(f"[round {rnd}] ranking: "
              + " > ".join(f"-{t}:{vv:.3f}" for t, vv in ranked[:5]), flush=True)
        print(f"[round {rnd}] best removal: -{removed} -> val={v:.4f} "
              f"(running best {best:.4f})", flush=True)
        if not accept:
            rounds.append({"round": rnd, "removed": "(stopped)", "n_tracks": len(selected),
                           "val": best, "kept": "+".join(sorted(selected)),
                           "all_removals": ";".join(f"{t}:{vv:.4f}" for t, vv in ranked)})
            write_out()
            print(f"[stop] removing any track now costs > {a.tol}; "
                  f"final set = {sorted(selected)} ({len(selected)} tracks)", flush=True)
            break
        selected = [t for t in selected if t != removed]   # kept = selected minus removed
        best = max(best, v)
        rounds.append({"round": rnd, "removed": removed, "n_tracks": len(selected),
                       "val": v, "kept": "+".join(sorted(selected)),
                       "all_removals": ";".join(f"{t}:{vv:.4f}" for t, vv in ranked)})
        write_out()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot([r["n_tracks"] for r in rounds], [r["val"] for r in rounds], "o-")
        for r in rounds[1:]:
            plt.annotate("-" + r["removed"], (r["n_tracks"], r["val"]), fontsize=7)
        plt.gca().invert_xaxis()
        plt.xlabel("number of tracks kept")
        plt.ylabel("val median per-region Pearson (100 held-out regions)")
        plt.title("Backward elimination of ChIP tracks")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_root / "backward.png", dpi=150)
    except Exception as e:
        print("plot failed:", e)

    print(f"\n[backward] wrote {out_root/'backward.csv'} and backward.png")
    print("round\tremoved\tn\tval\tkept")
    for r in rounds:
        print(f"{r['round']}\t{r['removed'] or '(all)'}\t{r['n_tracks']}\t{r['val']:.4f}\t{r['kept']}")
    # optimum along the elimination path (relevant in --full-path mode)
    valid = [r for r in rounds if r["removed"] != "(stopped)"]
    bestr = max(valid, key=lambda r: r["val"])
    print(f"\n[optimum] best point on path: {bestr['n_tracks']} tracks, "
          f"val={bestr['val']:.4f} -> {bestr['kept']}")


if __name__ == "__main__":
    main()
