"""Pairwise correlation of the 13 ChIP tracks at 2 kb (CPU only).

Reads the precomputed per-bin sym rows from the akita_native cache (rows 0-12 are
the 13 tracks, in channels.json order), pools every bin of every TRAIN window
(test/val windows are never touched), and computes the 13x13 Pearson correlation
matrix. Then greedily reports redundant clusters (|r| >= --thresh) so we can pick
combinations to try in the input search. Writes a CSV, a clustered heatmap, and a
short suggestion list. Numpy only; no GPU, no torch.

    python scripts/track_correlation.py --cache cache/akita_native
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="cache/akita_native")
    ap.add_argument("--split", default="train", help="which windows to pool")
    ap.add_argument("--thresh", type=float, default=0.7,
                    help="|r| above which two tracks are called redundant")
    ap.add_argument("--out-dir",
                    default="/mnt/processed/wolffjoa/hic-multiome/runs/track_correlation")
    a = ap.parse_args()

    cache = Path(a.cache)
    ch = json.load(open(cache / "channels.json"))
    tracks = ch["tracks"]
    nt = len(tracks)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(cache / f"*_{a.split}_chr*.npz")))
    print(f"{len(files)} {a.split} shards, {nt} tracks", flush=True)

    # streaming Pearson via pooled sums (co-moments), so we never hold all bins.
    n = 0
    s = np.zeros(nt, np.float64)          # sum x
    ss = np.zeros((nt, nt), np.float64)   # sum x_i x_j
    for i, f in enumerate(files):
        cond = np.load(f)["cond"][:, :nt, :].astype(np.float64)   # (W, nt, B)
        X = cond.transpose(0, 2, 1).reshape(-1, nt)               # (W*B, nt)
        finite = np.isfinite(X).all(1)
        X = X[finite]
        n += X.shape[0]
        s += X.sum(0)
        ss += X.T @ X
        print(f"  [{i+1}/{len(files)}] {Path(f).name}: +{X.shape[0]} bins "
              f"(total {n})", flush=True)

    mean = s / n
    cov = ss / n - np.outer(mean, mean)
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)

    # write CSV
    with open(out / "track_correlation.csv", "w") as fh:
        fh.write("track," + ",".join(tracks) + "\n")
        for i, t in enumerate(tracks):
            fh.write(t + "," + ",".join(f"{corr[i,j]:.3f}" for j in range(nt)) + "\n")

    # order tracks by hierarchical clustering (1-|r| distance) for a readable heatmap
    order = list(range(nt))
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        d = 1 - np.abs(corr)
        np.fill_diagonal(d, 0)
        Z = linkage(squareform(d, checks=False), method="average")
        order = list(leaves_list(Z))
    except Exception as e:
        print("clustering skipped:", e)

    # redundant clusters at threshold (connected components of |r|>=thresh graph)
    adj = np.abs(corr) >= a.thresh
    np.fill_diagonal(adj, False)
    seen, clusters = set(), []
    for i in range(nt):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            stack += [j for j in range(nt) if adj[k, j] and j not in seen]
        clusters.append(sorted(comp))

    print(f"\n=== redundant clusters (|r| >= {a.thresh}) ===", flush=True)
    for c in clusters:
        if len(c) > 1:
            print("  " + " ~ ".join(tracks[i] for i in c), flush=True)
    singles = [tracks[c[0]] for c in clusters if len(c) == 1]
    print("  singletons:", ", ".join(singles), flush=True)

    # a compact "one-per-cluster" candidate set
    reps = [tracks[c[0]] for c in clusters]
    print("\none-per-cluster candidate set:", "+".join(sorted(reps)), flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        C = corr[np.ix_(order, order)]
        labs = [tracks[i] for i in order]
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(nt)); ax.set_xticklabels(labs, rotation=90, fontsize=8)
        ax.set_yticks(range(nt)); ax.set_yticklabels(labs, fontsize=8)
        for i in range(nt):
            for j in range(nt):
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(C[i,j]) < 0.6 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
        ax.set_title(f"ChIP track correlation at 2 kb ({a.split}, {n:,} bins)")
        fig.tight_layout()
        fig.savefig(out / "track_correlation.png", dpi=150)
        print("wrote", out / "track_correlation.png", flush=True)
    except Exception as e:
        print("plot failed:", e)

    print("wrote", out / "track_correlation.csv", flush=True)


if __name__ == "__main__":
    main()
