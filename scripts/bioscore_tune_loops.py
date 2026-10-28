"""Choose the loop-caller parameters on an EXTERNAL criterion, not on a score.

A loop caller has knobs, and tuning them on how well some model reproduces the
calls would build the answer into the question. So they are tuned here on
something no model can see: CTCF occupancy at the called anchors, measured
from the GM12878 CTCF ChIP track. A caller whose calls are real CTCF/cohesin
loops puts its anchors on CTCF peaks; a caller that is picking up Poisson
noise does not.

For every setting the script reports

  loops/region   call density (Rao et al. find ~3 loops per Mb in GM12878)
  CTCF-AUC       measured anchors vs distance-matched random anchors, on the
                 CTCF track. 0.50 = the calls are noise.
  APA true/akita/hicgan
                 aggregate peak analysis at those anchors. Included only as a
                 sanity check that the setting can register a difference at
                 all; it plays no part in the choice.

Result that fixed the defaults in BioParams: on native 2048 bp bins the calls
are indistinguishable from noise (AUROC 0.50) whatever the threshold, because
a 2 kb pixel of a 4e9-contact map is Poisson-dominated and a single hot pixel
outranks a real, broad peak. Averaging 8x8 blocks first (16.4 kb) lifts the
same caller to AUROC 0.71 at ~3 loops per region.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np                                                  # noqa: E402
import pyBigWig                                                     # noqa: E402
from scipy.stats import mannwhitneyu                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible import bioscore as bs                              # noqa: E402

AKITA_RES, AKITA_CROP = 2048, 32
CTCF = ("/home/mh-hannover.local/wolffjoa/data/hic-multiome/hg38_gm12878/"
        "CTCF.bigwig")

GRID = [(1, 2, 7), (1, 5, 15), (1, 8, 20), (2, 2, 7), (2, 4, 10),
        (4, 1, 4), (4, 2, 5), (4, 2, 7), (8, 1, 3), (8, 1, 4), (16, 1, 3)]
THRESHOLDS = [1.3, 1.5, 1.8, 2.2]


def load(dirpath: Path, stride: int):
    T = np.load(dirpath / "rast_target.npy", mmap_mode="r")
    A = np.load(dirpath / "rast_akita.npy", mmap_mode="r")
    G = np.load(dirpath / "rast_hicgan.npy", mmap_mode="r")
    reg = [str(r) for r in np.load(dirpath / "rast_regions.npy",
                                   allow_pickle=True)]
    side, off = (int(x) for x in np.load(dirpath / "rast_meta.npy"))
    bw = pyBigWig.open(CTCF)
    out = []
    for k in range(0, len(reg), stride):
        chrom, sp = reg[k].split(":")
        st = int(sp.split("-")[0]) + AKITA_CROP * AKITA_RES
        v = np.nan_to_num(np.asarray(
            bw.stats(chrom, st, st + side * AKITA_RES, type="mean",
                     nBins=side), dtype=float))
        out.append((bs.oe_from_log(bs.dense_from_triu(np.asarray(T[k]), side, off)),
                    bs.oe_from_log(bs.dense_from_triu(np.asarray(A[k]), side, off)),
                    bs.oe_from_log(bs.dense_from_triu(np.asarray(G[k]), side, off)),
                    v))
    bw.close()
    return out


def evaluate(data, f: int, inner: int, outer: int, thr: float,
             min_oe: float = 1.15, cap: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    res_bp = AKITA_RES * f
    mind, maxd = max(2, int(60000 / res_bp)), int(600000 / res_bp)
    half = max(4, int(80000 / res_bp))
    corner = max(2, half // 2)
    p = bs.BioParams(loop_coarsen=f, donut_inner=inner, donut_outer=outer,
                     loop_min_enrich=thr, loop_min_oe=min_oe,
                     loop_min_dist=mind, loop_max_dist=maxd,
                     loop_nms=max(2, outer // 2), loop_max_peaks=cap,
                     apa_half=half, apa_corner=corner)
    w = 2 * half + 1
    st = np.zeros((w, w)); spA = np.zeros((w, w)); spG = np.zeros((w, w))
    na = nl = 0
    anc, ctl = [], []
    for T, A, G, v in data:
        Tc, Ac, Gc = (bs.coarsen(M, f) for M in (T, A, G))
        vc = v[:len(v) // f * f].reshape(-1, f).mean(1) if f > 1 else v
        ta, _ = bs.call_loops(Tc, p)
        nl += len(ta)
        n = Tc.shape[0]
        s1, s2, nn = bs.apa_pair(Tc, Ac, ta, half); st += s1; spA += s2; na += nn
        _, s3, _ = bs.apa_pair(Tc, Gc, ta, half); spG += s3
        for i, j in ta:
            anc += [vc[max(0, i - 1):i + 2].max(), vc[max(0, j - 1):j + 2].max()]
            d = int(j - i)
            for _ in range(4):
                a = int(rng.integers(0, max(1, n - d)))
                ctl += [vc[max(0, a - 1):a + 2].max(),
                        vc[max(0, a + d - 1):a + d + 2].max()]
    if nl < 20:
        return nl, *([float("nan")] * 4)
    anc, ctl = np.asarray(anc), np.asarray(ctl)
    u = mannwhitneyu(anc, ctl, alternative="greater")
    return (nl, u.statistic / (anc.size * ctl.size),
            bs.apa_score(st, na, corner), bs.apa_score(spA, na, corner),
            bs.apa_score(spG, na, corner))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/bioscore")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="runs/bioscore/loop_tuning.csv")
    a = ap.parse_args()

    data = load(Path(a.dir), a.stride)
    print(f"{len(data)} regions\n")
    hdr = (f"{'coarsen':>7} {'inner':>5} {'outer':>5} {'enrich':>6} "
           f"{'loops/reg':>9} {'CTCF-AUC':>8} {'APAtrue':>7} {'APAakita':>8} "
           f"{'APAhicgan':>9}")
    print(hdr)
    rows = []
    for f, inner, outer in GRID:
        for thr in THRESHOLDS:
            nl, auc, at, aa, ag = evaluate(data, f, inner, outer, thr)
            print(f"{f:>7} {inner:>5} {outer:>5} {thr:>6} "
                  f"{nl/len(data):>9.1f} {auc:>8.3f} {at:>7.3f} {aa:>8.3f} "
                  f"{ag:>9.3f}", flush=True)
            rows.append(dict(coarsen=f, inner=inner, outer=outer, enrich=thr,
                             loops_per_region=nl / len(data), ctcf_auroc=auc,
                             apa_true=at, apa_akita=aa, apa_hicgan=ag))
    import pandas as pd
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
