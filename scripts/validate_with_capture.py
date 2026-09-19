"""Do our predicted loops exist in an ORTHOGONAL assay?

The down-sampling experiment argued that prediction can beat a shallow
measurement, but it was a simulation: the "shallow experiment" was made by
thinning the same deep map. This asks the same question with an independent
assay. Promoter capture Hi-C (Mifsud 2015, GM12878) sequences far deeper at
baited promoters than genome-wide Hi-C can, so it sees contacts Hi-C at this
depth misses.

The decisive comparison is not "are our loops in capture data" -- popular,
open regions are enriched for everything. It is whether loops we predict that
the Hi-C map does NOT contain are supported more often than distance-matched
control pairs drawn from the same regions. If they are, the model is finding
real contacts the measurement missed. If they match the controls, our extra
loops are noise and the claim fails.

Coordinates: our work is hg38, Mifsud is hg19, so anchors are lifted with the
hg38ToHg19 chain. Anything that fails to lift is dropped from both the test
and the control set, never silently counted as unsupported.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome import bioscore as bs                    # noqa: E402
from hicmultiome.liftover import LiftOver                 # noqa: E402
from scripts.plot_regions_three_models import RAST, GM, dense   # noqa: E402

PCHIC = (Path.home() / "data/hic-multiome/pchic"
         / "TS5_GM12878_promoter-other_significant_interactions.txt")
CHAIN = Path.home() / "data/hic-multiome/chain/hg38ToHg19.over.chain.gz"


def load_pchic(min_score: float, binsize: int = 20000):
    """Significant promoter-other interactions, binned for fast lookup.

    Both orientations are stored: a Hi-C loop does not know which of its
    anchors is the promoter.
    """
    df = pd.read_csv(PCHIC, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    sc = df["log(observed/expected)"].astype(float)
    df = df[sc >= min_score]
    idx = defaultdict(set)
    for ch1, s1, e1, ch2, s2, e2 in zip(
            df["chr bait"], df["start bait"], df["end bait"],
            df["chr"], df["start"], df["end"]):
        if ch1 != ch2:
            continue
        a = ((int(s1) + int(e1)) // 2) // binsize
        b = ((int(s2) + int(e2)) // 2) // binsize
        idx[ch1].add((min(a, b), max(a, b)))
    n = sum(len(v) for v in idx.values())
    print(f"  capture: {len(df):,} significant interactions -> {n:,} binned pairs")
    return idx


def supported(idx, lo, chrom, a_bp, b_bp, binsize, tol_bins):
    """Is this hg38 anchor pair present in the capture set, after liftover?"""
    ma = lo.convert_interval(chrom, a_bp, a_bp + binsize)
    mb = lo.convert_interval(chrom, b_bp, b_bp + binsize)
    if ma is None or mb is None:
        return None                      # unliftable: excluded, not "absent"
    ca, pa = ma[0], (ma[1] + ma[2]) // 2
    cb, pb = mb[0], (mb[1] + mb[2]) // 2
    if ca != cb or ca not in idx:
        return None
    ia, ib = pa // binsize, pb // binsize
    lo_i, hi_i = min(ia, ib), max(ia, ib)
    pairs = idx[ca]
    for da in range(-tol_bins, tol_bins + 1):
        for db in range(-tol_bins, tol_bins + 1):
            if (lo_i + da, hi_i + db) in pairs:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="runs/bioscore/preds_akita_native.npz")
    ap.add_argument("--min-score", type=float, default=5.0)
    ap.add_argument("--bin", type=int, default=20000)
    ap.add_argument("--tol-bins", type=int, default=1)
    ap.add_argument("--n-controls", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/capture_validation.csv")
    a = ap.parse_args()

    idx = load_pchic(a.min_score, a.bin)
    lo = LiftOver(str(CHAIN))
    p = bs.BioParams()
    z = np.load(RAST, allow_pickle=True)
    side, off = int(z["side"]), int(z["offset"])
    regions = list(z["regions"])
    pz = np.load(a.preds, allow_pickle=True)
    pred_regions = list(pz["regions"])
    ours = pz["ours_native"]
    res = int(pz["resolution"])
    coarse = res * p.loop_coarsen

    rng = np.random.default_rng(0)
    rows = []
    todo = pred_regions[:a.limit] if a.limit else pred_regions
    for n_done, reg in enumerate(todo):
        k = regions.index(reg)
        chrom, span = reg.split(":")
        start = int(span.split("-")[0])
        T = bs.oe_from_log(dense(z["akita_target"][k, :, GM], side, off))
        Praw = ours[pred_regions.index(reg)]
        # predictions are stored as the full 512-bin window; the target is
        # Akita's central 448-bin crop. Without this every predicted loop sits
        # 32 bins (4 coarse bins) off and NOTHING can ever match, which
        # silently reclassifies every true positive as "invented".
        if Praw.shape[0] != side:
            c = (Praw.shape[0] - side) // 2
            Praw = Praw[c:c + side, c:c + side]
        P = bs.oe_from_log(Praw)
        Tc = bs.coarsen(T, p.loop_coarsen)
        Pc = bs.coarsen(P, p.loop_coarsen)
        t_anchor, _ = bs.call_loops(Tc, p)
        p_anchor, _ = bs.call_loops(Pc, p, top_k=max(len(t_anchor), 1))
        tset = {(int(i), int(j)) for i, j in t_anchor}

        def classify(i, j):
            for di in (-p.loop_match_tol, 0, p.loop_match_tol):
                for dj in (-p.loop_match_tol, 0, p.loop_match_tol):
                    if (i + di, j + dj) in tset:
                        return "matched"
            return "invented"

        for i, j in p_anchor:
            i, j = int(i), int(j)
            a_bp = start + i * coarse
            b_bp = start + j * coarse
            s = supported(idx, lo, chrom, a_bp, b_bp, a.bin, a.tol_bins)
            if s is None:
                continue
            rows.append({"region": reg, "kind": classify(i, j),
                         "sep_bins": j - i, "supported": bool(s)})
            # distance-matched controls: same separation, shifted position
            for _ in range(a.n_controls):
                shift = int(rng.integers(-Tc.shape[0] // 3, Tc.shape[0] // 3))
                ci, cj = i + shift, j + shift
                if ci < 0 or cj >= Tc.shape[0] or ci >= cj:
                    continue
                cs = supported(idx, lo, chrom, start + ci * coarse,
                               start + cj * coarse, a.bin, a.tol_bins)
                if cs is None:
                    continue
                rows.append({"region": reg, "kind": "control",
                             "sep_bins": cj - ci, "supported": bool(cs)})
        for i, j in t_anchor:
            i, j = int(i), int(j)
            s = supported(idx, lo, chrom, start + i * coarse,
                          start + j * coarse, a.bin, a.tol_bins)
            if s is not None:
                rows.append({"region": reg, "kind": "measured",
                             "sep_bins": j - i, "supported": bool(s)})
        if (n_done + 1) % 50 == 0:
            print(f"  {n_done+1}/{len(todo)} regions", flush=True)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\ncapture support (Mifsud 2015 GM12878 promoter capture Hi-C, "
          f"score >= {a.min_score}):\n")
    print(f"{'set':10s} {'n':>6s} {'supported':>10s} {'rate':>7s}")
    for kind in ("measured", "matched", "invented", "control"):
        g = df[df.kind == kind]
        if not len(g):
            continue
        print(f"{kind:10s} {len(g):6d} {int(g.supported.sum()):10d} "
              f"{100*g.supported.mean():6.1f}%")
    inv = df[df.kind == "invented"]
    ctl = df[df.kind == "control"]
    if len(inv) and len(ctl):
        from scipy.stats import fisher_exact
        tab = [[int(inv.supported.sum()), int((~inv.supported).sum())],
               [int(ctl.supported.sum()), int((~ctl.supported).sum())]]
        odds, pv = fisher_exact(tab, alternative="greater")
        print(f"\ninvented loops vs distance-matched controls: "
              f"odds ratio {odds:.2f}, p = {pv:.3g}")
        print("  (this is the test: >1 and significant means the model finds "
              "contacts the Hi-C map missed)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
