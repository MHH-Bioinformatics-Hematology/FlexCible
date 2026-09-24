"""Does the fast (few-chromosome) input search agree with the full-data one?

Compares the per-round drop rankings of two backward_tracks runs (a fast proxy
vs the full-data reference). For every round whose candidate pool matches in both
runs, reports Spearman and Kendall rank correlation of the per-track validation
score and whether the top-ranked (chosen) removal agrees. Round 1 (all 13 tracks)
is the clean test since both start from the same set.

    python scripts/compare_search.py \
        --fast /mnt/processed/wolffjoa/hic-multiome/runs/backward_fast/backward_rankings.csv \
        --full /mnt/processed/wolffjoa/hic-multiome/runs/backward_tracks/backward_rankings_FULLDATA_ref.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import kendalltau, spearmanr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", required=True)
    ap.add_argument("--full", required=True)
    a = ap.parse_args()

    F = pd.read_csv(a.fast)
    G = pd.read_csv(a.full)
    print(f"fast: rounds {sorted(F['round'].unique())}   full: rounds {sorted(G['round'].unique())}\n")

    for rnd in sorted(set(F["round"]) & set(G["round"])):
        f = F[F["round"] == rnd].set_index("removed")["val"]
        g = G[G["round"] == rnd].set_index("removed")["val"]
        common = sorted(set(f.index) & set(g.index))
        if set(f.index) != set(g.index):
            print(f"[round {rnd}] candidate pools differ "
                  f"(fast {len(f)}, full {len(g)}, common {len(common)}) "
                  f"-- path diverged earlier; comparing common {len(common)} only")
        if len(common) < 3:
            print(f"[round {rnd}] too few common candidates\n")
            continue
        fv = f.loc[common].values
        gv = g.loc[common].values
        rho = spearmanr(fv, gv).statistic
        tau = kendalltau(fv, gv).statistic
        f_top = f.loc[common].idxmax()
        g_top = g.loc[common].idxmax()
        # rank order (best drop first) in each
        f_order = list(f.loc[common].sort_values(ascending=False).index)
        g_order = list(g.loc[common].sort_values(ascending=False).index)
        print(f"[round {rnd}] n={len(common)}  Spearman={rho:.3f}  Kendall={tau:.3f}  "
              f"chosen-agree={'YES' if f_top == g_top else f'NO (fast -{f_top} / full -{g_top})'}")
        print(f"           fast order: {' > '.join('-'+t for t in f_order[:6])}")
        print(f"           full order: {' > '.join('-'+t for t in g_order[:6])}\n")


if __name__ == "__main__":
    main()
