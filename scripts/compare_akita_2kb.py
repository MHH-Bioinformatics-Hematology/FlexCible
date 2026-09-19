"""Compare our 2 kb Hi-C predictions with the RECORDED Akita scores.

Akita's numbers are reused, never recomputed. Its per-region Pearson is
measured in its own target space (observed/expected, natural log, clipped to
[-2, 2]) at 2 kb on 1 Mb regions, so our windows are scored exactly the same
way. Two caveats stated on the figure: Akita's regions are hg38 and ours are
hg19, so the comparison is between DISTRIBUTIONS over even chromosomes rather
than between matched loci; and Akita is sequence-only while ours reads
epigenomic tracks, so this compares end results, not like-for-like inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicmultiome.config import Config          # noqa: E402
from hicmultiome.matrix import open_cool        # noqa: E402

MAN = Path.home() / "src/Hi-cGAN/manuscript_gigascience/data"
WIN = 1_048_576                                  # Akita's region size
CLIP = 2.0


def akita_transform(m: np.ndarray) -> np.ndarray:
    """observed/expected -> natural log -> clip, Akita's target space."""
    n = m.shape[0]
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    exp = np.array([m[d == k].mean() if (d == k).any() else 0.0 for k in range(n)])
    e = np.maximum(exp[d], 1e-9)
    oe = np.log(np.maximum(m, 1e-9) / e)
    return np.clip(oe, -CLIP, CLIP)


def lifted_akita_regions(ak: pd.DataFrame, chain: str):
    """Akita's hg38 regions lifted to hg19 so the SAME loci can be scored."""
    from hicmultiome.liftover import LiftOver

    lo = LiftOver(chain)
    rows = []
    for _, r in ak.iterrows():
        chrom, span = r.region.split(":")
        s, e = (int(x) for x in span.split("-"))
        m = lo.convert_interval(chrom, s, e)
        if m is None:
            continue
        rows.append({"region": r.region, "chrom": m[0], "start": m[1],
                     "end": m[2], "akita": r["akita_gm12878"],
                     "hicgan": r["hicgan_fixed"]})
    out = pd.DataFrame(rows)
    print(f"lifted {len(out)}/{len(ak)} Akita regions hg38 -> hg19")
    return out


def _already_akita_space(clr) -> bool:
    """True if the matrix already holds log(obs/exp), so must NOT be
    transformed again. Raw-count matrices are non-negative; Akita-space ones
    are mostly negative and clipped at -2."""
    v = clr.pixels()[:200000]["count"].values.astype(float)
    v = v[np.isfinite(v)]
    return bool(v.size and (v < 0).mean() > 0.2)


def score_regions(cfg: Config, pred_cool: str, regions: pd.DataFrame):
    """Our per-region Pearson in Akita's space, on the given loci."""
    truth = open_cool(cfg.get_sample(None).cool_path)
    pred = open_cool(pred_cool)
    t_done, p_done = _already_akita_space(truth), _already_akita_space(pred)
    print(f"  already in Akita space: truth={t_done} pred={p_done}"
          " (transform skipped where True)")
    vals = []
    from hicmultiome.genome import match_name
    for _, r in regions.iterrows():
        # cools differ on 'chr1' vs '1'; resolve against the file, don't guess
        try:
            nm_t = match_name(r.chrom, list(truth.chromnames))
            nm_p = match_name(r.chrom, list(pred.chromnames))
        except Exception:
            vals.append(np.nan)
            continue
        if nm_t != nm_p:
            vals.append(np.nan)
            continue
        reg = f"{nm_t}:{r.start}-{r.end}"
        try:
            a = np.nan_to_num(np.asarray(
                truth.matrix(balance=False).fetch(reg), dtype=np.float64))
            b = np.nan_to_num(np.asarray(
                pred.matrix(balance=False).fetch(reg), dtype=np.float64))
        except Exception:
            vals.append(np.nan)
            continue
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n, :n], b[:n, :n]
        # NOT a.sum() > 0: in Akita space values are mostly negative, so a sum
        # test discards nearly every region. Variance is what scoring needs.
        if n < 20 or a.std() < 1e-9 or b.std() < 1e-9:
            vals.append(np.nan)
            continue
        ta = a if t_done else akita_transform(a)
        tb = b if p_done else akita_transform(b)
        iu = np.triu_indices(n, 1)
        x, y = ta[iu], tb[iu]
        vals.append(float(np.corrcoef(x, y)[0, 1])
                    if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
    regions = regions.copy()
    regions["ours"] = vals
    return regions


def our_window_scores(cfg: Config, pred_cool: str, per_chrom: int = 40):
    truth = open_cool(cfg.get_sample(None).cool_path)
    pred = open_cool(pred_cool)
    res = cfg.data.resolution
    rng = np.random.default_rng(0)
    rows = []
    even = [c for c in truth.chromnames
            if c.strip("chr").isdigit() and int(c.strip("chr")) % 2 == 0]
    for chrom in even:
        L = int(truth.chromsizes[chrom])
        starts = rng.choice(np.arange(0, max(L - WIN, WIN), WIN), 
                            size=min(per_chrom, max(1, (L - WIN) // WIN)),
                            replace=False)
        for s in sorted(starts):
            reg = f"{chrom}:{s}-{s + WIN}"
            try:
                a = np.nan_to_num(np.asarray(
                    truth.matrix(balance=False).fetch(reg), dtype=np.float64))
                b = np.nan_to_num(np.asarray(
                    pred.matrix(balance=False).fetch(reg), dtype=np.float64))
            except Exception:
                continue
            n = min(a.shape[0], b.shape[0])
            a, b = a[:n, :n], b[:n, :n]
            if a.sum() <= 0 or b.sum() <= 0:
                continue
            ta, tb = akita_transform(a), akita_transform(b)
            iu = np.triu_indices(n, 1)
            x, y = ta[iu], tb[iu]
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            rows.append({"chrom": chrom, "start": int(s),
                         "pearson": float(np.corrcoef(x, y)[0, 1])})
    return pd.DataFrame(rows)


def akita_regions_direct(ak):
    """Akita's regions used as-is, for a run already in hg38.

    The liftover path below exists only because an earlier run lived in hg19.
    Lifting when both sides are already hg38 would move every region for no
    reason, so the assembly decides which path is honest.
    """
    rows = []
    for _, r in ak.iterrows():
        chrom, span = r.region.split(":")
        s, e = (int(x) for x in span.split("-"))
        rows.append({"region": r.region, "chrom": chrom, "start": s, "end": e,
                     "akita": r["akita_gm12878"], "hicgan": r["hicgan_fixed"]})
    out = pd.DataFrame(rows)
    print(f"using {len(out)} Akita regions directly (both sides hg38)")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/akita_space_2kb.yaml")
    ap.add_argument("--pred", default="runs/akita_space_2kb/pred_holdout.cool")
    ap.add_argument("--assembly", choices=["hg19", "hg38"], default="hg19",
                    help="hg19 lifts Akita's regions; hg38 scores them directly")
    ap.add_argument("--out-prefix", default="runs/akita_space_2kb")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    ak = pd.read_csv(MAN / "akita_vs_hicgan_heldout.csv")
    ak["chrom"] = ak.region.str.split(":").str[0]
    ak["num"] = ak.chrom.str.replace("chr", "", regex=False)
    ak = ak[ak.num.str.isdigit()]
    ak_even = ak[ak.num.astype(int) % 2 == 0]
    # PAIRED comparison on identical loci: lift Akita's hg38 regions to hg19
    if a.assembly == "hg19":
        chain = str(Path.home() / "data/hic-multiome/chain/hg38ToHg19.over.chain.gz")
        regions = lifted_akita_regions(ak_even, chain)
    else:
        regions = akita_regions_direct(ak_even)
    paired = score_regions(cfg, a.pred, regions).dropna(subset=["ours", "akita"])
    Path(a.out_prefix).mkdir(parents=True, exist_ok=True)
    paired.to_csv(f"{a.out_prefix}/akita_paired_regions{a.tag}.csv", index=False)
    print(f"PAIRED on {len(paired)} identical regions: "
          f"Akita median {paired.akita.median():.3f} | "
          f"ours median {paired.ours.median():.3f} | "
          f"ours wins {100 * (paired.ours > paired.akita).mean():.0f}% of regions")
    a_scores = paired["akita"].values
    g_scores = paired["hicgan"].dropna().values
    o_scores = paired["ours"].values
    print(f"Akita (recorded, GM12878, {len(a_scores)} regions): "
          f"median {np.median(a_scores):.3f} mean {a_scores.mean():.3f}")
    print(f"Hi-cGAN (recorded, {len(g_scores)} regions): "
          f"median {np.median(g_scores):.3f} mean {g_scores.mean():.3f}")
    print(f"ours 2 kb ({len(o_scores)} windows): "
          f"median {np.median(o_scores):.3f} mean {o_scores.mean():.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    data = [a_scores, g_scores, o_scores]
    labels = [f"Akita\n(recorded, n={len(a_scores)})",
              f"Hi-cGAN\n(recorded, n={len(g_scores)})",
              f"ours 2 kb\n(same loci, n={len(o_scores)})"]
    colors = ["#5aa469", "#b05aa4", "#cf5b56"]
    parts = ax1.violinplot(data, showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.65)
    ax1.set_xticks([1, 2, 3], labels)
    ax1.set_ylabel("per-region Pearson (Akita space: log obs/exp, clipped)")
    ax1.set_title("Distribution over held-out regions, even chromosomes")
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0, color="k", lw=0.5, ls=":")
    for i, (d, c) in enumerate(zip(data, colors)):
        ax2.hist(d, bins=30, alpha=0.55, color=c, label=labels[i].split("\n")[0],
                 density=True)
        ax2.axvline(np.median(d), color=c, lw=2, ls="--")
    ax2.set_xlabel("per-region Pearson")
    ax2.set_ylabel("density")
    ax2.set_title("Same data as histograms (dashed = median)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    # paired scatter: same region, both models
    ax2.clear()
    ax2.scatter(paired.akita, paired.ours, s=14, alpha=0.55, color="#2c7fb8")
    lim = [min(paired.akita.min(), paired.ours.min()) - 0.05,
           max(paired.akita.max(), paired.ours.max()) + 0.05]
    ax2.plot(lim, lim, "k--", lw=1)
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("Akita per-region Pearson (recorded)")
    ax2.set_ylabel("ours 2 kb, same region")
    ax2.set_title(f"Paired on identical loci (n={len(paired)}); "
                  f"ours better in {100*(paired.ours>paired.akita).mean():.0f}%")
    ax2.grid(alpha=0.3)
    fig.suptitle("2 kb Hi-C on Akita's held-out regions, even chromosomes  |  "
                 "Akita scores reused as recorded; its hg38 regions lifted to "
                 "hg19 so both models are scored on the SAME loci in Akita's "
                 "space (log obs/exp, clipped)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"runs/akita_2kb_comparison{a.tag}.png"
    fig.savefig(out, dpi=150)
    paired.to_csv(f"{a.out_prefix}/akita_space_window_scores{a.tag}.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
