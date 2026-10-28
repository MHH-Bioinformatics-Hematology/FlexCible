"""Like-for-like rematch: our network trained on Akita's own data and split.

Everything is hg38 and already in Akita's target space (log observed/expected,
clipped), so regions are used as published with no liftover and the values are
compared directly. Akita's and Hi-cGAN's numbers are the recorded ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.config import Config          # noqa: E402
from flexcible.genome import match_name      # noqa: E402

MAN = Path.home() / "src/Hi-cGAN/manuscript_gigascience/data"


def main():
    import cooler

    cfg = Config.load("configs/akita_full_2kb.yaml")
    truth = cooler.Cooler(cfg.get_sample(None).cool_path)
    pred = cooler.Cooler("runs/akita_full_2kb/pred_holdout.cool")

    ak = pd.read_csv(MAN / "akita_vs_hicgan_heldout.csv")
    ak["chrom"] = ak.region.str.split(":").str[0]
    ak["num"] = ak.chrom.str.replace("chr", "", regex=False)
    ak = ak[ak.num.str.isdigit()]
    ak = ak[ak.num.astype(int) % 2 == 0]           # even chromosomes only

    rows = []
    for _, r in ak.iterrows():
        chrom, span = r.region.split(":")
        s, e = (int(x) for x in span.split("-"))
        reg_t = f"{match_name(chrom, list(truth.chromnames))}:{s}-{e}"
        reg_p = f"{match_name(chrom, list(pred.chromnames))}:{s}-{e}"
        try:
            a = np.nan_to_num(np.asarray(
                truth.matrix(balance=False).fetch(reg_t), dtype=np.float64))
            b = np.nan_to_num(np.asarray(
                pred.matrix(balance=False).fetch(reg_p), dtype=np.float64))
        except Exception:
            continue
        n = min(a.shape[0], b.shape[0])
        if n < 20:
            continue
        iu = np.triu_indices(n, 1)
        x, y = a[:n, :n][iu], b[:n, :n][iu]
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        rows.append({"region": r.region, "ours": float(np.corrcoef(x, y)[0, 1]),
                     "akita": r["akita_gm12878"], "hicgan": r["hicgan_fixed"]})
    d = pd.DataFrame(rows).dropna(subset=["ours", "akita"])
    d.to_csv("runs/akita_full_2kb/rematch_regions.csv", index=False)
    print(f"scored {len(d)} of {len(ak)} held-out regions (even chromosomes), "
          "all hg38, all in Akita's target space")
    for k, lab in (("akita", "Akita (recorded)"), ("hicgan", "Hi-cGAN (recorded)"),
                   ("ours", "ours, trained on Akita's data+split")):
        v = d[k].dropna()
        print(f"  {lab:38s} median {v.median():.3f}  mean {v.mean():.3f}")
    print(f"  ours better than Akita in {100*(d.ours > d.akita).mean():.0f}% of regions")
    print(f"  ours better than Hi-cGAN in "
          f"{100*(d.ours > d.hicgan).mean():.0f}% of regions")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    data = [d.akita.values, d.hicgan.dropna().values, d.ours.values]
    labels = ["Akita\n(recorded)", "Hi-cGAN\n(recorded)", "ours\n(same data+split)"]
    cols = ["#5aa469", "#b05aa4", "#cf5b56"]
    parts = ax1.violinplot(data, showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], cols):
        pc.set_facecolor(c); pc.set_alpha(0.65)
    ax1.set_xticks([1, 2, 3], labels)
    ax1.set_ylabel("per-region Pearson (Akita space)")
    ax1.set_title(f"Akita's held-out regions, even chromosomes (n={len(d)})")
    ax1.grid(axis="y", alpha=0.3); ax1.axhline(0, color="k", lw=.5, ls=":")
    ax2.scatter(d.akita, d.ours, s=16, alpha=.6, color="#2c7fb8")
    lim = [min(d.akita.min(), d.ours.min()) - .05, max(d.akita.max(), d.ours.max()) + .05]
    ax2.plot(lim, lim, "k--", lw=1); ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("Akita (recorded)"); ax2.set_ylabel("ours, same region")
    ax2.set_title(f"paired; ours better in {100*(d.ours>d.akita).mean():.0f}%")
    ax2.grid(alpha=.3)
    fig.suptitle("Like-for-like: our U-Net trained on Akita's 2 kb matrix, "
                 "target space and holdout, with the same single CTCF track "
                 "the recorded Hi-cGAN run used", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("runs/akita_rematch.png", dpi=150)
    print("wrote runs/akita_rematch.png")


if __name__ == "__main__":
    main()
