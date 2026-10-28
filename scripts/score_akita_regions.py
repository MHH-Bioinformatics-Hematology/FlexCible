"""Score ONLY Akita's held-out regions, with no genome-wide prediction.

The full predict path spends ~7 of its 10 minutes on work this comparison does
not need: it predicts every window of eleven chromosomes and writes a
genome-wide cool (894 MB) when the question concerns 222 one-megabase regions.
Here each region is predicted as a single window and scored in memory.

Both sides are already in Akita's space (log obs/exp, clipped), so no transform
is applied to either -- re-applying it to already-transformed data was one of
the bugs that made the previous number meaningless.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.config import Config                               # noqa: E402
from flexcible.genome import match_name                           # noqa: E402
from flexcible.matrix import (MatrixNormalizer, fetch_diagonal_window,
                                open_cool)                          # noqa: E402
from flexcible.model import build_backbone, build_spatial_cond, n_cond_channels  # noqa: E402
from flexcible.precompute import (assemble_chrom, chrom_nbins_of,
                                    sample_track_files)             # noqa: E402
from flexcible.tracks import TrackNormalizer                      # noqa: E402
from flexcible.utils import get_logger                            # noqa: E402

MAN = Path.home() / "src/Hi-cGAN/manuscript_gigascience/data"
log = get_logger("score-akita")


def load_model(cfg: Config, ckpt: Path, device):
    import json
    channels = json.loads((ckpt / "channels.json").read_text())
    n_sym = len(channels["sym_rows"])
    n_motifs = len(channels["motif_ids"])
    n_planes = 2 if channels.get("capture_input") else 0
    n_out = 2 if channels.get("has_capture") else 1
    in_ch = n_cond_channels(n_sym, n_motifs, n_planes, cfg.model.cond_mode)
    model = build_backbone(cfg.model.kind, in_ch, out_ch=n_out,
                           base=cfg.model.base_channels,
                           depth=cfg.model.depth).to(device)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
    model.eval()
    return model, channels, n_motifs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    sample = cfg.get_sample(None)
    ckpt = Path(a.checkpoint or (Path(cfg.train.output_dir) / "checkpoint-best"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, channels, n_motifs = load_model(cfg, ckpt, device)
    size = cfg.window.size
    res = cfg.data.resolution
    offset = len(channels["tracks"]) + len(channels["seq_features"])

    mat_norm = MatrixNormalizer.load(
        ckpt / f"norm_matrix_{sample.name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")

    ak = pd.read_csv(MAN / "akita_vs_hicgan_heldout.csv")
    ak["chrom"] = ak.region.str.split(":").str[0]
    ak["num"] = ak.chrom.str.replace("chr", "", regex=False)
    ak = ak[ak.num.str.isdigit()]
    ak = ak[ak.num.astype(int) % 2 == 0].reset_index(drop=True)
    log.info("scoring %d Akita held-out regions on even chromosomes", len(ak))

    truth = open_cool(sample.cool_path)
    strata: dict = {}
    track_files = sample_track_files(cfg, sample)
    rows = []

    for chrom, grp in ak.groupby("chrom"):
        try:
            name = match_name(chrom, list(truth.chromnames))
        except Exception:
            log.warning("%s absent from the matrix, skipped", chrom)
            continue
        nb = chrom_nbins_of(truth, name)
        # one track load per chromosome, not per region
        sym_full = track_norm.apply(
            assemble_chrom(cfg, sample, track_files, name, nb))

        starts, keep = [], []
        for idx, r in grp.iterrows():
            s, e = (int(x) for x in r.region.split(":")[1].split("-"))
            centre = (s + e) // 2 // res
            st = max(0, min(nb - size, centre - size // 2))
            starts.append(st)
            keep.append(idx)

        for st, idx in zip(starts, keep):
            ct = sym_full[:, st:st + size]
            if ct.shape[1] < size:
                ct = np.concatenate(
                    [ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
            cond = torch.from_numpy(ct[None]).to(device)
            pairs = (cond[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, size)
                     if n_motifs else None)
            with torch.no_grad():
                out = model(build_spatial_cond(cond, pairs, None, cfg.model.cond_mode))
                if a.tta:
                    from flexcible.dataset import reverse_cond, split_cond
                    cr = reverse_cond(cond, offset, n_motifs)
                    sr, pr = split_cond(cr, offset, n_motifs)
                    out_r = model(build_spatial_cond(sr, pr, None, cfg.model.cond_mode))
                    out = 0.5 * (out + torch.flip(out_r, dims=[-1, -2]))
            # sign preserved: these targets are log(obs/exp), mostly negative
            pred = mat_norm.decode(out[0, 0].float().cpu().numpy())
            pred = 0.5 * (pred + pred.T)
            tru = fetch_diagonal_window(truth, name, st, size)

            # The near-diagonal has huge dynamic range and is the easiest part
            # of the map to get right, so a full-triangle correlation flatters
            # any model. Report several exclusion radii; which one is
            # comparable to Akita's recorded number depends on their masking.
            rec = {"region": ak.loc[idx, "region"], "chrom": chrom,
                   "start_bin": st,
                   "akita": ak.loc[idx, "akita_gm12878"],
                   "hicgan": ak.loc[idx, "hicgan_fixed"]}
            for kdi in (1, 2, 5, 10, 25):
                iu = np.triu_indices(size, kdi)
                x, y = tru[iu], pred[iu]
                ok = np.isfinite(x) & np.isfinite(y)
                rec[f"ours_d{kdi}"] = (
                    float(np.corrcoef(x[ok], y[ok])[0, 1])
                    if ok.sum() > 100 and x[ok].std() > 1e-9 and y[ok].std() > 1e-9
                    else np.nan)
            rec["ours"] = rec["ours_d1"]
            # Full-triangle correlation is dominated by distance: on these
            # targets a distance-only prediction already scores ~0.74 and even
            # a DIFFERENT region's truth scores ~0.53, because empty pixels
            # clip to -2 and far diagonals are mostly empty. Accumulate the
            # per-stratum sums so a distance-stratified score -- the one that
            # actually measures folding -- can be pooled across regions.
            for dd in range(1, size):
                xs = np.diagonal(tru, offset=dd)
                ys = np.diagonal(pred, offset=dd)
                ok = np.isfinite(xs) & np.isfinite(ys)
                if ok.sum() < 8:
                    continue
                xs, ys = xs[ok], ys[ok]
                acc = strata.setdefault(dd, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
                acc[0] += xs.size
                acc[1] += xs.sum(); acc[2] += ys.sum()
                acc[3] += (xs * xs).sum(); acc[4] += (ys * ys).sum()
                acc[5] += (xs * ys).sum()
            rows.append(rec)
        log.info("%s: %d regions", chrom, len(grp))

    df = pd.DataFrame(rows).dropna(subset=["ours", "akita"])
    out = a.out or str(Path(cfg.train.output_dir) / "akita_region_scores.csv")
    df.to_csv(out, index=False)

    print(f"\nscored {len(df)} regions (same loci, Akita space, no re-transform)")
    print(f"  Akita   (recorded): median {df.akita.median():.3f}  "
          f"mean {df.akita.mean():.3f}")
    hg = df.hicgan.dropna()
    print(f"  Hi-cGAN (recorded): median {hg.median():.3f}  mean {hg.mean():.3f}")
    for kdi in (1, 2, 5, 10, 25):
        c = f"ours_d{kdi}"
        print(f"  ours (excl <{kdi} diag): median {df[c].median():.3f}  "
              f"mean {df[c].mean():.3f}  beats Akita "
              f"{100*(df[c] > df.akita).mean():.0f}%")
    rs = []
    for dd, (n_, sx, sy, sxx, syy, sxy) in sorted(strata.items()):
        if n_ < 50:
            continue
        cov = sxy / n_ - (sx / n_) * (sy / n_)
        vx = max(sxx / n_ - (sx / n_) ** 2, 0.0)
        vy = max(syy / n_ - (sy / n_) ** 2, 0.0)
        if vx > 1e-12 and vy > 1e-12:
            rs.append(cov / (vx * vy) ** 0.5)
    if rs:
        print(f"\n  DISTANCE-STRATIFIED (decay removed; {len(rs)} strata):")
        print(f"    mean per-stratum r = {np.mean(rs):.3f}   "
              f"median = {np.median(rs):.3f}")
        print("    this is the number that measures folding rather than decay")
    print(f"  ours beats Akita in {100*(df.ours > df.akita).mean():.0f}% of regions")
    print(f"  ours beats Hi-cGAN in {100*(df.ours > df.hicgan).mean():.0f}% of regions")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
