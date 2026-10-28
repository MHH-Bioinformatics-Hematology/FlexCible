"""Do the ChIP groups matter more than the individual mark?

Takes the trained all-13 model (NO retraining) and, at prediction time, edits the
per-bin track rows before they enter the network, then scores on Akita's test
target with the Hi-cGAN/Akita metric. Because the channel slots are positional
and this model was trained with track_dropout=0, every edit is out-of-distribution
to some degree; the informative signal is the RELATIVE change vs the unedited
baseline across edits, read together:

  swap A<->B   put B's (normalised) signal in A's slot and A's in B's. Within a
               correlation group this should barely move the score if the model
               keyed on the GROUP; across groups it should hurt.
  omit A       replace A's slot with its absent value (what an all-zero assay
               normalises to) -- the codebase's real "not given". A redundant
               track should cost little, a unique one a lot.

    python scripts/track_substitution.py --config configs/akita_native.yaml \
        --checkpoint runs/akita_native/checkpoint-best --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import cooler
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_like_hicgan as sc

from flexcible.config import Config
from flexcible.genome import match_name
from flexcible.matrix import MatrixNormalizer, open_cool
from flexcible.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
from flexcible.tracks import TrackNormalizer

# curated edits, from the 2 kb correlation structure
#   cohesin: RAD21~SMC3 (0.74); active block: H3K27ac/H3K4me1/H3K4me2/H3K4me3/H3K9ac
EXPERIMENTS = [
    ("baseline", None),
    # within-group swaps (expected: small change)
    ("swap_RAD21_SMC3",      ("swap", "RAD21", "SMC3")),
    ("swap_H3K4me1_H3K4me2", ("swap", "H3K4me1", "H3K4me2")),
    ("swap_H3K27ac_H3K9ac",  ("swap", "H3K27ac", "H3K9ac")),
    ("swap_H3K4me3_H3K9ac",  ("swap", "H3K4me3", "H3K9ac")),
    # cross-group swaps (negative control, expected: large drop)
    ("swap_CTCF_H3K27me3",   ("swap", "CTCF", "H3K27me3")),
    ("swap_RAD21_H3K27ac",   ("swap", "RAD21", "H3K27ac")),
    # omissions via absent value (redundant vs unique)
    ("omit_H3K4me2_redund",  ("omit", "H3K4me2")),
    ("omit_H3K9ac_redund",   ("omit", "H3K9ac")),
    ("omit_SMC3_redund",     ("omit", "SMC3")),
    ("omit_CTCF_unique",     ("omit", "CTCF")),
    ("omit_H3K27me3_unique", ("omit", "H3K27me3")),
    ("omit_H3K9me3_unique",  ("omit", "H3K9me3")),
]


def apply_edit(tracks_block, edit, row_of, absent):
    """tracks_block: (13, W) view of the normalised track rows; edited in place."""
    if edit is None:
        return
    kind = edit[0]
    if kind == "swap":
        a, b = row_of[edit[1]], row_of[edit[2]]
        tmp = tracks_block[a].copy()
        tracks_block[a] = tracks_block[b]
        tracks_block[b] = tmp
    elif kind == "omit":
        a = row_of[edit[1]]
        tracks_block[a] = absent[a]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--minCovered", type=float, default=0.5)
    ap.add_argument("--out-dir",
                    default="/mnt/processed/wolffjoa/hic-multiome/runs/track_substitution")
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = a.device
    cfg = Config.load(a.config)
    sample = cfg.get_sample(None)
    ckpt = Path(a.checkpoint)
    assert not cfg.train.select_tracks, "run on the full all-13 model"
    model, channels = sc.load_model(cfg, ckpt, device)
    size, res = cfg.window.size, cfg.data.resolution
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    n_chip = len(channels["tracks"])
    row_of = {t: i for i, t in enumerate(channels["tracks"])}   # rows 0..12
    mat_norm = MatrixNormalizer.load(ckpt / f"norm_matrix_{sample.name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    absent = track_norm.absent_values()                         # per sym row
    use_seq = cfg.sequence.uses_encoder
    fasta = sample.fasta or cfg.sequence.fasta

    regions = pd.read_csv(sc.REGIONS, sep="\t", names=["chrom", "start", "end"]).reset_index(drop=True)
    tgt = cooler.Cooler(str(sc.TARGET))
    if not Path(sample.cool_path).exists():        # training cool moved; chrom sizes only
        sample.cool_path = str(sc.TARGET)
    src = open_cool(sample.cool_path)
    track_files = sample_track_files(cfg, sample)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    # region -> {experiment: r}
    per_region = {}
    for chrom, grp in regions.groupby("chrom"):
        if match_name(chrom, list(tgt.chromnames)) is None:
            continue
        try:
            name = match_name(chrom, list(src.chromnames))
        except Exception:
            continue
        nb = chrom_nbins_of(src, name)
        sym_full = track_norm.apply(assemble_chrom(cfg, sample, track_files, name, nb))
        print(f"{chrom}: {len(grp)} regions", flush=True)

        for _, r in grp.iterrows():
            tname = match_name(chrom, list(tgt.chromnames))
            try:
                T = np.asarray(tgt.matrix(balance=False).fetch(
                    f"{tname}:{int(r.start)}-{int(r.end)}"), dtype=float)
            except Exception:
                continue
            n_out = T.shape[0]
            centre_bin = (int(r.start) + int(r.end)) // 2 // res
            st = max(0, min(nb - size, centre_bin - size // 2))
            base_ct = sym_full[:, st:st + size]
            if base_ct.shape[1] < size:
                base_ct = np.concatenate(
                    [base_ct, np.zeros((base_ct.shape[0], size - base_ct.shape[1]), np.float32)], 1)
            reg = f"{chrom}:{int(r.start)}-{int(r.end)}"
            per_region.setdefault(reg, {})
            seq_oh = (sc.region_onehot(fasta, name, st, size, res, device)
                      if use_seq else None)

            for ename, edit in EXPERIMENTS:
                ct = base_ct.copy()
                apply_edit(ct[:n_chip], edit, row_of, absent)
                cond = torch.from_numpy(ct[None]).to(device)
                pairs = (cond[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, size)
                         if n_motifs else None)
                with torch.no_grad():
                    o = model(cond, pairs, None, seq_oh)
                    if a.tta:
                        from flexcible.dataset import reverse_cond, split_cond
                        cr = reverse_cond(cond, offset, n_motifs)
                        sr, pr = split_cond(cr, offset, n_motifs)
                        seqr = (sc.region_onehot(fasta, name, st, size, res, device, revcomp=True)
                                if use_seq else None)
                        o = 0.5 * (o + torch.flip(model(sr, pr, None, seqr), dims=[-1, -2]))
                P = mat_norm.decode(o[0, 0].float().cpu().numpy())
                P = 0.5 * (P + P.T)
                P = sc.resample(P, st * res, res, int(r.start), n_out, tgt.binsize)
                live = np.abs(P).sum(1) > 0
                if live.mean() < a.minCovered:
                    continue
                M = np.where(live[:, None] & live[None, :], P, np.nan)
                Tm = np.where(T == 0, np.nan, T)
                ok = np.isfinite(M) & np.isfinite(Tm)
                if ok.sum() >= 100 and M[ok].std() > 0 and Tm[ok].std() > 0:
                    from scipy.stats import pearsonr
                    per_region[reg][ename] = float(pearsonr(M[ok], Tm[ok]).statistic)

        # incremental dump so progress is visible
        df = pd.DataFrame(per_region).T
        df.to_csv(out / "per_region.csv")
        _summary(df, out)

    df = pd.DataFrame(per_region).T
    df.to_csv(out / "per_region.csv")
    _summary(df, out, final=True)


def _summary(df, out, final=False):
    names = [n for n, _ in EXPERIMENTS if n in df.columns]
    base = df["baseline"] if "baseline" in df else None
    rows = []
    for n in names:
        col = df[n].dropna()
        rec = {"experiment": n, "n": len(col), "median_r": round(col.median(), 4),
               "mean_r": round(col.mean(), 4)}
        if base is not None and n != "baseline":
            paired = df[["baseline", n]].dropna()
            rec["delta_median"] = round((paired[n] - paired["baseline"]).median(), 4)
        rows.append(rec)
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    if final:
        print("\n=== summary (median per-region Pearson, Akita test) ===", flush=True)
        print(s.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
