"""Akita comparison (413 GM12878 test regions of Akita) for the release checkpoint all12one_fullrd (registry section 98).

Protocol of registry section 73 (scripts/structure_scores_vs_akita.py, stage akita, called unchanged): Akita's 2048 bp
grid, the 448-bin crop, the targets of Akita, one pixel mask per region (upper triangle, >= 2 bins off the diagonal,
finite in the target and in every method), all structure measures, pooled values with 2000 region bootstrap
resamples. The release checkpoint (2 kb view, the one network of the paper) predicts on its own 2000 bp grid and is
resampled onto Akita's bin centres by bilinear interpolation, exactly as score_like_hicgan.py does for the dedicated
models (`crosscell_predict.py` on the 2000 bp grid of each Akita sequence, then the same resampling).

Release rows (GM12878, 413 regions):
  all12one_fullrd_chip          marks only: 13 marks + Meth of GM12878, reference plane at 0 (`--no-reference`)
  all12one_fullrd_chip_nometh   marks only, 13 marks without Meth
  all12one_fullrd_cons          with the leave-one-out consensus of the other 11 training cells as reference
  all12one_fullrd_cons_nometh   the same without Meth
The paper's rows (akita_native_honest = ChIP only, akita_native_refres = consensus of four other cells as base) are
scored in the same pass. The reference of the release rows is the leave-one-out consensus of 11 cells, that of the
paper row the consensus of four cells. GM12878 is a training cell of the release checkpoint (the 413 regions are
excluded from its training windows).

    python scripts/one_model_remaining_akita.py convert
    python scripts/one_model_remaining_akita.py score
    python scripts/one_model_remaining_akita.py summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import structure_scores_vs_akita as sv                                   # noqa: E402

HM = sv.HM
PRED = HM / "runs/akita_one_model"
OUT = REPO / "results/one_model/remaining/akita"
NEW = {"all12one_fullrd_chip": "full_noref", "all12one_fullrd_chip_nometh": "nometh_noref",
       "all12one_fullrd_cons": "full", "all12one_fullrd_cons_nometh": "nometh"}
# control: the dedicated 2 kb models of the paper family (all12refin_drop with the leave-one-out consensus, all12honest_drop without
# reference; 13 marks, trained on `proper` O/E maps like the release checkpoint) through exactly the same pipeline
CTRL = {"dedicated2kb_cons": "all12refin_drop_full", "dedicated2kb_chip": "all12honest_drop_full"}
LABELS = {"dedicated2kb_cons": "control: dedicated 2 kb model (proper targets), with consensus",
          "dedicated2kb_chip": "control: dedicated 2 kb model (proper targets), ChIP only",
          "all12one_fullrd_chip": "release, marks + Meth only", "all12one_fullrd_chip_nometh": "release, 13 marks only",
          "all12one_fullrd_cons": "release, with consensus (+ Meth)",
          "all12one_fullrd_cons_nometh": "release, with consensus, 13 marks"}


class _NP:
    """numpy whose load() finds the release predictions in PRED"""
    def __getattr__(self, k):
        return getattr(np, k)

    def load(self, f, *a, **k):
        p = Path(str(f))
        if p.stem in NEW or p.stem in CTRL:
            return np.load(PRED / f"{p.stem}.npz", *a, **k)
        return np.load(f, *a, **k)


def resample(pred, our_start_bp, our_res, reg_start, n_out, out_res):
    """Our 2000 bp prediction sampled at Akita's 2048 bp bin centres (score_like_hicgan.resample, bilinear)."""
    from scipy.ndimage import map_coordinates
    centres = reg_start + (np.arange(n_out) + 0.5) * out_res
    coord = (centres - our_start_bp) / our_res - 0.5
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    return map_coordinates(pred, [yy.ravel(), xx.ravel()], order=1, mode="nearest").reshape(n_out, n_out)


def stage_convert(a):
    """raw/<set>.npz (crosscell_predict on the 2000 bp grid of every Akita sequence, GPU host) -> Akita space"""
    raw = PRED / "raw"
    for name, st in {**NEW, **CTRL}.items():
        if not (raw / f"{st}.npz").exists():
            continue
        z = np.load(raw / f"{st}.npz")
        out = {}
        for key in z.files:
            chrom, se = key.split(":")
            s0 = int(se.split("-")[0])
            out[key] = resample(np.asarray(z[key], float), s0, 2000, s0, 512, 2048).astype(np.float32)
        np.savez_compressed(PRED / f"{name}.npz", **out)
        print(name, len(out), "regions", flush=True)


def stage_score(a):
    sv.np = _NP()
    sv.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    ctrl = [k for k in CTRL if (PRED / f"{k}.npz").exists()]
    sv.OURS = ["akita_native_honest", "akita_native_refres", *NEW, *ctrl]
    sv.AK_METHODS = ["akita", "hicgan", "akita_native_honest", "akita_native_refres", *NEW, *ctrl, "consensus4", "map_hff"]
    sv.LABEL = {**sv.LABEL, **{k: LABELS[k] for k in [*NEW, *ctrl]}}
    sv.stage_akita(min(a.threads, 6), a.boot, a.seed, 0)


def stage_summary(a):
    S = pd.read_csv(OUT / "akita413_per_region_summary.csv")
    P = pd.read_csv(OUT / "akita413_pooled.csv")
    R = pd.read_csv(OUT / "akita413_paired.csv")
    old = pd.read_csv(REPO / "results/structure_vs_akita/akita413_per_region_summary.csv")
    oldp = pd.read_csv(REPO / "results/structure_vs_akita/akita413_pooled.csv")
    cols = ["pearson", "scc", "ins_r_10k", "ins_r_50k", "ins_r_250k", "bnd_f1",
            "bnd_strength_r", "moc", "stripe_r", "compartment_corr"]
    L = ["# Akita comparison, 413 test regions, GM12878 (median over regions; pooled values with 95% bootstrap interval)\n",
         "| predictor | regions | " + " | ".join(c for c in cols if f"{c}_median" in S) + " | pooled bnd F1 | hFT bnd recall | loop recall | APA ratio |",
         "|---|---|" + "---|" * (len([c for c in cols if f"{c}_median" in S]) + 4)]
    for _, s in S.iterrows():
        p = P[P.method == s.method].iloc[0]
        L.append(f"| {s.label} | {int(s.n_regions)} | " + " | ".join(f"{s[c + '_median']:.3f}" for c in cols if f"{c}_median" in S)
                 + f" | {p.bnd_f1_pooled:.3f} [{p.bnd_f1_pooled_lo:.3f}, {p.bnd_f1_pooled_hi:.3f}]"
                 f" | {p.hft_bnd_recall:.3f} [{p.hft_bnd_recall_lo:.3f}, {p.hft_bnd_recall_hi:.3f}]"
                 f" | {p.loop_recall:.3f} [{p.loop_recall_lo:.3f}, {p.loop_recall_hi:.3f}]"
                 f" | {p.apa_ratio:.3f} [{p.apa_ratio_lo:.3f}, {p.apa_ratio_hi:.3f}] |")
    L.append("\n## Paired against Akita (ours minus Akita, median over regions [95% interval]; regions ahead)\n")
    L.append("| predictor | metric | difference | regions ahead of n |")
    L.append("|---|---|---|---|")
    for o in ["akita_native_honest", "akita_native_refres", *NEW, *CTRL]:
        for _, r in R[(R.ours == o) & (R.other == "akita") & (R.kind == "per-region")].iterrows():
            L.append(f"| {sv.LABEL.get(o, o)} | {r.metric} | {r.median_diff:.3f} [{r.diff_lo:.3f}, {r.diff_hi:.3f}] | "
                     f"{int(r.wins)} of {int(r.n)} |")
        for _, r in R[(R.ours == o) & (R.other == "akita") & (R.kind == "pooled")].iterrows():
            L.append(f"| {sv.LABEL.get(o, o)} | {r.metric} (pooled) | {r.median_diff:.3f} [{r.diff_lo:.3f}, {r.diff_hi:.3f}] | |")
    L.append("\n## Paired against the consensus of four measured cells\n")
    L.append("| predictor | metric | difference | regions ahead of n |")
    L.append("|---|---|---|---|")
    for o in ["akita_native_refres", "all12one_fullrd_cons"]:
        for _, r in R[(R.ours == o) & (R.other == "consensus4") & (R.kind == "per-region") &
                      R.metric.isin(["pearson", "ins_r_50k", "bnd_f1", "compartment_corr"])].iterrows():
            L.append(f"| {sv.LABEL.get(o, o)} | {r.metric} | {r.median_diff:.3f} [{r.diff_lo:.3f}, {r.diff_hi:.3f}] | "
                     f"{int(r.wins)} of {int(r.n)} |")
    (OUT / "akita413_release.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


RASTER = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization/akita_testset_rasters.npz")


def _upper(m, k=2):
    return m[np.triu_indices(m.shape[0], k)]


def _pear(a, b):
    x, y = _upper(a), _upper(b)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 200 else np.nan


def stage_gain(a):
    """Share of regions ahead of both Akita and the consensus of four cells, and the region at the 90th percentile of
    the smaller of both gains with the correlation of (measured - consensus) and (prediction - consensus) after
    averaging to 80 bins of 10 240 bp (the numbers of the Fig. 1 strip of the manuscript, fig_regions.py)."""
    PR = pd.read_csv(OUT / "akita413_per_region.csv.gz")
    W = PR.pivot(index="region", columns="method", values="pearson")
    Z = np.load(RASTER, allow_pickle=True)
    side, off = int(Z["side"]), int(Z["offset"])
    crop = (512 - side) // 2
    regs = [str(r) for r in Z["regions"]]
    blk = np.load(sv.BENCH / "blocks/A_gm12878_2kb.npz")
    NB, P5 = 80, 5
    c0 = (side - NB * P5) // 2
    rows = []
    for m in ("akita_native_refres", "all12one_fullrd_cons", "all12one_fullrd_cons_nometh"):
        w = W.dropna(subset=["akita", m, "consensus4"]).copy()
        w["score"] = np.minimum(w[m] - w["consensus4"], w[m] - w["akita"])
        ahead = float(((w[m] > w["akita"]) & (w[m] > w["consensus4"])).mean())
        d = w.sort_values("score").reset_index()
        i = int(round(0.9 * (len(d) - 1)))
        reg = d.loc[i, "region"]
        k = regs.index(reg)
        import flexcible.bioscore as bs
        T = bs.dense_from_triu(Z["akita_target"][k, :, 1], side, off).astype(float)
        T[T == 0] = np.nan
        if m == "akita_native_refres":
            Pm = np.asarray(np.load(sv.BENCH / "pred/akita_space/akita_native_refres.npz")[reg], float)
        else:
            Pm = np.asarray(np.load(PRED / f"{m}.npz")[reg], float)
        Pm = Pm[crop:crop + side, crop:crop + side]
        C = blk[f"{reg}|consensus4"].astype(float)

        def pooled(M):
            S = M[c0:c0 + NB * P5, c0:c0 + NB * P5].reshape(NB, P5, NB, P5)
            with np.errstate(invalid="ignore"):
                return np.nanmean(S, axis=(1, 3))
        t, c, p = pooled(T), pooled(C), pooled(Pm)
        rows.append({"model": m, "regions": len(w), "share_ahead_of_akita_and_consensus4": ahead,
                     "regions_ahead_of_akita": int((w[m] > w["akita"]).sum()),
                     "median_gain_over_consensus4": float((w[m] - w["consensus4"]).median()),
                     "region_p90": reg, "r_vs_measured_p90": _pear(t, p), "r_consensus_p90": _pear(t, c),
                     "r_difference_maps_p90": _pear(t - c, p - c)})
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "akita413_gain_over_consensus.csv", index=False, float_format="%.4f")
    print(G.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["convert", "score", "summary", "gain"])
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"convert": stage_convert, "score": stage_score, "summary": stage_summary, "gain": stage_gain}[a.stage](a)


if __name__ == "__main__":
    main()
