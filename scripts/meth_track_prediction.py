"""How well does DNA methylation predict each ChIP track itself (registry
section 35, part B). CPU.

Bins of 10 kb on chr1 to chr22 of the nine cells with ENCODE WGBS (seven
training cells GM12878, IMR-90, H1, HepG2, A549, GM23248, CD14+ monocyte and
the held-out K562 and HMEC). Target: a ChIP track as the model sees it
(log1p, z-scored with the track statistics of the training checkpoint).
Features, from the cell's Meth.bigWig (1 kb, value - 1 = methylated fraction)
and the genome:

  local       mean, minimum, maximum and standard deviation of the ten 1 kb
              bins of the bin (the minimum is the methylation dip of a
              CpG island, promoter or enhancer)
  context     mean of +-50 kb and of +-500 kb, standard deviation of the 1 kb
              values in +-50 kb (local methylation variance), fraction of
              low-methylated (< 0.3) 1 kb bins in +-50 kb and in the bin,
              fraction of highly methylated (> 0.8) 1 kb bins in +-50 kb
              (partially methylated domain proxies)
  sequence    GC, CpG density, CTCF motif strength (forward + reverse)

Model: histogram gradient boosting. Schemes (test on even chromosomes, training
on odd chromosomes only, so no test chromosome and no test bin is seen):
  seen_cell    trained on the odd chromosomes of the seven training cells,
               tested on the even chromosomes of each of the nine cells (the
               seven training cells share their odd chromosomes with the fit)
  unseen_cell  leave one cell out: trained on the odd chromosomes of the other
               training cells, tested on the even chromosomes of the left-out
               cell (K562 and HMEC are never in a training set)
and two feature sets: sequence only, and sequence + methylation. R^2 and
Pearson per mark and cell. Residual: the out-of-fold residual of the unseen
scheme (observed - predicted from methylation + sequence) is regressed on the
other 12 tracks of the cell (ridge, 5-fold cross-validation over the even
chromosomes); permutation importance of the three feature groups is measured
on the unseen-cell fits.

    python scripts/meth_track_prediction.py --stage build fit summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import methylation as me                                            # noqa: E402
import multicell_expanded as mx                                     # noqa: E402

RES = 10000
OUT = mx.HM / "runs/meth_tracks"
RES_OUT = REPO / "results/methylation"
TRAIN_CELLS = ["gm12878", "imr90", "h1", "hepg2", "a549", "gm23248", "cd14mono"]
HELD_CELLS = ["k562", "hmec"]
CELLS = TRAIN_CELLS + HELD_CELLS
CHROMS = [f"chr{i}" for i in range(1, 23)]
ODD = [c for i, c in enumerate(CHROMS, 1) if i % 2 == 1]
EVEN = [c for i, c in enumerate(CHROMS, 1) if i % 2 == 0]
MARKS = mx.ALL13
F_LOCAL = ["m_mean", "m_min", "m_max", "m_sd"]
F_CONTEXT = ["m_ctx100k", "m_ctx1M", "m_sd100k", "low_frac100k", "low_frac_bin", "high_frac100k"]
F_SEQ = ["gc", "cpg", "ctcf_motif"]
FEATURES = F_LOCAL + F_CONTEXT + F_SEQ
NORM = mx.REF_OUT / "all12refres_drop_meth_10kb"
N_FIT = 500_000          # training rows per fit (random subsample)


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "meth_tracks.log", "a") as fh:
        fh.write(line + "\n")


def cell_dir(cell):
    return mx.CELLS[cell][1]


def norm_stats():
    from flexcible.tracks import TrackNormalizer
    tn = TrackNormalizer.load(mx.best_ckpt(NORM) / "norm_tracks.json")
    return tn


def build_cell(cell: str):
    """Feature and target table of one cell, saved as parquet-free npz."""
    import pyBigWig
    from scipy.ndimage import uniform_filter1d
    from flexcible.config import Config
    from flexcible.motifs import cached_motif_tracks
    from flexcible.sequence import cached_sequence_features
    from flexcible.tracks import read_track_binned
    f = OUT / "tables" / f"{cell}.npz"
    if f.exists():
        return
    cfg = Config.load(mx.best_ckpt(NORM) / "config.yaml")
    tn = norm_stats()
    sizes = me.chrom_sizes()
    ctcf_motif = next(t.motif for t in cfg.data.tracks if t.name == "CTCF")
    have = {p.stem.upper(): p for p in Path(cell_dir(cell)).glob("*.bigwig")}
    miss = mx.missing_marks(Path(cell_dir(cell)))
    marks = [m for m in MARKS if m not in miss and m.upper() in have]
    mbw = pyBigWig.open(str(me.meth_dir(cell) / f"{me.TRACK}.bigwig"))
    cols = {k: [] for k in FEATURES + ["chrom", "pos"]}
    for m in marks:
        cols["y_" + m] = []
    for ch in CHROMS:
        n = sizes[ch]
        nb = -(-n // RES)
        n1 = -(-n // 1000)
        v = np.nan_to_num(np.array(mbw.stats(ch, 0, n, type="mean", nBins=n1), dtype=np.float64))
        v = np.pad(v, (0, nb * 10 - n1), mode="edge")
        # the bigWig holds one value per 1 kb; value - 1 is the methylated fraction
        # (raw zero = no data, treated as the mean of the cell)
        frac = np.where(v > 0, v - 1.0, np.nan)
        frac = np.where(np.isnan(frac), np.nanmean(frac), frac)
        b = frac.reshape(nb, 10)
        mean = b.mean(1)
        sd1 = np.sqrt(np.maximum(uniform_filter1d(frac ** 2, 101, mode="nearest")
                                 - uniform_filter1d(frac, 101, mode="nearest") ** 2, 0))
        feat = {
            "m_mean": mean, "m_min": b.min(1), "m_max": b.max(1), "m_sd": b.std(1),
            "m_ctx100k": uniform_filter1d(mean, 11, mode="nearest"),
            "m_ctx1M": uniform_filter1d(mean, 101, mode="nearest"),
            "m_sd100k": sd1.reshape(nb, 10).mean(1),
            "low_frac100k": uniform_filter1d((frac < 0.3).astype(float), 101,
                                             mode="nearest").reshape(nb, 10).mean(1),
            "low_frac_bin": (b < 0.3).mean(1),
            "high_frac100k": uniform_filter1d((frac > 0.8).astype(float), 101,
                                              mode="nearest").reshape(nb, 10).mean(1),
        }
        sf = cached_sequence_features(cfg.sequence.fasta, ch, ["gc", "cpg"], nb, RES,
                                      cfg.sequence.cache_dir)
        feat["gc"], feat["cpg"] = sf[0], sf[1]
        mt = cached_motif_tracks(cfg.sequence.fasta, ch, ctcf_motif, cfg.motifs.dir, nb, RES,
                                 cfg.motifs.score_frac, cfg.sequence.cache_dir)
        feat["ctcf_motif"] = mt[0] + mt[1]
        for k in FEATURES:
            cols[k].append(np.asarray(feat[k], np.float32)[:nb])
        for m in marks:
            x = read_track_binned(str(have[m.upper()]), ch, nb, RES)
            k = mx.ALL13.index(m)
            cols["y_" + m].append(((np.log1p(np.clip(x, 0, None)) - tn.mean[k]) / tn.std[k]
                                   ).astype(np.float32))
        cols["chrom"].append(np.full(nb, CHROMS.index(ch), np.int8))
        cols["pos"].append(np.arange(nb, dtype=np.int32))
    mbw.close()
    d = {k: np.concatenate(v) for k, v in cols.items()}
    ok = d["gc"] > 0                                    # N-gaps and unmappable ends out
    d = {k: v[ok] for k, v in d.items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(f) + ".tmp.npz", **d)
    os.rename(str(f) + ".tmp.npz", f)
    log(f"table {cell}: {int(ok.sum())} bins, marks {marks}")


def load(cell):
    z = np.load(OUT / "tables" / f"{cell}.npz")
    return {k: z[k] for k in z.files}


def stage_build():
    for c in CELLS:
        build_cell(c)


def r2(y, p):
    sse = float(((y - p) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1 - sse / sst if sst > 0 else np.nan


def fit_model(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                      min_samples_leaf=100, random_state=seed)
    return m.fit(X, y)


def stage_fit():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    D = {c: load(c) for c in CELLS}
    rng = np.random.default_rng(0)
    rows, res_rows, imp_rows = [], [], []
    feats = {"seq": F_SEQ, "seq+meth": FEATURES}

    def stack(cells, mark, chroms):
        Xs, ys = [], []
        for c in cells:
            d = D[c]
            if "y_" + mark not in d:
                continue
            sel = np.isin(d["chrom"], [CHROMS.index(x) for x in chroms])
            Xs.append(np.stack([d[k][sel] for k in FEATURES], 1))
            ys.append(d["y_" + mark][sel])
        return np.concatenate(Xs), np.concatenate(ys)

    def sub(X, y):
        if len(y) > N_FIT:
            i = rng.choice(len(y), N_FIT, replace=False)
            return X[i], y[i]
        return X, y

    for mark in MARKS:
        have = [c for c in CELLS if "y_" + mark in D[c]]
        train_have = [c for c in TRAIN_CELLS if c in have]
        if len(train_have) < 2:
            log(f"{mark}: fewer than 2 training cells with the track, skipped")
            continue
        for fs, cols in feats.items():
            idx = [FEATURES.index(k) for k in cols]
            # seen_cell: all training cells with the mark
            X, y = sub(*stack(train_have, mark, ODD))
            mdl = fit_model(X[:, idx], y)
            for c in have:
                Xt, yt = stack([c], mark, EVEN)
                p = mdl.predict(Xt[:, idx])
                rows.append({"mark": mark, "cell": c, "scheme": "seen_cell", "features": fs,
                             "held_out_cell": c in HELD_CELLS, "n": len(yt),
                             "r2": r2(yt, p), "pearson": float(np.corrcoef(yt, p)[0, 1])})
            # unseen_cell: leave one training cell out; K562 and HMEC use the fit above
            for c in have:
                if c in TRAIN_CELLS:
                    others = [x for x in train_have if x != c]
                    if not others:
                        continue
                    X, y = sub(*stack(others, mark, ODD))
                    m2 = fit_model(X[:, idx], y)
                else:
                    m2 = mdl
                Xt, yt = stack([c], mark, EVEN)
                p = m2.predict(Xt[:, idx])
                rows.append({"mark": mark, "cell": c, "scheme": "unseen_cell", "features": fs,
                             "held_out_cell": c in HELD_CELLS, "n": len(yt),
                             "r2": r2(yt, p), "pearson": float(np.corrcoef(yt, p)[0, 1])})
                if fs == "seq+meth":
                    # residual predicted by the other tracks (ridge, 5-fold CV over chromosomes)
                    d = D[c]
                    sel = np.isin(d["chrom"], [CHROMS.index(x) for x in EVEN])
                    others_t = [m for m in MARKS if m != mark and "y_" + m in d]
                    Z = np.stack([d["y_" + m][sel] for m in others_t], 1)
                    resid = yt - p
                    grp = d["chrom"][sel]
                    pr = np.zeros_like(resid)
                    for tr, te in GroupKFold(5).split(Z, resid, grp):
                        pr[te] = Ridge(alpha=10.0).fit(Z[tr], resid[tr]).predict(Z[te])
                    res_rows.append({"mark": mark, "cell": c, "held_out_cell": c in HELD_CELLS,
                                     "n_other_tracks": len(others_t),
                                     "r2_methylation_model": r2(yt, p),
                                     "residual_r2_from_other_tracks": r2(resid, pr),
                                     "r2_after_adding_other_tracks": r2(yt, p + pr)})
                    # permutation importance of the feature groups on this unseen fit
                    base = r2(yt, p)
                    for gname, g in (("local", F_LOCAL), ("context", F_CONTEXT), ("sequence", F_SEQ)):
                        Xp = Xt[:, idx].copy()
                        gi = [FEATURES.index(k) for k in g]
                        perm = rng.permutation(len(yt))
                        Xp[:, gi] = Xp[perm][:, gi]
                        imp_rows.append({"mark": mark, "cell": c, "group": gname,
                                         "r2_drop": base - r2(yt, m2.predict(Xp))})
            log(f"fit {mark} {fs}")
    RES_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES_OUT / "meth_track_prediction.csv", index=False, float_format="%.4f")
    pd.DataFrame(res_rows).to_csv(RES_OUT / "meth_track_residual.csv", index=False,
                                  float_format="%.4f")
    pd.DataFrame(imp_rows).to_csv(RES_OUT / "meth_track_importance.csv", index=False,
                                  float_format="%.4f")


def stage_summary():
    R = pd.read_csv(RES_OUT / "meth_track_prediction.csv")
    t = (R.groupby(["mark", "scheme", "features"]).agg(r2=("r2", "median"),
                                                      pearson=("pearson", "median"),
                                                      n_cells=("cell", "nunique")).reset_index())
    P = t.pivot_table(index="mark", columns=["scheme", "features"], values="r2")
    print("median R^2 over cells (test: even chromosomes)\n", P.round(3).to_string())
    h = R[R.held_out_cell].groupby(["mark", "scheme", "features"]).r2.median().unstack(
        ["scheme", "features"])
    print("held-out cells K562 and HMEC\n", h.round(3).to_string())
    for f in ("meth_track_residual.csv", "meth_track_importance.csv"):
        d = pd.read_csv(RES_OUT / f)
        key = ["mark"] if "residual" in f else ["mark", "group"]
        print(f, "\n", d.groupby(key).median(numeric_only=True).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", choices=["build", "fit", "summary"], required=True)
    a = ap.parse_args()
    if "build" in a.stage:
        stage_build()
    if "fit" in a.stage:
        stage_fit()
    if "summary" in a.stage:
        stage_summary()


if __name__ == "__main__":
    main()
