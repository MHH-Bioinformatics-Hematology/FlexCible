"""HiC-Reg retrained leak-free on the chromosome split of our models (CPU only).

HiC-Reg (Zhang et al. 2019, ~/src/HiC-Reg) predicts the Hi-C count of a pair of
5 kb bins from ChIP/DNase signals: per pair the signal of both bins (E, P), the
mean signal of the bins between them (W) and the genomic distance, a random
forest (regForest: 20 trees, bootstrap sample per tree, one third of the
features per split, leaf size 10) on the target ln(1 + SQRTVC count), pairs up
to 1 Mb. The released evaluation is a region-wise 5-fold CV inside each
chromosome (registry sections 18 and 63 / results/benchmark/HICREG_LEAK_AUDIT.md).

This script rebuilds the pipeline for our cells and tracks and trains it on our
chromosome-held-out setting: train on chr1, 4-10, 12, 13, 15, 16, 18-22, predict
chr2, 3, 11, 14, 17 of the same cell, nothing of the five chromosomes in
training or in any choice.

  python scripts/hicreg_retrain.py --stage data            # features + targets (all cells)
  python scripts/hicreg_retrain.py --stage tune  --cell gm12878   # validation chromosome
  python scripts/hicreg_retrain.py --stage fit   --cell gm12878 --feat window
  python scripts/hicreg_retrain.py --stage cv    --cell gm12878   # same pipeline, region-wise CV
  python scripts/hicreg_retrain.py --stage cools                  # predictions -> 5/10/25 kb count cools + O/E

Features (13 tracks of our models; HiC-Reg's DNase and TBP do not exist for our
cells and are dropped, SMC3 is added): mean bigWig signal per 5 kb bin (missing
bin = 0.01 as in HiC-Reg), E = bin i, P = bin j, W = mean of the bins strictly
between i and j (0.01 when none), Distance = gap in bp (j - i - 1) * 5000.
Target: ln(1 + count / (w_i * w_j)) with the VC_SQRT vector of the 4DN mcool
(the SQRTVC normalisation HiC-Reg uses; the mcool stores Juicer-convention vectors, which
divide: checked against the ICE weights, log-correlation 0.84 at 50 kb against 0.49 when multiplying), zero-count pairs of mappable bins
included (target 0), so that every pair of the chromosome can be predicted and
the predicted pair list does not depend on the target.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
OUT = HM / "runs/benchmark/hicreg_retrain"
PP = Path("/mnt/processed/wolffjoa/restart_of_prediction_project")
RES = 5000
REACH = 200                                    # bins: 1 Mb, HiC-Reg's maximum distance
FLOOR = 0.01
TRACKS = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
          "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
MCOOL = {"gm12878": PP / "originalData/4Dnucleosome/GM12878_4DNFIXP4QG5B.mcool",
         "k562": PP / "originalData/4Dnucleosome/K562_4DNFI18UHVRO.mcool"}
BIGWIG = {"gm12878": Path.home() / "data/hic-multiome/hg38_gm12878",
          "k562": HM / "hg38_k562", "huvec": HM / "hg38_huvec", "hmec": HM / "hg38_hmec",
          "nhek": HM / "hg38_nhek"}
MC_CELLS = ["gm12878", "k562", "huvec", "hmec", "nhek"]      # the five cells of HiC-Reg MULTICELL
TEST = ["chr2", "chr3", "chr11", "chr14", "chr17"]
TRAIN = ["chr1"] + [f"chr{i}" for i in (4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21, 22)]
VALID = "chr10"                                # hyper-parameter selection; never a test chromosome
CHROMS = TRAIN + TEST


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def signal_cell(cell: str, chroms_sizes: dict) -> dict:
    import pyBigWig
    out = {}
    bws = {}
    for t in TRACKS:
        f = BIGWIG[cell] / f"{t}.bigwig"
        bws[t] = pyBigWig.open(str(f)) if f.exists() else None
    for ch, size in chroms_sizes.items():
        n = -(-size // RES)
        X = np.full((n, len(TRACKS)), FLOOR, np.float32)
        for k, t in enumerate(TRACKS):
            bw = bws[t]
            if bw is None or ch not in bw.chroms():
                continue
            v = np.array(bw.stats(ch, 0, min(size, n * RES), nBins=n, type="mean"), dtype=np.float64)
            X[:, k] = np.where(np.isfinite(v), v, FLOOR).astype(np.float32)
        out[ch] = X
    return out


def target_cell(cell: str, chroms: list[str]):
    """Banded ln(1 + SQRTVC count): array (n, REACH), column d-1 holds pair (i, i + d);
    NaN where either bin is unmappable (VC_SQRT = 0 or no ICE weight)."""
    import cooler
    c = cooler.Cooler(f"{MCOOL[cell]}::resolutions/{RES}")
    bins = c.bins()[:]
    out = {}
    for ch in chroms:
        lo, hi = c.extent(ch)
        b = bins.iloc[lo:hi]
        w = b["VC_SQRT"].values.astype(np.float64)
        ok = np.isfinite(w) & (w > 0) & np.isfinite(b["weight"].values)
        n = hi - lo
        T = np.full((n, REACH), np.nan, np.float32)
        m = c.matrix(balance=False, sparse=True).fetch(ch).tocoo()
        r, q, v = m.row, m.col, m.data.astype(np.float64)
        keep = (q > r) & (q - r <= REACH)
        r, q, v = r[keep], q[keep], v[keep]
        # mappable pair slots start at 0, observed pixels overwrite with the normalised count
        ii = np.arange(n)[:, None]
        dd = np.arange(1, REACH + 1)[None, :]
        jj = ii + dd
        valid = (jj < n) & ok[:, None] & ok[np.clip(jj, 0, n - 1)]
        T[valid] = 0.0
        val = np.log1p(v / (w[r] * w[q]))
        good = ok[r] & ok[q]
        T[r[good], (q - r)[good] - 1] = val[good].astype(np.float32)
        out[ch] = T
        log(f"target {cell} {ch}: {n} bins, {int(ok.sum())} mappable, {int(valid.sum())} pair slots, "
            f"{int((T > 0).sum())} nonzero")
    return out


def stage_data(cells):
    import cooler
    OUT.mkdir(parents=True, exist_ok=True)
    c = cooler.Cooler(f"{MCOOL['gm12878']}::resolutions/{RES}")
    sizes = {ch: int(c.chromsizes[ch]) for ch in CHROMS + ["chrX"]}
    for cell in (cells or MC_CELLS):
        f = OUT / f"sig_{cell}.npz"
        if not f.exists():
            np.savez(f, **signal_cell(cell, sizes))
            log(f"signal {cell} written")
    for cell in (cells or ["gm12878", "k562"]):
        if cell not in MCOOL:
            continue
        f = OUT / f"tgt_{cell}.npz"
        if not f.exists():
            np.savez(f, **target_cell(cell, CHROMS))
            log(f"target {cell} written")


# --------------------------------------------------------------------------- #
# pair features
# --------------------------------------------------------------------------- #
class Chrom:
    """Per-chromosome signal arrays of one or several cells and the banded target."""

    def __init__(self, cells: list[str], target_cell_name: str, chrom: str, with_target=True):
        self.chrom = chrom
        self.sigs = []
        for cell in cells:
            z = np.load(OUT / f"sig_{cell}.npz")
            self.sigs.append(z[chrom])
        self.cs = [np.vstack([np.zeros((1, s.shape[1]), np.float64), np.cumsum(s, axis=0, dtype=np.float64)])
                   for s in self.sigs]
        self.n = self.sigs[0].shape[0]
        self.T = None
        if with_target:
            self.T = np.load(OUT / f"tgt_{target_cell_name}.npz")[chrom]

    def feats(self, i: np.ndarray, d: np.ndarray) -> np.ndarray:
        """(len(i), 3 * tracks * cells + 1) float32 for the pairs (i, i + d)."""
        j = i + d
        cols = []
        wmean_ok = d >= 2
        den = np.where(wmean_ok, d - 1, 1).astype(np.float64)
        for s, cs in zip(self.sigs, self.cs):
            cols.append(s[i])
            cols.append(s[j])
            w = (cs[j] - cs[i + 1]) / den[:, None]
            w[~wmean_ok] = FLOOR
            cols.append(w.astype(np.float32))
        cols.append(((d - 1) * RES).astype(np.float32)[:, None])
        return np.concatenate(cols, axis=1)

    def target(self, i, d):
        return self.T[i, d - 1]

    def all_pairs(self, valid_only: bool):
        """All pair slots (i, d), d in 1..REACH, j < n; with a target only the mappable ones."""
        ii, dd = np.nonzero(np.ones((self.n, REACH), bool))
        dd = dd + 1
        keep = (ii + dd) < self.n
        if valid_only:
            keep &= np.isfinite(self.T[ii, dd - 1])
        return ii[keep].astype(np.int64), dd[keep].astype(np.int64)


def feature_names(cells):
    names = []
    for c in cells:
        for part in ("E", "P", "W"):
            names += [f"{t}_{part}" + ("" if len(cells) == 1 else f"_{c}") for t in TRACKS]
    return names + ["Distance"]


def sample_pairs(cells, tcell, chroms, n_total, seed, region_mask=None):
    """Uniform random sample of mappable pairs over the chromosomes (n_total in all)."""
    rng = np.random.default_rng(seed)
    objs = {ch: Chrom(cells, tcell, ch) for ch in chroms}
    cnt = {ch: int(np.isfinite(o.T).sum()) for ch, o in objs.items()}
    tot = sum(cnt.values())
    X, y = [], []
    for ch, o in objs.items():
        k = int(round(n_total * cnt[ch] / tot))
        ii, dd = o.all_pairs(True)
        sel = rng.choice(len(ii), min(k, len(ii)), replace=False)
        sel.sort()
        X.append(o.feats(ii[sel], dd[sel]))
        y.append(o.target(ii[sel], dd[sel]))
    return np.concatenate(X), np.concatenate(y).astype(np.float32)


def make_rf(n_trees=20, leaf=10, n_jobs=1, seed=0):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=leaf, max_features=1 / 3,
                                 bootstrap=True, n_jobs=n_jobs, random_state=seed)


def dist_centred_r(pred, truth, dist_bins):
    """Pearson of the residuals after removing the per-distance mean of each (the part of the
    score that is not the distance decay)."""
    import pandas as pd
    df = pd.DataFrame({"p": pred.astype(np.float64), "t": truth.astype(np.float64), "d": dist_bins})
    df["p"] -= df.groupby("d").p.transform("mean")
    df["t"] -= df.groupby("d").t.transform("mean")
    return float(np.corrcoef(df.p, df.t)[0, 1])


# --------------------------------------------------------------------------- #
# tune: validation chromosome only
# --------------------------------------------------------------------------- #
def stage_tune(cell, feat, jobs, grid_sizes, grid_leaf, n_val=1_000_000):
    import pandas as pd
    cells = MC_CELLS if feat == "multicell" else [cell]
    tcell = cell
    prev = OUT / f"tune_{cell}_{feat}.csv"
    rows = pd.read_csv(prev).to_dict("records") if prev.exists() else []
    train_chroms = [c for c in TRAIN if c != VALID]
    ov = Chrom(cells, tcell, VALID)
    ii, dd = ov.all_pairs(True)
    rng = np.random.default_rng(1)
    sel = np.sort(rng.choice(len(ii), min(n_val, len(ii)), replace=False))
    Xv, yv, dv = ov.feats(ii[sel], dd[sel]), ov.target(ii[sel], dd[sel]), dd[sel]
    for n in grid_sizes:
        X, y = sample_pairs(cells, tcell, train_chroms, n, seed=0)
        for leaf in grid_leaf:
            t0 = time.time()
            rf = make_rf(20, leaf, jobs).fit(X, y)
            p = rf.predict(Xv)
            row = dict(cell=cell, feat=feat, n_train=n, leaf=leaf, valid_chrom=VALID,
                       r_raw=float(np.corrcoef(p, yv)[0, 1]),
                       r_dist_centred=dist_centred_r(p, yv, dv),
                       rmse=float(np.sqrt(np.mean((p - yv) ** 2))), minutes=(time.time() - t0) / 60)
            rows.append(row)
            log(json.dumps(row))
            pd.DataFrame(rows).to_csv(OUT / f"tune_{cell}_{feat}.csv", index=False, float_format="%.4f")
    return rows


# --------------------------------------------------------------------------- #
# fit on all training chromosomes and predict the test chromosomes
# --------------------------------------------------------------------------- #
def stage_fit(cell, feat, jobs, n_train, leaf, n_trees, chroms, suffix=""):
    import joblib
    cells = MC_CELLS if feat == "multicell" else [cell]
    tag = f"{cell}_{feat}{suffix}"
    X, y = sample_pairs(cells, cell, TRAIN, n_train, seed=0)
    log(f"fit {tag}: {len(y)} pairs, {X.shape[1]} features, leaf {leaf}, trees {n_trees}")
    t0 = time.time()
    rf = make_rf(n_trees, leaf, jobs).fit(X, y)
    log(f"fit done {(time.time() - t0) / 60:.1f} min")
    del X
    for ch in chroms:
        o = Chrom(cells, cell, ch, with_target=False)
        ii, dd = np.nonzero(np.ones((o.n, REACH), bool))
        dd = dd + 1
        keep = (ii + dd) < o.n
        ii, dd = ii[keep].astype(np.int64), dd[keep].astype(np.int64)
        P = np.full((o.n, REACH), np.nan, np.float32)
        step = 2_000_000
        for s in range(0, len(ii), step):
            a, b = ii[s:s + step], dd[s:s + step]
            P[a, b - 1] = rf.predict(o.feats(a, b)).astype(np.float32)
        np.savez_compressed(OUT / f"pred_{tag}_{ch}.npz", P=P)
        log(f"predicted {tag} {ch}: {len(ii)} pairs")
    if feat == "window" and not suffix:  # this cell's model on the other cell's features (other chromosomes)
        other = "k562" if cell == "gm12878" else "gm12878"
        for ch in [c for c in chroms if c in ("chr14", "chr17")]:
            o = Chrom([other], other, ch, with_target=False)
            ii, dd = np.nonzero(np.ones((o.n, REACH), bool))
            dd = dd + 1
            keep = (ii + dd) < o.n
            ii, dd = ii[keep].astype(np.int64), dd[keep].astype(np.int64)
            P = np.full((o.n, REACH), np.nan, np.float32)
            for s_ in range(0, len(ii), 2_000_000):
                a, b = ii[s_:s_ + 2_000_000], dd[s_:s_ + 2_000_000]
                P[a, b - 1] = rf.predict(o.feats(a, b)).astype(np.float32)
            np.savez_compressed(OUT / f"pred_{cell}to{other}_crosscell_{ch}.npz", P=P)
            log(f"cross-cell {cell} -> {other} {ch}")
    json.dump(dict(cell=cell, feat=feat, n_train=int(len(y)), leaf=leaf, trees=n_trees,
                   train_chroms=TRAIN, test_chroms=chroms, features=feature_names(cells),
                   fit_minutes=(time.time() - t0) / 60), open(OUT / f"model_{tag}.json", "w"),
              indent=1)


# --------------------------------------------------------------------------- #
# cv: HiC-Reg's own protocol (region-wise 5-fold inside one chromosome) on our pipeline
# --------------------------------------------------------------------------- #
def stage_cv(cell, jobs, n_train, leaf, n_trees, chroms, folds=5):
    """Regions of the chromosome are assigned at random to five folds; the model of fold k
    trains on the pairs with both regions outside fold k and predicts every other pair
    (as in the released CV models, results/benchmark/hicreg_split.csv). Each pair is
    predicted once, by the model of the lowest fold among its two regions."""
    tag = f"{cell}_window"
    for ch in chroms:
        o = Chrom([cell], cell, ch)
        rng = np.random.default_rng(0)
        fold = rng.integers(0, folds, o.n)
        ii, dd = o.all_pairs(True)
        fi, fj = fold[ii], fold[ii + dd]
        P = np.full((o.n, REACH), np.nan, np.float32)
        allii, alldd = np.nonzero(np.ones((o.n, REACH), bool))
        alldd = alldd + 1
        k_ = (allii + alldd) < o.n
        allii, alldd = allii[k_].astype(np.int64), alldd[k_].astype(np.int64)
        afi, afj = fold[allii], fold[allii + alldd]
        rng2 = np.random.default_rng(1)
        for k in range(folds):
            tr = np.nonzero((fi != k) & (fj != k))[0]
            tr = rng2.choice(tr, min(n_train, len(tr)), replace=False)
            rf = make_rf(n_trees, leaf, jobs).fit(o.feats(ii[tr], dd[tr]), o.target(ii[tr], dd[tr]))
            te = np.nonzero(((afi == k) | (afj == k)) & (np.minimum(afi, afj) == k))[0]
            for s in range(0, len(te), 2_000_000):
                q = te[s:s + 2_000_000]
                P[allii[q], alldd[q] - 1] = rf.predict(o.feats(allii[q], alldd[q])).astype(np.float32)
            log(f"cv {tag} {ch} fold {k}: train {len(tr)}, test {len(te)}")
        np.savez_compressed(OUT / f"pred_{cell}_windowcvours_{ch}.npz", P=P)


# --------------------------------------------------------------------------- #
# cools: predictions -> count cools at 5/10/25 kb (summing 5 kb counts, as section 18) and O/E
# --------------------------------------------------------------------------- #
def counts_cool(name, tag, chrom, res, sizes):
    import cooler
    import pandas as pd
    P = np.load(OUT / f"pred_{tag}_{chrom}.npz")["P"]
    ii, dd = np.nonzero(np.isfinite(P))
    v = np.expm1(P[ii, dd].astype(np.float64))
    b1, b2 = ii.astype(np.int64), ii.astype(np.int64) + dd + 1
    f = res // RES
    df = pd.DataFrame({"bin1_id": b1 // f, "bin2_id": b2 // f, "count": v})
    df = df.groupby(["bin1_id", "bin2_id"], as_index=False)["count"].sum()
    size = sizes[chrom]
    starts = np.arange(0, size, res)
    bins = pd.DataFrame({"chrom": chrom, "start": starts, "end": np.minimum(starts + res, size)})
    df = df[(df.bin2_id < len(bins)) & (df.bin1_id <= df.bin2_id)].sort_values(["bin1_id", "bin2_id"])
    out = OUT / "counts" / f"{name}_{chrom}.cool"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    cooler.create_cooler(str(out), bins=bins, pixels=df, dtypes={"count": np.float64}, ordered=True)
    return out


def variant_names():
    """competitor name -> (tag of the prediction files, cell, res, chromosomes)."""
    v = {}
    for cell in ("gm12878", "k562"):
        for res in (25000, 10000, 5000):
            for feat, chroms in (("window", TEST), ("multicell", TEST), ("windowcvours", ["chr14", "chr17"]),
                                 ("windowsmall", ["chr14", "chr17"]), ("windowsupport", ["chr14", "chr17"])):
                v[f"hicregre{feat}_{cell}_{res // 1000}kb"] = (f"{cell}_{feat}", cell, res, chroms)
            other = "k562" if cell == "gm12878" else "gm12878"      # model of `other` applied to `cell`
            v[f"hicregrecrosscell_{other}to{cell}_{res // 1000}kb"] = (f"{other}to{cell}_crosscell", cell,
                                                                       res, ["chr14", "chr17"])
    return v


def comp_spec(name):
    tag, cell, res, chroms = variant_names()[name]
    return dict(label="HiC-Reg retrained (our split)", path=OUT / "counts" / (name + "_{chrom}.cool"),
                per_chrom=True, build="hg38", res=res, reach=1_000_000, space="counts",
                heldout=TEST, cell=cell, chroms=chroms)


def register():
    sys.path.insert(0, str(REPO / "scripts"))
    import benchmark_competitors as bc
    for name in variant_names():
        bc.COMPETITORS[name] = comp_spec(name)
    return bc


def stage_cools():
    import cooler
    bc = register()
    c = cooler.Cooler(f"{MCOOL['gm12878']}::resolutions/{RES}")
    sizes = {ch: int(c.chromsizes[ch]) for ch in CHROMS}
    for name, (tag, cell, res, chroms) in variant_names().items():
        for ch in chroms:
            if not (OUT / f"pred_{tag}_{ch}.npz").exists():
                continue
            out = OUT / "counts" / f"{name}_{ch}.cool"
            if not out.exists():
                counts_cool(name, tag, ch, res, sizes)
                log(f"cool {out.name}")
        if all((OUT / f"pred_{tag}_{ch}.npz").exists() for ch in chroms):
            bc.run_oe(name, bc.COMPETITORS[name])


# --------------------------------------------------------------------------- #
# check: the scikit-learn forest against the original regForest on the same pairs
# --------------------------------------------------------------------------- #
def stage_check(cell, n_train, n_test, leaf, trees, regforest, workdir):
    """Writes HiC-Reg training/test files (README format) of a random subsample of the training
    chromosomes (train) and of the validation chromosome (test), runs the original regForest
    binary (-k1 -l<leaf> -n<trees>) and the scikit-learn forest on the same rows and compares
    their predictions (check_<cell>.json)."""
    import subprocess
    import pandas as pd
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    names = feature_names([cell])
    X, y = sample_pairs([cell], cell, [c for c in TRAIN if c != VALID], n_train, seed=3)
    o = Chrom([cell], cell, VALID)
    ii, dd = o.all_pairs(True)
    sel = np.sort(np.random.default_rng(4).choice(len(ii), n_test, replace=False))
    Xt, yt = o.feats(ii[sel], dd[sel]), o.target(ii[sel], dd[sel])

    def write(path, X_, y_, tag):
        df = pd.DataFrame(X_, columns=names)
        df.insert(0, "Pair", [f"{tag}_{k}_{k + 5000}-{tag}_{k + 5000 * (1 + k % 7)}_{k + 5000 * (2 + k % 7)}"
                              for k in range(len(df))])
        df["Count"] = y_
        df.to_csv(path, sep="\t", index=False, float_format="%.6g")

    write(wd / "train.txt", X, y, "chrT")
    write(wd / "test.txt", Xt, yt, "chrV")
    (wd / "prior.txt").write_text("".join(f"{n}\tCount\n" for n in names))
    (wd / "out").mkdir(exist_ok=True)
    t0 = time.time()
    subprocess.run([regforest, "-t", str(wd / "train.txt"), "-o", str(wd / "out") + "/", "-k1",
                    f"-l{leaf}", f"-n{trees}", "-b", str(wd / "prior.txt"), "-d", str(wd / "test.txt")],
                   check=True, stdout=open(wd / "regforest.log", "w"), stderr=subprocess.STDOUT)
    t_rf = time.time() - t0
    te = pd.read_csv(wd / "out" / "testset_error.txt", sep="\t", skiprows=1, header=None)
    p_rf = te[2].values
    t0 = time.time()
    p_sk = make_rf(trees, leaf, 8).fit(X, y).predict(Xt)
    t_sk = time.time() - t0
    res = dict(cell=cell, n_train=int(len(y)), n_test=int(len(yt)), leaf=leaf, trees=trees,
               regforest_seconds=t_rf, sklearn_seconds=t_sk,
               r_regforest_truth=float(np.corrcoef(p_rf, yt)[0, 1]),
               r_sklearn_truth=float(np.corrcoef(p_sk, yt)[0, 1]),
               r_regforest_sklearn=float(np.corrcoef(p_rf, p_sk)[0, 1]),
               rmse_regforest=float(np.sqrt(np.mean((p_rf - yt) ** 2))),
               rmse_sklearn=float(np.sqrt(np.mean((p_sk - yt) ** 2))),
               r_dist_centred_regforest=dist_centred_r(p_rf, yt, dd[sel]),
               r_dist_centred_sklearn=dist_centred_r(p_sk, yt, dd[sel]))
    json.dump(res, open(OUT / f"check_{cell}.json", "w"), indent=1)
    log(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--feat", default="window", choices=["window", "multicell"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=200_000_000,
                    help="training pairs sampled over the training chromosomes (default: all of them)")
    ap.add_argument("--leaf", type=int, default=30,
                    help="minimum leaf size; HiC-Reg's default is 10, 30 is the validation choice (tune)")
    ap.add_argument("--trees", type=int, default=20)
    ap.add_argument("--grid-sizes", type=int, nargs="+", default=[1_000_000, 4_000_000])
    ap.add_argument("--grid-leaf", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--chroms", nargs="+", default=TEST)
    ap.add_argument("--regforest", default=str(Path.home() / "src/HiC-Reg/regForest"))
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--suffix", default="", help="fit: tag suffix of the prediction files (size control: small)")
    ap.add_argument("--n-test", type=int, default=20_000)
    a = ap.parse_args()
    for st in a.stage:
        if st == "data":
            stage_data([a.cell] if a.cell else None)
        elif st == "tune":
            stage_tune(a.cell, a.feat, a.jobs, a.grid_sizes, a.grid_leaf)
        elif st == "fit":
            stage_fit(a.cell, a.feat, a.jobs, a.n_train, a.leaf, a.trees, a.chroms, a.suffix)
        elif st == "cv":
            stage_cv(a.cell, a.jobs, a.n_train, a.leaf, a.trees, a.chroms)
        elif st == "check":
            stage_check(a.cell, a.n_train, a.n_test, a.leaf, a.trees, a.regforest,
                        a.workdir or str(OUT / f"check_{a.cell}"))
        elif st == "cools":
            stage_cools()
        else:
            raise SystemExit(f"unknown stage {st}")


if __name__ == "__main__":
    main()
