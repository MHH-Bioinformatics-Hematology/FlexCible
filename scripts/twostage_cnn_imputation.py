"""Two-stage model: do ChIP tracks predicted by the sequence CNN help the track-dropout
Hi-C model when the measured tracks are missing? (registry section 60)

Stage 1 is the sequence + methylation CNN of scripts/seq_cnn_chip.py (variant
seq_meth_aug, registry sections 42 and 59), which predicts the 13 ChIP marks at 10 kb
from DNA sequence and the cell's Meth track. Stage 2 is the existing Hi-C model
(default all12refres_drop_meth_10kb: 13 ChIP tracks + Meth, residual on the consensus,
track dropout 0.5), run with some tracks measured and the others

  absent   the absent value (current behaviour of the dropout model)
  imp      the CNN prediction of the mark
  shuf     the CNN prediction 1 Mb (100 bins) downstream on the same chromosome
           (control: right distribution and cell, wrong locus)

Scale mapping. The CNN target is log1p of the 10 kb bin mean of the bigWig, z-scored
per mark with the pooled statistics of the seven training cells (training
chromosomes, global target normalisation; Data.mu, Data.sd). The Hi-C model reads the
same bin mean as log1p, z-scored per track with its own training statistics
(norm_tracks.json, no smoothing). The CNN output z is therefore converted by
z*sd + mu (back to log1p units) and then by (v - mean_hic) / std_hic. No per-cell
scaling and no measured ChIP signal of the evaluated cell enters.

Scenarios (Meth always given to the Hi-C model unless 'nometh'): all13 (every mark
measured; the two cohesin marks HMEC lacks stay absent), none (no
ChIP: 'meth' with Meth, 'none' without), ctcf, ctcf_h3k27ac, five (CTCF, H3K27ac,
H3K4me3, H3K27me3, H3K36me3).

    python scripts/twostage_cnn_imputation.py --stage cnn distribution predict summary
"""
from __future__ import annotations

import argparse
import os
import subprocess
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
import seq_cnn_chip as S                                            # noqa: E402

WORK = mx.HM / "runs/seq_cnn_twostage"          # shared drive: CNN checkpoint, caches, dumps
CNN_RUNS = WORK / "runs"
PRED = WORK / "pred"
OUT = WORK / "crosscell"
RES_DIR = REPO / "results/twostage_cnn"
PY = os.environ.get("HICMULTI_PY", sys.executable)
RES = 10000
SHIFT_BINS = 100                                 # 1 Mb at 10 kb
CNN_VARIANT = ["--mode", "seq_meth", "--meth-aug"]
CELLS = ["k562", "hmec_deep", "gm12878", "imr90", "h1", "hepg2", "a549", "gm23248", "cd14mono"]
UNSEEN = ["k562", "hmec_deep"]
FIVE = ["CTCF", "H3K27ac", "H3K4me3", "H3K27me3", "H3K36me3"]
SCEN = {"all13": list(S.MARKS), "none": [], "ctcf": ["CTCF"],
        "ctcf_h3k27ac": ["CTCF", "H3K27ac"], "five": FIVE}
VARIANTS = ["", "@imp", "@shuf"]
NOISE = 0.013
CLEAN_CHROMS = set(S.TEST_CHROMS)               # chromosomes the CNN never saw (even, not chr2/chr4)


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def keep_sets() -> dict[str, list[str]]:
    """set name -> tracks kept (the rest absent, or imputed in the @imp / @shuf variants)."""
    out = {}
    for sc, marks in SCEN.items():
        for meth in (True, False):
            base = {"none+meth": "meth"}.get(sc + ("+meth" if meth else ""), sc + ("+meth" if meth else ""))
            for v in VARIANTS:
                out[base + v] = marks + ([me.TRACK] if meth else [])
    return out


def cnn_cell(cell):
    return S.ALIAS.get(cell, cell)


def regions_table():
    import score_like_hicgan as sc
    return pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])


# ----------------------------------------------------------------------------
# stage 1: CNN predictions over the bins the Hi-C windows read
# ----------------------------------------------------------------------------

def needed_bins(nb_cool: dict, size: int):
    """chrom -> bool mask of the 10 kb bins inside any evaluation window (window = size
    bins centred on the region centre, moved inside the chromosome as crosscell_predict does)."""
    need = {}
    for _, r in regions_table().iterrows():
        nb = nb_cool[r.chrom]
        centre = (int(r.start) + int(r.end)) // 2 // RES
        st = max(0, min(nb - size, centre - size // 2))
        need.setdefault(r.chrom, np.zeros(nb, bool))[st:st + size] = True
    return need


def stage_cnn(cells, gpu, force=False):
    import cooler
    import torch
    PRED.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(str(mx.target("k562", RES)))
    from hicmultiome.genome import match_name
    from hicmultiome.precompute import chrom_nbins_of
    nb_cool = {c: chrom_nbins_of(clr, match_name(c, list(clr.chromnames)))
               for c in regions_table().chrom.unique()}
    from hicmultiome.config import Config
    size = 192
    need = needed_bins(nb_cool, size)
    a = S.parse_args(["--config", str(REPO / "configs/seq_cnn.yaml"), "--stage", "eval",
                      "--out-root", str(CNN_RUNS), "--eval-batch", "16", "--num-workers", "4",
                      *CNN_VARIANT])
    assert a.run_name == "seq_meth_aug", a.run_name
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    dev = S.pick_device(a)
    D = S.Data(a)
    model = S.make_model(a, S.in_channels(a)).to(dev)
    model.load_state_dict(torch.load(Path(a.out_dir) / "ckpt_best.pt", map_location=dev,
                                     weights_only=False)["model"])
    np.savez(PRED / "cnn_norm.npz", mu=D.mu, sd=D.sd, marks=np.array(S.MARKS))
    # bins to predict: needed, and for the shifted control the bins SHIFT_BINS downstream
    need_pred = {}
    for c, m in need.items():
        n = D.nb[c]
        m2 = np.zeros(n, bool)
        mm = m[:n]
        m2[:len(mm)] = mm
        need_pred[c] = m2 | np.roll(m2, SHIFT_BINS)
    for cell in cells:
        cc = cnn_cell(cell)
        f1, f2 = PRED / f"pred_{cell}.npz", PRED / f"shuf_{cell}.npz"
        if f1.exists() and f2.exists() and not force:
            log(f"{cell}: done")
            continue
        items = []
        for c, m in need_pred.items():
            b0s = np.unique(np.flatnonzero(m) // D.NB * D.NB)
            items += [(c, int(b), cc) for b in b0s]
        log(f"{cell}: {len(items)} windows")
        t = time.time()
        P = S.predict(model, D, a, dev, items, True, lambda c: [])[0]          # z scale
        L = {c: np.full((len(S.MARKS), D.nb[c]), np.nan, np.float32) for c in need_pred}
        for (c, b0, _), p in zip(items, P):
            e = min(b0 + D.NB, D.nb[c])
            L[c][:, b0:e] = (p[:e - b0] * D.sd + D.mu).T
        marks = np.array(S.MARKS)
        np.savez_compressed(f1, marks=marks, **L)
        np.savez_compressed(f2, marks=marks, **{c: np.roll(v, -SHIFT_BINS, axis=1) for c, v in L.items()})
        log(f"{cell}: {time.time() - t:.0f} s")


# ----------------------------------------------------------------------------
# scale check: imputed against measured tracks in the Hi-C model's input space
# ----------------------------------------------------------------------------

def hic_norm():
    from hicmultiome.tracks import TrackNormalizer
    ck = mx.best_ckpt(mx.out_of(MODEL) / MODEL)
    tn = TrackNormalizer.load(ck / "norm_tracks.json")
    names = [t for t in __import__("yaml").safe_load(open(ck / "config.yaml"))["data"]["tracks"]]
    return tn, [t["name"] for t in names]


def stage_distribution(cells):
    """Per cell and mark: mean, sd and Pearson of imputed vs measured track in the Hi-C
    model's z units over the bins of the evaluation windows (all, and CNN test chromosomes)."""
    tn, tnames = hic_norm()
    row = {t: i for i, t in enumerate(tnames)}
    rows = []
    cache = CNN_RUNS / "cache"
    for cell in cells:
        cc = cnn_cell(cell)
        meas = np.load(cache / f"targets_{RES}" / f"{cc}.npz")
        pred = np.load(PRED / f"pred_{cell}.npz")
        for chrom in [k for k in pred.files if k != "marks"]:
            L = pred[chrom]
            M = meas[chrom].astype(np.float32).T                    # (13, nb) log1p measured
            n = min(L.shape[1], M.shape[1])
            for k, m in enumerate(S.MARKS):
                ok = np.isfinite(L[k, :n]) & np.isfinite(M[k, :n])
                if ok.sum() < 10:
                    continue
                zi = (L[k, :n][ok] - tn.mean[row[m]]) / tn.std[row[m]]
                zm = (M[k, :n][ok] - tn.mean[row[m]]) / tn.std[row[m]]
                rows.append(dict(cell=cell, mark=m, chrom=chrom, clean=chrom in CLEAN_CHROMS,
                                 n=int(ok.sum()), zi=zi, zm=zm))
    # aggregate per cell, mark, clean
    out = []
    df = pd.DataFrame(rows)
    for (cell, mark, clean), g in df.groupby(["cell", "mark", "clean"]):
        zi, zm = np.concatenate(g.zi.values), np.concatenate(g.zm.values)
        q = [0.05, 0.5, 0.95]
        out.append(dict(cell=cell, mark=mark, clean_chroms=clean, n_bins=len(zi),
                        mean_measured=zm.mean(), sd_measured=zm.std(),
                        mean_imputed=zi.mean(), sd_imputed=zi.std(),
                        sd_ratio=zi.std() / zm.std(),
                        q05_measured=np.quantile(zm, q[0]), q50_measured=np.quantile(zm, q[1]),
                        q95_measured=np.quantile(zm, q[2]),
                        q05_imputed=np.quantile(zi, q[0]), q50_imputed=np.quantile(zi, q[1]),
                        q95_imputed=np.quantile(zi, q[2]),
                        pearson=float(np.corrcoef(zi, zm)[0, 1]) if zi.std() > 0 and zm.std() > 0 else np.nan))
    RES_DIR.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(out)
    o.to_csv(RES_DIR / "scale_check.csv", index=False, float_format="%.4f")
    return o


# ----------------------------------------------------------------------------
# stage 2: Hi-C model, scenarios
# ----------------------------------------------------------------------------

MODEL = "all12refres_drop_meth_10kb"


def stage_predict(model, cells, gpu, structure):
    run = mx.out_of(model) / model
    ck = mx.best_ckpt(run)
    sets = keep_sets()
    for cell in cells:
        out = OUT / model
        if all((out / s / f"predict_{cell}.csv").exists() for s in sets):
            log(f"{cell}: done")
            continue
        _, bw, assay, _ = mx.CELLS[cell]
        ref = mx.reference_of(model, cell)
        tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
        cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
               "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw,
               "--target", mx.target(cell, RES), "--tta", "--assay", assay, "--binsize", RES,
               "--device", "cuda", "--out-dir", out, "--reference", ref,
               "--keep-sets", *[f"{s}={','.join(t)}" for s, t in sets.items()],
               "--impute", f"imp={PRED / f'pred_{cell}.npz'}", f"shuf={PRED / f'shuf_{cell}.npz'}",
               "--others", f"{tag}={ref}"]
        if structure:
            cmd.append("--structure")
        if me.has_meth(cell):
            cmd += ["--extra-bigwig-dir", me.meth_dir(cell)]
        env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3",
                   CUDA_VISIBLE_DEVICES=gpu)
        out.mkdir(parents=True, exist_ok=True)
        log(f"{cell}: crosscell_predict, {len(sets)} sets")
        with open(out / f"predict_{cell}.log", "a") as fh:
            fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
            fh.flush()
            rc = subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=REPO, env=env)
        log(f"{cell}: exit {rc}")


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------

METRICS = {"pearson": "model", "scc": "struct_scc", "insulation": "struct_insulation_corr",
           "boundary_f1": "struct_boundary_f1", "loop_enrichment": "struct_loop_enrich_corr",
           "loop_apa": "struct_loop_apa_ratio", "compartment": "struct_compartment_corr",
           "eigenvector": "struct_eigenvector_abs_corr"}


def load_all(model):
    rows = []
    for s in keep_sets():
        for cell in CELLS:
            f = OUT / model / s / f"predict_{cell}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d["set"], d["cell"] = s, cell
            rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["chrom"] = d.region.str.split(":").str[0]
    d["clean"] = d.chrom.isin(CLEAN_CHROMS)
    d["scenario"] = d.set.str.split("@").str[0]
    d["variant"] = np.where(d.set.str.contains("@"), d.set.str.split("@").str[1], "absent")
    return d


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), (n, len(x)))
    med = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))


def stage_summary(model):
    d = load_all(model)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    cons = d[d.variant == "absent"].groupby(["cell", "region"])[
        [c for c in d.columns if c.startswith("base_consensus")]].first()
    cons["consensus"] = cons.bfill(axis=1).iloc[:, 0]
    cons = cons["consensus"].reset_index()
    subsets = {"all413": lambda g: g, "clean169": lambda g: g[g.clean]}
    med_rows, diff_rows = [], []
    for sub, sel in subsets.items():
        for cell in CELLS:
            dc = sel(d[d.cell == cell])
            n_reg = dc.region.nunique()
            for scen in dict.fromkeys(d.scenario):
                piv = {v: dc[(dc.scenario == scen) & (dc.variant == v)].set_index("region") for v in
                       ["absent", "imp", "shuf"]}
                for mname, col in METRICS.items():
                    if col not in dc.columns:
                        continue
                    row = dict(subset=sub, cell=cell, scenario=scen, metric=mname)
                    for v in piv:
                        x = piv[v][col] if col in piv[v] else pd.Series(dtype=float)
                        row[f"median_{v}"] = float(np.nanmedian(x)) if len(x) else np.nan
                        row["n_regions"] = len(x)
                    med_rows.append(row)
                    for v in ("imp", "shuf"):
                        if col not in piv[v] or col not in piv["absent"]:
                            continue
                        dd = (piv[v][col] - piv["absent"][col]).dropna()
                        m, lo, hi = boot_ci(dd)
                        diff_rows.append(dict(subset=sub, cell=cell, scenario=scen, metric=mname,
                                              contrast=f"{v}_minus_absent", n=len(dd),
                                              median_diff=m, ci_lo=lo, ci_hi=hi,
                                              mean_diff=float(dd.mean()) if len(dd) else np.nan,
                                              n_improved=int((dd > 0).sum()),
                                              frac_improved=float((dd > 0).mean()) if len(dd) else np.nan))
                    if "imp" in piv and col in piv["imp"] and col in piv["shuf"]:
                        dd = (piv["imp"][col] - piv["shuf"][col]).dropna()
                        m, lo, hi = boot_ci(dd)
                        diff_rows.append(dict(subset=sub, cell=cell, scenario=scen, metric=mname,
                                              contrast="imp_minus_shuf", n=len(dd), median_diff=m,
                                              ci_lo=lo, ci_hi=hi, mean_diff=float(dd.mean()),
                                              n_improved=int((dd > 0).sum()),
                                              frac_improved=float((dd > 0).mean())))
            # consensus alone on the same regions
            cr = cons[cons.cell == cell].set_index("region")["consensus"]
            keep = dc.region.unique()
            cr = cr[cr.index.isin(keep)]
            med_rows.append(dict(subset=sub, cell=cell, scenario="consensus_alone", metric="pearson",
                                 median_absent=float(np.nanmedian(cr)), n_regions=len(cr)))
    med = pd.DataFrame(med_rows)
    dif = pd.DataFrame(diff_rows)
    med.to_csv(RES_DIR / "scenario_medians.csv", index=False, float_format="%.4f")
    dif.to_csv(RES_DIR / "paired_differences.csv", index=False, float_format="%.4f")
    d.drop(columns=["set"]).to_csv(RES_DIR / "per_region_scores.csv.gz", index=False, float_format="%.5f")
    make_table_and_figure(med, dif)
    return med, dif


def make_table_and_figure(med, dif):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = list(dict.fromkeys(med.scenario))
    p = med[(med.metric == "pearson") & (med.scenario != "consensus_alone")]
    tab = p.pivot_table(index=["subset", "cell", "scenario"],
                        values=["median_absent", "median_imp", "median_shuf"]).reset_index()
    tab.to_csv(RES_DIR / "scenario_table_pearson.csv", index=False, float_format="%.4f")
    scen = [s for s in order if s != "consensus_alone"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, cell in zip(axes, UNSEEN):
        t = tab[(tab.subset == "all413") & (tab.cell == cell)].set_index("scenario").reindex(scen)
        x = np.arange(len(scen))
        for j, (col, lab, c) in enumerate([("median_absent", "absent", "#999999"),
                                           ("median_imp", "CNN-imputed", "#1f77b4"),
                                           ("median_shuf", "shuffled (1 Mb)", "#ff7f0e")]):
            ax.bar(x + (j - 1) * 0.27, t[col], 0.27, label=lab, color=c)
        cr = med[(med.subset == "all413") & (med.cell == cell) & (med.scenario == "consensus_alone")]
        if len(cr):
            ax.axhline(cr.median_absent.iloc[0], color="k", ls="--", lw=1, label="consensus alone")
        ax.set_xticks(x)
        ax.set_xticklabels(scen, rotation=45, ha="right")
        ax.set_title(cell.replace("_deep", ""))
        ax.set_ylim(max(0, np.nanmin(tab.median_absent) - 0.05), None)
    axes[0].set_ylabel("median per-region Pearson (413 regions)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RES_DIR / "twostage_cnn.png", dpi=150)


def main():
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["cnn", "distribution", "predict", "summary"])
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--no-structure", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    MODEL = a.model
    if "cnn" in a.stage:
        stage_cnn(a.cells, a.gpu, a.force)
    if "distribution" in a.stage:
        print(stage_distribution(a.cells).groupby(["cell", "clean_chroms"])[
            ["mean_measured", "mean_imputed", "sd_measured", "sd_imputed", "pearson"]].mean())
    if "predict" in a.stage:
        stage_predict(a.model, a.cells, a.gpu, not a.no_structure)
    if "summary" in a.stage:
        stage_summary(a.model)


if __name__ == "__main__":
    main()
