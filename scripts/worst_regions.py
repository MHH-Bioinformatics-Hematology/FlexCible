"""Worst-predicted Akita test regions: which regions score low, and why.

Stages (run in order; each writes under --out, default
runs/worst_regions):

  predict   CPU re-prediction of the 413 regions with
            scripts/crosscell_predict.py --save-pred (no GPU): the 12-cell
            residual reference model at 25 kb (all12refres_25kb) for every
            cell and the 12-cell 2 kb model (all12) for GM12878 and K562. The
            saved maps feed the metric and figure stages; the per-region
            scores reproduce the committed crosscell tables.
  collect   one long table of every per-region score (models, measured maps
            of other cells, consensus) from the existing crosscell CSVs.
  reps      per-region replicate agreement (rep1 vs rep2 proper O/E, single
            lane in situ and deep intact pairs), 2 and 25 kb.
  features  per region and cell: target dynamic range, masked and zero
            pixels, raw coverage, CNV proxy (coverage deviation), distance to
            centromere/telomere/assembly gap, segmental duplication share,
            gene density, CTCF motif density, ChIP signal (H3K9me3, H3K27ac,
            CTCF, ...), A/B compartment, window position of the region.
  model     regression and classification of the score on the features.
  metrics   alternative metrics on the saved predictions (Spearman,
            stratum-adjusted correlation, masking rules, excluded diagonals).
  fixes     blend prediction with the consensus; evaluation rules and the
            target-side / model-side split against independent replicates.
  chipref   reference map from the other cells weighted by ChIP similarity.
  tables    bottom-10% tables; small tables copied to results/worst_regions.
  figures   per-region panels (target, prediction, consensus, replicate or
            other-cell map, ChIP tracks, per-distance Pearson) and a summary.

    python scripts/worst_regions.py --stage predict collect reps features model \
        metrics fixes chipref tables figures
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = HM / "crosscell_oe"
RUNS = HM / "runs"
OUT = RUNS / "worst_regions"
REF25 = RUNS / "multicell_reference/all12refres_25kb"
ALL12 = {2: RUNS / "multicell_expanded/all12", 5: RUNS / "multicell_expanded/all12_5kb",
         10: RUNS / "multicell_expanded/all12_10kb", 25: RUNS / "multicell_expanded/all12_25kb"}
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
           "gm23248", "mcf7", "cd14mono", "pc9"]
HELDOUT = ["k562", "hmec_deep", "huvec_deep"]
CANCER = {"k562", "hct116", "hepg2", "a549", "mcf7", "pc9"}


def sh(cmd, log, env):
    with open(log, "w") as fh:
        return subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=REPO).returncode


# --------------------------------------------------------------------- predict
def stage_predict(a):
    """CPU predictions with saved maps. Same arguments as
    multicell_expanded.crosscell, without the measured-map baselines (those
    are in the committed tables)."""
    import multicell_expanded as mx
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads))
    jobs = []
    for cell in TRAIN12 + HELDOUT:
        jobs.append(("ref25", REF25, RUNS / "multicell_reference/all12refres_25kb.yaml",
                     25000, cell,
                     ["--reference", mx.measured_reference("all12refres_25kb", cell)]))
    for cell in a.cells_2kb:
        jobs.append(("all12_2kb", ALL12[2], RUNS / "multicell_expanded/all12.yaml",
                     2000, cell, []))
    if a.pred_cells:
        jobs = [j for j in jobs if f"{j[0]}:{j[4]}" in a.pred_cells]
    for tag, run, cfg, res, cell, extra in jobs:
        od = a.out / f"pred_{tag}"
        od.mkdir(parents=True, exist_ok=True)
        npz = od / f"pred_{cell}.npz"
        if npz.exists():
            continue
        _, bw, assay, _ = mx.CELLS[cell]
        ckpt = run / (run / "best").read_text().strip()
        cmd = [sys.executable, "scripts/crosscell_predict.py", "--config", cfg,
               "--checkpoint", ckpt, "--cell", cell, "--bigwig-dir", bw,
               "--target", mx.target(cell, res), "--tta", "--assay", assay,
               "--out-dir", od, "--binsize", res, "--device", "cpu",
               "--save-pred", npz, *extra]
        print(f"[predict] {tag} {cell}", flush=True)
        rc = sh(cmd, od / f"predict_{cell}.log", env)
        if rc:
            print(f"[FAILED] {tag} {cell} rc={rc}", flush=True)


# ------------------------------------------------------------------ helpers
def regions_df():
    import score_like_hicgan as sc
    r = pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    r["region"] = r.chrom + ":" + r.start.astype(str) + "-" + r.end.astype(str)
    return r


def parse_region(s):
    c, se = s.split(":")
    a, b = se.split("-")
    return c, int(a), int(b)


def fetch(clr, region, balance=False):
    from flexcible.genome import match_name
    c, s, e = parse_region(region)
    nm = match_name(c, list(clr.chromnames))
    return np.asarray(clr.matrix(balance=balance).fetch(f"{nm}:{s}-{e}"), float)


def source_cool(cell, res):
    """Balanced count map behind a cell's proper O/E target (for weights,
    raw coverage and the CNV proxy)."""
    import cooler
    from encode_expand import LOCAL
    if cell in LOCAL:
        return cooler.Cooler(f"{LOCAL[cell]}::/resolutions/{res}")
    p = HM / "encode_hic" / f"{cell}_{res // 1000}kb.cool"
    return cooler.Cooler(str(p)) if p.exists() else None


def target_cool(cell, res):
    import cooler
    import multicell_expanded as mx
    return cooler.Cooler(str(mx.target(cell, res)))


def reference_cool(cell, res):
    import cooler
    import multicell_expanded as mx
    return cooler.Cooler(str(mx.measured_reference(f"all12refres_{res // 1000}kb", cell)))


def rep_cool(label, res):
    """label: gm12878_rep1 (single lane in situ) or gm12878_ENCSR965OVH_r75
    (deep intact); proper O/E of the ICE-balanced replicate."""
    import cooler
    k = res // 1000
    p = OE / (f"{label}_ice_{k}kb.cool" if "_ENCSR" not in label
              else f"intact_{label}_ice_{k}kb.cool")
    return cooler.Cooler(str(p)) if p.exists() else None


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20 or x[ok].std() == 0 or y[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def pair_r(A, B, k=2):
    """The measured-map baseline metric of crosscell_predict: upper
    triangle offset >= k, pixels nonzero in both maps."""
    if A.shape != B.shape:
        return np.nan
    iu = np.triu_indices(A.shape[0], k)
    x, y = A[iu], B[iu]
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    return pearson(np.where(ok, x, np.nan), np.where(ok, y, np.nan))


# ------------------------------------------------------------------ collect
def _read_run(d: Path, tag: str, res: int):
    rows = []
    for f in sorted(d.glob("predict_*.csv")):
        cell = f.stem[len("predict_"):]
        df = pd.read_csv(f)
        long = df.melt(id_vars="region", var_name="column", value_name="r").dropna()
        long["cell"], long["source"], long["res_kb"] = cell, tag, res
        rows.append(long)
    return rows


def stage_collect(a):
    parts = []
    parts += _read_run(REF25 / "crosscell", "all12refres_25kb", 25)
    for res, d in ALL12.items():
        parts += _read_run(d / "crosscell", f"all12_{res}kb", res)
    for d in sorted((RUNS / "coarse_gap").glob("*kb_w*__*")):
        if (d / "crosscell_test").is_dir():
            parts += _read_run(d / "crosscell_test", f"gm_{d.name}", int(d.name.split("kb")[0]))
    for d in sorted((RUNS / "resolution_inputs").glob("*kb_w*__all__s*")):
        if (d / "crosscell_test").is_dir():
            parts += _read_run(d / "crosscell_test", f"gm_{d.name}", int(d.name.split("kb")[0]))
    long = pd.concat(parts, ignore_index=True)
    long = long[~long.cell.isin(["hmec", "huvec"])]         # single-lane duplicates
    long.to_csv(a.out / "scores_long.csv.gz", index=False)

    # wide per-region table for the current best model
    ref = long[long.source == "all12refres_25kb"]
    mod = ref[ref.column == "model"].pivot(index="region", columns="cell", values="r")
    cons = ref[ref.column.isin(["base_consensus_loo", "base_consensus12"])].pivot(
        index="region", columns="cell", values="r")
    oth = (ref[ref.column.str.startswith("base_") & ~ref.column.str.contains("consensus")]
           .groupby(["region", "cell"]).r.max().unstack())
    wide = pd.DataFrame(index=mod.index)
    for c in mod.columns:
        wide[f"model_{c}"] = mod[c]
        wide[f"cons_{c}"] = cons.get(c)
        wide[f"bestother_{c}"] = oth.get(c)
    tr = [c for c in TRAIN12 if c in mod.columns]
    wide["model_med12"] = mod[tr].median(1)
    wide["model_min12"] = mod[tr].min(1)
    wide["cons_med12"] = cons[tr].median(1)
    wide["bestother_med12"] = oth[tr].median(1)
    wide["model_med_heldout"] = mod[[c for c in HELDOUT if c in mod]].median(1)
    for res in (2, 5, 10, 25):
        s = long[(long.source == f"all12_{res}kb") & (long.column == "model")
                 & long.cell.isin(tr)].pivot(index="region", columns="cell", values="r")
        wide[f"all12_{res}kb_med12"] = s.median(1)
        b = long[(long.source == f"all12_{res}kb") & long.column.str.startswith("base_")
                 & long.cell.isin(tr)].groupby(["region", "cell"]).r.max().unstack()
        wide[f"bestother_{res}kb_med12"] = b.median(1)
    wide["rank_med12"] = wide.model_med12.rank()
    wide = wide.sort_values("model_med12")
    wide.to_csv(a.out / "region_scores.csv")

    # consistency of the bad set
    n = len(wide)
    q = max(1, int(round(0.1 * n)))
    bad = lambda s: set(s.nsmallest(q).index)                       # noqa: E731
    rows = []
    cells = tr
    rho = mod[cells].corr(method="spearman").values
    rows.append({"comparison": "cell vs cell (ref25 model), mean pairwise Spearman",
                 "value": float(rho[np.triu_indices(len(cells), 1)].mean())})
    jac = []
    for i, c1 in enumerate(cells):
        for c2 in cells[i + 1:]:
            b1, b2 = bad(mod[c1]), bad(mod[c2])
            jac.append(len(b1 & b2) / q)
    rows.append({"comparison": "cell vs cell, share of bottom-10% shared (chance 0.10)",
                 "value": float(np.mean(jac))})
    for x, y in [("model_med12", "cons_med12"), ("model_med12", "bestother_med12"),
                 ("model_med12", "all12_25kb_med12"), ("model_med12", "all12_2kb_med12"),
                 ("all12_2kb_med12", "all12_25kb_med12"), ("model_med12", "model_med_heldout"),
                 ("all12_2kb_med12", "bestother_2kb_med12")]:
        rows.append({"comparison": f"{x} vs {y}, Spearman",
                     "value": float(wide[[x, y]].corr(method="spearman").iloc[0, 1])})
        rows.append({"comparison": f"{x} vs {y}, share of bottom-10% shared",
                     "value": len(bad(wide[x]) & bad(wide[y])) / q})
    pd.DataFrame(rows).to_csv(a.out / "consistency.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(wide[["model_med12", "cons_med12", "bestother_med12", "all12_25kb_med12",
                "all12_2kb_med12"]].head(15).round(3).to_string())


# --------------------------------------------------------------------- reps
REP_SETS = {
    # label: (cell of the target, replicate A, replicate B)
    "gm12878_insitu_lane": ("gm12878", "gm12878_rep1", "gm12878_rep2"),
    "imr90_insitu_lane": ("imr90", "imr90_rep1", "imr90_rep2"),
    "k562_insitu_lane": ("k562", "k562_rep1", "k562_rep2"),
    "hmec_insitu_lane": ("hmec_deep", "hmec_rep1", "hmec_rep2"),
    "huvec_insitu_lane": ("huvec_deep", "huvec_rep1", "huvec_rep2"),
    "gm12878_intact_965OVH": ("gm12878", "gm12878_ENCSR965OVH_r75", "gm12878_ENCSR965OVH_r76"),
    "gm12878_intact_653JLK": ("gm12878", "gm12878_ENCSR653JLK_r7", "gm12878_ENCSR653JLK_r9"),
    "gm12878_intact_946AKI": ("gm12878", "gm12878_ENCSR946AKI_r8", "gm12878_ENCSR946AKI_r12"),
    "k562_intact_676BEP": ("k562", "k562_ENCSR676BEP_r5", "k562_ENCSR676BEP_r7"),
    "k562_intact_178ENS": ("k562", "k562_ENCSR178ENS_r2", "k562_ENCSR178ENS_r3"),
}


def stage_reps(a):
    """Replicate A vs B, and each replicate vs the target map, per region.
    The intact experiments are independent of the in situ targets of
    GM12878 and K562, so 'repA_vs_target' there is an independent
    measurement scored like a prediction."""
    reg = regions_df()
    rows = []
    for res in (25000, 2000):
        for lab, (cell, ra, rb) in REP_SETS.items():
            A, B, T = rep_cool(ra, res), rep_cool(rb, res), target_cool(cell, res)
            if A is None or B is None:
                continue
            for region in reg.region:
                try:
                    MA, MB, MT = fetch(A, region), fetch(B, region), fetch(T, region)
                except Exception:
                    continue
                if MA.shape[0] < 20:
                    continue
                rows.append({"region": region, "set": lab, "cell": cell, "res_kb": res // 1000,
                             "repA_vs_repB": pair_r(MA, MB), "repA_vs_target": pair_r(MA, MT),
                             "repB_vs_target": pair_r(MB, MT)})
            print(f"[reps] {lab} {res}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "replicates_per_region.csv", index=False)
    print(df.groupby(["set", "res_kb"])[["repA_vs_repB", "repA_vs_target"]].median().round(3))


# ----------------------------------------------------------------- features
ANN = Path.home() / "data/hic-multiome/annotation/hg38"
CHIP_FEAT = ["CTCF", "RAD21", "H3K27ac", "H3K4me3", "H3K36me3", "H3K27me3", "H3K9me3"]


def _intervals(path, cols, filt=None):
    df = pd.read_csv(path, sep="\t", header=None, compression="infer", usecols=cols)
    df.columns = ["chrom", "start", "end"] + [f"x{i}" for i in range(len(cols) - 3)]
    if filt is not None:
        df = df[filt(df)]
    return df


def _merge(iv):
    out = {}
    for c, g in iv.groupby("chrom"):
        g = g.sort_values("start")
        s, e, m = g.start.values, g.end.values, []
        cs, ce = s[0], e[0]
        for a_, b_ in zip(s[1:], e[1:]):
            if a_ <= ce:
                ce = max(ce, b_)
            else:
                m.append((cs, ce)); cs, ce = a_, b_
        m.append((cs, ce))
        out[c] = np.array(m)
    return out


def _overlap_bp(merged, c, s, e):
    m = merged.get(c)
    if m is None or not len(m):
        return 0
    o = np.clip(np.minimum(m[:, 1], e) - np.maximum(m[:, 0], s), 0, None)
    return int(o.sum())


def genomic_features(reg):
    """Region-level features independent of the cell."""
    import pyBigWig
    fai = pd.read_csv(Path.home() / "data/hic-multiome/genome/hg38.fa.fai", sep="\t",
                      header=None, usecols=[0, 1], names=["chrom", "len"])
    clen = dict(zip(fai.chrom, fai.len))
    cen = _intervals(ANN / "centromeres.txt.gz", [1, 2, 3]).groupby("chrom").agg(
        start=("start", "min"), end=("end", "max"))
    gap = _intervals(ANN / "gap.txt.gz", [1, 2, 3, 7])
    gap = gap[gap.x0 != "telomere"]
    bigap = gap[(gap.end - gap.start) >= 50_000]
    sd = _merge(_intervals(ANN / "genomicSuperDups.txt.gz", [1, 2, 3]))
    bl = _merge(_intervals(ANN / "hg38-blacklist.v2.bed.gz", [0, 1, 2]))
    kg = pd.read_csv(ANN / "knownGene.txt.gz", sep="\t", header=None, usecols=[1, 2, 3, 4])
    kg.columns = ["chrom", "strand", "txStart", "txEnd"]
    kg["tss"] = np.where(kg.strand == "+", kg.txStart, kg.txEnd)
    tss = {c: np.unique(g.tss.values // 1000) * 1000 for c, g in kg.groupby("chrom")}
    ann = {k: pyBigWig.open(str(ANN / "bw_2kb" / f"ANN_{k}.bigwig"))
           for k in ("GeneBody", "Satellite", "LINE", "SINE", "CGI")}
    rows = []
    for _, r in reg.iterrows():
        c, s, e = r.chrom, int(r.start), int(r.end)
        rec = {"region": r.region, "chrom": c}
        if c in cen.index:
            cs, ce = cen.loc[c, "start"], cen.loc[c, "end"]
            rec["dist_centromere_mb"] = (0 if s < ce and e > cs else
                                         min(abs(s - ce), abs(cs - e))) / 1e6
        rec["dist_telomere_mb"] = min(s, clen[c] - e) / 1e6
        g = bigap[bigap.chrom == c]
        if len(g):
            d = np.maximum(0, np.maximum(g.start.values - e, s - g.end.values))
            rec["dist_gap_mb"] = float(d.min()) / 1e6
        else:
            rec["dist_gap_mb"] = 50.0
        rec["gap_bp_1mb_flank"] = sum(max(0, min(ge, e + 1_000_000) - max(gs, s - 1_000_000))
                                      for gs, ge in zip(gap[gap.chrom == c].start,
                                                        gap[gap.chrom == c].end))
        rec["segdup_frac"] = _overlap_bp(sd, c, s, e) / (e - s)
        rec["blacklist_frac"] = _overlap_bp(bl, c, s, e) / (e - s)
        t = tss.get(c, np.array([]))
        rec["tss_per_mb"] = int(((t >= s) & (t < e)).sum()) / ((e - s) / 1e6)
        for k, bw in ann.items():
            v = bw.stats(c, s, e, type="mean")[0]
            rec[f"ann_{k.lower()}"] = float(v) if v is not None else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    # CTCF motif sites per Mb (cached genome scan, MA0139.2, 25 kb bins)
    cache = Path.home() / "data/hic-multiome/genome_cache"
    motif = {}
    for f in cache.glob("motif_*_MA0139.2_25000_f0.80.npz"):
        ch = f.name.split("_")[2]
        motif["chr" + ch.replace("chr", "")] = np.load(f)["tracks"].sum(0)
    dens = []
    for _, r in reg.iterrows():
        m = motif.get(r.chrom)
        if m is None:
            dens.append(np.nan)
            continue
        dens.append(float(m[r.start // 25000:-(-r.end // 25000)].sum()) / ((r.end - r.start) / 1e6))
    df["ctcf_motif_per_mb"] = dens
    return df


def compartments(cell, reg):
    """Cis E1 at 250 kb (cooltools, phased by TSS density) for cells whose
    source mcool stores that level; per region: mean E1, share of A bins and
    the number of sign changes (compartment switches)."""
    import bioframe
    import cooltools
    from encode_expand import LOCAL
    if cell not in LOCAL:
        return None
    import cooler
    clr = cooler.Cooler(f"{LOCAL[cell]}::/resolutions/250000")
    bins = clr.bins()[:][["chrom", "start", "end"]]
    kg = pd.read_csv(ANN / "knownGene.txt.gz", sep="\t", header=None, usecols=[1, 2, 3, 4])
    kg.columns = ["chrom", "strand", "txStart", "txEnd"]
    kg["tss"] = np.where(kg.strand == "+", kg.txStart, kg.txEnd)
    cnt = kg.assign(b=kg.tss // 250000).groupby(["chrom", "b"]).size()
    bins["tss"] = [cnt.get((c, s // 250000), 0) for c, s in zip(bins.chrom, bins.start)]
    chroms = [c for c in clr.chromnames if c.replace("chr", "").isdigit()]
    view = bioframe.make_viewframe([(c, 0, clr.chromsizes[c]) for c in chroms])
    bins = bins[bins.chrom.isin(chroms)]
    _, eig = cooltools.eigs_cis(clr, phasing_track=bins[["chrom", "start", "end", "tss"]],
                                view_df=view, n_eigs=1)
    out = []
    for _, r in reg.iterrows():
        e = eig[(eig.chrom == r.chrom) & (eig.end > r.start) & (eig.start < r.end)].E1.values
        e = e[np.isfinite(e)]
        out.append({"region": r.region, "cell": cell,
                    "e1_mean": float(e.mean()) if len(e) else np.nan,
                    "a_frac": float((e > 0).mean()) if len(e) else np.nan,
                    "e1_switches": int((np.diff(np.sign(e)) != 0).sum()) if len(e) else np.nan})
    return pd.DataFrame(out)


def cell_features(cell, reg, res=25000):
    import pyBigWig
    import multicell_expanded as mx
    T = target_cool(cell, res)
    src = source_cool(cell, res)
    bwdir = mx.CELLS[cell][1]
    missing = mx.missing_marks(bwdir)
    bws, gmean = {}, {}
    for t in CHIP_FEAT:
        p = bwdir / f"{t}.bigwig"
        if t not in missing and p.exists():
            bws[t] = pyBigWig.open(str(p))
            h = bws[t].header()
            gmean[t] = h["sumData"] / max(h["nBasesCovered"], 1)
    w_all = src.bins()[:] if src is not None else None
    if w_all is not None:
        inv = 1.0 / w_all["weight"].values
        med = np.nanmedian(inv)
        w_all["log2cov"] = np.log2(inv / med)
    size = mx.WINDOW[res][0]
    rows = []
    for _, r in reg.iterrows():
        rec = {"region": r.region, "cell": cell, "n_missing_tracks": len(missing)}
        M = fetch(T, r.region)
        n = M.shape[0]
        iu = np.triu_indices(n, 2)
        v = M[iu]
        nz = v[v != 0]
        rec["tgt_sd"] = float(nz.std()) if len(nz) else np.nan
        rec["tgt_iqr"] = float(np.subtract(*np.percentile(nz, [75, 25]))) if len(nz) else np.nan
        rec["tgt_zero_frac"] = float((v == 0).mean())
        rec["tgt_clip_frac"] = float((np.abs(nz) > 1.9).mean()) if len(nz) else np.nan
        if w_all is not None:
            c = r.chrom
            sel = w_all[(w_all.chrom == c) & (w_all.end > r.start) & (w_all.start < r.end)]
            rec["unmap_frac"] = float(sel.weight.isna().mean())
            lc = sel.log2cov.values
            rec["log2cov_mean"] = float(np.nanmean(lc)) if np.isfinite(lc).any() else np.nan
            sm = pd.Series(lc).rolling(5, center=True, min_periods=3).median().values
            rec["log2cov_step"] = float(np.nanmax(np.abs(np.diff(sm)))) if np.isfinite(sm).sum() > 2 else np.nan
            rec["log2cov_sd"] = float(np.nanstd(lc))
            try:
                R = fetch(src, r.region, balance=False)
                rec["raw_per_px"] = float(np.nanmean(R[np.triu_indices(R.shape[0], 2)]))
            except Exception:
                rec["raw_per_px"] = np.nan
        # position of the region inside the model window (crosscell_predict)
        nb = int(T.chromsizes[r.chrom] // res) + 1
        centre = (int(r.start) + int(r.end)) // 2 // res
        st = max(0, min(nb - size, centre - size // 2))
        rs, re_ = int(r.start) // res, -(-int(r.end) // res)
        rec["window_margin_bins"] = min(rs - st, st + size - re_)
        for t in CHIP_FEAT:
            if t in bws:
                val = bws[t].stats(r.chrom, int(r.start), int(r.end), type="mean")[0]
                rec[f"chip_{t}"] = float(np.log2(((val or 0) + 1e-3 * gmean[t]) / gmean[t]))
                if t == "CTCF":
                    mx_ = np.array([x or 0 for x in bws[t].stats(
                        r.chrom, int(r.start), int(r.end), type="max",
                        nBins=(int(r.end) - int(r.start)) // 2000)])
                    rec["ctcf_peaks_per_mb"] = float((mx_ > 10 * gmean[t]).sum()) / (
                        (int(r.end) - int(r.start)) / 1e6)
            else:
                rec[f"chip_{t}"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def cell_variability(reg, res=25000):
    """Pixel-wise spread of the 12 training cells' maps per region: the share
    of the pixel variance that lies between cells (cell specificity), and
    the mean target correlation of a cell with the others."""
    Ts = {c: target_cool(c, res) for c in TRAIN12}
    rows = []
    for region in reg.region:
        mats = []
        for c in TRAIN12:
            M = fetch(Ts[c], region)
            mats.append(np.where(M == 0, np.nan, M)[np.triu_indices(M.shape[0], 2)])
        X = np.vstack(mats)
        within = np.nanmean(np.nanvar(X, axis=0))
        total = np.nanvar(X)
        rows.append({"region": region, "between_cell_var_share": float(within / total),
                     "mean_pixel_sd_across_cells": float(np.nanmean(np.nanstd(X, axis=0)))})
    return pd.DataFrame(rows)


def stage_features(a):
    reg = regions_df()
    g = genomic_features(reg)
    g = g.merge(cell_variability(reg), on="region")
    g.to_csv(a.out / "features_region.csv", index=False)
    print("[features] genomic done", flush=True)
    cf, comp = [], []
    for cell in TRAIN12 + HELDOUT:
        cf.append(cell_features(cell, reg))
        cp = compartments(cell, reg)
        if cp is not None:
            comp.append(cp)
        print(f"[features] {cell}", flush=True)
    cf = pd.concat(cf)
    comp = pd.concat(comp)
    cf = cf.merge(comp, on=["region", "cell"], how="left")
    cf.to_csv(a.out / "features_cell.csv", index=False)


# -------------------------------------------------------------------- model
FEATS_REGION = ["dist_centromere_mb", "dist_telomere_mb", "dist_gap_mb", "gap_bp_1mb_flank",
                "segdup_frac", "blacklist_frac", "tss_per_mb", "ann_genebody", "ann_satellite",
                "ann_line", "ann_sine", "ann_cgi", "ctcf_motif_per_mb",
                "between_cell_var_share", "mean_pixel_sd_across_cells"]
FEATS_CELL = ["tgt_sd", "tgt_iqr", "tgt_zero_frac", "tgt_clip_frac", "unmap_frac",
              "log2cov_mean", "log2cov_step", "log2cov_sd", "raw_per_px", "window_margin_bins",
              "chip_CTCF", "ctcf_peaks_per_mb", "chip_RAD21", "chip_H3K27ac", "chip_H3K4me3",
              "chip_H3K36me3", "chip_H3K27me3", "chip_H3K9me3", "n_missing_tracks",
              "e1_mean", "a_frac", "e1_switches"]
GROUPS = {
    "target map (dynamic range, masking, coverage, CNV proxy)":
        ["tgt_sd", "tgt_iqr", "tgt_zero_frac", "tgt_clip_frac", "unmap_frac", "log2cov_mean",
         "log2cov_step", "log2cov_sd", "raw_per_px"],
    "cell specificity (spread of the 12 maps)": ["between_cell_var_share",
                                                 "mean_pixel_sd_across_cells"],
    "genome (centromere, gaps, segdups, repeats, genes, motifs)":
        ["dist_centromere_mb", "dist_telomere_mb", "dist_gap_mb", "gap_bp_1mb_flank",
         "segdup_frac", "blacklist_frac", "tss_per_mb", "ann_genebody", "ann_satellite",
         "ann_line", "ann_sine", "ann_cgi", "ctcf_motif_per_mb"],
    "ChIP signal": ["chip_CTCF", "ctcf_peaks_per_mb", "chip_RAD21", "chip_H3K27ac",
                    "chip_H3K4me3", "chip_H3K36me3", "chip_H3K27me3", "chip_H3K9me3",
                    "n_missing_tracks"],
    "compartment": ["e1_mean", "a_frac", "e1_switches"],
    "window position": ["window_margin_bins"],
}


def _design(a):
    wide = pd.read_csv(a.out / "region_scores.csv")
    fr = pd.read_csv(a.out / "features_region.csv")
    fc = pd.read_csv(a.out / "features_cell.csv")
    long = pd.read_csv(a.out / "scores_long.csv.gz")
    ref = long[(long.source == "all12refres_25kb")]
    m = ref[ref.column == "model"][["region", "cell", "r"]].rename(columns={"r": "model"})
    c = ref[ref.column.str.startswith("base_consensus")][["region", "cell", "r"]].rename(
        columns={"r": "consensus"})
    o = (ref[ref.column.str.startswith("base_") & ~ref.column.str.contains("consensus")]
         .groupby(["region", "cell"]).r.max().rename("best_other").reset_index())
    d = m.merge(c, on=["region", "cell"]).merge(o, on=["region", "cell"])
    d = d.merge(fc, on=["region", "cell"], how="left").merge(fr, on="region", how="left")
    d["cancer"] = d.cell.isin(CANCER).astype(int)
    return wide, d


def stage_model(a):
    """Which features explain the score. Region level: the median over the
    12 training cells of the reference model's score (and of the consensus),
    against region features (cell features as medians over the cells).
    Pooled level: every region x cell, score centred per cell."""
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    wide, d = _design(a)
    tr = d[d.cell.isin(TRAIN12)]
    reg = tr.groupby("region")[FEATS_CELL + ["model", "consensus", "best_other"]].median()
    reg = reg.join(tr.groupby("region")[FEATS_REGION].first())
    reg["model_minus_cons"] = reg.model - reg.consensus
    reg = reg.join(wide.set_index("region")[["all12_2kb_med12", "all12_25kb_med12"]])
    feats = [f for f in FEATS_CELL + FEATS_REGION if reg[f].notna().mean() > 0.5]
    out_rows = []
    for resp in ("model", "consensus", "model_minus_cons", "all12_2kb_med12",
                 "all12_25kb_med12"):
        y = reg[resp].values
        bad = (reg[resp] <= reg[resp].quantile(0.1)).astype(int).values
        for f in feats:
            x = reg[f].values
            ok = np.isfinite(x) & np.isfinite(y)
            rho = spearmanr(x[ok], y[ok]).statistic
            try:
                auc = roc_auc_score(bad[ok], -x[ok])
            except ValueError:
                auc = np.nan
            out_rows.append({"level": "region", "response": resp, "feature": f,
                             "spearman": rho, "auc_bottom10_low_feature": auc,
                             "median_bottom10": float(np.nanmedian(x[bad == 1])),
                             "median_rest": float(np.nanmedian(x[bad == 0]))})
    uni = pd.DataFrame(out_rows)

    def fit(X, y, groups=None, label=""):
        cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)
        rf = make_pipeline(SimpleImputer(strategy="median"),
                           RandomForestRegressor(400, min_samples_leaf=5, random_state=0,
                                                 n_jobs=a.threads))
        lin = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            RidgeCV(alphas=np.logspace(-2, 3, 20)))
        res = {}
        for nm, mdl in (("rf", rf), ("ridge", lin)):
            p = cross_val_predict(mdl, X, y, cv=cv, groups=groups)
            res[f"{nm}_cv_r2"] = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        rf.fit(X, y)
        pi = permutation_importance(rf, X, y, n_repeats=10, random_state=0, n_jobs=a.threads)
        imp = pd.DataFrame({"feature": X.columns, "perm_importance": pi.importances_mean,
                            "fit": label})
        # grouped drop-column CV R2 (ridge and RF)
        grp = []
        for gname, cols in GROUPS.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue
            Xg = X.drop(columns=cols)
            p = cross_val_predict(rf, Xg, y, cv=cv, groups=groups)
            r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
            p1 = cross_val_predict(rf, X[cols], y, cv=cv, groups=groups)
            r21 = 1 - np.sum((y - p1) ** 2) / np.sum((y - y.mean()) ** 2)
            grp.append({"fit": label, "group": gname, "rf_cv_r2_without": r2,
                        "rf_cv_r2_drop": res["rf_cv_r2"] - r2, "rf_cv_r2_alone": r21})
        return res, imp, pd.DataFrame(grp)

    fits, imps, grps = [], [], []
    for resp in ("model", "consensus", "model_minus_cons", "all12_2kb_med12",
                 "all12_25kb_med12"):
        X = reg[feats]
        r, i, g = fit(X, reg[resp].values, label=f"region_{resp}")
        fits.append({"fit": f"region_{resp}", "n": len(X), **r})
        imps.append(i); grps.append(g)
    # pooled region x cell, score centred per cell; regions grouped in CV
    pc = tr.copy()
    pc["model_c"] = pc.model - pc.groupby("cell").model.transform("median")
    pf = [f for f in FEATS_CELL + FEATS_REGION + ["cancer"] if pc[f].notna().mean() > 0.3]
    r, i, g = fit(pc[pf], pc.model_c.values, groups=pc.region.values, label="pooled_model")
    fits.append({"fit": "pooled_model", "n": len(pc), **r})
    imps.append(i); grps.append(g)
    # same with the consensus score as a feature (how much is 'reference fails')
    pf2 = pf + ["consensus"]
    r, i, g = fit(pc[pf2], pc.model_c.values, groups=pc.region.values,
                  label="pooled_model_plus_consensus")
    fits.append({"fit": "pooled_model_plus_consensus", "n": len(pc), **r})
    imps.append(i); grps.append(g)
    fits = pd.DataFrame(fits)
    imps = pd.concat(imps).sort_values(["fit", "perm_importance"], ascending=[True, False])
    grps = pd.concat(grps)
    uni.to_csv(a.out / "feature_univariate.csv", index=False)
    fits.to_csv(a.out / "feature_fits.csv", index=False)
    imps.to_csv(a.out / "feature_importance.csv", index=False)
    grps.to_csv(a.out / "feature_groups.csv", index=False)
    reg.to_csv(a.out / "region_design.csv")
    print(fits.round(3).to_string(index=False))
    print(grps.round(3).to_string(index=False))
    for f in fits.fit:
        print(f, imps[imps.fit == f].head(8)[["feature", "perm_importance"]].round(4).to_string(index=False))
    u = uni[uni.response == "model"].reindex(
        uni[uni.response == "model"].spearman.abs().sort_values(ascending=False).index)
    print(u.head(15).round(3).to_string(index=False))


# ------------------------------------------------------------------ metrics
def per_diag_r(P, T, kmin=2):
    n = P.shape[0]
    out = np.full(n, np.nan)
    for k in range(kmin, n - 4):
        x, y = np.diagonal(P, k), np.diagonal(T, k)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 5 and x[ok].std() > 0 and y[ok].std() > 0:
            out[k] = np.corrcoef(x[ok], y[ok])[0, 1]
    return out


def scc(P, T, kmin=2):
    """Stratum-adjusted correlation (HiCRep weights: diagonal length times
    the product of the rank standard deviations), no extra smoothing."""
    from scipy.stats import rankdata
    n = P.shape[0]
    num = den = 0.0
    for k in range(kmin, n - 4):
        x, y = np.diagonal(P, k), np.diagonal(T, k)
        ok = np.isfinite(x) & np.isfinite(y)
        m = int(ok.sum())
        if m < 5 or x[ok].std() == 0 or y[ok].std() == 0:
            continue
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        w = m * np.std(rankdata(x[ok]) / m) * np.std(rankdata(y[ok]) / m)
        num += w * r; den += w
    return num / den if den > 0 else np.nan


def metric_set(P, T, mapped, cover=None, log2cov=None):
    """P prediction (or any map), T target, mapped: per-bin mappability.
    NaN = excluded. Returns the metric variants."""
    from scipy.stats import spearmanr
    n = T.shape[0]
    T0 = np.where(T == 0, np.nan, T)
    live = np.abs(np.nan_to_num(P)).sum(1) > 0
    Pl = np.where(live[:, None] & live[None, :], P, np.nan)
    if cover is not None:
        Pl = np.where(cover[:, None] & cover[None, :], Pl, np.nan)
    out = {"full_T0": pearson(Pl.ravel(), T0.ravel())}
    for k in (0, 1, 2, 5, 10):
        iu = np.triu_indices(n, k)
        out[f"triu{k}"] = pearson(Pl[iu], T0[iu])
    iu = np.triu_indices(n, 2)
    mp = mapped[:, None] & mapped[None, :]
    out["triu2_mapped"] = pearson(np.where(mp, Pl, np.nan)[iu], T0[iu])
    halo = mapped.copy()
    for s in (1, 2):
        halo[s:] &= mapped[:-s]; halo[:-s] &= mapped[s:]
    hp = halo[:, None] & halo[None, :]
    out["triu2_mapped_halo2"] = pearson(np.where(hp, Pl, np.nan)[iu], T0[iu])
    if log2cov is not None and len(log2cov) == n:
        # copy-number proxy: bins whose coverage departs from the region's
        # median by more than 0.7 log2 units (gain/loss segments, breakpoints)
        dev = np.abs(log2cov - np.nanmedian(log2cov))
        cn = mapped & (np.nan_to_num(dev, nan=9) <= 0.7)
        cp = cn[:, None] & cn[None, :]
        out["triu2_cnv_masked"] = pearson(np.where(cp, Pl, np.nan)[iu], T0[iu])
        out["cnv_masked_bin_frac"] = float(1 - cn.mean())
    x, y = Pl[iu], T0[iu]
    ok = np.isfinite(x) & np.isfinite(y)
    out["spearman_triu2"] = float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 20 else np.nan
    out["scc_k2"] = scc(Pl, T0)
    pdr = per_diag_r(Pl, T0)
    out["mean_diag_r"] = float(np.nanmean(pdr)) if np.isfinite(pdr).any() else np.nan
    out["tgt_sd_triu2"] = float(np.nanstd(T0[iu]))
    out["pred_sd_triu2"] = float(np.nanstd(Pl[iu]))
    return out


_MEDINV = {}


def mappable_bins(cell, region, res, with_cov=False):
    """Bins with a finite balancing weight; with_cov also returns log2 of
    the balancing bias (coverage) relative to the genome-wide median."""
    src = source_cool(cell, res)
    if src is None:
        return (None, None) if with_cov else None
    from flexcible.genome import match_name
    c, s, e = parse_region(region)
    nm = match_name(c, list(src.chromnames))
    w = src.bins().fetch(f"{nm}:{s}-{e}")["weight"].values
    if not with_cov:
        return np.isfinite(w)
    key = (cell, res)
    if key not in _MEDINV:
        _MEDINV[key] = np.nanmedian(1.0 / src.bins()[:]["weight"].values)
    return np.isfinite(w), np.log2((1.0 / w) / _MEDINV[key])


INDEP_REP = {"gm12878": "gm12878_ENCSR965OVH_r75", "k562": "k562_ENCSR676BEP_r5"}


def stage_metrics(a):
    reg = regions_df()
    rows = []
    for tag, res, cells in (("ref25", 25000, TRAIN12 + HELDOUT), ("all12_2kb", 2000, a.cells_2kb)):
        pdir = a.out / f"pred_{tag}"
        for cell in cells:
            f = pdir / f"pred_{cell}.npz"
            if not f.exists():
                print(f"[metrics] missing {f}", flush=True)
                continue
            Z = np.load(f)
            T = target_cool(cell, res)
            C = reference_cool(cell, 25000) if res == 25000 else None
            R = rep_cool(INDEP_REP[cell], res) if cell in INDEP_REP else None
            for region in reg.region:
                if region not in Z:
                    continue
                P = Z[region].astype(float)
                MT = fetch(T, region)
                mp, lcov = mappable_bins(cell, region, res, with_cov=True)
                if mp is None or len(mp) != MT.shape[0]:
                    mp, lcov = np.ones(MT.shape[0], bool), None
                cover = None
                if res == 2000:                       # bins inside the 512-bin model window
                    import multicell_expanded as mx
                    c, s, e = parse_region(region)
                    nb = int(T.chromsizes[c] // res) + 1
                    size = mx.WINDOW[res][0]
                    st = max(0, min(nb - size, (s + e) // 2 // res - size // 2))
                    cen = s + (np.arange(MT.shape[0]) + 0.5) * res
                    cover = (cen >= st * res) & (cen < (st + size) * res)
                maps = {"model": P}
                if C is not None:
                    maps["consensus"] = fetch(C, region)
                if R is not None:
                    maps["indep_replicate"] = fetch(R, region)
                for nm, X in maps.items():
                    if X.shape != MT.shape:
                        continue
                    Xn = X if nm == "model" else np.where(X == 0, np.nan, X)
                    rec = {"region": region, "cell": cell, "res_kb": res // 1000,
                           "map": nm, **metric_set(Xn, MT, mp, log2cov=lcov)}
                    if cover is not None and nm == "model":
                        rec["triu2_window_only"] = metric_set(Xn, MT, mp, cover)["triu2"]
                    rows.append(rec)
            print(f"[metrics] {tag} {cell}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "metric_variants.csv.gz", index=False)
    summarize_metrics(a, df)


def summarize_metrics(a, df):
    wide = pd.read_csv(a.out / "region_scores.csv", index_col=0)
    q = int(round(0.1 * len(wide)))
    bad25 = set(wide.model_med12.nsmallest(q).index)
    mcols = [c for c in df.columns if c not in ("region", "cell", "res_kb", "map")
             and not c.endswith("sd_triu2") and c != "cnv_masked_bin_frac"]
    rows = []
    for (res, mp), g in df.groupby(["res_kb", "map"]):
        if res == 25:
            bset = bad25
        else:
            wm = g[g.cell.isin(TRAIN12)].groupby("region").triu2.median()
            bset = set(wm.nsmallest(q).index)
        for m in mcols:
            if m not in g or g[m].isna().all():
                continue
            per = g[g.cell.isin(TRAIN12 + HELDOUT)]
            rows.append({"res_kb": res, "map": mp, "metric": m,
                         "median_all": per[m].median(),
                         "median_bad10": per[per.region.isin(bset)][m].median(),
                         "median_rest": per[~per.region.isin(bset)][m].median(),
                         "n_cells": per.cell.nunique()})
    s = pd.DataFrame(rows)
    s.to_csv(a.out / "metric_variants_summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    # does the bad set survive the metric change?
    g = df[(df.res_kb == 25) & (df["map"] == "model") & df.cell.isin(TRAIN12)]
    med = g.groupby("region")[mcols].median()
    rr = []
    for m in mcols:
        if med[m].notna().sum() < 100:
            continue
        bm = set(med[m].nsmallest(q).index)
        rr.append({"metric": m, "share_of_bad10_kept": len(bm & bad25) / q,
                   "spearman_with_full_T0": med[[m, "full_T0"]].corr("spearman").iloc[0, 1]})
    pd.DataFrame(rr).to_csv(a.out / "metric_badset_stability.csv", index=False)
    print(pd.DataFrame(rr).round(3).to_string(index=False))


# -------------------------------------------------------------------- fixes
def stage_fixes(a):
    """Cheap CPU tests on the saved 25 kb predictions.
    1. blend: P' = (1 - alpha) P + alpha C with the consensus C the model
       reads; alpha 0 is the model, 1 the consensus alone.
    2. evaluation rules: exclude regions whose target is flat (low SD) or
       whose independent replicate disagrees with the target, and a
       replicate-normalised score r_model / r_replicate."""
    reg = regions_df()
    wide = pd.read_csv(a.out / "region_scores.csv", index_col=0)
    q = int(round(0.1 * len(wide)))
    bad = set(wide.model_med12.nsmallest(q).index)
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    rows = []
    for cell in TRAIN12 + HELDOUT:
        f = a.out / "pred_ref25" / f"pred_{cell}.npz"
        if not f.exists():
            continue
        Z = np.load(f)
        T, C = target_cool(cell, 25000), reference_cool(cell, 25000)
        for region in reg.region:
            if region not in Z:
                continue
            P = Z[region].astype(float)
            MT = fetch(T, region)
            MC = fetch(C, region)
            if MC.shape != MT.shape:
                continue
            MC = np.where(MC == 0, np.nan, MC)
            iu = np.triu_indices(MT.shape[0], 2)
            T0 = np.where(MT == 0, np.nan, MT)[iu]
            rec = {"region": region, "cell": cell}
            for al in alphas:
                B = (1 - al) * P + al * np.nan_to_num(MC, nan=0.0)
                rec[f"alpha_{al:.1f}"] = pearson(B[iu], T0)
            rows.append(rec)
        print(f"[fixes] blend {cell}", flush=True)
    bl = pd.DataFrame(rows)
    bl.to_csv(a.out / "fix_blend_per_region.csv.gz", index=False)
    out = []
    for al in alphas:
        c = f"alpha_{al:.1f}"
        for grp, cells in (("train12", TRAIN12), ("heldout3", HELDOUT)):
            g = bl[bl.cell.isin(cells)]
            per_cell = g.groupby("cell")[c].median()
            out.append({"alpha": al, "cells": grp, "median_over_cells": per_cell.median(),
                        "median_bad10": g[g.region.isin(bad)][c].median(),
                        "median_rest": g[~g.region.isin(bad)][c].median()})
    bs = pd.DataFrame(out)
    bs.to_csv(a.out / "fix_blend_summary.csv", index=False)
    print(bs.round(4).to_string(index=False))

    # evaluation rules
    reps = pd.read_csv(a.out / "replicates_per_region.csv")
    fc = pd.read_csv(a.out / "features_cell.csv")
    long = pd.read_csv(a.out / "scores_long.csv.gz")
    ref = long[(long.source == "all12refres_25kb") & (long.column == "model")]
    ev = []
    sd_med = fc[fc.cell.isin(TRAIN12)].groupby("region").tgt_sd.median()
    for thr in (0.0, 0.15, 0.2, 0.25):
        keep = set(sd_med[sd_med >= thr].index)
        s = ref[ref.region.isin(keep) & ref.cell.isin(TRAIN12)].groupby("cell").r.median()
        ev.append({"rule": f"median target SD >= {thr}", "n_regions": len(keep),
                   "median_over_12_cells": s.median()})
    for src, res in (("all12refres_25kb", 25), ("all12_2kb", 2)):
        mdl = long[(long.source == src) & (long.column == "model")]
        for cell, lab in (("gm12878", "gm12878_intact_965OVH"), ("k562", "k562_intact_676BEP")):
            rp = reps[(reps.set == lab) & (reps.res_kb == res)].set_index("region")
            md = mdl[mdl.cell == cell].set_index("region").r
            j = pd.concat([md.rename("model"), rp.repA_vs_target.rename("rep_vs_target"),
                           rp.repA_vs_repB.rename("rep_vs_rep")], axis=1).dropna()
            tag = f"{src} {cell}"
            if res == 25:
                for thr in (0.0, 0.8, 0.9):
                    k = j[j.rep_vs_target >= thr]
                    ev.append({"rule": f"{tag}: independent replicate vs target >= {thr}",
                               "n_regions": len(k), "median_cell": k.model.median(),
                               "median_model_over_replicate":
                                   (k.model / k.rep_vs_target).median()})
            ev.append({"rule": f"{tag}: model beats independent replicate",
                       "n_regions": len(j), "share": float((j.model > j.rep_vs_target).mean()),
                       "spearman_model_replicate": j.model.corr(j.rep_vs_target, "spearman")})
            # bottom 10% of the model in this cell. Target-side: the
            # independent replicate agrees with the target in the lowest
            # quartile of regions (and, at 25 kb, below 0.85). Model-side:
            # replicate agreement in the upper half.
            b = j.nsmallest(int(round(0.1 * len(j))), "model")
            q25, q50 = j.rep_vs_target.quantile(0.25), j.rep_vs_target.median()
            ev.append({"rule": f"{tag}: bottom 10% with replicate in lowest quartile "
                               f"(< {q25:.3f})", "n_regions": len(b),
                       "share": float((b.rep_vs_target < q25).mean()),
                       "median_cell": b.model.median(),
                       "median_replicate_bottom10": b.rep_vs_target.median(),
                       "median_replicate_all": q50})
            ev.append({"rule": f"{tag}: bottom 10% with replicate in upper half "
                               f"(>= {q50:.3f})", "n_regions": len(b),
                       "share": float((b.rep_vs_target >= q50).mean())})
            if res == 25:
                ev.append({"rule": f"{tag}: bottom 10% with replicate < 0.85",
                           "n_regions": len(b), "share": float((b.rep_vs_target < 0.85).mean())})
    ev = pd.DataFrame(ev)
    ev.to_csv(a.out / "fix_eval_rules.csv", index=False)
    print(ev.round(4).to_string(index=False))


# ------------------------------------------------------------------ chipref
REF_TRACKS = ["CTCF", "H3K27ac", "H3K4me1", "H3K4me3", "H3K36me3", "H3K27me3", "H3K9me3",
              "H3K79me2", "RAD21"]


def _chip_profiles(cell, reg, res=25000, flank=500_000):
    """log ChIP per bin over the region plus flanks, z-scored per track."""
    import pyBigWig
    import multicell_expanded as mx
    bwdir = mx.CELLS[cell][1]
    miss = set(mx.missing_marks(bwdir))
    out = {}
    for t in REF_TRACKS:
        p = bwdir / f"{t}.bigwig"
        if t in miss or not p.exists():
            continue
        bw = pyBigWig.open(str(p))
        h = bw.header()
        g = h["sumData"] / max(h["nBasesCovered"], 1)
        cs = bw.chroms()
        prof = {}
        for _, r in reg.iterrows():
            s = max(0, int(r.start) - flank)
            e = min(cs[r.chrom], int(r.end) + flank)
            v = np.array([x or 0 for x in bw.stats(r.chrom, s, e, type="mean",
                                                   nBins=max(1, (e - s) // res))])
            v = np.log2(v / g + 0.1)
            prof[r.region] = (v - v.mean()) / (v.std() + 1e-9)
        out[t] = prof
    return out


def stage_chipref(a):
    """Reference chosen by ChIP similarity. The consensus averages the
    other cells' maps and flattens regions whose structure differs between
    cells. Here, per region, each other cell is weighted by how well its
    ChIP profile (shared marks, region +- 0.5 Mb) matches the target cell's,
    and the weighted mean of their maps (softmax, temperature tau) or the
    single most similar cell's map is scored against the target. No Hi-C of
    the scored cell is used."""
    reg = regions_df()
    cells = TRAIN12 + HELDOUT
    prof = {c: _chip_profiles(c, reg) for c in cells}
    print("[chipref] profiles done", flush=True)
    Ts = {c: target_cool(c, 25000) for c in cells}
    rows = []
    for region in reg.region:
        maps = {c: fetch(Ts[c], region) for c in cells}
        for cell in cells:
            pool = [o for o in TRAIN12 if o != cell]
            sims = []
            for o in pool:
                sh_ = [t for t in prof[cell] if t in prof[o]]
                x = np.concatenate([prof[cell][t][region] for t in sh_])
                y = np.concatenate([prof[o][t][region] for t in sh_])
                sims.append(pearson(x, y))
            sims = np.nan_to_num(np.array(sims), nan=-1)
            T = maps[cell]
            rec = {"region": region, "cell": cell,
                   "top_cell": pool[int(np.argmax(sims))], "top_sim": float(sims.max())}
            stack = np.stack([np.where(maps[o] == 0, np.nan, maps[o]) for o in pool])
            rec["mean_other"] = pair_r(np.nan_to_num(np.nanmean(stack, 0)), T)
            rec["top1"] = pair_r(maps[pool[int(np.argmax(sims))]], T)
            per = np.array([pair_r(maps[o], T) for o in pool])
            rec["oracle_best"] = float(np.nanmax(per))
            rec["sim_vs_score_spearman"] = float(pd.Series(sims).corr(pd.Series(per), "spearman"))
            for tau in (0.05, 0.1, 0.2):
                w = np.exp((sims - sims.max()) / tau)
                w /= w.sum()
                W = np.nansum(stack * w[:, None, None], 0)
                rec[f"weighted_tau{tau}"] = pair_r(W, T)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "chipref_per_region.csv.gz", index=False)
    wide = pd.read_csv(a.out / "region_scores.csv", index_col=0)
    q = int(round(0.1 * len(wide)))
    bad = set(wide.model_med12.nsmallest(q).index)
    cols = ["mean_other", "top1", "weighted_tau0.05", "weighted_tau0.1", "weighted_tau0.2",
            "oracle_best"]
    summ = []
    for grp, cs in (("train12", TRAIN12), ("heldout3", HELDOUT)):
        g = df[df.cell.isin(cs)]
        for c in cols:
            summ.append({"cells": grp, "reference": c,
                         "median_over_cells": g.groupby("cell")[c].median().median(),
                         "median_bad10": g[g.region.isin(bad)][c].median(),
                         "median_rest": g[~g.region.isin(bad)][c].median()})
        summ.append({"cells": grp, "reference": "ChIP similarity vs map score, Spearman",
                     "median_over_cells": g.sim_vs_score_spearman.median()})
    s = pd.DataFrame(summ)
    s.to_csv(a.out / "chipref_summary.csv", index=False)
    print(s.round(4).to_string(index=False))


# ------------------------------------------------------------------ figures
FIG_TRACKS = ["CTCF", "RAD21", "H3K27ac", "H3K36me3", "H3K27me3", "H3K9me3"]


def _bw_profile(cell, track, region, nbins):
    import pyBigWig
    import multicell_expanded as mx
    p = mx.CELLS[cell][1] / f"{track}.bigwig"
    if not p.exists():
        return None
    bw = pyBigWig.open(str(p))
    c, s, e = parse_region(region)
    v = np.array([x or 0 for x in bw.stats(c, s, e, type="mean", nBins=nbins)])
    h = bw.header()
    return v / (h["sumData"] / max(h["nBasesCovered"], 1))


def region_panel(a, region, cells, res, tag, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 6
    fig, axes = plt.subplots(len(cells), ncol, figsize=(3.1 * ncol, 3.0 * len(cells)),
                             squeeze=False)
    for ri, cell in enumerate(cells):
        Z = np.load(a.out / f"pred_{tag}" / f"pred_{cell}.npz")
        P = Z[region].astype(float)
        T = fetch(target_cool(cell, res), region)
        T = np.where(T == 0, np.nan, T)
        maps = [("target " + cell, T), ("prediction", P)]
        C = None
        if res == 25000:
            C = fetch(reference_cool(cell, res), region)
            C = np.where(C == 0, np.nan, C)
            maps.append(("consensus (input)", C))
        R = None
        if cell in INDEP_REP and rep_cool(INDEP_REP[cell], res) is not None:
            R = fetch(rep_cool(INDEP_REP[cell], res), region)
            R = np.where(R == 0, np.nan, R)
            maps.append(("independent intact rep", R))
        while len(maps) < 4:
            maps.append(("", None))
        for ci, (nm, M) in enumerate(maps):
            ax = axes[ri, ci]
            if M is None:
                ax.axis("off")
                continue
            iu = np.triu_indices(T.shape[0], 2)
            r = pair_r(np.nan_to_num(M), np.nan_to_num(T)) if ci else np.nan
            ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(nm + (f"\nr = {r:.2f}" if ci else f"\nsd = {np.nanstd(T[iu]):.2f}"),
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        # ChIP and coverage tracks
        ax = axes[ri, 4]
        n = T.shape[0]
        for k, t in enumerate(FIG_TRACKS):
            v = _bw_profile(cell, t, region, n)
            if v is None:
                ax.text(0.02, k + 0.4, f"{t} missing", fontsize=7, color="0.5",
                        transform=ax.get_yaxis_transform())
                continue
            v = np.log2(v + 0.05)
            v = (v - v.min()) / (np.ptp(v) + 1e-9)
            ax.fill_between(np.arange(n), k, k + 0.9 * v, step="mid", color=f"C{k}")
        mp = mappable_bins(cell, region, res)
        if mp is not None and len(mp) == n:
            ax.bar(np.arange(n)[~mp], len(FIG_TRACKS), bottom=0, width=1, color="0.8",
                   alpha=0.6, zorder=0)
        src = source_cool(cell, res)
        if src is not None:
            from flexcible.genome import match_name
            c, s, e = parse_region(region)
            w = src.bins().fetch(f"{match_name(c, list(src.chromnames))}:{s}-{e}")["weight"].values
            lc = np.log2((1 / w) / np.nanmedian(1 / src.bins()[:]["weight"].values))
            if len(lc) == n:
                ax.plot(np.arange(n), len(FIG_TRACKS) + 0.5 + 0.5 * np.clip(lc, -2, 2), "k-", lw=1)
        ax.set_yticks(np.arange(len(FIG_TRACKS) + 1) + 0.4)
        ax.set_yticklabels(FIG_TRACKS + ["log2 cov"], fontsize=7)
        ax.set_xlim(0, n - 1); ax.set_xticks([])
        ax.set_ylim(0, len(FIG_TRACKS) + 1.6)
        ax.set_title("ChIP (scaled), coverage; grey = unmappable", fontsize=9)
        # per-distance Pearson
        ax = axes[ri, 5]
        kk = np.arange(n) * res / 1e3
        ax.plot(kk, per_diag_r(P, T), label="prediction", color="C3")
        if C is not None:
            ax.plot(kk, per_diag_r(C, T), label="consensus", color="C0")
        if R is not None:
            ax.plot(kk, per_diag_r(R, T), label="intact rep", color="0.3", ls="--")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_ylim(-0.5, 1); ax.set_xlabel("distance (kb)", fontsize=8)
        ax.set_title("per-distance Pearson", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def stage_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    fd = a.out / "figures"
    fd.mkdir(exist_ok=True)
    wide = pd.read_csv(a.out / "region_scores.csv", index_col=0)
    worst = list(wide.model_med12.nsmallest(a.n_worst).index)
    s = wide.model_med12.sort_values()
    good = [s.index[int(q * len(s))] for q in (0.5, 0.7, 0.9)]
    paths = []
    for kind, regs in (("worst", worst), ("typical", good)):
        for i, region in enumerate(regs, 1):
            f = fd / f"ref25_{kind}{i:02d}_{region.replace(':', '_')}.png"
            row = wide.loc[region]
            region_panel(a, region, ["gm12878", "k562"], 25000, "ref25", f,
                         f"{region}  25 kb reference model; median over 12 training cells "
                         f"{row.model_med12:.3f}, consensus {row.cons_med12:.3f}, "
                         f"best other map {row.bestother_med12:.3f}")
            paths.append(f)
    # 2 kb: worst of the 2 kb model over the saved cells
    mv = pd.read_csv(a.out / "metric_variants.csv.gz")
    m2 = mv[(mv.res_kb == 2) & (mv["map"] == "model")].groupby("region").full_T0.median()
    for i, region in enumerate(m2.nsmallest(4).index, 1):
        f = fd / f"all12_2kb_worst{i:02d}_{region.replace(':', '_')}.png"
        region_panel(a, region, [c for c in ("gm12878", "k562") if c in a.cells_2kb], 2000,
                     "all12_2kb", f, f"{region}  2 kb all12 model, median {m2[region]:.3f}")
        paths.append(f)
    summary_figure(a, fd / "summary.png")
    paths.append(fd / "summary.png")
    print("\n".join(str(p) for p in paths))


def summary_figure(a, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wide = pd.read_csv(a.out / "region_scores.csv", index_col=0)
    reps = pd.read_csv(a.out / "replicates_per_region.csv")
    imp = pd.read_csv(a.out / "feature_groups.csv")
    ms = pd.read_csv(a.out / "metric_variants_summary.csv")
    long = pd.read_csv(a.out / "scores_long.csv.gz")
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    q = int(round(0.1 * len(wide)))
    bad = set(wide.model_med12.nsmallest(q).index)
    # a: model vs consensus per region
    x = ax[0, 0]
    isb = wide.index.isin(bad)
    x.scatter(wide.cons_med12, wide.model_med12, s=8, c=np.where(isb, "C3", "0.5"))
    x.plot([0.3, 1], [0.3, 1], "k:", lw=0.8)
    x.set_xlabel("consensus score (median of 12 cells)")
    x.set_ylabel("model score (median of 12 cells)")
    x.set_title("25 kb reference model per region (red: bottom 10%)", fontsize=10)
    # b: model vs independent replicate, GM12878 and K562 at 25 kb
    x = ax[0, 1]
    ref = long[(long.source == "all12refres_25kb") & (long.column == "model")]
    for k, (cell, lab) in enumerate((("gm12878", "gm12878_intact_965OVH"),
                                     ("k562", "k562_intact_676BEP"))):
        rp = reps[(reps.set == lab) & (reps.res_kb == 25)].set_index("region").repA_vs_target
        md = ref[ref.cell == cell].set_index("region").r
        j = pd.concat([rp, md], axis=1, keys=["rep", "model"]).dropna()
        x.scatter(j.rep, j.model, s=8, color=f"C{k}", label=f"{cell}", alpha=0.7)
    x.plot([0.3, 1], [0.3, 1], "k:", lw=0.8)
    x.set_xlabel("independent intact replicate vs target")
    x.set_ylabel("model vs target")
    x.legend(fontsize=8)
    x.set_title("target-side (both low) vs model-side (model low only)", fontsize=10)
    # c: score by resolution for 2 kb and 25 kb bad sets
    x = ax[0, 2]
    cols = ["all12_2kb_med12", "all12_5kb_med12", "all12_10kb_med12", "all12_25kb_med12",
            "model_med12"]
    labs = ["2 kb", "5 kb", "10 kb", "25 kb", "25 kb ref"]
    b2 = set(wide.all12_2kb_med12.nsmallest(q).index)
    for k, (nm, bs) in enumerate((("bottom 10% at 25 kb (ref)", bad),
                                  ("bottom 10% at 2 kb", b2), ("all regions", set(wide.index)))):
        x.plot(labs, [wide.loc[list(bs), c].median() for c in cols], "o-", label=nm)
    x.set_ylabel("median score"); x.legend(fontsize=8)
    x.set_title("bad sets differ by resolution", fontsize=10)
    # d: feature groups
    x = ax[1, 0]
    g = imp[imp.fit == "pooled_model"].sort_values("rf_cv_r2_alone")
    x.barh(g.group, g.rf_cv_r2_alone, color="C0", label="group alone")
    x.barh(g.group, g.rf_cv_r2_drop, color="C3", alpha=0.7, label="loss when dropped")
    x.set_xlabel("cross-validated R2 (region x cell, per-cell centred)")
    x.tick_params(axis="y", labelsize=7)
    x.legend(fontsize=8)
    x.set_title("what explains the score", fontsize=10)
    # e: metric variants, bad vs rest
    x = ax[1, 1]
    s = ms[(ms.res_kb == 25) & (ms["map"] == "model")].set_index("metric")
    order = [m for m in ["full_T0", "triu0", "triu2", "triu5", "triu10", "triu2_mapped",
                         "triu2_mapped_halo2", "spearman_triu2", "scc_k2", "mean_diag_r"]
             if m in s.index]
    yy = np.arange(len(order))
    x.barh(yy - 0.2, s.loc[order, "median_bad10"], 0.4, color="C3", label="bottom 10%")
    x.barh(yy + 0.2, s.loc[order, "median_rest"], 0.4, color="0.6", label="rest")
    x.set_yticks(yy); x.set_yticklabels(order, fontsize=8)
    x.set_xlim(0, 1); x.legend(fontsize=8)
    x.set_title("25 kb model score under metric variants", fontsize=10)
    # f: target dynamic range vs score
    x = ax[1, 2]
    d = pd.read_csv(a.out / "region_design.csv", index_col=0)
    x.scatter(d.tgt_sd, d.model, s=8, c=np.where(d.index.isin(bad), "C3", "0.5"))
    x.set_xlabel("target SD, median of 12 cells (O/E log units)")
    x.set_ylabel("model score")
    x.set_title("flat targets score lower", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)


SMALL_TABLES = ["consistency.csv", "feature_fits.csv", "feature_groups.csv",
                "feature_importance.csv", "feature_univariate.csv", "metric_variants_summary.csv",
                "metric_badset_stability.csv", "fix_blend_summary.csv", "fix_eval_rules.csv",
                "chipref_summary.csv", "worst_25kb.csv", "worst_2kb.csv"]


def stage_tables(a):
    """Bottom-10% tables at 25 kb (reference model) and 2 kb (all12), with
    the scores of the other maps, the independent replicate and the main
    features; copies the small tables to results/worst_regions."""
    import shutil
    w = pd.read_csv(a.out / "region_scores.csv", index_col=0)
    d = pd.read_csv(a.out / "region_design.csv", index_col=0)
    fr = pd.read_csv(a.out / "features_region.csv", index_col=0)
    rp = pd.read_csv(a.out / "replicates_per_region.csv")
    q = int(round(0.1 * len(w)))
    feat = ["tgt_sd", "raw_per_px", "log2cov_mean", "unmap_frac", "chip_H3K27ac",
            "chip_H3K9me3", "e1_mean"]
    for res, col, oth in ((25, "model_med12", ["cons_med12", "bestother_med12",
                                               "model_med_heldout", "all12_25kb_med12",
                                               "all12_2kb_med12"]),
                          (2, "all12_2kb_med12", ["bestother_2kb_med12", "all12_25kb_med12",
                                                  "model_med12"])):
        t = w.nsmallest(q, col)[[col] + oth + (["model_min12"] if res == 25 else [])]
        for cell, lab in (("gm12878", "gm12878_intact_965OVH"), ("k562", "k562_intact_676BEP")):
            r = rp[(rp.set == lab) & (rp.res_kb == res)].set_index("region")
            t[f"{cell}_indep_rep_vs_target"] = r.repA_vs_target
            t[f"{cell}_intact_repA_vs_repB"] = r.repA_vs_repB
        t = t.join(d[feat]).join(fr[["dist_centromere_mb", "dist_telomere_mb", "segdup_frac",
                                     "tss_per_mb", "between_cell_var_share"]])
        t.round(4).to_csv(a.out / f"worst_{res}kb.csv")
    dst = REPO / "results/worst_regions"
    dst.mkdir(parents=True, exist_ok=True)
    for f in SMALL_TABLES:
        shutil.copy(a.out / f, dst / f)
    print(f"copied {len(SMALL_TABLES)} tables to {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cells-2kb", nargs="*", default=["gm12878", "k562"])
    ap.add_argument("--n-worst", type=int, default=8)
    ap.add_argument("--pred-cells", nargs="*", default=None,
                    help="predict stage: only these tag:cell jobs (e.g. ref25:k562)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(a.threads)
    for s in a.stage:
        globals()[f"stage_{s}"](a)


if __name__ == "__main__":
    main()
