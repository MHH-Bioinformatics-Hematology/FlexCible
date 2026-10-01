"""Why does another AML patient's map predict a patient better than the model?

Separates three explanations for the primary AML patients of
scripts/aml_primary.py (Xu et al. 2022):

  sv        structural variants and copy-number changes. SVs from Xu et al.
            Supplementary Table 5 (WGS plus hic_breakfinder: translocations,
            inversions, deletions, duplications); copy number called here
            from Hi-C coverage (raw contacts per 500 kb bin of each patient's
            own 5 kb map over the mean of the four HSPC maps of the same lab
            and protocol, log2, median-centred, autosomes only (sex differs),
            rolling median over 5 bins; segments of >= 6 bins (3 Mb) with
            |log2 ratio| >= max(0.3, 4 robust SD of the sample)). Each of the 413
            test regions is flagged per patient when it lies within 1 Mb of
            an SV breakpoint, inside a deletion/inversion/duplication, or
            overlaps a CNV segment; scores are split on the flag.
  lineage   the four healthy CD34+ HSPC maps of the same study (the three
            PBMC controls of the paper have no public Hi-C), their consensus,
            and the blood cells of the training set (GM12878, CD14+
            monocytes) and K562 as baselines on the shared pixels; the
            agreement of a patient with other AML patients is set against its
            agreement with HSPC (same lab and protocol, healthy) and with
            blood cells from other labs.
  learn     regions where the patient differs from consensus12 and from the
            HSPC consensus (both deviation scores in the patient's top 20%,
            noise filters of the hard split, SV/CNV regions excluded): the
            change of the patient's own CUT&Tag tracks there against the
            training cells' tracks, and the residual correlation of each
            prediction with the measured deviation (from consensus12 and from
            HSPC).
  features  Xu et al.'s AML-specific loops (Suppl. Table 1; typed here by the
            patient's own CUT&Tag at the anchors: promoter-enhancer-like if
            both anchors carry H3K27ac, silencer-like if one carries
            H3K27me3) and stripes (Suppl. Table 4), predicted on windows
            centred on them: donut enrichment at the loop pixel and stripe
            strength in the prediction, the patient map, the HSPC consensus
            and consensus12.

Tables in results/aml_primary/explain_*.csv; Xu's tables as CSV in
results/aml_primary/xu_tables/.

    python scripts/aml_primary_explain.py --stage cnv annotate scores learn features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aml_primary as ap                                          # noqa: E402

XT = ap.RES_DIR / "xu_tables"
CNV_BIN = 500_000
CNV_MIN_LOG2 = 0.3          # and at least 4 robust SD of the sample's smoothed ratios
CNV_MIN_BINS = 6            # 3 Mb
HSPC = ["xu_HSPC-050", "xu_HSPC-213", "xu_HSPC-341", "xu_HSPC-822"]
BLOOD = ["gm12878", "cd14mono", "k562"]
log = ap.log


def norm_id(x) -> str:
    """Patient IDs of the supplementary tables ('18', '#018') as in GEO ('018')."""
    s = str(x).lstrip("#").strip()
    return s.zfill(3) if s.isdigit() and len(s) < 3 else s


def target_patients() -> list[str]:
    return sorted(p for p in ap.patients(25000) if (ap.INP / p).exists())


# ---------------------------------------------------------------- copy number
def marginal(label: str, binsize: int = CNV_BIN) -> pd.Series:
    """Raw contacts per `binsize` bin (row sums of the unbalanced 5 kb map,
    both triangles), indexed by (chrom, bin start)."""
    import cooler
    c = cooler.Cooler(str(ap.WORK / f"{label}_5kb.raw.cool"))
    bins = c.bins()[:][["chrom", "start"]]
    tot = np.zeros(len(bins))
    n = int(c.info["nnz"])
    for i in range(0, n, 50_000_000):
        px = c.pixels()[i:i + 50_000_000]
        np.add.at(tot, px["bin1_id"].values, px["count"].values)
        off = px["bin1_id"].values != px["bin2_id"].values
        np.add.at(tot, px["bin2_id"].values[off], px["count"].values[off])
    key = bins["start"] // binsize * binsize
    s = pd.Series(tot).groupby([bins["chrom"].values, key.values]).sum()
    return s


def stage_cnv() -> pd.DataFrame:
    """log2 coverage ratio against the HSPC mean and CNV segments per patient."""
    from concurrent.futures import ProcessPoolExecutor
    labs = ap.patients(25000)
    with ProcessPoolExecutor(max_workers=6) as ex:
        margs = dict(zip(labs, ex.map(marginal, labs)))
    M = pd.DataFrame(margs)
    M = M[M.index.get_level_values(0).isin(ap.MAIN)]
    M = M / M.sum()                                       # depth per sample
    ref = M[HSPC].mean(1)
    ok = ref > ref.median() * 0.2                         # mappable bins
    rows, segs = [], []
    auto = ~M.index.get_level_values(0).isin(["chrX", "chrY"])   # sex differs between samples
    for p in labs:
        r = np.log2(M.loc[ok & auto, p] / ref[ok & auto])
        r = r - r.median()
        sm_all = r.groupby(level=0).transform(
            lambda g: g.rolling(5, center=True, min_periods=3).median())
        mad = float(np.nanmedian(np.abs(sm_all - np.nanmedian(sm_all))))
        thr = max(CNV_MIN_LOG2, 4 * 1.4826 * mad)
        for chrom, g in r.groupby(level=0):
            sm = g.rolling(5, center=True, min_periods=3).median()
            for (c, st), v, raw in zip(g.index, sm.values, g.values):
                rows.append({"patient": p, "chrom": c, "start": st, "log2": raw, "smooth": v})
            call = np.where(sm.values >= thr, 1, np.where(sm.values <= -thr, -1, 0))
            starts = g.index.get_level_values(1).values
            i = 0
            while i < len(call):
                j = i
                while j < len(call) and call[j] == call[i]:
                    j += 1
                if call[i] != 0 and j - i >= CNV_MIN_BINS:
                    segs.append({"patient": p, "chrom": chrom, "start": int(starts[i]),
                                 "end": int(starts[j - 1] + CNV_BIN), "state": "gain" if call[i] > 0 else "loss",
                                 "mean_log2": float(np.nanmean(sm.values[i:j])), "bins": j - i})
                i = j
    pd.DataFrame(rows).to_csv(ap.WORK / "cnv_log2_500kb.csv.gz", index=False)
    S = pd.DataFrame(segs)
    S.to_csv(ap.RES_DIR / "explain_cnv_segments.csv", index=False)
    log(f"[cnv] {len(S)} segments over {S.patient.nunique() if len(S) else 0} samples")
    return S


# ---------------------------------------------------------------- annotation
def sv_table() -> pd.DataFrame:
    d = pd.read_csv(XT / "st5_svs.csv", dtype={"patient": str})
    d["patient"] = "xu_" + d["patient"].map(norm_id)
    return d


def region_flags() -> pd.DataFrame:
    """Per patient (every map) and test region: SV, CNV, neo-loop flags."""
    R = ap.region_list()
    R[["start", "end"]] = R[["start", "end"]].astype(int)
    sv = sv_table()
    cnv = pd.read_csv(ap.RES_DIR / "explain_cnv_segments.csv")
    neo = pd.read_csv(XT / "st6_neoloop_hijacking.csv", dtype={"patient": str})
    neo["patient"] = "xu_" + neo["patient"].map(norm_id)
    rows = []
    for p in ap.patients(25000):
        s, c, n = sv[sv.patient == p], cnv[cnv.patient == p], neo[neo.patient == p]
        for r in R.itertuples():
            lo, hi = r.start - 1_000_000, r.end + 1_000_000
            near = 0
            for x in s.itertuples():
                for ch, pos in ((x.chrom1, x.pos1), (x.chrom2, x.pos2)):
                    if ch == r.chrom and lo <= pos <= hi:
                        near += 1
                if x.sv_type != "translocation" and x.chrom1 == r.chrom == x.chrom2:
                    a, b = sorted((x.pos1, x.pos2))
                    if a < r.end and b > r.start:
                        near += 1
            cn = c[(c.chrom == r.chrom) & (c.start < r.end) & (c.end > r.start)]
            nl = 0
            for x in n.itertuples():
                for loc in (x.gene_region, x.enhancer):
                    ch, se = str(loc).split(":")
                    a, b = map(int, se.split("-"))
                    if ch == r.chrom and a < r.end and b > r.start:
                        nl += 1
            rows.append({"patient": p, "region": r.region, "sv": near > 0, "n_sv": near,
                         "cnv": len(cn) > 0, "cnv_state": ";".join(cn.state) if len(cn) else "",
                         "neoloop": nl > 0, "has_sv_table": len(s) > 0})
    F = pd.DataFrame(rows)
    F["affected"] = F["sv"] | F["cnv"]
    F.to_csv(ap.RES_DIR / "explain_region_flags.csv", index=False)
    log(f"[annotate] {int(F.affected.sum())} affected patient-regions of {len(F)}")
    return F


# ---------------------------------------------------------------- scores on shared pixels
MODELS_X = ["all12refres_drop", "all12honest_drop", "min_ctcf_h3k27ac_refres", "all12refres"]


def hspc_consensus_path(res: int) -> Path:
    return ap.CONS_DIR / f"hspccons_{res // 1000}kb.cool"


def build_hspc_consensus(res: int) -> Path:
    """Mean of the four HSPC maps (aml_primary.build_aml_consensus code path
    with another sample list)."""
    return ap.build_aml_consensus(res, None, labs=[h for h in HSPC if ap.map_path(h, res)],
                                  out=hspc_consensus_path(res))


def explain_scores_res(res: int) -> pd.DataFrame:
    """Per patient and test region, on one shared mask (patient map,
    consensus12, HSPC consensus, the patient's leave-one-out AML consensus,
    the four HSPC maps, GM12878, CD14+ monocytes, K562 and the model
    predictions): Pearson of every map and prediction with the patient map;
    the deviation scores from consensus12 (C) and the HSPC consensus (H);
    residual correlations corr(X - C, T - C) and corr(X - H, T - H)."""
    import cooler
    import benchmark_metrics as bm
    import hard_split as hs
    from aml_eval import fetch_region, pred_matrix
    k = res // 1000
    pats = target_patients()
    models = [f"{m}_{k}kb" for m in MODELS_X if f"{m}_{k}kb" in ap.PMODELS]
    maps = {"consensus12": ap.consensus12(res), "hspc_consensus": hspc_consensus_path(res)}
    maps.update({h: ap.map_path(h, res) for h in HSPC})
    maps.update({b: ap.map_path(b, res) for b in BLOOD})
    clr = {kk: cooler.Cooler(str(v)) for kk, v in maps.items() if v is not None and Path(v).exists()}
    rows = []
    for p in pats:
        tclr = cooler.Cooler(str(ap.map_path(p, res)))
        aclr = cooler.Cooler(str(ap.aml_consensus_path(res, p)))
        raw = cooler.Cooler(str(ap.WORK / f"{p}_{k}kb.cool"))
        preds = {}
        for m in models:
            for tag, nm in (("", m), (ap.AMLREF, m + "@amlref")):
                f = ap.pred_file(m, p + tag)
                if f.exists():
                    preds[nm] = dict(np.load(f))
        for r in ap.region_list().itertuples():
            cache: dict = {}
            T = fetch_region(tclr, r.region, cache)
            if T is None:
                continue
            mats = {f"map:{kk}": fetch_region(c, r.region, cache) for kk, c in clr.items()}
            mats["map:aml_consensus"] = fetch_region(aclr, r.region, cache)
            for nm, pr in preds.items():
                if r.region in pr:
                    mats[f"model:{nm}"] = pred_matrix(pr[r.region])
            mats = {kk: v for kk, v in mats.items() if v is not None and v.shape == T.shape}
            if "map:consensus12" not in mats or "map:hspc_consensus" not in mats:
                continue
            mask = bm.triu_mask(T.shape[0], 2) & np.isfinite(T)
            for M in mats.values():
                mask &= np.isfinite(M)
            if mask.sum() < 100:
                continue
            C, H = mats["map:consensus12"], mats["map:hspc_consensus"]
            dC, dH = (T - C)[mask], (T - H)[mask]
            ch, s, e = r.chrom, int(r.start), int(r.end)
            mb = hs.masked_bins(raw, ch, s, e)
            rc = hs.fetch(raw, ch, s, e)
            rec = {"patient": p, "res_kb": k, "region": r.region, "n_pixels": int(mask.sum()),
                   "coverage": float(np.nanmean(np.nan_to_num(rc)[hs.tri(T.shape[0])])),
                   "masked_frac": float(mb.mean()) if mb.shape[0] == T.shape[0] else np.nan}
            for X, nm in ((C, "C"), (H, "H")):
                x, t = X[mask], T[mask]
                r_ = hs.corr(t, x)
                rec[f"score_vs_{nm}"] = 0.5 * (1 - r_) + 0.5 * float(np.var(t - x) / np.var(t))
            for kk, M in mats.items():
                rec[f"r|{kk}"] = bm.pearson(M, T, mask)
                if kk not in ("map:consensus12",):
                    rec[f"residC|{kk}"] = hs.corr((M - C)[mask], dC)
                if kk not in ("map:hspc_consensus",):
                    rec[f"residH|{kk}"] = hs.corr((M - H)[mask], dH)
            rows.append(rec)
    return pd.DataFrame(rows)


def stage_scores(res_list: list[int]) -> None:
    from concurrent.futures import ProcessPoolExecutor
    for res in res_list:
        build_hspc_consensus(res)
    with ProcessPoolExecutor(max_workers=len(res_list)) as ex:
        parts = list(ex.map(explain_scores_res, res_list))
    D = pd.concat(parts, ignore_index=True)
    F = pd.read_csv(ap.RES_DIR / "explain_region_flags.csv")
    D = D.merge(F[["patient", "region", "sv", "cnv", "affected", "neoloop", "has_sv_table"]],
                on=["patient", "region"], how="left")
    D.to_csv(ap.WORK / "explain_per_region.csv.gz", index=False)
    rcols = [c for c in D.columns if c.startswith("r|")]
    out = []
    for (p, k), g in D.groupby(["patient", "res_kb"]):
        for sel, gg in (("all", g), ("sv_cnv_excluded", g[~g.affected.fillna(False)]),
                        ("sv_cnv_only", g[g.affected.fillna(False)])):
            rec = {"patient": p, "res_kb": k, "set": sel, "n_regions": len(gg)}
            rec.update({c[2:]: gg[c].median() for c in rcols})
            out.append(rec)
    S = pd.DataFrame(out)
    S.to_csv(ap.RES_DIR / "explain_scores.csv", index=False)
    lineage_table(D).to_csv(ap.RES_DIR / "explain_lineage.csv", index=False)
    log("[scores] explain_scores.csv, explain_lineage.csv")


def lineage_table(D: pd.DataFrame) -> pd.DataFrame:
    """Per resolution, medians over patients of the per-patient median
    Pearson: other AML patients (from aml_primary's other_patients table,
    best and median), single HSPC maps (best and median), the HSPC and AML
    consensus, blood training cells, K562 and consensus12; SV/CNV regions
    excluded and all regions. HSPC-to-HSPC agreement from patient_pairs.csv."""
    O = pd.read_csv(ap.RES_DIR / "other_patients.csv")
    P = pd.read_csv(ap.RES_DIR / "patient_pairs.csv")
    rows = []
    for k, g in D.groupby("res_kb"):
        for sel, gg in (("all", g), ("sv_cnv_excluded", g[~g.affected.fillna(False)])):
            per = gg.groupby("patient")[[c for c in gg.columns if c.startswith("r|")]].median()
            rec = {"res_kb": k, "set": sel, "n_patients": len(per)}
            for c in per.columns:
                rec[c[2:]] = per[c].median()
            hs_cols = [f"r|map:{h}" for h in HSPC if f"r|map:{h}" in per]
            rec["best_single_hspc"] = per[hs_cols].max(1).median() if hs_cols else np.nan
            o = O[(O.res_kb == k) & ~O.other.str.contains("HSPC") & O.target.isin(per.index)]
            if len(o):
                rec["best_other_aml_patient(all regions)"] = o.groupby("target").pearson_median.max().median()
                rec["median_other_aml_patient(all regions)"] = o.groupby("target").pearson_median.median().median()
            pp = P[P.res_kb == k]
            hh = pp[pp.a.isin(HSPC) & pp.b.isin(HSPC)]
            aa = pp[~pp.a.str.contains("HSPC") & ~pp.b.str.contains("HSPC")]
            ah = pp[pp.a.str.contains("HSPC") ^ pp.b.str.contains("HSPC")]
            rec["hspc_vs_hspc(all regions)"] = hh.pearson.median() if len(hh) else np.nan
            rec["aml_vs_aml(all regions)"] = aa.pearson.median() if len(aa) else np.nan
            rec["aml_vs_hspc(all regions)"] = ah.pearson.median() if len(ah) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- learnability
def track_matrix(bw_dir: Path, marks: list[str], chrom: str, start: int, end: int,
                 res: int) -> dict:
    """Bin means of the given tracks over a region (log1p)."""
    import pyBigWig
    out = {}
    nb = (end - start) // res
    for m in marks:
        f = Path(bw_dir) / f"{m}.bigwig"
        if not f.exists():
            continue
        bw = pyBigWig.open(str(f))
        v = bw.stats(chrom, start, start + nb * res, type="mean", nBins=nb)
        bw.close()
        out[m] = np.log1p(np.nan_to_num(np.asarray(v, float)))
    return out


def stage_learn(res_list: list[int], frac: float = 0.2) -> None:
    """Patient-specific regions (top `frac` of both deviation scores per
    patient, coverage and masked-bin filters of the hard split, SV/CNV
    excluded): input change (patient CUT&Tag against the mean of the training
    cells' tracks of the mark, as the quantile-matched scale, per bin; mean
    absolute difference and its correlation with the measured deviation's
    per-bin row mean) and residual correlations there."""
    D = pd.read_csv(ap.WORK / "explain_per_region.csv.gz")
    from aml_fetch import TRAIN_DIRS
    out, reg_rows = [], []
    for (p, k), g in D.groupby(["patient", "res_kb"]):
        if k * 1000 not in res_list:
            continue
        g = g[(g.masked_frac <= 0.2) & ~g.affected.fillna(False)]
        g = g[g.coverage >= g.coverage.quantile(0.15)]
        q = g[(g.score_vs_C.rank(pct=True, ascending=False) <= frac)
              & (g.score_vs_H.rank(pct=True, ascending=False) <= frac)]
        marks = sorted({b.stem for b in (ap.INP / p).glob("*.bigwig")})
        for r in q.itertuples():
            ch, se = r.region.split(":")
            s, e = map(int, se.split("-"))
            pt = track_matrix(ap.INP / p, marks, ch, s, e, k * 1000)
            rec = {"patient": p, "res_kb": k, "region": r.region}
            for m in marks:
                ref = [track_matrix(d, [m], ch, s, e, k * 1000).get(m) for d in TRAIN_DIRS]
                ref = [x for x in ref if x is not None]
                if not ref or m not in pt:
                    continue
                rec[f"input_change|{m}"] = float(np.mean(np.abs(pt[m] - np.mean(ref, 0))))
            for c in D.columns:
                if c.startswith(("residC|", "residH|")):
                    rec[c] = q.loc[r.Index, c]
            reg_rows.append(rec)
        rec = {"patient": p, "res_kb": k, "n_regions": len(q),
               "score_vs_C_median": q.score_vs_C.median(), "score_vs_H_median": q.score_vs_H.median()}
        for c in D.columns:
            if c.startswith(("residC|", "residH|")):
                rec[c] = q[c].median()
                rec[c + "|all_passing"] = g[c].median()
        out.append(rec)
    pd.DataFrame(out).to_csv(ap.RES_DIR / "explain_learn.csv", index=False)
    R = pd.DataFrame(reg_rows)
    R.to_csv(ap.RES_DIR / "explain_learn_regions.csv", index=False)
    log(f"[learn] {len(R)} patient-specific regions")


# ---------------------------------------------------------------- Xu features
FEAT_RES = 10000
WIN_BINS = 192                       # the 10 kb model window (1.92 Mb)


def feature_windows(max_per_patient: int = 150) -> pd.DataFrame:
    """AML-specific loops (Suppl. Table 1) and stripes (Suppl. Table 4) of
    the patients with CUT&Tag, intra-chromosomal, 60 kb to 1.5 Mb; each in a
    window of WIN_BINS bins centred on the feature."""
    pats = {p[3:] for p in target_patients()}
    L = pd.read_csv(XT / "st1_aml_specific_loops.csv", dtype={"patient": str})
    L["patient"] = L.patient.map(norm_id)
    L = L[L.patient.isin(pats) & (L.chrom1 == L.chrom2)]
    L["dist"] = L.start2 - L.start1
    L = L[(L.dist >= 60000) & (L.dist <= 1_500_000)]
    S = pd.read_csv(XT / "st4_aml_specific_stripes.csv", dtype={"patient": str})
    S["patient"] = S.patient.map(norm_id)
    S = S[S.patient.isin(pats) & (S.chrom_anchor == S.chrom_extent)]
    rows = []
    half = WIN_BINS // 2 * FEAT_RES
    for p, g in L.groupby("patient"):
        g = g.sample(min(len(g), max_per_patient), random_state=0)
        for r in g.itertuples():
            c = (r.start1 + r.end2) // 2 // FEAT_RES * FEAT_RES
            rows.append({"patient": "xu_" + p, "kind": "loop", "chrom": r.chrom1,
                         "start": max(0, c - half), "end": c + half,
                         "a1": int(r.start1), "a2": int(r.start2)})
    for p, g in S.groupby("patient"):
        g = g.sample(min(len(g), max_per_patient), random_state=0)
        for r in g.itertuples():
            lo = min(r.start_anchor, r.start_extent)
            hi = max(r.end_anchor, r.end_extent)
            if hi - lo > 1_500_000:
                continue
            c = (lo + hi) // 2 // FEAT_RES * FEAT_RES
            rows.append({"patient": "xu_" + p, "kind": "stripe", "chrom": r.chrom_anchor,
                         "start": max(0, c - half), "end": c + half,
                         "a1": int(r.start_anchor), "a2": int(r.end_anchor),
                         "e1": int(r.start_extent), "e2": int(r.end_extent)})
    return pd.DataFrame(rows)


FEAT_MODELS = ["all12refres_drop_10kb", "all12honest_drop_10kb", "min_ctcf_h3k27ac_refres_10kb"]


def stage_features_predict(devices: list[str]) -> None:
    W = feature_windows()
    W.to_csv(ap.WORK / "feature_windows.csv", index=False)
    jobs = []
    out = ap.PRED / "features"
    out.mkdir(parents=True, exist_ok=True)
    for p, g in W.groupby("patient"):
        reg = out / f"{p}.tsv"
        g[["chrom", "start", "end"]].drop_duplicates().to_csv(reg, sep="\t", header=False, index=False)
        for m in FEAT_MODELS:
            jobs.append((m, p, reg))
    import queue
    import threading
    import os
    import subprocess
    q = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(dev):
        while True:
            try:
                m, p, reg = q.get_nowait()
            except queue.Empty:
                return
            f = out / f"{m}__{p}.npz"
            if f.exists():
                continue
            res, run, fam, ref = ap.PMODELS[m]
            cmd = [ap.PY, "scripts/crosscell_predict.py", "--config", str(ap.pmodel_config(m)),
                   "--checkpoint", str(run / (run / "best").read_text().strip()),
                   "--cell", f"aml_{p}", "--bigwig-dir", str(ap.INP / p),
                   "--target", str(ap.grid_cool(res)), "--tta", "--assay", "hic",
                   "--binsize", str(res), "--device", "cuda", "--regions", str(reg),
                   "--out-dir", str(out / f"{m}__{p}_csv"), "--save-pred", str(f) + ".tmp.npz"]
            if ref:
                cmd += ["--reference", str(ap.consensus12(res))]
            env = dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=dev)
            with open(out / f"{m}__{p}.log", "w") as fh:
                subprocess.run(cmd, cwd=ap.REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
            if Path(str(f) + ".tmp.npz").exists():
                Path(str(f) + ".tmp.npz").rename(f)
                log(f"[features predict] {m} {p}")
    ts = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def anchor_marks(p: str, chrom: str, a: int) -> dict:
    """Mean quantile-matched H3K27ac / H3K27me3 / CTCF over a 10 kb anchor,
    and the genome-wide 90th percentile reference of the mark (ap quantiles)."""
    import pyBigWig
    out = {}
    for m in ("H3K27ac", "H3K27me3", "CTCF"):
        f = ap.INP / p / f"{m}.bigwig"
        if not f.exists():
            continue
        bw = pyBigWig.open(str(f))
        v = bw.stats(chrom, a, a + FEAT_RES, type="mean")[0]
        bw.close()
        q = ap.reference_quantiles(m)
        out[m] = (v or 0.0) >= q[int(0.9 * (ap.NQ - 1))]
    return out


def stage_features() -> None:
    """Loop pixels: log2 donut enrichment (hicmultiome.bioscore.local_enrichment,
    10 kb, no coarsening) in the patient map, each prediction, the HSPC
    consensus and consensus12. Stripes: mean log O/E over the stripe band
    (anchor rows x extent columns) minus the same band's mean in the reference."""
    import cooler
    from aml_eval import fetch_region, pred_matrix
    from hicmultiome.bioscore import local_enrichment
    W = pd.read_csv(ap.WORK / "feature_windows.csv")
    from aml_eval import loop_params
    prm = loop_params(FEAT_RES)
    C12 = cooler.Cooler(str(ap.consensus12(FEAT_RES)))
    HC = cooler.Cooler(str(hspc_consensus_path(FEAT_RES)))
    rows = []
    for p, g in W.groupby("patient"):
        T = cooler.Cooler(str(ap.map_path(p, FEAT_RES)))
        preds = {m: dict(np.load(ap.PRED / "features" / f"{m}__{p}.npz")) for m in FEAT_MODELS
                 if (ap.PRED / "features" / f"{m}__{p}.npz").exists()}
        for r in g.itertuples():
            win = f"{r.chrom}:{int(r.start)}-{int(r.end)}"
            cache: dict = {}
            mats = {"patient": fetch_region(T, win, cache), "hspc_consensus": fetch_region(HC, win, cache),
                    "consensus12": fetch_region(C12, win, cache)}
            for m, pr in preds.items():
                if win in pr:
                    mats[m] = pred_matrix(pr[win])
            if any(v is None for v in mats.values()):
                continue
            n = mats["patient"].shape[0]
            if any(v.shape != (n, n) for v in mats.values()):
                continue
            rec = {"patient": p, "kind": r.kind, "window": win}
            i = (int(r.a1) - int(r.start)) // FEAT_RES
            if r.kind == "loop":
                j = (int(r.a2) - int(r.start)) // FEAT_RES
                if not (0 <= i < n and 0 <= j < n):
                    continue
                rec["dist_kb"] = (j - i) * FEAT_RES / 1000
                for kk, M in mats.items():
                    e = local_enrichment(np.exp(np.nan_to_num(M)), prm)
                    rec[kk] = float(np.log2(e[min(i, j), max(i, j)]))
                am = anchor_marks(p, r.chrom, int(r.a1))
                bm_ = anchor_marks(p, r.chrom, int(r.a2))
                rec["type"] = ("P-S-like" if am.get("H3K27me3") or bm_.get("H3K27me3") else
                               "P-E-like" if am.get("H3K27ac") and bm_.get("H3K27ac") else "other")
            else:
                j1 = (int(r.e1) - int(r.start)) // FEAT_RES
                j2 = (int(r.e2) - int(r.start)) // FEAT_RES
                i2 = (int(r.a2) - int(r.start)) // FEAT_RES
                if not (0 <= i < n and 0 <= j1 < n and 0 < j2 <= n):
                    continue
                rows_ = slice(max(0, i), min(n, i2 + 1))
                cols_ = slice(max(0, j1), min(n, j2))
                for kk, M in mats.items():
                    band = M[rows_, cols_]
                    rec[kk] = float(np.nanmean(band)) if np.isfinite(band).any() else np.nan
                rec["type"] = "stripe"
            rows.append(rec)
    F = pd.DataFrame(rows)
    F.to_csv(ap.WORK / "explain_features.csv.gz", index=False)
    out = []
    for (kind, typ), g in F.groupby(["kind", "type"]):
        rec = {"kind": kind, "type": typ, "n": len(g), "patients": g.patient.nunique()}
        for ref in ("hspc_consensus", "consensus12"):
            rec[f"measured_minus_{ref}"] = (g["patient"] - g[ref]).median()
            for m in FEAT_MODELS:
                if m in g:
                    d_meas = g["patient"] - g[ref]
                    d_pred = g[m] - g[ref]
                    rec[f"{m}_minus_{ref}"] = d_pred.median()
                    rec[f"{m}_frac_above_{ref}"] = float((d_pred > 0).mean())
                    ok = d_meas.notna() & d_pred.notna()
                    rec[f"{m}_corr_with_measured_vs_{ref}"] = (
                        float(np.corrcoef(d_meas[ok], d_pred[ok])[0, 1]) if ok.sum() > 10 else np.nan)
        out.append(rec)
    pd.DataFrame(out).to_csv(ap.RES_DIR / "explain_features.csv", index=False)
    log("[features] explain_features.csv")


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--stage", nargs="+", required=True,
                   choices=["cnv", "annotate", "scores", "learn", "features_predict", "features"])
    a.add_argument("--res", nargs="*", type=int, default=[25000, 10000, 5000])
    a.add_argument("--devices", nargs="+", default=["0", "0", "1", "1"])
    a = a.parse_args()
    if "cnv" in a.stage:
        stage_cnv()
    if "annotate" in a.stage:
        region_flags()
    if "scores" in a.stage:
        stage_scores(a.res)
    if "learn" in a.stage:
        stage_learn(a.res)
    if "features_predict" in a.stage:
        stage_features_predict(a.devices)
    if "features" in a.stage:
        stage_features()


if __name__ == "__main__":
    main()
