"""Hard windows: where a cell's map differs most from the consensus it reads.

Deviation score of a window (or region) of cell c, on the upper triangle
(offset >= 2 bins) of the measured proper O/E map T and the consensus C that
c reads (leave-one-out for the 12 training cells, consensus12 for K562,
HMEC and HUVEC), over pixels measured in both and in no masked bin:

    score = 0.5 * (1 - Pearson(T, C)) + 0.5 * var(T - C) / var(T)

Noise-aware filters (the worst 2 kb regions of section 16 are sparse target
noise, not cell-specific structure):
  coverage     mean raw contacts per upper-triangle pixel (unbalanced counts of
               the cell's own matrix: ENCODE .hic conversions in encode_hic/,
               the 4DN mcools of encode_expand.LOCAL); the bottom 15% of the
               cell's windows are removed
  masked       more than 20% of the window's bins without a balancing weight
               (or without contacts) are removed
  replicates   where two replicate maps of the cell exist (GM12878 and K562
               deep intact pairs, IMR90, HMEC and HUVEC single-lane pairs, all
               oe_proper), the deviation must reproduce:
               Pearson(R1 - C, R2 - C) >= 0.3
Every filter's removals are counted.

Stages
  quick     no retraining: on the 413 Akita test regions, per cell the score,
            the filters and, for every existing 12-cell model at 25 and 10 kb,
            the prediction (TTA, as crosscell_predict) with Pearson, SCC and
            the residual correlation corr(P - C, T - C) (0: the model adds
            nothing beyond the consensus); also the consensus alone, the best
            other measured map and the replicate ceiling (replicate maps
            against each other, and their residual correlation). Summary over
            each cell's hardest 20% of passing regions and over all passing
            regions: runs/hard_split/quick_*.csv.

    python scripts/hard_split.py --stage quick --jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multicell_expanded as mx                                      # noqa: E402

REPO, PY, OE, HM = mx.REPO, mx.PY, mx.OE, mx.HM
OUT = HM / "runs/hard_split"
HELD = ["k562", "hmec_deep", "huvec_deep"]
CELLS = mx.TRAIN12 + HELD
RES = [25000, 10000]
FAMILIES = ["all12refres", "all12honest", "all12refres_drop", "all12honest_drop",
            "all12refres_refdrop", "all12multires"]
REF_FAMILIES = {"all12refres", "all12refres_drop", "all12refres_refdrop", "all12multires"}
COVERAGE_Q, MASK_MAX, REP_MIN, HARD_FRAC = 0.15, 0.20, 0.3, 0.20


def rep_pair(cell: str, res: int):
    k = res // 1000
    pairs = {
        "gm12878": ("intact_gm12878_ENCSR653JLK_r7", "intact_gm12878_ENCSR653JLK_r9"),
        "k562": ("intact_k562_ENCSR676BEP_r5", "intact_k562_ENCSR676BEP_r7"),
        "imr90": ("imr90_rep1", "imr90_rep2"),
        "hmec_deep": ("hmec_rep1", "hmec_rep2"),
        "huvec_deep": ("huvec_rep1", "huvec_rep2"),
    }
    if cell not in pairs:
        return None
    p = [OE / f"{x}_ice_{k}kb.cool" for x in pairs[cell]]
    return p if all(q.exists() for q in p) else None


def raw_uri(cell: str, res: int) -> str:
    import encode_expand as ee
    k = res // 1000
    enc = HM / "encode_hic" / f"{cell}_{k}kb.cool"
    if enc.exists():
        return str(enc)
    if cell in ee.LOCAL:
        return f"{ee.LOCAL[cell]}::/resolutions/{res}"
    raise SystemExit(f"no raw matrix for {cell} at {res}")


def consensus_of(cell: str, res: int) -> Path:
    return mx.consensus_loo(cell, res) if cell in mx.TRAIN12 else mx.consensus12(res)


def fetch(clr, chrom, s, e, balance=False):
    from hicmultiome.genome import match_name
    nm = match_name(chrom, list(clr.chromnames))
    m = np.asarray(clr.matrix(balance=balance).fetch(f"{nm}:{s}-{e}"), np.float64)
    return m


def masked_bins(raw, chrom, s, e):
    from hicmultiome.genome import match_name
    nm = match_name(chrom, list(raw.chromnames))
    b = raw.bins().fetch(f"{nm}:{s}-{e}")
    if "weight" in b.columns:
        return ~np.isfinite(b["weight"].values)
    M = fetch(raw, chrom, s, e)
    return np.nan_to_num(M).sum(1) == 0


def tri(n, k=2):
    i, j = np.indices((n, n))
    return (j - i) >= k


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return np.nan
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def region_stats(T, C, raw_counts, mbins, reps):
    """Score and filter quantities of one region (all maps on one bin grid)."""
    n = T.shape[0]
    tm = tri(n)
    good = ~mbins
    valid = (tm & good[:, None] & good[None, :] & np.isfinite(T) & (T != 0)
             & np.isfinite(C) & (C != 0))
    t, c = T[valid], C[valid]
    rec = {"n_pixels": int(valid.sum()),
           "masked_frac": float(mbins.mean()),
           "coverage": float(np.nanmean(np.nan_to_num(raw_counts)[tm])) if raw_counts is not None else np.nan}
    if valid.sum() >= 100 and t.std() > 0:
        rec["r_consensus"] = corr(t, c)
        rec["resid_share"] = float(np.var(t - c) / np.var(t))
        rec["score"] = 0.5 * (1 - rec["r_consensus"]) + 0.5 * rec["resid_share"]
    if reps is not None:
        R1, R2 = reps
        ok = valid & np.isfinite(R1) & (R1 != 0) & np.isfinite(R2) & (R2 != 0)
        rec["rep_dev_r"] = corr((R1 - C)[ok], (R2 - C)[ok])
        rec["rep_r"] = corr(R1[ok], R2[ok])
    return rec, valid


# --------------------------------------------------------------------------- #
# prediction of the existing models (crosscell_predict logic, tracks shared)
# --------------------------------------------------------------------------- #
HARD_FAMILIES = ["all12refres_drop_hard", "all12honest_drop_hard", "all12refres_refdrop_hard"]


def model_runs(res, families=None):
    out = {}
    for fam in (families or FAMILIES):
        run = mx.REF_OUT / mx.run_name(fam, res)
        if not (run / "best").exists():
            continue
        if fam == "all12multires":
            lg = mx.REF_OUT / "all12multires_train.log"
            ok = lg.exists() and "done." in lg.read_text(errors="replace")
        else:
            si = mx.stop_info(mx.run_name(fam, res), 10)
            ok = bool(si and si[2] == "patience")
        if ok:                                       # finished trainings only
            out[fam] = run
    return out


def load_bundle(run: Path, device, cell: str, res: int):
    import torch
    import score_like_hicgan as sc
    from hicmultiome.config import Config
    from hicmultiome.matrix import MatrixNormalizer
    from hicmultiome.train import _select_rows

    ck = run / (run / "best").read_text().strip()
    cfg = Config.load(ck / "config.yaml")
    model, channels = sc.load_model(cfg, ck, device)
    sel = None
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    n_motifs = len(channels["motif_ids"])
    if cfg.train.select_tracks is not None or cfg.train.select_motifs is not None:
        sel, offset, n_motifs, _ = _select_rows(cfg, channels)
    nm = ck / f"norm_matrix_{cell}.json"
    if not nm.exists():
        nm = ck / f"norm_matrix_{cfg.get_sample(None).name}.json"
    return {"cfg": cfg, "model": model, "sel": sel, "offset": offset, "n_motifs": n_motifs,
            "mat_norm": MatrixNormalizer.load(nm), "channels": channels,
            "size": cfg.window.size, "ref": cfg.model.reference != "none", "ck": ck}


def predict_region(b, sym_full, st, ref_block, device, reg_start, n_out, res, tgt_bin):
    import torch
    import score_like_hicgan as sc
    from hicmultiome.dataset import reverse_cond, split_cond

    size, off, nm = b["size"], b["offset"], b["n_motifs"]
    s = sym_full if b["sel"] is None else sym_full[b["sel"]]
    ct = s[:, st:st + size]
    if ct.shape[1] < size:
        ct = np.concatenate([ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
    cond = torch.from_numpy(ct[None]).to(device)
    pairs = cond[:, off:off + 2 * nm].reshape(1, nm, 2, size) if nm else None
    ref = torch.from_numpy(ref_block[None]).to(device) if b["ref"] else None
    with torch.no_grad():
        o = b["model"](cond, pairs, None, None, ref)
        cr = reverse_cond(cond, off, nm)
        sr, pr = split_cond(cr, off, nm)
        rr = torch.flip(ref, dims=[-1, -2]) if ref is not None else None
        o = 0.5 * (o + torch.flip(b["model"](sr, pr, None, None, rr), dims=[-1, -2]))
    P = b["mat_norm"].decode(o[0, 0].float().cpu().numpy())
    P = 0.5 * (P + P.T)
    return sc.resample(P, st * res, res, reg_start, n_out, tgt_bin)


def quick_task(cell: str, res: int, device: str, regions=None, families=None, out=None):
    """All regions of one cell at one resolution, every model (default: the
    413 Akita test regions and the existing models)."""
    import cooler
    import torch
    import score_like_hicgan as sc
    from hicmultiome.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
    from hicmultiome.reference import reference_block
    from hicmultiome.tracks import TrackNormalizer
    sys.path.insert(0, str(REPO / "scripts"))
    import benchmark_metrics as bm

    out = out or OUT / "quick" / f"{cell}_{res // 1000}kb.csv"
    if out.exists():
        return
    runs = model_runs(res, families)
    bundles = {f: load_bundle(r, device, cell, res) for f, r in runs.items()}
    any_b = next(iter(bundles.values()))
    cfg = any_b["cfg"]
    track_norm = TrackNormalizer.load(any_b["ck"] / "norm_tracks.json")
    absent = track_norm.absent_values()
    sample = cfg.get_sample(None)
    sample.bigwig_dir = str(mx.CELLS[cell][1])
    have = {p.stem.upper() for p in Path(sample.bigwig_dir).glob("*.bigwig")}
    missing = [t for t in any_b["channels"]["tracks"] if t.upper() not in have]
    sample.missing_tracks = list(missing)
    track_files = sample_track_files(cfg, sample)
    row_of = {t: i for i, t in enumerate(any_b["channels"]["tracks"])}

    tgt = cooler.Cooler(str(mx.target(cell, res)))
    cons = cooler.Cooler(str(consensus_of(cell, res)))
    raw = cooler.Cooler(raw_uri(cell, res))
    rp = rep_pair(cell, res)
    reps = [cooler.Cooler(str(p)) for p in rp] if rp else None
    # other cells' measured maps: from the section 15 crosscell table of the
    # same regions and target when it exists (per-region base_<cell> columns),
    # else fetched here
    base_tab = None
    cc = mx.REF_OUT / mx.run_name("all12refres", res) / "crosscell" / f"predict_{cell}.csv"
    if regions is None and cc.exists():
        bt = pd.read_csv(cc).set_index("region")
        base_tab = bt[[c for c in bt.columns if c.startswith("base_")
                       and not c.startswith("base_consensus")]]
    others = ({} if base_tab is not None else
              {o: cooler.Cooler(str(mx.target(o, res))) for o in mx.BASELINE_MAPS
               if mx.CELLS[o][3] != mx.CELLS[cell][3] and mx.target(o, res).exists()})
    if regions is None:
        regions = pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2],
                              names=["chrom", "start", "end"])
    rows = []
    for chrom, grp in regions.groupby("chrom"):
        from hicmultiome.genome import match_name
        if match_name(chrom, list(tgt.chromnames)) is None:
            continue
        nb = chrom_nbins_of(tgt, match_name(chrom, list(tgt.chromnames)))
        sym_full = track_norm.apply(assemble_chrom(cfg, sample, track_files, chrom, nb))
        for t in missing:
            sym_full[row_of[t]] = absent[row_of[t]]
        for r in grp.itertuples():
            s, e = int(r.start), int(r.end)
            T = fetch(tgt, chrom, s, e)
            C = fetch(cons, chrom, s, e)
            if C.shape != T.shape:
                continue
            RC = fetch(raw, chrom, s, e)
            mb = masked_bins(raw, chrom, s, e)
            if len(mb) != T.shape[0]:
                mb = np.resize(mb, T.shape[0])
            R = [fetch(x, chrom, s, e) for x in reps] if reps else None
            if R is not None and any(x.shape != T.shape for x in R):
                R = None                              # replicate grid differs at an edge
            if RC.shape != T.shape:
                RC = None
            rec, valid = region_stats(T, C, RC, mb, R)
            rec.update({"cell": cell, "res_kb": res // 1000, "region": f"{chrom}:{s}-{e}"})
            if "score" not in rec:
                rows.append(rec)
                continue
            mask = valid
            rec["consensus_pearson"] = bm.pearson(C, T, mask)
            rec["consensus_scc"] = bm.scc(C, T, mask)
            key = f"{chrom}:{s}-{e}"
            if base_tab is not None and key in base_tab.index:
                for c_, v in base_tab.loc[key].items():
                    rec[f"map_{c_[5:]}"] = v
            for o, oc in others.items():
                O = fetch(oc, chrom, s, e)
                if O.shape == T.shape:
                    rec[f"map_{o}"] = bm.pearson(np.where(O == 0, np.nan, O), T, mask)
            if R is not None:
                rec["replicate_pearson"] = bm.pearson(R[0], R[1], mask)
                ok = mask & np.isfinite(R[0]) & np.isfinite(R[1])
                rec["replicate_resid_corr"] = corr((R[0] - C)[ok], (R[1] - C)[ok])
            n_out = T.shape[0]
            centre = (s + e) // 2 // res
            for fam, b in bundles.items():
                size = b["size"]
                st = max(0, min(nb - size, centre - size // 2))
                rb = (reference_block(cons, chrom, st, size, b["mat_norm"])
                      if b["ref"] else None)
                P = predict_region(b, sym_full, st, rb, device, s, n_out, res, tgt.binsize)
                cen = s + (np.arange(n_out) + 0.5) * tgt.binsize
                cov = (cen >= st * res) & (cen < (st + size) * res)
                m = mask & cov[:, None] & cov[None, :]
                rec[f"{fam}_pearson"] = bm.pearson(P, T, m)
                rec[f"{fam}_scc"] = bm.scc(P, T, m)
                rec[f"{fam}_resid_corr"] = corr((P - C)[m], (T - C)[m])
            rows.append(rec)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(out)


def quick(jobs: int):
    tasks = [(c, r) for r in RES for c in CELLS
             if not (OUT / "quick" / f"{c}_{r // 1000}kb.csv").exists()]
    procs = []
    k = 0
    while tasks or procs:
        procs = [p for p in procs if p.poll() is None]
        while tasks and len(procs) < jobs:
            c, r = tasks.pop(0)
            gpu = str(k % 2)
            k += 1
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="3",
                       MKL_NUM_THREADS="3")
            logf = OUT / "quick" / f"{c}_{r // 1000}kb.log"
            logf.parent.mkdir(parents=True, exist_ok=True)
            procs.append(subprocess.Popen([PY, __file__, "--stage", "quick_task", "--cell", c,
                                           "--res", str(r)], env=env, cwd=REPO,
                                          stdout=open(logf, "w"), stderr=subprocess.STDOUT))
        time.sleep(10)
    quick_summary()


def apply_filters(df: pd.DataFrame, cell_col="cell") -> tuple[pd.DataFrame, list[dict]]:
    """Noise-aware filters per cell and resolution; returns kept rows and the
    number each filter removed (applied in order: scored, masked, coverage,
    replicate reproducibility)."""
    keep, counts = [], []
    for (cell, rk), g in df.groupby([cell_col, "res_kb"]):
        n0 = len(g)
        g = g[g["score"].notna()]
        n_scored = len(g)
        g = g[g["masked_frac"] <= MASK_MAX]
        n_mask = len(g)
        thr = g["coverage"].quantile(COVERAGE_Q)
        g = g[g["coverage"] > thr]
        n_cov = len(g)
        if "rep_dev_r" in g and g["rep_dev_r"].notna().any():
            g = g[g["rep_dev_r"] >= REP_MIN]
        counts.append({"cell": cell, "res_kb": rk, "candidates": n0,
                       "removed_unscored": n0 - n_scored, "removed_masked": n_scored - n_mask,
                       "removed_low_coverage": n_mask - n_cov,
                       "removed_not_reproducible": n_cov - len(g), "kept": len(g)})
        keep.append(g)
    return pd.concat(keep), counts


def quick_summary():
    files = sorted((OUT / "quick").glob("*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    kept, counts = apply_filters(df)
    pd.DataFrame(counts).to_csv(OUT / "quick_filters.csv", index=False)
    rows = []
    for (cell, rk), g in kept.groupby(["cell", "res_kb"]):
        hard = g.nlargest(max(1, int(round(HARD_FRAC * len(g)))), "score")
        for sub_name, sub in (("hardest20", hard), ("all_passing", g)):
            rec = {"cell": cell, "res_kb": rk, "subset": sub_name, "n": len(sub),
                   "trained_on_cell": cell in mx.TRAIN12,
                   "score": sub["score"].median(),
                   "consensus_pearson": sub["consensus_pearson"].median(),
                   "consensus_scc": sub["consensus_scc"].median()}
            # best other map: the single other cell with the highest median
            # over this subset (not a per-region choice)
            meds = {c[4:]: sub[c].median() for c in sub.columns
                    if c.startswith("map_") and sub[c].notna().any()}
            if meds:
                bo = max(meds, key=meds.get)
                rec["best_other_map"], rec["best_other_map_cell"] = meds[bo], bo
            for c in ("replicate_pearson", "replicate_resid_corr"):
                if c in sub:
                    rec[c] = sub[c].median()
            for fam in FAMILIES:
                for m in ("pearson", "scc", "resid_corr"):
                    col = f"{fam}_{m}"
                    if col in sub:
                        rec[col] = sub[col].median()
            rows.append(rec)
    s = pd.DataFrame(rows).round(4)
    s.to_csv(OUT / "quick_per_cell.csv", index=False)
    agg = []
    for (rk, sub_name, trained), g in s.groupby(["res_kb", "subset", "trained_on_cell"]):
        rec = {"res_kb": rk, "subset": sub_name,
               "cells": "12 training cells" if trained else "held-out cells", "n_cells": len(g)}
        for c in s.columns:
            if (c.endswith(("pearson", "scc", "resid_corr")) or c in ("best_other_map", "score")) \
                    and pd.api.types.is_numeric_dtype(s[c]):
                rec[c] = round(float(g[c].median()), 4) if g[c].notna().any() else None
        agg.append(rec)
    pd.DataFrame(agg).to_csv(OUT / "quick_summary.csv", index=False)
    print(pd.DataFrame(agg).T.to_string(), flush=True)


# --------------------------------------------------------------------------- #
# genome-wide candidate windows and the hard split
# --------------------------------------------------------------------------- #
# 1.5% of the genome per cell, preferring (bonus 0.2 on the score) windows
# another cell already claimed: the requested 3% without sharing removed 62%
# (25 kb) and 59% (10 kb) of all training windows through the union over 15
# cells; this choice removes 33% and 29% at 0.99 and 0.95 of the pure top
# windows' median score (runs/hard_split/options/tradeoff.csv)
GENOME_FRAC = 0.015
SHARE_BONUS = 0.2
WINDOW = {25000: 80, 10000: 192}
SPLIT_DIR = REPO / "results/hard_split"


def cache_windows(res: int) -> dict:
    """{sample: [(chrom, start_bin), ...]} of the train split of the 12-cell
    cache at res (the candidate grid: half-window stride inside Akita's
    training regions, disjoint from the 413 test regions)."""
    m = json.loads((mx.CACHE / f"mc_{mx.run_name('all12', res)}" / "manifest.json").read_text())
    out = {}
    for sh in m["shards"]:
        if sh["split"] != "train":
            continue
        st = np.load(sh["path"])["start"]
        out.setdefault(sh["sample"], []).extend((sh["chrom"], int(x)) for x in st)
    return out


def band(clr, chrom, W):
    """(nbins, W) array of a banded O/E map, [i, d] = value at (i, i + d);
    NaN where no pixel is stored."""
    from hicmultiome.genome import match_name
    nm = match_name(chrom, list(clr.chromnames))
    lo, hi = clr.extent(nm)
    n = hi - lo
    A = np.full((n, W), np.nan, np.float32)
    px = clr.matrix(balance=False, as_pixels=True).fetch(nm)
    i = px["bin1_id"].values - lo
    d = px["bin2_id"].values - px["bin1_id"].values
    k = (d >= 0) & (d < W)
    A[i[k], d[k]] = px["count"].values[k]
    A[A == 0] = np.nan
    return A


def score_task(cell: str, res: int):
    """Score every candidate window of one cell (training cells: its cache
    windows; held-out cells: the union of all cells' windows)."""
    import cooler

    out = OUT / "windows" / f"{cell}_{res // 1000}kb.csv"
    if out.exists():
        return
    W = WINDOW[res]
    cw = cache_windows(res)
    wins = (cw[cell] if cell in cw else
            sorted({w for v in cw.values() for w in v}))
    tgt = cooler.Cooler(str(mx.target(cell, res)))
    cons = cooler.Cooler(str(consensus_of(cell, res)))
    raw = cooler.Cooler(raw_uri(cell, res))
    rp = rep_pair(cell, res)
    reps = [cooler.Cooler(str(q)) for q in rp] if rp else None
    iw, dw = np.indices((W, W))
    tri_band = (dw >= 2) & (iw + dw < W)            # [local row, offset] inside the window
    rows = []
    from hicmultiome.genome import match_name
    for chrom in sorted({c for c, _ in wins}, key=lambda c: (len(c), c)):
        T, C = band(tgt, chrom, W), band(cons, chrom, W)
        R = [band(x, chrom, W) for x in reps] if reps else None
        rn = match_name(chrom, list(raw.chromnames))
        b = raw.bins().fetch(rn)
        good = (np.isfinite(b["weight"].values) if "weight" in b.columns else None)
        starts = [s for c, s in wins if c == chrom]
        for s in starts:
            e = s + W
            t, c_ = T[s:e], C[s:e]
            if t.shape[0] < W:
                continue
            if good is not None:
                g = good[s:e]
            else:
                g = np.ones(W, bool)
            gi = g[:, None] & g[np.clip(np.arange(W)[:, None] + np.arange(W)[None, :], 0, W - 1)]
            valid = tri_band & gi & np.isfinite(t) & np.isfinite(c_)
            rec = {"cell": cell, "res_kb": res // 1000, "chrom": chrom, "start_bin": s,
                   "masked_frac": float(1 - g.mean())}
            try:
                M = np.nan_to_num(np.asarray(raw.matrix(balance=False).fetch(
                    f"{rn}:{s * res}-{min(e * res, int(raw.chromsizes[rn]))}"), np.float64))
                rec["coverage"] = float(M[np.triu_indices(M.shape[0], 2)].mean())
                if good is None:
                    g2 = M.sum(1) > 0
                    rec["masked_frac"] = float(1 - g2.mean())
            except Exception:                         # noqa: BLE001
                rec["coverage"] = np.nan
            tv, cv = t[valid], c_[valid]
            if valid.sum() >= 100 and tv.std() > 0 and cv.std() > 0:
                rec["r_consensus"] = float(np.corrcoef(tv, cv)[0, 1])
                rec["resid_share"] = float(np.var(tv - cv) / np.var(tv))
                rec["score"] = 0.5 * (1 - rec["r_consensus"]) + 0.5 * rec["resid_share"]
            if R is not None:
                r1, r2 = R[0][s:e], R[1][s:e]
                ok = (valid & np.isfinite(r1) & np.isfinite(r2)
                      if r1.shape == valid.shape and r2.shape == valid.shape else None)
                if ok is not None and ok.sum() >= 100:
                    rec["rep_dev_r"] = corr((r1 - c_)[ok], (r2 - c_)[ok])
            rows.append(rec)
        print(f"{cell} {res} {chrom}: {len(starts)} windows", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(out)


def run_tasks(stage: str, tasks, jobs: int, gpu=False):
    procs, k = [], 0
    tasks = list(tasks)
    while tasks or procs:
        procs = [p for p in procs if p.poll() is None]
        while tasks and len(procs) < jobs:
            c, r = tasks.pop(0)
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       CUDA_VISIBLE_DEVICES=str(k % 2) if gpu else "")
            k += 1
            logf = OUT / stage / f"{c}_{r // 1000}kb.log"
            logf.parent.mkdir(parents=True, exist_ok=True)
            procs.append(subprocess.Popen([PY, __file__, "--stage", f"{stage}_task", "--cell", c,
                                           "--res", str(r)], env=env, cwd=REPO,
                                          stdout=open(logf, "w"), stderr=subprocess.STDOUT))
        time.sleep(10)


def select(frac=None, union_heldout=True, share_bonus=None, out_dir=None, write=True):
    """Per cell and resolution: filters, then the top-scoring non-overlapping
    windows covering GENOME_FRAC of the genome, at most a length-proportional
    share per chromosome (twice the chromosome's share, at least 2);
    training cells: odd chromosomes hard-validation, even chromosomes and chrX
    hard-test; held-out cells: every hard window is hard-test. Writes BED
    files and split_<k>kb.json (per cell val/test start bins and the union of
    all hard positions, which training removes for every cell)."""
    import cooler

    miss = [f"{c}_{r // 1000}kb" for r in RES for c in CELLS
            if not (OUT / "windows" / f"{c}_{r // 1000}kb.csv").exists()]
    if miss:
        raise SystemExit(f"window scores missing for {miss}; rerun --stage score")
    allw = pd.concat([pd.read_csv(f) for f in sorted((OUT / "windows").glob("*.csv"))],
                     ignore_index=True)
    kept, counts = apply_filters(allw)
    stats = []
    for res in RES:
        k, W = res // 1000, WINDOW[res]
        sizes = cooler.Cooler(str(mx.target("gm12878", res))).chromsizes
        auto = [c for c in sizes.index if c.replace("chr", "") in [str(i) for i in range(1, 23)] + ["X"]]
        glen = float(sizes[auto].sum())
        n_target = int(np.ceil((frac or GENOME_FRAC) * glen / (W * res)))
        claimed = {}                    # positions already hard for an earlier cell
        split = {"res": res, "window": W, "cells": {}, "union": [],
                 "rule": {"score": "0.5*(1-r(T,C)) + 0.5*var(T-C)/var(T)",
                          "coverage_bottom_removed": COVERAGE_Q, "masked_max": MASK_MAX,
                          "replicate_dev_r_min": REP_MIN, "genome_frac": frac or GENOME_FRAC,
                          "share_bonus": SHARE_BONUS if share_bonus is None else share_bonus,
                          "val": "odd chromosomes (training cells)",
                          "test": "even chromosomes and chrX; every window of a held-out cell"}}
        d = (out_dir or SPLIT_DIR) / f"{k}kb"
        d.mkdir(parents=True, exist_ok=True)
        for cell in CELLS:
            g = kept[(kept.cell == cell) & (kept.res_kb == k)].copy()
            share_bonus = SHARE_BONUS if share_bonus is None else share_bonus
            if share_bonus:
                # prefer windows another cell already claimed: same loci, smaller union
                g["rank_score"] = g["score"] + share_bonus * g.apply(
                    lambda r: any(abs(r.start_bin - x) < W for x in claimed.get(r.chrom, ())),
                    axis=1)
            else:
                g["rank_score"] = g["score"]
            g = g.sort_values("rank_score", ascending=False)
            cap = {c: max(2, int(np.ceil(2 * n_target * sizes[c] / glen))) for c in auto}
            chosen, per_chrom = [], {}
            for r in g.itertuples():
                if len(chosen) >= n_target:
                    break
                if per_chrom.get(r.chrom, 0) >= cap.get(r.chrom, 0):
                    continue
                if any(c == r.chrom and abs(s - r.start_bin) < W for c, s, _ in chosen):
                    continue
                chosen.append((r.chrom, int(r.start_bin), float(r.score)))
                per_chrom[r.chrom] = per_chrom.get(r.chrom, 0) + 1
            held = cell not in mx.TRAIN12
            parts = {"val": [], "test": []}
            for c, s, sc_ in chosen:
                odd = c.replace("chr", "").isdigit() and int(c.replace("chr", "")) % 2 == 1
                parts["test" if held or not odd else "val"].append((c, s, sc_))
            split["cells"][cell] = {p: [[c, s] for c, s, _ in v] for p, v in parts.items()}
            for p, v in (parts.items() if write else []):
                with open(d / f"{cell}.{p}.bed", "w") as fh:
                    for c, s, sc_ in sorted(v):
                        fh.write(f"{c}\t{s * res}\t{(s + W) * res}\t{sc_:.4f}\n")
            for c, s_, _ in chosen:
                claimed.setdefault(c, []).append(s_)
            if union_heldout or not held:
                split["union"] += [[c, s, s + W] for c, s, _ in chosen]
            stats.append({"res_kb": k, "cell": cell, "target_windows": n_target,
                          "hard_windows": len(chosen), "hard_val": len(parts["val"]),
                          "hard_test": len(parts["test"]), "chromosomes": len(per_chrom),
                          "median_score_hard": float(np.median([x[2] for x in chosen])) if chosen else None,
                          "median_score_all": float(g["score"].median()) if len(g) else None})
        tmp = d / "split.json.part"
        tmp.write_text(json.dumps(split))
        tmp.rename(d / "split.json")
    if write:
        pd.DataFrame(counts).to_csv((out_dir or SPLIT_DIR) / "filters.csv", index=False)
        pd.DataFrame(stats).to_csv((out_dir or SPLIT_DIR) / "selection.csv", index=False)
    print(pd.DataFrame(stats).to_string(), flush=True)


def hardeval_task(cell: str, res: int, device: str):
    """Hard-test windows of one cell: the hard-split models and, for
    comparison, the models trained on those loci (flagged in the summary)."""
    k = res // 1000
    rows = []
    for c, a, b in [(l.split()[0], int(l.split()[1]), int(l.split()[2])) for l in
                    open(SPLIT_DIR / f"{k}kb" / f"{cell}.test.bed") if l.strip()]:
        rows.append((c, a, b))
    if not rows:
        return
    regs = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    quick_task(cell, res, device, regions=regs, families=HARD_FAMILIES + FAMILIES,
               out=OUT / "hardtest" / f"{cell}_{k}kb.csv")


def hard_summary():
    files = sorted((OUT / "hardtest").glob("*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fams = [f for f in HARD_FAMILIES + FAMILIES if f"{f}_pearson" in df]
    rows = []
    for (cell, rk), g in df.groupby(["cell", "res_kb"]):
        rec = {"cell": cell, "res_kb": rk, "n": len(g), "trained_on_cell": cell in mx.TRAIN12,
               "score": g["score"].median(), "consensus_pearson": g["consensus_pearson"].median(),
               "consensus_scc": g["consensus_scc"].median()}
        meds = {c[4:]: g[c].median() for c in g.columns if c.startswith("map_")}
        if meds:
            bo = max(meds, key=meds.get)
            rec["best_other_map"], rec["best_other_map_cell"] = meds[bo], bo
        for c in ("replicate_pearson", "replicate_resid_corr"):
            if c in g:
                rec[c] = g[c].median()
        for f in fams:
            for m in ("pearson", "scc", "resid_corr"):
                rec[f"{f}_{m}"] = g[f"{f}_{m}"].median()
        rows.append(rec)
    s = pd.DataFrame(rows).round(4)
    s.to_csv(SPLIT_DIR / "hardtest_per_cell.csv", index=False)
    agg = []
    for (rk, tr), g in s.groupby(["res_kb", "trained_on_cell"]):
        rec = {"res_kb": rk, "cells": "12 training cells" if tr else "held-out cells",
               "n_cells": len(g)}
        for c in s.columns:
            if c.endswith(("pearson", "scc", "resid_corr")) or c in ("best_other_map", "score"):
                if pd.api.types.is_numeric_dtype(s[c]) and g[c].notna().any():
                    rec[c] = round(float(g[c].median()), 4)
        agg.append(rec)
    pd.DataFrame(agg).to_csv(SPLIT_DIR / "hardtest_summary.csv", index=False)
    print(pd.DataFrame(agg).T.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["quick", "quick_task", "quick_summary", "score", "score_task",
                             "select", "hardeval", "hardeval_task", "hard_summary"])
    ap.add_argument("--cell")
    ap.add_argument("--res", type=int)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "quick_task":
        import torch
        quick_task(a.cell, a.res, "cuda" if torch.cuda.is_available() else "cpu")
    elif a.stage == "quick":
        quick(a.jobs)
    elif a.stage == "score_task":
        score_task(a.cell, a.res)
    elif a.stage == "score":
        run_tasks("score", [(c, r) for r in RES for c in CELLS
                            if not (OUT / "windows" / f"{c}_{r // 1000}kb.csv").exists()], a.jobs)
    elif a.stage == "select":
        select()
    elif a.stage == "hardeval_task":
        import torch
        hardeval_task(a.cell, a.res, "cuda" if torch.cuda.is_available() else "cpu")
    elif a.stage == "hardeval":
        run_tasks("hardeval", [(c, r) for r in RES for c in CELLS
                               if not (OUT / "hardtest" / f"{c}_{r // 1000}kb.csv").exists()],
                  a.jobs, gpu=True)
        hard_summary()
    elif a.stage == "hard_summary":
        hard_summary()
    else:
        quick_summary()


if __name__ == "__main__":
    main()
