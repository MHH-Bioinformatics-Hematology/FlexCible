"""HiC-Reg audit for the 25 kb tie of registry section 18 (CPU only).

The 25 kb HiC-Reg maps of section 18 are the authors' published GM12878
predictions (HiC-Reg Zenodo records 3525432 / 3525510 / 3525514, files
named <Cell>_chr<N>_RF_tree20_prediction.txt, the name of the CV mode),
converted to 5 kb cools (log(count + 1) values) and summed into 25 kb bins.

Stages:
  split     reconstruct HiC-Reg's cross-validation folds from the training
            pair lists stored in the released models (model<k>.tar.gz:
            out/model/trainset_error.txt) for the chromosomes whose models
            are released (chr14, chr17; GM12878 and K562; WINDOW and
            MULTICELL features): fold of every region, share of pairs in
            training, share of the 8 neighbouring 5 kb pixels of a predicted
            pixel that were training pairs of the model that predicted it.
  cools     5 kb predictions -> count cools (expm1 of log(count + 1)) at 10
            and 25 kb (sum of the 5 kb counts): the published chr1-22 GM12878
            set (from the existing 5 kb cools) and the released chr14/chr17
            CV, cross-chromosome (CrossChrom: trained on the other of chr14 and
            chr17) and cross-cell predictions; then oe_proper.py (1 Mb reach).
  blocks    benchmark blocks (benchmark_blocks.oe_block) with these maps:
            C2 = block C with the count-space HiC-Reg map added;
            F_<cell>_<k>kb = chr14 and chr17, HiC-Reg CV against HiC-Reg
            cross-chromosome against our models, GM12878 and K562.
  extra     per-distance band Pearson and SCC, local compartment (E1)
            correlation, cell-specific difference test (GM12878 minus K562),
            noise-corrected 2 kb scores.

    python scripts/hicreg_audit.py --stage split cools blocks extra
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import benchmark_competitors as bc                                   # noqa: E402

HR = Path("/mnt/processed/wolffjoa/home_offload/HiC-Reg")        # released per-cell data
PUB = bc.PP / "HiC-Reg"                                           # published chr1-22 GM12878
OUT = bc.OUT / "hicreg_audit"
RES = bc.RES_OUT
PY = sys.executable
CELLS = {"gm12878": "Gm12878", "k562": "K562"}
REACH = 1_000_000


def log(msg):
    bc.log(f"[hicreg] {msg}")


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def _pairs(f, cols=(0,)):
    d = pd.read_csv(f, sep="\t", skiprows=1, header=None, usecols=list(cols))
    s = d[0].str.replace("-", "_").str.split("_", expand=True)
    return (s[1].astype(np.int64).values // 5000, s[4].astype(np.int64).values // 5000)


def split_one(cell: str, feat: str, chrom: str, n_sample=200_000) -> dict | None:
    cv = HR / CELLS[cell] / feat / chrom / "CV"
    pred = cv / f"{CELLS[cell]}_{chrom}_RF_tree20_prediction.txt"
    models = [cv / f"model{k}.tar.gz" for k in range(5)]
    if not pred.exists() or not all(m.exists() for m in models):
        return None
    pi, pj = _pairs(pred)
    allkey = pi * 100_000 + pj
    train, regions_in = [], []
    for m in models:
        with tarfile.open(m) as t:
            try:
                f = t.extractfile("out/model/trainset_error.txt")
            except KeyError:          # the MULTICELL models ship without it
                return None
            i, j = _pairs(f)
        k = i * 100_000 + j
        train.append(set(k.tolist()))
        regions_in.append(set(np.unique(np.concatenate([i, j])).tolist()))
    allreg = set(np.unique(np.concatenate([pi, pj])).tolist())
    fold = {}
    for k in range(5):
        for r in allreg - regions_in[k]:
            fold.setdefault(r, set()).add(k)
    in_train = np.zeros(len(allkey), int)
    for k in range(5):
        in_train += np.isin(allkey, np.fromiter(train[k], np.int64))
    rs = sorted(r for r in fold if len(fold[r]) == 1)
    fv = np.array([next(iter(fold[r])) for r in rs])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pi), min(n_sample, len(pi)), replace=False)
    share, anyn, same, share_one, any_one = [], [], [], [], []
    for t in idx:
        i, j = pi[t], pj[t]
        ks = fold.get(i, set()) | fold.get(j, set())
        if not ks:
            continue
        # the pair is a test pair of every model whose held-out regions contain i or j;
        # the released prediction is of one of them: count neighbours trained by ALL of them
        nb = [(i + a) * 100_000 + (j + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
        tr = [all(n in train[k] for k in ks) for n in nb]
        share.append(np.mean(tr))
        anyn.append(any(tr))
        # upper bound: trained by at least one of the candidate models
        tr1 = [any(n in train[k] for k in ks) for n in nb]
        share_one.append(np.mean(tr1))
        any_one.append(any(tr1))
        same.append(any(((i + a) * 100_000 + j) in train[k] for k in ks for a in (-1, 1)))
    return {"cell": cell, "features": feat, "chrom": chrom, "pairs": len(allkey),
            "regions": len(allreg),
            "regions_held_out_per_fold": np.mean([len(allreg - r) for r in regions_in]),
            "pairs_in_training_of_k_models_3": float(np.mean(in_train == 3)),
            "pairs_in_training_of_k_models_4": float(np.mean(in_train == 4)),
            "train_share_per_model": float(np.mean([len(t) for t in train]) / len(allkey)),
            "fold_changes_between_adjacent_regions": float(np.mean(np.diff(fv) != 0)),
            "expected_if_random": 0.8,
            "neighbour_pixels_in_training_mean": float(np.mean(share)),
            "neighbour_pixels_in_training_mean_upper": float(np.mean(share_one)),
            "pixels_with_any_trained_neighbour_upper": float(np.mean(any_one)),
            "pixels_with_any_trained_neighbour": float(np.mean(anyn)),
            "pixels_with_trained_row_neighbour": float(np.mean(same))}


def stage_split():
    rows = []
    for cell in CELLS:
        for feat in ("WINDOW", "MULTICELL"):
            for chrom in ("chr14", "chr17"):
                r = split_one(cell, feat, chrom)
                if r:
                    rows.append(r)
                    log(f"split {cell} {feat} {chrom}: train share {r['train_share_per_model']:.3f}, "
                        f"neighbours trained {r['neighbour_pixels_in_training_mean']:.3f}")
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "hicreg_split.csv", index=False, float_format="%.4f")


# --------------------------------------------------------------------------- #
# cools
# --------------------------------------------------------------------------- #
HG19 = Path.home() / "src/HiC-Reg/hg19.chrom.sizes"


def chromsize(chrom):
    s = pd.read_csv(HG19, sep="\t", header=None, index_col=0)[1]
    return int(s[chrom])


def write_counts(chrom: str, b1: np.ndarray, b2: np.ndarray, logv: np.ndarray, out: Path, res: int):
    """log(count + 1) at 5 kb -> counts summed into res bins (upper triangle)."""
    f = res // 5000
    c = np.expm1(logv.astype(np.float64))
    df = pd.DataFrame({"bin1_id": b1 // f, "bin2_id": b2 // f, "count": c})
    df = df.groupby(["bin1_id", "bin2_id"], as_index=False)["count"].sum()
    size = chromsize(chrom)
    starts = np.arange(0, size, res)
    bins = pd.DataFrame({"chrom": chrom, "start": starts, "end": np.minimum(starts + res, size)})
    df = df[(df.bin2_id < len(bins)) & (df.bin1_id <= df.bin2_id)].sort_values(["bin1_id", "bin2_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cooler.create_cooler(str(out), bins=bins, pixels=df, dtypes={"count": np.float64}, ordered=True)


def released(cell, feat, chrom, mode, src_chrom=None, src_cell=None):
    d = HR / CELLS[cell] / feat / chrom
    if mode == "CV":
        return d / "CV" / f"{CELLS[cell]}_{chrom}_RF_tree20_prediction.txt"
    if mode == "CrossChrom":
        other = "chr17" if chrom == "chr14" else "chr14"
        return d / "CrossChrom" / f"{CELLS[cell]}_{other}to{chrom}_RF_tree20_prediction.txt"
    if mode == "CrossCell":
        p = d / "CrossCell" / f"{CELLS[src_cell]}to{CELLS[cell]}_{chrom}_RF_tree20_prediction.txt"
        if not p.exists():
            p = d / "CrossCell" / f"{CELLS[src_cell]}to{CELLS[cell]}" / p.name
        return p
    raise ValueError(mode)


def variants():
    """name -> (source kind, args); every name is a competitor of the blocks."""
    v = {}
    for res in (25000, 10000):
        k = res // 1000
        v[f"hicregcounts_gm12878_{k}kb"] = ("published", "gm12878", res)
        for cell in CELLS:
            for feat in ("WINDOW", "MULTICELL"):
                for mode in ("CV", "CrossChrom"):
                    v[f"hicreg{feat.lower()}{mode.lower()}_{cell}_{k}kb"] = ("released", cell, feat, mode, res)
            other = "k562" if cell == "gm12878" else "gm12878"
            v[f"hicregwindowcrosscell_{other}to{cell}_{k}kb"] = ("crosscell", cell, other, res)
    if os.environ.get("BENCH_5KB") == "1":
        # native 5 kb of the released predictions (no summing); registered only on
        # request so that the default variants are unchanged
        for cell in CELLS:
            for feat in ("WINDOW", "MULTICELL"):
                for mode in ("CV", "CrossChrom"):
                    v[f"hicreg{feat.lower()}{mode.lower()}_{cell}_5kb"] = ("released", cell, feat, mode, 5000)
    return v


def comp_spec(name, kind):
    res = int(name.rsplit("_", 1)[1][:-2]) * 1000
    cell = name.split("_")[-2].split("to")[-1]
    chroms = bc.EVEN if kind[0] == "published" else ["chr14", "chr17"]
    lab = {"published": "HiC-Reg (published, counts)"}.get(kind[0])
    if kind[0] == "released":
        lab = f"HiC-Reg {kind[2]} {'CV' if kind[3] == 'CV' else 'cross-chromosome'}"
    if kind[0] == "crosscell":
        lab = f"HiC-Reg WINDOW cross-cell from {kind[2]}"
    return dict(label=lab, path=OUT / "counts" / (name + "_{chrom}.cool"), per_chrom=True,
                build="hg19", res=res, reach=REACH, space="counts", heldout="see hicreg_split.csv",
                cell=cell, chroms=chroms)


def stage_cools():
    for name, kind in variants().items():
        spec = comp_spec(name, kind)
        for chrom in spec["chroms"]:
            out = Path(str(spec["path"]).format(chrom=chrom))
            if out.exists():
                continue
            if kind[0] == "published":
                src = PUB / f"Gm12878_{chrom}_RF_tree20_prediction_predicted_coords.cool"
                if not src.exists():
                    continue
                c = cooler.Cooler(str(src))
                px = c.pixels()[:]
                off = c.offset(chrom)
                write_counts(chrom, px.bin1_id.values - off, px.bin2_id.values - off,
                             px["count"].values, out, spec["res"])
            else:
                if kind[0] == "released":
                    f = released(kind[1], kind[2], chrom, kind[3])
                else:
                    f = released(kind[1], "WINDOW", chrom, "CrossCell", src_cell=kind[2])
                if not f.exists():
                    log(f"missing {f}")
                    continue
                d = pd.read_csv(f, sep="\t", skiprows=1, header=None, usecols=[0, 2])
                s = d[0].str.replace("-", "_").str.split("_", expand=True)
                b1 = s[1].astype(np.int64).values // 5000
                b2 = s[4].astype(np.int64).values // 5000
                lo, hi = np.minimum(b1, b2), np.maximum(b1, b2)
                write_counts(chrom, lo, hi, d[2].values, out, spec["res"])
            log(f"cool {out.name}")
    # through oe_proper.py, as every competitor of section 18
    for name, kind in variants().items():
        spec = comp_spec(name, kind)
        bc.COMPETITORS[name] = spec
        bc.run_oe(name, spec)


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
def register():
    for name, kind in variants().items():
        bc.COMPETITORS[name] = comp_spec(name, kind)


def block_defs():
    import benchmark_blocks as bb
    d = {"C2_gm12878_25kb_even_hicreg": lambda: bb.oe_block(
        "C2_gm12878_25kb_even_hicreg", "gm12878", 25000, bc.EVEN,
        ["hicreg_gm12878_25kb", "hicregcounts_gm12878_25kb", "hicgan_gm12878_25kb"],
        {"hicreg_gm12878_25kb": "published CV predictions, log values summed (section 18)",
         "hicregcounts_gm12878_25kb": "published CV predictions, counts summed"})}
    for cell in CELLS:
        other = "k562" if cell == "gm12878" else "gm12878"
        for res in (25000, 10000):
            k = res // 1000
            comps = [f"hicreg{f}{m}_{cell}_{k}kb" for f in ("window", "multicell")
                     for m in ("cv", "crosschrom")] + [f"hicregwindowcrosscell_{other}to{cell}_{k}kb"]
            bid = f"F_{cell}_{k}kb_chr14_17"
            d[bid] = (lambda b=bid, c=cell, r=res, cs=comps: bb.oe_block(
                b, c, r, ["chr14", "chr17"], cs,
                {x: "HiC-Reg release; CV = region-wise 5-fold CV within the chromosome, "
                    "cross-chromosome = model of the other chromosome" for x in cs}))
    return d


def stage_blocks():
    import benchmark_blocks as bb
    register()
    for bid, fn in block_defs().items():
        bb.BLOCKS[bid] = fn
        if not (bb.BLK / f"{bid}.npz").exists():
            fn()
        bb.score_block(bid)


# --------------------------------------------------------------------------- #
# extra analyses
# --------------------------------------------------------------------------- #
def local_e1(M: np.ndarray) -> np.ndarray | None:
    """First eigenvector of the Pearson correlation of a region's O/E rows."""
    X = np.where(np.isfinite(M), M, np.nan)
    ok = np.isfinite(X).mean(1) > 0.8
    if ok.sum() < 10:
        return None
    X = X[np.ix_(ok, ok)]
    X = np.where(np.isfinite(X), X, np.nanmean(X))
    C = np.corrcoef(X)
    C = np.nan_to_num(C)
    w, v = np.linalg.eigh(C)
    e = np.full(M.shape[0], np.nan)
    e[ok] = v[:, -1]
    return e


def stage_extra():
    import benchmark_blocks as bb
    import benchmark_metrics as bm
    register()
    blocks = ["C_gm12878_25kb_even", "C2_gm12878_25kb_even_hicreg"] + [
        b for b in block_defs() if b.startswith("F_")]
    band_rows, e1_rows = [], []
    for bid in blocks:
        if not (bb.BLK / f"{bid}.npz").exists():
            continue
        rows, meta = bb.load_block(bid)
        res = meta["res"]
        methods = sorted({k for r in rows.values() for k in r if k not in ("b0", "target")})
        for key, r in rows.items():
            T = r["target"].astype(float)
            n = T.shape[0]
            mask = bm.triu_mask(n, 2, meta.get("reach_bins")) & np.isfinite(T)
            for m in methods:
                if not m.startswith("map_"):
                    mask &= np.isfinite(r[m])
            eT = local_e1(T)
            for m in methods:
                P = r[m].astype(float)
                for d, (a, b) in bm.per_distance(P, T, mask).items():
                    if a.size >= 5 and a.std() > 0 and b.std() > 0:
                        band_rows.append({"block": bid, "method": m, "region": key,
                                          "distance_bp": d * res,
                                          "r": float(np.corrcoef(a, b)[0, 1]), "n": a.size})
                eP = local_e1(np.where(mask | mask.T, P, np.nan))
                if eT is not None and eP is not None:
                    ok = np.isfinite(eT) & np.isfinite(eP)
                    if ok.sum() >= 10:
                        e1_rows.append({"block": bid, "method": m, "region": key,
                                        "e1_abs_r": abs(float(np.corrcoef(eT[ok], eP[ok])[0, 1]))})
    B = pd.DataFrame(band_rows)
    # per band: median over regions of the per-region band Pearson, and the
    # pooled-pixel band Pearson is in <block>_per_distance.csv
    Bs = B.groupby(["block", "method", "distance_bp"]).r.agg(["median", "count"]).reset_index()
    Bs.to_csv(RES / "hicreg_distance_bands.csv", index=False, float_format="%.4f")
    E = pd.DataFrame(e1_rows)
    E.groupby(["block", "method"]).e1_abs_r.agg(["median", "mean", "count"]).reset_index().to_csv(
        RES / "hicreg_local_e1.csv", index=False, float_format="%.4f")
    log(f"bands {len(Bs)} rows, e1 {len(E)} rows")
    stage_cellspecific()
    stage_noise()


def stage_cellspecific():
    """GM12878 minus K562: measured difference against each method's predicted
    difference, on the regions and pixels shared by both cells' blocks."""
    import benchmark_blocks as bb
    import benchmark_metrics as bm
    rows = []
    for res in (25000, 10000):
        k = res // 1000
        bg, bk = f"F_gm12878_{k}kb_chr14_17", f"F_k562_{k}kb_chr14_17"
        if not ((bb.BLK / f"{bg}.npz").exists() and (bb.BLK / f"{bk}.npz").exists()):
            continue
        G, _ = bb.load_block(bg)
        K, _ = bb.load_block(bk)
        pairs = {"all12refres": ("all12refres", "all12refres"),
                 "all12honest": ("all12honest", "all12honest")}
        for f in ("window", "multicell"):
            for m in ("cv", "crosschrom"):
                pairs[f"hicreg_{f}_{m}"] = (f"hicreg{f}{m}_gm12878_{k}kb", f"hicreg{f}{m}_k562_{k}kb")
        pairs["hicreg_window_crosscell"] = (f"hicregwindowcrosscell_k562togm12878_{k}kb",
                                            f"hicregwindowcrosscell_gm12878tok562_{k}kb")
        for key in sorted(set(G) & set(K)):
            dT = G[key]["target"].astype(float) - K[key]["target"].astype(float)
            n = dT.shape[0]
            for lab, (mg, mk) in pairs.items():
                if mg not in G[key] or mk not in K[key]:
                    continue
                dP = G[key][mg].astype(float) - K[key][mk].astype(float)
                mask = bm.triu_mask(n, 2, REACH // res) & np.isfinite(dT) & np.isfinite(dP)
                rows.append({"res_kb": k, "region": key, "method": lab,
                             "diff_pearson": bm.pearson(dP, dT, mask),
                             "diff_scc": bm.scc(dP, dT, mask),
                             "pred_diff_sd": float(np.nanstd(dP[mask])),
                             "meas_diff_sd": float(np.nanstd(dT[mask]))})
    D = pd.DataFrame(rows)
    D.to_csv(RES / "hicreg_cellspecific_per_region.csv", index=False, float_format="%.4f")
    D.groupby(["res_kb", "method"])[["diff_pearson", "diff_scc", "pred_diff_sd", "meas_diff_sd"]] \
        .median().reset_index().to_csv(RES / "hicreg_cellspecific.csv", index=False,
                                       float_format="%.4f")


REPS = Path("/mnt/processed/wolffjoa/hic-multiome/runs/worst_regions/replicates_per_region.csv")


def stage_noise():
    """Per region r_model / sqrt(r_replicate) for the 2 kb GM12878 block.

    r_replicate: GM12878 replicate agreement at 2 kb on the same region
    (runs/worst_regions, proper O/E): the single-lane in situ pair (same
    protocol as the target, shallower) and the deepest intact pair
    (ENCSR965OVH); regions with r_replicate < 0.1 are left out. HFF has no
    replicate pair, so the HFF block has no corrected score."""
    reps = pd.read_csv(REPS)
    reps = reps[(reps.res_kb == 2) & (reps.cell == "gm12878")]
    rows = []
    for bid in ("A_gm12878_2kb",):
        df = pd.read_csv(RES / f"{bid}_per_region.csv.gz")
        df = df[df.method != "target_self"]
        for rset in ("gm12878_insitu_lane", "gm12878_intact_965OVH"):
            rr = reps[reps.set == rset].set_index("region").repA_vs_repB
            for m, g in df.groupby("method"):
                g = g.set_index("region")
                j = pd.concat([g.pearson, g.scc, rr.rename("rep")], axis=1, join="inner").dropna()
                j = j[j.rep >= 0.1]
                rows.append({"block": bid, "replicate_set": rset, "method": m, "n": len(j),
                             "raw_pearson_median": j.pearson.median(),
                             "corrected_pearson_median": (j.pearson / np.sqrt(j.rep)).median(),
                             "raw_scc_median": j.scc.median(),
                             "corrected_scc_median": (j.scc / np.sqrt(j.rep)).median(),
                             "replicate_r_median": j.rep.median()})
    pd.DataFrame(rows).to_csv(RES / "noise_corrected_2kb.csv", index=False, float_format="%.4f")


def stage_match():
    """Which released mode do the published chr1-22 maps come from? Distance-
    centred log-count Pearson on chr14 (25 kb, 50 kb to 1 Mb) between the
    published map and the released CV and cross-chromosome predictions."""
    rows = []
    d = OUT / "counts"
    P = cooler.Cooler(str(d / "hicregcounts_gm12878_25kb_chr14.cool")).pixels()[:] \
        .set_index(["bin1_id", "bin2_id"])["count"]
    for n in ("hicregwindowcv", "hicregmulticellcv", "hicregwindowcrosschrom",
              "hicregmulticellcrosschrom"):
        X = cooler.Cooler(str(d / f"{n}_gm12878_25kb_chr14.cool")).pixels()[:] \
            .set_index(["bin1_id", "bin2_id"])["count"]
        j = pd.concat([P.rename("a"), X.rename("b")], axis=1, join="inner").dropna()
        j = j[(j.a > 0) & (j.b > 0)]
        j["d"] = j.index.get_level_values(1) - j.index.get_level_values(0)
        j = j[(j.d >= 2) & (j.d <= 40)]
        raw = float(np.corrcoef(np.log(j.a), np.log(j.b))[0, 1])
        for c in "ab":
            j[c] = np.log(j[c])
            j[c] = j[c] - j.groupby("d")[c].transform("mean")
        rows.append({"chrom": "chr14", "released": n, "pixels": len(j), "log_pearson": raw,
                     "distance_centred_pearson": float(np.corrcoef(j.a, j.b)[0, 1])})
    pd.DataFrame(rows).to_csv(RES / "hicreg_published_vs_released.csv", index=False,
                              float_format="%.4f")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    a = ap.parse_args()
    for st in a.stage:
        globals()[f"stage_{st}"]()


if __name__ == "__main__":
    main()
