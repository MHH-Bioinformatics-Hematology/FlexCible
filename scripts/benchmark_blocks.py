"""Blocks of the competitor benchmark: assemble, score, figures.

Called by scripts/benchmark_competitors.py (--stage assemble score figures).
Each block fixes one target map, one bin grid, one region set and one method
set; a region enters a block only when every method of the block covers it,
and every metric of a region uses the pixels finite in the target and in every
method (upper triangle, at least 2 bins off the diagonal, within the shortest
distance reach of any method in the block).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_competitors as bc                                   # noqa: E402
import benchmark_metrics as bm                                       # noqa: E402
import multicell_expanded as mx                                      # noqa: E402
from flexcible import bioscore as bs                               # noqa: E402
from flexcible.liftover import LiftOver                            # noqa: E402

BLK = bc.OUT / "blocks"
_LIFT = None


def lift():
    global _LIFT
    if _LIFT is None:
        _LIFT = LiftOver(bc.CHAIN)
    return _LIFT


def regions() -> pd.DataFrame:
    r = pd.read_csv(bc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    r["key"] = r.chrom + ":" + r.start.astype(str) + "-" + r.end.astype(str)
    return r


# --------------------------------------------------------------------------- #
# grid parameters per resolution
# --------------------------------------------------------------------------- #
def params(res: int) -> dict:
    if res == 2048:
        return dict(ins=25, btol=5, flank=25,
                    loop=bs.BioParams(loop_coarsen=5, donut_inner=1, donut_outer=3,
                                      loop_min_dist=3, loop_max_dist=80, loop_match_tol=1,
                                      apa_half=3, apa_corner=1, loop_nms=1))
    if res == 2000:
        # the 10 kb row with every window in bins multiplied by 5 (spans in bp equal to the 10 kb row)
        return dict(ins=25, btol=5, flank=25,
                    loop=bs.BioParams(loop_coarsen=1, donut_inner=5, donut_outer=15,
                                      loop_min_dist=15, loop_max_dist=500, loop_match_tol=5,
                                      apa_half=15, apa_corner=5, loop_nms=5))
    if res == 10000:
        return dict(ins=5, btol=1, flank=5,
                    loop=bs.BioParams(loop_coarsen=1, donut_inner=1, donut_outer=3,
                                      loop_min_dist=3, loop_max_dist=100, loop_match_tol=1,
                                      apa_half=3, apa_corner=1, loop_nms=1))
    if res == 5000:
        # the 10 kb row with every window in bins doubled, so that each span in bp
        # (insulation square 50 kb, donut 10 to 30 kb, loop distance 30 kb to 1 Mb,
        # APA half-width 30 kb, matching tolerance 10 kb) equals the 10 kb row
        return dict(ins=10, btol=2, flank=10,
                    loop=bs.BioParams(loop_coarsen=1, donut_inner=2, donut_outer=6,
                                      loop_min_dist=6, loop_max_dist=200, loop_match_tol=2,
                                      apa_half=6, apa_corner=2, loop_nms=2))
    # donut outer 3: with outer 2 the donut (cross removed) keeps only its 4
    # corner pixels, fewer than the 8 local_enrichment requires
    return dict(ins=3, btol=1, flank=4,
                loop=bs.BioParams(loop_coarsen=1, donut_inner=1, donut_outer=3,
                                  loop_min_dist=2, loop_max_dist=40, loop_match_tol=1,
                                  apa_half=2, apa_corner=1, loop_nms=1))


# --------------------------------------------------------------------------- #
# map sources
# --------------------------------------------------------------------------- #
_BAND: dict = {}


def _band(clr: cooler.Cooler, name: str, n_chr: int) -> np.ndarray:
    """Whole-chromosome banded copy (bins x distance), read once: a cooler
    fetch over the network share costs ~0.8 s, the band a few seconds."""
    key = (clr.filename, clr.root, name)
    if key not in _BAND:
        if len(_BAND) > 40:
            _BAND.pop(next(iter(_BAND)))
        D = int(2_600_000 // clr.binsize) + 1
        B = np.zeros((n_chr, D + 1), np.float32)
        px = clr.matrix(balance=False, as_pixels=True).fetch(name)
        off = clr.offset(name)
        i = px.bin1_id.values - off
        j = px.bin2_id.values - off
        k = (j >= i) & (j - i <= D)
        B[i[k], (j - i)[k]] = px["count"].values[k]
        _BAND[key] = B
    return _BAND[key]


def fetch_oe(clr: cooler.Cooler, chrom: str, b0: int, b1: int) -> np.ndarray | None:
    name = next((c for c in clr.chromnames if c == chrom or c == chrom.replace("chr", "")), None)
    if name is None:
        return None
    n_chr = int(np.ceil(clr.chromsizes[name] / clr.binsize))
    if b1 > n_chr:
        return None
    res = clr.binsize
    if res >= 8000 and b0 >= 0:
        B = _band(clr, name, n_chr)
        n = b1 - b0
        i, j = np.indices((n, n))
        lo, d = np.minimum(i, j) + b0, np.abs(i - j)
        M = np.where(d < B.shape[1], B[lo, np.minimum(d, B.shape[1] - 1)], 0.0)
        return M.astype(float)
    M = np.asarray(clr.matrix(balance=False).fetch(
        f"{name}:{b0 * res}-{min(b1 * res, int(clr.chromsizes[name]))}"), float)
    if M.shape[0] < b1 - b0:
        P = np.full((b1 - b0, b1 - b0), np.nan)
        P[:M.shape[0], :M.shape[0]] = M
        M = P
    return M


def sample_competitor(spec: dict, name: str, chrom: str, b0: int, n: int, res: int
                      ) -> np.ndarray | None:
    """Competitor O/E sampled at the target bin centres (hg19: lifted centres)."""
    f = bc.comp_oe(name, chrom) if spec["per_chrom"] else bc.comp_oe(name)
    if not f.exists():
        return None
    clr = _cool(f)
    cres = clr.binsize
    centres = (b0 + np.arange(n) + 0.5) * res
    if spec["build"] == "hg19":
        lo = lift()
        pos = np.full(n, np.nan)
        for k, c in enumerate(centres):
            for dx in (0, -res / 4, res / 4):
                hit = lo.convert(chrom, int(c + dx))
                if hit is not None and hit[0] == chrom:
                    pos[k] = hit[1] - dx
                    break
        if np.isfinite(pos).sum() < 0.8 * n:
            return None
        # a region spanning a rearrangement between the builds is dropped
        d = np.diff(pos[np.isfinite(pos)])
        if (d <= 0).any() or np.abs(d - res).max() > res:
            return None
    else:
        pos = centres
    u = pos / cres - 0.5
    lo_b = int(np.floor(np.nanmin(u))) - 2
    hi_b = int(np.ceil(np.nanmax(u))) + 3
    lo_b = max(lo_b, 0)
    M = fetch_oe(clr, chrom, lo_b, hi_b)
    if M is None:
        return None
    reach = spec["reach"] // cres
    i, j = np.indices(M.shape)
    M = np.where(np.abs(i - j) <= reach, M, np.nan)
    valid = np.isfinite(M).astype(float)
    Z = np.nan_to_num(M)
    uu = u - lo_b
    yy, xx = np.meshgrid(uu, uu, indexing="ij")
    ok = np.isfinite(yy) & np.isfinite(xx)
    out = np.full((n, n), np.nan)
    coords = [yy[ok], xx[ok]]
    v = map_coordinates(Z, coords, order=1, mode="nearest")
    w = map_coordinates(valid, coords, order=1, mode="nearest")
    vv = np.where(w > 0.999, v, np.nan)
    out[ok] = vv
    return out


_COOLS: dict[str, cooler.Cooler] = {}


def _cool(p) -> cooler.Cooler:
    p = str(p)
    if p not in _COOLS:
        _COOLS[p] = cooler.Cooler(p)
    return _COOLS[p]


def ours_map(model: str, res: int, cell: str, key: str, n: int) -> np.ndarray | None:
    f = bc.pred_path(model, res, cell)
    if not f.exists():
        return None
    z = _npz(f)
    if key not in z:
        return None
    P = np.asarray(z[key], float)
    if P.shape[0] != n:
        return None
    live = np.abs(P).sum(1) > 0
    P[~live, :] = np.nan
    P[:, ~live] = np.nan
    return P


_NPZ: dict[str, dict] = {}


def _npz(f) -> dict:
    f = str(f)
    if f not in _NPZ:
        z = np.load(f)
        _NPZ[f] = {k: z[k] for k in z.files}
    return _NPZ[f]


# --------------------------------------------------------------------------- #
# block definitions
# --------------------------------------------------------------------------- #
def other_maps(cell: str, res: int) -> dict[str, Path]:
    ident = mx.CELLS[cell][3]
    out = {f"map_{o}": mx.target(o, res) for o in mx.BASELINE_MAPS
           if mx.CELLS[o][3] != ident and mx.target(o, res).exists()}
    cons = mx.consensus_loo(cell, res) if cell in mx.TRAIN12 else mx.consensus12(res)
    if cons.exists():
        out["consensus"] = cons
    return out


def oe_block(bid: str, cell: str, res: int, chroms: list[str], comps: list[str],
             flags: dict | None = None):
    """Blocks B to E: proper O/E target at res, our 12-cell models, competitors
    and every other measured map on identical regions and pixels."""
    R = regions()
    R = R[R.chrom.isin(chroms)].sort_values(["chrom", "start"])
    tgt = _cool(mx.target(cell, res))
    maps = other_maps(cell, res)
    reach = min([bc.COMPETITORS[c]["reach"] for c in comps] + [2_000_000]) // res
    rows = {}
    for _, r in R.iterrows():
        b0, b1 = r.start // res, -(-r.end // res)
        n = b1 - b0
        T = fetch_oe(tgt, r.chrom, b0, b1)
        if T is None:
            continue
        T = np.where(T == 0, np.nan, T)
        m = {}
        for mod in bc.OURS[res]:
            m[mod] = ours_map(mod, res, cell, r.key, n)
        for c in comps:
            m[c] = sample_competitor(bc.COMPETITORS[c], c, r.chrom, b0, n, res)
        for k, p in maps.items():
            M = fetch_oe(_cool(p), r.chrom, b0, b1)
            m[k] = None if M is None else np.where(M == 0, np.nan, M)
        if any(v is None for k, v in m.items() if not k.startswith("map_")):
            continue
        rows[r.key] = (b0, T, {k: v for k, v in m.items() if v is not None})
    meta = dict(block=bid, cell=cell, res=res, chroms=chroms, competitors=comps,
                reach_bins=int(reach), flags=flags or {}, space="proper O/E",
                n_regions=len(rows))
    _save_block(bid, rows, meta)


def akita_block(bid: str, cell_idx: int):
    """Block A: Akita's own target space, 2048 bp grid, 448-bin crop."""
    Z = np.load(bc.RASTER, allow_pickle=True)
    side, off = int(Z["side"]), int(Z["offset"])
    cell = ["hff", "gm12878"][cell_idx]
    iu = np.triu_indices(side, off)
    crop = (512 - side) // 2

    def dense(v):
        return bs.dense_from_triu(v, side, off).astype(float)

    base = {"consensus4": "akita_consensus4_2048bp.cool", "map_hff": "akita_hff_all_2048bp.cool",
            "map_h1": "akita_h1_all_2048bp.cool", "map_imr90": "akita_imr90_all_2048bp.cool",
            "map_hct116": "akita_hct116_all_2048bp.cool",
            "map_gm12878": "akita_gm12878_all_2048bp.cool"}
    base = {k: bc.AK2 / v for k, v in base.items() if not k.endswith(cell)}
    if cell == "hff":
        base.pop("consensus4", None)          # consensus4 contains HFF
    rows = {}
    for i, reg in enumerate(Z["regions"]):
        reg = str(reg)
        chrom, se = reg.split(":")
        s, e = map(int, se.split("-"))
        T = dense(Z["akita_target"][i, :, cell_idx])
        T[T == 0] = np.nan
        m = {"akita": dense(Z["akita_pred"][i, :, cell_idx])}
        if cell == "gm12878":
            m["hicgan"] = dense(Z["hicgan"][i])
            for n in ("akita_native_honest", "akita_native_refres"):
                f = bc.OUT / "pred" / "akita_space" / f"{n}.npz"
                z = _npz(f) if f.exists() else {}
                P = z.get(reg)
                m[n] = None if P is None else np.asarray(P, float)[crop:crop + side,
                                                                 crop:crop + side]
        else:
            f = bc.pred_path("all12_2kb", 2000, "hff")
            z = _npz(f) if f.exists() else {}
            P = z.get(reg)
            if P is not None:
                P = np.asarray(P, float)
                src0 = (s // 2000) * 2000
                P = bs.resample_square(P, src0, 2000, s, 512, 2048)[crop:crop + side,
                                                                    crop:crop + side]
            m["all12_2kb"] = P
        for k, p in base.items():
            B = fetch_oe(_cool(p), chrom, s // 2048, s // 2048 + 512)
            m[k] = None if B is None else np.where(B == 0, np.nan, B)[crop:crop + side,
                                                                      crop:crop + side]
        if any(v is None or not np.isfinite(v).any() for k, v in m.items()
               if not k.startswith("map_") and k != "consensus4"):
            continue
        rows[reg] = (s // 2048 + crop, T, {k: v for k, v in m.items() if v is not None})
    meta = dict(block=bid, cell=cell, res=2048, space="Akita target (log O/E, clipped, smoothed)",
                reach_bins=side, n_regions=len(rows), offset=off)
    _save_block(bid, rows, meta)


def _save_block(bid, rows, meta):
    BLK.mkdir(parents=True, exist_ok=True)
    arr = {}
    for key, (b0, T, m) in rows.items():
        arr[f"{key}|b0"] = np.array([b0])
        arr[f"{key}|target"] = T.astype(np.float32)
        for k, v in m.items():
            arr[f"{key}|{k}"] = v.astype(np.float32)
    np.savez_compressed(BLK / f"{bid}.npz", **arr)
    (BLK / f"{bid}.json").write_text(json.dumps(meta, indent=1))
    bc.log(f"assembled block {bid}: {meta['n_regions']} regions")


BLOCKS = {
    "A_gm12878_2kb": lambda: akita_block("A_gm12878_2kb", 1),
    "A_hff_2kb": lambda: akita_block("A_hff_2kb", 0),
    "B_gm12878_10kb_epiphany": lambda: oe_block(
        "B_gm12878_10kb_epiphany", "gm12878", 10000, ["chr3", "chr11", "chr17"],
        ["epiphany_gm12878_10kb", "hicgan_gm12878_10kb"],
        {"hicgan_gm12878_10kb": "training chromosomes of the Hi-cGAN model (odd)"}),
    "B_gm12878_10kb_even": lambda: oe_block(
        "B_gm12878_10kb_even", "gm12878", 10000, bc.EVEN, ["hicgan_gm12878_10kb"]),
    "C_gm12878_25kb_even": lambda: oe_block(
        "C_gm12878_25kb_even", "gm12878", 25000, bc.EVEN,
        ["hicreg_gm12878_25kb", "hicgan_gm12878_25kb"],
        {"hicreg_gm12878_25kb": "published predictions, training split undocumented"}),
    "D_imr90_10kb": lambda: oe_block(
        "D_imr90_10kb", "imr90", 10000, ["chr2", "chr3"],
        ["corigami_imr90_8kb", "hicgan_imr90_on_imr90_10kb", "hicgan_gm12878_on_imr90_10kb"],
        {"hicgan_imr90_on_imr90_10kb": "IMR-90 model on IMR-90; chr3 odd (training side)",
         "hicgan_gm12878_on_imr90_10kb": "GM12878 model on IMR-90 inputs (cross-cell)"}),
}
for _c, _t in (("k562", "k562"), ("hmec", "hmec_deep"), ("huvec", "huvec_deep")):
    for _res in (10000, 25000):
        _k = _res // 1000
        BLOCKS[f"E_{_t}_{_k}kb"] = (lambda c=_c, t=_t, r=_res, k=_k: oe_block(
            f"E_{t}_{k}kb", t, r, bc.EVEN, [f"hicgan_gm12878_on_{c}_{k}kb"],
            {f"hicgan_gm12878_on_{c}_{k}kb": "GM12878 model on this cell's inputs"}))


def stage_assemble():
    for bid, fn in BLOCKS.items():
        if (BLK / f"{bid}.npz").exists() and os.environ.get("BENCH_REBUILD") != "1":
            continue
        try:
            fn()
        except Exception as e:                                      # noqa: BLE001
            bc.log(f"assemble {bid} FAILED: {e!r}")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def load_block(bid):
    z = np.load(BLK / f"{bid}.npz")
    meta = json.loads((BLK / f"{bid}.json").read_text())
    rows: dict[str, dict] = {}
    for k in z.files:
        key, what = k.split("|")
        rows.setdefault(key, {})[what] = z[k]
    return rows, meta


def ref_sets(meta: dict, key: str, b0: int, n: int):
    """Reference boundaries (block grid bins) and loops (loop grid anchors)."""
    cell = meta["cell"]
    cid = {"hmec_deep": "hmec_deep", "huvec_deep": "huvec_deep"}.get(cell, cell)
    res = meta["res"]
    chrom = key.split(":")[0]
    bres = 10000 if res == 2048 else res
    B = bc.ref_boundaries(cid, bres)
    bnd = np.zeros(0, int)
    if B is not None:
        pos = B[B.chrom == chrom].start.values + bres // 2 if (B.chrom == chrom).any() else np.zeros(0)
        idx = np.floor(pos / res).astype(int) - b0
        prm = params(res)
        bnd = np.unique(idx[(idx >= prm["ins"]) & (idx < n - prm["ins"])])
    L = bc.ref_loops(cid)
    loops = np.zeros((0, 2), int)
    if L is not None:
        l = L[(L.c1 == chrom) & (L.c2 == chrom)]
        f = params(res)["loop"].loop_coarsen
        g = res * f
        a = np.floor((l.s1.values + 5000) / g).astype(int) - b0 // f
        b = np.floor((l.s2.values + 5000) / g).astype(int) - b0 // f
        nc = n // f
        keep = (a >= 0) & (b < nc) & (b - a >= params(res)["loop"].loop_min_dist)
        loops = np.unique(np.stack([a[keep], b[keep]], 1), axis=0) if keep.any() else loops
    return bnd, loops


def score_block(bid):
    rows, meta = load_block(bid)
    res = meta["res"]
    prm = params(res)
    methods = sorted({k for r in rows.values() for k in r if k not in ("b0", "target")})
    per_region, perdist, loopacc, bacc = [], {}, {m: [] for m in methods + ["target"]}, {}
    for key, r in rows.items():
        T = r["target"].astype(float)
        n = T.shape[0]
        b0 = int(r["b0"][0])
        core = [m for m in methods if not m.startswith("map_") and m != "consensus4"]
        mask = bm.triu_mask(n, 2, meta.get("reach_bins"))
        mask &= np.isfinite(T)
        for m in core:
            mask &= np.isfinite(r[m])
        bnd, loops = ref_sets(meta, key, b0, n)
        for m in methods:
            P = r[m].astype(float)
            mm = mask & np.isfinite(P)
            rec = {"block": bid, "region": key, "method": m,
                   "pearson": bm.pearson(P, T, mm), "spearman": bm.spearman(P, T, mm),
                   "scc": bm.scc(P, T, mm), "n_pixels": int(mm.sum()),
                   "insulation_pearson": bm.insulation_pearson(
                       np.where(mm | mm.T, T, np.nan), np.where(mm | mm.T, P, np.nan), prm["ins"])}
            h, nb = bm.boundary_recall(np.where(np.isfinite(P), P, 0.0), bnd, prm["ins"],
                                       prm["btol"], prm["flank"])
            rec["boundary_hits"], rec["boundary_n"] = h, nb
            lc = bm.loop_counts(np.where(np.isfinite(T), T, np.nan), P, loops, prm["loop"],
                                seed=len(per_region))
            rec["loop_hits"], rec["loop_n"] = lc["loop_hits"], lc["loop_n"]
            loopacc[m].append(lc)
            per_region.append(rec)
            for d, (a, b) in bm.per_distance(P, T, mm).items():
                perdist.setdefault((m, d), []).append((a, b))
        # measured target's own insulation minima: ceiling of the boundary caller
        h, nb = bm.boundary_recall(np.nan_to_num(T), bnd, prm["ins"], prm["btol"], prm["flank"])
        per_region.append({"block": bid, "region": key, "method": "target_self",
                           "boundary_hits": h, "boundary_n": nb})
    df = pd.DataFrame(per_region)
    bc.RES_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(bc.RES_OUT / f"{bid}_per_region.csv.gz", index=False, float_format="%.5f")
    # per-distance Pearson
    pd_rows = []
    for (m, d), lst in perdist.items():
        a = np.concatenate([x for x, _ in lst]); b = np.concatenate([y for _, y in lst])
        if a.size > 50 and a.std() > 0 and b.std() > 0:
            pd_rows.append({"block": bid, "method": m, "distance_bins": d,
                            "distance_bp": d * res, "pearson": float(np.corrcoef(a, b)[0, 1]),
                            "n": int(a.size)})
    pdd = pd.DataFrame(pd_rows)
    pdd.to_csv(bc.RES_OUT / f"{bid}_per_distance.csv", index=False, float_format="%.5f")
    # summary
    summ = []
    for m in methods:
        d = df[df.method == m]
        s = {"block": bid, "method": m, "label": label_of(m), "flag": meta.get("flags", {}).get(m, ""),
             "n_regions": int(d.pearson.notna().sum())}
        for k in ("pearson", "scc", "spearman", "insulation_pearson"):
            s.update(bm.summarize(d[k], k))
        s["boundary_recall"] = (d.boundary_hits.sum() / d.boundary_n.sum()
                                if d.boundary_n.sum() else np.nan)
        s["boundary_n_ref"] = int(d.boundary_n.sum())
        s.update(bm.pool_loops(loopacc[m], prm["loop"]))
        summ.append(s)
    t = df[df.method == "target_self"]
    summ.append({"block": bid, "method": "target_self", "label": "measured target (own minima)",
                 "boundary_recall": t.boundary_hits.sum() / max(1, t.boundary_n.sum()),
                 "boundary_n_ref": int(t.boundary_n.sum())})
    S = pd.DataFrame(summ)
    # paired comparisons of every model of ours against every other method
    pairs = []
    ours = [m for m in methods if m in bc.OURS_LABEL]
    for o in ours:
        for m in methods:
            if m == o:
                continue
            for k in ("pearson", "scc"):
                x = df[df.method == o].set_index("region")[k]
                y = df[df.method == m].set_index("region")[k]
                j = pd.concat([x, y], axis=1, keys=["o", "m"]).dropna()
                if len(j) < 5:
                    continue
                try:
                    p = float(wilcoxon(j.o, j.m).pvalue)
                except ValueError:
                    p = float("nan")
                pairs.append({"block": bid, "ours": o, "other": m, "metric": k,
                              "n": len(j), "median_diff": float((j.o - j.m).median()),
                              "win_rate": float((j.o > j.m).mean()), "wilcoxon_p": p})
    P = pd.DataFrame(pairs)
    S.to_csv(bc.RES_OUT / f"{bid}_summary.csv", index=False, float_format="%.4f")
    P.to_csv(bc.RES_OUT / f"{bid}_paired.csv", index=False, float_format="%.4g")
    bc.log(f"scored block {bid}: {len(rows)} regions, {len(methods)} methods")
    return S, P


def label_of(m: str) -> str:
    if m in bc.OURS_LABEL:
        return bc.OURS_LABEL[m]
    if m in bc.COMPETITORS:
        s = bc.COMPETITORS[m]
        extra = f", {s['model_cell']} model" if "model_cell" in s else ""
        return f"{s['label']}{extra}"
    if m == "akita":
        return "Akita (released predictions)"
    if m == "hicgan":
        return "Hi-cGAN (submitted)"
    if m == "consensus":
        return "consensus of other cells' measured maps"
    if m == "consensus4":
        return "consensus4 (HFF, H1, IMR90, HCT116 measured)"
    if m.startswith("map_"):
        return f"measured map of {m[4:]}"
    return m


def stage_score():
    allS, allP = [], []
    for bid in BLOCKS:
        if not (BLK / f"{bid}.npz").exists():
            continue
        sf = bc.RES_OUT / f"{bid}_summary.csv"
        if sf.exists() and sf.stat().st_mtime > (BLK / f"{bid}.npz").stat().st_mtime:
            S, P = pd.read_csv(sf), pd.read_csv(bc.RES_OUT / f"{bid}_paired.csv")
        else:
            S, P = score_block(bid)
        allS.append(S); allP.append(P)
    if allS:
        S = pd.concat(allS)
        # best single other measured map per block, as its own row
        S.to_csv(bc.RES_OUT / "benchmark_summary.csv", index=False, float_format="%.4f")
        pd.concat(allP).to_csv(bc.RES_OUT / "benchmark_paired.csv", index=False,
                               float_format="%.4g")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def stage_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = bc.OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    for bid in BLOCKS:
        f = bc.RES_OUT / f"{bid}_per_distance.csv"
        s = bc.RES_OUT / f"{bid}_summary.csv"
        if not f.exists():
            continue
        D = pd.read_csv(f)
        S = pd.read_csv(s).set_index("method")
        maps = [m for m in D.method.unique() if m.startswith("map_")]
        best_map = (S.loc[maps, "pearson_median"].idxmax() if maps else None)
        show = [m for m in D.method.unique() if not m.startswith("map_")] + (
            [best_map] if best_map else [])
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for m in show:
            d = D[D.method == m].sort_values("distance_bp")
            ls = "--" if (m.startswith("map_") or m.startswith("consensus")) else "-"
            ax.plot(d.distance_bp / 1e3, d.pearson, ls, lw=1.6, label=label_of(m))
        ax.set_xlabel("genomic distance (kb)")
        ax.set_ylabel("Pearson per distance")
        ax.set_title(bid)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(fd / f"{bid}_per_distance.png", dpi=150)
        plt.close(fig)
        region_panels(bid, fd, best_map)


def region_panels(bid, fd, best_map, n_show=3):
    import matplotlib.pyplot as plt
    rows, meta = load_block(bid)
    df = pd.read_csv(bc.RES_OUT / f"{bid}_per_region.csv.gz")
    methods = [m for m in sorted({k for r in rows.values() for k in r})
               if m not in ("b0", "target") and not m.startswith("map_")]
    if best_map:
        methods.append(best_map)
    # regions around the median of the first model of ours
    ours = [m for m in methods if m in bc.OURS_LABEL]
    ref = df[df.method == (ours[0] if ours else methods[0])].dropna(subset=["pearson"])
    ref = ref.sort_values("pearson")
    pick = ref.region.values[np.linspace(len(ref) * 0.25, len(ref) * 0.75, n_show).astype(int)
                             .clip(0, len(ref) - 1)]
    fig, axes = plt.subplots(n_show, len(methods), figsize=(2.6 * len(methods), 2.7 * n_show),
                             squeeze=False)
    for i, key in enumerate(pick):
        T = rows[key]["target"]
        for j, m in enumerate(methods):
            ax = axes[i, j]
            P = rows[key][m]
            n = T.shape[0]
            iu = np.triu_indices(n, 1)
            M = np.array(P, float)
            M[iu] = T[iu]
            ax.imshow(M, cmap="RdBu_r", vmin=-1.2, vmax=1.2, interpolation="none")
            r = df[(df.region == key) & (df.method == m)].pearson.values
            ax.set_title(f"{label_of(m)[:28]}\nr={r[0]:.2f}" if len(r) else label_of(m)[:28],
                         fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(key, fontsize=6)
    fig.suptitle(f"{bid}: measured (upper) vs prediction (lower)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fd / f"{bid}_regions.png", dpi=130)
    plt.close(fig)
