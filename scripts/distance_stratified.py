"""Distance-stratified Pearson correlation between predictions and measured O/E maps.

Held-out chromosomes only. For every cell, resolution and comparison set, the pixels used are the
upper-triangle pixels (2 bins off the diagonal up to 2 Mb) that are stored and finite in the measured map
and in EVERY compared source of that set (identical pixels). Pearson r is computed per bin distance, pooled
over all pixels of the chromosomes of the set at that distance.
Non-native grids are read at the bin centres of the target grid (nearest native bin; hg19 sources after
lifting the centres with the hg38ToHg19 chain, bins whose lift is not locally monotone are dropped).
Reads existing files only. Usage: python scripts/distance_stratified.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexcible.liftover import LiftOver  # noqa: E402

SH = Path("/mnt/processed/wolffjoa/hic-multiome")
OE = SH / "crosscell_oe"
PRED = SH / "runs/hic_measures_chrho/pred"
CMP = SH / "runs/benchmark/competitor_oe"
CHAIN = Path.home() / "data/hic-multiome/chain/hg38ToHg19.over.chain.gz"
OUT = Path(__file__).resolve().parents[1] / "results/distance_stratified"
HELD = ["chr2", "chr3", "chr11", "chr14", "chr17"]
MAXD_BP = 2_000_000


class Table:
    """Sparse upper-triangle pixel table of one chromosome of one cooler (finite values only)."""

    def __init__(self, path: Path, chrom: str):
        self.clr = cooler.Cooler(str(path))
        name = next(c for c in self.clr.chromnames if c == chrom or c == chrom.replace("chr", ""))
        self.res = self.clr.binsize
        self.n = int(np.ceil(self.clr.chromsizes[name] / self.res)) + 1
        px = self.clr.pixels(join=False).fetch(name)
        off = self.clr.offset(name)
        i = px.bin1_id.values.astype(np.int64) - off
        j = px.bin2_id.values.astype(np.int64) - off
        v = px["count"].values.astype(np.float64)
        k = np.isfinite(v)
        key = i[k] * self.n + j[k]
        o = np.argsort(key)
        self.key, self.val = key[o], v[k][o]

    def at(self, pos_i, pos_j):
        """Values at genomic positions (bp, any grid); NaN where not stored or position invalid."""
        pi, pj = np.asarray(pos_i, float), np.asarray(pos_j, float)
        ok = np.isfinite(pi) & np.isfinite(pj)
        bi = np.floor(np.where(ok, pi, 0) / self.res).astype(np.int64)
        bj = np.floor(np.where(ok, pj, 0) / self.res).astype(np.int64)
        lo, hi = np.minimum(bi, bj), np.maximum(bi, bj)
        key = lo * self.n + hi
        idx = np.searchsorted(self.key, key)
        idx[idx >= len(self.key)] = len(self.key) - 1
        hit = ok & (self.key[idx] == key) & (lo >= 0) & (hi < self.n)
        return np.where(hit, self.val[idx], np.nan)


_LIFT = None


def lifted_centres(chrom: str, nbins: int, res: int) -> np.ndarray:
    """Centres of the target-grid bins (hg38) lifted to hg19; NaN where lift fails or is not locally monotone."""
    global _LIFT
    if _LIFT is None:
        _LIFT = LiftOver(CHAIN)
    pos = np.full(nbins, np.nan)
    for k in range(nbins):
        c = (k + 0.5) * res
        for dx in (0, -res / 4, res / 4):
            hit = _LIFT.convert(chrom, int(c + dx))
            if hit is not None and hit[0] == chrom:
                pos[k] = hit[1] - dx
                break
    step = np.diff(pos)
    bad = np.zeros(nbins, bool)
    badstep = ~np.isfinite(step) | (step <= 0) | (step > 2 * res)
    bad[:-1] |= badstep
    bad[1:] |= badstep
    pos[bad] = np.nan
    return pos


def resolve(spec, cell, res, chrom):
    """-> (path, build) of one source for one chromosome."""
    k = f"{res // 1000}kb"
    kind = spec
    if kind == "measured":
        return OE / f"{cell}_proper_{k}.cool", "hg38"
    if kind == "ours":
        return PRED / k / "all12refin_drop_chrho" / f"{cell}.cool", "hg38"
    if kind == "chip":
        return PRED / k / "all12honest_drop_chrho" / f"{cell}.cool", "hg38"
    if kind == "consensus":
        f = "consensus12" if cell == "k562" else f"consensus_loo_{cell}"
        return OE / f"{f}_proper_{k}.cool", "hg38"
    if kind == "corigami":
        return CMP / f"corigami_imr90_8kb_{chrom}.cool", "hg38"
    if kind == "epiphany":
        return CMP / f"epiphany_gm12878_10kb_{chrom}.cool", "hg38"
    if kind == "hicgan":
        return CMP / "hicgan_gm12878_10kb.cool", "hg19"
    raise KeyError(kind)


CORE = ["ours", "chip", "consensus"]
# (set name, sources, chromosomes) per cell; the competitor sets follow the task definition, and the
# strict sets keep each competitor to chromosomes outside its own training chromosomes.
SETS = {
    "imr90": [("core", CORE, HELD), ("all_sources", CORE + ["corigami"], ["chr2", "chr3"])],
    "gm12878": [("core", CORE, HELD),
                ("all_sources", CORE + ["epiphany", "hicgan"], ["chr2", "chr3", "chr11", "chr17"]),
                ("strict_epiphany", CORE + ["epiphany"], ["chr3", "chr11", "chr17"]),
                ("strict_hicgan", CORE + ["hicgan"], ["chr2", "chr14"])],
    "k562": [("core", CORE, HELD)],
}


def run_set(cell, res, sources, chroms):
    nd = MAXD_BP // res + 1
    acc = {s: np.zeros((6, nd)) for s in sources}  # n, sx, sy, sxx, syy, sxy
    npix = 0
    for ch in chroms:
        meas = Table(resolve("measured", cell, res, ch)[0], ch)
        i = (meas.key // meas.n)
        j = (meas.key % meas.n)
        d = j - i
        k = (d >= 2) & (d * res <= MAXD_BP)
        i, j, d, y = i[k], j[k], d[k], meas.val[k]
        ci, cj = (i + 0.5) * res, (j + 0.5) * res
        vals = {}
        for s in sources:
            path, build = resolve(s, cell, res, ch)
            t = Table(path, ch)
            if build == "hg19":
                L = lifted_centres(ch, meas.n, res)
                pi, pj = L[i], L[j]
                pj_ok = pj > pi
                pi, pj = np.where(pj_ok, pi, np.nan), np.where(pj_ok, pj, np.nan)
            else:
                pi, pj = ci, cj
            vals[s] = t.at(pi, pj)
        ok = np.ones(len(y), bool)
        for s in sources:
            ok &= np.isfinite(vals[s])
        npix += int(ok.sum())
        dd = d[ok]
        yy = y[ok]
        for s in sources:
            x = vals[s][ok]
            for r, w in enumerate((np.ones_like(x), x, yy, x * x, yy * yy, x * yy)):
                acc[s][r] += np.bincount(dd, weights=w, minlength=nd)[:nd]
        print(f"  {cell} {res//1000}kb {ch}: {int(ok.sum())} identical pixels", flush=True)
    rows = []
    for s in sources:
        n, sx, sy, sxx, syy, sxy = acc[s]
        for dist in range(2, nd):
            if n[dist] < 3:
                continue
            cov = sxy[dist] / n[dist] - sx[dist] * sy[dist] / n[dist] ** 2
            vx = sxx[dist] / n[dist] - (sx[dist] / n[dist]) ** 2
            vy = syy[dist] / n[dist] - (sy[dist] / n[dist]) ** 2
            r = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
            rows.append((dist * res, s, r, int(n[dist])))
    return pd.DataFrame(rows, columns=["distance_bp", "source", "pearson", "n_pixels"]), npix


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summ = []
    for cell in ("imr90", "gm12878", "k562"):
        for res in (10000, 25000):
            frames = []
            for name, srcs, chroms in SETS[cell]:
                if res == 25000 and name != "core":
                    continue
                print(cell, res, name, flush=True)
                df, npix = run_set(cell, res, srcs, chroms)
                df.insert(0, "set", name)
                df["chromosomes"] = ",".join(chroms)
                frames.append(df)
                summ.append((cell, res, name, ",".join(chroms), npix))
            pd.concat(frames).to_csv(OUT / f"{cell}_{res//1000}kb.csv", index=False)
    pd.DataFrame(summ, columns=["cell", "res_bp", "set", "chromosomes", "identical_pixels"]).to_csv(
        OUT / "pixel_counts.csv", index=False)
    # headline table
    rows = []
    for f in sorted(OUT.glob("*_10kb.csv")):
        df = pd.read_csv(f)
        for (st, s), g in df.groupby(["set", "source"]):
            for lab, d in (("100kb", 100_000), ("500kb", 500_000), ("1Mb", 1_000_000)):
                r = g.loc[g.distance_bp == d]
                rows.append((f.stem.split("_")[0], st, s, lab, float(r.pearson.iloc[0]) if len(r) else np.nan,
                             int(r.n_pixels.iloc[0]) if len(r) else 0))
    pd.DataFrame(rows, columns=["cell", "set", "source", "distance", "pearson", "n_pixels"]).to_csv(
        OUT / "headline_10kb.csv", index=False)


if __name__ == "__main__":
    main()
