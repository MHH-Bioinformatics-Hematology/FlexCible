"""Follow-up analyses of the benchmark subsection for the release twin all12one_fullrd_chrho (registry 101).

Stages (the twin = all12one_fullrd_chrho with the consensus, all12one_fullrd_chrho_noref = ChIP only):
  distance  distance-stratified Pearson (scripts/distance_stratified.py, Supplementary Fig. S8). Every set is run with
            the old sources plus the twin; the old sources must reproduce results/distance_stratified/<cell>_<k>kb.csv
            exactly and the pixel set must be unchanged, then the rows of the twin (source = model name) are appended.
  loops     cell-type-specific loops (scripts/celltype_specific_loops.py); same check, rows appended to
            results/celltype_loops/auroc.csv (predictor = model name).
  windows   window gains over the consensus and the competitors in 800 kb windows (the logic of the manuscript's
            figures/scripts/fig_regions.py score_set, repeated with the twin) -> results/one_model/window_gain_release_*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
NEWM = ["all12one_fullrd_chrho", "all12one_fullrd_chrho_noref"]


def stage_distance():
    import distance_stratified as ds
    out = ds.OUT
    summ = []
    for cell in ("imr90", "gm12878", "k562"):
        for res in (10000, 25000):
            f = out / f"{cell}_{res // 1000}kb.csv"
            old = pd.read_csv(f)
            if set(NEWM) <= set(old.source):
                print("already appended", f)
                continue
            frames = []
            for name, srcs, chroms in ds.SETS[cell]:
                if res == 25000 and name != "core":
                    continue
                df, npix = ds.run_set(cell, res, srcs + NEWM, chroms)
                o = old[old.set == name]
                m = df[~df.source.isin(NEWM)].merge(o, on=["distance_bp", "source"], suffixes=("", "_old"))
                assert len(m) == len(o), (cell, res, name, len(m), len(o))
                d = float((m.pearson - m.pearson_old).abs().max())
                dn = int((m.n_pixels != m.n_pixels_old).sum())
                print(f"{cell} {res} {name}: old sources max |dr| {d:.2g}, pixel-count differences {dn}", flush=True)
                assert d < 1e-9 and dn == 0
                new = df[df.source.isin(NEWM)].copy()
                new.insert(0, "set", name)
                new["chromosomes"] = ",".join(chroms)
                frames.append(new)
            pd.concat([old] + frames, ignore_index=True).to_csv(f, index=False)
    # headline table
    h = pd.read_csv(out / "headline_10kb.csv")
    rows = []
    for f in sorted(out.glob("*_10kb.csv")):
        if f.name.startswith("headline"):
            continue
        df = pd.read_csv(f)
        for (st, s), g in df[df.source.isin(NEWM)].groupby(["set", "source"]):
            for lab, dd in (("100kb", 100_000), ("500kb", 500_000), ("1Mb", 1_000_000)):
                r = g.loc[g.distance_bp == dd]
                rows.append((f.stem.split("_")[0], st, s, lab, float(r.pearson.iloc[0]) if len(r) else np.nan,
                             int(r.n_pixels.iloc[0]) if len(r) else 0))
    h = h[~h.source.isin(NEWM)]
    pd.concat([h, pd.DataFrame(rows, columns=h.columns)], ignore_index=True).to_csv(out / "headline_10kb.csv", index=False)


def stage_loops():
    import celltype_specific_loops as cl
    import distance_stratified  # noqa: F401
    from sklearn.metrics import roc_auc_score
    kinds = ["measured", "ours", "chip", "consensus"] + NEWM
    calls = {c: cl.read_calls(c) for c in cl.CELLS}
    sh_i = cl.near(calls["imr90"], calls["gm12878"])
    sh_g = cl.near(calls["gm12878"], calls["imr90"])
    A = calls["imr90"].assign(origin="imr90", shared=sh_i)
    B = calls["gm12878"].assign(origin="gm12878", shared=sh_g)
    loops = pd.concat([A, B[~B.shared]], ignore_index=True)
    tabs = {(k, c): {ch: cl.Table(cl.src_path(k, c), ch) for ch in cl.HELD} for k in kinds for c in cl.CELLS}
    for k in kinds:
        for c in cl.CELLS:
            loops[f"{k}_{c}"] = np.nan
    for ch in cl.HELD:
        m = (loops.chrom == ch).values
        i, j = loops.i.values[m], loops.j.values[m]
        for k in kinds:
            for c in cl.CELLS:
                loops.loc[m, f"{k}_{c}"] = cl.val3(tabs[(k, c)], ch, i, j)
    fin = np.ones(len(loops), bool)
    for k in kinds:
        for c in cl.CELLS:
            fin &= np.isfinite(loops[f"{k}_{c}"].values)
    L = loops[fin].copy()
    old_L = pd.read_csv(cl.OUT / "loops_scored.csv")
    assert len(L) == len(old_L), (len(L), len(old_L))
    for k in kinds:
        L[f"d_{k}"] = L[f"{k}_imr90"] - L[f"{k}_gm12878"]
    rows = []
    for thr in cl.THRS:
        cls = np.where(L.shared, "shared", np.where(L.d_measured >= thr, "imr90_specific",
                       np.where(L.d_measured <= -thr, "gm12878_specific", "unclassified")))
        cls = np.where((L.origin == "gm12878") & ~L.shared & (L.d_measured >= thr), "unclassified", cls)
        cls = np.where((L.origin == "imr90") & ~L.shared & (L.d_measured <= -thr), "unclassified", cls)
        L["cls"] = cls
        for k in kinds:
            for pos, negs in (("imr90_specific", ["shared", "gm12878_specific"]),
                              ("gm12878_specific", ["shared", "imr90_specific"])):
                sign = 1 if pos == "imr90_specific" else -1
                for neg in negs:
                    s = L[L.cls.isin([pos, neg])]
                    y = (s.cls == pos).astype(int).values
                    auc = roc_auc_score(y, sign * s[f"d_{k}"].values) if 0 < y.sum() < len(y) else np.nan
                    rows.append((thr, k, pos, neg, auc, int(y.sum()), int((1 - y).sum())))
    au = pd.DataFrame(rows, columns=["threshold", "predictor", "positives", "negatives", "auroc", "n_pos", "n_neg"])
    old = pd.read_csv(cl.OUT / "auroc.csv")
    names = {"measured": "measured (ceiling)", "ours": "ours with consensus", "chip": "ours ChIP only",
             "consensus": "consensus alone"}
    au["predictor"] = au.predictor.replace(names)
    chk = au[au.predictor.isin(old.predictor)].merge(old, on=["threshold", "predictor", "positives", "negatives"],
                                                     suffixes=("", "_old"))
    print("loops: old rows max |dAUROC|", float((chk.auroc - chk.auroc_old).abs().max()), "n rows", len(chk),
          "n_pos/n_neg equal", bool(((chk.n_pos == chk.n_pos_old) & (chk.n_neg == chk.n_neg_old)).all()))
    assert float((chk.auroc - chk.auroc_old).abs().max()) < 1e-9
    new = au[au.predictor.isin(NEWM)]
    old = old[~old.predictor.isin(NEWM)]
    pd.concat([old, new], ignore_index=True).to_csv(cl.OUT / "auroc.csv", index=False)
    print(new.to_string())


def stage_windows():
    import cooler
    SH = Path("/mnt/processed/wolffjoa/hic-multiome")
    CO = SH / "crosscell_oe"
    PRED = SH / "runs/hic_measures_chrho/pred/10kb"
    COMP = SH / "runs/benchmark/competitor_oe"
    NB, RESB = 80, 10_000
    pc = lambda pat: (lambda ch: COMP / pat.format(chrom=ch))  # noqa: E731

    def srcs(cell, extra):
        return extra

    SETS = {
        "imr90": dict(chroms=["chr2", "chr3"], meas=CO / "imr90_proper_10kb.cool", cons=CO / "consensus_loo_imr90_proper_10kb.cool",
                      comp={"C.Origami": pc("corigami_imr90_8kb_{chrom}.cool")}),
        "gm12878": dict(chroms=["chr2", "chr3", "chr11", "chr17"], meas=CO / "gm12878_proper_10kb.cool",
                        cons=CO / "consensus_loo_gm12878_proper_10kb.cool",
                        comp={"Epiphany": pc("epiphany_gm12878_10kb_{chrom}.cool"), "Hi-cGAN": COMP / "hicgan_gm12878_10kb.cool"}),
        "k562": dict(chroms=["chr2", "chr3", "chr11", "chr14", "chr17"], meas=CO / "k562_proper_10kb.cool",
                     cons=CO / "consensus12_proper_10kb.cool",
                     comp={"C.Origami": pc("corigami_pre_k562_{chrom}.cool"), "Epiphany": pc("epiphany_pre_k562_{chrom}.cool")}),
    }
    MODELS = {"FlexCible": "all12refin_drop_chrho", "FlexCible ChIP only": "all12honest_drop_chrho",
              "Twin": "all12one_fullrd_chrho", "Twin ChIP only": "all12one_fullrd_chrho_noref"}
    cache = {}

    def clr(p):
        p = str(p)
        if p not in cache:
            cache[p] = cooler.Cooler(p)
        return cache[p]

    def fetch(path, chrom, start):
        c = clr(path)
        name = chrom if chrom in c.chromnames else chrom.replace("chr", "")
        if name not in c.chromnames:
            return None
        end = start + NB * RESB
        if c.binsize == RESB:
            m = np.asarray(c.matrix(balance=False).fetch(f"{name}:{start}-{end}"), float)
        else:
            b = c.binsize
            lo = max(0, start - 2 * b)
            hi = min(c.chromsizes[name], end + 2 * b)
            m0 = np.asarray(c.matrix(balance=False).fetch(f"{name}:{lo}-{hi}"), float)
            idx = np.clip(((start + RESB * (np.arange(NB) + 0.5) - lo) // b).astype(int), 0, m0.shape[0] - 1)
            m = m0[np.ix_(idx, idx)]
        m[m == 0] = np.nan
        return m if m.shape == (NB, NB) else None

    def upper(m):
        return m[np.triu_indices(m.shape[0], 2)]

    def pearson(a, b):
        x, y = upper(a), upper(b)
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 200 else np.nan

    log = []
    for cell, S in SETS.items():
        src = {"Consensus": S["cons"]}
        src.update({n: PRED / m / f"{cell}.cool" for n, m in MODELS.items()})
        src.update(S["comp"])
        rows = []
        for chrom in S["chroms"]:
            L = clr(S["meas"]).chromsizes[chrom]
            for start in range(2_000_000, L - NB * RESB - 2_000_000, NB * RESB):
                meas = fetch(S["meas"], chrom, start)
                if meas is None or np.isfinite(upper(meas)).mean() < 0.9:
                    continue
                row = {"chrom": chrom, "start": start}
                for n, p in src.items():
                    path = p(chrom) if callable(p) else p
                    m = fetch(path, chrom, start) if Path(path).exists() else None
                    row[n] = np.nan if m is None else pearson(meas, m)
                rows.append(row)
        df = pd.DataFrame(rows).dropna()
        df["best_comp"] = df[list(S["comp"])].max(axis=1)
        for tag, col in (("", "FlexCible"), ("_twin", "Twin"), ("_twin_chip", "Twin ChIP only")):
            df[f"gain_cons{tag}"] = df[col] - df["Consensus"]
            df[f"gain_comp{tag}"] = df[col] - df["best_comp"]
            df[f"score{tag}"] = df[[f"gain_cons{tag}", f"gain_comp{tag}"]].min(axis=1)
        df.to_csv(REPO / f"results/one_model/window_gain_release_{cell}.csv", index=False, float_format="%.4f")
        for tag, nm in (("", "per-resolution all12refin_drop_chrho"), ("_twin", "twin all12one_fullrd_chrho"),
                        ("_twin_chip", "twin ChIP only")):
            gc, gp = df[f"gain_cons{tag}"], df[f"gain_comp{tag}"]
            msg = (f"{cell} {nm}: {len(df)} windows; > consensus {(gc > 0).mean():.1%}; > best competitor "
                   f"{(gp > 0).mean():.1%}; both {((gc > 0) & (gp > 0)).mean():.1%}; median gain over consensus "
                   f"{gc.median():+.3f}, over best competitor {gp.median():+.3f}")
            # window at the 90th percentile of the score and r between the two difference maps
            d = df.sort_values(f"score{tag}").reset_index(drop=True)
            i = int(round(0.9 * (len(d) - 1)))
            r = d.loc[i]
            col = {"": "FlexCible", "_twin": "Twin", "_twin_chip": "Twin ChIP only"}[tag]
            mm = fetch(S["meas"], r.chrom, int(r.start))
            mc = fetch(S["cons"], r.chrom, int(r.start))
            mo = fetch(src[col], r.chrom, int(r.start))
            rr = pearson(mm - mc, mo - mc)
            msg += f"; 90th-percentile window rank {i + 1}: {r.chrom}:{int(r.start)} score {r[f'score{tag}']:.3f}, r(measured-consensus, model-consensus) = {rr:.3f}"
            print(msg, flush=True)
            log.append(msg)
    (REPO / "results/one_model/window_gain_release_log.txt").write_text("\n".join(log) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", choices=["distance", "loops", "windows"], required=True)
    a = ap.parse_args()
    for s in a.stage:
        {"distance": stage_distance, "loops": stage_loops, "windows": stage_windows}[s]()


if __name__ == "__main__":
    main()
