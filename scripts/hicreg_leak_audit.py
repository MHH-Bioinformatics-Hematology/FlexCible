"""Quantitative checks behind results/benchmark/HICREG_LEAK_AUDIT.md (CPU only).

Stages
  support   pair lists of the released HiC-Reg predictions against the pair slots of the
            chromosome (hg19, 5 kb, 1 Mb reach): the predicted pairs are the non-zero
            pixels of the sparse Hi-C matrix, so the prediction file itself carries the
            sparsity of the target; and the share of mappable pair slots with zero count in
            our hg38 4DN maps (hicreg_retrain.py target files).
  mcfolds   the fold generator of Scripts/genMULTICELLfeats/gen-MULTICELL-features.py
            (contiguous blocks of the pair table sorted by pair name) applied to the pair
            list of the released CV prediction: share of the neighbouring pixels of a test
            pair that are training pairs, and of test pairs whose two regions are both
            anchors of training pairs; the region-wise CV of the WINDOW models for comparison.
            python scripts/hicreg_leak_audit.py --stage support mcfolds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
HR = Path("/mnt/processed/wolffjoa/home_offload/HiC-Reg")
RES = REPO / "results/benchmark"
CELLS = {"gm12878": "Gm12878", "k562": "K562"}
HG19 = pd.read_csv(Path.home() / "src/HiC-Reg/hg19.chrom.sizes", sep="\t", header=None, index_col=0)[1]


def read_pairs(f):
    d = pd.read_csv(f, sep="\t", skiprows=1, header=None, usecols=[0, 3])
    s = d[0].str.replace("-", "_").str.split("_", expand=True)
    return d[0].values, s[1].astype(np.int64).values // 5000, s[4].astype(np.int64).values // 5000


def slots(chrom, reach=200):
    n = -(-int(HG19[chrom]) // 5000)
    return int(sum(min(reach, n - 1 - i) for i in range(n))) if n < 10**6 else 0


def stage_support():
    rows = []
    for cell, C in CELLS.items():
        for chrom in ("chr14", "chr17"):
            n = -(-int(HG19[chrom]) // 5000)
            tot = n * 200 - 200 * 201 // 2
            for mode, sub in (("CV", f"CV/{C}_{chrom}_RF_tree20_prediction.txt"),
                              ("CrossChrom", None)):
                if mode == "CrossChrom":
                    other = "chr17" if chrom == "chr14" else "chr14"
                    sub = f"CrossChrom/{C}_{other}to{chrom}_RF_tree20_prediction.txt"
                f = HR / C / "WINDOW" / chrom / sub
                if not f.exists():
                    continue
                names, i, j = read_pairs(f)
                rows.append(dict(cell=cell, chrom=chrom, mode=mode, build="hg19", bins=n,
                                 pair_slots_1Mb=tot, predicted_pairs=len(i),
                                 share_of_slots_predicted=len(i) / tot,
                                 max_gap_bins=int((j - i).max()), min_gap_bins=int((j - i).min())))
    D = pd.DataFrame(rows)
    # our maps: share of mappable pair slots (1 Mb) with zero count, hg38 5 kb
    out = []
    import hicreg_retrain as hr
    for cell in ("gm12878", "k562"):
        f = hr.OUT / f"tgt_{cell}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for ch in ("chr14", "chr17", "chr2", "chr3", "chr11"):
            T = z[ch]
            fin = np.isfinite(T)
            out.append(dict(cell=cell, chrom=ch, build="hg38", mappable_pair_slots=int(fin.sum()),
                            zero_count_share=float((T[fin] == 0).mean()),
                            zero_share_beyond_500kb=float((T[:, 100:][fin[:, 100:]] == 0).mean())))
    D.to_csv(RES / "hicreg_leak_support_released.csv", index=False, float_format="%.4f")
    pd.DataFrame(out).to_csv(RES / "hicreg_leak_zero_share.csv", index=False, float_format="%.4f")
    print(D.to_string()); print(pd.DataFrame(out).to_string())


def neighbours_in_training(i, j, train, sel):
    """For the pairs `sel`: mean share of their 8 neighbouring pixels (i +- 1, j +- 1) that are
    training pairs, and share of pairs with at least one such neighbour."""
    key = i * 100_000 + j
    o = np.argsort(key)
    sk, st = key[o], train[o]
    H = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if (a, b) == (0, 0):
                continue
            kk = (i[sel] + a) * 100_000 + (j[sel] + b)
            pos = np.clip(np.searchsorted(sk, kk), 0, len(sk) - 1)
            H.append((sk[pos] == kk) & st[pos])
    H = np.array(H)
    return float(H.mean()), float(H.any(0).mean())


def stage_mcfolds():
    """gen-MULTICELL-features.py (lines 52 to 69): rows of the merged pair table sorted by the
    pair string; fold k = rows [k n/5, (k+1) n/5), the last fold the remainder."""
    rows = []
    for cell, C in CELLS.items():
        for chrom in ("chr14", "chr17"):
            f = HR / C / "WINDOW" / chrom / "CV" / f"{C}_{chrom}_RF_tree20_prediction.txt"
            names, i, j = read_pairs(f)
            n = len(i)
            order = np.argsort(names.astype(str), kind="stable")
            fold = np.empty(n, int)
            size = n // 5
            for k in range(5):
                fold[order[k * size:(n if k == 4 else (k + 1) * size)]] = k
            nb, anyn, r1, r2, ts = [], [], [], [], []
            for k in range(5):
                te = np.nonzero(fold == k)[0]
                sel = np.random.default_rng(k).choice(te, min(40_000, len(te)), replace=False)
                a, b = neighbours_in_training(i, j, fold != k, sel)
                nb.append(a); anyn.append(b)
                seen = np.union1d(i[fold != k], j[fold != k])
                r1.append(np.isin(i[te], seen).mean()); r2.append(np.isin(j[te], seen).mean())
                ts.append(np.mean(fold != k))
            rows.append(dict(cell=cell, chrom=chrom, scheme="MULTICELL generator: contiguous blocks of the sorted pair table",
                             pairs=n, train_share=float(np.mean(ts)),
                             neighbour_pixels_in_training=float(np.mean(nb)),
                             pixels_with_any_trained_neighbour=float(np.mean(anyn)),
                             test_pairs_region1_in_training=float(np.mean(r1)),
                             test_pairs_region2_in_training=float(np.mean(r2))))
    S = pd.read_csv(RES / "hicreg_split.csv")
    for _, r in S[S.features == "WINDOW"].iterrows():
        rows.append(dict(cell=r.cell, chrom=r.chrom,
                         scheme="WINDOW released models: region-wise CV (hicreg_split.csv)",
                         pairs=int(r.pairs), train_share=r.train_share_per_model,
                         neighbour_pixels_in_training=r.neighbour_pixels_in_training_mean,
                         pixels_with_any_trained_neighbour=r.pixels_with_any_trained_neighbour,
                         test_pairs_region1_in_training=np.nan, test_pairs_region2_in_training=np.nan))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "hicreg_leak_folds.csv", index=False, float_format="%.4f")
    print(D.to_string())


def stage_supportleak():
    """Effect of the target-dependent prediction support. The released HiC-Reg files hold a
    prediction only for the non-zero pixels of the held-out chromosome's own Hi-C matrix; in
    oe_proper.py a mappable pixel without a value becomes the clipped depletion (-2), which is
    also the target value of a zero-count pixel. The leak-free retrained predictions (every
    pair predicted) are re-scored with exactly that support: predictions of pixels whose
    target count is zero removed. Same retrained model, same pixels; only the support differs."""
    import hic_measures as hm
    import hicreg_retrain as hr
    hm.chrho_use_root()
    for cell in ("gm12878", "k562"):
        z = np.load(hr.OUT / f"tgt_{cell}.npz")
        for ch in ("chr14", "chr17"):
            f = hr.OUT / f"pred_{cell}_windowsupport_{ch}.npz"
            if not f.exists():
                P = np.load(hr.OUT / f"pred_{cell}_window_{ch}.npz")["P"]
                T = z[ch]
                np.savez_compressed(f, P=np.where(T > 0, P, np.nan).astype(np.float32))
    hr.stage_cools()
    bc = hr.register()
    rows = []
    for cell in ("gm12878", "k562"):
        for res in (25000, 10000, 5000):
            k = res // 1000
            D = 1_000_000 // res
            for ch in ("chr14", "chr17"):
                T = hm.load_band(hm.target_cool(cell, res), ch, _nbins(cell, res, ch), D, res)
                n = T.shape[0]
                T[T == 0] = np.nan
                M = {}
                for tag in ("window", "windowsupport"):
                    M[tag] = hm.competitor_band(f"hicregre{tag}_{cell}_{k}kb", ch, n, D, res)
                mask = hm.common_mask([T, M["window"], M["windowsupport"]])
                Tm = hm.masked(T, mask)
                for tag, P in M.items():
                    Pm = hm.masked(P, mask)
                    tp = hm.tile_pearsons(Tm, Pm, 1_000_000 // res)
                    rows.append(dict(cell=cell, res=res, chrom=ch, support=("target-nonzero pixels (as released)"
                                     if tag == "windowsupport" else "all pairs (retrained)"),
                                     oe_pearson=hm.band_pearson(Tm, Pm),
                                     oe_pearson_tile_median=float(np.median(tp)),
                                     oe_scc=hm.band_scc(Tm, Pm), pixels=int(mask.sum())))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "hicreg_leak_support_effect.csv", index=False, float_format="%.4f")
    print(D.to_string())


def _nbins(cell, res, chrom):
    import hic_measures as hm
    D = 1_000_000 // res
    ef = hm.expected_file(cell, res, chrom, D)
    if not ef.exists():
        hm.compute_expected(cell, res, chrom, D=max(200, D))
    return int(np.load(ef)["n"])


def stage_oracle():
    """Ceiling of HiC-Reg's native target in our scoring: the true SQRTVC-normalised 5 kb counts
    (ln(1 + count), every mappable pair) passed through the same conversion as a prediction
    (expm1, sum into 10/25 kb, oe_proper.py) and scored against the ICE-balanced O/E target."""
    import cooler
    import hic_measures as hm
    import hicreg_retrain as hr
    hm.chrho_use_root()
    bc = hr.register()
    c = cooler.Cooler(f"{hr.MCOOL['gm12878']}::resolutions/5000")
    sizes = {ch: int(c.chromsizes[ch]) for ch in hr.CHROMS}
    rows = []
    for cell in ("gm12878", "k562"):
        T5 = np.load(hr.OUT / f"tgt_{cell}.npz")
        for ch in ("chr14", "chr17"):
            np.savez_compressed(hr.OUT / f"pred_{cell}_oracle_{ch}.npz",
                                P=np.where(np.isfinite(T5[ch]), T5[ch], np.nan).astype(np.float32))
            for res in (25000, 10000, 5000):
                k = res // 1000
                name = f"hicregreoracle_{cell}_{k}kb"
                spec = dict(hr.comp_spec(f"hicregrewindow_{cell}_{k}kb"))
                spec["path"] = hr.OUT / "counts" / (name + "_{chrom}.cool")
                spec["chroms"] = [ch]
                bc.COMPETITORS[name] = spec
                hr.counts_cool(name, f"{cell}_oracle", ch, res, sizes)
                out = bc.comp_oe(name, ch)
                if out.exists():
                    out.unlink()
                bc.run_oe(name, spec)
                D = 1_000_000 // res
                n = _nbins(cell, res, ch)
                T = hm.load_band(hm.target_cool(cell, res), ch, n, D, res)
                T[T == 0] = np.nan
                P = hm.competitor_band(name, ch, n, D, res)
                m = hm.common_mask([T, P])
                Tm, Pm = hm.masked(T, m), hm.masked(P, m)
                rows.append(dict(cell=cell, res=res, chrom=ch, oe_pearson=hm.band_pearson(Tm, Pm),
                                 oe_pearson_tile_median=float(np.median(hm.tile_pearsons(Tm, Pm, 1_000_000 // res))),
                                 oe_scc=hm.band_scc(Tm, Pm), pixels=int(m.sum())))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "hicreg_leak_oracle.csv", index=False, float_format="%.4f")
    print(D.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    a = ap.parse_args()
    for s in a.stage:
        globals()[f"stage_{s}"]()


if __name__ == "__main__":
    main()
