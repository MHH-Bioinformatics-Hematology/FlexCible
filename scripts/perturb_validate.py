"""In-silico perturbation against degron experiments (registry section 21).

Experiments (measured Hi-C, hg38):
  rad21  HCT116 RAD21-mAC, untreated vs 6 h auxin (Rao et al. 2017, 4DN
         4DNFIFLDVASC / 4DNFILP99QJS)
  ctcf   HCT116 CTCF-mAID2, untreated vs 6 h 5-Ph-IAA (ENCODE intact Hi-C,
         ENCSR086RCC / ENCSR669VZH), when present under perturb/hct116_ctcf_aid
Both conditions of an experiment go through scripts/oe_proper.py; the measured
change is the difference of the two log O/E maps (auxin minus untreated).

Prediction: the 12-cell models predict HCT116 (a training cell; the 413 Akita
test regions are held out) from its ENCODE ChIP with all tracks (full) and with
one protein set to its absent value (crosscell_predict --keep-tracks): RAD21
(the HCT116 ChIP set has no SMC3, so no cohesin subunit remains), CTCF, and
H3K36me3 as a control mark. The CTCF motif channels stay (the DNA is
unchanged in a degron). Predicted change = perturbed minus full.

Metrics per region (upper triangle, 2 bins off the diagonal, pixels finite in
all maps): Pearson and SCC between predicted and measured change maps; the
change maps pooled per distance; at the loops called in the untreated map
(hicDetectLoops, runs/benchmark/calls) the change of loop strength (log O/E at
the loop pixel, maximum within +/- 1 bin) and APA; at the untreated domain
boundaries (hicFindTADs) the change of insulation.

    python scripts/perturb_validate.py --stage oe predict score figures
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import benchmark_metrics as bm                                       # noqa: E402
import multicell_expanded as mx                                      # noqa: E402
from hicmultiome import bioscore as bs                               # noqa: E402

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
PT = HM / "perturb"
OUT = HM / "runs/perturb_validate"
RES_OUT = REPO / "results/perturb_validate"
CALLS = HM / "runs/benchmark/calls"
PY = sys.executable
THREADS = int(os.environ.get("PERT_THREADS", "4"))
REGIONS = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization/"
               "akita_validation/sequences_test_list.tsv")

EXPERIMENTS = {
    "rad21": dict(untreated=HM / "matrices/hct116_untreated.mcool",
                  treated=HM / "matrices/hct116_auxin6h.mcool",
                  label="HCT116 RAD21-mAC, 6 h auxin (Rao 2017)", calls="hct116"),
    "ctcf": dict(untreated=PT / "hct116_ctcf_aid/hct116_ctcfaid_untreated_{k}kb.cool",
                 treated=PT / "hct116_ctcf_aid/hct116_ctcfaid_auxin6h_{k}kb.cool",
                 label="HCT116 CTCF-mAID2, 6 h 5-Ph-IAA (ENCODE)", calls="hct116"),
}
PERTS = {"full": None,
         "minus_RAD21": [t for t in mx.ALL13 if t != "RAD21"],
         "minus_CTCF": [t for t in mx.ALL13 if t != "CTCF"],
         "minus_H3K36me3": [t for t in mx.ALL13 if t != "H3K36me3"]}
MODELS = {  # name -> (run dir, res, reference run name or None)
    "all12honest_10kb": (mx.REF_OUT / "all12honest_10kb", 10000, None),
    "all12honest_25kb": (mx.REF_OUT / "all12honest_25kb", 25000, None),
    "all12refres_10kb": (mx.REF_OUT / "all12refres_10kb", 10000, "all12refres"),
    "all12refres_25kb": (mx.REF_OUT / "all12refres_25kb", 25000, "all12refres"),
    # trained with track dropout (registry section 22)
    **{f"{b}_{k}kb": (mx.REF_OUT / f"{b}_{k}kb", k * 1000,
                      None if b.startswith("all12honest") else b)
       for b in ("all12refres_drop", "all12honest_drop", "all12refres_refdrop",
                 "all12refres_drop_hard", "all12honest_drop_hard")
       for k in (10, 25)},
}
EXP_PERT = {"rad21": "minus_RAD21", "ctcf": "minus_CTCF"}
# measured ChIP of the degron cells (4DN GRCh38 fold change: RAD21 and CTCF from
# Kim et al. 2024, eight histone marks from Rao et al. 2017, untreated and 6 h
# auxin; H3K4me2, H3K9ac and H4K20me1 from the ENCODE HCT116 set in both):
# predicted change = prediction from the auxin ChIP minus from the untreated ChIP
CHIP_DIRS = {"chip_untreated": Path.home() / "data/hic-multiome/perturb_chip/untreated",
             "chip_auxin6h": Path.home() / "data/hic-multiome/perturb_chip/auxin6h"}
CHIP_EXP = {"rad21"}


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "perturb_validate.log", "a") as fh:
        fh.write(line + "\n")


def sh(cmd, logf: Path, env=None) -> int:
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def cpu_env():
    n = str(THREADS)
    return dict(os.environ, OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n,
                CUDA_VISIBLE_DEVICES="")


def oe_path(exp: str, cond: str, res: int) -> Path:
    if exp == "rad21" and cond == "untreated":
        return mx.target("hct116", res)          # the training target, same mcool
    return OUT / "oe" / f"{exp}_{cond}_proper_{res // 1000}kb.cool"


def stage_oe():
    for exp, e in EXPERIMENTS.items():
        for cond in ("untreated", "treated"):
            for res in (10000, 25000):
                src = str(e[cond]).replace("{k}", str(res // 1000))
                if not Path(src).exists():
                    log(f"oe {exp} {cond} {res}: {src} missing, skipped")
                    continue
                if src.endswith(".cool"):
                    import cooler as _c
                    if "weight" not in _c.Cooler(src).bins().columns:
                        log(f"oe {exp} {cond} {res}: {src} not balanced yet, skipped")
                        continue
                out = oe_path(exp, cond, res)
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                rc = sh([PY, "scripts/oe_proper.py", "--input", src, "--output", out,
                         "--binsize", res], OUT / "oe" / f"{exp}_{cond}.log", cpu_env())
                log(f"oe {exp} {cond} {res} rc={rc}")


def pred_file(model: str, pert: str) -> Path:
    return OUT / "pred" / model / f"{pert}.npz"


def stage_predict():
    for model, (run, res, ref) in MODELS.items():
        if not (run / "best").exists():          # not trained yet
            continue
        ck = run / (run / "best").read_text().strip()
        _, bw, assay, _ = mx.CELLS["hct116"]
        if all(pred_file(model, p).exists() for p in PERTS):
            continue
        od = pred_file(model, "x").parent
        extra = ["--keep-sets", *[f"{p}={','.join(keep or mx.ALL13)}" for p, keep in PERTS.items()]]
        if ref:
            extra += ["--reference", mx.reference_of(mx.run_name(ref, res), "hct116")]
        rc = sh([PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
                 "--checkpoint", ck, "--cell", "hct116", "--bigwig-dir", bw,
                 "--target", mx.target("hct116", res), "--tta", "--assay", assay,
                 "--binsize", res, "--device", "cpu", "--out-dir", od, *extra,
                 "--save-pred", str(od / "{set}.npz")], od / "predict.log", cpu_env())
        log(f"predict {model} rc={rc}")


def stage_predict_chip():
    for model, (run, res, ref) in MODELS.items():
        if not (run / "best").exists():          # not trained yet
            continue
        ck = run / (run / "best").read_text().strip()
        _, _, assay, _ = mx.CELLS["hct116"]
        for name, bw in CHIP_DIRS.items():
            f = pred_file(model, name)
            if f.exists():
                continue
            od = f.parent / name
            extra = []
            if ref:
                extra += ["--reference", mx.reference_of(mx.run_name(ref, res), "hct116")]
            rc = sh([PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
                     "--checkpoint", ck, "--cell", "hct116", "--bigwig-dir", bw,
                     "--target", mx.target("hct116", res), "--tta", "--assay", assay,
                     "--binsize", res, "--device", "cpu", "--out-dir", od, *extra,
                     "--save-pred", f], od / "predict.log", cpu_env())
            log(f"predict {model} {name} rc={rc}")


# --------------------------------------------------------------------------- #
def regions() -> pd.DataFrame:
    r = pd.read_csv(REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    r["key"] = r.chrom + ":" + r.start.astype(str) + "-" + r.end.astype(str)
    return r


_C: dict[str, cooler.Cooler] = {}


def _fetch(p: Path, chrom: str, b0: int, b1: int) -> np.ndarray | None:
    p = str(p)
    if p not in _C:
        _C[p] = cooler.Cooler(p)
    c = _C[p]
    res = c.binsize
    try:
        M = np.asarray(c.matrix(balance=False).fetch(
            f"{chrom}:{b0 * res}-{min(b1 * res, int(c.chromsizes[chrom]))}"), float)
    except Exception:                                                # noqa: BLE001
        return None
    if M.shape[0] != b1 - b0:
        return None
    return np.where(M == 0, np.nan, M)


def loop_anchors(cell: str, chrom: str, b0: int, n: int, res: int) -> np.ndarray:
    f = CALLS / f"{cell}_10kb_loops.bedgraph"
    if not f.exists():
        return np.zeros((0, 2), int)
    L = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 3, 4])
    L.columns = ["c1", "s1", "c2", "s2"]
    L = L[(L.c1 == chrom) & (L.c2 == chrom)]
    a = np.floor((L.s1.values + 5000) / res).astype(int) - b0
    b = np.floor((L.s2.values + 5000) / res).astype(int) - b0
    k = (a >= 0) & (b < n) & (b - a >= 2)
    return np.unique(np.stack([a[k], b[k]], 1), axis=0) if k.any() else np.zeros((0, 2), int)


def boundaries(cell: str, chrom: str, b0: int, n: int, res: int, w: int) -> np.ndarray:
    f = CALLS / f"{cell}_{res // 1000}kb_boundaries.bed"
    if not f.exists():
        return np.zeros(0, int)
    B = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1], comment="#")
    B.columns = ["chrom", "start"]
    idx = np.floor((B[B.chrom == chrom].start.values + res // 2) / res).astype(int) - b0
    return np.unique(idx[(idx >= w) & (idx < n - w)])


def strength(M: np.ndarray, ij: np.ndarray, tol: int = 1) -> np.ndarray:
    out = np.full(len(ij), np.nan)
    n = M.shape[0]
    for k, (i, j) in enumerate(ij):
        sub = M[max(0, i - tol):min(n, i + tol + 1), max(0, j - tol):min(n, j + tol + 1)]
        sub = sub[np.isfinite(sub)]
        if sub.size:
            out[k] = sub.max()
    return out


def stage_score():
    RES_OUT.mkdir(parents=True, exist_ok=True)
    R = regions()
    rows, loops_rows, bnd_rows, pdist = [], [], [], {}
    apa = {}
    for exp in EXPERIMENTS:
        for model, (run, res, _) in MODELS.items():
            u_p, t_p = oe_path(exp, "untreated", res), oe_path(exp, "treated", res)
            if not (u_p.exists() and t_p.exists()):
                continue
            preds = {p: pred_file(model, p) for p in PERTS}
            if not all(f.exists() for f in preds.values()):
                continue
            Z = {p: np.load(f) for p, f in preds.items()}
            pairs = [(p, "full", p) for p in PERTS if p != "full"]
            if exp in CHIP_EXP and all(pred_file(model, c).exists() for c in CHIP_DIRS):
                for c in CHIP_DIRS:
                    Z[c] = np.load(pred_file(model, c))
                pairs.append(("measured_chip", "chip_untreated", "chip_auxin6h"))
            w = 5 if res == 10000 else 3
            for _, r in R.iterrows():
                b0, b1 = r.start // res, -(-r.end // res)
                n = b1 - b0
                U, T = _fetch(u_p, r.chrom, b0, b1), _fetch(t_p, r.chrom, b0, b1)
                if U is None or T is None or r.key not in Z["full"].files:
                    continue
                P0 = np.asarray(Z["full"][r.key], float)
                if P0.shape[0] != n:
                    continue
                dM = T - U
                mask = bm.triu_mask(n, 2) & np.isfinite(dM) & np.isfinite(P0)
                anchors = loop_anchors(EXPERIMENTS[exp]["calls"], r.chrom, b0, n, res)
                bnds = boundaries(EXPERIMENTS[exp]["calls"], r.chrom, b0, n, res, w)
                for pert, base_k, pert_k in pairs:
                    if r.key not in Z[pert_k].files or r.key not in Z[base_k].files:
                        continue
                    P0 = np.asarray(Z[base_k][r.key], float)
                    P1 = np.asarray(Z[pert_k][r.key], float)
                    dP = P1 - P0
                    rows.append({"experiment": exp, "model": model, "perturbation": pert,
                                 "region": r.key,
                                 "pearson": bm.pearson(dP, dM, mask),
                                 "spearman": bm.spearman(dP, dM, mask),
                                 "scc": bm.scc(dP, dM, mask),
                                 "pred_abs_change": float(np.nanmean(np.abs(dP[mask]))),
                                 "meas_abs_change": float(np.nanmean(np.abs(dM[mask])))})
                    for d, (a, b) in bm.per_distance(dP, dM, mask).items():
                        pdist.setdefault((exp, model, pert, d * res), []).append((a, b))
                    if len(anchors):
                        su, st_ = strength(U, anchors), strength(T, anchors)
                        s0, s1 = strength(P0, anchors), strength(P1, anchors)
                        for k, (i, j) in enumerate(anchors):
                            loops_rows.append({"experiment": exp, "model": model,
                                               "perturbation": pert, "region": r.key,
                                               "i": int(i), "j": int(j),
                                               "distance_bp": int((j - i) * res),
                                               "meas_untreated": su[k], "meas_treated": st_[k],
                                               "pred_full": s0[k], "pred_pert": s1[k]})
                        key = (exp, model, pert)
                        acc = apa.setdefault(key, {"n": 0, "U": 0, "T": 0, "P0": 0, "P1": 0})
                        h = 3
                        for (i, j) in anchors:
                            if i - h < 0 or j + h >= n:
                                continue
                            subs = [M[i - h:i + h + 1, j - h:j + h + 1] for M in (U, T, P0, P1)]
                            if not all(np.isfinite(s).all() for s in subs):
                                continue
                            for nm, s in zip(("U", "T", "P0", "P1"), subs):
                                acc[nm] = acc[nm] + bs.oe_from_log(s)
                            acc["n"] += 1
                    if len(bnds):
                        iu_, it_ = bm.insulation(U, w), bm.insulation(T, w)
                        i0, i1 = bm.insulation(P0, w), bm.insulation(P1, w)
                        for b in bnds:
                            bnd_rows.append({"experiment": exp, "model": model,
                                             "perturbation": pert, "region": r.key, "bin": int(b),
                                             "meas_untreated": iu_[b], "meas_treated": it_[b],
                                             "pred_full": i0[b], "pred_pert": i1[b]})
            log(f"scored {exp} {model}")
    D = pd.DataFrame(rows)
    D.to_csv(RES_OUT / "change_per_region.csv.gz", index=False, float_format="%.5f")
    L = pd.DataFrame(loops_rows)
    L.to_csv(RES_OUT / "loops_per_anchor.csv.gz", index=False, float_format="%.5f")
    Bn = pd.DataFrame(bnd_rows)
    Bn.to_csv(RES_OUT / "boundaries_per_bin.csv.gz", index=False, float_format="%.5f")
    pdr = []
    for (exp, model, pert, dbp), lst in pdist.items():
        a = np.concatenate([x for x, _ in lst]); b = np.concatenate([y for _, y in lst])
        if a.size > 50 and a.std() > 0 and b.std() > 0:
            pdr.append({"experiment": exp, "model": model, "perturbation": pert,
                        "distance_bp": dbp, "pearson": float(np.corrcoef(a, b)[0, 1]),
                        "n": int(a.size)})
    PD = pd.DataFrame(pdr)
    PD.to_csv(RES_OUT / "change_per_distance.csv", index=False, float_format="%.5f")
    # summary
    summ = []
    for (exp, model, pert), g in D.groupby(["experiment", "model", "perturbation"]):
        s = {"experiment": exp, "model": model, "perturbation": pert}
        for k in ("pearson", "scc", "spearman"):
            s.update(bm.summarize(g[k], f"change_{k}"))
        s["pred_abs_change_mean"] = g.pred_abs_change.mean()
        s["meas_abs_change_mean"] = g.meas_abs_change.mean()
        l = L[(L.experiment == exp) & (L.model == model) & (L.perturbation == pert)] if len(L) else L
        if len(l):
            dm = l.meas_treated - l.meas_untreated
            dp = l.pred_pert - l.pred_full
            ok = np.isfinite(dm) & np.isfinite(dp)
            s.update({"loops_n": int(ok.sum()),
                      "loops_meas_median_change": float(dm[ok].median()),
                      "loops_pred_median_change": float(dp[ok].median()),
                      "loops_meas_frac_weaker": float((dm[ok] < 0).mean()),
                      "loops_pred_frac_weaker": float((dp[ok] < 0).mean()),
                      "loops_change_spearman": float(pd.Series(dm[ok]).corr(pd.Series(dp[ok]),
                                                                            method="spearman"))})
        a = apa.get((exp, model, pert))
        if a and a["n"]:
            for nm in ("U", "T", "P0", "P1"):
                s[f"apa_{nm}"] = bs.apa_score(a[nm], a["n"], 1)
            s["apa_n"] = a["n"]
            np.savez(OUT / f"apa_{exp}_{model}_{pert}.npz",
                     **{nm: a[nm] / a["n"] for nm in ("U", "T", "P0", "P1")})
        b = Bn[(Bn.experiment == exp) & (Bn.model == model) & (Bn.perturbation == pert)] if len(Bn) else Bn
        if len(b):
            dm = b.meas_treated - b.meas_untreated
            dp = b.pred_pert - b.pred_full
            ok = np.isfinite(dm) & np.isfinite(dp)
            s.update({"boundaries_n": int(ok.sum()),
                      "boundary_ins_meas_median_change": float(dm[ok].median()),
                      "boundary_ins_pred_median_change": float(dp[ok].median()),
                      "boundary_change_spearman": float(pd.Series(dm[ok]).corr(
                          pd.Series(dp[ok]), method="spearman"))})
        summ.append(s)
    S = pd.DataFrame(summ)
    S.to_csv(RES_OUT / "perturb_summary.csv", index=False, float_format="%.4f")
    log(f"summary {len(S)} rows")


def stage_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    PD = pd.read_csv(RES_OUT / "change_per_distance.csv")
    for exp, g in PD.groupby("experiment"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, res in zip(axes, (10000, 25000)):
            for (model, pert), h in g.groupby(["model", "perturbation"]):
                if not model.endswith(f"{res // 1000}kb"):
                    continue
                h = h.sort_values("distance_bp")
                ax.plot(h.distance_bp / 1e3, h.pearson,
                        "-" if pert == EXP_PERT.get(exp) else ":",
                        label=f"{model.split('_')[0]} {pert}")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("genomic distance (kb)")
            ax.set_title(f"{res // 1000} kb")
        axes[0].set_ylabel("Pearson, predicted vs measured change")
        axes[1].legend(fontsize=7, frameon=False)
        fig.suptitle(EXPERIMENTS[exp]["label"])
        fig.tight_layout()
        fig.savefig(fd / f"{exp}_change_per_distance.png", dpi=150)
        plt.close(fig)
    # APA panels
    for f in sorted(OUT.glob("apa_*.npz")):
        z = np.load(f)
        fig, axes = plt.subplots(1, 4, figsize=(11, 3))
        for ax, nm, t in zip(axes, ("U", "T", "P0", "P1"),
                             ("measured untreated", "measured treated", "predicted full",
                              "predicted perturbed")):
            ax.imshow(np.log2(z[nm]), cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            ax.set_title(f"{t}\nAPA {bs.apa_score(z[nm], 1, 1):.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f.stem, fontsize=8)
        fig.tight_layout()
        fig.savefig(fd / f"{f.stem}.png", dpi=130)
        plt.close(fig)
    region_figures(fd)


def region_figures(fd, n_show=3):
    import matplotlib.pyplot as plt
    D = pd.read_csv(RES_OUT / "change_per_region.csv.gz")
    for exp, pert in EXP_PERT.items():
        for model, (run, res, _) in MODELS.items():
            g = D[(D.experiment == exp) & (D.model == model) & (D.perturbation == pert)]
            g = g.dropna(subset=["pearson"]).sort_values("meas_abs_change")
            if g.empty:
                continue
            pick = g.region.values[-n_show:]            # strongest measured change
            Z0, Z1 = np.load(pred_file(model, "full")), np.load(pred_file(model, pert))
            fig, axes = plt.subplots(n_show, 4, figsize=(12, 3 * n_show), squeeze=False)
            for i, key in enumerate(pick):
                chrom, se = key.split(":")
                s, e = map(int, se.split("-"))
                b0, b1 = s // res, -(-e // res)
                U = _fetch(oe_path(exp, "untreated", res), chrom, b0, b1)
                T = _fetch(oe_path(exp, "treated", res), chrom, b0, b1)
                P0, P1 = np.asarray(Z0[key], float), np.asarray(Z1[key], float)
                iu = np.triu_indices(U.shape[0], 1)

                def comb(up, lo):
                    M = np.array(lo, float); M[iu] = up[iu]; return M
                r = g[g.region == key].pearson.values[0]
                for j, (M, t, v) in enumerate([
                        (comb(U, P0), "untreated: measured / predicted", 1.2),
                        (comb(T, P1), "treated: measured / predicted", 1.2),
                        (comb(T - U, P1 - P0), f"change: measured / predicted (r={r:.2f})", None),
                        (comb(T - U, (P1 - P0) * np.nanstd(T - U) /
                              max(np.nanstd(P1 - P0), 1e-9)), "change, predicted rescaled", None)]):
                    vv = v or np.nanquantile(np.abs(M), 0.98)
                    axes[i, j].imshow(M, cmap="RdBu_r", vmin=-vv, vmax=vv, interpolation="none")
                    axes[i, j].set_title(t, fontsize=7)
                    axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                axes[i, 0].set_ylabel(key, fontsize=7)
            fig.suptitle(f"{EXPERIMENTS[exp]['label']}; {model}, {pert}", fontsize=9)
            fig.tight_layout()
            fig.savefig(fd / f"{exp}_{model}_regions.png", dpi=120)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    a = ap.parse_args()
    for st in a.stage:
        globals()[f"stage_{st}"]()


if __name__ == "__main__":
    main()
