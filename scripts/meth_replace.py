"""How far can DNA methylation replace ChIP-seq signal (registry section 35, part A).

Inference only. A methylation model (all12refres_drop_meth_10kb, 13 ChIP tracks
+ Meth, track dropout, residual on the leave-one-out / all-12 consensus) is
predicted with ChIP groups removed, each set with and without the Meth channel,
next to the same model without the channel (all12refres_drop_10kb) on the
same sets without Meth:

  groups   CTCF; cohesin = RAD21 + SMC3; active = H3K27ac, H3K4me1, H3K4me2,
           H3K4me3, H3K9ac; genebody = H3K36me3, H3K79me2, H4K20me1;
           repressive = H3K27me3, H3K9me3
  sets     full (all 13); minus_<group> for each group; cumulative removals in
           the order cohesin, repressive, genebody, active, CTCF
           (cum1 .. cum3, then ctcf_only = cum4 and seq_only = cum5); the
           ChIP-light sets ctcf_only, ctcf_h3k27ac, ctcf_h3k27ac_h3k27me3,
           seq_only (motifs and sequence features only)
  every set also with Meth (suffix +meth) for the methylation model.

Scores per region (413 Akita test regions, crosscell_predict --structure):
Pearson, SCC, insulation correlation, boundary F1, loop enrichment correlation,
loop APA ratio, compartment correlation, first-eigenvector agreement; the
consensus the model reads is scored alike (floor). Replacement fraction of a
set S per cell and metric:

    RF = (score(S + Meth) - score(S)) / (score(full) - score(S))

reported where |score(full) - score(S)| exceeds the seed noise of the median
Pearson (0.013); smaller denominators are flagged, not reported as a fraction.

    python scripts/meth_replace.py --stage predict summary
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import methylation as me                                            # noqa: E402
import multicell_expanded as mx                                     # noqa: E402

OUT = mx.HM / "runs/meth_replace"
RES_OUT = REPO / "results/methylation"
PY = sys.executable
NOISE = 0.013            # seed noise of the median per-region Pearson (section 15)
GROUPS = {"CTCF": ["CTCF"], "cohesin": ["RAD21", "SMC3"],
          "active": ["H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac"],
          "genebody": ["H3K36me3", "H3K79me2", "H4K20me1"],
          "repressive": ["H3K27me3", "H3K9me3"]}
CUM_ORDER = ["cohesin", "repressive", "genebody", "active", "CTCF"]
METRICS = {"pearson": "model", "scc": "struct_scc", "insulation": "struct_insulation_corr",
           "boundary_f1": "struct_boundary_f1", "loop_enrichment": "struct_loop_enrich_corr",
           "loop_apa": "struct_loop_apa_ratio", "compartment": "struct_compartment_corr",
           "eigenvector": "struct_eigenvector_abs_corr"}
# model -> (has Meth channel, version 2 cell set)
MODELS = {"all12refres_drop_meth_10kb": (True, False),
          "all12refres_drop_10kb": (False, False),
          "all12refres_drop_meth_v2_10kb": (True, True)}
CELLS = ["k562", "hmec_deep", "gm12878", "imr90", "hepg2", "a549", "h1"]
CELLS_V2 = CELLS + ["hct116", "mcf7", "hff"]
HELD_OUT = {"k562", "hmec_deep"}


def chip_sets() -> dict[str, list[str]]:
    allt = mx.ALL13
    sets = {"full": list(allt)}
    for g, t in GROUPS.items():
        sets[f"minus_{g}"] = [x for x in allt if x not in t]
    gone: list[str] = []
    for i, g in enumerate(CUM_ORDER, 1):
        gone += GROUPS[g]
        sets[f"cum{i}"] = [x for x in allt if x not in gone]
    sets["ctcf_only"] = ["CTCF"]
    sets["ctcf_h3k27ac"] = ["CTCF", "H3K27ac"]
    sets["ctcf_h3k27ac_h3k27me3"] = ["CTCF", "H3K27ac", "H3K27me3"]
    sets["seq_only"] = []
    # cum4 equals ctcf_only and cum5 equals seq_only: one prediction each
    del sets["cum4"], sets["cum5"]
    return sets


SETS = chip_sets()
SET_ORDER = list(SETS)


def keep_sets(model: str) -> dict[str, list[str]]:
    has_meth = MODELS[model][0]
    out = dict(SETS)
    if has_meth:
        out.update({f"{k}+meth": v + [me.TRACK] for k, v in SETS.items()})
    return out


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "meth_replace.log", "a") as fh:
        fh.write(line + "\n")


def predict(model: str, cell: str, gpu: str) -> int:
    run = mx.out_of(model) / model
    ck = mx.best_ckpt(run)
    out = OUT / model
    sets = keep_sets(model)
    if all((out / s / f"predict_{cell}.csv").exists() for s in sets):
        return 0
    res = mx.split_name(model)[1]
    _, bw, assay, _ = mx.CELLS[cell]
    ref = mx.reference_of(model, cell)
    tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
           "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw,
           "--target", mx.target(cell, res), "--tta", "--assay", assay, "--binsize", res,
           "--device", "cuda", "--out-dir", out, "--reference", ref, "--structure",
           "--keep-sets", *[f"{s}={','.join(t)}" for s, t in sets.items()],
           "--others", f"{tag}={ref}"]
    v2 = MODELS[model][1]
    if MODELS[model][0] and me.has_meth(cell, v2):
        cmd += ["--extra-bigwig-dir", me.meth_dir(cell, v2)]
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3",
               CUDA_VISIBLE_DEVICES=gpu)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"predict_{cell}.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def stage_predict(models, procs_per_gpu=2):
    tasks = [(m, c) for m in models for c in (CELLS_V2 if MODELS[m][1] else CELLS)]
    slots = [str(g) for g in (0, 1) for _ in range(procs_per_gpu)]

    # one thread per slot, each walks its own list of tasks
    lanes = [[t for j, t in enumerate(tasks) if j % len(slots) == s] for s in range(len(slots))]

    def lane(s):
        for m, c in lanes[s]:
            rc = predict(m, c, slots[s])
            log(f"predict {m} {c} gpu{slots[s]} rc={rc}")
    with ThreadPoolExecutor(len(slots)) as ex:
        list(ex.map(lane, range(len(slots))))


def read_scores(models) -> pd.DataFrame:
    rows = []
    for model in models:
        cells = CELLS_V2 if MODELS[model][1] else CELLS
        for s in keep_sets(model):
            for cell in cells:
                f = OUT / model / s / f"predict_{cell}.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                base = s.removesuffix("+meth")
                rec = {"model": model, "set": base, "meth": s.endswith("+meth"), "cell": cell,
                       "held_out": cell in HELD_OUT, "n_regions": int(df.model.notna().sum())}
                for m, col in METRICS.items():
                    rec[m] = df[col].median() if col in df else np.nan
                cons = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
                rec["pearson_consensus"] = df[f"base_{cons}"].median() if f"base_{cons}" in df else np.nan
                for m, col in METRICS.items():
                    if m != "pearson" and f"struct_{cons}_{col[7:]}" in df:
                        rec[f"{m}_consensus"] = df[f"struct_{cons}_{col[7:]}"].median()
                rows.append(rec)
    return pd.DataFrame(rows)


def replacement(S: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    d = S[S.model == model]
    for cell, g in d.groupby("cell"):
        full = g[(g.set == "full") & (~g.meth)]
        if full.empty:
            continue
        for st in SET_ORDER:
            if st == "full":
                continue
            a, b = g[(g.set == st) & (~g.meth)], g[(g.set == st) & g.meth]
            if a.empty or b.empty:
                continue
            for m in METRICS:
                s0, s1, sf = float(a[m].iloc[0]), float(b[m].iloc[0]), float(full[m].iloc[0])
                den = sf - s0
                rows.append({"model": model, "set": st, "cell": cell, "metric": m,
                             "held_out": cell in HELD_OUT, "score_without_meth": s0,
                             "score_with_meth": s1, "score_full": sf,
                             "gain_from_meth": s1 - s0, "loss_without_group": den,
                             "replacement_fraction": (s1 - s0) / den if abs(den) >= NOISE else np.nan,
                             "denominator_below_noise": abs(den) < NOISE})
    return pd.DataFrame(rows)


def pooled(R: pd.DataFrame, n_boot=500, seed=0) -> pd.DataFrame:
    """Pooled replacement fraction over cells: mean gain from Meth / mean loss
    without the group, for the training cells with WGBS, the held-out cells and
    all, with a percentile bootstrap over cells; undefined (NaN) where the mean
    loss is below the seed noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for (st, m), g in R.groupby(["set", "metric"]):
        for lab, sel in (("training", g[~g.held_out]), ("held_out", g[g.held_out]), ("all", g)):
            if sel.empty:
                continue
            gain, loss = sel.gain_from_meth.values, sel.loss_without_group.values
            den = loss.mean()
            rf = gain.mean() / den if abs(den) >= NOISE else np.nan
            bs = []
            if len(sel) >= 3:
                for _ in range(n_boot):
                    i = rng.integers(0, len(sel), len(sel))
                    d = loss[i].mean()
                    if abs(d) >= NOISE:
                        bs.append(gain[i].mean() / d)
            rows.append({"set": st, "metric": m, "cells": lab, "n_cells": len(sel),
                         "mean_loss_without_group": den, "mean_gain_from_meth": gain.mean(),
                         "replacement_fraction": rf,
                         "ci_low": np.percentile(bs, 2.5) if len(bs) > 50 else np.nan,
                         "ci_high": np.percentile(bs, 97.5) if len(bs) > 50 else np.nan})
    return pd.DataFrame(rows)


def bootstrap_pearson_rf(model: str, st: str, cell: str, n=300, seed=0):
    """Paired bootstrap over regions of the Pearson replacement fraction."""
    def col(s):
        f = OUT / model / s / f"predict_{cell}.csv"
        return pd.read_csv(f).set_index("region").model if f.exists() else None
    a, b, f = col(st), col(st + "+meth"), col("full")
    if a is None or b is None or f is None:
        return np.nan, np.nan
    idx = a.index.intersection(b.index).intersection(f.index)
    a, b, f = a[idx].values, b[idx].values, f[idx].values
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(idx), len(idx))
        den = np.nanmedian(f[i]) - np.nanmedian(a[i])
        if abs(den) >= NOISE:
            out.append((np.nanmedian(b[i]) - np.nanmedian(a[i])) / den)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) > 30 \
        else (np.nan, np.nan)


def stage_summary(models):
    RES_OUT.mkdir(parents=True, exist_ok=True)
    S = read_scores(models)
    S.to_csv(RES_OUT / "meth_replace_scores.csv", index=False, float_format="%.4f")
    for model in models:
        if not MODELS[model][0]:
            continue
        R = replacement(S, model)
        if R.empty:
            continue
        ci = {}
        for (st, cell) in R[["set", "cell"]].drop_duplicates().itertuples(index=False):
            ci[(st, cell)] = bootstrap_pearson_rf(model, st, cell)
        R["rf_ci_low"] = [ci[(s, c)][0] if m == "pearson" else np.nan
                          for s, c, m in zip(R.set, R.cell, R.metric)]
        R["rf_ci_high"] = [ci[(s, c)][1] if m == "pearson" else np.nan
                           for s, c, m in zip(R.set, R.cell, R.metric)]
        R.to_csv(RES_OUT / f"meth_replacement_fraction_{model}.csv", index=False,
                 float_format="%.4f")
        # median over cells, training and held-out separately, reliable denominators only
        agg = (R.dropna(subset=["replacement_fraction"])
               .groupby(["set", "metric", "held_out"])
               .agg(rf_median=("replacement_fraction", "median"),
                    n_cells=("replacement_fraction", "size")).reset_index())
        agg.to_csv(RES_OUT / f"meth_replacement_fraction_{model}_median.csv", index=False,
                   float_format="%.3f")
        P = pooled(R)
        P.to_csv(RES_OUT / f"meth_replacement_fraction_{model}_pooled.csv", index=False,
                 float_format="%.3f")
        print(model)
        print(agg[agg.metric == "pearson"].to_string(index=False))
    return S


# --------------------------------------------------------------------------- #
# part C: dedicated 10 kb trainings with the Meth channel, against the existing
# minimal-input runs (runs/minimal_inputs, section 20), same cells and metrics
DEDICATED_OLD = ["min_ctcf_h3k27ac_refres_10kb", "min_ctcf_rad21_refres_10kb",
                 "min_one_per_group_refres_10kb", "min_motifs_only_refres_10kb"]
DEDICATED_METH = ["min_meth_only_refres_10kb", "min_ctcf_meth_refres_10kb",
                  "min_ctcf_h3k27ac_meth_refres_10kb"]
DEDICATED_CELLS = ["gm12878", "imr90", "k562", "hmec_deep"]      # all have Meth
MIN_OUT = mx.HM / "runs/minimal_inputs"


def predict_dedicated(run: str, cell: str, gpu: str) -> int:
    out = OUT / "dedicated" / run
    if (out / f"predict_{cell}.csv").exists():
        return 0
    ck = mx.best_ckpt(MIN_OUT / run)
    _, bw, assay, _ = mx.CELLS[cell]
    ref = mx.reference_of("all12refres_10kb", cell)
    tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
           "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw,
           "--target", mx.target(cell, 10000), "--tta", "--assay", assay, "--binsize", 10000,
           "--device", "cuda", "--out-dir", out, "--reference", ref, "--structure",
           "--others", f"{tag}={ref}"]
    if "meth" in run and me.has_meth(cell, True):
        cmd += ["--extra-bigwig-dir", me.meth_dir(cell, True)]
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", CUDA_VISIBLE_DEVICES=gpu)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"predict_{cell}.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def stage_dedicated(runs=None, procs_per_gpu=1):
    runs = runs or (DEDICATED_OLD + DEDICATED_METH)
    def finished(r):          # the training log of minimal_inputs.py ends with "done."
        f = MIN_OUT / f"{r}.train.log"
        return (MIN_OUT / r / "best").exists() and f.exists() and "done." in f.read_text(errors="replace")
    runs = [r for r in runs if finished(r)]
    tasks = [(r, c) for r in runs for c in DEDICATED_CELLS]
    slots = [str(g) for g in (0, 1) for _ in range(procs_per_gpu)]
    lanes = [[t for j, t in enumerate(tasks) if j % len(slots) == s] for s in range(len(slots))]

    def lane(s):
        for r, c in lanes[s]:
            log(f"dedicated {r} {c} gpu{slots[s]} rc={predict_dedicated(r, c, slots[s])}")
    with ThreadPoolExecutor(len(slots)) as ex:
        list(ex.map(lane, range(len(slots))))


def summary_dedicated():
    rows = []
    for run in DEDICATED_OLD + DEDICATED_METH:
        for cell in DEDICATED_CELLS:
            f = OUT / "dedicated" / run / f"predict_{cell}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            rec = {"run": run, "cell": cell, "held_out": cell in HELD_OUT,
                   "n_regions": int(df.model.notna().sum())}
            for m, col in METRICS.items():
                rec[m] = df[col].median() if col in df else np.nan
            cons = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
            rec["pearson_consensus"] = df[f"base_{cons}"].median() if f"base_{cons}" in df else np.nan
            rows.append(rec)
    D = pd.DataFrame(rows)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(RES_OUT / "meth_dedicated.csv", index=False, float_format="%.4f")
    if len(D):
        print(D.pivot_table(index="run", columns="cell", values="pearson").round(4).to_string())
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", choices=["predict", "summary", "dedicated", "dedicated_summary"],
                    required=True)
    ap.add_argument("--models", nargs="*", default=["all12refres_drop_meth_10kb",
                                                    "all12refres_drop_10kb"])
    ap.add_argument("--procs-per-gpu", type=int, default=2)
    a = ap.parse_args()
    if "predict" in a.stage:
        stage_predict(a.models, a.procs_per_gpu)
    if "summary" in a.stage:
        stage_summary(a.models)
    if "dedicated" in a.stage:
        stage_dedicated()
    if "dedicated_summary" in a.stage:
        summary_dedicated()


if __name__ == "__main__":
    main()
