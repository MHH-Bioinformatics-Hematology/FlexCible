"""Flexible input on the one model all12one: accuracy against the set of ChIP marks given.

Question: a model trained with 13 marks under track dropout (uniform_count) and with the
consensus map as an input plane predicts from any subset of marks. How does the accuracy
depend on how many and which marks are given, with and without the consensus?

Model: runs/multicell_reference/all12one_<k>kb (per-resolution views of the one shared
checkpoint). Cells: held-out k562, hmec, huvec; training cells gm12878, imr90 (reference =
their leave-one-out consensus; held-out cells: the 12-cell consensus). Metric: per-region
Pearson correlation on the 413 Akita test regions (crosscell_predict.py, --tta as in
multicell_expanded.py), median and quartiles over regions. A mark a cell lacks (RAD21 and
SMC3 of hmec and huvec) is absent anyway; a set that asks for it is scored with what exists
and `marks_present` records what was actually given.

Input sets (subsets.json):
  all13       all 13 marks              single      one mark alone (13)
  minus       one mark removed (13)     random      sizes 1..12, 20 draws each (numpy seed 0)
  seven       CTCF H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me3 H3K9me3
  ctcf_h3k27ac, ctcf_rad21              none        no marks (sequence and motifs only)
Every set is predicted with the consensus (with_reference=1) and with the consensus plane at
0, the expected level (with_reference=0; crosscell_predict.py keep sets named *_noref).
Programme: 25 and 10 kb, all sets, five cells; 5 and 2 kb, the smaller programme (all13,
single, seven, ctcf_h3k27ac, none, random sizes 3 and 6) for k562 and gm12878.

Stages (resumable: a chunk of sets whose output files all exist is skipped):
  python scripts/flexible_inputs.py predict --cells k562 --res 10 25 --gpu 0
  python scripts/flexible_inputs.py score          # per-set csvs -> per_region.csv, summary.csv
  python scripts/flexible_inputs.py summary        # curve.csv, single/minus rankings, tables.md
Options: --chunk N sets per crosscell_predict call (default 60), --set-batch (sets per forward
pass), env HICMULTI_HM (data root) and FLEX_REGIONS (regions TSV) for hosts without the share, --dry prints the plan only.
Raw per-set csvs: $HICMULTI_HM/runs/flexible_inputs/<cell>_<k>kb/<chunk>/<set>/predict_<cell>.csv

Other checkpoints (release candidate, 14 tracks with methylation): `--model all12one_fullrd` and
`--meth 1` (every set also gets the cell's methylation track "Meth", read from $HICMULTI_HM/meth/<cell>
or the share; the marks of a set are the 13 ChIP marks as before, so the subsets, the draws and the
scoring are unchanged) or `--meth 0` (Meth absent, the 13-mark design of all12one). Results go to
results/flexible_inputs/<model>/ (--meth 1) and results/flexible_inputs/<model>/nometh/ (--meth 0),
raw csvs to runs/flexible_inputs/<model>/{meth,nometh}/<cell>_<k>kb/. A cell without a methylation
track (huvec) is run with Meth absent in both variants. The default (--model all12one) writes where
section 87 wrote.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
OE = HM / "crosscell_oe"
RUNS = HM / "runs/multicell_reference"
RAW = HM / "runs/flexible_inputs"
OUT = REPO / "results/flexible_inputs"
PY = os.environ.get("HICMULTI_PY", sys.executable)
MODEL = "all12one"
METH = False


def set_model(model, meth):
    """point the module at another checkpoint family; meth: True/False, None = not applicable"""
    global MODEL, METH, RAW, OUT
    MODEL, METH = model, bool(meth)
    if model != "all12one":
        RAW = HM / "runs/flexible_inputs" / model / ("meth" if METH else "nometh")
        OUT = REPO / "results/flexible_inputs" / model / ("" if METH else "nometh")


def meth_dir(cell):
    for d in (HM / "meth" / cell, Path("/mnt/processed/wolffjoa/hic-multiome/meth") / cell):
        if (d / "Meth.bigwig").exists():
            return d
    return None

MARKS = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
         "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
SEVEN = ["CTCF", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]
HELD_OUT = ["k562", "hmec", "huvec"]
TRAINING = ["gm12878", "imr90"]
BIGWIG = {"gm12878": Path.home() / "data/hic-multiome/hg38_gm12878"}
REGIONS = os.environ.get("FLEX_REGIONS")     # optional regions TSV (remote host without the share)
TARGET_TAG = {"hmec": "rep1_ice", "huvec": "rep1_ice"}      # as in multicell_expanded.target()
FULL = {"cells": ["k562", "hmec", "huvec", "gm12878", "imr90"], "res": [25, 10]}
SMALL = {"cells": ["k562", "gm12878"], "res": [5, 2]}
SMALL_KINDS = {"all13", "single", "seven", "ctcf_h3k27ac", "none", "random"}
SMALL_SIZES = {3, 6}


def bigwig_dir(cell):
    if (HM / f"hg38_{cell}").exists():
        return HM / f"hg38_{cell}"
    return BIGWIG.get(cell, HM / f"hg38_{cell}")


def present_marks(cell):
    have = {p.stem.upper() for p in bigwig_dir(cell).glob("*.bigwig")}
    return [m for m in MARKS if m.upper() in have]


def make_subsets():
    rng = np.random.default_rng(0)
    draws = {str(k): [sorted(rng.choice(len(MARKS), k, replace=False).tolist()) for _ in range(20)]
             for k in range(1, 13)}
    return {"marks": MARKS,
            "seed": 0,
            "draws_note": "indices into marks; size -> 20 draws",
            "random": draws,
            "seven": SEVEN}


def load_subsets():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "subsets.json"
    if not p.exists():
        p.write_text(json.dumps(make_subsets(), indent=1))
    return json.loads(p.read_text())


def all_sets():
    """(set_kind, set_name, marks) for the full programme."""
    sub = load_subsets()
    s = [("all13", "all13", MARKS)]
    s += [("single", f"single_{m}", [m]) for m in MARKS]
    s += [("minus", f"minus_{m}", [x for x in MARKS if x != m]) for m in MARKS]
    for k in range(1, 13):
        for d, idx in enumerate(sub["random"][str(k)]):
            s.append(("random", f"rand_k{k:02d}_d{d:02d}", [MARKS[i] for i in idx]))
    s += [("seven", "seven", SEVEN), ("ctcf_h3k27ac", "ctcf_h3k27ac", ["CTCF", "H3K27ac"]),
          ("ctcf_rad21", "ctcf_rad21", ["CTCF", "RAD21"]), ("none", "none", [])]
    return s


def sets_for(res):
    s = all_sets()
    if res in SMALL["res"]:
        s = [x for x in s if x[0] in SMALL_KINDS
             and (x[0] != "random" or int(x[1][6:8]) in SMALL_SIZES)]
    return s


def programme():
    for res in FULL["res"]:
        for c in FULL["cells"]:
            yield c, res
    for res in SMALL["res"]:
        for c in SMALL["cells"]:
            yield c, res


def reference(cell, k):
    return OE / (f"consensus_loo_{cell}_proper_{k}kb.cool" if cell in TRAINING
                 else f"consensus12_proper_{k}kb.cool")


def target(cell, k):
    return OE / f"{cell}_{TARGET_TAG.get(cell, 'proper')}_{k}kb.cool"


def chunks(sets, n):
    items = []
    for _, nm, m in sets:
        items.append((nm, ",".join(m)))
        items.append((nm + "_noref", ",".join(m)))
    return [items[i:i + n] for i in range(0, len(items), n)]


def predict(a):
    for cell, k in programme():
        if (a.cells and cell not in a.cells) or (a.res and k not in a.res):
            continue
        view = RUNS / f"{MODEL}_{k}kb"
        ck = view / "checkpoint-best"
        base = RAW / f"{cell}_{k}kb"
        for ci, ch in enumerate(chunks(sets_for(k), a.chunk)):
            out = base / f"chunk{ci:02d}"
            if all((out / nm / f"predict_{cell}.csv").exists() for nm, _ in ch):
                continue
            print(f"[predict] {cell} {k}kb chunk{ci:02d} ({len(ch)} sets)", flush=True)
            if a.dry:
                continue
            out.mkdir(parents=True, exist_ok=True)
            md_ = meth_dir(cell)
            extra = ["--extra-bigwig-dir", str(md_)] if md_ is not None else []
            if METH and md_ is not None:
                ch = [(nm, (m + ",Meth") if m else "Meth") for nm, m in ch]
            cmd = [PY, "scripts/crosscell_predict.py", "--config", str(ck / "config.yaml"),
                   "--checkpoint", str(ck), "--cell", cell, "--bigwig-dir", str(bigwig_dir(cell)),
                   "--target", str(target(cell, k)), "--tta", "--assay", "hic",
                   "--binsize", str(k * 1000), "--out-dir", str(out), *extra,
                   "--reference", str(reference(cell, k)),
                   "--others", f"consensus={reference(cell, k)}",
                   "--device", "cuda", "--set-batch", str(a.set_batch),
                   *(["--regions", REGIONS] if REGIONS else []), "--keep-sets", *[f"{nm}={m}" for nm, m in ch]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
            with open(out / "predict.log", "w") as lg:
                rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=lg, stderr=subprocess.STDOUT)
            if rc:
                print(f"[FAILED] {cell} {k}kb chunk{ci:02d} rc={rc} (see {out}/predict.log)", flush=True)


def score(a):
    rows, per = [], []
    for cell, k in programme():
        present = present_marks(cell)
        base = RAW / f"{cell}_{k}kb"
        for kind, nm, marks in sets_for(k):
            for wr, suf in ((1, ""), (0, "_noref")):
                f = next(iter(sorted(base.glob(f"chunk*/{nm}{suf}/predict_{cell}.csv"))), None)
                if f is None:
                    continue
                d = pd.read_csv(f)
                r = d["model"].dropna()
                cons = d["base_consensus"].dropna()
                given = [m for m in marks if m in present]
                q = r.quantile([0.25, 0.75]) if len(r) else pd.Series([np.nan, np.nan], index=[0.25, 0.75])
                rows.append({"cell": cell, "res": k, "set_kind": kind, "set_name": nm,
                             "marks_present": "+".join(given), "n_marks": len(marks),
                             "n_present": len(given), "meth_given": int(METH and meth_dir(cell) is not None), "with_reference": wr, "n_regions": len(r),
                             "median_r": r.median(), "q25": q[0.25], "q75": q[0.75],
                             "mean_r": r.mean(), "consensus_median_r": cons.median()})
                x = d[["region", "model"]].dropna().assign(cell=cell, res=k, set_kind=kind,
                                                            set_name=nm, with_reference=wr)
                per.append(x)
    if not rows:
        raise SystemExit("no predictions found")
    s = pd.DataFrame(rows)
    s.round(5).to_csv(OUT / "summary.csv", index=False)
    p = pd.concat(per)
    p["r"] = p.pop("model").round(4)
    cols = ["cell", "res", "set_kind", "set_name", "with_reference", "region", "r"]
    p[p.set_kind != "random"][cols].to_csv(OUT / "per_region.csv", index=False)
    p[p.set_kind == "random"][cols].to_csv(OUT / "per_region_random.csv.gz", index=False)
    print(f"summary.csv {len(s)} rows; per-region rows {len(p)}")


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def summary(a):
    s = pd.read_csv(OUT / "summary.csv")
    s["wr"] = s.with_reference.map({1: "consensus", 0: "no consensus"})
    t = []
    # accuracy against number of marks: median over draws and its range, per cell/res/reference
    # at 5 and 2 kb no random draws of size 1 exist: the 13 single marks stand in for them
    rnd = s[(s.set_kind == "random") | ((s.set_kind == "single") & s.res.isin(SMALL["res"]))]
    g = rnd.groupby(["cell", "res", "with_reference", "n_marks"]).median_r
    curve = g.agg(median="median", lo="min", hi="max").reset_index()
    ends = s[s.set_kind.isin(["all13", "none"])].assign(n_marks=lambda d: d.n_marks)[
        ["cell", "res", "with_reference", "n_marks", "median_r"]].rename(columns={"median_r": "median"})
    ends = ends.assign(lo=ends["median"], hi=ends["median"])
    curve = pd.concat([ends[ends.n_marks == 0], curve, ends[ends.n_marks == 13]]).sort_values(
        ["cell", "res", "with_reference", "n_marks"])
    curve.round(4).to_csv(OUT / "curve.csv", index=False)
    cons = s.groupby(["cell", "res"]).consensus_median_r.first()
    t.append("## Accuracy against number of marks (median over draws [min, max] of the per-draw median r)\n")
    for (res, wr), d in curve.groupby(["res", "with_reference"]):
        tab = d.assign(v=lambda x: x.apply(
            lambda r: f"{r['median']:.3f} [{r.lo:.3f}, {r.hi:.3f}]" if r.lo != r.hi else f"{r['median']:.3f}",
            axis=1)).pivot(index="n_marks", columns="cell", values="v").reset_index()
        t.append(f"### {res} kb, {'with' if wr else 'without'} consensus "
                 f"(consensus alone: " + ", ".join(f"{c} {cons[(c, res)]:.3f}"
                                                     for c in tab.columns[1:]) + ")\n")
        t.append(md(tab) + "\n")
    # single marks
    for kind, title in (("single", "Single mark alone"), ("minus", "Single mark removed")):
        d = s[s.set_kind == kind]
        for res in sorted(d.res.unique(), reverse=True):
            for wr in (1, 0):
                x = d[(d.res == res) & (d.with_reference == wr)].copy()
                if x.empty:
                    continue
                x["mark"] = x.set_name.str.split("_", n=1).str[1]
                tab = x.pivot(index="mark", columns="cell", values="median_r")
                tab["mean"] = tab.mean(axis=1)
                tab = tab.sort_values("mean", ascending=(kind == "minus")).reset_index()
                t.append(f"### {title}, {res} kb, {'with' if wr else 'without'} consensus\n")
                t.append(md(tab) + "\n")
    # named sets
    nm = s[s.set_kind.isin(["all13", "seven", "ctcf_h3k27ac", "ctcf_rad21", "none"])]
    t.append("## Named sets (median r)\n")
    tab = nm.pivot_table(index=["res", "wr", "set_name"], columns="cell", values="median_r").reset_index()
    t.append(md(tab) + "\n")
    (OUT / "tables.md").write_text("\n".join(t))
    print(f"wrote curve.csv, tables.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["predict", "score", "summary"])
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--res", nargs="*", type=int, help="resolutions in kb")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--set-batch", type=int, default=20, help="sets per forward pass")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--model", default="all12one", help="checkpoint family, views runs/multicell_reference/<model>_<k>kb")
    ap.add_argument("--meth", type=int, choices=[0, 1], default=0,
                    help="1: give the cell's methylation track in every set (models with a Meth track)")
    a = ap.parse_args()
    set_model(a.model, a.meth)
    OUT.mkdir(parents=True, exist_ok=True)
    load_subsets()
    {"predict": predict, "score": score, "summary": summary}[a.stage](a)


if __name__ == "__main__":
    main()
