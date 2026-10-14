"""Scoring of the retrained and the released Akita with the measures of
registry sections 31, 37 and 58, on pixels identical to our chrho models.

The block machinery of scripts/hic_measures.py is reused unchanged
(score_block: whole chromosome, pixels finite in the target and in every
source, band cut to the shortest reach, hicFindTADs / hicDetectLoops of
HiCExplorer 3.7.6, HiCRep, rank-matched loop recall, insulation, ...). Akita
predictions are read from preds/<run>/<chrom>.npy (predict.py) and sampled at
the bin centres of the 10 kb grid by hic_measures.resample_band (bilinear in
(bin, bin) space, a pixel kept when its four neighbours are predicted, the
convention of the C.Origami and Epiphany blocks of registry sections 37 / 58).
Akita covers 448 bins of 2048 bp = 917 504 bp, so the band is 90 bins (900 kb).

Sources per block
  ours      all12refin_drop_chrho and all12honest_drop_chrho (chrho models,
            trained without chr2, 3, 11, 14, 17; clean on these chromosomes)
  akitaA    Akita retrained on split A (same chromosome split as the chrho models)
  akitaB    Akita retrained on split B (odd chromosomes train, even test)
  akitapre  the released Akita (five heads: hff, h1, gm12878, imr90, hct116)
  baselines consensus (leave-one-out consensus of the other cells), best other
            measured map, replicate map where one exists
Blocks
  A  every training cell x chr2, 3, 11, 14, 17 (+ K562, which no Akita head
     covers, for contrast: ours only)
  B  every training cell x even chromosomes 4..22 (Akita B only; our models
     trained on these, so they are not scored there); chr2 and chr14 belong to
     the A blocks, which also carry Akita B (even, held out of both).

    python evaluate.py --stage ours_predict              # on the machine with the GPU
    python evaluate.py --stage score --jobs 4
    python evaluate.py --stage report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("HICMULTI_REPO", str(HERE.parents[1])))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import common as C  # noqa: E402
import hic_measures as hm  # noqa: E402

RES = int(os.environ.get("AKITA_EVAL_RES", "10000"))   # 10000 or 25000 (registry 79b)
D = 900_000 // RES                                   # 900 kb band: 90 bins at 10 kb, 36 at 25 kb
FULL = os.environ.get("AKITA_EVAL_FULL", "0") == "1"  # 79b: full-schedule runs, set chrhoakitaF*
OURS = ["all12refin_drop_chrho", "all12honest_drop_chrho"]
# source prefix -> (run, bin size, bins per window map)
RUNS = {"akitaA": ("akita_A_all12", 2048, 448), "akitaB": ("akita_B_all12", 2048, 448),
        "akitapre": ("akita_pretrained", 2048, 448),
        # 79b: full-schedule runs (original stopping rule) and the native 8 kb / 32 kb models
        "akitaFA": ("akita_orig_A_2kb", 2048, 448), "akitaFB": ("akita_orig_B_2kb", 2048, 448),
        "akitaN8": ("akita_orig_A_8kb", 8192, 448), "akitaN32": ("akita_orig_A_32kb", 32768, 112)}
PRE_CELLS = ["gm12878", "imr90", "hct116", "hff", "h1"]
RESULTS = Path(os.environ.get("AKITA_RETRAIN_RESULTS", str(HERE.parents[1] / "results/akita_retrain")))
EVEN_B = [c for c in C.SPLITS["B"]["test"] if c not in ("chr2", "chr14")]
A_CHROMS = C.SPLITS["A"]["test"]


def log(m):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)


# ------------------------------------------------------------------ sources
_CACHE: dict = {}


def akita_band(name: str, chrom: str, n: int, Dn: int, res: int):
    kind, cell = name.split("_", 1)
    run, abin, anb = RUNS[kind]
    f = C.W / "preds" / run / f"{chrom}.npy"
    cj = C.W / "preds" / run / "cells.json"
    if not f.exists() or not cj.exists():
        return None
    cells = json.loads(cj.read_text())
    if cell not in cells:
        return None
    k = cells.index(cell)
    R = np.asarray(np.load(f, mmap_mode="r")[:, :, k], dtype=float)
    M = hm.resample_band(R, abin, n, Dn, res)
    reach = min(Dn, (anb - 1) * abin // res)
    M[:, reach + 1:] = np.nan
    return M


_orig_band = hm.competitor_band


def competitor_band(comp, chrom, n, Dn, res):
    if comp.startswith("akita"):
        return akita_band(comp, chrom, n, Dn, res)
    return _orig_band(comp, chrom, n, Dn, res)


def source_list(b):
    src = hm.block_sources(b["set"], b["cell"], b["res"], b["comps"], models=hm.CHRHO_MODELS)
    return [s for s in src if s["name"] != hm.CASCFOLD]


def _have(kind: str, chrom: str) -> bool:
    return (C.W / "preds" / RUNS[kind][0] / f"{chrom}.npy").exists()


def blocks() -> list[dict]:
    rows = []
    for cell in C.CELLS:
        for ch in A_CHROMS:
            if FULL:                                 # all A-run sources in one block
                comps = [f"{k}_{cell}" for k in ("akitaA", "akitaFA", "akitaN8", "akitaN32")
                         if _have(k, ch)]
                if ch in ("chr2", "chr14"):
                    comps += [f"{k}_{cell}" for k in ("akitaB", "akitaFB") if _have(k, ch)]
                if cell in PRE_CELLS:
                    comps.append(f"akitapre_{cell}")
                rows.append(("chrhoakitaFA", cell, ch, comps))
                continue
            comps = [f"akitaA_{cell}"]
            if ch in ("chr2", "chr14"):
                comps.append(f"akitaB_{cell}")
            if cell in PRE_CELLS:
                comps.append(f"akitapre_{cell}")
            rows.append(("chrhoakitaA", cell, ch, comps))
        for ch in EVEN_B:
            if FULL:
                comps = [f"{k}_{cell}" for k in ("akitaB", "akitaFB") if _have(k, ch)]
                rows.append(("chrhoakitaFB", cell, ch, comps))
            else:
                rows.append(("chrhoakitaB", cell, ch, [f"akitaB_{cell}"]))
    for ch in A_CHROMS:                              # unseen cell: ours only
        rows.append(("chrhoakitaFA" if FULL else "chrhoakitaA", "k562", ch, []))
    out = []
    for set_, cell, ch, comps in rows:
        out.append(dict(set=set_, cell=cell, res=RES, chrom=ch, D=D, comps=comps,
                        id=f"{set_}_{cell}_{RES // 1000}kb_{ch}"))
    return out


def setup():
    hm.OUT = hm.CHRHO_ROOT                          # expected maps and our predictions live here
    hm.CHRHO_MODELS = list(OURS)
    hm.competitor_band = competitor_band
    hm.block_source_list = source_list
    hm.RESOLUTIONS = [RES]


# ------------------------------------------------------------------ stages
def stage_ours_predict(threads: int, jobs: int, gpu: str):
    """Our chrho models at 10 kb for the 12 training cells (existing: gm12878,
    imr90, k562), whole chromosomes chr2, 3, 11, 14, 17, TTA, as chrho_predict."""
    hm.chrho_use_root()
    hm.CHRHO_MODELS = list(OURS)
    hm.CHRHO_CELLS = C.CELLS + ["k562"]
    hm.RESOLUTIONS = [RES]
    hm.chrho_predict(threads, jobs, gpu)


def stage_score(jobs: int, only: list[str] | None):
    import re
    setup()
    for b in blocks():
        if only and not any(re.search(p, b["id"]) for p in only):
            continue
        f = hm.OUT / "blocks" / b["id"] / "measures.csv"
        if f.exists():
            continue
        srcs = source_list(b)
        miss = [s["name"] for s in srcs if s["kind"] == "model"
                and not Path(s["path"]).exists()
                and b["set"] in ("chrhoakitaA", "chrhoakitaFA")]
        if miss:
            log(f"{b['id']}: predictions missing {miss}")
            continue
        if not all(_have(c.split("_", 1)[0], b["chrom"]) for c in b["comps"]):
            log(f"{b['id']}: Akita predictions missing")
            continue
        if not hm.claim(f):
            continue
        try:
            hm.score_block(b, jobs)
        except Exception as e:                       # noqa: BLE001
            import traceback
            log(f"FAILED {b['id']}: {e!r}\n{traceback.format_exc()}")
        finally:
            hm.release(f)


COLS = ["oe_pearson_tile_median", "oe_scc", "insulation_r", "bnd_recall", "loop_recall_rank",
        "apa_ratio", "anchor_auroc", "oe_pearson", "HiCRep", "loop_recall_call"]


def label(s: str) -> str:
    return {"all12refin_drop_chrho": "ours refin_drop_chrho",
            "all12honest_drop_chrho": "ours honest_drop_chrho (ChIP only)",
            "consensus": "consensus of the other cells", "best_other_map": "best other measured map",
            "replicate": "replicate map (ceiling)"}.get(s, s)


def group(s: str) -> str:
    for k, v in (("akitaA_", "Akita retrained, split A"), ("akitaB_", "Akita retrained, split B"),
                 ("akitapre_", "Akita released"),
                 ("akitaFA_", "Akita full schedule 2 kb, split A"),
                 ("akitaFB_", "Akita full schedule 2 kb, split B"),
                 ("akitaN8_", "Akita full schedule native 8 kb, split A"),
                 ("akitaN32_", "Akita full schedule native 32 kb, split A")):
        if s.startswith(k):
            return v
    return label(s)


GROUP_ORDER = ["ours refin_drop_chrho", "ours honest_drop_chrho (ChIP only)",
               "Akita retrained, split A", "Akita retrained, split B",
               "Akita full schedule 2 kb, split A", "Akita full schedule 2 kb, split B",
               "Akita full schedule native 8 kb, split A", "Akita full schedule native 32 kb, split A",
               "Akita released",
               "consensus of the other cells", "best other measured map", "replicate map (ceiling)"]
SHOW = [("oe_pearson_tile_median", "Pearson, median per 1 Mb region"), ("oe_scc", "SCC"),
        ("insulation_r", "insulation r"), ("bnd_recall", "boundary recall"),
        ("loop_recall_rank", "loop recall"), ("apa_ratio", "APA ratio"),
        ("anchor_auroc", "anchor AUROC")]


def md_table(df: pd.DataFrame, title: str) -> str:
    cols = [c for c, _ in SHOW if c in df.columns]
    head = "| method | cells | chromosomes | " + " | ".join(l for c, l in SHOW if c in cols) + " |\n"
    head += "|---|---|---|" + "---|" * len(cols) + "\n"
    rows = ""
    for g in GROUP_ORDER:
        x = df[df.group == g]
        if x.empty:
            continue
        r = x.iloc[0]
        rows += (f"| {g} | {int(r.n_cells)} | {r.chroms} | "
                 + " | ".join(f"{r[c]:.3f}" for c in cols) + " |\n")
    return f"**{title}**\n\n{head}{rows}\n"


def tables(per_block: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """name -> table: mean over the chromosomes of a block per cell and source,
    then the median over cells."""
    cols = [c for c, _ in SHOW if c in per_block.columns] + [c for c in COLS if c in per_block.columns]
    cols = list(dict.fromkeys(cols))
    out = {}

    def agg(x: pd.DataFrame) -> pd.DataFrame:
        pc = x.groupby(["cell", "group"])[cols].mean().reset_index()
        rows = []
        for g, y in pc.groupby("group"):
            r = dict(group=g, n_cells=int(y.cell.nunique()))
            ch = sorted(x[x.group == g].chrom.unique(), key=lambda c: int(c[3:]))
            r["chroms"] = ", ".join(c[3:] for c in ch)
            for c in cols:
                r[c] = float(y[c].median())
            rows.append(r)
        return pd.DataFrame(rows)

    A = per_block[per_block.set.isin(["chrhoakitaA", "chrhoakitaFA"])]
    B = per_block[per_block.set.isin(["chrhoakitaB", "chrhoakitaFB"])]
    seen = A[A.cell != "k562"]
    out["A_12cells"] = agg(seen[~seen.group.str.contains("split B")])
    out["A_5cells_with_released_akita"] = agg(
        seen[seen.cell.isin(PRE_CELLS) & ~seen.group.str.contains("split B")])
    both = seen[seen.chrom.isin(["chr2", "chr14"])]
    out["chr2_chr14_even_held_out_by_all"] = agg(both)
    out["chr2_chr14_5cells"] = agg(both[both.cell.isin(PRE_CELLS)])
    if len(B):
        out["B_even_chromosomes"] = agg(B)
    k = A[A.cell == "k562"]
    if len(k):
        out["unseen_cell_k562"] = agg(k)
    return out


def stage_report():
    out = RESULTS / (f"full_{RES // 1000}kb" if FULL else "")
    out.mkdir(parents=True, exist_ok=True)
    pat = f"chrhoakitaF*_{RES // 1000}kb_*" if FULL else "chrhoakita[AB]_*_10kb_*"
    fs = sorted((hm.CHRHO_ROOT / "blocks").glob(pat + "/measures.csv"))
    if not fs:
        log("no blocks")
        return
    D_ = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    D_ = D_[D_.kind.isin(["model", "cool", "competitor"])].copy()
    D_["group"] = D_.source.map(group)
    D_.to_csv(out / "per_block.csv", index=False, float_format="%.4f")
    cols = [c for c in COLS if c in D_.columns]
    pc = D_.groupby(["set", "cell", "group"])[cols].mean().reset_index()
    pc["n_chromosomes"] = D_.groupby(["set", "cell", "group"]).chrom.nunique().values
    pc.to_csv(out / "per_cell.csv", index=False, float_format="%.4f")
    md = ""
    titles = {"A_12cells": "Split A chromosomes (2, 3, 11, 14, 17), the 12 training cells",
              "A_5cells_with_released_akita":
                  "Split A chromosomes, the 5 cells the released Akita has heads for",
              "chr2_chr14_even_held_out_by_all":
                  "chr2 and chr14 (even; held out by split A, split B and our models), 12 cells",
              "chr2_chr14_5cells": "chr2 and chr14, the 5 cells of the released Akita",
              "B_even_chromosomes":
                  "Split B test chromosomes 4, 6, 8, 10, 12, 16, 18, 20, 22 (our models trained on these)",
              "unseen_cell_k562": "Unseen cell K562 on chr2, 3, 11, 14, 17 (no Akita head exists)"}
    for name, t in tables(D_).items():
        t.to_csv(out / f"summary_{name}.csv", index=False, float_format="%.4f")
        md += md_table(t, titles[name])
    (out / "summary_tables.md").write_text(md)
    log(f"report: {len(fs)} blocks -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["ours_predict", "score", "report"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gpu", default="")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for st in a.stage:
        if st == "ours_predict":
            stage_ours_predict(a.threads, 1, a.gpu)
        elif st == "score":
            stage_score(a.jobs, a.only)
        else:
            setup()
            stage_report()


if __name__ == "__main__":
    main()
