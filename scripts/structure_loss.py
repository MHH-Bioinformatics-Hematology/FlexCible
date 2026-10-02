"""Structure-aware learning (registry section 30).

Models (scripts/multicell_expanded.py, all variants of all12refin_drop, the
best model on new cells at 10 kb): one structure loss term added to L1+MSE,
insul (insulation profile, TADs), donut (local enrichment, loops) or compart
(row-correlation matrix inside the window, a compartment proxy), each at the
weights STRUCT_GRID at 25 kb; per term the weight is chosen on the honest
validation windows; the chosen weight per term at 10 kb; all three together
(struct3) at 25 and 10 kb; checkpoint selection on the combined structure
score (selcomb) and, once scripts/hic_measures.py exists, on its combined
score (selhm).

Stages:
  choose    per term, the grid weight whose selected checkpoint has the
            highest combined validation score (mean over Pearson, stratified
            correlation, insulation correlation and APA agreement of the rank
            among the term's grid runs, scaled to [0, 1]) -> chosen_weights.json
  evaluate  crosscell_predict --structure of every trained model and the
            base all12refin_drop on held-out and training cells (CPU)
  summary   results/structure_loss/*.csv

    python scripts/structure_loss.py --stage choose
    python scripts/structure_loss.py --stage evaluate summary [--res 25000]
"""
from __future__ import annotations

import argparse
import json
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
import multicell_expanded as mx                                     # noqa: E402
from hicmultiome.structure import SELECT_KEYS, rank_combined        # noqa: E402

OUT = mx.HM / "runs/structure_eval"
RES_OUT = REPO / "results/structure_loss"
PY = sys.executable
CELLS = ["k562", "hmec_deep", "huvec_deep", "gm12878", "imr90", "hepg2"]
MEASURES = ["pearson", "scc", "insulation_corr", "boundary_f1", "loop_enrich_corr",
            "loop_apa_ratio", "compartment_corr", "eigenvector_abs_corr"]


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "structure_loss.log", "a") as fh:
        fh.write(line + "\n")


def grid_runs(res: int = 25000) -> dict[str, list[tuple[float, str]]]:
    return {t: [(w, mx.run_name(f"all12refin_drop_{t}_{mx.wtag(w)}", res))
                for w in mx.STRUCT_GRID] for t in mx.STRUCT_TERMS}


def models(res: int) -> list[str]:
    """Every model of the section at res (trained or not)."""
    names = [mx.run_name("all12refin_drop", res)]
    if res == 25000:
        names += [n for v in grid_runs(res).values() for _, n in v]
    else:
        names += [mx.run_name(f"all12refin_drop_{t}", res) for t in mx.STRUCT_TERMS]
    names += [mx.run_name(b, res) for b in ("all12refin_drop_struct3",
                                            "all12refin_drop_selcomb",
                                            "all12refin_drop_selhm")]
    return names


def trained(name: str) -> bool:
    si = mx.stop_info(name, 10)
    return bool(si and si[2] == "patience")


def selected_row(name: str) -> dict | None:
    """Validation structure measures at the selected checkpoint (the epoch of
    the highest validation Pearson, the dist_pearson selection)."""
    f = mx.out_of(name) / name / "val_structure.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if mx.split_name(name)[0] == "all12refin_drop_selcomb":
        # replay the combined selection: the last evaluation that ranked
        # first among all evaluations up to it
        hist, pick = [], 0
        for i, r in d.iterrows():
            hist.append({k: r[f"sel_{k}"] for k in SELECT_KEYS})
            sc = rank_combined(hist)
            if len(sc) == 1 or sc[-1] > max(sc[:-1]):
                pick = i
        return d.loc[pick].to_dict()
    if mx.split_name(name)[0] == "all12refin_drop_selhm":
        return d.loc[d["hic_measures"].idxmax()].to_dict()
    return d.loc[d["pearson"].idxmax()].to_dict()


def stage_choose():
    rows, chosen = [], {}
    for t, runs in grid_runs().items():
        sel = []
        for w, n in runs:
            r = selected_row(n) if trained(n) else None
            if r is None:
                raise SystemExit(f"{n}: not trained or no val_structure.csv")
            sel.append((w, n, r))
        hist = [{k: r[f"sel_{k}"] for k in SELECT_KEYS} for _, _, r in sel]
        sc = rank_combined(hist)
        for (w, n, r), c in zip(sel, sc):
            rows.append({"term": t, "weight": w, "run": n, "combined_rank_score": c,
                         **{k: r.get(k) for k in MEASURES}})
        chosen[t] = float(sel[int(np.argmax(sc))][0])
    RES_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES_OUT / "weight_grid_val.csv", index=False,
                              float_format="%.4f")
    mx.STRUCT_CHOSEN.write_text(json.dumps(chosen, indent=1) + "\n")
    log(f"chosen weights {chosen}")


def predict(name: str, cell: str) -> int:
    run = mx.out_of(name) / name
    out = OUT / name
    if (out / f"predict_{cell}.csv").exists():
        return 0
    res = mx.split_name(name)[1]
    ck = mx.best_ckpt(run)
    _, bw, assay, _ = mx.CELLS[cell]
    ref = mx.reference_of(name, cell)
    tag = "consensus_loo" if cell in mx.TRAIN12 else "consensus12"
    cmd = [PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
           "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw,
           "--target", mx.target(cell, res), "--tta", "--assay", assay, "--binsize", res,
           "--device", "cpu", "--out-dir", out, "--reference", ref, "--structure",
           "--others", f"{tag}={ref}"]
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", CUDA_VISIBLE_DEVICES="")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"predict_{cell}.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def stage_evaluate(res_list, jobs: int):
    todo = [(n, c) for r in res_list for n in models(r) if trained(n) for c in CELLS]
    with ThreadPoolExecutor(jobs) as ex:
        for (n, c), rc in zip(todo, ex.map(lambda a: predict(*a), todo)):
            if rc:
                log(f"[FAILED] {n} {c} rc={rc}")


def stage_summary():
    rows = []
    for r in (25000, 10000):
        for n in models(r):
            for c in CELLS:
                f = OUT / n / f"predict_{c}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f)
                cons = "consensus_loo" if c in mx.TRAIN12 else "consensus12"
                row = {"res": r, "model": n, "cell": c, "held_out": c not in mx.TRAIN12,
                       "n_regions": int(d["struct_pearson"].notna().sum())}
                for k in MEASURES:
                    row[k] = d[f"struct_{k}"].median()
                    ck = f"struct_{cons}_{k}"
                    row[f"consensus_{k}"] = d[ck].median() if ck in d else np.nan
                rows.append(row)
    T = pd.DataFrame(rows)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(RES_OUT / "structure_eval.csv", index=False, float_format="%.4f")
    if len(T):
        H = T[T.held_out].groupby(["res", "model"])[MEASURES].median().reset_index()
        H.to_csv(RES_OUT / "structure_eval_heldout_median.csv", index=False,
                 float_format="%.4f")
        print(H.to_string(index=False))
    # validation curves of every trained structure run
    vrows = []
    for r in (25000, 10000):
        for n in models(r)[1:]:
            s = selected_row(n)
            if s is not None:
                vrows.append({"res": r, "model": n, **{k: s.get(k) for k in MEASURES}})
    if vrows:
        pd.DataFrame(vrows).to_csv(RES_OUT / "val_selected.csv", index=False,
                                   float_format="%.4f")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["choose", "evaluate", "summary"])
    ap.add_argument("--res", type=int, nargs="*", default=[25000, 10000])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    for s in a.stage:
        if s == "choose":
            stage_choose()
        elif s == "evaluate":
            stage_evaluate(a.res, a.jobs)
        elif s == "summary":
            stage_summary()


if __name__ == "__main__":
    main()
