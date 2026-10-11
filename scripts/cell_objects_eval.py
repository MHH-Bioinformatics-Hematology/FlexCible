"""Do the models predict the cell-specific loops and stripes of held-out cells? (registry section 66)

Cell-agnostic object scoring on the evaluation table of scripts/cell_objects_table.py --heldout
(K562, HMEC, HUVEC; objects relative to consensus12, specific objects and matched shared controls
A_ref). For every model it reads the prediction in 1.92 Mb windows around the objects (10 kb, as
scripts/aml_loop_detection.py), scores each object like that script (loops: maximum of the log2 donut
enrichment over the 3 x 3 pixels around the annotated pixel; stripes: mean log O/E of the rectangle) in
the measured map, the reference (consensus12) and the prediction, and reports

  auroc_gain      AUROC of the gain over the reference (prediction minus reference) for specific objects against
                  their matched shared controls (all, and private = carried by no training cell)
  amplitude       median predicted gain / median measured gain of the specific objects
  spearman_gain   Spearman correlation of the predicted and the measured gain over the specific objects

The reference is the map the model also reads, so a model without any grasp of the marks scores 0.5
(controls and specific objects share the reference level only through the matching).

    python scripts/cell_objects_eval.py --stage windows                               # CPU
    python scripts/cell_objects_eval.py --stage predict --models <labels> --gpu 0     # GPU, inference
    python scripts/cell_objects_eval.py --stage score                                 # CPU
Run with the hicdiff conda env. Outputs in results/cell_objects_eval/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OUTD = HM / "runs/cell_objects_eval"
RES_OUT = REPO / "results/cell_objects_eval"
RES = 10000
WIN = 192
SEED = 0
# model label -> run (reference-residual models read consensus12 for a held-out cell)
MODELS = {
    "base_refres": "all12refres_drop_10kb",
    **{f"diff_loopfocus_f{f}": f"all12diff_loopfocus_refres_drop_f{f}_10kb" for f in (10, 11, 12)},
    **{f"diff_f{f}": f"all12diff_refres_drop_f{f}_10kb" for f in (10, 11, 12)},
    **{f"allobj_f{f}": f"all12allobj_refres_drop_f{f}_10kb" for f in (10, 11, 12)},
    **{f"diffmeth_loopfocus_f{f}": f"all12diffmeth_loopfocus_refres_drop_f{f}_10kb" for f in (10, 11, 12)},
    **{f"diff_loopfocus2_f{f}": f"all12diff_loopfocus2_refres_drop_f{f}_10kb" for f in (10, 11, 12)},
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def items_path() -> Path:
    return RES_OUT / "items.csv.gz"


def build_items(tab: pd.DataFrame, sizes: dict) -> pd.DataFrame:
    """One row per object with its 192-bin window (centred on the object, clipped to the chromosome)."""
    d = tab.copy()
    d["a_lo"], d["a_hi"] = d.a_start // RES, -(-d.a_end // RES)
    d["b_lo"], d["b_hi"] = d.b_start // RES, -(-d.b_end // RES)
    lo, hi = np.minimum(d.a_lo, d.b_lo), np.maximum(d.a_hi, d.b_hi)
    nb = d.chrom.map(lambda c: int(sizes[c]) // RES)
    d = d[(hi - lo <= WIN - 20) & (nb >= WIN)].copy()
    lo, hi = np.minimum(d.a_lo, d.b_lo), np.maximum(d.a_hi, d.b_hi)
    nb = d.chrom.map(lambda c: int(sizes[c]) // RES)
    d["win_bin"] = np.clip((lo + hi) // 2 - WIN // 2, 0, nb - WIN).astype(int)
    d["win"] = d.chrom + ":" + (d.win_bin * RES).astype(str) + "-" + ((d.win_bin + WIN) * RES).astype(str)
    d["cls"] = np.where(d.role == "ctrl", "control", np.where(d.n_train_other.fillna(0) >= 1, "shared", "private"))
    d = d.reset_index(drop=True)
    d.insert(0, "item", d.index)
    return d


def stage_windows() -> None:
    import cooler
    import cell_objects_table as cot
    tab = pd.read_csv(cot.table_path(True) if cot.table_path(True).exists() else cot.OUT / "cell_objects_heldout.csv.gz")
    sizes = cooler.Cooler(str(HM / "crosscell_oe/k562_proper_10kb.cool")).chromsizes
    items = build_items(tab, sizes)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    items.to_csv(items_path(), index=False)
    (OUTD / "windows").mkdir(parents=True, exist_ok=True)
    for c, g in items.groupby("cell"):
        w = g[["chrom", "win_bin"]].drop_duplicates()
        pd.DataFrame({"c": w.chrom, "s": w.win_bin * RES, "e": (w.win_bin + WIN) * RES}).to_csv(
            OUTD / "windows" / f"{c}.tsv", sep="\t", header=False, index=False)
    print(items.groupby(["cell", "kind", "role"]).size().unstack([1, 2]))


def predict(label: str, cell: str, gpu: str, threads: int) -> int:
    import multicell_expanded as mx
    run = MODELS[label]
    out = OUTD / label / cell
    if (out / "marks.npz").exists() and (out / "none.npz").exists():
        return 0
    ck = mx.best_ckpt(mx.REF_OUT / run)
    _, bw, assay, _ = mx.CELLS[cell]
    marks = sorted(p.stem for p in Path(bw).glob("*.bigwig"))
    cmd = [mx.PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml", "--checkpoint", ck,
           "--cell", cell, "--bigwig-dir", bw, "--target", mx.target(cell, RES), "--tta", "--assay", assay,
           "--binsize", RES, "--device", "cuda", "--regions", OUTD / "windows" / f"{cell}.tsv",
           "--out-dir", out, "--save-pred", str(out / "{set}.npz"),
           "--keep-sets", f"marks={','.join(marks)}", "none=", "--reference", mx.consensus12(RES)]
    if "diff" in label:
        cmd += ["--diff-group", "cell"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES=gpu)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predict.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        rc = subprocess.call(["nice", "-n", "10"] + [str(c) for c in cmd], stdout=fh,
                             stderr=subprocess.STDOUT, cwd=REPO, env=env)
    log(f"{label} {cell} rc={rc}")
    return rc


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def object_scores(items: pd.DataFrame, maps_of) -> pd.DataFrame:
    """Score of every object in every map. maps_of(win) -> {name: window matrix or None}; the loop score
    is the 3 x 3 maximum of the log2 donut enrichment, the stripe score the mean log O/E of the rectangle."""
    from cell_objects_table import donut_log2
    rows = []
    for win, gw in items.groupby("win"):
        maps = maps_of(win)
        if any(maps.get(k) is None or maps[k].shape != (WIN, WIN) for k in ("measured", "reference")):
            continue
        enr = {nm: donut_log2(M) for nm, M in maps.items() if M is not None and M.shape == (WIN, WIN)}
        for r in gw.itertuples():
            ws = r.win_bin
            rec = {"item": r.item}
            if r.kind == "loop":
                i, j = r.a_lo - ws, r.b_lo - ws
                if not (1 <= i < WIN - 1 and 1 <= j < WIN - 1):
                    continue
                for nm, E in enr.items():
                    blk = E[i - 1:i + 2, j - 1:j + 2]
                    rec[nm] = float(np.nanmax(blk)) if np.isfinite(blk).any() else np.nan
            else:
                a0, a1, b0, b1 = r.a_lo - ws, r.a_hi - ws, r.b_lo - ws, r.b_hi - ws
                if min(a0, b0) < 0 or max(a1, b1) > WIN:
                    continue
                for nm, M in maps.items():
                    if M is None or M.shape != (WIN, WIN):
                        continue
                    band = M[a0:a1, b0:b1]
                    rec[nm] = float(np.nanmean(band)) if np.isfinite(band).any() else np.nan
            rows.append(rec)
    return items.merge(pd.DataFrame(rows), on="item", how="inner")


def summarise(D: pd.DataFrame, models: list[str], nboot: int = 500) -> pd.DataFrame:
    """Per model, cell ('all' pooled), kind and class: AUROC of the gain over the reference of specific
    objects against their matched controls, amplitude and Spearman of the gain."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(SEED)
    out = []
    for m in models:
        if m not in D:
            continue
        D = D.assign(**{"gain|" + m: D[m] - D["reference"]})
    D = D.assign(**{"gain|measured": D["measured"] - D["reference"]})
    for m in [x for x in models if x in D] + ["measured"]:
        col = "gain|" + m
        for cell in ["all"] + sorted(D.cell.unique()):
            sub = D if cell == "all" else D[D.cell == cell]
            for kind in ("loop", "stripe"):
                for cls in ("all", "private"):
                    pos = sub[(sub.kind == kind) & (sub.role == "pos")]
                    if cls == "private":
                        pos = pos[pos.cls == "private"]
                    neg = sub[(sub.kind == kind) & (sub.role == "ctrl")]
                    if len(pos) < 10 or len(neg) < 10:
                        continue
                    a = auroc(pos[col].to_numpy(), neg[col].to_numpy())
                    boots = [auroc(pos[col].to_numpy()[rng.integers(len(pos), size=len(pos))],
                                   neg[col].to_numpy()[rng.integers(len(neg), size=len(neg))])
                             for _ in range(nboot)]
                    ok = np.isfinite(pos[col]) & np.isfinite(pos["gain|measured"])
                    out.append({"model": m, "cell": cell, "kind": kind, "class": cls, "n_pos": len(pos),
                                "n_ctl": len(neg), "auroc_gain": a,
                                "auroc_lo": float(np.nanpercentile(boots, 2.5)),
                                "auroc_hi": float(np.nanpercentile(boots, 97.5)),
                                "amplitude": float(pos[col].median() / pos["gain|measured"].median()),
                                "spearman_gain": float(spearmanr(pos.loc[ok, col], pos.loc[ok, "gain|measured"])[0])
                                if ok.sum() >= 10 else np.nan})
    return pd.DataFrame(out)


def stage_score(models: list[str]) -> None:
    import cooler
    import multicell_expanded as mx
    from aml_eval import fetch_region, pred_matrix
    items = pd.read_csv(items_path())
    parts = []
    for cell, g in items.groupby("cell"):
        T = cooler.Cooler(str(mx.target(cell, RES)))
        R = cooler.Cooler(str(mx.consensus12(RES)))
        preds = {}
        for lab in models:
            for k in ("marks", "none"):
                f = OUTD / lab / cell / f"{k}.npz"
                if f.exists():
                    preds[lab if k == "marks" else f"{lab}|none"] = np.load(f)
        log(f"score {cell}: {len(g)} objects, models {list(preds)}")
        cache: dict = {}

        def maps_of(win, T=T, R=R, preds=preds, cache=cache):
            m = {"measured": fetch_region(T, win, cache), "reference": fetch_region(R, win, cache)}
            for nm, z in preds.items():
                if win in z.files:
                    m[nm] = pred_matrix(z[win])
            return m
        parts.append(object_scores(g, maps_of))
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(RES_OUT / "object_scores.csv.gz", index=False, float_format="%.5f")
    S = summarise(D, [c for c in D.columns if c in models or c.split("|")[0] in models])
    S.to_csv(RES_OUT / "auroc_by_model.csv", index=False, float_format="%.4f")
    print(S[(S.cell == "all")].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["windows", "predict", "score"])
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--cells", nargs="*", default=["k562", "hmec_deep", "huvec_deep"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if "windows" in a.stage:
        stage_windows()
    if "predict" in a.stage:
        for lab in a.models:
            for c in a.cells:
                predict(lab, c, a.gpu, a.threads)
    if "score" in a.stage:
        stage_score(a.models)


if __name__ == "__main__":
    main()
