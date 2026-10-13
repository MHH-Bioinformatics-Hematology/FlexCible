"""Do the difference input tracks help the prediction of cell-specific loops and stripes of held-out cells? (registry section 71)

Re-uses scripts/cell_objects_eval.py (windows, prediction, object scoring) with
  * controls A_refrand instead of the shared loops A_ref: random objects of the same kind matched on the
    reference (consensus12) enrichment, distance and orientation (scripts/refrand_controls.py, as the training
    cells in scripts/cell_objects_refrand.py), built here for the held-out evaluation table
    (K562, HMEC deep, HUVEC deep; the reference of a held-out cell is consensus12, no validity mask);
  * four models: all12refres_drop_10kb with its two seed copies and all12diffcells_refres_drop_10kb;
  * the object classes of the rule-transfer features (loops promoter-enhancer PE / promoter-silencer PS /
    other, stripes enh / rep / other; $HM/crosscell_rules/cell_features.csv.gz, joined on pair_id);
  * the marks effect (AUROC with the cell's own marks minus the track-dropout condition 'none'), seed spread
    and a paired bootstrap of the difference between the difference-channel model and each seed.

    python scripts/cell_objects_eval_diff.py --stage table windows          # CPU
    python scripts/cell_objects_eval_diff.py --stage predict --gpu 1        # GPU, inference only
    python scripts/cell_objects_eval_diff.py --stage score summarise        # CPU
Run with the hicdiff conda env. Outputs in results/cell_objects_eval_diff/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import cell_objects_eval as ce  # noqa: E402

HM = ce.HM
RES_OUT = REPO / "results/cell_objects_eval_diff"
OUTD = HM / "runs/cell_objects_eval_diff"
ce.RES_OUT, ce.OUTD = RES_OUT, OUTD
TABLE = RES_OUT / "cell_objects_heldout_refrand.csv.gz"
FEATURES = HM / "crosscell_rules/cell_features.csv.gz"
BASE = ["base_refres", "base_refres_s1", "base_refres_s2"]
DIFF = "diffcells"
ce.MODELS.update({"base_refres": "all12refres_drop_10kb", "base_refres_s1": "all12refres_drop_s1_10kb",
                  "base_refres_s2": "all12refres_drop_s2_10kb", DIFF: "all12diffcells_refres_drop_10kb"})
MODELS = BASE + [DIFF]
NBOOT = 500


def stage_table() -> None:
    import cell_objects_refrand as cr
    import cell_objects_table as cot
    import multicell_expanded as mx
    d = pd.read_csv(cot.table_path(True))
    cot.assert_no_leak(d, True)
    cells = sorted(d.cell.unique())
    out = cr.build(d, cells, lambda c: mx.consensus12(ce.RES), workers=3)
    out = out[out["set"].isin(["pos", "A_refrand"])]
    RES_OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE, index=False)
    rows = []
    for (c, kind), g in out.groupby(["cell", "kind"]):
        P, C = g[g["set"] == "pos"], g[g["set"] == "A_refrand"]
        m = P.merge(C, on="pair_id", suffixes=("_p", "_c"))
        rows.append({"cell": c, "kind": kind, "n_pos": len(P), "n_matched": len(m),
                     "mean_pos_ref": m.ref_val_p.mean(), "mean_ctl_ref": m.ref_val_c.mean(),
                     "max_abs_diff_ref": (m.ref_val_p - m.ref_val_c).abs().max()})
    Q = pd.DataFrame(rows)
    Q.to_csv(RES_OUT / "refrand_match_quality_heldout.csv", index=False, float_format="%.4f")
    print(Q.round(3).to_string(index=False))


def stage_windows() -> None:
    import cooler
    tab = pd.read_csv(TABLE)
    sizes = cooler.Cooler(str(HM / "crosscell_oe/k562_proper_10kb.cool")).chromsizes
    items = ce.build_items(tab, sizes)
    feat = pd.read_csv(FEATURES, usecols=["kind", "pair_id", "set", "pair_type"])
    feat = feat[feat["set"] == "pos"].drop_duplicates(["kind", "pair_id"])[["kind", "pair_id", "pair_type"]]
    items = items.drop(columns="pair_type").merge(feat, on=["kind", "pair_id"], how="left")
    items["pair_type"] = items.pair_type.fillna("unknown")
    items = items.sort_values("item").reset_index(drop=True)
    # controls inherit the class of the positive they were matched to
    cls = items[items.role == "pos"].set_index(["kind", "pair_id"]).pair_type
    ctl = items.role == "ctrl"
    items.loc[ctl, "pair_type"] = [cls.get(k, "unknown") for k in zip(items.kind[ctl], items.pair_id[ctl])]
    items.to_csv(ce.items_path(), index=False)
    (OUTD / "windows").mkdir(parents=True, exist_ok=True)
    for c, g in items.groupby("cell"):
        w = g[["chrom", "win_bin"]].drop_duplicates()
        pd.DataFrame({"c": w.chrom, "s": w.win_bin * ce.RES, "e": (w.win_bin + ce.WIN) * ce.RES}).to_csv(
            OUTD / "windows" / f"{c}.tsv", sep="\t", header=False, index=False)
    print(items.groupby(["cell", "kind", "role", "pair_type"]).size().unstack([1, 2, 3]).T.to_string())


def stage_score() -> None:
    import cooler
    import multicell_expanded as mx
    from aml_eval import fetch_region, pred_matrix
    items = pd.read_csv(ce.items_path())
    parts = []
    for cell, g in items.groupby("cell"):
        T = cooler.Cooler(str(mx.target(cell, ce.RES)))
        R = cooler.Cooler(str(mx.consensus12(ce.RES)))
        preds = {}
        for lab in MODELS:
            for k in ("marks", "none"):
                f = OUTD / lab / cell / f"{k}.npz"
                if f.exists():
                    preds[lab if k == "marks" else f"{lab}|none"] = np.load(f)
        ce.log(f"score {cell}: {len(g)} objects, maps {list(preds)}")
        cache: dict = {}

        def maps_of(win, T=T, R=R, preds=preds, cache=cache):
            m = {"measured": fetch_region(T, win, cache), "reference": fetch_region(R, win, cache)}
            for nm, z in preds.items():
                if win in z.files:
                    m[nm] = pred_matrix(z[win])
            return m
        parts.append(ce.object_scores(g, maps_of))
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(RES_OUT / "object_scores.csv.gz", index=False, float_format="%.5f")
    print(D.groupby(["cell", "kind", "role"]).size().unstack([1, 2]))


def groups(D: pd.DataFrame, cell: str, kind: str):
    """Yield (class name, positives, matched controls) of one cell ('all' pooled) and kind."""
    sub = D[D.kind == kind]
    if cell != "all":
        sub = sub[sub.cell == cell]
    pos, ctl = sub[sub.role == "pos"], sub[sub.role == "ctrl"]
    classes = ["all", "private"] + (["PE", "PS", "other"] if kind == "loop" else ["enh", "rep", "other"])
    for c in classes:
        if c == "all":
            p = pos
        elif c == "private":
            p = pos[pos.cls == "private"]
        else:
            p = pos[pos.pair_type == c]
        yield c, p, ctl[ctl.pair_id.isin(set(p.pair_id))] if c not in ("all",) else ctl


def metrics(p: pd.DataFrame, n: pd.DataFrame, m: str, cond: str) -> dict:
    from scipy.stats import spearmanr
    col = m if cond == "marks" else f"{m}|none"
    if col not in p:
        return {}
    gp, gn = (p[col] - p.reference).to_numpy(), (n[col] - n.reference).to_numpy()
    gm = (p.measured - p.reference).to_numpy()
    ok = np.isfinite(gp) & np.isfinite(gm)
    return {"auroc": ce.auroc(gp, gn),
            "amplitude": float(np.nanmedian(gp) / np.nanmedian(gm)),
            "spearman": float(spearmanr(gp[ok], gm[ok])[0]) if ok.sum() >= 10 else np.nan}


def stage_summarise() -> None:
    D = pd.read_csv(RES_OUT / "object_scores.csv.gz")
    rng = np.random.default_rng(ce.SEED)
    rows, diffrows = [], []
    for cell in ["all"] + sorted(D.cell.unique()):
        for kind in ("loop", "stripe"):
            for cls, p, n in groups(D, cell, kind):
                if len(p) < 10 or len(n) < 10:
                    continue
                base = {"cell": cell, "kind": kind, "class": cls, "n_pos": len(p), "n_ctl": len(n)}
                pt = {(m, c): metrics(p, n, m, c) for m in MODELS for c in ("marks", "none")}
                # paired bootstrap: the same positives / controls for every model
                bs = {k: [] for k in pt}
                for _ in range(NBOOT):
                    pb, nb = p.iloc[rng.integers(len(p), size=len(p))], n.iloc[rng.integers(len(n), size=len(n))]
                    for k in pt:
                        bs[k].append(metrics(pb, nb, *k))
                for (m, c), v in pt.items():
                    if not v:
                        continue
                    r = dict(base, model=m, cond=c, **v)
                    for q in ("auroc", "spearman"):
                        a = np.array([x.get(q, np.nan) for x in bs[(m, c)]])
                        r[q + "_lo"], r[q + "_hi"] = np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)
                    rows.append(r)
                # derived: marks effect and difference of the diff model to each seed and to the seed mean
                def stat(k, q):
                    return pt[k].get(q, np.nan)

                def boot(f):
                    a = np.array([f(i) for i in range(NBOOT)])
                    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
                for q in ("auroc", "spearman"):
                    for tag, fn_pt, fn_bs in (
                        ("marks_effect", lambda m: stat((m, "marks"), q) - stat((m, "none"), q),
                         lambda m, i: bs[(m, "marks")][i][q] - bs[(m, "none")][i][q]),
                        ("value", lambda m: stat((m, "marks"), q), lambda m, i: bs[(m, "marks")][i][q])):
                        if tag == "marks_effect" and q != "auroc":
                            continue
                        sd = [fn_pt(b) for b in BASE]
                        d_ = fn_pt(DIFF) - np.nanmean(sd)
                        lo, hi = boot(lambda i: fn_bs(DIFF, i) - np.nanmean([fn_bs(b, i) for b in BASE]))
                        rec = dict(base, metric=f"{q}_{tag}" if tag != "value" else q,
                                   diff_model=fn_pt(DIFF), seed_mean=float(np.nanmean(sd)),
                                   seed_min=float(np.nanmin(sd)), seed_max=float(np.nanmax(sd)),
                                   seed_range=float(np.nanmax(sd) - np.nanmin(sd)),
                                   diff_minus_seedmean=float(d_), ci_lo=lo, ci_hi=hi,
                                   diff_above_all_seeds=bool(fn_pt(DIFF) > np.nanmax(sd)),
                                   diff_below_all_seeds=bool(fn_pt(DIFF) < np.nanmin(sd)))
                        for b in BASE:
                            rec["diff_minus_" + b] = float(fn_pt(DIFF) - fn_pt(b))
                        diffrows.append(rec)
    S = pd.DataFrame(rows)
    S.to_csv(RES_OUT / "metrics_by_model.csv", index=False, float_format="%.4f")
    X = pd.DataFrame(diffrows)
    X.to_csv(RES_OUT / "diff_vs_seeds.csv", index=False, float_format="%.4f")
    A = S[S.cond == "marks"].groupby(["cell", "kind", "class", "model"])["amplitude"].first().unstack("model")
    A.to_csv(RES_OUT / "amplitude.csv", float_format="%.4f")
    print(X[X.metric.isin(["auroc", "auroc_marks_effect", "spearman"]) & (X["class"] == "all")]
          .round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["table", "windows", "predict", "score", "summarise"])
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cells", nargs="*", default=["k562", "hmec_deep", "huvec_deep"])
    a = ap.parse_args()
    if "table" in a.stage:
        stage_table()
    if "windows" in a.stage:
        stage_windows()
    if "predict" in a.stage:
        for lab in MODELS:
            for c in a.cells:
                ce.predict(lab, c, a.gpu, a.threads)
    if "score" in a.stage:
        stage_score()
    if "summarise" in a.stage:
        stage_summarise()


if __name__ == "__main__":
    main()
