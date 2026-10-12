"""Seed spread of the AML folds 10 to 12 (registry section 72): tidy table (seed, family, fold, metric, value)
and a mean / min / max summary over the seeds 0, 1, 2, from the evaluation tables on the share.

Families: refin = all12aml_refin_drop (label aml_refin), diff = all12diff_refres_drop (aml_diff, channels only),
honest = all12aml_honest_drop (aml_honest). Seed 0 is the original run, seeds 1 and 2 the replicates.
Sources (HM = /mnt/processed/wolffjoa/hic-multiome):
  HM/runs/aml_train_remote/fold<N>/eval_f<N>_patients.csv         aml_refin, aml_honest (seed 0)
  HM/runs/aml_train_remote/fold<N>/eval_f<N>_extra_patients.csv   aml_diff (seed 0)
  HM/runs/aml_train_remote/fold<N>/eval_f<N>_seeds_patients.csv   *_s1, *_s2
  HM/runs/diff_marks/aml_loop_detection_seeds/auroc.csv           loop / stripe AUROC and share above the p90 of the controls
  HM/runs/diff_marks/seeds_criteria2.csv                          C1, C2, C3 per model (pooled over the ten donors)
Metrics per fold: median test-patient Pearson at 10 kb (marks input) and its median gain over the AML training
mean; pooled (fold "all"): detection share above the p90 of random pairs (B_rand) and AUROC of the residual / gain
for private loops and stripes, C1 (marks effect), C2 (residual AUROC against A_refrand), C3 (residual Spearman).

    python scripts/aml_seeds_table.py            # writes results/aml_seeds/seeds_tidy.csv and seeds_summary.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OUT = Path(__file__).resolve().parents[1] / "results/aml_seeds"
FAM = {"aml_refin": "refin", "aml_diff": "diff", "aml_honest": "honest"}


def split_label(lab: str):
    """('aml_refin_s2' -> ('refin', 2)); a label without seed suffix is seed 0; None when not a family model."""
    m = re.fullmatch(r"(aml_refin|aml_diff|aml_honest)(?:_s(\d+))?(\|none)?", lab)
    if not m or m.group(3):
        return None
    return FAM[m.group(1)], int(m.group(2) or 0)


def main():
    rows = []
    for fold in (10, 11, 12):
        d = HM / f"runs/aml_train_remote/fold{fold}"
        for suffix in ("", "_extra", "_seeds"):
            f = d / f"eval_f{fold}{suffix}_patients.csv"
            if not f.exists():
                continue
            P = pd.read_csv(f, dtype={"cell": str})
            P = P[(P.res_kb == 10) & (P.group == "patient") & (P.condition == "marks")]
            for model, g in P.groupby("model"):
                sl = split_label(model)
                if sl is None:
                    continue
                fam, seed = sl
                rows.append((seed, fam, fold, "median_pearson", float(g.pearson.median())))
                if "resid_vs_aml_train_mean" in g:
                    rows.append((seed, fam, fold, "median_resid_corr_vs_aml_mean", float(g.resid_vs_aml_train_mean.median())))
    a = HM / "runs/diff_marks/aml_loop_detection_seeds/auroc.csv"
    if a.exists():
        A = pd.read_csv(a)
        A = A[(A["class"] == "private") & (A.control == "B_rand")]
        for r in A.itertuples():
            sl = split_label(r.predictor)
            if sl is None:
                continue
            fam, seed = sl
            rows.append((seed, fam, "all", f"{r.kind}_{r.score}_frac_above_p90_B_rand", float(r.frac_above_ctl_p90)))
            rows.append((seed, fam, "all", f"{r.kind}_{r.score}_auroc_B_rand", float(r.auroc)))
    c = HM / "runs/diff_marks/seeds_criteria2.csv"
    if c.exists():
        for r in pd.read_csv(c).itertuples():
            sl = split_label(str(r.model))
            if sl is None:
                continue
            fam, seed = sl
            for k, col in (("C1_marks_effect", "C1_marks_delta_resid_A_refrand"), ("C2_resid_auroc", "C2_resid_auroc_A_refrand"),
                           ("C3_resid_spearman", "C3_resid_spearman")):
                if col in c_cols(c) and pd.notna(getattr(r, col)):
                    rows.append((seed, fam, "all", k, float(getattr(r, col))))
    T = pd.DataFrame(rows, columns=["seed", "family", "fold", "metric", "value"]).drop_duplicates()
    OUT.mkdir(parents=True, exist_ok=True)
    T.sort_values(["family", "metric", "fold", "seed"]).to_csv(OUT / "seeds_tidy.csv", index=False, float_format="%.4f")
    S = (T.groupby(["family", "fold", "metric"]).value.agg(n_seeds="count", mean="mean", min="min", max="max")
         .assign(range=lambda d: d["max"] - d["min"]).reset_index())
    S.to_csv(OUT / "seeds_summary.csv", index=False, float_format="%.4f")
    print(S.to_string(index=False))


def c_cols(path):
    return list(pd.read_csv(path, nrows=0).columns)


if __name__ == "__main__":
    main()
