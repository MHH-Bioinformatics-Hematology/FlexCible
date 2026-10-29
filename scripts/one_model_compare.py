"""One model against the dedicated models (registry section 88).

Reads the evaluation of the one-model views all12one_<k>kb (run with
`scripts/multicell_expanded.py --runs all12one --res <bp> --stage eval`, exactly as the
dedicated runs) and of the dedicated track-dropout reference-input runs
all12refin_drop[_<k>kb], and writes results/one_model/compare_dedicated.csv and RESULTS.md.

Targets: the 12 training cells on their test windows (median window Pearson r of log O/E,
eval_val_windows.py; the consensus alone is the leave-one-out map read as the prediction)
and the held-out cells K562, HMEC and HUVEC on the 413 Akita test regions
(crosscell_predict.py; consensus alone = the 12-cell consensus). The 413-region scores of the
training cells are added as a second target set (their leave-one-out consensus alone).

    python scripts/one_model_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REF = Path("/mnt/processed/wolffjoa/hic-multiome/runs/multicell_reference")
OUT = REPO / "results/one_model"
RES = [2, 5, 10, 25]
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248", "mcf7",
           "cd14mono", "pc9"]
HELD = ["k562", "hmec", "huvec"]


def run_of(base, k):
    return REF / (base if k == 2 else f"{base}_{k}kb")


def test_windows(run, cell):
    f = run / f"test_eval_{cell}.json"
    g = run / f"refalone_test_eval_{cell}.json"
    if not f.exists():
        return None
    a = json.loads(f.read_text())
    b = json.loads(g.read_text()) if g.exists() else {}
    return a["n"], a["oe_median_r"], b.get("oe_median_r", np.nan)


def regions(run, cell):
    f = run / "crosscell" / f"predict_{cell}.csv"
    return pd.read_csv(f) if f.exists() else None


def boot_ci(d, n=2000, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.median(d[rng.integers(0, len(d), (n, len(d)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def build():
    rows = []
    for k in RES:
        one, ded = run_of("all12one", k) if k != 2 else REF / "all12one_2kb", run_of("all12refin_drop", k)
        for cell in TRAIN12:
            a, b = test_windows(one, cell), test_windows(ded, cell)
            if a and b:
                rows.append(dict(res_kb=k, target_set="test_windows", group="training", cell=cell, n=a[0],
                                 one_model=a[1], dedicated=b[1], consensus=a[2],
                                 dedicated_consensus=b[2]))
        for cell in TRAIN12 + HELD:
            a, b = regions(one, cell), regions(ded, cell)
            if a is None or b is None:
                continue
            cons = "base_consensus_loo" if cell in TRAIN12 else "base_consensus12"
            m = a.merge(b[["region", "model"]], on="region", suffixes=("", "_ded"))
            d = (m["model"] - m["model_ded"]).values
            lo, hi = boot_ci(d)
            rows.append(dict(res_kb=k, target_set="akita_413_regions",
                             group="training" if cell in TRAIN12 else "held-out", cell=cell,
                             n=int(m["model"].notna().sum()), one_model=a["model"].median(),
                             dedicated=b["model"].median(), consensus=a[cons].median() if cons in a else np.nan,
                             dedicated_consensus=b[cons].median() if cons in b else np.nan,
                             paired_diff_median=float(np.nanmedian(d)), paired_diff_lo=lo, paired_diff_hi=hi,
                             one_wins_frac=float(np.nanmean(d > 0))))
    df = pd.DataFrame(rows)
    df["one_minus_dedicated"] = df["one_model"] - df["dedicated"]
    df["one_minus_consensus"] = df["one_model"] - df["consensus"]
    df["dedicated_minus_consensus"] = df["dedicated"] - df["consensus"]
    df["dedicated_run"] = df.res_kb.map(lambda k: run_of("all12refin_drop", k).name)
    df["one_model_run"] = df.res_kb.map(lambda k: f"all12one_{k}kb")
    return df.round(4)


def summary(df):
    s = []
    for (k, ts, g), x in df.groupby(["res_kb", "target_set", "group"], sort=False):
        s.append(dict(res_kb=k, target_set=ts, group=g, n_cells=len(x), one_model=x.one_model.median(),
                      dedicated=x.dedicated.median(), consensus=x.consensus.median(),
                      one_minus_dedicated=(x.one_model - x.dedicated).median(),
                      one_beats_dedicated=f"{int((x.one_model > x.dedicated).sum())}/{len(x)}",
                      one_beats_consensus=f"{int((x.one_model > x.consensus).sum())}/{len(x)}",
                      dedicated_beats_consensus=f"{int((x.dedicated > x.consensus).sum())}/{len(x)}"))
    return pd.DataFrame(s).round(4)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(OUT / "compare_dedicated.csv", index=False)
    sm = summary(df)
    sm.to_csv(OUT / "compare_dedicated_summary.csv", index=False)
    held = df[(df.target_set == "akita_413_regions") & (df.group == "held-out")]
    txt = ["# One model against the dedicated models", "",
           "Generated by `scripts/one_model_compare.py` from the evaluation stage of "
           "`scripts/multicell_expanded.py` (registry section 88). `compare_dedicated.csv` holds one row per "
           "bin size, target set and cell; `compare_dedicated_summary.csv` the medians over cells.", "",
           "Models: one model `all12one` (2, 5, 10 and 25 kb in one network, continuous log2(bin size) "
           "embedding, reference input, track dropout), evaluated on its per-resolution views "
           "`all12one_<k>kb`; dedicated: `all12refin_drop` (2 kb), `all12refin_drop_5kb`, `_10kb`, `_25kb` "
           "(reference input, track dropout, one network per bin size). Consensus alone: leave-one-out "
           "consensus for the training cells, the 12-cell consensus for the held-out cells. Pearson r of log "
           "O/E; test windows: median over the windows of the chromosome-held-out test split; Akita 413: "
           "median over the 413 Akita test regions.", "",
           "## Medians over cells", "", md_table(sm), "",
           "## Held-out cells on the 413 Akita regions, with the paired difference over regions", "",
           md_table(held[["res_kb", "cell", "n", "one_model", "dedicated", "consensus",
                          "paired_diff_median", "paired_diff_lo", "paired_diff_hi", "one_wins_frac"]]), ""]
    (OUT / "RESULTS.md").write_text("\n".join(txt))
    print(md_table(sm))


if __name__ == "__main__":
    main()
