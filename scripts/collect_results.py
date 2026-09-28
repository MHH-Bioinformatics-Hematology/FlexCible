"""Copy every experiment's result tables into results/ (versioned with the code).

Run outputs live on the processing share; this gathers the small tables the
paper cites into results/<experiment>/ and derives the ones that only existed
as run logs. Re-run after any experiment finishes; missing sources are listed,
not fatal. See results/REGISTRY.md for what each table is and how it was made.

    python scripts/collect_results.py
"""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
R = Path("/mnt/processed/wolffjoa/hic-multiome/runs")
X = Path("/mnt/processed/wolffjoa/hic-multiome/crosscell_raw")

COPY = {
    "akita_comparison": [REPO / "runs/akita_native/score_like_hicgan.csv",
                         R / "akita_ref/section1_overlapval.score.csv",
                         R / "akita_ref/akita_native_honest.score.csv",
                         R / "akita_ref/akita_native_refres.score.csv",
                         Path("/mnt/processed/wolffjoa/hic-multiome/akita2kb/akita_targets_check.json"),
                         Path("/mnt/processed/wolffjoa/hic-multiome/akita2kb/akita_cache_check.json")],
    "backward_elimination": [R / "backward_tracks/backward.csv",
                             R / "backward_tracks/backward_rankings.csv",
                             R / "backward_tracks/backward_FULLDATA_ref.csv",
                             R / "backward_tracks/backward_rankings_FULLDATA_ref.csv"],
    "track_redundancy": [R / "track_correlation/track_correlation.csv",
                         R / "track_correlation/combos.csv",
                         R / "track_substitution/summary.csv",
                         R / "track_substitution/per_region.csv"],
    "track_dropout_subsets": [R / "subset_curve/summary.csv",
                              R / "subset_curve/per_region.csv"],
    "crosscell": [R / "crosscell/predict_gm12878.csv", R / "crosscell/predict_imr90.csv",
                  R / "crosscell/predict_k562.csv", R / "crosscell/predict_hmec_vs_rep1.csv",
                  R / "crosscell/predict_gm12878_2048check.csv"],
    "crosscell_superseded_inflated": [R / "crosscell/crosscell_matrix.csv",
                                      R / "crosscell/crosscell_per_region.csv"],
    "replicates": [X / "replicate_corr.csv"],
    "sequence_chip_grid": [R / "motif_grid/results_all.csv", R / "motif_grid/motif_grid.csv",
                           R / "motif_grid/annot_grid.csv"],
    "sharpening": [R / "sharpen/sharpen_median.csv"],
    "capture": [R / "capture/capture_summary.csv",
                R / "capture/capture_input_diagnostics.json"],
    "capture_lift": [Path("~/data/hic-multiome/pchic/lift_capture_hg38.json").expanduser()],
    # tables written by `scripts/multicell_akita.py --stage summary`
    "multicell_akita": [R / "multicell/test730.csv", R / "multicell/crosscell.csv",
                        R / "multicell/train_windows.csv", R / "multicell/training_stop.csv"],
    # tables written by `scripts/multicell_expanded.py --stage summary`
    "multicell_expanded": [R / f"multicell_expanded/{n}.csv" for n in
                           ("test730", "crosscell", "scaling",
                            "resolution_vs_replicates", "training_stop")],
    # tables written by `scripts/resolution_inputs.py` and `averaging_ceiling.py`
    "resolution_inputs": [R / f"resolution_inputs/{n}.csv" for n in
                          ("summary", "runs", "resolution_compare",
                           "resolution_compare_summary", "averaging_ceiling")],
    # tables written by `scripts/coarse_gap.py --stage summary` (GM12878 only)
    "coarse_gap": [R / f"coarse_gap/{n}.csv" for n in
                   ("summary", "runs", "heldout", "maps_gm12878")],
    # tables written by `scripts/multicell_expanded.py --stage refsummary`
    "multicell_reference": [R / f"multicell_reference/{n}.csv" for n in
                            ("reference_compare", "reference_summary", "training_stop",
                             "split_disjoint")],
}
ALL13 = {"CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"}


def best_val(text):
    v = [float(x) for x in re.findall(r"dist-pearson=(-?[0-9.]+)", text)]
    return max(v) if v else None


def proxy_round1():
    """Round-1 single-track drops of the 3-chromosome proxy search, from logs."""
    rows = []
    for log in sorted((R / "backward_fast").glob("cand_*.log")):
        t = log.read_text(errors="replace")
        if "early stop at epoch" not in t and "done." not in t:
            continue
        cfg = yaml.safe_load(open(log.with_suffix(".yaml")))
        sel = set(cfg["train"].get("select_tracks") or [])
        dropped = sorted(ALL13 - sel)
        rows.append({"n_tracks": len(sel), "dropped": "+".join(dropped) or "(none)",
                     "val": best_val(t), "chroms": ",".join(cfg["train"].get("chroms") or [])})
    return rows


def main():
    missing = []
    for exp, srcs in COPY.items():
        d = RES / exp
        d.mkdir(parents=True, exist_ok=True)
        for s in srcs:
            if s.exists():
                shutil.copy2(s, d / s.name)
            else:
                missing.append(str(s))
    rows = proxy_round1()
    if rows:
        d = RES / "backward_elimination"
        with open(d / "proxy_chr2_3_5_round1.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(sorted(rows, key=lambda r: -(r["val"] or 0)))
    print(f"collected into {RES}")
    for m in missing:
        print("missing (not yet produced?):", m)


if __name__ == "__main__":
    main()
