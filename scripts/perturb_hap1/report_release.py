"""Writes results/one_model/PERTURB_HAP1_CTCF.md (registry section 99): HAP1 CTCF-site deletions and the HCT116 CTCF degron
for the dedicated ChIP-only model of the manuscript, the release model all12one_fullrd with the consensus plane and without."""
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results"
H = pd.read_csv(R / "perturb_hap1/all12one_fullrd_comparison.csv")
rows = [("loops at a deleted site, predicted contact decreased", lambda r: f"{r.loops_all_pred_decrease}/{r.loops_all}"),
        ("  same sign as measured", lambda r: f"{r.loops_all_sign_agree}/{r.loops_all}"),
        ("  convergent anchors only: decreased / sign agrees", lambda r: f"{r.loops_convergent_pred_decrease}/{r.loops_convergent}, {r.loops_convergent_sign_agree}/{r.loops_convergent}"),
        ("  mean predicted change at convergent loops [min, max] (log O/E)", lambda r: f"{r.loop_pred_change_mean:.3f} [{r.loop_pred_change_min:.3f}, {r.loop_pred_change_max:.3f}]"),
        ("  mean measured change at the same loops", lambda r: f"{r.loop_meas_change_mean:.3f}"),
        ("  measured / predicted (ratio of means; median of per-loop ratios)", lambda r: f"{r.loop_ratio_meas_over_pred:.1f}; {r.loop_ratio_median_per_loop:.1f}"),
        ("  control site 1 Mb away, largest |change| at the 12 loops", lambda r: f"{r.loops_all_ctrl_abs_max:.4f}"),
        ("insulation, sign agrees, 100 kb + 200 kb windows (8 sites each)", lambda r: r.ins_total_sign_agree),
        ("  100 kb window: sign agrees; mean predicted / measured change", lambda r: f"{r.ins100kb_sign_agree}/{r.ins100kb_n}; {r.ins100kb_pred_mean:.3f} / {r.ins100kb_meas_mean:.3f}"),
        ("  200 kb window: sign agrees; mean predicted / measured change", lambda r: f"{r.ins200kb_sign_agree}/{r.ins200kb_n}; {r.ins200kb_pred_mean:.3f} / {r.ins200kb_meas_mean:.3f}"),
        ("  control site, largest |insulation change| (100 / 200 kb)", lambda r: f"{r.ins100kb_ctrl_abs_max:.4f} / {r.ins200kb_ctrl_abs_max:.4f}"),
        ("whole-region Pearson, mean of clones: min / median / max over 14 deletions x regions", lambda r: f"{r.pearson_mean_of_clones_min:.2f} / {r.pearson_mean_of_clones_median:.2f} / {r.pearson_mean_of_clones_max:.2f}"),
        ("  single clones: min to max", lambda r: f"{r.pearson_single_clone_min:.2f} to {r.pearson_single_clone_max:.2f}"),
        ("  control-site prediction: min to max", lambda r: f"{r.pearson_ctrl_min:.2f} to {r.pearson_ctrl_max:.2f}"),
        ("  clone vs clone agreement (ceiling): min to max", lambda r: f"{r.clone_vs_clone_min:.2f} to {r.clone_vs_clone_max:.2f}"),
        ("SD of change map: predicted (median) / measured (median)", lambda r: f"{r.pred_sd_median:.3f} / {r.meas_sd_median:.2f}")]
out = ["## A. HAP1 single CTCF-site deletions (Sanborn et al. 2015)", "",
       "Units: mean over clones per deletion (7 deletion experiments, 19 clones), 10 kb and 5 kb. Six histone marks + CTCF of HAP1, RAD21, SMC3, H3K4me2, H3K79me2, H3K9ac, H4K20me1 and Meth absent. TTA, GPU 1.",
       "Dedicated = all12honest_drop (section 85, no consensus; the paper's current model). Consensus = all12one_fullrd with `--reference consensus12_proper_<res>.cool` (mean of the 12 training cells; HAP1 is none of them, so no leave-one-out is needed). No reference = the same plus `--no-reference` (plane at 0).", ""]
cols = [(m, rl) for m in H.model.unique() for rl in ("10kb", "5kb")]
hdr = "| statistic | " + " | ".join(f"{m} {rl}" for m, rl in cols) + " |"
out += [hdr, "|---|" + "---|" * len(cols)]
for name, f in rows:
    out.append(f"| {name} | " + " | ".join(f(H[(H.model == m) & (H.res == rl)].iloc[0]) for m, rl in cols) + " |")
out += ["", "Per deletion, clone and loop values: `results/perturb_hap1/all12one_fullrd_{consensus,noref}_*` (summary, region{1,2,3}_{map,loop,insulation}_scores). Cooler predictions: `/mnt/processed/wolffjoa/hic-multiome/perturb/hap1_insilico/cools_all12one_fullrd_{consensus,noref}/{10kb,5kb}/{wt_<region>,del_<deletion>,ctrl_<deletion>}.cool` (same layout as `cools/` of the dedicated model, read by `figures/scripts/fig_perturb_hap1.py` through `HAPOUT / \"cools\" / rl`).", ""]
# part B
def load(f, models):
    d = pd.read_csv(R / "perturb_validate" / f)
    return d[(d.experiment == "ctcf") & (d["subset"] == "all") & d.model.isin(models)]
cons = load("all12one_fullrd_perturb_summary.csv", ["all12one_fullrd_10kb", "all12one_fullrd_25kb"])
nore = load("all12one_fullrd_noref_perturb_summary.csv", ["all12one_fullrd_noref_10kb", "all12one_fullrd_noref_25kb"])
ded = load("perturb_summary.csv", ["all12honest_drop_10kb", "all12honest_drop_25kb"])
sets = [("dedicated all12honest_drop", ded), ("release, consensus", cons), ("release, no reference", nore)]
out += ["## B. HCT116 CTCF degron (ENCODE CTCF-mAID2, 6 h; all chromosomes, never trained)", "",
        "413 Akita test regions (411 / 410 with a finite Pearson at 10 / 25 kb for the section 95 and dedicated rows; see n). Predicted change = prediction without the track minus prediction with all tracks, HCT116 methylation given to the release model. Consensus plane for HCT116 = leave-one-out consensus (as in section 95). Pearson over pixels >= 2 bins off the diagonal per region; loops = untreated-map loops (hicDetectLoops), change = log O/E at the loop pixel.", "",
        "Paper (current text, dedicated model): CTCF removal median Pearson 0.34 (10 kb) and 0.40 (25 kb), 91% of the loops weaker, H3K36me3 removal near zero.", ""]
def line(label, f):
    cells = []
    for _, d in sets:
        for k in (10, 25):
            r = d[d.model.str.endswith(f"_{k}kb")]
            cells.append("n/a" if r.empty else f(r))
    return f"| {label} | " + " | ".join(cells) + " |"
hdr = "| statistic | " + " | ".join(f"{n} {k} kb" for n, _ in sets for k in (10, 25)) + " |"
for pert, title in (("minus_CTCF", "CTCF removed"), ("minus_H3K36me3", "H3K36me3 removed (control)")):
    out += [f"### {title}", "", hdr, "|---|" + "---|" * 6]
    g = lambda col: (lambda r: r[r.perturbation == pert].iloc[0][col])
    P = lambda col: (lambda r: f"{g(col)(r):.3f}")
    out += [line("change-map Pearson, median [q25, q75]", lambda r: f"{g('change_pearson_median')(r):.3f} [{g('change_pearson_q25')(r):.3f}, {g('change_pearson_q75')(r):.3f}]"),
            line("  mean (SD)", lambda r: f"{g('change_pearson_mean')(r):.3f} ({g('change_pearson_sd')(r):.3f})"),
            line("  regions", lambda r: f"{int(g('change_pearson_n')(r))}"),
            line("SCC median", P("change_scc_median")), line("Spearman median", P("change_spearman_median")),
            line("mean |change| predicted / measured", lambda r: f"{g('pred_abs_change_mean')(r):.4f} / {g('meas_abs_change_mean')(r):.4f}"),
            line("loops (n)", lambda r: f"{int(g('loops_n')(r))}"),
            line("loop change, median predicted / measured (log O/E)", lambda r: f"{g('loops_pred_median_change')(r):.3f} / {g('loops_meas_median_change')(r):.3f}"),
            line("loops weaker, predicted / measured", lambda r: f"{100 * g('loops_pred_frac_weaker')(r):.1f}% / {100 * g('loops_meas_frac_weaker')(r):.1f}%"),
            line("loop change Spearman (pred vs meas, per loop)", P("loops_change_spearman")),
            line("boundary insulation change, predicted median / measured median (10 kb only)", lambda r: "n/a" if pd.isna(g('boundary_ins_pred_median_change')(r)) else f"{g('boundary_ins_pred_median_change')(r):.3f} / {g('boundary_ins_meas_median_change')(r):.3f}"),
            line("boundary change Spearman (10 kb only)", lambda r: "n/a" if pd.isna(g('boundary_change_spearman')(r)) else f"{g('boundary_change_spearman')(r):.3f}"), ""]
out += ["Notes. (1) The 12 loops are all Sanborn HiCCUPS loops with an anchor at a deleted motif (10 convergent, 2 non-convergent); the paper counts all 12. (2) The insulation count of the dedicated model recomputed from its committed score files is 14/16 at 10 kb and 15/16 at 5 kb (section 85 text and README say 15 of 16 at 10 kb); both release uses give 16/16 at 10 kb and 15/16 at 5 kb. (3) The dedicated-model degron row recomputed from `perturb_summary.csv` gives 89.3% / 76.7% of the loops weaker (paper text 91%); Pearson 0.347 / 0.401 agrees with 0.34 / 0.40. (4) `flexcible predict --no-reference` still needs a reference map for the release checkpoint (no map is embedded in it), so it was given `--reference consensus12_proper_<res>.cool` and `--no-reference`, which zeroes the plane after reading it.", "", "Full tables: `results/perturb_validate/all12one_fullrd_noref_perturb_summary.csv` (this analysis), `all12one_fullrd_perturb_summary.csv` (section 95, with consensus), `perturb_summary.csv` (dedicated). Per-region, per-loop and per-boundary files: `all12one_fullrd_noref_{change_per_region,loops_per_anchor,boundaries_per_bin}.csv.gz` (gitignored *.gz, kept on disk), `all12one_fullrd_noref_change_per_distance.csv`.", ""]
(R / "one_model/PERTURB_HAP1_CTCF.md").write_text("# HAP1 CTCF-site deletions and CTCF degron on the release candidate all12one_fullrd, with and without the consensus\n\nRegistry section 99, 2026-11-01. Generated by `scripts/perturb_hap1/report_release.py`.\n\n" + "\n".join(out))
print("\n".join(out))
