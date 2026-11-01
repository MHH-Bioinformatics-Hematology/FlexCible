"""Tables of registry section 100: encodings of CTCF removal on the release checkpoint all12one_fullrd.

Reads results/one_model/ctcf_encoding/perturb_summary.csv (scripts/perturb_degron_zero.py --set encodings --stage score),
the section 95/99 summaries of the track-absent encoding (scripts/perturb_validate.py, crosscell_predict pipeline) and the
HAP1 comparison (scripts/perturb_hap1/compare_models.py). Writes ctcf_encoding/ENCODINGS_degron.csv,
ENCODINGS_hap1.csv and results/one_model/CTCF_ENCODING.md (tables plus the reading in ctcf_encoding/READING.md).
"""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
E = R / "one_model/ctcf_encoding"
ENC = ["b/c", "a/d", "e1", "e2/e3"]
LABEL = {"b/c": "b = c: CTCF removed, motif rows stay (signal zero = track absent)",
         "a/d": "a = d: CTCF removed + motif rows zero",
         "e1": "e1: motif rows zero, CTCF ChIP intact",
         "e2/e3": "e2 = e3: H3K36me3 removed (control)"}
# (pipeline label, summary file, model -> use/kb pattern, perturbation -> encoding)
SRC = [
    ("P1 region windows, ENCODE HCT116 ChIP (sections 95, 99)", "crosscell_perturb_summary.csv", "fullrd_enc_",
     {"minus_CTCF": "b/c", "minus_CTCF_nomotif": "a/d", "minus_nomotif": "e1", "minus_H3K36me3": "e2/e3"}),
    ("P2 region windows, ChIP of the untreated degron cells", "crosscell_perturb_summary.csv", "fullrd_encu_",
     {"minus_CTCF": "b/c", "minus_CTCF_nomotif": "a/d", "minus_nomotif": "e1", "minus_H3K36me3": "e2/e3"}),
    ("P3 tiled chromosomes (flexcible predict), ChIP of the untreated degron cells", "perturb_summary.csv", "fullrd_",
     {"zero_CTCF": "b/c", "zero_CTCF_motif": "a/d", "zero_motif": "e1", "zero_H3K36me3": "e2/e3"}),
]
frames = []
for pipe, f, pat, pm in SRC:
    d = pd.read_csv(E / f)
    d = d[(d.experiment == "ctcf") & (d["subset"] == "all" if "subset" in d else True)]
    d = d[d.model.str.contains(pat) & d.perturbation.isin(pm)].copy()
    if pat == "fullrd_":
        d = d[d.model.str.match(r"all12one_fullrd_(noref|cons)_\d+kb$")]
    d["use"] = d.model.str.extract(r"_(noref|cons)_")[0].map({"noref": "no consensus", "cons": "consensus"})
    d["res_kb"] = d.model.str.extract(r"_(\d+)kb")[0].astype(int)
    d["code"] = d.perturbation.map(pm)
    d["encoding"] = d.code.map(LABEL)
    d["pipeline"] = pipe
    d["run_perturbation"] = d.perturbation
    frames.append(d)
A = pd.concat(frames, ignore_index=True)
A["o"] = A.code.map({c: i for i, c in enumerate(ENC)})
A = A.sort_values(["pipeline", "use", "res_kb", "o"], ascending=[True, False, True, True])
cols = ["pipeline", "use", "res_kb", "code", "encoding", "run_perturbation", "model", "change_pearson_median", "change_pearson_q25",
        "change_pearson_q75", "change_pearson_mean", "change_pearson_n", "change_scc_median", "change_spearman_median",
        "pred_abs_change_mean", "meas_abs_change_mean", "loops_n", "loops_pred_frac_weaker", "loops_meas_frac_weaker",
        "loops_pred_median_change", "loops_meas_median_change", "loops_change_spearman", "apa_P0", "apa_P1", "apa_U", "apa_T",
        "boundaries_n", "boundary_ins_pred_median_change", "boundary_ins_meas_median_change", "boundary_change_spearman"]
A[cols].to_csv(E / "ENCODINGS_degron.csv", index=False, float_format="%.4f")

f = lambda v, d=3: "" if pd.isna(v) else f"{v:.{d}f}"
pc = lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%"
PIPE = {SRC[0][0]: "P1", SRC[1][0]: "P2", SRC[2][0]: "P3"}
L = ["| input and pipeline | use | kb | encoding | change Pearson median [q25, q75] (mean, n) | SCC / Spearman median | "
     "mean abs change pred / meas | loops n | loops weaker pred (meas) | median loop change pred (meas) | loop Spearman | "
     "APA no change / perturbed (meas U / T) | boundaries n; insulation change pred (meas); Spearman |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in A.iterrows():
    L.append(f"| {PIPE[r.pipeline]} | {r.use} | {r.res_kb} | {r.encoding} | {f(r.change_pearson_median)} [{f(r.change_pearson_q25)}, "
             f"{f(r.change_pearson_q75)}] ({f(r.change_pearson_mean)}, {int(r.change_pearson_n)}) | "
             f"{f(r.change_scc_median)} / {f(r.change_spearman_median)} | {f(r.pred_abs_change_mean, 4)} / {f(r.meas_abs_change_mean, 4)} | "
             f"{int(r.loops_n)} | {pc(r.loops_pred_frac_weaker)} ({pc(r.loops_meas_frac_weaker)}) | "
             f"{f(r.loops_pred_median_change)} ({f(r.loops_meas_median_change)}) | {f(r.loops_change_spearman, 2)} | "
             f"{f(r.apa_P0, 2)} / {f(r.apa_P1, 2)} ({f(r.apa_U, 2)} / {f(r.apa_T, 2)}) | "
             + ("n/a" if pd.isna(r.boundaries_n) or pd.isna(r.boundary_ins_pred_median_change)
                else f"{int(r.boundaries_n)}; {f(r.boundary_ins_pred_median_change)} ({f(r.boundary_ins_meas_median_change)}); "
                     f"{f(r.boundary_change_spearman, 2)}") + " |")
(E / "ENCODINGS_degron.md").write_text("\n".join(L) + "\n")

# follow-up: cohesin and the ChIP set (registry section 100, second part)
FU = [("ENCODE ChIP, per-region (P1)", "crosscell_perturb_summary.csv", "fullrd_enc_", {"minus_CTCF": "reference: CTCF removed", "minus_H3K36me3": "control: H3K36me3 removed"}),
      ("degron-cell ChIP, per-region (P2 = h)", "crosscell_perturb_summary.csv", "fullrd_encu_",
       {"minus_CTCF": "h: CTCF removed (same 12 marks as ENCODE)", "minus_CTCF_RAD21": "f: CTCF and RAD21 (and SMC3) removed",
        "minus_RAD21": "g: RAD21 (and SMC3) removed, CTCF intact", "minus_H3K36me3": "control: H3K36me3 removed"}),
      ("degron-cell ChIP, tiled (P3, figure input)", "perturb_summary.csv", "fullrd_",
       {"zero_CTCF_RAD21": "f: CTCF and RAD21 zero", "zero_RAD21": "g: RAD21 zero, CTCF intact"}),
      ("ENCODE ChIP + RAD21 of the degron cells, per-region", "crosscell_perturb_summary.csv", "fullrd_enck_",
       {"minus_CTCF": "i: CTCF removed", "minus_H3K36me3": "control: H3K36me3 removed"}),
      ("ENCODE ChIP + CTCF of the degron cells, per-region", "crosscell_perturb_summary.csv", "fullrd_encc_",
       {"minus_CTCF": "j: CTCF removed", "minus_H3K36me3": "control: H3K36me3 removed"}),
      ("degron-cell ChIP + CTCF of ENCODE, per-region", "crosscell_perturb_summary.csv", "fullrd_encd_",
       {"minus_CTCF": "k: CTCF removed", "minus_H3K36me3": "control: H3K36me3 removed"})]
FR = []
for pipe, fn, pat, pm in FU:
    d = pd.read_csv(E / fn)
    d = d[(d.experiment == "ctcf") & ((d["subset"] == "all") if "subset" in d else True)]
    d = d[d.model.str.contains(pat) & d.perturbation.isin(pm)].copy()
    if pat == "fullrd_":
        d = d[d.model.str.match(r"all12one_fullrd_(noref|cons)_\d+kb$")]
    d["use"] = d.model.str.extract(r"_(noref|cons)_")[0].map({"noref": "no consensus", "cons": "consensus"})
    d["res_kb"] = d.model.str.extract(r"_(\d+)kb")[0].astype(int)
    d["encoding"] = d.perturbation.map(pm)
    d["pipeline"] = pipe
    d["o"] = d.perturbation.map({k: i for i, k in enumerate(pm)})
    FR.append(d)
F = pd.concat(FR, ignore_index=True)
F["po"] = F.pipeline.map({x[0]: i for i, x in enumerate(FU)})
F = F.sort_values(["po", "use", "res_kb", "o"], ascending=[True, False, True, True])
F[[c for c in cols if c in F.columns]].to_csv(E / "ENCODINGS_cohesin.csv", index=False, float_format="%.4f")
FL = ["| input and pipeline | use | kb | encoding | change Pearson median [q25, q75] (n) | mean abs change pred / meas | loops n | "
      "loops weaker pred (meas) | median loop change pred (meas) | loop Spearman | APA no change / perturbed |", "|" + "---|" * 11]
for _, r in F.iterrows():
    FL.append(f"| {r.pipeline} | {r.use} | {r.res_kb} | {r.encoding} | {f(r.change_pearson_median)} [{f(r.change_pearson_q25)}, "
              f"{f(r.change_pearson_q75)}] ({int(r.change_pearson_n)}) | {f(r.pred_abs_change_mean, 4)} / {f(r.meas_abs_change_mean, 4)} | "
              f"{int(r.loops_n)} | {pc(r.loops_pred_frac_weaker)} ({pc(r.loops_meas_frac_weaker)}) | "
              f"{f(r.loops_pred_median_change)} ({f(r.loops_meas_median_change)}) | {f(r.loops_change_spearman, 2)} | "
              f"{f(r.apa_P0, 2)} / {f(r.apa_P1, 2)} |")
(E / "ENCODINGS_cohesin.md").write_text("\n".join(FL) + "\n")

# HAP1
H = pd.read_csv(R / "perturb_hap1/all12one_fullrd_comparison.csv")
H.to_csv(E / "ENCODINGS_hap1.csv", index=False, float_format="%.4f")
HL = {"release, consensus": ("consensus", "CTCF signal zero + motif rows zero (protocol of section 99)"),
      "release, no reference": ("no consensus", "CTCF signal zero + motif rows zero (protocol of section 99)"),
      "release, consensus, CTCF signal zero only": ("consensus", "CTCF signal zero only, motif rows stay"),
      "release, no reference, CTCF signal zero only": ("no consensus", "CTCF signal zero only, motif rows stay")}
H = H[H.model.isin(HL)]
out = ["| use | kb | encoding at the deleted site | 12 loops decreased | mean predicted / measured change at the 10 convergent loops | "
       "measured / predicted | control site 1 Mb away, max abs loop change | insulation direction (100 + 200 kb, x/16) | "
       "whole-region Pearson of the mean of clones, min / median / max | single clones | control-site Pearson |", "|" + "---|" * 11]
H = H.assign(u=H.model.map(lambda m: HL[m][0]), e=H.model.map(lambda m: HL[m][1]))
H = H.sort_values(["u", "res", "e"], ascending=[False, False, False])
for _, r in H.iterrows():
    out.append(f"| {r.u} | {r.res[:-2]} | {r.e} | {r.loops_all_pred_decrease}/{r.loops_all} | {f(r.loop_pred_change_mean)} / "
               f"{f(r.loop_meas_change_mean, 2)} | {r.loop_ratio_meas_over_pred:.1f} | {r.loops_all_ctrl_abs_max:.4f} | "
               f"{r.ins_total_sign_agree} | {r.pearson_mean_of_clones_min:.2f} / {r.pearson_mean_of_clones_median:.2f} / "
               f"{r.pearson_mean_of_clones_max:.2f} | {r.pearson_single_clone_min:.2f} to {r.pearson_single_clone_max:.2f} | "
               f"{r.pearson_ctrl_min:.2f} to {r.pearson_ctrl_max:.2f} |")
(E / "ENCODINGS_hap1.md").write_text("\n".join(out) + "\n")
READING = (E / "READING.md").read_text() if (E / "READING.md").exists() else ""
PIPES = "\n".join(f"- {PIPE[k]}: {k}." for k in PIPE)
(R / "one_model/CTCF_ENCODING.md").write_text(
    "# Encodings of CTCF removal on the release candidate all12one_fullrd\n\nRegistry section 100, 2026-11-01. Generated by "
    "`scripts/ctcf_encoding_report.py`; tables in `results/one_model/ctcf_encoding/ENCODINGS_degron.{csv,md}` and "
    "`ENCODINGS_hap1.{csv,md}`.\n\n" + READING + "\n## A. HCT116 CTCF degron (ENCODE CTCF-AID, 413 test regions)\n\n"
    "Encodings: a = CTCF signal zero and the CTCF motif rows (MA0139.2 forward and reverse) zero, genome-wide; b = CTCF signal zero, motif rows stay; "
    "c = CTCF track declared absent; d = absent and motif rows zero; e1 = motif rows zero with the CTCF ChIP intact; e2, e3 = H3K36me3 "
    "zero or absent. Inputs and pipelines:\n\n" + PIPES + "\n\n" + "\n".join(L) + "\n\n## B. HAP1 single-site deletions (Sanborn et al. 2015)\n\n"
    "Deleted site +-1 kb; 7 deletions, 19 clones, 12 loops at the deleted sites (10 convergent); measured mean change at the convergent loops as stated.\n\n"
    + "\n".join(out) + "\n\n## C. Cohesin and the ChIP set (follow-up)\n\n"
    + (E / "READING_cohesin.md").read_text() + "\n" + "\n".join(FL) + "\n")
print("\n".join(L)); print(); print("\n".join(out))
