"""results/one_model/REMAINING.md: the six analyses of registry section 98 side by side with the paper's numbers.

Builds the numeric tables from the result files of `results/one_model/remaining/*` and of the existing sections
(the paper's current numbers are read from the files the figure scripts read, never retyped). The text parts that
state which sentences of main.tex change are in `results/one_model/remaining/sentences_*.md` (written by hand after
reading the tables) and are appended.

    python scripts/one_model_remaining_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
REM = R / "one_model/remaining"


def f(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"



# ------------------------------------------------------------------ 1. AML
AML_ROWS = [("measured", "truth (measured patient map)"), ("reference", "AML reference map of the fold (other patients)"),
            ("consensus12", "consensus12 (12 cell lines)"),
            ("base_refin@aml", "PAPER 12-cell model, AML reference input (all12refin_drop)"),
            ("rel_aml", "RELEASE, AML reference input"),
            ("c12_refin", "PAPER 12-cell refin, consensus12"), ("c12_refres", "PAPER 12-cell refres, consensus12"),
            ("c12_diffcells", "PAPER 12-cell differential, consensus12"),
            ("rel_c12", "RELEASE, consensus12"),
            ("honest_old", "PAPER-model family, no reference (all12honest_drop, run here)"),
            ("rel_noref", "RELEASE, marks only (reference plane 0)"),
            ("akita", "Akita (sequence only)"), ("corigami_q", "C.Origami, quantile-matched"), ("corigami_raw", "C.Origami, as given"),
            ("epiphany_zero", "Epiphany, H3K4me3 = 0"), ("epiphany_mean", "Epiphany, H3K4me3 = mean"),
            ("aml_refin", "(trained with AML patients, not comparable) aml_refin"), ("aml_diff", "(trained with AML) aml_diff")]
COLS6 = [("loop", "all"), ("loop", "private"), ("loop", "recurrent"), ("stripe", "all"), ("stripe", "private"), ("stripe", "recurrent")]


def _cell(x, nd=2):
    return "" if x is None or not np.isfinite(x.v) else f"{x.v:.{nd}f} ({x.lo:.{nd}f}-{x.hi:.{nd}f})"


def aml_wide(df, val, filt, cols=COLS6, nd=2):
    d = df[filt]
    rows = []
    for m, lab in AML_ROWS:
        row = [lab]
        for k, c in cols:
            x = d[(d.model == m) & (d.kind == k) & (d["class"] == c)]
            if len(x):
                x = x.iloc[0]
                row.append(f"{x[val]:.{nd}f} ({x['lo']:.{nd}f}-{x['hi']:.{nd}f})" if np.isfinite(x[val]) else "")
            else:
                row.append("")
        if any(row[1:]):
            rows.append(row)
    return rows


def sec_aml():
    A = REM / "aml"
    sub0 = "all objects (each model on its own coverage)"
    det = pd.read_csv(A / "detection_rates.csv")
    au = pd.read_csv(A / "auroc.csv")
    sp = pd.read_csv(A / "spearman.csv")
    detg = pd.read_csv(A / "detection_rates_gain.csv")
    aug = pd.read_csv(A / "auroc_gain.csv")
    L = ["## 1. AML-specific loops and stripes in the ten held-out patients\n"]
    L.append("Protocol of registry sections 62 / 72 (`scripts/aml_competitors_loops.py`, functions called unchanged by "
             "`scripts/one_model_remaining_aml.py`): 2276 AML-specific loops and 1075 stripes of 10 donors (Xu et al.), "
             "controls B_rand (distance-matched random pairs) and A_refrand (matched on AML reference and HSPC enrichment, "
             "distance, orientation), 192-bin windows of 10 kb, patient marks CTCF / H3K27ac / H3K27me3 (1360 and 1916 lack "
             "CTCF), detection = share of the objects above the 90th percentile of the same predictor at the B_rand pairs of "
             "the same donors (false-positive rate 10 percent by construction), 95 percent intervals over donors "
             "(500 resamples, the paper tables used 500 too). The old rows (PAPER) are recomputed from the same score table "
             "and reproduce the registry-72 values; the intervals differ in the third decimal because the resampling stream "
             "differs. The release checkpoint (`all12one_fullrd_10kb` view) has seen no AML sample; the lines of the "
             "table correspond to the paper numbers as follows:\n\n"
             "- RELEASE, AML reference input <-> `base_refin@aml` (paper: 48 % of the loops, 67 % of the stripes): the model "
             "without AML in training that reads the AML reference map of the fold's training patients as Hi-C input (AML data "
             "used as input only).\n"
             "- RELEASE, consensus12 <-> `c12_refin`, `c12_refres`, `c12_diffcells` (paper: 40 to 42 % of the loops, 52 to "
             "57 % of the stripes): reference = mean map of the 12 cell lines, no AML data anywhere.\n"
             "- RELEASE, marks only <-> the paper sentence \"from the marks alone, without any reference, the model detected "
             "28 % of the loops and 10 % of the stripes\". I could not find the source of 28 % / 10 % in the result files "
             "(the `aml_refin|none` columns give 0.48 / 0.68 above their own B_rand threshold and 0.34 / 0.58 as the share above "
             "the reference in `aml_train_v2`); the like-for-like comparison is therefore the dedicated no-reference "
             "model all12honest_drop_10kb, run here on the same windows and scored in the same table (row \"PAPER-model "
             "family, no reference\").\n"
             "- The AML-trained rows are listed for completeness; the release checkpoint was not trained on AML data, so there "
             "is no release counterpart. The paper's AML-inclusive result (Supplementary Fig. S4, Tables S7, S16) is not "
             "affected.\n")
    L.append("**Detection rate** (loops / stripes; all, private, recurrent; level of the predicted enrichment)\n")
    L.append(md(["predictor"] + [f"{k} {c}" for k, c in COLS6], aml_wide(det, "detection_rate", det.subset == sub0)))
    L.append("**AUROC of the predicted level, AML-specific objects against B_rand**\n")
    L.append(md(["predictor"] + [f"{k} {c}" for k, c in COLS6], aml_wide(au, "auroc", (au.subset == sub0) & (au.control == "B_rand"))))
    L.append("**AUROC against the A_refrand controls** (matched on AML reference and HSPC enrichment, distance, orientation)\n")
    L.append(md(["predictor"] + [f"{k} {c}" for k, c in COLS6], aml_wide(au, "auroc", (au.subset == sub0) & (au.control == "A_refrand"))))
    L.append("**Spearman of the predicted with the measured enrichment among the AML-specific objects** (level)\n")
    rows = []
    for m, lab in AML_ROWS:
        if m == "measured":
            continue
        row = [lab]
        for k, c in COLS6:
            x = sp[(sp.model == m) & (sp.kind == k) & (sp["class"] == c) & (sp.target == "level")]
            row.append(f"{x.spearman.iloc[0]:.2f} ({x.lo.iloc[0]:.2f}-{x.hi.iloc[0]:.2f})" if len(x) else "")
        if any(row[1:]):
            rows.append(row)
    L.append(md(["predictor"] + [f"{k} {c}" for k, c in COLS6], rows))
    L.append("**Contrast with HSPC** (score = prediction minus the same predictor on HSPC CD34+ inputs; truth, reference, consensus12 "
             "and AML-trained rows minus the measured HSPC consensus): detection rate\n")
    L.append(md(["predictor"] + [f"{k} {c}" for k, c in COLS6], aml_wide(detg, "detection_rate", detg.subset == sub0)))
    L.append("AUROC of the contrast against A_refrand\n")
    L.append(md(["predictor"] + [f"{k} {c}" for k, c in COLS6], aml_wide(aug, "auroc", (aug.subset == sub0) & (aug.control == "A_refrand"))))
    spg = pd.read_csv(A / "spearman_gain.csv")
    L.append("Spearman of the predicted gain with the measured gain (all loops / all stripes)\n")
    rows = []
    for m, lab in AML_ROWS:
        if m == "measured":
            continue
        row = [lab]
        for k in ("loop", "stripe"):
            x = spg[(spg.model == m) & (spg.kind == k) & (spg["class"] == "all") & (spg.target == "level")]
            row.append(f"{x.spearman.iloc[0]:.2f} ({x.lo.iloc[0]:.2f}-{x.hi.iloc[0]:.2f})" if len(x) else "")
        if any(row[1:]):
            rows.append(row)
    L.append(md(["predictor", "loops all", "stripes all"], rows))
    wf = A / "windows/aml_windows_summary.csv"
    if wf.exists():
        S = pd.read_csv(wf).set_index("method")
        cols = ["pearson", "ins_r_50k", "bnd_f1", "compartment_corr", "moc", "stripe_r", "loop_enrich_corr"]
        L.append("**Structure measures in the object windows of the eight patients with CTCF** (Supplementary Table S13; median over "
                 "patients, range in brackets; identical pixels for all methods)\n")
        names = {"consensus12": "consensus12", "c12_refin": "PAPER 12-cell refin", "c12_refres": "PAPER 12-cell refres",
                 "rel_c12": "RELEASE, consensus12", "rel_noref": "RELEASE, marks only", "rel_aml": "RELEASE, AML reference",
                 "honest_old": "PAPER-model family, no reference", "akita": "Akita", "corigami_q": "C.Origami (q)",
                 "epiphany_mean": "Epiphany (mean)", "epiphany_zero": "Epiphany (zero)"}
        rows = [[names[m]] + [f"{S.loc[m, c + '_median']:.3f} ({S.loc[m, c + '_min']:.3f}-{S.loc[m, c + '_max']:.3f})" for c in cols]
                for m in names if m in S.index]
        L.append(md(["method"] + cols, rows))
        P = pd.read_csv(A / "windows/aml_windows_paired.csv")
        L.append("Patients (of 8) in which the release row is ahead, per measure\n")
        rows = []
        for o in ("rel_c12", "rel_noref"):
            for other in ("akita", "corigami_q", "epiphany_mean", "epiphany_zero", "consensus12"):
                x = P[(P.ours == o) & (P.other == other)].set_index("metric")
                rows.append([names[o], names[other]] + [f"{int(x.loc[c, 'patients_ours_ahead'])}/{int(x.loc[c, 'n_patients'])}"
                                                        for c in sv_aml_metrics(P)])
        L.append(md(["release row", "against"] + sv_aml_metrics(P), rows))
    return "\n".join(L)


def sv_aml_metrics(P):
    return [m for m in ["pearson", "scc", "ins_r_50k", "ins_r_100k", "ins_r_250k", "bnd_f1", "moc", "stripe_r", "loop_enrich_corr",
                        "compartment_corr", "eigenvector_abs_corr"] if m in set(P.metric)]


# ------------------------------------------------------------------ 4. replicate ceiling
def sec_ceiling():
    d = pd.read_csv(REM / "replicate_ceiling/replicate_ceiling_table.csv")
    s = pd.read_csv(REM / "replicate_ceiling/replicate_ceiling_summary.csv")
    L = ["## 4. Replicate ceiling\n"]
    L.append("Statistic: median per-region Pearson of the reference-input model on the 413 Akita regions divided by the "
             "ceiling sqrt(r) (single-replicate target) or sqrt(2r/(1+r)) (merged target) of the agreement r of two "
             "biological replicates; deepest pair for GM12878 and K562. paper = all12refin_drop (dedicated per size), "
             "release = all12one_fullrd with the consensus (leave-one-out for training cells, consensus12 for "
             "held-out cells), noref = all12one_fullrd with the reference plane at 0. The ceilings are the existing "
             "ones; nothing was predicted again (the release 413-region tables of registry section 93 were used). "
             "Script `scripts/one_model_remaining_ceiling.py`, files `results/one_model/remaining/replicate_ceiling/`.\n")
    for res in (25, 10, 5, 2):
        x = d[(d.res_kb == res) & d.ceiling.notna()]
        rows = [[r["name"], r.group, f(r.paper_model), f(r.release_model), f(r.noref_model), f(r.consensus_alone), f(r.replicate_r),
                 f(r.ceiling), f(r.ratio_paper), f(r.ratio_release), f(r.ratio_noref)] for _, r in x.iterrows()]
        L.append(f"**{res} kb** (cells with a replicate pair)\n")
        L.append(md(["cell", "set", "paper", "release", "noref", "consensus", "replicate r", "ceiling", "paper/ceiling",
                     "release/ceiling", "noref/ceiling"], rows))
    L.append("**Summary over the cells with a ceiling** (training: GM12878, IMR-90; held-out: K562, HMEC, HMEC deep, HUVEC, HUVEC deep)\n")
    rows = [[r.res_kb, r.group, r.model, f"{r.n_with_ceiling}", f(r.median_model_where_ceiling), f(r.median_ceiling),
             f(r.median_ratio), f(r.min_ratio), f(r.max_ratio), f"{r.n_ge_95pct}/{r.n_with_ceiling}"] for _, r in s.iterrows()]
    L.append(md(["kb", "set", "model", "cells", "median model", "median ceiling", "median fraction", "min", "max", ">= 95%"], rows))
    x = d.dropna(subset=["ceiling_deep_pair"])
    rows = [[r["name"], r.res_kb, f(r.paper_model), f(r.release_model), f(r.noref_model), f(r.deep_replicate_r), f(r.ceiling_deep_pair),
             f(r.ratio_deep_paper), f(r.ratio_deep_release), f(r.ratio_deep_noref)] for _, r in x.iterrows()]
    L.append("**Deepest intact replicate pair as the ceiling** (GM12878 ENCSR965OVH, K562 ENCSR676BEP)\n")
    L.append(md(["cell", "kb", "paper", "release", "noref", "deep pair r", "ceiling", "paper/ceiling", "release/ceiling",
                 "noref/ceiling"], rows))
    return "\n".join(L)


# ------------------------------------------------------------------ 6. timing
def sec_timing():
    new = pd.read_csv(REM / "timing/runs.csv")
    old = pd.read_csv(R / "inference_benchmark/runs.csv")
    old = old[old.method.str.startswith("ours")].copy()
    old["key"] = old.job.str.replace("ours_", "")
    new["key"] = new.job.str.replace("release_", "").str.replace("paper_", "")
    L = ["## 6. Prediction time and memory\n"]
    L.append("Chromosome 2 of GM12878 (242 Mb) into one cool file, one job at a time, wall time from process start to exit "
             "(model and inputs loading and cool writing included), peak GPU memory (nvidia-smi over the process tree) and peak "
             "resident memory (psutil), \"default\" batch 8, \"lowest\" batch 1, test-time reverse-complement averaging, reference "
             "input; CPU rows on 8 threads, batch 1. Script `scripts/one_model_remaining_timing.py`, "
             "`results/one_model/remaining/timing/runs.csv`. **Hardware.** The paper rows (registry section 82, columns "
             "\"paper, section 82\") were measured on the Ryzen 9 7950X workstation with an RTX 4090 (CUDA device 1); the local GPUs "
             "were busy during this work, so both models were measured in one session on the second computer (AMD Ryzen "
             "Threadripper 7970X, one RTX 4090, host PC3301501): columns \"paper model, this host\" and \"release, this host\". "
             "Comparisons between models use the two columns of this host. The release rows read the 13 marks and "
             "methylation of GM12878 (14 tracks) and the cohesin fraction, the paper model the 13 marks.\n")
    rows = []
    for key in ["25kb_gpu_default", "25kb_gpu_lowest", "10kb_gpu_default", "10kb_gpu_lowest", "5kb_gpu_default",
                "5kb_gpu_lowest", "2kb_gpu_default", "2kb_gpu_lowest", "25kb_cpu_lowest", "10kb_cpu_lowest"]:
        o = old[old.key == key].iloc[0]
        p = new[(new.key == key) & (new.model == "paper")].iloc[0]
        q = new[(new.key == key) & (new.model == "release")].iloc[0]
        res, dev, st = key.split("_")
        rows.append([res.replace("kb", " kb"), dev.upper(), st, f"{o.wall_s:.1f} / {o.peak_gpu_mib} / {o.peak_rss_mib}",
                     f"{p.wall_s:.1f} / {p.peak_gpu_mib} / {p.peak_rss_mib}", f"{q.wall_s:.1f} / {q.peak_gpu_mib} / {q.peak_rss_mib}",
                     f"{q.wall_s / p.wall_s:.2f}", f"{q.out_mb:.1f}"])
    L.append(md(["grid", "device", "setting", "paper, section 82: s / GPU MiB / RAM MiB",
                 "paper model, this host: s / GPU MiB / RAM MiB", "release, this host: s / GPU MiB / RAM MiB",
                 "time release / paper model", "cool (MB)"], rows))
    return "\n".join(L)


# ------------------------------------------------------------------ 2. new cell types
def sec_iv():
    A = REM / "new_cell_types"
    T = pd.read_csv(A / "summary_test_windows.csv")
    P = pd.read_csv(A / "paired_contrasts.csv")
    ce = pd.read_csv(R / "independent_validation/ceilings.csv")
    names = {"pc3": "PC-3", "panc1": "Panc1", "nk": "NK", "caco2": "Caco-2"}
    cells = list(names)
    L = ["## 2. Pre-registered new cell types PC-3, Panc1, NK, Caco-2\n"]
    L.append("Protocol of registry section 74 and its pre-registration (`scripts/independent_validation.py` functions, "
             "`scripts/one_model_remaining_iv.py`): 413 Akita test windows (central 917 504 bp), seven marks, identical "
             "pixels, per-window Pearson, medians over windows, paired bootstrap over windows (2000 resamples, seed 0), "
             "consensus12 as Hi-C reference input. The release checkpoint is used unchanged (one network for 10 and 25 kb; "
             "Meth absent, cohesin fraction 1); \"noref\" is the same checkpoint with the reference plane at 0 "
             "(`--no-reference`) and corresponds to the paper's model without consensus (honest). The PAPER rows are "
             "recomputed in the same pass (same pixels) and equal the values of section 74. Endpoint of the "
             "pre-registration applied to the release model: A = model with consensus above consensus12 in >= 3 of 4 cells, "
             "B = above the model without consensus in all 4.\n")
    order = [("ceiling", "split-half ceiling (Spearman-Brown)"), ("consensus12", "consensus12 alone"), ("best_cell", "best other measured cell (oracle)"),
             ("akita", "Akita (GM12878 head)"), ("honest:all", "PAPER no consensus, 7 marks (honest)"),
             ("refin:all", "PAPER reference input, 7 marks (refin)"), ("refres:all", "PAPER reference residual (refres)"),
             ("diff:all", "PAPER difference channels (diff)"),
             ("rel:all", "RELEASE with consensus12, 7 marks"), ("rel:ctcf_k27ac", "RELEASE with consensus12, CTCF + H3K27ac"),
             ("rel:ctcf", "RELEASE with consensus12, CTCF"), ("rel:none", "RELEASE with consensus12, no marks"),
             ("rel:all_noref", "RELEASE no consensus, 7 marks"), ("rel:ctcf_k27ac_noref", "RELEASE no consensus, CTCF + H3K27ac"),
             ("rel:ctcf_noref", "RELEASE no consensus, CTCF"), ("rel:none_noref", "RELEASE no consensus, no marks")]
    for res in (10000, 25000):
        rows = []
        for m, lab in order:
            row = [lab]
            for c in cells:
                if m == "ceiling":
                    x = ce[(ce.cell == c) & (ce.res == res)].median_r_full_SB
                else:
                    x = T[(T.cell == c) & (T.res == res) & (T.method == m)].pearson
                row.append(f(x.iloc[0]) if len(x) else "")
            rows.append(row)
        L.append(f"**Median per-window Pearson, {res // 1000} kb**\n")
        L.append(md(["predictor"] + [names[c] for c in cells], rows))
    L.append("**Paired contrasts (median difference of per-window Pearson, 95 % bootstrap interval over the 413 windows; share of windows in which the first is better)**\n")
    rows = []
    for res in (10000, 25000):
        for _, r in P[P.res == res].iterrows():
            rows.append([names[r.cell], f"{res // 1000} kb", r.contrast, f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]",
                         "yes" if r.above else "no", f"{r.share_x_better:.2f}"])
    L.append(md(["cell", "kb", "contrast", "difference", "interval excludes 0", "share better"], rows))
    import json
    v = json.loads((A / "primary_endpoint_verdict.json").read_text())
    L.append(f"Pre-registered endpoint with the release model at 10 kb: A (above consensus12) in {v['A_count']} of {v['n']} cells "
             f"({', '.join(k for k, x in v['A_above_consensus12'].items() if x)}), success {v['A_success']}; B (above the model without "
             f"consensus) in {sum(v['B_above_noref'].values())} of {v['n']} cells, success {v['B_success']}.\n")
    mets = ["pearson", "scc", "insulation_corr", "boundary_f1", "loop_enrich_corr", "loop_apa_ratio", "compartment_corr", "eigenvector_abs_corr"]
    L.append("**Structure measures at 10 kb** (median over windows, all eight measures)\n")
    rows = []
    for m, lab in [("consensus12", "consensus12"), ("best_cell", "best other cell"), ("akita", "Akita"), ("refin:all", "PAPER refin"),
                   ("honest:all", "PAPER honest"), ("rel:all", "RELEASE with consensus12"), ("rel:all_noref", "RELEASE no consensus")]:
        for c in cells:
            x = T[(T.cell == c) & (T.res == 10000) & (T.method == m)]
            if len(x):
                rows.append([lab, names[c]] + [f(x[k].iloc[0]) for k in mets])
    L.append(md(["predictor", "cell"] + mets, rows))
    return "\n".join(L)


# ------------------------------------------------------------------ 3. mouse
def sec_mouse():
    A = REM / "mouse"
    S = pd.read_csv(A / "summary_autosomes.csv")
    C = pd.read_csv(R / "mouse_transfer/ceilings.csv")
    P = pd.read_csv(A / "paired_contrasts.csv")
    cells = ["mesc", "ch12", "heart"]
    names = {"mesc": "mESC", "ch12": "CH12.LX", "heart": "heart"}
    L = ["## 3. Transfer to mouse\n"]
    L.append("Protocol of registry section 75 and its pre-registration (`scripts/mouse_transfer.py` functions, "
             "`scripts/one_model_remaining_mouse.py`): the 5629 autosome windows of the Akita-v2 tiles at 10 kb (ENCODE "
             "mm10 blacklist and assembly gaps excluded), mESC E14, CH12.LX, adult heart, the seven marks (heart has six: its "
             "H3K9me3 file does not exist, the mark is absent), human motifs and normalisers, the leave-one-out mean of the other "
             "mouse samples as reference, identical pixels, per-window Pearson and insulation correlation, medians over windows. The "
             "release checkpoint is used unchanged (13 marks + Meth; the six marks and Meth that mouse lacks are absent). "
             "noref = reference plane 0, the use that corresponds to the paper's model without consensus (honest). Shuffled "
             "marks and the marks of another sample as in section 75. PAPER rows are re-scored in the same pass; the untrained "
             "network (an architecture control) and the CH12 RAD21/SMC3 set were not rerun. The Akita-v2 mouse heads "
             "(1500-window random subset) and the replicate-experiment map are those of section 75.\n")
    order = [("untrained:all7", "PAPER untrained network"),
             ("honest_shuf:all7", "PAPER shuffled marks (no consensus)"), ("rel_shuf:all7_noref", "RELEASE shuffled marks (no consensus)"),
             ("honest:none", "PAPER sequence only (no consensus)"), ("rel:none_noref", "RELEASE sequence only (no consensus)"),
             ("honest:ctcf", "PAPER CTCF"), ("rel:ctcf_noref", "RELEASE CTCF"),
             ("honest:ctcf_k27ac", "PAPER CTCF + H3K27ac"), ("rel:ctcf_k27ac_noref", "RELEASE CTCF + H3K27ac"),
             ("honest_xm:all7", "PAPER marks of another sample"), ("rel_xm:all7_noref", "RELEASE marks of another sample"),
             ("honest:all7", "PAPER own seven marks (no consensus)"), ("rel:all7_noref", "RELEASE own marks (no consensus)"),
             ("consensus", "mouse consensus alone"), ("distance_decay", "distance decay only"), ("distance_oracle", "per-window diagonal means (in-sample bound)"),
             ("refin:all7", "PAPER reference input"), ("rel:all7", "RELEASE with the mouse consensus"),
             ("rel_shuf:all7", "RELEASE with consensus, shuffled marks"), ("rel_xm:all7", "RELEASE with consensus, marks of another sample"),
             ("rel:none", "RELEASE with consensus, no marks"),
             ("akita:Hsieh_mESC_uC", "Akita v2 mouse head Hsieh mESC (mESC only, 1500 windows)"), ("akita:Bonev_ES", "Akita v2 mouse head Bonev ES (mESC only)")]
    for res in (10000, 25000):
        for metric in ("pearson", "insulation_corr"):
            rows = []
            for m, lab in order:
                row = [lab]
                for c in cells:
                    x = S[(S.cell == c) & (S.res == res) & (S.method == m)]
                    row.append(f(x[metric + "_median"].iloc[0]) if len(x) else "")
                if any(row[1:]):
                    rows.append(row)
            L.append(f"**Median per-window {metric.replace('_', ' ')}, {res // 1000} kb**\n")
            L.append(md(["predictor"] + [names[c] for c in cells], rows))
    rows = []
    for c in cells:
        cc = C[(C.cell == c) & (C.res == 10000)]
        row = [names[c], f(cc.full_depth_median.iloc[0]) if len(cc) else "", f(cc.half_depth_median.iloc[0]) if len(cc) else ""]
        for m in ("honest:all7", "rel:all7_noref"):
            x = S[(S.cell == c) & (S.res == 10000) & (S.method == m)].pearson_median
            row += [f(x.iloc[0]), f(x.iloc[0] / cc.full_depth_median.iloc[0]) if len(cc) else ""]
        rows.append(row)
    L.append("**Fraction of the split-half ceiling at 10 kb** (median Pearson / median full-depth split-half ceiling of section 75)\n")
    L.append(md(["sample", "ceiling (full depth)", "split half", "PAPER honest", "PAPER fraction", "RELEASE noref", "RELEASE fraction"], rows))
    L.append("**Paired contrasts of the per-window Pearson** (median difference, 95 % bootstrap interval over windows; share of windows in which the first is better)\n")
    rows = []
    for _, r in P.iterrows():
        rows.append([names[r.cell], f"{r.res // 1000} kb", f"{r.m1} - {r.m2}", int(r.n), f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]",
                     f"{r.frac_windows_m1_above:.2f}"])
    L.append(md(["sample", "kb", "contrast", "windows", "difference", "share better"], rows))
    SC = pd.read_csv(A / "structure_class_contrasts.csv")
    L.append("**Structure classes at 10 kb: own marks minus shuffled marks / minus no marks / minus marks of another sample** (median over windows; boundary F1 mean)\n")
    rows = []
    for _, r in SC.iterrows():
        rows.append([names[r.cell], r.m1, r.ref, r.metric, f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"])
    L.append(md(["sample", "own marks", "minus", "measure", "difference"], rows))
    import json
    v = json.loads((A / "primary_endpoint_verdict.json").read_text())
    L.append(f"Pre-registered endpoint with the release model (no consensus, 10 kb; above distance decay and above shuffled marks in every "
             f"sample): met = {v['endpoint_met']}.\n")
    return "\n".join(L)


# ------------------------------------------------------------------ 5. Akita
def sec_akita():
    A = REM / "akita"
    S = pd.read_csv(A / "akita413_per_region_summary.csv")
    Pp = pd.read_csv(A / "akita413_pooled.csv")
    R_ = pd.read_csv(A / "akita413_paired.csv")
    G = pd.read_csv(A / "akita413_gain_over_consensus.csv")
    L = ["## 5. Comparison with Akita on its 413 GM12878 test regions\n"]
    L.append("Protocol of registry section 73 (`scripts/structure_scores_vs_akita.py` stage akita, called unchanged by "
             "`scripts/one_model_remaining_akita.py`): Akita's 2048 bp grid and targets, the 448-bin crop, one pixel mask per region, "
             "median over regions and pooled values with 2000 region bootstrap resamples. PAPER rows: `akita_native_honest` (ChIP only) "
             "and `akita_native_refres` (with the consensus of four other cells; models trained on Akita's own normalised targets). "
             "RELEASE rows: the 2 kb view of the release checkpoint predicts the 2000 bp grid of every sequence (GM12878 marks, "
             "leave-one-out consensus of the other 11 training cells as reference) and is resampled to Akita's bin centres by "
             "bilinear interpolation, the procedure of `score_like_hicgan.py`. Differences from the paper rows: the release "
             "checkpoint is trained on `proper` O/E maps and not on Akita's normalisation (the scoring target is Akita's own); "
             "the reference is the 11-cell leave-one-out consensus, not the four-cell consensus; GM12878 is a training cell "
             "(the 413 regions are excluded from the training windows); the Meth track of GM12878 is part of the \"+ Meth\" rows.\n")
    cols = ["pearson", "scc", "ins_r_10k", "ins_r_50k", "ins_r_250k", "bnd_f1", "bnd_strength_r", "moc", "stripe_r", "compartment_corr"]
    rows = []
    for _, s_ in S.iterrows():
        p = Pp[Pp.method == s_.method].iloc[0]
        rows.append([s_.label, int(s_.n_regions)] + [f(s_[c + "_median"]) for c in cols] +
                    [f"{p.bnd_f1_pooled:.3f} [{p.bnd_f1_pooled_lo:.3f}, {p.bnd_f1_pooled_hi:.3f}]",
                     f"{p.hft_bnd_recall:.3f} [{p.hft_bnd_recall_lo:.3f}, {p.hft_bnd_recall_hi:.3f}]",
                     f"{p.loop_recall:.3f} [{p.loop_recall_lo:.3f}, {p.loop_recall_hi:.3f}]",
                     f"{p.apa_ratio:.3f} [{p.apa_ratio_lo:.3f}, {p.apa_ratio_hi:.3f}]"])
    L.append(md(["predictor", "regions"] + cols + ["pooled bnd F1", "hFT bnd recall", "loop recall", "APA ratio"], rows))
    L.append("**Paired against Akita** (ours minus Akita, median over regions [95 % interval]; regions ahead)\n")
    rows = []
    for o in ["akita_native_honest", "akita_native_refres", "all12one_fullrd_chip", "all12one_fullrd_chip_nometh",
              "all12one_fullrd_cons", "all12one_fullrd_cons_nometh", "dedicated2kb_chip", "dedicated2kb_cons"]:
        x = R_[(R_.ours == o) & (R_.other == "akita")]
        if not len(x):
            continue
        lab = S[S.method == o].label.iloc[0] if (S.method == o).any() else o
        for _, r in x[x.kind == "per-region"].iterrows():
            rows.append([lab, r.metric, f"{r.median_diff:.3f} [{r.diff_lo:.3f}, {r.diff_hi:.3f}]", f"{int(r.wins)} of {int(r.n)}"])
        for _, r in x[x.kind == "pooled"].iterrows():
            rows.append([lab, r.metric + " (pooled)", f"{r.median_diff:.3f} [{r.diff_lo:.3f}, {r.diff_hi:.3f}]", ""])
    L.append(md(["predictor", "measure", "difference to Akita", "regions ahead"], rows))
    L.append("**Against the consensus of four measured cells**\n")
    rows = []
    for o in ["akita_native_refres", "all12one_fullrd_cons", "all12one_fullrd_chip"]:
        x = R_[(R_.ours == o) & (R_.other == "consensus4")]
        lab = S[S.method == o].label.iloc[0]
        for _, r in x[x.kind == "per-region"].iterrows():
            rows.append([lab, r.metric, f"{r.median_diff:.3f} [{r.diff_lo:.3f}, {r.diff_hi:.3f}]", f"{int(r.wins)} of {int(r.n)}"])
        for _, r in x[x.kind == "pooled"].iterrows():
            rows.append([lab, r.metric + " (pooled)", f"{r.median_diff:.3f} [{r.diff_lo:.3f}, {r.diff_hi:.3f}]", ""])
    L.append(md(["predictor", "measure", "difference to the consensus of 4 cells", "regions ahead"], rows))
    L.append("**Gain over the consensus** (regions ahead of both Akita and the consensus; the region at the 90th percentile of the smaller of both gains and the correlation of the two difference maps after averaging to 80 bins of 10 240 bp, the strip of the manuscript)\n")
    rows = [[r.model, int(r.regions), int(r.regions_ahead_of_akita), f"{r.share_ahead_of_akita_and_consensus4:.3f}",
             f"{r.median_gain_over_consensus4:+.3f}", r.region_p90, f(r.r_vs_measured_p90, 2), f(r.r_consensus_p90, 2),
             f(r.r_difference_maps_p90, 2)] for _, r in G.iterrows()]
    L.append(md(["model", "regions", "ahead of Akita", "share ahead of Akita and consensus", "median gain over consensus (Pearson)",
                 "p90 region", "r(pred, measured)", "r(consensus, measured)", "r(measured - cons, pred - cons)"], rows))
    return "\n".join(L)


INTRO = """# Remaining analyses on the release checkpoint all12one_fullrd (registry section 98)

Written 2026-11-01. The manuscript (`~/src/manuscripts/hic-multiome-natcomms/main.tex`, read only) switches to ONE released
checkpoint, `all12one_fullrd` (13 marks + Meth, cohesin-fraction row, reference dropout 0.3, one network for 2, 5, 10 and 25 kb;
views `runs/multicell_reference/all12one_fullrd_<k>kb`). This file reruns the six analyses that still rest on the older dedicated
models, with the protocol, regions and metrics of the paper numbers, and gives for each the paper's current number, the release
number, all statistics, and the sentences of main.tex that change. Existing results were not modified; everything new is under
`results/one_model/remaining/<analysis>/` (scripts `scripts/one_model_remaining_*.py`, predictions on the share under
`runs/aml_one_model`, `validation_new/pred_rel`, `validation_mouse/pred_rel`, `runs/akita_one_model`). Predictions ran on the
remote RTX 4090 (data staged under `~/scratch`, only files created for this work were removed afterwards), scoring on this workstation.

Two uses of the checkpoint appear everywhere: **with a reference** (the Hi-C reference input as in the paper's reference-input models:
consensus12, the leave-one-out consensus, the AML reference or the mouse consensus, depending on the analysis) and **marks only**
(`flexcible predict --no-reference`, the reference plane at 0), which is the counterpart of the paper's "model without consensus"
(all12honest_drop). Rows are labelled PAPER (the dedicated models behind the current numbers, recomputed in the same pass or read from
the existing files) and RELEASE. In every analysis the PAPER rows reproduce the registry values (checked: sections 72, 74, 75, 73, 49).

Not run / not available: the whole-genome 1 Mb tiles of the new cell types (section 74 secondary), the untrained-network control and the CH12.LX
RAD21/SMC3 set of the mouse analysis, the chromosome-held-out twin `all12one_fullrd_chrho` (still training when this was written; section 94).
The release checkpoint is one run (no seed spread). The timing was measured on the second computer, not on the workstation of section 82.

"""

FOOT = """
## Problems met

- The mouse predictions were produced twice: the first pull script excluded all sub-directories, copied nothing and deleted the
  remote copies. The remote job was rerun in full (1 h) and the pull script now verifies the file counts before deleting anything
  (`pull` logic described in registry section 98). No result was affected.
- The wall clock of the shell jumped by about seven hours once (02:40 to 08:45) while background jobs were waiting; no job was affected.
- The Akita comparison cannot use the paper protocol one to one (see its section): the paper rows are dedicated models trained in Akita's
  target space; a control run of the dedicated proper-target models separates the effect of the space from that of the checkpoint.
- The paper sentence "28% of the loops and 10% of the stripes" (marks only, AML) could not be traced to a result file.
"""


def read(name):
    f = REM / name
    return f.read_text() if f.exists() else ""


def main():
    parts = [INTRO, sec_aml(), read("sentences_aml.md"), sec_iv(), read("sentences_iv.md"), sec_mouse(), read("sentences_mouse.md"),
             sec_ceiling(), read("sentences_ceiling.md"), sec_akita(), read("sentences_akita.md"), sec_timing(), read("sentences_timing.md"), FOOT]
    out = R / "one_model/REMAINING.md"
    out.write_text("\n".join(parts))
    print(out, len("\n".join(parts)))


if __name__ == "__main__":
    main()
