"""Degron perturbation test with zero-signal semantics (registry section 21b).

Section 21 set one protein to its ABSENT value, the encoding of a mark that was
not measured. Here `flexcible predict --zero-region TRACK:chrom:0-length` sets the
raw signal of the protein to ZERO on every chromosome of the test regions (the
state of a region without binding; the motif channels stay, as the DNA does in a
degron). Perturbations of HCT116 (training sample hct116, ChIP of the training
set):

  full             unperturbed
  zero_CTCF        CTCF signal zero genome-wide
  zero_RAD21       RAD21 signal zero genome-wide
  zero_H3K36me3    control mark
  zero_CTCF_motif  CTCF zero plus motif MA0139.2 removed (site deletion)

Predicted change = perturbed minus full, scored against the measured change of
the RAD21-mAC and CTCF-AID experiments with the section-21 regions (413 Akita
test regions), O/E maps, loop and boundary calls and metrics (scripts/
perturb_validate.py is imported, not copied).

    CUDA_VISIBLE_DEVICES=1 python scripts/perturb_degron_zero.py --stage predict score
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import perturb_validate as pv                                        # noqa: E402
import benchmark_metrics as bm                                       # noqa: E402
from flexcible import bioscore as bs                                 # noqa: E402,F401

HM = pv.HM
RUNS = HM / "runs/multicell_reference"
D = HM / "perturb/degron_zero"
COOLS = D / "cool"
SCORE = D / "score"
RES_OUT = REPO / "results/perturb_degron_zero"
OE_DIR = pv.OUT                       # section-21 O/E maps (read only)
THREADS = 8
BATCH = 4
PY = sys.executable

PERTS = {"full": [],
         "zero_CTCF": [("CTCF",)],
         "zero_RAD21": [("RAD21",)],
         "zero_H3K36me3": [("H3K36me3",)],
         "zero_CTCF_motif": [("CTCF",), ("motif", "MA0139.2")],
         # encodings of registry section 100 (release checkpoint): the track declared absent
         # (not measured, --absent-track) and the CTCF motif rows (MA0139.2 fwd and rev) zero
         "zero_motif": [("motif", "MA0139.2")],
         "absent_CTCF": [("absent", "CTCF")],
         "absent_H3K36me3": [("absent", "H3K36me3")],
         "absent_CTCF_motif": [("absent", "CTCF"), ("motif", "MA0139.2")],
         # cohesin follow-up of section 100: CTCF and RAD21 signal zero (SMC3 is absent in this ChIP set)
         "zero_CTCF_RAD21": [("CTCF",), ("RAD21",)]}
MODELS = {  # name -> (resolution, perturbations)
    "all12honest_drop_10kb": (10000, list(PERTS)),
    "all12honest_drop_25kb": (25000, list(PERTS)),
    "all12refin_drop_10kb": (10000, ["full", "zero_CTCF"]),
}
EXP_MATCH = {"ctcf": "zero_CTCF", "rad21": "zero_RAD21"}
# Release checkpoint in its no-consensus use (registry section 99/100): the per-resolution view
# all12one_fullrd_<k>kb of the one checkpoint, `flexcible predict --no-reference` (consensus plane at 0).
# `--set release` selects it; results go to results/perturb_degron_zero/all12one_fullrd_noref and the
# cool files to <D>/cool/all12one_fullrd_noref_<k>kb. The default set is the dedicated models above.
RELEASE_MODELS = {"all12one_fullrd_noref_10kb": (10000, ["full", "zero_CTCF", "zero_H3K36me3"])}
VIEW_OF = {"all12one_fullrd_noref_10kb": "all12one_fullrd_10kb"}
NO_REFERENCE = {"all12one_fullrd_noref_10kb"}
# Encodings of CTCF removal (registry section 100): the release views at 10 and 25 kb without (plane at 0) and
# with the consensus (leave-one-out map of HCT116, the checkpoint's reference for the cell).
ENC_PERTS = ["full", "zero_CTCF", "zero_CTCF_motif", "zero_motif", "absent_CTCF", "absent_CTCF_motif",
             "zero_H3K36me3", "absent_H3K36me3", "zero_CTCF_RAD21", "zero_RAD21"]
# the track-absent input equals the raw-zero input (TrackNormalizer.absent_values is the normalised value of a raw-zero
# track), checked bit for bit at 10 kb without consensus; the absent_* runs are made there only
ENC_MODELS = {f"all12one_fullrd_{u}_{k}kb": (k * 1000, ENC_PERTS if (u, k) == ("noref", 10) else
                                             [p for p in ENC_PERTS if not p.startswith("absent_")])
              for k in (10, 25) for u in ("noref", "cons")}
ENC_VIEW = {m: f"all12one_fullrd_{m.rsplit('_', 1)[1]}" for m in ENC_MODELS}
ENC_NOREF = {m for m in ENC_MODELS if "_noref_" in m}
EXPS = ("ctcf", "rad21")


def oe_path(exp, cond, res):
    if exp == "rad21" and cond == "untreated":
        return pv.mx.target("hct116", res)
    return OE_DIR / "oe" / f"{exp}_{cond}_proper_{res // 1000}kb.cool"


def cool_path(model, pert):
    return COOLS / model / f"{pert}.cool"


def test_chroms():
    return list(dict.fromkeys(pv.regions().chrom))


def stage_predict():
    chroms = test_chroms()
    for model, (res, perts) in MODELS.items():
        run = VIEW_OF.get(model, model)
        cfg = RUNS / f"{run}.yaml"
        ck = RUNS / run / (RUNS / run / "best").read_text().strip()
        sizes = cooler.Cooler(str(pv.mx.target("hct116", res))).chromsizes
        for pert in perts:
            out = cool_path(model, pert)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            zr = []
            for spec in PERTS[pert]:
                if spec[0] == "absent":
                    zr += ["--absent-track", spec[1]]           # genome-wide, declared not measured
                    continue
                for c in chroms:
                    prefix = ":".join(spec)                     # TRACK or motif:ID
                    zr += ["--zero-region", f"{prefix}:{c}:0-{int(sizes[c])}"]
            cmd = [PY, "-m", "flexcible.cli", "predict", "--config", cfg, "--checkpoint", ck,
                   "--sample", "hct116", "--chroms", ",".join(chroms), "--tta",
                   "--batch-size", BATCH, *zr, *(["--no-reference"] if model in NO_REFERENCE else []),
                   "--out", out]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS=str(THREADS),
                       MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS))
            logf = D / "logs" / f"{model}_{pert}.log"
            logf.parent.mkdir(parents=True, exist_ok=True)
            with open(logf, "w") as fh:
                fh.write("[cmd] " + " ".join(map(str, cmd)) + "\n")
                fh.flush()
                rc = subprocess.call(["nice", "-n", "10"] + [str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                                     cwd=REPO, env=env)
            print(f"predict {model} {pert} rc={rc}", flush=True)
            if rc != 0 and out.exists():
                out.unlink()


def stage_score():
    SCORE.mkdir(parents=True, exist_ok=True)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    pv.OUT = SCORE                       # the summary helper writes APA arrays there
    R = pv.regions()
    rows, loops_rows, bnd_rows, pdist, apa = [], [], [], {}, {}
    for exp, e in pv.EXPERIMENTS.items():
        if exp not in EXPS:
            continue
        for model, (res, perts) in MODELS.items():
            u_p, t_p = oe_path(exp, "untreated", res), oe_path(exp, "treated", res)
            have = [p for p in perts if cool_path(model, p).exists()]
            if "full" not in have or not (u_p.exists() and t_p.exists()):
                continue
            w = {10000: 5, 25000: 3}[res]
            for _, r in R.iterrows():
                b0, b1 = r.start // res, -(-r.end // res)
                n = b1 - b0
                U, T = pv._fetch(u_p, r.chrom, b0, b1), pv._fetch(t_p, r.chrom, b0, b1)
                P = {p: pv._fetch(cool_path(model, p), r.chrom, b0, b1) for p in have}
                if U is None or T is None or P["full"] is None:
                    continue
                P0 = P["full"]
                dM = T - U
                anchors = pv.loop_anchors(e["calls"], r.chrom, b0, n, res)
                bnds = pv.boundaries(e["calls"], r.chrom, b0, n, res, w)
                for pert in have:
                    if pert == "full" or P[pert] is None:
                        continue
                    P1 = P[pert]
                    dP = P1 - P0
                    mask = bm.triu_mask(n, 2) & np.isfinite(dM) & np.isfinite(dP)
                    rows.append({"experiment": exp, "model": model, "perturbation": pert,
                                 "region": r.key,
                                 "pearson": bm.pearson(dP, dM, mask),
                                 "spearman": bm.spearman(dP, dM, mask),
                                 "scc": bm.scc(dP, dM, mask),
                                 "pred_abs_change": float(np.nanmean(np.abs(dP[mask]))),
                                 "meas_abs_change": float(np.nanmean(np.abs(dM[mask])))})
                    for d, (a, b) in bm.per_distance(dP, dM, mask).items():
                        pdist.setdefault((exp, model, pert, d * res), []).append((a, b))
                    if len(anchors):
                        su, st_ = pv.strength(U, anchors), pv.strength(T, anchors)
                        s0, s1 = pv.strength(P0, anchors), pv.strength(P1, anchors)
                        for k, (i, j) in enumerate(anchors):
                            loops_rows.append({"experiment": exp, "model": model,
                                               "perturbation": pert, "region": r.key,
                                               "i": int(i), "j": int(j),
                                               "distance_bp": int((j - i) * res),
                                               "meas_untreated": su[k], "meas_treated": st_[k],
                                               "pred_full": s0[k], "pred_pert": s1[k]})
                        acc = apa.setdefault((exp, model, pert),
                                             {"n": 0, "U": 0, "T": 0, "P0": 0, "P1": 0})
                        h = 3
                        for (i, j) in anchors:
                            if i - h < 0 or j + h >= n:
                                continue
                            subs = [M[i - h:i + h + 1, j - h:j + h + 1] for M in (U, T, P0, P1)]
                            if not all(np.isfinite(s).all() for s in subs):
                                continue
                            for nm, s in zip(("U", "T", "P0", "P1"), subs):
                                acc[nm] = acc[nm] + bs.oe_from_log(s)
                            acc["n"] += 1
                    if len(bnds):
                        iu_, it_ = bm.insulation(U, w), bm.insulation(T, w)
                        i0, i1 = bm.insulation(P0, w), bm.insulation(P1, w)
                        for b in bnds:
                            bnd_rows.append({"experiment": exp, "model": model,
                                             "perturbation": pert, "region": r.key, "bin": int(b),
                                             "meas_untreated": iu_[b], "meas_treated": it_[b],
                                             "pred_full": i0[b], "pred_pert": i1[b]})
            print(f"scored {exp} {model}", flush=True)
    Dd = pd.DataFrame(rows)
    Dd.to_csv(RES_OUT / "change_per_region.csv.gz", index=False, float_format="%.5f")
    L = pd.DataFrame(loops_rows)
    L.to_csv(RES_OUT / "loops_per_anchor.csv.gz", index=False, float_format="%.5f")
    Bn = pd.DataFrame(bnd_rows)
    Bn.to_csv(RES_OUT / "boundaries_per_bin.csv.gz", index=False, float_format="%.5f")
    pdr = []
    for (exp, model, pert, dbp), lst in pdist.items():
        a = np.concatenate([x for x, _ in lst]); b = np.concatenate([y for _, y in lst])
        if a.size > 50 and a.std() > 0 and b.std() > 0:
            pdr.append({"experiment": exp, "model": model, "perturbation": pert,
                        "distance_bp": dbp, "pearson": float(np.corrcoef(a, b)[0, 1]),
                        "n": int(a.size)})
    pd.DataFrame(pdr).to_csv(RES_OUT / "change_per_distance.csv", index=False, float_format="%.5f")
    S = pd.DataFrame(pv._summ_rows(Dd, L, Bn, apa, "all"))
    S.to_csv(RES_OUT / "perturb_summary.csv", index=False, float_format="%.4f")
    print(f"summary {len(S)} rows")


PROT = {"zero_CTCF": "CTCF", "zero_RAD21": "RAD21", "zero_H3K36me3": "H3K36me3",
        "zero_CTCF_motif": "CTCF", "minus_CTCF": "CTCF", "minus_RAD21": "RAD21",
        "minus_H3K36me3": "H3K36me3"}
EXPLAB = {"ctcf": "CTCF degron", "rad21": "RAD21 degron"}
PERTLAB = {"zero_CTCF": "CTCF zero", "zero_RAD21": "RAD21 zero", "zero_H3K36me3": "H3K36me3 zero (control)",
           "zero_CTCF_motif": "CTCF zero + MA0139.2 sites removed",
           "minus_CTCF": "CTCF absent", "minus_RAD21": "RAD21 absent",
           "minus_H3K36me3": "H3K36me3 absent (control)"}


def stage_compare():
    """Zero-signal rows next to the section-21 rows (absent-track encoding, same scorer, same
    regions): all12honest (no track dropout, the models of section 21) and the dropout models of
    this test. Written to comparison.csv and comparison.md."""
    new = pd.read_csv(RES_OUT / "perturb_summary.csv")
    old = pd.read_csv(pv.RES_OUT / "perturb_summary.csv")
    old = old[(old.subset == "all") & old.model.isin(
        ["all12honest_10kb", "all12honest_25kb", "all12honest_drop_10kb", "all12honest_drop_25kb"])
        & old.perturbation.str.startswith("minus_")]
    new = new.assign(encoding="zero signal")
    old = old.assign(encoding="absent track")
    A = pd.concat([old, new], ignore_index=True)
    A["input change"] = A.perturbation.map(PERTLAB)
    keep = ["experiment", "model", "encoding", "input change", "perturbation", "change_pearson_median",
            "change_pearson_q25", "change_pearson_q75", "change_scc_median", "pred_abs_change_mean",
            "meas_abs_change_mean", "loops_n", "loops_pred_frac_weaker", "loops_meas_frac_weaker",
            "loops_pred_median_change", "loops_meas_median_change", "loops_change_spearman",
            "apa_U", "apa_T", "apa_P0", "apa_P1", "boundary_ins_pred_median_change",
            "boundary_ins_meas_median_change", "boundary_change_spearman"]
    A = A[keep]
    order = {"all12honest_10kb": 0, "all12honest_drop_10kb": 1, "all12refin_drop_10kb": 2,
             "all12honest_25kb": 3, "all12honest_drop_25kb": 4}
    A = A.sort_values(by=["experiment", "model", "input change", "encoding"],
                      key=lambda c: c.map(order) if c.name == "model" else c).reset_index(drop=True)
    A.to_csv(RES_OUT / "comparison.csv", index=False, float_format="%.4f")
    f = lambda v, d=2: "" if pd.isna(v) else f"{v:.{d}f}"
    lines = ["| experiment | model | encoding | input change | change Pearson median (IQR) | change SCC | "
             "loops weaker, predicted (measured) | loop change Spearman | APA unperturbed / perturbed "
             "(measured untreated / treated) | boundary insulation change, predicted (measured) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in A.iterrows():
        lines.append(
            f"| {EXPLAB[r.experiment]} | {r.model} | {r.encoding} | {r['input change']} | "
            f"{f(r.change_pearson_median, 3)} ({f(r.change_pearson_q25, 3)} to {f(r.change_pearson_q75, 3)}) | "
            f"{f(r.change_scc_median, 3)} | {f(r.loops_pred_frac_weaker)} ({f(r.loops_meas_frac_weaker)}) | "
            f"{f(r.loops_change_spearman)} | {f(r.apa_P0)} / {f(r.apa_P1)} ({f(r.apa_U)} / {f(r.apa_T)}) | "
            f"{f(r.boundary_ins_pred_median_change, 3)} ({f(r.boundary_ins_meas_median_change, 3)}) |")
    (RES_OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--set", choices=["dedicated", "release", "encodings"], default="dedicated")
    a = ap.parse_args()
    global MODELS, RES_OUT, SCORE
    if a.set == "release":
        MODELS = RELEASE_MODELS
        RES_OUT = REPO / "results/perturb_degron_zero/all12one_fullrd_noref"
        SCORE = D / "score_all12one_fullrd_noref"
    if a.set == "encodings":
        global VIEW_OF, NO_REFERENCE, EXPS
        MODELS, VIEW_OF, NO_REFERENCE, EXPS = ENC_MODELS, ENC_VIEW, ENC_NOREF, ("ctcf",)
        RES_OUT = REPO / "results/one_model/ctcf_encoding"
        SCORE = D / "score_encodings"
    for st in a.stage:
        globals()[f"stage_{st}"]()


if __name__ == "__main__":
    main()
