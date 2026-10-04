"""AML-inclusive training (registry section 36).

The 12 non-AML training cells plus AML samples: MOLM-13 and the training
patients of Xu et al. 2022 (GSE152135 Hi-C, GSE152132 CUT&Tag: H3K27ac and
H3K27me3 for all 12 patients, CTCF for 9). U-937 WT and the test patients are
never trained on.

Patient split. The 12 patients with CUT&Tag form three folds of four. Fold f is
the test set of run ``..._f<f>`` (fold 0 is the unsuffixed run); the other
eight patients train. Folds are stratified by the two properties that change
what a patient offers the model: whether CTCF CUT&Tag exists (9 of 12) and
whether the patient carries a copy-number segment called from Hi-C coverage
(``results/aml_primary/explain_cnv_segments.csv``); within a stratum patients
are ordered by Hi-C depth (contacts at 25 kb) and dealt to the folds in turn,
the starting fold rotating between strata. Mutation subtypes are not
assigned to patients in the paper's tables (Supplementary Table 3 lists
subtype-specific loops with gene:count labels only), so no stratification by
subtype is possible. Written to ``results/aml_train/split.json``.

Reference map of an AML sample (``samples[].reference_cool``): the mean proper
O/E of the training patients of the fold, leave-one-out for a training patient
and the mean of all eight for MOLM-13; the same mean serves the test patients
and U-937 at inference. The json next to each map lists its cells
(``amlxu<patient>``, hicmultiome.reference.cell_identity), so a test patient
is never in its own reference; the non-AML cells keep their consensus_loo.

Hard validation (``train.hard_split``): windows where a training patient
deviates most from both the 12-cell consensus and the HSPC consensus (the
deviation score of scripts/hard_split.py, both scores in the patient's top
20 percent of the windows that pass the mask and CNV filters), 25 per patient,
held out from training for every sample and used for checkpoint selection.

    python scripts/aml_train.py --stage split refs
    python scripts/aml_train.py --stage hard --res 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
AML = HM / "aml"
PRIM = AML / "primary"
OE_DIR = AML / "oe"
INP = PRIM / "inputs"
CONS_DIR = PRIM / "consensus"
RES_OUT = REPO / "results/aml_train"
PATIENTS = ["018", "027", "103", "168", "270", "472", "546", "629", "673", "773", "1360", "1916"]
N_FOLDS = 3
ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
         "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
OVERSAMPLE = 3


def guard(p: str) -> str:
    """Sample name and leakage-guard identity of a patient (no underscore)."""
    return "amlxu" + p


def label(p: str) -> str:
    return f"xu_{p}"


def make_split() -> dict:
    cnv = pd.read_csv(REPO / "results/aml_primary/explain_cnv_segments.csv")
    cnv_p = set(cnv.patient.str.replace("xu_", "", regex=False))
    rows = []
    for p in PATIENTS:
        t = json.loads((REPO / f"results/aml_primary/targets/xu_{p}_25kb.json").read_text())
        marks = sorted(b.stem for b in (INP / label(p)).glob("*.bigwig"))
        rows.append({"patient": p, "contacts_M": round(t["contacts"] / 1e6, 1),
                     "ctcf": "CTCF" in marks, "cnv": p in cnv_p, "marks": marks})
    df = pd.DataFrame(rows)
    df["stratum"] = df.ctcf.map({True: "ctcf", False: "no_ctcf"}) + "/" + \
        df.cnv.map({True: "cnv", False: "no_cnv"})
    fold = {}
    start = 0
    for _, g in df.groupby("stratum", sort=True):
        for i, p in enumerate(g.sort_values("contacts_M").patient):
            fold[p] = (start + i) % N_FOLDS
        start += len(g)
    df["fold"] = df.patient.map(fold)
    folds = {str(f): sorted(df[df.fold == f].patient) for f in range(N_FOLDS)}
    return {"patients": rows, "folds": folds,
            "note": "fold f is the test set of the run suffixed _f<f> (fold 0: no suffix); "
                    "the other folds train; MOLM-13 trains in every fold, U-937 WT is "
                    "held out in every fold"}


def split() -> dict:
    f = RES_OUT / "split.json"
    if not f.exists():
        RES_OUT.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(make_split(), indent=1))
    return json.loads(f.read_text())


def fold_of(base: str) -> int:
    """Fold of a run base name (all12aml_..., suffix _f1 / _f2)."""
    import re
    m = re.search(r"_f(\d)$", base)
    return int(m.group(1)) if m else 0


def train_patients(fold: int) -> list[str]:
    sp = split()
    test = set(sp["folds"][str(fold)])
    return [p for p in PATIENTS if p not in test]


def test_patients(fold: int) -> list[str]:
    return split()["folds"][str(fold)]


def ref_path(fold: int, res: int, patient: str | None) -> Path:
    """Reference of a training patient (leave-one-out) or, patient None, the
    mean of the fold's training patients."""
    tag = f"lo_{patient}" if patient else "all"
    return CONS_DIR / f"amltrain_f{fold}_{tag}_{res // 1000}kb.cool"


def build_refs(fold: int, res: int) -> None:
    import aml_primary as ap
    tr = train_patients(fold)
    for p in [None] + tr:
        labs = [label(x) for x in tr if x != p]
        ap.build_aml_consensus(res, None, labs=labs, out=ref_path(fold, res, p))


def aml_samples(fold: int, res: int, with_ref: bool) -> list[dict]:
    """Sample dicts (config data.samples entries) of the AML training cells of
    a fold: the training patients and MOLM-13."""
    import multicell_expanded as mx
    out = []
    for p in train_patients(fold):
        have = {b.stem for b in (INP / label(p)).glob("*.bigwig")}
        s = {"name": guard(p), "cool_path": str(OE_DIR / f"xu_{p}_proper_{res // 1000}kb.cool"),
             "bigwig_dir": str(INP / label(p)), "missing_tracks": [t for t in ALL13 if t not in have],
             "assay": "hic"}
        if with_ref:
            s["reference_cool"] = str(ref_path(fold, res, p))
        out.append(s)
    d = AML / "hg38_molm13"
    s = {"name": "molm13", "cool_path": str(OE_DIR / f"molm13_proper_{res // 1000}kb.cool"),
         "bigwig_dir": str(d), "missing_tracks": mx.missing_marks(d), "assay": "hic"}
    if with_ref:
        s["reference_cool"] = str(ref_path(fold, res, None))
    out.append(s)
    return out


def aml_names(fold: int) -> list[str]:
    return [guard(p) for p in train_patients(fold)] + ["molm13"]


def hard_split_path(fold: int, res: int) -> Path:
    return RES_OUT / f"hard_f{fold}_{res // 1000}kb.json"


# --------------------------------------------------------------------------- #
def deviation(T: np.ndarray, C: np.ndarray) -> float:
    """0.5 * (1 - Pearson(T, C)) + 0.5 * var(T - C) / var(T) on the pixels
    finite in both (upper triangle, offset >= 2)."""
    n = T.shape[0]
    iu = np.triu_indices(n, 2)
    t, c = T[iu], C[iu]
    ok = np.isfinite(t) & np.isfinite(c) & (t != 0) & (c != 0)
    if ok.sum() < 200:
        return np.nan
    t, c = t[ok], c[ok]
    if t.std() == 0 or c.std() == 0:
        return np.nan
    return float(0.5 * (1 - np.corrcoef(t, c)[0, 1]) + 0.5 * np.var(t - c) / np.var(t))


def stage_hard(fold: int, res: int, per_patient: int = 25, frac: float = 0.2, seed: int = 0):
    """Patient-specific hard-validation windows of the training patients."""
    import cooler
    import multicell_expanded as mx
    from hicmultiome.matrix import fetch_diagonal_window
    import aml_primary_explain as ex
    cache = mx.CACHE / (f"mc_all12aml_ref_{res // 1000}kb" + (f"_f{fold}" if fold else ""))
    man = json.loads((cache / "manifest.json").read_text())
    W = mx.WINDOW[res][0]
    c12 = cooler.Cooler(str(mx.consensus12(res)))
    hs = cooler.Cooler(str(ex.hspc_consensus_path(res)))
    cnv = pd.read_csv(REPO / "results/aml_primary/explain_cnv_segments.csv")
    rng = np.random.default_rng(seed)
    cells, union = {}, []
    for p in train_patients(fold):
        shards = [s for s in man["shards"] if s["sample"] == guard(p) and s["split"] == "train"]
        pos = []
        for s in shards:
            z = np.load(s["path"])
            pos += [(s["chrom"], int(st)) for st in z["start"]]
        T = cooler.Cooler(str(OE_DIR / f"xu_{p}_proper_{res // 1000}kb.cool"))
        segs = cnv[cnv.patient == label(p)]
        rows = []
        for ch, st in pos:
            if any(r.chrom == ch and r.start < (st + W) * res and st * res < r.end
                   for r in segs.itertuples()):
                continue
            t = fetch_diagonal_window(T, ch, st, W)
            if np.isfinite(t).mean() < 0.8:
                continue
            rows.append((ch, st, deviation(t, fetch_diagonal_window(c12, ch, st, W)),
                         deviation(t, fetch_diagonal_window(hs, ch, st, W))))
        d = pd.DataFrame(rows, columns=["chrom", "start", "sC", "sH"]).dropna()
        d["rC"] = d.sC.rank(pct=True, ascending=False)
        d["rH"] = d.sH.rank(pct=True, ascending=False)
        q = d[(d.rC <= frac) & (d.rH <= frac)].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        chosen, taken = [], {}
        for r in q.itertuples():
            if any(r.start < b and a < r.start + W for a, b in taken.get(r.chrom, ())):
                continue
            taken.setdefault(r.chrom, []).append((r.start, r.start + W))
            chosen.append(r)
            if len(chosen) == per_patient:
                break
        cells[guard(p)] = {"val": [[r.chrom, int(r.start)] for r in chosen], "test": []}
        union += [[r.chrom, int(r.start), int(r.start) + W] for r in chosen]
        print(f"hard f{fold} {res // 1000}kb {p}: {len(d)} scored windows, {len(q)} patient-specific, "
              f"{len(chosen)} chosen", flush=True)
    out = {"res": res, "window": W, "cells": cells, "union": union,
           "rule": {"score": "0.5*(1-r)+0.5*var(T-C)/var(T) against consensus12 and HSPC",
                    "top_fraction_each": frac, "per_patient": per_patient, "cnv_windows_excluded": True}}
    hard_split_path(fold, res).write_text(json.dumps(out, indent=1))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--stage", nargs="+", choices=["split", "refs", "hard"], required=True)
    ap_.add_argument("--fold", type=int, nargs="*", default=[0])
    ap_.add_argument("--res", type=int, nargs="*", default=[25000, 10000])
    a = ap_.parse_args()
    if "split" in a.stage:
        sp = split()
        print(pd.DataFrame(sp["patients"]).to_string(index=False))
        print(sp["folds"])
    if "refs" in a.stage:
        for f in a.fold:
            for r in a.res:
                build_refs(f, r)
    if "hard" in a.stage:
        for f in a.fold:
            for r in a.res:
                stage_hard(f, r)


if __name__ == "__main__":
    main()
