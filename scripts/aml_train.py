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

Folds 10, 11 and 12 (registry section 51) replace the folds 0 to 2 for new runs:
the "patients" 103 and 673 (and 576, without CUT&Tag) are healthy PBMC donors
(registry section 50), so they are excluded from the patient list, from the
training patients, from the AML reference and from every baseline built here.
The split of the ten AML patients with CUT&Tag is in
``results/aml_train/split_v2.json``; the old split (``split.json``, healthy donors
included, fold ids 0 to 2) stays reproducible with ``--stage split
--include-healthy`` (default ``--exclude-healthy``).

Healthy controls. The four CD34+ HSPC donors of the same study (same lab and
protocol; Hi-C only, no donor ChIP) are samples too: donors 050 and 213 train,
donors 341 and 822 are held out in every fold. They carry the marks of one
public CD34+ ChIP experiment per mark (ChIP-Atlas, aml/picks_hspc_cd34.tsv:
H3K27ac SRX1761289, H3K27me3 SRX1761288, CTCF SRX037935), quantile-matched
like the patients' tracks; the donor of those experiments is not one of the
Hi-C donors. Their reference is the mean of the training donors (leave-one-
out for a training donor), never the AML reference.

Laboratory indicator (``data.lab_classes: [xu]``, ``samples[].lab``): one
constant row, 1 for the Xu et al. Hi-C (patients and HSPC donors), 0 for every
other cell, MOLM-13 and U-937 included. The healthy HSPC separate the lab and
protocol component of a patient map from the AML component. The ablation run
``all12aml_refin_drop_nolab`` has the same data without the row.

Reference map of an AML sample (``samples[].reference_cool``): the mean proper
O/E of the training patients of the fold, leave-one-out for a training patient
and the mean of all eight for MOLM-13; the same mean serves the test patients
and U-937 at inference. The json next to each map lists its cells
(``amlxu<patient>``, flexcible.reference.cell_identity), so a test patient
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
HEALTHY = ["103", "673", "576"]          # healthy PBMC donors (registry section 50), never AML
NEW_FOLD_MIN = 10                        # fold ids >= 10: split_v2, healthy donors excluded
AML_PATIENTS = [p for p in PATIENTS if p not in HEALTHY]
# patients that are the same donor (registry section 54): they stay on one side of a split, and
# a baseline mean for a test patient excludes the patient's partners and counts a donor once.
# 496 and 260/424 have no CUT&Tag, so they are in no fold; they only matter for the Hi-C-only
# baseline means (168/496: two samples of one leukaemia, certain; 260/424: probable)
SAME_DONOR = [("168", "496"), ("260", "424")]
# test patients of the folds 10 to 12: TET2 once per fold, RUNX1 and KMT2B in two folds,
# CEBPA, STAG2 and FLT3-ITD only once, 3 to 4 test patients per fold
V2_FOLDS = {"10": ["018", "027", "270"], "11": ["168", "546", "629", "1360"],
            "12": ["472", "773", "1916"]}
SUBTYPE_TABLE = REPO / "results/aml_subtype/patient_subtype_table.csv"
ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
         "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
OVERSAMPLE = 3
HSPC_TRAIN = ["050", "213"]
HSPC_TEST = ["341", "822"]
HSPC_MARKS = INP / "hspc_cd34"


def guard(p: str) -> str:
    """Sample name and leakage-guard identity of a patient (no underscore)."""
    return "amlxu" + p


def label(p: str) -> str:
    return f"xu_{p}"


def hspc_label(d: str) -> str:
    return f"xu_HSPC-{d}"


def hspc_guard(d: str) -> str:
    return "amlxuhspc" + d


def hspc_ref_path(res: int, donor: str | None) -> Path:
    """Reference of a training HSPC donor (the other training donor) or, donor
    None, the mean of the training donors (test donors)."""
    tag = f"lo_{donor}" if donor else "all"
    return CONS_DIR / f"hspctrain_{tag}_{res // 1000}kb.cool"


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
            "hspc_train": HSPC_TRAIN, "hspc_test": HSPC_TEST,
            "note": "fold f is the test set of the run suffixed _f<f> (fold 0: no suffix); "
                    "the other folds train; MOLM-13 trains in every fold, U-937 WT is "
                    "held out in every fold"}


def donor_group(p: str) -> set:
    """p and its same-donor partners."""
    g = {p}
    for a, b in SAME_DONOR:
        if p in (a, b):
            g |= {a, b}
    return g


def baseline_labels(all_labels: list[str], test_patient: str | None) -> list[str]:
    """Maps of a baseline mean for a test patient: no healthy donor, no map of the test
    patient's donor, and the second sample of every other same-donor pair dropped (a donor
    counts once)."""
    excl = {label(x) for x in donor_group(test_patient)} if test_patient else set()
    second = {label(b) for _, b in SAME_DONOR}
    bad = {label(h) for h in HEALTHY}
    return [x for x in all_labels if x not in excl and x not in second and x not in bad]


def assert_same_donor_together(fold: int) -> None:
    test = set(test_patients(fold))
    train = set(train_patients(fold))
    for a, b in SAME_DONOR:
        assert not ({a, b} & test and {a, b} & train), f"{a}/{b} split across train and test in fold {fold}"


def make_split_v2() -> dict:
    """Folds 10 to 12: patient-disjoint folds over the AML patients with CUT&Tag, healthy
    donors excluded, subtype groups (consensus mutation group of Xu 2022 Fig. 1c) spread."""
    tab = pd.read_csv(SUBTYPE_TABLE, dtype={"patient": str}).set_index("patient")
    rows = []
    for p in AML_PATIENTS:
        t = json.loads((REPO / f"results/aml_primary/targets/xu_{p}_25kb.json").read_text())
        marks = sorted(b.stem for b in (INP / label(p)).glob("*.bigwig"))
        rows.append({"patient": p, "group": str(tab.loc[p, "consensus_group"]),
                     "contacts_M": round(t["contacts"] / 1e6, 1), "ctcf": "CTCF" in marks,
                     "marks": marks})
    test = [p for f in V2_FOLDS.values() for p in f]
    assert sorted(test) == sorted(AML_PATIENTS), "folds must partition the AML patients"
    assert not set(test) & set(HEALTHY)
    assert all(3 <= len(f) <= 4 for f in V2_FOLDS.values())
    for a, b in SAME_DONOR:
        side = {f for f, ps in V2_FOLDS.items() if a in ps or b in ps}
        assert len(side) <= 1, f"{a} and {b} are the same donor"
        assert not ({a, b} & set(HEALTHY))
    return {"patients": rows, "folds": V2_FOLDS, "healthy_excluded": HEALTHY,
            "hspc_train": HSPC_TRAIN, "hspc_test": HSPC_TEST,
            "note": "fold f (10 to 12) is the test set of the run suffixed _f<f>; the other folds' "
                    "AML patients train; healthy PBMC donors 103, 673, 576 are in no fold and in no "
                    "reference; HSPC donors 050 and 213 and MOLM-13 train in every fold, U-937 WT "
                    "and HSPC donors 341 and 822 are held out"}


def split(fold: int = 0) -> dict:
    """Split of a fold id: folds >= 10 from split_v2.json, older folds from split.json."""
    v2 = fold >= NEW_FOLD_MIN
    f = RES_OUT / ("split_v2.json" if v2 else "split.json")
    if not f.exists():
        RES_OUT.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(make_split_v2() if v2 else make_split(), indent=1))
    return json.loads(f.read_text())


def fold_of(base: str) -> int:
    """Fold of a run base name (all12aml_..., suffix _f1 / _f2)."""
    import re
    m = re.search(r"_f(\d+)$", base)
    return int(m.group(1)) if m else 0


def train_patients(fold: int) -> list[str]:
    sp = split(fold)
    test = set(sp["folds"][str(fold)])
    return [p for p in (AML_PATIENTS if fold >= NEW_FOLD_MIN else PATIENTS) if p not in test]


def test_patients(fold: int) -> list[str]:
    return split(fold)["folds"][str(fold)]


def ref_path(fold: int, res: int, patient: str | None) -> Path:
    """Reference of a training patient (leave-one-out) or, patient None, the
    mean of the fold's training patients."""
    tag = f"lo_{patient}" if patient else "all"
    return CONS_DIR / f"amltrain_f{fold}_{tag}_{res // 1000}kb.cool"


def clean_lopo(p: str | None, res: int) -> Path:
    """Baseline of folds >= 10: mean of the AML patients other than p and p's same-donor
    partners (patient p None: all AML patients), healthy donors excluded, each donor once."""
    tag = f"lopo_xu_{p}" if p else "all"
    return CONS_DIR / f"amlclean_{tag}_{res // 1000}kb.cool"


def build_clean_baselines(fold: int, res: int) -> None:
    import aml_primary as ap
    assert_same_donor_together(fold)
    allp = ap.aml_patients(res)
    for p in [None] + test_patients(fold):
        ap.build_aml_consensus(res, None, labs=baseline_labels(allp, p), out=clean_lopo(p, res))


def build_refs(fold: int, res: int) -> None:
    import aml_primary as ap
    if fold >= NEW_FOLD_MIN:
        build_clean_baselines(fold, res)
    tr = train_patients(fold)
    for p in [None] + tr:
        labs = [label(x) for x in tr if x != p]
        ap.build_aml_consensus(res, None, labs=labs, out=ref_path(fold, res, p))
    for d in [None] + HSPC_TRAIN:
        labs = [hspc_label(x) for x in HSPC_TRAIN if x != d]
        ap.build_aml_consensus(res, None, labs=labs, out=hspc_ref_path(res, d))


def refs_done(fold: int, res: int) -> bool:
    if fold >= NEW_FOLD_MIN and not all(clean_lopo(p, res).exists() for p in [None] + test_patients(fold)):
        return False
    return (all(ref_path(fold, res, p).exists() for p in [None] + train_patients(fold))
            and all(hspc_ref_path(res, d).exists() for d in [None] + HSPC_TRAIN))


def aml_samples(fold: int, res: int, with_ref: bool) -> list[dict]:
    """Sample dicts (config data.samples entries) of the AML training cells of
    a fold: the training patients and MOLM-13."""
    import multicell_expanded as mx
    out = []
    for p in train_patients(fold):
        have = {b.stem for b in (INP / label(p)).glob("*.bigwig")}
        s = {"name": guard(p), "cool_path": str(OE_DIR / f"xu_{p}_proper_{res // 1000}kb.cool"),
             "bigwig_dir": str(INP / label(p)), "missing_tracks": [t for t in ALL13 if t not in have],
             "assay": "hic", "lab": "xu"}
        if with_ref:
            s["reference_cool"] = str(ref_path(fold, res, p))
        out.append(s)
    have = {b.stem for b in HSPC_MARKS.glob("*.bigwig")}
    for dn in HSPC_TRAIN:
        s = {"name": hspc_guard(dn),
             "cool_path": str(OE_DIR / f"{hspc_label(dn)}_proper_{res // 1000}kb.cool"),
             "bigwig_dir": str(HSPC_MARKS), "missing_tracks": [t for t in ALL13 if t not in have],
             "assay": "hic", "lab": "xu"}
        if with_ref:
            s["reference_cool"] = str(hspc_ref_path(res, dn))
        out.append(s)
    d = AML / "hg38_molm13"
    s = {"name": "molm13", "cool_path": str(OE_DIR / f"molm13_proper_{res // 1000}kb.cool"),
         "bigwig_dir": str(d), "missing_tracks": mx.missing_marks(d), "assay": "hic"}
    if with_ref:
        s["reference_cool"] = str(ref_path(fold, res, None))
    out.append(s)
    return out


def patient_meth_dir(name: str) -> Path | None:
    """Folder of the Meth.bigwig of an AML sample (Xu WGBS built by
    scripts/aml_primary_meth.py), None when the sample has none."""
    if not name.startswith("amlxu") or name.startswith("amlxuhspc"):
        return None
    d = PRIM / "meth" / label(name[5:])
    return d if (d / "Meth.bigwig").exists() else None


def aml_names(fold: int) -> list[str]:
    return [guard(p) for p in train_patients(fold)] + [hspc_guard(d) for d in HSPC_TRAIN] + \
        ["molm13"]


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
    from flexcible.matrix import fetch_diagonal_window
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
    g = ap_.add_mutually_exclusive_group()
    g.add_argument("--exclude-healthy", dest="include_healthy", action="store_false",
                   help="stage split: print the split of the folds 10 to 12 (default)")
    g.add_argument("--include-healthy", dest="include_healthy", action="store_true",
                   help="stage split: print/verify the old split of the folds 0 to 2")
    ap_.add_argument("--res", type=int, nargs="*", default=[25000, 10000])
    a = ap_.parse_args()
    if "split" in a.stage:
        if a.include_healthy:
            old = make_split()
            if (RES_OUT / "split.json").exists():
                cur = json.loads((RES_OUT / "split.json").read_text())
                print("old split identical to split.json (patients, folds):",
                      all(old[k] == cur[k] for k in ("patients", "folds")))
            sp = old
        else:
            sp = split(NEW_FOLD_MIN)
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
