"""Objects tables of the loop-centric AML recipe (registry section 63).

Per fold (10, 11, 12 of results/aml_train/split_v2.json) the AML-specific loops and stripes of
the TRAINING patients of the fold and their matched shared controls, with genomic coordinates,
for train.aml_object_* (hicmultiome/aml_objects.py).

Source: the objects and matched controls of the loop classifier (scripts/aml_loop_classifier.py,
results/aml_loop_classifier/features.csv.gz, 10 kb): AML-specific loops of Xu et al. 2022
Suppl. Table 1 and stripes of Suppl. Table 4 (set pos); shared loops / stripes matched 1:1 on
distance, orientation and marks (A_hspc, A_full, A_do) and random pairs (B_rand), the control
inheriting the pair_id of its positive within a patient. Objects wider than 150 bins are dropped as
in scripts/aml_loop_detection.py (they do not fit a 1.92 Mb window).

Leakage rules. Only the training patients of the fold contribute (aml_train.train_patients); a
patient of the same donor as a test patient (aml_train.SAME_DONOR, donor_group) is test-side and
excluded, so no object of a test patient or its partner is in the table. The recurrence column
n_train_other counts the OTHER training donors whose Suppl. Table 1 / 4 entry matches the object
(same rule as the detection analysis); test patients and their partners are not counted, so the
column carries no test information. The table is checked by assert_no_leak before it is written.

Output: results/aml_objects/objects_f<N>.csv.gz (+ the same file on the share, aml/objects/, which
the generated configs point at so both machines read it), results/aml_objects/summary.csv.

    python scripts/aml_objects_table.py [--folds 10 11 12]
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

import aml_train as at  # noqa: E402

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
FEATS = REPO / "results/aml_loop_classifier/features.csv.gz"
XT = REPO / "results/aml_primary/xu_tables"
OUT = REPO / "results/aml_objects"
SHARE = HM / "aml/objects"
RES = 10000                      # resolution of the classifier's object coordinates
MAX_SPAN = 150                   # bins; wider objects do not fit a window
FOLDS = [10, 11, 12]
COLS = ["fold", "patient", "cell", "kind", "role", "set", "pair_id", "chrom", "a_start", "a_end",
        "b_start", "b_end", "pair_type", "n_train_other"]


def table_path(fold: int) -> Path:
    """Table the generated configs read (on the share, identical on both machines)."""
    return SHARE / f"objects_f{fold}.csv.gz"


def _tab(name: str) -> pd.DataFrame:
    d = pd.read_csv(XT / name, dtype={"patient": str})
    d["patient"] = d.patient.str.lstrip("#")
    d["patient"] = d.patient.map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)
    return d


def load_features(path: Path = FEATS) -> pd.DataFrame:
    F = pd.read_csv(path, usecols=["patient", "kind", "set", "pair_id", "chrom", "a_lo", "a_hi",
                                   "b_lo", "b_hi", "pair_type"], dtype={"patient": str})
    F["patient"] = F.patient.map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)
    return F


def train_donors(fold: int) -> list[str]:
    """Training patients of the fold (donor-clean: none shares a donor with a test patient)."""
    at.assert_same_donor_together(fold)
    train = at.train_patients(fold)
    test_group = set()
    for t in at.test_patients(fold):
        test_group |= at.donor_group(t)
    return [p for p in train if p not in test_group]


def recurrence_train(pos: pd.DataFrame, train: list[str], L: pd.DataFrame, S: pd.DataFrame) -> np.ndarray:
    """Per positive object: the number of other TRAINING donors whose table entry matches
    (loops: both anchors within one bin; stripes: overlapping anchor and extent)."""
    Ld = {p: g for p, g in L[L.chrom1 == L.chrom2].groupby("patient")}
    Sd = {p: g for p, g in S[S.chrom_anchor == S.chrom_extent].groupby("patient")}
    n = np.zeros(len(pos), dtype=int)
    for k, r in enumerate(pos.itertuples()):
        seen = set()
        for q in train:
            if at.donor_group(q) & at.donor_group(r.patient):
                continue
            if r.kind == "loop":
                g = Ld.get(q)
                m = g is not None and ((g.chrom1 == r.chrom) & ((g.start1 // RES - r.a_lo).abs() <= 1)
                                       & ((g.start2 // RES - r.b_lo).abs() <= 1)).any()
            else:
                g = Sd.get(q)
                m = g is not None and ((g.chrom_anchor == r.chrom) & (g.start_anchor // RES < r.a_hi)
                                       & (-(-g.end_anchor // RES) > r.a_lo)
                                       & (g.start_extent // RES < r.b_hi)
                                       & (-(-g.end_extent // RES) > r.b_lo)).any()
            if m:
                seen.add(min(at.donor_group(q)))
        n[k] = len(seen)
    return n


def build_table(fold: int, F: pd.DataFrame, L: pd.DataFrame | None = None,
                S: pd.DataFrame | None = None) -> pd.DataFrame:
    train = train_donors(fold)
    d = F[F.patient.isin(train)].copy()
    span = np.maximum(d.a_hi, d.b_hi) - np.minimum(d.a_lo, d.b_lo)
    d = d[span <= MAX_SPAN].copy()
    d["fold"] = fold
    d["cell"] = d.patient.map(at.guard)
    d["role"] = np.where(d["set"] == "pos", "pos", "ctrl")
    d["a_start"], d["a_end"] = d.a_lo * RES, d.a_hi * RES
    d["b_start"], d["b_end"] = d.b_lo * RES, d.b_hi * RES
    d["n_train_other"] = np.nan
    if L is not None and S is not None:
        pm = d["set"] == "pos"
        d.loc[pm, "n_train_other"] = recurrence_train(d[pm], train, L, S)
    d = d.sort_values(["patient", "kind", "set", "pair_id"]).reset_index(drop=True)
    assert_no_leak(d, fold)
    return d[COLS]


def assert_no_leak(d: pd.DataFrame, fold: int) -> None:
    """No object of a test patient of the fold or of a patient of the same donor."""
    test = set(at.test_patients(fold))
    bad = set()
    for t in test:
        bad |= at.donor_group(t)
    got = set(d.patient)
    assert not got & bad, f"fold {fold}: objects of test-side patients {sorted(got & bad)}"
    assert got <= set(at.train_patients(fold)), f"fold {fold}: patients outside the training set"
    assert not set(d.cell) & {at.guard(p) for p in bad}, f"fold {fold}: test-side cells"
    assert set(d.role) <= {"pos", "ctrl"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", nargs="*", type=int, default=FOLDS)
    ap.add_argument("--no-share", action="store_true", help="do not write the copy on the share")
    a = ap.parse_args()
    F = load_features()
    L = _tab("st1_aml_specific_loops.csv")
    S = _tab("st4_aml_specific_stripes.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in a.folds:
        d = build_table(fold, F, L, S)
        d.to_csv(OUT / f"objects_f{fold}.csv.gz", index=False)
        if not a.no_share:
            SHARE.mkdir(parents=True, exist_ok=True)
            d.to_csv(table_path(fold), index=False)
        for (kind, st), g in d.groupby(["kind", "set"]):
            rows.append({"fold": fold, "kind": kind, "set": st, "n": len(g), "n_patients": g.patient.nunique(),
                         "n_private_among_train": int((g.n_train_other == 0).sum()) if st == "pos" else np.nan})
        print(f"fold {fold}: training patients {train_donors(fold)}, {len(d)} objects "
              f"({int((d.role == 'pos').sum())} AML-specific)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
