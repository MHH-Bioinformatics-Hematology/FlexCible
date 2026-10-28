"""Objects tables of the loop-centric AML recipe (registry section 63).

Per fold (10, 11, 12 of results/aml_train/split_v2.json) the AML-specific loops and stripes of
the TRAINING patients of the fold and their matched shared controls, with genomic coordinates,
for train.aml_object_* (flexcible/aml_objects.py).

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

Version 2 (registry section 70, `--refrand`): objects2_f<N>.csv.gz = the table above plus the control set
A_refrand (scripts/refrand_controls.py): per AML-specific object of a training patient a random object matched
on the reference enrichment (the leave-one-out reference the model reads), the HSPC enrichment, distance and
orientation, so that neither enrichment separates the controls from the AML-specific objects. Columns ref_val
and hspc_val hold the matching values (loops: log2 donut enrichment, 3 x 3 maximum; stripes: mean log O/E).
The shared-loop sets stay in the table for the weak term only. Leakage rules unchanged (training patients of
the fold only; the controls are drawn from the training patient's own chromosomes and map).

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
        "b_start", "b_end", "pair_type", "n_train_other", "ref_val", "hspc_val"]
WORK = SHARE / "work"


def table_path(fold: int, version: int = 1) -> Path:
    """Table the generated configs read (on the share, identical on both machines). Version 2 holds the
    A_refrand controls."""
    return SHARE / (f"objects_f{fold}.csv.gz" if version == 1 else f"objects{version}_f{fold}.csv.gz")


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
    d["ref_val"] = d["hspc_val"] = np.nan
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


def _refrand_patient(job):
    """A_refrand controls of one training patient of a fold (cached on the share)."""
    import pickle

    import aml_loop_classifier as lc
    fold, p, pos = job
    dest = WORK / f"refrand_f{fold}_{p}.pkl"
    if dest.exists():
        return p, pickle.load(open(dest, "rb"))
    ref = at.ref_path(fold, RES, p)
    hspc = lc.CONS / "hspccons_10kb.cool"
    valid = lc.CACHE / f"valid_{p}.npz"
    al = lc.positives_loops(p).rename(columns={"i": "i", "j": "j"})[["chrom", "i", "j"]]
    st = lc.read_tab("st4_aml_specific_stripes.csv", "stripe")
    st = st[st.patient == p]
    anchors = {c: sorted(set((g.start_anchor // RES).astype(int))) for c, g in st.groupby("chrom_anchor")}
    import refrand_controls as rc
    out = rc.refrand_controls(pos, str(ref), str(hspc), str(valid), al, anchors, seed=int(fold) * 1000 + int(p))
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(dest, "wb"))
    return p, out


def add_refrand(fold: int, d: pd.DataFrame, workers: int = 3) -> pd.DataFrame:
    """Table version 2: `d` (build_table output) plus the A_refrand controls and the matching values."""
    from concurrent.futures import ProcessPoolExecutor
    pos = d[d["set"] == "pos"].copy()
    for c in ("a", "b"):
        pos[c + "_lo"] = (pos[c + "_start"] // RES).astype(int)
        pos[c + "_hi"] = (pos[c + "_end"] // RES).astype(int)
    jobs = [(fold, p, g[["kind", "pair_id", "chrom", "a_lo", "a_hi", "b_lo", "b_hi"]].reset_index(drop=True))
            for p, g in pos.groupby("patient")]
    new, vals = [], []
    with ProcessPoolExecutor(workers) as ex:
        for p, (C, V) in ex.map(_refrand_patient, jobs):
            tp = pos[pos.patient == p].set_index(["kind", "pair_id"])
            if len(C):
                C = C.copy()
                key = list(zip(C.kind, C.pair_id))
                C["patient"], C["fold"], C["cell"] = p, fold, at.guard(p)
                C["role"], C["set"] = "ctrl", "A_refrand"
                C["pair_type"] = [tp.pair_type.get(k, "other") for k in key]
                C["a_start"], C["a_end"] = C.a_lo * RES, C.a_hi * RES
                C["b_start"], C["b_end"] = C.b_lo * RES, C.b_hi * RES
                C["n_train_other"] = np.nan
                new.append(C)
            V = V.assign(patient=p)
            vals.append(V)
    V = pd.concat(vals, ignore_index=True)
    vi = V.set_index(["patient", "kind", "pair_id"])
    m = d["set"] == "pos"
    key = pd.MultiIndex.from_arrays([d.patient[m], d.kind[m], d.pair_id[m]])
    vi = vi[~vi.index.duplicated()]
    d.loc[m, "ref_val"] = vi.ref_val.reindex(key).to_numpy()
    d.loc[m, "hspc_val"] = vi.hspc_val.reindex(key).to_numpy()
    out = pd.concat([d] + new, ignore_index=True) if new else d
    out = out.sort_values(["patient", "kind", "set", "pair_id"]).reset_index(drop=True)
    assert_no_leak(out, fold)
    return out[COLS]


def refrand_quality(d: pd.DataFrame, fold: int) -> list[dict]:
    """Match quality of A_refrand against the AML-specific objects that have a control."""
    from scipy.stats import rankdata
    rows = []
    for kind in ("loop", "stripe"):
        P = d[(d.kind == kind) & (d["set"] == "pos")]
        C = d[(d.kind == kind) & (d["set"] == "A_refrand")]
        m = P.merge(C, on=["patient", "pair_id"], suffixes=("_p", "_c"))
        rec = {"fold": fold, "kind": kind, "n_pos": len(P), "n_matched": len(m), "frac_matched": len(m) / max(1, len(P))}
        for v in ("ref_val", "hspc_val"):
            a, b = m[v + "_p"].to_numpy(float), m[v + "_c"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            rec[f"mean_pos_{v}"], rec[f"mean_ctl_{v}"] = a[ok].mean() if ok.any() else np.nan, b[ok].mean() if ok.any() else np.nan
            rec[f"max_abs_diff_{v}"] = np.abs(a[ok] - b[ok]).max() if ok.any() else np.nan
            if ok.sum() > 3:
                r = rankdata(np.concatenate([-a[ok], -b[ok]]))
                rec[f"auroc_{v}"] = (r[:ok.sum()].sum() - ok.sum() * (ok.sum() + 1) / 2) / ok.sum() ** 2
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", nargs="*", type=int, default=FOLDS)
    ap.add_argument("--no-share", action="store_true", help="do not write the copy on the share")
    ap.add_argument("--refrand", action="store_true", help="version 2 tables with the A_refrand controls")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    F = load_features()
    L = _tab("st1_aml_specific_loops.csv")
    S = _tab("st4_aml_specific_stripes.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, qrows = [], []
    for fold in a.folds:
        d = build_table(fold, F, L, S)
        ver = 1
        if a.refrand:
            ver = 2
            d = add_refrand(fold, d, a.workers)
            qrows.extend(refrand_quality(d, fold))
        d.to_csv(OUT / (f"objects_f{fold}.csv.gz" if ver == 1 else f"objects2_f{fold}.csv.gz"), index=False)
        if not a.no_share:
            SHARE.mkdir(parents=True, exist_ok=True)
            d.to_csv(table_path(fold, ver), index=False)
        for (kind, st), g in d.groupby(["kind", "set"]):
            rows.append({"fold": fold, "kind": kind, "set": st, "n": len(g), "n_patients": g.patient.nunique(),
                         "n_private_among_train": int((g.n_train_other == 0).sum()) if st == "pos" else np.nan})
        print(f"fold {fold}: training patients {train_donors(fold)}, {len(d)} objects "
              f"({int((d.role == 'pos').sum())} AML-specific)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / ("summary2.csv" if a.refrand else "summary.csv"), index=False)
    if qrows:
        pd.DataFrame(qrows).to_csv(OUT / "refrand_match_quality.csv", index=False, float_format="%.4f")
        print(pd.DataFrame(qrows).round(3).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
