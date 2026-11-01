"""AML patients: the measured map of every other AML patient and their mean map as the prediction
for a target patient, against the published predictors, on the same pixels (registry section 103).

Setting of registry 72 / 73: 10 kb object windows (192 bins) of the eight patients with CTCF
(018, 027, 168, 270, 472, 546, 629, 773), the saved predictions of C.Origami (quantile-matched
inputs), Epiphany (H3K4me3 = mean and = 0) and Akita (scripts/aml_competitors_loops.py, nothing
rerun), the measured O/E map of the target patient as the reference, the same metric definitions
(structure_scores_vs_akita.region_scores, unchanged). Added sources: the measured O/E map
(aml/oe/xu_<q>_proper_10kb.cool) of every other AML patient q with a 10 kb map (healthy HSPC samples
and the *_h0 / *_h1 haplotype files are not used), their nan-mean ("mean of the other patients"), and
consensus12 (12 cell lines, as in section 73). One pixel mask per window: upper triangle, >= 2 bins
off the diagonal, finite in the target and in every source. Per patient the median over windows, as in
section 73. Cost: 32 maps per window; a fixed random sample of 400 windows per patient (seed 0) is scored.

    python scripts/motivation_aml.py --workers 8 --limit 400
Output: results/motivation/aml_per_window.csv.gz, aml_per_patient.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import structure_scores_vs_akita as S  # noqa: E402

OUT = REPO / "results/motivation"
AML_OE = S.HM / "aml/oe"
COMP = {k: v for k, v in S.AML_SRC.items() if k in ("corigami_q", "epiphany_mean", "epiphany_zero", "akita")}
METRICS = ["pearson", "scc", "ins_r_50k", "ins_r_100k", "ins_r_250k", "bnd_f1", "bnd_recall",
           "bnd_precision", "moc", "stripe_r", "loop_enrich_corr", "compartment_corr",
           "eigenvector_abs_corr"]
_JOBS: list = []


def patients() -> list[str]:
    out = []
    for f in sorted(AML_OE.glob("xu_*_proper_10kb.cool")):
        m = re.fullmatch(r"xu_(\d+)_proper_10kb\.cool", f.name)
        if m:
            out.append(m.group(1))
    return out


def work(i):
    p, win, T, maps, mask = _JOBS[i]
    Tm = S.symmetrize(T, mask)
    rows = []
    for m, P in maps.items():
        r, _ = S.region_scores(Tm, S.symmetrize(P, mask), mask, 10000, hft=False)
        r = {k: r.get(k, np.nan) for k in METRICS}
        r.update(patient=p, window=win, method=m, n_pixels=int(mask.sum()))
        rows.append(r)
    return rows


def main() -> None:
    global _JOBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="random sample of windows per patient, seed 0 (0 = all)")
    a = ap.parse_args()
    import cooler
    import multicell_expanded as mx
    from flexcible.genome import match_name
    OUT.mkdir(parents=True, exist_ok=True)
    pats = patients()
    clr = {q: cooler.Cooler(str(AML_OE / f"xu_{q}_proper_10kb.cool")) for q in pats}
    c12 = cooler.Cooler(str(mx.consensus12(10000)))
    S.log(f"{len(pats)} AML patients with a 10 kb map: {pats}")

    def fetch(c, chrom, s, e):
        try:
            M = np.asarray(c.matrix(balance=False).fetch(f"{match_name(chrom, list(c.chromnames))}:{s}-{e}"), float)
        except Exception:                                      # noqa: BLE001
            return None
        if M.shape != (192, 192):
            return None
        M[M == 0] = np.nan
        return M

    rows_all, nwin = [], {}
    for p in S.AML_PATIENTS:
        files = {m: S.AML_COMP / rel.format(p=p) for m, (_, rel) in COMP.items()}
        if not all(f.exists() for f in files.values()):
            S.log(f"{p}: predictions missing")
            continue
        zs = {m: np.load(f) for m, f in files.items()}
        keys = set.intersection(*[set(z.files) for z in zs.values()])
        others = [q for q in pats if q != p]
        tasks, n_lost = [], 0
        wins = sorted(keys)
        if a.limit and len(wins) > a.limit:                   # fixed random sample of the windows
            rng = np.random.default_rng(0)
            wins = [wins[i] for i in sorted(rng.choice(len(wins), a.limit, replace=False))]
        for win in wins:
            chrom, se = win.split(":")
            s, e = map(int, se.split("-"))
            T = fetch(clr[p], chrom, s, e)
            C = fetch(c12, chrom, s, e)
            if T is None or C is None:
                continue
            maps = {"consensus12": C}
            for m in COMP:
                maps[m] = zs[m][win].astype(float)
            O = {}
            for q in others:
                M = fetch(clr[q], chrom, s, e)
                if M is not None:
                    O[q] = M
            if len(O) < len(others):                          # a patient without this window: not identical pixels
                n_lost += 1
                continue
            maps["mean_other_patients"] = np.nanmean(np.stack(list(O.values())), 0)
            for q, M in O.items():
                maps[f"patient_{q}"] = M
            mask = np.triu(np.ones((192, 192), bool), 2) & np.isfinite(T)
            for P in maps.values():
                mask &= np.isfinite(P)
            if mask.sum() < 500:
                n_lost += 1
                continue
            tasks.append((p, win, T.astype(np.float32), {k2: v.astype(np.float32) for k2, v in maps.items()}, mask))
        S.log(f"{p}: {len(tasks)} windows scored, {n_lost} dropped (of {len(keys)})")
        nwin[p] = (len(tasks), n_lost, len(keys))
        _JOBS = tasks
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(work, range(len(tasks)), chunksize=8)
        rows_all += [r for rs in res for r in rs]
        S.log(f"{p}: scored in {time.time() - t0:.0f} s")
        _JOBS = []
    df = pd.DataFrame(rows_all)
    df.to_csv(OUT / "aml_per_window.csv.gz", index=False, float_format="%.5f")
    pm = df.groupby(["patient", "method"])[METRICS].median().reset_index()
    pm["n_windows"] = df.groupby(["patient", "method"]).size().values
    pm.to_csv(OUT / "aml_per_patient.csv", index=False, float_format="%.4f")
    pd.DataFrame([dict(patient=p, windows_scored=v[0], windows_dropped=v[1], windows_total=v[2])
                  for p, v in nwin.items()]).to_csv(OUT / "aml_windows_count.csv", index=False)


if __name__ == "__main__":
    main()
