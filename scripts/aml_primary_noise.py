"""Noise ceiling of the patient-specific deviation (primary AML).

Each patient's pairs file is split into two halves by the parity of the last
character of the read name (the y coordinate of the cluster, a deterministic
function of the read name; header lines go to both), each half is built
exactly like the full map (5 kb cooler, coarsening to 10 and 25 kb, ICE
with the ENCODE parameters, oe_proper). On the regions the full-depth
analysis calls patient-specific (aml_primary_explain.stage_learn: top 20% of
the deviation score against both consensus12 and the HSPC consensus, hard
split coverage and masked-bin filters, SV/CNV regions excluded) and on the
patient's hardest 20% against consensus12 alone, the deviation of one half
from the reference is correlated with the deviation of the other half
(upper triangle, offset >= 2 bins, pixels measured in both). The half-split
correlation r_half estimates the reliability of one half; the full map has
r_full = 2 r_half / (1 + r_half) (Spearman-Brown), and a predictor that
knows the true signal exactly correlates with the full-depth deviation at
most at sqrt(r_full).

    python scripts/aml_primary_noise.py --stage build score
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aml_primary as ap                                          # noqa: E402
import aml_eval as ae                                             # noqa: E402

RES_LIST = [10000, 25000]
log = ap.log


def half_label(p: str, h: int) -> str:
    return f"{p}_h{h}"


def pairs_of(p: str) -> Path:
    r = ap.xu_pairs()
    return ap.PRIM / "pairs" / r[r.patient == p[3:]].name.iloc[0]


def build_half(p: str, h: int) -> None:
    label = half_label(p, h)
    if all(ap.oe_of(label, r).exists() for r in RES_LIST):
        return
    raw5 = ap.WORK / f"{label}_5kb.raw.cool"
    if not raw5.exists():
        tmp = ap.WORK / f"{label}_5kb.raw.tmp.cool"
        tmp.unlink(missing_ok=True)
        awk = ('{ if (substr($0,1,1)=="#") print; else { c=substr($1,length($1),1); '
               f'if ((c % 2)=={h}) print }} }}')
        log(f"[cload half] {label}")
        with open(ap.PRIM / "logs" / f"cload_{label}.log", "a") as lg:
            z = subprocess.Popen(["zcat", str(pairs_of(p))], stdout=subprocess.PIPE)
            f = subprocess.Popen(["awk", awk], stdin=z.stdout, stdout=subprocess.PIPE)
            z.stdout.close()
            c = subprocess.Popen([f"{ae.ENV}/cooler", "cload", "pairs", "-c1", "2", "-p1", "3",
                                  "-c2", "4", "-p2", "5", "--chunksize", "20000000",
                                  f"{ap.chromsizes()}:5000", "-", str(tmp)],
                                 stdin=f.stdout, stdout=lg, stderr=subprocess.STDOUT)
            f.stdout.close()
            rc = c.wait()
            z.wait(); f.wait()
        if rc or z.returncode or f.returncode:
            raise RuntimeError(f"cload {label} failed ({rc}, {z.returncode}, {f.returncode})")
        tmp.rename(raw5)
    for res in RES_LIST:
        k = res // 1000
        o = ap.oe_of(label, res)
        if o.exists():
            continue
        bal = ap.WORK / f"{label}_{k}kb.cool"
        if not bal.exists():
            tmp = ap.WORK / f"{label}_{k}kb.bal.tmp.cool"
            tmp.unlink(missing_ok=True)
            ae.sh(f"{ae.ENV}/cooler coarsen -k {res // 5000} -p 2 -o '{tmp}' '{raw5}'")
            ae.sh(f"{ae.ENV}/cooler balance --force -p 2 --ignore-diags 2 --max-iters 500 '{tmp}'",
                  ap.PRIM / "logs" / f"balance_{label}_{k}kb.log")
            tmp.rename(bal)
            ap.target_stats(label, res, bal)
        tmp = ap.OE_DIR / f"{label}_proper_{k}kb.tmp.cool"
        tmp.unlink(missing_ok=True)
        ae.sh(f"{ap.PY} scripts/oe_proper.py --input '{bal}' --output '{tmp}' --binsize {res}",
              ap.PRIM / "logs" / f"oe_{label}_{k}kb.log")
        tmp.rename(o)
        log(f"[oe ok] {o.name}")


def stage_build(workers: int) -> None:
    pats = ax_patients()
    jobs = [(p, h) for p in pats for h in (0, 1)]
    with ThreadPoolExecutor(workers) as ex:
        for f in [ex.submit(build_half, *j) for j in jobs]:
            try:
                f.result()
            except Exception as e:                       # noqa: BLE001
                log(f"[FAILED] {e}")


def ax_patients() -> list[str]:
    import aml_primary_explain as ax
    return ax.target_patients()


def dev_corr(a, b, ref, mask):
    from hard_split import corr
    return corr((a - ref)[mask], (b - ref)[mask])


def score_res(res: int) -> pd.DataFrame:
    import cooler
    import benchmark_metrics as bm
    import aml_primary_explain as ax
    from aml_eval import fetch_region
    k = res // 1000
    D = pd.read_csv(ap.WORK / "explain_per_region.csv.gz")
    D = D[D.res_kb == k]
    C = cooler.Cooler(str(ap.consensus12(res)))
    H = cooler.Cooler(str(ax.hspc_consensus_path(res)))
    rows = []
    for p in ax_patients():
        g = D[D.patient == p]
        g = g[(g.masked_frac <= 0.2) & ~g.affected.fillna(False)]
        g = g[g.coverage >= g.coverage.quantile(0.15)]
        both = set(g[(g.score_vs_C.rank(pct=True, ascending=False) <= 0.2)
                     & (g.score_vs_H.rank(pct=True, ascending=False) <= 0.2)].region)
        conly = set(g[g.score_vs_C.rank(pct=True, ascending=False) <= 0.2].region)
        c = [cooler.Cooler(str(ap.oe_of(half_label(p, h), res))) for h in (0, 1)]
        full = cooler.Cooler(str(ap.map_path(p, res)))
        for r in ap.region_list().itertuples():
            if r.region not in set(g.region):
                continue
            cache: dict = {}
            T0, T1, TF = (fetch_region(x, r.region, cache) for x in (*c, full))
            Cm, Hm = fetch_region(C, r.region, cache), fetch_region(H, r.region, cache)
            if any(x is None for x in (T0, T1, TF, Cm, Hm)) or T0.shape != Cm.shape:
                continue
            mask = (bm.triu_mask(T0.shape[0], 2) & np.isfinite(T0) & np.isfinite(T1)
                    & np.isfinite(TF) & np.isfinite(Cm) & np.isfinite(Hm))
            if mask.sum() < 100:
                continue
            rows.append({"patient": p, "res_kb": k, "region": r.region,
                         "hard_both": r.region in both, "hard_c": r.region in conly,
                         "r_half_C": dev_corr(T0, T1, Cm, mask),
                         "r_half_H": dev_corr(T0, T1, Hm, mask),
                         "r_half_plain": bm.pearson(T0, T1, mask),
                         "r_half_vs_full_C": dev_corr(T0, TF, Cm, mask),
                         "r_half1_vs_full_C": dev_corr(T1, TF, Cm, mask)})
    return pd.DataFrame(rows)


def sb(r):
    r = np.asarray(r, float)
    return 2 * r / (1 + r)


def stage_score() -> None:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(len(RES_LIST)) as ex:
        parts = list(ex.map(score_res, RES_LIST))
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(ap.WORK / "noise_per_region.csv.gz", index=False)
    out = []
    for (p, k), g in D.groupby(["patient", "res_kb"]):
        for name, gg, col in (("patient_specific_vs_C", g[g.hard_both], "r_half_C"),
                              ("patient_specific_vs_H", g[g.hard_both], "r_half_H"),
                              ("hardest20_vs_C", g[g.hard_c], "r_half_C"),
                              ("all_passing_vs_C", g, "r_half_C")):
            r = gg[col].median()
            out.append({"patient": p, "res_kb": k, "set": name, "n_regions": len(gg),
                        "r_half": r, "r_full_spearman_brown": float(sb(r)),
                        "ceiling_predictor_corr": float(np.sqrt(max(sb(r), 0)))})
    P = pd.DataFrame(out)
    P.to_csv(ap.RES_DIR / "noise_ceiling_patients.csv", index=False)
    S = P.groupby(["res_kb", "set"])[["n_regions", "r_half", "r_full_spearman_brown",
                                      "ceiling_predictor_corr"]].median().reset_index()
    S.to_csv(ap.RES_DIR / "noise_ceiling.csv", index=False)
    log("[noise] noise_ceiling.csv")


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--stage", nargs="+", required=True, choices=["build", "score"])
    a.add_argument("--workers", type=int, default=6)
    a = a.parse_args()
    if "build" in a.stage:
        stage_build(a.workers)
    if "score" in a.stage:
        stage_score()


if __name__ == "__main__":
    main()
