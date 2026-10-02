"""Cohesin recovery after auxin washout as a held-out perturbation test.

Rao et al. 2017 (HCT116 RAD21-mAC) measured Hi-C 20, 40, 60 and 180 min after
withdrawing auxin from 6 h treated cells (4DN 4DNESCOJ3ADI, 4DNES7RDRS69,
4DNES4KDIQNN, 4DNESOGIV7NG; GRCh38 mcools of the 4DN pipeline). No ChIP
exists at these time points: Kim et al. 2024 measured RAD21 and CTCF only
untreated, after 6 h auxin and at 9 h and 12 h washout. The washout maps are
therefore never trained on; the models predict them only approximately:

  RAD21 interpolated   every input from the untreated degron ChIP, except
                       RAD21 = (1 - f) * auxin + f * untreated in raw signal,
                       for f in 0, 0.25, 0.5, 0.75, 1 (f = 0: the 6 h auxin
                       RAD21 track; f = 1: the untreated inputs)
and compare the predicted loop strength at the loops of the untreated map
with the measured loop strength at each time point. Reported per model: the
measured recovery of loop strength (as a fraction of the auxin-to-untreated
difference) at 20, 40, 60, 180 min, the predicted recovery as a function of
f, and the f whose prediction matches each time point; this is an
approximation (the real RAD21 occupancy at each time point is unknown).

Stages: fetch (download, md5), oe (single-resolution copies at 10 and 25 kb,
oe_proper.py), predict, score.

    python scripts/washout.py --stage fetch oe
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multicell_expanded as mx                                      # noqa: E402

REPO, PY, HM = mx.REPO, mx.PY, mx.HM
DATA = HM / "perturb/hct116_rad21_washout"
OUT = HM / "runs/washout"
TIMES = {20: ("4DNFI2NUHMIM", "627bb8ab-3a86-4d99-859e-9cee8850e46c",
              "3035e14735e5d4369ff84357b89cf893"),
         40: ("4DNFINKNVML6", "25590bcb-a616-4ab1-bd45-d5346a973ff9",
              "e12a38e5c894251f190e86671c308997"),
         60: ("4DNFIACWCWO1", "7adb3ed0-7066-4148-9cfe-55ff13630fd4",
              "da72f2d00680e8d36c740c1454702145"),
         180: ("4DNFIBSLD9S9", "cafe7c52-40b1-4ccf-81b7-34b135c4ff5e",
               "69b02f9635ca80f38e5bc6fd95e361bf")}
URL = "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/{uuid}/{acc}.mcool"
RES = [25000, 10000]


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "washout.log", "a") as fh:
        fh.write(line + "\n")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def mcool(t: int) -> Path:
    return DATA / f"hct116_rad21mac_washout{t}min.mcool"


def oe_path(t: int, res: int) -> Path:
    return OUT / "oe" / f"washout{t}min_proper_{res // 1000}kb.cool"


def stage_fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for t, (acc, uuid, m) in TIMES.items():
        dst = mcool(t)
        if dst.exists():
            continue
        part = dst.with_name(dst.name + ".part")
        log(f"fetch {t} min {acc}")
        rc = subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "5", "-o", str(part),
                             URL.format(uuid=uuid, acc=acc)]).returncode
        if rc or md5(part) != m:
            raise SystemExit(f"download of {acc} failed (rc {rc}) or md5 mismatch")
        part.rename(dst)
        log(f"fetched {dst.name} md5 ok")


def stage_oe():
    for t in TIMES:
        for res in RES:
            out = oe_path(t, res)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part.cool")
            tmp.unlink(missing_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS="4")
            with open(OUT / "oe" / f"washout{t}min_{res // 1000}kb.log", "w") as fh:
                rc = subprocess.run([PY, "scripts/oe_proper.py", "--input", str(mcool(t)),
                                     "--output", str(tmp), "--binsize", str(res)],
                                    cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc:
                raise SystemExit(f"oe_proper failed for {t} min at {res}")
            tmp.rename(out)
            log(f"oe {t} min {res}")


F_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CHIP = Path.home() / "data/hic-multiome/perturb_chip"
FAMILIES = ["all12refres_drop", "all12honest_drop", "all12deg_refres_drop",
            "all12deg_honest_drop", "all12hffcoh_refres_drop", "all12refres", "all12honest",
            "all12cohunk_refres_drop", "all12cohunk_honest_drop"]


def _trained(fam, res):
    si = mx.stop_info(mx.run_name(fam, res), 10)
    return bool(si and si[2] == "patience")


def stage_predict():
    """Per model and resolution: HCT116 on the 413 Akita test regions from the
    untreated degron ChIP with RAD21 = (1 - f) * auxin + f * untreated (raw
    signal), for f in F_GRID; loop strength (max log O/E within +/- 1 bin) at
    the loops called in the untreated map, with the measured untreated, auxin
    and washout maps at the same pixels. One table per model."""
    import numpy as np
    import pandas as pd
    import torch
    import hard_split as hs
    import perturb_validate as pv
    import score_like_hicgan as sc
    from hicmultiome.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
    from hicmultiome.reference import reference_block
    from hicmultiome.tracks import TrackNormalizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    regs = pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    for res in RES:
        maps = {"meas_untreated": mx.target("hct116", res),
                "meas_auxin": pv.oe_path("rad21", "treated", res),
                **{f"meas_washout{t}": oe_path(t, res) for t in TIMES}}
        if not all(Path(m).exists() for m in maps.values()):
            log(f"predict {res}: measured maps missing, skipped")
            continue
        for fam in FAMILIES:
            if not _trained(fam, res):
                continue
            out = OUT / "pred" / f"{fam}_{res // 1000}kb.csv.gz"
            if out.exists():
                continue
            run = mx.REF_OUT / mx.run_name(fam, res)
            b = hs.load_bundle(run, device, "hct116", res)
            cfg = b["cfg"]
            tn = TrackNormalizer.load(b["ck"] / "norm_tracks.json")
            absent = tn.absent_values()
            tracks = b["channels"]["tracks"]
            cons = None
            if b["ref"]:
                import cooler
                cons = cooler.Cooler(str(mx.consensus_loo("hct116", res)))
            files = {}
            for cond in ("untreated", "auxin6h"):
                smp = cfg.get_sample("hct116") if "hct116" in [x.name for x in cfg.data.samples] \
                    else cfg.get_sample(None)
                smp = type(smp)(**{**smp.__dict__})
                smp.bigwig_dir = str(CHIP / cond)
                have = {q.stem.upper() for q in (CHIP / cond).glob("*.bigwig")}
                smp.missing_tracks = [t for t in tracks if t.upper() not in have]
                files[cond] = (smp, sample_track_files(cfg, smp))
            ri = tracks.index("RAD21")
            rows = []
            for chrom, grp in regs.groupby("chrom"):
                import cooler
                tgt = cooler.Cooler(str(maps["meas_untreated"]))
                nb = chrom_nbins_of(tgt, chrom)
                raw_u = assemble_chrom(cfg, files["untreated"][0], files["untreated"][1], chrom, nb)
                raw_a = assemble_chrom(cfg, files["auxin6h"][0], files["auxin6h"][1], chrom, nb)
                syms = {}
                for f in F_GRID:
                    raw = raw_u.copy()
                    raw[ri] = (1 - f) * raw_a[ri] + f * raw_u[ri]
                    sy = tn.apply(raw)
                    for t in files["untreated"][0].missing_tracks:
                        sy[tracks.index(t)] = absent[tracks.index(t)]
                    syms[f] = sy
                for r in grp.itertuples():
                    b0, b1 = int(r.start) // res, -(-int(r.end) // res)
                    n = b1 - b0
                    ij = pv.loop_anchors("hct116", chrom, b0, n, res)
                    if not len(ij):
                        continue
                    M = {k: pv._fetch(v, chrom, b0, b1) for k, v in maps.items()}
                    if any(v is None for v in M.values()):
                        continue
                    rec = {k: pv.strength(v, ij) for k, v in M.items()}
                    centre = (b0 + b1) // 2
                    st = max(0, min(nb - b["size"], centre - b["size"] // 2))
                    rb = reference_block(cons, chrom, st, b["size"], b["mat_norm"]) if cons else None
                    for f in F_GRID:
                        P = hs.predict_region(b, syms[f], st, rb, device, b0 * res, n, res, res)
                        rec[f"pred_f{f}"] = pv.strength(P, ij)
                    df = pd.DataFrame(rec)
                    df["region"], df["chrom"] = f"{chrom}:{r.start}-{r.end}", chrom
                    df["i"], df["j"] = ij[:, 0], ij[:, 1]
                    rows.append(df)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            pd.concat(rows).to_csv(tmp, index=False, compression="gzip")
            tmp.rename(out)
            log(f"predict {fam} {res}: {sum(len(x) for x in rows)} loops")


def stage_score():
    """Measured recovery at each time point, rec_t = median(S_t - S_aux) /
    median(S_untreated - S_aux) over the loops; predicted recovery for each f,
    rec_f = median(P_f - P_0) / median(P_1 - P_0); f matching each time point
    (linear interpolation of rec_f); the share of loops predicted weaker at
    f = 0 than at f = 1 (measured: auxin weaker than untreated); on all regions
    and on the even chromosomes."""
    import numpy as np
    import pandas as pd

    rows = []
    for f in sorted((OUT / "pred").glob("*.csv.gz")):
        fam, k = f.name.replace(".csv.gz", "").rsplit("_", 1)
        D = pd.read_csv(f)
        num = pd.to_numeric(D.chrom.str.replace("chr", "", regex=False), errors="coerce")
        for subset, d in (("all", D), ("even_chromosomes", D[num % 2 == 0])):
            d = d.dropna()
            if not len(d):
                continue
            base = np.median(d.meas_untreated - d.meas_auxin)
            pbase = np.median(d["pred_f1.0"] - d["pred_f0.0"])
            rec = {"model": fam, "res_kb": int(k.replace("kb", "")), "subset": subset,
                   "loops": len(d),
                   "meas_frac_weaker_auxin": float((d.meas_auxin < d.meas_untreated).mean()),
                   "pred_frac_weaker_rad21_auxin": float((d["pred_f0.0"] < d["pred_f1.0"]).mean()),
                   "meas_median_change_auxin": float(np.median(d.meas_auxin - d.meas_untreated)),
                   "pred_median_change_rad21_auxin": float(np.median(d["pred_f0.0"] - d["pred_f1.0"]))}
            recf = {fv: (np.median(d[f"pred_f{fv}"] - d["pred_f0.0"]) / pbase if pbase else np.nan)
                    for fv in F_GRID}
            for fv, v in recf.items():
                rec[f"pred_recovery_f{fv}"] = float(v)
            for t in TIMES:
                rt = float(np.median(d[f"meas_washout{t}"] - d.meas_auxin) / base) if base else np.nan
                rec[f"meas_recovery_{t}min"] = rt
                xs, ys = np.array(F_GRID), np.array([recf[x] for x in F_GRID])
                ok = np.isfinite(ys)
                rec[f"matching_f_{t}min"] = (float(np.interp(rt, ys[ok], xs[ok]))
                                            if ok.sum() > 1 and np.all(np.diff(ys[ok]) > 0) else np.nan)
            rows.append(rec)
    S = pd.DataFrame(rows).round(4)
    S.to_csv(OUT / "washout_summary.csv", index=False)
    (mx.REPO / "results/washout").mkdir(parents=True, exist_ok=True)
    S.to_csv(mx.REPO / "results/washout/washout_summary.csv", index=False)
    print(S.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", nargs="+", required=True)
    a = ap.parse_args()
    for st in a.stage:
        globals()[f"stage_{st}"]()


if __name__ == "__main__":
    main()
