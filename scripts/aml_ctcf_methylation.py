"""CTCF binding and motif CpG methylation in the primary AML patients of Xu et al. 2022
(registry section 42). CPU only.

Inputs (all already in the repo data tree):
  * patient WGBS: GSE152099 processed bigWigs, CG methylation per 10 bp bin, no read
    coverage (aml/primary/wgbs/*.CGN.10bp.bw); a bin with an interval is covered, 0 is
    covered and unmethylated.
  * patient CUT&Tag (CTCF, H3K27ac, H3K27me3): MACS2 -log10 p-value bigWigs
    (aml/primary/cuttag). H3K4me3 was not measured in the patients.
  * the 72,626 CTCF motif sites (JASPAR MA0139.2, PWM >= 0.6 of the maximum) of
    scripts/chip_seq_meth_relationships.py (section 41) and its non-AML cell tables
    (ENCODE WGBS CpG beds with read counts, CD14+ monocytes among them).
  * healthy references: HSPC CTCF = CD34+ ChIP-seq SRX037935 (ChIP-Atlas, 9.8 M reads;
    no HSPC or PBMC WGBS exists in the repo or in the GEO deposit), healthy WGBS =
    ENCODE CD14+ monocytes (CpG-level, >= 10 reads).
  * Hi-C: patient and HSPC O/E maps (aml/oe/xu_*_proper_10kb.cool, ICE + oe_proper,
    the maps of the AML scoring scripts), 9 patients and 4 HSPC donors.

stages
  sites     per-site table per patient: motif/flank methylation (10 bp bins), CUT&Tag
            heights; HSPC CTCF height.            -> WORK/tables
  q1        partial Spearman of CTCF height with motif methylation given motif strength
            (+ flank, + H3K27ac), per patient and pooled, chromosome bootstrap, and the
            same statistics for the non-AML cells of section 41 for comparison.
  q2        sites that gain motif methylation against CD14+ monocytes and lose CTCF
            against HSPC: 2x2 tables, sensitivity and precision, coverage dependence;
            same-assay check against the other patients.
  q3        insulation at gain-and-lost sites, patient against HSPC, against matched
            control sites; paired difference with chromosome bootstrap.
  figures

    python scripts/aml_ctcf_methylation.py --stage sites q1 q2 q3 refcheck figures

healthy donors (registry section 50: xu_103, xu_576, xu_673 are healthy PBMC donors, not AML)
  --exclude-healthy (default)  the AML analysis is over the AML patients only; 103 (the only
                    healthy donor with CTCF CUT&Tag and WGBS) is reported as a separate
                    healthy-donor row and never enters a pooled number.
                    Output: results/aml_ctcf_methylation_aml_only/
  --include-healthy reproduces the original section 46 run (103 counted as a patient).
                    Output: results/aml_ctcf_methylation/ (pass --out to write elsewhere)
  Extra variants (AML-only mode), each labelled in the output tables:
    d103                 gain/loss against the healthy PBMC donor 103 (same lab, same assays):
                         methylation reference = 103 WGBS, CTCF reference = 103 CUT&Tag.
    d103ctcf_cd14meth    CTCF reference = 103 CUT&Tag, methylation reference = CD14+ monocytes.
    Hi-C baseline variants for the insulation test: HSPC (primary), healthy PBMC maps
    (576 = GSM4604271, 673, and 103 for the other patients), 576 alone.
  refcheck stage: how well each candidate reference predicts the patients (rank agreement of
  CTCF height and of motif methylation, insulation profile similarity).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
PRIM = HM / "aml" / "primary"
OE_DIR = HM / "aml" / "oe"
S41 = HM / "runs" / "chip_meth_rel" / "sites"
WORK = HM / "runs" / "aml_ctcf_meth"
OUT = REPO / "results" / "aml_ctcf_methylation"
OUT_AML_ONLY = REPO / "results" / "aml_ctcf_methylation_aml_only"
EXCLUDE_HEALTHY = True       # set from the command line
HEALTHY = {"103", "576", "673"}      # healthy PBMC donors of GSE152135 (registry section 50)
PBMC_HIC = ["576", "673", "103"]     # healthy PBMC Hi-C maps (GSM4604271 = 576, GSM4604272/73 = 103, 673)
FASTA = Path.home() / "data/hic-multiome/genome/hg38.fa"
HSPC_CTCF_BW = HM / "aml" / "bw" / "SRX037935.bw"
CHROMS = [f"chr{i}" for i in range(1, 23)]
EVEN = [c for c in CHROMS if int(c[3:]) % 2 == 0]
L = 15                       # motif length
FLANK = 250
SEED = 20261010
HSPC_DONORS = ["HSPC-050", "HSPC-213", "HSPC-341", "HSPC-822"]


def log(msg):
    WORK.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(WORK / "aml_ctcf_methylation.log", "a") as f:
        f.write(line + "\n")


def pid(s: str) -> str:
    return s.zfill(3)


def wgbs_files() -> dict[str, Path]:
    out = {}
    for p in sorted((PRIM / "wgbs").glob("*.CGN.10bp.bw")):
        out[pid(re.search(r"allc_(\d+)\.", p.name).group(1))] = p
    return out


def cuttag_files() -> dict[tuple[str, str], Path]:
    out = {}
    for p in sorted((PRIM / "cuttag").glob("*.bigwig")):
        m = re.search(r"_(\d+)_CUT_TAG_(CTCF|H3K27ac|H3K27me3)\.", p.name)
        out[(pid(m.group(1)), m.group(2))] = p
    return out


def all_with_both() -> list[str]:
    w, c = wgbs_files(), cuttag_files()
    return sorted(p for p in w if (p, "CTCF") in c)


def both_patients() -> list[str]:
    """Patients that enter the AML analysis and every pooled number."""
    a = all_with_both()
    return [p for p in a if p not in HEALTHY] if EXCLUDE_HEALTHY else a


def healthy_patients() -> list[str]:
    """Healthy donors with CTCF CUT&Tag and WGBS, reported as separate rows (empty with --include-healthy)."""
    return [p for p in all_with_both() if p in HEALTHY] if EXCLUDE_HEALTHY else []


def grp(p: str) -> str:
    return "healthy donor" if p in HEALTHY and EXCLUDE_HEALTHY else "AML"


# ------------------------------------------------------------------ base site table
def base_table() -> pd.DataFrame:
    f = WORK / "tables" / "base.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    from hicmultiome.sequence import read_chrom_codes
    parts = []
    for ch in CHROMS:
        z = np.load(S41 / f"motif_{ch}.npz")
        pos = z["pos"].astype(np.int64)
        codes = read_chrom_codes(str(FASTA), ch)
        cg = np.zeros((len(pos), L - 1), bool)
        for k in range(L - 1):
            cg[:, k] = (codes[pos + k] == 1) & (codes[pos + k + 1] == 2)
        mask = np.zeros(len(pos), np.int64)
        for k in range(L - 1):
            mask |= cg[:, k].astype(np.int64) << (L - 2 - k)
        t = pd.DataFrame({"chrom": ch, "pos": pos, "strand": z["strand"], "score": z["score"],
                          "frac": z["frac"], "cg_mask": mask})
        t["n_cpg_motif"] = cg.sum(1)
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(f, index=False)
    log(f"base table {len(t)} sites, {int((t.n_cpg_motif > 0).sum())} with a motif CpG")
    return t


def _stat(bw, ch, s, e, nb=None):
    try:
        if nb is None:
            v = bw.stats(ch, int(s), int(e), type="mean")[0]
            return np.nan if v is None else v
        return np.array([np.nan if x is None else x for x in bw.stats(ch, int(s), int(e), type="mean", nBins=nb)],
                        float)
    except RuntimeError:
        return np.nan if nb is None else np.full(nb, np.nan)


def wgbs_job(args):
    """Motif and flank methylation of every site from the 10 bp bins."""
    import pyBigWig
    p, path = args
    f = WORK / "tables" / f"wgbs_{p}.csv.gz"
    if f.exists():
        return p
    base = base_table()
    bw = pyBigWig.open(str(path))
    sizes = bw.chroms()
    mm = np.full(len(base), np.nan); nm = np.zeros(len(base), np.int16)
    mf = np.full(len(base), np.nan); nf = np.zeros(len(base), np.int16)
    for i, (ch, pos, mask) in enumerate(zip(base.chrom.to_numpy(), base.pos.to_numpy(), base.cg_mask.to_numpy())):
        if pos < FLANK + 20 or pos + L + FLANK + 20 > sizes[ch]:
            continue
        s0 = ((pos - FLANK) // 10) * 10
        e0 = ((pos + L + FLANK + 9) // 10) * 10
        nb = (e0 - s0) // 10
        v = _stat(bw, ch, s0, e0, nb)
        lo, hi = (pos - s0) // 10, (pos + L - 1 - s0) // 10
        span = np.zeros(nb, bool); span[lo:hi + 1] = True
        cp = np.zeros(nb, bool)
        for k in range(L - 1):
            if (int(mask) >> (L - 2 - k)) & 1:
                cp[(pos + k - s0) // 10] = True
                cp[(pos + k + 1 - s0) // 10] = True
        x = v[cp & np.isfinite(v)]
        if len(x):
            mm[i] = x.mean(); nm[i] = len(x)
        y = v[~span & np.isfinite(v)]
        if len(y):
            mf[i] = y.mean(); nf[i] = len(y)
    out = pd.DataFrame({"m_motif": mm, "n_motif_bins": nm, "m_flank": mf, "n_flank_bins": nf})
    out.to_csv(f, index=False, float_format="%.5f")
    log(f"wgbs {p}: {int(np.isfinite(mm).sum())} sites with a covered motif CpG bin")
    return p


def height_job(args):
    import pyBigWig
    key, path, half = args
    name = "_".join(key)
    f = WORK / "tables" / f"h_{name}.csv.gz"
    if f.exists():
        return name
    base = base_table()
    bw = pyBigWig.open(str(path))
    sizes = bw.chroms()
    h = np.zeros(len(base))
    for i, (ch, pos) in enumerate(zip(base.chrom.to_numpy(), base.pos.to_numpy())):
        c = pos + L // 2
        v = _stat(bw, ch, max(c - half, 0), min(c + half, sizes[ch]))
        h[i] = 0.0 if not np.isfinite(v) else v
    pd.DataFrame({"h": np.log1p(np.clip(h, 0, None))}).to_csv(f, index=False, float_format="%.5f")
    log(f"height {name}")
    return name


def stage_sites(workers):
    base_table()
    jobs_w = list(wgbs_files().items())
    jobs_h = []
    for (p, m), path in cuttag_files().items():
        jobs_h.append(((p, m), path, 100 if m == "CTCF" else 500))
    jobs_h.append((("hspc", "CTCF"), HSPC_CTCF_BW, 100))
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(wgbs_job, jobs_w))
        list(ex.map(height_job, jobs_h))


# ------------------------------------------------------------------ loaders
def load_patient(p: str) -> pd.DataFrame:
    t = pd.concat([base_table(), pd.read_csv(WORK / "tables" / f"wgbs_{p}.csv.gz")], axis=1)
    for m in ("CTCF", "H3K27ac", "H3K27me3"):
        f = WORK / "tables" / f"h_{p}_{m}.csv.gz"
        t["h_" + m] = pd.read_csv(f).h.to_numpy() if f.exists() else np.nan
    t["h_hspc_CTCF"] = pd.read_csv(WORK / "tables" / "h_hspc_CTCF.csv.gz").h.to_numpy()
    t["patient"] = p
    return t


def load_s41(cell: str) -> pd.DataFrame:
    return pd.read_csv(S41 / f"table_{cell}.csv.gz")


S41_CELLS = ["gm12878", "imr90", "h1", "hepg2", "a549", "cd14mono", "k562", "hmec"]
S41_NAMES = {"gm12878": "GM12878", "imr90": "IMR-90", "h1": "H1", "hepg2": "HepG2", "a549": "A549",
             "cd14mono": "CD14+ monocyte", "k562": "K562 (held out)", "hmec": "HMEC (held out)"}


# ------------------------------------------------------------------ q1
def pstat(d: pd.DataFrame, kind: str) -> float:
    import chip_seq_meth_relationships as csm
    h, m, fr = d.h.to_numpy(), d.m.to_numpy(), d.frac.to_numpy()
    if kind == "rho_strength":
        from scipy.stats import spearmanr
        return float(spearmanr(d.h, d.frac)[0])
    if kind == "rho_meth":
        from scipy.stats import spearmanr
        return float(spearmanr(d.h, d.m)[0])
    if kind == "partial_strength":
        return csm.partial_spearman(h, m, [fr])
    if kind == "partial_strength_flank":
        return csm.partial_spearman(h, m, [fr, d.flank.to_numpy()])
    if kind == "partial_strength_k27ac":
        return csm.partial_spearman(h, m, [fr, d.act.to_numpy()])
    if kind == "partial_strength_k27ac_flank":
        return csm.partial_spearman(h, m, [fr, d.act.to_numpy(), d.flank.to_numpy()])
    if kind == "partial_flank_given_strength":
        return csm.partial_spearman(h, d.flank.to_numpy(), [fr])
    raise KeyError(kind)


KINDS = ["rho_strength", "rho_meth", "partial_strength", "partial_strength_flank", "partial_strength_k27ac",
         "partial_strength_k27ac_flank", "partial_flank_given_strength"]
POOL_KINDS = ["partial_strength", "partial_strength_flank", "partial_strength_k27ac",
              "partial_strength_k27ac_flank", "partial_flank_given_strength", "rho_meth"]


def q1_frame(t: pd.DataFrame, hcol, mcol, fcol, acol):
    return pd.DataFrame({"chrom": t.chrom, "frac": t.frac, "h": t[hcol], "m": t[mcol], "flank": t[fcol],
                         "act": t[acol]})


def stage_q1():
    import chip_seq_meth_relationships as csm
    rng = np.random.default_rng(SEED)
    pats = both_patients()
    hp = healthy_patients()
    frames = {}
    for p in pats + hp:
        frames[p] = q1_frame(load_patient(p), "h_CTCF", "m_motif", "m_flank", "h_H3K27ac")
    for c in S41_CELLS:
        t = load_s41(c)
        d = q1_frame(t, "h_CTCF", "meth_motif", "meth_flank", "h_H3K27ac")
        d.loc[t.cov_motif < 10, "m"] = np.nan
        frames["s41_" + c] = d

    rows = []
    for key, d in frames.items():
        for scope in ("even", "all"):
            dd = d[d.chrom.isin(EVEN)] if scope == "even" else d
            sub = dd[dd.m.notna()].dropna(subset=["flank", "act"])
            if len(sub) < 100:
                continue
            for kind in KINDS:
                use = dd if kind == "rho_strength" else sub
                v = pstat(use, kind)
                lo, hi = csm.chrom_bootstrap(use, lambda x, k=kind: pstat(x, k), rng, B=300)
                rows.append({"sample": key, "group": "non-AML" if key.startswith("s41_") else grp(key),
                             "scope": scope, "statistic": kind, "value": v, "ci_lo": lo,
                             "ci_hi": hi, "n_sites": len(use), "n_chrom": use.chrom.nunique()})
        log(f"q1 {key}")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "q1_partial_spearman.csv", index=False, float_format="%.5f")
    R[(R.scope == "even") & (R.statistic == "partial_strength") & R["sample"].isin(pats + hp)].to_csv(
        OUT / "q1_per_patient_even.csv", index=False, float_format="%.5f")

    # pooled: the same chromosomes are resampled in every sample; median and mean over samples
    split = {}
    for key, d in frames.items():
        sub = d[d.m.notna()].dropna(subset=["flank", "act"])
        split[key] = {c: g for c, g in sub.groupby("chrom")}

    def val(keys, chs, kind):
        vs = []
        for k in keys:
            parts = [split[k][c] for c in chs if c in split[k]]
            if not parts:
                continue
            sub = pd.concat(parts)
            if len(sub) >= 100:
                vs.append(pstat(sub, kind))
        return float(np.median(vs)), float(np.mean(vs))

    prow = []
    groups = {f"AML all {len(pats)}": pats, "AML without 773": [p for p in pats if p != "773"],
              "non-AML 8 cells": ["s41_" + c for c in S41_CELLS]}
    for g, keys in groups.items():
        for scope in ("even", "all"):
            chs = EVEN if scope == "even" else CHROMS
            for kind in POOL_KINDS:
                a = val(keys, chs, kind)
                bs = np.array([val(keys, list(rng.choice(chs, len(chs), replace=True)), kind) for _ in range(300)])
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
                prow.append({"group": g, "scope": scope, "statistic": kind, "n_samples": len(keys),
                             "median": a[0], "median_lo": lo[0], "median_hi": hi[0],
                             "mean": a[1], "mean_lo": lo[1], "mean_hi": hi[1]})
        log(f"q1 pooled {g}")
    pd.DataFrame(prow).to_csv(OUT / "q1_pooled.csv", index=False, float_format="%.5f")

    # CTCF height by motif methylation level
    lv = []
    for key in pats + hp + ["s41_" + c for c in S41_CELLS]:
        d = frames[key]
        d = d[d.chrom.isin(EVEN)]
        z = (d.h - d.h.mean()) / d.h.std()
        for name, (a, b) in {"low": (0, 0.2), "mid": (0.2, 0.8), "high": (0.8, 1.01)}.items():
            s = d[(d.m >= a) & (d.m < b)]
            lv.append({"sample": key, "group": "non-AML" if key.startswith("s41_") else grp(key), "level": name, "n_sites": len(s), "median_log1p_height": s.h.median(),
                       "median_z": z[s.index].median()})
    pd.DataFrame(lv).to_csv(OUT / "q1_height_by_methylation_level.csv", index=False, float_format="%.4f")


# ------------------------------------------------------------------ q2
GAIN = 0.3
PRIMARY = (0.3, 0.8)          # drop in CTCF percentile rank, HSPC-bound percentile
REL_Q = 0.25                  # definition B: patient rank among the HSPC-bound sites in the lowest quartile


def pct(x: pd.Series) -> pd.Series:
    return x.rank(pct=True, method="average")


def ref_table() -> pd.DataFrame:
    """CD14+ monocyte reference columns on the common site order."""
    base = base_table()
    r = load_s41("cd14mono")
    assert (r.pos.to_numpy() == base.pos.to_numpy()).all() and (r.chrom.to_numpy() == base.chrom.to_numpy()).all()
    return pd.DataFrame({"ref_m": r.meth_motif.where(r.cov_motif >= 10), "ref_cov": r.cov_motif,
                         "ref_h_CTCF": r.h_CTCF})


def annotate(t: pd.DataFrame, R: pd.DataFrame, drop: float, bound: float, mode: str = "A") -> pd.DataFrame:
    """Definition A: the percentile rank of the patient's CTCF height (over all 72,626 sites) is at least
    `drop` below the HSPC percentile. Definition B: the patient height ranks in the lowest `drop` fraction
    among the HSPC-bound sites (fixed prevalence, removes the CUT&Tag depth and noise difference between
    patients). HSPC-bound: HSPC percentile >= `bound`."""
    t = pd.concat([t, R], axis=1)
    t["pct_pat"] = pct(t.h_CTCF)
    t["pct_hspc"] = pct(t.h_hspc_CTCF)
    t["bound_h"] = t.pct_hspc >= bound
    if mode == "A":
        t["lost"] = t.bound_h & ((t.pct_hspc - t.pct_pat) >= drop)
    else:
        rk = t.h_CTCF.where(t.bound_h).rank(pct=True, method="average")
        t["lost"] = t.bound_h & (rk <= drop)
    t["dm"] = t.m_motif - t.ref_m
    t["gain"] = t.dm >= GAIN
    return t


def tab(t: pd.DataFrame) -> dict:
    d = t[t.dm.notna() & t.bound_h]
    a = int((d.gain & d.lost).sum()); b = int((d.gain & ~d.lost).sum())
    c = int((~d.gain & d.lost).sum()); e = int((~d.gain & ~d.lost).sum())
    prec = a / (a + b) if a + b else np.nan
    sens = a / (a + c) if a + c else np.nan
    base = c / (c + e) if c + e else np.nan
    from scipy.stats import rankdata
    auc = np.nan
    if (a + c) and (b + e):
        r = rankdata(d.dm.to_numpy())
        auc = (r[d.lost.to_numpy()].sum() - (a + c) * (a + c + 1) / 2) / ((a + c) * (b + e))
    return {"auc_dm_lost": auc, "n_bound_testable": len(d), "gain_lost": a, "gain_kept": b, "nogain_lost": c, "nogain_kept": e,
            "precision": prec, "sensitivity": sens, "p_lost_no_gain": base,
            "rel_risk": prec / base if base and base > 0 else np.nan,
            "lost_total_bound": int(t[t.bound_h].lost.sum()), "lost_testable": a + c}


def boot_tab(t: pd.DataFrame, rng, B=300):
    chs = t.chrom.unique()
    gp = {c: t[t.chrom == c] for c in chs}
    vals = []
    for _ in range(B):
        r = tab(pd.concat([gp[c] for c in rng.choice(chs, len(chs), replace=True)]))
        vals.append([r["precision"], r["sensitivity"], r["rel_risk"]])
    return np.nanpercentile(np.array(vals), [2.5, 97.5], axis=0)


def rates(a, b, c, e):
    prec = a / max(a + b, 1); base = c / max(c + e, 1)
    return {"precision": prec, "sensitivity": a / max(a + c, 1), "p_lost_no_gain": base,
            "rel_risk": prec / base if base > 0 else np.nan}


def sites_file(variant: str) -> Path:
    tag = "aml_only" if EXCLUDE_HEALTHY else "with_healthy"
    return WORK / "tables" / f"q2_sites_{tag}_{variant}.csv.gz"


def q2_variants(allp: dict, t103: pd.DataFrame, R: pd.DataFrame):
    """Variant: the healthy PBMC donor 103 (same laboratory, same CUT&Tag and WGBS assays) as the healthy
    reference in place of CD14+ monocyte WGBS and CD34+ CTCF ChIP-seq. The CTCF reference percentile and the
    HSPC-bound set come from the 103 CUT&Tag height (column h_hspc_CTCF is overwritten with it, so annotate()
    and the q3 matching run unchanged); the methylation reference is the 103 motif methylation."""
    rng = np.random.default_rng(SEED + 1)
    PRIM = {"A": PRIMARY, "B": (REL_Q, PRIMARY[1])}
    R103 = pd.DataFrame({"ref_m": t103.m_motif, "ref_cov": t103.n_motif_bins, "ref_h_CTCF": t103.h_CTCF})
    variants = {"d103": R103, "d103ctcf_cd14meth": R}
    pats = list(allp)
    grid, pool_rows = [], []
    for vname, Rv in variants.items():
        tabs0 = {}
        for p, t0 in allp.items():
            t1 = t0.copy()
            t1["h_hspc_CTCF"] = t103.h_CTCF.to_numpy()
            tabs0[p] = t1
        for mode, drops in (("A", (0.2, 0.3, 0.4)), ("B", (0.15, 0.25, 0.35))):
            for p, t1 in tabs0.items():
                for drop in drops:
                    for bound in (0.7, 0.8, 0.9):
                        r = tab(annotate(t1, Rv, drop, bound, mode))
                        r.update({"variant": vname, "mode": mode, "patient": p, "drop": drop, "bound": bound})
                        grid.append(r)

        def pooled_vals(tabs, chs):
            acc = np.zeros(4, int)
            for t in tabs.values():
                r = tab(pd.concat([t[t.chrom == c] for c in chs]))
                acc += [r["gain_lost"], r["gain_kept"], r["nogain_lost"], r["nogain_kept"]]
            r = rates(*acc)
            return [r["precision"], r["sensitivity"], r["rel_risk"]], acc

        for (mode, name), keep in {("A", f"all {len(pats)}"): pats, ("A", "without 773"): [p for p in pats if p != "773"],
                                   ("B", f"all {len(pats)}"): pats, ("B", "without 773"): [p for p in pats if p != "773"]}.items():
            tabs = {p: annotate(tabs0[p], Rv, *PRIM[mode], mode) for p in keep}
            v0, acc = pooled_vals(tabs, CHROMS)
            bs = np.array([pooled_vals(tabs, list(rng.choice(CHROMS, len(CHROMS), replace=True)))[0] for _ in range(200)])
            q = np.nanpercentile(bs, [2.5, 97.5], axis=0)
            pool_rows.append({"variant": vname, "mode": mode, "group": name, "patients": len(keep),
                              "gain_lost": acc[0], "gain_kept": acc[1], "nogain_lost": acc[2], "nogain_kept": acc[3],
                              "precision": v0[0], "precision_lo": q[0, 0], "precision_hi": q[1, 0],
                              "sensitivity": v0[1], "sens_lo": q[0, 1], "sens_hi": q[1, 1],
                              "rel_risk": v0[2], "rr_lo": q[0, 2], "rr_hi": q[1, 2]})
            log(f"q2 variant {vname} pooled {mode} {name}")
        # per-patient primary rows with intervals
        for mode in "AB":
            for p in pats:
                t = annotate(tabs0[p], Rv, *PRIM[mode], mode)
                r = tab(t)
                q = boot_tab(t, rng)
                r.update({"variant": vname, "mode": mode, "patient": p, "precision_lo": q[0, 0],
                          "precision_hi": q[1, 0], "sens_lo": q[0, 1], "sens_hi": q[1, 1], "rr_lo": q[0, 2],
                          "rr_hi": q[1, 2]})
                grid.append({**r, "primary": True})
        if vname == "d103":
            lst = []
            for p in pats:
                t = annotate(tabs0[p], Rv, *PRIMARY, "A")
                tb = annotate(tabs0[p], Rv, *PRIM["B"], "B")
                t["lost_B"] = tb.lost
                t["gain_lost"] = t.gain & t.lost
                t["gain_lost_B"] = t.gain & t.lost_B
                t["group"] = "AML"
                lst.append(t[["patient", "group", "chrom", "pos", "frac", "score", "h_CTCF", "h_hspc_CTCF", "pct_pat",
                              "pct_hspc", "m_motif", "ref_m", "dm", "bound_h", "lost", "lost_B", "gain", "gain_lost",
                              "gain_lost_B", "n_cpg_motif", "n_motif_bins", "n_flank_bins"]])
            S = pd.concat(lst)
            S.to_csv(sites_file("d103"), index=False)
    G = pd.DataFrame(grid)
    G.to_csv(OUT / "q2_variant_d103_grid_and_primary.csv", index=False, float_format="%.4f")
    gp = []
    for (v, m, d, b_), g in G[G.primary.isna()].groupby(["variant", "mode", "drop", "bound"]):
        r = rates(*[int(g[k].sum()) for k in ("gain_lost", "gain_kept", "nogain_lost", "nogain_kept")])
        r.update({"variant": v, "mode": m, "drop": d, "bound": b_, "gain_lost": int(g.gain_lost.sum()),
                  "gain_kept": int(g.gain_kept.sum()), "nogain_lost": int(g.nogain_lost.sum()),
                  "nogain_kept": int(g.nogain_kept.sum())})
        gp.append(r)
    pd.DataFrame(gp).to_csv(OUT / "q2_variant_d103_grid_pooled.csv", index=False, float_format="%.4f")
    pd.DataFrame(pool_rows).to_csv(OUT / "q2_variant_d103_pooled.csv", index=False, float_format="%.4f")
    log("q2 variants done")


def stage_q2():
    rng = np.random.default_rng(SEED)
    R = ref_table()
    pats = both_patients()
    hp = healthy_patients()
    allp = {p: load_patient(p) for p in pats}
    allh = {p: load_patient(p) for p in hp}          # separate healthy-donor rows, never pooled
    both = {**allp, **allh}
    PRIM = {"A": PRIMARY, "B": (REL_Q, PRIMARY[1])}
    grid = []
    for mode, drops in (("A", (0.2, 0.3, 0.4)), ("B", (0.15, 0.25, 0.35))):
        for p, t0 in both.items():
            for drop in drops:
                for bound in (0.7, 0.8, 0.9):
                    t = annotate(t0, R, drop, bound, mode)
                    r = tab(t)
                    r.update({"mode": mode, "patient": p, "group": grp(p), "drop": drop, "bound": bound})
                    if (drop, bound) == PRIM[mode]:
                        q = boot_tab(t, rng)
                        r.update({"precision_lo": q[0, 0], "precision_hi": q[1, 0], "sens_lo": q[0, 1],
                                  "sens_hi": q[1, 1], "rr_lo": q[0, 2], "rr_hi": q[1, 2]})
                    grid.append(r)
    G = pd.DataFrame(grid)
    G.to_csv(OUT / "q2_2x2_grid.csv", index=False, float_format="%.4f")
    pm = pd.concat([G[(G["mode"] == m) & (G["drop"] == PRIM[m][0]) & (G.bound == PRIM[m][1])] for m in "AB"])
    pm.to_csv(OUT / "q2_2x2_primary.csv", index=False, float_format="%.4f")
    gp = []
    for (m, d, b), g in G[G.group == "AML"].groupby(["mode", "drop", "bound"]):
        r = rates(*[int(g[k].sum()) for k in ("gain_lost", "gain_kept", "nogain_lost", "nogain_kept")])
        r.update({"mode": m, "drop": d, "bound": b, "gain_lost": int(g.gain_lost.sum()),
                  "gain_kept": int(g.gain_kept.sum()), "nogain_lost": int(g.nogain_lost.sum()),
                  "nogain_kept": int(g.nogain_kept.sum()), "median_auc_dm_lost": g.auc_dm_lost.median()})
        gp.append(r)
    pd.DataFrame(gp).to_csv(OUT / "q2_2x2_grid_pooled.csv", index=False, float_format="%.4f")

    # pooled primary with a chromosome bootstrap resampling the same chromosomes in all patients
    def pooled_vals(tabs, chs):
        acc = np.zeros(4, int)
        for t in tabs.values():
            r = tab(pd.concat([t[t.chrom == c] for c in chs]))
            acc += [r["gain_lost"], r["gain_kept"], r["nogain_lost"], r["nogain_kept"]]
        r = rates(*acc)
        return [r["precision"], r["sensitivity"], r["rel_risk"]], acc

    pool_rows = []
    for (mode, name), keep in {("A", f"all {len(pats)}"): pats, ("A", "without 773"): [p for p in pats if p != "773"],
                               ("B", f"all {len(pats)}"): pats, ("B", "without 773"): [p for p in pats if p != "773"]}.items():
        tabs = {p: annotate(allp[p], R, *PRIM[mode], mode) for p in keep}
        v0, acc = pooled_vals(tabs, CHROMS)
        bs = np.array([pooled_vals(tabs, list(rng.choice(CHROMS, len(CHROMS), replace=True)))[0] for _ in range(200)])
        q = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        pool_rows.append({"mode": mode, "group": name, "patients": len(keep), "gain_lost": acc[0], "gain_kept": acc[1],
                          "nogain_lost": acc[2], "nogain_kept": acc[3], "precision": v0[0], "precision_lo": q[0, 0],
                          "precision_hi": q[1, 0], "sensitivity": v0[1], "sens_lo": q[0, 1], "sens_hi": q[1, 1],
                          "rel_risk": v0[2], "rr_lo": q[0, 2], "rr_hi": q[1, 2]})
        log(f"q2 pooled {mode} {name}")
    pd.DataFrame(pool_rows).to_csv(OUT / "q2_2x2_pooled.csv", index=False, float_format="%.4f")

    # how many lost sites (including those without any motif CpG) are explained
    ex = []
    for (mode, p), t0 in [((m, p), both[p]) for m in "AB" for p in pats + hp]:
        t = annotate(t0, R, *PRIM[mode], mode)
        lost = t[t.lost]
        ex.append({"mode": mode, "patient": p, "group": grp(p), "lost_sites": len(lost),
                   "lost_no_motif_cpg": int((lost.n_cpg_motif == 0).sum()),
                   "lost_cpg_patient_uncovered": int(((lost.n_cpg_motif > 0) & lost.m_motif.isna()).sum()),
                   "lost_cpg_ref_uncovered": int(((lost.n_cpg_motif > 0) & lost.m_motif.notna() & lost.ref_m.isna()).sum()),
                   "lost_testable": int(lost.dm.notna().sum()), "lost_testable_gain": int(lost.gain.sum()),
                   "gain_share_of_all_lost": float(lost.gain.sum() / max(len(lost), 1)),
                   "gain_share_of_testable_lost": float(lost.gain.sum() / max(lost.dm.notna().sum(), 1)),
                   "lost_motif_methylation_ge_0.5": int((lost.m_motif >= 0.5).sum())})
    pd.DataFrame(ex).to_csv(OUT / "q2_explained_lost_sites.csv", index=False, float_format="%.4f")

    # coverage dependence
    cov_rows = []
    for p, t0 in both.items():
        t = annotate(t0, R, *PRIMARY, "A")
        med = float(t.n_flank_bins[t.n_flank_bins > 0].median())
        for name, mask in {
            "ref reads >= 10 (primary)": t.ref_cov >= 10,
            "ref reads >= 20": t.ref_cov >= 20,
            "ref reads >= 40": t.ref_cov >= 40,
            "patient motif bins >= 2": t.n_motif_bins >= 2,
            "patient flank bins >= median": t.n_flank_bins >= med,
            "patient flank bins < median": (t.n_flank_bins > 0) & (t.n_flank_bins < med),
            "patient motif value neither 0 nor 1": t.m_motif.between(0.001, 0.999),
        }.items():
            r = tab(t[mask.fillna(False)])
            r.update({"patient": p, "group": grp(p), "filter": name})
            cov_rows.append(r)
    C = pd.DataFrame(cov_rows)
    C.to_csv(OUT / "q2_coverage_dependence.csv", index=False, float_format="%.4f")
    cs = []
    for name, g in C[C.group == "AML"].groupby("filter", sort=False):
        a, b, c, e = [int(g[k].sum()) for k in ("gain_lost", "gain_kept", "nogain_lost", "nogain_kept")]
        r = rates(a, b, c, e)
        r.update({"filter": name, "patients": g.patient.nunique(), "gain_lost": a, "gain_kept": b,
                  "nogain_lost": c, "nogain_kept": e})
        cs.append(r)
    pd.DataFrame(cs).to_csv(OUT / "q2_coverage_dependence_pooled.csv", index=False, float_format="%.4f")

    # same-assay check: patient against the median of the other patients (CUT&Tag and WGBS)
    same = []
    Hm = pd.DataFrame({p: pct(both[p].h_CTCF) for p in pats + hp})
    Mm = pd.DataFrame({p: both[p].m_motif for p in pats + hp})
    for p in pats + hp:
        others = [q for q in pats if q != p]          # healthy donors are never part of the AML median
        t = both[p].copy()
        t["pct_pat"] = Hm[p]
        t["pct_oth"] = Hm[others].median(axis=1)
        t["m_oth"] = Mm[others].median(axis=1)
        t["n_oth"] = Mm[others].notna().sum(axis=1)
        t["dm"] = t.m_motif - t.m_oth
        t["gain"] = t.dm >= GAIN
        t["bound_h"] = t.pct_oth >= 0.8
        t["lost"] = t.bound_h & ((t.pct_oth - t.pct_pat) >= 0.3)
        d = t[t.bound_h & t.dm.notna() & (t.n_oth >= 4)]
        a = int((d.gain & d.lost).sum()); b = int((d.gain & ~d.lost).sum())
        c = int((~d.gain & d.lost).sum()); e = int((~d.gain & ~d.lost).sum())
        r = rates(a, b, c, e)
        r.update({"patient": p, "group": grp(p), "n_testable": len(d), "gain_lost": a, "gain_kept": b, "nogain_lost": c,
                  "nogain_kept": e})
        same.append(r)
    Sm = pd.DataFrame(same)
    Sa = Sm[Sm.group == "AML"]
    tot = rates(*[int(Sa[k].sum()) for k in ("gain_lost", "gain_kept", "nogain_lost", "nogain_kept")])
    tot.update({"patient": "pooled", "group": "AML", "n_testable": int(Sa.n_testable.sum()),
                **{k: int(Sa[k].sum()) for k in ("gain_lost", "gain_kept", "nogain_lost", "nogain_kept")}})
    pd.concat([Sm, pd.DataFrame([tot])]).to_csv(OUT / "q2_same_assay_vs_other_patients.csv", index=False,
                                                float_format="%.4f")

    lst = []
    for p, t0 in both.items():
        t = annotate(t0, R, *PRIMARY, "A")
        tb = annotate(t0, R, *PRIM["B"], "B")
        t["lost_B"] = tb.lost
        t["gain_lost"] = t.gain & t.lost
        t["gain_lost_B"] = t.gain & t.lost_B
        t["group"] = grp(p)
        lst.append(t[["patient", "group", "chrom", "pos", "frac", "score", "h_CTCF", "h_hspc_CTCF", "pct_pat", "pct_hspc",
                      "m_motif", "ref_m", "dm", "bound_h", "lost", "lost_B", "gain", "gain_lost", "gain_lost_B",
                      "n_cpg_motif", "n_motif_bins", "n_flank_bins"]])
    S = pd.concat(lst)
    S.to_csv(sites_file("primary"), index=False)
    S[S.gain_lost | S.gain_lost_B].to_csv(OUT / "q2_gain_and_lost_sites.csv.gz", index=False, float_format="%.4f")
    log("q2 done")
    if EXCLUDE_HEALTHY and "103" in allh:
        q2_variants(allp, allh["103"], R)


# ------------------------------------------------------------------ q3
INS_W = 10          # insulation square of 10 x 10 bins (100 kb) at 10 kb; 5 x 5 as a check
MATCH_K = 5


def band_matrix(path: Path, chrom: str, dmax: int) -> np.ndarray:
    """O/E values as a band B[i, d] = M[i, i+d]; 0 (not measured) -> NaN."""
    import cooler
    c = cooler.Cooler(str(path))
    lo, hi = c.extent(chrom)
    m = c.matrix(balance=False, sparse=True).fetch(chrom).tocoo()
    r, cc, v = m.row, m.col, m.data
    keep = (cc >= r) & (cc - r <= dmax)
    B = np.full((hi - lo, dmax + 1), np.nan)
    B[r[keep], (cc - r)[keep]] = v[keep]
    B[B == 0] = np.nan
    return B


def ins_job(args):
    from hic_measures import insulation_band
    label, w = args
    f = WORK / "ins" / f"{label}_w{w}.npz"
    if f.exists():
        return label
    path = OE_DIR / f"xu_{label}_proper_10kb.cool"
    prof = {ch: insulation_band(band_matrix(path, ch, 2 * w), w) for ch in CHROMS}
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **prof)
    log(f"insulation {label} w={w}")
    return label


def hic_baseline(kind: str, p: str) -> list[str]:
    if kind == "hspc":
        return HSPC_DONORS
    if kind == "pbmc":                       # healthy PBMC maps; a donor never serves as its own baseline
        return [x for x in PBMC_HIC if x != p]
    if kind == "p576":                       # GSM4604271 alone
        return ["576"]
    raise KeyError(kind)


def stage_q3(workers):
    from scipy.spatial import cKDTree
    pats = both_patients()
    hp = healthy_patients()
    # (variant label, sites table, Hi-C baseline)
    variants = [("primary", "primary", "hspc")]
    if EXCLUDE_HEALTHY:
        variants += [("hic_pbmc", "primary", "pbmc"), ("hic_576", "primary", "p576")]
        if sites_file("d103").exists():
            variants += [("d103", "d103", "hspc"), ("d103_hic_pbmc", "d103", "pbmc")]
    labels = pats + hp + HSPC_DONORS + (PBMC_HIC if EXCLUDE_HEALTHY else [])
    labels = list(dict.fromkeys(labels))
    jobs = [(x, w) for x in labels for w in (INS_W, 5)]
    with ProcessPoolExecutor(min(workers, 4)) as ex:
        list(ex.map(ins_job, jobs))
    cnv = pd.read_csv(REPO / "results/aml_primary/explain_cnv_segments.csv")
    pairs = []
    for vname, sname, bkind in variants:
        S = pd.read_csv(sites_file(sname), dtype={"patient": str})
        for w in (INS_W, 5):
            prof = {x: np.load(WORK / "ins" / f"{x}_w{w}.npz") for x in labels}
            for p in pats + (hp if sname == "primary" else []):
                base = hic_baseline(bkind, p)
                t = S[S.patient == p].reset_index(drop=True)
                t["bin"] = (t.pos + L // 2) // 10_000
                ip = np.full(len(t), np.nan); ih = np.full(len(t), np.nan)
                for ch in CHROMS:
                    sel = np.where(t.chrom.to_numpy() == ch)[0]
                    b = t.bin.to_numpy()[sel]
                    n = len(prof[p][ch])
                    ok = b < n
                    ip[sel[ok]] = prof[p][ch][b[ok]]
                    ih[sel[ok]] = np.nanmean(np.stack([prof[h][ch][b[ok]] for h in base]), axis=0)
                t["ins_pat"], t["ins_hspc"] = ip, ih
                t["d_ins"] = t.ins_pat - t.ins_hspc          # > 0: more cross-boundary contact = weaker insulation
                for _, r in cnv[cnv.patient == "xu_" + p].iterrows():
                    t.loc[(t.chrom == r.chrom) & (t.pos >= r.start) & (t.pos < r.end), "d_ins"] = np.nan
                t = t.dropna(subset=["d_ins", "ins_hspc", "dm"])
                t = t.sort_values("h_hspc_CTCF", ascending=False).drop_duplicates(["chrom", "bin"])
                pool = t[(t.dm < 0.1) & t.bound_h]          # methylation covered, no gain, bound in the reference
                for mode, lc, glc in (("A", "lost", "gain_lost"), ("B", "lost_B", "gain_lost_B")):
                  for name, tst, ctl in (
                        (f"{mode}:gain_lost_vs_nogain", t[t[glc]], pool),
                        (f"{mode}:gain_lost_vs_nogain_kept", t[t[glc]], pool[~pool[lc]]),
                        (f"{mode}:lost_nogain_vs_nogain", t[t[lc] & (t.dm < 0.1)], pool[~pool[lc]]),
                        (f"{mode}:gain_kept_vs_nogain", t[t.gain & ~t[lc]], pool),
                        (f"{mode}:gain_lost_vs_gain_kept", t[t[glc]], t[t.gain & ~t[lc] & t.bound_h])):
                    if len(tst) == 0 or len(ctl) < MATCH_K:
                        continue
                    for sfx, scale in (("", None), ("_matched_hspc_ins", 0.05)):
                        X = [ctl.frac.to_numpy() / 0.02, ctl.h_hspc_CTCF.to_numpy() / 0.25]
                        Y = [tst.frac.to_numpy() / 0.02, tst.h_hspc_CTCF.to_numpy() / 0.25]
                        if scale:
                            X.append(ctl.ins_hspc.to_numpy() / scale); Y.append(tst.ins_hspc.to_numpy() / scale)
                        dist, nn = cKDTree(np.c_[tuple(X)]).query(np.c_[tuple(Y)], k=MATCH_K, p=np.inf)
                        ok = dist <= 1.0                     # caliper: |dPWM fraction| <= 0.02, |d log1p height| <= 0.25
                        dc = np.where(ok, ctl.d_ins.to_numpy()[np.minimum(nn, len(ctl) - 1)], np.nan)
                        keep = ok.sum(1) >= 1
                        if keep.sum() == 0:
                            continue
                        cm = np.nanmean(np.where(keep[:, None], dc, np.nan), axis=1)
                        tt = tst[keep].copy()
                        tt["ctrl_d_ins"] = cm[keep]
                        tt["paired_diff"] = tt.d_ins - tt.ctrl_d_ins
                        tt["n_ctrl"] = ok.sum(1)[keep]
                        tt["contrast"] = name + sfx
                        tt["w"] = w
                        tt["variant"] = vname
                        tt["group"] = grp(p)
                        tt["n_test_before_match"] = len(tst)
                        pairs.append(tt[["variant", "group", "patient", "chrom", "pos", "w", "contrast", "d_ins",
                                         "ctrl_d_ins", "paired_diff", "ins_pat", "ins_hspc", "frac", "h_hspc_CTCF",
                                         "dm", "n_ctrl", "n_test_before_match"]])
            log(f"q3 {vname} matched w={w}")
    P = pd.concat(pairs, ignore_index=True)
    P.to_csv(OUT / "q3_paired_sites.csv.gz", index=False, float_format="%.5f")

    summ = []
    for vname, _, _ in variants:
        rng = np.random.default_rng(SEED if vname == "primary" else SEED + 2)

        def boot(df, B):
            chs = df.chrom.unique()
            gp = {c: df.paired_diff.to_numpy()[(df.chrom == c).to_numpy()] for c in chs}
            vals = [np.mean(np.concatenate([gp[c] for c in rng.choice(chs, len(chs), replace=True)])) for _ in range(B)]
            return np.percentile(vals, [2.5, 97.5])

        Pv = P[P.variant == vname]
        for (w, con), g in Pv.groupby(["w", "contrast"]):
            ga = g[g.group == "AML"]
            for p, gg in list(ga.groupby("patient")) + [("pooled", ga)] + list(g[g.group != "AML"].groupby("patient")):
                if len(gg) < 5 and not (p != "pooled" and gg.group.iloc[0] != "AML"):
                    continue
                if len(gg) >= 5:
                    lo, hi = boot(gg, 2000 if p == "pooled" else 1000)
                else:
                    lo = hi = np.nan                      # healthy-donor row with too few sites for an interval
                summ.append({"variant": vname, "w": w, "contrast": con, "patient": p,
                             "group": "AML" if p == "pooled" else gg.group.iloc[0], "n_test_sites": len(gg),
                             "n_test_before_match": int(gg.n_test_before_match.iloc[0]) if p != "pooled"
                             else int(gg.groupby("patient").n_test_before_match.first().sum()),
                             "mean_d_ins_test": gg.d_ins.mean(), "mean_d_ins_ctrl": gg.ctrl_d_ins.mean(),
                             "paired_diff_mean": gg.paired_diff.mean(), "ci_lo": lo, "ci_hi": hi,
                             "paired_diff_median": gg.paired_diff.median(),
                             "frac_weaker": float((gg.paired_diff > 0).mean())})
    pd.DataFrame(summ).to_csv(OUT / "q3_insulation_summary.csv", index=False, float_format="%.5f")
    log("q3 done")


# ------------------------------------------------------------------ refcheck
def stage_refcheck():
    """Which healthy reference matches the patients better? Rank agreement of CTCF height and of motif
    methylation between each AML patient and each candidate reference, and the similarity of the genome-wide
    insulation profile to the HSPC and the healthy PBMC Hi-C baselines. Descriptive; even chromosomes for the
    site statistics, all autosomes for insulation."""
    from scipy.stats import spearmanr
    pats = both_patients()
    t103 = load_patient("103")
    R = ref_table()
    rows = []
    hs = pd.read_csv(WORK / "tables" / "h_hspc_CTCF.csv.gz").h.to_numpy()
    ev = base_table().chrom.isin(EVEN).to_numpy()
    cd = R.ref_m.to_numpy(); m103 = t103.m_motif.to_numpy(); h103 = t103.h_CTCF.to_numpy()
    cdh = R.ref_h_CTCF.to_numpy()
    # reference against reference
    def sp(a, b, mask):
        k = mask & np.isfinite(a) & np.isfinite(b)
        return float(spearmanr(a[k], b[k])[0]), int(k.sum())
    r, n = sp(h103, hs, ev); rows.append({"comparison": "CTCF height: 103 CUT&Tag vs HSPC CD34+ ChIP-seq", "rho": r, "n": n})
    r, n = sp(h103, cdh, ev); rows.append({"comparison": "CTCF height: 103 CUT&Tag vs CD14+ ChIP-seq (section 41)", "rho": r, "n": n})
    r, n = sp(m103, cd, ev); rows.append({"comparison": "motif methylation: 103 WGBS vs CD14+ WGBS", "rho": r, "n": n})
    k = ev & np.isfinite(m103) & np.isfinite(cd)
    rows.append({"comparison": "motif methylation: mean (103 minus CD14+), sites covered in both", "rho": float(np.mean(m103[k] - cd[k])), "n": int(k.sum())})
    rows.append({"comparison": "motif methylation: share |103 minus CD14+| >= 0.3", "rho": float(np.mean(np.abs(m103[k] - cd[k]) >= 0.3)), "n": int(k.sum())})
    pd.DataFrame(rows).to_csv(OUT / "refcheck_reference_vs_reference.csv", index=False, float_format="%.4f")
    # each AML patient against each candidate reference
    pr = []
    allm = pd.DataFrame({p: load_patient(p).m_motif for p in pats})
    allh = pd.DataFrame({p: load_patient(p).h_CTCF for p in pats})
    for p in pats:
        tp = load_patient(p)
        h, m = tp.h_CTCF.to_numpy(), tp.m_motif.to_numpy()
        others = [q for q in pats if q != p]
        ho = allh[others].median(axis=1).to_numpy(); mo = allm[others].median(axis=1).to_numpy()
        row = {"patient": p}
        for name, ref in (("HSPC CD34+ ChIP", hs), ("103 CUT&Tag", h103), ("median other AML", ho)):
            row["h_rho_" + name], row["h_n_" + name] = sp(h, ref, ev)
        for name, ref in (("CD14+ WGBS", cd), ("103 WGBS", m103), ("median other AML", mo)):
            row["m_rho_" + name], row["m_n_" + name] = sp(m, ref, ev)
        pr.append(row)
    PR = pd.DataFrame(pr)
    PR.to_csv(OUT / "refcheck_patient_vs_reference.csv", index=False, float_format="%.4f")
    # insulation profile similarity: Pearson over all bins with data
    ir = []
    for w in (INS_W, 5):
        prof = {x: np.load(WORK / "ins" / f"{x}_w{w}.npz") for x in pats + HSPC_DONORS + PBMC_HIC}
        def vec(label_or_list):
            ls = [label_or_list] if isinstance(label_or_list, str) else label_or_list
            return np.concatenate([np.nanmean(np.stack([prof[x][ch] for x in ls]), axis=0) for ch in CHROMS])
        hs_v = vec(HSPC_DONORS); pb_v = vec(PBMC_HIC); p576 = vec("576")
        def corr(a, b):
            k = np.isfinite(a) & np.isfinite(b)
            return float(np.corrcoef(a[k], b[k])[0, 1]), float(np.mean(np.abs(a[k] - b[k])))
        for p in pats:
            v = vec(p)
            vb = vec([x for x in PBMC_HIC if x != p]) if p in PBMC_HIC else pb_v
            row = {"w": w, "patient": p}
            for name, ref in (("HSPC mean of 4", hs_v), ("PBMC mean (576, 673, 103)", vb), ("576 (GSM4604271)", p576)):
                row["r_" + name], row["mad_" + name] = corr(v, ref)
            ir.append(row)
        row = {"w": w, "patient": "HSPC mean vs PBMC mean"}
        row["r_HSPC mean of 4"], row["mad_HSPC mean of 4"] = corr(hs_v, pb_v)
        ir.append(row)
    pd.DataFrame(ir).to_csv(OUT / "refcheck_insulation_profiles.csv", index=False, float_format="%.4f")
    log("refcheck done")


# ------------------------------------------------------------------ figures
def stage_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    pats = both_patients()
    hp = healthy_patients()
    if (OUT / "q1_partial_spearman.csv").exists():
        R = pd.read_csv(OUT / "q1_partial_spearman.csv")
        R = R[(R.scope == "even") & (R.statistic == "partial_strength")]
        fig, ax = plt.subplots(figsize=(7, 5))
        keys = ["s41_" + c for c in S41_CELLS] + pats + hp
        labels = [S41_NAMES[c] for c in S41_CELLS] + [f"AML {p}" for p in pats] + [f"healthy donor {p}" for p in hp]
        for i, k in enumerate(keys):
            r = R[R["sample"] == k]
            if r.empty:
                continue
            col = "#999999" if k.startswith("s41_") else ("#0072B2" if k in hp else "#D55E00")
            ax.errorbar(r.value, i, xerr=[r.value - r.ci_lo, r.ci_hi - r.value], fmt="o", color=col, ms=4, lw=1)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("partial Spearman: CTCF height vs motif CpG methylation given motif strength\n"
                      "(even chromosomes, chromosome bootstrap 95% interval)")
        ax.invert_yaxis()
        fig.tight_layout(); fig.savefig(OUT / "fig_q1_partial_spearman.png", dpi=170); plt.close(fig)
    if (OUT / "q2_2x2_primary.csv").exists():
        S = pd.read_csv(OUT / "q2_2x2_primary.csv")
        S = S[S["mode"] == "A"]
        S = pd.concat([S[S.group == "AML"], S[S.group != "AML"]]) if "group" in S else S
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
        x = np.arange(len(S))
        axs[0].bar(x - 0.2, S.precision, 0.4, label="P(CTCF lost | methylation gain)", color="#D55E00")
        axs[0].bar(x + 0.2, S.p_lost_no_gain, 0.4, label="P(CTCF lost | no gain)", color="#999999")
        axs[0].set_xticks(x); axs[0].set_xticklabels(S.patient, rotation=90); axs[0].legend(frameon=False, fontsize=7)
        axs[0].set_ylabel("fraction of HSPC-bound testable sites")
        axs[1].bar(x, S.sensitivity, color="#0072B2")
        axs[1].set_xticks(x); axs[1].set_xticklabels(S.patient, rotation=90)
        axs[1].set_ylabel("P(methylation gain | CTCF lost), testable sites")
        fig.tight_layout(); fig.savefig(OUT / "fig_q2_precision_sensitivity.png", dpi=170); plt.close(fig)
    if (OUT / "q3_insulation_summary.csv").exists():
        Q = pd.read_csv(OUT / "q3_insulation_summary.csv")
        cons = ["A:gain_lost_vs_nogain", "A:gain_lost_vs_nogain_kept", "A:lost_nogain_vs_nogain", "A:gain_kept_vs_nogain",
                "A:gain_lost_vs_gain_kept", "B:gain_lost_vs_nogain", "B:gain_lost_vs_gain_kept"]
        Q = Q[(Q.w == INS_W) & Q.contrast.isin(cons) & (Q.variant == "primary")]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        cols = ["#D55E00", "#E69F00", "#0072B2", "#999999", "#CC79A7", "#009E73", "#56B4E9"]
        pl = sorted(Q.patient.unique(), key=lambda s: (s == "pooled", s))
        for j, c in enumerate(cons):
            q = Q[Q.contrast == c].set_index("patient")
            for i, p in enumerate(pl):
                if p not in q.index:
                    continue
                r = q.loc[p]
                ax.errorbar(r.paired_diff_mean, i + 0.15 * j, xerr=[[r.paired_diff_mean - r.ci_lo],
                            [r.ci_hi - r.paired_diff_mean]], fmt="o", color=cols[j], ms=3, lw=1,
                            label=c if i == 0 else None)
        ax.set_yticks(range(len(pl))); ax.set_yticklabels(pl, fontsize=7); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("(patient - HSPC) insulation, site minus matched controls; > 0 = weaker insulation")
        ax.legend(frameon=False, fontsize=6); ax.invert_yaxis()
        fig.tight_layout(); fig.savefig(OUT / "fig_q3_insulation.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True, choices=["sites", "q1", "q2", "q3", "refcheck", "figures"])
    ap.add_argument("--workers", type=int, default=4)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--exclude-healthy", dest="exclude_healthy", action="store_true", default=True,
                   help="AML patients only; the healthy donor 103 as a separate row (default)")
    g.add_argument("--include-healthy", dest="exclude_healthy", action="store_false",
                   help="original section 46 behaviour: 103 counted as a patient")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default depends on the mode)")
    a = ap.parse_args()
    global EXCLUDE_HEALTHY, OUT
    EXCLUDE_HEALTHY = a.exclude_healthy
    OUT = a.out if a.out else (OUT_AML_ONLY if EXCLUDE_HEALTHY else OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if "sites" in a.stage:
        stage_sites(a.workers)
    if "q1" in a.stage:
        stage_q1()
    if "q2" in a.stage:
        stage_q2()
    if "q3" in a.stage:
        stage_q3(a.workers)
    if "refcheck" in a.stage:
        stage_refcheck()
    if "figures" in a.stage:
        stage_figures()


if __name__ == "__main__":
    main()
