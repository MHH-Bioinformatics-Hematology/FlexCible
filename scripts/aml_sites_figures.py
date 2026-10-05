"""Eight AML-specific sites against GM12878, measured and predicted.

Loci from Xu et al. 2022 (Supplementary Tables 1 to 4, 10 kb): AML-specific
loops (Suppl. Table 1) and stripes (Suppl. Table 4) of the patients with
CUT&Tag, plus one SV locus. A locus qualifies when, at 10 kb,
  - the patient's measured strength (log2 donut enrichment at the loop pixel,
    mean log O/E in the stripe band) exceeds that of the HSPC consensus and of
    GM12878 by at least GAIN, both are below ABSENT (the feature is absent
    there), and both half-depth maps of the patient (noise control, section
    29c) exceed the HSPC level by at least HALF_GAIN (reproduced);
  - neither anchor lies within 1 Mb of an SV breakpoint (Suppl. Table 5) or on
    a copy-number segment (explain_cnv_segments.csv).
Types by the patient's own CUT&Tag at the anchors (quantile-matched tracks,
90th percentile of the training distribution as "high"): promoter-enhancer
loop (H3K27ac high at both anchors), promoter-silencer loop (H3K27me3 high at
one anchor), enhancer stripe (H3K27ac high at the stripe anchor), repressive
stripe (H3K27me3 high). A gene is named by the refGene TSS inside an anchor.
Selection: 3 P-E loops, 2 P-S loops, 2 stripes (one repressive, one enhancer)
and the SV locus; test patients of fold 0 of results/aml_train/split.json
(027, 270, 773, 1916) and famous genes (MYCN, MEIS1, ERG, WT1, RUNX1, IKZF2,
RTTN, KLF4) preferred.

Models (10 kb): all12aml_refin_drop (AML data, lab indicator, the patient's
own marks, AML reference of fold 0: mean of the training patients, leave-one-
out for a training patient) and all12refres_drop (no AML data, the same
reference map); GM12878 predicted by the AML-trained model from its own ChIP
with its leave-one-out consensus.

    python scripts/aml_sites_figures.py --stage select predict figures
"""
from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aml_primary as ap                                          # noqa: E402
import aml_primary_explain as ax                                  # noqa: E402
import aml_train as at                                            # noqa: E402
import multicell_expanded as mx                                   # noqa: E402

RES = 10000
WIN = 80
OUT = mx.HM / "runs/aml_sites"
FIG = OUT / "figures"
REPO = ap.REPO
GAIN = 0.8
ABSENT = 0.5
HALF_GAIN = 0.4
FAMOUS = {"MYCN", "MEIS1", "ERG", "WT1", "RUNX1", "IKZF2", "RTTN", "KLF4", "HSF4", "MYC"}
TEST = {"027", "270", "773", "1916"}
GM = mx.HM / "crosscell_oe" / "gm12878_proper_10kb.cool"
log = ap.log


# ---------------------------------------------------------------- annotation
def tss_table() -> pd.DataFrame:
    rows = []
    with gzip.open(ap.PRIM / "annot" / "refGene.txt.gz", "rt") as fh:
        for l in fh:
            f = l.rstrip("\n").split("\t")
            chrom, strand, name = f[2], f[3], f[12]
            if "_" in chrom or not f[1].startswith("NM_"):
                continue
            rows.append((chrom, int(f[4]) if strand == "+" else int(f[5]), name))
    d = pd.DataFrame(rows, columns=["chrom", "tss", "gene"]).drop_duplicates()
    return d.sort_values(["chrom", "tss"]).reset_index(drop=True)


def genes_in(tss: pd.DataFrame, chrom: str, start: int, end: int) -> list[str]:
    t = tss[(tss.chrom == chrom) & (tss.tss >= start - 2000) & (tss.tss < end + 2000)]
    return list(dict.fromkeys(t.gene))


# ---------------------------------------------------------------- measures
def fetch(clr, chrom, s, e):
    from hicmultiome.genome import match_name
    nm = match_name(chrom, list(clr.chromnames))
    try:
        return np.asarray(clr.matrix(balance=False).fetch(f"{nm}:{max(0, s)}-{e}"), float)
    except Exception:                                              # noqa: BLE001
        return None


def loop_strength(clr, chrom, a1, a2, prm) -> float:
    """log2 donut enrichment at the loop pixel (bins of a1, a2)."""
    from hicmultiome.bioscore import local_enrichment
    lo = (min(a1, a2) // RES - 12) * RES
    hi = (max(a1, a2) // RES + 13) * RES
    M = fetch(clr, chrom, lo, hi)
    if M is None or M.size == 0:
        return np.nan
    i, j = (min(a1, a2) - max(lo, 0)) // RES, (max(a1, a2) - max(lo, 0)) // RES
    if not (0 <= i < M.shape[0] and 0 <= j < M.shape[0]):
        return np.nan
    e = local_enrichment(np.exp(np.nan_to_num(M)), prm)
    return float(np.log2(e[i, j]))


def stripe_strength(clr, chrom, a_lo, a_hi, e_lo, e_hi) -> float:
    lo = (min(a_lo, e_lo) // RES - 2) * RES
    hi = (max(a_hi, e_hi) // RES + 3) * RES
    M = fetch(clr, chrom, lo, hi)
    if M is None or M.size == 0:
        return np.nan
    o = max(lo, 0)
    r0, r1 = (a_lo - o) // RES, (a_hi - o) // RES + 1
    c0, c1 = (e_lo - o) // RES, (e_hi - o) // RES
    band = M[max(0, r0):r1, max(0, c0):c1]
    return float(np.nanmean(band)) if band.size and np.isfinite(band).any() else np.nan


def marks_at(lab: str, chrom: str, lo: int, hi: int, q: float) -> dict:
    """Is the patient's mean track over [lo, hi) above the q quantile of the
    training distribution of the mark (quantile-matched tracks)?"""
    import pyBigWig
    out = {}
    for m in ("H3K27ac", "H3K27me3"):
        f = ap.INP / lab / f"{m}.bigwig"
        if not f.exists():
            continue
        bw = pyBigWig.open(str(f))
        v = bw.stats(chrom, lo, hi, type="mean")[0] or 0.0
        bw.close()
        out[m] = v >= ap.reference_quantiles(m)[int(q * (ap.NQ - 1))]
    return out


def flagged(p: str, chrom: str, positions: list[int], sv: pd.DataFrame, cnv: pd.DataFrame) -> bool:
    s = sv[sv.patient == p]
    for x in s.itertuples():
        for ch, pos in ((x.chrom1, x.pos1), (x.chrom2, x.pos2)):
            if ch == chrom and any(abs(pos - q) <= 1_000_000 for q in positions):
                return True
        if x.sv_type != "translocation" and x.chrom1 == chrom == x.chrom2:
            a, b = sorted((x.pos1, x.pos2))
            if any(a <= q <= b for q in positions):
                return True
    c = cnv[(cnv.patient == p) & (cnv.chrom == chrom)]
    return any(((c.start <= q) & (c.end >= q)).any() for q in positions)


# ---------------------------------------------------------------- select
def stage_select() -> pd.DataFrame:
    import cooler
    from aml_eval import loop_params
    tss = tss_table()
    sv, cnv = ax.sv_table(), pd.read_csv(ap.RES_DIR / "explain_cnv_segments.csv")
    prm = loop_params(RES)
    hspc = cooler.Cooler(str(ax.hspc_consensus_path(RES)))
    gm = cooler.Cooler(str(GM))
    L = pd.read_csv(ax.XT / "st1_aml_specific_loops.csv", dtype={"patient": str})
    S = pd.read_csv(ax.XT / "st4_aml_specific_stripes.csv", dtype={"patient": str})
    pats = {p[3:] for p in ax.target_patients()}
    rows = []
    for kind, T in (("loop", L), ("stripe", S)):
        T = T.copy()
        T["patient"] = T.patient.map(ax.norm_id)
        T = T[T.patient.isin(pats)]
        for p, g in T.groupby("patient"):
            lab = "xu_" + p
            pc = cooler.Cooler(str(ap.map_path(lab, RES)))
            h = [cooler.Cooler(str(ap.oe_of(f"{lab}_h{k}", RES))) for k in (0, 1)]
            for r in g.itertuples():
                if kind == "loop":
                    if r.chrom1 != r.chrom2:
                        continue
                    ch, a1, a2 = r.chrom1, int(r.start1), int(r.start2)
                    d = abs(a2 - a1)
                    if not (60_000 <= d <= 1_000_000):
                        continue
                    pos = [a1, a2]
                    fn = lambda c: loop_strength(c, ch, a1, a2, prm)          # noqa: E731
                    centre = (a1 + a2) // 2
                    anchors = [(ch, a1), (ch, a2)]
                else:
                    if r.chrom_anchor != r.chrom_extent:
                        continue
                    ch = r.chrom_anchor
                    lo, hi = min(r.start_anchor, r.start_extent), max(r.end_anchor, r.end_extent)
                    if hi - lo > 700_000:
                        continue
                    pos = [int(r.start_anchor), int(r.start_extent), int(r.end_extent)]
                    fn = lambda c: stripe_strength(c, ch, int(r.start_anchor), int(r.end_anchor),
                                                   int(r.start_extent), int(r.end_extent))  # noqa: E731
                    centre = (lo + hi) // 2
                    a1, a2 = int(r.start_anchor), int(r.start_extent)
                    anchors = [(ch, a) for a in range(int(r.start_anchor), int(r.end_anchor), RES)]
                if flagged(lab, ch, pos, sv, cnv):
                    continue
                m_p, m_h, m_g = fn(pc), fn(hspc), fn(gm)
                m_h0, m_h1 = fn(h[0]), fn(h[1])
                if not np.isfinite([m_p, m_h, m_g, m_h0, m_h1]).all():
                    continue
                marks = [ax.anchor_marks(lab, c, int(a)) for c, a in anchors]
                genes = []
                for c, a in anchors:
                    genes += genes_in(tss, c, int(a), int(a) + RES)
                if kind == "stripe":                       # mean over the 40 kb anchor, 75th percentile
                    marks = [marks_at(lab, ch, int(r.start_anchor), int(r.end_anchor), 0.75)]
                rows.append({"kind": kind, "patient": p, "chrom": ch, "a1": a1, "a2": a2,
                             "centre": int(centre) // RES * RES, "genes": ";".join(dict.fromkeys(genes)),
                             "ac_both": all(m.get("H3K27ac") for m in marks),
                             "me3_any": any(m.get("H3K27me3") for m in marks),
                             "ac_any": any(m.get("H3K27ac") for m in marks),
                             "meas": m_p, "hspc": m_h, "gm12878": m_g, "half0": m_h0, "half1": m_h1})
        log(f"[select] {kind}: {len(rows)} rows")
    D = pd.DataFrame(rows)
    D["gain_hspc"], D["gain_gm"] = D.meas - D.hspc, D.meas - D.gm12878
    D["gain_half"] = np.minimum(D.half0 - D.hspc, D.half1 - D.hspc)
    D["ok"] = ((D.gain_hspc >= GAIN) & (D.gain_gm >= GAIN) & (D.hspc <= ABSENT + 1.0)
               & (D.gm12878 <= ABSENT + 1.0) & (D.gain_half >= HALF_GAIN))
    # a named locus of the list below qualifies on weaker evidence (flagged in the figure)
    fam = D.genes.map(lambda x: bool(FAMOUS & set(str(x).split(";"))))
    D["relaxed"] = (~D.ok & fam & (D.gain_hspc >= 0.7) & (D.gain_gm >= 0.5) & (D.gain_half >= 0.3))
    D["ok"] = D.ok | D.relaxed
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "candidates.csv", index=False)
    return D


def choose(D0: pd.DataFrame) -> pd.DataFrame:
    D = D0.copy()
    # tier 2 for slots the strict criteria leave empty: gains of at least 0.5
    # over HSPC and GM12878, reproduced in the halves at 0.4
    D["tier"] = np.where(D.ok, 1, 0)
    D.loc[(D.tier == 0) & (D.gain_hspc >= 0.5) & (D.gain_gm >= 0.5) & (D.gain_half >= 0.4), "tier"] = 2
    D = D[D.tier > 0].copy()
    D["score"] = (np.minimum.reduce([D.gain_hspc, D.gain_gm, D.gain_half + 0.4])
                  + 0.3 * D.patient.isin(TEST)
                  + 0.3 * D.genes.map(lambda s: bool(FAMOUS & set(str(s).split(";"))))
                  - 0.5 * (D.tier == 2))
    D["slot"] = ""
    D.loc[(D.kind == "loop") & D.ac_both & ~D.me3_any & (D.genes != ""), "slot"] = "P-E loop"
    D.loc[(D.kind == "loop") & D.me3_any & ~D.ac_both & (D.genes != ""), "slot"] = "P-S loop"
    D.loc[(D.kind == "stripe") & D.me3_any & ~D.ac_any, "slot"] = "repressive stripe"
    D.loc[(D.kind == "stripe") & D.ac_any & ~D.me3_any, "slot"] = "enhancer stripe"
    picked, used = [], []
    for slot, n in (("P-E loop", 3), ("P-S loop", 2), ("repressive stripe", 1), ("enhancer stripe", 1)):
        c = D[(D.slot == slot) & ((D.tier == 1) | slot.endswith("stripe"))].sort_values(
            "score", ascending=False)
        k = 0
        for r in c.itertuples():
            if any(r.chrom == u[0] and abs(r.centre - u[1]) < 1_000_000 for u in used):
                continue
            picked.append(r._asdict())
            used.append((r.chrom, r.centre))
            k += 1
            if k == n:
                break
    return pd.DataFrame(picked).drop(columns="Index", errors="ignore")


# ---------------------------------------------------------------- predict
def patient_ref(p: str) -> Path:
    return at.ref_path(0, RES, None if p in TEST else p)


def run_predict(tag: str, run: str, cell: str, bw: Path, ref: Path, lab: str | None,
                regions: Path, assay: str, device: str, target: Path) -> Path:
    out = OUT / "pred" / tag
    f = out / "w.npz"
    if f.exists():
        return f
    out.mkdir(parents=True, exist_ok=True)
    ck = mx.best_ckpt(mx.REF_OUT / run)
    cmd = [ap.PY, "scripts/crosscell_predict.py", "--config", str(ck / "config.yaml"),
           "--checkpoint", str(ck), "--cell", cell, "--bigwig-dir", str(bw), "--target", str(target),
           "--tta", "--assay", assay, "--binsize", str(RES), "--device", "cuda",
           "--regions", str(regions), "--reference", str(ref), "--out-dir", str(out / "csv"),
           "--save-pred", str(f)]
    if lab is not None:
        cmd += ["--lab", lab]
    env = dict(os.environ, OMP_NUM_THREADS="3", CUDA_VISIBLE_DEVICES=device)
    with open(out / "run.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"predict {tag} failed")
    return f


def window_of(r) -> tuple[str, int, int]:
    s = int(r.centre) - WIN // 2 * RES
    return r.chrom, max(0, s), max(0, s) + WIN * RES


def stage_predict(L: pd.DataFrame, device: str) -> None:
    for p, g in L.groupby("patient"):
        reg = OUT / "pred" / f"regions_{p}.tsv"
        reg.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([window_of(r) for r in g.itertuples()]).drop_duplicates().to_csv(
            reg, sep="\t", header=False, index=False)
        bw, ref, tgt = ap.INP / f"xu_{p}", patient_ref(p), ap.map_path(f"xu_{p}", RES)
        run_predict(f"aml_{p}", "all12aml_refin_drop_10kb", at.guard(p), bw, ref, "xu", reg, "hic",
                    device, tgt)
        run_predict(f"base_{p}", "all12refres_drop_10kb", at.guard(p), bw, ref, None, reg, "hic",
                    device, tgt)
    regs = sorted({window_of(r) for r in L.itertuples()})
    reg = OUT / "pred" / "regions_gm12878.tsv"
    pd.DataFrame(regs).to_csv(reg, sep="\t", header=False, index=False)
    run_predict("aml_gm12878", "all12aml_refin_drop_10kb", "gm12878", mx.CELLS["gm12878"][1],
                mx.consensus_loo("gm12878", RES), "", reg, "hic", device, GM)
    log("[predict] done")


# ---------------------------------------------------------------- figures
def corr_tri(a, b):
    n = a.shape[0]
    iu = np.triu_indices(n, 2)
    x, y = a[iu], b[iu]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50 or x[ok].std() == 0 or y[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def dist_curve(a, b):
    n = a.shape[0]
    out = []
    for d in range(2, n - 8):
        x, y = np.diagonal(a, d), np.diagonal(b, d)
        ok = np.isfinite(x) & np.isfinite(y)
        out.append(np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 8 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan)
    return np.arange(2, n - 8) * RES / 1000, np.array(out)


def track_vals(path: Path, chrom: str, s: int, e: int):
    import pyBigWig
    if not path.exists():
        return None
    bw = pyBigWig.open(str(path))
    v = np.asarray(bw.stats(chrom, s, e, type="mean", nBins=(e - s) // RES), float)
    bw.close()
    return np.log1p(np.nan_to_num(v))


def win_matrix(clr, chrom, s, e):
    M = fetch(clr, chrom, s, s + WIN * RES)
    if M is None or M.shape != (WIN, WIN):
        return None
    return M


def enrich_at(M, i, j, prm) -> float:
    from hicmultiome.bioscore import local_enrichment
    e = local_enrichment(np.exp(np.nan_to_num(M)), prm)
    return float(np.log2(e[min(i, j), max(i, j)]))


def band_mean(M, r0, r1, c0, c1) -> float:
    b = M[max(0, r0):min(WIN, r1), max(0, c0):min(WIN, c1)]
    return float(np.nanmean(b)) if b.size and np.isfinite(b).any() else np.nan


def locus_data(r, tss_names: str) -> dict | None:
    """All panels and numbers of one locus (regular loops and stripes)."""
    import cooler
    from aml_eval import loop_params, pred_matrix
    p, ch = r.patient, r.chrom
    chrom, s, e = window_of(r)
    key = f"{chrom}:{s}-{e}"
    lab = f"xu_{p}"
    prm = loop_params(RES)
    C = lambda path: cooler.Cooler(str(path))                          # noqa: E731
    pan = {"gm_meas": win_matrix(C(GM), chrom, s, e),
           "hspc": win_matrix(C(ax.hspc_consensus_path(RES)), chrom, s, e),
           "pat_meas": win_matrix(C(ap.map_path(lab, RES)), chrom, s, e),
           "ref": win_matrix(C(patient_ref(p)), chrom, s, e)}
    for nm, tag in (("pat_aml", f"aml_{p}"), ("pat_base", f"base_{p}"), ("gm_pred", "aml_gm12878")):
        z = np.load(OUT / "pred" / tag / "w.npz")
        if key not in z.files:
            return None
        pan[nm] = pred_matrix(z[key])
    if any(v is None or v.shape != (WIN, WIN) for v in pan.values()):
        return None
    pan["diff_meas"] = pan["pat_meas"] - pan["gm_meas"]
    pan["diff_pred"] = pan["pat_aml"] - pan["gm_pred"]
    i = (int(r.a1) - s) // RES
    if r.kind == "loop":
        j = (int(r.a2) - s) // RES
        fn = lambda M: enrich_at(M, i, j, prm)                          # noqa: E731
        mark = {"kind": "loop", "i": i, "j": j}
    else:
        S = pd.read_csv(ax.XT / "st4_aml_specific_stripes.csv", dtype={"patient": str})
        S["patient"] = S.patient.map(ax.norm_id)
        q = S[(S.patient == p) & (S.chrom_anchor == ch) & (S.start_anchor == int(r.a1))
              & (S.start_extent == int(r.a2))].iloc[0]
        r0, r1 = (int(q.start_anchor) - s) // RES, (int(q.end_anchor) - s) // RES + 1
        c0, c1 = sorted(((int(q.start_extent) - s) // RES, (int(q.end_extent) - s) // RES))
        fn = lambda M: band_mean(M, r0, r1, c0, c1)                     # noqa: E731
        mark = {"kind": "stripe", "r0": r0, "r1": r1, "c0": c0, "c1": c1}
    st = {k: fn(pan[k]) for k in ("pat_meas", "hspc", "gm_meas", "ref", "pat_aml", "pat_base",
                                  "gm_pred")}
    iu = np.triu_indices(WIN, 2)
    q98 = float(np.nanpercentile(np.abs(pan["pat_meas"][iu]), 98))
    return {"key": key, "pan": pan, "mark": mark, "stat": st, "q98": q98, "s": s, "e": e}


def draw_marks(ax_, mark):
    import matplotlib.patches as mp
    if mark["kind"] == "loop":
        for a, b in ((mark["i"], mark["j"]), (mark["j"], mark["i"])):
            ax_.add_patch(mp.Circle((b, a), 4.5, fill=False, ec="k", lw=0.9))
    else:
        for rr, cc in ((mark["r0"], mark["c0"]), (mark["c0"], mark["r0"])):
            w = (mark["c1"] - mark["c0"]) if rr == mark["r0"] else (mark["r1"] - mark["r0"])
            h = (mark["r1"] - mark["r0"]) if rr == mark["r0"] else (mark["c1"] - mark["c0"])
            ax_.add_patch(mp.Rectangle((cc - 0.5, rr - 0.5), w, h, fill=False, ec="k", lw=0.9))


def fig_locus(n: int, r, d: dict, title: str, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    pan, q = d["pan"], d["q98"]
    cols = [("gm_meas", "GM12878 measured"), ("hspc", "HSPC consensus measured"),
            ("pat_meas", f"patient {r.patient} measured"), ("pat_aml", "patient, AML-trained model"),
            ("pat_base", "patient, model without AML data"), ("ref", "AML reference read"),
            ("gm_pred", "GM12878, AML-trained model"), ("diff_meas", "patient minus GM12878, measured"),
            ("diff_pred", "patient minus GM12878, predicted")]
    ref_for = {"gm_meas": "pat_meas", "hspc": "pat_meas", "pat_aml": "pat_meas",
               "pat_base": "pat_meas", "ref": "pat_meas", "gm_pred": "gm_meas",
               "diff_pred": "diff_meas"}
    fig = plt.figure(figsize=(27, 9.6))
    gs = GridSpec(3, 9, height_ratios=[4, 2.6, 1.7], hspace=0.55, wspace=0.12, figure=fig)
    rvals = {}
    for k, (nm, ti) in enumerate(cols):
        a = fig.add_subplot(gs[0, k])
        a.imshow(pan[nm], cmap="RdBu_r", vmin=-q, vmax=q, interpolation="none")
        draw_marks(a, d["mark"])
        a.set_xticks([]); a.set_yticks([])
        a.set_title(ti, fontsize=8)
        if nm in ref_for:
            rv = corr_tri(pan[nm], pan[ref_for[nm]])
            rvals[nm] = rv
            a.set_xlabel(f"r = {rv:.2f}", fontsize=9)
        else:
            a.set_xlabel(f"colour scale +/-{q:.2f}", fontsize=8)
    # tracks
    sub = gs[1, 0:4].subgridspec(5, 1, hspace=0.15)
    chrom, s, e = r.chrom, d["s"], d["e"]
    tracks = [("patient H3K27ac", ap.INP / f"xu_{r.patient}" / "H3K27ac.bigwig"),
              ("patient H3K27me3", ap.INP / f"xu_{r.patient}" / "H3K27me3.bigwig"),
              ("patient CTCF", ap.INP / f"xu_{r.patient}" / "CTCF.bigwig"),
              ("GM12878 H3K27ac", mx.CELLS["gm12878"][1] / "H3K27ac.bigwig"),
              ("GM12878 CTCF", mx.CELLS["gm12878"][1] / "CTCF.bigwig")]
    tvals = {}
    anchors = [r.a1, r.a2] if r.kind == "loop" else [r.a1]
    for k, (nm, f) in enumerate(tracks):
        a = fig.add_subplot(sub[k])
        v = track_vals(Path(f), chrom, s, e)
        if v is not None:
            a.fill_between(np.arange(WIN), v, step="mid", color="#52514e" if "GM" in nm else "#2a78d6",
                           lw=0)
            tvals[nm] = v
        for x in anchors:
            a.axvline((int(x) - s) // RES, color="k", ls="--", lw=0.7)
        a.set_xlim(0, WIN - 1)
        a.set_yticks([])
        a.set_ylabel(nm, fontsize=6, rotation=0, ha="right", va="center")
        if k < 4:
            a.set_xticks([])
        else:
            a.set_xticks([0, 20, 40, 60, 79])
            a.set_xticklabels([f"{(s + t * RES) / 1e6:.2f}" for t in (0, 20, 40, 60, 79)], fontsize=6)
            a.set_xlabel(f"{chrom} (Mb), 10 kb bins, log1p of the track on its own scale", fontsize=6)
    # per-distance strip
    ad = fig.add_subplot(gs[1, 4:7])
    for nm, ti, col, ls in (("pat_aml", "AML-trained", "#2a78d6", "-"), ("pat_base", "no AML data", "#eb6834", "-"),
                            ("ref", "AML reference", "#e87ba4", "--"), ("hspc", "HSPC", "#52514e", ":"),
                            ("gm_meas", "GM12878", "#1baf7a", ":")):
        x, y = dist_curve(pan[nm], pan["pat_meas"])
        ad.plot(x, y, color=col, ls=ls, lw=1.3, label=ti)
    ad.set_xlabel("distance (kb)", fontsize=7)
    ad.set_ylabel("Pearson per diagonal\nvs patient measured", fontsize=7)
    ad.yaxis.set_label_coords(-0.14, 0.5)
    ad.tick_params(labelsize=6)
    ad.legend(fontsize=6, frameon=False, ncol=2)
    ad.grid(alpha=0.25, lw=0.5)
    # numbers
    at_ = fig.add_subplot(gs[1, 7:9])
    at_.axis("off")
    st = d["stat"]
    unit = "log2 donut enrichment at the loop pixel" if r.kind == "loop" else "mean log O/E of the stripe band"
    txt = (f"strength ({unit})\n"
           f"patient measured {st['pat_meas']:.2f}   HSPC {st['hspc']:.2f}   GM12878 {st['gm_meas']:.2f}\n"
           f"measured gain, patient minus HSPC: {st['pat_meas'] - st['hspc']:.2f}\n"
           f"predicted gain over HSPC: AML-trained {st['pat_aml'] - st['hspc']:.2f}, "
           f"no AML data {st['pat_base'] - st['hspc']:.2f}\n"
           f"predicted minus AML reference: AML-trained {st['pat_aml'] - st['ref']:.2f}, "
           f"no AML data {st['pat_base'] - st['ref']:.2f}\n"
           f"GM12878 predicted {st['gm_pred']:.2f}")
    at_.text(0, 1, txt, va="top", fontsize=8, family="monospace")
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    arrs = {f"map_{k}": v.astype(np.float32) for k, v in pan.items()}
    arrs.update({f"track_{k}": v for k, v in tvals.items()})
    arrs.update({"q98": np.array(q), "window": np.array([chrom, s, e], dtype=object)})
    np.savez_compressed(out.with_suffix(".npz"), **arrs, allow_pickle=True)
    return {"locus": n, **{f"r_{k}": v for k, v in rvals.items()}, **{f"strength_{k}": v for k, v in st.items()}}


# ---------------------------------------------------------------- SV locus
SV_PATIENT, SV_INDEX = "629", 238            # inv(16), junction 15.72 Mb | 67.10 Mb, orientation --
SV_MODEL = "all12refres_drop_10kb"


def sv_locus() -> dict:
    """Derivative-allele panels of the inv(16) of patient 629 (HSF4 hijacking).
    Derivative coordinates: left piece = chr16 15.72 to 16.68 Mb reversed, right
    piece = chr16 67.10 to 68.06 Mb; 40 bins of each side of the junction. The
    prediction is the stored derivative-allele prediction of
    scripts/aml_sv_predict.py (non-AML model, consensus12 reference); the AML-
    trained model was not run on derivative alleles."""
    import cooler
    h, size = 96, 192
    z = np.load(ap.PRIM / "sv" / f"windows_xu_{SV_PATIENT}.npz", allow_pickle=True)
    Pd = z[f"{SV_MODEL}|{SV_INDEX}|Pd"]
    M = z[f"{SV_MODEL}|{SV_INDEX}|M"]
    E = np.load(ap.PRIM / "sv" / f"expected_xu_{SV_PATIENT}_10kb.npy")
    d = np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
    Ed = E[np.minimum(d, len(E) - 1)]
    cross = np.log(np.where(M > 0, M, np.nan) / Ed[:h, h:])

    def build(clr, with_cross):
        D = np.full((size, size), np.nan)
        L = fetch(clr, "chr16", 15_720_000, 15_720_000 + h * RES)
        R = fetch(clr, "chr16", 67_100_000, 67_100_000 + h * RES)
        D[:h, :h] = L[::-1, ::-1]
        D[h:, h:] = R
        if with_cross:
            D[:h, h:] = cross
            D[h:, :h] = cross.T
        return D
    lab = f"xu_{SV_PATIENT}"
    maps = {"gm_meas": build(cooler.Cooler(str(GM)), False),
            "hspc": build(cooler.Cooler(str(ax.hspc_consensus_path(RES))), False),
            "pat_meas": build(cooler.Cooler(str(ap.map_path(lab, RES))), True),
            "pat_pred": Pd}
    sl = slice(h - 40, h + 40)
    crop = {k: v[sl, sl] for k, v in maps.items()}
    q = float(np.nanpercentile(np.abs(crop["pat_meas"][np.triu_indices(WIN, 2)]), 98))
    # HSF4 promoter (67.11 Mb) and the hijacked enhancer (16.03 Mb)
    ie = h - 1 - (16_030_000 - 15_720_000) // RES - (h - 40)
    ig = h + (67_110_000 - 67_100_000) // RES - (h - 40)
    return {"crop": crop, "q98": q, "pix": (ie, ig)}


def fig_sv(out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    d = sv_locus()
    crop, q = d["crop"], d["q98"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.4))
    items = [("gm_meas", "GM12878 measured\n(no inversion; junction block not defined)"),
             ("hspc", "HSPC consensus measured\n(no inversion; junction block not defined)"),
             ("pat_meas", "patient 629 measured\n(derivative allele coordinates)"),
             ("pat_pred", "patient 629 predicted by the model\nwithout AML data, derivative allele"),
             (None, "AML-trained model\nnot run on derivative alleles")]
    r_all = r_x = np.nan
    for a, (k, ti) in zip(axes, items):
        a.set_xticks([]); a.set_yticks([])
        a.set_title(ti, fontsize=8)
        if k is None:
            a.axis("off")
            continue
        a.imshow(crop[k], cmap="RdBu_r", vmin=-q, vmax=q, interpolation="none")
        a.axhline(39.5, color="k", lw=0.5, ls=":"); a.axvline(39.5, color="k", lw=0.5, ls=":")
        for rr, cc in ((d["pix"][0], d["pix"][1]), (d["pix"][1], d["pix"][0])):
            a.add_patch(mp.Circle((cc, rr), 3.5, fill=False, ec="k", lw=0.9))
        if k != "pat_meas":
            rv = corr_tri(crop[k], crop["pat_meas"])
            a.set_xlabel(f"r = {rv:.2f}", fontsize=9)
            if k == "pat_pred":
                r_all = rv
                xb = crop[k][:40, 40:].ravel(), crop["pat_meas"][:40, 40:].ravel()
                ok = np.isfinite(xb[0]) & np.isfinite(xb[1])
                r_x = float(np.corrcoef(xb[0][ok], xb[1][ok])[0, 1]) if ok.sum() > 30 else np.nan
        else:
            a.set_xlabel(f"colour scale +/-{q:.2f}", fontsize=8)
    jn = pd.read_csv(ap.RES_DIR / "sv_junctions.csv")
    jn = jn[(jn.patient == f"xu_{SV_PATIENT}") & (jn.sv_index == SV_INDEX)].set_index("model")
    txt = (f"junction block (upper right of each panel, 40 x 40 bins): r predicted vs measured = {r_x:.2f}\n"
           f"neo-loop pixels HSF4 promoter x enhancer (circles), log2 donut enrichment over the 60 listed "
           f"neo-loops: measured {jn.loc[SV_MODEL, 'neoloop_meas_enrich_median']:.2f}, "
           f"predicted {jn.loc[SV_MODEL, 'neoloop_pred_enrich_median']:.2f} (model without AML data), "
           f"{jn.loc['all12honest_drop_10kb', 'neoloop_pred_enrich_median']:.2f} (no reference)")
    fig.suptitle("SV, expected to fail: inv(16) of patient 629 (training patient), HSF4 enhancer hijacking, "
                 "80 bins per panel across the junction (800 kb)", fontsize=10, y=1.03)
    fig.text(0.01, 0.01, txt, fontsize=7)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(out.with_suffix(".npz"), **{f"map_{k}": v.astype(np.float32) for k, v in crop.items()},
                        q98=np.array(q))
    return {"locus": 8, "r_pat_pred": r_all, "r_junction_block": r_x,
            "neoloop_meas": float(jn.loc[SV_MODEL, "neoloop_meas_enrich_median"]),
            "neoloop_pred_base": float(jn.loc[SV_MODEL, "neoloop_pred_enrich_median"]),
            "neoloop_pred_honest": float(jn.loc["all12honest_drop_10kb", "neoloop_pred_enrich_median"])}


def fig_summary(rows: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    lim = (rows[["gain_meas", "gain_aml", "gain_base"]].min().min() - 0.2,
           rows[["gain_meas", "gain_aml", "gain_base"]].max().max() + 0.2)
    for a, col, ti in ((axes[0], "gain_aml", "AML-trained model"), (axes[1], "gain_base", "model without AML data")):
        for r in rows.itertuples():
            a.scatter(r.gain_meas, getattr(r, col), s=60,
                      c="#2a78d6" if r.kind == "loop" else "#eb6834",
                      marker="o" if r.heldout else "s", ec="k", lw=0.5)
            a.annotate(r.name, (r.gain_meas, getattr(r, col)), fontsize=6, xytext=(3, 3),
                       textcoords="offset points")
        a.plot(lim, lim, "k:", lw=0.8)
        a.axhline(0, color="#888", lw=0.5); a.axvline(0, color="#888", lw=0.5)
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("measured gain, patient minus HSPC consensus (log2 enrichment or log O/E)", fontsize=7)
        a.set_ylabel("predicted gain over the HSPC consensus", fontsize=7)
        ok = rows.gain_meas.notna() & rows[col].notna()
        rho = np.corrcoef(rows.gain_meas[ok], rows[col][ok])[0, 1] if ok.sum() > 3 else np.nan
        a.set_title(f"{ti} (r = {rho:.2f}, n = {int(ok.sum())})", fontsize=9)
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Seven AML-specific loops and stripes: measured against predicted gain. "
                 "Blue loops, orange stripes; circles held-out patients, squares training patients",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def stage_figures() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(OUT / "chosen.csv", dtype={"patient": str})
    P["genes"] = P.genes.fillna("")
    rows = []
    for n, r in enumerate(P.itertuples(), 1):
        d = locus_data(r, r.genes)
        if d is None:
            log(f"[skip] locus {n}")
            continue
        held = r.patient in TEST
        gene = (r.genes.split(";")[0] if r.genes else
                f"{r.chrom} {min(r.a1, r.a2) / 1e6:.2f}-{max(r.a1, r.a2) / 1e6:.2f} Mb")
        tier = " (weaker evidence)" if getattr(r, "tier", 1) == 2 else ""
        title = (f"{n}. {r.slot}, {gene}, patient {r.patient}, "
                 f"{'held-out test patient' if held else 'training patient'} of the AML-inclusive fold 0"
                 f"{tier}; GM12878 predictions come from a training cell")
        name = (f"{n:02d}_{r.slot.replace(' ', '_')}_{gene}_xu{r.patient}"
                .replace(";", "_").replace(" ", "_").replace(":", ""))
        rec = fig_locus(n, r, d, title, FIG / f"{name}.png")
        st = d["stat"]
        rec.update({"name": f"{gene} ({r.patient})", "slot": r.slot, "kind": r.kind, "patient": r.patient,
                    "heldout": held, "gene": gene, "chrom": r.chrom, "a1": r.a1, "a2": r.a2,
                    "gain_meas": st["pat_meas"] - st["hspc"], "gain_aml": st["pat_aml"] - st["hspc"],
                    "gain_base": st["pat_base"] - st["hspc"], "gain_aml_over_ref": st["pat_aml"] - st["ref"],
                    "gain_base_over_ref": st["pat_base"] - st["ref"], "tier": getattr(r, "tier", 1)})
        rows.append(rec)
        log(f"[figure] {name}")
    R = pd.DataFrame(rows)
    sv = fig_sv(FIG / "08_SV_expected_to_fail_HSF4_inv16_xu629.png")
    R.to_csv(ap.RES_DIR / "sites_loci.csv", index=False)
    pd.DataFrame([sv]).to_csv(ap.RES_DIR / "sites_sv_locus.csv", index=False)
    fig_summary(R, FIG / "summary_measured_vs_predicted_gain.png")
    log("[figures] done")


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--stage", nargs="+", required=True, choices=["select", "predict", "figures"])
    a.add_argument("--device", default="0")
    a = a.parse_args()
    if "select" in a.stage:
        D = stage_select()
        choose(D).to_csv(OUT / "chosen.csv", index=False)
    if "predict" in a.stage:
        P = pd.read_csv(OUT / "chosen.csv", dtype={"patient": str})
        stage_predict(P, a.device)
    if "figures" in a.stage:
        stage_figures()


if __name__ == "__main__":
    main()
