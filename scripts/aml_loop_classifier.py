"""Does the patient's own epigenome distinguish AML-specific Hi-C loops from shared loops?

Loop-level classification with held-out patients (registry section on the loop classifier).

Objects
  loops    positives: AML-specific loops of Xu et al. 2022 (Suppl. Table 1, 10 kb anchors,
           60 kb to 1.5 Mb, intra-chromosomal) of the patients in LOOP_PATIENTS.
  stripes  positives: AML-specific stripes (Suppl. Table 4; anchor 40 kb, extent block).

Negatives (labels only use Hi-C; no feature uses the Hi-C of the target pair)
  A_full   shared loops (log2 enrichment >= 0.6 in both): local-maximum donut enrichment (hicmultiome.bioscore, 10 kb, same
           parameters as the AML scripts) in the patient map AND in the HSPC consensus,
           not within 2 bins of an AML-specific loop of the patient; matched 1:1 within the
           patient on distance, CTCF motif orientation class and the anchors' mean H3K27ac
           (patients with CUT&Tag only)
  A_hspc   the same pool, matched on distance, orientation and the anchors' HSPC CTCF and H3K27ac
  A_do     the same pool, matched on distance and orientation class only (all patients)
  B_rand   random anchor pairs on mappable bins of the same chromosome, same bin distance
           (distance only)
  Stripes: the stripe rectangle (anchor rows x extent columns, same gap and extent length) moved
  along the chromosome; shared = band mean log O/E high in patient and HSPC consensus.

Features (patient data only): H3K27ac, H3K27me3, CTCF CUT&Tag (MACS2 -log10 p, quantile-matched to
the training scale, the inputs of the AML models) at the anchor bin (mean, max) and +-20 kb
(flank mean); WGBS methylation (1 kb bins; anchor mean, min, 1 kb bin at the strongest CTCF
motif); CTCF motif strength (JASPAR MA0139.2, hg38 scan, hits at >= 70 percent of the maximal log-odds score) and orientation class; distance; promoter
(TSS in anchor, refGene). HSPC: the same marks of one public CD34+ ChIP experiment per mark
(ChIP-Atlas, quantile-matched), plus patient minus HSPC. Per pair every anchor feature enters as
min and max over the two anchors.

Models: logistic regression and histogram gradient boosting; leave-one-patient-out and
leave-subtype-out (subtype inferred from Suppl. Table 3, see `infer_subtypes`); patient bootstrap.

    python scripts/aml_loop_classifier.py --stage hic tracks build model figures
Run with the hicdiff conda env (cooler, pyBigWig, pyfaidx, scikit-learn).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
PRIM = HM / "aml/primary"
OE = HM / "aml/oe"
CONS = PRIM / "consensus"
INP = PRIM / "inputs"
METH = PRIM / "meth"
XT = REPO / "results/aml_primary/xu_tables"
OUT = REPO / "results/aml_loop_classifier"
CACHE = PRIM / "work/loopclf"
FASTA = Path.home() / "data/hic-multiome/genome/hg38.fa"
JASPAR = Path("/mnt/processed/wolffjoa/home_offload/hic-multiome_data_copy/motifs/MA0139.2.jaspar")
REFGENE = PRIM / "annot/refGene.txt.gz"

RES = 10_000
MIN_BINS, MAX_BINS = 6, 150           # 60 kb to 1.5 Mb
CT_PATIENTS = ["018", "027", "168", "270", "472", "546", "629", "773", "1360", "1916"]
CTCF_PATIENTS = ["018", "027", "168", "270", "472", "546", "629", "773"]   # CTCF + K27ac + me3 + WGBS
WGBS_ONLY = ["1071", "1324", "260", "424", "496", "504", "566", "798", "838", "933"]
LOOP_PATIENTS = CT_PATIENTS + WGBS_ONLY
# Registry sections 50 and 54: xu_103, xu_673, xu_576 are healthy PBMC donors; 168/496 are two samples of
# one leukaemia and 260/424 probably too (results/aml_duplicate_check/). Both corrections are on by default
# (--exclude-healthy, --group-donors); the opposite flags reproduce section 47.
HEALTHY = ["103", "673", "576"]
DONOR_GROUPS = {"168": "168+496", "496": "168+496", "260": "260+424", "424": "260+424"}
MARKS = ["H3K27ac", "H3K27me3", "CTCF"]
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
SEED = 0
E_MIN = 0.6            # log2 donut enrichment of a shared loop in the patient and in the HSPC consensus
MOTIF_FRAC = 0.70      # CTCF PWM hit threshold (fraction of the maximal log-odds score)


def donor_of(p: str, group: bool = True) -> str:
    """Donor unit of a sample: the two samples of one leukaemia share a label (fold unit)."""
    return DONOR_GROUPS.get(p, p) if group else p


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def norm_id(x) -> str:
    s = str(x).lstrip("#")
    return s.lstrip("0") or "0" if s.isdigit() and not s.startswith("0") else s


def pid(p: str) -> str:
    """Patient id as in the tables ('018' stays '018', '#27' -> '027')."""
    s = str(p).lstrip("#")
    return s.zfill(3) if s.isdigit() else s


# ====================================================================== Hi-C stage (labels only)
WIN, STRIDE, MARGIN = 420, 250, 10


def _windows(n: int):
    s = 0
    while s < n - 2 * MARGIN:
        yield s, min(n, s + WIN)
        s += STRIDE


def _fetch(clr, chrom: str, s: int, e: int) -> np.ndarray:
    size = int(clr.chromsizes[chrom])
    M = np.asarray(clr.matrix(balance=False).fetch(f"{chrom}:{s * RES}-{min(e * RES, size)}"), np.float64)
    M = np.where(M == 0, np.nan, M)
    if M.shape[0] < WIN:
        P = np.full((WIN, WIN), np.nan)
        P[:M.shape[0], :M.shape[1]] = M
        M = P
    return M


def hic_one(p: str) -> str:
    """Per patient: donut enrichments (log2) of the patient and the HSPC consensus at every local
    maximum of the patient map (distance 6 to 150 bins), mappable bins, and stripe-rectangle
    integral images are NOT stored; stripes are scored in `stripe_scores`."""
    import cooler
    from hicmultiome.bioscore import BioParams, local_enrichment
    prm = BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4)
    cp = cooler.Cooler(str(OE / f"xu_{p}_proper_10kb.cool"))
    ch = cooler.Cooler(str(CONS / "hspccons_10kb.cool"))
    from scipy.ndimage import maximum_filter
    rows, valid = [], {}
    for chrom in CHROMS:
        n = cp.chromsizes[chrom] // RES + 1
        v = np.zeros(n)
        for s, e in _windows(n):
            Mp = _fetch(cp, chrom, s, e)
            Mh = _fetch(ch, chrom, s, e)
            Lp = np.log2(local_enrichment(np.exp(Mp), prm))
            Lh = np.log2(local_enrichment(np.exp(Mh), prm))
            mp = maximum_filter(np.nan_to_num(Lp, nan=-9), size=3)
            mh = maximum_filter(np.nan_to_num(Lh, nan=-9), size=3)
            r0 = MARGIN if s > 0 else 0
            own = np.arange(r0, MARGIN + STRIDE)
            isum = (np.nan_to_num(Lp, nan=-9) == mp) & (Lp > 0.4)
            ii, jj = np.nonzero(np.triu(isum, MIN_BINS))
            keep = (jj - ii <= MAX_BINS) & (ii >= r0) & (ii < MARGIN + STRIDE) & (s + jj < n)
            ii, jj = ii[keep], jj[keep]
            rows.append(pd.DataFrame({"chrom": chrom, "i": s + ii, "j": s + jj, "e_pat": Lp[ii, jj],
                                      "e_hspc3": mh[ii, jj], "e_hspc": Lh[ii, jj]}))
            ok = np.isfinite(Mp)
            band = np.zeros(WIN)
            for d in (2, 10, 40, 100):
                band += np.pad(np.diagonal(ok, d).astype(float), (0, d))
            idx = s + own
            m = idx < n
            v[idx[m]] = band[own][m] / 4.0
        valid[chrom] = v >= 0.5
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(CACHE / f"peaks_{p}.csv.gz", index=False)
    np.savez_compressed(CACHE / f"valid_{p}.npz", **valid)
    return p


def stage_hic(patients: list[str], workers: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [p for p in patients if not (CACHE / f"peaks_{p}.csv.gz").exists()]
    log(f"[hic] {len(todo)} patients")
    with ProcessPoolExecutor(workers) as ex:
        for p in ex.map(hic_one, todo):
            log(f"[hic] {p} done")



# ====================================================================== track arrays (10 kb bins)
def n_bins(chrom: str, sizes: dict) -> int:
    return sizes[chrom] // RES + 1


def chrom_sizes() -> dict:
    d = {}
    for line in (PRIM / "hg38.chrom.sizes").read_text().splitlines():
        c, n = line.split()[:2]
        d[c] = int(n)
    return d


def track_path(src: str, mark: str) -> Path | None:
    if mark == "Meth":
        f = METH / f"xu_{src}" / "Meth.bigwig"
    elif src == "hspc":
        f = INP / "hspc_cd34" / f"{mark}.bigwig"
    else:
        f = INP / f"xu_{src}" / f"{mark}.bigwig"
    return f if f.exists() else None


def tracks_one(job):
    """10 kb bin mean and max of a quantile-matched CUT&Tag/ChIP track; for methylation the 1 kb
    bins themselves (value 1 + methylated fraction, bins without data at 1 + genome mean)."""
    import pyBigWig
    src, mark = job
    f = track_path(src, mark)
    dest = CACHE / f"trk_{src}_{mark}.npz"
    if f is None or dest.exists():
        return job, f is not None
    sizes = chrom_sizes()
    bw = pyBigWig.open(str(f))
    out = {}
    for c in CHROMS:
        n = n_bins(c, sizes)
        if mark == "Meth":
            nb = sizes[c] // 1000
            kb = np.array(bw.stats(c, 0, nb * 1000, type="mean", nBins=nb), dtype=float)
            out[c] = kb.astype(np.float32)
        else:
            nb = sizes[c] // RES
            mean = np.full(n, np.nan)
            mx = np.full(n, np.nan)
            mean[:nb] = np.array(bw.stats(c, 0, nb * RES, type="mean", nBins=nb), dtype=float)
            mx[:nb] = np.array(bw.stats(c, 0, nb * RES, type="max", nBins=nb), dtype=float)
            out[c + "|mean"] = mean.astype(np.float32)
            out[c + "|max"] = mx.astype(np.float32)
    bw.close()
    np.savez_compressed(dest, **out)
    return job, True


def stage_tracks(workers: int) -> None:
    jobs = [(s, m) for s in LOOP_PATIENTS + ["hspc"] for m in MARKS + ["Meth"]
            if not (s == "hspc" and m == "Meth")]
    with ProcessPoolExecutor(workers) as ex:
        for job, ok in ex.map(tracks_one, jobs):
            if not ok:
                log(f"[tracks] {job}: no file")
    log("[tracks] done")


# ====================================================================== CTCF motif scan and genes
def _scan_chunk(codes, lut, thresh, L):
    n = codes.shape[0] - L + 1
    score = np.zeros(n, dtype=np.float32)
    for k in range(L):
        score += lut[codes[k:k + n], k].astype(np.float32)
    idx = np.nonzero(score >= thresh)[0]
    return idx, score[idx]


def motif_one(chrom: str) -> str:
    from hicmultiome.motifs import pwm_logodds, read_jaspar, revcomp_pwm
    from hicmultiome.sequence import read_chrom_codes
    dest = CACHE / f"motif70_{chrom}.npz"
    if dest.exists():
        return chrom
    pwm = pwm_logodds(read_jaspar(JASPAR))
    L = pwm.shape[1]
    mx = float(pwm.max(axis=0).sum())
    codes = read_chrom_codes(str(FASTA), chrom)
    res = {}
    for name, w in (("fwd", pwm), ("rev", revcomp_pwm(pwm))):
        lut = np.vstack([w, w.min(axis=0, keepdims=True)])
        pos, sc = [], []
        step = 10_000_000
        for a in range(0, codes.shape[0], step):
            sub = codes[a:a + step + L - 1]
            if sub.shape[0] < L:
                continue
            i, v = _scan_chunk(sub, lut, MOTIF_FRAC * mx, L)
            pos.append(i + a)
            sc.append(v / mx)
        res[name + "_pos"] = np.concatenate(pos).astype(np.int64)
        res[name + "_score"] = np.concatenate(sc).astype(np.float32)
    np.savez_compressed(dest, **res)
    return chrom


def stage_motifs(workers: int) -> None:
    with ProcessPoolExecutor(workers) as ex:
        for c in ex.map(motif_one, CHROMS):
            log(f"[motifs] {c}")


def load_motif_bins(sizes: dict) -> dict:
    """Per chromosome and 10 kb bin: best hit score fraction, its strand (+1 forward, -1 reverse,
    0 none), its base-pair position, the number of hits."""
    out = {}
    for c in CHROMS:
        z = np.load(CACHE / f"motif70_{c}.npz")
        n = n_bins(c, sizes)
        pos = np.concatenate([z["fwd_pos"], z["rev_pos"]]) + 9           # motif centre
        sc = np.concatenate([z["fwd_score"], z["rev_score"]])
        st = np.concatenate([np.ones(len(z["fwd_pos"])), -np.ones(len(z["rev_pos"]))])
        b = np.minimum(pos // RES, n - 1)
        order = np.lexsort((-sc, b))
        b, sc, st, pos = b[order], sc[order], st[order], pos[order]
        first = np.r_[True, b[1:] != b[:-1]]
        best = np.zeros(n, np.float32)
        strand = np.zeros(n, np.int8)
        bpos = np.full(n, -1, np.int64)
        best[b[first]] = sc[first]
        strand[b[first]] = st[first]
        bpos[b[first]] = pos[first]
        out[c] = {"score": best, "strand": strand, "pos": bpos,
                  "n": np.bincount(b, minlength=n).astype(np.float32)}
    return out


def load_genes(sizes: dict) -> dict:
    """Per chromosome: TSS bin flag, coding-TSS flag, log10 distance (bp) to the nearest TSS of
    the bin centre (cap 1 Mb)."""
    tss = {c: [] for c in CHROMS}
    cod = {c: [] for c in CHROMS}
    with gzip.open(REFGENE, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[2] not in tss:
                continue
            t = int(f[4]) if f[3] == "+" else int(f[5])
            tss[f[2]].append(t)
            cod[f[2]].append(f[1].startswith("NM_"))
    out = {}
    for c in CHROMS:
        n = n_bins(c, sizes)
        t = np.array(tss[c], dtype=np.int64)
        cd = np.array(cod[c], dtype=bool)
        flag = np.zeros(n)
        cflag = np.zeros(n)
        if len(t):
            np.add.at(flag, np.minimum(t // RES, n - 1), 1)
            np.add.at(cflag, np.minimum(t[cd] // RES, n - 1), 1)
        ts = np.sort(t)
        cen = np.arange(n) * RES + RES // 2
        if len(ts):
            k = np.clip(np.searchsorted(ts, cen), 1, len(ts) - 1)
            d = np.minimum(np.abs(cen - ts[k - 1]), np.abs(ts[k] - cen))
        else:
            d = np.full(n, 1e6)
        out[c] = {"tss": flag, "tss_coding": cflag, "dtss": np.log10(np.minimum(d, 1e6) + 1.0)}
    return out



# ====================================================================== feature extraction
class Ctx:
    """Everything a feature call needs; loaded lazily once per process."""

    def __init__(self):
        self.sizes = chrom_sizes()
        self.motif = load_motif_bins(self.sizes)
        self.genes = load_genes(self.sizes)
        self._trk = {}
        import aml_primary as ap
        self.q90 = {m: float(ap.reference_quantiles(m)[int(0.7 * (ap.NQ - 1))]) for m in ("H3K27ac", "H3K27me3")}

    def trk(self, src: str, mark: str):
        k = (src, mark)
        if k not in self._trk:
            f = CACHE / f"trk_{src}_{mark}.npz"
            self._trk[k] = dict(np.load(f)) if f.exists() else None
        return self._trk[k]


def _nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def region_feats(cx: Ctx, p: str, chrom: str, lo: int, hi: int, anchor_only_extras: bool = True) -> dict:
    """Features of the bins [lo, hi) of 10 kb for patient p (own, HSPC, difference, methylation,
    motif, gene). NaN where the track does not exist for the patient."""
    f = {}
    n = n_bins(chrom, cx.sizes)
    lo_f, hi_f = max(0, lo - 2), min(n, hi + 2)
    for m in MARKS:
        vals = {}
        for src in (p, "hspc"):
            T = cx.trk(src, m)
            if T is None:
                vals[src] = (np.nan, np.nan, np.nan)
                continue
            mean, mx = T[chrom + "|mean"], T[chrom + "|max"]
            flank = np.r_[mean[lo_f:lo], mean[hi:hi_f]]
            vals[src] = (_nanmean(mean[lo:hi]), float(np.nanmax(mx[lo:hi])) if np.isfinite(mx[lo:hi]).any() else np.nan,
                         _nanmean(flank))
        for k, nm in enumerate(("mean", "max", "flank")):
            f[f"own_{m}_{nm}"] = vals[p][k]
            f[f"hspc_{m}_{nm}"] = vals["hspc"][k]
            f[f"diff_{m}_{nm}"] = vals[p][k] - vals["hspc"][k]
    Tm = cx.trk(p, "Meth")
    mo = cx.motif[chrom]
    seg = mo["score"][lo:hi]
    kbest = int(np.argmax(seg)) if seg.size else 0
    best = float(seg[kbest]) if seg.size else 0.0
    f["motif_score"] = best
    f["motif_nhits"] = float(mo["n"][lo:hi].sum())
    f["_strand"] = float(mo["strand"][lo + kbest]) if best > 0 else 0.0
    if Tm is not None:
        kb = Tm[chrom][lo * 10:hi * 10].astype(float)
        f["meth_mean"] = _nanmean(kb) - 1.0
        f["meth_min"] = float(np.nanmin(kb)) - 1.0 if kb.size else np.nan
        if best > 0:
            pos = int(mo["pos"][lo + kbest]) // 1000
            f["meth_motif"] = float(Tm[chrom][min(pos, len(Tm[chrom]) - 1)]) - 1.0
        else:
            f["meth_motif"] = np.nan
    else:
        f["meth_mean"] = f["meth_min"] = f["meth_motif"] = np.nan
    g = cx.genes[chrom]
    f["gene_tss"] = float(g["tss"][lo:hi].sum() > 0)
    f["gene_coding"] = float(g["tss_coding"][lo:hi].sum() > 0)
    f["gene_dtss"] = float(g["dtss"][lo:hi].min())
    return f


ORIENT = ["conv", "div", "tandem", "none"]


def orient_class(sa: float, sb: float, score_a: float, score_b: float) -> str:
    if score_a <= 0 or score_b <= 0 or sa == 0 or sb == 0:
        return "none"
    if sa == 1 and sb == -1:
        return "conv"
    if sa == -1 and sb == 1:
        return "div"
    return "tandem"


def pair_feats(cx: Ctx, p: str, chrom: str, a: tuple, b: tuple, kind: str) -> dict:
    """Feature row of a loop (a, b = one bin each) or a stripe (a = anchor bins, b = extent bins).
    Region features enter as min and max over the two regions; for stripes the motif, methylation
    at the motif and gene features describe the anchor only (the extent is a block, not a site)."""
    fa = region_feats(cx, p, chrom, *a)
    fb = region_feats(cx, p, chrom, *b)
    row = {}
    only_a = ("motif_", "meth_motif", "gene_")
    for k in fa:
        if k.startswith("_"):
            continue
        if kind == "stripe" and k.startswith(only_a):
            row[k + "_min"] = row[k + "_max"] = fa[k]
        else:
            row[k + "_min"] = np.nanmin([fa[k], fb[k]]) if np.isfinite([fa[k], fb[k]]).any() else np.nan
            row[k + "_max"] = np.nanmax([fa[k], fb[k]]) if np.isfinite([fa[k], fb[k]]).any() else np.nan
    if kind == "loop":
        oc = orient_class(fa["_strand"], fb["_strand"], fa["motif_score"], fb["motif_score"])
        dist = (b[0] - a[0]) * RES
    else:
        direction = 1 if b[0] > a[0] else -1
        facing = fa["_strand"] == direction and fa["motif_score"] > 0
        oc = "conv" if facing else ("none" if fa["motif_score"] <= 0 else "div")
        dist = abs(b[0] - a[0]) * RES
    for o in ORIENT:
        row[f"motif_orient_{o}"] = float(oc == o)
    row["dist_log10"] = float(np.log10(max(dist, 1)))
    if kind == "stripe":
        row["dist_extent_log10"] = float(np.log10((b[1] - b[0]) * RES))
    row["gene_nprom"] = float(fa["gene_tss"] + (fb["gene_tss"] if kind == "loop" else 0.0))
    # type from the patient's own anchors (loops: PE = promoter anchor and partner anchor above the 70th percentile of the training H3K27ac scale, PS = partner above that of H3K27me3 (priority); stripes: enh, rep, same thresholds on the anchor)
    q = cx.q90
    ac_a, ac_b = fa["own_H3K27ac_mean"], fb["own_H3K27ac_mean"]
    me_a, me_b = fa["own_H3K27me3_mean"], fb["own_H3K27me3_mean"]
    typ = "other"
    if np.isfinite(ac_a):
        if kind == "loop":
            for (x, y_) in (((fa, ac_a, me_a), (fb, ac_b, me_b)), ((fb, ac_b, me_b), (fa, ac_a, me_a))):
                if x[0]["gene_tss"] > 0:
                    if y_[2] >= q["H3K27me3"]:
                        typ = "PS"
                        break
                    if y_[1] >= q["H3K27ac"] and typ != "PS":
                        typ = "PE"
        else:
            if me_a >= q["H3K27me3"]:
                typ = "rep"
            elif ac_a >= q["H3K27ac"]:
                typ = "enh"
    row["_type"] = typ
    row["_orient"] = oc
    row["_prom_both"] = float(kind == "loop" and fa["gene_tss"] > 0 and fb["gene_tss"] > 0)
    return row



# ====================================================================== loops: positives, pools, matching
def read_tab(name: str, kind: str) -> pd.DataFrame:
    d = pd.read_csv(XT / name, dtype={"patient": str})
    d["patient"] = d.patient.str.lstrip("#")
    d["patient"] = d.patient.map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)
    return d


def positives_loops(p: str) -> pd.DataFrame:
    L = read_tab("st1_aml_specific_loops.csv", "loop")
    L = L[(L.patient == p) & (L.chrom1 == L.chrom2) & L.chrom1.isin(CHROMS)]
    out = pd.DataFrame({"chrom": L.chrom1.values, "i": (L.start1 // RES).values, "j": (L.start2 // RES).values})
    out = out[(out.j - out.i >= MIN_BINS) & (out.j - out.i <= MAX_BINS)].drop_duplicates()
    return out.reset_index(drop=True)


def attrs(cx: Ctx, p: str, df: pd.DataFrame) -> pd.DataFrame:
    """Matching variables of loops (chrom, i, j): bin distance, orientation class, mean H3K27ac."""
    d = (df.j - df.i).values
    cls = np.empty(len(df), dtype=object)
    ac = np.full(len(df), np.nan)
    hc = np.full(len(df), np.nan)
    hk = np.full(len(df), np.nan)
    T = cx.trk(p, "H3K27ac")
    TH, TC = cx.trk("hspc", "CTCF"), cx.trk("hspc", "H3K27ac")
    for c, g in df.groupby("chrom"):
        ix = np.where(df.chrom.values == c)[0]
        mo = cx.motif[c]
        si, sj = mo["strand"][g.i.values], mo["strand"][g.j.values]
        sci, scj = mo["score"][g.i.values], mo["score"][g.j.values]
        for k, a, b, x, y in zip(ix, si, sj, sci, scj):
            cls[k] = orient_class(a, b, x, y)
        if T is not None:
            ac[ix] = 0.5 * (T[c + "|mean"][g.i.values] + T[c + "|mean"][g.j.values])
        hc[ix] = 0.5 * (TH[c + "|mean"][g.i.values] + TH[c + "|mean"][g.j.values])
        hk[ix] = 0.5 * (TC[c + "|mean"][g.i.values] + TC[c + "|mean"][g.j.values])
    return pd.DataFrame({"d": d, "cls": cls, "ac": ac, "hc": hc, "hk": hk})


def pool_loops(p: str, pos: pd.DataFrame) -> pd.DataFrame:
    pk = pd.read_csv(CACHE / f"peaks_{p}.csv.gz")
    v = np.load(CACHE / f"valid_{p}.npz")
    pk = pk[(pk.e_pat >= E_MIN) & (pk.e_hspc3 >= E_MIN)]
    ok = np.array([v[c][i] and v[c][j] for c, i, j in zip(pk.chrom, pk.i, pk.j)], dtype=bool)
    pk = pk[ok]
    # not within 2 bins (Chebyshev) of an AML-specific loop of the patient
    from scipy.spatial import cKDTree
    keep = np.ones(len(pk), bool)
    for c, g in pos.groupby("chrom"):
        ix = np.where(pk.chrom.values == c)[0]
        if not len(ix):
            continue
        t = cKDTree(g[["i", "j"]].values)
        dd, _ = t.query(pk.iloc[ix][["i", "j"]].values, p=np.inf)
        keep[ix[dd <= 2]] = False
    return pk[keep].reset_index(drop=True)


def label_strength(p: str, pos: pd.DataFrame) -> pd.DataFrame:
    """Hi-C donut enrichment at the nearest local maximum within 1.5 bins (diagnostic of the labels
    only; never a feature)."""
    from scipy.spatial import cKDTree
    pk = pd.read_csv(CACHE / f"peaks_{p}.csv.gz")
    e1 = np.full(len(pos), np.nan)
    e2 = np.full(len(pos), np.nan)
    for c, g in pos.groupby("chrom"):
        q = pk[pk.chrom == c]
        if not len(q):
            continue
        dd, ix = cKDTree(q[["i", "j"]].values).query(g[["i", "j"]].values, p=np.inf)
        good = dd <= 1
        idx = g.index.values[good]
        e1[pos.index.get_indexer(idx)] = q.e_pat.values[ix[good]]
        e2[pos.index.get_indexer(idx)] = q.e_hspc3.values[ix[good]]
    return pd.DataFrame({"lab_e_pat": e1, "lab_e_hspc": e2})


def match_greedy(pa: pd.DataFrame, qa: pd.DataFrame, covs: list, rng) -> np.ndarray:
    """1:1 nearest matching without replacement; exact orientation class, distance caliper
    max(2 bins, 15 percent), caliper 0.25 SD of the pool on every covariate in `covs`.
    -1 where no match."""
    out = np.full(len(pa), -1, int)
    avail = np.ones(len(qa), bool)
    qd, qc = qa.d.values, qa.cls.values
    cal = {c: 0.25 * np.nanstd(qa[c].values) for c in covs}
    for k in rng.permutation(len(pa)):
        dcal = max(2.0, 0.15 * pa.d.values[k])
        m = avail & (qc == pa.cls.values[k]) & (np.abs(qd - pa.d.values[k]) <= dcal)
        for c in covs:
            m &= np.abs(qa[c].values - pa[c].values[k]) <= cal[c]
        idx = np.nonzero(m)[0]
        if not len(idx):
            continue
        cost = ((qd[idx] - pa.d.values[k]) / dcal) ** 2
        for c in covs:
            cost = cost + ((qa[c].values[idx] - pa[c].values[k]) / cal[c]) ** 2
        b = idx[int(np.argmin(cost))]
        out[k] = b
        avail[b] = False
    return out


def random_pairs(p: str, pos: pd.DataFrame, cx: Ctx, rng) -> pd.DataFrame:
    v = np.load(CACHE / f"valid_{p}.npz")
    from scipy.spatial import cKDTree
    trees = {c: cKDTree(g[["i", "j"]].values) for c, g in pos.groupby("chrom")}
    rows = []
    for r in pos.itertuples():
        n = len(v[r.chrom])
        d = r.j - r.i
        for _ in range(500):
            i = int(rng.integers(0, n - d))
            j = i + d
            if not (v[r.chrom][i] and v[r.chrom][j]):
                continue
            if trees[r.chrom].query([i, j], p=np.inf)[0] <= 2:
                continue
            rows.append((r.Index, r.chrom, i, j))
            break
        else:
            rows.append((r.Index, r.chrom, -1, -1))
    return pd.DataFrame(rows, columns=["pair", "chrom", "i", "j"])


def smd(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def build_loops_patient(p: str, cx: Ctx) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(SEED + int(p))
    pos = positives_loops(p)
    pa = attrs(cx, p, pos)
    pool = pool_loops(p, pos)
    qa = attrs(cx, p, pool)
    has_ac = cx.trk(p, "H3K27ac") is not None
    rows, qual = [], []
    ls = label_strength(p, pos)
    feats_pos = [pair_feats(cx, p, r.chrom, (r.i, r.i + 1), (r.j, r.j + 1), "loop") for r in pos.itertuples()]
    types = [f["_type"] for f in feats_pos]
    for k, r in enumerate(pos.itertuples()):
        rows.append({"patient": p, "kind": "loop", "set": "pos", "pair_id": f"{p}:L{k}", "y": 1, "chrom": r.chrom,
                     "a_lo": r.i, "a_hi": r.i + 1, "b_lo": r.j, "b_hi": r.j + 1, "pair_type": types[k],
                     "lab_e_pat": ls.lab_e_pat[k], "lab_e_hspc": ls.lab_e_hspc[k], **feats_pos[k]})
    for name, covs in (("A_full", ["ac"]), ("A_do", []), ("A_hspc", ["hc", "hk"])):
        if name in ("A_full",) and not has_ac:
            continue
        m = match_greedy(pa, qa, covs, rng)
        ok = m >= 0
        for k in np.nonzero(ok)[0]:
            q = pool.iloc[m[k]]
            f = pair_feats(cx, p, q.chrom, (q.i, q.i + 1), (q.j, q.j + 1), "loop")
            rows.append({"patient": p, "kind": "loop", "set": name, "pair_id": f"{p}:L{k}", "y": 0, "chrom": q.chrom,
                         "a_lo": q.i, "a_hi": q.i + 1, "b_lo": q.j, "b_hi": q.j + 1, "pair_type": types[k],
                         "lab_e_pat": q.e_pat, "lab_e_hspc": q.e_hspc3, **f})
        mq = qa.iloc[m[ok]]
        qual.append({"patient": p, "kind": "loop", "set": name, "n_pos": len(pos), "n_matched": int(ok.sum()),
                     "pool": len(pool), "smd_logdist": smd(np.log(pa.d[ok]), np.log(mq.d)),
                     "smd_k27ac_anchor_mean": smd(pa.ac[ok], mq.ac),
                     "smd_hspc_ctcf": smd(pa.hc[ok], mq.hc), "smd_hspc_k27ac": smd(pa.hk[ok], mq.hk),
                     "orient_equal_frac": float((pa.cls[ok].values == mq.cls.values).mean()) if ok.any() else np.nan,
                     "dist_med_pos": float(np.median(pa.d[ok])) if ok.any() else np.nan,
                     "dist_med_neg": float(np.median(mq.d)) if ok.any() else np.nan})
    rp = random_pairs(p, pos, cx, rng)
    ok = rp.i >= 0
    for r in rp[ok].itertuples():
        f = pair_feats(cx, p, r.chrom, (r.i, r.i + 1), (r.j, r.j + 1), "loop")
        rows.append({"patient": p, "kind": "loop", "set": "B_rand", "pair_id": f"{p}:L{r.pair}", "y": 0,
                     "chrom": r.chrom, "a_lo": r.i, "a_hi": r.i + 1, "b_lo": r.j, "b_hi": r.j + 1,
                     "pair_type": types[r.pair], "lab_e_pat": np.nan, "lab_e_hspc": np.nan, **f})
    qual.append({"patient": p, "kind": "loop", "set": "B_rand", "n_pos": len(pos), "n_matched": int(ok.sum()),
                 "pool": np.nan, "smd_logdist": 0.0, "smd_k27ac_anchor_mean": np.nan,
                 "orient_equal_frac": np.nan, "dist_med_pos": float(np.median(pa.d)),
                 "dist_med_neg": float(np.median(pa.d[ok.values]))})
    return rows, qual


def _loops_job(p):
    cx = Ctx()
    r, q = build_loops_patient(p, cx)
    return p, r, q



# ====================================================================== stripes
STRIPE_MAX = 150
NMAX = 320
WIN2 = 600


def stripe_positives(p: str) -> pd.DataFrame:
    S = read_tab("st4_aml_specific_stripes.csv", "stripe")
    S = S[(S.patient == p) & (S.chrom_anchor == S.chrom_extent) & S.chrom_anchor.isin(CHROMS)]
    d = pd.DataFrame({"chrom": S.chrom_anchor.values, "a_lo": (S.start_anchor // RES).values,
                      "a_hi": -(-S.end_anchor // RES).values, "e_lo": (S.start_extent // RES).values,
                      "e_hi": -(-S.end_extent // RES).values}).drop_duplicates()
    d["ra"], d["eb"] = d.a_hi - d.a_lo, d.e_hi - d.e_lo
    d["o"] = d.e_lo - d.a_lo
    right, left = d.e_lo >= d.a_hi, d.e_hi <= d.a_lo
    dmax = np.where(right, d.o + d.eb - 1, d.ra - 1 - d.o)
    d = d[(right | left) & (dmax <= NMAX)].reset_index(drop=True)
    if len(d) > STRIPE_MAX:
        d = d.sample(STRIPE_MAX, random_state=SEED).reset_index(drop=True)
    return d


def banded(clr, chrom: str) -> np.ndarray:
    """D[i, d] = log O/E of pixel (i, i + d), d = 0..NMAX; NaN where unmeasured."""
    n = int(clr.chromsizes[chrom]) // RES + 1
    D = np.full((n, NMAX + 1), np.nan, np.float32)
    dd = np.arange(NMAX + 1)
    s = 0
    while s < n - 2 * MARGIN:
        size = int(clr.chromsizes[chrom])
        M = np.asarray(clr.matrix(balance=False).fetch(f"{chrom}:{s * RES}-{min((s + WIN2) * RES, size)}"), np.float64)
        M = np.where(M == 0, np.nan, M)
        if M.shape[0] < WIN2:
            P = np.full((WIN2, WIN2), np.nan)
            P[:M.shape[0], :M.shape[1]] = M
            M = P
        r0 = MARGIN if s > 0 else 0
        own = np.arange(r0, MARGIN + STRIDE)
        own = own[s + own < n]
        D[s + own] = M[own[:, None], own[:, None] + dd[None, :]]
        s += STRIDE
    return D


def shift_scores(D: np.ndarray, ra: int, eb: int, o: int) -> np.ndarray:
    """Mean log O/E of the stripe rectangle (rows a..a+ra-1, columns a+o..a+o+eb-1) for every
    anchor start a (NaN where the rectangle leaves the chromosome or nothing is measured)."""
    n = D.shape[0]
    P = 800
    Dp = np.pad(D, ((P, P), (0, 0)), constant_values=np.nan)
    tot = np.zeros(n)
    cnt = np.zeros(n)
    for k in range(ra):
        for t in range(eb):
            if o >= ra:
                ro, dd = k, o + t - k
            else:
                ro, dd = o + t, k - o - t
            x = Dp[P + ro:P + ro + n, dd]
            m = np.isfinite(x)
            tot += np.where(m, x, 0.0)
            cnt += m
    out = np.where(cnt >= 0.5 * ra * eb, tot / np.maximum(cnt, 1), np.nan)
    return out.astype(np.float32)


def stripes_hic_one(p: str) -> str:
    import cooler
    dest = CACHE / f"stripes_{p}.pkl"
    if dest.exists():
        return p
    import pickle
    pos = stripe_positives(p)
    cp = cooler.Cooler(str(OE / f"xu_{p}_proper_10kb.cool"))
    ch = cooler.Cooler(str(CONS / "hspccons_10kb.cool"))
    allp = read_tab("st4_aml_specific_stripes.csv", "stripe")
    allp = allp[allp.patient == p]
    pos["sp0"] = np.nan
    pos["sh0"] = np.nan
    arrs = {}
    for c, g in pos.groupby("chrom"):
        Dp, Dh = banded(cp, c), banded(ch, c)
        for r in g.itertuples():
            sp = shift_scores(Dp, r.ra, r.eb, r.o)
            sh = shift_scores(Dh, r.ra, r.eb, r.o)
            pos.loc[r.Index, "sp0"] = sp[r.a_lo]
            pos.loc[r.Index, "sh0"] = sh[r.a_lo]
            arrs[r.Index] = (sp.astype(np.float16), sh.astype(np.float16))
    pickle.dump({"pos": pos, "arrs": arrs, "all_anchor": allp[["chrom_anchor", "start_anchor"]].assign(
        a=lambda x: x.start_anchor // RES).drop(columns="start_anchor").values}, open(dest, "wb"))
    return p


def stage_stripes_hic(patients: list[str], workers: int) -> None:
    with ProcessPoolExecutor(workers) as ex:
        for p in ex.map(stripes_hic_one, [q for q in patients if q in CT_PATIENTS]):
            log(f"[stripes_hic] {p}")


def build_stripes_patient(p: str, cx: Ctx) -> tuple[list[dict], list[dict]]:
    import pickle
    rng = np.random.default_rng(SEED + 7 + int(p))
    Z = pickle.load(open(CACHE / f"stripes_{p}.pkl", "rb"))
    pos, arrs = Z["pos"], Z["arrs"]
    v = np.load(CACHE / f"valid_{p}.npz")
    has_ac = cx.trk(p, "H3K27ac") is not None
    thr = float(np.nanpercentile(pos.sp0, 25))
    allanch = {}
    for c, a in Z["all_anchor"]:
        allanch.setdefault(c, []).append(int(a))
    # positive rows
    rows, qual = [], []
    pf, ptype, pac, pcls = [], [], [], []
    for r in pos.itertuples():
        f = pair_feats(cx, p, r.chrom, (r.a_lo, r.a_hi), (r.e_lo, r.e_hi), "stripe")
        pf.append(f)
        pac.append(f["own_H3K27ac_mean_min"] if False else np.nan)
        pcls.append(f["_orient"])
    # anchor K27ac mean for matching (anchor region only)
    def anchor_ac(c, a, ra):
        T = cx.trk(p, "H3K27ac")
        return _nanmean(T[c + "|mean"][a:a + ra]) if T is not None else np.nan
    def anchor_h(c, a, ra, mark):
        T = cx.trk("hspc", mark)
        return _nanmean(T[c + "|mean"][a:a + ra])
    for k, r in enumerate(pos.itertuples()):
        rows.append({"patient": p, "kind": "stripe", "set": "pos", "pair_id": f"{p}:S{k}", "y": 1, "chrom": r.chrom,
                     "a_lo": r.a_lo, "a_hi": r.a_hi, "b_lo": r.e_lo, "b_hi": r.e_hi, "pair_type": pf[k]["_type"],
                     "lab_e_pat": r.sp0, "lab_e_hspc": r.sh0, **pf[k]})
    used = set()
    stats = {"A_full": [], "A_do": [], "A_hspc": [], "B_rand": []}
    for k, r in enumerate(pos.itertuples()):
        sp, sh = (a.astype(float) for a in arrs[r.Index])
        n = len(sp)
        vv = v[r.chrom]
        rowsok = np.zeros(n, bool)
        # rectangle rows and the extent's end bins mappable
        for a in range(0):
            pass
        base = np.arange(n)
        okpos = np.isfinite(sp) & np.isfinite(sh)
        lo_bin = np.minimum(base, base + r.o)
        hi_bin = np.maximum(base + r.ra, base + r.o + r.eb)
        okpos &= (lo_bin >= 0) & (hi_bin < n)
        okpos &= vv[np.clip(base, 0, n - 1)] & vv[np.clip(base + r.o, 0, n - 1)]
        near = np.zeros(n, bool)
        for a0 in allanch.get(r.chrom, []):
            near[max(0, a0 - 8):a0 + 9] = True
        shared = okpos & ~near & (sp >= thr) & (sh >= thr)
        cand = np.nonzero(shared)[0]
        cand = np.array([a for a in cand if (r.chrom, a) not in used])
        dirn = 1 if r.o > 0 else -1
        # orientation class of candidates (anchor strand facing the extent)
        if len(cand):
            mo = cx.motif[r.chrom]
            sc = np.array([mo["score"][a:a + r.ra].max() for a in cand])
            kb = np.array([int(np.argmax(mo["score"][a:a + r.ra])) for a in cand])
            st = np.array([mo["strand"][a + kk] for a, kk in zip(cand, kb)])
            cls = np.where(sc <= 0, "none", np.where(st == dirn, "conv", "div"))
            ac = np.array([anchor_ac(r.chrom, a, r.ra) for a in cand])
            hcc = np.array([anchor_h(r.chrom, a, r.ra, "CTCF") for a in cand])
            hkk = np.array([anchor_h(r.chrom, a, r.ra, "H3K27ac") for a in cand])
        for name, use_ac in (("A_full", True), ("A_do", False), ("A_hspc", None)):
            if name == "A_full" and not has_ac:
                continue
            if not len(cand):
                continue
            sel = np.nonzero(cls == pcls[k])[0]
            if use_ac is None:
                if not len(sel):
                    continue
                p1, p2 = anchor_h(r.chrom, r.a_lo, r.ra, "CTCF"), anchor_h(r.chrom, r.a_lo, r.ra, "H3K27ac")
                c1, c2 = 0.3 * np.nanstd(hcc), 0.3 * np.nanstd(hkk)
                ok_ = (np.abs(hcc[sel] - p1) <= c1) & (np.abs(hkk[sel] - p2) <= c2)
                if not ok_.any():
                    continue
                sel = sel[ok_]
                pick = sel[int(np.argmin(np.abs(hcc[sel] - p1) / c1 + np.abs(hkk[sel] - p2) / c2))]
            elif use_ac:
                pa_ = anchor_ac(r.chrom, r.a_lo, r.ra)
                sel = sel[np.isfinite(ac[sel])]
                if not len(sel):
                    continue
                dif = np.abs(ac[sel] - pa_)
                if dif.min() > 0.3 * np.nanstd(ac):
                    continue
                pick = sel[int(np.argmin(dif))]
            else:
                if not len(sel):
                    continue
                pick = sel[int(rng.integers(len(sel)))]
            a = int(cand[pick])
            f = pair_feats(cx, p, r.chrom, (a, a + r.ra), (a + r.o, a + r.o + r.eb), "stripe")
            rows.append({"patient": p, "kind": "stripe", "set": name, "pair_id": f"{p}:S{k}", "y": 0, "chrom": r.chrom,
                         "a_lo": a, "a_hi": a + r.ra, "b_lo": a + r.o, "b_hi": a + r.o + r.eb,
                         "pair_type": pf[k]["_type"], "lab_e_pat": sp[a], "lab_e_hspc": sh[a], **f})
            stats[name].append((k, anchor_ac(r.chrom, r.a_lo, r.ra), anchor_ac(r.chrom, a, r.ra), 1.0))
            if use_ac is not None and use_ac:
                used.add((r.chrom, a))
        # random geometry-matched (distance and extent only)
        rnd = np.nonzero(okpos & ~near)[0]
        if len(rnd):
            a = int(rng.choice(rnd))
            f = pair_feats(cx, p, r.chrom, (a, a + r.ra), (a + r.o, a + r.o + r.eb), "stripe")
            rows.append({"patient": p, "kind": "stripe", "set": "B_rand", "pair_id": f"{p}:S{k}", "y": 0,
                         "chrom": r.chrom, "a_lo": a, "a_hi": a + r.ra, "b_lo": a + r.o, "b_hi": a + r.o + r.eb,
                         "pair_type": pf[k]["_type"], "lab_e_pat": sp[a], "lab_e_hspc": sh[a], **f})
            stats["B_rand"].append((k, anchor_ac(r.chrom, r.a_lo, r.ra), anchor_ac(r.chrom, a, r.ra), 1.0))
    for name, st in stats.items():
        if not st:
            continue
        a = np.array(st)
        qual.append({"patient": p, "kind": "stripe", "set": name, "n_pos": len(pos), "n_matched": len(st),
                     "pool": np.nan, "smd_logdist": 0.0, "smd_k27ac_anchor_mean": smd(a[:, 1], a[:, 2]),
                     "orient_equal_frac": np.nan, "dist_med_pos": np.nan, "dist_med_neg": np.nan})
    return rows, qual


def _stripes_job(p):
    cx = Ctx()
    r, q = build_stripes_patient(p, cx)
    return p, r, q



# ====================================================================== subtypes
def infer_subtypes() -> pd.DataFrame:
    """Subtype of a patient from its structural variants (Suppl. Table 5, hg38 coordinates of the
    recurrent fusions): BCR-ABL1 (chr22 BCR / chr9 ABL1 translocation), inv(16) (chr16 inversion
    between the MYH11 and CBFB loci), t(8;21) (RUNX1T1 / RUNX1), KMT2A-rearranged (chr11 118.4 to
    118.6 Mb). Mutation subtypes (TET2, CEBPA, NPM1, FLT3-ITD ...) are not assigned to patients in
    the paper's tables; patients without a recurrent fusion are their own group."""
    S = pd.read_csv(XT / "st5_svs.csv", dtype={"patient": str})
    S["patient"] = S.patient.str.lstrip("#").map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)

    def near(c, x, chrom, lo, hi):
        return (c == chrom) & (x >= lo * 1e6) & (x <= hi * 1e6)

    flags = {}
    for p, g in S.groupby("patient"):
        f = set()
        for r in g.itertuples():
            for (c1, x1, c2, x2) in ((r.chrom1, r.pos1, r.chrom2, r.pos2), (r.chrom2, r.pos2, r.chrom1, r.pos1)):
                if near(c1, x1, "chr22", 23.0, 23.6) and near(c2, x2, "chr9", 130.4, 131.2):
                    f.add("BCR-ABL1")
                if r.sv_type == "inversion" and near(c1, x1, "chr16", 15.5, 16.2) and near(c2, x2, "chr16", 66.9, 67.4):
                    f.add("inv16")
                if near(c1, x1, "chr8", 91.9, 92.4) and near(c2, x2, "chr21", 34.5, 36.5):
                    f.add("t(8;21)")
                if near(c1, x1, "chr11", 118.4, 118.6) and c2 != "chr11":
                    f.add("KMT2A-r")
        flags[p] = sorted(f)
    rows = []
    for p in LOOP_PATIENTS:
        f = flags.get(p, [])
        rows.append({"patient": p, "sv_flags": ";".join(f), "subtype": f[0] if len(f) == 1 else
                     (("multi:" + "+".join(f)) if f else f"unclassified_{p}")})
    return pd.DataFrame(rows)


# ====================================================================== dataset assembly
def stage_build(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, qual = [], []
    with ProcessPoolExecutor(workers) as ex:
        for p, r, q in ex.map(_loops_job, LOOP_PATIENTS):
            rows += r
            qual += q
            log(f"[build] loops {p}: {len(r)} rows")
        for p, r, q in ex.map(_stripes_job, [x for x in CT_PATIENTS if (CACHE / f"stripes_{x}.pkl").exists()]):
            rows += r
            qual += q
            log(f"[build] stripes {p}: {len(r)} rows")
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "features.csv.gz", index=False)
    pd.DataFrame(qual).to_csv(OUT / "matching_quality.csv", index=False)
    infer_subtypes().to_csv(OUT / "subtypes.csv", index=False)
    log(f"[build] features.csv.gz {F.shape}")


# ====================================================================== models
GROUPS = ["own", "hspc", "diff", "meth", "motif", "dist", "gene"]
NONFEAT = {"patient", "kind", "set", "pair_id", "y", "chrom", "a_lo", "a_hi", "b_lo", "b_hi", "pair_type",
           "lab_e_pat", "lab_e_hspc"}


def feature_cols(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in NONFEAT and not c.startswith("_")]


def group_of(c: str) -> str:
    return c.split("_")[0]


def fine_group(c: str) -> str:
    g = group_of(c)
    if g in ("own", "hspc", "diff"):
        return f"{g}_{c.split('_')[1]}"
    return g


# name, kind, negative set, patients, description
EXPERIMENTS = [
    ("loops_all4_A", "loop", "A_full", CTCF_PATIENTS, "own K27ac, K27me3, CTCF, methylation; negatives matched on distance, orientation, K27ac"),
    ("loops_all4_Ado", "loop", "A_do", CTCF_PATIENTS, "same features; negatives matched on distance and orientation only"),
    ("loops_all4_B", "loop", "B_rand", CTCF_PATIENTS, "same features; random pairs matched on distance"),
    ("loops_all4_Ahspc", "loop", "A_hspc", CTCF_PATIENTS, "same features; negatives matched on distance, orientation and the anchors' HSPC CTCF and H3K27ac"),
    ("loops_marks2_Ahspc", "loop", "A_hspc", CT_PATIENTS, "K27ac and K27me3 only; 10 patients; negatives matched on HSPC CTCF and H3K27ac"),
    ("loops_meth_Ahspc", "loop", "A_hspc", [p for p in LOOP_PATIENTS if p not in ("1360", "1916")], "methylation, motif, gene, distance; 18 patients; negatives matched on HSPC CTCF and H3K27ac"),
    ("loops_marks2_A", "loop", "A_full", CT_PATIENTS, "K27ac and K27me3 only (no CTCF, no methylation); 10 patients"),
    ("loops_marks2_Ado", "loop", "A_do", CT_PATIENTS, "K27ac and K27me3 only; 10 patients; no K27ac matching"),
    ("loops_meth_Ado", "loop", "A_do", [p for p in LOOP_PATIENTS if p not in ("1360", "1916")], "methylation, motif, gene, distance only; 18 patients with WGBS"),
    ("loops_meth_B", "loop", "B_rand", [p for p in LOOP_PATIENTS if p not in ("1360", "1916")], "methylation, motif, gene, distance only; 18 patients; random pairs"),
    ("stripes_all4_A", "stripe", "A_full", CTCF_PATIENTS, "stripes; negatives shared stripes matched on geometry, orientation, anchor K27ac"),
    ("stripes_all4_Ado", "stripe", "A_do", CTCF_PATIENTS, "stripes; shared stripes matched on geometry and orientation"),
    ("stripes_all4_Ahspc", "stripe", "A_hspc", CTCF_PATIENTS, "stripes; shared stripes matched on geometry, orientation, anchor HSPC CTCF and H3K27ac"),
    ("stripes_all4_B", "stripe", "B_rand", CTCF_PATIENTS, "stripes; random shifted rectangles (geometry only)"),
]


def exp_data(F: pd.DataFrame, kind: str, negset: str, pats: list[str]):
    d = F[(F.kind == kind) & F.patient.isin(pats)]
    have = set(d[d.set == negset].pair_id)
    d = d[((d.set == negset) | ((d.set == "pos") & d.pair_id.isin(have)))].reset_index(drop=True)
    cols = []
    for c in feature_cols(F):
        if d[c].isna().groupby(d.patient).all().any():
            continue            # track missing for at least one patient
        if d[c].isna().mean() > 0.9:
            continue
        cols.append(c)
    return d, cols


def make_model(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=2000))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                          l2_regularization=2.0, random_state=0)


def _fit_pred(kind, X, y, tr, te):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        m = make_model(kind)
        m.fit(X[tr], y[tr])
        return te, m.predict_proba(X[te])[:, 1]


def cv_scores(kind: str, X: np.ndarray, y: np.ndarray, folds: list, n_jobs: int = 1) -> np.ndarray:
    """Out-of-fold scores; folds = list of (train_idx, test_idx)."""
    from joblib import Parallel, delayed
    out = np.full(len(y), np.nan)
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_pred)(kind, X, y, tr, te) for tr, te in folds)
    for te, pr in res:
        out[te] = pr
    return out


def row_keys(d: pd.DataFrame) -> list:
    """Locus keys of each row: chromosome index * 1e7 + 10 kb bin of every anchor bin."""
    ci = d.chrom.map({c: k for k, c in enumerate(CHROMS)}).values
    out = []
    for c, a0, a1, b0, kind in zip(ci, d.a_lo.values, d.a_hi.values, d.b_lo.values, d.kind.values):
        bins = [a0, b0] if kind == "loop" else list(range(a0, a1))
        out.append({int(c) * 10_000_000 + int(b) for b in bins})
    return out


def make_folds(patients: np.ndarray, group_of_patient: dict | None = None, keys: list | None = None,
               pad: int = 2) -> list:
    """Leave-one-patient-out, or leave-group-out when group_of_patient maps patient -> group.
    With `keys` the folds are also locus-disjoint: a training row is dropped when one of its anchor
    bins lies within `pad` bins of an anchor bin of a test row (recurrent loops of other patients
    share the anchors and, through the HSPC features, identify the locus)."""
    g = np.array([group_of_patient[p] for p in patients]) if group_of_patient else patients
    folds = []
    for u in np.unique(g):
        te = np.nonzero(g == u)[0]
        tr = np.nonzero(g != u)[0]
        if keys is not None:
            S = set()
            for i in te:
                for k in keys[i]:
                    S.update(range(k - pad, k + pad + 1))
            tr = np.array([i for i in tr if not (keys[i] & S)], dtype=int)
        folds.append((tr, te))
    return folds


def metrics(y, s, pats, bootstrap=1000, seed=0) -> dict:
    """Macro AUROC (mean of per-patient AUROCs), pooled AUROC and AUPRC with patient-bootstrap
    95 percent intervals (patients resampled with replacement; rows of a patient move together)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    up = np.unique(pats)
    idx = {p: np.nonzero(pats == p)[0] for p in up}
    per = {p: roc_auc_score(y[i], s[i]) if 0 < y[i].sum() < len(i) else np.nan for p, i in idx.items()}
    rng = np.random.default_rng(seed)
    macro, pool_a, pool_p = [], [], []
    for _ in range(bootstrap):
        pick = rng.choice(up, len(up))
        ii = np.concatenate([idx[p] for p in pick])
        macro.append(np.nanmean([per[p] for p in pick]))
        if 0 < y[ii].sum() < len(ii):
            pool_a.append(roc_auc_score(y[ii], s[ii]))
            pool_p.append(average_precision_score(y[ii], s[ii]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return {"auroc_macro": float(np.nanmean(list(per.values()))), "auroc_macro_lo": q(macro)[0], "auroc_macro_hi": q(macro)[1],
            "auroc_pooled": float(roc_auc_score(y, s)), "auroc_pooled_lo": q(pool_a)[0], "auroc_pooled_hi": q(pool_a)[1],
            "auprc_pooled": float(average_precision_score(y, s)), "auprc_lo": q(pool_p)[0], "auprc_hi": q(pool_p)[1],
            "prevalence": float(y.mean()), "per_patient": per}


def macro_auc(y, s, pats) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    per = []
    for p in np.unique(pats):
        i = pats == p
        if 0 < y[i].sum() < i.sum():
            per.append(roc_auc_score(y[i], s[i]))
    return float(np.mean(per)), float(roc_auc_score(y, s))


def shuffle_null(kind, X, y, pats, folds, reps, n_jobs, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        ys = y.copy()
        for p in np.unique(pats):
            i = np.nonzero(pats == p)[0]
            ys[i] = rng.permutation(ys[i])
        s = cv_scores(kind, X, ys, folds, n_jobs)
        out.append(macro_auc(ys, s, pats))
    return np.array(out)



def run_experiment(F, sub, name, kind, negset, pats, desc, args):
    d, cols = exp_data(F, kind, negset, pats)
    if d.empty:
        return None
    y = d.y.values.astype(int)
    pt = np.array([donor_of(p, args.group_donors) for p in d.patient.values])   # fold and bootstrap unit
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    log(f"[model] {name}: {len(pats)} patients, {len(np.unique(pt))} donor units, {n_pos} positives, {n_neg} negatives, {len(cols)} features")
    gmap = {}
    for q, st in zip(sub.patient, sub.subtype):        # subtype per fold unit (donor); a unit takes the
        u = donor_of(q, args.group_donors)             # first recognised fusion among its samples
        if u not in gmap or (gmap[u].startswith("unclassified") and not st.startswith("unclassified")):
            gmap[u] = st if not st.startswith("unclassified") else f"unclassified_{u}"
    keys = row_keys(d)
    folds = {"lopo": make_folds(pt, keys=keys), "lopo_plain": make_folds(pt), "lso": make_folds(pt, gmap, keys=keys)}
    X = d[cols].values.astype(float)
    groups = {c: group_of(c) for c in cols}
    fine = {c: fine_group(c) for c in cols}
    # exact feature-group column sets present in this experiment
    G = sorted(set(groups.values()))
    FG = sorted(set(fine.values()))
    summ, per_pat, imp, preds, bytype = [], [], [], {}, []
    combo_rows = []
    dist_cols = [i for i, c in enumerate(cols) if c.startswith("dist_")]
    null_cache = {}
    for model in ("lr", "gb"):
        for cv in ("lopo", "lopo_plain", "lso"):
            fo = folds[cv]
            s_full = cv_scores(model, X, y, fo, args.jobs)
            m = metrics(y, s_full, pt, args.boot)
            s_dist = cv_scores("lr", X[:, dist_cols], y, fo, args.jobs)
            md = metrics(y, s_dist, pt, args.boot)
            reps = args.null_reps_lr if model == "lr" else args.null_reps_gb
            nl = shuffle_null(model, X, y, pt, fo, reps, args.jobs, seed=1)
            # paired patient bootstrap of full minus distance-only (macro AUROC)
            pf, pdist = m["per_patient"], md["per_patient"]
            up = np.array(sorted(pf))
            rng = np.random.default_rng(5)
            dd = np.array([np.nanmean([pf[q] - pdist[q] for q in rng.choice(up, len(up))]) for _ in range(args.boot)])
            rec = {"experiment": name, "kind": kind, "negatives": negset, "model": model, "cv": cv,
                   "n_patients": len(pats), "n_donors": len(np.unique(pt)), "n_pos": n_pos, "n_neg": n_neg, "n_features": len(cols),
                   **{k: v for k, v in m.items() if k != "per_patient"},
                   "dist_auroc_macro": md["auroc_macro"], "dist_auroc_pooled": md["auroc_pooled"],
                   "shuffle_macro_mean": float(nl[:, 0].mean()), "shuffle_macro_p97.5": float(np.percentile(nl[:, 0], 97.5)),
                   "shuffle_macro_max": float(nl[:, 0].max()), "shuffle_pooled_mean": float(nl[:, 1].mean()),
                   "shuffle_pooled_p97.5": float(np.percentile(nl[:, 1], 97.5)), "shuffle_reps": reps,
                   "delta_vs_dist_macro": float(np.nanmean([pf[q] - pdist[q] for q in up])),
                   "delta_vs_dist_lo": float(np.percentile(dd, 2.5)), "delta_vs_dist_hi": float(np.percentile(dd, 97.5)),
                   "desc": desc}
            summ.append(rec)
            for q, v in pf.items():
                per_pat.append({"experiment": name, "model": model, "cv": cv, "patient": q, "auroc": v,
                                "dist_auroc": pdist[q], "n_pos": int(y[pt == q].sum())})
            preds[(model, cv)] = s_full
            # by pair type
            for t in sorted(set(d.pair_type)):
                i = (d.pair_type == t).values
                if y[i].sum() < 15 or (1 - y[i]).sum() < 15:
                    continue
                mt = metrics(y[i], s_full[i], pt[i], args.boot)
                mdt = metrics(y[i], s_dist[i], pt[i], args.boot)
                bytype.append({"experiment": name, "model": model, "cv": cv, "pair_type": t,
                               "n_pos": int(y[i].sum()), "n_patients": int(len(np.unique(pt[i]))),
                               **{k: v for k, v in mt.items() if k not in ("per_patient",)},
                               "dist_auroc_pooled": mdt["auroc_pooled"]})
            if cv == "lopo":
                # feature-group importance: group alone, all but the group
                for gname, members in (("group", G), ("mark", FG)):
                    for g in members:
                        key = (lambda c: groups[c]) if gname == "group" else (lambda c: fine[c])
                        only = [i for i, c in enumerate(cols) if key(c) == g]
                        s_only = cv_scores(model, X[:, only], y, fo, args.jobs)
                        mo_ = metrics(y, s_only, pt, 300)
                        rec_i = {"experiment": name, "model": model, "level": gname, "feature_group": g,
                                 "n_features": len(only), "only_auroc_macro": mo_["auroc_macro"],
                                 "only_lo": mo_["auroc_macro_lo"], "only_hi": mo_["auroc_macro_hi"],
                                 "only_auroc_pooled": mo_["auroc_pooled"]}
                        if gname == "group" and len(G) > 1:
                            rest = [i for i, c in enumerate(cols) if key(c) != g]
                            s_rest = cv_scores(model, X[:, rest], y, fo, args.jobs)
                            mr = metrics(y, s_rest, pt, 300)
                            rec_i.update({"without_auroc_macro": mr["auroc_macro"], "without_lo": mr["auroc_macro_lo"],
                                          "without_hi": mr["auroc_macro_hi"],
                                          "drop_vs_full": m["auroc_macro"] - mr["auroc_macro"]})
                        imp.append(rec_i)
    # incremental value of the patient's own data over the HSPC state (leave-one-patient-out, locus-disjoint)
    combos = {"hspc": ["hspc"], "hspc+own": ["hspc", "own"], "hspc+diff": ["hspc", "diff"],
              "hspc+meth": ["hspc", "meth"], "hspc+motif+gene": ["hspc", "motif", "gene"],
              "hspc+own+meth+motif+gene": ["hspc", "own", "meth", "motif", "gene"], "own": ["own"],
              "own+meth+motif+gene": ["own", "meth", "motif", "gene"], "all": G}
    for model in ("lr", "gb"):
        base = None
        for cname, gs in combos.items():
            gs = [g for g in gs if g in G]
            if not gs or (cname != "hspc" and base is None and cname != "all"):
                pass
            cidx = [i for i, c in enumerate(cols) if groups[c] in gs]
            if not cidx:
                continue
            sc = cv_scores(model, X[:, cidx], y, folds["lopo"], args.jobs)
            mc = metrics(y, sc, pt, 300)
            if cname == "hspc":
                base = mc["per_patient"]
            rec_c = {"experiment": name, "model": model, "combo": cname, "groups": "+".join(gs),
                     "auroc_macro": mc["auroc_macro"], "lo": mc["auroc_macro_lo"], "hi": mc["auroc_macro_hi"],
                     "auroc_pooled": mc["auroc_pooled"]}
            if base is not None and cname != "hspc":
                up2 = np.array(sorted(base))
                rg = np.random.default_rng(9)
                dl = np.array([np.nanmean([mc["per_patient"][q] - base[q] for q in rg.choice(up2, len(up2))])
                               for _ in range(500)])
                rec_c.update({"delta_vs_hspc": float(np.nanmean([mc["per_patient"][q] - base[q] for q in up2])),
                              "delta_lo": float(np.percentile(dl, 2.5)), "delta_hi": float(np.percentile(dl, 97.5))})
            combo_rows.append(rec_c)
    # univariate: mean within-patient AUROC of every feature
    from sklearn.metrics import roc_auc_score
    uni = []
    for j, c in enumerate(cols):
        per = []
        for q in np.unique(pt):
            i = pt == q
            x = X[i, j]
            ok = np.isfinite(x)
            if ok.sum() > 20 and 0 < y[i][ok].sum() < ok.sum() and np.ptp(x[ok]) > 0:
                per.append(roc_auc_score(y[i][ok], x[ok]))
        if per:
            pa = np.array(per)
            uni.append({"experiment": name, "feature": c, "group": group_of(c), "mean_patient_auroc": float(pa.mean()),
                        "n_patients": len(pa), "n_patients_above_half": int((pa > 0.5).sum())})
    pr = d[["patient", "pair_id", "set", "y", "pair_type"]].copy()
    pr.insert(1, "donor", pt)
    for (mname, cv), sc in preds.items():
        pr[f"{mname}_{cv}"] = sc
    pr.insert(0, "experiment", name)
    return summ, per_pat, imp, bytype, uni, pr, combo_rows


def stage_model(args) -> None:
    F = pd.read_csv(OUT / "features.csv.gz", low_memory=False, dtype={"patient": str})
    sub = pd.read_csv(OUT / "subtypes.csv", dtype={"patient": str})
    names = set(args.experiments) if args.experiments else None
    OUT.mkdir(parents=True, exist_ok=True)
    S, P, I, T, U, PR, CB = [], [], [], [], [], [], []
    for name, kind, negset, pats, desc in EXPERIMENTS:
        if names and name not in names:
            continue
        if args.exclude_healthy:
            pats = [p for p in pats if p not in HEALTHY]
        assert not (args.exclude_healthy and set(F[F.patient.isin(HEALTHY)].patient)), "healthy donor in features"
        r = run_experiment(F, sub, name, kind, negset, pats, desc, args)
        if r is None:
            continue
        for lst, x in zip((S, P, I, T, U, PR, CB), r):
            if isinstance(x, pd.DataFrame):
                lst.append(x)
            else:
                lst.append(pd.DataFrame(x))
        suffix = "" if not names else "_" + "_".join(sorted(names))
        for nm, lst in (("summary", S), ("per_patient", P), ("group_importance", I), ("by_type", T),
                        ("univariate", U), ("oof_predictions", PR), ("incremental", CB)):
            pd.concat(lst, ignore_index=True).to_csv(OUT / (f"{nm}{suffix}.csv" + (".gz" if nm == "oof_predictions" else "")),
                                                     index=False)
    log("[model] done")


# ====================================================================== figures
def stage_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = pd.read_csv(OUT / "summary.csv")
    I = pd.read_csv(OUT / "group_importance.csv")
    P = pd.read_csv(OUT / "per_patient.csv", dtype={"patient": str})
    Q = pd.read_csv(OUT / "matching_quality.csv", dtype={"patient": str})
    # 1: macro AUROC with patient-bootstrap interval, controls
    d = S[(S.cv == "lopo")]
    exps = list(dict.fromkeys(d.experiment))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(exps) * 2 + 1.5))
    y = 0
    ticks = []
    for e in exps:
        for model, col in (("lr", "#1f77b4"), ("gb", "#d62728")):
            r = d[(d.experiment == e) & (d.model == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.auroc_macro, y, xerr=[[r.auroc_macro - r.auroc_macro_lo], [r.auroc_macro_hi - r.auroc_macro]],
                        fmt="o", color=col, label=model if e == exps[0] else None)
            ax.plot(r["shuffle_macro_p97.5"], y, "|", color="grey", ms=12)
            ax.plot(r.dist_auroc_macro, y, "x", color="black", ms=6)
            ticks.append((y, f"{e} {model}"))
            y += 1
        y += 0.5
    ax.axvline(0.5, color="grey", lw=0.8)
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks], fontsize=7)
    ax.set_xlabel("leave-one-patient-out AUROC (mean over patients; bar: patient bootstrap 95%; | shuffle 97.5%; x distance only)")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_auroc_summary.png", dpi=150)
    plt.close(fig)
    # 2: group importance (only-group AUROC) for the main experiments
    ex2 = [e for e in exps if e in ("loops_all4_A", "loops_all4_Ado", "stripes_all4_Ado", "loops_all4_B")]
    fig, axes = plt.subplots(1, max(1, len(ex2)), figsize=(4.5 * max(1, len(ex2)), 5), squeeze=False)
    for ax, e in zip(axes[0], ex2):
        g = I[(I.experiment == e) & (I.level == "mark") & (I.model == "gb")].sort_values("only_auroc_macro")
        ax.barh(g.feature_group, g.only_auroc_macro - 0.5, left=0.5, color="#d62728")
        ax.errorbar(g.only_auroc_macro, range(len(g)), xerr=[g.only_auroc_macro - g.only_lo, g.only_hi - g.only_auroc_macro],
                    fmt="none", color="k", lw=0.8)
        ax.axvline(0.5, color="grey", lw=0.8)
        ax.set_title(e + " (gradient boosting, one group alone)", fontsize=8)
        ax.set_xlabel("LOPO AUROC")
    fig.tight_layout()
    fig.savefig(OUT / "fig_group_importance.png", dpi=150)
    plt.close(fig)
    # 3: per patient
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, e in enumerate([e for e in exps if e in ("loops_all4_A", "loops_all4_Ado", "loops_all4_B", "stripes_all4_Ado")]):
        g = P[(P.experiment == e) & (P.model == "gb") & (P.cv == "lopo")]
        ax.scatter(np.full(len(g), k) + np.random.default_rng(0).uniform(-.15, .15, len(g)), g.auroc, s=14)
        ax.plot([k - .3, k + .3], [g.auroc.mean()] * 2, "k")
        ax.set_xticks(range(4))
    ax.set_xticklabels([e for e in exps if e in ("loops_all4_A", "loops_all4_Ado", "loops_all4_B", "stripes_all4_Ado")], fontsize=7)
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set_ylabel("AUROC of held-out patient (gradient boosting)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_patient.png", dpi=150)
    plt.close(fig)
    # 4: matching balance
    fig, ax = plt.subplots(figsize=(6, 4))
    for st, col in (("A_full", "C0"), ("A_do", "C1")):
        g = Q[(Q.set == st) & (Q.kind == "loop")]
        ax.scatter(g.smd_logdist, g.smd_k27ac_anchor_mean, label=st, color=col)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("standardised mean difference, log distance")
    ax.set_ylabel("standardised mean difference, anchor H3K27ac")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_matching_balance.png", dpi=150)
    plt.close(fig)
    log("[figures] done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["hic"])
    ap.add_argument("--patients", nargs="+", default=LOOP_PATIENTS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=6, help="joblib workers inside the model stage")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--null-reps-lr", type=int, default=30)
    ap.add_argument("--null-reps-gb", type=int, default=15)
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--exclude-healthy", action=argparse.BooleanOptionalAction, default=True,
                    help="drop the healthy PBMC donors 103, 673, 576 from every patient list")
    ap.add_argument("--group-donors", action=argparse.BooleanOptionalAction, default=True,
                    help="folds and bootstrap over donor units: {168, 496} and {260, 424} are one unit")
    ap.add_argument("--out", default=None, help="output folder (default results/aml_loop_classifier_v2, "
                    "_legacy suffix when a flag is switched off; the section 47 folder is never the default)")
    a = ap.parse_args()
    global OUT
    OUT = REPO / "results" / (a.out or ("aml_loop_classifier_v2" if a.exclude_healthy and a.group_donors
                                         else "aml_loop_classifier_v2_legacy"))
    if a.exclude_healthy:
        for lst in (CT_PATIENTS, CTCF_PATIENTS, WGBS_ONLY, LOOP_PATIENTS):
            lst[:] = [p for p in lst if p not in HEALTHY]
        a.patients = [p for p in a.patients if p not in HEALTHY]
    if "hic" in a.stage:
        stage_hic(a.patients, a.workers)
    if "tracks" in a.stage:
        stage_tracks(a.workers)
    if "motifs" in a.stage:
        stage_motifs(a.workers)
    if "stripes_hic" in a.stage:
        stage_stripes_hic(a.patients, a.workers)
    if "build" in a.stage:
        stage_build(a.workers)
    if "subtypes" in a.stage:
        print(infer_subtypes())
    if "model" in a.stage:
        stage_model(a)
    if "figures" in a.stage:
        stage_figures()
    if "testloops" in a.stage:
        for p in a.patients:
            _, r, q = _loops_job(p)
            df = pd.DataFrame(r)
            print(df.groupby("set").size().to_dict(), df[df.set == "pos"]._type.value_counts().to_dict())
            print(pd.DataFrame(q).T)
    if "teststripes" in a.stage:
        for p in a.patients:
            _, r, q = _stripes_job(p)
            df = pd.DataFrame(r)
            print(df.groupby("set").size().to_dict(), df[df.set == "pos"]._type.value_counts().to_dict())
            print(pd.DataFrame(q).T)


if __name__ == "__main__":
    main()
