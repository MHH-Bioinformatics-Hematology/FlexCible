"""Is there information in ATAC-seq and RNA-seq of the patient that predicts AML-specific loops and stripes
that are private to the patient? (registry section on private-loop information)

Data (Xu et al. 2022, GSE152136 superseries): ATAC-seq GSE152134 (10 bp RPGC bigWig + MACS2 peaks),
RNA-seq GSE152096 (RSEM gene quantification, GENCODE v24). Healthy contrast: three PBMC donors of the same
cohort and a CD34+ HSPC reference (ATAC: five public CD34+ ATAC experiments of healthy donors from
ChIP-Atlas, RNA: ENCODE ENCSR830HIN CD34+ common myeloid progenitor, GENCODE V24). The download is
`scripts/aml_private_loop_download.sh`.

Stages
  inventory  per-patient overlap table of Hi-C / CUT&Tag / WGBS / ATAC / RNA from the GEO sample titles
  atac       10 kb bin mean and max of the log2(1 + RPGC) track, quantile-matched to the mean quantile
             function of the AML patients (one HSPC track = mean of the five matched replicates)
  rna        per-gene log2(TPM + 1), quantile-matched across samples; genes placed on refGene TSS (by symbol,
             GENCODE TSS where the symbol is absent); per 10 kb bin the maximal expression of the genes with a TSS
             in the bin and the maximal up/down difference to HSPC
  features   new feature groups for every row of the loop classifier table (results/aml_loop_classifier_v2),
             plus the recurrence of each AML-specific object over the other donors (Suppl. Tables 1, 4)
  model      same matched controls, locus-disjoint leave-one-donor-out folds and donor bootstrap as the loop
             classifier; increments of every new group over the baseline
  figures

New feature groups (every region feature enters per pair as min and max over the two regions)
  atacown   patient ATAC at the anchor bin: mean, max, flank (+-20 kb mean)
  atachspc  the same for HSPC
  atacdiff  patient minus HSPC
  rnaown    log2 TPM of the genes with the TSS in the anchor bin (prom) and within +-20 kb (near), patient
  rnahspc   the same, HSPC
  rnadiff   patient minus HSPC (region level), and the strongest up- and down-regulated gene within +-20 kb
  acxatac   H3K27ac x ATAC interaction: product of the genome-wide percentiles of both at the region (patient,
            and patient minus HSPC)

Targets: all AML-specific objects, private objects (present in no other donor with one-bin tolerance;
{168, 496} and {260, 424} one donor), non-private objects. Negatives: A_hspc (matched shared loops / stripes)
and B_rand (random pairs / shifted rectangles), as in the classifier.

    python scripts/aml_private_loop_info.py --stage inventory atac rna features model figures
Run with the hicdiff conda env.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import aml_loop_classifier as alc  # noqa: E402

PRIM = alc.PRIM
CACHE = PRIM / "work/privinfo"
OUT = REPO / "results/aml_private_loop_info"
CLF = REPO / "results/aml_loop_classifier_v2"
ATAC_DIR, RNA_DIR = PRIM / "atac", PRIM / "rna"
GENCODE = PRIM / "annot/gencode.v24.annotation.gtf.gz"
HSPC_ATAC = ["SRX19490150", "SRX19490151", "SRX19490152", "SRX10942129", "SRX10942130"]
HSPC_RNA = RNA_DIR / "hspc/ENCFF690QPA.tsv"
PBMC = ["103", "673", "576"]
RES, CHROMS = alc.RES, alc.CHROMS
SEED = 0


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


# ====================================================================== inventory
GEO = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={}&targ=gsm&form=text&view=brief"


def stage_inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    txt = urllib.request.urlopen(GEO.format("GSE152136"), timeout=300).read().decode()
    rows, cur = [], {}
    for line in txt.splitlines():
        if line.startswith("^SAMPLE"):
            if cur:
                rows.append(cur)
            cur = {"gsm": line.split("=")[1].strip()}
        elif line.startswith("!Sample_title"):
            cur["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_library_strategy"):
            cur["strategy"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_series_id"):
            cur.setdefault("series", []).append(line.split("=")[1].strip())
    rows.append(cur)
    S = pd.DataFrame(rows)
    S["series"] = S.series.map(lambda x: ";".join(x) if isinstance(x, list) else "")
    S["patient"] = S.title.str.extract(r"^([^_]+)_")[0]
    S["assay"] = S.title.str.replace(r"^[^_]+_", "", regex=True)
    S.to_csv(OUT / "geo_samples.csv", index=False)
    piv = S.assign(v=1).pivot_table(index="patient", columns="assay", values="v", aggfunc="sum").fillna(0).astype(int)
    piv = piv.rename(columns=lambda c: c.replace("CUT&TAG_", "CT_"))
    keep = ["Hi-C", "RNA-seq", "ATAC-seq", "WGBS", "CT_H3K27ac", "CT_H3K27me3", "CT_CTCF"]
    piv = piv.reindex(columns=[k for k in keep if k in piv.columns] + [c for c in piv.columns if c not in keep and c != "CUT&TAG_IgG" and c != "CT_IgG"])
    piv["in_loop_classifier"] = piv.index.isin(alc.LOOP_PATIENTS)
    piv["healthy_PBMC"] = piv.index.isin(alc.HEALTHY)
    piv.to_csv(OUT / "inventory_per_patient.csv")
    fl = []
    for g, d in (("GSE152134", "ATAC-seq"), ("GSE152096", "RNA-seq (RSEM genes)"), ("GSE152099", "WGBS"),
                 ("GSE152132", "CUT&Tag"), ("GSE152135", "Hi-C")):
        f = PRIM / f"{g}_filelist.txt"
        if not f.exists():
            continue
        t = pd.read_csv(f, sep="\t")
        t = t[t["#Archive/File"] == "File"]
        t["ext"] = t.Name.str.extract(r"((?:\.[A-Za-z0-9&-]+)+)$")[0]
        t["kind"] = t.Name.map(lambda n: "bigWig" if re.search(r"bigwig|\.bw$", n, re.I) else
                               "peaks" if "peaks" in n else "gene counts/TPM" if "quant.genes" in n else
                               "mcool" if "mcool" in n else "pairs" if "pairs" in n else "other")
        s = t.groupby("kind").agg(n_files=("Name", "size"), total_gb=("Size", lambda x: x.sum() / 1e9), median_mb=("Size", lambda x: x.median() / 1e6))
        s.insert(0, "assay", d)
        s.insert(0, "series", g)
        fl.append(s.reset_index())
    pd.concat(fl).to_csv(OUT / "inventory_files.csv", index=False, float_format="%.3f")
    log("[inventory] done")
    print(piv.to_string())


# ====================================================================== ATAC
def atac_path(src: str) -> Path | None:
    g = sorted(ATAC_DIR.glob(f"*_{src}_ATAC-seq.10bp.RPGC.bigwig"))
    return g[0] if g else None


def atac_raw_one(job):
    import pyBigWig
    src, path = job
    dest = CACHE / f"atacraw_{src}.npz"
    if dest.exists():
        return src
    sizes = alc.chrom_sizes()
    bw = pyBigWig.open(str(path))
    out = {}
    for c in CHROMS:
        n = alc.n_bins(c, sizes)
        nb = sizes[c] // RES
        for st in ("mean", "max"):
            a = np.full(n, np.nan, np.float32)
            v = bw.stats(c, 0, nb * RES, type=st, nBins=nb)
            a[:nb] = np.array([np.nan if x is None else x for x in v], np.float32)
            out[f"{c}|{st}"] = a
    bw.close()
    np.savez_compressed(dest, **out)
    return src


def qmap(vals_by_key: dict, ref: np.ndarray, grid: np.ndarray) -> dict:
    allv = np.concatenate([v[np.isfinite(v)] for v in vals_by_key.values()])
    sq = np.quantile(allv, grid)
    return {k: np.interp(v, sq, ref).astype(np.float32) for k, v in vals_by_key.items()}


def stage_atac(workers: int):
    CACHE.mkdir(parents=True, exist_ok=True)
    srcs = [p for p in alc.LOOP_PATIENTS + PBMC if atac_path(p)]
    jobs = [(p, atac_path(p)) for p in srcs] + [(s, ATAC_DIR / f"hspc/{s}.bw") for s in HSPC_ATAC]
    with ProcessPoolExecutor(workers) as ex:
        for s in ex.map(atac_raw_one, jobs):
            log(f"[atac] raw {s}")
    grid = np.linspace(0, 1, 1001)
    sizes = alc.chrom_sizes()
    raw = {s: dict(np.load(CACHE / f"atacraw_{s}.npz")) for s, _ in jobs}
    aml = [p for p in srcs if p not in PBMC]
    for st in ("mean", "max"):
        qs = []
        for s in aml:
            v = np.concatenate([raw[s][f"{c}|{st}"] for c in CHROMS])
            v = v[np.isfinite(v)]
            qs.append(np.quantile(v, grid))
        ref = np.mean(qs, 0)
        for s in raw:
            # chromosomes without data (None) count as zero coverage
            m = qmap({c: np.nan_to_num(raw[s][f"{c}|{st}"], nan=0.0) for c in CHROMS}, ref, grid)
            for c in CHROMS:
                raw[s][f"{c}|{st}"] = np.log2(1.0 + m[c])
    for s in srcs:
        np.savez_compressed(CACHE / f"atac_{s}.npz", **raw[s])
    hs = {k: np.mean([raw[s][k] for s in HSPC_ATAC], 0).astype(np.float32) for k in raw[HSPC_ATAC[0]]}
    np.savez_compressed(CACHE / "atac_hspc.npz", **hs)
    # replicate agreement of the five HSPC experiments (Pearson on 10 kb mean bins, genome-wide)
    M = np.array([np.concatenate([raw[s][f"{c}|mean"] for c in CHROMS]) for s in HSPC_ATAC])
    C = np.corrcoef(M)
    pd.DataFrame(C, index=HSPC_ATAC, columns=HSPC_ATAC).to_csv(OUT / "atac_hspc_replicate_corr.csv", float_format="%.3f")
    # agreement of patients with HSPC and with each other
    P = np.array([np.concatenate([raw[s][f"{c}|mean"] for c in CHROMS]) for s in srcs])
    H = np.concatenate([hs[f"{c}|mean"] for c in CHROMS])
    pd.DataFrame({"sample": srcs, "pearson_vs_hspc": [np.corrcoef(p, H)[0, 1] for p in P],
                  "pearson_vs_mean_other": [np.corrcoef(P[i], np.delete(P, i, 0).mean(0))[0, 1] for i in range(len(srcs))]}).to_csv(
        OUT / "atac_sample_corr.csv", index=False, float_format="%.3f")
    log("[atac] done")


# ====================================================================== RNA
def load_genes():
    """GENCODE v24 genes (protein coding and lincRNA) with TSS lists (refGene by symbol, else GENCODE)."""
    ref = {}
    with gzip.open(alc.REFGENE, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[2] not in CHROMS:
                continue
            ref.setdefault(f[12], set()).add((f[2], int(f[4]) if f[3] == "+" else int(f[5])))
    rows = []
    with gzip.open(GENCODE, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t")
            if f[2] != "gene" or f[0] not in CHROMS:
                continue
            a = f[8]
            gid = re.search(r'gene_id "([^"]+)"', a).group(1)
            typ = re.search(r'gene_type "([^"]+)"', a).group(1)
            name = re.search(r'gene_name "([^"]+)"', a).group(1)
            if typ not in ("protein_coding", "lincRNA"):
                continue
            tss = int(f[3]) if f[6] == "+" else int(f[4])
            rows.append((gid, name, typ, f[0], tss))
    G = pd.DataFrame(rows, columns=["gene_id", "name", "type", "chrom", "tss_gencode"])
    tl = []
    for r in G.itertuples():
        if r.name in ref:
            for c, t in ref[r.name]:
                tl.append((r.gene_id, c, t, "refGene"))
        else:
            tl.append((r.gene_id, r.chrom, r.tss_gencode, "gencode"))
    T = pd.DataFrame(tl, columns=["gene_id", "chrom", "tss", "source"])
    return G, T


def stage_rna():
    CACHE.mkdir(parents=True, exist_ok=True)
    G, T = load_genes()
    log(f"[rna] genes {len(G)}; TSS from refGene {(T.source == 'refGene').mean():.3f} of {len(T)}")
    files = {}
    for f in sorted(RNA_DIR.glob("GSM*genes.txt.gz")):
        m = re.match(r"GSM\d+_(\d+)", f.name)
        files[m.group(1)] = f
    gid = G.gene_id.values
    E = {}
    for s, f in files.items():
        d = pd.read_csv(f, sep="\t", usecols=["gene_id", "TPM"]).set_index("gene_id").TPM
        E[s] = np.log2(1.0 + d.reindex(gid).fillna(0).values)
    d = pd.read_csv(HSPC_RNA, sep="\t", usecols=["gene_id", "TPM"]).set_index("gene_id").TPM
    E["hspc"] = np.log2(1.0 + d.reindex(gid).fillna(0).values)
    aml = [s for s in E if s not in PBMC and s != "hspc"]
    # quantile matching (rank to mean sorted AML profile)
    ref = np.mean([np.sort(E[s]) for s in aml], 0)
    pct = np.linspace(0, 1, len(gid))
    for s in E:
        r = pd.Series(E[s]).rank(method="average").values
        E[s] = np.interp((r - 1) / (len(r) - 1), pct, ref).astype(np.float32)
    sizes = alc.chrom_sizes()
    gidx = {g: i for i, g in enumerate(gid)}
    T = T[T.chrom.isin(CHROMS)].copy()
    T["gi"] = T.gene_id.map(gidx)
    T["bin"] = np.minimum(T.tss // RES, T.chrom.map(lambda c: alc.n_bins(c, sizes) - 1))
    Eh = E["hspc"]
    pd.DataFrame({"gene_id": gid, "name": G.name.values, **{f"log2tpm_{s}": E[s] for s in E}}).to_csv(
        CACHE / "gene_expression_qn.csv.gz", index=False)
    for s in E:
        out = {}
        for c in CHROMS:
            n = alc.n_bins(c, sizes)
            t = T[T.chrom == c]
            prom = np.full(n, -np.inf)
            up = np.full(n, -np.inf)
            dn = np.full(n, np.inf)
            np.maximum.at(prom, t.bin.values, E[s][t.gi.values])
            dif = E[s][t.gi.values] - Eh[t.gi.values]
            np.maximum.at(up, t.bin.values, dif)
            np.minimum.at(dn, t.bin.values, dif)
            out[c + "|prom"] = np.where(np.isfinite(prom), prom, 0).astype(np.float32)
            out[c + "|up"] = np.where(np.isfinite(up), up, 0).astype(np.float32)
            out[c + "|down"] = np.where(np.isfinite(dn), dn, 0).astype(np.float32)
        np.savez_compressed(CACHE / f"rna_{s}.npz", **out)
    ex = pd.DataFrame({s: E[s] for s in aml + ["hspc"] + PBMC if s in E})
    expr = ex[(ex[aml].max(1) > 1)]
    cor = expr.corr(method="spearman")
    cor.to_csv(OUT / "rna_sample_spearman.csv", float_format="%.3f")
    log(f"[rna] {len(E)} samples done; median Spearman patient-patient {np.nanmedian(cor.loc[aml, aml].values[np.triu_indices(len(aml), 1)]):.3f}, "
        f"patient-HSPC {cor.loc[aml, 'hspc'].median():.3f}")


# ====================================================================== recurrence (private objects)
def recurrence(F: pd.DataFrame) -> pd.DataFrame:
    """Number of other donors that carry the same AML-specific object (loops: both anchors within one 10 kb
    bin; stripes: overlapping anchor and overlapping extent), for positive rows."""
    from scipy.spatial import cKDTree
    L = alc.read_tab("st1_aml_specific_loops.csv", "loop")
    L = L[L.chrom1 == L.chrom2]
    S = alc.read_tab("st4_aml_specific_stripes.csv", "stripe")
    S = S[S.chrom_anchor == S.chrom_extent]
    donor = lambda p: alc.donor_of(p, True)  # noqa: E731
    allp = sorted((set(L.patient) | set(S.patient)) - set(alc.HEALTHY))
    trees = {}
    for q, g in L.groupby("patient"):
        for c, h in g.groupby("chrom1"):
            trees[(q, c)] = cKDTree(np.c_[h.start1 // RES, h.start2 // RES])
    Sd = {q: g for q, g in S.groupby("patient")}
    n_other = np.full(len(F), -1)
    pos = np.nonzero((F.set == "pos").values)[0]
    for k in pos:
        r = F.iloc[k]
        seen = set()
        for q in allp:
            if donor(q) == donor(r.patient):
                continue
            if r.kind == "loop":
                t = trees.get((q, r.chrom))
                if t is not None and t.query([r.a_lo, r.b_lo], p=np.inf)[0] <= 1:
                    seen.add(donor(q))
            else:
                g = Sd.get(q)
                if g is None:
                    continue
                m = ((g.chrom_anchor == r.chrom) & (g.start_anchor // RES < r.a_hi) & (-(-g.end_anchor // RES) > r.a_lo) &
                     (g.start_extent // RES < r.b_hi) & (-(-g.end_extent // RES) > r.b_lo)).any()
                if m:
                    seen.add(donor(q))
        n_other[k] = len(seen)
    F["n_other"] = n_other
    return F


# ====================================================================== features
class Tr:
    def __init__(self):
        self.sizes = alc.chrom_sizes()
        self._c = {}
        self.cdf = {}

    def get(self, kind, src):
        k = (kind, src)
        if k not in self._c:
            f = CACHE / f"{kind}_{src}.npz"
            self._c[k] = dict(np.load(f)) if f.exists() else None
        return self._c[k]

    def ac(self, src):
        k = ("ac", src)
        if k not in self._c:
            f = alc.CACHE / f"trk_{src}_H3K27ac.npz"
            self._c[k] = dict(np.load(f)) if f.exists() else None
        return self._c[k]

    def pct(self, kind, src, arr_by_chr_key):
        """Genome-wide empirical CDF of the mean bins of a track."""
        k = (kind, src)
        if k not in self.cdf:
            T = arr_by_chr_key
            v = np.concatenate([T[c + "|mean"] for c in CHROMS])
            v = np.sort(v[np.isfinite(v)])
            self.cdf[k] = v
        return self.cdf[k]


def _nm(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _nx(x):
    x = np.asarray(x, float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def _nmin(x):
    x = np.asarray(x, float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan


def region_new(tr: Tr, p: str, chrom: str, lo: int, hi: int) -> dict:
    n = alc.n_bins(chrom, tr.sizes)
    lo2, hi2 = max(0, lo - 2), min(n, hi + 2)
    f = {}
    A, H = tr.get("atac", p), tr.get("atac", "hspc")
    ac_p, ac_h = tr.ac(p), tr.ac("hspc")
    vals = {}
    for nm, T in (("own", A), ("hspc", H)):
        if T is None:
            vals[nm] = (np.nan, np.nan, np.nan)
            continue
        mean, mx = T[chrom + "|mean"], T[chrom + "|max"]
        flank = np.r_[mean[lo2:lo], mean[hi:hi2]]
        vals[nm] = (_nm(mean[lo:hi]), _nx(mx[lo:hi]), _nm(flank))
    for k, s in enumerate(("mean", "max", "flank")):
        f[f"atacown_{s}"] = vals["own"][k]
        f[f"atachspc_{s}"] = vals["hspc"][k]
        f[f"atacdiff_{s}"] = vals["own"][k] - vals["hspc"][k]
    # interaction of H3K27ac and ATAC (genome-wide percentiles, region mean)
    inter = {}
    for nm, T, Tac, src in (("own", A, ac_p, p), ("hspc", H, ac_h, "hspc")):
        if T is None or Tac is None:
            inter[nm] = np.nan
            continue
        ca, ct = tr.pct("ac", src, Tac), tr.pct("atac", src, T)
        ua = np.searchsorted(ca, _nm(Tac[chrom + "|mean"][lo:hi])) / len(ca)
        ut = np.searchsorted(ct, _nm(T[chrom + "|mean"][lo:hi])) / len(ct)
        inter[nm] = ua * ut
    f["acxatac_own"] = inter["own"]
    f["acxatac_hspc"] = inter["hspc"]
    f["acxatac_diff"] = inter["own"] - inter["hspc"]
    R, RH = tr.get("rna", p), tr.get("rna", "hspc")
    if R is not None and RH is not None:
        po, ph = R[chrom + "|prom"], RH[chrom + "|prom"]
        f["rnaown_prom"], f["rnaown_near"] = float(po[lo:hi].max()), float(po[lo2:hi2].max())
        f["rnahspc_prom"], f["rnahspc_near"] = float(ph[lo:hi].max()), float(ph[lo2:hi2].max())
        f["rnadiff_prom"] = f["rnaown_prom"] - f["rnahspc_prom"]
        f["rnadiff_near"] = f["rnaown_near"] - f["rnahspc_near"]
        f["rnadiff_up"] = float(R[chrom + "|up"][lo2:hi2].max())
        f["rnadiff_down"] = float(R[chrom + "|down"][lo2:hi2].min())
    else:
        for k in ("rnaown_prom", "rnaown_near", "rnahspc_prom", "rnahspc_near", "rnadiff_prom", "rnadiff_near",
                  "rnadiff_up", "rnadiff_down"):
            f[k] = np.nan
    return f


def feats_job(args):
    p, rows = args
    tr = Tr()
    out = []
    for r in rows:
        fa = region_new(tr, p, r["chrom"], r["a_lo"], r["a_hi"])
        fb = region_new(tr, p, r["chrom"], r["b_lo"], r["b_hi"])
        d = {"row": r["row"]}
        for k in fa:
            d[k + "_min"] = _nmin([fa[k], fb[k]])
            d[k + "_max"] = _nx([fa[k], fb[k]])
        out.append(d)
    return out


def stage_features(workers: int):
    F = pd.read_csv(CLF / "features.csv.gz", low_memory=False, dtype={"patient": str},
                    usecols=["patient", "kind", "set", "pair_id", "y", "chrom", "a_lo", "a_hi", "b_lo", "b_hi", "pair_type"])
    F = F[~F.patient.isin(alc.HEALTHY)].reset_index(drop=True)
    F = recurrence(F)
    F["row"] = np.arange(len(F))
    jobs = []
    for p, g in F.groupby("patient"):
        if not (CACHE / f"atac_{p}.npz").exists() and not (CACHE / f"rna_{p}.npz").exists():
            continue
        jobs.append((p, g[["row", "chrom", "a_lo", "a_hi", "b_lo", "b_hi"]].to_dict("records")))
    res = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(feats_job, jobs):
            res += r
    N = pd.DataFrame(res).set_index("row")
    out = F.join(N, on="row")
    out.to_csv(CACHE / "new_features.csv.gz", index=False)
    pos = out[out.set == "pos"]
    pos.groupby(["kind", "patient"]).agg(n=("y", "size"), private=("n_other", lambda x: int((x == 0).sum())),
                                          shared1=("n_other", lambda x: int((x == 1).sum())),
                                          recurrent=("n_other", lambda x: int((x >= 2).sum()))).to_csv(OUT / "private_counts.csv")
    log(f"[features] {len(out)} rows, {N.shape[1]} new columns; private fraction loops "
        f"{(pos[pos.kind == 'loop'].n_other == 0).mean():.3f}, stripes {(pos[pos.kind == 'stripe'].n_other == 0).mean():.3f}")


# ====================================================================== models
NEW_GROUPS = ["atacown", "atachspc", "atacdiff", "rnaown", "rnahspc", "rnadiff", "acxatac"]


def paired(y, s1, s0, donors, B=1000, seed=0):
    """Difference s1 minus s0 in macro AUROC, pooled AUROC and pooled AUPRC with donor bootstrap."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    up = np.unique(donors)
    idx = {u: np.nonzero(donors == u)[0] for u in up}
    ok = {u: 0 < y[i].sum() < len(i) for u, i in idx.items()}
    pm = {u: (roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s0[i])) if ok[u] else np.nan for u, i in idx.items()}
    rng = np.random.default_rng(seed)
    dm, dp, dr = [], [], []
    for _ in range(B):
        pick = rng.choice(up, len(up))
        dm.append(np.nanmean([pm[u] for u in pick]))
        ii = np.concatenate([idx[u] for u in pick])
        if 0 < y[ii].sum() < len(ii):
            dp.append(roc_auc_score(y[ii], s1[ii]) - roc_auc_score(y[ii], s0[ii]))
            dr.append(average_precision_score(y[ii], s1[ii]) - average_precision_score(y[ii], s0[ii]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return {"d_macro": float(np.nanmean(list(pm.values()))), "d_macro_lo": q(dm)[0], "d_macro_hi": q(dm)[1],
            "d_pooled": float(roc_auc_score(y, s1) - roc_auc_score(y, s0)), "d_pooled_lo": q(dp)[0], "d_pooled_hi": q(dp)[1],
            "d_auprc": float(average_precision_score(y, s1) - average_precision_score(y, s0)), "d_auprc_lo": q(dr)[0], "d_auprc_hi": q(dr)[1]}


def build_sets(base_groups, avail):
    A = [g for g in ("atacown", "atachspc", "atacdiff") if g in avail]
    R = [g for g in ("rnaown", "rnahspc", "rnadiff") if g in avail]
    X = ["acxatac"] if "acxatac" in avail else []
    sets = {"base": []}
    for g in avail:
        sets[f"+{g}"] = [g]
    if A:
        sets["+atac_all"] = A
    if R:
        sets["+rna_all"] = R
    if A and R:
        sets["+atac_all+rna_all"] = A + R
    if X:
        sets["+atac_all+acxatac"] = A + X
    sets["+all_new"] = A + R + X
    sets = {k: v for k, v in sets.items() if k == "base" or v}
    return sets


DATASETS = {
    # name: (patients, base groups, description)
    "D1_marks_wgbs_atac_rna": (alc.CTCF_PATIENTS, ["hspc", "own", "meth", "motif", "gene", "dist"],
                               "8 patients with CTCF, H3K27ac, H3K27me3, WGBS, ATAC and RNA; baseline HSPC state + own marks + methylation"),
    "D2_marks_atac": (alc.CT_PATIENTS, ["hspc", "own", "motif", "gene", "dist"],
                      "10 patients with H3K27ac, H3K27me3 and ATAC (no CTCF track in 1360, 1916; no WGBS, no RNA in them); baseline HSPC + own marks"),
    "D3_rna_wgbs": (alc.CTCF_PATIENTS + alc.WGBS_ONLY, ["hspc", "meth", "motif", "gene", "dist"],
                    "patients with RNA and WGBS (8 + 10), no own marks; baseline HSPC state + methylation"),
    "D4_atac_rna_wgbs": (alc.CTCF_PATIENTS + ["260", "566"], ["hspc", "meth", "motif", "gene", "dist"],
                         "patients with ATAC, RNA and WGBS (8 + 260 + 566), no own marks; baseline HSPC state + methylation"),
}


def make_data(F, base, kind, negset, target, pats):
    d = F[(F.kind == kind) & F.patient.isin(pats)]
    pos = d[d.set == "pos"]
    if target == "private":
        pos = pos[pos.n_other == 0]
    elif target == "nonprivate":
        pos = pos[pos.n_other >= 1]
    neg = d[d.set == negset]
    keep = set(pos.pair_id) & set(neg.pair_id)
    d = pd.concat([pos[pos.pair_id.isin(keep)], neg[neg.pair_id.isin(keep)]]).reset_index(drop=True)
    return d


def cols_for(d, groups, allcols):
    out = []
    for c in allcols:
        g = alc.group_of(c)
        if g not in groups:
            continue
        if d[c].isna().groupby(d.patient).all().any() or d[c].isna().mean() > 0.9:
            continue
        out.append(c)
    return out


def stage_model(args):
    from sklearn.metrics import roc_auc_score
    Fn = pd.read_csv(CACHE / "new_features.csv.gz", low_memory=False, dtype={"patient": str})
    Fo = pd.read_csv(CLF / "features.csv.gz", low_memory=False, dtype={"patient": str})
    Fo = Fo[~Fo.patient.isin(alc.HEALTHY)].reset_index(drop=True)
    newc = [c for c in Fn.columns if alc.group_of(c) in NEW_GROUPS]
    F = pd.concat([Fo.reset_index(drop=True), Fn[["n_other"] + newc]], axis=1)
    assert (Fo.pair_id.values == Fn.pair_id.values).all()
    oldc = alc.feature_cols(Fo)
    OUT.mkdir(parents=True, exist_ok=True)
    summ, incr, bytype, uni, preds = [], [], [], [], []
    runs = []
    for dsn, (pats, bg, desc) in DATASETS.items():
        pats = [p for p in pats if p not in alc.HEALTHY]
        kinds = ["loop", "stripe"] if dsn.startswith("D1") else ["loop"]
        for kind in kinds:
            for negset in ("A_hspc", "B_rand"):
                for target in ("all", "private", "nonprivate"):
                    if target == "nonprivate" and not dsn.startswith("D1"):
                        continue
                    runs.append((dsn, pats, bg, desc, kind, negset, target))
    for dsn, pats, bg, desc, kind, negset, target in runs:
        d = make_data(F, bg, kind, negset, target, pats)
        y = d.y.values.astype(int)
        if y.sum() < 40 or (1 - y).sum() < 40:
            continue
        pt = np.array([alc.donor_of(p, True) for p in d.patient])
        # available groups with usable columns in this dataset
        allc = oldc + newc
        base_cols = cols_for(d, bg, allc)
        avail = [g for g in NEW_GROUPS if cols_for(d, [g], allc)]
        sets = build_sets(bg, avail)
        keys = alc.row_keys(d)
        folds = alc.make_folds(pt, keys=keys)
        tag = f"{dsn}|{kind}|{negset}|{target}"
        log(f"[model] {tag}: {len(np.unique(pt))} donors, {y.sum()} pos, {(1 - y).sum()} neg, base {len(base_cols)} cols, new {avail}")
        for model in ("lr", "gb"):
            S = {}
            for sn, gs in sets.items():
                cols = base_cols + cols_for(d, gs, allc) if gs else base_cols
                X = d[cols].values.astype(float)
                S[sn] = alc.cv_scores(model, X, y, folds, args.jobs)
            # new groups alone (+ distance as the only baseline column)
            nc = cols_for(d, avail, allc)
            if nc:
                S["new_only"] = alc.cv_scores(model, d[cols_for(d, ["dist"], allc) + nc].values.astype(float), y, folds, args.jobs)
            S["dist_only"] = alc.cv_scores("lr", d[cols_for(d, ["dist"], allc)].values.astype(float), y, folds, args.jobs)
            for sn, s in S.items():
                m = alc.metrics(y, s, pt, args.boot)
                rec = {"dataset": dsn, "kind": kind, "negatives": negset, "target": target, "model": model, "featureset": sn,
                       "n_donors": len(np.unique(pt)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                       **{k: v for k, v in m.items() if k != "per_patient"}}
                summ.append(rec)
                if sn not in ("base",) and sn != "dist_only":
                    dl = paired(y, s, S["base"], pt, args.boot)
                    incr.append({"dataset": dsn, "kind": kind, "negatives": negset, "target": target, "model": model,
                                 "featureset": sn, "n_pos": int(y.sum()), "n_donors": len(np.unique(pt)),
                                 "base_auroc_macro": summ[[i for i, r in enumerate(summ) if r["featureset"] == "base"
                                                           and r["dataset"] == dsn and r["kind"] == kind and r["negatives"] == negset
                                                           and r["target"] == target and r["model"] == model][0]]["auroc_macro"], **dl})
            pr = d[["patient", "pair_id", "set", "y", "pair_type", "n_other"]].copy()
            pr.insert(0, "dataset", dsn)
            for sn in ("base", "+all_new"):
                if sn in S:
                    pr[sn] = S[sn]
            pr["model"] = model
            pr["negatives"], pr["target"], pr["kind"] = negset, target, kind
            preds.append(pr)
            for t in sorted(set(d.pair_type)):
                i = (d.pair_type == t).values
                if y[i].sum() < 15 or (1 - y[i]).sum() < 15 or "+all_new" not in S:
                    continue
                if len(np.unique(pt[i])) < 3:
                    continue
                m1 = alc.metrics(y[i], S["+all_new"][i], pt[i], args.boot)
                m0 = alc.metrics(y[i], S["base"][i], pt[i], args.boot)
                dl = paired(y[i], S["+all_new"][i], S["base"][i], pt[i], args.boot)
                bytype.append({"dataset": dsn, "kind": kind, "negatives": negset, "target": target, "model": model,
                               "pair_type": t, "n_pos": int(y[i].sum()), "n_donors": len(np.unique(pt[i])),
                               "base_auroc_pooled": m0["auroc_pooled"], "base_lo": m0["auroc_pooled_lo"], "base_hi": m0["auroc_pooled_hi"],
                               "new_auroc_pooled": m1["auroc_pooled"], "new_lo": m1["auroc_pooled_lo"], "new_hi": m1["auroc_pooled_hi"],
                               "base_auprc": m0["auprc_pooled"], "new_auprc": m1["auprc_pooled"], **dl})
        # univariate: mean donor-wise AUROC of every new feature (and key baseline features)
        if negset == "A_hspc":
            for c in nc + [c for c in base_cols if c.startswith(("own_H3K27ac_mean", "hspc_H3K27ac_mean", "own_CTCF_mean", "hspc_CTCF_mean"))]:
                per = []
                for u in np.unique(pt):
                    i = pt == u
                    x = d[c].values[i]
                    ok = np.isfinite(x)
                    if ok.sum() > 20 and 0 < y[i][ok].sum() < ok.sum() and np.ptp(x[ok]) > 0:
                        per.append(roc_auc_score(y[i][ok], x[ok]))
                if per:
                    uni.append({"dataset": dsn, "kind": kind, "target": target, "feature": c, "group": alc.group_of(c),
                                "mean_donor_auroc": float(np.mean(per)), "n_donors": len(per), "n_above_half": int((np.array(per) > 0.5).sum())})
        pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
        pd.DataFrame(incr).to_csv(OUT / "increments.csv", index=False, float_format="%.4f")
        pd.DataFrame(bytype).to_csv(OUT / "by_type.csv", index=False, float_format="%.4f")
        pd.DataFrame(uni).to_csv(OUT / "univariate.csv", index=False, float_format="%.4f")
    pd.concat(preds).to_csv(OUT / "oof_predictions.csv.gz", index=False, float_format="%.5f")
    log("[model] done")


# ====================================================================== figure
def stage_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    I = pd.read_csv(OUT / "increments.csv")
    S = pd.read_csv(OUT / "summary.csv")
    U = pd.read_csv(OUT / "univariate.csv")
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.1, 1.5, 1.0]})
    # a: AUROC of base and +all_new for private / all, loops, D1, gb, both negatives
    d = S[(S.dataset == "D1_marks_wgbs_atac_rna") & (S.kind == "loop") & (S.model == "gb") & S.featureset.isin(["base", "+all_new", "new_only", "dist_only"])]
    lab, y0 = [], 0
    cols = {"base": "#555555", "+all_new": "#b2182b", "new_only": "#2166ac", "dist_only": "#bbbbbb"}
    for tgt in ("private", "all"):
        for neg in ("A_hspc", "B_rand"):
            for fs in ("dist_only", "new_only", "base", "+all_new"):
                r = d[(d.target == tgt) & (d.negatives == neg) & (d.featureset == fs)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax[0].errorbar(r.auroc_pooled, y0, xerr=[[r.auroc_pooled - r.auroc_pooled_lo], [r.auroc_pooled_hi - r.auroc_pooled]],
                               fmt="o", color=cols[fs], capsize=2)
                lab.append(f"{tgt} vs {neg}  {fs}")
                y0 += 1
            y0 += 0.6
    ax[0].set_yticks([i for i in range(len(lab))][:0])
    ax[0].set_yticks(np.arange(len(lab)) + np.repeat(np.arange(0, len(lab) // 4 + 1), 4)[:len(lab)] * 0.6)
    ax[0].set_yticklabels(lab, fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("pooled AUROC (GB, leave-one-donor-out, donor bootstrap 95%)")
    ax[0].set_title("a  Loops, 8 patients: baseline and + ATAC/RNA", fontsize=9)
    ax[0].axvline(0.5, color="k", lw=0.5)
    # b: increments per feature set, private vs all, A_hspc, D1/D3/D4
    z = I[(I.negatives == "A_hspc") & (I.kind == "loop") & (I.model == "gb") & I.target.isin(["private", "all"]) &
          I.featureset.isin(["+atac_all", "+rna_all", "+acxatac", "+atac_all+rna_all", "+all_new"])]
    lab2, y1 = [], 0
    for dsn in ("D1_marks_wgbs_atac_rna", "D2_marks_atac", "D3_rna_wgbs", "D4_atac_rna_wgbs"):
        for fs in ("+atac_all", "+rna_all", "+all_new"):
            for tgt, col in (("all", "#999999"), ("private", "#b2182b")):
                r = z[(z.dataset == dsn) & (z.featureset == fs) & (z.target == tgt)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax[1].errorbar(r.d_macro, y1, xerr=[[r.d_macro - r.d_macro_lo], [r.d_macro_hi - r.d_macro]], fmt="o", color=col, capsize=2)
                lab2.append(f"{dsn.split('_')[0]} {fs} {tgt}")
                y1 += 1
            y1 += 0.3
    ax[1].set_yticks(np.arange(len(lab2)) + np.repeat(np.arange(0, len(lab2) // 2 + 1), 2)[:len(lab2)] * 0.3)
    ax[1].set_yticklabels(lab2, fontsize=7)
    ax[1].invert_yaxis()
    ax[1].axvline(0, color="k", lw=0.5)
    ax[1].axvline(0.05, color="#2166ac", lw=0.8, ls="--")
    ax[1].set_xlabel("increment in macro AUROC over baseline (GB, grey: all AML-specific, red: private; dashed: 0.05)")
    ax[1].set_title("b  Increment of the new groups, loops vs matched shared loops", fontsize=9)
    # c: univariate
    u = U[(U.dataset == "D1_marks_wgbs_atac_rna") & (U.kind == "loop") & (U.target == "private")].copy()
    u = u[u.group.isin(NEW_GROUPS)].sort_values("mean_donor_auroc")
    u = pd.concat([u.head(6), u.tail(6)])
    ax[2].barh(range(len(u)), u.mean_donor_auroc - 0.5, left=0.5, color="#2166ac")
    ax[2].set_yticks(range(len(u)))
    ax[2].set_yticklabels(u.feature, fontsize=7)
    ax[2].axvline(0.5, color="k", lw=0.5)
    ax[2].set_xlabel("mean donor AUROC, private loops vs A_hspc")
    ax[2].set_title("c  Strongest single new features (8 patients)", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "fig_private_loop_info.png", dpi=150)
    log("[figures] done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["inventory"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    for lst in (alc.CT_PATIENTS, alc.CTCF_PATIENTS, alc.WGBS_ONLY, alc.LOOP_PATIENTS):
        lst[:] = [p for p in lst if p not in alc.HEALTHY]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if "inventory" in a.stage:
        stage_inventory()
    if "atac" in a.stage:
        stage_atac(a.workers)
    if "rna" in a.stage:
        stage_rna()
    if "features" in a.stage:
        stage_features(a.workers)
    if "model" in a.stage:
        stage_model(a)
    if "figures" in a.stage:
        stage_figures()


if __name__ == "__main__":
    main()
