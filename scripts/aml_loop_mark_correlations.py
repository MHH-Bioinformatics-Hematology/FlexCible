"""Which marks and methylation features go with AML-specific loops and stripes, and can they be boosted?

Registry section 63. Objects: the 2276 AML-specific loops and 1075 stripes of the ten held-out donors of
the loop detection (scripts/aml_loop_detection.py, results/aml_loop_detection/object_scores.csv.gz; Xu
et al. 2022 Suppl. Tables 1 and 4) with their matched shared controls (A_hspc, A_full, A_do, B_rand).
Target: measured gain over the HSPC consensus (column gain|measured; donut enrichment for loops,
mean log O/E of the stripe rectangle for stripes).

  inventory  all AML-related data (one row per sample) -> inventory.csv / inventory.md
  tracks     10 kb / 1 kb track caches (healthy donors, CD14+ methylation, extra HSPC ChIP, CpG islands)
  features   feature matrix of every object: mark x representation x position x scale, methylation,
             motif-level methylation, engineered pairwise / normalised / alternative-reference features
  corr       Spearman (pooled, donor bootstrap, per donor, partial) and AUROC of every feature per class
  boost      engineered feature sets, held-out donors (donor-grouped, locus-disjoint folds)
  maps       2D outer-product maps of the 1D track deltas against the Hi-C residual and the model residual
  figures    heatmaps and increment plots

    python scripts/aml_loop_mark_correlations.py --stage inventory tracks features corr boost maps figures
Run with the hicdiff conda env (pyBigWig, cooler, scikit-learn), CPU only (nice 10, <= 6 threads).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
AMLD = HM / "aml"
PRIM = AMLD / "primary"
INP = PRIM / "inputs"
CACHE = PRIM / "work/loopclf"            # track caches of the loop classifier (10 kb bin mean / max)
WORK = PRIM / "work/loopmarks"           # caches of this script
OUT = REPO / "results/aml_loop_marks"
DET = REPO / "results/aml_loop_detection"
CLF = REPO / "results/aml_loop_classifier_v2/features.csv.gz"
XT = REPO / "results/aml_primary/xu_tables"
CGI = Path.home() / "data/hic-multiome/annotation/hg38/cpgIslandExt.txt.gz"
RES = 10_000
MARKS = ["H3K27ac", "H3K27me3", "CTCF"]
SHORT = {"H3K27ac": "ac", "H3K27me3": "me3", "CTCF": "ct"}
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
DONORS10 = ["018", "027", "270", "168", "546", "629", "1360", "472", "773", "1916"]
CTCF_DONORS = ["018", "027", "270", "168", "546", "629", "472", "773"]      # CTCF + WGBS available
HEALTHY = ["103", "673", "576"]
GROUP = {"168": "168+496", "496": "168+496", "260": "260+424", "424": "260+424"}
SCALES = {"s0": 0, "s2": 2, "s10": 10}          # bin only, +-20 kb, +-100 kb (bins of 10 kb on each side)
SEED = 0
NBOOT = 500


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sizes() -> dict:
    d = {}
    for line in (PRIM / "hg38.chrom.sizes").read_text().splitlines():
        c, n = line.split()[:2]
        d[c] = int(n)
    return d


def nb(chrom: str, S: dict) -> int:
    return S[chrom] // RES + 1


# ============================================================================================ inventory
def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024
    return str(n)


def fsize(p) -> int:
    try:
        return os.stat(p).st_size
    except OSError:
        return 0


def gl(pat: str) -> list:
    return sorted(glob.glob(pat))


def tot(pat: str) -> tuple:
    f = gl(pat)
    return len(f), sum(fsize(x) for x in f) / 2**30


def inventory_types() -> None:
    """One row per data type: raw reads or processed, format, size, and whether the model / analyses use it."""
    rows = []

    def add(dtype, level, fmt, pat, desc, used, notused=""):
        n, gb = tot(pat) if pat else (np.nan, np.nan)
        rows.append({"data_type": dtype, "level": level, "format": fmt, "files": n, "size_GB": round(gb, 1) if pat else np.nan,
                     "description": desc, "used_now": used, "not_used_yet": notused})
    add("Hi-C read pairs (Xu 2022)", "processed (aligned, de-duplicated pairs; no FASTQ)", "nodups.pairs.gz", str(PRIM / "pairs/*.pairs.gz"),
        "32 samples (25 AML, 3 healthy PBMC, 4 HSPC), hg38", "all 32: 5 kb cooler, 10 / 25 kb ICE and oe_proper targets", "2 kb maps not built")
    add("Hi-C iced matrices (Xu 2022)", "processed", ".iced.mcool (10 kb and coarser)", str(PRIM / "hic/*.mcool"), "same 32 samples", "targets rebuilt from the pairs", "5 kb and finer are not in the mcools")
    add("Hi-C targets and consensus maps", "processed", "cooler 5/10/25 kb, oe_proper", str(PRIM / "consensus/*.cool"), "AML / HSPC / leave-one-patient-out consensus maps", "reference maps of the models and the loop detection")
    add("CUT&Tag signal (H3K27ac, H3K27me3, CTCF)", "processed (MACS2 -log10 p-value signal, nodup_x_ctl)", "bigWig", str(PRIM / "cuttag/*.bigwig"),
        "33 files: 12 H3K27ac, 12 H3K27me3, 9 CTCF (8 AML + healthy 103); one sample per mark and donor, no replicate series; GEO record GSE152132",
        "quantile-matched to the training scale (inputs/xu_*) = model inputs; 10 kb bin means in the loop analyses", "")
    add("CUT&Tag peaks (MACS2 filt.peaks.txt.gz)", "processed", "txt.gz", None, "listed in GSE152132 (one file per sample and mark), not downloaded", "no", "peak calls: not used, not downloaded (MACS2 command line is not in the files we hold)")
    add("CUT&Tag raw reads", "raw", "FASTQ", None, "SRA only (GSE152132 gives bigWig and peaks; reads not public per the 2026-10-01 note)", "no", "not available locally")
    add("WGBS methylation (Xu 2022)", "processed (CpG methylation fraction per 10 bp bin, no read counts)", ".CGN.10bp.bw", str(PRIM / "wgbs/*.bw"),
        "20 samples (GSE152099): 10 with CUT&Tag, 10 WGBS only, healthy 103 / 673 included, no 576; no per-CpG coverage, no CD34+ WGBS",
        "1 kb Meth.bigwig (unweighted mean of covered 10 bp bins, >= 3 per kb) in the Meth model variants and the loop classifier; 10 bp values at CTCF motifs in this analysis", "per-CpG read counts / coverage (not in the bigWigs)")
    add("ATAC-seq (Xu 2022)", "processed (RPGC bigWig + MACS2 peaks)", "bigWig, peaks.txt.gz", str(PRIM / "atac/*"), "being fetched / tested by the private-loop agent", "not here", "in progress (other agent)")
    add("RNA-seq (Xu 2022)", "processed (RSEM gene quantification)", "txt.gz", str(PRIM / "rna/*.gz"), "being fetched / tested by the private-loop agent", "not here", "in progress (other agent)")
    add("Fischer 2024 ChIP (RAD21, STAG2, CTCF, H3K27ac per AML patient)", "processed (CNV-normalised bigWig)", "bigWig", str(PRIM / "fischer/chip/*"),
        "45 patient folders in inputs (21 control, 9 RAD21-mutant, 15 STAG2-mutant by name; 160 bigWigs, not every mark for every patient); no per-patient Hi-C", "group-level inputs (figroup_*) for the Fischer cohesin predictions", "per-patient RAD21 / STAG2 ChIP not used in any loop analysis")
    add("Fischer 2024 pooled Hi-C", "processed (FAN-C HDF5, 10 kb)", ".fanc.h5", str(PRIM / "fischer/*.h5"), "AML control, RAD21-mutant, STAG2-mutant pooled maps", "cooler conversion and group predictions", "")
    add("AML cell-line Hi-C and ChIP", "processed", "mcool / hic, ChIP-Atlas bigWig", None, "MOLM-13 (3 Hi-C), U-937 (5 Micro-C), K562 controls (files under aml/molm13, aml/u937, aml/hg38_*)", "model test cells (aml_primary sections)", "THP-1 / HL-60 / OCI-AML3 / MV4-11 listed in the survey only; cell-line loops / stripes not analysed")
    add("SV junctions", "processed (published table)", "csv", str(XT / "st5_svs.csv"), "Xu 2022 Suppl. Table 5: 595 junctions in 22 samples (none for 103, 673)", "SV-aware prediction (aml_sv_predict)", "")
    add("CNV segments", "derived from Hi-C coverage vs HSPC", "csv", str(REPO / "results/aml_primary/explain_cnv_segments.csv"), "54 segments in 18 samples", "region flags", "")
    add("Mutation / fusion labels", "published figure, hand-read", "csv", str(REPO / "results/aml_subtype/patient_subtype_table.csv"), "fusion class + consensus mutation group per patient (Xu 2022 Fig. 1c)", "subtype folds of the loop classifier", "no per-mutation effect analysis on loop gain yet")
    add("AML-specific loops and stripes", "processed (published tables)", "csv", str(XT / "st1_aml_specific_loops.csv"), "Suppl. Table 1 (loops, 22 patients with calls) and Table 4 (stripes, 18 patients)", "loop classifier, loop detection, this analysis", "")
    add("HSPC ChIP reference", "processed (ChIP-Atlas bigWig, 1 experiment per mark)", "bigWig", str(INP / "hspc_cd34/*.bigwig"), "CD34+ H3K27ac SRX1761289, H3K27me3 SRX1761288, CTCF SRX037935 (quantile-matched)", "the HSPC reference of all loop analyses",
        "7 further CD34+/HSC experiments fetched in this analysis (WORK/hspc_extra)")
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "inventory_data_types.csv", index=False)
    md = ["| data type | level | format | files | GB | used now | not used yet |", "|---|---|---|---|---|---|---|"]
    for _, r in D.iterrows():
        md.append(f"| {r.data_type} | {r.level} | {r.format} | {'' if pd.isna(r.files) else int(r.files)} | {'' if pd.isna(r.size_GB) else r.size_GB} | {r.used_now} | {r.not_used_yet} |")
    (OUT / "inventory_data_types.md").write_text("\n".join(md) + "\n")


def stage_inventory() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(REPO / "results/aml_subtype/patient_subtype_table.csv", dtype=str).set_index("patient")
    st1 = pd.read_csv(XT / "st1_aml_specific_loops.csv", dtype=str)
    st4 = pd.read_csv(XT / "st4_aml_specific_stripes.csv", dtype=str)
    st5 = pd.read_csv(XT / "st5_svs.csv", dtype=str)
    cnv = pd.read_csv(REPO / "results/aml_primary/explain_cnv_segments.csv", dtype={"patient": str})
    nl = st1.patient.value_counts().to_dict()
    ns = st4.patient.value_counts().to_dict()
    nsv = st5.patient.str.zfill(3).where(st5.patient.str.isdigit() & (st5.patient.str.len() < 3), st5.patient).value_counts().to_dict()
    ncnv = cnv.patient.str.replace("xu_", "").value_counts().to_dict()
    hic = {Path(f).name.split("_")[1].split(".")[0]: f for f in gl(str(PRIM / "hic/*.mcool"))}
    pairs = {Path(f).name.split("_")[1].split(".")[0]: f for f in gl(str(PRIM / "pairs/*.pairs.gz"))}
    rows = []
    ids = list(hic)
    train_p = ["018", "027", "103", "168", "270", "472", "546", "629", "673", "773", "1360", "1916"]
    for pid in ids:
        is_h = pid in HEALTHY
        is_hspc = pid.startswith("HSPC")
        r = {"sample": pid, "group": "HSPC donor (CD34+)" if is_hspc else ("healthy PBMC donor" if is_h else "AML patient"),
             "cohort": "Xu 2022 (GSE152135)"}
        r["hic_mcool"] = Path(hic[pid]).name
        r["hic_mcool_GB"] = round(fsize(hic[pid]) / 2**30, 2)
        r["hic_pairs"] = Path(pairs[pid]).name if pid in pairs else ""
        r["hic_pairs_GB"] = round(fsize(pairs.get(pid, "")) / 2**30, 2)
        jf = REPO / f"results/aml_primary/targets/xu_{pid}_5kb.json"
        r["hic_contacts_5kb_M"] = round(json.load(open(jf))["contacts"] / 1e6, 0) if jf.exists() else np.nan
        for m in MARKS:
            f = gl(str(PRIM / f"cuttag/*_{pid}_CUT_TAG_{m}.MACS2*.bigwig"))
            r[f"cuttag_{SHORT[m]}_bigwig"] = Path(f[0]).name.split("_CUT_TAG")[0].split("_")[0] if f else ""
            r[f"cuttag_{SHORT[m]}_MB"] = round(fsize(f[0]) / 2**20, 0) if f else np.nan
        r["cuttag_qm_inputs"] = ";".join(m for m in MARKS if (INP / f"xu_{pid}/{m}.bigwig").exists())
        n_ = pid.lstrip("0")
        w = gl(str(PRIM / f"wgbs/*_allc_{n_}.CGN.10bp.bw"))
        r["wgbs_10bp_bigwig"] = Path(w[0]).name if w else ""
        r["wgbs_MB"] = round(fsize(w[0]) / 2**20, 0) if w else np.nan
        r["wgbs_meth_1kb"] = (PRIM / f"meth/xu_{pid}/Meth.bigwig").exists()
        a = gl(str(PRIM / f"atac/*_{pid}_ATAC-seq.10bp.RPGC.bigwig"))
        r["atac_bigwig"] = Path(a[0]).name if a else ""
        r["atac_peaks"] = bool(gl(str(PRIM / f"atac/*_{pid}_ATAC-seq.MACS2.filt.peaks.txt.gz")))
        rn = gl(str(PRIM / f"rna/*_{pid}*RSEM.quant.genes.txt.gz")) + gl(str(PRIM / f"rna/*_{pid}_RNA-seq.RSEM.quant.genes.txt.gz"))
        r["rna_rsem"] = Path(rn[0]).name if rn else ""
        r["atac_rna_status"] = "downloaded by the private-loop agent, not analysed here" if (a or rn) else "not downloaded"
        k = pid
        r["loops_specific_n"] = int(nl.get(k, 0))
        r["stripes_specific_n"] = int(ns.get(k, 0))
        r["sv_junctions_n"] = int(nsv.get(k, 0))
        r["cnv_segments_n"] = int(ncnv.get(k, 0))
        if k in sub.index:
            s = sub.loc[k]
            r["fusion"] = s.fusion_class
            r["mutation_group"] = s.consensus_group
            r["mutations"] = s.consensus_mutations + (";" + s.other_mutations if isinstance(s.other_mutations, str) and s.other_mutations else "")
        r["model_inputs_chip"] = pid in train_p
        r["loop_classifier"] = pid in set(DONORS10) or (pid in ["1071", "1324", "260", "424", "496", "504", "566", "798", "838", "933"])
        r["loop_detection_donor"] = pid in DONORS10
        rows.append(r)
    Inv = pd.DataFrame(rows)
    # non-sample rows
    extra = []

    def add(**kw):
        extra.append(kw)

    add(sample="HSPC consensus", group="HSPC ChIP reference", cohort="ChIP-Atlas CD34+ (3 experiments)",
        cuttag_ac_bigwig="SRX1761289 (GSM2152595)", cuttag_me3_bigwig="SRX1761288 (GSM2152594)", cuttag_ct_bigwig="SRX037935 (GSM651541)",
        notes="one ChIP-seq experiment per mark, quantile-matched to the CUT&Tag scale (inputs/hspc_cd34); Hi-C consensus of the 4 HSPC maps (consensus/hspccons_*)")
    add(sample="CD14+ monocyte", group="healthy reference", cohort="ENCODE ENCSR017BUL", wgbs_10bp_bigwig="ENCFF689TNG (1 kb Meth.bigwig only)",
        notes="WGBS methylation reference (coverage-weighted 1 kb bins); no CD34+ WGBS exists locally")
    for nm, d_, hic_, ch_ in [("MOLM-13", "molm13", "GSM4417606 MboI hg38 .hic/.mcool 252 M; GSM9769670 Arima 703 M .mcool", "hg38_molm13: CTCF, RAD21, H3K27ac, H3K27me3, H3K4me1/3, H3K9ac, H3K9me3, H3K36me3, H3K79me2 (ChIP-Atlas bigwigs)"),
                              ("U-937", "u937", "GSM9916468-70 Micro-C WT clones (250 bp mcool), STAG2-KO GSM9916471-72", "hg38_u937: CTCF, RAD21, SMC3, H3K27ac, H3K4me3, H3K9ac, H3K9me3 (+ ko_ctcfonly, ko_half input sets)")]:
        add(sample=nm, group="AML cell line", cohort="other labs", hic_mcool=hic_, notes=ch_ + "; Hi-C not in the loop analyses")
    add(sample="THP-1 / HL-60 / OCI-AML3 / MV4-11 / others", group="AML cell line", cohort="survey only",
        notes="aml_data_survey.md: accessions listed, not downloaded (THP-1 ENCODE Hi-C 5.9 B pairs hg19; no local files)")
    add(sample="K562 (control)", group="CML cell line (control)", cohort="ENCODE / ChIP-Atlas",
        notes="hg38_k562ca (11 marks + cohesin), hg38_k562ct (CUT&Tag CTCF, H3K27ac, H3K27me3), k562_pval (p-value bigWigs); used as input control only")
    grp = {}
    for f in gl(str(PRIM / "fischer/chip/*.bigWig")):
        p_ = Path(f).name.split("_")
        key = f"{p_[3]}_{p_[4]}"
        grp.setdefault(key, []).append((p_[5], fsize(f)))
    for key, v in grp.items():
        add(sample=f"Fischer {key}", group="AML patient (cohesin cohort)", cohort="Fischer 2024 (GSE268031)",
            cuttag_ct_bigwig="CTCF" if any(m == "CTCF" for m, _ in v) else "", notes="ChIP-seq bigWigs (CNV-normalised): " + ",".join(sorted(m for m, _ in v))
            + f"; {sum(s for _, s in v) / 2**20:.0f} MB; mutation class = {key.split('_')[0]}; input folder fi_{key}; no per-patient Hi-C")
    for g, f in [("fi_aml_ctrl", "GSE267780_merged_aml_ctrl_10kb.fanc.h5"), ("fi_rad21_mut", "GSE267780_merged_rad21_mut_10kb.fanc.h5"), ("fi_sa2_mut", "GSE267780_merged_sa2_mut_10kb.fanc.h5")]:
        jf = REPO / f"results/aml_primary/targets/{g}_10kb.json"
        add(sample=f"Fischer pooled Hi-C {g}", group="AML pooled Hi-C", cohort="Fischer 2024 (GSE267780)", hic_mcool=f,
            hic_mcool_GB=round(fsize(PRIM / "fischer" / f) / 2**30, 2), hic_contacts_5kb_M=round(json.load(open(jf))["contacts"] / 1e6, 0) if jf.exists() else np.nan,
            notes="FAN-C HDF5 converted to cooler, 10 kb only (contacts at 10 kb)")
    for c in sorted(os.listdir(PRIM / "inputs_nonaml")):
        add(sample=f"non-AML {c}", group="non-AML cell (model training / control)", cohort="ENCODE", notes="inputs_nonaml/" + c)
    E = pd.DataFrame(extra)
    Inv = pd.concat([Inv, E], ignore_index=True)
    Inv.to_csv(OUT / "inventory.csv", index=False)
    inventory_types()
    log(f"inventory: {len(Inv)} rows")
    # markdown: compact table
    cols = ["sample", "group", "hic_mcool_GB", "hic_contacts_5kb_M", "cuttag_ac_MB", "cuttag_me3_MB", "cuttag_ct_MB", "wgbs_MB",
            "atac_bigwig", "rna_rsem", "loops_specific_n", "stripes_specific_n", "sv_junctions_n", "cnv_segments_n", "fusion", "mutation_group"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in Inv.iterrows():
        def fmt(v):
            if isinstance(v, float) and np.isnan(v):
                return ""
            if isinstance(v, (bool, np.bool_)):
                return "yes" if v else "no"
            if isinstance(v, str) and len(v) > 40:
                return v[:37] + "..."
            return str(v)
        md.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    (OUT / "inventory.md").write_text("\n".join(md) + "\n")


# ============================================================================================ tracks
EXTRA_HSPC = {"H3K27ac": ["SRX1901616", "SRX1089816", "SRX1089818"],
              "H3K27me3": ["SRX2018538", "SRX2018539", "SRX1089820"],
              "CTCF": ["SRX1738529"]}


def bin_chip(bw_path: str, S: dict, kb: int = 10) -> dict:
    """Mean per bin of a bigWig, {chrom: float32 array} (10 kb bins with one spare, or 1 kb bins)."""
    import pyBigWig
    bw = pyBigWig.open(str(bw_path))
    out = {}
    res = kb * 1000
    for c in CHROMS:
        n_full = S[c] // res
        n_arr = S[c] // RES + 1 if kb == 10 else S[c] // res
        a = np.full(n_arr, np.nan)
        a[:n_full] = np.array(bw.stats(c, 0, n_full * res, type="mean", nBins=n_full), dtype=float)
        out[c] = a.astype(np.float32)
    bw.close()
    return out


def save_tr(path: Path, d: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)


def _bin_job(j):
    dest, f, kb = j
    if dest.exists():
        return dest.name
    save_tr(dest, bin_chip(f, sizes(), kb))
    return dest.name


def stage_tracks() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    S = sizes()
    jobs = []
    for src in ("103", "673"):
        for m in MARKS:
            f = INP / f"xu_{src}/{m}.bigwig"
            if f.exists():
                jobs.append((WORK / f"trk10_{src}_{m}.npz", f, 10))
        jobs.append((WORK / f"meth1k_{src}.npz", PRIM / f"meth/xu_{src}/Meth.bigwig", 1))
    jobs.append((WORK / "meth1k_cd14.npz", HM / "meth/cd14mono/Meth.bigwig", 1))
    for m, srx in EXTRA_HSPC.items():
        for s in srx:
            f = WORK / f"hspc_extra/{s}.bw"
            if f.exists():
                jobs.append((WORK / f"raw10_{s}.npz", f, 10))

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(5) as ex:
        for nm in ex.map(_bin_job, jobs):
            log(f"[tracks] {nm}")
    # extra HSPC experiments: quantile-matched to the CD34+ track of the model inputs
    for m, srx in EXTRA_HSPC.items():
        base = dict(np.load(CACHE / f"trk_hspc_{m}.npz"))
        bm = np.concatenate([base[c + "|mean"][np.isfinite(base[c + "|mean"])] for c in CHROMS])
        bm = np.sort(bm)
        for s in srx:
            raw = WORK / f"raw10_{s}.npz"
            dest = WORK / f"hspcx_{s}_{m}.npz"
            if not raw.exists() or dest.exists():
                continue
            r = dict(np.load(raw))
            allv = np.concatenate([r[c] for c in CHROMS]).astype(float)
            ok = np.isfinite(allv)
            from scipy.stats import rankdata
            ranks = rankdata(allv[ok], method="average") - 1.0       # ties (zero coverage) share one quantile
            q = np.interp((ranks + 0.5) / ok.sum(), np.linspace(0, 1, len(bm)), bm)
            full = np.full(len(allv), np.nan)
            full[ok] = q
            out, k = {}, 0
            for c in CHROMS:
                n = len(r[c])
                out[c + "|mean"] = full[k:k + n].astype(np.float32)
                k += n
            save_tr(dest, out)
            log(f"[tracks] quantile-matched HSPC {s} ({m})")
    # CpG islands: fraction of every 10 kb bin
    dest = WORK / "cgi10.npz"
    if not dest.exists():
        cg = {c: np.zeros(nb(c, S)) for c in CHROMS}
        with gzip.open(CGI, "rt") as fh:
            for line in fh:
                f = line.split("\t")
                c, a, b = f[1], int(f[2]), int(f[3])
                if c not in cg:
                    continue
                for bi in range(a // RES, min((b - 1) // RES, len(cg[c]) - 1) + 1):
                    cg[c][bi] += (min(b, (bi + 1) * RES) - max(a, bi * RES)) / RES
        save_tr(dest, {c: v.astype(np.float32) for c, v in cg.items()})
        log("[tracks] CpG islands")


class Store:
    """Lazy track access: 10 kb mean arrays per (source, mark) and the methylation summaries."""

    def __init__(self):
        self.S = sizes()
        self._c = {}

    def chip(self, src: str, mark: str):
        k = (src, mark)
        if k in self._c:
            return self._c[k]
        f = CACHE / f"trk_{src}_{mark}.npz"
        if not f.exists():
            f = WORK / f"trk10_{src}_{mark}.npz"
            key = lambda c: c
        else:
            key = lambda c: c + "|mean"
        if not f.exists():
            self._c[k] = None
            return None
        z = np.load(f)
        d = {c: z[key(c)].astype(np.float64) for c in CHROMS}
        self._c[k] = d
        return d

    def hspcx(self, mark: str):
        """Mean of all HSPC experiments (the model-input CD34+ track and the quantile-matched extra ones)."""
        k = ("hspcm", mark)
        if k in self._c:
            return self._c[k]
        arrs = [self.chip("hspc", mark)]
        for s in EXTRA_HSPC[mark]:
            f = WORK / f"hspcx_{s}_{mark}.npz"
            if f.exists():
                z = np.load(f)
                arrs.append({c: z[c + "|mean"].astype(np.float64) for c in CHROMS})
        d = {c: np.mean([a[c] for a in arrs], axis=0) for c in CHROMS}
        self._c[k] = (d, len(arrs))
        return self._c[k]

    def meth(self, src: str):
        """{'mean','min','lmr'} -> {chrom: 10 kb array} of the methylated fraction (1 kb bins; lmr = fraction of
        1 kb bins below 0.5)."""
        k = ("meth", src)
        if k in self._c:
            return self._c[k]
        if src == "cd14":
            f = WORK / "meth1k_cd14.npz"
            raw = dict(np.load(f)) if f.exists() else None
        elif src in ("103", "673"):
            f = WORK / f"meth1k_{src}.npz"
            raw = dict(np.load(f)) if f.exists() else None
        else:
            f = CACHE / f"trk_{src}_Meth.npz"
            raw = dict(np.load(f)) if f.exists() else None
        if raw is None:
            self._c[k] = None
            return None
        out = {"mean": {}, "min": {}, "lmr": {}}
        for c in CHROMS:
            v = raw[c].astype(np.float64) - 1.0
            n10 = nb(c, self.S)
            pad = np.full(n10 * 10, np.nan)
            pad[:len(v)] = v[:n10 * 10]
            M = pad.reshape(n10, 10)
            with np.errstate(all="ignore"):
                out["mean"][c] = np.nanmean(M, axis=1)
                out["min"][c] = np.nanmin(M, axis=1)
                lm = np.where(np.isfinite(M), (M < 0.5).astype(float), np.nan)
                out["lmr"][c] = np.nanmean(lm, axis=1)
        self._c[k] = out
        return out

    def cgi(self):
        if "cgi" not in self._c:
            z = np.load(WORK / "cgi10.npz")
            self._c["cgi"] = {c: z[c].astype(np.float64) for c in CHROMS}
        return self._c["cgi"]


def genome_flat(d: dict) -> np.ndarray:
    return np.concatenate([d[c][np.isfinite(d[c])] for c in CHROMS])


def apply_chrom(d: dict, fn) -> dict:
    return {c: fn(d[c]) for c in CHROMS}


def arr_pct(d: dict) -> dict:
    """Genome-wide percentile (0 to 1) of every bin of one track."""
    allv = np.sort(genome_flat(d))
    return {c: np.where(np.isfinite(d[c]), np.searchsorted(allv, d[c]) / len(allv), np.nan) for c in CHROMS}


def sub_d(a: dict, b: dict) -> dict:
    return {c: a[c] - b[c] for c in CHROMS}


def mean_d(ds: list) -> dict:
    return {c: np.nanmean(np.stack([d[c] for d in ds]), axis=0) for c in CHROMS}


# ============================================================================================ features
class Iv:
    """Interval statistics of one 10 kb array: mean (prefix sums), min and max (sparse tables)."""

    def __init__(self, x: np.ndarray):
        self.n = len(x)
        fin = np.isfinite(x)
        self.cs = np.r_[0.0, np.cumsum(np.where(fin, x, 0.0))]
        self.cc = np.r_[0.0, np.cumsum(fin)]
        self.x = x
        self._mn = None
        self._mx = None

    def mean(self, lo, hi):
        lo = np.clip(lo, 0, self.n)
        hi = np.clip(hi, 0, self.n)
        c = self.cc[hi] - self.cc[lo]
        with np.errstate(all="ignore"):
            return np.where((hi > lo) & (c > 0), (self.cs[hi] - self.cs[lo]) / np.maximum(c, 1), np.nan)

    def _table(self, fill, fn):
        t = [np.where(np.isfinite(self.x), self.x, fill)]
        j = 1
        while 2 ** j <= self.n:
            p = t[-1]
            h = 2 ** (j - 1)
            t.append(fn(p[:-h], p[h:]))
            j += 1
        return t

    def _q(self, t, lo, hi, fn, fill):
        lo = np.clip(lo, 0, self.n)
        hi = np.clip(hi, 0, self.n)
        ln = np.maximum(hi - lo, 1)
        j = np.floor(np.log2(ln)).astype(int)
        out = np.full(len(lo), fill, float)
        for jj in np.unique(j):
            s = j == jj
            tj = t[jj]
            a = tj[np.minimum(lo[s], len(tj) - 1)]
            b = tj[np.minimum(np.maximum(hi[s] - 2 ** jj, 0), len(tj) - 1)]
            out[s] = fn(a, b)
        out[hi <= lo] = np.nan
        out[np.isinf(out)] = np.nan
        return out

    def min(self, lo, hi):
        if self._mn is None:
            self._mn = self._table(np.inf, np.minimum)
        return self._q(self._mn, lo, hi, np.minimum, np.inf)

    def max(self, lo, hi):
        if self._mx is None:
            self._mx = self._table(-np.inf, np.maximum)
        return self._q(self._mx, lo, hi, np.maximum, -np.inf)


def combos(a: np.ndarray, b: np.ndarray) -> dict:
    with np.errstate(all="ignore"):
        return {"A": a, "B": b, "min": np.fmin(a, b), "max": np.fmax(a, b), "sum": a + b, "prod": a * b}


def object_intervals(D: pd.DataFrame, k: int, n: int):
    """(lo, hi) of anchor A, anchor B and the 'mid' piece of every object for smoothing k (bins each side).
    Loops: A and B are one bin, mid the bins strictly between. Stripes: A the 4 anchor bins (+-k), B the
    extent block (not smoothed), mid the extent block (maximum)."""
    al, ah, bl, bh = (D[c].values.astype(int) for c in ("a_lo", "a_hi", "b_lo", "b_hi"))
    loop = (D.kind.values == "loop")
    Alo, Ahi = al - k, ah + k
    Blo = np.where(loop, bl - k, bl)
    Bhi = np.where(loop, bh + k, bh)
    Mlo = np.where(loop, ah, bl)
    Mhi = np.where(loop, bl, bh)
    return Alo, Ahi, Blo, Bhi, Mlo, Mhi


def block(F: dict, idx: np.ndarray, D: pd.DataFrame, iv: Iv, stat: str, tag: str, flags=None,
          scales=SCALES, mid=True) -> None:
    """Fill F[tag|pos|scale] for the objects idx (rows of D) from one interval object."""
    Ds = D.iloc[idx]
    for sc, k in scales.items():
        Alo, Ahi, Blo, Bhi, Mlo, Mhi = object_intervals(Ds, k, iv.n)
        f = getattr(iv, "min" if stat == "min" else "mean")
        a, b = f(Alo, Ahi), f(Blo, Bhi)
        cb = combos(a, b)
        if flags is not None and k in (0, 2):
            ga, gb = flags[0], flags[1]
            loop = (Ds.kind.values == "loop")
            pa = np.where(ga > 0, a, np.where(gb > 0, b, np.nan))
            pt = np.where(ga > 0, b, np.where(gb > 0, a, np.nan))
            cb["prom"] = np.where(loop, pa, np.nan)
            cb["partner"] = np.where(loop, pt, np.nan)
        if mid and sc == "s0":
            m = f(Mlo, Mhi) if stat != "mean" else iv.mean(Mlo, Mhi)
            ext = iv.max(Mlo, Mhi) if stat == "mean" else m
            cb["mid"] = np.where(Ds.kind.values == "loop", m, ext)
        for pos, v in cb.items():
            key = f"{tag}|{pos}|{sc}"
            if key not in F:
                F[key] = np.full(len(D), np.nan)
            F[key][idx] = v


CHIP_REPS = ["own", "hspc", "diff", "diffz", "diffq", "diffH", "diffA", "hspcm", "diffM"]


def chip_reps(st: Store, p: str, mark: str, reps: list) -> dict:
    out = {}
    own = st.chip(p, mark)
    hsp = st.chip("hspc", mark)
    if own is None:
        return out
    if "own" in reps:
        out["own"] = own
    if "hspc" in reps:
        out["hspc"] = hsp
    diff = sub_d(own, hsp)
    if "diff" in reps:
        out["diff"] = diff
    if "diffz" in reps:
        fl = genome_flat(diff)
        med = np.median(fl)
        mad = 1.4826 * np.median(np.abs(fl - med)) + 1e-9
        out["diffz"] = apply_chrom(diff, lambda a: (a - med) / mad)
    if "diffq" in reps:
        out["diffq"] = sub_d(arr_pct(own), arr_pct(hsp))
    if "diffH" in reps:
        hs = [st.chip(h, mark) for h in ("103", "673")]
        hs = [h for h in hs if h is not None]
        if hs:
            out["diffH"] = sub_d(own, mean_d(hs))
    if "diffA" in reps:
        oth = [st.chip(q, mark) for q in DONORS10 if q != p]
        oth = [o for o in oth if o is not None]
        if oth:
            out["diffA"] = sub_d(own, mean_d(oth))
    if "hspcm" in reps or "diffM" in reps:
        hm, nh = st.hspcx(mark)
        if "hspcm" in reps:
            out["hspcm"] = hm
        if "diffM" in reps:
            out["diffM"] = sub_d(own, hm)
    return out


METH_REPS = ["own", "H", "C", "dH", "dC", "dA"]
METH_WGBS = ["018", "027", "168", "270", "472", "546", "629", "773", "1071", "1324", "260", "424", "496", "504", "566", "798", "838", "933"]


def meth_reps(st: Store, p: str, stat: str, reps: list) -> dict:
    own = st.meth(p)
    out = {}
    if own is None:
        return out
    o = own[stat]
    if "own" in reps:
        out["own"] = o
    H = [st.meth(h) for h in ("103", "673")]
    H = [h[stat] for h in H if h is not None]
    Hm = mean_d(H) if H else None
    C = st.meth("cd14")
    Cm = C[stat] if C is not None else None
    if "H" in reps and Hm is not None:
        out["H"] = Hm
    if "C" in reps and Cm is not None:
        out["C"] = Cm
    if "dH" in reps and Hm is not None:
        out["dH"] = sub_d(o, Hm)
    if "dC" in reps and Cm is not None:
        out["dC"] = sub_d(o, Cm)
    if "dA" in reps:
        part = GROUP.get(p)
        oth = [st.meth(q) for q in METH_WGBS if q != p and not (part and GROUP.get(q) == part)]
        oth = [x[stat] for x in oth if x is not None]
        if oth:
            out["dA"] = sub_d(o, mean_d(oth))
    return out


def build_features(D: pd.DataFrame, chip_reps_use=None, meth_reps_use=None) -> pd.DataFrame:
    """Feature matrix (rows of D = objects). Columns: <tag>|<position>|<scale> with tag = <mark>:<rep>
    (ac, me3, ct), meth:<stat>:<rep>, cgi."""
    import aml_loop_classifier as clf
    st = Store()
    genes = None
    S = st.S
    try:
        genes = clf.load_genes(S)
    except Exception as e:                      # noqa: BLE001
        log(f"genes unavailable: {e}")
    chip_reps_use = chip_reps_use or CHIP_REPS
    meth_reps_use = meth_reps_use or METH_REPS
    F: dict = {}
    pats = sorted(D.patient.unique())
    for p in pats:
        ip = np.where(D.patient.values == p)[0]
        for mark in MARKS:
            R = chip_reps(st, p, mark, chip_reps_use)
            for rep, tr in R.items():
                for c in CHROMS:
                    ic = ip[D.chrom.values[ip] == c]
                    if not len(ic):
                        continue
                    iv = Iv(tr[c])
                    gf = None
                    if genes is not None:
                        g = genes[c]["tss"]
                        Ds = D.iloc[ic]
                        ga = g[np.clip(Ds.a_lo.values.astype(int), 0, len(g) - 1)]
                        gb = g[np.clip(Ds.b_lo.values.astype(int), 0, len(g) - 1)]
                        gf = (ga, gb)
                    block(F, ic, D, iv, "mean", f"{SHORT[mark]}:{rep}", gf, mid=True)
        for stat in ("mean", "min", "lmr"):
            R = meth_reps(st, p, stat, meth_reps_use)
            for rep, tr in R.items():
                for c in CHROMS:
                    ic = ip[D.chrom.values[ip] == c]
                    if not len(ic):
                        continue
                    iv = Iv(tr[c])
                    block(F, ic, D, iv, "min" if stat == "min" else "mean", f"meth:{stat}:{rep}", None, mid=(stat != "min"))
        for c in CHROMS:
            ic = ip[D.chrom.values[ip] == c]
            if not len(ic):
                continue
            block(F, ic, D, Iv(st.cgi()[c]), "mean", "cgi", None, mid=True)
        log(f"[features] {p}: {len(F)} columns")
    return pd.DataFrame(F)


def load_sites() -> dict:
    """CTCF motif hits (JASPAR MA0139.2, hg38, >= 70 percent of the maximal log-odds score; the scan of the loop
    classifier): {chrom: (centre, score, strand)} sorted by position."""
    out = {}
    for c in CHROMS:
        z = np.load(CACHE / f"motif70_{c}.npz")
        pos = np.concatenate([z["fwd_pos"], z["rev_pos"]]) + 9
        sc = np.concatenate([z["fwd_score"], z["rev_score"]]).astype(float)
        stn = np.r_[np.ones(len(z["fwd_pos"])), -np.ones(len(z["rev_pos"]))]
        o = np.argsort(pos)
        out[c] = (pos[o], sc[o], stn[o])
    return out


def motif_features(D: pd.DataFrame) -> pd.DataFrame:
    """Methylation and CTCF signal at the CTCF motif hits inside each anchor (anchor bin +-1 bin; stripes: the anchor block only).
    Per anchor: number of hits, best hit (score) methylation / patient minus healthy, the CTCF-bound hit (largest
    own CTCF signal at the hit) with its methylation / patient-minus-healthy methylation / CTCF minus HSPC CTCF, and
    the mean methylation over the hits. Pair level: A, B, min, max over the two anchors (loops)."""
    import pyBigWig
    sites = load_sites()
    bws = {}

    def bw(path):
        if path not in bws:
            bws[path] = pyBigWig.open(str(path)) if os.path.exists(str(path)) else None
        return bws[path]

    def val(b, c, s, e):
        if b is None:
            return np.nan
        try:
            v = b.stats(c, max(int(s), 0), int(e), type="mean")[0]
        except Exception:                           # noqa: BLE001
            return np.nan
        return np.nan if v is None else float(v)

    hctcf = bw(INP / "hspc_cd34/CTCF.bigwig")
    wg = {Path(f).name.split("_allc_")[1].split(".")[0]: f for f in gl(str(PRIM / "wgbs/*_allc_*.CGN.10bp.bw"))}
    cache = {}
    names = ["n", "best_score", "best_meth", "best_dmeth", "bound_ctcf", "bound_dctcf", "bound_meth", "bound_dmeth", "mean_meth", "mean_dmeth"]

    def region(p, c, lo, hi):
        key = (p, c, lo, hi)
        if key in cache:
            return cache[key]
        pos, sc, stn = sites[c]
        i0, i1 = np.searchsorted(pos, (lo - 1) * RES), np.searchsorted(pos, (hi + 1) * RES)
        res = dict.fromkeys(names, np.nan)
        res["n"] = float(i1 - i0)
        if i1 > i0:
            own = bw(wg[p.lstrip("0")]) if p.lstrip("0") in wg else None
            Hs = [bw(wg[h]) for h in ("103", "673") if h in wg]
            cb = bw(INP / f"xu_{p}/CTCF.bigwig")
            rows = []
            for j in range(i0, i1):
                ctr = int(pos[j])
                mo = val(own, c, ctr - 9, ctr + 10)
                mh = np.nanmean([val(h, c, ctr - 9, ctr + 10) for h in Hs]) if Hs else np.nan
                co = val(cb, c, ctr - 100, ctr + 100)
                ch = val(hctcf, c, ctr - 100, ctr + 100)
                rows.append((sc[j], mo, mh, co, ch))
            R = np.array(rows, float)
            jb = int(np.nanargmax(R[:, 0]))
            res["best_score"] = R[jb, 0]
            res["best_meth"] = R[jb, 1]
            res["best_dmeth"] = R[jb, 1] - R[jb, 2]
            if np.isfinite(R[:, 3]).any():
                jc = int(np.nanargmax(R[:, 3]))
                res["bound_ctcf"] = R[jc, 3]
                res["bound_dctcf"] = R[jc, 3] - R[jc, 4]
                res["bound_meth"] = R[jc, 1]
                res["bound_dmeth"] = R[jc, 1] - R[jc, 2]
            with np.errstate(all="ignore"):
                res["mean_meth"] = np.nanmean(R[:, 1]) if np.isfinite(R[:, 1]).any() else np.nan
                res["mean_dmeth"] = np.nanmean(R[:, 1] - R[:, 2]) if np.isfinite(R[:, 1] - R[:, 2]).any() else np.nan
        cache[key] = res
        return res

    out = {f"mot:{n}|{pos}": np.full(len(D), np.nan) for n in names for pos in ("A", "B", "min", "max")}
    for r, (p, c, kind, al, ah, bl) in enumerate(zip(D.patient.values, D.chrom.values, D.kind.values, D.a_lo.values, D.a_hi.values, D.b_lo.values)):
        ra = region(p, c, int(al), int(ah))
        rb = region(p, c, int(bl), int(bl) + 1) if kind == "loop" else None
        for n in names:
            a = ra[n]
            b = rb[n] if rb is not None else np.nan
            out[f"mot:{n}|A"][r] = a
            out[f"mot:{n}|B"][r] = b
            if n != "n":
                out[f"mot:{n}|min"][r] = np.fmin(a, b) if rb is not None else a
                out[f"mot:{n}|max"][r] = np.fmax(a, b) if rb is not None else a
            else:
                out[f"mot:{n}|min"][r] = min(a, b) if rb is not None else a
                out[f"mot:{n}|max"][r] = max(a, b) if rb is not None else a
        if r % 4000 == 0:
            log(f"[motif] {r}/{len(D)}")
    return pd.DataFrame(out)


def load_objects() -> pd.DataFrame:
    """Objects of the loop detection (positives and controls) with measured scores and the loop classifier features."""
    O = pd.read_csv(DET / "object_scores.csv.gz", dtype={"patient": str})
    O["patient"] = O.patient.astype(str)
    O["donor"] = O.patient.map(lambda p: GROUP.get(p, p))
    O["logdist"] = np.where(O.kind == "loop", np.log10(np.maximum((O.b_lo - O.a_lo).abs() * RES, 1)),
                            np.log10(np.maximum((O.b_lo - O.a_lo).abs() * RES, 1)))
    O["ptype"] = O.pair_type
    return O


def stage_features() -> None:
    O = load_objects()
    OUT.mkdir(parents=True, exist_ok=True)
    F = build_features(O)
    M = motif_features(O)
    base = pd.read_csv(CLF, dtype={"patient": str})
    keep = [c for c in base.columns if c not in {"patient", "kind", "set", "pair_id", "y", "chrom", "a_lo", "a_hi", "b_lo", "b_hi", "pair_type",
                                                  "lab_e_pat", "lab_e_hspc"} and not c.startswith("_")]
    B = O[["patient", "set", "pair_id"]].merge(base[["patient", "set", "pair_id"] + keep], on=["patient", "set", "pair_id"], how="left")
    B = B[keep].add_prefix("clf:")
    B.index = O.index
    FF = pd.concat([F, M, B], axis=1)
    FF.to_pickle(WORK / "features.pkl")
    log(f"[features] matrix {FF.shape}")


# ============================================================================================ correlations
LOOP_CLASSES = {"all": lambda d: np.ones(len(d), bool), "PE": lambda d: d.ptype.values == "PE", "PS": lambda d: d.ptype.values == "PS",
                "other": lambda d: d.ptype.values == "other", "recurrent": lambda d: d.cls.values == "recurrent",
                "private": lambda d: d.cls.values == "private", "shared1": lambda d: d.cls.values == "shared1"}
STRIPE_CLASSES = {"all": lambda d: np.ones(len(d), bool), "enh": lambda d: d.ptype.values == "enh", "rep": lambda d: d.ptype.values == "rep",
                  "other": lambda d: d.ptype.values == "other", "recurrent": lambda d: d.cls.values == "recurrent",
                  "private": lambda d: d.cls.values == "private", "shared1": lambda d: d.cls.values == "shared1"}
MAIN_CHIP = {"own", "hspc", "diff"}
MAIN_METH = {"own", "dH", "dC", "H", "C"}


def parse_feature(f: str) -> dict:
    """Name -> family, tag, representation, position, scale; tier 'main' for the requested representations."""
    if f.startswith("clf:"):
        return dict(family="clf", tag=f[4:], rep="", pos="", scale="", tier="base")
    parts = f.split("|")
    tag = parts[0]
    pos = parts[1] if len(parts) > 1 else ""
    sc = parts[2] if len(parts) > 2 else ""
    if tag.startswith("mot:"):
        return dict(family="motif", tag=tag[4:], rep="", pos=pos, scale="", tier="main")
    if tag.startswith("meth:"):
        _, stat, rep = tag.split(":")
        return dict(family="meth", tag=f"meth_{stat}", rep=rep, pos=pos, scale=sc, tier="main" if rep in MAIN_METH else "extra")
    if tag == "cgi":
        return dict(family="cgi", tag="cgi", rep="static", pos=pos, scale=sc, tier="main")
    mk, rep = tag.split(":")
    return dict(family=mk, tag=mk, rep=rep, pos=pos, scale=sc, tier="main" if rep in MAIN_CHIP else "extra")


def rank_cols(X: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(X, axis=0)


def col_groups(X: pd.DataFrame, donors: np.ndarray, min_frac: float = 0.1, max_nan: float = 0.05):
    """Columns with the same set of usable rows: donors without data (finite fraction < min_frac) are dropped, sporadic
    NaN (<= max_nan of the usable rows) are median-imputed, other columns get their own finite-row mask.
    Returns [(columns, row mask, imputed matrix)]."""
    ud = np.unique(donors)
    fin = np.isfinite(X.values)
    ok_d = {c: np.array([fin[donors == d, j].mean() >= min_frac for d in ud]) for j, c in enumerate(X.columns)}
    groups: dict = {}
    for j, c in enumerate(X.columns):
        okd = ok_d[c]
        if not okd.any():
            continue
        rows = np.isin(donors, ud[okd])
        miss = 1 - fin[rows, j].mean()
        if miss <= max_nan:
            key = ("donors", tuple(okd))
        else:
            key = ("own", c)
        groups.setdefault(key, []).append((j, c, rows))
    out = []
    for key, lst in groups.items():
        cols = [c for _, c, _ in lst]
        rows = lst[0][2]
        M = X.values[:, [j for j, _, _ in lst]].astype(float).copy()
        if key[0] == "donors":
            med = np.nanmedian(M[rows], axis=0)
            for k in range(M.shape[1]):
                bad = ~np.isfinite(M[:, k])
                M[bad, k] = med[k]
        else:
            rows = rows & np.isfinite(M[:, 0])
        out.append((cols, rows, M))
    return out


def _partial(rxy, rxz1, rxz2, ryz1, ryz2, rz1z2):
    with np.errstate(all="ignore"):
        a = (rxy - rxz1 * ryz1) / np.sqrt((1 - rxz1 ** 2) * (1 - ryz1 ** 2))
        b = (rxz2 - rxz1 * rz1z2) / np.sqrt((1 - rxz1 ** 2) * (1 - rz1z2 ** 2))
        c = (ryz2 - ryz1 * rz1z2) / np.sqrt((1 - ryz1 ** 2) * (1 - rz1z2 ** 2))
        return (a - b * c) / np.sqrt((1 - b ** 2) * (1 - c ** 2))


def pooled_stats(Rx, Ry, R1, R2, di, nd, W):
    """Pearson of the ranks (= pooled Spearman) and the partial correlation controlling for R1, R2, for every
    donor resample (rows of W: multiplicities per donor). Sufficient statistics per donor."""
    def dsum(A):
        out = np.zeros((nd,) + A.shape[1:])
        np.add.at(out, di, A)
        return out
    n = dsum(np.ones(len(Ry)))
    sx, sxx = dsum(Rx), dsum(Rx * Rx)
    sy, syy = dsum(Ry), dsum(Ry * Ry)
    s1, s11 = dsum(R1), dsum(R1 * R1)
    s2, s22 = dsum(R2), dsum(R2 * R2)
    sxy = dsum(Rx * Ry[:, None])
    sx1 = dsum(Rx * R1[:, None])
    sx2 = dsum(Rx * R2[:, None])
    sy1, sy2, s12 = dsum(Ry * R1), dsum(Ry * R2), dsum(R1 * R2)

    def corr(N, a, b, aa, bb, ab):
        with np.errstate(all="ignore"):
            cov = ab - a * b / N
            return cov / np.sqrt((aa - a * a / N) * (bb - b * b / N))
    N = W @ n
    T = lambda v: W @ v
    Nx = N[:, None]
    rxy = corr(Nx, T(sx), T(sy)[:, None], T(sxx), T(syy)[:, None], T(sxy))
    rx1 = corr(Nx, T(sx), T(s1)[:, None], T(sxx), T(s11)[:, None], T(sx1))
    rx2 = corr(Nx, T(sx), T(s2)[:, None], T(sxx), T(s22)[:, None], T(sx2))
    ry1 = corr(N, T(sy), T(s1), T(syy), T(s11), T(sy1))[:, None]
    ry2 = corr(N, T(sy), T(s2), T(syy), T(s22), T(sy2))[:, None]
    r12 = corr(N, T(s1), T(s2), T(s11), T(s22), T(s12))[:, None]
    return rxy, _partial(rxy, rx1, rx2, ry1, ry2, r12)


def boot_weights(nd: int, B: int, rng) -> np.ndarray:
    W = np.zeros((B + 1, nd))
    W[0] = 1
    for b in range(1, B + 1):
        W[b] = np.bincount(rng.integers(0, nd, nd), minlength=nd)
    return W


def corr_class(Xp: pd.DataFrame, y: np.ndarray, z1: np.ndarray, z2: np.ndarray, donors: np.ndarray, B: int, seed: int,
               min_donor_n: int = 10) -> pd.DataFrame:
    """Pooled Spearman and partial Spearman with donor bootstrap and per-donor Spearman of every column."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    rows_out = []
    for cols, rows, M in col_groups(Xp, donors):
        r = np.where(rows)[0]
        dn = donors[r]
        ud, di = np.unique(dn, return_inverse=True)
        if len(ud) < 3 or len(r) < 30:
            continue
        Rx = rank_cols(M[r])
        Ry, R1, R2 = rankdata(y[r]), rankdata(z1[r]), rankdata(z2[r])
        W = boot_weights(len(ud), B, rng)
        rho, par = pooled_stats(Rx, Ry, R1, R2, di, len(ud), W)
        # per-donor Spearman (ranks within donor)
        pd_rho = np.full((len(ud), len(cols)), np.nan)
        for k in range(len(ud)):
            s = di == k
            if s.sum() < min_donor_n:
                continue
            a = rank_cols(M[r][s]) - (s.sum() + 1) / 2
            b = rankdata(y[r][s]) - (s.sum() + 1) / 2
            with np.errstate(all="ignore"):
                pd_rho[k] = (a * b[:, None]).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum())
        for j, c in enumerate(cols):
            v = pd_rho[:, j]
            ok = np.isfinite(v)
            rows_out.append({"feature": c, "n": len(r), "n_donors": len(ud),
                             "rho": rho[0, j], "rho_lo": np.nanpercentile(rho[1:, j], 2.5), "rho_hi": np.nanpercentile(rho[1:, j], 97.5),
                             "partial": par[0, j], "partial_lo": np.nanpercentile(par[1:, j], 2.5), "partial_hi": np.nanpercentile(par[1:, j], 97.5),
                             "rho_donor_median": np.nanmedian(v) if ok.any() else np.nan, "donors_evaluable": int(ok.sum()),
                             "donors_positive": int((v[ok] > 0).sum())})
    return pd.DataFrame(rows_out)


def perm_max_null(Xp: pd.DataFrame, y: np.ndarray, donors: np.ndarray, B: int, seed: int, cols: list) -> np.ndarray:
    """Maximum |pooled Spearman| over the given columns when y is permuted within donor (family-wise null)."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    out = np.zeros(B)
    sub = Xp[cols]
    groups = col_groups(sub, donors)
    prep = []
    for cs, rows, M in groups:
        r = np.where(rows)[0]
        prep.append((r, rank_cols(M[r]) - (len(r) + 1) / 2))
    for b in range(B):
        yp = y.copy()
        for d in np.unique(donors):
            ix = np.where(donors == d)[0]
            yp[ix] = y[rng.permutation(ix)]
        m = 0.0
        for r, R in prep:
            ry = rankdata(yp[r]) - (len(r) + 1) / 2
            with np.errstate(all="ignore"):
                c = (R * ry[:, None]).sum(0) / np.sqrt((R * R).sum(0) * (ry * ry).sum())
            m = max(m, np.nanmax(np.abs(c)))
        out[b] = m
    return out


def auroc_class(Xp: pd.DataFrame, Xc: pd.DataFrame, dp: np.ndarray, dc: np.ndarray, B: int, seed: int, min_donor_n: int = 10) -> pd.DataFrame:
    """AUROC (positives vs matched controls) of every column, pooled with donor bootstrap and per donor.
    The bootstrap uses the pooled ranks of the full sample (AUROC = 0.5 + (mean rank positives - mean rank controls) / N)."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    X = pd.concat([Xp, Xc], ignore_index=True)
    don = np.r_[dp, dc]
    isp = np.r_[np.ones(len(dp), bool), np.zeros(len(dc), bool)]
    rows_out = []
    for cols, rows, M in col_groups(X, don):
        r = np.where(rows)[0]
        ud, di = np.unique(don[r], return_inverse=True)
        pm = isp[r]
        if len(ud) < 3 or pm.sum() < 20 or (~pm).sum() < 20:
            continue
        R = rank_cols(M[r])
        N = len(r)
        npos = np.zeros(len(ud))
        nneg = np.zeros(len(ud))
        np.add.at(npos, di[pm], 1)
        np.add.at(nneg, di[~pm], 1)
        spos = np.zeros((len(ud), len(cols)))
        sneg = np.zeros((len(ud), len(cols)))
        np.add.at(spos, di[pm], R[pm])
        np.add.at(sneg, di[~pm], R[~pm])
        W = boot_weights(len(ud), B, rng)
        Np, Nn = W @ npos, W @ nneg
        with np.errstate(all="ignore"):
            a = 0.5 + ((W @ spos) / Np[:, None] - (W @ sneg) / Nn[:, None]) / N
        # exact pooled AUROC for the point estimate and per donor
        n1, n0 = pm.sum(), (~pm).sum()
        exact = (R[pm].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        per = np.full((len(ud), len(cols)), np.nan)
        for k in range(len(ud)):
            s = di == k
            if pm[s].sum() < min_donor_n or (~pm[s]).sum() < min_donor_n:
                continue
            Rk = rank_cols(M[r][s])
            n1k, n0k = pm[s].sum(), (~pm[s]).sum()
            per[k] = (Rk[pm[s]].sum(0) - n1k * (n1k + 1) / 2) / (n1k * n0k)
        for j, c in enumerate(cols):
            v = per[:, j]
            ok = np.isfinite(v)
            rows_out.append({"feature": c, "auroc": exact[j], "auroc_lo": np.nanpercentile(a[1:, j], 2.5), "auroc_hi": np.nanpercentile(a[1:, j], 97.5),
                             "auroc_donor_median": np.nanmedian(v) if ok.any() else np.nan, "auroc_donors_above": int((v[ok] > 0.5).sum()),
                             "auroc_donors_evaluable": int(ok.sum()), "n_pos_auc": int(n1), "n_ctl_auc": int(n0)})
    return pd.DataFrame(rows_out)


def stage_corr(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    O = load_objects()
    F = pd.read_pickle(WORK / "features.pkl")
    tab, tab_auc = [], []
    perm_rows = []
    for kind, classes in (("loop", LOOP_CLASSES), ("stripe", STRIPE_CLASSES)):
        sel = (O.kind == kind).values
        P = O[sel & (O.set == "pos").values & O["gain|measured"].notna().values].copy()
        Xall = F.loc[P.index]
        feats = [c for c in F.columns if not (np.isfinite(Xall[c].values).sum() < 40)]
        Xall = Xall[feats]
        main = [c for c in feats if parse_feature(c)["tier"] in ("main", "base")]
        for cname, cfn in classes.items():
            m = cfn(P)
            if m.sum() < 60:
                continue
            Pc, Xc_ = P[m], Xall[m]
            log(f"[corr] {kind}/{cname}: {len(Pc)} objects, {Pc.donor.nunique()} donors, {len(feats)} features")
            res = corr_class(Xc_.reset_index(drop=True), Pc["gain|measured"].values, Pc.logdist.values, Pc.hspc.values,
                             Pc.donor.values, args.boot, SEED)
            res.insert(0, "class", cname)
            res.insert(0, "kind", kind)
            tab.append(res)
            # family-wise null for the main features (columns with data for >= 60 percent of the class objects)
            ncols = [c for c in main if c in Xc_.columns and parse_feature(c)["family"] != "clf"
                     and np.isfinite(Xc_[c].values).sum() >= max(100, 0.6 * len(Pc))]
            nul = perm_max_null(Xc_.reset_index(drop=True), Pc["gain|measured"].values, Pc.donor.values, 200, SEED + 1, ncols)
            perm_rows.append({"kind": kind, "class": cname, "n": len(Pc), "n_features": len(ncols), "max_abs_rho_null_median": np.median(nul),
                              "max_abs_rho_null_p95": np.percentile(nul, 95), "max_abs_rho_null_max": nul.max()})
            # AUROC against matched controls. Pair-type classes (PE, PS, enh, rep, other) are defined from the patient's own
            # marks, so their controls are the shared loops / stripes of the same own-mark type; all / recurrent / private use the
            # controls matched (pair id) to the class positives.
            for ctl in ("A_hspc", "A_do"):
                C = O[sel & (O.set == ctl).values]
                if cname in ("PE", "PS", "enh", "rep", "other"):
                    C = C[(C.ptype == cname) & C.patient.isin(Pc.patient)]
                else:
                    key = set(zip(Pc.patient, Pc.pair_id))
                    C = C[[(p, q) in key for p, q in zip(C.patient, C.pair_id)]]
                if len(C) < 40:
                    continue
                a = auroc_class(Xc_.reset_index(drop=True), F.loc[C.index, feats].reset_index(drop=True), Pc.donor.values, C.donor.values, args.boot, SEED)
                a.insert(0, "control", ctl)
                a.insert(0, "class", cname)
                a.insert(0, "kind", kind)
                tab_auc.append(a)
    T = pd.concat(tab, ignore_index=True)
    A = pd.concat(tab_auc, ignore_index=True)
    meta = pd.DataFrame([parse_feature(f) for f in T.feature]).drop(columns=[])
    T = pd.concat([T.reset_index(drop=True), meta], axis=1)
    meta = pd.DataFrame([parse_feature(f) for f in A.feature])
    A = pd.concat([A.reset_index(drop=True), meta], axis=1)
    T.to_csv(OUT / "correlations_all_features.csv.gz", index=False, float_format="%.4f")
    A.to_csv(OUT / "auroc_all_features.csv.gz", index=False, float_format="%.4f")
    pd.DataFrame(perm_rows).to_csv(OUT / "correlation_familywise_null.csv", index=False, float_format="%.4f")
    log(f"[corr] {len(T)} correlation rows, {len(A)} AUROC rows")
    summarize_corr(T, A, pd.DataFrame(perm_rows))


def summarize_corr(T: pd.DataFrame, A: pd.DataFrame, nul: pd.DataFrame) -> None:
    """Ranked tables of the top features per class."""
    Tm = T[T.tier != "extra"].copy()
    ncls = Tm.groupby(["kind", "class"]).n.transform("max")
    Tm = Tm[Tm.n >= np.maximum(100, 0.6 * ncls)]          # features with data for most objects of the class
    Tm["abs_rho"] = Tm.rho.abs()
    Tm["ci_excludes_0"] = (Tm.rho_lo > 0) | (Tm.rho_hi < 0)
    Tm["sign_consistent"] = np.where(Tm.rho > 0, Tm.donors_positive, Tm.donors_evaluable - Tm.donors_positive)
    top = []
    for (k, c), g in Tm[Tm.family != "clf"].groupby(["kind", "class"]):
        top.append(g.sort_values("abs_rho", ascending=False).head(25))
    pd.concat(top).to_csv(OUT / "top_features_correlation.csv", index=False, float_format="%.4f")
    Am = A[(A.tier != "extra") & (A.control == "A_hspc")].copy()
    Am = Am[Am.n_pos_auc >= np.maximum(100, 0.6 * Am.groupby(["kind", "class"]).n_pos_auc.transform("max"))]
    Am["eff"] = (Am.auroc - 0.5).abs()
    top = []
    for (k, c), g in Am[Am.family != "clf"].groupby(["kind", "class"]):
        top.append(g.sort_values("eff", ascending=False).head(25))
    pd.concat(top).to_csv(OUT / "top_features_auroc.csv", index=False, float_format="%.4f")
    expected_table(T, A)


EXPECTED = [("loop", "PE", f) for f in ["ac:diff|min|s0", "ac:diff|max|s0", "ac:diff|prod|s0", "ac:diff|prom|s0", "ac:diff|partner|s0", "ac:diff|min|s2", "ac:own|min|s0"]] + \
    [("loop", "PS", f) for f in ["me3:diff|partner|s0", "me3:diff|max|s0", "me3:diff|min|s0", "me3:diff|prom|s0", "me3:own|partner|s0", "me3:diff|max|s2"]] + \
    [("stripe", "rep", f) for f in ["me3:diff|A|s0", "me3:diff|A|s2", "me3:diff|B|s0", "me3:own|A|s0", "me3:diff|prod|s0"]] + \
    [("stripe", "enh", f) for f in ["ac:diff|A|s0", "ac:diff|A|s2", "ac:diff|B|s0", "ac:own|A|s0", "ac:diff|prod|s0"]] + \
    [(k, c, f) for k in ("loop", "stripe") for c in ("all", "recurrent", "private") for f in ["ct:diff|min|s0", "ct:diff|max|s0", "ct:diff|A|s2", "ct:own|min|s0"]] + \
    [(k, c, f) for k in ("loop", "stripe") for c in ("all", "recurrent", "private") for f in ["mot:bound_meth|min", "mot:bound_dmeth|min", "mot:best_meth|min", "mot:best_dmeth|min", "mot:bound_dctcf|min"]] + \
    [(k, c, f) for k in ("loop", "stripe") for c in ("all", "private") for f in ["meth:mean:own|min|s0", "meth:mean:dH|min|s0", "meth:mean:dC|min|s0", "meth:lmr:own|max|s0"]]


def expected_table(T: pd.DataFrame, A: pd.DataFrame) -> None:
    """The relationships named in the question, one row per (class, canonical feature)."""
    rows = []
    for kind, cl, f in EXPECTED:
        t = T[(T.kind == kind) & (T["class"] == cl) & (T.feature == f)]
        a = A[(A.kind == kind) & (A["class"] == cl) & (A.feature == f) & (A.control == "A_hspc")]
        if t.empty:
            continue
        t = t.iloc[0]
        r = {"kind": kind, "class": cl, "feature": f, "n": t.n, "n_donors": t.n_donors, "rho": t.rho, "rho_lo": t.rho_lo, "rho_hi": t.rho_hi,
             "partial": t.partial, "partial_lo": t.partial_lo, "partial_hi": t.partial_hi, "donors_positive": t.donors_positive,
             "donors_evaluable": t.donors_evaluable}
        if len(a):
            a = a.iloc[0]
            r.update({"auroc": a.auroc, "auroc_lo": a.auroc_lo, "auroc_hi": a.auroc_hi, "auroc_donors_above": a.auroc_donors_above,
                      "auroc_donors_evaluable": a.auroc_donors_evaluable})
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "expected_relationships.csv", index=False, float_format="%.4f")


# ============================================================================================ boosting
def sel(F: pd.DataFrame, fam=None, reps=None, pos=None, scales=None, tags=None) -> list:
    out = []
    for c in F.columns:
        if c.startswith("clf:"):
            continue
        m = parse_feature(c)
        if fam and m["family"] not in fam:
            continue
        if reps and m["rep"] not in reps:
            continue
        if pos and m["pos"] not in pos:
            continue
        if scales and m["scale"] not in scales:
            continue
        if tags and m["tag"] not in tags:
            continue
        out.append(c)
    return out


def engineered(F: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Pairwise / interaction features (a) computed from the anchor tracks of one object kind."""
    g = lambda n: F[n].values if n in F else np.full(len(F), np.nan)
    conv = g("clf:motif_orient_conv")
    X = {}
    for m in ("ac", "me3", "ct"):
        X[f"x:{m}_dprod"] = g(f"{m}:diff|prod|s0")
        X[f"x:{m}_dmin"] = g(f"{m}:diff|min|s0")
        X[f"x:{m}_dprod_s2"] = g(f"{m}:diff|prod|s2")
        X[f"x:{m}_ownprod"] = g(f"{m}:own|prod|s0")
    X["x:conv_dct_min"] = conv * g("ct:diff|min|s0")
    X["x:conv_dct_prod"] = conv * g("ct:diff|prod|s0")
    X["x:conv_ctown_prod"] = conv * g("ct:own|prod|s0")
    X["x:pe_prom_partner"] = g("ac:diff|prom|s0") * g("ac:diff|partner|s0")
    X["x:ps_prom_partner"] = g("ac:diff|prom|s0") * g("me3:diff|partner|s0")
    X["x:rep_prom_partner"] = g("me3:diff|prom|s0") * g("me3:diff|partner|s0")
    with np.errstate(all="ignore"):
        bal_a = g("ac:diff|A|s0") - g("me3:diff|A|s0")
        bal_b = g("ac:diff|B|s0") - g("me3:diff|B|s0")
    X["x:balance_min"] = np.fmin(bal_a, bal_b)
    X["x:balance_prod"] = bal_a * bal_b
    X["x:meth_dH_ct_min"] = g("meth:mean:dH|min|s0") * g("ct:diff|min|s0")
    return pd.DataFrame(X, index=F.index)


def feature_sets(F: pd.DataFrame, kind: str) -> dict:
    clfc = [c for c in F.columns if c.startswith("clf:")]
    cov = [c for c in clfc if not c.startswith(("clf:own_", "clf:hspc_", "clf:diff_", "clf:meth_"))]
    Xe = engineered(F, kind)
    sets = {"BASE": clfc, "COV": cov}
    sets["+pairwise"] = clfc + list(Xe.columns)
    sets["+multiscale"] = clfc + sel(F, fam=["ac", "me3", "ct"], reps=["own", "hspc", "diff"], pos=["A", "B", "min", "max", "mid"], scales=["s0", "s2", "s10"])
    sets["+normalised"] = clfc + sel(F, fam=["ac", "me3", "ct"], reps=["diffz", "diffq"], pos=["A", "B", "min", "max", "prod"], scales=["s0", "s2", "s10"])
    sets["+altref"] = clfc + sel(F, fam=["ac", "me3", "ct"], reps=["diffM", "diffH", "diffA", "hspcm"], pos=["A", "B", "min", "max", "prod"], scales=["s0", "s2"])
    sets["+hspcmean"] = clfc + sel(F, fam=["ac", "me3", "ct"], reps=["diffM", "hspcm"], pos=["A", "B", "min", "max", "prod"], scales=["s0", "s2"])
    sets["+healthyref"] = clfc + sel(F, fam=["ac", "me3", "ct"], reps=["diffH"], pos=["A", "B", "min", "max", "prod"], scales=["s0", "s2"])
    sets["+amlmean"] = clfc + sel(F, fam=["ac", "me3", "ct"], reps=["diffA"], pos=["A", "B", "min", "max", "prod"], scales=["s0", "s2"])
    sets["+meth"] = clfc + sel(F, fam=["meth", "cgi", "motif"], pos=["A", "B", "min", "max", "mid"], scales=["s0", "s2", "s10", ""])
    allc = sorted(set(sets["+pairwise"]) | set(sets["+multiscale"]) | set(sets["+normalised"]) | set(sets["+altref"]) | set(sets["+meth"]))
    sets["ALL"] = allc
    sets["LOCUS"] = cov + [c for c in clfc if c.startswith("clf:hspc_")]          # no patient data at all
    added = ["+pairwise", "+multiscale", "+normalised", "+altref", "+meth"]
    for a_ in added:
        own_cols = set(sets[a_]) - set(clfc) - set().union(*[set(sets[b]) for b in added if b != a_])
        sets["ALL-" + a_.lstrip("+")] = [c for c in allc if c not in own_cols]
    return sets, Xe


def donor_idx(d: np.ndarray):
    ud, di = np.unique(d, return_inverse=True)
    return ud, di


def hgb_model(task: str, mono=None, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    kw = dict(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40, l2_regularization=2.0, random_state=seed,
              monotonic_cst=mono)
    return HistGradientBoostingClassifier(**kw) if task == "clf" else HistGradientBoostingRegressor(**kw)


def lin_model(task: str):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    est = LogisticRegression(C=0.02, max_iter=3000) if task == "clf" else Ridge(alpha=300.0)
    return make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                         StandardScaler(), est)


def fit_fold(task, model, X, y, tr, te, mono_signs=None, centre=None):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        Xtr, Xte = X[tr], X[te]
        good = np.isfinite(Xtr).sum(0) >= 20                    # columns without data in the training donors are dropped
        Xtr, Xte = Xtr[:, good], Xte[:, good]
        ytr = y[tr]
        if centre is not None:                          # per-donor centring of features and (regression) target
            dtr, dte = centre[tr], centre[te]
            Xtr, Xte = Xtr.copy(), Xte.copy()
            for d in np.unique(dtr):
                m = dtr == d
                Xtr[m] -= np.nanmedian(Xtr[m], axis=0)
            for d in np.unique(dte):
                m = dte == d
                Xte[m] -= np.nanmedian(Xte[m], axis=0)
            if task == "reg":
                ytr = ytr.copy()
                for d in np.unique(dtr):
                    m = dtr == d
                    ytr[m] -= np.median(ytr[m])
        if model == "hgb":
            mono = None
            m_ = hgb_model(task)
        elif model == "hgbmono":
            from scipy.stats import spearmanr
            sg = np.zeros(Xtr.shape[1], int)
            r = rank_cols(np.where(np.isfinite(Xtr), Xtr, np.nanmedian(Xtr, axis=0)))
            ry = rank_cols(ytr[:, None])[:, 0]
            r = r - r.mean(0)
            ry = ry - ry.mean()
            with np.errstate(all="ignore"):
                c = (r * ry[:, None]).sum(0) / np.sqrt((r * r).sum(0) * (ry * ry).sum())
            sg = np.where(np.abs(c) > 0.05, np.sign(c), 0).astype(int)
            m_ = hgb_model(task, mono=list(sg))
        else:
            m_ = lin_model(task)
        m_.fit(Xtr, ytr)
        p = m_.predict_proba(Xte)[:, 1] if task == "clf" else m_.predict(Xte)
        return te, p


def oof(task, model, X, y, folds, centre=None, jobs=6):
    from joblib import Parallel, delayed
    out = np.full(len(y), np.nan)
    res = Parallel(n_jobs=jobs)(delayed(fit_fold)(task, model, X, y, tr, te, None, centre) for tr, te in folds)
    for te, p in res:
        out[te] = p
    return out


def macro_boot(M: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Mean over the resampled donors of per-donor metrics M (donors x variants), NaN donors dropped."""
    ok = np.isfinite(M)
    num = W @ np.where(ok, M, 0.0)
    den = W @ ok.astype(float)
    with np.errstate(all="ignore"):
        return num / den


def spearman_per_donor(P: np.ndarray, y: np.ndarray, di: np.ndarray, nd: int, min_n=15) -> np.ndarray:
    M = np.full((nd, P.shape[1]), np.nan)
    for k in range(nd):
        s = di == k
        if s.sum() < min_n:
            continue
        a = rank_cols(P[s]) - (s.sum() + 1) / 2
        b = rank_cols(y[s][:, None])[:, 0] - (s.sum() + 1) / 2
        with np.errstate(all="ignore"):
            M[k] = (a * b[:, None]).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum())
    return M


def pooled_rank_corr(P, y, di, nd, W):
    Rx = rank_cols(P)
    Ry = rank_cols(y[:, None])[:, 0]

    def dsum(A):
        out = np.zeros((nd,) + A.shape[1:])
        np.add.at(out, di, A)
        return out
    n = dsum(np.ones(len(y)))
    sx, sxx, sy, syy, sxy = dsum(Rx), dsum(Rx * Rx), dsum(Ry), dsum(Ry * Ry), dsum(Rx * Ry[:, None])
    N = (W @ n)[:, None]
    with np.errstate(all="ignore"):
        cov = W @ sxy - (W @ sx) * (W @ sy)[:, None] / N
        return cov / np.sqrt((W @ sxx - (W @ sx) ** 2 / N) * ((W @ syy) - (W @ sy) ** 2 / N[:, 0])[:, None])


def auroc_pooled_boot(S, ispos, di, nd, W):
    """AUROC per variant (columns of S) of positives against controls, pooled, donor bootstrap through mean ranks."""
    R = rank_cols(S)
    N = len(S)
    npos = np.zeros(nd)
    nneg = np.zeros(nd)
    np.add.at(npos, di[ispos], 1)
    np.add.at(nneg, di[~ispos], 1)
    sp = np.zeros((nd, S.shape[1]))
    sn = np.zeros((nd, S.shape[1]))
    np.add.at(sp, di[ispos], R[ispos])
    np.add.at(sn, di[~ispos], R[~ispos])
    with np.errstate(all="ignore"):
        return 0.5 + ((W @ sp) / (W @ npos)[:, None] - (W @ sn) / (W @ nneg)[:, None]) / N


def auroc_per_donor(S, ispos, di, nd, min_n=15):
    M = np.full((nd, S.shape[1]), np.nan)
    for k in range(nd):
        s = di == k
        pm = ispos[s]
        if pm.sum() < min_n or (~pm).sum() < min_n:
            continue
        R = rank_cols(S[s])
        n1, n0 = pm.sum(), (~pm).sum()
        M[k] = (R[pm].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return M


def ci(x):
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def stage_boost(args) -> None:
    import aml_loop_classifier as clf
    OUT.mkdir(parents=True, exist_ok=True)
    O = load_objects()
    F = pd.read_pickle(WORK / "features.pkl")
    rng = np.random.default_rng(SEED)
    rows, inc_rows = [], []
    for kind, classes in (("loop", LOOP_CLASSES), ("stripe", STRIPE_CLASSES)):
        sel_k = (O.kind == kind).values
        D = O[sel_k & O.set.isin(["pos", "A_hspc"]).values].copy()
        D = pd.concat([D[D.set == "pos"], D[D.set == "A_hspc"]])          # positives first, then the matched controls
        D = D.reset_index().rename(columns={"index": "oidx"})
        Fk = F.loc[D.oidx].reset_index(drop=True)
        sets, Xe = feature_sets(Fk, kind)
        Xfull = pd.concat([Fk, Xe], axis=1)
        # class of each control = class of the matched positive
        pos = D[D.set == "pos"]
        cmap = {(p, q): (c, t) for p, q, c, t in zip(pos.patient, pos.pair_id, pos.cls, pos.ptype)}
        D["cls_m"] = [cmap.get((p, q), (None, None))[0] for p, q in zip(D.patient, D.pair_id)]
        D["pt_m"] = [cmap.get((p, q), (None, None))[1] for p, q in zip(D.patient, D.pair_id)]
        don = D.donor.values
        ispos = (D.set == "pos").values
        y_cls = ispos.astype(int)
        y_gain = D["gain|measured"].values
        keys = clf.row_keys(D.rename(columns={"set": "set_"}).assign(kind=D.kind))
        folds_all = clf.make_folds(don, keys=keys)
        # regression: positives only
        pidx = np.where(ispos & np.isfinite(y_gain))[0]
        sub_keys = [keys[i] for i in pidx]
        folds_reg = [(pidx[tr], pidx[te]) for tr, te in clf.make_folds(don[pidx], keys=sub_keys)]
        ud, di_all = donor_idx(don)
        nd = len(ud)
        W = boot_weights(nd, args.boot, rng)
        variants = [("hgb", s) for s in ["BASE", "COV", "LOCUS", "+pairwise", "+multiscale", "+normalised", "+altref", "+hspcmean", "+healthyref", "+amlmean", "+meth", "ALL",
                                         "ALL-pairwise", "ALL-multiscale", "ALL-normalised", "ALL-altref", "ALL-meth"]]
        variants += [("lin", "BASE"), ("lin", "ALL"), ("hgbmono", "ALL"), ("hgbcentre", "ALL"), ("hgbcentre", "BASE")]
        log(f"[boost] {kind}: {len(D)} rows ({ispos.sum()} positives), {nd} donors, {len(variants)} variants")
        P_clf, P_reg = {}, {}
        for model, sname in variants:
            cols = [c for c in sets[sname] if c in Xfull.columns]
            X = Xfull[cols].values.astype(float)
            mdl = "hgb" if model == "hgbcentre" else model
            cen = don if model == "hgbcentre" else None
            # centring leaves covariates untouched
            if cen is not None:
                keepc = np.array([not (c.startswith("clf:dist") or c.startswith("clf:motif_orient") or c.startswith("clf:gene")) for c in cols])
                Xc = X.copy()
                Xc2 = X.copy()
                for d in np.unique(don):
                    m = don == d
                    Xc2[np.ix_(m, keepc)] = X[np.ix_(m, keepc)] - np.nanmedian(X[np.ix_(m, keepc)], axis=0)
                X = Xc2
                cen = None
            tag = f"{model}:{sname}"
            t0 = time.time()
            P_clf[tag] = oof("clf", mdl, X, y_cls, folds_all, None, args.jobs)
            Pr = oof("reg", mdl, X, y_gain, folds_reg, None, args.jobs)
            if model == "hgbcentre":
                # target centred per donor in training as well
                yc = y_gain.copy()
                for d in np.unique(don):
                    m = (don == d) & ispos
                    yc[m] -= np.nanmedian(yc[m])
                Pr = oof("reg", mdl, X, yc, folds_reg, None, args.jobs)
            P_reg[tag] = Pr
            log(f"[boost] {kind} {tag}: {len(cols)} features, {time.time() - t0:.0f} s")
        tags = list(P_clf)
        Sc = np.column_stack([P_clf[t] for t in tags])
        Sr = np.column_stack([P_reg[t] for t in tags])
        for cname, cfn in classes.items():
            if cname == "shared1":
                continue
            # classification: positives of the class vs their matched controls
            mpos = ispos & cfn(D)
            keyset = set(zip(D.patient[mpos], D.pair_id[mpos]))
            mctl = (~ispos) & np.array([(p, q) in keyset for p, q in zip(D.patient, D.pair_id)])
            mc = mpos | mctl
            if mpos.sum() >= 60 and mctl.sum() >= 60:
                di = di_all[mc]
                A = auroc_pooled_boot(Sc[mc], ispos[mc], di, nd, W)
                Pd = auroc_per_donor(Sc[mc], ispos[mc], di, nd)
                Mb = macro_boot(Pd, W)
                for j, t in enumerate(tags):
                    lo, hi = ci(A[1:, j])
                    mlo, mhi = ci(Mb[1:, j])
                    rows.append({"kind": kind, "class": cname, "task": "classification AUROC", "variant": t, "n": int(mc.sum()),
                                 "pooled": A[0, j], "pooled_lo": lo, "pooled_hi": hi, "macro": Mb[0, j], "macro_lo": mlo, "macro_hi": mhi})
                    for bt in ("hgb:BASE", "hgb:LOCUS"):
                        if t == bt:
                            continue
                        b = tags.index(bt)
                        d_ = A[:, j] - A[:, b]
                        dm = Mb[:, j] - Mb[:, b]
                        inc_rows.append({"kind": kind, "class": cname, "task": "classification AUROC", "variant": t, "baseline": bt,
                                         "inc_pooled": d_[0], "inc_pooled_lo": ci(d_[1:])[0], "inc_pooled_hi": ci(d_[1:])[1],
                                         "inc_macro": dm[0], "inc_macro_lo": ci(dm[1:])[0], "inc_macro_hi": ci(dm[1:])[1]})
            # regression: positives of the class
            mr = mpos & np.isfinite(y_gain)
            if mr.sum() >= 60:
                di = di_all[mr]
                C = pooled_rank_corr(Sr[mr], y_gain[mr], di, nd, W)
                Pd = spearman_per_donor(Sr[mr], y_gain[mr], di, nd)
                Mb = macro_boot(Pd, W)
                for j, t in enumerate(tags):
                    lo, hi = ci(C[1:, j])
                    mlo, mhi = ci(Mb[1:, j])
                    rows.append({"kind": kind, "class": cname, "task": "gain Spearman", "variant": t, "n": int(mr.sum()),
                                 "pooled": C[0, j], "pooled_lo": lo, "pooled_hi": hi, "macro": Mb[0, j], "macro_lo": mlo, "macro_hi": mhi})
                    for bt in ("hgb:BASE", "hgb:LOCUS"):
                        if t == bt:
                            continue
                        b = tags.index(bt)
                        d_ = C[:, j] - C[:, b]
                        dm = Mb[:, j] - Mb[:, b]
                        inc_rows.append({"kind": kind, "class": cname, "task": "gain Spearman", "variant": t, "baseline": bt,
                                         "inc_pooled": d_[0], "inc_pooled_lo": ci(d_[1:])[0], "inc_pooled_hi": ci(d_[1:])[1],
                                         "inc_macro": dm[0], "inc_macro_lo": ci(dm[1:])[0], "inc_macro_hi": ci(dm[1:])[1]})
        D[["oidx", "patient", "donor", "set", "pair_id", "cls", "ptype"]].assign(
            **{f"clf|{t}": P_clf[t] for t in tags}, **{f"reg|{t}": P_reg[t] for t in tags}, gain=y_gain).to_csv(
            OUT / f"boost_oof_{kind}.csv.gz", index=False, float_format="%.5f")
    R = pd.DataFrame(rows)
    I = pd.DataFrame(inc_rows)
    R.to_csv(OUT / "boost_metrics.csv", index=False, float_format="%.4f")
    I.to_csv(OUT / "boost_increments.csv", index=False, float_format="%.4f")
    log(f"[boost] {len(R)} metric rows")


# ============================================================================================ 2D maps
WIN = 192
FOLD_OF = {"018": 10, "027": 10, "270": 10, "168": 11, "546": 11, "629": 11, "1360": 11, "472": 12, "773": 12, "1916": 12}
DETRUN = HM / "runs/aml_loop_detection"


def detrend_diag(M: np.ndarray) -> np.ndarray:
    """Subtract the per-diagonal median (distance decay) from a square map."""
    n = M.shape[0]
    out = M.copy()
    for k in range(n):
        d = np.diagonal(M, k)
        f = np.isfinite(d)
        if f.sum() > 3:
            med = np.median(d[f])
            idx = np.arange(n - k)
            out[idx, idx + k] = d - med
            out[idx + k, idx] = d - med
    return out


def spear(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return np.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def partial_given(a, b, z1, z2) -> float:
    """Spearman partial correlation of a and b given the row and column marginals z1, z2 (pixel vectors)."""
    from scipy.stats import rankdata
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(z1) & np.isfinite(z2)
    if ok.sum() < 60:
        return np.nan
    Z = np.column_stack([np.ones(ok.sum()), rankdata(z1[ok]), rankdata(z2[ok])])
    out = []
    for v in (a, b):
        r = rankdata(v[ok])
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        out.append(r - Z @ beta)
    d = np.sqrt((out[0] ** 2).sum() * (out[1] ** 2).sum())
    return float((out[0] * out[1]).sum() / d) if d > 0 else np.nan


def stage_maps(args) -> None:
    from scipy.stats import rankdata
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cooler
    OUT.mkdir(parents=True, exist_ok=True)
    O = load_objects()
    st = Store()
    P = O[(O.set == "pos") & (O.kind.isin(["loop", "stripe"]))].copy()
    P = P[P.patient.isin(DONORS10)]
    recs = []
    hs = cooler.Cooler(str(PRIM / "consensus/hspccons_10kb.cool"))

    def fetch(clr, win):
        c, se = win.split(":")
        a, b = map(int, se.split("-"))
        try:
            M = np.asarray(clr.matrix(balance=False).fetch(f"{c}:{a}-{b}"), float)
        except Exception:                                  # noqa: BLE001
            return None
        return np.where(M == 0, np.nan, M)

    def pred(z, win):
        Pm = z[win]
        live = np.abs(np.nan_to_num(Pm)).sum(1) > 0
        return np.where(live[:, None] & live[None, :], Pm, np.nan)

    ii, jj = np.triu_indices(WIN, 6)
    band = np.zeros((WIN, WIN), bool)
    band[ii, jj] = True
    band = band | band.T
    for p in DONORS10:
        fold = FOLD_OF[p]
        g = P[P.patient == p]
        T = cooler.Cooler(str(AMLD / f"oe/xu_{p}_proper_10kb.cool"))
        ref = cooler.Cooler(str(PRIM / f"consensus/amltrain_f{fold}_all_10kb.cool"))
        z = np.load(DETRUN / f"f{fold}/aml_refin__xu_{p}/marks.npz")
        tr = {m: chip_reps(st, p, m, ["own", "diff"]) for m in MARKS}
        mt = meth_reps(st, p, "mean", ["dH"])
        log(f"[maps] {p}: {len(g)} objects")
        for win, gw in g.groupby("win"):
            Mm, Mr, Mh = fetch(T, win), fetch(ref, win), fetch(hs, win)
            if Mm is None or Mr is None or Mh is None or win not in z.files or Mm.shape != (WIN, WIN):
                continue
            Mp = pred(z, win)
            ws = int(gw.win_bin.iloc[0])
            c = gw.chrom.iloc[0]

            def sl(d):
                if d is None:
                    return None
                a = d[c][ws:ws + WIN]
                return np.pad(a, (0, WIN - len(a)), constant_values=np.nan) if len(a) < WIN else a
            t = {"dac": sl(tr["H3K27ac"].get("diff")), "dme3": sl(tr["H3K27me3"].get("diff")), "dct": sl(tr["CTCF"].get("diff")),
                 "ownac": sl(tr["H3K27ac"].get("own")), "dmeth": sl(mt.get("dH"))}
            tg = {"vs_ref": detrend_diag(Mm - Mr), "vs_hspc": detrend_diag(Mm - Mh), "model_resid": detrend_diag(Mm - Mp)}
            tnames = list(tg)
            ch, marg = {}, {}
            for nm, v in (("dac_op", t["dac"]), ("dme3_op", t["dme3"]), ("dct_op", t["dct"]), ("ownac_op", t["ownac"]), ("dmeth_op", t["dmeth"])):
                if v is not None:
                    ch[nm] = np.outer(v, v)
                    marg[nm] = v
            if t["dac"] is not None and t["dme3"] is not None:
                X = np.outer(t["dac"], t["dme3"])
                ch["dac_me3_op"] = (X + X.T) / 2
                marg["dac_me3_op"] = t["dac"]
            if t["dac"] is not None:
                ch["dac_min"] = np.fmin.outer(t["dac"], t["dac"])
                ch["dac_add"] = np.add.outer(t["dac"], t["dac"])
                marg["dac_min"] = marg["dac_add"] = t["dac"]
            for nm, key in (("dme3_add", "dme3"), ("dct_add", "dct")):
                if t[key] is not None:
                    ch[nm] = np.add.outer(t[key], t[key])
                    marg[nm] = t[key]
            ch = {k: detrend_diag(v) for k, v in ch.items()}
            cn = list(ch)
            for r in gw.itertuples():
                i0, j0 = r.a_lo - ws, r.b_lo - ws
                loc = np.zeros((WIN, WIN), bool)
                if r.kind == "loop":
                    loc[max(i0 - 15, 0):i0 + 16, max(j0 - 15, 0):j0 + 16] = True
                else:
                    loc[max(i0 - 5, 0):r.a_hi - ws + 5, max(j0 - 5, 0):r.b_hi - ws + 5] = True
                loc &= band
                for scope, mask in (("window", band), ("local", loc)):
                    Tm = np.column_stack([tg[k][mask] for k in tnames])
                    Cm = np.column_stack([ch[k][mask] for k in cn])
                    ok = np.isfinite(Tm).all(1) & np.isfinite(Cm).all(1)
                    if ok.sum() < 60:
                        continue
                    Rt = rankdata(Tm[ok], axis=0)
                    Rc = rankdata(Cm[ok], axis=0)
                    Rt -= Rt.mean(0)
                    Rc -= Rc.mean(0)
                    CC = (Rc.T @ Rt) / np.sqrt(np.outer((Rc ** 2).sum(0), (Rt ** 2).sum(0)))
                    rows_i, cols_i = np.nonzero(mask)
                    for a_i, nm in enumerate(cn):
                        for b_i, tn in enumerate(tnames):
                            rec = {"item": r.item, "patient": p, "donor": r.donor, "kind": r.kind, "ptype": r.ptype, "cls": r.cls, "channel": nm,
                                   "target": tn, "scope": scope, "rho": float(CC[a_i, b_i])}
                            if scope == "local" and nm.endswith("_op"):
                                v = marg[nm]
                                rec["partial_marginals"] = partial_given(Cm[ok][:, a_i], Tm[ok][:, b_i], v[rows_i[ok]], v[cols_i[ok]])
                            recs.append(rec)
    Rc_ = pd.DataFrame(recs)
    Rc_.to_pickle(WORK / "maps_window_corr.pkl")
    summarize_maps(Rc_)


def summarize_maps(Rc: pd.DataFrame) -> None:
    out = []
    rng = np.random.default_rng(SEED)
    for (kind, ch, tg, scope), g in Rc.groupby(["kind", "channel", "target", "scope"]):
        for cls_name, gg in [("all", g), ("private", g[g.cls == "private"]), ("recurrent", g[g.cls == "recurrent"])]:
            for col in ("rho", "partial_marginals"):
                if col not in gg or gg[col].notna().sum() < 50:
                    continue
                v = gg.groupby("donor")[col].mean()
                if len(v) < 4:
                    continue
                bs = [v.values[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
                out.append({"kind": kind, "channel": ch, "target": tg, "scope": scope, "class": cls_name, "statistic": col, "n_windows": int(gg[col].notna().sum()),
                            "n_donors": len(v), "mean_of_donor_means": v.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                            "donors_positive": int((v > 0).sum()), "median_window": gg[col].median(), "frac_windows_positive": float((gg[col].dropna() > 0).mean())})
    pd.DataFrame(out).to_csv(OUT / "maps2d_summary.csv", index=False, float_format="%.4f")


# ============================================================================================ figures
ROWS = [("H3K27ac own", "ac", "own"), ("H3K27ac HSPC", "ac", "hspc"), ("H3K27ac patient-HSPC", "ac", "diff"),
        ("H3K27me3 own", "me3", "own"), ("H3K27me3 HSPC", "me3", "hspc"), ("H3K27me3 patient-HSPC", "me3", "diff"),
        ("CTCF own", "ct", "own"), ("CTCF HSPC", "ct", "hspc"), ("CTCF patient-HSPC", "ct", "diff"),
        ("WGBS mean", "meth_mean", "own"), ("WGBS mean patient-healthy", "meth_mean", "dH"), ("WGBS mean patient-CD14", "meth_mean", "dC"),
        ("WGBS min 1 kb", "meth_min", "own"), ("WGBS min patient-healthy", "meth_min", "dH"),
        ("WGBS low-meth fraction", "meth_lmr", "own"), ("WGBS low-meth patient-healthy", "meth_lmr", "dH"),
        ("CpG island", "cgi", "static")]
MOTIF_ROWS = [("motif methylation (best hit)", ["best_meth"]), ("motif methylation patient-healthy (best hit)", ["best_dmeth"]),
              ("motif methylation (CTCF-bound hit)", ["bound_meth"]), ("motif methylation patient-healthy (bound hit)", ["bound_dmeth"]),
              ("CTCF at bound motif patient-HSPC", ["bound_dctcf"]), ("CTCF motif hits (n)", ["n"])]
CLASS_ORDER = {"loop": ["all", "PE", "PS", "other", "recurrent", "private"], "stripe": ["all", "enh", "rep", "other", "recurrent", "private"]}


def best_grid(T: pd.DataFrame, A: pd.DataFrame, kind: str):
    """Rows x classes: the feature with the largest |rho| of every row group, its rho, partial rho and AUROC - 0.5
    (AUROC best per row group separately)."""
    classes = CLASS_ORDER[kind]
    labs = [r[0] for r in ROWS] + [r[0] for r in MOTIF_ROWS]
    G = {k: pd.DataFrame(np.nan, index=labs, columns=classes) for k in ("rho", "partial", "auc")}
    lab = {k: pd.DataFrame("", index=labs, columns=classes) for k in ("rho", "partial", "auc")}
    for c in classes:
        for (name, tag, rep) in ROWS:
            g = T[(T.kind == kind) & (T["class"] == c) & (T.tag == tag) & (T.rep == rep)]
            if len(g):
                b = g.loc[g.rho.abs().idxmax()]
                G["rho"].loc[name, c], G["partial"].loc[name, c] = b.rho, b.partial
                lab["rho"].loc[name, c] = f"{b.pos}/{b.scale}"
            a = A[(A.kind == kind) & (A["class"] == c) & (A.control == "A_hspc") & (A.tag == tag) & (A.rep == rep)]
            if len(a):
                b = a.loc[(a.auroc - 0.5).abs().idxmax()]
                G["auc"].loc[name, c] = b.auroc - 0.5
                lab["auc"].loc[name, c] = f"{b.pos}/{b.scale}"
        for (name, tags) in MOTIF_ROWS:
            g = T[(T.kind == kind) & (T["class"] == c) & (T.family == "motif") & (T.tag.isin(tags))]
            if len(g):
                b = g.loc[g.rho.abs().idxmax()]
                G["rho"].loc[name, c], G["partial"].loc[name, c] = b.rho, b.partial
                lab["rho"].loc[name, c] = f"{b.pos}"
            a = A[(A.kind == kind) & (A["class"] == c) & (A.control == "A_hspc") & (A.family == "motif") & (A.tag.isin(tags))]
            if len(a):
                b = a.loc[(a.auroc - 0.5).abs().idxmax()]
                G["auc"].loc[name, c] = b.auroc - 0.5
                lab["auc"].loc[name, c] = f"{b.pos}"
    return G, lab


def stage_figures(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    md_tables()
    T = pd.read_csv(OUT / "correlations_all_features.csv.gz")
    A = pd.read_csv(OUT / "auroc_all_features.csv.gz")
    T = T[T.tier != "extra"]
    A = A[A.tier != "extra"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    titles = {"rho": "Spearman with measured gain (best position and scale per cell)",
              "partial": "partial Spearman (same feature; controls log distance and HSPC enrichment)",
              "auc": "AUROC - 0.5 against matched shared controls (best per cell)"}
    for col, kind in enumerate(("loop", "stripe")):
        G, lab = best_grid(T, A, kind)
        for row, k in enumerate(("rho", "partial", "auc")):
            ax = axes[row, col]
            M = G[k].values.astype(float)
            vmax = 0.45 if k != "auc" else 0.3
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(M.shape[1]))
            ax.set_xticklabels(G[k].columns, rotation=0)
            ax.set_yticks(range(M.shape[0]))
            ax.set_yticklabels(G[k].index, fontsize=8)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
            ax.set_title(f"{'loops' if kind == 'loop' else 'stripes'}: {titles[k]}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mark_heatmap.png", dpi=150)
    plt.close(fig)
    # boosting increments
    I = pd.read_csv(OUT / "boost_increments.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for r, kind in enumerate(("loop", "stripe")):
        for c, task in enumerate(("classification AUROC", "gain Spearman")):
            ax = axes[r, c]
            g = I[(I.kind == kind) & (I.task == task) & (I["class"] == "private") & (I.baseline == "hgb:BASE")]
            if g.empty:
                continue
            g = g.iloc[::-1]
            y = np.arange(len(g))
            ax.errorbar(g.inc_macro, y - 0.12, xerr=[g.inc_macro - g.inc_macro_lo, g.inc_macro_hi - g.inc_macro], fmt="o", color="#1f77b4", label="mean per donor", capsize=2)
            ax.errorbar(g.inc_pooled, y + 0.12, xerr=[g.inc_pooled - g.inc_pooled_lo, g.inc_pooled_hi - g.inc_pooled], fmt="s", color="#d95f02", label="pooled", capsize=2)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(g.variant, fontsize=8)
            ax.set_title(f"{kind}s, private: increment of {task} over hgb:BASE", fontsize=9)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_boost_increments.png", dpi=150)
    plt.close(fig)
    # 2D maps
    f = OUT / "maps2d_summary.csv"
    if f.exists():
        Ms = pd.read_csv(f)
        Ms = Ms[(Ms["class"] == "all") & (Ms.scope == "local") & (Ms.statistic == "rho")]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, kind in zip(axes, ("loop", "stripe")):
            g = Ms[Ms.kind == kind]
            piv = g.pivot_table(index="channel", columns="target", values="mean_of_donor_means")
            im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels(piv.columns, fontsize=8)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels(piv.index, fontsize=8)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    ax.text(j, i, f"{piv.values[i, j]:+.3f}", ha="center", va="center", fontsize=7)
            ax.set_title(f"{kind}s: local pixel Spearman, detrended (mean over donors)", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(OUT / "fig_maps2d.png", dpi=150)
        plt.close(fig)


def md_tables() -> None:
    """Readable markdown of the ranked tables (top 10 per class), the expected relationships and the boosting increments."""
    out = []
    T = pd.read_csv(OUT / "top_features_correlation.csv")
    A = pd.read_csv(OUT / "top_features_auroc.csv")
    out.append("# Ranked features per class\n")
    out.append("Spearman rho with the measured gain (pooled over donors; 95 percent interval from donor bootstrap), partial rho (controls log distance and the HSPC enrichment of the object), donors with positive per-donor rho / donors evaluable. Features with data for at least 60 percent of the class objects.\n")
    for (k, c), g in T.groupby(["kind", "class"], sort=False):
        out.append(f"\n## {k}s, class {c}: Spearman\n")
        out.append("| feature | n | rho [95% CI] | partial | donors + / n |\n|---|---|---|---|---|")
        for _, r in g.head(10).iterrows():
            out.append(f"| {r.feature.replace('|', '/')} | {int(r.n)} | {r.rho:+.3f} [{r.rho_lo:+.3f}, {r.rho_hi:+.3f}] | {r.partial:+.3f} | {int(r.donors_positive)}/{int(r.donors_evaluable)} |")
    for (k, c), g in A.groupby(["kind", "class"], sort=False):
        out.append(f"\n## {k}s, class {c}: AUROC against matched shared controls\n")
        out.append("| feature | n pos / n ctl | AUROC [95% CI] | donors above 0.5 / n |\n|---|---|---|---|")
        for _, r in g.head(10).iterrows():
            out.append(f"| {r.feature.replace('|', '/')} | {int(r.n_pos_auc)} / {int(r.n_ctl_auc)} | {r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}] | {int(r.auroc_donors_above)}/{int(r.auroc_donors_evaluable)} |")
    (OUT / "top_features.md").write_text("\n".join(out) + "\n")
    I = pd.read_csv(OUT / "boost_increments.csv")
    o2 = ["# Boosting increments over hgb:BASE (the loop classifier feature set), held-out donors\n",
          "| object | class | task | variant | increment pooled [95% CI] | increment mean per donor [95% CI] |", "|---|---|---|---|---|---|"]
    for _, r in I[I["class"].isin(["private", "all", "recurrent", "PE", "PS", "enh", "rep"])].iterrows():
        o2.append(f"| {r.kind} | {r['class']} | {r.task} | {r.variant} vs {r.baseline} | {r.inc_pooled:+.3f} [{r.inc_pooled_lo:+.3f}, {r.inc_pooled_hi:+.3f}] | {r.inc_macro:+.3f} [{r.inc_macro_lo:+.3f}, {r.inc_macro_hi:+.3f}] |")
    (OUT / "boost_increments.md").write_text("\n".join(o2) + "\n")


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["inventory"],
                    choices=["inventory", "tracks", "features", "corr", "boost", "maps", "figures", "all"])
    ap.add_argument("--boot", type=int, default=NBOOT)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    stages = a.stage if "all" not in a.stage else ["inventory", "tracks", "features", "corr", "boost", "maps", "figures"]
    for s in stages:
        t0 = time.time()
        log(f"=== stage {s}")
        if s == "inventory":
            stage_inventory()
        elif s == "tracks":
            stage_tracks()
        elif s == "features":
            stage_features()
        elif s == "corr":
            stage_corr(a)
        elif s == "boost":
            stage_boost(a)
        elif s == "maps":
            stage_maps(a)
        elif s == "figures":
            stage_figures(a)
        log(f"=== stage {s} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
