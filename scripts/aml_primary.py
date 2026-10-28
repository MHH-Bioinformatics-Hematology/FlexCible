"""Primary AML patients: predicted against measured Hi-C, per patient.

Data
  Xu et al., Nature 2022 (GSE152136): per-patient in situ Hi-C (GSE152135,
  hg38 .iced.mcool, 28 AML patients and 4 CD34+ HSPC donors) and CUT&Tag
  (GSE152132: H3K27ac and H3K27me3 for 12 patients, CTCF for 9; MACS2
  -log10 p-value bigWigs only, raw reads are not public).
  Fischer et al., Cell Rep 2024 (GSE268035): per-patient ChIP (GSE268031:
  CTCF, RAD21, STAG2, H3K27ac, CNV-normalised bigWigs) and Hi-C pooled per
  cohesin genotype (GSE267780: AML cohesin wild type, RAD21-mutant,
  STAG2-mutant; named .hic, but FAN-C HDF5 objects with 10 kb bins).

Stages
  fetch     resumable downloads with size and HDF5/bigWig/.hic checks
  targets   per-patient proper O/E (scripts/aml_eval.py target path: ICE with
            the ENCODE parameters, scripts/oe_proper.py) at 5, 10 and 25 kb,
            2 kb where the depth allows
  inputs    model-scale tracks (see `quantile_track`)
  predict   crosscell_predict per model x patient (aml_eval.predict_one)
  score     413 Akita test regions per patient: models, consensus12, the best
            training map, MOLM-13 / U-937, every other patient's map
  specific  patient-specific structure and residual correlation
  fischer   cohesin-mutant against wild-type group maps
  figures

    python scripts/aml_primary.py --stage fetch
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
AML = HM / "aml"
PRIM = AML / "primary"
RES_DIR = REPO / "results/aml_primary"
PY = sys.executable
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def filelist(series: str) -> pd.DataFrame:
    """GEO filelist.txt of a series (cached)."""
    f = PRIM / f"{series}_filelist.txt"
    if not f.exists() or f.stat().st_size < 200:
        url = f"{GEO}/series/{series[:-3]}nnn/{series}/suppl/filelist.txt"
        subprocess.run(["curl", "-fsS", "--retry", "5", "-o", str(f), url], check=True)
    d = pd.read_csv(f, sep="\t")
    return d[d["#Archive/File"] == "File"]


def sample_url(name: str) -> str:
    gsm = name.split("_")[0]
    return f"{GEO}/samples/{gsm[:-3]}nnn/{gsm}/suppl/{name}"


def remote_size(url: str) -> int | None:
    out = subprocess.run(["curl", "-sIL", "--max-time", "120", url], capture_output=True,
                         text=True).stdout
    s = [int(l.split()[1]) for l in out.splitlines() if l.lower().startswith("content-length:")]
    return s[-1] if s else None


def check(path: Path) -> bool:
    try:
        if path.name.endswith(".mcool"):
            import h5py
            with h5py.File(path, "r") as f:
                return len(f["resolutions"]) > 0
        if path.suffix.lower() == ".bigwig":
            import pyBigWig
            bw = pyBigWig.open(str(path))
            ok = "chr1" in bw.chroms()
            bw.close()
            return ok
        if ".pairs.gz" in path.name:
            return subprocess.run(["gzip", "-t", str(path)]).returncode == 0
        if path.suffix == ".hic":
            import hicstraw
            return len(hicstraw.HiCFile(str(path)).getResolutions()) > 0
        return path.stat().st_size > 0
    except Exception:                                   # noqa: BLE001
        return False


def get(url: str, dest: Path) -> bool:
    """Resumable, retried, stall-aborted download; size and format checked
    before the rename into place."""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    want = remote_size(url)
    part = Path(str(dest) + ".part")
    for _ in range(8):
        subprocess.run(["curl", "-fL", "-sS", "--retry", "10", "--retry-delay", "30",
                        "--retry-all-errors", "--speed-limit", "50000", "--speed-time", "120",
                        "-C", "-", "-o", str(part), url])
        if part.exists() and (want is None or part.stat().st_size == want):
            break
        time.sleep(30)
    if want is not None and (not part.exists() or part.stat().st_size != want):
        log(f"[FAILED] size {dest.name}")
        return False
    tmp = dest.with_name(dest.name + ".chk" + dest.suffix)   # checkers look at the suffix
    part.rename(tmp)
    if not check(tmp):
        log(f"[FAILED] format {dest.name}")
        tmp.rename(part)
        return False
    tmp.rename(dest)
    log(f"[ok] {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
    return True


# ---------------------------------------------------------------- inventory
def xu_hic() -> pd.DataFrame:
    """patient -> mcool name (GSE152135)."""
    d = filelist("GSE152135")
    d = d[d["Name"].str.endswith(".iced.mcool")]
    pat = d["Name"].str.extract(r"^(GSM\d+)_(.+)\.iced\.mcool$")
    return pd.DataFrame({"patient": pat[1], "gsm": pat[0], "name": d["Name"].values,
                         "size": d["Size"].values})


def xu_cuttag() -> pd.DataFrame:
    """patient, mark -> p-value bigWig name (GSE152132)."""
    d = filelist("GSE152132")
    d = d[d["Name"].str.endswith(".pval.signal.bigwig")]
    m = d["Name"].str.extract(r"^(GSM\d+)_(.+?)_CUT_TAG_(\w+?)\.MACS2")
    return pd.DataFrame({"patient": m[1], "gsm": m[0], "mark": m[2], "name": d["Name"].values,
                         "size": d["Size"].values})


FISCHER_HIC = ["merged_aml_ctrl_10kb", "merged_rad21_mut_10kb", "merged_sa2_mut_10kb"]


def fischer_chip() -> pd.DataFrame:
    """AML patients' ChIP bigWigs (GSE268031): group, patient, mark."""
    d = filelist("GSE268031")
    m = d["Name"].str.extract(r"^(GSM\d+)_ChIP_AML_(ctr|Rad21|SA2)_([^_]+)_(.+?)_CNVnorm\.bigWig$")
    out = pd.DataFrame({"gsm": m[0], "group": m[1], "patient": m[2], "mark": m[3],
                        "name": d["Name"].values, "size": d["Size"].values}).dropna()
    return out[out["mark"].isin(["CTCF", "Rad21", "SA2", "H3K27ac"])]


def xu_pairs() -> pd.DataFrame:
    """patient -> nodups.pairs.gz (GSE152135); CUT&Tag patients first."""
    d = filelist("GSE152135")
    d = d[d["Name"].str.endswith(".nodups.pairs.gz")]
    m = d["Name"].str.extract(r"^(GSM\d+)_(.+?)(?:_Hi-C)?\.nodups\.pairs\.gz$")
    out = pd.DataFrame({"patient": m[1], "gsm": m[0], "name": d["Name"].values,
                        "size": d["Size"].values})
    ct = set(xu_cuttag()["patient"])
    out["cuttag"] = out["patient"].isin(ct)
    return out.sort_values("cuttag", ascending=False, kind="stable")


def fetch_pairs(workers: int) -> None:
    """The pairs files: the mcools start at 10 kb in 2x steps, so 5 and 25 kb
    maps are built from the pairs (181 GB for 32 samples)."""
    jobs = [(sample_url(r.name), PRIM / "pairs" / r.name) for r in xu_pairs().itertuples()]
    log(f"[fetch_pairs] {len(jobs)} files")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = list(ex.map(lambda j: get(*j), jobs))
    log(f"[fetch_pairs] done: {sum(ok)}/{len(ok)} ok")


def fetch(workers: int) -> None:
    jobs = []
    for r in xu_hic().itertuples():
        jobs.append((sample_url(r.name), PRIM / "hic" / r.name))
    for r in xu_cuttag().itertuples():
        jobs.append((sample_url(r.name), PRIM / "cuttag" / r.name))
    for r in fischer_chip().itertuples():
        jobs.append((sample_url(r.name), PRIM / "fischer" / "chip" / r.name))
    for n in FISCHER_HIC:
        jobs.append((f"{GEO}/series/GSE267nnn/GSE267780/suppl/GSE267780_{n}.hic",
                     PRIM / "fischer" / f"GSE267780_{n}.hic"))
    log(f"[fetch] {len(jobs)} files")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = list(ex.map(lambda j: get(*j), jobs))
    log(f"[fetch] done: {sum(ok)}/{len(ok)} ok")


# ---------------------------------------------------------------- targets
WORK = PRIM / "work"
OE_DIR = AML / "oe"
TARGET_RES = [5000, 10000, 25000]


def chromsizes() -> Path:
    """chr1-22, X, Y of hg38 (lengths from a training target cooler)."""
    import cooler
    f = PRIM / "hg38.chrom.sizes"
    if not f.exists():
        c = cooler.Cooler(str(HM / "crosscell_oe" / "gm12878_proper_25kb.cool"))
        keep = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        cs = c.chromsizes
        f.write_text("".join(f"{k}\t{int(cs[k])}\n" for k in keep if k in cs.index))
    return f


def patient_label(p: str) -> str:
    return f"xu_{p}"


def oe_of(label: str, res: int) -> Path:
    return OE_DIR / f"{label}_proper_{res // 1000}kb.cool"


def build_patient(patient: str, pairs: Path) -> None:
    """pairs -> 5 kb cooler (cooler cload pairs) -> 10 and 25 kb (coarsen) ->
    ICE per level (ENCODE parameters) -> proper O/E. Temporary names, rename
    when complete; the 5 kb raw cooler is kept for the coverage filter."""
    import aml_eval as ae
    label = patient_label(patient)
    if all(oe_of(label, r).exists() for r in TARGET_RES):
        return
    WORK.mkdir(parents=True, exist_ok=True)
    raw5 = WORK / f"{label}_5kb.raw.cool"
    if not raw5.exists():
        tmp = WORK / f"{label}_5kb.raw.tmp.cool"
        tmp.unlink(missing_ok=True)
        ae.sh(f"{ae.ENV}/cooler cload pairs -c1 2 -p1 3 -c2 4 -p2 5 --chunksize 20000000 "
              f"'{chromsizes()}:5000' '{pairs}' '{tmp}'",
              PRIM / "logs" / f"cload_{label}.log")
        tmp.rename(raw5)
    for res in TARGET_RES:
        k = res // 1000
        o = oe_of(label, res)
        if o.exists():
            continue
        bal = WORK / f"{label}_{k}kb.cool"
        if not bal.exists():
            tmp = WORK / f"{label}_{k}kb.bal.tmp.cool"
            tmp.unlink(missing_ok=True)
            if res == 5000:
                ae.sh(f"cp '{raw5}' '{tmp}'")
            else:
                ae.sh(f"{ae.ENV}/cooler coarsen -k {res // 5000} -p 4 -o '{tmp}' '{raw5}'")
            ae.sh(f"{ae.ENV}/cooler balance --force -p 4 --ignore-diags 2 --max-iters 500 '{tmp}'",
                  PRIM / "logs" / f"balance_{label}_{k}kb.log")
            tmp.rename(bal)
            target_stats(label, res, bal)
        tmp = OE_DIR / f"{label}_proper_{k}kb.tmp.cool"
        tmp.unlink(missing_ok=True)
        ae.sh(f"{PY} scripts/oe_proper.py --input '{bal}' --output '{tmp}' --binsize {res}",
              PRIM / "logs" / f"oe_{label}_{k}kb.log")
        tmp.rename(o)
        log(f"[oe ok] {o.name}")


def target_stats(label: str, res: int, bal: Path) -> None:
    import cooler
    c = cooler.Cooler(str(bal))
    n = int(c.info["nnz"])
    tot = sum(float(c.pixels()[i:i + 50_000_000]["count"].sum())
              for i in range(0, n, 50_000_000))
    w = c.bins()[:]["weight"]
    st = {"label": label, "res": res, "contacts": tot, "nnz": n,
          "ice_masked_frac": float(w.isna().mean())}
    out = RES_DIR / "targets"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}_{res // 1000}kb.json").write_text(json.dumps(st, indent=1))
    log(f"[stats] {label} {res}: {tot / 1e6:.1f} M contacts, masked "
        f"{100 * st['ice_masked_frac']:.1f}%")


def fanc_to_cool(src: Path, dest: Path) -> None:
    """The GSE267780 '.hic' files are FAN-C Hi-C objects (PyTables HDF5,
    blosc: regions/regions with 1-based bins, edges/chrpair_i_j with source,
    sink and integer weight). Written as a cooler of the raw counts (no
    balancing weights), via a temporary file."""
    import cooler
    import tables                                # blosc-compressed PyTables
    with tables.open_file(str(src), "r") as h:
        R = h.get_node("/regions/regions").read()
        bins = pd.DataFrame({"chrom": [c.decode() for c in R["chromosome"]],
                             "start": R["start"] - 1, "end": R["end"]})
        parts = []
        for node in h.get_node("/edges"):
            if node._v_name.startswith("_i_") or not isinstance(node, tables.Table):
                continue
            E = node.read()
            E = E[E["_mask"] == 0]               # FAN-C masked edges dropped
            if len(E):
                i, j = E["source"].astype(np.int64), E["sink"].astype(np.int64)
                parts.append(pd.DataFrame({"bin1_id": np.minimum(i, j),
                                           "bin2_id": np.maximum(i, j),
                                           "count": E["weight"].astype(np.int64)}))
    pix = pd.concat(parts, ignore_index=True)
    pix = pix.groupby(["bin1_id", "bin2_id"], as_index=False)["count"].sum()
    tmp = dest.with_name(dest.stem + ".tmp.cool")
    tmp.unlink(missing_ok=True)
    cooler.create_cooler(str(tmp), bins, pix, assembly="hg38")
    tmp.rename(dest)


def fischer_targets() -> None:
    """The three pooled AML group maps (FAN-C 10 kb, raw weights) through
    aml_eval's target path at 10 kb (ICE with the ENCODE parameters,
    oe_proper); 25 kb is not a multiple of the 10 kb bins."""
    import aml_eval as ae
    for n in FISCHER_HIC:
        part = PRIM / "fischer" / f"GSE267780_{n}.hic.part"
        h5 = PRIM / "fischer" / f"GSE267780_{n}.fanc.h5"
        if part.exists() and not h5.exists():
            part.rename(h5)                    # size verified at download; FAN-C, not Juicer
        lab = "fi_" + n.replace("merged_", "").replace("_10kb", "")
        o = OE_DIR / f"{lab}_proper_10kb.cool"
        if o.exists() or not h5.exists():
            continue
        WORK.mkdir(parents=True, exist_ok=True)
        raw = WORK / f"{lab}_10kb.raw.cool"
        if not raw.exists():
            log(f"[fanc] {h5.name} -> {raw.name}")
            fanc_to_cool(h5, raw)
        bal = WORK / f"{lab}_10kb.cool"
        if not bal.exists():
            tmp = WORK / f"{lab}_10kb.bal.tmp.cool"
            ae.sh(f"cp '{raw}' '{tmp}'")
            ae.sh(f"{ae.ENV}/cooler balance --force -p 4 --ignore-diags 2 --max-iters 500 '{tmp}'",
                  PRIM / "logs" / f"balance_{lab}_10kb.log")
            tmp.rename(bal)
            target_stats(lab, 10000, bal)
        tmp = OE_DIR / f"{lab}_proper_10kb.tmp.cool"
        tmp.unlink(missing_ok=True)
        ae.sh(f"{PY} scripts/oe_proper.py --input '{bal}' --output '{tmp}' --binsize 10000",
              PRIM / "logs" / f"oe_{lab}_10kb.log")
        tmp.rename(o)
        log(f"[oe ok] {o.name}")


def stage_targets(workers: int, only: list[str] | None = None, skip: list[str] | None = None,
                  fischer: bool = True) -> None:
    jobs = []
    for r in xu_pairs().itertuples():
        f = PRIM / "pairs" / r.name
        if (only and r.patient not in only) or (skip and r.patient in skip):
            continue
        if f.exists():
            jobs.append((r.patient, f))
    log(f"[targets] {len(jobs)} patients with pairs")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in [ex.submit(build_patient, p, f) for p, f in jobs]:
            try:
                fut.result()
            except Exception as e:                       # noqa: BLE001
                log(f"[FAILED] target: {e}")
    if fischer:
        fischer_targets()


# ---------------------------------------------------------------- inputs
# The CUT&Tag tracks are MACS2 -log10 p-values and the Fischer ChIP tracks
# CNV-normalised coverage, neither on the fold-change scale of the training
# bigWigs, and CUT&Tag has a far lower background than ChIP. Each track is
# therefore quantile-matched per mark: its 200 bp bin values (mean over all
# bases, uncovered bases 0, chr1-22 and X) are ranked, and each rank interval
# of equal source values gets the mean of the training distribution over that
# interval, where the training distribution is the pooled 200 bp values of
# the 12 training cells' ENCODE fold-change bigWigs of the mark (every 7th bin
# of chr1-22 and X). The mapping is monotone, so the order of the source is
# kept; it fixes background level, peak height and their proportion to the
# training scale.
STEP = 200
SUB = 7
NQ = 20001
MAIN = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def reference_quantiles(mark: str) -> np.ndarray:
    """Quantile function (NQ points) of the pooled training fold-change
    values of one mark at 200 bp; cached in aml/primary/ref_quantiles.npz."""
    import pyBigWig
    from aml_fetch import TRAIN_DIRS, binned_mean
    cache = PRIM / "ref_quantiles.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    if mark in have:
        return have[mark]
    vals = []
    for d in TRAIN_DIRS:
        f = d / f"{mark}.bigwig"
        if not f.exists():
            continue
        bw = pyBigWig.open(str(f))
        for c in MAIN:
            if c in bw.chroms():
                vals.append(binned_mean(bw, c, STEP, all_bases=True)[::SUB])
        bw.close()
    v = np.concatenate(vals)
    q = np.quantile(v, np.linspace(0, 1, NQ))
    have[mark] = q
    np.savez(cache, **have)
    log(f"[quantiles] {mark}: {len(vals) // len(MAIN)} cells, median {q[NQ // 2]:.3f}, "
        f"p99 {q[int(0.99 * (NQ - 1))]:.2f}")
    return q


def quantile_map(src_values: np.ndarray, q: np.ndarray):
    """Monotone map from source values to the training scale: every block of
    equal source values (ties, e.g. the zero background) gets the mean of the
    reference quantile function over its rank interval."""
    u, cnt = np.unique(src_values, return_counts=True)
    c1 = np.cumsum(cnt) / cnt.sum()
    c0 = c1 - cnt / cnt.sum()
    p = np.linspace(0, 1, NQ)
    # integral of the piecewise-linear quantile function from 0 to x
    seg = 0.5 * (q[1:] + q[:-1]) * np.diff(p)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    def integral(x):
        i = np.clip(np.searchsorted(p, x, side="right") - 1, 0, NQ - 2)
        t = x - p[i]
        slope = (q[i + 1] - q[i]) / (p[i + 1] - p[i])
        return cum[i] + q[i] * t + 0.5 * slope * t * t

    w = c1 - c0
    mapped = (integral(c1) - integral(c0)) / np.where(w > 0, w, 1)

    def f(x):
        j = np.clip(np.searchsorted(u, x), 0, len(u) - 1)
        return mapped[j]
    return f


def quantile_track(src: Path, dest: Path, mark: str) -> dict:
    """Write src quantile-matched to the training scale of `mark` (200 bp
    grid, temporary file then rename); returns summary numbers."""
    import pyBigWig
    from aml_fetch import binned_mean, check_bigwig
    q = reference_quantiles(mark)
    bi = pyBigWig.open(str(src))
    chroms = [(c, L) for c, L in bi.chroms().items() if c.startswith("chr") and "_" not in c]
    per = {c: binned_mean(bi, c, STEP, all_bases=True) for c, L in chroms if L // STEP}
    allv = np.concatenate([per[c] for c in MAIN if c in per])
    f = quantile_map(allv, q)
    tmp = Path(str(dest) + ".tmp")
    bo = pyBigWig.open(str(tmp), "w")
    bo.addHeader(chroms, maxZooms=10)
    for c, L in chroms:
        if c in per:
            bo.addEntries(c, 0, values=f(per[c]).astype(np.float64), span=STEP, step=STEP)
    bo.close()
    bi.close()
    if not check_bigwig(tmp):
        raise SystemExit(f"{dest}: written bigWig unreadable")
    tmp.rename(dest)
    return {"src": str(src), "zero_frac": float((allv == 0).mean()),
            "src_median": float(np.median(allv)), "mapped_mean": float(f(allv).mean()),
            "train_mean": float(q.mean())}


INP = PRIM / "inputs"
MARK_NAME = {"Rad21": "RAD21"}
K562_PVAL = {"CTCF": "CTCF.ENCFF336UPT", "H3K27ac": "H3K27ac.ENCFF465GBD",
             "H3K27me3": "H3K27me3.ENCFF582IMB"}


def input_jobs() -> list[tuple[str, str, Path]]:
    """(input set, mark, source bigWig) of every quantile-matched track."""
    jobs = []
    for r in xu_cuttag().itertuples():
        jobs.append((f"xu_{r.patient}", r.mark, PRIM / "cuttag" / r.name))
    for r in fischer_chip().itertuples():
        m = MARK_NAME.get(r.mark, r.mark)
        if m in ("CTCF", "RAD21", "H3K27ac"):                 # STAG2 is no model track
            jobs.append((f"fi_{r.group}_{r.patient}", m, PRIM / "fischer" / "chip" / r.name))
    # K562 controls: the same three marks from four sources
    for m, stem in K562_PVAL.items():
        jobs.append(("k562_pvalq", m, PRIM / "k562_pval" / f"{stem}.bigwig"))
        jobs.append(("k562_encq", m, HM / "hg38_k562" / f"{m}.bigwig"))
    for line in open(AML / "picks_k562ct.tsv").read().splitlines()[1:]:
        m, srx = line.split("\t")[:2]
        jobs.append(("k562_ctq", m, AML / "bw" / f"{srx}.bw"))
    return jobs


def _input_one(job):
    name, mark, src = job
    dest = INP / name / f"{mark}.bigwig"
    if dest.exists():
        return {"input": name, "mark": mark, "done": True}
    if not src.exists():
        return {"input": name, "mark": mark, "missing_source": str(src)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    rec = quantile_track(src, dest, mark)
    log(f"[input] {name} {mark}")
    return {"input": name, "mark": mark, **rec}


def stage_inputs(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    jobs = input_jobs()
    for m in sorted({j[1] for j in jobs}):
        reference_quantiles(m)                                 # fill the cache once
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_input_one, jobs))
    # ENCODE fold change of the same three marks, unchanged (control reference)
    d = INP / "k562_enc3"
    d.mkdir(parents=True, exist_ok=True)
    for m in K562_PVAL:
        if not (d / f"{m}.bigwig").exists():
            import shutil
            shutil.copyfile(HM / "hg38_k562" / f"{m}.bigwig", str(d / f"{m}.bigwig") + ".tmp")
            Path(str(d / f"{m}.bigwig") + ".tmp").rename(d / f"{m}.bigwig")
    # CUT&Tag through the section 19 rescaling (aml_fetch convert -> hg38_k562ct)
    d = INP / "k562_ctm"
    d.mkdir(parents=True, exist_ok=True)
    for f in (AML / "hg38_k562ct").glob("*.bigwig"):
        if not (d / f.name).exists():
            import shutil
            shutil.copyfile(f, str(d / f.name) + ".tmp")
            Path(str(d / f.name) + ".tmp").rename(d / f.name)
    fischer_group_inputs()
    old = pd.read_csv(RES_DIR / "inputs.csv") if (RES_DIR / "inputs.csv").exists() else None
    new = pd.DataFrame([r for r in rows if not r.get("done")])
    df = pd.concat([old, new]) if old is not None else new
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.drop_duplicates(["input", "mark"], keep="last").to_csv(RES_DIR / "inputs.csv", index=False)


def fischer_group_inputs() -> None:
    """Per cohesin genotype, the mean of the patients' quantile-matched tracks
    of each mark (200 bp grid), for the group-map prediction."""
    import pyBigWig
    from aml_fetch import binned_mean, check_bigwig
    fc = fischer_chip()
    for g, gg in fc.groupby("group"):
        out = INP / f"figroup_{g}"
        out.mkdir(parents=True, exist_ok=True)
        for m in ("CTCF", "RAD21", "H3K27ac"):
            dest = out / f"{m}.bigwig"
            pats = gg[gg["mark"].map(lambda x: MARK_NAME.get(x, x)) == m]["patient"].unique()
            srcs = [INP / f"fi_{g}_{p}" / f"{m}.bigwig" for p in pats]
            if dest.exists() or not srcs or not all(s.exists() for s in srcs):
                continue                       # every patient of the group first
            bws = [pyBigWig.open(str(s)) for s in srcs]
            chroms = [(c, L) for c, L in bws[0].chroms().items()]
            tmp = Path(str(dest) + ".tmp")
            bo = pyBigWig.open(str(tmp), "w")
            bo.addHeader(chroms, maxZooms=10)
            for c, L in chroms:
                if L // STEP == 0:
                    continue
                v = np.mean([binned_mean(b, c, STEP, all_bases=True) for b in bws], axis=0)
                bo.addEntries(c, 0, values=v.astype(np.float64), span=STEP, step=STEP)
            bo.close()
            for b in bws:
                b.close()
            if check_bigwig(tmp):
                tmp.rename(dest)
            log(f"[group input] {g} {m} from {len(srcs)} patients")


# ---------------------------------------------------------------- predict
RUNS = HM / "runs"
MR = RUNS / "multicell_reference"
PRED = PRIM / "pred"
CONS = HM / "crosscell_oe"
# name: (res, run dir, family, reference)
PMODELS = {}
for _k in (25, 10):
    PMODELS[f"all12refres_drop_{_k}kb"] = (_k * 1000, MR / f"all12refres_drop_{_k}kb", "refres_drop", True)
    PMODELS[f"all12honest_drop_{_k}kb"] = (_k * 1000, MR / f"all12honest_drop_{_k}kb", "honest_drop", False)
    PMODELS[f"min_ctcf_h3k27ac_refres_{_k}kb"] = (
        _k * 1000, RUNS / "minimal_inputs" / f"min_ctcf_h3k27ac_refres_{_k}kb", "min_ctcf_h3k27ac", True)
for _k in (25, 10, 5):
    PMODELS[f"all12refres_{_k}kb"] = (_k * 1000, MR / f"all12refres_{_k}kb", "refres", True)
PMODELS["all12honest_5kb"] = (5000, MR / "all12honest_5kb", "honest", False)


def pmodel_config(name: str) -> Path:
    run = PMODELS[name][1]
    y = run.parent / f"{run.name}.yaml"
    return y if y.exists() else run / "checkpoint-best" / "config.yaml"


def pmodel_ready(name: str) -> bool:
    return (PMODELS[name][1] / "best").exists() and pmodel_config(name).exists()


def consensus12(res: int) -> Path:
    return CONS / f"consensus12_proper_{res // 1000}kb.cool"


def grid_cool(res: int) -> Path:
    """Any cooler of the resolution: predictions lie on the region grid of
    the bin size, the same for every target."""
    return CONS / f"imr90_proper_{res // 1000}kb.cool"


def pred_file(model: str, inp: str) -> Path:
    return PRED / model / f"{inp}.npz"


def expected_marks(inp: str) -> set[str]:
    """Marks an input set must hold before it is predicted (no partial sets)."""
    if inp.startswith("xu_"):
        c = xu_cuttag()
        return set(c[c.patient == inp[3:]]["mark"])
    if inp.startswith("k562_"):
        return set(K562_PVAL)
    if inp.startswith("figroup_"):
        return {"CTCF", "RAD21", "H3K27ac"}
    return set()


def input_complete(inp: str) -> bool:
    have = {p.stem for p in (INP / inp).glob("*.bigwig")}
    need = expected_marks(inp)
    return bool(have) and need <= have


def predict_jobs() -> list[tuple[str, str]]:
    inputs = sorted(p.name for p in INP.iterdir() if p.is_dir() and input_complete(p.name))
    jobs = []
    for m in PMODELS:
        if not pmodel_ready(m):
            log(f"[skip] {m}: not trained")
            continue
        for inp in inputs:
            if inp.startswith("fi_") and not inp.startswith("figroup_"):
                continue                       # Fischer patients: group tracks only
            if inp.startswith("figroup_") and PMODELS[m][0] == 5000:
                continue                       # pooled maps are 10 kb and coarser
            jobs.append((m, inp))
    return jobs


def predict_one(model: str, inp: str, device: str, threads: int,
                reference: Path | None = None, tag: str = "", cell: str | None = None,
                bigwig_dir: Path | None = None, assay: str = "hic") -> None:
    """crosscell_predict of one model on one input set. `reference` replaces
    consensus12 for reference models (saved as <inp><tag>.npz); `cell` is
    the name the leakage guard checks against the reference's cell list."""
    import os
    res, run, fam, ref = PMODELS[model]
    done = pred_file(model, inp + tag)
    if done.exists():
        return
    out = PRED / model
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "scripts/crosscell_predict.py", "--config", str(pmodel_config(model)),
           "--checkpoint", str(run / (run / "best").read_text().strip()),
           "--cell", cell or f"aml_{inp}", "--bigwig-dir", str(bigwig_dir or INP / inp),
           "--target", str(grid_cool(res)), "--tta", "--assay", assay,
           "--binsize", str(res), "--device", "cuda" if device != "cpu" else "cpu",
           "--out-dir", str(out / f"{inp}{tag}_csv"), "--save-pred", str(done) + ".tmp.npz"]
    if ref:
        cmd += ["--reference", str(reference or consensus12(res))]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               CUDA_VISIBLE_DEVICES="" if device == "cpu" else device)
    with open(out / f"{inp}{tag}.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        log(f"[FAILED] {model} {inp}{tag} rc={rc}")
        return
    Path(str(done) + ".tmp.npz").rename(done)
    log(f"[predict ok] {model} {inp}{tag} ({device})")


def stage_predict(devices: list[str], threads: int, jobs: list | None = None) -> None:
    """One worker per entry of `devices` ('0', '1' = GPU ids, 'cpu')."""
    import queue
    import threading
    q = queue.Queue()
    for j in (jobs if jobs is not None else predict_jobs()):
        q.put(j)
    log(f"[predict] {q.qsize()} jobs on {devices}")

    def worker(dev):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            if isinstance(job, dict):
                predict_one(job.pop("model"), job.pop("inp"), dev, threads, **job)
            else:
                predict_one(job[0], job[1], dev, threads)
    ts = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


# ---------------------------------------------------------------- AML consensus reference
# A disease-matched reference: the pixel-wise mean of AML patients' proper O/E
# maps (HSPC donors excluded). For a scored patient the mean leaves that
# patient out (leave-one-patient-out); MOLM-13 and U-937 read the mean of
# all AML patients. The reference models use it in place of consensus12 at
# inference, without retraining. The cell list is written to <cool>.json as
# "amlxu<patient>" and the prediction runs as cell "amlxu<patient>", so
# flexcible.reference.assert_reference_excludes rejects a reference that
# contains the scored patient.
CONS_DIR = PRIM / "consensus"
REF_MODELS = [m for m, s in PMODELS.items() if s[3]]
AMLREF = "__amlref"


def guard_name(patient_label: str) -> str:
    return "amlxu" + patient_label.replace("xu_", "").replace("-", "").lower()


def aml_patients(res: int) -> list[str]:
    return [p for p in patients(res) if "HSPC" not in p]


def aml_consensus_path(res: int, exclude: str | None) -> Path:
    tag = f"lopo_{exclude}" if exclude else "all"
    return CONS_DIR / f"amlcons_{tag}_{res // 1000}kb.cool"


def build_aml_consensus(res: int, exclude: str | None, labs: list[str] | None = None,
                        out: Path | None = None) -> Path:
    """Mean of the AML patients' proper O/E maps (all share one bin grid and
    pixel table, checked chunk by chunk), excluding `exclude`; or of `labs`
    written to `out`."""
    import cooler
    out = out or aml_consensus_path(res, exclude)
    if out.exists():
        return out
    labs = labs or [p for p in aml_patients(res) if p != exclude]
    clrs = [cooler.Cooler(str(map_path(p, res))) for p in labs]
    n = int(clrs[0].info["nnz"])
    if any(int(c.info["nnz"]) != n for c in clrs):
        raise SystemExit(f"{res}: pixel tables differ between patient maps")
    bins = clrs[0].bins()[:][["chrom", "start", "end"]]
    CONS_DIR.mkdir(parents=True, exist_ok=True)

    def chunks():
        step = 20_000_000
        for i in range(0, n, step):
            ref = clrs[0].pixels()[i:i + step]
            acc = ref["count"].to_numpy(np.float64).copy()
            for c in clrs[1:]:
                px = c.pixels()[i:i + step]
                if not (np.array_equal(px["bin1_id"].values, ref["bin1_id"].values)
                        and np.array_equal(px["bin2_id"].values, ref["bin2_id"].values)):
                    raise SystemExit(f"{c.filename}: pixel ids differ")
                acc += px["count"].to_numpy(np.float64)
            yield pd.DataFrame({"bin1_id": ref["bin1_id"].values, "bin2_id": ref["bin2_id"].values,
                                "count": (acc / len(clrs)).astype(np.float32)})
    tmp = out.with_name(out.stem + ".tmp.cool")
    tmp.unlink(missing_ok=True)
    cooler.create_cooler(str(tmp), bins, chunks(), dtypes={"count": np.float32},
                         ordered=True, assembly="hg38")
    meta = {"cells": [guard_name(p) for p in labs], "patients": labs,
            "excluded": exclude, "res": res}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    tmp.rename(out)
    log(f"[aml consensus] {out.name}: {len(labs)} patients")
    return out


def stage_amlref(res_list: list[int], devices: list[str], threads: int) -> None:
    """Leave-one-patient-out AML consensus per scored patient (and the
    all-patient consensus for MOLM-13 / U-937), then the reference models
    with it as reference."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = []
    for res in res_list:
        tasks.append((res, None))
        for p in aml_patients(res):
            if (INP / p).exists():
                tasks.append((res, p))
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(_build_cons, tasks))
    jobs = []
    for m in REF_MODELS:
        res = PMODELS[m][0]
        if res not in res_list or not pmodel_ready(m):
            continue
        for p in aml_patients(res):
            if (INP / p).exists() and input_complete(p):
                jobs.append({"model": m, "inp": p, "tag": AMLREF, "cell": guard_name(p),
                             "reference": aml_consensus_path(res, p)})
        for line, bw, assay in (("molm13", AML / "hg38_molm13", "hic"),
                                ("u937", AML / "hg38_u937", "microc")):
            jobs.append({"model": m, "inp": f"line_{line}", "tag": AMLREF, "cell": line,
                         "bigwig_dir": bw, "assay": assay,
                         "reference": aml_consensus_path(res, None)})
            jobs.append({"model": m, "inp": f"line_{line}", "cell": line, "bigwig_dir": bw,
                         "assay": assay, "reference": consensus12(res)})
    stage_predict(devices, threads, jobs)


def _build_cons(t):
    return build_aml_consensus(*t)


# ---------------------------------------------------------------- score
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek", "gm23248",
           "mcf7", "cd14mono", "pc9"]
LINES = {"molm13": "molm13", "u937": "u937_wt_c1"}          # section 19 AML lines


def map_path(label: str, res: int) -> Path | None:
    k = res // 1000
    if label == "consensus12":
        p = consensus12(res)
    elif label in TRAIN12 or label == "k562":
        p = CONS / f"{label}_proper_{k}kb.cool"
    elif label in LINES:
        p = OE_DIR / f"{LINES[label]}_proper_{k}kb.cool"
    else:
        p = OE_DIR / f"{label}_proper_{k}kb.cool"
    return p if p.exists() else None


def patients(res: int) -> list[str]:
    """Patient and HSPC map labels built at this resolution."""
    return sorted(f"xu_{p}" for p in xu_pairs()["patient"] if map_path(f"xu_{p}", res))


def region_list() -> pd.DataFrame:
    import score_like_hicgan as sc
    r = pd.read_csv(sc.REGIONS, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    r["region"] = r.chrom + ":" + r.start.astype(str) + "-" + r.end.astype(str)
    return r


def add_dist(dist: dict, method: str, M, T, mask) -> None:
    """Running sums per diagonal (n, x, y, xx, yy, xy), pooled over regions
    and patients, for the per-distance Pearson."""
    import benchmark_metrics as bm
    acc = dist.setdefault(method, {})
    for d, (a, b) in bm.per_distance(M, T, mask).items():
        a, b = a.astype(np.float64), b.astype(np.float64)
        s = acc.setdefault(d, np.zeros(6))
        s += (a.size, a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum())


def dist_rows(dist: dict, k: int, res: int) -> list[dict]:
    out = []
    for m, dd in dist.items():
        for d, (n, sx, sy, sxx, syy, sxy) in sorted(dd.items()):
            vx, vy = sxx - sx * sx / max(n, 1), syy - sy * sy / max(n, 1)
            if n >= 50 and vx > 0 and vy > 0:
                out.append({"res_kb": k, "method": m, "distance_kb": d * res / 1000,
                            "pearson": float((sxy - sx * sy / n) / np.sqrt(vx * vy)),
                            "n_pixels": int(n)})
    return out


K562_SETS = ["k562_enc3", "k562_encq", "k562_pvalq", "k562_ctq", "k562_ctm"]


def score_k562(res: int, models: list[str]) -> list[dict]:
    """Input-processing control on K562 (held out from every model), only
    CTCF, H3K27ac and H3K27me3 given: ENCODE fold change as is, ENCODE fold
    change quantile-matched, ENCODE p-value quantile-matched, K562 CUT&Tag
    quantile-matched and CUT&Tag with the section 19 mean scaling; one mask
    over every prediction and consensus12."""
    import cooler
    import benchmark_metrics as bm
    from aml_eval import fetch_region, pred_matrix
    tp = map_path("k562", res)
    if tp is None:
        return []
    tclr, cclr = cooler.Cooler(str(tp)), cooler.Cooler(str(consensus12(res)))
    preds = {(m, i): dict(np.load(pred_file(m, i))) for m in models for i in K562_SETS
             if pred_file(m, i).exists()}
    rows = []
    for r in region_list().itertuples():
        cache: dict = {}
        T, C = fetch_region(tclr, r.region, cache), fetch_region(cclr, r.region, cache)
        if T is None or C is None or C.shape != T.shape:
            continue
        mats = {f"model:{m}:{i}": pred_matrix(p[r.region]) for (m, i), p in preds.items()
                if r.region in p and p[r.region].shape == T.shape}
        mats["map:consensus12"] = C
        mask = bm.triu_mask(T.shape[0], 2) & np.isfinite(T)
        for M in mats.values():
            mask &= np.isfinite(M)
        if mask.sum() < 100:
            continue
        for kk, M in mats.items():
            rows.append({"target": "k562", "res_kb": res // 1000, "method": kk,
                         "region": r.region, "pearson": bm.pearson(M, T, mask),
                         "scc": bm.scc(M, T, mask), "n_pixels": int(mask.sum())})
    return rows


def score_lines(res: int, models: list[str]) -> list[dict]:
    """MOLM-13 and U-937 (section 19 targets) predicted by the reference
    models with consensus12 and with the consensus of all AML patients as
    reference, against both references alone; one mask per target."""
    import cooler
    import benchmark_metrics as bm
    from aml_eval import fetch_region, pred_matrix
    ac = aml_consensus_path(res, None)
    if not ac.exists():
        return []
    aclr, cclr = cooler.Cooler(str(ac)), cooler.Cooler(str(consensus12(res)))
    rows = []
    for line in LINES:
        tp = map_path(line, res)
        if tp is None:
            continue
        tclr = cooler.Cooler(str(tp))
        preds = {}
        for m in models:
            for tag, nm in (("", m), (AMLREF, m + "@amlref")):
                f = pred_file(m, f"line_{line}{tag}")
                if f.exists():
                    preds[nm] = dict(np.load(f))
        for r in region_list().itertuples():
            cache: dict = {}
            T = fetch_region(tclr, r.region, cache)
            if T is None:
                continue
            mats = {f"model:{m}": pred_matrix(p[r.region]) for m, p in preds.items()
                    if r.region in p and p[r.region].shape == T.shape}
            for nm, c in (("map:consensus12", cclr), ("map:aml_consensus", aclr)):
                M = fetch_region(c, r.region, cache)
                if M is not None and M.shape == T.shape:
                    mats[nm] = M
            mask = bm.triu_mask(T.shape[0], 2) & np.isfinite(T)
            for M in mats.values():
                mask &= np.isfinite(M)
            if mask.sum() < 100:
                continue
            for kk, M in mats.items():
                rows.append({"target": f"line_{line}", "res_kb": res // 1000, "method": kk,
                             "region": r.region, "pearson": bm.pearson(M, T, mask),
                             "scc": bm.scc(M, T, mask), "n_pixels": int(mask.sum())})
    return rows


def score_res(res: int) -> Path | None:
    """Per patient target at one resolution, per region: every model
    prediction from the patient's own tracks, consensus12, the 12 training
    maps, K562, MOLM-13 and U-937 on one shared mask (target, models,
    consensus and these maps); every other patient's map on the shared mask
    restricted to that patient's measured pixels, with the models re-scored
    on the same restricted mask (`*_on_<other>`). Pixels: upper triangle,
    offset >= 2 bins."""
    import pickle
    import cooler
    import benchmark_metrics as bm
    from aml_eval import fetch_region, pred_matrix
    k = res // 1000
    pats = patients(res)
    targets = [p for p in pats if (INP / p).exists()]           # patients with tracks
    base = ["consensus12"] + TRAIN12 + ["k562", "molm13", "u937"]
    clrs = {lab: cooler.Cooler(str(map_path(lab, res))) for lab in base + pats
            if map_path(lab, res) is not None}
    models = [m for m, s in PMODELS.items() if s[0] == res]
    preds = {(m, t): dict(np.load(pred_file(m, t))) for m in models for t in targets
             if pred_file(m, t).exists()}
    for m in models:                         # the same models with the AML reference
        for t in targets:
            if pred_file(m, t + AMLREF).exists():
                preds[(m + "@amlref", t)] = dict(np.load(pred_file(m, t + AMLREF)))
    lopo = {t: cooler.Cooler(str(aml_consensus_path(res, t))) for t in targets
            if aml_consensus_path(res, t).exists()}
    models = models + [m + "@amlref" for m in models]
    rows, other_rows, dist = [], [], {}
    for r in region_list().itertuples():
        cache: dict = {}
        get = {lab: fetch_region(c, r.region, cache) for lab, c in clrs.items()}
        for t in targets:
            T = get.get(t)
            if T is None:
                continue
            mats = {f"model:{m}": pred_matrix(preds[(m, t)][r.region])
                    for m in models if (m, t) in preds and r.region in preds[(m, t)]}
            mats = {kk: v for kk, v in mats.items() if v.shape == T.shape}
            for lab in base:
                M = get.get(lab)
                if M is not None and M.shape == T.shape:
                    mats[f"map:{lab}"] = M
            if t in lopo:                         # leave-one-patient-out AML consensus
                M = fetch_region(lopo[t], r.region, cache)
                if M is not None and M.shape == T.shape:
                    mats["map:aml_consensus"] = M
            mask = bm.triu_mask(T.shape[0], 2) & np.isfinite(T)
            for M in mats.values():
                mask &= np.isfinite(M)
            if mask.sum() < 100:
                continue
            for kk, M in mats.items():
                rows.append({"target": t, "res_kb": k, "method": kk, "region": r.region,
                             "pearson": bm.pearson(M, T, mask), "scc": bm.scc(M, T, mask),
                             "n_pixels": int(mask.sum())})
                add_dist(dist, kk, M, T, mask)
            for q in pats:
                Q = get.get(q)
                if q == t or Q is None or Q.shape != T.shape:
                    continue
                mq = mask & np.isfinite(Q)
                if mq.sum() < 100:
                    continue
                add_dist(dist, "map:other_aml_patient" if "HSPC" not in q else "map:hspc",
                         Q, T, mq)
                rec = {"target": t, "res_kb": k, "other": q, "region": r.region,
                       "pearson": bm.pearson(Q, T, mq), "scc": bm.scc(Q, T, mq),
                       "n_pixels": int(mq.sum())}
                for kk, M in mats.items():
                    if kk.startswith("model:") or kk in ("map:consensus12", "map:aml_consensus"):
                        rec[f"{kk}|pearson"] = bm.pearson(M, T, mq)
                other_rows.append(rec)
    rows += score_k562(res, [m for m in models if "@" not in m])
    rows += score_lines(res, [m for m in models if "@" not in m])
    out = WORK / "score_parts" / f"{k}kb.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".tmp", "wb") as fh:
        pickle.dump({"rows": rows, "other": other_rows, "dist": dist_rows(dist, k, res)}, fh)
    Path(str(out) + ".tmp").rename(out)
    log(f"[score] {k} kb: {len(rows)} rows, {len(other_rows)} patient-pair rows")
    return out


def patient_pairs_res(res: int) -> Path:
    """Patient-to-patient agreement (every pair of patient/HSPC maps), the
    context for 'another patient's map'; pixels measured in both."""
    import cooler
    import benchmark_metrics as bm
    from aml_eval import fetch_region
    k = res // 1000
    pats = patients(res)
    clrs = {p: cooler.Cooler(str(map_path(p, res))) for p in pats}
    rows = []
    for r in region_list().itertuples():
        cache: dict = {}
        get = {p: fetch_region(c, r.region, cache) for p, c in clrs.items()}
        for i, a in enumerate(pats):
            for b in pats[i + 1:]:
                A, B = get[a], get[b]
                if A is None or B is None or A.shape != B.shape:
                    continue
                m = bm.triu_mask(A.shape[0], 2) & np.isfinite(A) & np.isfinite(B)
                if m.sum() >= 100:
                    rows.append({"res_kb": k, "a": a, "b": b, "region": r.region,
                                 "pearson": bm.pearson(A, B, m)})
    out = WORK / "score_parts" / f"pairs_{k}kb.csv.gz"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def stage_score(res_list: list[int]) -> None:
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2 * len(res_list)) as ex:
        list(ex.map(score_res, res_list))
        list(ex.map(patient_pairs_res, res_list))
    rows, other, drows = [], [], []
    for f in sorted((WORK / "score_parts").glob("*kb.pkl")):
        z = pickle.load(open(f, "rb"))
        rows += z["rows"]
        other += z["other"]
        drows += z.get("dist", [])
    RES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(drows).to_csv(RES_DIR / "per_distance.csv", index=False)
    L, O = pd.DataFrame(rows), pd.DataFrame(other)
    L.to_csv(WORK / "scores_long.csv.gz", index=False)
    O.to_csv(WORK / "scores_other_patients.csv.gz", index=False)
    P = pd.concat([pd.read_csv(f) for f in (WORK / "score_parts").glob("pairs_*kb.csv.gz")])
    RES_DIR.mkdir(parents=True, exist_ok=True)
    summ = (L.groupby(["target", "res_kb", "method"])
             .agg(pearson_median=("pearson", "median"), scc_median=("scc", "median"),
                  n_regions=("pearson", "count")).reset_index())
    summ.to_csv(RES_DIR / "summary.csv", index=False)
    # other patients: median per (target, other), and the models on those pixels
    cols = [c for c in O.columns if c.endswith("|pearson")]
    osum = O.groupby(["target", "res_kb", "other"]).agg(
        pearson_median=("pearson", "median"), scc_median=("scc", "median"),
        n_regions=("pearson", "count"), **{c: (c, "median") for c in cols}).reset_index()
    osum.to_csv(RES_DIR / "other_patients.csv", index=False)
    pm = P.groupby(["res_kb", "a", "b"])["pearson"].median().reset_index()
    pm.to_csv(RES_DIR / "patient_pairs.csv", index=False)
    per_patient_table(summ[summ.target.str.startswith("xu_")], osum).to_csv(RES_DIR / "per_patient.csv",
                                                                 index=False)
    summ[summ.target == "k562"].to_csv(RES_DIR / "k562_control.csv", index=False)
    summ[summ.target.str.startswith("line_")].to_csv(RES_DIR / "lines_aml_reference.csv",
                                                     index=False)


def per_patient_table(summ: pd.DataFrame, osum: pd.DataFrame) -> pd.DataFrame:
    """One row per patient and resolution: each model, consensus12, the best
    training map, MOLM-13 / U-937, and the best other AML patient's map with
    the models on that patient's pixels."""
    rows = []
    for (t, k), g in summ.groupby(["target", "res_kb"]):
        g = g.set_index("method")
        rec = {"patient": t, "res_kb": k, "n_regions": int(g["n_regions"].max())}
        for m in g.index:
            if m.startswith("model:") or m in ("map:consensus12", "map:molm13", "map:u937",
                                               "map:k562", "map:aml_consensus"):
                rec[f"{m.split(':', 1)[1]}|pearson"] = g.loc[m, "pearson_median"]
                rec[f"{m.split(':', 1)[1]}|scc"] = g.loc[m, "scc_median"]
        tm = g[g.index.str[4:].isin(TRAIN12)]
        if len(tm):
            b = tm["pearson_median"].idxmax()
            rec.update({"best_train_map": b[4:], "best_train_map|pearson": tm.loc[b, "pearson_median"],
                        "best_train_map|scc": tm.loc[b, "scc_median"]})
        o = osum[(osum.target == t) & (osum.res_kb == k)]
        aml = o[~o.other.str.contains("HSPC")]
        if len(aml):
            b = aml.loc[aml.pearson_median.idxmax()]
            rec.update({"best_other_patient": b.other, "best_other_patient|pearson": b.pearson_median,
                        "best_other_patient|scc": b.scc_median,
                        "median_other_patient|pearson": aml.pearson_median.median()})
            for c in o.columns:
                if c.endswith("|pearson") and c.startswith(("model:", "map:consensus12",
                                                             "map:aml_consensus")):
                    rec[f"{c.split('|')[0].split(':', 1)[1]}|pearson_on_best_patient_pixels"] = b[c]
        hs = o[o.other.str.contains("HSPC")]
        if len(hs):
            rec["best_hspc|pearson"] = hs.pearson_median.max()
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- patient-specific structure
HARD_FRAC = 0.2


def specific_res(res: int) -> pd.DataFrame:
    """Per patient and region: deviation of the measured map T from
    consensus12 C and from the mean of the other AML patients' maps A
    (scripts/hard_split.py score: 0.5 (1 - r) + 0.5 var(T - X) / var(T)),
    the noise quantities of the hard split (raw contacts per pixel of the
    patient's own unbalanced map, fraction of unbalanced bins), and how much
    of the deviation each model and each other patient's map reproduces:
    residual correlation corr(P - C, T - C) and corr(P - A, T - A)."""
    import cooler
    import hard_split as hs
    from aml_eval import fetch_region, pred_matrix
    k = res // 1000
    pats = patients(res)
    aml_pats = [p for p in pats if "HSPC" not in p]
    targets = [p for p in pats if (INP / p).exists()]
    clrs = {p: cooler.Cooler(str(map_path(p, res))) for p in pats}
    cclr = cooler.Cooler(str(consensus12(res)))
    raws = {p: cooler.Cooler(str(WORK / f"{p}_{k}kb.cool")) for p in targets
            if (WORK / f"{p}_{k}kb.cool").exists()}
    models = [m for m, s in PMODELS.items() if s[0] == res]
    preds = {(m, t): dict(np.load(pred_file(m, t))) for m in models for t in targets
             if pred_file(m, t).exists()}
    for m in models:
        for t in targets:
            if pred_file(m, t + AMLREF).exists():
                preds[(m + "@amlref", t)] = dict(np.load(pred_file(m, t + AMLREF)))
    rows = []
    for r in region_list().itertuples():
        cache: dict = {}
        get = {p: fetch_region(c, r.region, cache) for p, c in clrs.items()}
        C = fetch_region(cclr, r.region, cache)
        for t in targets:
            T = get.get(t)
            if T is None or C is None or C.shape != T.shape or t not in raws:
                continue
            others = [get[q] for q in aml_pats if q != t and get.get(q) is not None
                      and get[q].shape == T.shape]
            A = np.nanmean(np.stack(others), 0) if others else None
            raw = hs.fetch(raws[t], r.chrom, r.start, r.end)
            mb = hs.masked_bins(raws[t], r.chrom, r.start, r.end)
            if raw.shape != T.shape or mb.shape[0] != T.shape[0]:
                continue
            st, valid = hs.region_stats(np.nan_to_num(T), np.nan_to_num(C), raw, mb, None)
            rec = {"patient": t, "res_kb": k, "region": r.region, **st}
            if A is not None:
                sa, _ = hs.region_stats(np.nan_to_num(T), np.nan_to_num(A), raw, mb, None)
                rec["score_vs_patients"] = sa.get("score")
                rec["r_patient_mean"] = sa.get("r_consensus")
            v = valid & np.isfinite(A) if A is not None else valid
            dT = (T - C)[v]
            for (m, tt), pr in preds.items():
                if tt != t or r.region not in pr:
                    continue
                P = pred_matrix(pr[r.region])
                if P.shape != T.shape:
                    continue
                rec[f"resid:{m}"] = hs.corr((P - C)[v], dT)
                if A is not None:
                    rec[f"resid_vs_patients:{m}"] = hs.corr((P - A)[v], (T - A)[v])
            if A is not None:
                rec["resid:patient_mean"] = hs.corr((A - C)[v], dT)
            oc = [hs.corr((get[q] - C)[v], dT) for q in aml_pats
                  if q != t and get.get(q) is not None and get[q].shape == T.shape]
            oc = [x for x in oc if np.isfinite(x)]
            if oc:
                rec["resid:best_other_patient"] = max(oc)
                rec["resid:median_other_patient"] = float(np.median(oc))
            rows.append(rec)
    return pd.DataFrame(rows)


def stage_specific(res_list: list[int]) -> None:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(res_list)) as ex:
        parts = list(ex.map(specific_res, res_list))
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(WORK / "specific_per_region.csv.gz", index=False)
    # noise filters of the hard split (no replicate maps per patient)
    keep = D["score"].notna() & (D["masked_frac"] <= 0.2)
    cut = D.groupby(["patient", "res_kb"])["coverage"].transform(lambda x: x.quantile(0.15))
    D["pass"] = keep & (D["coverage"] >= cut)
    rank = D[D["pass"]].groupby(["patient", "res_kb"])["score"].rank(ascending=False, pct=True)
    D["hard"] = False
    D.loc[rank.index, "hard"] = rank <= HARD_FRAC
    D.to_csv(WORK / "specific_per_region.csv.gz", index=False)
    cols = [c for c in D.columns if c.startswith("resid")]
    out = []
    for (p, k), g in D[D["pass"]].groupby(["patient", "res_kb"]):
        for sel, gg in (("all_passing", g), ("hardest_20pct", g[g["hard"]])):
            rec = {"patient": p, "res_kb": k, "set": sel, "n_regions": len(gg),
                   "score_median": gg["score"].median(),
                   "score_vs_patients_median": gg["score_vs_patients"].median(),
                   "r_consensus_median": gg["r_consensus"].median()}
            rec.update({c: gg[c].median() for c in cols})
            out.append(rec)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(RES_DIR / "specific_summary.csv", index=False)
    filt = D.groupby(["patient", "res_kb"]).agg(
        regions=("region", "count"), scored=("score", "count"),
        masked_gt20=("masked_frac", lambda x: int((x > 0.2).sum())),
        passing=("pass", "sum"), hard=("hard", "sum")).reset_index()
    filt.to_csv(RES_DIR / "specific_filters.csv", index=False)
    log("[specific] wrote specific_summary.csv")


# ---------------------------------------------------------------- Fischer groups
FI_GROUPS = {"ctr": "fi_aml_ctrl", "Rad21": "fi_rad21_mut", "SA2": "fi_sa2_mut"}


def stage_fischer(res_list: list[int]) -> None:
    """Pooled group maps: each group prediction (from the group's mean
    tracks) against each group map, and the predicted mutant - wild-type
    difference against the measured one (change map Pearson per region,
    donut-loop strength change at loops called on the measured group maps,
    insulation change at wild-type boundaries; aml_eval's perturbation
    measures)."""
    import cooler
    import benchmark_metrics as bm
    from scipy.stats import pearsonr, spearmanr
    from aml_eval import fetch_region, pred_matrix, loop_params, insulation_log
    from flexcible.bioscore import call_loops, coarsen, local_enrichment
    from flexcible.loops import boundaries
    rows, chg, loops, bnds = [], [], [], []
    for res in res_list:
        k = res // 1000
        maps = {g: map_path(lab, res) for g, lab in FI_GROUPS.items()}
        if any(p is None for p in maps.values()):
            continue
        clr = {g: cooler.Cooler(str(p)) for g, p in maps.items()}
        cclr = cooler.Cooler(str(consensus12(res)))
        models = [m for m, s in PMODELS.items() if s[0] == res]
        preds = {(m, g): dict(np.load(pred_file(m, f"figroup_{g}"))) for m in models
                 for g in FI_GROUPS if pred_file(m, f"figroup_{g}").exists()}
        prm = loop_params(res)
        w = max(3, 100000 // res)
        for r in region_list().itertuples():
            cache: dict = {}
            M = {g: fetch_region(c, r.region, cache) for g, c in clr.items()}
            C = fetch_region(cclr, r.region, cache)
            if any(v is None for v in M.values()) or C is None:
                continue
            n = C.shape[0]
            if any(v.shape != C.shape for v in M.values()):
                continue
            base = bm.triu_mask(n, 2) & np.isfinite(C)
            for v in M.values():
                base &= np.isfinite(v)
            if base.sum() < 100:
                continue
            P = {}
            for (m, g), pr in preds.items():
                if r.region in pr and pr[r.region].shape == C.shape:
                    P[(m, g)] = pred_matrix(pr[r.region])
            # group maps: each prediction, the consensus and the other groups' maps
            for g, T in M.items():
                mk = base.copy()
                for (m, gg), X in P.items():
                    mk &= np.isfinite(X)
                for (m, gg), X in P.items():
                    rows.append({"res_kb": k, "target": g, "method": f"model:{m}:{gg}",
                                 "region": r.region, "pearson": bm.pearson(X, T, mk)})
                rows.append({"res_kb": k, "target": g, "method": "map:consensus12",
                             "region": r.region, "pearson": bm.pearson(C, T, mk)})
                for g2, T2 in M.items():
                    if g2 != g:
                        rows.append({"res_kb": k, "target": g, "method": f"map:{g2}",
                                     "region": r.region, "pearson": bm.pearson(T2, T, mk)})
            f = prm.loop_coarsen
            for mut in ("Rad21", "SA2"):
                dT = M[mut] - M["ctr"]
                ew = local_enrichment(coarsen(np.exp(M["ctr"]), f), prm)
                ek = local_enrichment(coarsen(np.exp(M[mut]), f), prm)
                aw, _ = call_loops(coarsen(np.exp(M["ctr"]), f), prm)
                ak, _ = call_loops(coarsen(np.exp(M[mut]), f), prm)
                anchors = sorted({tuple(x) for x in aw} | {tuple(x) for x in ak})
                iw, ik = insulation_log(M["ctr"], w), insulation_log(M[mut], w)
                bnd = boundaries(iw, 0.1)
                for m in models:
                    if (m, mut) not in P or (m, "ctr") not in P:
                        continue
                    dP = P[(m, mut)] - P[(m, "ctr")]
                    mk = base & np.isfinite(dP)
                    chg.append({"res_kb": k, "mutant": mut, "model": m, "region": r.region,
                                "change_pearson": bm.pearson(dP, dT, mk),
                                "change_scc": bm.scc(dP, dT, mk)})
                    epw = local_enrichment(coarsen(np.exp(P[(m, "ctr")]), f), prm)
                    epk = local_enrichment(coarsen(np.exp(P[(m, mut)]), f), prm)
                    for i, j in anchors:
                        loops.append({"res_kb": k, "mutant": mut, "model": m, "region": r.region,
                                      "dist_kb": (j - i) * res * f / 1000,
                                      "meas": np.log2(ek[i, j] / ew[i, j]),
                                      "pred": np.log2(epk[i, j] / epw[i, j])})
                    ipw, ipk = insulation_log(P[(m, "ctr")], w), insulation_log(P[(m, mut)], w)
                    for b in bnd:
                        bnds.append({"res_kb": k, "mutant": mut, "model": m, "region": r.region,
                                     "meas": ik[b] - iw[b], "pred": ipk[b] - ipw[b]})
                # measured clone-free context: the change of the other mutant group
                other = "SA2" if mut == "Rad21" else "Rad21"
                dO = M[other] - M["ctr"]
                mk = base & np.isfinite(dO)
                chg.append({"res_kb": k, "mutant": mut, "model": f"measured_{other}_change",
                            "region": r.region, "change_pearson": bm.pearson(dO, dT, mk)})
    R, Cg = pd.DataFrame(rows), pd.DataFrame(chg)
    Lp, Bp = pd.DataFrame(loops), pd.DataFrame(bnds)
    for nm, D in (("fischer_groups_long", R), ("fischer_change_long", Cg),
                  ("fischer_loops", Lp), ("fischer_boundaries", Bp)):
        D.to_csv(WORK / f"{nm}.csv.gz", index=False)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    R.groupby(["res_kb", "target", "method"])["pearson"].agg(["median", "count"]).reset_index() \
        .to_csv(RES_DIR / "fischer_groups.csv", index=False)
    out = []
    for (k, mut, m), g in Cg.groupby(["res_kb", "mutant", "model"]):
        rec = {"res_kb": k, "mutant": mut, "model": m, "n_regions": len(g),
               "change_pearson_median": g["change_pearson"].median(),
               "change_scc_median": g.get("change_scc", pd.Series(dtype=float)).median()}
        for nm, D in (("loops", Lp), ("boundaries", Bp)):
            if not len(D):
                continue
            d = D[(D.res_kb == k) & (D.mutant == mut) & (D.model == m)].replace(
                [np.inf, -np.inf], np.nan).dropna(subset=["meas", "pred"])
            if len(d) >= 10:
                rec.update({f"{nm}_n": len(d), f"{nm}_meas_mean": d.meas.mean(),
                            f"{nm}_pred_mean": d.pred.mean(),
                            f"{nm}_pearson": pearsonr(d.meas, d.pred).statistic,
                            f"{nm}_spearman": spearmanr(d.meas, d.pred).statistic,
                            f"{nm}_sign_agreement": float((np.sign(d.meas) == np.sign(d.pred)).mean())})
                if nm == "loops":
                    for lab, sel in (("short", d.dist_kb < 200), ("long", d.dist_kb >= 200)):
                        rec[f"loops_{lab}_n"] = int(sel.sum())
                        rec[f"loops_{lab}_meas_mean"] = d.meas[sel].mean()
                        rec[f"loops_{lab}_pred_mean"] = d.pred[sel].mean()
        out.append(rec)
    pd.DataFrame(out).to_csv(RES_DIR / "fischer_change.csv", index=False)
    log("[fischer] wrote fischer_groups.csv, fischer_change.csv")


# ---------------------------------------------------------------- figures
PANEL_BINS = 80
SHOWCASE_MODEL = {25000: "all12refres_drop_25kb", 10000: "all12refres_drop_10kb",
                  5000: "all12refres_5kb"}


def showcase_windows(res: int, n_patients: int = 3, per_patient: int = 2) -> pd.DataFrame:
    """Hard regions (passing the noise filters) of the patients with the
    largest deviation from the consensus: a window of PANEL_BINS bins centred
    on each region."""
    D = pd.read_csv(WORK / "specific_per_region.csv.gz")
    D = D[(D.res_kb == res // 1000) & D["hard"]]
    if not len(D):
        return pd.DataFrame()
    top = D.groupby("patient")["score"].median().sort_values(ascending=False).index[:n_patients]
    rows = []
    for p in top:
        g = D[D.patient == p].sort_values("score", ascending=False).head(per_patient)
        for r in g.itertuples():
            chrom, se = r.region.split(":")
            s, e = map(int, se.split("-"))
            c = (s + e) // 2 // res * res
            half = PANEL_BINS // 2 * res
            rows.append({"patient": p, "region": r.region, "chrom": chrom,
                         "start": max(0, c - half), "end": c + half, "score": r.score})
    return pd.DataFrame(rows)


def showcase_predict(res: int, W: pd.DataFrame, device: str) -> dict:
    """Predictions on the showcase windows (crosscell_predict --regions)."""
    import os
    m = SHOWCASE_MODEL[res]
    if not pmodel_ready(m):
        return {}
    out = PRED / "showcase"
    out.mkdir(parents=True, exist_ok=True)
    preds = {}
    for p, g in W.groupby("patient"):
        f = out / f"{m}_{p}.npz"
        if not f.exists():
            reg = out / f"{m}_{p}.tsv"
            g[["chrom", "start", "end"]].to_csv(reg, sep="\t", header=False, index=False)
            run = PMODELS[m][1]
            cmd = [PY, "scripts/crosscell_predict.py", "--config", str(pmodel_config(m)),
                   "--checkpoint", str(run / (run / "best").read_text().strip()),
                   "--cell", f"aml_{p}", "--bigwig-dir", str(INP / p),
                   "--target", str(grid_cool(res)), "--tta", "--assay", "hic",
                   "--binsize", str(res), "--device", "cuda" if device != "cpu" else "cpu",
                   "--regions", str(reg), "--out-dir", str(out / f"{m}_{p}_csv"),
                   "--save-pred", str(f) + ".tmp.npz"]
            if PMODELS[m][3]:
                cmd += ["--reference", str(consensus12(res))]
            env = dict(os.environ, OMP_NUM_THREADS="2",
                       CUDA_VISIBLE_DEVICES="" if device == "cpu" else device)
            with open(out / f"{m}_{p}.log", "w") as fh:
                subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
            if Path(str(f) + ".tmp.npz").exists():
                Path(str(f) + ".tmp.npz").rename(f)
        if f.exists():
            preds[p] = dict(np.load(f))
    return preds


def fig_showcase(res: int, device: str) -> None:
    """Rows: hard regions; columns: measured patient map, consensus12, mean of
    the other AML patients, the prediction, measured - consensus, predicted -
    consensus; PANEL_BINS bins per panel."""
    import cooler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aml_eval import fetch_region, pred_matrix
    W = showcase_windows(res)
    if not len(W):
        return
    preds = showcase_predict(res, W, device)
    k = res // 1000
    pats = [p for p in patients(res) if "HSPC" not in p]
    clrs = {p: cooler.Cooler(str(map_path(p, res))) for p in pats}
    cclr = cooler.Cooler(str(consensus12(res)))
    fig, axes = plt.subplots(len(W), 6, figsize=(16, 2.8 * len(W)), squeeze=False)
    for i, r in enumerate(W.itertuples()):
        win = f"{r.chrom}:{r.start}-{r.end}"
        cache: dict = {}
        T = fetch_region(clrs[r.patient], win, cache)
        C = fetch_region(cclr, win, cache)
        others = [fetch_region(clrs[q], win, cache) for q in pats if q != r.patient]
        others = [o for o in others if o is not None and T is not None and o.shape == T.shape]
        A = np.nanmean(np.stack(others), 0) if others else None
        P = preds.get(r.patient, {}).get(win)
        P = pred_matrix(P) if P is not None else None
        mats = [T, C, A, P,
                T - C if T is not None and C is not None else None,
                P - C if P is not None and C is not None else None]
        titles = [f"{r.patient} measured", "consensus12", f"mean of {len(others)} AML patients",
                  f"{SHOWCASE_MODEL[res]}", "measured - consensus", "predicted - consensus"]
        for j, (M, ti) in enumerate(zip(mats, titles)):
            ax = axes[i, j]
            if M is not None:
                lim = 1.5 if j < 4 else 1.0
                ax.imshow(M[:PANEL_BINS, :PANEL_BINS], cmap="RdBu_r", vmin=-lim, vmax=lim,
                          interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(ti, fontsize=8)
        axes[i, 0].set_ylabel(f"{win}\nscore {r.score:.2f}", fontsize=7)
    fig.suptitle(f"Patient-specific structure, {k} kb, {PANEL_BINS} bins per panel "
                 f"({PANEL_BINS * k} kb), log O/E", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES_DIR / f"showcase_{k}kb.png", dpi=110)
    plt.close(fig)


def fig_per_distance() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = RES_DIR / "per_distance.csv"
    if not f.exists():
        return
    D = pd.read_csv(f)
    ress = sorted(D.res_kb.unique())
    colors = {"refres_drop": "#2a78d6", "honest_drop": "#eb6834", "min_ctcf_h3k27ac": "#1baf7a",
              "refres": "#4a3aa7", "honest": "#eda100"}
    fig, axes = plt.subplots(1, len(ress), figsize=(4.2 * len(ress), 3.4), squeeze=False)
    for ax, k in zip(axes[0], ress):
        d = D[D.res_kb == k]
        for m, g in d.groupby("method"):
            if m.startswith("model:"):
                nm = m.split(":")[1]
                fam = PMODELS[nm.split("@")[0]][2]
                ax.plot(g.distance_kb, g.pearson, color=colors[fam], lw=1.6,
                        ls="--" if "@" in nm else "-", label=nm)
        for m, col, ls in (("map:consensus12", "#52514e", "--"), ("map:aml_consensus", "#e87ba4", "--"),
                           ("map:other_aml_patient", "#0b0b0b", ":"),
                           ("map:hspc", "#e34948", ":"), ("map:molm13", "#008300", "-.")):
            g = d[d.method == m]
            if len(g):
                ax.plot(g.distance_kb, g.pearson, color=col, ls=ls, lw=1.4, label=m[4:])
        ax.set_title(f"{k} kb, pooled over patients", fontsize=9)
        ax.set_xlabel("distance (kb)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6, frameon=False)
    axes[0, 0].set_ylabel("Pearson per diagonal")
    fig.tight_layout()
    fig.savefig(RES_DIR / "per_distance.png", dpi=130)
    plt.close(fig)


def fig_per_patient() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = RES_DIR / "per_patient.csv"
    if not f.exists():
        return
    T = pd.read_csv(f)
    ress = sorted(T.res_kb.unique())
    fig, axes = plt.subplots(1, len(ress), figsize=(5 * len(ress), 3.6), squeeze=False)
    cols = [("all12refres_drop_{k}kb|pearson", "#2a78d6", "o", "all12refres_drop"),
            ("all12refres_{k}kb|pearson", "#4a3aa7", "s", "all12refres"),
            ("all12refres_drop_{k}kb@amlref|pearson", "#1baf7a", "P", "all12refres_drop, AML reference"),
            ("consensus12|pearson", "#52514e", "D", "consensus12"),
            ("aml_consensus|pearson", "#e87ba4", "X", "AML consensus (other patients)"),
            ("best_other_patient|pearson", "#0b0b0b", "^", "best other AML patient"),
            ("best_train_map|pearson", "#eda100", "v", "best training map")]
    for ax, k in zip(axes[0], ress):
        t = T[T.res_kb == k].sort_values("patient")
        x = np.arange(len(t))
        for c, col, mk, lab in cols:
            c = c.format(k=k)
            if c in t:
                ax.plot(x, t[c], mk, color=col, ms=6, label=lab, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(t.patient.str.replace("xu_", ""), rotation=90, fontsize=7)
        ax.set_title(f"{k} kb", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6, frameon=False)
    axes[0, 0].set_ylabel("median Pearson, 413 regions")
    fig.tight_layout()
    fig.savefig(RES_DIR / "per_patient.png", dpi=130)
    plt.close(fig)


def stage_figures(res_list: list[int], device: str) -> None:
    fig_per_distance()
    fig_per_patient()
    for res in res_list:
        if (WORK / "specific_per_region.csv.gz").exists():
            fig_showcase(res, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["fetch", "fetch_pairs", "amlref", "targets", "inputs", "predict", "score", "specific",
                             "fischer", "figures"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--devices", nargs="+", default=["0", "0", "1", "1"],
                    help="prediction workers: GPU ids (two per GPU at most) or cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--patients", nargs="*", default=None, help="targets: only these")
    ap.add_argument("--skip", nargs="*", default=None, help="targets: not these")
    ap.add_argument("--no-fischer", action="store_true")
    ap.add_argument("--res", nargs="*", type=int, default=[25000, 10000, 5000])
    a = ap.parse_args()
    if "fetch" in a.stage:
        fetch(a.workers)
    if "fetch_pairs" in a.stage:
        fetch_pairs(a.workers)
    if "targets" in a.stage:
        stage_targets(a.workers, a.patients, a.skip, not a.no_fischer)
    if "inputs" in a.stage:
        stage_inputs(a.workers)
    if "predict" in a.stage:
        stage_predict(a.devices, a.threads)
    if "amlref" in a.stage:
        stage_amlref(a.res, a.devices, a.threads)
    if "score" in a.stage:
        stage_score(a.res)
    if "specific" in a.stage:
        stage_specific(a.res)
    if "fischer" in a.stage:
        stage_fischer([r for r in a.res if r >= 10000])
    if "figures" in a.stage:
        stage_figures(a.res, a.devices[0])


if __name__ == "__main__":
    main()
