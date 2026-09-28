"""Head-to-head benchmark against published Hi-C predictors (registry section 17).

Every method is scored on identical regions, pixels and targets. Competitor
predictions are reused as they exist on disk (never re-run): Akita (released
predictions, Akita space), Hi-cGAN (submitted; whole-genome predictions of the
Hi-cGAN manuscript and its Zenodo records), C.Origami (released IMR-90 model),
Epiphany (released GM12878 weights) and HiC-Reg (published GM12878 predictions).

Blocks (one target, one grid, one region set each):
  A   2048 bp, Akita space, Akita test regions (448-bin crop), GM12878 and HFF
  B   10 kb GM12878, chr3/11/17 (Epiphany held out) and even chromosomes
      (Hi-cGAN held out)
  C   25 kb GM12878, even chromosomes and chrX (Hi-cGAN held out; HiC-Reg split
      undocumented)
  D   10 kb IMR-90, chr2 and chr3 (C.Origami test and validation chromosomes)
  E   10 and 25 kb, cells never seen by our models or by the Hi-cGAN model
      (K562, HMEC, HUVEC), even chromosomes
Blocks B to E score against our proper O/E targets (scripts/oe_proper.py); each
competitor's predicted matrix goes through the same oe_proper.py (expected over
mappable pairs, log O/E, per-diagonal centring, clip, Gaussian sigma 2 bins),
limited to the competitor's own distance reach, and is sampled at the target's
bin centres (hg19 predictions after lifting the bin centres with the UCSC
hg38ToHg19 chain). Block A scores in Akita's own target space.

Stages:
  extract   band-limited single-resolution count cools of the measured maps
  calls     hicDetectLoops and hicFindTADs (HiCExplorer v4 C++) on them
  oe        oe_proper.py on each competitor prediction
  predict   our models on the CPU (crosscell_predict / score_like_hicgan
            --save-pred)
  assemble  every method's map per region on the block grid (npz per block)
  score     metrics per region and method, summaries, paired statistics
  figures   per-distance Pearson and region panels

    python scripts/benchmark_competitors.py --stage extract calls oe predict assemble score figures
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
OUT = HM / "runs/benchmark"
RES_OUT = REPO / "results/benchmark"
PY = sys.executable
TOOLS = Path.home() / "src/HiCExplorer-v4/cpp/build/tools"
THREADS = int(os.environ.get("BENCH_THREADS", "8"))
OE = HM / "crosscell_oe"
PP = Path("/mnt/processed/wolffjoa/restart_of_prediction_project")
ZR = PP / "zenodo_records"
AKN = PP / "akita_normalization"
REGIONS = AKN / "akita_validation/sequences_test_list.tsv"
RASTER = AKN / "akita_testset_rasters.npz"
CHAIN = Path.home() / "data/hic-multiome/chain/hg38ToHg19.over.chain.gz"
MC_REF = HM / "runs/multicell_reference"
MC_EXP = HM / "runs/multicell_expanded"
AKITA_REF = HM / "runs/akita_ref"
AK2 = HM / "akita2kb"

# measured count maps (4DN / ENCODE, ICE weights) behind the proper O/E targets
MEASURED = {
    "gm12878": PP / "originalData/4Dnucleosome/GM12878_4DNFIXP4QG5B.mcool",
    "imr90": Path("/mnt/processed/wolffjoa/hyperparameter_publication/imr90.mcool"),
    "k562": PP / "originalData/4Dnucleosome/K562_4DNFI18UHVRO.mcool",
    "hff": HM / "matrices/hff_microc.mcool",
    "hct116": HM / "matrices/hct116_untreated.mcool",
    "hct116_auxin": HM / "matrices/hct116_auxin6h.mcool",
    "hmec_deep": HM / "encode_hic/hmec_deep_{k}kb.cool",
    "huvec_deep": HM / "encode_hic/huvec_deep_{k}kb.cool",
}
CALL_JOBS = [(c, 10000) for c in MEASURED] + [(c, 25000) for c in
                                              ("gm12878", "k562", "hmec_deep", "huvec_deep")]
EVEN = [f"chr{i}" for i in range(2, 23, 2)] + ["chrX"]

# --------------------------------------------------------------------------- #
# competitor predictions on disk (never re-run)
# --------------------------------------------------------------------------- #
HICGAN_BEST = ZR / "02_best_models/05_predictions/gm12878_sweeps"
HICGAN_X10 = ZR / "05_sweep_10kb/05_predictions/cross_celltype/10kb_CTCF_H3K4me2_w512"
HICGAN_X25 = ZR / "06_sweep_25kb/05_predictions/cross_celltype/25kb_H3K27ac_CTCF_w256"
COMPETITORS = {
    # name: dict(path per chromosome or whole genome, build, res, reach (bp),
    #            value space, held-out chromosomes, cell)
    "hicreg_gm12878_25kb": dict(
        label="HiC-Reg", path=PP / "HiC-Reg/Gm12878_{chrom}_RF_tree20_prediction_predicted_coords_25kb.cool",
        per_chrom=True, build="hg19", res=25000, reach=1_000_000, space="counts",
        heldout="undocumented", cell="gm12878"),
    "hicgan_gm12878_25kb": dict(
        label="Hi-cGAN (submitted)",
        path=HICGAN_BEST / "25kb/best_plus_one/training_odd_gm12878_pred_all_chromosomes_L1_gen_00099_H3K27ac_CTCF.cool",
        per_chrom=False, build="hg19", res=25000, reach=6_400_000, space="counts",
        heldout="even", cell="gm12878"),
    "hicgan_gm12878_10kb": dict(
        label="Hi-cGAN (submitted)",
        path=HICGAN_BEST / "10kb/best_plus_one/training_odd_gm12878_pred_all_chromosomes_L1_gen_00099_CTCF_H3K4me2.cool",
        per_chrom=False, build="hg19", res=10000, reach=5_120_000, space="counts",
        heldout="even", cell="gm12878", chroms=EVEN + ["chr3", "chr11", "chr17"]),
    "epiphany_gm12878_10kb": dict(
        label="Epiphany", path=PP / "epiphany_pretrained/assemble_matrix_{chrom}.cool",
        per_chrom=True, build="hg38", res=10000, reach=990_000, space="signal",
        heldout=["chr3", "chr11", "chr17"], chroms=["chr2", "chr3", "chr11", "chr17"],
        cell="gm12878"),
    "corigami_imr90_8kb": dict(
        label="C.Origami", path=Path.home() / "src/C.Origami/testNovember/IMR90/{chrom}.cool",
        per_chrom=True, build="hg38", res=8192, reach=2_097_152, space="log1p",
        heldout=["chr2", "chr3"], chroms=["chr2", "chr3"], cell="imr90"),
}
# Hi-cGAN cross-cell-type predictions: GM12878 (and IMR-90) models applied to
# another cell's inputs; the IMR-90 model on IMR-90 for block D
for _res, _dir in ((10000, HICGAN_X10), (25000, HICGAN_X25)):
    pairs = [("gm12878", "k562"), ("gm12878", "hmec"), ("gm12878", "huvec")]
    if _res == 10000:
        pairs += [("gm12878", "imr90"), ("imr90", "imr90")]
    for _m, _c in pairs:
        COMPETITORS[f"hicgan_{_m}_on_{_c}_{_res // 1000}kb"] = dict(
            label="Hi-cGAN (submitted)", path=_dir / f"pred_{_m}_on_{_c}_epoch_00099.cool",
            per_chrom=False, build="hg19", res=_res,
            reach=(5_120_000 if _res == 10000 else 6_400_000), space="counts",
            heldout="even", cell=_c, model_cell=_m,
            chroms=(["chr2", "chr3"] if _c == "imr90" else EVEN))

# our models: (label, run dir, cells) per resolution
OURS = {
    25000: {"all12refres": MC_REF / "all12refres_25kb", "all12honest": MC_REF / "all12honest_25kb"},
    10000: {"all12refres": MC_REF / "all12refres_10kb", "all12honest": MC_REF / "all12honest_10kb"},
    2000: {"all12_2kb": MC_EXP / "all12"},
}
OURS_LABEL = {"all12refres": "U-Net 12-cell, reference residual",
              "all12honest": "U-Net 12-cell, no reference",
              "akita_native_honest": "U-Net GM12878 2 kb (akita_native, honest)",
              "akita_native_refres": "U-Net GM12878 2 kb, reference residual (consensus4)",
              "all12_2kb": "U-Net 12-cell 2 kb (section 13)"}


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "benchmark.log", "a") as fh:
        fh.write(line + "\n")


def sh(cmd, logf: Path, env=None) -> int:
    with open(logf, "a") as fh:
        fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=REPO, env=env)


def thread_env(n=THREADS):
    return dict(os.environ, OMP_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n),
                OPENBLAS_NUM_THREADS=str(n), CUDA_VISIBLE_DEVICES="")


# --------------------------------------------------------------------------- #
# stage: extract / calls
# --------------------------------------------------------------------------- #
def measured_uri(cell: str, res: int) -> str:
    p = str(MEASURED[cell])
    if "{k}" in p:
        return p.format(k=res // 1000)
    return f"{p}::/resolutions/{res}"


def call_cool(cell: str, res: int) -> Path:
    return OUT / "calls" / f"{cell}_{res // 1000}kb.cool"


def extract(cell: str, res: int, band_bp: int = 3_000_000):
    """Single-weight, band-limited copy (the C++ tools crash on the 4DN mcools
    that carry several weight columns)."""
    import cooler
    out = call_cool(cell, res)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(measured_uri(cell, res))
    bins = clr.bins()[:][["chrom", "start", "end", "weight"]]
    band = band_bp // res
    keep = [c for c in clr.chromnames if c in [f"chr{i}" for i in range(1, 23)] + ["chrX"]]

    step = 10_000_000

    def gen():
        # row blocks of 10 Mb against columns up to band_bp further: cis, banded,
        # upper triangle, in bin1 order
        mat = clr.matrix(balance=False, as_pixels=True)
        for ch in keep:
            size = int(clr.chromsizes[ch])
            for s in range(0, size, step):
                e = min(s + step, size)
                p = mat.fetch(f"{ch}:{s}-{e}", f"{ch}:{s}-{min(e + band_bp, size)}")
                p = p[(p.bin2_id >= p.bin1_id) & (p.bin2_id - p.bin1_id <= band)]
                p = p.drop_duplicates(["bin1_id", "bin2_id"])
                yield p[["bin1_id", "bin2_id", "count"]].sort_values(["bin1_id", "bin2_id"])
    tmp = out.with_suffix(".tmp.cool")
    cooler.create_cooler(str(tmp), bins=bins, pixels=gen(), ordered=True,
                         dtypes={"count": np.float64 if clr.pixels()[:1]["count"].dtype.kind == "f"
                                 else np.int32})
    tmp.rename(out)
    log(f"extract {cell} {res}: {out}")
    return out


HX376 = Path.home() / "miniconda3/envs/__hicexplorer@3.7.6/bin"


def calls(cell: str, res: int):
    """Reference loops (hicDetectLoops, HiCExplorer 3.7.6 release) and domain
    boundaries (hicFindTADs, HiCExplorer v4 C++) on the measured map.

    The v4 C++ hicDetectLoops aborts in neighborhood_merge (double free) on the
    4DN GM12878 10 kb map, so the loops come from the released Python 3.7.6
    tool. --peakInteractionsThreshold 0: the 4DN ICE weights make pixel values
    fractions, so the default minimum of 10 interactions rejects every peak;
    the p-value and obs/exp (1.5) filters stay at their defaults."""
    d = OUT / "calls"
    d.mkdir(parents=True, exist_ok=True)
    uri = measured_uri(cell, res)
    stem = d / f"{cell}_{res // 1000}kb"
    lf = d / f"{cell}_{res // 1000}kb.calls.log"
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    if res == 10000 and not Path(f"{stem}_loops.bedgraph").exists():
        rc = sh([HX376 / "hicDetectLoops", "-m", uri, "-o", f"{stem}_loops.bedgraph",
                 "--maxLoopDistance", "2000000", "-pit", "0", "-t", str(THREADS), "-tpc", "1",
                 "--chromosomes", *chroms], lf)
        log(f"hicDetectLoops {cell} {res} rc={rc}")
    if not Path(f"{stem}_boundaries.bed").exists():
        rc = sh([TOOLS / "hicFindTADs", "-m", uri, "--outPrefix", stem,
                 "--correctForMultipleTesting", "fdr", "--minDepth", 3 * res,
                 "--maxDepth", 10 * res, "--step", res, "--thresholdComparisons", "0.05",
                 "--delta", "0.01", "-p", str(THREADS), "--chromosomes", *chroms], lf)
        log(f"hicFindTADs {cell} {res} rc={rc}")


def ref_boundaries(cell: str, res: int) -> pd.DataFrame | None:
    f = OUT / "calls" / f"{cell}_{res // 1000}kb_boundaries.bed"
    if not f.exists():
        return None
    b = pd.read_csv(f, sep="\t", header=None, comment="#", usecols=[0, 1, 2])
    b.columns = ["chrom", "start", "end"]
    return b


def ref_loops(cell: str) -> pd.DataFrame | None:
    f = OUT / "calls" / f"{cell}_10kb_loops.bedgraph"
    if not f.exists():
        return None
    lp = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 5])
    lp.columns = ["c1", "s1", "e1", "c2", "s2", "e2"]
    return lp


# --------------------------------------------------------------------------- #
# stage: oe (competitor predictions through oe_proper.py)
# --------------------------------------------------------------------------- #
def comp_oe(name: str, chrom: str | None = None) -> Path:
    tag = f"_{chrom}" if chrom else ""
    return OUT / "competitor_oe" / f"{name}{tag}.cool"


def comp_chroms(spec: dict) -> list[str]:
    if spec.get("chroms"):
        return spec["chroms"]
    return EVEN


def to_counts_cool(spec: dict, chrom: str, src: Path) -> Path:
    """oe_proper.py reads counts; C.Origami stores log(count + 1)."""
    if spec["space"] != "log1p":
        return src
    import cooler
    dst = OUT / "competitor_oe" / f"_counts_{src.parent.name}_{src.stem}.cool"
    if dst.exists():
        return dst
    c = cooler.Cooler(str(src))
    px = c.pixels()[:]
    px["count"] = np.expm1(px["count"].values.astype(np.float64))
    cooler.create_cooler(str(dst), bins=c.bins()[:][["chrom", "start", "end"]], pixels=px,
                         dtypes={"count": np.float64}, ordered=True)
    return dst


def run_oe(name: str, spec: dict):
    (OUT / "competitor_oe").mkdir(parents=True, exist_ok=True)
    lf = OUT / "competitor_oe" / f"{name}.log"
    env = thread_env()
    if spec["per_chrom"]:
        for ch in comp_chroms(spec):
            chn = ch if spec["build"] == "hg38" else ch   # per-chrom files use chr names
            src = Path(str(spec["path"]).format(chrom=chn))
            out = comp_oe(name, ch)
            if out.exists() or not src.exists():
                continue
            src = to_counts_cool(spec, ch, src)
            rc = sh([PY, "scripts/oe_proper.py", "--input", src, "--output", out,
                     "--binsize", spec["res"], "--chroms", ch, "--max-dist", spec["reach"]], lf, env)
            log(f"oe {name} {ch} rc={rc}")
    else:
        out = comp_oe(name)
        if out.exists():
            return
        import cooler
        src = cooler.Cooler(str(spec["path"]))
        chroms = [c.replace("chr", "") for c in comp_chroms(spec)]
        # oe_proper writes the pixels chromosome by chromosome with ordered=True,
        # so the chromosomes must follow the input's bin order ('1', '10', ...)
        chroms = sorted(chroms, key=lambda c: src.offset(c))
        rc = sh([PY, "scripts/oe_proper.py", "--input", spec["path"], "--output", out,
                 "--binsize", spec["res"], "--chroms", *chroms, "--max-dist",
                 min(spec["reach"], 2_000_000)], lf, env)
        log(f"oe {name} rc={rc}")


# --------------------------------------------------------------------------- #
# stage: predict (our models, CPU)
# --------------------------------------------------------------------------- #
def best_ckpt(run: Path) -> Path:
    return run / (run / "best").read_text().strip()


def pred_path(model: str, res: int, cell: str) -> Path:
    return OUT / "pred" / f"{model}_{res // 1000}kb" / f"pred_{cell}.npz"


def predict_ours(model: str, res: int, cells: list[str]):
    import multicell_expanded as mx
    run = OURS[res][model]
    ck = best_ckpt(run)
    od = pred_path(model, res, "x").parent
    od.mkdir(parents=True, exist_ok=True)
    env = thread_env()
    for cell in cells:
        if pred_path(model, res, cell).exists():
            continue
        # the same crosscell_predict call (--tta, leave-one-out or consensus12
        # reference) already ran for registry section 16
        prev = HM / "runs/worst_regions/pred_ref25" / f"pred_{cell}.npz"
        if model == "all12refres" and res == 25000 and prev.exists():
            import shutil
            shutil.copyfile(prev, pred_path(model, res, cell))
            continue
        _, bw, assay, ident = mx.CELLS[cell]
        # measured-map baselines are scored by the blocks (benchmark_blocks.py),
        # so no --others here: every cooler fetch over the share costs ~0.8 s
        extra = []
        if model == "all12refres":
            extra = ["--reference", mx.reference_of(mx.run_name("all12refres", res), cell)]
        rc = sh([PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
                 "--checkpoint", ck, "--cell", cell, "--bigwig-dir", bw, "--target",
                 mx.target(cell, res), "--tta", "--assay", assay, "--out-dir", od,
                 "--binsize", res, "--device", "cpu", *extra,
                 "--save-pred", pred_path(model, res, cell)],
                od / f"predict_{cell}.log", env)
        log(f"predict {model} {res} {cell} rc={rc}")


def predict_akita_space():
    """Our 2 kb GM12878 models on Akita's grid (score_like_hicgan --save-pred)."""
    od = OUT / "pred" / "akita_space"
    od.mkdir(parents=True, exist_ok=True)
    env = thread_env()
    base = (f"consensus4={AK2}/akita_consensus4_2048bp.cool hff={AK2}/akita_hff_all_2048bp.cool "
            f"h1={AK2}/akita_h1_all_2048bp.cool imr90={AK2}/akita_imr90_all_2048bp.cool "
            f"hct116={AK2}/akita_hct116_all_2048bp.cool").split()
    for n in ("akita_native_honest", "akita_native_refres"):
        out = od / f"{n}.npz"
        if out.exists():
            continue
        ck = best_ckpt(AKITA_REF / n)
        rc = sh([PY, "scripts/score_like_hicgan.py", "--config", ck / "config.yaml",
                 "--checkpoint", ck, "--tta", "--minCovered", "0",
                 "--sample-cool", AK2 / "akita_gm12878_train_2048bp.cool",
                 "--out", od / f"{n}.score.csv", "--save-pred", out],
                od / f"{n}.log", env)
        log(f"predict akita-space {n} rc={rc}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--cells", nargs="*", default=None)
    a = ap.parse_args()
    for st in a.stage:
        if st == "extract":
            for c, r in CALL_JOBS:
                if a.cells and c not in a.cells:
                    continue
                extract(c, r)
        elif st == "calls":
            for c, r in CALL_JOBS:
                if a.cells and c not in a.cells:
                    continue
                calls(c, r)
        elif st == "oe":
            for n, spec in COMPETITORS.items():
                run_oe(n, spec)
        elif st == "predict":
            predict_akita_space()
            for res, cells in ((2000, ["hff"]),
                               (25000, ["gm12878", "k562", "hmec_deep", "huvec_deep"]),
                               (10000, ["gm12878", "imr90", "k562", "hmec_deep", "huvec_deep"])):
                for m in OURS[res]:
                    predict_ours(m, res, cells)
        else:
            import benchmark_blocks as bb
            getattr(bb, f"stage_{st}")()


if __name__ == "__main__":
    main()
