"""Sequence CNN for ChIP-seq peak height: how much of a ChIP track is explained
by DNA sequence, by DNA methylation, and by both (registry section 42).

A dilated residual 1D CNN reads a one-hot DNA window (default 131072 bp) and
predicts, for the bins of the central region, the height of 13 ChIP tracks
(CTCF, RAD21, SMC3, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me2, H3K4me3,
H3K79me2, H3K9ac, H3K9me3, H4K20me1; mean bigWig signal per bin) in one
multitask head. Bins are 10 kb (default) or 1 kb (--res 1000). Three modes:

  seq        one-hot sequence only. The input does not contain the cell, so the
             prediction is the same for every cell. Training uses all training
             cells at once (the loss compares the one prediction to the track of
             every training cell that has the mark, so the optimum is the
             across-cell mean). Reported per cell against that cell's own
             measured track, next to the across-cell-average ceiling (the best
             a cell-agnostic prediction can do, rows mode == "ceiling_incl" and
             "ceiling_loo").
  seq_meth   sequence + methylation. Two extra input channels at base-pair
             resolution: the 1 kb Meth track of the cell (methylated fraction,
             each 1 kb value repeated over its 1000 bp, scaled to -1..1) and a
             constant channel with the genome-wide mean methylation of the cell.
             The prediction is cell specific through these inputs, so it also
             works for cells never seen in training (k562, hmec).
  meth       the two methylation channels only (no sequence), as a control.

--cell-embedding adds a learned embedding of the training cell to any mode. It
exists only for cells seen in training and CANNOT generalise to an unseen
cell: unseen cells are skipped in the evaluation of such a run. It serves as a
ceiling on what cell-specific information could add, never as a model for new
cells.

Robustness to cells outside the training range (registry section 59; all defaults
reproduce the earlier results, the variants get a name suffix):
  --meth-norm cell|rank  per-cell z-score or quantile normalisation of the Meth
                         channel from the cell's own Meth track (suffix _cellnorm,
                         _ranknorm); the mean-methylation channel is then 0
  --meth-aug             training only: random per-window scale and shift of the
                         Meth channels (suffix _aug)
  --meth-dropout P       training only: whole windows with zeroed Meth channels
                         (suffix _mdropP)
  --target-norm cell     per-cell z-scoring of the targets and metrics on per-cell
                         standardised predictions and targets (suffix _tcell); tested
                         by the CPU smoke run only, not trained on the full data

Splits (by chromosome only, autosomes chr1 to chr22):
  train        odd chromosomes (chr1, 3, ..., 21) of the training cells
  validation   chr2 and chr4 of the training cells (model selection)
  test         the remaining even chromosomes (chr6, 8, ..., 22) of every cell:
               the seven training cells (seen cells) and the held-out cells
               k562 and hmec, which have Meth and ChIP but are never trained on
  --holdout-cell X removes the training cell X from training and validation
  (leave one training cell out); X is then scored on the test chromosomes as an
  unseen cell.
Validation never overlaps training windows: the three chromosome sets are
disjoint, every sampled input window (output region plus context) is clipped to
its own chromosome (positions outside are padded, never taken from a
neighbouring chromosome), and the script asserts the disjointness at start.
The z-score statistics are computed from training chromosomes of training cells
only. hmec_deep uses the same ChIP and Meth inputs as hmec, so it is not scored
separately; the number of independent cells is 9 (7 + k562 + hmec).

Bins that contain an N in hg38 or overlap the ENCODE blacklist v2 are excluded
from training, validation and test (loss and metrics). A mark missing for a
cell is ignored for that cell.

Target: log1p of the bin mean, z-scored per mark with the training statistics;
loss = MSE (default) averaged over marks; --loss poisson uses the Poisson
negative log-likelihood of the raw bin mean with rate expm1(prediction in
log1p space), scaled per mark. Metrics are computed on the z-scored scale
(R2 = 1 - SSE/SST per mark and cell, Pearson, Spearman) pooled over the test
chromosomes. Augmentation: reverse complement (sequence reverse-complemented,
Meth channel and bin order reversed; the marks keep their order). The test
evaluation averages the forward and the reverse-complement prediction.

Pipeline (no bigWig read during training):
  --stage cache      bin means of all ChIP bigWigs and the 1 kb Meth bigWigs ->
                     compact npz files under --cache-dir; hg38 -> uint8 .npy
                     per chromosome (memory-mapped)
  --stage train      resumable training (ckpt_last.pt, ckpt_best.pt, train_log.csv)
  --stage eval       per mark and cell metrics -> eval.csv, ceiling.csv
  --stage summarize  all runs under --out-root -> seq_cnn_results.csv,
                     seq_cnn_summary.csv, seq_cnn_partition.csv (sequence,
                     methylation, shared and unique R2) in --results-dir
  --benchmark        parameter count and CPU throughput of the real model

Examples
    python scripts/seq_cnn_chip.py --smoke --data-root DATA --mode seq_meth
    python scripts/seq_cnn_chip.py --config configs/seq_cnn.yaml --mode seq

README (running on another computer)
  Required: DATA_ROOT with the layout of /mnt/processed/wolffjoa/hic-multiome:
    hg38_<cell>/<MARK>.bigwig (cells gm12878 imr90 h1 hepg2 a549 gm23248
    cd14mono k562 hmec; marks as above; some marks may be absent for a cell,
    e.g. cd14mono lacks four), meth/<cell>/Meth.bigwig, genome/hg38.fa (the .fai
    index is created next to it, so the folder must be writable or the .fai
    copied along) and annotation/hg38/hg38-blacklist.v2.bed.gz (optional;
    without it only N bins are removed). hg38_gm12878 lives under
    ~/data/hic-multiome on the origin machine, not in the hic-multiome data
    folder; copy or link it into DATA_ROOT.
  Python: torch (CUDA build), numpy, scipy, pandas, pyyaml, pyBigWig, pyfaidx.
    The script needs no other module of the repository.
  Run: bash scripts/run_seq_cnn_remote.sh. One RTX 4090; runtime estimate in
    registry section 42. Cache: about 1 hour, 2 GB disk. RAM below 8 GB.
  Copy back: results/seq_cnn/seq_cnn_results.csv, seq_cnn_summary.csv and
    seq_cnn_partition.csv (optionally OUT_ROOT/*/train_log.csv).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))

MARKS = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
         "H3K4me2", "H3K4me3", "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
TRAIN_CELLS = ["gm12878", "imr90", "h1", "hepg2", "a549", "gm23248", "cd14mono"]
HELD_CELLS = ["k562", "hmec"]
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
TRAIN_CHROMS = [c for i, c in enumerate(AUTOSOMES, 1) if i % 2 == 1]
VAL_CHROMS = ["chr2", "chr4"]
TEST_CHROMS = [c for i, c in enumerate(AUTOSOMES, 1) if i % 2 == 0 and c not in VAL_CHROMS]
ALIAS = {"hmec_deep": "hmec"}          # same ChIP and Meth inputs: one independent cell
METH_BIN = 1000
DOWNSAMPLE = 128                       # trunk stride in bp
COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)    # A C G T N -> T G C A N


def log(msg, fh=None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    E = os.environ.get
    ap.add_argument("--config", help="YAML file; keys are the long option names (underscore or "
                    "hyphen); explicit command-line options win")
    ap.add_argument("--mode", choices=["seq", "seq_meth", "meth"], default="seq")
    ap.add_argument("--stage", nargs="+", default=["cache", "train", "eval"],
                    choices=["cache", "train", "eval", "summarize"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run: chr21 train, chr22 validation and test")
    ap.add_argument("--benchmark", action="store_true", help="params and CPU throughput only")
    ap.add_argument("--data-root", default=E("DATA_ROOT"))
    ap.add_argument("--fasta", default=E("FASTA"), help="default DATA_ROOT/genome/hg38.fa")
    ap.add_argument("--blacklist", default=E("BLACKLIST"),
                    help="default DATA_ROOT/annotation/hg38/hg38-blacklist.v2.bed.gz if present")
    ap.add_argument("--chip-dirs", type=json.loads, default={},
                    help="map cell -> directory with <MARK>.bigwig (default DATA_ROOT/hg38_<cell>)")
    ap.add_argument("--out-root", default=E("OUT_ROOT", "runs/seq_cnn"))
    ap.add_argument("--cache-dir", default=E("CACHE_DIR"), help="default OUT_ROOT/cache")
    ap.add_argument("--out-dir", help="run directory (default OUT_ROOT/<run name>)")
    ap.add_argument("--results-dir", default=E("RESULTS_DIR", str(REPO / "results/seq_cnn")))
    ap.add_argument("--cells", nargs="+", default=TRAIN_CELLS + HELD_CELLS)
    ap.add_argument("--train-cells", nargs="+", default=TRAIN_CELLS)
    ap.add_argument("--train-chroms", nargs="+", default=TRAIN_CHROMS)
    ap.add_argument("--val-chroms", nargs="+", default=VAL_CHROMS)
    ap.add_argument("--test-chroms", nargs="+", default=TEST_CHROMS)
    ap.add_argument("--holdout-cell", default=None, help="leave this training cell out")
    ap.add_argument("--cell-embedding", action="store_true",
                    help="learned cell embedding; seen cells only, cannot generalise")
    ap.add_argument("--res", type=int, default=10000, help="output bin size (10000 or 1000)")
    ap.add_argument("--window", type=int, default=131072, help="input window in bp")
    ap.add_argument("--out-bins", type=int, default=None,
                    help="output bins per window (default 8 at 10 kb, 80 at 1 kb)")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--stem", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss", choices=["mse", "poisson"], default="mse")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-windows", type=int, default=1024)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many validations without improvement (0 = never)")
    ap.add_argument("--meth-norm", choices=["none", "cell", "rank"], default="none",
                    help="Meth channel scaling. none: methylated fraction scaled to -1..1 plus a "
                    "constant channel with the cell's genome-wide mean (original behaviour). "
                    "cell: z-score with the cell's own mean and sd over its valid bins; rank: "
                    "per-cell quantile normalisation to a standard normal. In both the mean "
                    "channel is set to 0 (a cell's mean methylation is its fingerprint). "
                    "The statistics use the cell's Meth track only (no ChIP, no label)")
    ap.add_argument("--meth-aug", action="store_true",
                    help="training only: per window random scale U(1-s, 1+s) and shift N(0, t) "
                    "of the Meth channels (mean channel gets the same affine map)")
    ap.add_argument("--meth-aug-scale", type=float, default=0.5, help="s of --meth-aug")
    ap.add_argument("--meth-aug-shift", type=float, default=0.5, help="t of --meth-aug")
    ap.add_argument("--meth-dropout", type=float, default=0.0,
                    help="training only: probability that a whole window gets zeros in both Meth "
                    "channels (the sequence path stays usable without Meth)")
    ap.add_argument("--target-norm", choices=["global", "cell"], default="global",
                    help="global (default): z-score per mark with the pooled training-cell "
                    "statistics. cell: z-score per mark and cell with that cell's own statistics "
                    "(training chromosomes of a fitted cell, all used chromosomes of an unseen "
                    "cell); predictions are then scale-free per cell, and the test metrics "
                    "standardise predictions and targets per mark and cell first")
    ap.add_argument("--no-rc-aug", action="store_true")
    ap.add_argument("--no-tta", action="store_true", help="no reverse-complement averaging at test")
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--force", action="store_true", help="redo a finished stage")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        import yaml
        cfg = yaml.safe_load(open(pre.config)) or {}
        cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
        known = {x.dest for x in ap._actions}
        bad = set(cfg) - known
        if bad:
            raise SystemExit(f"unknown keys in {pre.config}: {sorted(bad)}")
        ap.set_defaults(**cfg)
    a = ap.parse_args(argv)
    if a.smoke:
        apply_smoke(a, argv)
    if a.data_root is None and "cache" in a.stage and not a.benchmark:
        raise SystemExit("set --data-root or the DATA_ROOT environment variable")
    if a.out_bins is None:
        a.out_bins = 8 if a.res == 10000 else max(1, 80000 // a.res)
    if a.cache_dir is None or (a.smoke and "cache_dir" not in given_opts(argv)):
        a.cache_dir = str(Path(a.out_root) / "cache")
    if a.holdout_cell:
        if a.holdout_cell not in a.train_cells:
            raise SystemExit(f"--holdout-cell {a.holdout_cell} is not a training cell")
        a.fit_cells = [c for c in a.train_cells if c != a.holdout_cell]
    else:
        a.fit_cells = list(a.train_cells)
    mflag = ""
    if a.mode != "seq":
        mflag = {"none": "", "cell": "_cellnorm", "rank": "_ranknorm"}[a.meth_norm] + \
            ("_aug" if a.meth_aug else "") + (f"_mdrop{a.meth_dropout:g}" if a.meth_dropout > 0 else "")
    a.run_name = a.mode + ("_embed" if a.cell_embedding else "") + mflag + \
        ("_tcell" if a.target_norm == "cell" else "") + \
        (f"_ho-{a.holdout_cell}" if a.holdout_cell else "") + \
        (f"_{a.res // 1000}kb" if a.res != 10000 else "") + ("_poisson" if a.loss == "poisson" else "")
    if a.out_dir is None:
        a.out_dir = str(Path(a.out_root) / a.run_name)
    return a


def given_opts(argv):
    return {x.lstrip("-").replace("-", "_").split("=")[0] for x in argv if x.startswith("--")}


def apply_smoke(a, argv):
    """Smoke defaults; options given explicitly on the command line are kept."""
    given = given_opts(argv)
    sm = dict(cells=["gm12878", "imr90", "k562"], train_cells=["gm12878", "imr90"],
              train_chroms=["chr21"], val_chroms=["chr22"], test_chroms=["chr22"],
              window=32768, out_bins=2, width=48, blocks=2, stem=16, steps=30, batch_size=4,
              val_every=15, val_windows=24, ckpt_every=10, warmup=5, num_workers=2,
              eval_batch=8, device="cpu", build_workers=3, no_tta=True)
    for k, v in sm.items():
        if k not in given:
            setattr(a, k, v)
    a.out_root = str(Path(a.out_root) / "smoke")     # own folder, never mixed with real runs
    if "results_dir" not in given:
        a.results_dir = str(Path(a.out_root) / "results")


# ----------------------------------------------------------------------------
# cache
# ----------------------------------------------------------------------------

def chip_dir(a, cell):
    if cell in a.chip_dirs:
        return Path(a.chip_dirs[cell])
    return Path(a.data_root) / f"hg38_{cell}"


def meth_path(a, cell):
    return Path(a.data_root) / "meth" / ALIAS.get(cell, cell) / "Meth.bigwig"


def fasta_path(a):
    return Path(a.fasta) if a.fasta else Path(a.data_root) / "genome/hg38.fa"


def blacklist_path(a):
    if a.blacklist:
        return Path(a.blacklist)
    p = Path(a.data_root) / "annotation/hg38/hg38-blacklist.v2.bed.gz"
    return p if p.exists() else None


def needed_chroms(a):
    return sorted(set(a.train_chroms) | set(a.val_chroms) | set(a.test_chroms),
                  key=lambda c: AUTOSOMES.index(c))


def validate_data(a):
    """Fail early and list everything that is missing."""
    miss = []
    if not fasta_path(a).exists():
        miss.append(str(fasta_path(a)))
    for c in a.cells:
        if not meth_path(a, c).exists():
            miss.append(str(meth_path(a, c)))
        d = chip_dir(a, c)
        have = [m for m in MARKS if (d / f"{m}.bigwig").exists()]
        if len(have) < 5:
            miss.append(f"{d} (only {len(have)} of 13 marks)")
    if miss:
        raise SystemExit("missing data:\n  " + "\n  ".join(miss))


def encode_seq(s: str) -> np.ndarray:
    lut = np.full(256, 4, dtype=np.uint8)
    for i, ch in enumerate("ACGT"):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i
    return lut[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]


def _targets_task(args):
    """All marks of one cell: log1p float16 [nbins, 13] per chromosome (NaN = mark absent)."""
    import pyBigWig
    cache, cell, chip_d, chroms, sizes, res = args
    out = Path(cache) / f"targets_{res}" / f"{cell}.npz"
    if out.exists():
        return cell
    arrays = {c: np.full((sizes[c] // res, len(MARKS)), np.nan, np.float16) for c in chroms}
    for k, m in enumerate(MARKS):
        f = Path(chip_d) / f"{m}.bigwig"
        if not f.exists():
            continue
        bw = pyBigWig.open(str(f))
        for c in chroms:
            nb = sizes[c] // res
            v = bw.stats(c, 0, nb * res, type="mean", nBins=nb, exact=True)
            v = np.array([np.nan if x is None else x for x in v], dtype=np.float64)
            v = np.nan_to_num(v, nan=0.0)                  # no coverage = 0
            arrays[c][:, k] = np.log1p(np.clip(v, 0, None)).astype(np.float16)
        bw.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    return cell


def _meth_task(args):
    """Meth.bigWig (1 kb, value - 1 = methylated fraction; 0 = no data) -> float32 fraction."""
    import pyBigWig
    cache, cell, path, chroms, sizes = args
    out = Path(cache) / "meth" / f"{cell}.npz"
    if out.exists():
        return cell
    bw = pyBigWig.open(str(path))
    arr = {}
    tot, cnt = 0.0, 0
    for c in chroms:
        n1 = sizes[c] // METH_BIN
        v = bw.stats(c, 0, n1 * METH_BIN, type="mean", nBins=n1, exact=True)
        v = np.array([np.nan if x is None else x for x in v], dtype=np.float64)
        frac = np.where(np.nan_to_num(v, nan=0.0) > 0, v - 1.0, np.nan)
        ok = ~np.isnan(frac)
        tot += float(frac[ok].sum())
        cnt += int(ok.sum())
        arr[c] = frac
    bw.close()
    mean = tot / max(cnt, 1)
    arr = {c: np.where(np.isnan(v), mean, v).astype(np.float32) for c, v in arr.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, _mean=np.float32(mean), **arr)
    os.replace(tmp, out)
    return cell


def cache_complete(a):
    c = Path(a.cache_dir)
    return ((c / f"valid_{a.res}.npz").exists() and (c / "sizes.json").exists() and
            all((c / f"targets_{a.res}" / f"{x}.npz").exists() and
                (c / "meth" / f"{x}.npz").exists() for x in a.cells) and
            all((c / "seq" / f"{x}.npy").exists() for x in needed_chroms(a)))


def stage_cache(a):
    import pyfaidx
    validate_data(a)
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    chroms = needed_chroms(a)
    fa = pyfaidx.Fasta(str(fasta_path(a)))
    sizes = {c: len(fa[c]) for c in chroms}
    (cache / "seq").mkdir(exist_ok=True)
    for c in chroms:
        f = cache / "seq" / f"{c}.npy"
        if not f.exists():
            t = time.time()
            with open(str(f) + ".tmp", "wb") as fh:
                np.save(fh, encode_seq(str(fa[c][:])))
            os.replace(str(f) + ".tmp", f)
            log(f"seq {c} {sizes[c]} bp {time.time() - t:.0f}s")
    vf = cache / f"valid_{a.res}.npz"
    if not vf.exists():
        bl = {}
        bp = blacklist_path(a)
        if bp is not None and Path(bp).exists():
            op = gzip.open if str(bp).endswith(".gz") else open
            with op(bp, "rt") as fh:
                for line in fh:
                    p = line.split("\t")
                    if len(p) >= 3:
                        bl.setdefault(p[0], []).append((int(p[1]), int(p[2])))
        else:
            log("WARNING: no blacklist file, only N bins are excluded")
        valid = {}
        for c in chroms:
            nb = sizes[c] // a.res
            seq = np.load(cache / "seq" / f"{c}.npy", mmap_mode="r")
            ok = ~(np.asarray(seq[:nb * a.res]) == 4).reshape(nb, a.res).any(1)
            for s, e in bl.get(c, []):
                ok[max(s // a.res, 0):min((e - 1) // a.res + 1, nb)] = False
            valid[c] = ok
        np.savez(str(vf) + ".tmp.npz", **valid)
        os.replace(str(vf) + ".tmp.npz", vf)
        log("valid bins: " + ", ".join(f"{c} {valid[c].mean():.3f}" for c in chroms))
    (cache / "sizes.json").write_text(json.dumps(sizes))
    jobs = [(str(cache), c, str(meth_path(a, c)), chroms, sizes) for c in a.cells]
    with ProcessPoolExecutor(max(1, min(a.build_workers, len(jobs)))) as ex:
        for c in ex.map(_meth_task, jobs):
            log(f"meth {c}")
    jobs = [(str(cache), c, str(chip_dir(a, c)), chroms, sizes, a.res) for c in a.cells]
    with ProcessPoolExecutor(max(1, min(a.build_workers, len(jobs)))) as ex:
        for c in ex.map(_targets_task, jobs):
            log(f"targets {c} ({a.res} bp)")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

class Data:
    """Everything the loaders need, read from the cache; z-scored targets in float16."""

    def __init__(self, a):
        cache = Path(a.cache_dir)
        self.a = a
        self.res, self.L, self.NB = a.res, a.window, a.out_bins
        self.span = self.NB * self.res
        if (self.L - self.span) % 2 or self.L < self.span:
            raise SystemExit("window must be >= out_bins * res and window - span even")
        self.margin = (self.L - self.span) // 2
        self.sizes = json.loads((cache / "sizes.json").read_text())
        self.chroms = needed_chroms(a)
        self.seq = {c: np.load(cache / "seq" / f"{c}.npy", mmap_mode="r") for c in self.chroms}
        vz = np.load(cache / f"valid_{a.res}.npz")
        self.valid = {c: vz[c] for c in self.chroms}
        self.nb = {c: len(self.valid[c]) for c in self.chroms}
        raw, self.meth, self.meth_mean = {}, {}, {}
        for cell in a.cells:
            z = np.load(cache / f"targets_{a.res}" / f"{cell}.npz")
            raw[cell] = {c: z[c] for c in self.chroms}
            z = np.load(cache / "meth" / f"{cell}.npz")
            self.meth[cell] = {c: z[c] for c in self.chroms}
            self.meth_mean[cell] = float(z["_mean"])
        if a.meth_norm != "none":
            self.normalise_meth()
        # z-score statistics: training chromosomes of the fitted cells, valid bins only
        self.mu = np.zeros(len(MARKS))
        self.sd = np.ones(len(MARKS))
        for k in range(len(MARKS)):
            v = np.concatenate([raw[cell][c][self.valid[c], k].astype(np.float64)
                                for cell in a.fit_cells for c in a.train_chroms])
            v = v[~np.isnan(v)]
            self.mu[k], self.sd[k] = v.mean(), max(v.std(), 1e-6)
        self.y = {cell: {c: ((raw[cell][c].astype(np.float32) - self.mu) / self.sd).astype(np.float16)
                         for c in self.chroms} for cell in a.cells}
        if a.target_norm == "cell":
            self.cell_target_norm(raw)
        self.check_disjoint()

    def valid_meth_values(self, cell):
        """1 kb Meth values of one cell inside its valid 10 kb bins (all used chromosomes)."""
        out = []
        for c in self.chroms:
            v = self.meth[cell][c]
            n = min(len(v), self.nb[c] * (self.res // METH_BIN)) if self.res >= METH_BIN else len(v)
            ok = np.repeat(self.valid[c], max(self.res // METH_BIN, 1))[:n]
            out.append(v[:n][ok[:n]])
        return np.concatenate(out).astype(np.float64)

    def normalise_meth(self):
        """Per-cell scaling of the Meth channel from the cell's own genome-wide Meth values."""
        from scipy.special import ndtri
        for cell in self.a.cells:
            ref = self.valid_meth_values(cell)
            if self.a.meth_norm == "cell":
                mu, sd = ref.mean(), max(ref.std(), 1e-6)
                f = lambda v: ((v - mu) / sd).astype(np.float32)
            else:
                srt = np.sort(ref)
                def f(v, srt=srt):
                    u = (np.searchsorted(srt, v, "left") + np.searchsorted(srt, v, "right")) / (2.0 * len(srt))
                    return ndtri(np.clip(u, 0.5 / len(srt), 1 - 0.5 / len(srt))).astype(np.float32)
            self.meth[cell] = {c: f(v) for c, v in self.meth[cell].items()}
            # padding value outside the chromosome = the cell's typical level; the mean channel
            # is not informative any more and is 0 for every cell
            self.meth_mean[cell] = float(f(np.array([self.meth_mean[cell]]))[0])

    def cell_target_norm(self, raw):
        """z-score every cell's targets with its own statistics (see --target-norm)."""
        a = self.a
        for cell in a.cells:
            chs = a.train_chroms if cell in a.fit_cells else self.chroms
            for k in range(len(MARKS)):
                v = np.concatenate([raw[cell][c][self.valid[c], k].astype(np.float64) for c in chs])
                v = v[~np.isnan(v)]
                if len(v) < 10:
                    continue
                mu, sd = v.mean(), max(v.std(), 1e-6)
                for c in self.chroms:
                    self.y[cell][c][:, k] = ((raw[cell][c][:, k].astype(np.float32) - mu) / sd).astype(np.float16)

    def check_disjoint(self):
        a = self.a
        tr, va, te = set(a.train_chroms), set(a.val_chroms), set(a.test_chroms)
        assert not (tr & va), f"training and validation chromosomes overlap: {tr & va}"
        assert not (tr & te), f"training and test chromosomes overlap: {tr & te}"
        if not a.smoke:
            assert not (va & te), f"validation and test chromosomes overlap: {va & te}"
        if a.holdout_cell:
            assert a.holdout_cell not in a.fit_cells

    def codes(self, chrom, start):
        n = self.sizes[chrom]
        out = np.full(self.L, 4, dtype=np.uint8)
        s, e = max(start, 0), min(start + self.L, n)
        if e > s:
            out[s - start:e - start] = self.seq[chrom][s:e]
        return out

    def meth_bp(self, cell, chrom, start):
        arr = self.meth[cell][chrom]
        idx = np.arange(start, start + self.L) // METH_BIN
        ok = (idx >= 0) & (idx < len(arr))
        out = np.full(self.L, self.meth_mean[cell], np.float32)
        out[ok] = arr[idx[ok]]
        return out

    def target(self, cell, chrom, b0):
        """[NB, 13] z-scored, NaN where the bin is invalid or the mark absent."""
        out = np.full((self.NB, len(MARKS)), np.nan, np.float32)
        e = min(b0 + self.NB, self.nb[chrom])
        if e > b0:
            y = self.y[cell][chrom][b0:e].astype(np.float32)
            y[~self.valid[chrom][b0:e]] = np.nan
            out[:e - b0] = y
        return out

    def window(self, chrom, b0, cell, cells_y, rc=False):
        """cell: cell whose Meth is used (None in seq mode); cells_y: cells whose targets are returned."""
        start = b0 * self.res - self.margin
        codes = self.codes(chrom, start)
        meth = self.meth_bp(cell, chrom, start) if cell is not None else np.zeros(self.L, np.float32)
        mm = self.meth_mean[cell] if cell is not None else 0.0
        if cell is not None and self.a.meth_norm != "none":
            mm = 0.0
        item = {"codes": codes, "meth": meth, "mmean": np.float32(mm)}
        item["y"] = (np.stack([self.target(c, chrom, b0) for c in cells_y]) if cells_y
                     else np.zeros((0, self.NB, len(MARKS)), np.float32))
        if rc:
            item["codes"] = COMP[item["codes"][::-1]]
            item["meth"] = item["meth"][::-1].copy()
            item["y"] = item["y"][:, ::-1].copy()
        return item


def celldep(a):
    return a.mode != "seq" or a.cell_embedding


class TrainStream:
    """Infinite stream of random training windows (chromosome by length, bin-aligned start)."""

    def __init__(self, D, a, seed):
        self.D, self.a, self.seed = D, a, seed
        self.chroms = [c for c in a.train_chroms if D.nb[c] > D.NB]
        w = np.array([D.nb[c] for c in self.chroms], float)
        self.p = w / w.sum()

    def __iter__(self):
        import torch
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        D, a = self.D, self.a
        while True:
            for _ in range(100):
                c = self.chroms[rng.choice(len(self.chroms), p=self.p)]
                b0 = int(rng.integers(0, D.nb[c] - D.NB + 1))
                if D.valid[c][b0:b0 + D.NB].mean() < 0.5:
                    continue
                if celldep(a):
                    ci = int(rng.integers(len(a.fit_cells)))
                    cell, ys = a.fit_cells[ci], [a.fit_cells[ci]]
                else:
                    ci, cell, ys = -1, None, a.fit_cells
                it = D.window(c, b0, cell, ys, rc=(not a.no_rc_aug and rng.random() < 0.5))
                if np.isfinite(it["y"]).any():
                    it["cell"] = np.int64(ci if a.cell_embedding else -1)
                    yield {k: torch.as_tensor(np.asarray(v)) for k, v in it.items()}
                    break


class _TrainDS:
    pass


def train_dataset(stream):
    import torch

    class DS(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(stream)
    return DS()


class EvalSet:
    def __init__(self, D, items, a, ys_fn):
        self.D, self.items, self.a, self.f = D, items, a, ys_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        chrom, b0, cell = self.items[i]
        a = self.a
        it = self.D.window(chrom, b0, cell if a.mode != "seq" else None, self.f(cell))
        it["cell"] = np.int64(a.fit_cells.index(cell) if (a.cell_embedding and cell in a.fit_cells)
                              else -1)
        return it


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

def pool_matrix(T, a):
    """[NB, T]: overlap of trunk position t (input bp t*128 .. (t+1)*128) with output bin j."""
    margin = (a.window - a.out_bins * a.res) // 2
    P = np.zeros((a.out_bins, T))
    for j in range(a.out_bins):
        lo, hi = margin + j * a.res, margin + (j + 1) * a.res
        for t in range(max(lo // DOWNSAMPLE, 0), min(-(-hi // DOWNSAMPLE), T)):
            ov = min(hi, (t + 1) * DOWNSAMPLE) - max(lo, t * DOWNSAMPLE)
            P[j, t] = max(ov, 0) / a.res
    return P


def make_model(a, in_ch):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, c, dil, drop):
            super().__init__()
            self.n1, self.c1 = nn.BatchNorm1d(c), nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
            self.n2, self.c2 = nn.BatchNorm1d(c), nn.Conv1d(c, c, 1)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            h = self.c1(F.gelu(self.n1(x)))
            return x + self.c2(self.drop(F.gelu(self.n2(h))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            s = a.stem
            self.stem = nn.Sequential(nn.Conv1d(in_ch, s, 15, padding=7), nn.BatchNorm1d(s), nn.GELU())
            chans = [s, min(int(1.5 * s), a.width), min(2 * s, a.width), min(3 * s, a.width), a.width]
            strides = [4, 4, 2, 4]            # product 128
            assert math.prod(strides) == DOWNSAMPLE
            self.down = nn.ModuleList()
            for i, st in enumerate(strides):
                self.down.append(nn.Sequential(
                    nn.Conv1d(chans[i], chans[i + 1], 2 * st + 1, stride=st, padding=st),
                    nn.BatchNorm1d(chans[i + 1]), nn.GELU(),
                    nn.Conv1d(chans[i + 1], chans[i + 1], 5, padding=2),
                    nn.BatchNorm1d(chans[i + 1]), nn.GELU()))
            self.emb = None
            if a.cell_embedding:
                self.emb = nn.Embedding(len(a.fit_cells), 32)
                self.emb_proj = nn.Linear(32, a.width)
            self.blocks = nn.ModuleList([Block(a.width, 2 ** (i % 8), a.dropout)
                                         for i in range(a.blocks)])
            self.norm = nn.BatchNorm1d(a.width)
            self.head = nn.Sequential(nn.Conv1d(a.width, a.width, 1), nn.GELU(),
                                      nn.Dropout(a.dropout), nn.Conv1d(a.width, len(MARKS), 1))
            self.register_buffer("pool", torch.tensor(pool_matrix(a.window // DOWNSAMPLE, a),
                                                      dtype=torch.float32))

        def forward(self, x, cell=None):
            h = self.stem(x)
            for d in self.down:
                h = d(h)
            if self.emb is not None:
                h = h + self.emb_proj(self.emb(cell.clamp(min=0)))[:, :, None]
            for b in self.blocks:
                h = b(h)
            h = F.gelu(self.norm(h))
            h = torch.einsum("bct,jt->bcj", h, self.pool.to(h.dtype))   # exact bin pooling
            return self.head(h).transpose(1, 2)                          # [B, NB, 13]

    return Net()


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def in_channels(a):
    return {"seq": 4, "seq_meth": 6, "meth": 2}[a.mode]


def make_input(batch, a, dev, train=False):
    import torch
    import torch.nn.functional as F
    chans = []
    if a.mode in ("seq", "seq_meth"):
        codes = batch["codes"].to(dev).long()
        chans.append(F.one_hot(codes, 5)[..., :4].permute(0, 2, 1).float())
    if a.mode in ("seq_meth", "meth"):
        m = batch["meth"].to(dev).float()
        mm = batch["mmean"].to(dev).float()[:, None].expand_as(m)
        if a.meth_norm == "none":
            mt = torch.stack([m * 2 - 1, mm * 2 - 1], 1)
        else:
            mt = torch.stack([m, mm], 1)               # already per-cell normalised, mean channel 0
        if train and a.meth_aug:
            B = mt.shape[0]
            sc = 1 + a.meth_aug_scale * (torch.rand(B, 1, 1, device=dev) * 2 - 1)
            sh = a.meth_aug_shift * torch.randn(B, 1, 1, device=dev)
            mt = mt * sc + sh
        if train and a.meth_dropout > 0:
            keep = (torch.rand(mt.shape[0], 1, 1, device=dev) >= a.meth_dropout).float()
            mt = mt * keep
        chans.append(mt)
    return torch.cat(chans, 1), batch["cell"].to(dev)


def reverse_complement_input(x, a):
    import torch
    xr = x.flip(-1)
    if a.mode in ("seq", "seq_meth"):
        xr = torch.cat([xr[:, :4].flip(1), xr[:, 4:]], 1)     # A C G T -> T G C A
    return xr


def loss_fn(pred, y, D, loss, dev):
    """pred [B, NB, 13], y [B, nc, NB, 13] with NaN; mean per mark, then over marks."""
    import torch
    mask = torch.isfinite(y)
    yz = torch.where(mask, y, torch.zeros_like(y))
    p = pred[:, None].float()
    if loss == "mse":
        e = (p - yz) ** 2
    else:
        mu = torch.tensor(D.mu, device=dev, dtype=torch.float32)
        sd = torch.tensor(D.sd, device=dev, dtype=torch.float32)
        rate = torch.expm1((p * sd + mu).clamp(max=12.0)).clamp_min(1e-3)
        t = torch.expm1((yz * sd + mu).clamp(max=12.0)).clamp_min(0.0)
        e = (rate - t * torch.log(rate)) / torch.expm1(mu).clamp_min(1e-3)
    e = torch.where(mask, e, torch.zeros_like(e))
    cnt = mask.sum((0, 1, 2))
    per_mark = e.sum((0, 1, 2)) / cnt.clamp_min(1)
    return per_mark[cnt > 0].mean()


# ----------------------------------------------------------------------------
# prediction and metrics
# ----------------------------------------------------------------------------

def pick_device(a):
    import torch
    if a.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(a.device)


def predict(model, D, a, dev, items, tta, ys_fn):
    """items [(chrom, b0, cell)] -> pred [N, NB, 13] (z scale), y [N, nc, NB, 13]."""
    import torch
    dl = torch.utils.data.DataLoader(EvalSet(D, items, a, ys_fn), batch_size=a.eval_batch,
                                     shuffle=False, num_workers=a.num_workers)
    model.eval()
    P, Y = [], []
    amp = dev.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
        for b in dl:
            x, cell = make_input(b, a, dev)
            p = model(x, cell).float()
            if tta:
                p = 0.5 * (p + model(reverse_complement_input(x, a), cell).float().flip(1))
            P.append(p.cpu().numpy())
            Y.append(b["y"].numpy())
    return np.concatenate(P), np.concatenate(Y)


def corr_stats(y, p):
    from scipy.stats import spearmanr
    n = len(y)
    if n < 10 or np.std(y) == 0 or np.std(p) == 0:
        return n, np.nan, np.nan, np.nan
    sse = float(((y - p) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return n, 1 - sse / sst, float(np.corrcoef(y, p)[0, 1]), float(spearmanr(y, p)[0])


def standardise_pair(a, y, p):
    """--target-norm cell: both sides standardised over the scored bins (R2 is then scale free)."""
    if a.target_norm != "cell" or len(y) < 10 or y.std() == 0 or p.std() == 0:
        return y, p
    return (y - y.mean()) / y.std(), (p - p.mean()) / p.std()


def tile_items(D, chroms, cell):
    return [(c, b0, cell) for c in chroms for b0 in range(0, D.nb[c], D.NB)]


def cell_scheme(a, cell):
    return "seen_cell" if cell in a.fit_cells else "unseen_cell"


def valid_y(D, cell, chrom):
    y = D.y[cell][chrom].astype(np.float32).copy()
    y[~D.valid[chrom]] = np.nan
    return y


def stage_eval(a):
    import pandas as pd
    import torch
    out = Path(a.out_dir)
    if (out / "eval.csv").exists() and not a.force:
        log(f"eval of {a.run_name} already done")
        return
    dev = pick_device(a)
    D = Data(a)
    model = make_model(a, in_channels(a)).to(dev)
    ck = out / "ckpt_best.pt"
    if not ck.exists():
        raise SystemExit(f"no checkpoint {ck}; run --stage train first")
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
    tta = not a.no_tta
    rows = []
    test_cells = [c for c in a.cells if (c in a.fit_cells or not a.cell_embedding)]
    skipped = [c for c in a.cells if c not in test_cells]
    if skipped:
        log(f"cell embedding: unseen cells {skipped} cannot be scored")
    if not celldep(a):
        shared, _ = predict(model, D, a, dev, tile_items(D, a.test_chroms, None), tta, lambda c: [])
    for cell in test_cells:
        t0 = time.time()
        p = shared if not celldep(a) else predict(
            model, D, a, dev, tile_items(D, a.test_chroms, cell), tta, lambda c: [])[0]
        k, yy, pp = 0, [], []
        for c in a.test_chroms:
            nwin = len(range(0, D.nb[c], D.NB))
            pp.append(p[k:k + nwin].reshape(-1, len(MARKS))[:D.nb[c]])
            yy.append(valid_y(D, cell, c))
            k += nwin
        yy, pp = np.concatenate(yy), np.concatenate(pp)
        for m, mark in enumerate(MARKS):
            ok = np.isfinite(yy[:, m])
            n, r2, pr, sp = corr_stats(*standardise_pair(a, yy[ok, m].astype(np.float64),
                                                         pp[ok, m].astype(np.float64)))
            rows.append(dict(mode=a.mode, variant=a.run_name, mark=mark, cell=cell,
                             scheme=cell_scheme(a, cell), split="test", res=a.res, n_bins=n,
                             r2=r2, pearson=pr, spearman=sp))
        log(f"eval {a.run_name} {cell}: median pearson "
            f"{np.nanmedian([r['pearson'] for r in rows if r['cell'] == cell]):.3f} "
            f"({time.time() - t0:.0f}s)")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "eval.csv", index=False, float_format="%.5f")
    ceiling_rows(a, D).to_csv(out / "ceiling.csv", index=False, float_format="%.5f")


def ceiling_rows(a, D):
    """Best cell-agnostic prediction: the mean z-scored track of the fitted cells.

    ceiling_incl  mean over all fitted cells with the mark (includes the scored cell when it
                  was fitted: the upper bound for any prediction without a cell input);
    ceiling_loo   mean over the fitted cells other than the scored one."""
    import pandas as pd
    rows = []
    for kind in ("ceiling_incl", "ceiling_loo"):
        for cell in a.cells:
            yy, pp = [], []
            for c in a.test_chroms:
                others = [o for o in a.fit_cells if not (kind == "ceiling_loo" and o == cell)]
                if not others:
                    pp.append(np.full((D.nb[c], len(MARKS)), np.nan, np.float32))
                    yy.append(valid_y(D, cell, c))
                    continue
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pp.append(np.nanmean(np.stack([valid_y(D, o, c) for o in others]), 0))
                yy.append(valid_y(D, cell, c))
            yy, pp = np.concatenate(yy), np.concatenate(pp)
            for m, mark in enumerate(MARKS):
                ok = np.isfinite(yy[:, m]) & np.isfinite(pp[:, m])
                n, r2, pr, sp = corr_stats(*standardise_pair(a, yy[ok, m].astype(np.float64),
                                                             pp[ok, m].astype(np.float64)))
                rows.append(dict(mode=kind, variant=kind + ("_tcell" if a.target_norm == "cell" else ""), mark=mark, cell=cell,
                                 scheme=cell_scheme(a, cell), split="test", res=a.res, n_bins=n,
                                 r2=r2, pearson=pr, spearman=sp))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def val_items(D, a):
    rng = np.random.default_rng(12345)
    pool = [(c, b0) for c in a.val_chroms for b0 in range(0, D.nb[c] - D.NB + 1, D.NB)
            if D.valid[c][b0:b0 + D.NB].mean() >= 0.5]
    sel = sorted(rng.choice(len(pool), min(a.val_windows, len(pool)), replace=False))
    return [(*pool[j], a.fit_cells[i % len(a.fit_cells)]) for i, j in enumerate(sel)]


def validate(model, D, a, dev, items):
    ys = (lambda c: [c]) if celldep(a) else (lambda c: a.fit_cells)
    p, y = predict(model, D, a, dev, items, False, ys)
    pr = np.broadcast_to(p[:, None], y.shape)
    cors, mse = [], []
    for m in range(len(MARKS)):
        ok = np.isfinite(y[..., m])
        if ok.sum() > 10 and y[..., m][ok].std() > 0 and pr[..., m][ok].std() > 0:
            cors.append(float(np.corrcoef(y[..., m][ok], pr[..., m][ok])[0, 1]))
            mse.append(float(((y[..., m][ok] - pr[..., m][ok]) ** 2).mean()))
    return float(np.mean(cors)), float(np.mean(mse))


def save_atomic(obj, path):
    import torch
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def stage_train(a):
    import torch
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_done.json").exists() and not a.force:
        log(f"training of {a.run_name} already finished")
        return
    fh = open(out / "log.txt", "a")
    dev = pick_device(a)
    torch.manual_seed(a.seed)
    D = Data(a)
    model = make_model(a, in_channels(a)).to(dev)
    log(f"{a.run_name}: {n_params(model) / 1e6:.2f} M parameters, device {dev}, fit cells "
        f"{a.fit_cells}, train {len(a.train_chroms)} chr, val {a.val_chroms}", fh)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        return a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) /
                                                         max(a.steps - a.warmup, 1))))

    step, best, bad = 0, -1e9, 0
    last = out / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best, bad = ck["step"], ck["best"], ck["bad"]
        log(f"resumed from step {step}", fh)
    vitems = val_items(D, a)
    dl = torch.utils.data.DataLoader(
        train_dataset(TrainStream(D, a, a.seed * 1000003 + step)), batch_size=a.batch_size,
        num_workers=a.num_workers, prefetch_factor=4 if a.num_workers else None,
        persistent_workers=a.num_workers > 0)
    amp = dev.type == "cuda"
    if amp:
        torch.backends.cudnn.benchmark = True
    it = iter(dl)
    logf = out / "train_log.csv"
    if not logf.exists():
        logf.write_text("step,loss,lr,samples_per_s,val_pearson,val_mse\n")
    t0, run_loss, nl = time.time(), 0.0, 0
    log_every = max(1, min(100, a.val_every))
    while step < a.steps:
        model.train()
        b = next(it)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, cell = make_input(b, a, dev, train=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(x, cell)
        loss = loss_fn(pred.float(), b["y"].to(dev), D, a.loss, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run_loss += loss.item()
        nl += 1
        vp = vm = ""
        if step % a.val_every == 0 or step == a.steps:
            vpe, vme = validate(model, D, a, dev, vitems)
            vp, vm = f"{vpe:.4f}", f"{vme:.4f}"
            if vpe > best:
                best, bad = vpe, 0
                save_atomic({"model": model.state_dict(), "step": step, "val_pearson": vpe},
                            out / "ckpt_best.pt")
            else:
                bad += 1
        if step % log_every == 0 or vp or step == a.steps:
            sps = nl * a.batch_size / (time.time() - t0)
            with open(logf, "a") as f:
                f.write(f"{step},{run_loss / nl:.5f},{lr_at(step):.3e},{sps:.1f},{vp},{vm}\n")
            log(f"step {step}/{a.steps} loss {run_loss / nl:.4f} {sps:.1f} samples/s"
                + (f" val_pearson {vp} val_mse {vm}" if vp else ""), fh)
            t0, run_loss, nl = time.time(), 0.0, 0
        if step % a.ckpt_every == 0 or step == a.steps:
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                         "best": best, "bad": bad}, last)
        if a.patience and bad >= a.patience:
            log(f"early stop at step {step} ({bad} validations without improvement)", fh)
            break
    if not (out / "ckpt_best.pt").exists():
        save_atomic({"model": model.state_dict(), "step": step, "val_pearson": best},
                    out / "ckpt_best.pt")
    (out / "train_done.json").write_text(json.dumps(
        {"step": step, "best_val_pearson": best, "params": n_params(model)}))
    log(f"training done, best val pearson {best:.4f}", fh)


# ----------------------------------------------------------------------------
# summary and benchmark
# ----------------------------------------------------------------------------

def stage_summarize(a):
    import pandas as pd
    root = Path(a.out_root)
    ev = [pd.read_csv(f) for f in sorted(root.glob("*/eval.csv"))]
    ce = [pd.read_csv(f) for f in sorted(root.glob("*/ceiling.csv"))]
    if not ev:
        raise SystemExit(f"no eval.csv under {root}")
    allr = pd.concat(ev + ce, ignore_index=True)
    allr = allr.drop_duplicates(["variant", "mark", "cell"]).reset_index(drop=True)
    allr["independent_cell"] = allr["cell"].replace(ALIAS)
    res = Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    allr.to_csv(res / "seq_cnn_results.csv", index=False, float_format="%.5f")
    summ = (allr.groupby(["variant", "mode", "scheme", "mark"])
            .agg(r2_median=("r2", "median"), pearson_median=("pearson", "median"),
                 spearman_median=("spearman", "median"), n_cells=("cell", "nunique"),
                 n_independent_cells=("independent_cell", "nunique")).reset_index())
    summ.to_csv(res / "seq_cnn_summary.csv", index=False, float_format="%.5f")
    base = allr[allr.variant.isin(["seq", "seq_meth", "meth"])]
    if set(base["mode"]) >= {"seq", "seq_meth", "meth"}:
        pv = base.pivot_table(index=["mark", "cell", "scheme"], columns="mode", values="r2").reset_index()
        pv["shared"] = pv["seq"] + pv["meth"] - pv["seq_meth"]
        pv["unique_seq"] = pv["seq_meth"] - pv["meth"]
        pv["unique_meth"] = pv["seq_meth"] - pv["seq"]
        pv["unexplained"] = 1 - pv["seq_meth"]
        pv = pv.rename(columns={"seq": "r2_seq", "meth": "r2_meth", "seq_meth": "r2_seq_meth"})
        pv.to_csv(res / "seq_cnn_partition.csv", index=False, float_format="%.5f")
        print(pv.groupby(["scheme", "mark"])[["r2_seq", "r2_meth", "r2_seq_meth", "shared",
                                              "unique_seq", "unique_meth"]].median().round(3).to_string())
    keep = summ[~summ.variant.str.contains("_embed|_ho-|_poisson|_1kb")]
    print(keep.pivot_table(index="mark", columns=["variant", "scheme"], values="pearson_median")
          .round(3).to_string())
    log(f"wrote {res / 'seq_cnn_results.csv'}")


def stage_benchmark(a):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    if a.out_bins is None:
        a.out_bins = 8
    for mode in ("seq", "seq_meth", "meth"):
        a.mode = mode
        model = make_model(a, in_channels(a))
        x = torch.randn(2, in_channels(a), a.window)
        cell = torch.zeros(2, dtype=torch.long)
        with FlopCounterMode(display=False) as fc:
            model(x, cell).sum().backward()
        flops = fc.get_total_flops() / 2
        ts = []
        for _ in range(3):
            t = time.time()
            model(x, cell).sum().backward()
            ts.append(time.time() - t)
        dt = min(ts[1:])
        print(f"{mode}: {n_params(model) / 1e6:.2f} M params, {flops / 1e9:.1f} GFLOP per sample "
              f"(fwd+bwd), CPU {2 / dt:.2f} samples/s, {flops * 2 / dt / 1e12:.3f} TFLOP/s achieved")


def main():
    a = parse_args()
    if a.benchmark:
        stage_benchmark(a)
        return
    for st in a.stage:
        if st == "cache":
            if cache_complete(a) and not a.force:
                log("cache already built")
            else:
                stage_cache(a)
        elif st == "train":
            stage_train(a)
        elif st == "eval":
            stage_eval(a)
        elif st == "summarize":
            stage_summarize(a)


if __name__ == "__main__":
    main()
