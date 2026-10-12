"""Do published Hi-C predictors detect the AML-specific loops and stripes? (registry section 72)

Same objects, windows and rule as scripts/aml_loop_detection.py (10 held-out donors, 2276 loops, 1075
stripes with B_rand and A_* controls): the table object_scores.csv.gz of the diff_loopfocus_after_T1b
run holds the measured, consensus (AML reference), HSPC and our models' enrichments. This script adds

  * our models WITHOUT any AML sample in training (stage ours): all12refin_drop_10kb and
    all12refres_drop_10kb with consensus12 (cell lines only) as Hi-C reference, and
    all12diffcells_refres_drop_10kb (marks relative to the mean of the 12 cell lines,
    --diff-group cell) with consensus12; and the same models on the HSPC CD34+ marks (contrast);
  * competitors on the AML windows (stages corigami, epiphany, hicreg, akita);
  * the scoring of every map with the estimator of aml_loop_detection (donut for loops, rectangle
    mean for stripes), the 90th-percentile detection rule, AUROC vs B_rand and A_refrand with donor
    bootstrap, and Spearman with the measured enrichment among AML-specific (private) objects.

    python scripts/aml_competitors_loops.py --stage ours --gpu 1
    python scripts/aml_competitors_loops.py --stage score stats figure
Run with the hicdiff conda env. Outputs: results/aml_competitors_loops/.
"""
from __future__ import annotations

import argparse
import os
import warnings
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
SRC = HM / "runs/diff_marks/aml_loop_detection_diff_loopfocus_after_T1b/object_scores.csv.gz"
WINDOWS = HM / "runs/aml_loop_detection/windows"
OUTD = HM / "runs/aml_competitors_loops"
RES_OUT = REPO / "results/aml_competitors_loops"
PARTS = OUTD / "scores"          # per-patient score tables (large; merged into RES_OUT/scores.csv.gz)
MR = HM / "runs/multicell_reference"
RES, WIN = 10000, 192
PATIENTS = ["018", "027", "270", "168", "546", "629", "1360", "472", "773", "1916"]
FOLD = {"018": 10, "027": 10, "270": 10, "168": 11, "546": 11, "629": 11, "1360": 11,
        "472": 12, "773": 12, "1916": 12}

# no-AML models of the 12 cell lines: (label, run, extra predict flags)
NOAML = [("c12_refin", "all12refin_drop_10kb", []),
         ("c12_refres", "all12refres_drop_10kb", []),
         ("c12_diffcells", "all12diffcells_refres_drop_10kb", ["--diff-group", "cell"])]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ===================================================================== our no-AML models
def stage_ours(gpu: str, threads: int, only: list[str] | None) -> None:
    import aml_train as at
    import multicell_expanded as mx
    import aml_train_eval as te
    c12 = mx.consensus12(RES)
    for p in PATIENTS:
        if only and p not in only:
            continue
        for lab, run, flags in NOAML:
            ck = te.ckpt(run)
            for group, cell, d, tgt in (
                    ("patient", at.guard(p), at.INP / at.label(p), at.OE_DIR / f"xu_{p}_proper_10kb.cool"),
                    ("hspc", at.hspc_guard("341"), at.HSPC_MARKS, at.OE_DIR / f"{at.hspc_label('341')}_proper_10kb.cool")):
                marks = sorted(b.stem for b in d.glob("*.bigwig"))
                out = OUTD / "ours" / f"{lab}@{group}__xu_{p}"
                if (out / "marks.npz").exists():
                    continue
                cmd = [te.PY, "scripts/crosscell_predict.py", "--config", ck / "config.yaml",
                       "--checkpoint", ck, "--cell", cell, "--bigwig-dir", d, "--target", tgt,
                       "--tta", "--assay", "hic", "--binsize", RES, "--device", "cuda",
                       "--regions", WINDOWS / f"xu_{p}.tsv", "--out-dir", out,
                       "--save-pred", str(out / "{set}.npz"),
                       "--keep-sets", f"marks={','.join(marks)}", "--reference", c12, *flags]
                env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                           CUDA_VISIBLE_DEVICES=gpu)
                out.mkdir(parents=True, exist_ok=True)
                with open(out / "predict.log", "a") as fh:
                    fh.write(f"[cmd] {' '.join(map(str, cmd))}\n")
                    fh.flush()
                    rc = subprocess.call(["nice", "-n", "10"] + [str(c) for c in cmd], stdout=fh,
                                         stderr=subprocess.STDOUT, cwd=REPO, env=env)
                log(f"{lab} {group} {p} rc={rc}")


# ===================================================================== map conversion (shared)
CLIP, SIGMA = 2.0, 2.0          # oe_proper.py constants (natural log O/E, clip, Gaussian sigma in native bins)


def log_to_final(L: np.ndarray, reach: int) -> np.ndarray:
    """Per-diagonal centring, clip, Gaussian smoothing (sigma 2 native bins, truncate 1) of a native
    log O/E block L (n x n, NaN = no prediction); pixels beyond `reach` bins become NaN. The window
    counterpart of the last passes of oe_proper.py (emit_chrom)."""
    from scipy.ndimage import gaussian_filter
    n = L.shape[0]
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    off = np.zeros(n)
    for k in range(n):
        v = np.diagonal(L, k)
        v = v[np.isfinite(v)]
        off[k] = v.mean() if v.size else 0.0
    val = np.clip(L - off[d], -CLIP, CLIP)
    val = np.where((d <= reach) & np.isfinite(val), val, 0.0)       # unpredicted pixels: 0 before smoothing
    sm = gaussian_filter(val, sigma=SIGMA, truncate=1.0, mode="nearest")
    return np.where(d <= reach, sm, np.nan)


def counts_to_log_oe(C: np.ndarray, reach: int) -> np.ndarray:
    """oe_proper.py on a dense predicted count block: expected(d) = mean count over all pairs of
    diagonal d, log(obs / exp) clipped to +-2, a zero count = -2 (a pair that is mappable and absent)."""
    n = C.shape[0]
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    C = np.where(np.isfinite(C), np.maximum(C, 0.0), np.nan)
    exp = np.full(n, np.nan)
    for k in range(min(n, reach + 1)):
        v = np.diagonal(C, k)
        v = v[np.isfinite(v)]
        if v.size and v.mean() > 0:
            exp[k] = v.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.log(C / exp[d])
    L = np.where(np.isfinite(C) & (C == 0), -CLIP, L)
    return np.clip(L, -CLIP, CLIP)


def sample_grid(M: np.ndarray, cres: int, x0_bp: float, b0: int, n: int = WIN, res: int = RES) -> np.ndarray:
    """Bilinear (in bin, bin space) sampling at the target 10 kb bin centres, as benchmark_blocks.
    sample_competitor: native bin k covers [x0 + k * cres, x0 + (k + 1) * cres); a value is kept only
    where the interpolation weights touch no missing pixel."""
    from scipy.ndimage import map_coordinates
    centres = (b0 + np.arange(n) + 0.5) * res
    u = (centres - x0_bp) / cres - 0.5
    valid = np.isfinite(M).astype(float)
    Z = np.nan_to_num(M)
    yy, xx = np.meshgrid(u, u, indexing="ij")
    ok = (yy >= -0.5) & (yy <= M.shape[0] - 0.5) & (xx >= -0.5) & (xx <= M.shape[1] - 0.5)
    out = np.full((n, n), np.nan)
    v = map_coordinates(Z, [yy[ok], xx[ok]], order=1, mode="nearest")
    w = map_coordinates(valid, [yy[ok], xx[ok]], order=1, mode="nearest")
    out[ok] = np.where(w > 0.999, v, np.nan)
    return out


def read_windows(p: str) -> list[tuple[str, int]]:
    w = pd.read_csv(WINDOWS / f"xu_{p}.tsv", sep="\t", header=None, names=["c", "s", "e"])
    return [(c, int(s) // RES) for c, s in zip(w.c, w.s)]


def win_name(c: str, b0: int) -> str:
    return f"{c}:{b0 * RES}-{(b0 + WIN) * RES}"


def hg38_seq(chrom: str) -> np.ndarray:
    """Chromosome as codes a0 t1 c2 g3 n4 (C.Origami's sequence folder, hg38)."""
    import gzip
    cache = OUTD / "seq" / f"{chrom}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    f = Path.home() / f"src/C.Origami/corigami_data/data/hg38/dna_sequence/{chrom}.fa.gz"
    raw = gzip.open(f, "rb").read()
    raw = raw[raw.find(b"\n") + 1:].replace(b"\n", b"").lower()
    lut = np.full(256, 4, np.uint8)
    for k, ch in enumerate(b"atcg"):
        lut[ch] = k
    out = lut[np.frombuffer(raw, np.uint8)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache.with_suffix(".tmp.npy"), out)
    cache.with_suffix(".tmp.npy").rename(cache)
    return out


def bw_values(bw, chrom: str, s: int, e: int) -> np.ndarray:
    """bigWig values of [s, e) at 1 bp; positions outside the chromosome and NaN = 0."""
    size = bw.chroms()[chrom]
    out = np.zeros(e - s, np.float32)
    a, b = max(s, 0), min(e, size)
    if b > a:
        v = np.array(bw.values(chrom, a, b), np.float32)
        out[a - s:b - s] = np.nan_to_num(v)
    return out


def bw_binned(bw, chrom: str, s: int, e: int, nb: int) -> np.ndarray:
    """Mean per bin of [s, e) in nb bins (positions outside the chromosome = 0)."""
    size = bw.chroms()[chrom]
    step = (e - s) // nb
    out = np.zeros(nb, np.float32)
    a, b = max(s, 0), min(e, size)
    k0, k1 = (a - s) // step, (b - s) // step
    if k1 > k0:
        v = np.array(bw.stats(chrom, s + k0 * step, s + k1 * step, nBins=k1 - k0), dtype=float)
        out[k0:k1] = np.nan_to_num(v)
    return out


def genome_mean(path: str) -> float:
    import pyBigWig
    h = pyBigWig.open(str(path)).header()
    return h["sumData"] / h["nBasesCovered"]


def quantile_map(src: str, ref: str, cache: Path):
    """(x, y) of the monotone map from the distribution of bigWig `src` (1 bp values of 600 random 25 kb
    segments of chr1 to 22) onto that of `ref`; applied with np.interp."""
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    import pyBigWig

    rg = np.random.default_rng(0)
    sizes = pyBigWig.open(str(ref)).chroms()
    chroms = [f"chr{i}" for i in range(1, 23)]
    w = np.array([sizes[c] for c in chroms], float)
    segs = [(chroms[k], int(rg.integers(0, sizes[chroms[k]] - 25000))) for k in rg.choice(len(chroms), 600, p=w / w.sum())]

    def sample(f):
        """1 bp values of 600 random 25 kb segments (the same segments for source and reference)."""
        bw = pyBigWig.open(str(f))
        v = np.concatenate([np.array(bw.values(c, s0, s0 + 25000), dtype=float) for c, s0 in segs])
        return v[np.isfinite(v)]
    q = np.concatenate([np.linspace(0, 99, 100), np.linspace(99, 99.99, 100), [100]]) / 100
    x = np.quantile(sample(src), q)
    y = np.quantile(sample(ref), q)
    x, ix = np.unique(x, return_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y[ix])
    return x, y[ix]


# ===================================================================== C.Origami
CO_DATA = Path.home() / "src/C.Origami/corigami_data/data/hg38/imr90/genomic_features"
GALAXY_MODELS = Path("/mnt/processed/wolffjoa/galaxy-hic-models")
AML = HM / "aml/primary"
HSPC_ATAC = [AML / "atac/hspc" / f"{s}.bw" for s in ("SRX19490150", "SRX19490151", "SRX19490152",
                                                     "SRX10942129", "SRX10942130")]
CO_PATIENTS = ["018", "027", "168", "270", "472", "546", "629", "773"]


def atac_path(p: str) -> Path:
    return next((AML / "atac").glob(f"GSM*_{p}_ATAC-seq.10bp.RPGC.bigwig"))


def run_corigami(gpu_dev: str, only, limit, variants=("q", "raw")):
    import pyBigWig
    import torch
    sys.path.insert(0, str(Path.home() / "src/C.Origami/src"))
    from corigami.inference.utils.model_utils import load_default
    dev = torch.device(gpu_dev)
    model = load_default(str(GALAXY_MODELS / "corigami/corigami_base.ckpt"), device=dev).eval()
    W, RESN = 2097152, 8192
    ref_ctcf, ref_atac = CO_DATA / "ctcf_log2fc.bw", CO_DATA / "atac.bw"
    out_dir = OUTD / "corigami"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in CO_PATIENTS:
        if only and p not in only:
            continue
        ctcf = AML / f"inputs/xu_{p}/CTCF.bigwig"
        jobs.append((p, "patient", [ctcf], [atac_path(p)]))
        jobs.append((p, "hspc", [AML / "inputs/hspc_cd34/CTCF.bigwig"], HSPC_ATAC))
    for p, group, ctcfs, atacs in jobs:
        todo = [v for v in variants if not (group == "hspc" and v == "raw")
                and not (out_dir / f"{v}@{group}__xu_{p}.npz").exists()]
        if not todo:
            continue
        qm = {}
        for kind, files, ref in (("ctcf", ctcfs, ref_ctcf), ("atac", atacs, ref_atac)):
            qm[kind] = [quantile_map(f, ref, OUTD / "qmap" / f"{Path(f).parent.name}_{Path(f).stem}__{kind}.npz")
                        for f in files]
        bwc = [pyBigWig.open(str(f)) for f in ctcfs]
        bwa = [pyBigWig.open(str(f)) for f in atacs]
        res = {v: {} for v in todo}
        wins = read_windows(p)
        wins = sorted(wins)[::max(1, len(wins) // limit)][:limit] if limit else sorted(wins)
        cur, seq = None, None
        t0 = time.time()
        for k, (c, b0) in enumerate(wins):
            if c != cur:
                seq, cur = hg38_seq(c), c
            size = len(seq)
            centre = b0 * RES + WIN * RES // 2
            s0 = int(np.clip((centre - W // 2) // RESN * RESN, 0, (size - W) // RESN * RESN))
            oh = np.zeros((W, 5), np.float32)
            oh[np.arange(W), seq[s0:s0 + W]] = 1
            ct = np.mean([bw_values(b, c, s0, s0 + W) for b in bwc], 0)
            at = np.mean([bw_values(b, c, s0, s0 + W) for b in bwa], 0) if len(bwa) == 1 else None
            for v in todo:
                if v == "q":
                    cq = np.mean([np.interp(bw_values(b, c, s0, s0 + W), *qm["ctcf"][i]) for i, b in enumerate(bwc)], 0)
                    aq = np.mean([np.interp(bw_values(b, c, s0, s0 + W), *qm["atac"][i]) for i, b in enumerate(bwa)], 0)
                    ctf, atf = cq, np.log(np.maximum(aq, 0) + 1)
                else:
                    ctf, atf = ct, np.log(np.maximum(at, 0) + 1)
                x = np.concatenate([oh, ctf[:, None], atf[:, None]], 1).astype(np.float32)
                with torch.no_grad():
                    pred = model(torch.from_numpy(x)[None].to(dev))[0].cpu().numpy().astype(np.float64)
                pred = (pred + pred.T) / 2
                C = np.expm1(pred)
                L = log_to_final(counts_to_log_oe(C, 256), 256)
                res[v][win_name(c, b0)] = sample_grid(L, RESN, s0, b0).astype(np.float32)
            if k % 200 == 0:
                log(f"corigami {p} {group}: {k}/{len(wins)} windows, {time.time() - t0:.0f} s")
        for v in todo:
            np.savez_compressed(out_dir / f"{v}@{group}__xu_{p}.npz", **res[v])
        log(f"corigami {p} {group}: written {todo}")


# ===================================================================== Epiphany
EP_ORDER = ["DNaseI", "H3K27ac", "H3K4me3", "H3K27me3", "CTCF"]       # channel order of data_load()
EP_PATIENTS = PATIENTS
EP_SEQ, EP_WIN = 200, 14000       # rows per forward pass (2 Mb along the diagonal), model window (100 bp bins)


def epiphany_pairs(i: int, first: int = EP_WIN // 2 // 100):
    """Bin pairs (10 kb bin index, as pred_assemble) of the 100 outputs of global row i."""
    c1 = [v for v in range(i + first - 50 + 1, i + first + 2) for _ in (0, 1)][::-1][1:-1]
    c2 = [v for v in range(i + first + 1, i + first + 50 + 1) for _ in (0, 1)]
    return np.array(c1), np.array(c2)


def epiphany_block(net, dev, x: np.ndarray, s: int, bin0: int, nbins: int):
    """Counts (nbins x nbins block starting at 10 kb bin bin0, NaN = no prediction) from one forward
    pass of the pretrained 10 kb model. x: (5, 34000) 100 bp means (channels in EP_ORDER) of the
    3.4 Mb slice starting at 10 kb bin s; the 200 output rows are the global rows s .. s + 199, whose
    pairs are placed as in pred_assemble (centre bin row + 71). With s = bin0 - 71 and nbins = 200
    the block is covered completely by this single pass."""
    import torch
    C = np.full((nbins, nbins), np.nan)
    with torch.no_grad():
        y = net(torch.from_numpy(x.astype(np.float32)).to(dev))[0][0].cpu().numpy()   # (200, 100)
    for r in range(EP_SEQ):
        c1, c2 = epiphany_pairs(s + r)
        a, b = c1 - bin0, c2 - bin0
        ok = (a >= 0) & (b < nbins) & (a < nbins) & (b >= 0)
        C[a[ok], b[ok]] = 2.0 ** y[r][ok] - 1.0
        C[b[ok], a[ok]] = C[a[ok], b[ok]]
    return C


class ChromTracks:
    """100 bp means of the Epiphany input tracks of one group on one chromosome (read once)."""

    def __init__(self, specs: dict, chrom: str):
        self.chrom = chrom
        self.a = {}
        for k, items in specs.items():            # items: [(bigWig, scale)]; several = summed (ATAC replicates)
            acc = None
            for bw, sc in items:
                size = bw.chroms()[chrom]
                n = size // 100
                v = np.zeros(n, np.float32)
                step = 20_000_000                                  # 1 bp values (fast), 100 bp means
                for a in range(0, n * 100, step):
                    b = min(a + step, n * 100)
                    r = np.array(bw.values(chrom, a, b), np.float32).reshape(-1, 100)
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = np.nanmean(r, axis=1)
                    v[a // 100:b // 100] = np.nan_to_num(m)
                v *= sc
                acc = v if acc is None else acc + v
            self.a[k] = acc

    def cut(self, k: str, i0: int, n: int) -> np.ndarray:
        v = self.a[k]
        out = np.zeros(n, np.float32)
        a, b = max(i0, 0), min(i0 + n, len(v))
        if b > a:
            out[a - i0:b - i0] = v[a:b]
        return out


def run_epiphany(only, limit):
    import pyBigWig
    import torch
    sys.path.insert(0, str(Path.home() / "src/epiphany/epiphany"))
    import utils.model_architecture_util as dmau
    dev = torch.device("cuda")
    net = dmau.Net(window_size=EP_WIN)
    dmau.restore(net, str(GALAXY_MODELS / "epiphany/pretrained_10kb.pt_model"))
    net.float().eval().to(dev)
    out_dir = OUTD / "epiphany"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in EP_PATIENTS:
        if only and p not in only:
            continue
        for group in ("patient", "hspc"):
            d = AML / (f"inputs/xu_{p}" if group == "patient" else "inputs/hspc_cd34")
            atac = [atac_path(p)] if group == "patient" else HSPC_ATAC
            if not (d / "CTCF.bigwig").exists():
                continue
            variants = ["zero", "mean"]
            todo = [v for v in variants if not (out_dir / f"{v}@{group}__xu_{p}.npz").exists()]
            if not todo:
                continue
            # tracks scaled to a genome mean of 1 (the GM12878 tracks of the released model are
            # RPGC-like); ATAC substitutes DNase; the replicate ATAC tracks of HSPC are averaged
            def one(f):
                return (pyBigWig.open(str(f)), 1.0 / genome_mean(f))
            specs = {"DNaseI": [(b, sc / len(atac)) for b, sc in (one(f) for f in atac)],
                     "H3K27ac": [one(d / "H3K27ac.bigwig")], "H3K27me3": [one(d / "H3K27me3.bigwig")],
                     "CTCF": [one(d / "CTCF.bigwig")]}
            res = {v: {} for v in todo}
            wins = sorted(read_windows(p))
            wins = wins[::max(1, len(wins) // limit)][:limit] if limit else wins
            t0, cur, ct = time.time(), None, None
            NB = 200
            for k, (c, b0) in enumerate(wins):
                if c != cur:
                    ct, cur = ChromTracks(specs, c), c
                bin0 = b0 - (NB - WIN) // 2
                s0 = bin0 - EP_WIN // 2 // 100
                x = np.stack([ct.cut("DNaseI", s0 * 100, 34000), ct.cut("H3K27ac", s0 * 100, 34000),
                              np.zeros(34000, np.float32), ct.cut("H3K27me3", s0 * 100, 34000),
                              ct.cut("CTCF", s0 * 100, 34000)])
                for v in todo:
                    x[2] = 0.0 if v == "zero" else 1.0
                    C = epiphany_block(net, dev, x, s0, bin0, NB)
                    L = log_to_final(counts_to_log_oe(C, 99), 99)
                    res[v][win_name(c, b0)] = sample_grid(L, RES, bin0 * RES, b0).astype(np.float32)
                if k % 200 == 0:
                    log(f"epiphany {p} {group}: {k}/{len(wins)} windows, {time.time() - t0:.0f} s")
            for v in todo:
                np.savez_compressed(out_dir / f"{v}@{group}__xu_{p}.npz", **res[v])
            log(f"epiphany {p} {group}: written {todo}")


# ===================================================================== Akita
def run_akita(limit):
    """Akita (released model, GM12878 head = target index 2), sequence only: the 1 048 576 bp sequence
    centred on each window, forward and reverse-complement averaged. The prediction does not depend on
    the patient, so every distinct window is predicted once. 448 central 2048 bp bins."""
    import json
    from basenji import seqnn
    d = GALAXY_MODELS / "akita"
    params = json.load(open(d / "params.json"))
    m = seqnn.SeqNN(params["model"])
    m.restore(str(d / "model_best.h5"))
    wins = sorted({w for p in PATIENTS for w in read_windows(p)})
    wins = wins[::max(1, len(wins) // limit)][:limit] if limit else wins
    out_f = OUTD / "akita" / "akita.npz"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    L_SEQ, NB, HEAD = 1048576, 448, 2
    iu = np.triu_indices(NB, 2)
    res, cur, seq, t0 = {}, None, None, time.time()
    for k, (c, b0) in enumerate(wins):
        if c != cur:
            seq, cur = hg38_seq(c), c
        centre = (b0 + WIN // 2) * RES
        s0 = centre - L_SEQ // 2
        codes = np.full(L_SEQ, 4, np.uint8)
        a, b = max(s0, 0), min(s0 + L_SEQ, len(seq))
        codes[a - s0:b - s0] = seq[a:b]
        oh = np.zeros((L_SEQ, 4), np.float32)           # codes a0 t1 c2 g3 -> columns A C G T; n = 0
        for code, col in ((0, 0), (1, 3), (2, 1), (3, 2)):
            oh[codes == code, col] = 1
        x = np.stack([oh, oh[::-1, ::-1]])
        y = m.model.predict(x, verbose=0)[..., HEAD]        # (2, 99681)
        mats = []
        for r in range(2):
            M = np.full((NB, NB), np.nan)
            M[iu] = y[r]
            M = np.where(np.isfinite(M), M, M.T)
            mats.append(M if r == 0 else M[::-1, ::-1])
        Mm = np.nanmean(mats, 0) * np.log(2.0)            # log2 O/E of Akita's target -> natural log
        Lf = log_to_final(Mm, NB)
        res[win_name(c, b0)] = sample_grid(Lf, 2048, centre - NB * 2048 // 2, b0).astype(np.float16)
        if k % 500 == 0:
            log(f"akita {k}/{len(wins)} windows, {time.time() - t0:.0f} s")
    np.savez_compressed(out_f, **res)
    log(f"akita written {len(res)} windows")


# ===================================================================== scoring of all maps
KEEP = ["item", "patient", "kind", "set", "pair_id", "chrom", "a_lo", "a_hi", "b_lo", "b_hi", "pair_type", "win",
        "win_bin", "n_other", "cls", "measured", "measured#px", "reference", "reference#px", "hspc", "hspc#px",
        "aml_refin", "aml_refin#px", "aml_diff", "aml_diff#px", "aml_diffhspc", "aml_diffhspc#px",
        "base_refin@aml", "base_refin@aml#px"]


def sources(p: str) -> dict[str, tuple]:
    """name -> ('npz', path) maps of one patient; '@hspc' = the same model on the HSPC CD34+ inputs."""
    S = {}
    for lab, _, _ in NOAML:
        for grp, suf in (("patient", ""), ("hspc", "@hspc")):
            f = OUTD / "ours" / f"{lab}@{grp}__xu_{p}" / "marks.npz"
            if f.exists():
                S[lab + suf] = ("npz", f)
    for tool, variants in (("corigami", ("q", "raw")), ("epiphany", ("zero", "mean"))):
        for v in variants:
            for grp, suf in (("patient", ""), ("hspc", "@hspc")):
                f = OUTD / tool / f"{v}@{grp}__xu_{p}.npz"
                if f.exists():
                    S[f"{tool}_{v}" + suf] = ("own", f)
    f = OUTD / "akita" / "akita.npz"
    if f.exists():
        S["akita"] = ("own", f)
    return S


def stage_score(only) -> None:
    import cooler
    from aml_eval import fetch_region, pred_matrix
    from aml_loop_detection import enrich_maps
    from hicmultiome.bioscore import BioParams
    import multicell_expanded as mx
    prm = BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4)
    T = pd.read_csv(SRC, dtype={"patient": str})
    T["patient"] = T.patient.map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)
    c12 = cooler.Cooler(str(mx.consensus12(RES)))
    for p in PATIENTS:
        if only and p not in only:
            continue
        rows = []
        g = T[T.patient == p]
        S = sources(p)
        zs = {n: np.load(f) for n, (_, f) in S.items()}
        log(f"score {p}: {len(g)} objects; maps {sorted(S)}")
        cache = {}
        for win, gw in g.groupby("win"):
            maps = {}
            M12 = fetch_region(c12, win, cache)
            if M12 is not None and M12.shape == (WIN, WIN):
                maps["consensus12"] = M12
            for n, z in zs.items():
                if win in z.files:
                    A = z[win].astype(np.float64)
                    maps[n] = pred_matrix(A) if S[n][0] == "npz" else A
            enr = {n: enrich_maps(M, prm) for n, M in maps.items() if M.shape == (WIN, WIN)}
            for r in gw.itertuples():
                ws = r.win_bin
                rec = {"item": r.item}
                if r.kind == "loop":
                    i, j = r.a_lo - ws, r.b_lo - ws
                    if not (1 <= i < WIN - 1 and 1 <= j < WIN - 1):
                        continue
                    for n, E in enr.items():
                        blk = E[i - 1:i + 2, j - 1:j + 2]
                        rec[n] = float(np.nanmax(blk)) if np.isfinite(blk).any() else np.nan
                        rec[n + "#px"] = float(E[i, j])
                else:
                    a0, a1, b0, b1 = r.a_lo - ws, r.a_hi - ws, r.b_lo - ws, r.b_hi - ws
                    if min(a0, b0) < 0 or max(a1, b1) > WIN:
                        continue
                    for n, M in maps.items():
                        band = M[a0:a1, b0:b1]
                        rec[n] = float(np.nanmean(band)) if np.isfinite(band).any() else np.nan
                rows.append(rec)
        R = pd.DataFrame(rows)
        D = T[[c for c in KEEP if c in T]].merge(R, on="item", how="inner")
        PARTS.mkdir(parents=True, exist_ok=True)
        D.to_csv(PARTS / f"scores_{p}.csv.gz", index=False, float_format="%.5f")
        log(f"scored {p}: {len(D)} of {len(g)} objects")


def stage_merge() -> None:
    """scores_<patient>.csv.gz of every patient -> results/aml_competitors_loops/scores.csv.gz"""
    D = pd.concat([pd.read_csv(PARTS / f"scores_{p}.csv.gz", dtype={"patient": str}) for p in PATIENTS],
                  ignore_index=True)
    D.to_csv(RES_OUT / "scores.csv.gz", index=False, float_format="%.5f")
    log(f"merged {len(D)} objects, {D.patient.nunique()} patients")


# ===================================================================== statistics
BLOCKS = {
    "reference": ["measured", "reference", "consensus12"],
    "no AML in training": ["base_refin@aml", "c12_refin", "c12_refres", "c12_diffcells"],
    "competitors (published models, no AML in training)": ["corigami_q", "corigami_raw", "epiphany_zero",
                                                           "epiphany_mean", "akita"],
    "trained with AML patients (not comparable to competitors)": ["aml_refin", "aml_diff", "aml_diffhspc"],
}
LABEL = {"measured": "truth (measured)", "reference": "consensus (AML reference map)",
         "consensus12": "consensus12 (12 cell lines, no AML)",
         "base_refin@aml": "12-cell model, AML reference as input",
         "c12_refin": "12-cell model (refin), consensus12",
         "c12_refres": "12-cell model (refres), consensus12",
         "c12_diffcells": "12-cell differential-channel model, consensus12",
         "corigami_q": "C.Origami (CTCF, ATAC quantile-matched to IMR-90)",
         "corigami_raw": "C.Origami (CTCF, ATAC as given)",
         "epiphany_zero": "Epiphany (ATAC for DNase, H3K4me3 = 0)",
         "epiphany_mean": "Epiphany (ATAC for DNase, H3K4me3 = genome mean)",
         "akita": "Akita (sequence only, GM12878 head)",
         "aml_refin": "AML-trained, reference input (aml_refin)",
         "aml_diff": "AML-trained, difference channels (aml_diff)",
         "aml_diffhspc": "AML-trained, difference channels vs HSPC (aml_diffhspc)"}
NBOOT_ = 500
SUF = ""
CLASS_DEF = [("all", None), ("private", ("cls", "private")), ("recurrent", ("cls", "recurrent")),
             ("shared1", ("cls", "shared1")), ("PE", ("pair_type", "PE")), ("PS", ("pair_type", "PS")),
             ("enh", ("pair_type", "enh")), ("rep", ("pair_type", "rep")), ("other", ("pair_type", "other"))]


def _auroc(a, b):
    from aml_loop_detection import auroc
    return auroc(a, b)


def match_refrand(D: pd.DataFrame, kind: str, sel_pos: pd.Series) -> pd.DataFrame:
    """criteria2 matching (aml_loop_detection.match_controls): the positives of `sel_pos` to B_rand
    candidates on AML reference, HSPC enrichment (0.15 each), log10 distance (0.10), orientation."""
    from aml_loop_detection import match_controls
    ren = {"_ref": "reference", "_hspc": "hspc"} if "_ref" in D else {}
    keep = ["patient", "chrom", "a_lo", "b_lo", "dist_log10", "_orient"] + list(ren)
    X = D[(D.kind == kind) & (D.set.isin(["pos", "B_rand"]))][keep + ["set"]].rename(columns=ren)
    if not ren:
        X["reference"], X["hspc"] = D.loc[X.index, "reference"], D.loc[X.index, "hspc"]
    P = X[(X.set == "pos") & sel_pos.reindex(X.index, fill_value=False)]
    C = X[X.set == "B_rand"]
    return match_controls(P, C)


def to_gain(D: pd.DataFrame) -> pd.DataFrame:
    """Score = contrast with the healthy cell: model minus the same model on the HSPC CD34+ inputs
    (12-cell models, C.Origami, Epiphany); truth, consensus, consensus12 and the models trained with
    AML minus the measured HSPC consensus (the convention of aml_loop_detection). Akita has no
    patient-dependent output and drops out."""
    D = D.copy()
    D["_ref"], D["_hspc"] = D["reference"], D["hspc"]
    for m in [m for b in BLOCKS.values() for m in b if m in D]:
        if m + "@hspc" in D:
            D[m] = D[m] - D[m + "@hspc"]
        elif m != "akita":
            D[m] = D[m] - D["_hspc"]
        else:
            D[m] = np.nan
    return D


def stage_stats(nboot: int, scores: str = "", score: str = "level") -> None:
    global SUF
    SUF = "" if score == "level" else "_gain"
    D = pd.read_csv(scores or RES_OUT / "scores.csv.gz", dtype={"patient": str})
    F = pd.read_csv(REPO / "results/aml_loop_classifier/features.csv.gz",
                    usecols=["patient", "kind", "set", "pair_id", "_orient", "dist_log10"], dtype={"patient": str})
    F["patient"] = F.patient.map(lambda x: x.zfill(3) if x.isdigit() and len(x) < 3 else x)
    D = D.merge(F, on=["patient", "kind", "set", "pair_id"], how="left").reset_index(drop=True)
    if score == "gain":
        D = to_gain(D)
    rng = np.random.default_rng(7)
    models = [m for b in BLOCKS.values() for m in b if m in D]
    block_of = {m: b for b, ms in BLOCKS.items() for m in ms}
    pats = sorted(D.patient.unique())
    # subsets of objects: every model of the comparison is evaluated on the same rows
    sub = {"all objects (each model on its own coverage)": pd.Series(True, index=D.index),
           "C.Origami patients": D.patient.isin(CO_PATIENTS)}
    if "akita" in D:
        sub["Akita coverage (object inside the 917 kb window)"] = D.akita.notna()
    if "epiphany_zero" in D:
        sub["Epiphany coverage (reach 990 kb)"] = D.epiphany_zero.notna()
    com = D.patient.isin(CO_PATIENTS)
    for m in ("akita", "epiphany_zero", "corigami_q"):
        if m in D:
            com &= D[m].notna()
    sub["objects scored by all competitors"] = com

    # ---- coverage
    cov = []
    for m in models:
        for kind in ("loop", "stripe"):
            d = D[(D.kind == kind) & (D.set == "pos")]
            cov.append({"model": m, "kind": kind, "n_pos": len(d), "n_pos_scored": int(d[m].notna().sum()),
                        "n_patients_with_prediction": int(d.groupby("patient")[m].apply(lambda x: x.notna().any()).sum())})
    pd.DataFrame(cov).to_csv(RES_OUT / f"coverage{SUF}.csv", index=False)

    # ---- (1) detection rate
    def rate(pos, ctl):
        c = ctl[np.isfinite(ctl)]
        pp = pos[np.isfinite(pos)]
        if len(c) < 20 or len(pp) < 5:
            return np.nan, np.nan, np.nan
        thr = np.percentile(c, 90)
        return float((pp > thr).mean()), float(thr), float((c > thr).mean())

    rows = []
    for sname, mask in sub.items():
        for kind in ("loop", "stripe"):
            dk = D[mask & (D.kind == kind)]
            for cname, cdef in CLASS_DEF:
                pos = dk[dk.set == "pos"]
                if cdef:
                    pos = pos[pos[cdef[0]] == cdef[1]]
                ctl = dk[dk.set == "B_rand"]
                if len(pos) < 10:
                    continue
                donors = sorted(set(pos.patient))
                for m in models:
                    if m not in D:
                        continue
                    r0, thr, fpr = rate(pos[m].values, ctl[m].values)
                    if not np.isfinite(r0):
                        continue
                    pg = {d: pos[pos.patient == d][m].values for d in donors}
                    cg = {d: ctl[ctl.patient == d][m].values for d in sorted(set(ctl.patient))}
                    bs = []
                    for _ in range(nboot):
                        pick = rng.choice(donors, len(donors))
                        a = np.concatenate([pg[d] for d in pick])
                        b = np.concatenate([cg[d] for d in pick if d in cg])
                        bs.append(rate(a, b)[0])
                    bs = np.array([x for x in bs if np.isfinite(x)])
                    rows.append({"subset": sname, "kind": kind, "class": cname, "model": m, "block": block_of[m],
                                 "n_pos": int(np.isfinite(pos[m]).sum()), "n_B_rand": int(np.isfinite(ctl[m]).sum()),
                                 "n_donors": len(donors), "threshold_p90_B_rand": thr, "detection_rate": r0,
                                 "lo": np.percentile(bs, 2.5) if len(bs) > 20 else np.nan,
                                 "hi": np.percentile(bs, 97.5) if len(bs) > 20 else np.nan,
                                 "false_positive_rate": fpr})
    pd.DataFrame(rows).to_csv(RES_OUT / f"detection_rates{SUF}.csv", index=False, float_format="%.4f")
    log("detection rates written")

    # ---- (2) AUROC vs B_rand and vs the A_refrand matched controls
    arows, mq = [], []
    for sname in ("all objects (each model on its own coverage)", "objects scored by all competitors"):
        mask = sub[sname]
        Dm = D[mask].reset_index(drop=True) if sname.startswith("objects") else D
        for kind in ("loop", "stripe"):
            for cname, cdef in CLASS_DEF:
                sp = (Dm[cdef[0]] == cdef[1]) if cdef else pd.Series(True, index=Dm.index)
                pos = Dm[(Dm.kind == kind) & (Dm.set == "pos") & sp]
                if len(pos) < 20:
                    continue
                allctl = Dm[(Dm.kind == kind) & (Dm.set == "B_rand")]
                mt = match_refrand(Dm, kind, sp)
                mq.append({"subset": sname, "kind": kind, "class": cname, "n_pos": len(pos), "n_matched": len(mt),
                           "frac_matched": len(mt) / len(pos)})
                sets = {"B_rand": (pos.index.values, allctl.index.values, None),
                        "A_refrand": (mt.pos.values, mt.ctl.values, "matched")}
                for cn, (pi, ci, _) in sets.items():
                    if len(pi) < 20 or len(ci) < 20:
                        continue
                    donors = sorted(set(Dm.loc[pi, "patient"]))
                    pby = {d: pi[Dm.loc[pi, "patient"].values == d] for d in donors}
                    cby = {d: ci[Dm.loc[ci, "patient"].values == d] for d in donors}
                    for m in models:
                        v = Dm[m].values
                        a0 = _auroc(v[pi], v[ci])
                        if not np.isfinite(a0):
                            continue
                        bs = []
                        for _ in range(nboot):
                            pick = rng.choice(donors, len(donors))
                            bs.append(_auroc(v[np.concatenate([pby[d] for d in pick])],
                                             v[np.concatenate([cby[d] for d in pick])]))
                        bs = np.array([x for x in bs if np.isfinite(x)])
                        arows.append({"subset": sname, "kind": kind, "class": cname, "control": cn, "model": m,
                                      "block": block_of[m], "n_pos": len(pi), "n_ctl": len(ci), "n_donors": len(donors),
                                      "auroc": a0, "lo": np.percentile(bs, 2.5) if len(bs) > 20 else np.nan,
                                      "hi": np.percentile(bs, 97.5) if len(bs) > 20 else np.nan})
    pd.DataFrame(arows).to_csv(RES_OUT / f"auroc{SUF}.csv", index=False, float_format="%.4f")
    pd.DataFrame(mq).to_csv(RES_OUT / f"match_quality{SUF}.csv", index=False, float_format="%.4f")
    log("AUROC written")

    # ---- (3) Spearman of the predicted with the measured enrichment among the AML-specific objects
    from scipy.stats import spearmanr
    srows = []
    for kind in ("loop", "stripe"):
        for cname in ("all", "private", "recurrent"):
            cdef = dict(CLASS_DEF)[cname]
            pos = D[(D.kind == kind) & (D.set == "pos")]
            if cdef:
                pos = pos[pos[cdef[0]] == cdef[1]]
            donors = sorted(set(pos.patient))
            for m in models:
                if m == "measured":
                    continue
                hs = m + "@hspc"
                tgt = {"level": (pos[m], pos.measured)}
                if score == "gain":
                    pass
                elif hs in pos:
                    tgt["gain_vs_own_hspc"] = (pos[m] - pos[hs], pos.measured - pos.hspc)
                elif m in ("reference", "consensus12"):
                    tgt["gain_vs_own_hspc"] = (pos[m] - pos.hspc, pos.measured - pos.hspc)
                elif m.startswith("aml_") or m == "base_refin@aml":
                    tgt["gain_vs_own_hspc"] = (pos[m] - pos.hspc, pos.measured - pos.hspc)
                for tn, (x, y) in tgt.items():
                    ok = np.isfinite(x.values) & np.isfinite(y.values)
                    if ok.sum() < 30 or np.nanstd(x.values[ok]) == 0:
                        continue
                    r0 = spearmanr(x.values[ok], y.values[ok])[0]
                    byd = {d: (x.values[ok][pos.patient.values[ok] == d], y.values[ok][pos.patient.values[ok] == d])
                           for d in donors}
                    bs = []
                    for _ in range(nboot):
                        pick = rng.choice(donors, len(donors))
                        a = np.concatenate([byd[d][0] for d in pick])
                        b = np.concatenate([byd[d][1] for d in pick])
                        bs.append(spearmanr(a, b)[0] if len(a) > 10 and np.std(a) > 0 else np.nan)
                    bs = np.array([t for t in bs if np.isfinite(t)])
                    per = [spearmanr(*byd[d])[0] for d in donors if len(byd[d][0]) > 10 and np.std(byd[d][0]) > 0]
                    srows.append({"kind": kind, "class": cname, "model": m, "block": block_of[m], "target": tn,
                                  "n": int(ok.sum()), "spearman": r0,
                                  "lo": np.percentile(bs, 2.5) if len(bs) > 20 else np.nan,
                                  "hi": np.percentile(bs, 97.5) if len(bs) > 20 else np.nan,
                                  "median_donor_spearman": float(np.median(per)) if per else np.nan})
    pd.DataFrame(srows).to_csv(RES_OUT / f"spearman{SUF}.csv", index=False, float_format="%.4f")
    log("Spearman written")


def stage_summary(suf: str = "") -> None:
    global SUF
    SUF = suf
    """Wide tables (rows = models, columns = loops / stripes by class) of the detection rate, the
    AUROC against B_rand and against A_refrand, and the Spearman correlation; markdown to stdout."""
    det = pd.read_csv(RES_OUT / f"detection_rates{SUF}.csv")
    au = pd.read_csv(RES_OUT / f"auroc{SUF}.csv")
    sp = pd.read_csv(RES_OUT / f"spearman{SUF}.csv")
    cols = [(k, c) for k in ("loop", "stripe") for c in ("all", "private", "recurrent")]
    order = [m for b in BLOCKS.values() for m in b]
    sub0 = "all objects (each model on its own coverage)"

    def wide(df, val, filt):
        d = df[filt]
        out = pd.DataFrame(index=[m for m in order if m in set(d.model)])
        for k, c in cols:
            x = d[(d.kind == k) & (d["class"] == c)].set_index("model")
            out[f"{k} {c}"] = [("%.2f (%.2f-%.2f)" % (x.loc[m, val], x.loc[m, "lo"], x.loc[m, "hi"]))
                               if m in x.index and np.isfinite(x.loc[m, val]) else "n/a" for m in out.index]
        out.insert(0, "block", [next(b for b, ms in BLOCKS.items() if m in ms) for m in out.index])
        return out
    tabs = {"detection_rate": wide(det, "detection_rate", det.subset == sub0),
            "auroc_vs_B_rand": wide(au, "auroc", (au.subset == sub0) & (au.control == "B_rand")),
            "auroc_vs_A_refrand": wide(au, "auroc", (au.subset == sub0) & (au.control == "A_refrand")),
            "spearman_level_vs_measured": wide(sp, "spearman", sp.target == "level")}
    for sname in det.subset.unique():
        if sname != sub0:
            tabs["detection_rate | " + sname] = wide(det, "detection_rate", det.subset == sname)
    with open(RES_OUT / f"summary_tables{suf}.md", "w") as fh:
        for n, t in tabs.items():
            t.to_csv(RES_OUT / ("summary_" + "".join(ch if ch.isalnum() else "_" for ch in n)[:60].strip("_") + suf + ".csv"))
            fh.write(f"\n### {n}\n\n")
            fh.write("| model | block | " + " | ".join(t.columns[1:]) + " |\n|---|---|" + "---|" * (len(t.columns) - 1) + "\n")
            for m, r in t.iterrows():
                fh.write(f"| {LABEL.get(m, m)} | {r['block']} | " + " | ".join(r.values[1:]) + " |\n")
    print(open(RES_OUT / f"summary_tables{suf}.md").read())


def stage_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    det = pd.read_csv(RES_OUT / f"detection_rates{SUF}.csv")
    au = pd.read_csv(RES_OUT / f"auroc{SUF}.csv")
    sub0 = "all objects (each model on its own coverage)"
    order = [m for b in BLOCKS.values() for m in b]
    colors = {"reference": "#555555", "no AML in training": "#1b6ca8",
              "competitors (published models, no AML in training)": "#c0522d",
              "trained with AML patients (not comparable to competitors)": "#9aa6b2"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for r, kind in enumerate(("loop", "stripe")):
        for c, cname in enumerate(("all", "private", "recurrent")):
            ax = axes[r, c]
            d = det[(det.subset == sub0) & (det.kind == kind) & (det["class"] == cname)].set_index("model")
            ms = [m for m in order if m in d.index]
            y = np.arange(len(ms))[::-1]
            ax.barh(y, d.loc[ms, "detection_rate"], color=[colors[d.loc[m, "block"]] for m in ms])
            ax.errorbar(d.loc[ms, "detection_rate"], y,
                        xerr=[d.loc[ms, "detection_rate"] - d.loc[ms, "lo"], d.loc[ms, "hi"] - d.loc[ms, "detection_rate"]],
                        fmt="none", ecolor="k", lw=0.8)
            ax.axvline(0.10, color="k", ls=":", lw=1)
            ax.set_yticks(y)
            ax.set_yticklabels([LABEL[m] for m in ms] if c == 0 else [], fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_xlabel("share above the 90th percentile of B_rand")
            ax.set_title(f"{kind}s, {cname} (n = {int(d['n_pos'].max())})", fontsize=10)
    fig.suptitle("Detection of AML-specific loops and stripes (10 held-out donors); dotted line = 10% false-positive rate by construction",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RES_OUT / "fig_aml_competitors_loops.png", dpi=150)
    log("figure written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--nboot", type=int, default=NBOOT_)
    ap.add_argument("--score", choices=["level", "gain"], default="level",
                    help="stats / summary: predicted enrichment (level) or its contrast with the healthy HSPC prediction (gain)")
    ap.add_argument("--scores", default="", help="stats: scores table (default results/aml_competitors_loops/scores.csv.gz)")
    ap.add_argument("--limit", type=int, default=0, help="competitor stages: first N windows per patient (test)")
    a = ap.parse_args()
    for st in a.stage:
        if st == "ours":
            stage_ours(a.gpu, a.threads, a.only)
        elif st == "corigami":
            run_corigami("cuda", a.only, a.limit)
        elif st == "epiphany":
            run_epiphany(a.only, a.limit)
        elif st == "akita":
            run_akita(a.limit)
        elif st == "score":
            stage_score(a.only)
        elif st == "merge":
            stage_merge()
        elif st == "stats":
            stage_stats(a.nboot, a.scores, a.score)
        elif st == "figure":
            stage_figure()
        elif st == "summary":
            stage_summary("" if a.score == "level" else "_gain")


if __name__ == "__main__":
    main()
