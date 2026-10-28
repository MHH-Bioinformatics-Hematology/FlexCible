"""Transfer of the human-trained models to mouse (mm10), no retraining, no tuning (registry section 75).

Windows are the 5 938 test/valid/train tiles of Akita v2 (mm10, 1 310 720 bp, fold labelled); the scored
region is the central 1 048 576 bp (131 072 bp cropped each side), whole 10 kb (25 kb) bins inside it. A
window is used when the model's input span (192 bins at 10 kb, 80 bins at 25 kb) lies inside the
chromosome and contains no assembly gap (N run >= 1 kb). The Akita v2 model whose test fold is the tile's
fold predicts the tile (mouse head), so the tile was never seen by it.

Stages (each resumes; heavy ones are run from the shell, see the registry section):

    python scripts/mouse_transfer.py regions
    python scripts/mouse_transfer.py consensus             # leave-one-out mouse consensus O/E maps
    python scripts/mouse_transfer.py views                 # config + checkpoint views of the human models
    python scripts/mouse_transfer.py predict --res 10000 25000 --gpu 0 [--variants main shuf xm]
    ~/miniconda3/envs/basenji/bin/python scripts/mouse_transfer.py akita [--part i --nparts n]
    python scripts/mouse_transfer.py score --procs 6
    python scripts/mouse_transfer.py ceiling
    python scripts/mouse_transfer.py summary
    python scripts/mouse_transfer.py figure

Inputs the human models read in mouse: 7 ChIP tracks (CTCF, H3K27ac, H3K27me3, H3K36me3, H3K4me1,
H3K4me3, H3K9me3; plus RAD21 and SMC3 in the optional "avail" set of CH12), GC and CpG per bin of mm10, and
the four motif planes (CTCF MA0139.2, YY1 MA0095.3, ZNF143 MA0088.2, MAZ MA1522.1) scanned with the human
PWMs on mm10 sequence. RAD21, SMC3, H3K4me2, H3K79me2, H3K9ac, H4K20me1 are absent (absent value).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
VM = HM / "validation_mouse"
MR = HM / "runs/multicell_reference"
OEH = HM / "crosscell_oe"
RESULTS = REPO / "results/mouse_transfer"
LOCAL = Path(os.environ.get("MT_CACHE", str(Path.home() / "scratch/mt_cache")))

TILE, PAD = 1_310_720, 131_072
CHRS = [f"chr{i}" for i in range(1, 20)]
CHRX = ["chrX"]
SIZE = {10000: 192, 25000: 80}
CORE7 = ["CTCF", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]
ALL13 = ["CTCF", "RAD21", "SMC3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
         "H3K79me2", "H3K9ac", "H3K9me3", "H4K20me1"]
POOL = ["mesc", "ch12", "heart"]                     # samples with a Hi-C map; the consensus pool
ASSAY = {"mesc": "hic", "ch12": "hic", "heart": "intact"}
# variant of the marks: (label of the marks folder, Hi-C sample)
MARKS = {"mesc": "mm10_mesc", "ch12": "mm10_ch12", "heart": "mm10_heart", "mesc_b4": "mm10_mesc_b4"}
XM_NEXT = {"mesc": "heart", "ch12": "mesc", "heart": "ch12"}      # marks of a different mouse sample
SETS = {"all7": CORE7, "ctcf_k27ac": ["CTCF", "H3K27ac"], "ctcf": ["CTCF"], "none": [],
        "avail": CORE7 + ["RAD21", "SMC3"]}
MODELS = {"honest": "all12honest_drop", "refin": "all12refin_drop", "refres": "all12refres_drop",
          "diff": "all12diffcells_refres_drop"}
METRICS = ["pearson", "scc", "insulation_corr", "boundary_f1", "loop_enrich_corr", "loop_apa_ratio",
           "compartment_corr", "eigenvector_abs_corr", "stripe_corr"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def kb(res):
    return f"{res // 1000}kb"


def oe_cool(cell, res):
    p = VM / "oe" / f"{cell}_proper_{kb(res)}.cool"
    q = LOCAL / p.name
    return q if q.exists() else p


def loo_cool(cell, res):
    c = "mesc" if cell == "mesc_b4" else cell
    p = VM / "oe" / f"consensus_loo_{c}_proper_{kb(res)}.cool"
    q = LOCAL / p.name
    return q if q.exists() else p


# ================================================================== regions
def crop_bins(start, end, res):
    c0, c1 = start + PAD, end - PAD
    return -(-c0 // res), c1 // res


def gap_runs(chrom, fasta, min_len=1000):
    """(start, end) of N runs >= min_len of a chromosome."""
    from pyfaidx import Fasta
    fa = Fasta(str(fasta), rebuild=False)
    s = np.frombuffer(fa[chrom][:].seq.encode("ascii"), dtype=np.uint8)
    isn = (s == ord("N")) | (s == ord("n"))
    if not isn.any():
        return [], len(s)
    d = np.diff(np.concatenate([[0], isn.view(np.int8), [0]]))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(st, en) if b - a >= min_len], len(s)


def blacklist_by_chrom():
    f = VM / "ref/mm10-blacklist.v2.bed.gz"
    b = pd.read_csv(f, sep="\t", header=None, names=["chrom", "start", "end"], usecols=[0, 1, 2])
    return {c: g[["start", "end"]].values for c, g in b.groupby("chrom")}


def stage_regions(a):
    d = VM / "regions"
    d.mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(VM / "akita_v2/sequences.bed", sep="\t", header=None,
                        names=["chrom", "start", "end", "fold"])
    tiles = tiles[tiles.chrom.isin(CHRS + CHRX)].copy()
    fasta = VM / "ref/mm10.fa"
    gaps, clen = {}, {}
    for c in sorted(tiles.chrom.unique()):
        gaps[c], clen[c] = gap_runs(c, fasta)
    rows = []
    for res in (10000, 25000):
        for r in tiles.itertuples():
            lo, hi = crop_bins(r.start, r.end, res)
            a0, a1 = lo * res, hi * res
            centre = (a0 + a1) // 2 // res
            st = centre - SIZE[res] // 2
            span = (st * res, (st + SIZE[res]) * res)
            nb = -(-clen[r.chrom] // res)
            ok = st >= 0 and st + SIZE[res] <= nb and span[1] <= clen[r.chrom]
            gap = any(g0 < span[1] and g1 > span[0] for g0, g1 in gaps[r.chrom])
            rows.append({"res": res, "chrom": r.chrom, "tile_start": r.start, "tile_end": r.end,
                         "fold": int(str(r.fold).replace("fold", "")), "b_lo": lo, "b_hi": hi,
                         "a_start": a0, "a_end": a1, "span_start": span[0], "span_end": span[1],
                         "in_chrom": bool(ok), "gap": bool(gap), "ok": bool(ok and not gap),
                         "set": "x" if r.chrom == "chrX" else "auto",
                         "key": f"{r.chrom}:{a0}-{a1}", "seq_key": f"{r.chrom}:{r.start}"})
    R = pd.DataFrame(rows)
    R.to_csv(d / "windows.tsv", sep="\t", index=False)
    for res in (10000, 25000):
        S = R[(R.res == res) & R.ok]
        for nm, sub in (("all", S), ("auto", S[S.set == "auto"]), ("x", S[S.set == "x"])):
            sub[["chrom", "a_start", "a_end"]].to_csv(d / f"crop_{kb(res)}_{nm}.tsv", sep="\t",
                                                      header=False, index=False)
        log(f"{kb(res)}: {len(S)} windows usable of {int((R.res == res).sum())} tiles "
            f"(gap {int(R[(R.res == res)].gap.sum())}, off-chromosome {int((~R[(R.res == res)].in_chrom).sum())}); "
            f"autosomes {int((S.set == 'auto').sum())}, X {int((S.set == 'x').sum())}")


def load_windows(res, sets=("auto",)):
    R = pd.read_csv(VM / "regions/windows.tsv", sep="\t")
    return R[(R.res == res) & R.ok & R.set.isin(sets)].reset_index(drop=True)


# ================================================================== consensus
def stage_consensus(a):
    import cooler
    import h5py
    for res in (10000, 25000):
        k = kb(res)
        files = {c: VM / "oe" / f"{c}_proper_{k}.cool" for c in POOL}
        if not all(f.exists() for f in files.values()):
            log(f"[skip] {k}: not all pool maps exist")
            continue
        clrs = {c: cooler.Cooler(str(f)) for c, f in files.items()}
        grid = clrs[POOL[0]]
        bins = grid.bins()[:][["chrom", "start", "end"]]
        offs = {}
        for c, cl in clrs.items():
            with h5py.File(cl.filename, "r") as h:
                offs[c] = h["indexes/bin1_offset"][:]
        chroms = [c for c in grid.chromnames if all(c in cl.chromnames for cl in clrs.values())]
        outs = {"full": VM / "oe" / f"consensus{len(POOL)}_proper_{k}.cool"}
        outs.update({c: VM / "oe" / f"consensus_loo_{c}_proper_{k}.cool" for c in POOL})
        if all(p.exists() for p in outs.values()):
            continue

        def gen(name):
            for chrom in chroms:
                nb = int(grid.extent(chrom)[1] - grid.offset(chrom))
                go = int(grid.offset(chrom))
                vals, b1, b2 = [], None, None
                for c in POOL:
                    cl = clrs[c]
                    fo = int(cl.offset(chrom))
                    px = cl.pixels()[int(offs[c][fo]):int(offs[c][fo + nb])]
                    i, j = px["bin1_id"].values - fo, px["bin2_id"].values - fo
                    if b1 is None:
                        b1, b2 = i, j
                    elif not (np.array_equal(i, b1) and np.array_equal(j, b2)):
                        raise SystemExit(f"{c} {chrom}: pixel table differs")
                    vals.append(px["count"].values.astype(np.float64))
                V = np.stack(vals)
                use = [i for i, c in enumerate(POOL) if name == "full" or c != name]
                m = V[use].mean(axis=0).astype(np.float32)
                yield pd.DataFrame({"bin1_id": b1 + go, "bin2_id": b2 + go, "count": m})

        for name, path in outs.items():
            if path.exists():
                continue
            part = path.with_name(path.name + ".part")
            part.unlink(missing_ok=True)
            cooler.create_cooler(str(part), bins=bins, pixels=gen(name), dtypes={"count": np.float32},
                                 ordered=True)
            part.rename(path)
            cells = POOL if name == "full" else [c for c in POOL if c != name]
            Path(str(path) + ".json").write_text(json.dumps({
                "res": res, "cells": cells, "inputs": [str(files[c]) for c in cells],
                "method": "pixel-wise mean of oe_proper.py maps (identical pixel tables); a mouse, "
                          "species-matched consensus of the other pool samples (leave-one-out)"}, indent=1))
            log(f"-> {path.name} (cells {cells})")


# ================================================================== config + checkpoint views
def view_dir(model, res):
    return VM / "views" / f"{MODELS.get(model, model)}_{kb(res)}"


def stage_views(a):
    import yaml
    for res in (10000, 25000):
        for m, run in MODELS.items():
            src = MR / f"{run}_{kb(res)}" / "checkpoint-best"
            if not (src / "model.pt").exists():
                log(f"[skip] {run}_{kb(res)}: no checkpoint")
                continue
            dst = view_dir(m, res) / "checkpoint-best"
            dst.mkdir(parents=True, exist_ok=True)
            for fn in ("model.pt", "norm_tracks.json", "channels.json"):
                t = dst / fn
                if not t.exists():
                    shutil.copyfile(src / fn, t)
            nm = dst / "norm_matrix_mt_target.json"
            if not nm.exists():
                shutil.copyfile(src / "norm_matrix_gm12878.json", nm)       # the matrix normaliser of the first
            cfg = yaml.safe_load((src / "config.yaml").read_text())            # sample, as for unseen human cells
            first = dict(cfg["data"]["samples"][0])
            first["reference_cool"] = None
            refc = (lambda c: str(loo_cool(c, res))) if m != "honest" else (lambda c: None)
            tg = dict(first, name="mt_target", cool_path=str(oe_cool("heart", res)),
                      bigwig_dir=str(VM / MARKS["heart"]), missing_tracks=[], zeroed_tracks=[],
                      assay="hic", fasta=None, reference_cool=refc("heart"))
            samples = [tg]
            have = {}
            for c in POOL:
                d = VM / MARKS[c]
                have[c] = {p.stem for p in d.glob("*.bigwig")}
                samples.append(dict(first, name=c, cool_path=str(oe_cool(c, res)), bigwig_dir=str(d),
                                    missing_tracks=[t for t in ALL13 if t not in have[c]], zeroed_tracks=[],
                                    assay=ASSAY[c], fasta=None, reference_cool=refc(c)))
            cfg["data"]["samples"] = samples
            if cfg["data"].get("diff_marks"):
                cfg["data"]["diff_reference_sets"] = {"consensus": list(POOL)}
                cfg["data"]["diff_default_set"] = "consensus"
            cfg["sequence"]["fasta"] = str(VM / "ref/mm10.fa")
            cfg["sequence"]["cache_dir"] = str(VM / "seqcache")
            cfg["train"]["cache_dir"] = str(VM / "cfgcache" / f"{run}_{kb(res)}")
            cfg["train"]["output_dir"] = str(VM / "cfgcache" / f"{run}_{kb(res)}_out")
            (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            log(f"view {dst}")
    # untrained control: same architecture, default initialisation, same normalisers
    import torch
    import score_like_hicgan as sc
    from flexcible.config import Config
    res = 10000
    dst = view_dir("untrained", res) / "checkpoint-best"
    base = view_dir("honest", res) / "checkpoint-best"
    if (base / "config.yaml").exists() and not (dst / "model.pt").exists():
        dst.mkdir(parents=True, exist_ok=True)
        cfg = Config.load(base / "config.yaml")
        model, _ = sc.load_model(cfg, base, "cpu")
        torch.manual_seed(0)
        for mod in model.modules():
            if hasattr(mod, "reset_parameters") and mod is not model:
                mod.reset_parameters()
        torch.save(model.state_dict(), dst / "model.pt")
        for fn in ("norm_tracks.json", "channels.json", "norm_matrix_mt_target.json", "config.yaml"):
            t = dst / fn
            if not t.exists():
                shutil.copyfile(base / fn, t)
        log(f"view {dst} (untrained, seed 0)")


# ================================================================== predict
def claim(name, done):
    r = subprocess.run([sys.executable, str(REPO / "scripts/claim_job.py"), "claim", name,
                        "--done-path", str(done), "--cmd", f"mouse_transfer.py predict {name}"])
    return r.returncode


def pred_jobs(res, variants):
    """(model, variant, hic sample, marks sample, keep sets) of one resolution."""
    jobs = []
    for v in variants:
        if v == "main":
            for s in POOL:
                for m in MODELS:
                    sets = ["all7", "ctcf_k27ac", "ctcf", "none"] if m in ("honest", "refin") else ["all7"]
                    if s == "ch12" and m in ("honest", "refin"):
                        sets = sets + ["avail"]
                    jobs.append((m, "main", s, s, sets))
            jobs.append(("honest", "b4", "mesc", "mesc_b4", ["all7"]))
            jobs.append(("refin", "b4", "mesc", "mesc_b4", ["all7"]))
        elif v == "shuf":
            for s in POOL:
                for m in ("honest", "refin"):
                    jobs.append((m, "shuf", s, s, ["all7"]))
        elif v == "xm":
            for s in POOL:
                for m in ("honest", "refin"):
                    jobs.append((m, "xm", s, XM_NEXT[s], ["all7"]))
        elif v == "untrained" and res == 10000:
            for s in POOL:
                jobs.append(("untrained", "main", s, s, ["all7"]))
    return jobs


def stage_predict(a):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads))
    py = os.environ.get("HICMULTI_PY", sys.executable)
    regs_kind = {"auto": "auto", "all": "all"}[a.regions]
    for res in a.res:
        for m, var, hic, marks, sets in pred_jobs(res, a.variants):
            if a.only and hic not in a.only:
                continue
            if a.models and m not in a.models:
                continue
            ck = view_dir(m, res) / "checkpoint-best"
            if not (ck / "model.pt").exists():
                log(f"[skip] {m} {kb(res)}: no view")
                continue
            bw = VM / MARKS[marks]
            tgt = oe_cool(hic, res)
            out = VM / "pred" / kb(res) / (m if var == "main" else f"{m}_{var}") / hic
            done = out / "DONE"
            if done.exists():
                continue
            if not tgt.exists() or not bw.exists():
                log(f"[skip] {hic} {kb(res)}: target or marks missing")
                continue
            have = {p.stem for p in bw.glob("*.bigwig")}
            ks = [f"{n}={','.join(t for t in SETS[n] if t in have)}" for n in sets]
            regs = VM / "regions" / f"crop_{kb(res)}_{regs_kind}.tsv"
            job = f"mouse_transfer_{m}_{var}_{hic}_{kb(res)}"
            if claim(job, done) != 0:
                log(f"[claim] {job} skipped")
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = [py, "scripts/mouse_transfer_predict.py", "--config", str(ck / "config.yaml"),
                   "--checkpoint", str(ck), "--cell", hic, "--bigwig-dir", str(bw), "--target", str(tgt),
                   "--tta", "--assay", ASSAY[hic], "--binsize", str(res), "--device", a.device,
                   "--regions", str(regs), "--out-dir", str(out), "--save-pred", str(out / "{set}.npz"),
                   "--keep-sets", *ks]
            if m in ("refin", "refres", "diff"):
                cmd += ["--reference", str(loo_cool(hic, res))]
            if m == "diff":
                cmd += ["--diff-group", "cell"]
            e = dict(env)
            if var == "shuf":
                e["MT_SHUFFLE"] = "1"
            log(f"[run] {job}")
            with open(out / "predict.log", "w") as fh:
                fh.write("[cmd] " + " ".join(cmd) + "\n")
                fh.flush()
                r = subprocess.run(["nice", "-n", "10"] + cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=e)
            if r.returncode == 0 and all((out / f"{s}.npz").exists() for s in sets):
                done.write_text(time.strftime("%F %T"))
            else:
                log(f"[FAILED] {job} rc={r.returncode}")
                subprocess.run([sys.executable, str(REPO / "scripts/claim_job.py"), "release", job])


# ================================================================== akita v2 (mouse head)
AK_HEADS = {0: "Hsieh_mESC_uC", 1: "Bonev_ES", 2: "Bonev_CN", 4: "Bonev_NPC"}
CLIP, SIGMA = 2.0, 2.0


def log_to_final(L, reach):
    from scipy.ndimage import gaussian_filter
    n = L.shape[0]
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    off = np.zeros(n)
    for k in range(n):
        v = np.diagonal(L, k)
        v = v[np.isfinite(v)]
        off[k] = v.mean() if v.size else 0.0
    val = np.clip(L - off[d], -CLIP, CLIP)
    val = np.where((d <= reach) & np.isfinite(val), val, 0.0)
    sm = gaussian_filter(val, sigma=SIGMA, truncate=1.0, mode="nearest")
    return np.where(d <= reach, sm, np.nan)


def sample_grid(M, cres, x0_bp, b0, n, res):
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


def read_seq(fasta, fai, chrom, s0, n):
    """codes a0 c1 g2 t3 n4 of fasta[chrom][s0:s0+n] (zero padding outside = N)."""
    ent = fai[chrom]
    length, off, lb, lw = ent
    lo, hi = max(s0, 0), min(s0 + n, length)
    codes = np.full(n, 4, np.uint8)
    if hi > lo:
        a0 = off + lo // lb * lw + lo % lb
        a1 = off + (hi - 1) // lb * lw + (hi - 1) % lb + 1
        fasta.seek(a0)
        raw = np.frombuffer(fasta.read(a1 - a0), dtype=np.uint8)
        raw = raw[(raw != 10) & (raw != 13)][:hi - lo]
        lut = np.full(256, 4, np.uint8)
        for i, ch in enumerate(b"ACGT"):
            lut[ch] = i
            lut[ch + 32] = i
        codes[lo - s0:hi - s0] = lut[raw]
    return codes


def _patch_squeeze_excite():
    """The Akita v2 models of gs://basenji_hic/3-2021 were trained with a batch norm on the excitation of
    every squeeze-excite block (4 more weights per block); the local basenji checkout has it commented
    out. Same layer with the norm enabled, as in calico/basenji."""
    import tensorflow as tf
    from basenji import layers

    orig_build = layers.SqueezeExcite.build

    def build(self, input_shape):
        orig_build(self, input_shape)
        self.norm = (tf.keras.layers.BatchNormalization(momentum=self.bn_momentum, gamma_initializer="zeros")
                     if self.norm_type == "batch" else None)

    def call(self, x):
        x = layers.activate(x, self.activation)
        excite = self.dense2(self.dense1(self.gap(x)))
        if self.norm is not None:
            excite = self.norm(excite)
        excite = tf.reshape(excite, [-1, 1, 1, self.num_channels] if self.one_or_two == "two"
                            else [-1, 1, self.num_channels])
        return x + excite if self.additive else x * tf.keras.activations.sigmoid(excite)

    layers.SqueezeExcite.build = build
    layers.SqueezeExcite.call = call


def stage_akita(a):
    from basenji import seqnn
    _patch_squeeze_excite()
    from scipy.ndimage import map_coordinates                                  # noqa: F401
    d = VM / "akita_v2"
    params = json.load(open(d / "params.json"))
    fai = {}
    for ln in open(VM / "ref/mm10.fa.fai"):
        p = ln.split("\t")
        fai[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    fasta = open(VM / "ref/mm10.fa", "rb")
    W = pd.read_csv(VM / "regions/windows.tsv", sep="\t")
    W = W[W.ok & (W.set == "auto")].drop_duplicates(["seq_key"]).sort_values(["fold", "chrom", "tile_start"])
    if a.part >= 0:
        W = W.iloc[a.part::a.nparts]
    if a.akita_n:                         # random subset (seed 0) of the windows, ordered by fold
        W = W.sample(min(a.akita_n, len(W)), random_state=0).sort_values(["fold", "chrom", "tile_start"])
    if a.limit:
        W = W.iloc[::max(1, len(W) // a.limit)][:a.limit]
    out_dir = VM / "akita"
    out_dir.mkdir(parents=True, exist_ok=True)
    f_out = out_dir / ("akita_limited.npz" if a.limit else
                       f"akita_part{a.part}.npz" if a.part >= 0 else "akita.npz")
    res_d = dict(np.load(f_out)) if f_out.exists() else {}
    NB, ASZ = 512, 2048
    iu = np.triu_indices(NB, 2)
    models = {}
    t0, n = time.time(), 0
    for r in W.itertuples():
        key = r.seq_key
        if f"{kb(10000)}:0:{key}" in res_d:
            continue
        k = int(r.fold)
        if k not in models:
            m = seqnn.SeqNN(params["model"])
            m.restore(str(d / f"f{k}c0_model1_best.h5"), head_i=1)
            models[k] = m
            if len(models) > 2:
                models.pop(next(iter(models)))
        m = models[k]
        codes = read_seq(fasta, fai, r.chrom, int(r.tile_start), TILE)
        oh = np.zeros((TILE, 4), np.float32)
        for code in range(4):
            oh[codes == code, code] = 1
        x = np.stack([oh, oh[::-1, ::-1]])
        y = m.models[1].predict(x, verbose=0)                                 # (2, pairs, 6)
        x0 = int(r.tile_start) + PAD
        for h in AK_HEADS:
            mats = []
            for rr in range(2):
                M = np.full((NB, NB), np.nan)
                M[iu] = y[rr][:, h]
                M = np.where(np.isfinite(M), M, M.T)
                mats.append(M if rr == 0 else M[::-1, ::-1])
            Lf = log_to_final(np.nanmean(mats, 0) * np.log(2.0), NB)
            for res in (10000, 25000):
                lo, hi = crop_bins(int(r.tile_start), int(r.tile_end), res)
                res_d[f"{kb(res)}:{h}:{key}"] = sample_grid(Lf, ASZ, x0, lo, hi - lo, res).astype(np.float32)
        n += 1
        if n % 50 == 0:
            log(f"akita {n}/{len(W)} tiles, {time.time() - t0:.0f} s")
            np.savez(f_out.with_suffix(".tmp.npz"), **res_d)
            os.replace(f_out.with_suffix(".tmp.npz"), f_out)
    np.savez(f_out.with_suffix(".tmp.npz"), **res_d)
    os.replace(f_out.with_suffix(".tmp.npz"), f_out)
    log(f"akita written, {len(res_d) // (2 * len(AK_HEADS))} tiles")


# ================================================================== scoring
_W = {}


def _fetch(clr, chrom, lo, hi, res):
    try:
        return np.asarray(clr.matrix(balance=False).fetch((chrom, lo * res, hi * res)), float)
    except Exception:                                                         # noqa: BLE001
        return None


def _mask_zero(T):
    return None if T is None else np.where(T == 0, np.nan, T)


def _init_worker(cell, res, methods, akita_files, replicate):
    import cooler
    _W.clear()
    bl = blacklist_by_chrom()
    _W.update(cell=cell, res=res, tgt=cooler.Cooler(str(oe_cool(cell, res))),
              cons=cooler.Cooler(str(loo_cool(cell, res))),
              npz={m: np.load(f) for m, f in methods.items()},
              akita=[np.load(f) for f in akita_files], bl=bl,
              rep=cooler.Cooler(str(replicate)) if replicate else None)


def _bl_mask(chrom, lo, hi, res):
    """bins of the window that overlap an ENCODE mm10 blacklist v2 interval"""
    n = hi - lo
    m = np.zeros(n, bool)
    iv = _W["bl"].get(chrom)
    if iv is None:
        return m
    s, e = lo * res, hi * res
    for b0, b1 in iv[(iv[:, 1] > s) & (iv[:, 0] < e)]:
        m[max(0, (b0 - s) // res):min(n, (b1 - 1 - s) // res + 1)] = True
    return m


def _diag_mean_baseline(T):
    """distance-only predictor with the window's own per-diagonal mean (an in-sample upper bound of any
    function of distance alone)."""
    n = T.shape[0]
    out = np.full_like(T, np.nan)
    for k in range(n):
        v = np.diagonal(T, k)
        mu = np.nanmean(v) if np.isfinite(v).any() else np.nan
        idx = np.arange(n - k)
        out[idx, idx + k] = mu
        out[idx + k, idx] = mu
    return out


def stripe_profile(M, d0=3, d1=30):
    """per bin mean O/E over d0..d1 bins downstream and upstream (the base of stripes), concatenated"""
    n = M.shape[0]
    dn, up = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        v = M[i, i + d0:i + d1 + 1]
        if np.isfinite(v).sum() >= 5:
            dn[i] = np.nanmean(v)
        v = M[max(0, i - d1):max(0, i - d0 + 1), i]
        if np.isfinite(v).sum() >= 5:
            up[i] = np.nanmean(v)
    return np.concatenate([dn, up])


def _score_chunk(rows):
    from flexcible.structure import structure_metrics
    W = _W
    res = W["res"]
    out = []
    for key, chrom, lo, hi, seqkey, st in rows:
        T = _mask_zero(_fetch(W["tgt"], chrom, lo, hi, res))
        if T is None or T.shape[0] != hi - lo or T.shape[0] < 20:
            continue
        bm = _bl_mask(chrom, lo, hi, res)
        T = np.where(bm[:, None] | bm[None, :], np.nan, T)
        maps = {}
        c = _mask_zero(_fetch(W["cons"], chrom, lo, hi, res))
        if c is not None and c.shape == T.shape:
            maps["consensus"] = c
        else:
            continue
        for m, z in W["npz"].items():
            if key in z.files:
                P = np.asarray(z[key], float)
                if P.shape != T.shape:
                    continue
                live = np.abs(np.nan_to_num(P)).sum(1) > 0
                maps[m] = np.where(live[:, None] & live[None, :], P, np.nan)
        for z in W["akita"]:
            for h in AK_HEADS:
                kk = f"{kb(res)}:{h}:{seqkey}"
                if kk in z.files:
                    maps[f"akita:{AK_HEADS[h]}"] = np.asarray(z[kk], float)
        if W["rep"] is not None:
            R = _mask_zero(_fetch(W["rep"], chrom, lo, hi, res))
            if R is not None and R.shape == T.shape:
                maps["replicate_experiment"] = R
        mask = np.isfinite(T)
        for m, M in maps.items():
            mask &= np.isfinite(M)
        iu = np.triu_indices(T.shape[0], 2)
        n_px = int(mask[iu].sum())
        if n_px < 100:
            continue
        Tm = np.where(mask, T, np.nan)
        base = {"cell": W["cell"], "res": res, "key": key, "chrom": chrom, "set": st, "n_px": n_px}
        maps["distance_oracle"] = _diag_mean_baseline(Tm)
        for m, M in maps.items():
            Mm = np.where(mask, M, np.nan)
            r = structure_metrics(Mm, Tm, res)
            sp, st_ = stripe_profile(Mm), stripe_profile(Tm)
            ok = np.isfinite(sp) & np.isfinite(st_)
            r["stripe_corr"] = (float(np.corrcoef(sp[ok], st_[ok])[0, 1])
                                if ok.sum() > 20 and sp[ok].std() > 0 and st_[ok].std() > 0 else np.nan)
            out.append({**base, "method": m, **{k: r.get(k, np.nan) for k in METRICS}})
        # chromosome-wide distance-decay of an O/E map is zero by construction: a constant predictor
        out.append({**base, "method": "distance_decay", **{k: (0.0 if k == "pearson" else np.nan) for k in METRICS}})
    return out


def stage_score(a):
    import multiprocessing as mp
    for res in a.res:
        k = kb(res)
        Wn = load_windows(res, a.sets)
        for cell in a.cells:
            outf = VM / "scores" / f"{cell}_{k}.csv.gz"
            if outf.exists() and not a.force:
                log(f"[have] {outf.name}")
                continue
            methods = {}
            for pdir in sorted((VM / "pred" / k).glob("*")):
                f = pdir / cell / "DONE"
                if not f.exists():
                    continue
                for npz in sorted((pdir / cell).glob("*.npz")):
                    methods[f"{pdir.name}:{npz.stem}"] = npz
            akf = []
            if cell == "mesc":
                akf = [p for p in sorted((VM / "akita").glob("akita*.npz")) if "tmp" not in p.name]
            rep = None
            if cell == "mesc":
                p = VM / "oe" / f"mesc_rep_proper_{k}.cool"
                rep = p if p.exists() else None
            sub = Wn if not a.limit else Wn.sample(a.limit, random_state=0)
            rows = list(zip(sub.key, sub.chrom, sub.b_lo, sub.b_hi, sub.seq_key, sub.set))
            chunks = [rows[i:i + 40] for i in range(0, len(rows), 40)]
            log(f"score {cell} {k}: {len(rows)} windows, {len(methods)} model outputs, akita {len(akf)}")
            res_l = []
            with mp.Pool(a.procs, initializer=_init_worker, initargs=(cell, res, methods, akf, rep)) as pool:
                for i, r in enumerate(pool.imap_unordered(_score_chunk, chunks)):
                    res_l.extend(r)
                    if i % 25 == 0:
                        log(f"  {i + 1}/{len(chunks)} chunks")
            df = pd.DataFrame(res_l)
            outf.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outf if not a.limit else outf.with_suffix(".limited.gz"), index=False, float_format="%.5g")
            t = df[df.set == "auto"].groupby("method").pearson.median().sort_values(ascending=False)
            log(f"median Pearson ({cell}, {k}, autosomes):\n{t.to_string()}")


# ================================================================== ceilings
def stage_ceiling(a):
    import cooler
    import replicate_ceiling as rc
    for res in a.res:
        k = kb(res)
        Wn = load_windows(res, ("auto",))
        for cell in POOL:
            f = VM / "scores" / f"ceiling_{cell}_{k}.csv"
            bal = VM / "balanced" / f"{cell}_{k}.cool"
            if f.exists() or not bal.exists():
                continue
            clr = cooler.Cooler(str(bal))
            sub = Wn.sample(min(a.n_windows, len(Wn)), random_state=0)
            rng = np.random.default_rng(0)
            rows = []
            for r in sub.itertuples():
                reg = (r.chrom, int(r.b_lo) * res, int(r.b_hi) * res)
                try:
                    C = np.asarray(clr.matrix(balance=False).fetch(reg), float)
                    w = clr.bins().fetch(reg)["weight"].values.astype(float)
                except Exception:                                                 # noqa: BLE001
                    continue
                n = C.shape[0]
                if n < 20 or not np.isfinite(w).any():
                    continue
                U = np.triu(np.rint(C).astype(np.int64))
                H1 = rng.binomial(U, 0.5)
                H2 = U - H1
                H1 = H1 + np.triu(H1, 1).T
                H2 = H2 + np.triu(H2, 1).T
                outs = []
                for H in (H1, H2):
                    B = H * w[:, None] * w[None, :]
                    B = np.where(np.isfinite(B), B, np.nan)
                    outs.append(rc.akita_space(B, max_d=n, smooth=True))
                iu = np.triu_indices(n, 2)
                x, y = outs[0][iu], outs[1][iu]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 100 or x[ok].std() < 1e-9 or y[ok].std() < 1e-9:
                    continue
                rh = float(np.corrcoef(x[ok], y[ok])[0, 1])
                rows.append({"cell": cell, "res": res, "key": r.key, "r_half": rh, "r_full": 2 * rh / (1 + rh),
                             "n_px": int(ok.sum())})
            df = pd.DataFrame(rows)
            df.to_csv(f, index=False)
            log(f"ceiling {cell} {k}: half {df.r_half.median():.3f} full {df.r_full.median():.3f} ({len(df)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["regions", "consensus", "views", "predict", "akita", "score", "ceiling",
                                      "summary", "figure"])
    ap.add_argument("--res", type=int, nargs="+", default=[10000, 25000])
    ap.add_argument("--cells", nargs="+", default=POOL)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--variants", nargs="+", default=["main"], choices=["main", "shuf", "xm", "untrained"])
    ap.add_argument("--regions", choices=["auto", "all"], default="auto")
    ap.add_argument("--sets", nargs="+", default=["auto"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=1000)
    ap.add_argument("--akita-n", type=int, default=0, help="akita: random subset of this many windows (seed 0)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--part", type=int, default=-1)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "regions":
        stage_regions(a)
    elif a.stage == "consensus":
        stage_consensus(a)
    elif a.stage == "views":
        stage_views(a)
    elif a.stage == "predict":
        stage_predict(a)
    elif a.stage == "akita":
        stage_akita(a)
    elif a.stage == "score":
        stage_score(a)
    elif a.stage == "ceiling":
        stage_ceiling(a)
    else:
        import mouse_transfer_report as rep
        rep.run(a.stage)


if __name__ == "__main__":
    main()
