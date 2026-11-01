"""FlexCible at bin sizes it was not trained on (registry section 89).

The one-model checkpoint all12one is trained at 2, 5, 10 and 25 kb with a
continuous log2(bin size) embedding. This script predicts GM12878 (a training
cell, leave-one-out consensus of the other 11 as reference) and K562 (held out,
12-cell consensus) at 1, 4, 20 and 50 kb, scores the Akita test regions on
chr2, chr3 and chr10 (chr2 and chr10 at 1 kb) against the measured map, with the
consensus alone as baseline, and compares with the nearest trained bin size on
the same regions.

Stages (each resumable: a finished output is never rebuilt; run them in this
order, or --stage all):

  cools      per cell and bin size: a cool restricted to the chromosomes and to a
             distance band, coarsened from the finest available zoom level
             (4DN mcools: 1, 2, 5, 10, 25, 50 kb; ENCODE cools made with hicxchange:
             2, 5, 10, 25 kb), ICE balanced (cooler.balance_cooler, ignore_diags 2,
             max_iters 500 as the training maps), then scripts/oe_proper.py.
             1 kb exists only for the six 4DN cells (GM12878, IMR-90, HCT116,
             HFF, H1, K562): the 1 kb consensus is the mean of the five training
             cells among them (the ENCODE .hic files of the other seven cells were
             deleted after conversion at 2 kb and coarser).
  consensus  scripts/consensus_loo.py on those maps (leave-one-out per training
             cell, full consensus for K562)
  config     one yaml per bin size from the nearest trained view's config.yaml
  precompute flexcible precompute at that bin size (normalisers and windows)
  score      flexcible's crosscell scoring (scripts/crosscell_predict.py) of the
             shared checkpoint at the new bin size with three normaliser variants:
               nearest   track and matrix normalisers of the nearest trained bin size
               refit_tracks  track normalisers refitted on the cell's own signal
                        (what `flexcible predict --renormalize-tracks` does)
               refit_all  track and matrix normalisers from `flexcible precompute`
  summary    results/zero_shot_binsize/summary.csv and RESULTS.md

    python scripts/zero_shot_binsize.py --stage all
    python scripts/zero_shot_binsize.py --stage cools --bins 4000 --workers 3

``--model all12one_fullrd`` (registry section 102) repeats the protocol with the release candidate
(13 marks plus Meth, consensus reference): the maps and consensus cools of the shared folder are
reused, config, cache, normalisers and scores get a tag of the model name
(``scores_<model>``, ``zeroshot_<model>_<k>kb``), Meth is given to every cell that has a
methylation folder and the summary is written to ``results/zero_shot_binsize/<model>/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PY = os.environ.get("HICMULTI_PY", sys.executable)
HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
DATA = Path(os.environ.get("HICMULTI_DATA", str(Path.home() / "data/hic-multiome")))
ZS = HM / "zero_shot_binsize"                       # maps, configs, logs (share)
LOCAL = Path(os.environ.get("HICMULTI_ZS_LOCAL", str(Path.home() / "scratch/zero_shot_binsize")))
CACHE = HM / "cache_expanded"
REFOUT = HM / "runs/multicell_reference"
RESULTS = REPO / "results/zero_shot_binsize"
MODEL = "all12one"                                  # set by --model
TAG = ""                                            # "" for all12one, "_<model>" otherwise
ENC = HM / "encode_hic"
ORIG = HM.parent / "restart_of_prediction_project/originalData/4Dnucleosome"

REMOTE = os.environ.get("HICMULTI_REMOTE", "wolffjoa@172.17.42.159")      # idle RTX 4090 host
REMOTE_ROOT = "scratch/zeroshot"                                            # under the remote home
BINS = [1000, 4000, 20000, 50000]
TRAINED = [2000, 5000, 10000, 25000]
CHROMS = ["chr2", "chr3", "chr10"]
CHROMS_1KB = ["chr2", "chr10"]
TRAIN12 = ["gm12878", "imr90", "hct116", "hff", "h1", "hepg2", "a549", "nhek",
           "gm23248", "mcf7", "cd14mono", "pc9"]
CELLS = ["gm12878", "k562"]
# 4DN mcools (ICE weights stored; used here as raw counts) and ENCODE cools
MCOOL = {
    "gm12878": ORIG / "GM12878_4DNFIXP4QG5B.mcool",
    "h1": ORIG / "H1_4DNFI6HDY7WZ.mcool",
    "k562": ORIG / "K562_4DNFI18UHVRO.mcool",
    "imr90": HM.parent / "hyperparameter_publication/imr90.mcool",
    "hff": HM / "matrices/hff_microc.mcool",
    "hct116": HM / "matrices/hct116_untreated.mcool",
}
MCOOL_RES = [1000, 2000, 5000, 10000, 25000, 50000]
ENC_RES = [2000, 5000, 10000, 25000]
# O/E band (bp). 1 kb windows span 1.02 Mb, so a 1.1 Mb band holds every window pixel
# (the per-diagonal expectation does not depend on the band); the others use the 2 Mb of training.
OE_BAND = {1000: 1_100_000}
WINDOW = {1000: (1024, 512), 4000: (512, 256), 20000: (96, 48), 50000: (48, 24)}
NEAREST = {1000: 2000, 4000: 5000, 20000: 25000, 50000: 25000}      # nearest in log2(bin size)
BW = {"gm12878": DATA / "hg38_gm12878", "k562": HM / "hg38_k562"}
REGIONS = Path("/mnt/processed/wolffjoa/restart_of_prediction_project/akita_normalization/"
               "akita_validation/sequences_test_list.tsv")


def kb(b):
    return f"{b // 1000}kb"


def chroms_of(b):
    return CHROMS_1KB if b == 1000 else CHROMS


def cells_with_data(b):
    """Cells that can be built at bin size b."""
    return [c for c in TRAIN12 + ["k562"] if c in MCOOL or b != 1000]


def train_cells(b):
    return [c for c in TRAIN12 if c in MCOOL or b != 1000]


def oe_path(cell, b):
    return ZS / "oe" / f"{cell}_proper_{kb(b)}.cool"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cmd, logfile=None, env=None, check=True):
    cmd = [str(c) for c in cmd]
    log("[cmd] " + " ".join(cmd))
    fh = open(logfile, "a") if logfile else None
    r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT if fh else None, env=env, cwd=REPO)
    if fh:
        fh.close()
    if check and r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}; see {logfile}")
    return r.returncode


# ---------------------------------------------------------------- stage: cools
def source(cell, b):
    """(cooler, base resolution, factor) of the finest-division zoom level for bin size b."""
    import cooler
    if cell in MCOOL:
        avail, path = MCOOL_RES, lambda r: f"{MCOOL[cell]}::/resolutions/{r}"
    else:
        avail, path = ENC_RES, lambda r: str(ENC / f"{cell}_{kb(r)}.cool")
    base = max((r for r in avail if b % r == 0 and r <= b), default=None)
    if base is None:
        raise SystemExit(f"{cell}: no zoom level divides {b} bp (has {avail})")
    return cooler.Cooler(path(base)), base, b // base


def build_raw(cell, b, chroms, out, band_bp):
    """Restricted, coarsened, balanced cool: counts of `chroms` within `band_bp`, bin size b."""
    import cooler
    src, base, f = source(cell, b)
    sizes = pd.Series({c: int(src.chromsizes[c]) for c in chroms})
    bins = cooler.util.binnify(sizes, b)
    new_off = {c: int(bins.index[bins.chrom == c][0]) for c in chroms}
    maxd = int(math.ceil(band_bp / base))              # fine bins
    h5 = src.open("r")
    offsets = h5["indexes/bin1_offset"]
    ROWS = 20000                                        # coarse rows per block

    def blocks():
        for ch in chroms:
            lo, hi = (int(x) for x in src.extent(ch))
            nfine = hi - lo
            nnew = -(-nfine // f)
            offs = offsets[lo:hi + 1]
            for r0 in range(0, nnew, ROWS):
                r1 = min(r0 + ROWS, nnew)
                fa, fb = r0 * f, min(r1 * f, nfine)
                px = src.pixels()[int(offs[fa]):int(offs[fb])]
                b1 = px["bin1_id"].values - lo
                b2 = px["bin2_id"].values - lo
                keep = (b2 < nfine) & (b2 - b1 <= maxd)
                b1, b2, cnt = b1[keep], b2[keep], px["count"].values[keep]
                # also at f == 1: the 4DN mcools hold repeated (bin1, bin2) entries, which
                # cooler's matrix() sums; the sum is what oe_proper saw for the training cells
                df = pd.DataFrame({"bin1_id": b1 // f, "bin2_id": b2 // f, "count": cnt})
                df = df.groupby(["bin1_id", "bin2_id"], sort=True)["count"].sum().reset_index()
                if len(df):
                    df["bin1_id"] += new_off[ch]
                    df["bin2_id"] += new_off[ch]
                    yield df

    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)
    cooler.create_cooler(str(part), bins=bins, pixels=blocks(), dtypes={"count": np.int32},
                         ordered=True, ensure_sorted=False)
    h5.close()
    clr = cooler.Cooler(str(part))
    ok, stats = cooler.balance_cooler(clr, cis_only=True, ignore_diags=2, mad_max=5, min_nnz=10,
                                      min_count=0, max_iters=500, tol=1e-5, store=True,
                                      chunksize=20_000_000)
    part.rename(out)
    return base, f


def build_one(args):
    cell, b = args
    chroms = chroms_of(b)
    out = oe_path(cell, b)
    if out.exists():
        return cell, b, "have"
    t0 = time.time()
    band = OE_BAND.get(b, 2_000_000)
    raw = LOCAL / "raw" / f"{cell}_{kb(b)}.cool"
    if not raw.exists():
        base, f = build_raw(cell, b, chroms, raw, band + 100_000)
        log(f"[raw] {cell} {kb(b)} from {kb(base)} x{f} ({(time.time() - t0) / 60:.1f} min)")
    tmp = LOCAL / "oe" / f"{cell}_proper_{kb(b)}.cool"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)
    (ZS / "logs").mkdir(parents=True, exist_ok=True)
    run([PY, "scripts/oe_proper.py", "--input", raw, "--output", tmp, "--binsize", b,
         "--chroms", *chroms, "--max-dist", band], ZS / "logs" / f"oe_{cell}_{kb(b)}.log")
    (ZS / "oe").mkdir(parents=True, exist_ok=True)
    cp = out.with_name(out.name + ".copy")
    shutil.copyfile(tmp, cp)
    cp.rename(out)
    tmp.unlink()
    raw.unlink()
    return cell, b, f"built {(time.time() - t0) / 60:.1f} min"


def stage_cools(bins, workers, cells=None):
    tasks = [(c, b) for b in bins for c in (cells or cells_with_data(b))]
    todo = [t for t in tasks if not oe_path(*t).exists()]
    log(f"cools: {len(todo)} of {len(tasks)} maps to build")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for cell, b, msg in ex.map(build_one, todo):
            log(f"[oe] {cell} {kb(b)}: {msg}")


# ------------------------------------------------------------ stage: consensus
def stage_consensus(bins):
    for b in bins:
        cells = train_cells(b)
        n = len(cells)
        full = ZS / "oe" / f"consensus{n}_proper_{kb(b)}.cool"
        loo = [ZS / "oe" / f"consensus_loo_{c}_proper_{kb(b)}.cool" for c in cells]
        if full.exists() and all(p.exists() for p in loo):
            log(f"consensus {kb(b)}: have")
            continue
        run([PY, "scripts/consensus_loo.py", "--res", b, "--dir", ZS / "oe", "--cells", *cells,
             "--tmp", LOCAL / f"consensus_tmp_{kb(b)}"], ZS / "logs" / f"consensus_{kb(b)}.log")


def consensus_cool(cell, b):
    """The reference of `cell` at bin size b: leave-one-out for a training cell."""
    if cell in train_cells(b):
        return ZS / "oe" / f"consensus_loo_{cell}_proper_{kb(b)}.cool"
    return ZS / "oe" / f"consensus{len(train_cells(b))}_proper_{kb(b)}.cool"


# ---------------------------------------------------------------- stage: config
def view_dir(res):
    return REFOUT / f"{MODEL}_{kb(res)}"


def zs_cfg_path(b):
    return ZS / "configs" / f"zeroshot{TAG}_{kb(b)}.yaml"


def meth_dirs(cell):
    """Methylation folder(s) of a cell for models with a Meth track (lifted RRBS included)."""
    import methylation as me
    if MODEL == "all12one" or not me.has_meth(cell, v2=True):
        return []
    return [str(me.meth_dir(cell, v2=True))]


def cache_dir(b):
    return CACHE / f"zeroshot{TAG}_{kb(b)}"


def stage_config(bins):
    (ZS / "configs").mkdir(parents=True, exist_ok=True)
    for b in bins:
        cfg = yaml.safe_load(open(view_dir(NEAREST[b]) / "checkpoint-best/config.yaml"))
        near = cfg["data"]["resolution"]
        size, stride = WINDOW[b]
        cfg["data"]["resolution"] = b
        cfg["data"]["chroms"] = chroms_of(b)
        cfg["window"]["size"], cfg["window"]["stride"] = size, stride
        template = {s["name"]: s for s in cfg["data"]["samples"]}
        samples = []
        for cell in CELLS:
            s = dict(template.get(cell, template["gm12878"]))
            s.update({"name": cell, "cool_path": str(oe_path(cell, b)),
                      "reference_cool": str(consensus_cool(cell, b)),
                      "bigwig_dir": str(BW[cell])})
            if cell == "k562":
                s["missing_tracks"] = json.loads(
                    (BW[cell] / "missing.json").read_text()) if (BW[cell] / "missing.json").exists() else []
            s.pop("train_chroms", None)
            if MODEL != "all12one":
                s["extra_bigwig_dirs"] = meth_dirs(cell)
            samples.append(s)
        cfg["data"]["samples"] = samples
        cfg["train"].update({"cache_dir": str(cache_dir(b)),
                             "output_dir": str(ZS / "runs" / f"zeroshot{TAG}_{kb(b)}"),
                             "resume": False, "val_regions": 0})
        yaml.safe_dump(cfg, open(zs_cfg_path(b), "w"), sort_keys=False)
        log(f"config {kb(b)}: window {size}/{stride}, nearest trained view {kb(near)} -> {zs_cfg_path(b)}")


def stage_precompute(bins, workers):
    for b in bins:
        if (cache_dir(b) / "manifest.json").exists():
            log(f"precompute {kb(b)}: have")
            continue
        run([PY, "-m", "flexcible.precompute", "--config", zs_cfg_path(b), "--workers", workers],
            ZS / "logs" / f"precompute_{kb(b)}.log")


# ----------------------------------------------------------------- stage: score
def regions_on(chroms, out):
    r = pd.read_csv(REGIONS, sep="\t", names=["chrom", "start", "end"])
    r = r[r.chrom.isin(chroms)]
    r.to_csv(out, sep="\t", header=False, index=False)
    return r


def variant_ckpt(b, cell, variant):
    """Checkpoint folder (local, symlinked model.pt) with the normalisers of a variant."""
    near = NEAREST[b]
    src = view_dir(near) / "checkpoint-best"
    d = LOCAL / "ckpt" / f"{kb(b)}{TAG}_{cell}_{variant}"
    if (d / "norm_tracks.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    for f in ("model.pt", "channels.json"):
        (d / f).unlink(missing_ok=True)
        (d / f).symlink_to(src / f)
    for f in src.glob("norm_matrix_*.json"):
        shutil.copyfile(f, d / f.name)
    shutil.copyfile(src / "norm_tracks.json", d / "norm_tracks.json")
    if variant == "refit_all":
        shutil.copyfile(cache_dir(b) / "norm_tracks.json", d / "norm_tracks.json")
        for f in cache_dir(b).glob("norm_matrix_*.json"):
            shutil.copyfile(f, d / f.name)
    elif variant == "refit_tracks":
        refit_tracks(b, cell, d)
    return d


def refit_tracks(b, cell, d):
    """Track statistics refitted on the cell's own signal, as predict --renormalize-tracks."""
    import cooler
    from flexcible.config import Config
    from flexcible.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
    from flexcible.tracks import TrackNormalizer
    cfg = Config.load(str(zs_cfg_path(b)))
    sample = cfg.get_sample(cell)
    clr = cooler.Cooler(str(oe_path(cell, b)))
    files = sample_track_files(cfg, sample)
    old = TrackNormalizer.load(d / "norm_tracks.json")
    rows, valid = [], []
    dead = set(sample.missing_tracks) | set(sample.zeroed_tracks)
    for ch in chroms_of(b):
        rows.append(assemble_chrom(cfg, sample, files, ch, chrom_nbins_of(clr, ch)))
        valid.append(np.array([t not in dead for t in cfg.track_names]
                              + [True] * (rows[-1].shape[0] - len(cfg.track_names))))
    new = TrackNormalizer(transform=old.transform, norm=old.norm, smooth_bins=old.smooth_bins,
                          passthrough=old.passthrough).fit(rows, valid)
    new.save(d / "norm_tracks.json")


def score_dir(b, cell, variant):
    return ZS / f"scores{TAG}" / kb(b) / variant / cell


def score_jobs(bins, variants):
    """(bin size, cell, variant, command words) of every scoring run that is still missing;
    paths are local, remote runs map them with remote_path."""
    jobs = []
    for b in bins:
        chroms = chroms_of(b)
        (ZS / "regions").mkdir(parents=True, exist_ok=True)
        reg = ZS / "regions" / f"akita_test_{'_'.join(chroms)}.tsv"
        regions_on(chroms, reg)
        for cell in CELLS:
            for variant in variants:
                out = score_dir(b, cell, variant)
                if (out / f"predict_{cell}.csv").exists():
                    continue
                ck = variant_ckpt(b, cell, variant)
                tag = "consensus_loo" if cell in train_cells(b) else "consensus"
                xb = ["--extra-bigwig-dir", meth_dirs(cell)[0]] if meth_dirs(cell) else []
                jobs.append((b, cell, variant, [
                    "scripts/crosscell_predict.py", "--config", zs_cfg_path(b),
                    "--checkpoint", ck, "--cell", cell, "--bigwig-dir", BW[cell],
                    "--target", oe_path(cell, b), "--tta", "--binsize", b,
                    "--reference", consensus_cool(cell, b), "--regions", reg, *xb,
                    "--out-dir", out, "--others", f"{tag}={consensus_cool(cell, b)}"]))
    return jobs


def stage_score(bins, variants, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    for b, cell, variant, words in score_jobs(bins, variants):
        out = score_dir(b, cell, variant)
        out.parent.mkdir(parents=True, exist_ok=True)
        run([PY, *words], out.parent / f"{cell}_{variant}.log", env)


def ssh(cmd, check=True):
    log(f"[ssh] {cmd[:200]}")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, cmd])
    if check and r.returncode:
        raise SystemExit(f"remote command failed ({r.returncode}): {cmd[:200]}")
    return r.returncode


def stage_score_remote(bins, variants):
    """Scoring on the remote GPU host, whose disk does not see the share: the inputs of the
    missing runs (view checkpoints without model.pt copies, bigWigs, restricted O/E and
    consensus cools, regions) are staged under ~/scratch/zeroshot, the runs executed there and the
    score tables copied back. Refuses to start while a process holds the remote GPU."""
    home = subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, "echo $HOME"], capture_output=True,
                          text=True).stdout.strip()
    root = f"{home}/{REMOTE_ROOT}"
    jobs = score_jobs(bins, variants)
    if not jobs:
        log("score remote: nothing to do")
        return
    busy = subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE,
                           "nvidia-smi --query-compute-apps=pid --format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    if busy:
        raise SystemExit(f"remote GPU is in use (pids {busy}); not starting")
    lh = str(Path.home())
    mounts = [(str(ZS), f"{root}/zs"), (str(LOCAL), f"{root}/local"), (str(HM / "hg38_k562"), f"{root}/hg38_k562"),
              (str(REGIONS), f"{root}/regions.tsv"), (str(REFOUT), f"{root}/refout"), (lh, home)]

    def rp(x):
        x = str(x)
        if "=" in x and not x.startswith(("/", "-")):          # key=path words (--others)
            k, v = x.split("=", 1)
            return k + "=" + rp(v)
        for a_, b_ in mounts:
            if x.startswith(a_):
                return b_ + x[len(a_):]
        return x

    ssh(f"mkdir -p {root}/zs/oe {root}/zs/configs {root}/zs/regions {root}/local/ckpt")
    subprocess.run(["rsync", "-a", "--exclude", ".git", "--exclude", "results/", "--exclude", "REMOTE.md",
                    "--exclude", "remote_queue*", f"{REPO}/", f"{REMOTE}:{home}/hic-multiome-unet/"], check=True)
    need_b = sorted({j[0] for j in jobs})
    for b in need_b:
        txt = zs_cfg_path(b).read_text()
        for a_, b_ in mounts:
            txt = txt.replace(a_, b_)
        tmp = LOCAL / f"remote_{kb(b)}.yaml"
        tmp.write_text(txt)
        subprocess.run(["rsync", "-a", str(tmp), f"{REMOTE}:{root}/zs/configs/zeroshot_{kb(b)}.yaml"], check=True)
        for cell in CELLS:
            for f in (oe_path(cell, b), consensus_cool(cell, b)):
                subprocess.run(["rsync", "-a", "--partial", str(f), f"{REMOTE}:{root}/zs/oe/"], check=True)
    subprocess.run(["rsync", "-a", f"{ZS}/regions/", f"{REMOTE}:{root}/zs/regions/"], check=True)
    for cell in CELLS:
        if cell == "k562":
            subprocess.run(["rsync", "-a", "--partial", f"{HM / 'hg38_k562'}/", f"{REMOTE}:{root}/hg38_k562/"],
                           check=True)
    ssh(f"mkdir -p {root}/refout && ls {root}/refout/model_* >/dev/null 2>&1 || true")
    script = ["set -e", f"cd {home}/hic-multiome-unet", "source ~/hicenv/bin/activate"]
    sent_model = set()
    for b, cell, variant, words in jobs:
        ck = variant_ckpt(b, cell, variant)
        rck = rp(ck)
        subprocess.run(["rsync", "-aL", "--exclude", "model.pt", f"{ck}/", f"{REMOTE}:{rck}/"], check=True)
        real = (ck / "model.pt").resolve()
        rm = f"{root}/refout/model_{kb(NEAREST[b])}.pt"
        if rm not in sent_model:
            subprocess.run(["rsync", "-a", "--partial", str(real), f"{REMOTE}:{rm}"], check=True)
            sent_model.add(rm)
        script.append(f"ln -sfn {rm} {rck}/model.pt")
        out = rp(score_dir(b, cell, variant))
        script.append(f"mkdir -p {out}")
        script.append("CUDA_VISIBLE_DEVICES=0 python " + " ".join(rp(w) for w in map(str, words))
                      + f" > {out}.log 2>&1")
    sh_local = LOCAL / "remote_score.sh"
    sh_local.write_text("\n".join(script) + "\n")
    subprocess.run(["rsync", "-a", str(sh_local), f"{REMOTE}:{root}/remote_score.sh"], check=True)
    ssh(f"bash {root}/remote_score.sh")
    subprocess.run(["rsync", "-a", f"{REMOTE}:{root}/zs/scores/", f"{ZS}/scores/"], check=False)


# --------------------------------------------------------------- stage: summary
def q(x, p):
    return float(np.nanpercentile(x, p)) if len(x) else float("nan")


def trained_rows(cell, res, regs):
    """The one model at a trained bin size on the same regions (from the eval stage of section 88)."""
    f = view_dir(res) / "crosscell" / f"predict_{cell}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df[df.region.isin(regs)]


def stage_summary():
    out_dir = RESULTS if MODEL == "all12one" else RESULTS / MODEL
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in sorted(set(BINS + TRAINED)):
        chroms = chroms_of(b if b in BINS else 1000) if b in BINS else CHROMS
        regs_df = pd.read_csv(REGIONS, sep="\t", names=["chrom", "start", "end"])
        regs_df = regs_df[regs_df.chrom.isin(chroms)]
        regs = {f"{r.chrom}:{r.start}-{r.end}" for r in regs_df.itertuples()}
        for cell in CELLS:
            ref_cells = len(train_cells(b)) - (1 if cell in train_cells(b) else 0) if b in BINS else (
                11 if cell == "gm12878" else 12)
            if b in TRAINED:
                df = trained_rows(cell, b, regs)
                if df is None:
                    continue
                cons = df["base_consensus_loo" if cell == "gm12878" else "base_consensus12"]
                rows.append(dict(bin_size=b, cell=cell, chroms=" ".join(chroms), normalisers="trained",
                                 n_regions=int(df["model"].notna().sum()),
                                 median_r=q(df["model"], 50), q25=q(df["model"], 25), q75=q(df["model"], 75),
                                 consensus_median_r=q(cons, 50), nearest_trained_binsize=b,
                                 nearest_trained_median_r=q(df["model"], 50), n_consensus_cells=ref_cells,
                                 trained_bin_size=True))
                continue
            near = NEAREST[b]
            nd = trained_rows(cell, near, regs)
            for variant in ("nearest", "refit_tracks", "refit_all"):
                f = score_dir(b, cell, variant) / f"predict_{cell}.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                cons = df[[c for c in df.columns if c.startswith("base_")][0]]
                nm = (nd[nd.region.isin(set(df.region))]["model"] if nd is not None else pd.Series(dtype=float))
                rows.append(dict(bin_size=b, cell=cell, chroms=" ".join(chroms), normalisers=variant,
                                 n_regions=int(df["model"].notna().sum()),
                                 median_r=q(df["model"], 50), q25=q(df["model"], 25), q75=q(df["model"], 75),
                                 consensus_median_r=q(cons, 50), nearest_trained_binsize=near,
                                 nearest_trained_median_r=q(nm, 50), n_consensus_cells=ref_cells,
                                 trained_bin_size=False))
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out_dir / "summary.csv", index=False)
    log(f"wrote {out_dir / 'summary.csv'} ({len(df)} rows)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["cools", "consensus", "config", "precompute", "score", "summary", "all"])
    ap.add_argument("--bins", nargs="*", type=int, default=BINS)
    ap.add_argument("--cells", nargs="*", default=None, help="cools: only these cells")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--variants", nargs="*", default=["nearest", "refit_tracks", "refit_all"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", default="all12one",
                    help="shared checkpoint whose views <model>_<k>kb are scored (default all12one)")
    ap.add_argument("--remote", action="store_true",
                    help="score: run on the remote 4090 host (inputs staged to ~/scratch/zeroshot there)")
    a = ap.parse_args()
    global MODEL, TAG
    MODEL = a.model
    TAG = "" if MODEL == "all12one" else f"_{MODEL}"
    ZS.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    st = a.stage
    if st in ("cools", "all"):
        stage_cools(a.bins, a.workers, a.cells)
    if st in ("consensus", "all"):
        stage_consensus(a.bins)
    if st in ("config", "all"):
        stage_config(a.bins)
    if st in ("precompute", "all"):
        stage_precompute(a.bins, a.workers)
    if st in ("score", "all"):
        if a.remote:
            stage_score_remote(a.bins, a.variants)
        else:
            stage_score(a.bins, a.variants, a.gpu)
    if st in ("summary", "all"):
        stage_summary()
    print("ZERO_SHOT_DONE", flush=True)


if __name__ == "__main__":
    main()
