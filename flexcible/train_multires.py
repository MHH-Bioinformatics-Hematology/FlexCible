"""Train ONE resolution-conditioned model on windows of several resolutions.

Each resolution has its own config (same tracks, motifs, samples, reference
mode and model settings; its own data.resolution, window, cache and
reference maps) and model.resolutions listing all of them. The model is the
usual HiCModel with a learned resolution embedding that modulates every U-Net
block by FiLM (flexcible.model). Window sizes stay those of the dedicated
models (e.g. 512 / 384 / 192 / 80 bins at 2 / 5 / 10 / 25 kb): the U-Net is
fully convolutional, so each step draws one resolution and a batch of its
windows. Resolutions are drawn uniformly per step (the same draw on every DDP
rank), so each gets the same share of gradient steps; an epoch is as many
steps as one pass over all training windows of all resolutions.

Per resolution: the honest validation split of train.split_val_regions (the
same windows eval_val_windows reproduces from the per-resolution config),
the reference leakage guard, rank stripes of the training windows kept in
CPU memory (float16) and moved to the GPU per batch. Track dropout and the
cohesin-loss augmentation (train.cohesin_aug, one degron operator per
resolution) act per batch exactly as in flexcible.train: augmentation first,
then dropout. Validation after every
epoch: the median per-window Pearson (val_akita_median) per resolution; the
mean over resolutions drives checkpointing and patience. Per-resolution
training loss and validation are logged and written to learning_curve.csv.

    torchrun --standalone --nproc_per_node=2 -m flexcible.train_multires \
        --configs res2.yaml res5.yaml res10.yaml res25.yaml --out runs/x

The checkpoint folder holds model.pt, optim.pt, progress.json and, per
resolution, res_<bp>/ with that cache's normalisers, channels.json and
config.yaml. write_views() turns it into one run folder per resolution
(best pointer, checkpoint-best with a model.pt link and the resolution's
config and normalisers), which every single-resolution tool can load.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import Config
from .dataset import WindowDataset, reverse_cond, split_cond
from .losses import build_loss
from .model import make_model, n_cond_channels
from .train import (_coords, _select_rows, ddp_env, log_val_split, split_val_regions,
                    val_akita_median)
from .utils import get_logger, set_seed

log = get_logger()


def load_rows(ds: WindowDataset, rows, dtype=np.float16) -> dict:
    """target (N, W, W), cond (N, R, W) and ref (N, W, W) of the given rows of
    ds, decompressing each shard once."""
    rows = list(rows)
    W = ds.window_size
    n = len(rows)
    tg = np.empty((n, W, W), dtype)
    cd = np.empty((n, ds.n_sym, W), dtype)
    rf = np.empty((n, W, W), dtype) if ds.reference else None
    by = {}
    for k, r in enumerate(rows):
        path, local = ds._index[r][0], ds._index[r][1]
        by.setdefault(path, []).append((k, local))
    for path, pairs in by.items():
        z = np.load(path)
        ks = np.array([k for k, _ in pairs])
        loc = np.array([l for _, l in pairs])
        tg[ks] = z["target"][loc, 0]
        cd[ks] = z["cond"][loc]
        if rf is not None:
            rf[ks] = z["ref"][loc]
    out = {"target": torch.from_numpy(tg), "cond": torch.from_numpy(cd)}
    if rf is not None:
        out["ref"] = torch.from_numpy(rf)
    return out


def stage_arrays(cfgs: list[Config], stage_dir: Path):
    """Write each resolution's training-split windows (the `train` split of
    its cache, before the validation split) uncompressed to memory-mapped .npy
    files in stage_dir/res_<bp>/ (target, cond with the model's rows, ref),
    in the dataset's index order, one shard decompressed at a time. Training
    with stage_dir then maps these files: every DDP rank shares one page
    cache instead of holding its own copy, and nothing is decompressed."""
    stage_dir = Path(stage_dir)
    for c in cfgs:
        d = stage_dir / res_dir(c.data.resolution)
        if (d / "meta.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        ds = WindowDataset(c.train.cache_dir, "train")
        rows, *_ = _select_rows(c, ds.channels)
        n, W = len(ds), ds.window_size
        mm = {"target": np.lib.format.open_memmap(d / "target.npy.part", "w+", np.float16, (n, W, W)),
              "cond": np.lib.format.open_memmap(d / "cond.npy.part", "w+", np.float16,
                                                (n, len(rows), W))}
        if ds.reference:
            mm["ref"] = np.lib.format.open_memmap(d / "ref.npy.part", "w+", np.float16, (n, W, W))
        for path, pairs in ds.grouped_index().items():
            z = np.load(path)
            ks = np.array([k for k, _ in pairs])
            loc = np.array([l for _, l in pairs])
            mm["target"][ks] = z["target"][loc, 0]
            mm["cond"][ks] = z["cond"][loc][:, rows]
            if "ref" in mm:
                mm["ref"][ks] = z["ref"][loc]
            log.info("staged %s", Path(path).name)
        for k, m in mm.items():
            m.flush()
            del m
            (d / f"{k}.npy.part").rename(d / f"{k}.npy")
        (d / "meta.json").write_text(json.dumps({
            "cache": str(c.train.cache_dir), "n": n, "window": W, "rows": rows,
            "manifest_mtime": (Path(c.train.cache_dir) / "manifest.json").stat().st_mtime}))
        log.info("staged resolution %d: %d windows -> %s", c.data.resolution, n, d)


def open_stage(stage_dir: Path, c: Config, n_expected: int, rows) -> dict:
    d = Path(stage_dir) / res_dir(c.data.resolution)
    meta = json.loads((d / "meta.json").read_text())
    if meta["n"] != n_expected or meta["rows"] != list(rows) or meta["cache"] != str(c.train.cache_dir):
        raise RuntimeError(f"{d} does not match {c.train.cache_dir}; restage")
    return {k: np.load(d / f"{k}.npy", mmap_mode="r")
            for k in ("target", "cond", "ref") if (d / f"{k}.npy").exists()}


def _take(src, key, rows, device):
    """Rows of a resident tensor or a memory-mapped array, on device, float32."""
    a = src[key]
    if isinstance(a, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(a[rows])).to(device).float()
    return a[torch.as_tensor(rows)].to(device).float()


class _AtRes:
    """Bind a resolution to the model for the single-resolution helpers."""

    def __init__(self, model, res):
        self.model, self.res = model, res

    def __call__(self, *args, **kw):
        return self.model(*args, res=self.res, **kw)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()


def res_dir(res: int) -> str:
    return f"res_{int(res)}"


def build_cohesin_aug(c: Config, ds: WindowDataset, chip_rows, offset: int, device, is_main: bool):
    """CohesinAugmenter of one resolution (train.cohesin_aug of its config) and
    the contact scale of every window of its cache, in the dataset's index
    order (the order of the staged arrays and of `_coords`)."""
    from .cohesin import CohesinAugmenter, load_operator
    from .precompute import COHESIN_TRACKS, FRACTION_ROW
    from .tracks import TrackNormalizer

    cache = Path(c.train.cache_dir)
    seqf = ds.channels["seq_features"]
    if FRACTION_ROW not in seqf:
        raise RuntimeError(f"{cache}: train.cohesin_aug but no cohesin_fraction row; "
                           "set data.cohesin_fraction and re-run flexcible.precompute")
    tracks_all = ds.channels["tracks"]
    tn = TrackNormalizer.load(cache / "norm_tracks.json")
    absent_all = tn.absent_values()
    keep = [t for t in COHESIN_TRACKS if t in tracks_all and tracks_all.index(t) in chip_rows]
    if not keep:
        raise ValueError("train.cohesin_aug needs RAD21 or SMC3 among the model tracks")
    rows = [chip_rows.index(tracks_all.index(t)) for t in keep]
    op, f_ref = load_operator(c.train.cohesin_aug_operator, device)
    aug = CohesinAugmenter(
        op, f_ref, c.train.cohesin_aug, c.train.cohesin_aug_fractions, c.train.cohesin_aug_absent,
        rows, [tn.mean[tracks_all.index(t)] for t in keep], [tn.std[tracks_all.index(t)] for t in keep],
        [float(absent_all[tracks_all.index(t)]) for t in keep],
        offset - len(seqf) + seqf.index(FRACTION_ROW), device)
    names = _coords(ds)[2]
    scale_of = {n: float(json.loads((cache / f"norm_matrix_{n}.json").read_text())["scale"])
                for n in set(names)}
    scale = torch.tensor([scale_of[n] for n in names], dtype=torch.float32)
    if is_main:
        log.info("resolution %d cohesin_aug: p=%.2f fractions %s absent-track share %.2f, operator %s "
                 "(fitted at cohesin fraction %.2f), tracks %s", c.data.resolution, c.train.cohesin_aug,
                 c.train.cohesin_aug_fractions, c.train.cohesin_aug_absent, c.train.cohesin_aug_operator,
                 f_ref, keep)
    return aug, scale


def train_multires(cfgs: list[Config], out_dir: Path, resume: bool = False,
                   max_epochs: int | None = None, stage_dir: Path | None = None):
    rank, world, local_rank = ddp_env()
    is_ddp = world > 1
    if is_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0
    base = cfgs[0]
    set_seed(base.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    reslist = [int(c.data.resolution) for c in cfgs]
    for c in cfgs:
        if list(c.model.resolutions or []) != list(base.model.resolutions or []):
            raise ValueError("every config needs the same model.resolutions")
        if int(c.data.resolution) not in (c.model.resolutions or []):
            raise ValueError(f"{c.data.resolution} not in model.resolutions")
    if sorted(reslist) != sorted(base.model.resolutions):
        raise ValueError(f"configs cover {reslist}, model.resolutions {base.model.resolutions}")

    if any(c.train.hard_split or c.train.sv_mask for c in cfgs):
        raise NotImplementedError("hard split and SV masks are not supported in "
                                  "multi-resolution training")
    # reference dropout (train.reference_dropout of the first config): the reference plane of
    # that share of the training windows is set to 0 (its expected level), as in flexcible.train,
    # so that one checkpoint also predicts from the tracks alone (predict --no-reference)
    ref_p = float(base.train.reference_dropout)
    if ref_p > 0 and base.model.reference == "none":
        raise ValueError("train.reference_dropout needs a model.reference model")
    if any(c.data.cohesin_unknown or c.data.diff_marks or c.motifs.motif_meth for c in cfgs):
        raise NotImplementedError("track dropout with cohesin_unknown, diff_marks or motif_meth is "
                                  "not supported in multi-resolution training")
    # track dropout (train.track_dropout of the first config): per window the ChIP rows drawn as
    # absent get the cache's absent value of that row, exactly as in flexcible.train
    drop_p = float(base.train.track_dropout)
    drop_mode = base.train.track_dropout_mode
    if base.model.prior:
        raise NotImplementedError("model.prior is not supported in multi-resolution training")
    use_ref = base.model.reference != "none"
    data, vdata, info = {}, {}, {}
    in_ch = None
    for c in cfgs:
        r = int(c.data.resolution)
        ds = WindowDataset(c.train.cache_dir, "train")
        if use_ref:
            if not ds.reference:
                raise RuntimeError(f"{c.train.cache_dir}: no reference windows")
            ds.check_reference_leakage()
        co = _coords(ds)
        vi, ti = split_val_regions(co, c.train.val_regions, ds.window_size, c.train.seed,
                                   c.train.val_exclude_overlap)
        if is_main:
            log.info("resolution %d: %s", r, c.train.cache_dir)
            log_val_split(co, vi, ti, c.train.val_exclude_overlap)
        rows, offset, n_motifs, n_sym = _select_rows(c, ds.channels)
        absent_t, n_chip = None, 0
        from .train import dropout_chip_rows

        chip_rows = dropout_chip_rows(c, ds.channels)
        if drop_p > 0:
            from .tracks import TrackNormalizer

            n_chip = len(chip_rows)
            tn = TrackNormalizer.load(Path(c.train.cache_dir) / "norm_tracks.json")
            absent_t = torch.from_numpy(tn.absent_values()[chip_rows]).to(device)
        aug, scale = None, None
        if c.train.cohesin_aug > 0:
            aug, scale = build_cohesin_aug(c, ds, chip_rows, offset, device, is_main)
        stripe = ti[rank::world]
        if stage_dir is not None:
            src = open_stage(stage_dir, c, len(ds), rows)
            data[r] = {"src": src, "rows": np.asarray(stripe)}
            if scale is not None:
                scale = scale.to(device)            # indexed by the global row
            if is_main:
                vs = np.sort(vi)
                v = {k: torch.from_numpy(np.ascontiguousarray(a[vs])) for k, a in src.items()}
                v["target"] = v["target"][:, None]
                vdata[r] = v
        else:
            d = load_rows(ds, stripe)
            d["cond"] = d["cond"][:, rows, :]
            data[r] = {"src": d, "rows": np.arange(len(stripe))}
            if scale is not None:
                scale = scale[torch.as_tensor(stripe)].to(device)   # indexed within the stripe
            if is_main:
                v = load_rows(ds, vi)
                v["cond"] = v["cond"][:, rows, :]
                v["target"] = v["target"][:, None]
                vdata[r] = v
        ich = n_cond_channels(n_sym, n_motifs, c.model.reference_planes, c.model.cond_mode)
        if in_ch is not None and ich != in_ch:
            raise ValueError("input channels differ between resolutions")
        in_ch = ich
        info[r] = {"offset": offset, "n_motifs": n_motifs, "n_train": len(ti),
                   "n_stripe": len(stripe), "batch": c.train.batch_size,
                   "cache": c.train.cache_dir, "W": ds.window_size,
                   "absent": absent_t, "n_chip": n_chip, "aug": aug, "scale": scale}
    model = make_model(base, in_ch, 1).to(device)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    core = model.module if is_ddp else model
    opt = torch.optim.AdamW(model.parameters(), lr=base.train.lr)
    hic_loss = build_loss(base.train.loss, max_dist=base.train.loss_max_dist,
                          stride=base.train.loss_stride)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(base.train.amp)
    total_train = sum(v["n_train"] for v in info.values())
    steps_per_epoch = max(1, total_train // (world * int(np.mean([v["batch"] for v in info.values()]))))
    if is_main:
        log.info("multi-resolution model %s (%s embedding): %.2fM params, %d cond channels, %d steps "
                 "per epoch (one resolution per step, drawn uniformly), world %d, track dropout %s",
                 reslist, base.model.res_embed_mode, sum(p.numel() for p in core.parameters()) / 1e6,
                 in_ch, steps_per_epoch, world, f"{drop_mode}" if drop_p > 0 else "off")
    step = epoch = 0
    best = float("-inf")
    curve = []
    ck_latest = out_dir / "checkpoint-latest"
    if resume and (ck_latest / "model.pt").exists():
        core.load_state_dict(torch.load(ck_latest / "model.pt", map_location=device))
        opt.load_state_dict(torch.load(ck_latest / "optim.pt", map_location=device))
        pr = json.loads((ck_latest / "progress.json").read_text())
        step, epoch, best = pr["step"], pr["epoch"], pr["best"]
        if (out_dir / "learning_curve.csv").exists():
            curve = list(csv.DictReader(open(out_dir / "learning_curve.csv")))
        if is_main:
            log.info("resumed at epoch %d step %d (best %.4f)", epoch, step, best)
    max_epochs = max_epochs or base.train.max_epochs or 1000
    patience = base.train.patience_epochs or 10

    def save(name):
        if not is_main:
            return
        ck = out_dir / name
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(core.state_dict(), ck / "model.pt")
        torch.save(opt.state_dict(), ck / "optim.pt")
        (ck / "progress.json").write_text(json.dumps({"step": step, "epoch": epoch,
                                                      "best": best}))
        for c in cfgs:
            d = ck / res_dir(c.data.resolution)
            d.mkdir(exist_ok=True)
            cache = Path(c.train.cache_dir)
            for f in list(cache.glob("norm_*.json")) + [cache / "channels.json"]:
                shutil.copyfile(f, d / f.name)
            c.save(d / "config.yaml")
        (out_dir / name.replace("checkpoint-", "")).write_text(name)

    def stop_flag(v: bool) -> bool:
        if not is_ddp:
            return v
        t = torch.tensor([1.0 if v else 0.0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    no_imp = 0
    if curve:
        vals = [float(x["val_mean"]) for x in curve]
        no_imp = len(vals) - 1 - int(np.argmax(vals)) if vals else 0
    t0 = last_log = time.time()
    cursors = {r: 0 for r in reslist}
    perms = {}
    while epoch < max_epochs:
        epoch += 1
        rng = np.random.default_rng(base.train.seed * 100003 + epoch)   # same on all ranks
        seq = rng.choice(reslist, size=steps_per_epoch)
        prng = np.random.default_rng(base.train.seed * 7919 + epoch * 31 + rank)
        loss_sum = {r: 0.0 for r in reslist}
        loss_n = {r: 0 for r in reslist}
        n_aug = {r: 0 for r in reslist}
        n_refdrop = {r: 0 for r in reslist}
        model.train()
        for r in seq:
            r = int(r)
            d, bs = data[r], info[r]["batch"]
            n = len(d["rows"])
            if r not in perms or cursors[r] + bs > len(perms[r]):
                perms[r] = prng.permutation(n)
                cursors[r] = 0
            idx = np.sort(d["rows"][perms[r][cursors[r]:cursors[r] + bs]])
            cursors[r] += bs
            cond = _take(d["src"], "cond", idx, device)
            tgt = _take(d["src"], "target", idx, device)
            ref = _take(d["src"], "ref", idx, device) if use_ref else None
            if ref is not None and ref_p > 0:
                keep = torch.rand(ref.shape[0], device=device) >= ref_p
                ref = ref * keep[:, None, None].to(ref.dtype)
                n_refdrop[r] += int(ref.shape[0] - keep.sum().item())
            off, nm = info[r]["offset"], info[r]["n_motifs"]
            if base.train.augment_reverse:
                m = torch.rand(bs, device=device) < 0.5
                if m.any():
                    cond = torch.where(m[:, None, None], reverse_cond(cond, off, nm), cond)
                    tgt = torch.where(m[:, None, None], torch.flip(tgt, dims=[-1, -2]), tgt)
                    if ref is not None:
                        ref = torch.where(m[:, None, None], torch.flip(ref, dims=[-1, -2]), ref)
            if info[r]["aug"] is not None:
                cond, tgt4, n_a = info[r]["aug"](cond, tgt[:, None], info[r]["scale"][torch.as_tensor(idx)])
                tgt = tgt4[:, 0]
                n_aug[r] += n_a
            absent_t, n_chip = info[r]["absent"], info[r]["n_chip"]
            if absent_t is not None and n_chip > 0:
                if drop_mode == "uniform_count":
                    k = torch.randint(1, n_chip + 1, (bs, 1), device=device)
                    rk = torch.rand(bs, n_chip, device=device).argsort(1).argsort(1)
                    drop = rk >= k
                else:
                    drop = torch.rand(bs, n_chip, device=device) < drop_p
                if drop.any():
                    cond = cond.clone()
                    cond[:, :n_chip] = torch.where(
                        drop[:, :, None],
                        absent_t[None, :, None].expand(bs, n_chip, cond.shape[-1]).to(cond.dtype),
                        cond[:, :n_chip])
            sym, pairs = split_cond(cond, off, nm)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(sym, pairs, None, None, ref, res=r)
            loss = hic_loss(pred[:, 0].float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            loss_sum[r] += loss.item()
            loss_n[r] += 1
            if is_main and time.time() - last_log >= 60:
                log.info("epoch %d step %d  %.1f min", epoch, step, (time.time() - t0) / 60)
                last_log = time.time()
        stop = False
        row = {"epoch": epoch, "step": step}
        for r in reslist:
            row[f"loss_{r}"] = loss_sum[r] / max(1, loss_n[r])
            row[f"steps_{r}"] = loss_n[r]
            if info[r]["aug"] is not None:
                row[f"aug_{r}"] = n_aug[r]
            if ref_p > 0:
                row[f"refdrop_{r}"] = n_refdrop[r]
        save("checkpoint-latest")
        if is_main:
            vs, vs0 = [], []
            for c in cfgs:
                r = int(c.data.resolution)
                vb = max(1, min(info[r]["batch"], vdata[r]["target"].shape[0]))
                v = val_akita_median(_AtRes(model, r), vdata[r], device, vb,
                                     info[r]["offset"], info[r]["n_motifs"],
                                     cond_mode=c.model.cond_mode, use_ref=use_ref)
                row[f"val_{r}"] = v
                vs.append(v)
                if ref_p > 0:
                    # the same windows with the reference plane at 0: what the checkpoint gives
                    # from the tracks alone; logged, not used for model selection
                    v0 = val_akita_median(_AtRes(model, r),
                                          {**vdata[r], "ref": torch.zeros_like(vdata[r]["ref"])},
                                          device, vb, info[r]["offset"], info[r]["n_motifs"],
                                          cond_mode=c.model.cond_mode, use_ref=use_ref)
                    row[f"valnoref_{r}"] = v0
                    vs0.append(v0)
            row["val_mean"] = float(np.mean(vs))
            if ref_p > 0:
                row["valnoref_mean"] = float(np.mean(vs0))
            imp = row["val_mean"] > best
            log.info("epoch %d done: " + "  ".join(f"{r // 1000}kb loss %.4f val %.4f"
                                                  for r in reslist) + "  mean %.4f%s%s",
                     epoch, *[x for r in reslist for x in (row[f"loss_{r}"], row[f"val_{r}"])],
                     row["val_mean"],
                     f"  (no reference {row['valnoref_mean']:.4f})" if ref_p > 0 else "",
                     "  <- best" if imp else "")
            if imp:
                best = row["val_mean"]
                no_imp = 0
                save("checkpoint-best")
            else:
                no_imp += 1
            curve.append(row)
            with open(out_dir / "learning_curve.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(curve)
            stop = no_imp >= patience
            if stop:
                log.info("early stop at epoch %d (best val %.4f, %d epochs no improvement)",
                         epoch, best, no_imp)
        if stop_flag(stop):
            break
    if is_main and epoch >= max_epochs and no_imp < patience:
        log.info("early stop at epoch %d (best val %.4f, %d epochs no improvement)",
                 epoch, best, no_imp)
    if is_main:
        log.info("done. %d epochs, %d steps, best mean val %.4f", epoch, step, best)
    if is_ddp:
        dist.destroy_process_group()


def write_views(out_dir: Path, views: dict[int, Path], ckpt: str = "checkpoint-best"):
    """One run folder per resolution: <view>/best -> checkpoint-best, whose
    model.pt links to the shared weights and whose config.yaml, normalisers
    and channels.json are that resolution's (with train.output_dir = view)."""
    src = Path(out_dir) / ckpt
    for res, view in views.items():
        view = Path(view)
        ck = view / "checkpoint-best"
        ck.mkdir(parents=True, exist_ok=True)
        rd = src / res_dir(res)
        for f in rd.iterdir():
            if f.name != "config.yaml":
                shutil.copyfile(f, ck / f.name)
        cfg = Config.load(rd / "config.yaml")
        cfg.train.output_dir = str(view)
        cfg.save(ck / "config.yaml")
        link = ck / "model.pt"
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(os.path.relpath(src / "model.pt", ck))
        except OSError:                 # SMB/CIFS shares refuse symlinks
            shutil.copyfile(src / "model.pt", link)
        (view / "best").write_text("checkpoint-best")
        (view / "multires_source.json").write_text(json.dumps(
            {"shared_checkpoint": str(src), "resolution": int(res)}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--stage-dir", default=None,
                    help="memory-mapped training windows (built by --stage-only); "
                         "without it each rank decompresses its stripe into RAM")
    ap.add_argument("--stage-only", action="store_true",
                    help="write the memory-mapped training windows and exit")
    a = ap.parse_args()
    cfgs = [Config.load(c) for c in a.configs]
    if a.stage_only:
        stage_arrays(cfgs, Path(a.stage_dir))
        return
    train_multires(cfgs, Path(a.out), a.resume, a.max_epochs,
                   Path(a.stage_dir) if a.stage_dir else None)


if __name__ == "__main__":
    main()
