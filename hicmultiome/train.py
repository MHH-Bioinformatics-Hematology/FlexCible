"""Train the multi-omic Hi-C U-Net -- single GPU or multi-GPU (DDP).

Multi-GPU: launch with torchrun, one process per GPU; the window cache is small
enough that every rank keeps its own stripe fully resident on its GPU:

    torchrun --nproc_per_node=2 -m hicmultiome.train --config configs/multiome_25kb.yaml

Single GPU:  python -m hicmultiome.train --config configs/multiome_25kb.yaml

Loss: L1+MSE on the Hi-C channel, plus (when capture data is configured) a
bait-masked L1+MSE on the capture-Hi-C channel weighted by
capture.loss_weight. Validation metric (checkpoint selection): mean
per-genomic-distance Pearson of the Hi-C channel on held-out windows.
A learning-curve CSV and PNG are written to the output dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import Config
from .dataset import WindowDataset, reverse_cond, split_cond
from .losses import build_loss
from .model import (HiCModel, HiCUNet, SeqEncoder, build_backbone,
                    build_spatial_cond, n_cond_channels)
from .sequence import window_codes
from .utils import get_logger, set_seed

log = get_logger()


def ddp_env():
    """(rank, world_size, local_rank); (0, 1, 0) when not under torchrun."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def _stack(cache_dir, split, val_fraction, want_val, seed):
    ds = WindowDataset(cache_dir, split, val_fraction, want_val, seed)
    if len(ds) == 0:
        return None, ds
    n, w = len(ds), ds.window_size
    tg = torch.empty((n, ds.n_out, w, w), dtype=torch.float32)
    cd = torch.empty((n, ds.n_sym, w), dtype=torch.float32)
    bt = torch.zeros((n, w), dtype=torch.float32)
    for path, pairs in ds.grouped_index().items():        # one decompress each
        z = np.load(path)
        rows = np.array([r for r, _ in pairs])
        loc = np.array([l for _, l in pairs])
        tg[rows] = torch.from_numpy(z["target"][loc].astype(np.float32))
        cd[rows] = torch.from_numpy(z["cond"][loc].astype(np.float32))
        if "bait" in z:
            bt[rows] = torch.from_numpy(z["bait"][loc].astype(np.float32))
    return {"target": tg, "cond": cd, "bait": bt}, ds


def _coords(ds):
    """Per-window (chrom, start_bin, sample) aligned to the stacked tensors.
    `_stack` fills row r from ds._index[r], so this order matches."""
    chroms = [c for _, _, c, _, _ in ds._index]
    starts = np.array([s for _, _, _, s, _ in ds._index], dtype=np.int64)
    samples = [sm for _, _, _, _, sm in ds._index]
    return chroms, starts, samples


def _fasta_map(cfg: Config) -> dict:
    """sample name -> FASTA path (per-sample override, else sequence.fasta).
    Includes a 'default' entry for caches whose windows carry no sample name."""
    m = {s.name: (s.fasta or cfg.sequence.fasta) for s in cfg.data.samples}
    m.setdefault("default", cfg.sequence.fasta)
    return m


def batch_onehot(coords, idx, rmask, W, binsize, fasta_map, device):
    """One-hot DNA (len(idx), 4, W*binsize) for the given rank-local rows.
    `rmask` (bool per row) requests the reverse complement, matching the
    reverse-strand augmentation applied to the rest of the conditioning.

    The CPU only slices raw base codes (cheap); the small uint8 buffer is sent to
    the GPU and the one-hot scatter runs there, so the GPU is not stalled waiting
    on a Python one-hot loop."""
    chroms, starts, samples = coords
    L = W * binsize
    codes = np.empty((len(idx), L), dtype=np.uint8)
    for k, i in enumerate(idx):
        codes[k] = window_codes(fasta_map[samples[i]], chroms[i], int(starts[i]),
                                W, binsize, revcomp=bool(rmask[k]))
    c = torch.from_numpy(codes).to(device, non_blocking=True).long()   # (B, L)
    oh = torch.zeros(len(idx), 4, L, device=device)
    oh.scatter_(1, c.clamp(max=3).unsqueeze(1), 1.0)   # 1 at [b, base, l]
    oh.mul_((c < 4).unsqueeze(1).to(oh.dtype))         # N (code 4) -> all zero
    return oh


def masked_pair_loss(pred, tgt, bait):
    """L1+MSE over pixels where either bin has bait coverage."""
    mask = torch.maximum(bait[:, :, None], bait[:, None, :])   # (B, W, W)
    denom = mask.sum().clamp_min(1.0)
    d = (pred - tgt) * mask
    return (d.abs().sum() + d.pow(2).sum()) / denom


def capture_planes(tgt, bait, present):
    """(B,2,W,W) capture-input planes: encoded capture map + bait-pair mask.

    present: (B,) float 0/1 -- windows with 0 get the 'absent' encoding
    (plane=-1, mask=0), which is also the inference default without capture."""
    p = present[:, None, None]
    plane = tgt[:, 1] * p + (-1.0) * (1 - p)
    mask = torch.maximum(bait[:, :, None], bait[:, None, :]) * p
    return torch.stack([plane, mask], dim=1)


@torch.no_grad()
def val_distance_pearson(model, data, device, batch, offset, n_motifs, maxd=64,
                         capin=False, cond_mode="full", seq_fn=None):
    """Validation in the tracks-only mode (capture plane absent) -- the honest
    default, since prediction must not require capture data. `seq_fn(i, n)`, when
    given, returns the one-hot DNA for val rows [i:i+n] for the sequence encoder."""
    if data is None or data["target"].shape[0] == 0:
        return float("nan")
    model.eval()
    preds = []
    for i in range(0, data["target"].shape[0], batch):
        tgt_b = data["target"][i : i + batch].to(device)
        sym, pairs = split_cond(data["cond"][i : i + batch].to(device), offset, n_motifs)
        planes = None
        if capin:
            absent = torch.zeros(tgt_b.shape[0], device=device)
            planes = capture_planes(tgt_b, data["bait"][i : i + batch].to(device), absent)
        seq_oh = seq_fn(i, tgt_b.shape[0]) if seq_fn is not None else None
        preds.append(model(sym, pairs, planes, seq_oh).cpu())
    pred = torch.cat(preds)[:, 0].numpy()
    true = data["target"][:, 0].cpu().numpy()
    w = true.shape[-1]
    rs = []
    for d in range(1, min(maxd, w)):
        x = np.concatenate([np.diagonal(true[k], d) for k in range(true.shape[0])])
        y = np.concatenate([np.diagonal(pred[k], d) for k in range(pred.shape[0])])
        if x.std() > 1e-8 and y.std() > 1e-8:
            rs.append(np.corrcoef(x, y)[0, 1])
    model.train()
    return float(np.nanmean(rs)) if rs else float("nan")


def _plot_curve(rows, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(steps, [r["loss"] for r in rows], color="#888", lw=1, label="train loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    vs = [(r["step"], r["val_pearson"]) for r in rows if r["val_pearson"] == r["val_pearson"]]
    if vs:
        ax2.plot(*zip(*vs), color="#2c7fb8", lw=2, marker="o", ms=3,
                 label="val dist-Pearson")
        ax2.set_ylabel("val per-distance Pearson")
    fig.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def train(cfg: Config):
    rank, world, local_rank = ddp_env()
    is_ddp = world > 1
    if is_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0
    set_seed(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    out_dir = Path(cfg.train.output_dir)
    cache_dir = Path(cfg.train.cache_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    data, ds = _stack(str(cache_dir), "train", cfg.train.val_fraction, False, cfg.train.seed)
    vdata, vds = _stack(str(cache_dir), "train", cfg.train.val_fraction, True, cfg.train.seed)
    if data is None:
        raise RuntimeError("no training windows; run hicmultiome.precompute first")
    offset, n_motifs = ds.motif_row_offset(), ds.n_motifs

    use_seq = cfg.sequence.uses_encoder
    binsize = int(cfg.data.resolution)
    fasta_map = _fasta_map(cfg)
    W = ds.window_size
    # full-set window coordinates (aligned to the stacked rows), for the encoder
    full_coords = _coords(ds) if use_seq else None

    # data-efficiency knob: keep every k-th window (effective stride =
    # window.stride * k). A cheap way to sweep training-set size without
    # re-precomputing the cache -- every k-th of the 50%-overlap windows is
    # exactly the coarser-stride tiling.
    k = max(1, int(cfg.train.window_subsample))
    if k > 1:
        keep = list(range(0, data["target"].shape[0], k))
        data = {kk: v[keep] for kk, v in data.items()}
        if full_coords is not None:
            ch, st, sm = full_coords
            full_coords = ([ch[i] for i in keep], st[keep], [sm[i] for i in keep])
        if is_main:
            log.info("window_subsample=%d: %d training windows (effective "
                     "stride %d bins)", k, len(keep), cfg.window.stride * k)
    N_total = data["target"].shape[0]

    # Each rank keeps its stripe of the windows resident on its GPU. Stripes are
    # trimmed to equal length so every rank runs the same number of steps --
    # the per-step stop_flag all_reduce would deadlock otherwise.
    n_even = (N_total // world) * world if world > 1 else N_total
    sl = slice(rank, n_even, world)
    tg = data["target"][sl].to(device)
    cd = data["cond"][sl].to(device)
    bt = data["bait"][sl].to(device)
    vdata_d = None
    if is_main and vdata is not None:
        vdata_d = {k2: v.to(device) for k2, v in vdata.items()}
    N = tg.shape[0]

    tr_coords = val_coords = None
    if use_seq:
        ch, st, sm = full_coords                  # rank stripe, aligned to tg/cd
        rows = list(range(rank, n_even, world))
        tr_coords = ([ch[i] for i in rows], st[rows], [sm[i] for i in rows])
        if is_main and vds is not None:
            val_coords = _coords(vds)

    n_chip = len(ds.channels["tracks"])
    absent_t = None
    if cfg.train.track_dropout > 0:
        from .tracks import TrackNormalizer

        tn = TrackNormalizer.load(cache_dir / "norm_tracks.json")
        absent_t = torch.from_numpy(tn.absent_values()[:n_chip]).to(device)

    capin = bool(cfg.capture.use_as_input and ds.n_out == 2)
    cond_mode = cfg.model.cond_mode
    emb = cfg.sequence.embed_dim if use_seq else 0
    in_ch = n_cond_channels(ds.n_sym + emb, n_motifs, 2 if capin else 0, cond_mode)
    backbone = build_backbone(cfg.model.kind, in_ch, out_ch=ds.n_out,
                              base=cfg.model.base_channels, depth=cfg.model.depth)
    encoder = (SeqEncoder(embed_dim=emb, width=cfg.sequence.enc_width)
               if use_seq else None)
    model = HiCModel(backbone, cond_mode, encoder).to(device)
    if use_seq and is_main:
        log.info("sequence encoder: dilated, embed_dim=%d width=%d (bp->%d bins)",
                 emb, cfg.sequence.enc_width, W)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    core = model.module if is_ddp else model
    if cfg.train.compile:
        # `core` deliberately keeps pointing at the uncompiled module: the
        # compiled wrapper prefixes every state-dict key with "_orig_mod.",
        # which would save checkpoints that no plain run could load back.
        # Parameters are shared, so training the wrapper trains `core`.
        model = torch.compile(model)
        if is_main:
            log.info("torch.compile enabled (first steps include compilation)")
    if is_main:
        log.info(
            "hic-multiome-unet: %d train windows (%d on this rank, world=%d), "
            "%d val, %d sym rows, %d motifs, %d cond channels, %d out channels, "
            "%.2fM params",
            N_total, N, world, 0 if vdata is None else vdata["target"].shape[0],
            ds.n_sym, n_motifs, in_ch, ds.n_out,
            sum(p.numel() for p in core.parameters()) / 1e6,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    batch = cfg.train.batch_size
    cap_w = cfg.capture.loss_weight if ds.n_out == 2 else 0.0
    hic_loss = build_loss(cfg.train.loss, max_dist=cfg.train.loss_max_dist,
                          stride=cfg.train.loss_stride)
    if is_main:
        log.info("objective: %s", cfg.train.loss)
    step = seen = epoch = 0
    best = float("-inf")

    if cfg.train.resume:
        src = Path(cfg.train.resume_from) if cfg.train.resume_from else None
        if src is None:
            for ptr in ("best", "latest"):
                q = out_dir / ptr
                if q.exists():
                    src = out_dir / q.read_text().strip()
                    break
        if src is None or not (src / "model.pt").exists():
            log.warning("resume requested but no checkpoint found; "
                        "starting from scratch")
        else:
            core.load_state_dict(torch.load(src / "model.pt", map_location=device))
            if (src / "optim.pt").exists():
                opt.load_state_dict(torch.load(src / "optim.pt",
                                               map_location=device))
            if (src / "progress.json").exists():
                pr = json.loads((src / "progress.json").read_text())
                step, epoch, best = pr["step"], pr["epoch"], pr["best"]
            log.info("resumed from %s at step %d (epoch %d, best val %.4f)",
                     src, step, epoch, best)
    t0 = last_log = time.time()
    last_eval = 0
    stop = False
    curve: list[dict] = []

    def save(name, ptr):
        if not is_main:
            return
        ck = out_dir / name
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(core.state_dict(), ck / "model.pt")
        torch.save(opt.state_dict(), ck / "optim.pt")
        (ck / "progress.json").write_text(json.dumps(
            {"step": step, "epoch": epoch, "best": best}))
        for s in list(cache_dir.glob("norm_*.json")) + [cache_dir / "channels.json"]:
            if s.exists():
                (ck / s.name).write_text(s.read_text())
        cfg.save(ck / "config.yaml")
        (out_dir / ptr).write_text(name)

    def validate(cur_loss):
        nonlocal best, last_eval
        last_eval = step
        save("checkpoint-latest", "latest")
        if not is_main:
            return
        r = float("nan")
        if vdata_d is not None:
            seq_fn = None
            if use_seq and val_coords is not None:
                seq_fn = lambda i, n: batch_onehot(
                    val_coords, list(range(i, i + n)), [False] * n, W, binsize,
                    fasta_map, device)
            r = val_distance_pearson(
                model, vdata_d, device,
                max(1, min(batch, vdata_d["target"].shape[0])), offset, n_motifs,
                capin=capin, cond_mode=cond_mode, seq_fn=seq_fn,
            )
            imp = r > best
            log.info("  [val] epoch %d step %d  dist-pearson=%.4f%s",
                     epoch, step, r, "  <- best" if imp else "")
            if imp:
                best = r
                save("checkpoint-best", "best")
        curve.append({"step": step, "loss": cur_loss, "val_pearson": r})

    def stop_flag(val: bool) -> bool:
        """Agree on stopping across ranks so DDP never deadlocks."""
        if not is_ddp:
            return val
        t = torch.tensor([1.0 if val else 0.0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.train.amp)
    scaler = torch.amp.GradScaler("cuda") if cfg.train.amp == "fp16" else None
    if amp_dtype is not None and is_main:
        log.info("mixed precision: %s autocast", cfg.train.amp)

    loss_val = float("nan")
    while step < cfg.train.max_steps and not stop:
        timed_out = bool(
            cfg.train.max_minutes and (time.time() - t0) / 60.0 >= cfg.train.max_minutes
        )
        if stop_flag(timed_out):
            if is_main:
                log.info("reached max_minutes=%.1f at step %d", cfg.train.max_minutes, step)
            break
        g = torch.Generator(device="cpu").manual_seed(cfg.train.seed * 100003 + epoch)
        perm = torch.randperm(N, generator=g).to(device)
        epoch += 1
        bs = min(batch, N)
        ep_loss = nb = 0
        b = 0
        while b + bs <= N:
            idx = perm[b : b + bs]
            cond_b, tgt, bait_b = cd[idx], tg[idx], bt[idx]
            rev_mask = None
            if cfg.train.augment_reverse:
                rmask = torch.rand(bs, device=device) < 0.5
                rev_mask = rmask
                if rmask.any():
                    cond_r = reverse_cond(cond_b, offset, n_motifs)
                    tgt_r = torch.flip(tgt, dims=[-1, -2])
                    bait_r = torch.flip(bait_b, dims=[-1])
                    m1 = rmask[:, None, None]
                    cond_b = torch.where(m1, cond_r, cond_b)
                    tgt = torch.where(rmask[:, None, None, None], tgt_r, tgt)
                    bait_b = torch.where(m1[:, :, 0], bait_r, bait_b)
            if absent_t is not None:
                drop = (torch.rand(bs, n_chip, device=device)
                        < cfg.train.track_dropout)
                if drop.any():
                    cond_b = cond_b.clone()
                    cond_b[:, :n_chip] = torch.where(
                        drop[:, :, None],
                        absent_t[None, :, None].expand(bs, n_chip, cond_b.shape[-1]),
                        cond_b[:, :n_chip])
            sym, pairs = split_cond(cond_b, offset, n_motifs)
            planes = None
            dropped = None
            if capin:
                present = (torch.rand(bs, device=device)
                           >= cfg.capture.input_dropout).float()
                planes = capture_planes(tgt, bt[idx], present)
                dropped = 1.0 - present
            seq_oh = None
            if use_seq:
                idx_cpu = idx.detach().cpu().tolist()
                rmask_cpu = (rev_mask.detach().cpu().tolist()
                             if rev_mask is not None else [False] * bs)
                seq_oh = batch_onehot(tr_coords, idx_cpu, rmask_cpu, W, binsize,
                                      fasta_map, device)
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                pred = model(cond_b, pairs, planes, seq_oh)
            pred = pred.float()
            loss = hic_loss(pred[:, 0], tgt[:, 0])
            if cap_w > 0:
                bait_eff = bt[idx]
                if dropped is not None:
                    # aux target only where the input plane was blanked --
                    # otherwise the task degenerates to copying the input
                    bait_eff = bait_eff * dropped[:, None]
                loss = loss + cap_w * masked_pair_loss(pred[:, 1], tgt[:, 1], bait_eff)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            step += 1
            seen += bs * world
            b += bs
            loss_val = loss.item()
            ep_loss += loss_val
            nb += 1
            now = time.time()
            timed_out = bool(
                cfg.train.max_minutes and (now - t0) / 60.0 >= cfg.train.max_minutes
            )
            if step >= cfg.train.max_steps or stop_flag(timed_out):
                stop = True
                break
            if is_main and now - last_log >= 15.0:
                log.info("epoch %d step %d  loss=%.4f  %.0f win/s  %.1fmin",
                         epoch, step, loss_val, seen / (now - t0), (now - t0) / 60.0)
                last_log = now
        if is_main and nb:
            log.info("epoch %d done  avg_loss=%.4f (step %d)", epoch, ep_loss / nb, step)
        if step - last_eval >= cfg.train.eval_every or stop:
            validate(loss_val)
    if step != last_eval:
        validate(loss_val)

    if is_main:
        log.info("done. %d epochs, %d steps, best val dist-pearson=%.4f",
                 epoch, step, best)
        if curve:
            with open(out_dir / "learning_curve.csv", "w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=["step", "loss", "val_pearson"])
                wtr.writeheader()
                wtr.writerows(curve)
            _plot_curve(curve, out_dir / "learning_curve.png")
            log.info("wrote %s", out_dir / "learning_curve.png")
    if is_ddp:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description="Train the multi-omic Hi-C U-Net.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(Config.load(args.config))


if __name__ == "__main__":
    main()
