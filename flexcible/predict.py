"""Predict full-chromosome Hi-C (and the capture-Hi-C channel) -> .cool files.

Sample-aware: --sample picks which configured sample's tracks/cool template
to use (default: the first). A degron sample's zeroed_tracks are applied
automatically, so predicting the cohesin-degron sample already shows the
cohesin-free structure the model learned.

CLI:  python -m flexcible.predict --config configs/multicell_25kb.yaml \
          --sample gm12878 --chroms test --out pred.cool
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .capture import CaptureNormalizer
from .config import Config
from .genome import select_by_parity
from .matrix import (
    BandedAccumulator,
    MatrixNormalizer,
    hann2d,
    open_cool,
    plan_windows,
    write_cool,
)
from .model import (HiCModel, HiCUNet, SeqEncoder, build_backbone,
                    build_spatial_cond, n_cond_channels)
from .precompute import assemble_chrom, chrom_nbins_of, sample_track_files
from .sequence import window_codes
from .tracks import TrackNormalizer
from .utils import get_logger, set_seed

log = get_logger()


def _resolve_ckpt(cfg: Config, override):
    if override:
        return Path(override)
    out = Path(cfg.train.output_dir)
    for ptr in ("best", "latest"):
        p = out / ptr
        if p.exists():
            log.info("using %s checkpoint", ptr)
            return out / p.read_text().strip()
    raise FileNotFoundError(f"no checkpoint in {out}")


def _resolve_chroms(spec, available, cfg):
    if spec in ("odd", "even", "all"):
        return select_by_parity(available, spec)
    if spec in ("test", "train"):
        par = cfg.data.test_parity if spec == "test" else cfg.data.train_parity
        return select_by_parity(available, par)
    return [c.strip() for c in spec.split(",") if c.strip()]


def scaled_window(cfg: Config, bin_size: int) -> tuple[int, int]:
    """Window size and stride in bins at `bin_size` that keep the genomic span of
    the configured window (size * data.resolution); the size is a multiple of
    2 ** model.depth so that every U-Net pooling divides it."""
    old = int(cfg.data.resolution)
    mult = 2 ** int(cfg.model.depth)
    size = max(mult, int(round(cfg.window.size * old / bin_size / mult)) * mult)
    stride = max(1, min(size, int(round(cfg.window.stride * old / bin_size))))
    return size, stride


def check_bin_size_model(cfg: Config) -> None:
    """--bin-size is for models with the continuous bin-size embedding only; a lookup embedding
    or a single-resolution model knows its trained bin sizes and nothing else."""
    m = cfg.model
    if not m.resolutions:
        raise ValueError("--bin-size needs a model trained at several bin sizes "
                         "(model.resolutions); this model predicts at "
                         f"{cfg.data.resolution} bp only")
    if m.res_embed_mode != "continuous":
        raise ValueError("--bin-size needs a model with the continuous bin-size embedding "
                         "(model.res_embed_mode: continuous); this model uses a lookup "
                         f"embedding and predicts only its trained bin sizes {m.resolutions}")
    if m.prior:
        raise ValueError("--bin-size is not available for cascade models (model.prior)")


def apply_bin_size(cfg: Config, bin_size: int) -> int:
    """Switch a bin-size-conditioned model to `bin_size` (bp). Changes cfg.data.resolution and
    the window (same genomic span). Returns the previous resolution."""
    check_bin_size_model(cfg)
    if int(bin_size) < 1:
        raise ValueError(f"--bin-size must be positive, got {bin_size}")
    old = int(cfg.data.resolution)
    cfg.window.size, cfg.window.stride = scaled_window(cfg, int(bin_size))
    cfg.data.resolution = int(bin_size)
    return old


def nearest_trained(resolutions, bin_size: int) -> int:
    """The trained bin size closest to `bin_size` on the log scale."""
    import math

    return min((int(r) for r in resolutions), key=lambda r: abs(math.log2(r / bin_size)))


def norm_dir_for(ckpt: Path, resolutions, bin_size: int) -> Path:
    """Folder with the normalisers of the trained bin size nearest to `bin_size`: the
    res_<bp> folder of the shared multi-resolution checkpoint that the view `ckpt` points to
    (multires_source.json beside it); else the checkpoint's own."""
    near = nearest_trained(resolutions, bin_size)
    for cand in (ckpt.parent / "multires_source.json", ckpt / "multires_source.json"):
        if cand.exists():
            shared = Path(json.loads(cand.read_text())["shared_checkpoint"])
            if not shared.is_absolute():   # relative to the json (released checkpoints)
                shared = (cand.parent / shared).resolve()
            d = shared / f"res_{near}"
            if (d / "norm_tracks.json").exists():
                log.info("bin size %d bp: normalisers of the nearest trained bin size, %d bp (%s)",
                         bin_size, near, d)
                return d
    if (ckpt / f"res_{near}" / "norm_tracks.json").exists():
        return ckpt / f"res_{near}"
    log.warning("bin size %d bp: no normalisers of the nearest trained bin size (%d bp) beside "
                "%s; using the checkpoint's own", bin_size, near, ckpt)
    return ckpt


def grid_cool_at(template_cool: str, bin_size: int, path: Path) -> str:
    """A pixel-free cool with the chromosomes of `template_cool` binned at `bin_size`: the
    prediction grid when the bin size differs from the sample's cool."""
    import cooler
    import pandas as pd

    chromsizes = open_cool(template_cool).chromsizes
    bins = cooler.util.binnify(chromsizes, int(bin_size))
    one = pd.DataFrame({"bin1_id": [0], "bin2_id": [0], "count": [0.0]})   # cooler needs a pixel table
    cooler.create_cooler(str(path), bins=bins, pixels=one, dtypes={"count": "float64"})
    return str(path)


def load_norm(ckpt: Path, stem: str, sample_name: str, cls,
              borrow_from: str | None = None):
    """Per-sample normaliser, with an explicit path for UNSEEN cell types.

    A sample that was not in training has no fitted normaliser. Since the whole
    point is to predict cell types the model never saw, we borrow one rather
    than fail: the caller's choice if given, otherwise any sibling, and we say
    loudly which one was used because it sets the output count scale.
    """
    p = ckpt / f"{stem}_{sample_name}.json"
    if p.exists():
        return cls.load(p)
    legacy = ckpt / f"{stem}.json"
    if legacy.exists():
        return cls.load(legacy)
    if borrow_from:
        q = ckpt / f"{stem}_{borrow_from}.json"
        if q.exists():
            log.warning("sample %r was not in training; borrowing the %r "
                        "%s normaliser", sample_name, borrow_from, stem)
            return cls.load(q)
    siblings = sorted(ckpt.glob(f"{stem}_*.json"))
    if not siblings:
        raise FileNotFoundError(f"no {stem} normaliser in {ckpt}")
    log.warning("sample %r was not in training; borrowing %s (contact scale "
                "comes from that sample, so absolute counts are indicative "
                "only -- structure is unaffected)",
                sample_name, siblings[0].name)
    return cls.load(siblings[0])


def build_model(cfg: Config, ckpt: Path, device):
    channels = json.loads((ckpt / "channels.json").read_text())
    n_sym = len(channels["sym_rows"])
    n_motifs = len(channels["motif_ids"])
    if cfg.train.select_tracks is not None or cfg.train.select_motifs is not None:
        from .train import _select_rows       # model trained on a subset of the rows
        _, _, n_motifs, n_sym = _select_rows(cfg, channels)
    n_out = 2 if channels["has_capture"] else 1
    n_planes = (2 if channels.get("capture_input") else 0) + cfg.model.reference_planes
    mode = cfg.model.cond_mode
    emb = cfg.sequence.embed_dim if cfg.sequence.uses_encoder else 0
    encoder = (SeqEncoder(embed_dim=emb, width=cfg.sequence.enc_width)
               if cfg.sequence.uses_encoder else None)
    from .model import make_model
    model = make_model(cfg, n_cond_channels(n_sym + emb, n_motifs, n_planes, mode),
                       n_out, encoder).to(device)
    weights = ckpt / "model.pt"
    if not weights.exists():   # a view of a multi-resolution checkpoint: weights in the shared folder
        for cand in (ckpt.parent / "multires_source.json", ckpt / "multires_source.json"):
            if cand.exists():
                shared = Path(json.loads(cand.read_text())["shared_checkpoint"])
                if not shared.is_absolute():
                    shared = (cand.parent / shared).resolve()
                weights = shared / "model.pt"
                break
    sd = torch.load(weights, map_location=device)
    # older bare-backbone checkpoints have no "backbone." prefix
    if not any(k.startswith("backbone.") for k in sd):
        sd = {f"backbone.{k}": v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, channels


def _batch_seq(fasta, chrom, starts, size, res, device, revcomp=False):
    """One-hot DNA (len(starts), 4, size*res) for a batch of windows, for the
    sequence encoder; N and past-chromosome-end padding are all-zero."""
    L = size * res
    oh = torch.zeros(len(starts), 4, L)
    ar = torch.arange(L)
    for k, s in enumerate(starts):
        c = torch.from_numpy(
            window_codes(fasta, chrom, s, size, res, revcomp=revcomp).astype(np.int64))
        m = c < 4
        oh[k, c[m], ar[m]] = 1.0
    return oh.to(device)


def predict(cfg: Config, chroms_spec, out_path, checkpoint=None, batch=8,
            pchic_out: str | None = None, sample_name: str | None = None,
            zero_tracks: list[str] | None = None, capture_input: bool = False,
            tta: bool = False, obs_exp_out: bool = False,
            capture_cool: str | None = None,
            zero_regions: list[tuple[str, str, int, int]] | None = None,
            renormalize_tracks: bool = False, bigwig_dir: str | None = None,
            cell: str | None = None, reference: str | None = None,
            prior: str | None = None, extra_bigwig_dirs: list[str] | None = None,
            diff_group: str | None = None, bin_size: int | None = None,
            no_reference: bool = False,
            absent_tracks: list[str] | None = None):
    """absent_tracks: tracks declared NOT MEASURED genome-wide (the normalised row is set to the
            track's absent value, the encoding of track dropout in training), unlike zero_tracks
            and zero_regions, which set the raw signal to 0 (measured, no binding).
    no_reference: reference-map models read a zero reference plane (the expected level in
            the normalised space) instead of the reference map, so the prediction rests on the
            ChIP tracks alone (the reference map is still opened for its shape).
    bin_size: predict at this bin size (bp) with a model that has the continuous bin-size
    embedding (the window keeps its genomic span; the reference map must be at this bin size;
    the track and contact normalisers are those of the nearest trained bin size, or refitted
    on the sample with renormalize_tracks). Refused for any other model.
    bigwig_dir: predict a cell from this folder of ChIP bigWigs (tracks it
    lacks enter as absent) with the configured sample's bin grid and
    normalisers; `cell` names it (default: the sample name, or the folder
    name with bigwig_dir). Reference-map models read `reference`, else the
    cell's own training reference, else the map embedded in the checkpoint
    (flexcible.reference.resolve_reference), looked up per window."""
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_ckpt(cfg, checkpoint)
    log.info("loading %s", ckpt)
    sample = cfg.get_sample(sample_name)
    norm_src = ckpt
    grid_tmp = None
    if bin_size is not None:
        check_bin_size_model(cfg)
    if bin_size is not None and int(bin_size) != int(cfg.data.resolution):
        import copy

        cfg = copy.deepcopy(cfg)                      # the caller's config stays at its resolution
        resolutions = list(cfg.model.resolutions or [])
        old = apply_bin_size(cfg, int(bin_size))      # refuses lookup and single-resolution models
        if cfg.model.reference != "none" and not reference:
            raise ValueError(f"--bin-size {bin_size} needs --reference: a reference O/E cool at "
                             f"{bin_size} bp (the configured and embedded consensus maps are at "
                             f"{old} bp)")
        log.info("bin size %d bp (trained %s, config %d bp): window %d bins, stride %d",
                 bin_size, resolutions, old, cfg.window.size, cfg.window.stride)
        if int(bin_size) not in resolutions:
            log.warning("%d bp was not a training bin size (%s): outside the trained range this is "
                        "extrapolation and untested", bin_size, resolutions)
        norm_src = norm_dir_for(ckpt, resolutions, int(bin_size))
        grid_tmp = Path(out_path).with_name(Path(out_path).name + f".grid{bin_size}.cool")
        sample = type(sample)(**{**sample.__dict__})
        sample.cool_path = grid_cool_at(sample.cool_path, int(bin_size), grid_tmp)
    if zero_tracks:
        sample = type(sample)(**{**sample.__dict__})
        sample.zeroed_tracks = sorted(set(sample.zeroed_tracks) | set(zero_tracks))
        log.info("zeroing tracks at inference: %s", sample.zeroed_tracks)
    missing_new: list[str] = []
    if bigwig_dir:
        sample = type(sample)(**{**sample.__dict__})
        sample.bigwig_dir = str(bigwig_dir)
        sample.extra_bigwig_dirs = [str(d) for d in (extra_bigwig_dirs or [])]
        have = {q.stem.upper() for d in [bigwig_dir, *sample.extra_bigwig_dirs]
                for q in Path(d).glob("*.bigwig")}
        missing_new = [t for t in cfg.track_names if t.upper() not in have]
        sample.missing_tracks = list(missing_new)
        cell = cell or Path(bigwig_dir).name
        log.info("cell %s from %s; absent tracks: %s", cell, bigwig_dir, missing_new)
        if cfg.data.diff_marks:
            from .diffmarks import configure_unseen
            configure_unseen(cfg, sample, cell or Path(bigwig_dir).name, diff_group)
    cell = cell or sample.name
    use_seq = cfg.sequence.uses_encoder
    fasta = sample.fasta or cfg.sequence.fasta
    track_norm = TrackNormalizer.load(norm_src / "norm_tracks.json")
    mat_norm = load_norm(norm_src, "norm_matrix", sample.name, MatrixNormalizer)
    model, channels = build_model(cfg, ckpt, device)
    ref_clr = None
    if cfg.model.reference != "none":
        from .reference import open_reference, resolve_reference

        ref_path = resolve_reference(cfg, ckpt, cell, reference)
        ref_clr = open_reference(ref_path, cfg.data.resolution,
                                 dict(open_cool(sample.cool_path).chromsizes))
        log.info("reference map for %s: %s (%s)", cell, ref_path, cfg.model.reference)
    prior_clr = None
    if cfg.model.prior:
        own = next((s for s in cfg.data.samples if s.name == cell), None)
        pp = prior or (cfg.sample_prior(own) if own else cfg.data.prior_cool)
        if not pp:
            raise ValueError(f"{cell}: cascade model (model.prior) needs a prior map")
        prior_clr = open_cool(pp)
        log.info("prior map for %s: %s", cell, pp)
    if renormalize_tracks:
        # An unseen cell type usually comes from a different processing
        # pipeline (base-resolution fold-change vs pre-binned signal, other
        # depth and dynamic range). Re-fitting the per-track statistics on
        # THIS sample puts its assays on the footing the model expects, so the
        # comparison is about biology rather than about file provenance.
        log.info("re-fitting track statistics on this sample's own signal")
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    sel_rows = None
    if cfg.train.select_tracks is not None or cfg.train.select_motifs is not None:
        from .train import _select_rows
        sel_rows, offset, n_motifs, _ = _select_rows(cfg, channels)
    cap_norm = None
    if channels["has_capture"] and (pchic_out or capture_input or capture_cool):
        cap_norm = load_norm(ckpt, "norm_capture",
                             sample.name if sample.capture_interactions
                             else channels.get("samples", ["default"])[0],
                             CaptureNormalizer)
    cap_data = None
    if capture_input:
        if not channels.get("capture_input"):
            raise ValueError(
                "--capture-input given but this checkpoint was not trained "
                "with capture.use_as_input"
            )
        if not sample.capture_interactions:
            raise ValueError(
                f"--capture-input given but sample {sample.name!r} has no "
                "capture_interactions file"
            )
        from .capture import CaptureData

        cap_data = CaptureData(sample.capture_interactions, cfg.capture.format,
                               cfg.capture.min_score, baits=sample.capture_baits)
        log.info("feeding capture Hi-C as conditioning input")
    cap_clr = None
    if capture_cool:
        if not channels.get("capture_input"):
            raise ValueError(
                "--capture-cool given but this checkpoint was not trained "
                "with capture.use_as_input"
            )
        if cap_data is not None:
            raise ValueError("--capture-input and --capture-cool are exclusive")
        cap_clr = open_cool(capture_cool)
        log.info("feeding PREDICTED capture map %s as conditioning input "
                 "(dense mask)", capture_cool)

    clr = open_cool(sample.cool_path)
    chroms = _resolve_chroms(chroms_spec, list(clr.chromnames), cfg)
    log.info("predicting sample %s, chroms: %s", sample.name, chroms)

    track_files = sample_track_files(cfg, sample)
    if zero_regions:
        if cfg.data.diff_marks:
            log.warning("regional knockouts act on the raw tracks only; the reference/difference "
                        "rows of a diff-marks model keep the unperturbed values")
        for t, ch, a, b in zero_regions:
            if t in channels["tracks"]:
                continue
            if t.startswith("motif:") and any(
                    r.startswith(t) for r in channels["sym_rows"]):
                continue
            raise ValueError(
                f"unknown track {t!r} in a regional knockout (assays: "
                f"{channels['tracks']}; motifs: "
                f"{['motif:' + m for m in channels['motif_ids']]})")
        log.info("regional knockouts: %s",
                 [f"{t}@{ch}:{a}-{b}" for t, ch, a, b in zero_regions])
    size = cfg.window.size
    w2d = hann2d(size) if cfg.predict.merge == "hann" else np.ones((size, size), np.float32)
    chrom_pixels: dict = {}
    chrom_pixels_cap: dict = {}

    if renormalize_tracks:
        from .precompute import assemble_chrom as _ac

        fit_rows, valid = [], []
        for ch in chroms:
            nbf = chrom_nbins_of(clr, ch)
            fit_rows.append(_ac(cfg, sample, track_files, ch, nbf))
            dead = set(sample.missing_tracks) | set(sample.zeroed_tracks)
            v = np.ones(fit_rows[-1].shape[0], dtype=bool)
            for i, tname in enumerate(cfg.track_names):
                if tname in dead:
                    v[i] = False
            valid.append(v)
        track_norm = TrackNormalizer(
            transform=track_norm.transform, norm=track_norm.norm,
            smooth_bins=track_norm.smooth_bins,
            passthrough=track_norm.passthrough).fit(fit_rows, valid)

    for chrom in chroms:
        nb = chrom_nbins_of(clr, chrom)
        raw = assemble_chrom(cfg, sample, track_files, chrom, nb)
        if zero_regions:
            # knock the assay out only inside the interval, in RAW space, so
            # the normaliser maps it to the same "absent" level the degron and
            # missing-assay samples are trained with
            from .genome import match_name as _mn

            res_bp = cfg.data.resolution
            for t, ch, a, b in zero_regions:
                if _mn(ch, [chrom]) is None and _mn(chrom, [ch]) is None:
                    continue
                if t.startswith("motif:"):
                    # remove the BINDING SITE: the model no longer sees a
                    # recognisable motif here, as if the sequence were changed
                    rows = [i for i, r in enumerate(channels["sym_rows"])
                            if r.startswith(t)]
                else:
                    # remove the PROTEIN only: the site stays in the DNA and
                    # the motif channels still fire, which is degron semantics
                    rows = [channels["tracks"].index(t)]
                lo, hi = max(0, a // res_bp), min(nb, -(-b // res_bp))
                if hi > lo:
                    for r in rows:
                        raw[r, lo:hi] = 0.0
        sym_full = track_norm.apply(raw)
        if absent_tracks:
            absent_v = track_norm.absent_values()
            for t in absent_tracks:
                k = cfg.track_names.index(t)
                sym_full[k] = absent_v[k]
        if missing_new:
            absent = track_norm.absent_values()
            for t in missing_new:
                k = cfg.track_names.index(t)
                sym_full[k] = absent[k]
        if sel_rows is not None:
            sym_full = sym_full[sel_rows]
        wins = plan_windows(nb, size, cfg.window.stride, chrom)
        acc = BandedAccumulator(nb, band=size)
        acc_cap = BandedAccumulator(nb, band=size) if (pchic_out and cap_norm) else None
        for i in tqdm(range(0, len(wins), batch), desc=f"{sample.name}:{chrom}"):
            bw = wins[i : i + batch]
            conds = []
            for w in bw:
                ct = sym_full[:, w.start : w.start + size]
                if ct.shape[1] < size:
                    ct = np.concatenate(
                        [ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
                conds.append(ct)
            cond = torch.from_numpy(np.stack(conds)).to(device)
            pairs = None
            if n_motifs:
                b = cond.shape[0]
                pairs = cond[:, offset : offset + 2 * n_motifs].reshape(
                    b, n_motifs, 2, size)
            planes = None
            if channels.get("capture_input"):
                b = cond.shape[0]
                if cap_data is not None:
                    pl = np.empty((b, 2, size, size), dtype=np.float32)
                    for k, w in enumerate(bw):
                        cb, bt = cap_data.window_block(
                            chrom, w.start, size, cfg.data.resolution)
                        pl[k, 0] = cap_norm.encode(cb)
                        pl[k, 1] = np.maximum(bt[:, None], bt[None, :])
                    planes = torch.from_numpy(pl).to(device)
                elif cap_clr is not None:
                    from .matrix import fetch_diagonal_window as _fdw

                    pl = np.empty((b, 2, size, size), dtype=np.float32)
                    for k, w in enumerate(bw):
                        cb = _fdw(cap_clr, chrom, w.start, size)
                        pl[k, 0] = cap_norm.encode(cb)
                        # pseudo-bait mask: rows where the predicted map has
                        # real signal (mimics the bait-limited mask the model
                        # saw during training better than all-ones)
                        pb = (cb.max(axis=1) > 1.0).astype(np.float32)
                        pl[k, 1] = np.maximum(pb[:, None], pb[None, :])
                    planes = torch.from_numpy(pl).to(device)
                else:
                    planes = torch.full((b, 2, size, size), -1.0, device=device)
                    planes[:, 1] = 0.0
            seq_oh = (_batch_seq(fasta, chrom, [w.start for w in bw], size,
                                 cfg.data.resolution, device) if use_seq else None)
            ref = None
            if ref_clr is not None:
                from .reference import reference_block

                ref = torch.from_numpy(np.stack([
                    reference_block(ref_clr, chrom, w.start, size, mat_norm)
                    for w in bw])).to(device)
                if no_reference:
                    ref = torch.zeros_like(ref)
            prior_t = None
            if prior_clr is not None:
                from .reference import prior_block

                prior_t = torch.from_numpy(np.stack([
                    prior_block(prior_clr, chrom, w.start, size, cfg.data.resolution, mat_norm)
                    for w in bw])).to(device)
            with torch.no_grad():
                out = model(cond, pairs, planes, seq_oh, ref, res=cfg.data.resolution,
                            prior=prior_t)
                if tta:
                    from .dataset import reverse_cond, split_cond as _sc

                    cond_r = reverse_cond(cond, offset, n_motifs)
                    sym_r, pairs_r = _sc(cond_r, offset, n_motifs)
                    planes_r = (torch.flip(planes, dims=[-1, -2])
                                if planes is not None else None)
                    seq_r = (_batch_seq(fasta, chrom, [w.start for w in bw], size,
                                        cfg.data.resolution, device, revcomp=True)
                             if use_seq else None)
                    ref_r = torch.flip(ref, dims=[-1, -2]) if ref is not None else None
                    pr_r = (torch.flip(prior_t, dims=[-1, -2])
                            if prior_t is not None else None)
                    out_r = model(sym_r, pairs_r, planes_r, seq_r, ref_r,
                                  res=cfg.data.resolution, prior=pr_r)
                    out = 0.5 * (out + torch.flip(out_r, dims=[-1, -2]))
                imgs = out.float().cpu().numpy()
            for w, img in zip(bw, imgs):
                if obs_exp_out:
                    block = mat_norm.decode_oe(img[0])
                else:
                    block = mat_norm.decode(img[0])
                    # Counts cannot be negative, but `prescaled` targets are
                    # log(obs/exp) where most of the signal IS negative (-2 is
                    # the strongest depletion). Clamping there deletes the
                    # majority of the prediction and leaves a matrix that
                    # cannot even represent the target space.
                    if mat_norm.transform != "prescaled":
                        block = np.maximum(block, 0.0)
                acc.add_block(w.start, 0.5 * (block + block.T), w2d)
                if acc_cap is not None:
                    cb = np.maximum(cap_norm.decode(img[1]), 0.0)
                    acc_cap.add_block(w.start, 0.5 * (cb + cb.T), w2d)
        off = int(clr.extent(chrom)[0])
        chrom_pixels[chrom] = acc.to_pixels(off)
        if acc_cap is not None:
            chrom_pixels_cap[chrom] = acc_cap.to_pixels(off)

    ints = cfg.predict.integer_counts and not obs_exp_out
    write_cool(out_path, sample.cool_path, chrom_pixels, integer=ints)
    log.info("wrote %s", out_path)
    if grid_tmp is not None:
        grid_tmp.unlink(missing_ok=True)
    if pchic_out and chrom_pixels_cap:
        write_cool(pchic_out, sample.cool_path, chrom_pixels_cap)
        log.info("wrote %s (capture channel)", pchic_out)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Predict Hi-C from multi-omic tracks.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", default=None, help="sample name (default: first)")
    ap.add_argument("--chroms", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pchic-out", default=None,
                    help="also write the capture-Hi-C channel to this .cool")
    ap.add_argument("--zero-tracks", default=None,
                    help="comma-separated tracks to zero at inference "
                         "(genome-wide in-silico deletion)")
    ap.add_argument("--absent-track", action="append", default=[],
                    help="track declared absent (not measured) genome-wide, as in track dropout "
                         "(repeatable); --zero-tracks and --zero-region set the raw signal to 0 instead")
    ap.add_argument("--capture-input", action="store_true",
                    help="feed the sample's capture Hi-C as conditioning "
                         "(optional; without it the model runs tracks-only)")
    ap.add_argument("--zero-region", action="append", default=[],
                    help="regional knockout TRACK:chrom:start-end (repeatable), "
                         "e.g. H3K27ac:chr1:10000000-50000000")
    ap.add_argument("--capture-cool", default=None,
                    help="feed a (predicted) capture-score .cool as the "
                         "conditioning plane (e.g. pchicunet output); "
                         "exclusive with --capture-input")
    ap.add_argument("--renormalize-tracks", action="store_true",
                    help="re-fit per-track statistics on this sample "
                         "(use for a cell type whose tracks come from a "
                         "different processing pipeline)")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation: average with the "
                         "reversed-input prediction")
    ap.add_argument("--obs-exp", action="store_true",
                    help="write obs/expected ratios instead of counts "
                         "(requires window.transform=obs_exp; skips the "
                         "distance-decay inversion)")
    ap.add_argument("--bigwig-dir", default=None,
                    help="predict a new cell from this folder of ChIP bigWigs")
    ap.add_argument("--extra-bigwig-dir", nargs="*", default=[],
                    help="with --bigwig-dir: further folders of the cell's track bigWigs")
    ap.add_argument("--cell", default=None,
                    help="name of the predicted cell (reference choice and "
                         "leakage guard; default: sample or bigWig folder name)")
    ap.add_argument("--reference", default=None,
                    help="reference-map models: reference O/E cool (default: "
                         "the cell's training reference, else the embedded map)")
    ap.add_argument("--no-reference", action="store_true",
                    help="reference-map models: set the reference plane to 0 in the normalised space "
                         "(the expected level), so the prediction uses the ChIP tracks alone")
    ap.add_argument("--prior", default=None,
                    help="cascade models: the coarser model's prediction for this cell")
    ap.add_argument("--diff-group", choices=["cell", "aml", "hspc"], default=None,
                    help="diff-marks models with --bigwig-dir: group of the cell, which picks its "
                         "reference set (default cell)")
    ap.add_argument("--bin-size", "--binSize", dest="bin_size", type=int, default=None,
                    help="bin size in bp (for example 4000) for the one model trained at several bin "
                         "sizes (continuous bin-size embedding). Keeps the genomic span of the "
                         "window, bins the bigWigs at this size and needs --reference at this bin "
                         "size. Uses the track and contact normalisers of the nearest trained bin "
                         "size unless --renormalize-tracks is given. Trained: 2000 to 25000; other "
                         "sizes are untested extrapolation. Refused for models with a lookup "
                         "embedding or a single resolution")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    zt = [t.strip() for t in args.zero_tracks.split(",")] if args.zero_tracks else None
    zr = []
    for spec in args.zero_region:
        parts = spec.split(":")
        if parts[0] == "motif":                       # motif:ID:chrom:start-end
            track, chrom, span = f"motif:{parts[1]}", parts[2], parts[3]
        else:                                         # TRACK:chrom:start-end
            track, chrom, span = parts
        a, b = (int(x) for x in span.replace(",", "").split("-"))
        zr.append((track, chrom, a, b))
    predict(Config.load(args.config), args.chroms, args.out, args.checkpoint,
            args.batch_size, args.pchic_out, args.sample, zt, args.capture_input,
            args.tta, args.obs_exp, args.capture_cool, zr or None,
            args.renormalize_tracks, args.bigwig_dir, args.cell, args.reference,
            args.prior, args.extra_bigwig_dir, args.diff_group, args.bin_size,
            args.no_reference, args.absent_track or None)


if __name__ == "__main__":
    main()
