"""Predict full-chromosome Hi-C (and the capture-Hi-C channel) -> .cool files.

Sample-aware: --sample picks which configured sample's tracks/cool template
to use (default: the first). A degron sample's zeroed_tracks are applied
automatically, so predicting the cohesin-degron sample already shows the
cohesin-free structure the model learned.

CLI:  python -m hicmultiome.predict --config configs/multicell_25kb.yaml \
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
    n_out = 2 if channels["has_capture"] else 1
    n_planes = 2 if channels.get("capture_input") else 0
    mode = cfg.model.cond_mode
    emb = cfg.sequence.embed_dim if cfg.sequence.uses_encoder else 0
    backbone = build_backbone(cfg.model.kind,
                              n_cond_channels(n_sym + emb, n_motifs, n_planes, mode),
                              out_ch=n_out, base=cfg.model.base_channels,
                              depth=cfg.model.depth)
    encoder = (SeqEncoder(embed_dim=emb, width=cfg.sequence.enc_width)
               if cfg.sequence.uses_encoder else None)
    model = HiCModel(backbone, mode, encoder,
                     cfg.window.size if cfg.model.dist_bias else 0).to(device)
    sd = torch.load(ckpt / "model.pt", map_location=device)
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
            renormalize_tracks: bool = False):
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_ckpt(cfg, checkpoint)
    log.info("loading %s", ckpt)
    sample = cfg.get_sample(sample_name)
    if zero_tracks:
        sample = type(sample)(**{**sample.__dict__})
        sample.zeroed_tracks = sorted(set(sample.zeroed_tracks) | set(zero_tracks))
        log.info("zeroing tracks at inference: %s", sample.zeroed_tracks)
    use_seq = cfg.sequence.uses_encoder
    fasta = sample.fasta or cfg.sequence.fasta
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    mat_norm = load_norm(ckpt, "norm_matrix", sample.name, MatrixNormalizer)
    model, channels = build_model(cfg, ckpt, device)
    if renormalize_tracks:
        # An unseen cell type usually comes from a different processing
        # pipeline (base-resolution fold-change vs pre-binned signal, other
        # depth and dynamic range). Re-fitting the per-track statistics on
        # THIS sample puts its assays on the footing the model expects, so the
        # comparison is about biology rather than about file provenance.
        log.info("re-fitting track statistics on this sample's own signal")
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
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
                               cfg.capture.min_score)
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
            smooth_bins=track_norm.smooth_bins).fit(fit_rows, valid)

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
            with torch.no_grad():
                out = model(cond, pairs, planes, seq_oh)
                if tta:
                    from .dataset import reverse_cond, split_cond as _sc

                    cond_r = reverse_cond(cond, offset, n_motifs)
                    sym_r, pairs_r = _sc(cond_r, offset, n_motifs)
                    planes_r = (torch.flip(planes, dims=[-1, -2])
                                if planes is not None else None)
                    seq_r = (_batch_seq(fasta, chrom, [w.start for w in bw], size,
                                        cfg.data.resolution, device, revcomp=True)
                             if use_seq else None)
                    out_r = model(sym_r, pairs_r, planes_r, seq_r)
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
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    zt = [t.strip() for t in args.zero_tracks.split(",")] if args.zero_tracks else None
    zr = []
    for spec in args.zero_region:
        track, chrom, span = spec.split(":")
        a, b = (int(x) for x in span.replace(",", "").split("-"))
        zr.append((track, chrom, a, b))
    predict(Config.load(args.config), args.chroms, args.out, args.checkpoint,
            args.batch_size, args.pchic_out, args.sample, zt, args.capture_input,
            args.tta, args.obs_exp, args.capture_cool, zr or None,
            args.renormalize_tracks)


if __name__ == "__main__":
    main()
