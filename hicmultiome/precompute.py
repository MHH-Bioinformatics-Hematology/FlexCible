"""Precompute windowed training data (multi-omic conditioning + targets).

Multi-sample: every sample (cell line / condition) contributes windows from
its own cool + bigwigs. Track rows listed in a sample's missing_tracks or
zeroed_tracks are set to raw zero BEFORE the transform/normalisation, so
"protein absent" is encoded as genuine no-signal (a cohesin-degron sample
with zeroed RAD21/SMC3 teaches the model contact maps without cohesin).

Memory safety: Hi-C values are read per *window* through cooler region
queries (never a dense chromosome matrix), so 5 kb / 2 kb resolutions work
within bounded RAM. Capture interactions are binned per window the same way.

Normalisers: one shared TrackNormalizer across all samples (so signal levels
are comparable and zero stays "absent"), one MatrixNormalizer per sample
(sequencing depth differs), one CaptureNormalizer per capture-bearing sample.

CLI:  python -m hicmultiome.precompute --config configs/multicell_25kb.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .capture import CaptureData, CaptureNormalizer
from .config import Config, SampleSpec
from .genome import (match_name, merge_regions, read_bed_regions, split_chroms,
                     window_region_state)
from .matrix import (
    MatrixNormalizer,
    fetch_diagonal_window,
    open_cool,
    plan_windows,
)
from .motifs import cached_motif_tracks
from .reference import assert_reference_excludes
from .sequence import cached_sequence_features
from .tracks import TrackNormalizer, discover_tracks, read_all_tracks
from .utils import get_logger

log = get_logger()


def _prep_target(block, sigma):
    """Akita-style target denoising: light 2D Gaussian smoothing."""
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(block, sigma=sigma)
    return block


def _zero_fraction(block: np.ndarray) -> float:
    return float((block == 0).mean())


_MAPPABLE_CACHE: dict = {}
_FORK_FN = None


def _fork_call(args):
    return _FORK_FN(*args)


def _pmap(fn, tasks: list, workers: int) -> list:
    """[fn(*t) for t in tasks], in order; with workers > 1 on a fork pool whose
    children inherit fn (a closure over the build state) instead of pickling it."""
    global _FORK_FN
    if workers <= 1 or len(tasks) <= 1:
        return [fn(*t) for t in tasks]
    import multiprocessing as mp

    _FORK_FN = fn
    try:
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            return pool.map(_fork_call, tasks, chunksize=1)
    finally:
        _FORK_FN = None


def load_mappable(cool_path: str):
    """Per-bin mappability mask beside a matrix, or None if absent."""
    key = str(cool_path)
    if key in _MAPPABLE_CACHE:
        return _MAPPABLE_CACHE[key]
    import os
    cand = str(cool_path).split("::")[0] + ".mappable.npz"
    m = None
    if os.path.exists(cand):
        import numpy as _np
        z = _np.load(cand)
        m = {k: z[k] for k in z.files}
    _MAPPABLE_CACHE[key] = m
    return m


def mappable_fraction(mask, chrom: str, start: int, size: int) -> float:
    """Fraction of a window's bins that are usable; 1.0 when unknown."""
    if mask is None:
        return 1.0
    from .genome import match_name
    try:
        name = match_name(chrom, list(mask.keys()))
    except Exception:
        return 1.0
    v = mask.get(name)
    if v is None:
        return 1.0
    seg = v[start:start + size]
    return float(seg.mean()) if seg.size else 0.0


ASSAY_ROW = "assay:microc"          # data.assay_indicator row (0 Hi-C, 1 Micro-C)
ASSAY_ROW_INTACT = "assay:intact"   # extra row of assay_indicator_mode "three"


def assay_classes(cfg: Config) -> list[str]:
    """Assays with an indicator row, in row order (empty when the indicator is
    off): ["microc"] in binary mode, ["intact", "microc"] in three-class mode."""
    if not cfg.data.assay_indicator:
        return []
    return ["intact", "microc"] if cfg.data.assay_indicator_mode == "three" else ["microc"]


def seq_feature_names(cfg: Config) -> list[str]:
    """channels["seq_features"]: the rows between the ChIP tracks and the motif
    rows (consumers compute the motif offset as len(tracks) + len(this)). The
    assay indicator row, when enabled, sits at the end of this block."""
    names = list(cfg.sequence.features) if cfg.sequence.fasta else []
    names += [f"assay_{a}" for a in assay_classes(cfg)]
    return names


def sym_row_names(cfg: Config) -> list[str]:
    rows = list(cfg.track_names)
    rows += [f"seq:{f}" for f in cfg.sequence.features] if cfg.any_fasta else []
    rows += [f"assay:{a}" for a in assay_classes(cfg)]
    for mid in cfg.active_motif_ids:
        rows += [f"motif:{mid}:fwd", f"motif:{mid}:rev"]
    return rows


def chrom_nbins_of(clr, chrom: str) -> int:
    ext = clr.extent(chrom)
    return int(ext[1] - ext[0])


def assemble_chrom(
    cfg: Config, sample: SampleSpec, track_files, chrom: str, nb: int
) -> np.ndarray:
    """Raw (unnormalised) stacked sym rows for one sample chromosome: (R, nb)."""
    chip = read_all_tracks(track_files, chrom, nb, cfg.data.resolution)
    dead = set(sample.missing_tracks) | set(sample.zeroed_tracks)
    for i, name in enumerate(cfg.track_names):
        if name in dead:
            chip[i] = 0.0                       # raw zero = protein absent
    parts = [chip]
    fasta = cfg.sample_fasta(sample)
    if not fasta and cfg.any_fasta:
        # another sample has a genome, so the layout carries sequence/motif
        # rows: emit zeros here rather than a differently shaped array
        if cfg.sequence.features:
            parts.append(np.zeros((len(cfg.sequence.features), nb), dtype=np.float32))
    if fasta and cfg.sequence.features:
        parts.append(
            cached_sequence_features(
                fasta, chrom, cfg.sequence.features,
                nb, cfg.data.resolution, cfg.sequence.cache_dir,
            )
        )
    for a in assay_classes(cfg):
        parts.append(np.full((1, nb), float(sample.assay == a), dtype=np.float32))
    if not fasta and cfg.any_fasta and cfg.active_motif_ids:
        parts.append(np.zeros((2 * len(cfg.active_motif_ids), nb), dtype=np.float32))
    if fasta:
        for mid in cfg.active_motif_ids:
            parts.append(
                cached_motif_tracks(
                    fasta, chrom, mid, cfg.motifs.dir,
                    nb, cfg.data.resolution, cfg.motifs.score_frac,
                    cfg.sequence.cache_dir,
                )
            )
    return np.concatenate(parts, axis=0)


def sample_track_files(cfg: Config, sample: SampleSpec):
    """discover_tracks, tolerating files absent for missing/zeroed tracks."""
    dead = set(sample.missing_tracks) | set(sample.zeroed_tracks)
    files = {n: p for n, p in discover_tracks(sample.bigwig_dir, None)}
    out = []
    for name in cfg.track_names:
        if name in files:
            out.append((name, files[name]))
        elif name in dead:
            out.append((name, None))            # read_all_tracks -> zeros
        else:
            raise FileNotFoundError(
                f"sample {sample.name!r}: track {name!r} not in "
                f"{sample.bigwig_dir} and not declared missing/zeroed"
            )
    return out


def build(cfg: Config, workers: int = 1) -> dict:
    """Fit the normalisers and cache the windows. workers > 1 runs the
    per-sample normaliser pass and the per-chromosome window pass in parallel
    processes (fork); results, their order and the cache are the same as with
    workers=1, the default."""
    rows = sym_row_names(cfg)
    holdout = read_bed_regions(cfg.data.holdout_bed) if cfg.data.holdout_bed else None
    if holdout:
        log.info("region holdout: %d chroms, %d regions",
                 len(holdout), sum(len(v) for v in holdout.values()))
    train_regions = (merge_regions(read_bed_regions(cfg.data.train_regions_bed))
                     if cfg.data.train_regions_bed else None)
    if train_regions is not None:
        log.info("training windows restricted to %d merged regions (%.1f Mb) of %s",
                 sum(len(v) for v in train_regions.values()),
                 sum(e - s for v in train_regions.values() for s, e in v) / 1e6,
                 cfg.data.train_regions_bed)
    cache = Path(cfg.train.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    samples = cfg.data.samples
    # reference map per sample (samples[].reference_cool, else data.reference_cool)
    ref_clrs = {}
    if cfg.has_reference:
        for s in samples:
            path = cfg.sample_reference(s)
            if not path:
                raise ValueError(f"sample {s.name!r}: no reference map while "
                                 "another sample has one")
            assert_reference_excludes(path, s.name)
            ref_clrs[s.name] = open_cool(path)
            if ref_clrs[s.name].binsize != cfg.data.resolution:
                raise ValueError(f"reference {path} binsize "
                                 f"{ref_clrs[s.name].binsize} != resolution "
                                 f"{cfg.data.resolution}")
            log.info("sample %s: reference map %s cached per window", s.name, path)
    prior_clrs = {}
    if cfg.has_prior:
        for s in samples:
            path = cfg.sample_prior(s)
            if not path:
                raise ValueError(f"sample {s.name!r}: no prior map while another sample has one")
            prior_clrs[s.name] = open_cool(path)
            log.info("sample %s: prior map %s (%d bp) cached per window", s.name, path,
                     prior_clrs[s.name].binsize)
    coolers, sample_chroms, sample_tracks, captures = {}, {}, {}, {}
    for s in samples:
        clr = open_cool(s.cool_path)
        if clr.binsize != cfg.data.resolution:
            raise ValueError(
                f"sample {s.name}: cool binsize {clr.binsize} != "
                f"resolution {cfg.data.resolution}"
            )
        coolers[s.name] = clr
        if cfg.data.holdout_bed:
            # every chromosome contributes to BOTH splits; which windows land
            # where is decided per window by the held-out regions
            from .genome import resolve_chroms

            allc = resolve_chroms(cfg.data.chroms, list(clr.chromnames))
            sample_chroms[s.name] = (allc, allc)
        else:
            sample_chroms[s.name] = split_chroms(
                cfg.data.chroms, list(clr.chromnames),
                cfg.data.train_parity, cfg.data.test_parity,
            )
        sample_tracks[s.name] = sample_track_files(cfg, s)
        if s.capture_interactions:
            captures[s.name] = CaptureData(
                s.capture_interactions, cfg.capture.format, cfg.capture.min_score,
                baits=s.capture_baits,
            )
        log.info("sample %s: train %s | test %s", s.name,
                 sample_chroms[s.name][0], sample_chroms[s.name][1])

    # ---- pass 1: fit normalisers on training chromosomes only ----
    log.info("fitting normalisers (train chroms, all samples)...")
    all_rows_arrays, all_valid_rows = [], []
    mat_norms: dict[str, MatrixNormalizer] = {}
    cap_norms: dict[str, CaptureNormalizer] = {}
    if workers > 1:
        # the per-chromosome sequence/motif caches are written without locking:
        # fill them once here, so the parallel passes only read them
        for s in samples:
            fasta = cfg.sample_fasta(s)
            if not fasta:
                continue
            for chrom in sorted(set(sample_chroms[s.name][0]) | set(sample_chroms[s.name][1])):
                nb = chrom_nbins_of(coolers[s.name], chrom)
                if cfg.sequence.features:
                    cached_sequence_features(fasta, chrom, cfg.sequence.features, nb,
                                             cfg.data.resolution, cfg.sequence.cache_dir)
                for mid in cfg.active_motif_ids:
                    cached_motif_tracks(fasta, chrom, mid, cfg.motifs.dir, nb,
                                        cfg.data.resolution, cfg.motifs.score_frac,
                                        cfg.sequence.cache_dir)

    def fit_sample(s: SampleSpec):
        clr = coolers[s.name]
        train_chroms, _ = sample_chroms[s.name]
        band_sample, oe_blocks, arrays, valids = [], [], [], []
        for chrom in tqdm(train_chroms, desc=f"fit:{s.name}"):
            nb = chrom_nbins_of(clr, chrom)
            arrays.append(
                assemble_chrom(cfg, s, sample_tracks[s.name], chrom, nb))
            # rows this sample actually measures (missing/zeroed excluded)
            dead = set(s.missing_tracks) | set(s.zeroed_tracks)
            valid = np.ones(arrays[-1].shape[0], dtype=bool)
            for i, tname in enumerate(cfg.track_names):
                if tname in dead:
                    valid[i] = False
            valids.append(valid)
            wins = plan_windows(nb, cfg.window.size, cfg.window.stride, chrom)
            for w in wins[:: max(1, len(wins) // 40)]:      # ~40 windows/chrom
                block = _prep_target(
                    fetch_diagonal_window(clr, chrom, w.start, w.size),
                    cfg.window.smooth_sigma)
                if cfg.window.transform == "obs_exp":
                    oe_blocks.append(block)
                else:
                    pos = block[block > 0]
                    if pos.size:
                        band_sample.append(pos[:20000])
        norm = MatrixNormalizer(
            transform=cfg.window.transform, clip_quantile=cfg.window.clip_quantile,
            headroom=cfg.window.clip_headroom,
        )
        if cfg.window.transform == "obs_exp":
            norm = norm.fit_obs_exp(oe_blocks)
        else:
            norm = norm.fit(np.concatenate(band_sample))
        return arrays, valids, norm

    fitted = _pmap(fit_sample, [(s,) for s in samples], workers)
    for s, (arrays, valids, norm) in zip(samples, fitted):
        clr = coolers[s.name]
        train_chroms, _ = sample_chroms[s.name]
        all_rows_arrays += arrays
        all_valid_rows += valids
        mat_norms[s.name] = norm
        mat_norms[s.name].save(cache / f"norm_matrix_{s.name}.json")
        log.info("  %s contact scale = %.4f", s.name, mat_norms[s.name].scale)
        if s.name in captures:
            def nb_of(c, _clr=clr):
                return chrom_nbins_of(_clr, c)
            cap_norms[s.name] = CaptureNormalizer(cfg.capture.clip_quantile).fit(
                captures[s.name].train_score_sample(
                    train_chroms, nb_of, cfg.data.resolution)
            )
            cap_norms[s.name].save(cache / f"norm_capture_{s.name}.json")
            log.info("  %s capture scale = %.4f", s.name, cap_norms[s.name].scale)

    track_norm = TrackNormalizer(
        transform=cfg.track.transform, norm=cfg.track.norm,
        smooth_bins=cfg.track.smooth_bins,
    ).fit(all_rows_arrays, all_valid_rows)
    track_norm.save(cache / "norm_tracks.json")

    channels = {
        "sym_rows": rows,
        "tracks": cfg.track_names,
        "seq_features": seq_feature_names(cfg),
        "motif_ids": cfg.active_motif_ids,
        "has_capture": bool(captures),
        "capture_input": bool(captures) and cfg.capture.use_as_input,
        "samples": [s.name for s in samples],
    }
    if ref_clrs:
        refs_by_sample = {s.name: str(cfg.sample_reference(s)) for s in samples}
        paths = set(refs_by_sample.values())
        channels["reference"] = (paths.pop() if len(paths) == 1 else refs_by_sample)
    if prior_clrs:
        channels["prior"] = {s.name: str(cfg.sample_prior(s)) for s in samples}
    (cache / "channels.json").write_text(json.dumps(channels, indent=2))

    # ---- pass 2: slice + cache windows per sample ----
    manifest = {
        "config_resolution": cfg.data.resolution,
        "window_size": cfg.window.size,
        "stride": cfg.window.stride,
        **channels,
        "shards": [],
    }

    def process(s: SampleSpec, chrom: str, split: str):
        clr = coolers[s.name]
        nb = chrom_nbins_of(clr, chrom)
        sym = track_norm.apply(
            assemble_chrom(cfg, s, sample_tracks[s.name], chrom, nb))
        cap = captures.get(s.name)
        wins = plan_windows(nb, cfg.window.size, cfg.window.stride, chrom)

        targets, conds, baits, starts, refs, priors = [], [], [], [], [], []
        prior_clr = prior_clrs.get(s.name)
        ref_clr = ref_clrs.get(s.name)
        ref_name = (match_name(chrom, list(ref_clr.chromnames))
                    if ref_clr is not None else None)
        res_bp = cfg.data.resolution
        mappable = load_mappable(s.cool_path)
        n_unmappable = 0
        n_empty = 0
        n_outside_train = 0
        for w in wins:
            if holdout is not None:
                state = window_region_state(
                    holdout, chrom, w.start * res_bp, (w.start + w.size) * res_bp)
                # train on windows that never touch a held-out region; test on
                # windows lying wholly inside one
                if split == "train" and state != "outside":
                    continue
                if split == "test" and state != "inside":
                    continue
            if split == "train" and train_regions is not None and window_region_state(
                    train_regions, chrom, w.start * res_bp,
                    (w.start + w.size) * res_bp) != "inside":
                n_outside_train += 1
                continue
            # Mappability is known from the mask alone, so this rejects the
            # window before paying for the fetch. Applies to both splits: an
            # unmappable window is no more useful as a test case.
            if (mappable_fraction(mappable, chrom, w.start, w.size)
                    < cfg.window.min_mappable_fraction):
                n_unmappable += 1
                continue
            block = fetch_diagonal_window(clr, chrom, w.start, w.size)
            # An empty window is useless in ANY split, and on a matrix whose
            # held-out regions were cut out by the publisher the whole test
            # split is empty -- previously it fetched every one of those
            # windows and wrote shards of zeros.
            if _zero_fraction(block) > cfg.window.max_zero_fraction:
                n_empty += 1
                continue
            imgs = [mat_norms[s.name].encode(
                _prep_target(block, cfg.window.smooth_sigma))]
            if channels["has_capture"]:
                if cap is not None:
                    cb, bt = cap.window_block(chrom, w.start, w.size, cfg.data.resolution)
                    imgs.append(cap_norms[s.name].encode(cb))
                else:
                    imgs.append(np.full_like(imgs[0], -1.0))
                    bt = np.zeros(w.size, dtype=np.uint8)
                baits.append(bt)
            ct = sym[:, w.start : w.start + w.size]
            if ct.shape[1] < w.size:
                pad = np.zeros((ct.shape[0], w.size - ct.shape[1]), dtype=np.float32)
                ct = np.concatenate([ct, pad], axis=1)
            targets.append(np.stack(imgs).astype(np.float16))
            if ref_clr is not None:
                # same scale as the target; absent chromosome -> 0 (expected)
                rb = (fetch_diagonal_window(ref_clr, ref_name, w.start, w.size)
                      if ref_name is not None else np.zeros((w.size, w.size), np.float32))
                refs.append(mat_norms[s.name].encode(rb).astype(np.float16))
            if prior_clr is not None:
                from .reference import prior_block
                priors.append(prior_block(prior_clr, chrom, w.start, w.size,
                                          cfg.data.resolution,
                                          mat_norms[s.name]).astype(np.float16))
            conds.append(ct.astype(np.float16))
            starts.append(w.start)

        if not targets:
            # (a chromosome with no shard left still reports what it dropped)
            return None, ((s.name, chrom, n_outside_train) if n_outside_train else None)
        shard = cache / f"{s.name}_{split}_{chrom}.npz"
        arrays = dict(
            target=np.stack(targets),
            cond=np.stack(conds),
            start=np.asarray(starts, dtype=np.int64),
            nbins=np.asarray([nb], dtype=np.int64),
        )
        if baits:
            arrays["bait"] = np.stack(baits).astype(np.uint8)
        if refs:
            arrays["ref"] = np.stack(refs)
        if priors:
            arrays["prior"] = np.stack(priors)
        np.savez_compressed(shard, **arrays)
        return {
            "sample": s.name, "chrom": chrom, "split": split,
            "path": str(shard), "n_windows": len(targets), "nbins": nb,
            "n_unmappable_skipped": n_unmappable,
            "n_empty_skipped": n_empty,
            "n_outside_train_regions": n_outside_train,
        }, None

    skipped_outside: list[tuple[str, str, int]] = []   # chroms with no shard left
    if workers <= 1:
        for s in samples:
            train_chroms, test_chroms = sample_chroms[s.name]
            for chrom in tqdm(train_chroms, desc=f"cache-train:{s.name}"):
                rec, skip = process(s, chrom, "train")
                if rec:
                    manifest["shards"].append(rec)
                if skip:
                    skipped_outside.append(skip)
            for chrom in tqdm(test_chroms, desc=f"cache-test:{s.name}"):
                rec, skip = process(s, chrom, "test")
                if rec:
                    manifest["shards"].append(rec)
                if skip:
                    skipped_outside.append(skip)
    else:
        tasks = [(s, chrom, split) for s in samples
                 for split, chroms in zip(("train", "test"), sample_chroms[s.name])
                 for chrom in chroms]
        log.info("caching %d (sample, split, chromosome) tasks on %d workers",
                 len(tasks), workers)
        for rec, skip in _pmap(process, tasks, workers):
            if rec:
                manifest["shards"].append(rec)
            if skip:
                skipped_outside.append(skip)

    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_train = sum(x["n_windows"] for x in manifest["shards"] if x["split"] == "train")
    n_test = sum(x["n_windows"] for x in manifest["shards"] if x["split"] == "test")
    n_unmap = sum(x.get("n_unmappable_skipped", 0) for x in manifest["shards"])
    log.info("cached %d train + %d test windows (%d samples) -> %s",
             n_train, n_test, len(samples), cache)
    # a filter that drops things without saying so is how unusable regions
    # reached training unnoticed in the first place
    n_emp = sum(x.get("n_empty_skipped", 0) for x in manifest["shards"])
    log.info("skipped %d windows below %.0f%% mappability, %d empty windows",
             n_unmap, 100 * cfg.window.min_mappable_fraction, n_emp)
    if train_regions is not None:
        n_out = (sum(x.get("n_outside_train_regions", 0) for x in manifest["shards"])
                 + sum(n for *_, n in skipped_outside))
        manifest["n_outside_train_regions"] = n_out
        (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("train_regions_bed removed %d training windows (not wholly inside "
                 "a training region)", n_out)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Precompute multi-omic Hi-C windows.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes for the per-sample and per-chromosome "
                         "passes (default 1: serial); the cache is identical")
    args = ap.parse_args()
    build(Config.load(args.config), workers=args.workers)


if __name__ == "__main__":
    main()
