"""Synthetic chromatin with KNOWN causal rules, for teaching the model what a
boundary is rather than letting it learn a correlational shortcut.

The motivation is a concrete failure: a model trained only on wild-type genomes
learned "a sharp discontinuity in signal means a boundary", so it invented a
boundary at every duplication junction, including junctions where no boundary
element had been duplicated. Real rearranged genomes with measured Hi-C are
scarce, but the rules are not, so we can generate the cases the model needs.

Rules encoded here, each from a well-established observation:

* a convergent pair of CTCF sites forms a loop and insulates between them
  (the orientation rule; inverting one anchor abolishes the loop);
* insulation strength grows with the occupancy of the boundary sites;
* a strong anchor that faces mostly one way produces a STRIPE, contacts
  spreading from that anchor along one direction rather than a focal dot --
  the architectural stripe seen at strongly extruding anchors, and reported at
  rearranged/oncogenic loci;
* contact frequency decays with genomic distance;
* active marks raise local contact, heterochromatic marks lower it.

Because the generator knows where the boundaries ARE, it can also emit the
rearranged versions with their correct answers: duplicating a span that
contains a boundary yields a new domain, duplicating a span that contains none
yields a larger domain and NO new boundary. That pair is the training signal
the real data cannot supply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SimParams:
    n_bins: int = 512
    site_rate: float = 0.02          # CTCF sites per bin
    boundary_frac: float = 0.35      # of convergent pairs that insulate strongly
    decay: float = 1.0               # contact ~ distance^-decay
    loop_gain: float = 2.5           # focal enrichment at a convergent pair
    stripe_gain: float = 1.6         # enrichment along a one-sided strong anchor
    stripe_frac: float = 0.15        # anchors that produce a stripe
    noise: float = 0.15
    seed: int = 0


@dataclass
class SimChrom:
    """A synthetic chromosome plus the ground truth that generated it."""

    ctcf_fwd: np.ndarray             # per-bin forward-site occupancy
    ctcf_rev: np.ndarray
    active: np.ndarray               # active-mark-like track
    repressed: np.ndarray
    matrix: np.ndarray
    boundaries: np.ndarray           # bins that truly insulate
    stripes: np.ndarray              # bins that anchor a stripe
    tracks: dict = field(default_factory=dict)


def simulate(p: SimParams) -> SimChrom:
    rng = np.random.default_rng(p.seed)
    n = p.n_bins
    fwd = np.zeros(n)
    rev = np.zeros(n)
    n_sites = max(4, int(n * p.site_rate))
    pos = np.sort(rng.choice(np.arange(2, n - 2), size=n_sites, replace=False))
    strength = rng.uniform(0.3, 1.0, size=n_sites)
    orient = rng.random(n_sites) < 0.5
    for k, b in enumerate(pos):
        (fwd if orient[k] else rev)[b] = strength[k]

    # convergent pairs: a forward site followed by a reverse site
    pairs = []
    for a_i, a in enumerate(pos):
        if not orient[a_i]:
            continue
        for b_i in range(a_i + 1, len(pos)):
            if not orient[b_i]:
                pairs.append((a, pos[b_i], strength[a_i] * strength[b_i]))
                break

    # a fraction of convergent pairs insulate; the boundary sits at the
    # downstream anchor, with strength set by site occupancy
    bnd = np.zeros(n)
    for a, b, s in pairs:
        if rng.random() < p.boundary_frac:
            bnd[b] = max(bnd[b], min(0.9, 0.35 + 0.6 * s))

    # stripes: strong anchors whose partner sites are weak
    stripe = np.zeros(n)
    for k, b in enumerate(pos):
        if strength[k] > 0.75 and rng.random() < p.stripe_frac:
            stripe[b] = strength[k]

    active = np.clip(rng.normal(0.5, 0.25, n), 0, None)
    active = np.convolve(active, np.ones(5) / 5, mode="same")
    repressed = np.clip(1.2 - active + rng.normal(0, 0.1, n), 0, None)

    mat = _matrix_from_rules(n, bnd, stripe, active, p, rng)
    return SimChrom(fwd, rev, active, repressed, mat, bnd, stripe,
                    tracks={"active": active, "repressed": repressed})


def _matrix_from_rules(n, bnd, stripe, active, p: SimParams, rng) -> np.ndarray:
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    m = 1.0 / np.power(np.maximum(d, 1), p.decay)

    # insulation: a contact is attenuated by every boundary it must cross
    cross = np.cumsum(bnd)
    barrier = np.abs(cross[:, None] - cross[None, :])
    m *= np.exp(-1.8 * barrier)

    # focal loop at each insulating convergent pair
    b_bins = np.nonzero(bnd)[0]
    for k in range(len(b_bins) - 1):
        i, j = b_bins[k], b_bins[k + 1]
        m[i, j] += p.loop_gain * bnd[i] * bnd[j] / max(1, (j - i) ** 0.5)
        m[j, i] = m[i, j]

    # stripes: elevated contact fanning out from a single strong anchor
    for a in np.nonzero(stripe)[0]:
        reach = min(n - 1, a + int(0.25 * n))
        for j in range(a + 1, reach):
            v = p.stripe_gain * stripe[a] / np.power(max(1, j - a), 0.7)
            m[a, j] += v
            m[j, a] += v

    act = 0.5 * (active[:, None] + active[None, :])
    m *= (0.7 + 0.6 * act)
    m *= np.exp(rng.normal(0, p.noise, size=m.shape) * 0.5)
    m = 0.5 * (m + m.T)
    return m * 100.0


def duplicate(sim: SimChrom, a: int, b: int, p: SimParams) -> tuple[SimChrom, bool]:
    """Tandem-duplicate bins [a, b) and regenerate the matrix FROM THE RULES.

    Returns the edited chromosome and whether the duplicated span contained a
    boundary, which is exactly the distinction the model has to learn: with a
    boundary the copies become separate domains, without one they merge into a
    single larger domain and the junction is not a border.
    """
    def dup(v):
        return np.concatenate([v[:b], v[a:b], v[b:]])

    fwd, rev = dup(sim.ctcf_fwd), dup(sim.ctcf_rev)
    act, rep = dup(sim.active), dup(sim.repressed)
    bnd, stripe = dup(sim.boundaries), dup(sim.stripes)
    had_boundary = bool(sim.boundaries[a:b].max() > 0)
    n = fwd.shape[0]
    rng = np.random.default_rng(p.seed + 977)
    mat = _matrix_from_rules(n, bnd, stripe, act, p, rng)
    return SimChrom(fwd, rev, act, rep, mat, bnd, stripe,
                    tracks={"active": act, "repressed": rep}), had_boundary


def invert(sim: SimChrom, a: int, b: int, p: SimParams) -> SimChrom:
    """Invert bins [a, b): orientations swap, so a convergent pair can become
    divergent and its loop should disappear -- the orientation rule."""
    fwd, rev = sim.ctcf_fwd.copy(), sim.ctcf_rev.copy()
    fwd[a:b], rev[a:b] = sim.ctcf_rev[a:b][::-1], sim.ctcf_fwd[a:b][::-1]
    act, rep = sim.active.copy(), sim.repressed.copy()
    act[a:b], rep[a:b] = act[a:b][::-1], rep[a:b][::-1]
    bnd = sim.boundaries.copy(); bnd[a:b] = bnd[a:b][::-1]
    stripe = sim.stripes.copy(); stripe[a:b] = stripe[a:b][::-1]
    # an inverted anchor no longer faces its former partner
    bnd[a:b] *= 0.2
    rng = np.random.default_rng(p.seed + 1553)
    mat = _matrix_from_rules(len(fwd), bnd, stripe, act, p, rng)
    return SimChrom(fwd, rev, act, rep, mat, bnd, stripe,
                    tracks={"active": act, "repressed": rep})


def build_cache(out_dir: str, n_train: int = 40, n_test: int = 8,
                p: SimParams | None = None, window: int = 128,
                stride: int = 64, rearrange: bool = True) -> dict:
    """Write a training cache of synthetic chromosomes in the normal format.

    Half of the training chromosomes are REARRANGED, each paired with the
    matrix the rules say it should have. That is the part real data cannot
    provide: a duplication that captures a boundary and one that does not,
    with the correct answer for both, so the model can learn the causal rule
    instead of "sharp signal change means boundary".

    Conditioning rows are laid out exactly like the real pipeline
    (assays, then sequence features, then motif fwd/rev), so a model trained
    here can be applied to, or mixed with, real data.
    """
    import json

    from .matrix import MatrixNormalizer, plan_windows
    from .tracks import TrackNormalizer

    p = p or SimParams()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_name = ["active", "repressed", "seq:gc", "seq:cpg",
                 "motif:SIM:fwd", "motif:SIM:rev"]

    def rows_of(sim: SimChrom) -> np.ndarray:
        n = sim.active.shape[0]
        gc = 0.4 + 0.1 * (sim.active / max(sim.active.max(), 1e-9))
        cpg = 0.02 + 0.03 * (sim.active / max(sim.active.max(), 1e-9))
        return np.stack([sim.active, sim.repressed, gc, cpg,
                         sim.ctcf_fwd, sim.ctcf_rev]).astype(np.float32)

    def make(seed: int, kind: str):
        pp = SimParams(**{**p.__dict__, "seed": seed})
        base = simulate(pp)
        if kind == "ref":
            return base, "reference"
        n = base.matrix.shape[0]
        rng = np.random.default_rng(seed + 4242)
        span = int(rng.integers(n // 12, n // 5))
        b_bins = np.nonzero(base.boundaries)[0]
        if kind == "dup_boundary" and len(b_bins):
            b = int(rng.choice(b_bins))
            a = max(0, b - span // 2)
            ed, had = duplicate(base, a, min(n - 1, a + span), pp)
            return ed, "dup_with_boundary" if had else "dup_no_boundary"
        if kind == "dup_plain":
            for _ in range(20):
                a = int(rng.integers(1, n - span - 1))
                if base.boundaries[a:a + span].max() == 0:
                    ed, had = duplicate(base, a, a + span, pp)
                    return ed, "dup_no_boundary"
            return base, "reference"
        if kind == "inv":
            a = int(rng.integers(1, n - span - 1))
            return invert(base, a, a + span, pp), "inversion"
        return base, "reference"

    kinds = (["ref", "dup_boundary", "dup_plain", "inv"] if rearrange else ["ref"])
    specs = [(s, kinds[s % len(kinds)]) for s in range(n_train)]
    specs += [(10000 + s, kinds[s % len(kinds)]) for s in range(n_test)]

    all_rows, band, shards = [], [], []
    made = []
    for seed, kind in specs:
        sim, label = make(seed, kind)
        made.append((seed, kind, label, sim))
        r = rows_of(sim)
        if seed < 10000:
            all_rows.append(r)
            band.append(sim.matrix[np.triu_indices_from(sim.matrix)][:20000])
    tn = TrackNormalizer(transform="log1p", norm="zscore").fit(all_rows)
    mn = MatrixNormalizer(transform="log1p", clip_quantile=0.999).fit(
        np.concatenate(band))
    tn.save(out / "norm_tracks.json")
    mn.save(out / "norm_matrix_sim.json")

    for seed, kind, label, sim in made:
        split = "train" if seed < 10000 else "test"
        rows = tn.apply(rows_of(sim))
        n = sim.matrix.shape[0]
        wins = plan_windows(n, window, stride, "sim")
        tg, cd, st = [], [], []
        for w in wins:
            blk = sim.matrix[w.start:w.start + window, w.start:w.start + window]
            if blk.shape != (window, window):
                pad = np.zeros((window, window)); pad[:blk.shape[0], :blk.shape[1]] = blk
                blk = pad
            ct = rows[:, w.start:w.start + window]
            if ct.shape[1] < window:
                ct = np.concatenate(
                    [ct, np.zeros((ct.shape[0], window - ct.shape[1]), np.float32)], 1)
            tg.append(mn.encode(blk)[None].astype(np.float16))
            cd.append(ct.astype(np.float16))
            st.append(w.start)
        name = f"sim{seed}_{label}"
        shard = out / f"{name}_{split}_sim.npz"
        np.savez_compressed(shard, target=np.stack(tg), cond=np.stack(cd),
                            start=np.asarray(st, dtype=np.int64),
                            nbins=np.asarray([n], dtype=np.int64),
                            boundaries=sim.boundaries)
        shards.append({"sample": name, "chrom": "sim", "split": split,
                       "path": str(shard), "n_windows": len(tg), "nbins": n,
                       "label": label})

    channels = {"sym_rows": rows_name, "tracks": ["active", "repressed"],
                "seq_features": ["gc", "cpg"], "motif_ids": ["SIM"],
                "has_capture": False, "capture_input": False,
                "samples": [s["sample"] for s in shards]}
    (out / "channels.json").write_text(json.dumps(channels, indent=2))
    manifest = {"config_resolution": 25000, "window_size": window,
                "stride": stride, **channels, "shards": shards}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    import collections
    print("synthetic cache:", collections.Counter(s["label"] for s in shards))
    return manifest
