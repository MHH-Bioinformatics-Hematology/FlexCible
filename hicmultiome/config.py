"""Configuration schema for hic-multiome-unet.

A single YAML file fully specifies a run: the Hi-C cool target, the protein
tracks (each DNA-binding protein MUST declare its binding motif), the genome
FASTA for sequence features and motif scanning, the capture Hi-C interactions
used as auxiliary supervision, the windowing, the chromosome split, and the
optimisation hyper-parameters. Every stage (precompute, train, predict,
evaluate) loads the same config so runs are reproducible.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TF_TYPES = ("tf", "cofactor")           # DNA-binding: motif REQUIRED
NONMOTIF_TYPES = ("histone", "accessibility", "methylation", "annotation")
# SampleSpec.assay: "hic" = in situ Hi-C, "intact" = intact Hi-C (MNase and/or
# DNaseI digestion), "microc" = Micro-C
ASSAYS = ("hic", "intact", "microc")
# data.assay_indicator_mode: "binary" = one row (1 Micro-C, 0 either Hi-C);
# "three" = two rows (intact, microc), in situ Hi-C is the all-zero reference
ASSAY_MODES = ("binary", "three")
# model.reference: how the cached reference map enters the model
REFERENCE_MODES = ("none", "input", "residual")


@dataclass
class TrackSpec:
    """One protein/assay bigwig track.

    type:
      tf            transcription factor / architectural protein -> motif required
      cofactor      protein recruited at motif sites (e.g. cohesin at CTCF
                    motifs) -> motif required (name the recruiting motif)
      histone       histone modification -> no motif (not a sequence-specific
                    binder)
      accessibility DNase/ATAC -> no motif
      methylation   DNA methylation level (WGBS/RRBS fraction) -> no motif
      annotation    cell-invariant genome annotation (CpG islands, TSS, gene
                    bodies, repeat classes; scripts/build_annotation_tracks.py)
                    -> no motif
    motif: JASPAR matrix ID; the PFM must exist as <motifs.dir>/<motif>.jaspar
    """

    name: str
    type: str = "histone"
    motif: str | None = None

    def __post_init__(self):
        if self.type not in TF_TYPES + NONMOTIF_TYPES:
            raise ValueError(
                f"track {self.name!r}: unknown type {self.type!r} "
                f"(expected one of {TF_TYPES + NONMOTIF_TYPES})"
            )
        if self.type in TF_TYPES and not self.motif:
            raise ValueError(
                f"track {self.name!r} has type {self.type!r} but no motif. "
                "Every DNA-binding protein track must declare its binding "
                "motif (JASPAR ID, e.g. 'MA0139.2' for CTCF). For cohesin "
                "subunits use the motif of the recruiting factor."
            )
        if self.type in NONMOTIF_TYPES and self.motif:
            raise ValueError(
                f"track {self.name!r} is a {self.type} track; a binding motif "
                "does not apply -- remove the 'motif' key."
            )


@dataclass
class SampleSpec:
    """One training sample: a cell line / condition with its Hi-C + tracks.

    missing_tracks: tracks with no data for this sample -> raw zeros (the
        shared normaliser then encodes them as clean "no signal").
    zeroed_tracks:  tracks deliberately forced to raw zero, e.g. RAD21+SMC3
        for a cohesin-degron condition whose Hi-C shows the protein-free
        genome. This teaches the model what absence looks like, so in-silico
        deletions (hicmultiome.perturb) are grounded in real perturbation data.
    capture_interactions: optional per-sample capture Hi-C file (format/score
        options come from the global `capture` section).
    capture_baits: optional BED of every baited fragment of the capture
        design (chrom, start, end, ...). The bait mask then covers all
        designed baits, including baits without any significant interaction
        (their rows are measured and hold no significant contact). None keeps
        the original behaviour: baits are the bait ends seen in the
        interaction file.
    fasta: per-sample genome override (cross-assembly samples); defaults to
        sequence.fasta.
    assay: contact assay of the target map: "hic" (in situ Hi-C), "intact"
        (intact Hi-C, MNase/DNaseI) or "microc". Only read when
        data.assay_indicator is on, which then feeds it to the network as
        constant input rows (see data.assay_indicator_mode): the assays differ
        systematically in ways no ChIP track can explain. In the default
        binary mode "intact" counts as Hi-C (row value 0).
    """

    name: str
    cool_path: str
    bigwig_dir: str
    fasta: str | None = None
    capture_interactions: str | None = None
    capture_baits: str | None = None
    missing_tracks: list[str] = field(default_factory=list)
    zeroed_tracks: list[str] = field(default_factory=list)
    assay: str = "hic"
    # per-sample reference map (overrides data.reference_cool for this
    # sample), e.g. a leave-one-out consensus that excludes the sample's own
    # cell. None = data.reference_cool (or no reference).
    reference_cool: str | None = None

    def __post_init__(self):
        if self.assay not in ASSAYS:
            raise ValueError(f"sample {self.name!r}: unknown assay {self.assay!r} "
                             f"(expected one of {ASSAYS})")


@dataclass
class DataConfig:
    tracks: list[TrackSpec] = field(default_factory=list)
    samples: list[SampleSpec] = field(default_factory=list)
    # single-sample shorthand (back-compat): builds samples=[default] on load
    cool_path: str | None = None
    bigwig_dir: str | None = None
    resolution: int = 25000              # bin size in bp; must match the cools
    chroms: list[str] | None = None      # None -> all autosomes
    train_parity: str = "odd"
    test_parity: str = "even"
    # Region-based split instead of chromosome parity: a BED of held-out
    # regions (e.g. Akita's valid+test windows). Training then uses every
    # window that does NOT overlap them, and the test set is the windows
    # contained inside them -- the same containment/overlap rule the
    # published comparison used.
    holdout_bed: str | None = None
    # Restrict TRAINING windows to these regions: a window is kept only if it
    # lies wholly inside one region after overlapping/adjacent regions are
    # merged (e.g. the union of Akita's training sequences, which leaves out
    # exactly the stretches Akita's filters dropped: assembly gaps, blacklist,
    # unmappable and sparse-target sequence). Test windows are unaffected.
    # None = no restriction (previous behaviour).
    train_regions_bed: str | None = None
    # Feed each sample's SampleSpec.assay as one constant conditioning row
    # (0 = Hi-C, 1 = Micro-C), placed after the sequence features. Off = no row.
    assay_indicator: bool = False
    # "binary" (default): the single row above, in situ and intact Hi-C both 0.
    # "three": two constant rows, assay:intact then assay:microc, so in situ
    # Hi-C (0, 0), intact Hi-C (1, 0) and Micro-C (0, 1) are three classes.
    # Only read when assay_indicator is on.
    assay_indicator_mode: str = "binary"
    # Reference contact map: an O/E cool on the same bin grid as the samples
    # (e.g. the mean of other cells' maps, scripts/consensus_oe.py). When set,
    # precompute caches each window's reference block as `ref`, encoded with
    # the sample's matrix normaliser so it shares the target's scale; the model
    # reads it only when model.reference is not "none". Requires
    # window.transform "prescaled" (the reference holds log O/E values).
    # None = no reference (previous behaviour, previous caches).
    reference_cool: str | None = None

    def __post_init__(self):
        if self.assay_indicator_mode not in ASSAY_MODES:
            raise ValueError(f"unknown data.assay_indicator_mode "
                             f"{self.assay_indicator_mode!r} (expected one of {ASSAY_MODES})")


@dataclass
class SequenceConfig:
    """DNA sequence input: per-bin scalar features computed from the FASTA, and
    optionally a learned DNA sequence encoder (a dilated conv trunk over the
    one-hot bases, the way Akita reads sequence) whose per-bin embedding is added
    to the conditioning. The scalar features (gc/cpg) are a summary; the encoder
    gives the model the actual sequence, which is what decides the fine
    distance-normalised (O/E) structure at 2 kb."""

    fasta: str | None = None
    features: list[str] = field(default_factory=lambda: ["gc", "cpg"])
    cache_dir: str = "cache/genome"      # per-chrom feature/motif scan cache
    encoder: str = "none"                # "none" or "dilated"
    embed_dim: int = 8                   # learned per-bin sequence channels
    enc_width: int = 48                  # encoder hidden width (multiple of 8)

    def __post_init__(self):
        for f in self.features:
            if f not in ("gc", "cpg"):
                raise ValueError(f"unknown sequence feature {f!r} (gc|cpg)")
        if self.encoder not in ("none", "dilated"):
            raise ValueError(f"unknown sequence encoder {self.encoder!r} (none|dilated)")

    @property
    def uses_encoder(self) -> bool:
        return self.encoder != "none"


@dataclass
class MotifConfig:
    """PWM scanning of the genome for every distinct motif named by the tracks."""

    dir: str = "motifs"                  # <ID>.jaspar PFM files live here
    # Motifs to scan even when no ChIP track declares them. A binding site is a
    # property of the DNA, so it is available for any species and any cell type
    # without an experiment -- and it is the only way the model can see that a
    # rearrangement duplicated or destroyed a boundary ELEMENT rather than just
    # creating a discontinuity in the signal.
    extra: list[str] = field(default_factory=list)
    enabled: bool = True                 # False -> no motif channels are built
                                         # (feature ablations; declaring motifs
                                         # on tf tracks stays mandatory)
    score_frac: float = 0.80             # hit = log-odds >= frac * max score
    directed_pairs: bool = True          # add fwd_i*rev_j / rev_i*fwd_j channels
                                         # (captures convergent CTCF loop anchors)


@dataclass
class CaptureConfig:
    """Capture Hi-C interactions used as an auxiliary training target.

    The model predicts a second output channel regressing the binned capture
    interaction score; the loss is applied only where the capture design has
    coverage (rows/columns containing baited promoter fragments). This sharpens
    enhancer-promoter loops and, through them, TAD structure -- without leaking
    capture data into the Hi-C prediction at inference time.
    """

    interactions: str | None = None      # significant-interactions text file
    format: str = "mifsud"               # parser to use (Mifsud 2015 E-MTAB-2323)
    min_score: float = 0.0               # drop interactions below this score
    loss_weight: float = 0.5             # weight of the auxiliary capture loss
    clip_quantile: float = 0.999         # score value mapped to +1
    # Optional-input mode: feed the binned capture map as an extra 2D
    # conditioning plane, randomly blanked during training (modality dropout)
    # so prediction never REQUIRES capture data. When blanked, the plane is
    # the "absent" encoding (-1) and the bait mask plane is 0; the auxiliary
    # target loss applies only to blanked windows (otherwise it's a copy).
    use_as_input: bool = False
    input_dropout: float = 0.5           # fraction of windows seen WITHOUT capture
    # --- pairwise (window-free) capture modelling: see pchicunet.pairwise ---
    # Candidate pairs are real fragment pairs, so no bins and no windows are
    # involved; these bound the candidate space and the negative sampling.
    pair_max_distance: int = 2_000_000   # bp; 99th pct of real interactions ~444 kb
    pair_negatives_per_positive: int = 3
    pair_feature_binsize: int = 200      # bp grid used ONLY to integrate track
                                         # signal over a fragment (cumsum trick)
    # What "strength" means for a pair. Read counts are integers drawn from a
    # count process, so `count` targets are trained with a Poisson likelihood
    # and predicted on a log link; `obs_exp` keeps the CHiCAGO-style
    # log(observed/expected) score and an L1/L2 loss.
    strength: str = "obs_exp"            # obs_exp | count

    def __post_init__(self):
        if self.interactions and self.format not in ("mifsud",):
            raise ValueError(f"unknown capture format {self.format!r}")


@dataclass
class WindowConfig:
    size: int = 128
    stride: int = 64
    transform: str = "log1p"      # log1p | none | obs_exp | prescaled
    # The model head is tanh, so a target sitting exactly at +/-1 can only be
    # approached asymptotically. Map the clip bound to this value instead,
    # keeping every target strictly inside the saturating region. Only
    # affects `prescaled`, where the input is already clipped (e.g. Akita's
    # log obs/exp in [-2, 2]) and a hard edge would otherwise land on +/-1.
    clip_headroom: float = 0.9
    clip_quantile: float = 0.999
    max_zero_fraction: float = 0.95
    # Minimum fraction of a window's bins that must be mappable, judged from a
    # "<cool>.mappable.npz" mask beside the matrix (scripts/make_mappable_mask.py).
    # max_zero_fraction above cannot do this job on obs/exp-style targets: there
    # an empty-but-mappable pixel is -2, not 0, and smoothing leaves nothing
    # exactly zero, so it drops nothing at all and unmappable pericentromeric
    # windows reach training unnoticed. Ignored when no mask file exists.
    min_mappable_fraction: float = 0.5
    # Akita-style target denoising: 2D Gaussian smoothing (sigma in bins) of
    # the raw count window before encoding. 0 = off. Applied consistently in
    # normaliser fitting and window caching; evaluation stays vs raw truth.
    smooth_sigma: float = 0.0


@dataclass
class TrackNormConfig:
    transform: str = "log1p"
    norm: str = "zscore"
    smooth_bins: int = 0


@dataclass
class ModelConfig:
    base_channels: int = 48
    depth: int = 4
    kind: str = "unet"        # architecture selector (used by pchicunet's zoo)
    # How much of the 1D->2D wiring is handed to the network. "full" gives the
    # pairwise features; "pair"/"tile" withhold them so the network has to
    # discover the correspondence itself (see model.COND_MODES).
    cond_mode: str = "full"
    # learned per-distance bias b[|i-j|] (length window.size) added to the
    # Hi-C output, to absorb the residual distance trend left in O/E targets
    dist_bias: bool = False
    # Use of the cached reference map (data.reference_cool):
    #   none      not read (default; old configs unchanged)
    #   input     one extra 2D conditioning plane
    #   residual  input plane AND the Hi-C output is reference + network
    #             output, so the network predicts the deviation of this cell
    #             from the reference
    reference: str = "none"
    # Reference map to embed in every checkpoint (copied once as
    # embedded_reference.cool, hicmultiome.reference.embed_reference), so that
    # prediction for a NEW cell needs only its ChIP: predict.py and
    # crosscell_predict.py look windows up in the embedded map by chromosome
    # and bin. Typically the consensus of all training cells; training itself
    # reads data.reference_cool / samples[].reference_cool. None = off.
    embed_reference: str | None = None

    def __post_init__(self):
        if self.reference not in REFERENCE_MODES:
            raise ValueError(f"unknown model.reference {self.reference!r} "
                             f"(expected one of {REFERENCE_MODES})")

    @property
    def reference_planes(self) -> int:
        """2D conditioning planes taken by the reference map (0 or 1)."""
        return 0 if self.reference == "none" else 1


@dataclass
class TrainConfig:
    output_dir: str = "runs/default"
    cache_dir: str = "cache/default"
    batch_size: int = 32                 # per GPU
    # Reversal augmentation: each window may also be seen genomically
    # reversed (tracks flipped, matrix flipped on both axes, motif fwd/rev
    # channels swapped) -- doubles the effective data.
    augment_reverse: bool = False
    # Track dropout: probability per ChIP/assay track per window of being
    # replaced by its "absent" encoding during training. A model trained with
    # this predicts sensibly from ANY track subset, enabling cheap greedy
    # minimal-input search at inference (scripts/minimal_set.py).
    track_dropout: float = 0.0
    # Dropout scheme. "independent": each track dropped i.i.d. with prob
    # track_dropout (expected ~ (1-p)*T kept, rarely very sparse). "uniform_count":
    # per window draw k ~ Uniform{1..T} kept tracks and drop the rest, so the model
    # trains on every subset size from 1 to all -- needed for robust prediction
    # from only a couple of tracks. Active whenever track_dropout > 0 (its value
    # is ignored in uniform_count mode).
    track_dropout_mode: str = "independent"
    # Restrict training/validation to these chromosomes only (e.g. ["chr1","chr2"]).
    # Used for a fast proxy input-search: far fewer windows -> quick candidates,
    # to check whether the track ranking is stable without all chromosomes.
    chroms: list[str] | None = None
    # Continue an interrupted or finished run instead of starting from zero:
    # restores the weights, the optimizer state (momentum matters, restarting
    # it costs progress) and the step/epoch counters, and keeps the previous
    # best validation score so a worse checkpoint cannot overwrite a better one.
    # bf16 autocast: ~1.4x faster on Ada with fp32's exponent range, so no
    # loss scaler is needed. Numerics differ from fp32, so a run trained this
    # way is not bit-comparable with fp32 results.
    amp: str = "none"                    # none | bf16 | fp16
    # Training objective, '+'-joined with optional weights (hicmultiome.losses):
    # 'l1+mse' is the historical default; terms l1, mse, scc, ssim. Adding scc
    # optimises the stratum-adjusted correlation we actually report, which L1
    # does not -- L1 is dominated by the near-diagonal pixels that carry the
    # largest values, exactly the ones SCC's stratification discounts.
    # torch.compile the model: fuses the conv/GroupNorm/SiLU chain, which is
    # memory-traffic bound. Costs a one-off compilation of a minute or two;
    # window shapes are fixed here so it should not recompile afterwards.
    compile: bool = False
    loss: str = "l1+mse"
    loss_max_dist: int | None = None     # strata to include; None = whole window
    loss_stride: int = 1                 # subsample strata (cost vs resolution)
    resume: bool = False
    resume_from: str | None = None       # defaults to output_dir's checkpoint
    lr: float = 3.0e-4
    max_steps: int = 200000
    max_minutes: float | None = None
    eval_every: int = 150
    val_fraction: float = 0.1
    seed: int = 0
    # keep every k-th training window (effective stride = window.stride * k); a
    # cheap way to sweep training-set size without re-precomputing the cache
    window_subsample: int = 1
    # dtype of the windows kept resident (CPU stack and GPU stripe). The cache
    # stores float16, so "float16" is lossless and halves memory (several
    # cells at 2 kb); batches are cast to float32 before use.
    resident_dtype: str = "float32"
    # stream windows from the cooler on the fly instead of the resident cache,
    # so any stride works (incl. the dense, high-overlap regime). Reuses the
    # fitted normalisers in cache_dir. stream_stride overrides window.stride for
    # the streamed windows; num_workers feeds the DataLoader.
    stream: bool = False
    stream_stride: int | None = None
    num_workers: int = 4
    # train on only these ChIP tracks (a subset of the cache's tracks), for
    # greedy feature selection without re-precomputing: the matching cond rows
    # (plus the sequence features and any motif whose track is kept) are sliced
    # from the full cache. None = use all tracks.
    select_tracks: list[str] | None = None
    # motif channels to keep, independently of which ChIP tracks are kept
    # (e.g. the CTCF motif without any CTCF/cohesin ChIP, or no motif at all).
    # None = motifs of the kept tracks plus every motifs.extra entry.
    select_motifs: list[str] | None = None
    # early stopping by epoch: stop after max_epochs, or when validation has not
    # improved for patience_epochs (validated once per epoch). None disables.
    max_epochs: int | None = None
    patience_epochs: int | None = None
    # hold out this many fixed windows (seeded) as an Akita-style validation set
    # -- validation and early stopping then use the median per-region Pearson of
    # log(obs/exp) over them, not the per-distance Pearson. None = old behaviour.
    val_regions: int | None = None
    # Honest validation (default since 2026-09-28): validation windows are
    # drawn by genomic position (every sample's window at a drawn position),
    # and every training window, of any sample, sharing a bin with a validation
    # position is dropped, so checkpoint selection and early stopping cannot
    # be passed by memorised position. Applies to val_regions and to the
    # val_fraction split. Runs before that date used false (validation windows
    # overlapping training windows by up to size - stride bins); their configs
    # reproduce with an explicit val_exclude_overlap: false.
    val_exclude_overlap: bool = True
    # write checkpoints during training. Off for the greedy screen, where only
    # the logged best-val matters, to avoid a 128 MB write every epoch stalling
    # the GPU.
    save_checkpoints: bool = True


@dataclass
class PredictConfig:
    checkpoint: str | None = None
    # Observed Hi-C contacts are integer counts, but the model predicts their
    # EXPECTATION, which is legitimately fractional (an expected 0.7 contacts
    # is meaningful), and an observed/expected ratio is never an integer at
    # all. So predictions stay float by default; set this to emit a rounded
    # integer count matrix as a deliverable, knowing that at fine resolution
    # rounding sends sub-1 expectations to zero.
    integer_counts: bool = False
    merge: str = "hann"
    out_cool: str = "predicted.cool"


@dataclass
class Config:
    data: DataConfig
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    motifs: MotifConfig = field(default_factory=MotifConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    track: TrackNormConfig = field(default_factory=TrackNormConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    # ---- derived helpers ----
    @property
    def track_names(self) -> list[str]:
        return [t.name for t in self.data.tracks]

    @property
    def motif_ids(self) -> list[str]:
        """Distinct motif IDs: those declared by DNA-binding tracks, plus any
        listed in motifs.extra (which need no ChIP track at all)."""
        seen: list[str] = []
        for t in self.data.tracks:
            if t.motif and t.motif not in seen:
                seen.append(t.motif)
        for m in self.motifs.extra:
            if m not in seen:
                seen.append(m)
        return seen

    @property
    def any_fasta(self) -> bool:
        """True if a genome is available for at least one sample. The row
        layout must be identical across samples (they share one tensor), so
        sequence/motif rows are decided globally here, not per sample."""
        return bool(self.sequence.fasta or any(s.fasta for s in self.data.samples))

    @property
    def active_motif_ids(self) -> list[str]:
        """Motifs actually scanned: declaring them is always required for
        DNA-binding tracks, but without a genome FASTA no channels are built
        (ablation configs)."""
        return self.motif_ids if (self.any_fasta and self.motifs.enabled) else []

    @property
    def has_reference(self) -> bool:
        """A reference map is configured (globally or for any sample)."""
        return bool(self.data.reference_cool
                    or any(s.reference_cool for s in self.data.samples))

    def sample_reference(self, sample: "SampleSpec") -> str | None:
        return sample.reference_cool or self.data.reference_cool

    @property
    def has_capture(self) -> bool:
        return any(s.capture_interactions for s in self.data.samples)

    def sample_fasta(self, sample: "SampleSpec") -> str | None:
        return sample.fasta or self.sequence.fasta

    def get_sample(self, name: str | None = None) -> "SampleSpec":
        """Sample by name; default = the first sample."""
        if name is None:
            return self.data.samples[0]
        for s in self.data.samples:
            if s.name == name:
                return s
        raise KeyError(
            f"no sample {name!r} (have {[s.name for s in self.data.samples]})"
        )

    def validate(self) -> None:
        if not self.data.tracks:
            raise ValueError("data.tracks must list at least one track")
        names = self.track_names
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate track names: {names}")
        if not self.data.samples:
            raise ValueError(
                "no samples: give data.samples, or cool_path+bigwig_dir "
                "for the single-sample shorthand"
            )
        snames = [s.name for s in self.data.samples]
        if len(set(snames)) != len(snames):
            raise ValueError(f"duplicate sample names: {snames}")
        for s in self.data.samples:
            for t in s.missing_tracks + s.zeroed_tracks:
                if t not in names:
                    raise ValueError(
                        f"sample {s.name!r}: unknown track {t!r} in "
                        "missing_tracks/zeroed_tracks"
                    )
        if self.model.reference != "none":
            lack = [s.name for s in self.data.samples
                    if not (s.reference_cool or self.data.reference_cool)]
            if lack:
                raise ValueError(f"model.reference={self.model.reference!r} needs "
                                 "data.reference_cool or samples[].reference_cool "
                                 f"(the cached reference map); none for {lack}")
        if self.has_reference and self.window.transform != "prescaled":
            raise ValueError("a reference map (data.reference_cool or "
                             "samples[].reference_cool) needs window.transform "
                             "'prescaled' (the reference holds log O/E values)")
        for mid in self.active_motif_ids:
            p = Path(self.motifs.dir).expanduser() / f"{mid}.jaspar"
            if not p.exists():
                raise FileNotFoundError(
                    f"motif PFM not found: {p} (download it from JASPAR, e.g. "
                    f"https://jaspar.elixir.no/api/v1/matrix/{mid}.jaspar)"
                )

    # ---- (de)serialisation ----
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        def build(dc, d):
            d = dict(d or {})
            fields = {f.name for f in dataclasses.fields(dc)}
            unknown = set(d) - fields
            if unknown:
                raise ValueError(f"Unknown keys for {dc.__name__}: {sorted(unknown)}")
            return dc(**d)

        if "data" not in raw:
            raise ValueError("config must contain a 'data' section")
        draw = dict(raw["data"])
        tracks = draw.pop("tracks", [])
        parsed = []
        for t in tracks:
            if isinstance(t, str):
                raise ValueError(
                    f"track {t!r}: tracks must be mappings with name/type"
                    "/motif, e.g. {name: CTCF, type: tf, motif: MA0139.2} -- "
                    "a bare name does not say what protein this is or what "
                    "motif it binds."
                )
            parsed.append(TrackSpec(**t))
        samples = [SampleSpec(**s) for s in draw.pop("samples", [])]
        if not samples and draw.get("cool_path"):
            samples = [SampleSpec(
                name="default",
                cool_path=draw["cool_path"],
                bigwig_dir=draw["bigwig_dir"],
                capture_interactions=(raw.get("capture") or {}).get("interactions"),
            )]
        cfg = cls(
            data=DataConfig(tracks=parsed, samples=samples, **draw),
            sequence=build(SequenceConfig, raw.get("sequence")),
            motifs=build(MotifConfig, raw.get("motifs")),
            capture=build(CaptureConfig, raw.get("capture")),
            window=build(WindowConfig, raw.get("window")),
            track=build(TrackNormConfig, raw.get("track")),
            model=build(ModelConfig, raw.get("model")),
            train=build(TrainConfig, raw.get("train")),
            predict=build(PredictConfig, raw.get("predict")),
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
