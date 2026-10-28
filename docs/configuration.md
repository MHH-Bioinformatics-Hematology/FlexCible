# Configuration reference

A run is described by one YAML file. The schema lives in `flexcible/config.py`; every field has a comment there, and `Config.from_dict` rejects unknown or inconsistent fields. This page lists the fields that ordinary training and prediction use. Fields of experimental variants (cascade priors, differential mark channels, cohesin indicators, AML objective terms) are documented in the source only.

## data

```yaml
data:
  resolution: 10000                 # bin size in bp; must match every cool file
  chroms: null                      # chromosomes to use; null = all autosomes
  train_parity: odd                 # chromosome split by number: odd chromosomes train,
  test_parity: even                 #   even chromosomes test (or the reverse)
  holdout_bed: null                 # instead of parity: BED of held-out regions; windows inside them are the test set
  train_regions_bed: null           # restrict training windows to these regions
  reference_cool: path/consensus.cool   # consensus map for every sample without its own
  tracks:
    - {name: CTCF,  type: tf,       motif: MA0139.2}
    - {name: RAD21, type: cofactor, motif: MA0139.2}
    - {name: SMC3,  type: cofactor, motif: MA0139.2}
    - {name: H3K27ac, type: histone}
    - {name: Meth,  type: methylation}
  samples:
    - name: gm12878
      cool_path: oe/gm12878_proper_10kb.cool       # log O/E target (scripts/oe_proper.py)
      bigwig_dir: tracks/gm12878                   # <MARK>.bigwig per track
      reference_cool: oe/consensus_loo_gm12878_proper_10kb.cool   # leave-one-out consensus
      missing_tracks: [SMC3]                       # marks not measured for this cell
      assay: hic                                   # hic or microc
```

- `tracks` is the set of marks the model can read. The order defines the input rows. A `tf` or `cofactor` track must name a JASPAR motif; the motif is scanned on both strands and gives the directed anchor channels. A `histone` or `methylation` track has no motif.
- `samples` are the cells. `missing_tracks` lists marks that were not measured; they receive the absent-track encoding, the same that track dropout uses. A sample's `reference_cool` overrides `data.reference_cool`; for a training cell it should exclude that cell (leave-one-out).
- The single-sample shorthand `cool_path` and `bigwig_dir` directly under `data` builds one sample called `default`.

## sequence and motifs

```yaml
sequence:
  fasta: genome/hg38.fa
  features: [gc, cpg]               # per-bin sequence summaries
  cache_dir: cache/genome           # motif scans and features, per chromosome
motifs:
  dir: motifs                       # <ID>.jaspar files
  extra: [MA0095.3, MA0088.2, MA1522.1]   # YY1, ZNF143, MAZ: scanned without a ChIP track
  score_frac: 0.80                  # a hit is a log-odds score of at least this fraction of the maximum
  directed_pairs: true              # fwd_i * rev_j and rev_i * fwd_j planes
```

## window and track

```yaml
window:
  size: 192                         # bins per window: 80 at 25 kb, 192 at 10 kb, 384 at 5 kb, 512 at 2 kb
  stride: 96                        # half a window
  transform: prescaled              # targets already hold log O/E (required with a reference)
  clip_quantile: 0.999              # per-sample scale of the target: quantile of |log O/E|
  max_zero_fraction: 0.95           # drop windows with more empty pixels than this
  min_mappable_fraction: 0.5        # drop windows with fewer mappable bins than this
track:
  transform: log1p                  # per-bin track values
  norm: zscore                      # with statistics of the training chromosomes
```

## model

```yaml
model:
  kind: unet
  base_channels: 48                 # channels of the first level; 96, 192, 384 follow, bottleneck 768
  depth: 4                          # number of halvings
  reference: input                  # none: no consensus; input: as an input plane; residual: added to the output
  embed_reference: oe/consensus12_proper_10kb.cool   # copied into every checkpoint for new cells
  resolutions: null                 # [2000, 5000, 10000, 25000] trains one model for all bin sizes
```

The released models use `reference: input` with the consensus of all training cells embedded. With `embed_reference`, `flexcible predict --bigwig-dir ... --cell NAME` needs no reference argument.

## train

```yaml
train:
  output_dir: runs/my_model
  cache_dir: cache/my_model         # written by precompute
  stage_dir: null                   # local disk copy of the cache for large runs
  batch_size: 3                     # per process
  lr: 0.0003
  loss: l1+mse
  amp: bf16                         # mixed precision; none on the CPU
  compile: true                     # torch.compile; false on the CPU
  augment_reverse: true             # reverse-complement augmentation
  track_dropout: 0.5                # > 0 switches track dropout on
  track_dropout_mode: uniform_count # per window keep k ~ Uniform{1..T} tracks
  max_epochs: 500
  patience_epochs: 10               # stop after this many epochs without validation improvement
  max_minutes: null                 # optional wall-clock limit
  val_fraction: 0.1
  val_regions: 1200                 # validation positions, drawn once; training windows overlapping them are removed
  val_exclude_overlap: true
  num_workers: 4
  seed: 0
  resume: false
predict:
  merge: hann                       # weight of overlapping windows
```

`val_exclude_overlap: true` keeps the validation windows disjoint from every training window of every cell. Leave it on; a validation set that overlaps the training windows reports an optimistic score.

## Chromosome-held-out training

To hold whole chromosomes out of every cell, list them nowhere in `data.chroms` for training and predict them afterwards with `--chroms chr2,chr3`. The normalisers are then fitted without those chromosomes as well. The manuscript's benchmark models were trained without chromosomes 2, 3, 11, 14 and 17.
