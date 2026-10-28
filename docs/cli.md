# Command line

`flexcible <command> --help` prints the options of each command. Every command is also available as `python -m flexcible.<command>`.

## precompute

```bash
flexcible precompute --config CONFIG [--workers N]
```

Builds the window cache of a configuration: motif scans and sequence features per chromosome (cached under `sequence.cache_dir`), per-track normalisers fitted on the training chromosomes, the log O/E targets and, when a reference is configured, the reference block of every window. Output under `train.cache_dir`. Rerun after any change to `data`, `window`, `track`, `sequence` or `motifs`.

## train

```bash
flexcible train --config CONFIG
torchrun --nproc_per_node=2 -m flexcible.train --config CONFIG     # two graphics cards
flexcible train --config CONFIG --stage-only                        # copy the cache to train.stage_dir and stop
```

Trains until `train.patience_epochs` epochs without improvement of the validation correlation or `train.max_epochs`, whichever comes first. Writes `checkpoint-best/` and `checkpoint-last/` under `train.output_dir`, a `best` file naming the best checkpoint, `history.json` and the embedded reference when `model.embed_reference` is set.

## predict

```bash
flexcible predict --config CONFIG --out OUT.cool [--sample NAME] [--chroms test|train|all|chr1,chr2]
                  [--tta] [--batch-size N] [--checkpoint DIR]
                  [--bigwig-dir DIR [--extra-bigwig-dir DIR ...] --cell NAME] [--reference REF.cool]
                  [--zero-tracks A,B] [--zero-region TRACK:chrom:start-end ...] [--renormalize-tracks]
```

| option | meaning |
|---|---|
| `--sample` | a sample of the configuration (default: the first) |
| `--chroms` | `test` or `train` by the configured split, `all`, or a comma-separated list |
| `--tta` | average the prediction with that of the reverse-complement orientation |
| `--bigwig-dir`, `--cell` | predict a cell that is not in the configuration from its bigWig folder; the reference is the embedded consensus unless `--reference` names another map |
| `--zero-tracks` | treat these tracks as absent (the absent-track encoding of track dropout) |
| `--zero-region` | set one track to its absent value inside a region, repeatable, for in silico deletions of a binding site |
| `--renormalize-tracks` | fit the per-track statistics on this sample instead of using the training normalisers; for tracks from another assay such as CUT&Tag |
| `--checkpoint` | checkpoint folder (default: the `best` of `train.output_dir`) |

Overlapping windows are merged with a Hann weight into one map per chromosome. With `window.transform: prescaled` (the released models) the output holds log O/E values on the model's bin grid; with count targets it holds counts.

## evaluate

```bash
flexcible evaluate --config CONFIG --pred PRED.cool [--sample NAME] [--chroms test] [--h 1] [--max-dist-mb 5] [--out summary.json]
```

Pearson and Spearman correlation and the stratum-adjusted correlation coefficient (HiCRep, smoothing radius `--h`) between the prediction and the sample's measured map, per chromosome and summarised.

## plot and diagnose

```bash
flexcible plot --config CONFIG --pred PRED.cool --region chr2:18000000-25000000 --out region.png [--track CTCF] [--depth BP]
flexcible diagnose --config CONFIG --chrom 2 [--pred PRED.cool] --out-dir DIR
```

`plot` draws the predicted and the measured map of a region with one track. `diagnose` writes the per-distance Pearson correlation of the prediction and of the baselines along one chromosome.

## loops

```bash
flexcible loops --config CONFIG --pred PRED.cool [--chroms test]
```

Loop and domain measures of a prediction against the measured map: insulation profile correlation, boundary recall and loop recall (see the Methods of the manuscript for the definitions).

## Helper scripts

| script | purpose |
|---|---|
| `scripts/oe_proper.py --input X.mcool --output X_proper_25kb.cool --binsize 25000` | balanced counts to the distance-normalised log O/E target |
| `scripts/consensus_oe.py --res 25000 --cells a b c --out consensus.cool` | mean O/E map of several cells |
| `scripts/consensus_loo.py --res 25000 --cells a b c --tmp DIR` | all leave-one-out consensus maps and the full consensus in one pass |
| `scripts/independent_validation_fetch.py` | example of downloading and preparing a new cell type from ENCODE |
