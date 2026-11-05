# Predicting with the released models

The released checkpoint is one model for every bin size from 25 to 2 kb and for any subset of the marks. It predicts a human cell from its bigWig tracks, with the consensus of the 12 training cell lines as reference, so no Hi-C data of the target cell are needed. A second checkpoint, trained with chromosomes 2, 3, 11, 14 and 17 held out, exists only for benchmarks on those chromosomes.

!!! note
    The checkpoints, the consensus maps and the split files are deposited with the manuscript in the Zenodo record [10.5281/zenodo.23922797](https://doi.org/10.5281/zenodo.23922797).

## Files you need

- the bigWig tracks of your cell, one file per mark, named as in the configuration (`CTCF.bigwig`, `RAD21.bigwig`, `SMC3.bigwig`, `H3K27ac.bigwig`, `H3K27me3.bigwig`, `H3K36me3.bigwig`, `H3K4me1.bigwig`, `H3K4me2.bigwig`, `H3K4me3.bigwig`, `H3K79me2.bigwig`, `H3K9ac.bigwig`, `H3K9me3.bigwig`, `H4K20me1.bigwig`, optionally `Meth.bigwig`), in hg38, as fold-change-over-control signal or a comparable scale;
- the hg38 FASTA and the four JASPAR motif files (downloaded once, see the [tutorial](tutorial.md#motifs));
- `flexcible_released_model.zip` and `consensus12_maps.zip` from the Zenodo record, unpacked. The first holds the shared weights (`flexcible_release/flexcible/checkpoint-best/`), one view per bin size (`flexcible_release/flexcible_<k>kb/checkpoint-best/`, k = 25, 10, 5, 2) and one prediction configuration per bin size (`flexcible_release/flexcible_<k>kb.yaml`); the second the 12-cell consensus at the four bin sizes.

Any subset of the marks is enough. Marks you do not have are left out of the folder or declared with `--absent-track`.

The prediction configurations contain no local paths. They take four folders from the environment:

```bash
export FLEXCIBLE_GENOME=data/genome/hg38.fa     # the hg38 FASTA
export FLEXCIBLE_MOTIFS=data/motifs             # folder with the four .jaspar files
export FLEXCIBLE_CACHE=cache                    # any writable folder; genome features are cached here once
export FLEXCIBLE_MAPS=consensus12               # the folder unpacked from consensus12_maps.zip
```

## Predict a whole genome

```bash
flexcible predict --config flexcible_release/flexcible_10kb.yaml \
    --checkpoint flexcible_release/flexcible_10kb/checkpoint-best \
    --bigwig-dir tracks/mycell --cell mycell --chroms all --tta \
    --absent-track RAD21 --absent-track SMC3 --absent-track H3K4me2 \
    --absent-track H3K79me2 --absent-track H3K9ac --absent-track H4K20me1 \
    --out mycell_10kb.cool
```

The output is a cool file at 10 kb with log O/E values within 2 Mb of the diagonal. It opens in HiGlass, HiCExplorer (`hicPlotMatrix`, `hicFindTADs`, `hicDetectLoops`) and cooltools.

For CUT&Tag or other assays whose signal scale differs from ENCODE fold change, add `--renormalize-tracks`, which fits the per-track statistics on your sample.

## Choose the resolution

The released model was trained at 2, 5, 10 and 25 kb together and comes with one configuration and one view per trained bin size, all sharing the weights. Pick the configuration and the view of the bin size you want; any other bin size is given at prediction time:

```bash
flexcible predict --config flexcible_release/flexcible_10kb.yaml \
    --checkpoint flexcible_release/flexcible_10kb/checkpoint-best \
    --bigwig-dir tracks/mycell --cell mycell --chroms all --tta \
    --bin-size 4000 --reference oe/consensus12_proper_4kb.cool \
    --out mycell_4kb.cool
```

`--bin-size` (alias `--binSize`) accepts any bin size in bp. The reference has to be a log O/E cool at the same bin size, built with `scripts/oe_proper.py` and `scripts/consensus_loo.py` from Hi-C maps at that bin size. The window keeps its genomic span; the normalisers are those of the nearest trained bin size, and `--renormalize-tracks` refits the track statistics on your sample. Trained bin sizes are 2, 5, 10 and 25 kb; bin sizes between them are interpolation, and bin sizes below 2 kb or above 25 kb are untested extrapolation. The accuracy at 1, 4, 20 and 50 kb is reported in the supplement of the article. The option is refused for models with a lookup embedding or one resolution.

## With or without the consensus

The released model reads the marks and the consensus. `--no-reference` sets the consensus plane to zero, so the prediction rests on the marks alone; use it to see what the marks carry by themselves and for in silico perturbations. The prediction minus the consensus is the predicted cell-specific difference.

```bash
flexcible predict ... --no-reference --out mycell_10kb_marks_alone.cool
```

## Use your own reference

A consensus of cells closer to yours (for example other patients of a cohort) replaces the 12-cell consensus:

```bash
flexcible predict ... --reference oe/consensus_other_patients_10kb.cool
```

The map has to be a log O/E cool on the same bin grid (`scripts/consensus_oe.py`).

## In silico deletion of a binding site

`--zero-region CTCF:chr8:127730000-127740000` sets the CTCF signal to zero inside the region, and `--zero-region motif:MA0139.2:chr8:127730000-127740000` removes the CTCF motif sites there; both repeatable. Zero signal is the state of a region without binding; `--zero-tracks CTCF` removes the protein genome-wide, as in a degron, and `--absent-track` declares a mark as not measured. The difference between the perturbed and the unperturbed prediction is the predicted effect of losing that site. Add `--no-reference` for such tests: the consensus carries the wild-type structure of the other cells and damps a local deletion.
