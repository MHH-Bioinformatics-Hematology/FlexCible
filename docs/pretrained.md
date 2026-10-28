# Predicting with the released models

The released checkpoints predict a human cell from its bigWig tracks. Each checkpoint carries the consensus of the 12 training cell lines as `embedded_reference.cool`, so no Hi-C data of the target cell are needed.

!!! note
    The checkpoints, the consensus maps and the split files are deposited with the manuscript (Zenodo record to be announced). Until the deposit is public, this page describes the procedure.

## Files you need

- the bigWig tracks of your cell, one file per mark, named as in the configuration (`CTCF.bigwig`, `RAD21.bigwig`, `SMC3.bigwig`, `H3K27ac.bigwig`, `H3K27me3.bigwig`, `H3K36me3.bigwig`, `H3K4me1.bigwig`, `H3K4me2.bigwig`, `H3K4me3.bigwig`, `H3K79me2.bigwig`, `H3K9ac.bigwig`, `H3K9me3.bigwig`, `H4K20me1.bigwig`, optionally `Meth.bigwig`), in hg38, as fold-change-over-control signal or a comparable scale;
- the hg38 FASTA and the four JASPAR motif files (downloaded once, see the [tutorial](tutorial.md#motifs));
- a released checkpoint folder and its configuration file.

Any subset of the marks is enough. Marks you do not have are listed with `--zero-tracks`.

## Predict a whole genome

```bash
flexcible predict --config released/refin_10kb.yaml --checkpoint released/refin_10kb/checkpoint-best \
    --bigwig-dir tracks/mycell --cell mycell --chroms all --tta \
    --zero-tracks RAD21,SMC3,H3K4me2,H3K79me2,H3K9ac,H4K20me1 \
    --out mycell_10kb.cool
```

The output is a cool file at 10 kb with log O/E values within 2 Mb of the diagonal. It opens in HiGlass, HiCExplorer (`hicPlotMatrix`, `hicFindTADs`, `hicDetectLoops`) and cooltools.

For CUT&Tag or other assays whose signal scale differs from ENCODE fold change, add `--renormalize-tracks`, which fits the per-track statistics on your sample.

## Choose the resolution

Released checkpoints exist at 25, 10, 5 and 2 kb (`refin_25kb`, `refin_10kb`, `refin_5kb`, `refin_2kb`), each with its own configuration. A single model for all bin sizes, selected with `--binSize` at prediction time, is in preparation.

## Which model

| model | reads | use |
|---|---|---|
| `refin_*` (reference input) | marks and the consensus | the default |
| `refres_*` (reference residual) | marks; adds its prediction to the consensus | the same accuracy; its output minus the consensus is the predicted cell-specific difference |
| `honest_*` (ChIP only) | marks only | when no consensus should enter, for example to test what the marks alone predict |

## Use your own reference

A consensus of cells closer to yours (for example other patients of a cohort) replaces the embedded one:

```bash
flexcible predict ... --reference oe/consensus_other_patients_10kb.cool
```

The map has to be a log O/E cool on the same bin grid (`scripts/consensus_oe.py`).

## In silico deletion of a binding site

`--zero-region CTCF:chr8:127730000-127740000` sets the CTCF signal to zero inside the region, and `--zero-region motif:MA0139.2:chr8:127730000-127740000` removes the CTCF motif sites there; both repeatable. Zero signal is the state of a region without binding, which differs from the absent-track encoding of `--zero-tracks` (a mark that was not measured). The difference between the perturbed and the unperturbed prediction is the predicted effect of losing that site. Use a model without the consensus for such tests: the consensus carries the wild-type structure of the other cells and damps a local deletion.
