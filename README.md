# FlexCible

FlexCible predicts the Hi-C contact map of a cell from its chromatin marks. It reads ChIP-seq or CUT&Tag tracks of CTCF, the cohesin subunits RAD21 and SMC3 and up to ten histone modifications, DNA methylation, the GC and CpG content of the genome and the positions of CTCF, YY1, ZNF143 and MAZ motifs on both strands. As a further input it reads a consensus, the mean contact map of other cell types, and adapts it with the marks of the target cell. The output is a whole-chromosome contact map in cool format at 25, 10, 5 or 2 kb.

One trained model serves every use: it accepts any subset of the marks, works with or without the consensus and predicts at any bin size from 25 to 2 kb. A patient sample profiled by CUT&Tag for three marks and a cell line profiled for all thirteen are analysed with the same checkpoint. The model is a U-Net with 32 million parameters; a chromosome takes seconds on a graphics card and about a minute on eight processor threads.

## Installation

```bash
mamba create -n flexcible -c conda-forge -c bioconda python=3.11 pytorch cooler pybigwig pyfaidx numpy scipy pyyaml tqdm matplotlib
mamba activate flexcible
git clone https://github.com/MHH-Bioinformatics-Hematology/FlexCible.git
cd FlexCible
pip install -e .
```

`flexcible --help` lists the commands. Details and alternatives are in `docs/installation.md`.

## Predicting with the released model

The released checkpoint, the consensus maps and the split files are deposited on Zenodo (doi:10.5281/zenodo.23922797). Unpack `flexcible_released_model.zip` and `consensus12_maps.zip`, point four environment variables at the genome, the motifs, a cache folder and the consensus maps, and predict a human cell from its bigWig tracks in hg38; marks that were not measured are left out or declared with `--absent-track`:

```bash
export FLEXCIBLE_GENOME=data/genome/hg38.fa FLEXCIBLE_MOTIFS=data/motifs \
       FLEXCIBLE_CACHE=cache FLEXCIBLE_MAPS=consensus12
flexcible predict --config flexcible_release/flexcible_10kb.yaml \
    --checkpoint flexcible_release/flexcible_10kb/checkpoint-best \
    --bigwig-dir tracks/mycell --cell mycell --chroms all --tta \
    --absent-track RAD21 --absent-track SMC3 \
    --out mycell_10kb.cool
```

The output holds log observed/expected values within 2 Mb of the diagonal and opens in HiGlass, HiCExplorer and cooltools. `--no-reference` predicts from the marks alone, `--bin-size` sets another bin size, and `--renormalize-tracks` fits the track statistics on a sample whose signal scale differs from ENCODE fold change. See `docs/pretrained.md`.

## Training a model

```bash
flexcible precompute --config my_model.yaml   # motif scans, normalisers, window cache
flexcible train --config my_model.yaml        # single GPU; torchrun for several
flexcible predict --config my_model.yaml --sample mycell --chroms test --out pred.cool
flexcible evaluate --config my_model.yaml --pred pred.cool --chroms test
```

The Hi-C targets are distance-normalised log observed/expected maps made with `scripts/oe_proper.py`, and the consensus of the other cells with `scripts/consensus_loo.py`. `docs/tutorial.md` walks through a complete training from download to evaluation, and `docs/configuration.md` documents every field of the configuration file.

## Quickstart on toy data

```bash
bash examples/quickstart/run.sh toy
```

runs precompute, training, prediction and evaluation on a synthetic genome in a few minutes on a CPU.

## Documentation

The documentation in `docs/` is built with MkDocs (`mkdocs serve`): overview, installation, quickstart, tutorial, released models, configuration reference, command line and troubleshooting.

## Tests

```bash
python -m pytest tests/ -q
```

## Citation

Wolff J. FlexCible: flexible-input prediction of chromatin architecture from chromatin marks and a consensus matrix. Submitted.

## Licence

MIT, see `LICENSE`.
