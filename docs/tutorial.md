# Tutorial: from public data to a trained model

This tutorial trains a FlexCible model on two human cell lines, GM12878 and IMR-90, at 25 kb, and predicts a third cell from its marks. It uses the same sources and steps as the models of the manuscript. A short variant that trains on two chromosomes in about ten minutes is given at the end.

All commands assume the repository root as working directory, the `flexcible` environment activated, and a data folder `data/`:

```bash
mkdir -p data/hic data/oe data/tracks data/genome data/motifs cache runs
```

## 1. Genome and motifs {#motifs}

```bash
curl -L -o data/genome/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip data/genome/hg38.fa.gz
for id in MA0139.2 MA0095.3 MA0088.2 MA1522.1; do      # CTCF, YY1, ZNF143, MAZ
  curl -L -o data/motifs/$id.jaspar "https://jaspar.elixir.no/api/v1/matrix/$id.jaspar"
done
```

The motif scans and the GC and CpG content are computed once per chromosome on the first `precompute` and cached under `sequence.cache_dir`.

## 2. Chromatin marks

The released ENCODE signal of each mark is the fold change over control in GRCh38, one bigWig per mark. The accessions of all cells used in the manuscript are listed in `docs/data/accessions_full.csv` (columns `sample`, `assay_or_mark`, `accession`). For GM12878 and IMR-90:

```bash
for cell in gm12878 imr90; do
  mkdir -p data/tracks/$cell
  awk -F, -v c=$cell '$2==c && $3=="train" && $4!="Hi-C" && $4!="methylation" {print $4, $5}' docs/data/accessions_full.csv | sort -u |
  while read mark acc; do
    curl -L -C - -o data/tracks/$cell/$mark.bigwig "https://www.encodeproject.org/files/$acc/@@download/$acc.bigWig"
  done
done
```

The file name has to be `<MARK>.bigwig` with the mark written as in the configuration. A mark that was not measured for a cell is simply absent from the folder and listed under `missing_tracks` in the configuration. DNA methylation (`Meth.bigwig`, ENCODE WGBS methylation level) is optional.

## 3. Hi-C targets

Training needs a measured contact map per cell. GM12878 and IMR-90 are 4D Nucleome multi-resolution coolers:

```bash
for acc in 4DNFIXP4QG5B 4DNFIJTOIGOI; do        # GM12878, IMR-90
  curl -L -C - -o data/hic/$acc.mcool "https://data.4dnucleome.org/files-processed/$acc/@@download/$acc.mcool"
done
```

The model does not learn counts. Each map is converted into a distance-normalised log O/E map: the balanced counts are divided by the mean over all mappable pixel pairs at the same distance, the log ratio is centred per diagonal, clipped to plus or minus 2, smoothed and cut to 2 Mb from the diagonal.

```bash
python scripts/oe_proper.py --input data/hic/4DNFIXP4QG5B.mcool --output data/oe/gm12878_proper_25kb.cool --binsize 25000
python scripts/oe_proper.py --input data/hic/4DNFIJTOIGOI.mcool --output data/oe/imr90_proper_25kb.cool --binsize 25000
```

An ENCODE `.hic` file is first converted to a cool file and balanced (`hicxchange hic2cool`, then `cooler balance --ignore-diags 2 --max-iters 500`); `scripts/independent_validation_fetch.py` does this for four cell types and serves as a template.

## 4. Consensus maps

The consensus of a set of cells is the pixel-wise mean of their O/E maps. A training cell must read a consensus that excludes itself (leave-one-out), otherwise its own target leaks into its input. `consensus_loo.py` writes all leave-one-out maps and the full consensus in one pass:

```bash
python scripts/consensus_loo.py --dir data/oe --res 25000 --cells gm12878 imr90 --tmp cache/consensus_tmp
```

This produces `data/oe/consensus_loo_gm12878_proper_25kb.cool` (here the IMR-90 map), `consensus_loo_imr90_proper_25kb.cool` and `consensus2_proper_25kb.cool`, each with a `.json` naming its cells. With two cells the leave-one-out consensus is the other cell's map; the manuscript's models use 12 cells. The software refuses a consensus whose recorded cells include the cell being trained.

## 5. Configuration

`configs/tutorial_25kb.yaml`:

```yaml
data:
  resolution: 25000
  train_parity: odd
  test_parity: even
  reference_cool: data/oe/consensus2_proper_25kb.cool
  tracks:
    - {name: CTCF,  type: tf,       motif: MA0139.2}
    - {name: RAD21, type: cofactor, motif: MA0139.2}
    - {name: SMC3,  type: cofactor, motif: MA0139.2}
    - {name: H3K27ac, type: histone}
    - {name: H3K27me3, type: histone}
    - {name: H3K36me3, type: histone}
    - {name: H3K4me1, type: histone}
    - {name: H3K4me2, type: histone}
    - {name: H3K4me3, type: histone}
    - {name: H3K79me2, type: histone}
    - {name: H3K9ac, type: histone}
    - {name: H3K9me3, type: histone}
    - {name: H4K20me1, type: histone}
  samples:
    - name: gm12878
      cool_path: data/oe/gm12878_proper_25kb.cool
      bigwig_dir: data/tracks/gm12878
      reference_cool: data/oe/consensus_loo_gm12878_proper_25kb.cool
    - name: imr90
      cool_path: data/oe/imr90_proper_25kb.cool
      bigwig_dir: data/tracks/imr90
      reference_cool: data/oe/consensus_loo_imr90_proper_25kb.cool
sequence:
  fasta: data/genome/hg38.fa
  cache_dir: cache/genome
motifs:
  dir: data/motifs
  extra: [MA0095.3, MA0088.2, MA1522.1]
window:
  size: 80
  stride: 40
  transform: prescaled
model:
  base_channels: 48
  depth: 4
  reference: input
  embed_reference: data/oe/consensus2_proper_25kb.cool
train:
  output_dir: runs/tutorial_25kb
  cache_dir: cache/tutorial_25kb
  batch_size: 3
  lr: 0.0003
  amp: bf16
  compile: true
  track_dropout: 0.5
  track_dropout_mode: uniform_count
  patience_epochs: 10
  num_workers: 4
```

Odd chromosomes train, even chromosomes test. To hold specific chromosomes out of every cell instead, list the others under `data.chroms` and predict the held-out ones afterwards. If IMR-90 lacks a mark in your download, add it under `missing_tracks` of that sample.

## 6. Precompute

```bash
flexcible precompute --config configs/tutorial_25kb.yaml --workers 8
```

Scans motifs, computes sequence features, fits the track normalisers on the training chromosomes, cuts windows of 80 bins with a stride of 40 and writes them with their reference blocks to `cache/tutorial_25kb`. The log reports the number of training and test windows per cell.

## 7. Train {#train}

```bash
torchrun --nproc_per_node=2 -m flexcible.train --config configs/tutorial_25kb.yaml    # two graphics cards
flexcible train --config configs/tutorial_25kb.yaml                                   # one card, or the CPU
```

Training stops when the validation correlation has not improved for 10 epochs. At 25 kb with two cells this takes about an hour on two consumer graphics cards; the 12-cell models of the manuscript took several hours at 25 kb and about a day at 2 kb. The run folder holds `checkpoint-best/` with the weights, the normalisers, the configuration and the embedded consensus, plus `history.json` with the learning curve. On a machine without a graphics card set `amp: none` and `compile: false`.

Check the learning curve before using a model:

```bash
python -c "import json; h=json.load(open('runs/tutorial_25kb/history.json')); print([round(e['val_pearson'],3) for e in h])"
```

## 8. Predict

The test chromosomes of a training cell:

```bash
flexcible predict --config configs/tutorial_25kb.yaml --sample gm12878 --chroms test --tta --out runs/tutorial_25kb/gm12878_even.cool
```

A new cell from its marks, with the embedded consensus as reference. Here HCT116, for which SMC3 is not available:

```bash
flexcible predict --config configs/tutorial_25kb.yaml --bigwig-dir data/tracks/hct116 --cell hct116 --chroms all --tta \
    --zero-tracks SMC3 --out runs/tutorial_25kb/hct116_25kb.cool
```

The output is a cool file at 25 kb with log O/E values within 2 Mb of the diagonal.

## 9. Evaluate

```bash
flexcible evaluate --config configs/tutorial_25kb.yaml --sample gm12878 --pred runs/tutorial_25kb/gm12878_even.cool --chroms test --out runs/tutorial_25kb/eval_gm12878.json
flexcible loops --config configs/tutorial_25kb.yaml --pred runs/tutorial_25kb/gm12878_even.cool --chroms test
flexcible diagnose --config configs/tutorial_25kb.yaml --chrom 2 --pred runs/tutorial_25kb/gm12878_even.cool --out-dir runs/tutorial_25kb/diagnose
flexcible plot --config configs/tutorial_25kb.yaml --pred runs/tutorial_25kb/gm12878_even.cool --region chr2:18000000-25000000 --out runs/tutorial_25kb/chr2.png
```

`evaluate` gives Pearson, Spearman and the stratum-adjusted correlation coefficient per chromosome; `loops` the insulation correlation and the recall of boundaries and loops; `diagnose` the Pearson correlation as a function of genomic distance; `plot` a region with the predicted and the measured map. Compare every number with the consensus alone, scored the same way, since the consensus is the baseline a prediction has to exceed:

```bash
flexcible evaluate --config configs/tutorial_25kb.yaml --sample gm12878 --pred data/oe/consensus_loo_gm12878_proper_25kb.cool --chroms test
```

## 10. Short variant: two chromosomes, ten minutes

For a first real run, restrict the data to two chromosomes and the training to a few epochs:

```yaml
data:
  chroms: [chr21, chr22]        # chr21 trains, chr22 tests
train:
  max_epochs: 5
  patience_epochs: 5
  num_workers: 2
```

Everything else stays as above. `oe_proper.py` and `consensus_loo.py` accept `--chroms chr21 chr22` to convert only these chromosomes. Precompute then takes about a minute and training about ten minutes on one graphics card. The resulting model is not useful for prediction; it shows the pipeline on real files and the format of every output.

## 11. In silico perturbation

A binding site is deleted by setting its signal to zero inside a region, together with the motif sites there; use the model without consensus (`model.reference: none`) for perturbations, since the consensus carries the wild-type structure:

```bash
flexcible predict --config configs/tutorial_25kb.yaml --sample gm12878 --chroms chr8 --tta \
    --zero-region CTCF:chr8:127730000-127740000 --zero-region RAD21:chr8:127730000-127740000 \
    --zero-region motif:MA0139.2:chr8:127730000-127740000 \
    --out runs/tutorial_25kb/gm12878_chr8_del.cool
```

The difference between this map and the unperturbed prediction is the predicted effect of losing the site. The same mechanism with `--zero-tracks` removes a whole mark.
