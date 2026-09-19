# hic-multiome-unet

Predict **Hi-C contact matrices** from **protein ChIP tracks + DNA sequence +
binding motifs**, with **capture Hi-C** as an auxiliary training target, using
a plain spatial-conditioning **U-Net** and **multi-GPU (DDP)** training.

Design decisions (and why):

- **U-Net only, no adversary.** The Hi-cGAN experiments showed the adversarial
  term does not help this task; the purpose-built regression U-Net (from
  `hic-unet`, SCC 0.84 held-out from CTCF alone) is the base and this project
  extends it.
- **Every DNA-binding protein must declare its binding motif.** The config
  rejects a `tf`/`cofactor` track without a JASPAR motif ID. The genome is
  PWM-scanned per motif, strand-separated, so the model gets *directed*
  conditioning channels `fwd_i * rev_j` -- the convergent-orientation geometry
  of CTCF/cohesin loop anchors, which no symmetric feature can express.
  Cohesin subunits (RAD21, SMC3) declare the CTCF motif that positions them.
- **DNA sequence** enters as per-bin GC / CpG-density features plus the motif
  scans (bins are 10-25 kb; base-resolution one-hot is not meaningful here).
- **Capture Hi-C as auxiliary supervision, not input.** The model has a second
  output channel regressing binned significant promoter-capture interactions
  (CHiCAGO-style log(obs/exp)), with the loss masked to bait-covered
  rows/columns. This sharpens enhancer-promoter loops and TAD structure while
  keeping inference input-only (no capture leakage at predict time).

## Data (GM12878, hg19 -- the cell type where everything exists)

| layer | source | location |
|---|---|---|
| Hi-C 25 kb | Rao 2014 GSE63525 in-situ | `~/data/Hi-cGAN/data/GM12878_25kb_cooler_coarsen.cool` (local) |
| 14 ChIP/DNase bigwigs | ENCODE | `~/data/Hi-cGAN/data/gm12878/bigwig/` (local) |
| genome FASTA | hg19 | `/mnt/processed/wolffjoa/vardictcpp_validation_zenodo/reference/hg19.fa` (local) |
| CTCF motif | JASPAR MA0139.2 | `~/data/hic-multiome/motifs/` |
| promoter-capture Hi-C | Mifsud 2015, E-MTAB-2323 (hg19) | `~/data/hic-multiome/pchic/TS5_GM12878_promoter-other_significant_interactions.txt` |

All layers share hg19 coordinates; K562 lacks matched promoter-capture Hi-C,
which is why GM12878 is the training cell type.

## Usage

```bash
conda activate hicdiff          # torch-cu13 + cooler + pyBigWig + pyfaidx + hicrep

# 1) Precompute windows (fits normalisers on train chroms; caches motif scans)
python -m hicmultiome.precompute --config configs/multiome_25kb.yaml

# 2) Train on BOTH GPUs (or drop torchrun for single-GPU)
torchrun --nproc_per_node=2 -m hicmultiome.train --config configs/multiome_25kb.yaml

# 3) Predict held-out chromosomes; also write the capture channel
python -m hicmultiome.predict --config configs/multiome_25kb.yaml --chroms test \
    --out runs/multiome_25kb/pred_even.cool \
    --pchic-out runs/multiome_25kb/pred_even_capture.cool

# 4) Global metric: HiCRep SCC
python -m hicmultiome.evaluate --config configs/multiome_25kb.yaml \
    --pred runs/multiome_25kb/pred_even.cool --chroms test

# 5) Loop + TAD metrics: E-P contacts vs capture Hi-C, insulation/boundaries
python -m hicmultiome.loops --config configs/multiome_25kb.yaml \
    --pred runs/multiome_25kb/pred_even.cool \
    --pred-capture runs/multiome_25kb/pred_even_capture.cool --chroms test

# 6) Visuals: per-distance Pearson + region plot
python -m hicmultiome.diagnose --config configs/multiome_25kb.yaml --chrom 2 \
    --pred runs/multiome_25kb/pred_even.cool --out-dir runs/multiome_25kb
python -m hicmultiome.plot --config configs/multiome_25kb.yaml \
    --pred runs/multiome_25kb/pred_even.cool \
    --region chr2:18000000-25000000 --out runs/multiome_25kb/chr2.png
```

`configs/chip_only_25kb.yaml` is the like-for-like ablation (same 14 tracks,
no sequence/motif/capture) for measuring what the multi-omic additions buy.

## Modules

- `config.py` -- typed YAML schema; **enforces motif declaration** for
  DNA-binding tracks
- `sequence.py` / `motifs.py` -- FASTA features + cached strand-separated PWM
  scans (pure numpy)
- `capture.py` -- Mifsud-format parsing, binning, bait mask, score normaliser
- `model.py` -- `HiCUNet` (1-2 output channels) + `build_spatial_cond` with
  directed motif-pair planes
- `precompute.py` / `dataset.py` -- window cache (cond + Hi-C + capture + mask)
- `train.py` -- DDP-capable trainer, masked auxiliary loss, learning-curve
  CSV/PNG
- `predict.py` -- full-genome cools for both channels
- `evaluate.py` -- HiCRep SCC; `loops.py` -- E-P loop + insulation/TAD metrics
- `diagnose.py` / `plot.py` -- per-distance Pearson curves, pyGenomeTracks
  region figures

## Tests

```bash
python -m pytest tests/ -q     # synthetic end-to-end (CPU, ~6 s)
```
