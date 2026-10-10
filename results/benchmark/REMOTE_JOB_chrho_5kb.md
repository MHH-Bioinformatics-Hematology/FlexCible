# Remote job: chromosome-held-out final models at 5 kb (registry section 58, PENDING)

Prepared 2026-10-10. Nothing in this file has been run on a GPU. The two caches
in step 1 are built on the CPU of the local machine (nice 19, 2 workers) while
this file is written; check `ls $HM/cache_expanded/<cache>/manifest.json`
before starting.

## What exists and what is missing

- chrho models (chromosomes 2, 3, 11, 14, 17 absent from the data of every
  cell, `data.chroms` of `scripts/multicell_expanded.py`, `CHRHO_HELD`) exist at
  25 and 10 kb only: `all12refin_drop_chrho`, `all12honest_drop_chrho`,
  `all12refres_drop_chrho` (25 kb and 10 kb folders in
  `runs/multicell_reference/`, section 37). No `*_chrho_5kb` folder, yaml or
  cache existed before this preparation.
- The 5 kb models of section 33 (`all12refin_drop_5kb`, `all12honest_drop_5kb`,
  `all12refres_drop_5kb`) were trained on data that include chr2, 3, 11, 14, 17
  and cannot be compared with competitors on those chromosomes.
- The generic run name works at 5 kb without a code change: `RUNS`,
  `CHRHO_RUNS`, `WINDOW[5000] = (384, 192)` and the cache rule
  (`mc_all12ref_chrho_<k>kb`, `mc_all12chrho_<k>kb`) of
  `scripts/multicell_expanded.py`. `--stage ready` prints READY for the three
  runs at `--res 5000` (checked 2026-10-10: 5 kb proper O/E targets of the 12
  cells and their leave-one-out consensus maps exist in `crosscell_oe`).

## Jobs

| run name (claim key) | config | cache | note |
|---|---|---|---|
| `all12refin_drop_chrho_5kb` | `runs/multicell_reference/all12refin_drop_chrho_5kb.yaml` (written by the stage) | `cache_expanded/mc_all12ref_chrho_5kb` | input-only reference, track dropout 0.5; the model compared with C.Origami, Epiphany, HiC-Reg, Hi-cGAN |
| `all12honest_drop_chrho_5kb` | `.../all12honest_drop_chrho_5kb.yaml` | `cache_expanded/mc_all12chrho_5kb` | ChIP only, no consensus; the fair comparison with competitors that get none |
| `all12refres_drop_chrho_5kb` (optional, third) | `.../all12refres_drop_chrho_5kb.yaml` | shares `mc_all12ref_chrho_5kb` | reference as residual; adds the refres column of section 37 |

Same recipe as the 25 and 10 kb chrho runs: 12 training cells, honest
validation (`val_exclude_overlap`), track dropout 0.5 (`uniform_count`),
window 384 bins (1.92 Mb) stride 192, base channels 48, depth 4, batch 3 per
rank (6 global; the single-GPU wrapper doubles it), patience 10 epochs,
`resident_dtype` float16, Akita valid and test regions excluded, chr2, 3, 11,
14, 17 removed from `data.chroms`.

## Step 1: caches (CPU, local, already started)

    HM=/mnt/processed/wolffjoa/hic-multiome
    cd ~/src/hic-multiome-unet
    nice -n 19 python scripts/multicell_expanded.py --runs all12refin_drop_chrho --res 5000 --stage precompute --workers 2
    nice -n 19 python scripts/multicell_expanded.py --runs all12honest_drop_chrho --res 5000 --stage precompute --workers 2

The existing full 5 kb caches (`mc_all12ref_5kb` 12 GB, `mc_all12_5kb` 5.7 GB)
cannot be reused: they hold windows and normaliser statistics of the five
held-out chromosomes, and the chrho caches are separate by design (section
37). The precompute of the casc 5 kb cache took 30 min with 8 workers (log
`cascade_5kb.precompute.log`). Expected size of the chrho caches: about 0.74
of the full ones (396 of 516 shards at 10 and 25 kb), roughly 9 GB and 4 GB;
the share has 3.6 TB free. Local log: `runs/chrho5_precompute.out` and
`runs/multicell_reference/<run>_5kb.precompute.log`.

## Step 2: training on the remote 4090 (one job after the other)

Check `nvidia-smi` and `pgrep -f seq_cnn_chip` first (REMOTE.md: the sequence
CNN holds the GPU; do not start while it runs).

    source ~/hicenv/bin/activate
    HM=/mnt/processed/wolffjoa/hic-multiome
    cd ~/hic-multiome-unet          # rsync of the local tree, after this commit
    for RUN in all12honest_drop_chrho all12refin_drop_chrho; do   # add all12refres_drop_chrho last if time permits
      NAME=${RUN}_5kb
      python scripts/claim_job.py claim $NAME --done-path $HM/runs/multicell_reference/$NAME/best \
          --cmd "run_single_gpu $RUN 5000" || continue
      python scripts/run_single_gpu.py multicell_expanded --runs $RUN --res 5000 --stage train --gpus 0 \
          > $HM/runs/multicell_reference/$NAME.remote.out 2>&1 \
          || python scripts/claim_job.py release $NAME
    done

The stage writes the yaml next to the log, trains with
`torchrun --standalone --nproc_per_node=1` (batch doubled by
`run_single_gpu.py`, so steps per epoch, learning rate and early stopping
equal the local 2-GPU recipe) and stops by early stopping. It never touches the
local scheduler (PID 1902496) or `gpu_queue_extra.py`.

Per-cell test evaluation after training (inference only, 1 GPU):

    python scripts/multicell_expanded.py --runs all12honest_drop_chrho --res 5000 --stage eval --gpus 0 \
        --cells gm12878 imr90 hct116 hff h1 hepg2 a549 k562 hmec_deep nhek gm23248 mcf7 cd14mono pc9 hmec huvec huvec_deep
    (same for all12refin_drop_chrho)

## GPU time

Measured (run logs in `runs/multicell_reference/`, wall time from the
"train windows" line to "done."):

| run | windows | epochs | wall | min per epoch |
|---|---|---|---|---|
| remote 1x4090, all12honest_drop_meth_25kb | 14325 | 24 | 10.8 min | 0.45 |
| remote 1x4090, all12honest_drop_meth_10kb | 15227 | 26 | 22.0 min | 0.85 |
| local 2 GPUs, all12honest_drop_chrho_25kb | 9634 | 25 | 19.7 min | 0.79 |
| local 2 GPUs, all12refin_drop_chrho_25kb | 9634 | 72 | 66.2 min | 0.92 |
| local 2 GPUs, all12honest_drop_chrho_10kb | 10257 | 29 | 26.4 min | 0.91 |
| local 2 GPUs, all12refin_drop_chrho_10kb | 10257 | 55 | 50.2 min | 0.91 |
| local 2 GPUs, all12honest_drop_5kb (full) | 15143 | 24 | 51.7 min | 2.15 |
| local 2 GPUs, all12refin_drop_5kb (full) | 15143 | 125 | 250.4 min | 2.00 |
| local 2 GPUs, all12honest_drop_10kb (full) | 15227 | 28 | 37.3 min | 1.33 |
| local 2 GPUs, all12refin_drop_10kb (full) | 15227 | 76 | 110.8 min | 1.46 |

Estimate for the 5 kb chrho runs on the remote 4090. Windows: the chrho
caches hold 0.674 of the full cache's training windows at 10 kb (10257 of
15227), so about 10200 at 5 kb (15143 x 0.674). Cost per window at 5 kb over
10 kb at equal window count: 2.05 / 1.40 = 1.46 on the local machine (the
4x larger window area costs 1.46x, not 4x). Remote per-epoch cost at 10 kb
with 15227 windows: 0.85 min, so 0.85 x 0.674 x 1.46 = 0.84 min per epoch at
5 kb chrho.

| run | epochs (expected, range) | GPU time |
|---|---|---|
| all12honest_drop_chrho_5kb | 24 to 30 (24 at 5 kb full, 25 and 29 at 25 and 10 kb chrho) | 20 to 25 min |
| all12refin_drop_chrho_5kb | 90 (70 to 125; 72 and 55 epochs at 25 and 10 kb chrho, 125 at 5 kb full) | 75 min (60 to 105 min) |
| sum of the two required runs | | about 1.7 h (1.3 to 2.2 h) |
| all12refres_drop_chrho_5kb (optional) | 100 to 140 | 85 to 120 min |
| three runs | | about 3.3 h (2.5 to 4.2 h) |

If the cost per window scaled with the window area (4x, the worst case
against the measured 1.46x), the two required runs would take 3.3 h. The
runs include per-epoch validation and the `torch.compile` warm-up; the final
per-cell test evaluation adds about 10 min.

## Step 3: predictions and scoring (CPU; GPU optional)

Prediction on the five held-out chromosomes for GM12878, IMR-90 and K562
(`hicmultiome.predict --chroms chr2,chr3,chr11,chr14,chr17 --tta`) goes through
the existing chrho stage. `scripts/hic_measures.py` does not yet carry 5 kb
(`RESOLUTIONS = [25000, 10000]`, `WINDOW`, `BEST_OTHER`, `REPLICATE`, the
insulation width `ins_w` and `benchmark_blocks.params`), so the following is
needed before `--stage chrho_predict chrho_score` runs at 5 kb, and can be done
on the CPU while the GPU job trains:

1. add 5000 to `WINDOW` (384), `BEST_OTHER` (the 10 kb table), `ins_w`
   (insulation square in bins: 5 at 10 kb, so 10 at 5 kb for the same 50 kb span), `benchmark_blocks.params(5000)` (loop and APA windows
   in bins, doubled from 10 kb), and a flag `--chrho-res 5000` so that the
   default 25 and 10 kb behaviour stays unchanged;
2. measured reference calls (hicFindTADs, hicDetectLoops) at 5 kb for GM12878,
   IMR-90, K562 (`benchmark_competitors.py --stage extract calls`, `CALL_JOBS`);
3. competitor O/E at 5 kb through `oe_proper.py` (never re-running a competitor,
   only converting what exists, see the competitor table below):
   Hi-cGAN GM12878 5 kb (`04_sweep_5kb` / `02_best_models`, 875 MB, odd
   chromosomes trained, even held out: chr2 and chr14 shared with us),
   HiC-Reg released 5 kb predictions (`variants()` of `scripts/hicreg_audit.py`
   loops over `(25000, 10000)`; add 5000, no summing needed), C.Origami (8192 bp
   native, sampled bilinearly onto the 5 kb centres, an up-sampling) and
   Epiphany (10 kb native, up-sampled the same way).

Blocks at 5 kb then follow `chrho_blocks()`: C.Origami IMR-90 chr2, chr3
(2 Mb band; 5 kb window reach is 1.92 Mb, so the band is 383 bins); Epiphany
GM12878 chr3, 11, 17 (990 kb = 198 bins); HiC-Reg GM12878 and K562 chr14,
chr17 (1 Mb = 200 bins, WINDOW and MULTICELL cross-chromosome and CV);
Hi-cGAN GM12878 chr2, chr14.

    python scripts/hic_measures.py --stage chrho_predict chrho_score chrho_report --chrho-res 5000 --threads 4 --jobs 2

The scoring cost at 5 kb is about four times the 10 kb cost (10 kb chr2 block:
about 20 min native measures); budget a day of CPU.

## Competitors at 5 kb (verified from the files)

| competitor | 5 kb output | file | held-out chromosomes |
|---|---|---|---|
| Hi-cGAN | GM12878 only (best_plus_two RAD21 + H3K4me3 + SMC3 at 5 kb; 5 kb sweep with up to best_plus_three); no cross-cell-type 5 kb predictions (the cross-cell sets exist at 10 and 25 kb only) | `restart_of_prediction_project/zenodo_records/02_best_models/05_predictions/gm12878_sweeps/5kb/best_plus_two/*.cool` (bin size 5000, hg19) | trained on odd chromosomes, even held out |
| HiC-Reg | native 5 kb: the published GM12878 chr1 to 22 CV files `Gm12878_chr<N>_RF_tree20_prediction_predicted_coords.cool` (bin size 5000, hg19, 1 Mb reach) and the released text predictions (CV, cross-chromosome and cross-cell for chr14, chr17; K562 too) under `home_offload/HiC-Reg` | as listed | cross-chromosome files leak-free for chr14, chr17; CV files within-chromosome |
| C.Origami | 8192 bp native (`testNovember/IMR90/chr2.cool`, bin size 8192), no 5 kb output | | chr2 test, chr3 validation |
| Epiphany | 10 kb native only (`epiphany_pretrained/assemble_matrix_chr3.cool`, bin size 10000) | | chr3, 11, 17 |
| Akita | 2048 bp / 2 kb, 1 Mb windows on the 413 test regions only | | all test regions excluded from our training |

## Result tables to fill (section 58, 5 kb rows)

Same columns as the 25 kb table (Pearson, tile median, SCC, insulation r,
boundary recall, loop recall, APA, compartment r, HiCRep), written by
`scripts/chrho_25kb_table.py` after the `--chrho-res` generalisation (the script
reads `chrho_25kb_*.csv`; copy it with the file tag changed).
