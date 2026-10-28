# Troubleshooting

**`ValueError: track ... declares no binding motif`.** Tracks of type `tf` and `cofactor` need a JASPAR motif ID (`motif: MA0139.2`), and the `<ID>.jaspar` file has to be in `motifs.dir`. Histone tracks take no motif.

**The precompute step finds no windows.** The chromosome names of the cool file, the bigWigs and the FASTA have to agree (`chr1` or `1`, consistently). Check `data.chroms` and the parity split; with a single chromosome in the data, either the training or the test set is empty.

**`window.transform prescaled` versus counts.** A reference map (`data.reference_cool`) holds log O/E values and requires `window.transform: prescaled` with targets from `scripts/oe_proper.py`. Raw count maps as targets work only without a reference.

**Out of memory during training.** Reduce `train.batch_size` (3 per 24 GB card at 10 kb with 192-bin windows), set `train.resident_dtype: float16`, or stage the cache on local disk with `train.stage_dir` and `--stage-only` first. At 2 kb, windows of 512 bins with a reference plane need about 40 GB of host memory per process when held in RAM; staging avoids this.

**`torch.compile` errors or very long first steps.** Set `train.compile: false`. Compilation is skipped automatically for multi-resolution models.

**Prediction of a new cell is poor although its marks look fine.** Check that the bigWigs are fold-change-over-control signal (or a comparable scale) and that the files are named exactly `<MARK>.bigwig` as in the configuration. For CUT&Tag or other assays, add `--renormalize-tracks`. Marks that were not measured are given with `--zero-tracks`, never as empty files.

**The predicted map stops 2 Mb from the diagonal.** The band of the target maps is 2 Mb; pixels beyond it are not predicted.

**`hicrep` not installed.** `flexcible evaluate` falls back to its own implementation of the stratum-adjusted correlation coefficient, which agrees with the package to the third decimal.

**Old scripts import `hicmultiome`.** The package was renamed to `flexcible`; `hicmultiome` remains as an alias and resolves to the same modules.
