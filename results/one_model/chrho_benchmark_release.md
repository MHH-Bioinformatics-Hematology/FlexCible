# The release twin on the held-out-chromosome benchmark

Mean over the held-out chromosomes of each block (the value the benchmark figure reads). Release twin = `all12one_fullrd_chrho` (one network for 2, 5, 10 and 25 kb; 13 marks and methylation, cohesin-fraction row, depletion augmentation, reference dropout 0.3; trained without chr2, 3, 11, 14, 17 of every cell), with the consensus as reference input and, from the same checkpoint, ChIP only (reference plane 0, methylation given). All sources of a block are scored on the same pixels. Per-chromosome values: `chrho_benchmark_release.csv`.

## C.Origami block, imr90, 10 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.927 | 0.930 | 0.944 | 0.668 | 0.350 | 0.964 | 2 |
| release twin, ChIP only | 0.812 | 0.818 | 0.870 | 0.528 | 0.223 | 0.889 | 2 |
| one model (no dropout), consensus input | 0.933 | 0.936 | 0.951 | 0.647 | 0.362 | 0.966 | 2 |
| one model (no dropout), ChIP only | 0.506 | 0.525 | 0.700 | 0.360 | 0.053 | 0.718 | 2 |
| per-resolution, consensus input | 0.926 | 0.927 | 0.924 | 0.654 | 0.355 | 0.960 | 2 |
| per-resolution, ChIP only | 0.808 | 0.812 | 0.822 | 0.497 | 0.192 | 0.885 | 2 |
| per-resolution, consensus residual | 0.928 | 0.931 | 0.943 | 0.658 | 0.364 | 0.964 | 2 |
| consensus | 0.866 | 0.862 | 0.796 | 0.641 | 0.383 | 0.893 | 2 |
| best_other_map | 0.873 | 0.868 | 0.783 | 0.623 | 0.286 | 0.917 | 2 |
| replicate | 0.885 | 0.901 | 0.976 | 0.684 | 0.034 | 0.983 | 2 |
| cascfold | 0.932 | 0.935 | 0.947 | 0.658 | 0.365 | 0.967 | 2 |
| corigami_imr90_8kb | 0.920 | 0.913 | 0.866 | 0.473 | 0.162 | 0.972 | 2 |
| hicgan_gm12878_on_imr90_10kb | 0.291 | 0.314 | 0.564 | 0.137 | 0.040 | 0.717 | 1 |
| hicgan_imr90_on_imr90_10kb | 0.567 | 0.567 | 0.584 | 0.105 | 0.014 | 0.734 | 1 |

Chromosomes: chr2, chr3.

## Epiphany block, gm12878, 10 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.921 | 0.917 | 0.905 | 0.660 | 0.478 | 0.957 | 3 |
| release twin, ChIP only | 0.791 | 0.787 | 0.783 | 0.529 | 0.274 | 0.870 | 3 |
| one model (no dropout), consensus input | 0.925 | 0.922 | 0.911 | 0.656 | 0.498 | 0.956 | 3 |
| one model (no dropout), ChIP only | 0.522 | 0.539 | 0.595 | 0.334 | 0.065 | 0.690 | 3 |
| per-resolution, consensus input | 0.914 | 0.908 | 0.875 | 0.655 | 0.495 | 0.951 | 3 |
| per-resolution, ChIP only | 0.760 | 0.751 | 0.715 | 0.497 | 0.220 | 0.844 | 3 |
| per-resolution, consensus residual | 0.919 | 0.915 | 0.892 | 0.676 | 0.474 | 0.956 | 3 |
| consensus | 0.826 | 0.818 | 0.778 | 0.660 | 0.452 | 0.875 | 3 |
| best_other_map | 0.748 | 0.744 | 0.711 | 0.601 | 0.258 | 0.821 | 3 |
| replicate | 0.871 | 0.874 | 0.899 | 0.702 | 0.145 | 0.963 | 3 |
| cascfold | 0.920 | 0.916 | 0.904 | 0.664 | 0.460 | 0.957 | 3 |
| epiphany_gm12878_10kb | 0.512 | 0.501 | 0.318 | 0.208 | 0.052 | 0.776 | 3 |

Chromosomes: chr11, chr17, chr3.

## Hi-cGAN block, gm12878, 10 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.926 | 0.924 | 0.903 | 0.624 | 0.409 | 0.963 | 2 |
| release twin, ChIP only | 0.803 | 0.799 | 0.781 | 0.498 | 0.215 | 0.885 | 2 |
| one model (no dropout), consensus input | 0.931 | 0.929 | 0.910 | 0.629 | 0.414 | 0.963 | 2 |
| one model (no dropout), ChIP only | 0.445 | 0.478 | 0.601 | 0.304 | 0.059 | 0.730 | 2 |
| per-resolution, consensus input | 0.921 | 0.917 | 0.875 | 0.620 | 0.400 | 0.953 | 2 |
| per-resolution, ChIP only | 0.797 | 0.787 | 0.726 | 0.497 | 0.164 | 0.876 | 2 |
| per-resolution, consensus residual | 0.924 | 0.921 | 0.890 | 0.640 | 0.407 | 0.960 | 2 |
| consensus | 0.841 | 0.833 | 0.757 | 0.611 | 0.383 | 0.882 | 2 |
| best_other_map | 0.735 | 0.732 | 0.684 | 0.540 | 0.146 | 0.819 | 2 |
| replicate | 0.860 | 0.869 | 0.896 | 0.674 | 0.133 | 0.960 | 2 |
| cascfold | 0.922 | 0.921 | 0.900 | 0.617 | 0.399 | 0.960 | 2 |
| hicgan_gm12878_10kb | 0.328 | 0.330 | 0.260 | 0.113 | 0.031 | 0.700 | 2 |

Chromosomes: chr14, chr2.

## Hi-cGAN block, k562, 10 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.855 | 0.856 | 0.843 | 0.558 | 0.390 | 0.942 | 2 |
| release twin, ChIP only | 0.730 | 0.730 | 0.728 | 0.468 | 0.186 | 0.858 | 2 |
| one model (no dropout), consensus input | 0.861 | 0.860 | 0.837 | 0.573 | 0.388 | 0.946 | 2 |
| one model (no dropout), ChIP only | 0.523 | 0.553 | 0.674 | 0.342 | 0.080 | 0.769 | 2 |
| per-resolution, consensus input | 0.844 | 0.844 | 0.825 | 0.559 | 0.364 | 0.937 | 2 |
| per-resolution, ChIP only | 0.731 | 0.728 | 0.732 | 0.438 | 0.160 | 0.872 | 2 |
| per-resolution, consensus residual | 0.849 | 0.850 | 0.830 | 0.571 | 0.382 | 0.940 | 2 |
| consensus | 0.792 | 0.794 | 0.798 | 0.577 | 0.360 | 0.890 | 2 |
| best_other_map | 0.724 | 0.715 | 0.576 | 0.494 | 0.248 | 0.827 | 2 |
| replicate | 0.838 | 0.836 | 0.762 | 0.592 | 0.266 | 0.959 | 2 |
| hicgan_gm12878_on_k562_10kb | 0.125 | 0.141 | 0.339 | 0.108 | 0.008 | 0.519 | 2 |

Chromosomes: chr14, chr2.

## C.Origami block, imr90, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.932 | 0.935 | 0.945 | 0.821 | 0.350 | 0.953 | 2 |
| release twin, ChIP only | 0.778 | 0.785 | 0.833 | 0.652 | 0.219 | 0.860 | 2 |
| one model (no dropout), consensus input | 0.943 | 0.946 | 0.952 | 0.797 | 0.352 | 0.958 | 2 |
| one model (no dropout), ChIP only | 0.502 | 0.518 | 0.645 | 0.444 | 0.057 | 0.705 | 2 |
| per-resolution, consensus input | 0.940 | 0.941 | 0.943 | 0.817 | 0.349 | 0.958 | 2 |
| per-resolution, ChIP only | 0.804 | 0.800 | 0.775 | 0.647 | 0.206 | 0.862 | 2 |
| per-resolution, consensus residual | 0.940 | 0.942 | 0.946 | 0.807 | 0.335 | 0.960 | 2 |
| consensus | 0.877 | 0.872 | 0.817 | 0.788 | 0.374 | 0.885 | 2 |
| best_other_map | 0.894 | 0.887 | 0.817 | 0.792 | 0.364 | 0.915 | 2 |
| replicate | 0.976 | 0.979 | 0.988 | 0.891 | 0.318 | 0.993 | 2 |
| cascfold | 0.940 | 0.943 | 0.952 | 0.827 | 0.336 | 0.959 | 2 |
| corigami_imr90_8kb | 0.942 | 0.933 | 0.898 | 0.752 | 0.313 | 0.980 | 2 |
| hicgan_gm12878_on_imr90_25kb | 0.335 | 0.349 | 0.443 | 0.309 | 0.075 | 0.739 | 1 |
| hicgan_imr90_on_imr90_25kb | 0.808 | 0.792 | 0.630 | 0.618 | 0.258 | 0.945 | 1 |

Chromosomes: chr2, chr3.

## Epiphany block, gm12878, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.919 | 0.914 | 0.899 | 0.828 | 0.447 | 0.951 | 3 |
| release twin, ChIP only | 0.744 | 0.739 | 0.725 | 0.625 | 0.256 | 0.842 | 3 |
| one model (no dropout), consensus input | 0.923 | 0.920 | 0.907 | 0.825 | 0.452 | 0.949 | 3 |
| one model (no dropout), ChIP only | 0.476 | 0.493 | 0.545 | 0.336 | 0.081 | 0.638 | 3 |
| per-resolution, consensus input | 0.917 | 0.913 | 0.900 | 0.827 | 0.453 | 0.948 | 3 |
| per-resolution, ChIP only | 0.754 | 0.739 | 0.668 | 0.608 | 0.241 | 0.846 | 3 |
| per-resolution, consensus residual | 0.922 | 0.916 | 0.902 | 0.837 | 0.451 | 0.950 | 3 |
| consensus | 0.841 | 0.832 | 0.804 | 0.805 | 0.456 | 0.878 | 3 |
| best_other_map | 0.775 | 0.769 | 0.744 | 0.780 | 0.389 | 0.821 | 3 |
| replicate | 0.946 | 0.939 | 0.922 | 0.906 | 0.417 | 0.976 | 3 |
| cascfold | 0.922 | 0.918 | 0.911 | 0.843 | 0.442 | 0.951 | 3 |
| epiphany_gm12878_10kb | 0.518 | 0.501 | 0.377 | 0.437 | 0.254 | 0.775 | 3 |

Chromosomes: chr11, chr17, chr3.

## Hi-cGAN block, gm12878, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.932 | 0.927 | 0.902 | 0.809 | 0.381 | 0.954 | 2 |
| release twin, ChIP only | 0.774 | 0.765 | 0.737 | 0.570 | 0.201 | 0.863 | 2 |
| one model (no dropout), consensus input | 0.936 | 0.932 | 0.912 | 0.776 | 0.391 | 0.960 | 2 |
| one model (no dropout), ChIP only | 0.388 | 0.426 | 0.574 | 0.368 | 0.095 | 0.686 | 2 |
| per-resolution, consensus input | 0.934 | 0.929 | 0.901 | 0.808 | 0.405 | 0.959 | 2 |
| per-resolution, ChIP only | 0.796 | 0.778 | 0.698 | 0.616 | 0.200 | 0.883 | 2 |
| per-resolution, consensus residual | 0.932 | 0.927 | 0.893 | 0.810 | 0.373 | 0.957 | 2 |
| consensus | 0.860 | 0.848 | 0.784 | 0.777 | 0.389 | 0.897 | 2 |
| best_other_map | 0.793 | 0.782 | 0.719 | 0.737 | 0.320 | 0.836 | 2 |
| replicate | 0.941 | 0.938 | 0.923 | 0.856 | 0.348 | 0.970 | 2 |
| cascfold | 0.928 | 0.926 | 0.907 | 0.818 | 0.358 | 0.956 | 2 |
| hicgan_gm12878_25kb | 0.221 | 0.225 | 0.237 | 0.218 | 0.048 | 0.644 | 2 |

Chromosomes: chr14, chr2.

## Hi-cGAN block, k562, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.910 | 0.900 | 0.839 | 0.745 | 0.364 | 0.947 | 2 |
| release twin, ChIP only | 0.772 | 0.756 | 0.682 | 0.555 | 0.156 | 0.865 | 2 |
| one model (no dropout), consensus input | 0.908 | 0.899 | 0.838 | 0.733 | 0.351 | 0.941 | 2 |
| one model (no dropout), ChIP only | 0.574 | 0.585 | 0.582 | 0.436 | 0.116 | 0.746 | 2 |
| per-resolution, consensus input | 0.904 | 0.896 | 0.843 | 0.727 | 0.356 | 0.941 | 2 |
| per-resolution, ChIP only | 0.765 | 0.743 | 0.657 | 0.550 | 0.141 | 0.854 | 2 |
| per-resolution, consensus residual | 0.901 | 0.894 | 0.853 | 0.751 | 0.352 | 0.938 | 2 |
| consensus | 0.853 | 0.849 | 0.816 | 0.718 | 0.350 | 0.895 | 2 |
| best_other_map | 0.812 | 0.785 | 0.616 | 0.680 | 0.323 | 0.844 | 2 |
| replicate | 0.936 | 0.923 | 0.836 | 0.831 | 0.349 | 0.963 | 2 |
| hicgan_gm12878_on_k562_25kb | 0.143 | 0.147 | 0.218 | 0.213 | 0.050 | 0.655 | 2 |

Chromosomes: chr14, chr2.

## HiC-Reg block, gm12878, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.934 | 0.928 | 0.907 | 0.812 | 0.382 | 0.958 | 2 |
| release twin, ChIP only | 0.764 | 0.756 | 0.732 | 0.563 | 0.198 | 0.837 | 2 |
| one model (no dropout), consensus input | 0.937 | 0.931 | 0.911 | 0.800 | 0.385 | 0.954 | 2 |
| one model (no dropout), ChIP only | 0.449 | 0.476 | 0.573 | 0.345 | 0.070 | 0.612 | 2 |
| per-resolution, consensus input | 0.931 | 0.925 | 0.905 | 0.833 | 0.396 | 0.953 | 2 |
| per-resolution, ChIP only | 0.782 | 0.763 | 0.674 | 0.608 | 0.191 | 0.867 | 2 |
| per-resolution, consensus residual | 0.930 | 0.923 | 0.898 | 0.830 | 0.383 | 0.950 | 2 |
| consensus | 0.868 | 0.853 | 0.804 | 0.805 | 0.396 | 0.894 | 2 |
| best_other_map | 0.797 | 0.791 | 0.776 | 0.801 | 0.335 | 0.830 | 2 |
| replicate | 0.947 | 0.941 | 0.923 | 0.895 | 0.344 | 0.979 | 2 |
| cascfold | 0.933 | 0.929 | 0.918 | 0.837 | 0.365 | 0.958 | 2 |
| hicregmulticellcrosschrom_gm12878_25kb | 0.615 | 0.567 | 0.455 | 0.345 | 0.038 | 0.856 | 2 |
| hicregwindowcrosschrom_gm12878_25kb | 0.579 | 0.538 | 0.447 | 0.322 | 0.032 | 0.846 | 2 |
| hicregwindowcv_gm12878_25kb | 0.908 | 0.870 | 0.677 | 0.723 | 0.323 | 0.987 | 2 |

Chromosomes: chr14, chr17.

## HiC-Reg block, k562, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.897 | 0.886 | 0.841 | 0.760 | 0.337 | 0.928 | 2 |
| release twin, ChIP only | 0.721 | 0.706 | 0.640 | 0.569 | 0.160 | 0.825 | 2 |
| one model (no dropout), consensus input | 0.904 | 0.894 | 0.847 | 0.748 | 0.329 | 0.932 | 2 |
| one model (no dropout), ChIP only | 0.577 | 0.561 | 0.492 | 0.422 | 0.071 | 0.697 | 2 |
| per-resolution, consensus input | 0.897 | 0.889 | 0.857 | 0.738 | 0.347 | 0.927 | 2 |
| per-resolution, ChIP only | 0.730 | 0.711 | 0.622 | 0.568 | 0.125 | 0.834 | 2 |
| per-resolution, consensus residual | 0.899 | 0.892 | 0.865 | 0.785 | 0.334 | 0.929 | 2 |
| consensus | 0.857 | 0.853 | 0.839 | 0.769 | 0.333 | 0.887 | 2 |
| best_other_map | 0.805 | 0.776 | 0.633 | 0.718 | 0.330 | 0.854 | 2 |
| replicate | 0.936 | 0.918 | 0.840 | 0.862 | 0.325 | 0.974 | 2 |
| hicregmulticellcrosschrom_k562_25kb | 0.740 | 0.700 | 0.500 | 0.519 | 0.030 | 0.927 | 2 |
| hicregwindowcrosschrom_k562_25kb | 0.736 | 0.696 | 0.484 | 0.523 | 0.038 | 0.927 | 2 |
| hicregwindowcv_k562_25kb | 0.888 | 0.864 | 0.670 | 0.688 | 0.178 | 0.973 | 2 |

Chromosomes: chr14, chr17.

## C.Origami block, imr90, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.884 | 0.886 | 0.942 | 0.575 | 0.316 | 0.964 | 2 |
| release twin, ChIP only | 0.777 | 0.782 | 0.871 | 0.483 | 0.217 | 0.901 | 2 |
| one model (no dropout), consensus input | 0.887 | 0.890 | 0.949 | 0.550 | 0.335 | 0.967 | 2 |
| one model (no dropout), ChIP only | 0.483 | 0.503 | 0.704 | 0.325 | 0.049 | 0.728 | 2 |
| per-resolution, consensus input | 0.882 | 0.884 | 0.936 | 0.547 | 0.302 | 0.964 | 2 |
| per-resolution, ChIP only | 0.756 | 0.760 | 0.834 | 0.462 | 0.177 | 0.893 | 2 |
| per-resolution, consensus residual | 0.878 | 0.880 | 0.928 | 0.555 | 0.319 | 0.961 | 2 |
| consensus | 0.819 | 0.815 | 0.803 | 0.581 | 0.339 | 0.902 | 2 |
| best_other_map | 0.796 | 0.798 | 0.779 | 0.537 | 0.168 | 0.919 | 2 |
| replicate | 0.670 | 0.755 | 0.937 | 0.513 | 0.152 | 0.902 | 2 |
| cascfold | 0.887 | 0.890 | 0.948 | 0.578 | 0.314 | 0.966 | 2 |
| corigami_imr90_8kb | 0.849 | 0.854 | 0.852 | 0.348 | 0.164 | 0.956 | 2 |

Chromosomes: chr2, chr3.

## Epiphany block, gm12878, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.898 | 0.897 | 0.914 | 0.544 | 0.460 | 0.955 | 3 |
| release twin, ChIP only | 0.773 | 0.774 | 0.800 | 0.476 | 0.289 | 0.866 | 3 |
| one model (no dropout), consensus input | 0.901 | 0.901 | 0.918 | 0.577 | 0.478 | 0.953 | 3 |
| one model (no dropout), ChIP only | 0.520 | 0.538 | 0.617 | 0.362 | 0.035 | 0.680 | 3 |
| per-resolution, consensus input | 0.894 | 0.891 | 0.889 | 0.577 | 0.430 | 0.953 | 3 |
| per-resolution, ChIP only | 0.737 | 0.734 | 0.726 | 0.472 | 0.222 | 0.844 | 3 |
| per-resolution, consensus residual | 0.887 | 0.881 | 0.864 | 0.566 | 0.424 | 0.947 | 3 |
| consensus | 0.793 | 0.790 | 0.784 | 0.560 | 0.416 | 0.876 | 3 |
| best_other_map | 0.650 | 0.654 | 0.697 | 0.483 | 0.103 | 0.820 | 3 |
| replicate | 0.700 | 0.732 | 0.895 | 0.571 | 0.212 | 0.918 | 3 |
| cascfold | 0.899 | 0.897 | 0.909 | 0.560 | 0.450 | 0.957 | 3 |
| epiphany_gm12878_10kb | 0.523 | 0.504 | 0.310 | 0.133 | 0.043 | 0.770 | 3 |

Chromosomes: chr11, chr17, chr3.

## Hi-cGAN block, gm12878, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.888 | 0.891 | 0.910 | 0.526 | 0.371 | 0.960 | 2 |
| release twin, ChIP only | 0.765 | 0.770 | 0.799 | 0.459 | 0.204 | 0.892 | 2 |
| one model (no dropout), consensus input | 0.891 | 0.894 | 0.912 | 0.551 | 0.380 | 0.961 | 2 |
| one model (no dropout), ChIP only | 0.438 | 0.467 | 0.618 | 0.359 | 0.027 | 0.708 | 2 |
| per-resolution, consensus input | 0.882 | 0.882 | 0.886 | 0.540 | 0.337 | 0.954 | 2 |
| per-resolution, ChIP only | 0.734 | 0.740 | 0.737 | 0.459 | 0.161 | 0.881 | 2 |
| per-resolution, consensus residual | 0.873 | 0.873 | 0.851 | 0.547 | 0.318 | 0.953 | 2 |
| consensus | 0.791 | 0.792 | 0.759 | 0.538 | 0.319 | 0.882 | 2 |
| best_other_map | 0.575 | 0.584 | 0.652 | 0.434 | 0.020 | 0.816 | 2 |
| replicate | 0.664 | 0.713 | 0.898 | 0.555 | 0.151 | 0.908 | 2 |
| cascfold | 0.882 | 0.883 | 0.888 | 0.535 | 0.367 | 0.959 | 2 |
| hicgan_gm12878_5kb | 0.346 | 0.353 | 0.397 | 0.124 | 0.068 | 0.638 | 2 |

Chromosomes: chr14, chr2.

## HiC-Reg block, gm12878, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.911 | 0.910 | 0.922 | 0.525 | 0.393 | 0.960 | 2 |
| release twin, ChIP only | 0.789 | 0.789 | 0.800 | 0.447 | 0.227 | 0.885 | 2 |
| one model (no dropout), consensus input | 0.916 | 0.914 | 0.920 | 0.572 | 0.410 | 0.961 | 2 |
| one model (no dropout), ChIP only | 0.509 | 0.529 | 0.629 | 0.356 | 0.030 | 0.696 | 2 |
| per-resolution, consensus input | 0.906 | 0.902 | 0.898 | 0.562 | 0.363 | 0.957 | 2 |
| per-resolution, ChIP only | 0.749 | 0.749 | 0.741 | 0.455 | 0.188 | 0.871 | 2 |
| per-resolution, consensus residual | 0.899 | 0.894 | 0.873 | 0.568 | 0.349 | 0.954 | 2 |
| consensus | 0.821 | 0.817 | 0.798 | 0.556 | 0.354 | 0.885 | 2 |
| best_other_map | 0.640 | 0.652 | 0.744 | 0.481 | 0.063 | 0.818 | 2 |
| replicate | 0.693 | 0.740 | 0.900 | 0.589 | 0.206 | 0.911 | 2 |
| cascfold | 0.908 | 0.906 | 0.911 | 0.541 | 0.395 | 0.961 | 2 |
| hicregmulticellcrosschrom_gm12878_5kb | 0.626 | 0.571 | 0.525 | 0.195 | 0.000 | 0.890 | 2 |
| hicregwindowcrosschrom_gm12878_5kb | 0.603 | 0.554 | 0.515 | 0.161 | 0.000 | 0.891 | 2 |
| hicregwindowcv_gm12878_5kb | 0.893 | 0.859 | 0.647 | 0.252 | 0.020 | 0.992 | 2 |

Chromosomes: chr14, chr17.

## HiC-Reg block, k562, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| release twin, consensus input | 0.769 | 0.776 | 0.859 | 0.530 | 0.367 | 0.925 | 2 |
| release twin, ChIP only | 0.657 | 0.663 | 0.734 | 0.438 | 0.199 | 0.847 | 2 |
| one model (no dropout), consensus input | 0.777 | 0.783 | 0.859 | 0.510 | 0.366 | 0.930 | 2 |
| one model (no dropout), ChIP only | 0.527 | 0.542 | 0.697 | 0.364 | 0.071 | 0.692 | 2 |
| per-resolution, consensus input | 0.771 | 0.776 | 0.842 | 0.510 | 0.338 | 0.925 | 2 |
| per-resolution, ChIP only | 0.641 | 0.653 | 0.730 | 0.415 | 0.169 | 0.831 | 2 |
| per-resolution, consensus residual | 0.784 | 0.789 | 0.869 | 0.509 | 0.338 | 0.922 | 2 |
| consensus | 0.673 | 0.685 | 0.810 | 0.505 | 0.323 | 0.875 | 2 |
| best_other_map | 0.570 | 0.580 | 0.581 | 0.417 | 0.130 | 0.837 | 2 |
| replicate | 0.706 | 0.707 | 0.774 | 0.546 | 0.183 | 0.965 | 2 |
| hicregmulticellcrosschrom_k562_5kb | 0.783 | 0.768 | 0.640 | 0.271 | 0.003 | 0.962 | 2 |
| hicregwindowcrosschrom_k562_5kb | 0.774 | 0.755 | 0.599 | 0.226 | 0.003 | 0.959 | 2 |
| hicregwindowcv_k562_5kb | 0.845 | 0.840 | 0.728 | 0.282 | 0.005 | 0.978 | 2 |

Chromosomes: chr14, chr17.

## Differences of the release twin

Mean over the held-out chromosomes of the block; positive = the first model is higher. Wins = held-out chromosomes (of all blocks of the table family) on which the first model is higher.

### release twin, consensus input minus per-resolution, consensus input

| block | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | +0.002 | +0.004 | +0.020 | +0.014 | -0.004 | +0.003 |
| Epiphany block, gm12878, 10 kb | +0.007 | +0.010 | +0.030 | +0.005 | -0.017 | +0.005 |
| Hi-cGAN block, gm12878, 10 kb | +0.005 | +0.007 | +0.028 | +0.004 | +0.009 | +0.009 |
| Hi-cGAN block, k562, 10 kb | +0.011 | +0.012 | +0.017 | -0.001 | +0.026 | +0.005 |
| C.Origami block, imr90, 25 kb | -0.008 | -0.006 | +0.001 | +0.003 | +0.001 | -0.005 |
| Epiphany block, gm12878, 25 kb | +0.002 | +0.002 | -0.001 | +0.001 | -0.006 | +0.003 |
| Hi-cGAN block, gm12878, 25 kb | -0.002 | -0.002 | +0.001 | +0.001 | -0.024 | -0.005 |
| Hi-cGAN block, k562, 25 kb | +0.005 | +0.004 | -0.004 | +0.019 | +0.007 | +0.006 |
| HiC-Reg block, gm12878, 25 kb | +0.003 | +0.002 | +0.001 | -0.021 | -0.014 | +0.004 |
| HiC-Reg block, k562, 25 kb | -0.000 | -0.003 | -0.015 | +0.022 | -0.010 | +0.000 |
| C.Origami block, imr90, 5 kb | +0.002 | +0.002 | +0.006 | +0.028 | +0.013 | +0.000 |
| Epiphany block, gm12878, 5 kb | +0.004 | +0.007 | +0.025 | -0.034 | +0.030 | +0.002 |
| Hi-cGAN block, gm12878, 5 kb | +0.006 | +0.009 | +0.024 | -0.014 | +0.034 | +0.005 |
| HiC-Reg block, gm12878, 5 kb | +0.005 | +0.007 | +0.024 | -0.037 | +0.030 | +0.003 |
| HiC-Reg block, k562, 5 kb | -0.002 | -0.000 | +0.017 | +0.020 | +0.029 | +0.001 |
| mean of the 4 figure blocks, 10 kb | +0.006 | +0.008 | +0.024 | +0.005 | +0.003 | +0.006 |
| mean of the 4 figure blocks, 25 kb | -0.001 | -0.001 | -0.000 | +0.006 | -0.005 | -0.000 |
| mean of the 3 figure blocks, 5 kb | +0.004 | +0.006 | +0.018 | -0.006 | +0.026 | +0.002 |
| chromosome wins, figure blocks, 10 kb | 9 of 9 | 9 of 9 | 9 of 9 | 7 of 9 | 3 of 9 | 8 of 9 |
| chromosome wins, figure blocks, 25 kb | 5 of 9 | 4 of 9 | 4 of 9 | 6 of 9 | 4 of 9 | 5 of 9 |
| chromosome wins, figure blocks, 5 kb | 7 of 7 | 7 of 7 | 7 of 7 | 2 of 7 | 7 of 7 | 5 of 7 |

### release twin, ChIP only minus per-resolution, ChIP only

| block | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | +0.003 | +0.006 | +0.047 | +0.031 | +0.032 | +0.004 |
| Epiphany block, gm12878, 10 kb | +0.031 | +0.036 | +0.068 | +0.032 | +0.054 | +0.026 |
| Hi-cGAN block, gm12878, 10 kb | +0.006 | +0.012 | +0.056 | +0.002 | +0.051 | +0.008 |
| Hi-cGAN block, k562, 10 kb | -0.001 | +0.001 | -0.004 | +0.030 | +0.026 | -0.014 |
| C.Origami block, imr90, 25 kb | -0.026 | -0.015 | +0.058 | +0.006 | +0.013 | -0.002 |
| Epiphany block, gm12878, 25 kb | -0.011 | +0.000 | +0.056 | +0.017 | +0.015 | -0.004 |
| Hi-cGAN block, gm12878, 25 kb | -0.021 | -0.013 | +0.039 | -0.046 | +0.000 | -0.020 |
| Hi-cGAN block, k562, 25 kb | +0.007 | +0.013 | +0.025 | +0.005 | +0.015 | +0.011 |
| HiC-Reg block, gm12878, 25 kb | -0.019 | -0.007 | +0.058 | -0.046 | +0.007 | -0.031 |
| HiC-Reg block, k562, 25 kb | -0.009 | -0.005 | +0.019 | +0.001 | +0.035 | -0.009 |
| C.Origami block, imr90, 5 kb | +0.021 | +0.022 | +0.036 | +0.021 | +0.040 | +0.009 |
| Epiphany block, gm12878, 5 kb | +0.036 | +0.040 | +0.074 | +0.004 | +0.067 | +0.022 |
| Hi-cGAN block, gm12878, 5 kb | +0.031 | +0.030 | +0.063 | +0.000 | +0.043 | +0.012 |
| HiC-Reg block, gm12878, 5 kb | +0.040 | +0.040 | +0.059 | -0.008 | +0.039 | +0.014 |
| HiC-Reg block, k562, 5 kb | +0.016 | +0.010 | +0.004 | +0.023 | +0.031 | +0.015 |
| mean of the 4 figure blocks, 10 kb | +0.010 | +0.014 | +0.042 | +0.024 | +0.041 | +0.006 |
| mean of the 4 figure blocks, 25 kb | -0.013 | -0.004 | +0.044 | -0.005 | +0.011 | -0.004 |
| mean of the 3 figure blocks, 5 kb | +0.030 | +0.031 | +0.058 | +0.008 | +0.050 | +0.014 |
| chromosome wins, figure blocks, 10 kb | 6 of 9 | 8 of 9 | 8 of 9 | 8 of 9 | 9 of 9 | 6 of 9 |
| chromosome wins, figure blocks, 25 kb | 2 of 9 | 4 of 9 | 9 of 9 | 5 of 9 | 7 of 9 | 4 of 9 |
| chromosome wins, figure blocks, 5 kb | 7 of 7 | 7 of 7 | 7 of 7 | 4 of 7 | 7 of 7 | 7 of 7 |

### release twin, consensus input minus one model (no dropout), consensus input

| block | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | -0.005 | -0.005 | -0.007 | +0.021 | -0.012 | -0.003 |
| Epiphany block, gm12878, 10 kb | -0.004 | -0.005 | -0.007 | +0.004 | -0.020 | +0.001 |
| Hi-cGAN block, gm12878, 10 kb | -0.005 | -0.005 | -0.008 | -0.005 | -0.005 | -0.000 |
| Hi-cGAN block, k562, 10 kb | -0.006 | -0.004 | +0.006 | -0.015 | +0.001 | -0.004 |
| C.Origami block, imr90, 25 kb | -0.011 | -0.010 | -0.007 | +0.023 | -0.002 | -0.005 |
| Epiphany block, gm12878, 25 kb | -0.004 | -0.006 | -0.008 | +0.003 | -0.006 | +0.002 |
| Hi-cGAN block, gm12878, 25 kb | -0.004 | -0.005 | -0.010 | +0.033 | -0.010 | -0.006 |
| Hi-cGAN block, k562, 25 kb | +0.001 | +0.001 | +0.001 | +0.012 | +0.012 | +0.006 |
| HiC-Reg block, gm12878, 25 kb | -0.003 | -0.003 | -0.005 | +0.012 | -0.003 | +0.004 |
| HiC-Reg block, k562, 25 kb | -0.008 | -0.008 | -0.006 | +0.012 | +0.009 | -0.004 |
| C.Origami block, imr90, 5 kb | -0.003 | -0.004 | -0.007 | +0.026 | -0.019 | -0.003 |
| Epiphany block, gm12878, 5 kb | -0.003 | -0.004 | -0.004 | -0.033 | -0.018 | +0.002 |
| Hi-cGAN block, gm12878, 5 kb | -0.003 | -0.003 | -0.002 | -0.025 | -0.009 | -0.001 |
| HiC-Reg block, gm12878, 5 kb | -0.006 | -0.005 | +0.002 | -0.047 | -0.016 | -0.001 |
| HiC-Reg block, k562, 5 kb | -0.008 | -0.007 | +0.000 | +0.020 | +0.001 | -0.004 |
| mean of the 4 figure blocks, 10 kb | -0.005 | -0.005 | -0.004 | +0.001 | -0.009 | -0.002 |
| mean of the 4 figure blocks, 25 kb | -0.004 | -0.005 | -0.006 | +0.018 | -0.001 | -0.001 |
| mean of the 3 figure blocks, 5 kb | -0.003 | -0.004 | -0.004 | -0.011 | -0.015 | -0.001 |
| chromosome wins, figure blocks, 10 kb | 2 of 9 | 0 of 9 | 2 of 9 | 5 of 9 | 2 of 9 | 3 of 9 |
| chromosome wins, figure blocks, 25 kb | 2 of 9 | 2 of 9 | 1 of 9 | 8 of 9 | 3 of 9 | 4 of 9 |
| chromosome wins, figure blocks, 5 kb | 1 of 7 | 0 of 7 | 1 of 7 | 2 of 7 | 0 of 7 | 3 of 7 |

### release twin, ChIP only minus one model (no dropout), ChIP only

| block | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | +0.306 | +0.293 | +0.170 | +0.168 | +0.170 | +0.171 |
| Epiphany block, gm12878, 10 kb | +0.269 | +0.248 | +0.188 | +0.195 | +0.209 | +0.181 |
| Hi-cGAN block, gm12878, 10 kb | +0.358 | +0.320 | +0.180 | +0.194 | +0.157 | +0.155 |
| Hi-cGAN block, k562, 10 kb | +0.207 | +0.177 | +0.054 | +0.126 | +0.106 | +0.089 |
| C.Origami block, imr90, 25 kb | +0.276 | +0.267 | +0.188 | +0.209 | +0.162 | +0.155 |
| Epiphany block, gm12878, 25 kb | +0.268 | +0.246 | +0.179 | +0.288 | +0.175 | +0.203 |
| Hi-cGAN block, gm12878, 25 kb | +0.386 | +0.339 | +0.163 | +0.202 | +0.105 | +0.178 |
| Hi-cGAN block, k562, 25 kb | +0.199 | +0.171 | +0.099 | +0.119 | +0.040 | +0.119 |
| HiC-Reg block, gm12878, 25 kb | +0.315 | +0.280 | +0.159 | +0.218 | +0.128 | +0.225 |
| HiC-Reg block, k562, 25 kb | +0.144 | +0.145 | +0.148 | +0.147 | +0.089 | +0.128 |
| C.Origami block, imr90, 5 kb | +0.294 | +0.280 | +0.167 | +0.158 | +0.168 | +0.173 |
| Epiphany block, gm12878, 5 kb | +0.253 | +0.237 | +0.183 | +0.114 | +0.254 | +0.185 |
| Hi-cGAN block, gm12878, 5 kb | +0.328 | +0.303 | +0.182 | +0.100 | +0.177 | +0.184 |
| HiC-Reg block, gm12878, 5 kb | +0.279 | +0.259 | +0.172 | +0.091 | +0.197 | +0.189 |
| HiC-Reg block, k562, 5 kb | +0.130 | +0.121 | +0.037 | +0.074 | +0.128 | +0.154 |
| mean of the 4 figure blocks, 10 kb | +0.285 | +0.260 | +0.148 | +0.171 | +0.160 | +0.149 |
| mean of the 4 figure blocks, 25 kb | +0.282 | +0.256 | +0.157 | +0.205 | +0.121 | +0.164 |
| mean of the 3 figure blocks, 5 kb | +0.292 | +0.273 | +0.177 | +0.124 | +0.200 | +0.181 |
| chromosome wins, figure blocks, 10 kb | 9 of 9 | 9 of 9 | 9 of 9 | 9 of 9 | 9 of 9 | 9 of 9 |
| chromosome wins, figure blocks, 25 kb | 9 of 9 | 9 of 9 | 9 of 9 | 9 of 9 | 9 of 9 | 9 of 9 |
| chromosome wins, figure blocks, 5 kb | 7 of 7 | 7 of 7 | 7 of 7 | 7 of 7 | 7 of 7 | 7 of 7 |

### release twin, ChIP only minus release twin, consensus input

| block | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | -0.116 | -0.113 | -0.074 | -0.140 | -0.127 | -0.074 |
| Epiphany block, gm12878, 10 kb | -0.130 | -0.130 | -0.122 | -0.131 | -0.204 | -0.086 |
| Hi-cGAN block, gm12878, 10 kb | -0.123 | -0.125 | -0.121 | -0.126 | -0.193 | -0.078 |
| Hi-cGAN block, k562, 10 kb | -0.125 | -0.126 | -0.115 | -0.090 | -0.204 | -0.083 |
| C.Origami block, imr90, 25 kb | -0.153 | -0.150 | -0.112 | -0.168 | -0.131 | -0.093 |
| Epiphany block, gm12878, 25 kb | -0.175 | -0.175 | -0.175 | -0.204 | -0.190 | -0.110 |
| Hi-cGAN block, gm12878, 25 kb | -0.157 | -0.163 | -0.165 | -0.239 | -0.181 | -0.091 |
| Hi-cGAN block, k562, 25 kb | -0.137 | -0.145 | -0.157 | -0.190 | -0.208 | -0.082 |
| HiC-Reg block, gm12878, 25 kb | -0.171 | -0.172 | -0.175 | -0.249 | -0.183 | -0.121 |
| HiC-Reg block, k562, 25 kb | -0.175 | -0.180 | -0.201 | -0.191 | -0.177 | -0.102 |
| C.Origami block, imr90, 5 kb | -0.107 | -0.104 | -0.071 | -0.093 | -0.099 | -0.063 |
| Epiphany block, gm12878, 5 kb | -0.125 | -0.123 | -0.114 | -0.068 | -0.170 | -0.089 |
| Hi-cGAN block, gm12878, 5 kb | -0.123 | -0.121 | -0.111 | -0.067 | -0.167 | -0.067 |
| HiC-Reg block, gm12878, 5 kb | -0.122 | -0.121 | -0.122 | -0.078 | -0.166 | -0.074 |
| HiC-Reg block, k562, 5 kb | -0.112 | -0.113 | -0.125 | -0.092 | -0.168 | -0.079 |
| mean of the 4 figure blocks, 10 kb | -0.124 | -0.124 | -0.108 | -0.122 | -0.182 | -0.080 |
| mean of the 4 figure blocks, 25 kb | -0.156 | -0.158 | -0.152 | -0.200 | -0.177 | -0.094 |
| mean of the 3 figure blocks, 5 kb | -0.118 | -0.116 | -0.099 | -0.076 | -0.146 | -0.073 |
| chromosome wins, figure blocks, 10 kb | 0 of 9 | 0 of 9 | 0 of 9 | 0 of 9 | 0 of 9 | 0 of 9 |
| chromosome wins, figure blocks, 25 kb | 0 of 9 | 0 of 9 | 0 of 9 | 0 of 9 | 0 of 9 | 0 of 9 |
| chromosome wins, figure blocks, 5 kb | 0 of 7 | 0 of 7 | 0 of 7 | 0 of 7 | 0 of 7 | 0 of 7 |

## Competitors that exceed the release twin

Competitor block mean above the twin's block mean in a measure (the twin with the consensus and ChIP only listed separately); the per-resolution counterpart is given for comparison.

### release twin, consensus input

| block | competitor | measure | competitor | twin | difference | per-resolution model |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | corigami_imr90_8kb | compartment r | 0.972 | 0.964 | +0.009 | 0.960 |
| C.Origami block, imr90, 25 kb | corigami_imr90_8kb | Pearson | 0.942 | 0.932 | +0.010 | 0.940 |
| C.Origami block, imr90, 25 kb | corigami_imr90_8kb | compartment r | 0.980 | 0.953 | +0.027 | 0.958 |
| HiC-Reg block, gm12878, 25 kb | hicregwindowcv_gm12878_25kb | compartment r | 0.987 | 0.958 | +0.030 | 0.953 |
| HiC-Reg block, k562, 25 kb | hicregwindowcv_k562_25kb | compartment r | 0.973 | 0.928 | +0.045 | 0.927 |
| HiC-Reg block, gm12878, 5 kb | hicregwindowcv_gm12878_5kb | compartment r | 0.992 | 0.960 | +0.032 | 0.957 |
| HiC-Reg block, k562, 5 kb | hicregmulticellcrosschrom_k562_5kb | Pearson | 0.783 | 0.769 | +0.014 | 0.771 |
| HiC-Reg block, k562, 5 kb | hicregmulticellcrosschrom_k562_5kb | compartment r | 0.962 | 0.925 | +0.037 | 0.925 |
| HiC-Reg block, k562, 5 kb | hicregwindowcrosschrom_k562_5kb | Pearson | 0.774 | 0.769 | +0.004 | 0.771 |
| HiC-Reg block, k562, 5 kb | hicregwindowcrosschrom_k562_5kb | compartment r | 0.959 | 0.925 | +0.033 | 0.925 |
| HiC-Reg block, k562, 5 kb | hicregwindowcv_k562_5kb | Pearson | 0.845 | 0.769 | +0.075 | 0.771 |
| HiC-Reg block, k562, 5 kb | hicregwindowcv_k562_5kb | SCC | 0.840 | 0.776 | +0.064 | 0.776 |
| HiC-Reg block, k562, 5 kb | hicregwindowcv_k562_5kb | compartment r | 0.978 | 0.925 | +0.052 | 0.925 |

### release twin, ChIP only

| block | competitor | measure | competitor | twin | difference | per-resolution model |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | corigami_imr90_8kb | Pearson | 0.920 | 0.812 | +0.108 | 0.808 |
| C.Origami block, imr90, 10 kb | corigami_imr90_8kb | SCC | 0.913 | 0.818 | +0.095 | 0.812 |
| C.Origami block, imr90, 10 kb | corigami_imr90_8kb | compartment r | 0.972 | 0.889 | +0.083 | 0.885 |
| C.Origami block, imr90, 25 kb | corigami_imr90_8kb | Pearson | 0.942 | 0.778 | +0.163 | 0.804 |
| C.Origami block, imr90, 25 kb | corigami_imr90_8kb | SCC | 0.933 | 0.785 | +0.148 | 0.800 |
| C.Origami block, imr90, 25 kb | corigami_imr90_8kb | insulation r | 0.898 | 0.833 | +0.065 | 0.775 |
| C.Origami block, imr90, 25 kb | corigami_imr90_8kb | boundary recall | 0.752 | 0.652 | +0.099 | 0.647 |
| C.Origami block, imr90, 25 kb | corigami_imr90_8kb | loop recall | 0.313 | 0.219 | +0.094 | 0.206 |
| C.Origami block, imr90, 25 kb | corigami_imr90_8kb | compartment r | 0.980 | 0.860 | +0.120 | 0.862 |
| C.Origami block, imr90, 25 kb | hicgan_imr90_on_imr90_25kb | Pearson | 0.808 | 0.778 | +0.030 | 0.804 |
| C.Origami block, imr90, 25 kb | hicgan_imr90_on_imr90_25kb | SCC | 0.792 | 0.785 | +0.007 | 0.800 |
| C.Origami block, imr90, 25 kb | hicgan_imr90_on_imr90_25kb | loop recall | 0.258 | 0.219 | +0.040 | 0.206 |
| C.Origami block, imr90, 25 kb | hicgan_imr90_on_imr90_25kb | compartment r | 0.945 | 0.860 | +0.085 | 0.862 |
| HiC-Reg block, gm12878, 25 kb | hicregmulticellcrosschrom_gm12878_25kb | compartment r | 0.856 | 0.837 | +0.019 | 0.867 |
| HiC-Reg block, gm12878, 25 kb | hicregwindowcrosschrom_gm12878_25kb | compartment r | 0.846 | 0.837 | +0.009 | 0.867 |
| HiC-Reg block, gm12878, 25 kb | hicregwindowcv_gm12878_25kb | Pearson | 0.908 | 0.764 | +0.145 | 0.782 |
| HiC-Reg block, gm12878, 25 kb | hicregwindowcv_gm12878_25kb | SCC | 0.870 | 0.756 | +0.114 | 0.763 |
| HiC-Reg block, gm12878, 25 kb | hicregwindowcv_gm12878_25kb | boundary recall | 0.723 | 0.563 | +0.161 | 0.608 |
| HiC-Reg block, gm12878, 25 kb | hicregwindowcv_gm12878_25kb | loop recall | 0.323 | 0.198 | +0.125 | 0.191 |
| HiC-Reg block, gm12878, 25 kb | hicregwindowcv_gm12878_25kb | compartment r | 0.987 | 0.837 | +0.151 | 0.867 |
| HiC-Reg block, k562, 25 kb | hicregmulticellcrosschrom_k562_25kb | Pearson | 0.740 | 0.721 | +0.019 | 0.730 |
| HiC-Reg block, k562, 25 kb | hicregmulticellcrosschrom_k562_25kb | compartment r | 0.927 | 0.825 | +0.101 | 0.834 |
| HiC-Reg block, k562, 25 kb | hicregwindowcrosschrom_k562_25kb | Pearson | 0.736 | 0.721 | +0.015 | 0.730 |
| HiC-Reg block, k562, 25 kb | hicregwindowcrosschrom_k562_25kb | compartment r | 0.927 | 0.825 | +0.101 | 0.834 |
| HiC-Reg block, k562, 25 kb | hicregwindowcv_k562_25kb | Pearson | 0.888 | 0.721 | +0.166 | 0.730 |
| HiC-Reg block, k562, 25 kb | hicregwindowcv_k562_25kb | SCC | 0.864 | 0.706 | +0.158 | 0.711 |
| HiC-Reg block, k562, 25 kb | hicregwindowcv_k562_25kb | insulation r | 0.670 | 0.640 | +0.030 | 0.622 |
| HiC-Reg block, k562, 25 kb | hicregwindowcv_k562_25kb | boundary recall | 0.688 | 0.569 | +0.119 | 0.568 |
| HiC-Reg block, k562, 25 kb | hicregwindowcv_k562_25kb | loop recall | 0.178 | 0.160 | +0.018 | 0.125 |
| HiC-Reg block, k562, 25 kb | hicregwindowcv_k562_25kb | compartment r | 0.973 | 0.825 | +0.147 | 0.834 |
| C.Origami block, imr90, 5 kb | corigami_imr90_8kb | Pearson | 0.849 | 0.777 | +0.072 | 0.756 |
| C.Origami block, imr90, 5 kb | corigami_imr90_8kb | SCC | 0.854 | 0.782 | +0.071 | 0.760 |
| C.Origami block, imr90, 5 kb | corigami_imr90_8kb | compartment r | 0.956 | 0.901 | +0.055 | 0.893 |
| HiC-Reg block, gm12878, 5 kb | hicregmulticellcrosschrom_gm12878_5kb | compartment r | 0.890 | 0.885 | +0.005 | 0.871 |
| HiC-Reg block, gm12878, 5 kb | hicregwindowcrosschrom_gm12878_5kb | compartment r | 0.891 | 0.885 | +0.006 | 0.871 |
| HiC-Reg block, gm12878, 5 kb | hicregwindowcv_gm12878_5kb | Pearson | 0.893 | 0.789 | +0.104 | 0.749 |
| HiC-Reg block, gm12878, 5 kb | hicregwindowcv_gm12878_5kb | SCC | 0.859 | 0.789 | +0.070 | 0.749 |
| HiC-Reg block, gm12878, 5 kb | hicregwindowcv_gm12878_5kb | compartment r | 0.992 | 0.885 | +0.106 | 0.871 |
| HiC-Reg block, k562, 5 kb | hicregmulticellcrosschrom_k562_5kb | Pearson | 0.783 | 0.657 | +0.126 | 0.641 |
| HiC-Reg block, k562, 5 kb | hicregmulticellcrosschrom_k562_5kb | SCC | 0.768 | 0.663 | +0.105 | 0.653 |
| HiC-Reg block, k562, 5 kb | hicregmulticellcrosschrom_k562_5kb | compartment r | 0.962 | 0.847 | +0.116 | 0.831 |
| HiC-Reg block, k562, 5 kb | hicregwindowcrosschrom_k562_5kb | Pearson | 0.774 | 0.657 | +0.117 | 0.641 |
| HiC-Reg block, k562, 5 kb | hicregwindowcrosschrom_k562_5kb | SCC | 0.755 | 0.663 | +0.093 | 0.653 |
| HiC-Reg block, k562, 5 kb | hicregwindowcrosschrom_k562_5kb | compartment r | 0.959 | 0.847 | +0.112 | 0.831 |
| HiC-Reg block, k562, 5 kb | hicregwindowcv_k562_5kb | Pearson | 0.845 | 0.657 | +0.188 | 0.641 |
| HiC-Reg block, k562, 5 kb | hicregwindowcv_k562_5kb | SCC | 0.840 | 0.663 | +0.177 | 0.653 |
| HiC-Reg block, k562, 5 kb | hicregwindowcv_k562_5kb | compartment r | 0.978 | 0.847 | +0.131 | 0.831 |

## Check of the older models

Largest absolute difference between the stored rows of the per-resolution models and the all12one_chrho pair and their rows in the blocks scored together with the release twin (`chrho_benchmark_release_oldmodel_check.csv`).

| table | measure | rows | max abs difference |
|---|---|---|---|
| chrho | oe_pearson | 85 | 0 |
| chrho | oe_scc | 85 | 0 |
| chrho | insulation_r | 85 | 0 |
| chrho | bnd_recall | 85 | 0 |
| chrho | loop_recall_rank | 85 | 0 |
| chrho | comp_pref_r | 85 | 0 |
| chrho_25kb | oe_pearson | 65 | 0 |
| chrho_25kb | oe_scc | 65 | 0 |
| chrho_25kb | insulation_r | 65 | 0 |
| chrho_25kb | bnd_recall | 65 | 0 |
| chrho_25kb | loop_recall_rank | 65 | 0 |
| chrho_25kb | comp_pref_r | 65 | 0 |
| chrho_5kb | oe_pearson | 55 | 0 |
| chrho_5kb | oe_scc | 55 | 0 |
| chrho_5kb | insulation_r | 55 | 0 |
| chrho_5kb | bnd_recall | 55 | 0 |
| chrho_5kb | loop_recall_rank | 55 | 0 |
| chrho_5kb | comp_pref_r | 55 | 0 |
