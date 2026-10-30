# One model on the held-out-chromosome benchmark

Mean over the held-out chromosomes of each block (the value the benchmark figure reads); one model = `all12one_chrho` (one network for 2, 5, 10 and 25 kb, trained without chr2, 3, 11, 14, 17 of every cell), consensus input and, from the same checkpoint, ChIP only (reference plane 0); per-resolution = `all12refin_drop_chrho` and `all12honest_drop_chrho`. Per-chromosome values: `chrho_benchmark.csv`.

## C.Origami block, imr90, 10 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.933 | 0.936 | 0.951 | 0.647 | 0.362 | 0.966 | 2 |
| one model, ChIP only | 0.506 | 0.525 | 0.700 | 0.360 | 0.053 | 0.718 | 2 |
| per-resolution, consensus input | 0.926 | 0.927 | 0.924 | 0.654 | 0.355 | 0.960 | 2 |
| per-resolution, ChIP only | 0.808 | 0.812 | 0.822 | 0.497 | 0.192 | 0.885 | 2 |
| all12refres_drop_chrho | 0.928 | 0.931 | 0.943 | 0.658 | 0.364 | 0.964 | 2 |
| consensus | 0.866 | 0.862 | 0.796 | 0.641 | 0.383 | 0.893 | 2 |
| cascfold | 0.932 | 0.935 | 0.947 | 0.658 | 0.365 | 0.967 | 2 |
| best_other_map | 0.873 | 0.868 | 0.783 | 0.623 | 0.286 | 0.917 | 2 |
| corigami_imr90_8kb | 0.920 | 0.913 | 0.866 | 0.473 | 0.162 | 0.972 | 2 |
| hicgan_gm12878_on_imr90_10kb | 0.291 | 0.314 | 0.564 | 0.137 | 0.040 | 0.717 | 1 |
| hicgan_imr90_on_imr90_10kb | 0.567 | 0.567 | 0.584 | 0.105 | 0.014 | 0.734 | 1 |
| replicate | 0.885 | 0.901 | 0.976 | 0.684 | 0.034 | 0.983 | 2 |

Chromosomes: chr2, chr3.

## Epiphany block, gm12878, 10 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.925 | 0.922 | 0.911 | 0.656 | 0.498 | 0.956 | 3 |
| one model, ChIP only | 0.522 | 0.539 | 0.595 | 0.334 | 0.065 | 0.690 | 3 |
| per-resolution, consensus input | 0.914 | 0.908 | 0.875 | 0.655 | 0.495 | 0.951 | 3 |
| per-resolution, ChIP only | 0.760 | 0.751 | 0.715 | 0.497 | 0.220 | 0.844 | 3 |
| all12refres_drop_chrho | 0.919 | 0.915 | 0.892 | 0.676 | 0.474 | 0.956 | 3 |
| consensus | 0.826 | 0.818 | 0.778 | 0.660 | 0.452 | 0.875 | 3 |
| cascfold | 0.920 | 0.916 | 0.904 | 0.664 | 0.460 | 0.957 | 3 |
| best_other_map | 0.748 | 0.744 | 0.711 | 0.601 | 0.258 | 0.821 | 3 |
| epiphany_gm12878_10kb | 0.512 | 0.501 | 0.318 | 0.208 | 0.052 | 0.776 | 3 |
| replicate | 0.871 | 0.874 | 0.899 | 0.702 | 0.145 | 0.963 | 3 |

Chromosomes: chr11, chr17, chr3.

## Hi-cGAN block, gm12878, 10 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.931 | 0.929 | 0.910 | 0.629 | 0.414 | 0.963 | 2 |
| one model, ChIP only | 0.445 | 0.478 | 0.601 | 0.304 | 0.059 | 0.730 | 2 |
| per-resolution, consensus input | 0.921 | 0.917 | 0.875 | 0.620 | 0.400 | 0.953 | 2 |
| per-resolution, ChIP only | 0.797 | 0.787 | 0.726 | 0.497 | 0.164 | 0.876 | 2 |
| all12refres_drop_chrho | 0.924 | 0.921 | 0.890 | 0.640 | 0.407 | 0.960 | 2 |
| consensus | 0.841 | 0.833 | 0.757 | 0.611 | 0.383 | 0.882 | 2 |
| cascfold | 0.922 | 0.921 | 0.900 | 0.617 | 0.399 | 0.960 | 2 |
| best_other_map | 0.735 | 0.732 | 0.684 | 0.540 | 0.146 | 0.819 | 2 |
| hicgan_gm12878_10kb | 0.328 | 0.330 | 0.260 | 0.113 | 0.031 | 0.700 | 2 |
| replicate | 0.860 | 0.869 | 0.896 | 0.674 | 0.133 | 0.960 | 2 |

Chromosomes: chr14, chr2.

## Hi-cGAN block, k562, 10 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.861 | 0.860 | 0.837 | 0.573 | 0.388 | 0.946 | 2 |
| one model, ChIP only | 0.523 | 0.553 | 0.674 | 0.342 | 0.080 | 0.769 | 2 |
| per-resolution, consensus input | 0.844 | 0.844 | 0.825 | 0.559 | 0.364 | 0.937 | 2 |
| per-resolution, ChIP only | 0.731 | 0.728 | 0.732 | 0.438 | 0.160 | 0.872 | 2 |
| all12refres_drop_chrho | 0.849 | 0.850 | 0.830 | 0.571 | 0.382 | 0.940 | 2 |
| consensus | 0.792 | 0.794 | 0.798 | 0.577 | 0.360 | 0.890 | 2 |
| best_other_map | 0.724 | 0.715 | 0.576 | 0.494 | 0.248 | 0.827 | 2 |
| hicgan_gm12878_on_k562_10kb | 0.125 | 0.141 | 0.339 | 0.108 | 0.008 | 0.519 | 2 |
| replicate | 0.838 | 0.836 | 0.762 | 0.592 | 0.266 | 0.959 | 2 |

Chromosomes: chr14, chr2.

## C.Origami block, imr90, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.943 | 0.946 | 0.952 | 0.797 | 0.352 | 0.958 | 2 |
| one model, ChIP only | 0.502 | 0.518 | 0.645 | 0.444 | 0.057 | 0.705 | 2 |
| per-resolution, consensus input | 0.940 | 0.941 | 0.943 | 0.817 | 0.349 | 0.958 | 2 |
| per-resolution, ChIP only | 0.804 | 0.800 | 0.775 | 0.647 | 0.206 | 0.862 | 2 |
| all12refres_drop_chrho | 0.940 | 0.942 | 0.946 | 0.807 | 0.335 | 0.960 | 2 |
| consensus | 0.877 | 0.872 | 0.817 | 0.788 | 0.374 | 0.885 | 2 |
| cascfold | 0.940 | 0.943 | 0.952 | 0.827 | 0.336 | 0.959 | 2 |
| best_other_map | 0.894 | 0.887 | 0.817 | 0.792 | 0.364 | 0.915 | 2 |
| corigami_imr90_8kb | 0.942 | 0.933 | 0.898 | 0.752 | 0.313 | 0.980 | 2 |
| hicgan_gm12878_on_imr90_25kb | 0.335 | 0.349 | 0.443 | 0.309 | 0.075 | 0.739 | 1 |
| hicgan_imr90_on_imr90_25kb | 0.808 | 0.792 | 0.630 | 0.618 | 0.258 | 0.945 | 1 |
| replicate | 0.976 | 0.979 | 0.988 | 0.891 | 0.318 | 0.993 | 2 |

Chromosomes: chr2, chr3.

## Epiphany block, gm12878, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.923 | 0.920 | 0.907 | 0.825 | 0.452 | 0.949 | 3 |
| one model, ChIP only | 0.476 | 0.493 | 0.545 | 0.336 | 0.081 | 0.638 | 3 |
| per-resolution, consensus input | 0.917 | 0.913 | 0.900 | 0.827 | 0.453 | 0.948 | 3 |
| per-resolution, ChIP only | 0.754 | 0.739 | 0.668 | 0.608 | 0.241 | 0.846 | 3 |
| all12refres_drop_chrho | 0.922 | 0.916 | 0.902 | 0.837 | 0.451 | 0.950 | 3 |
| consensus | 0.841 | 0.832 | 0.804 | 0.805 | 0.456 | 0.878 | 3 |
| cascfold | 0.922 | 0.918 | 0.911 | 0.843 | 0.442 | 0.951 | 3 |
| best_other_map | 0.775 | 0.769 | 0.744 | 0.780 | 0.389 | 0.821 | 3 |
| epiphany_gm12878_10kb | 0.518 | 0.501 | 0.377 | 0.437 | 0.254 | 0.775 | 3 |
| replicate | 0.946 | 0.939 | 0.922 | 0.906 | 0.417 | 0.976 | 3 |

Chromosomes: chr11, chr17, chr3.

## Hi-cGAN block, gm12878, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.936 | 0.932 | 0.912 | 0.776 | 0.391 | 0.960 | 2 |
| one model, ChIP only | 0.388 | 0.426 | 0.574 | 0.368 | 0.095 | 0.686 | 2 |
| per-resolution, consensus input | 0.934 | 0.929 | 0.901 | 0.808 | 0.405 | 0.959 | 2 |
| per-resolution, ChIP only | 0.796 | 0.778 | 0.698 | 0.616 | 0.200 | 0.883 | 2 |
| all12refres_drop_chrho | 0.932 | 0.927 | 0.893 | 0.810 | 0.373 | 0.957 | 2 |
| consensus | 0.860 | 0.848 | 0.784 | 0.777 | 0.389 | 0.897 | 2 |
| cascfold | 0.928 | 0.926 | 0.907 | 0.818 | 0.358 | 0.956 | 2 |
| best_other_map | 0.793 | 0.782 | 0.719 | 0.737 | 0.320 | 0.836 | 2 |
| hicgan_gm12878_25kb | 0.221 | 0.225 | 0.237 | 0.218 | 0.048 | 0.644 | 2 |
| replicate | 0.941 | 0.938 | 0.923 | 0.856 | 0.348 | 0.970 | 2 |

Chromosomes: chr14, chr2.

## Hi-cGAN block, k562, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.908 | 0.899 | 0.838 | 0.733 | 0.351 | 0.941 | 2 |
| one model, ChIP only | 0.574 | 0.585 | 0.582 | 0.436 | 0.116 | 0.746 | 2 |
| per-resolution, consensus input | 0.904 | 0.896 | 0.843 | 0.727 | 0.356 | 0.941 | 2 |
| per-resolution, ChIP only | 0.765 | 0.743 | 0.657 | 0.550 | 0.141 | 0.854 | 2 |
| all12refres_drop_chrho | 0.901 | 0.894 | 0.853 | 0.751 | 0.352 | 0.938 | 2 |
| consensus | 0.853 | 0.849 | 0.816 | 0.718 | 0.350 | 0.895 | 2 |
| best_other_map | 0.812 | 0.785 | 0.616 | 0.680 | 0.323 | 0.844 | 2 |
| hicgan_gm12878_on_k562_25kb | 0.143 | 0.147 | 0.218 | 0.213 | 0.050 | 0.655 | 2 |
| replicate | 0.936 | 0.923 | 0.836 | 0.831 | 0.349 | 0.963 | 2 |

Chromosomes: chr14, chr2.

## HiC-Reg block, gm12878, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.937 | 0.931 | 0.911 | 0.800 | 0.385 | 0.954 | 2 |
| one model, ChIP only | 0.449 | 0.476 | 0.573 | 0.345 | 0.070 | 0.612 | 2 |
| per-resolution, consensus input | 0.931 | 0.925 | 0.905 | 0.833 | 0.396 | 0.953 | 2 |
| per-resolution, ChIP only | 0.782 | 0.763 | 0.674 | 0.608 | 0.191 | 0.867 | 2 |
| all12refres_drop_chrho | 0.930 | 0.923 | 0.898 | 0.830 | 0.383 | 0.950 | 2 |
| consensus | 0.868 | 0.853 | 0.804 | 0.805 | 0.396 | 0.894 | 2 |
| cascfold | 0.933 | 0.929 | 0.918 | 0.837 | 0.365 | 0.958 | 2 |
| best_other_map | 0.797 | 0.791 | 0.776 | 0.801 | 0.335 | 0.830 | 2 |
| hicregmulticellcrosschrom_gm12878_25kb | 0.615 | 0.567 | 0.455 | 0.345 | 0.038 | 0.856 | 2 |
| hicregwindowcrosschrom_gm12878_25kb | 0.579 | 0.538 | 0.447 | 0.322 | 0.032 | 0.846 | 2 |
| hicregwindowcv_gm12878_25kb | 0.908 | 0.870 | 0.677 | 0.723 | 0.323 | 0.987 | 2 |
| replicate | 0.947 | 0.941 | 0.923 | 0.895 | 0.344 | 0.979 | 2 |

Chromosomes: chr14, chr17.

## HiC-Reg block, k562, 25 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.904 | 0.894 | 0.847 | 0.748 | 0.329 | 0.932 | 2 |
| one model, ChIP only | 0.577 | 0.561 | 0.492 | 0.422 | 0.071 | 0.697 | 2 |
| per-resolution, consensus input | 0.897 | 0.889 | 0.857 | 0.738 | 0.347 | 0.927 | 2 |
| per-resolution, ChIP only | 0.730 | 0.711 | 0.622 | 0.568 | 0.125 | 0.834 | 2 |
| all12refres_drop_chrho | 0.899 | 0.892 | 0.865 | 0.785 | 0.334 | 0.929 | 2 |
| consensus | 0.857 | 0.853 | 0.839 | 0.769 | 0.333 | 0.887 | 2 |
| best_other_map | 0.805 | 0.776 | 0.633 | 0.718 | 0.330 | 0.854 | 2 |
| hicregmulticellcrosschrom_k562_25kb | 0.740 | 0.700 | 0.500 | 0.519 | 0.030 | 0.927 | 2 |
| hicregwindowcrosschrom_k562_25kb | 0.736 | 0.696 | 0.484 | 0.523 | 0.038 | 0.927 | 2 |
| hicregwindowcv_k562_25kb | 0.888 | 0.864 | 0.670 | 0.688 | 0.178 | 0.973 | 2 |
| replicate | 0.936 | 0.918 | 0.840 | 0.862 | 0.325 | 0.974 | 2 |

Chromosomes: chr14, chr17.

## C.Origami block, imr90, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.887 | 0.890 | 0.949 | 0.550 | 0.335 | 0.967 | 2 |
| one model, ChIP only | 0.483 | 0.503 | 0.704 | 0.325 | 0.049 | 0.728 | 2 |
| per-resolution, consensus input | 0.882 | 0.884 | 0.936 | 0.547 | 0.302 | 0.964 | 2 |
| per-resolution, ChIP only | 0.756 | 0.760 | 0.834 | 0.462 | 0.177 | 0.893 | 2 |
| all12refres_drop_chrho | 0.878 | 0.880 | 0.928 | 0.555 | 0.319 | 0.961 | 2 |
| consensus | 0.819 | 0.815 | 0.803 | 0.581 | 0.339 | 0.902 | 2 |
| cascfold | 0.887 | 0.890 | 0.948 | 0.578 | 0.314 | 0.966 | 2 |
| best_other_map | 0.796 | 0.798 | 0.779 | 0.537 | 0.168 | 0.919 | 2 |
| corigami_imr90_8kb | 0.849 | 0.854 | 0.852 | 0.348 | 0.164 | 0.956 | 2 |
| replicate | 0.670 | 0.755 | 0.937 | 0.513 | 0.152 | 0.902 | 2 |

Chromosomes: chr2, chr3.

## Epiphany block, gm12878, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.901 | 0.901 | 0.918 | 0.577 | 0.478 | 0.953 | 3 |
| one model, ChIP only | 0.520 | 0.538 | 0.617 | 0.362 | 0.035 | 0.680 | 3 |
| per-resolution, consensus input | 0.894 | 0.891 | 0.889 | 0.577 | 0.430 | 0.953 | 3 |
| per-resolution, ChIP only | 0.737 | 0.734 | 0.726 | 0.472 | 0.222 | 0.844 | 3 |
| all12refres_drop_chrho | 0.887 | 0.881 | 0.864 | 0.566 | 0.424 | 0.947 | 3 |
| consensus | 0.793 | 0.790 | 0.784 | 0.560 | 0.416 | 0.876 | 3 |
| cascfold | 0.899 | 0.897 | 0.909 | 0.560 | 0.450 | 0.957 | 3 |
| best_other_map | 0.650 | 0.654 | 0.697 | 0.483 | 0.103 | 0.820 | 3 |
| epiphany_gm12878_10kb | 0.523 | 0.504 | 0.310 | 0.133 | 0.043 | 0.770 | 3 |
| replicate | 0.700 | 0.732 | 0.895 | 0.571 | 0.212 | 0.918 | 3 |

Chromosomes: chr11, chr17, chr3.

## Hi-cGAN block, gm12878, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.891 | 0.894 | 0.912 | 0.551 | 0.380 | 0.961 | 2 |
| one model, ChIP only | 0.438 | 0.467 | 0.618 | 0.359 | 0.027 | 0.708 | 2 |
| per-resolution, consensus input | 0.882 | 0.882 | 0.886 | 0.540 | 0.337 | 0.954 | 2 |
| per-resolution, ChIP only | 0.734 | 0.740 | 0.737 | 0.459 | 0.161 | 0.881 | 2 |
| all12refres_drop_chrho | 0.873 | 0.873 | 0.851 | 0.547 | 0.318 | 0.953 | 2 |
| consensus | 0.791 | 0.792 | 0.759 | 0.538 | 0.319 | 0.882 | 2 |
| cascfold | 0.882 | 0.883 | 0.888 | 0.535 | 0.367 | 0.959 | 2 |
| best_other_map | 0.575 | 0.584 | 0.652 | 0.434 | 0.020 | 0.816 | 2 |
| hicgan_gm12878_5kb | 0.346 | 0.353 | 0.397 | 0.124 | 0.068 | 0.638 | 2 |
| replicate | 0.664 | 0.713 | 0.898 | 0.555 | 0.151 | 0.908 | 2 |

Chromosomes: chr14, chr2.

## HiC-Reg block, gm12878, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.916 | 0.914 | 0.920 | 0.572 | 0.410 | 0.961 | 2 |
| one model, ChIP only | 0.509 | 0.529 | 0.629 | 0.356 | 0.030 | 0.696 | 2 |
| per-resolution, consensus input | 0.906 | 0.902 | 0.898 | 0.562 | 0.363 | 0.957 | 2 |
| per-resolution, ChIP only | 0.749 | 0.749 | 0.741 | 0.455 | 0.188 | 0.871 | 2 |
| all12refres_drop_chrho | 0.899 | 0.894 | 0.873 | 0.568 | 0.349 | 0.954 | 2 |
| consensus | 0.821 | 0.817 | 0.798 | 0.556 | 0.354 | 0.885 | 2 |
| cascfold | 0.908 | 0.906 | 0.911 | 0.541 | 0.395 | 0.961 | 2 |
| best_other_map | 0.640 | 0.652 | 0.744 | 0.481 | 0.063 | 0.818 | 2 |
| hicregmulticellcrosschrom_gm12878_5kb | 0.626 | 0.571 | 0.525 | 0.195 | 0.000 | 0.890 | 2 |
| hicregwindowcrosschrom_gm12878_5kb | 0.603 | 0.554 | 0.515 | 0.161 | 0.000 | 0.891 | 2 |
| hicregwindowcv_gm12878_5kb | 0.893 | 0.859 | 0.647 | 0.252 | 0.020 | 0.992 | 2 |
| replicate | 0.693 | 0.740 | 0.900 | 0.589 | 0.206 | 0.911 | 2 |

Chromosomes: chr14, chr17.

## HiC-Reg block, k562, 5 kb

| source | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r | chromosomes |
|---|---|---|---|---|---|---|---|
| one model, consensus input | 0.777 | 0.783 | 0.859 | 0.510 | 0.366 | 0.930 | 2 |
| one model, ChIP only | 0.527 | 0.542 | 0.697 | 0.364 | 0.071 | 0.692 | 2 |
| per-resolution, consensus input | 0.771 | 0.776 | 0.842 | 0.510 | 0.338 | 0.925 | 2 |
| per-resolution, ChIP only | 0.641 | 0.653 | 0.730 | 0.415 | 0.169 | 0.831 | 2 |
| all12refres_drop_chrho | 0.784 | 0.789 | 0.869 | 0.509 | 0.338 | 0.922 | 2 |
| consensus | 0.673 | 0.685 | 0.810 | 0.505 | 0.323 | 0.875 | 2 |
| best_other_map | 0.570 | 0.580 | 0.581 | 0.417 | 0.130 | 0.837 | 2 |
| hicregmulticellcrosschrom_k562_5kb | 0.783 | 0.768 | 0.640 | 0.271 | 0.003 | 0.962 | 2 |
| hicregwindowcrosschrom_k562_5kb | 0.774 | 0.755 | 0.599 | 0.226 | 0.003 | 0.959 | 2 |
| hicregwindowcv_k562_5kb | 0.845 | 0.840 | 0.728 | 0.282 | 0.005 | 0.978 | 2 |
| replicate | 0.706 | 0.707 | 0.774 | 0.546 | 0.183 | 0.965 | 2 |

Chromosomes: chr14, chr17.

## Difference, one model minus per-resolution model

Mean over the held-out chromosomes of the block; positive = the one model is higher.

### one model, consensus input minus per-resolution, consensus input

| block | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | +0.007 | +0.009 | +0.027 | -0.007 | +0.008 | +0.006 |
| Epiphany block, gm12878, 10 kb | +0.011 | +0.015 | +0.037 | +0.001 | +0.003 | +0.005 |
| Hi-cGAN block, gm12878, 10 kb | +0.010 | +0.012 | +0.035 | +0.009 | +0.014 | +0.009 |
| Hi-cGAN block, k562, 10 kb | +0.017 | +0.016 | +0.012 | +0.013 | +0.024 | +0.009 |
| C.Origami block, imr90, 25 kb | +0.003 | +0.004 | +0.009 | -0.020 | +0.002 | -0.000 |
| Epiphany block, gm12878, 25 kb | +0.007 | +0.007 | +0.007 | -0.002 | -0.001 | +0.001 |
| Hi-cGAN block, gm12878, 25 kb | +0.002 | +0.003 | +0.011 | -0.032 | -0.014 | +0.000 |
| Hi-cGAN block, k562, 25 kb | +0.004 | +0.003 | -0.005 | +0.007 | -0.005 | -0.000 |
| HiC-Reg block, gm12878, 25 kb | +0.005 | +0.005 | +0.006 | -0.033 | -0.011 | +0.001 |
| HiC-Reg block, k562, 25 kb | +0.007 | +0.005 | -0.009 | +0.010 | -0.018 | +0.004 |
| C.Origami block, imr90, 5 kb | +0.005 | +0.006 | +0.013 | +0.003 | +0.033 | +0.003 |
| Epiphany block, gm12878, 5 kb | +0.007 | +0.011 | +0.028 | -0.000 | +0.047 | -0.000 |
| Hi-cGAN block, gm12878, 5 kb | +0.009 | +0.012 | +0.026 | +0.011 | +0.043 | +0.007 |
| HiC-Reg block, gm12878, 5 kb | +0.010 | +0.012 | +0.023 | +0.011 | +0.046 | +0.004 |
| HiC-Reg block, k562, 5 kb | +0.006 | +0.006 | +0.017 | +0.000 | +0.028 | +0.005 |
| mean of the 4 figure blocks, 10 kb | +0.011 | +0.013 | +0.028 | +0.004 | +0.012 | +0.007 |
| mean of the 4 figure blocks, 25 kb | +0.004 | +0.005 | +0.006 | -0.012 | -0.004 | +0.000 |
| mean of the 3 figure blocks, 5 kb | +0.007 | +0.009 | +0.022 | +0.005 | +0.041 | +0.003 |

### one model, ChIP only minus per-resolution, ChIP only

| block | Pearson | SCC | insulation r | boundary recall | loop recall | compartment r |
|---|---|---|---|---|---|---|
| C.Origami block, imr90, 10 kb | -0.303 | -0.286 | -0.123 | -0.137 | -0.138 | -0.167 |
| Epiphany block, gm12878, 10 kb | -0.238 | -0.212 | -0.120 | -0.163 | -0.155 | -0.154 |
| Hi-cGAN block, gm12878, 10 kb | -0.352 | -0.309 | -0.125 | -0.192 | -0.105 | -0.146 |
| Hi-cGAN block, k562, 10 kb | -0.208 | -0.176 | -0.058 | -0.095 | -0.079 | -0.102 |
| C.Origami block, imr90, 25 kb | -0.302 | -0.282 | -0.130 | -0.203 | -0.149 | -0.158 |
| Epiphany block, gm12878, 25 kb | -0.278 | -0.246 | -0.123 | -0.271 | -0.160 | -0.208 |
| Hi-cGAN block, gm12878, 25 kb | -0.408 | -0.352 | -0.124 | -0.248 | -0.105 | -0.198 |
| Hi-cGAN block, k562, 25 kb | -0.191 | -0.158 | -0.075 | -0.114 | -0.025 | -0.108 |
| HiC-Reg block, gm12878, 25 kb | -0.333 | -0.287 | -0.102 | -0.264 | -0.121 | -0.256 |
| HiC-Reg block, k562, 25 kb | -0.153 | -0.150 | -0.130 | -0.146 | -0.054 | -0.137 |
| C.Origami block, imr90, 5 kb | -0.273 | -0.257 | -0.130 | -0.138 | -0.129 | -0.165 |
| Epiphany block, gm12878, 5 kb | -0.217 | -0.197 | -0.109 | -0.109 | -0.186 | -0.164 |
| Hi-cGAN block, gm12878, 5 kb | -0.296 | -0.273 | -0.119 | -0.100 | -0.134 | -0.172 |
| HiC-Reg block, gm12878, 5 kb | -0.240 | -0.220 | -0.112 | -0.099 | -0.158 | -0.175 |
| HiC-Reg block, k562, 5 kb | -0.114 | -0.111 | -0.033 | -0.051 | -0.098 | -0.139 |
| mean of the 4 figure blocks, 10 kb | -0.275 | -0.246 | -0.106 | -0.147 | -0.119 | -0.143 |
| mean of the 4 figure blocks, 25 kb | -0.295 | -0.259 | -0.113 | -0.209 | -0.110 | -0.168 |
| mean of the 3 figure blocks, 5 kb | -0.262 | -0.242 | -0.120 | -0.116 | -0.150 | -0.167 |

## Check of the older models

Largest absolute difference between the stored rows of the per-resolution models and their rows in the blocks scored together with the one model (`chrho_benchmark_oldmodel_check.csv`).

| table | measure | rows | max abs difference |
|---|---|---|---|
| chrho | oe_pearson | 51 | 0 |
| chrho | oe_scc | 51 | 0 |
| chrho | insulation_r | 51 | 0 |
| chrho | bnd_recall | 51 | 0 |
| chrho | loop_recall_rank | 51 | 0 |
| chrho | comp_pref_r | 51 | 0 |
| chrho_25kb | oe_pearson | 39 | 0 |
| chrho_25kb | oe_scc | 39 | 0 |
| chrho_25kb | insulation_r | 39 | 0 |
| chrho_25kb | bnd_recall | 39 | 0 |
| chrho_25kb | loop_recall_rank | 39 | 0 |
| chrho_25kb | comp_pref_r | 39 | 0 |
| chrho_5kb | oe_pearson | 33 | 0 |
| chrho_5kb | oe_scc | 33 | 0 |
| chrho_5kb | insulation_r | 33 | 0 |
| chrho_5kb | bnd_recall | 33 | 0 |
| chrho_5kb | loop_recall_rank | 33 | 0 |
| chrho_5kb | comp_pref_r | 33 | 0 |
