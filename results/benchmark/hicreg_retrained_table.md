
GM12878 25 kb, chr14+chr17 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.931 | 0.950 | 0.926 | 0.905 | 0.833 | 0.396 | 0.792 | 0.953 | 0.935 |
| ours: all12honest_drop_chrho (ChIP only) | 0.783 | 0.798 | 0.763 | 0.674 | 0.608 | 0.191 | 0.319 | 0.867 | 0.791 |
| ours: all12refres_drop_chrho | 0.930 | 0.950 | 0.923 | 0.899 | 0.830 | 0.383 | 0.834 | 0.950 | 0.930 |
| consensus of the other cells | 0.867 | 0.901 | 0.853 | 0.804 | 0.805 | 0.396 | 0.818 | 0.894 | 0.857 |
| best other measured map | 0.797 | 0.855 | 0.791 | 0.776 | 0.801 | 0.335 | 0.809 | 0.830 | 0.823 |
| HiC-Reg retrained (our split), WINDOW | 0.430 | 0.524 | 0.482 | 0.502 | 0.331 | 0.050 | -0.110 | 0.756 | 0.599 |
| HiC-Reg retrained (our split), MULTICELL | 0.383 | 0.569 | 0.473 | 0.498 | 0.342 | 0.039 | -0.124 | 0.741 | 0.601 |
| HiC-Reg retrained, cross-cell (model of the other cell) | 0.327 | 0.430 | 0.375 | 0.467 | 0.243 | 0.054 | -0.117 | 0.637 | 0.523 |
| HiC-Reg retrained, WINDOW, 4M-pair training (size control) | 0.432 | 0.537 | 0.469 | 0.481 | 0.314 | 0.038 | -0.110 | 0.763 | 0.591 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.579 | 0.619 | 0.538 | 0.447 | 0.322 | 0.032 | -0.002 | 0.846 | 0.668 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.615 | 0.635 | 0.567 | 0.455 | 0.346 | 0.038 | -0.032 | 0.856 | 0.692 |
| HiC-Reg CV protocol on our pipeline (within-chromosome) | 0.879 | 0.886 | 0.855 | 0.615 | 0.554 | 0.216 | 0.357 | 0.979 | 0.885 |
| HiC-Reg CV (released, within-chromosome) | 0.908 | 0.964 | 0.870 | 0.677 | 0.723 | 0.323 | 0.765 | 0.987 | 0.915 |

GM12878 10 kb, chr14+chr17 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.925 | 0.937 | 0.919 | 0.890 | 0.665 | 0.408 | 0.728 | 0.952 | 0.927 |
| ours: all12honest_drop_chrho (ChIP only) | 0.788 | 0.810 | 0.777 | 0.735 | 0.495 | 0.182 | 0.301 | 0.854 | 0.797 |
| ours: all12refres_drop_chrho | 0.931 | 0.936 | 0.925 | 0.897 | 0.692 | 0.418 | 0.942 | 0.958 | 0.939 |
| consensus of the other cells | 0.853 | 0.882 | 0.842 | 0.793 | 0.674 | 0.397 | 0.902 | 0.880 | 0.861 |
| best other measured map | 0.756 | 0.805 | 0.756 | 0.756 | 0.628 | 0.181 | 0.897 | 0.819 | 0.809 |
| HiC-Reg retrained (our split), WINDOW | 0.446 | 0.467 | 0.473 | 0.523 | 0.215 | 0.014 | -0.009 | 0.776 | 0.549 |
| HiC-Reg retrained (our split), MULTICELL | 0.428 | 0.540 | 0.470 | 0.515 | 0.224 | 0.011 | -0.038 | 0.761 | 0.555 |
| HiC-Reg retrained, cross-cell (model of the other cell) | 0.331 | 0.378 | 0.359 | 0.452 | 0.116 | 0.025 | 0.078 | 0.645 | 0.452 |
| HiC-Reg retrained, WINDOW, 4M-pair training (size control) | 0.446 | 0.477 | 0.461 | 0.497 | 0.150 | 0.004 | -0.024 | 0.785 | 0.540 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.590 | 0.561 | 0.542 | 0.491 | 0.208 | 0.004 | 0.016 | 0.857 | 0.601 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.623 | 0.574 | 0.569 | 0.501 | 0.229 | 0.000 | 0.006 | 0.858 | 0.622 |
| HiC-Reg CV protocol on our pipeline (within-chromosome) | 0.879 | 0.846 | 0.843 | 0.556 | 0.254 | 0.057 | 0.140 | 0.981 | 0.883 |
| HiC-Reg CV (released, within-chromosome) | 0.920 | 0.939 | 0.881 | 0.626 | 0.393 | 0.148 | 0.494 | 0.991 | 0.925 |

GM12878 5 kb, chr14+chr17 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.906 | 0.911 | 0.902 | 0.898 | 0.562 | 0.363 | 0.675 | 0.957 | 0.923 |
| ours: all12honest_drop_chrho (ChIP only) | 0.749 | 0.758 | 0.749 | 0.741 | 0.455 | 0.188 | 0.385 | 0.871 | 0.770 |
| ours: all12refres_drop_chrho | 0.899 | 0.900 | 0.894 | 0.873 | 0.568 | 0.349 | 0.772 | 0.954 | 0.919 |
| consensus of the other cells | 0.821 | 0.842 | 0.817 | 0.798 | 0.556 | 0.354 | 0.946 | 0.885 | 0.853 |
| best other measured map | 0.640 | 0.699 | 0.652 | 0.743 | 0.481 | 0.063 | 0.957 | 0.818 | 0.764 |
| HiC-Reg retrained (our split), WINDOW | 0.454 | 0.434 | 0.475 | 0.537 | 0.146 | 0.005 | 0.026 | 0.773 | 0.526 |
| HiC-Reg retrained (our split), MULTICELL | 0.448 | 0.503 | 0.474 | 0.534 | 0.163 | 0.034 | 0.005 | 0.761 | 0.534 |
| HiC-Reg retrained, cross-cell (model of the other cell) | 0.338 | 0.309 | 0.359 | 0.459 | 0.086 | 0.029 | 0.133 | 0.650 | 0.420 |
| HiC-Reg retrained, WINDOW, 4M-pair training (size control) | 0.454 | 0.421 | 0.464 | 0.509 | 0.117 | 0.002 | 0.009 | 0.784 | 0.514 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.603 | 0.520 | 0.554 | 0.516 | 0.161 | 0.000 | 0.045 | 0.891 | 0.593 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.626 | 0.519 | 0.571 | 0.525 | 0.195 | 0.000 | 0.036 | 0.890 | 0.611 |
| HiC-Reg CV protocol on our pipeline (within-chromosome) | 0.872 | 0.814 | 0.836 | 0.565 | 0.166 | 0.030 | 0.101 | 0.980 | 0.874 |
| HiC-Reg CV (released, within-chromosome) | 0.893 | 0.898 | 0.859 | 0.647 | 0.252 | 0.020 | 0.398 | 0.992 | 0.922 |

K562 25 kb, chr14+chr17 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.897 | 0.928 | 0.889 | 0.857 | 0.738 | 0.347 | 0.843 | 0.927 | 0.904 |
| ours: all12honest_drop_chrho (ChIP only) | 0.730 | 0.777 | 0.711 | 0.622 | 0.568 | 0.125 | 0.187 | 0.834 | 0.748 |
| ours: all12refres_drop_chrho | 0.899 | 0.933 | 0.892 | 0.865 | 0.785 | 0.334 | 0.862 | 0.929 | 0.901 |
| consensus of the other cells | 0.857 | 0.905 | 0.853 | 0.839 | 0.769 | 0.333 | 0.876 | 0.887 | 0.868 |
| best other measured map | 0.805 | 0.888 | 0.776 | 0.633 | 0.718 | 0.330 | 0.766 | 0.854 | 0.766 |
| HiC-Reg retrained (our split), WINDOW | 0.286 | 0.436 | 0.347 | 0.497 | 0.255 | 0.049 | -0.097 | 0.667 | 0.465 |
| HiC-Reg retrained (our split), MULTICELL | 0.297 | 0.486 | 0.392 | 0.528 | 0.291 | 0.029 | -0.250 | 0.680 | 0.518 |
| HiC-Reg retrained, cross-cell (model of the other cell) | 0.168 | 0.305 | 0.237 | 0.414 | 0.304 | 0.019 | -0.051 | 0.592 | 0.377 |
| HiC-Reg retrained, WINDOW, 4M-pair training (size control) | 0.257 | 0.441 | 0.311 | 0.471 | 0.210 | 0.023 | -0.077 | 0.661 | 0.440 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.736 | 0.820 | 0.696 | 0.484 | 0.523 | 0.038 | 0.292 | 0.927 | 0.797 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.740 | 0.819 | 0.700 | 0.500 | 0.519 | 0.030 | 0.260 | 0.927 | 0.800 |
| HiC-Reg CV protocol on our pipeline (within-chromosome) | 0.801 | 0.844 | 0.799 | 0.603 | 0.493 | 0.133 | 0.232 | 0.960 | 0.841 |
| HiC-Reg CV (released, within-chromosome) | 0.888 | 0.950 | 0.864 | 0.671 | 0.688 | 0.178 | 0.698 | 0.973 | 0.923 |

K562 10 kb, chr14+chr17 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.858 | 0.900 | 0.857 | 0.848 | 0.627 | 0.366 | 0.568 | 0.924 | 0.887 |
| ours: all12honest_drop_chrho (ChIP only) | 0.715 | 0.724 | 0.714 | 0.721 | 0.484 | 0.139 | 0.220 | 0.848 | 0.745 |
| ours: all12refres_drop_chrho | 0.863 | 0.901 | 0.862 | 0.847 | 0.655 | 0.393 | 0.848 | 0.928 | 0.894 |
| consensus of the other cells | 0.811 | 0.866 | 0.814 | 0.836 | 0.660 | 0.386 | 0.811 | 0.879 | 0.847 |
| best other measured map | 0.746 | 0.833 | 0.735 | 0.628 | 0.546 | 0.252 | 0.595 | 0.841 | 0.748 |
| HiC-Reg retrained (our split), WINDOW | 0.336 | 0.417 | 0.378 | 0.554 | 0.117 | 0.028 | 0.071 | 0.664 | 0.452 |
| HiC-Reg retrained (our split), MULTICELL | 0.361 | 0.420 | 0.426 | 0.584 | 0.228 | 0.016 | -0.027 | 0.685 | 0.500 |
| HiC-Reg retrained, cross-cell (model of the other cell) | 0.169 | 0.240 | 0.219 | 0.413 | 0.213 | 0.004 | 0.013 | 0.579 | 0.280 |
| HiC-Reg retrained, WINDOW, 4M-pair training (size control) | 0.311 | 0.386 | 0.350 | 0.524 | 0.091 | 0.012 | 0.064 | 0.656 | 0.427 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.747 | 0.760 | 0.708 | 0.496 | 0.267 | 0.002 | 0.210 | 0.954 | 0.782 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.756 | 0.764 | 0.719 | 0.541 | 0.326 | 0.000 | 0.186 | 0.954 | 0.793 |
| HiC-Reg CV protocol on our pipeline (within-chromosome) | 0.817 | 0.794 | 0.808 | 0.618 | 0.259 | 0.064 | 0.126 | 0.967 | 0.857 |
| HiC-Reg CV (released, within-chromosome) | 0.878 | 0.900 | 0.858 | 0.650 | 0.413 | 0.009 | 0.456 | 0.985 | 0.925 |

K562 5 kb, chr14+chr17 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.771 | 0.817 | 0.776 | 0.842 | 0.510 | 0.338 | 0.718 | 0.925 | 0.814 |
| ours: all12honest_drop_chrho (ChIP only) | 0.641 | 0.641 | 0.653 | 0.730 | 0.415 | 0.169 | 0.283 | 0.831 | 0.664 |
| ours: all12refres_drop_chrho | 0.784 | 0.812 | 0.789 | 0.869 | 0.509 | 0.338 | 0.689 | 0.922 | 0.829 |
| consensus of the other cells | 0.673 | 0.777 | 0.685 | 0.810 | 0.505 | 0.323 | 0.818 | 0.875 | 0.727 |
| best other measured map | 0.570 | 0.717 | 0.580 | 0.581 | 0.417 | 0.130 | 0.562 | 0.837 | 0.602 |
| HiC-Reg retrained (our split), WINDOW | 0.414 | 0.324 | 0.440 | 0.625 | 0.089 | 0.030 | 0.108 | 0.636 | 0.428 |
| HiC-Reg retrained (our split), MULTICELL | 0.439 | 0.333 | 0.479 | 0.654 | 0.150 | 0.026 | 0.038 | 0.677 | 0.453 |
| HiC-Reg retrained, cross-cell (model of the other cell) | 0.193 | 0.158 | 0.223 | 0.451 | 0.161 | 0.002 | 0.025 | 0.569 | 0.237 |
| HiC-Reg retrained, WINDOW, 4M-pair training (size control) | 0.403 | 0.290 | 0.428 | 0.598 | 0.064 | 0.018 | 0.092 | 0.635 | 0.401 |
| HiC-Reg cross-chromosome (released), WINDOW | 0.774 | 0.703 | 0.755 | 0.599 | 0.226 | 0.003 | 0.260 | 0.959 | 0.790 |
| HiC-Reg cross-chromosome (released), MULTICELL | 0.783 | 0.716 | 0.768 | 0.640 | 0.271 | 0.003 | 0.242 | 0.962 | 0.810 |
| HiC-Reg CV protocol on our pipeline (within-chromosome) | 0.781 | 0.697 | 0.783 | 0.692 | 0.178 | 0.039 | 0.114 | 0.959 | 0.758 |
| HiC-Reg CV (released, within-chromosome) | 0.845 | 0.814 | 0.840 | 0.728 | 0.282 | 0.005 | 0.441 | 0.978 | 0.899 |

GM12878 25 kb, chr2+chr3+chr11 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.924 | 0.945 | 0.917 | 0.898 | 0.809 | 0.472 | 0.807 | 0.952 | 0.929 |
| ours: all12honest_drop_chrho (ChIP only) | 0.756 | 0.809 | 0.739 | 0.684 | 0.605 | 0.253 | 0.375 | 0.853 | 0.761 |
| ours: all12refres_drop_chrho | 0.926 | 0.946 | 0.919 | 0.898 | 0.817 | 0.455 | 0.815 | 0.954 | 0.929 |
| consensus of the other cells | 0.837 | 0.885 | 0.824 | 0.783 | 0.788 | 0.459 | 0.761 | 0.875 | 0.826 |
| best other measured map | 0.772 | 0.841 | 0.759 | 0.705 | 0.740 | 0.386 | 0.765 | 0.814 | 0.767 |
| HiC-Reg retrained (our split), WINDOW | 0.319 | 0.550 | 0.384 | 0.380 | 0.355 | 0.040 | -0.044 | 0.700 | 0.497 |

GM12878 10 kb, chr2+chr3+chr11 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.916 | 0.934 | 0.907 | 0.862 | 0.637 | 0.482 | 0.707 | 0.953 | 0.920 |
| ours: all12honest_drop_chrho (ChIP only) | 0.765 | 0.794 | 0.753 | 0.709 | 0.498 | 0.214 | 0.337 | 0.856 | 0.769 |
| ours: all12refres_drop_chrho | 0.921 | 0.935 | 0.915 | 0.884 | 0.647 | 0.470 | 0.956 | 0.957 | 0.928 |
| consensus of the other cells | 0.822 | 0.864 | 0.810 | 0.744 | 0.625 | 0.451 | 0.891 | 0.872 | 0.823 |
| best other measured map | 0.741 | 0.781 | 0.730 | 0.655 | 0.549 | 0.224 | 0.865 | 0.815 | 0.764 |
| HiC-Reg retrained (our split), WINDOW | 0.366 | 0.518 | 0.393 | 0.386 | 0.206 | 0.012 | 0.007 | 0.716 | 0.467 |

GM12878 5 kb, chr2+chr3+chr11 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.892 | 0.907 | 0.888 | 0.882 | 0.550 | 0.412 | 0.649 | 0.954 | 0.916 |
| ours: all12honest_drop_chrho (ChIP only) | 0.743 | 0.755 | 0.737 | 0.720 | 0.457 | 0.207 | 0.427 | 0.860 | 0.764 |
| ours: all12refres_drop_chrho | 0.884 | 0.896 | 0.876 | 0.844 | 0.535 | 0.409 | 0.746 | 0.949 | 0.904 |
| consensus of the other cells | 0.787 | 0.808 | 0.780 | 0.750 | 0.535 | 0.404 | 0.923 | 0.875 | 0.816 |
| best other measured map | 0.627 | 0.676 | 0.626 | 0.635 | 0.448 | 0.078 | 0.922 | 0.816 | 0.726 |
| HiC-Reg retrained (our split), WINDOW | 0.396 | 0.478 | 0.410 | 0.419 | 0.146 | 0.021 | 0.038 | 0.713 | 0.462 |

K562 25 kb, chr2+chr3+chr11 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.888 | 0.932 | 0.880 | 0.846 | 0.751 | 0.429 | 0.876 | 0.924 | 0.876 |
| ours: all12honest_drop_chrho (ChIP only) | 0.708 | 0.810 | 0.696 | 0.650 | 0.575 | 0.201 | 0.310 | 0.799 | 0.688 |
| ours: all12refres_drop_chrho | 0.889 | 0.929 | 0.882 | 0.853 | 0.760 | 0.423 | 0.842 | 0.920 | 0.883 |
| consensus of the other cells | 0.831 | 0.892 | 0.831 | 0.823 | 0.743 | 0.418 | 0.831 | 0.850 | 0.839 |
| best other measured map | 0.784 | 0.861 | 0.763 | 0.644 | 0.692 | 0.394 | 0.725 | 0.806 | 0.750 |
| HiC-Reg retrained (our split), WINDOW | 0.261 | 0.528 | 0.319 | 0.367 | 0.249 | 0.050 | 0.051 | 0.685 | 0.393 |

K562 10 kb, chr2+chr3+chr11 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.857 | 0.884 | 0.854 | 0.837 | 0.595 | 0.436 | 0.550 | 0.922 | 0.876 |
| ours: all12honest_drop_chrho (ChIP only) | 0.720 | 0.770 | 0.717 | 0.716 | 0.465 | 0.204 | 0.258 | 0.853 | 0.724 |
| ours: all12refres_drop_chrho | 0.862 | 0.885 | 0.860 | 0.847 | 0.609 | 0.458 | 0.855 | 0.929 | 0.885 |
| consensus of the other cells | 0.797 | 0.840 | 0.800 | 0.813 | 0.597 | 0.429 | 0.825 | 0.849 | 0.829 |
| best other measured map | 0.737 | 0.791 | 0.727 | 0.621 | 0.510 | 0.339 | 0.625 | 0.803 | 0.739 |
| HiC-Reg retrained (our split), WINDOW | 0.335 | 0.484 | 0.370 | 0.447 | 0.122 | 0.020 | 0.114 | 0.700 | 0.432 |

K562 5 kb, chr2+chr3+chr11 (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.766 | 0.809 | 0.769 | 0.825 | 0.482 | 0.359 | 0.702 | 0.918 | 0.807 |
| ours: all12honest_drop_chrho (ChIP only) | 0.636 | 0.647 | 0.642 | 0.722 | 0.409 | 0.174 | 0.340 | 0.838 | 0.647 |
| ours: all12refres_drop_chrho | 0.771 | 0.803 | 0.775 | 0.852 | 0.475 | 0.368 | 0.671 | 0.915 | 0.812 |
| consensus of the other cells | 0.660 | 0.746 | 0.669 | 0.784 | 0.492 | 0.367 | 0.826 | 0.845 | 0.701 |
| best other measured map | 0.552 | 0.669 | 0.564 | 0.586 | 0.418 | 0.161 | 0.618 | 0.793 | 0.594 |
| HiC-Reg retrained (our split), WINDOW | 0.440 | 0.366 | 0.464 | 0.583 | 0.099 | 0.027 | 0.131 | 0.694 | 0.433 |

GM12878 25 kb, all five (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.927 | 0.947 | 0.921 | 0.901 | 0.819 | 0.442 | 0.801 | 0.953 | 0.931 |
| ours: all12honest_drop_chrho (ChIP only) | 0.766 | 0.805 | 0.749 | 0.680 | 0.606 | 0.228 | 0.353 | 0.859 | 0.773 |
| ours: all12refres_drop_chrho | 0.927 | 0.948 | 0.920 | 0.898 | 0.822 | 0.426 | 0.823 | 0.952 | 0.929 |
| consensus of the other cells | 0.849 | 0.891 | 0.835 | 0.791 | 0.795 | 0.434 | 0.784 | 0.882 | 0.838 |
| best other measured map | 0.782 | 0.847 | 0.772 | 0.734 | 0.764 | 0.365 | 0.783 | 0.820 | 0.789 |
| HiC-Reg retrained (our split), WINDOW | 0.364 | 0.539 | 0.423 | 0.429 | 0.345 | 0.044 | -0.071 | 0.722 | 0.538 |

GM12878 10 kb, all five (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.920 | 0.935 | 0.912 | 0.873 | 0.648 | 0.452 | 0.715 | 0.953 | 0.923 |
| ours: all12honest_drop_chrho (ChIP only) | 0.774 | 0.800 | 0.762 | 0.719 | 0.497 | 0.201 | 0.323 | 0.855 | 0.780 |
| ours: all12refres_drop_chrho | 0.925 | 0.935 | 0.919 | 0.889 | 0.665 | 0.449 | 0.950 | 0.957 | 0.933 |
| consensus of the other cells | 0.834 | 0.871 | 0.823 | 0.763 | 0.645 | 0.429 | 0.896 | 0.876 | 0.838 |
| best other measured map | 0.747 | 0.791 | 0.741 | 0.695 | 0.581 | 0.207 | 0.878 | 0.817 | 0.782 |
| HiC-Reg retrained (our split), WINDOW | 0.398 | 0.498 | 0.425 | 0.441 | 0.209 | 0.013 | 0.000 | 0.740 | 0.500 |

GM12878 5 kb, all five (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.898 | 0.909 | 0.893 | 0.888 | 0.555 | 0.392 | 0.659 | 0.955 | 0.919 |
| ours: all12honest_drop_chrho (ChIP only) | 0.745 | 0.756 | 0.742 | 0.728 | 0.456 | 0.199 | 0.410 | 0.865 | 0.766 |
| ours: all12refres_drop_chrho | 0.890 | 0.897 | 0.883 | 0.855 | 0.548 | 0.385 | 0.756 | 0.951 | 0.910 |
| consensus of the other cells | 0.801 | 0.821 | 0.795 | 0.769 | 0.543 | 0.384 | 0.932 | 0.879 | 0.831 |
| best other measured map | 0.632 | 0.685 | 0.636 | 0.678 | 0.462 | 0.072 | 0.936 | 0.817 | 0.741 |
| HiC-Reg retrained (our split), WINDOW | 0.419 | 0.460 | 0.436 | 0.466 | 0.146 | 0.015 | 0.033 | 0.737 | 0.487 |

K562 25 kb, all five (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.892 | 0.931 | 0.884 | 0.850 | 0.746 | 0.396 | 0.863 | 0.926 | 0.887 |
| ours: all12honest_drop_chrho (ChIP only) | 0.717 | 0.797 | 0.702 | 0.639 | 0.572 | 0.171 | 0.261 | 0.813 | 0.712 |
| ours: all12refres_drop_chrho | 0.893 | 0.930 | 0.886 | 0.858 | 0.770 | 0.387 | 0.850 | 0.923 | 0.890 |
| consensus of the other cells | 0.842 | 0.897 | 0.840 | 0.830 | 0.753 | 0.384 | 0.849 | 0.865 | 0.851 |
| best other measured map | 0.793 | 0.872 | 0.768 | 0.640 | 0.702 | 0.369 | 0.741 | 0.826 | 0.756 |
| HiC-Reg retrained (our split), WINDOW | 0.271 | 0.491 | 0.330 | 0.419 | 0.252 | 0.050 | -0.008 | 0.678 | 0.422 |

K562 10 kb, all five (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.857 | 0.891 | 0.855 | 0.841 | 0.608 | 0.408 | 0.557 | 0.923 | 0.880 |
| ours: all12honest_drop_chrho (ChIP only) | 0.718 | 0.752 | 0.716 | 0.718 | 0.472 | 0.178 | 0.243 | 0.851 | 0.733 |
| ours: all12refres_drop_chrho | 0.862 | 0.892 | 0.861 | 0.847 | 0.628 | 0.432 | 0.852 | 0.929 | 0.889 |
| consensus of the other cells | 0.803 | 0.851 | 0.805 | 0.822 | 0.622 | 0.412 | 0.819 | 0.861 | 0.836 |
| best other measured map | 0.741 | 0.808 | 0.730 | 0.624 | 0.524 | 0.304 | 0.613 | 0.818 | 0.742 |
| HiC-Reg retrained (our split), WINDOW | 0.336 | 0.457 | 0.374 | 0.490 | 0.120 | 0.023 | 0.097 | 0.685 | 0.440 |

K562 5 kb, all five (mean over the chromosomes; identical pixels within a chromosome block)

| source | Pearson | tile median | SCC | insulation r | boundary recall | loop recall | APA ratio | compartment r | HiCRep |
|---|---|---|---|---|---|---|---|---|---|
| ours: all12refin_drop_chrho | 0.768 | 0.813 | 0.772 | 0.832 | 0.493 | 0.350 | 0.708 | 0.921 | 0.810 |
| ours: all12honest_drop_chrho (ChIP only) | 0.638 | 0.645 | 0.646 | 0.725 | 0.411 | 0.172 | 0.317 | 0.835 | 0.653 |
| ours: all12refres_drop_chrho | 0.776 | 0.807 | 0.781 | 0.859 | 0.489 | 0.356 | 0.678 | 0.918 | 0.818 |
| consensus of the other cells | 0.665 | 0.758 | 0.675 | 0.794 | 0.497 | 0.349 | 0.823 | 0.857 | 0.712 |
| best other measured map | 0.559 | 0.688 | 0.570 | 0.584 | 0.418 | 0.149 | 0.596 | 0.810 | 0.597 |
| HiC-Reg retrained (our split), WINDOW | 0.430 | 0.349 | 0.454 | 0.599 | 0.095 | 0.028 | 0.122 | 0.671 | 0.431 |
