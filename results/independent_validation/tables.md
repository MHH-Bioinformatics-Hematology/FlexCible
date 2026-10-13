
### Median per-window Pearson, 10 kb, Akita test windows

| predictor | PC-3 | Panc1 | NK cell | Caco-2 |
|---|---|---|---|---|
| consensus12 alone | 0.883 | 0.613 | 0.658 | 0.870 |
| best other cell (measured) | 0.830 | 0.577 | 0.694 | 0.835 |
| Akita (sequence only) | 0.673 | 0.434 | 0.420 | 0.688 |
| refin, 7 marks | 0.895 | 0.615 | 0.766 | 0.894 |
| refin:ctcf_k27ac | 0.893 | 0.598 | 0.781 | 0.888 |
| refin:ctcf | 0.880 | 0.610 | 0.677 | 0.871 |
| refin:none | 0.879 | 0.620 | 0.644 | 0.871 |
| refres, 7 marks | 0.896 | 0.617 | 0.782 | 0.897 |
| refres:ctcf_k27ac | 0.892 | 0.596 | 0.793 | 0.895 |
| refres:ctcf | 0.880 | 0.620 | 0.670 | 0.873 |
| refres:none | 0.881 | 0.623 | 0.642 | 0.875 |
| honest (no consensus), 7 marks | 0.772 | 0.526 | 0.676 | 0.737 |
| honest:ctcf_k27ac | 0.726 | 0.487 | 0.652 | 0.702 |
| honest:ctcf | 0.691 | 0.474 | 0.454 | 0.659 |
| honest:none | 0.519 | 0.450 | 0.318 | 0.514 |
| differential channels, 7 marks | 0.902 | 0.623 | 0.802 | 0.898 |
| diff:ctcf_k27ac | 0.897 | 0.608 | 0.800 | 0.893 |
| diff:ctcf | 0.890 | 0.615 | 0.664 | 0.879 |
| diff:none | 0.885 | 0.621 | 0.640 | 0.879 |
| windows | 413 | 413 | 413 | 413 |
| split-half ceiling (Spearman-Brown) | 0.952 | 0.738 | 0.935 | 0.949 |

### Structure measures, 10 kb, Akita test windows (median over windows)

| predictor | sample | pearson | scc | insulation_corr | boundary_f1 | loop_enrich_corr | loop_apa_ratio | compartment_corr | eigenvector_abs_corr |
|---|---|---|---|---|---|---|---|---|---|
| consensus12 alone | PC-3 | 0.883 | 0.885 | 0.971 | 1.000 | 0.725 | 0.520 | 0.922 | 0.963 |
| consensus12 alone | Panc1 | 0.613 | 0.517 | 0.936 | 0.000 | 0.204 | 0.114 | 0.762 | 0.899 |
| consensus12 alone | NK cell | 0.658 | 0.633 | 0.773 | 0.000 | 0.469 | 0.581 | 0.737 | 0.823 |
| consensus12 alone | Caco-2 | 0.870 | 0.876 | 0.960 | 1.000 | 0.716 | 0.650 | 0.906 | 0.949 |
| best other cell (measured) | PC-3 | 0.830 | 0.834 | 0.939 | 0.000 | 0.657 | 0.744 | 0.880 | 0.939 |
| best other cell (measured) | Panc1 | 0.577 | 0.485 | 0.911 | 0.000 | 0.167 | 0.157 | 0.760 | 0.888 |
| best other cell (measured) | NK cell | 0.694 | 0.674 | 0.866 | 0.000 | 0.361 | 0.549 | 0.746 | 0.836 |
| best other cell (measured) | Caco-2 | 0.835 | 0.844 | 0.956 | 0.667 | 0.610 | 0.904 | 0.891 | 0.943 |
| Akita (sequence only) | PC-3 | 0.673 | 0.757 | 0.883 | 0.000 | 0.574 | 0.197 | 0.739 | 0.772 |
| Akita (sequence only) | Panc1 | 0.434 | 0.461 | 0.863 | 0.000 | 0.181 | 0.048 | 0.591 | 0.659 |
| Akita (sequence only) | NK cell | 0.420 | 0.497 | 0.634 | 0.000 | 0.344 | 0.187 | 0.625 | 0.684 |
| Akita (sequence only) | Caco-2 | 0.688 | 0.766 | 0.901 | 0.000 | 0.593 | 0.245 | 0.744 | 0.803 |
| honest (no consensus), 7 marks | PC-3 | 0.772 | 0.758 | 0.894 | 0.000 | 0.579 | 0.256 | 0.843 | 0.916 |
| honest (no consensus), 7 marks | Panc1 | 0.526 | 0.400 | 0.801 | 0.000 | 0.155 | 0.045 | 0.692 | 0.843 |
| honest (no consensus), 7 marks | NK cell | 0.676 | 0.645 | 0.821 | 0.000 | 0.376 | 0.172 | 0.745 | 0.852 |
| honest (no consensus), 7 marks | Caco-2 | 0.737 | 0.725 | 0.879 | 0.000 | 0.528 | 0.252 | 0.820 | 0.894 |
| refin, 7 marks | PC-3 | 0.895 | 0.895 | 0.974 | 1.000 | 0.747 | 0.589 | 0.933 | 0.970 |
| refin, 7 marks | Panc1 | 0.615 | 0.519 | 0.929 | 0.000 | 0.206 | 0.120 | 0.769 | 0.905 |
| refin, 7 marks | NK cell | 0.766 | 0.750 | 0.890 | 0.000 | 0.546 | 0.591 | 0.819 | 0.898 |
| refin, 7 marks | Caco-2 | 0.894 | 0.890 | 0.974 | 1.000 | 0.737 | 0.699 | 0.927 | 0.966 |
| refres, 7 marks | PC-3 | 0.896 | 0.898 | 0.975 | 1.000 | 0.743 | 0.601 | 0.930 | 0.969 |
| refres, 7 marks | Panc1 | 0.617 | 0.516 | 0.932 | 0.000 | 0.214 | 0.134 | 0.773 | 0.908 |
| refres, 7 marks | NK cell | 0.782 | 0.791 | 0.915 | 0.000 | 0.547 | 0.641 | 0.834 | 0.907 |
| refres, 7 marks | Caco-2 | 0.897 | 0.900 | 0.976 | 1.000 | 0.742 | 0.746 | 0.931 | 0.966 |
| differential channels, 7 marks | PC-3 | 0.902 | 0.902 | 0.976 | 1.000 | 0.747 | 0.625 | 0.935 | 0.972 |
| differential channels, 7 marks | Panc1 | 0.623 | 0.520 | 0.930 | 0.000 | 0.212 | 0.117 | 0.778 | 0.905 |
| differential channels, 7 marks | NK cell | 0.802 | 0.794 | 0.921 | 0.000 | 0.562 | 0.598 | 0.834 | 0.909 |
| differential channels, 7 marks | Caco-2 | 0.898 | 0.898 | 0.976 | 1.000 | 0.744 | 0.685 | 0.935 | 0.970 |

### Median per-window Pearson, 25 kb, Akita test windows

| predictor | PC-3 | Panc1 | NK cell | Caco-2 |
|---|---|---|---|---|
| consensus12 alone | 0.932 | 0.839 | 0.707 | 0.915 |
| best other cell (measured) | 0.897 | 0.811 | 0.790 | 0.899 |
| Akita (sequence only) | 0.663 | 0.549 | 0.404 | 0.688 |
| refin, 7 marks | 0.942 | 0.841 | 0.823 | 0.934 |
| refin:ctcf_k27ac | 0.932 | 0.836 | 0.829 | 0.928 |
| refin:ctcf | 0.931 | 0.840 | 0.705 | 0.914 |
| refin:none | 0.927 | 0.846 | 0.692 | 0.913 |
| refres, 7 marks | 0.940 | 0.841 | 0.863 | 0.930 |
| refres:ctcf_k27ac | 0.936 | 0.831 | 0.848 | 0.926 |
| refres:ctcf | 0.931 | 0.836 | 0.714 | 0.910 |
| refres:none | 0.932 | 0.841 | 0.694 | 0.916 |
| honest (no consensus), 7 marks | 0.817 | 0.709 | 0.734 | 0.759 |
| honest:ctcf_k27ac | 0.773 | 0.653 | 0.686 | 0.741 |
| honest:ctcf | 0.737 | 0.652 | 0.459 | 0.657 |
| honest:none | 0.581 | 0.601 | 0.410 | 0.519 |
| differential channels, 7 marks | 0.944 | 0.841 | 0.858 | 0.939 |
| diff:ctcf_k27ac | 0.938 | 0.828 | 0.842 | 0.930 |
| diff:ctcf | 0.932 | 0.839 | 0.724 | 0.909 |
| diff:none | 0.933 | 0.842 | 0.704 | 0.916 |
| windows | 413 | 413 | 413 | 413 |
| split-half ceiling (Spearman-Brown) | 0.978 | 0.857 | 0.977 | 0.978 |

### Structure measures, 25 kb, Akita test windows (median over windows)

| predictor | sample | pearson | scc | insulation_corr | boundary_f1 | loop_enrich_corr | loop_apa_ratio | compartment_corr | eigenvector_abs_corr |
|---|---|---|---|---|---|---|---|---|---|
| consensus12 alone | PC-3 | 0.932 | 0.933 | 0.972 | 1.000 | 0.917 | 0.648 | 0.942 | 0.975 |
| consensus12 alone | Panc1 | 0.839 | 0.770 | 0.956 | 1.000 | 0.666 | 0.738 | 0.889 | 0.958 |
| consensus12 alone | NK cell | 0.707 | 0.661 | 0.808 | 0.000 | 0.690 | 0.722 | 0.747 | 0.858 |
| consensus12 alone | Caco-2 | 0.915 | 0.915 | 0.965 | 1.000 | 0.911 | 0.795 | 0.923 | 0.964 |
| best other cell (measured) | PC-3 | 0.897 | 0.899 | 0.958 | 1.000 | 0.867 | 0.628 | 0.915 | 0.965 |
| best other cell (measured) | Panc1 | 0.811 | 0.743 | 0.935 | 0.667 | 0.646 | 0.743 | 0.863 | 0.948 |
| best other cell (measured) | NK cell | 0.790 | 0.767 | 0.876 | 0.000 | 0.688 | 0.717 | 0.784 | 0.884 |
| best other cell (measured) | Caco-2 | 0.899 | 0.900 | 0.958 | 1.000 | 0.884 | 0.996 | 0.914 | 0.953 |
| Akita (sequence only) | PC-3 | 0.663 | 0.771 | 0.877 | 0.667 | 0.715 | 0.334 | 0.706 | 0.784 |
| Akita (sequence only) | Panc1 | 0.549 | 0.634 | 0.882 | 0.000 | 0.537 | 0.398 | 0.638 | 0.704 |
| Akita (sequence only) | NK cell | 0.404 | 0.507 | 0.640 | 0.000 | 0.484 | 0.389 | 0.605 | 0.713 |
| Akita (sequence only) | Caco-2 | 0.688 | 0.766 | 0.902 | 1.000 | 0.747 | 0.414 | 0.721 | 0.783 |
| honest (no consensus), 7 marks | PC-3 | 0.817 | 0.777 | 0.904 | 1.000 | 0.749 | 0.306 | 0.858 | 0.935 |
| honest (no consensus), 7 marks | Panc1 | 0.709 | 0.561 | 0.830 | 0.000 | 0.486 | 0.306 | 0.794 | 0.907 |
| honest (no consensus), 7 marks | NK cell | 0.734 | 0.705 | 0.844 | 0.000 | 0.618 | 0.336 | 0.765 | 0.858 |
| honest (no consensus), 7 marks | Caco-2 | 0.759 | 0.742 | 0.872 | 0.000 | 0.715 | 0.309 | 0.815 | 0.914 |
| refin, 7 marks | PC-3 | 0.942 | 0.930 | 0.974 | 1.000 | 0.920 | 0.665 | 0.951 | 0.979 |
| refin, 7 marks | Panc1 | 0.841 | 0.768 | 0.950 | 1.000 | 0.678 | 0.759 | 0.881 | 0.959 |
| refin, 7 marks | NK cell | 0.823 | 0.817 | 0.910 | 0.000 | 0.783 | 0.760 | 0.836 | 0.921 |
| refin, 7 marks | Caco-2 | 0.934 | 0.929 | 0.975 | 1.000 | 0.922 | 0.820 | 0.940 | 0.975 |
| refres, 7 marks | PC-3 | 0.940 | 0.929 | 0.973 | 1.000 | 0.923 | 0.687 | 0.953 | 0.981 |
| refres, 7 marks | Panc1 | 0.841 | 0.768 | 0.949 | 1.000 | 0.672 | 0.829 | 0.891 | 0.960 |
| refres, 7 marks | NK cell | 0.863 | 0.859 | 0.930 | 0.000 | 0.790 | 0.781 | 0.871 | 0.941 |
| refres, 7 marks | Caco-2 | 0.930 | 0.926 | 0.977 | 1.000 | 0.925 | 0.864 | 0.934 | 0.972 |
| differential channels, 7 marks | PC-3 | 0.944 | 0.940 | 0.980 | 1.000 | 0.924 | 0.728 | 0.956 | 0.983 |
| differential channels, 7 marks | Panc1 | 0.841 | 0.765 | 0.957 | 1.000 | 0.677 | 0.771 | 0.891 | 0.956 |
| differential channels, 7 marks | NK cell | 0.858 | 0.838 | 0.931 | 0.667 | 0.800 | 0.750 | 0.860 | 0.932 |
| differential channels, 7 marks | Caco-2 | 0.939 | 0.933 | 0.978 | 1.000 | 0.933 | 0.863 | 0.948 | 0.979 |

### Chromosomes 2, 3, 11, 14, 17 (held out of the chrho models), 10 kb

| windows | predictor | PC-3 | Panc1 | NK cell | Caco-2 |
|---|---|---|---|---|---|
| test windows | consensus12 alone | 0.886 | 0.609 | 0.657 | 0.851 |
| test windows | honest_chrho:all | 0.730 | 0.522 | 0.593 | 0.711 |
| test windows | refin_chrho:all | 0.890 | 0.626 | 0.768 | 0.869 |
| test windows | refres_chrho:all | 0.900 | 0.614 | 0.716 | 0.868 |
| test windows | honest (no consensus), 7 marks | 0.756 | 0.515 | 0.622 | 0.706 |
| test windows | refin, 7 marks | 0.892 | 0.613 | 0.760 | 0.879 |
| test windows | refres, 7 marks | 0.893 | 0.612 | 0.778 | 0.882 |
| test windows | differential channels, 7 marks | 0.897 | 0.623 | 0.787 | 0.885 |
| 1 Mb tiles | consensus12 alone | 0.892 | 0.601 | 0.633 | 0.870 |
| 1 Mb tiles | honest_chrho:all | 0.754 | 0.510 | 0.604 | 0.726 |
| 1 Mb tiles | refin_chrho:all | 0.900 | 0.612 | 0.768 | 0.887 |
| 1 Mb tiles | refres_chrho:all | 0.903 | 0.608 | 0.733 | 0.895 |
| 1 Mb tiles | honest (no consensus), 7 marks | 0.764 | 0.513 | 0.646 | 0.729 |
| 1 Mb tiles | refin, 7 marks | 0.902 | 0.605 | 0.756 | 0.894 |
| 1 Mb tiles | refres, 7 marks | 0.905 | 0.604 | 0.771 | 0.899 |
| 1 Mb tiles | differential channels, 7 marks | 0.910 | 0.608 | 0.789 | 0.902 |

### Whole genome, 1 Mb tiles (chr1-22), median Pearson

| resolution | predictor | PC-3 | Panc1 | NK cell | Caco-2 |
|---|---|---|---|---|---|
| 10 kb | consensus12 alone | 0.878 | 0.590 | 0.637 | 0.867 |
| 10 kb | honest (no consensus), 7 marks | 0.762 | 0.498 | 0.644 | 0.713 |
| 10 kb | refin, 7 marks | 0.893 | 0.591 | 0.763 | 0.887 |
| 10 kb | refres, 7 marks | 0.895 | 0.595 | 0.782 | 0.892 |
| 10 kb | differential channels, 7 marks | 0.899 | 0.594 | 0.794 | 0.896 |
| 10 kb | Akita (sequence only) | 0.696 | 0.416 | 0.466 | 0.714 |
| 25 kb | consensus12 alone | 0.928 | 0.824 | 0.688 | 0.911 |
| 25 kb | honest (no consensus), 7 marks | 0.801 | 0.687 | 0.702 | 0.737 |
| 25 kb | refin, 7 marks | 0.936 | 0.828 | 0.819 | 0.931 |
| 25 kb | refres, 7 marks | 0.935 | 0.832 | 0.850 | 0.929 |
| 25 kb | differential channels, 7 marks | 0.942 | 0.830 | 0.848 | 0.937 |
| 25 kb | Akita (sequence only) | 0.698 | 0.539 | 0.457 | 0.721 |

### Paired contrasts on the test windows (median difference of per-window Pearson, 95% bootstrap interval)

| sample | res | contrast | windows | difference | share of windows refin better |
|---|---|---|---|---|---|
| PC-3 | 10 kb | A: refin:all - consensus12 | 413 | +0.0116 [+0.0072, +0.0206] | 0.71 |
| PC-3 | 10 kb | B: refin:all - honest:all | 413 | +0.1231 [+0.1109, +0.1399] | 0.95 |
| PC-3 | 10 kb | refin:all - akita | 413 | +0.2218 [+0.1984, +0.2535] | 0.98 |
| PC-3 | 10 kb | refin:all - best_cell | 413 | +0.0645 [+0.0523, +0.0764] | 0.90 |
| PC-3 | 10 kb | refin:all - diff:all | 413 | -0.0072 [-0.0106, -0.0021] | 0.38 |
| PC-3 | 10 kb | refin:all - diff:ctcf | 413 | +0.0050 [+0.0004, +0.0113] | 0.60 |
| PC-3 | 10 kb | refin:all - diff:ctcf_k27ac | 413 | -0.0027 [-0.0053, +0.0038] | 0.50 |
| PC-3 | 10 kb | refin:all - diff:none | 413 | +0.0096 [+0.0046, +0.0155] | 0.66 |
| PC-3 | 10 kb | refin:all - honest:ctcf | 413 | +0.2034 [+0.1861, +0.2301] | 0.99 |
| PC-3 | 10 kb | refin:all - honest:ctcf_k27ac | 413 | +0.1684 [+0.1492, +0.1865] | 0.98 |
| PC-3 | 10 kb | refin:all - honest:none | 413 | +0.3759 [+0.3423, +0.4119] | 0.99 |
| PC-3 | 10 kb | refin:all - honest_chrho:all | 86 | +0.1624 [+0.1286, +0.2006] | 0.98 |
| PC-3 | 10 kb | refin:all - honest_s1:all | 413 | +0.1310 [+0.1186, +0.1475] | 0.96 |
| PC-3 | 10 kb | refin:all - honest_s2:all | 413 | +0.1396 [+0.1284, +0.1534] | 0.94 |
| PC-3 | 10 kb | refin:all - refin:ctcf | 413 | +0.0146 [+0.0069, +0.0214] | 0.69 |
| PC-3 | 10 kb | refin:all - refin:ctcf_k27ac | 413 | +0.0016 [-0.0011, +0.0059] | 0.58 |
| PC-3 | 10 kb | refin:all - refin:none | 413 | +0.0160 [+0.0092, +0.0235] | 0.74 |
| PC-3 | 10 kb | refin:all - refin_chrho:all | 86 | +0.0021 [-0.0036, +0.0133] | 0.66 |
| PC-3 | 10 kb | refin:all - refin_s1:all | 413 | -0.0015 [-0.0054, +0.0018] | 0.47 |
| PC-3 | 10 kb | refin:all - refin_s2:all | 413 | -0.0028 [-0.0068, +0.0019] | 0.46 |
| PC-3 | 10 kb | refin:all - refres:all | 413 | -0.0008 [-0.0045, +0.0034] | 0.46 |
| PC-3 | 10 kb | refin:all - refres:ctcf | 413 | +0.0151 [+0.0092, +0.0222] | 0.72 |
| PC-3 | 10 kb | refin:all - refres:ctcf_k27ac | 413 | +0.0030 [-0.0006, +0.0080] | 0.58 |
| PC-3 | 10 kb | refin:all - refres:none | 413 | +0.0140 [+0.0087, +0.0211] | 0.71 |
| PC-3 | 10 kb | refin:all - refres_chrho:all | 86 | -0.0081 [-0.0116, +0.0030] | 0.47 |
| PC-3 | 25 kb | A: refin:all - consensus12 | 413 | +0.0099 [+0.0040, +0.0146] | 0.62 |
| PC-3 | 25 kb | B: refin:all - honest:all | 413 | +0.1250 [+0.1074, +0.1487] | 0.91 |
| PC-3 | 25 kb | refin:all - akita | 413 | +0.2781 [+0.2378, +0.3125] | 0.98 |
| PC-3 | 25 kb | refin:all - best_cell | 413 | +0.0441 [+0.0341, +0.0507] | 0.84 |
| PC-3 | 25 kb | refin:all - diff:all | 413 | -0.0026 [-0.0066, +0.0007] | 0.34 |
| PC-3 | 25 kb | refin:all - diff:ctcf | 413 | +0.0096 [+0.0045, +0.0153] | 0.61 |
| PC-3 | 25 kb | refin:all - diff:ctcf_k27ac | 413 | +0.0030 [-0.0006, +0.0092] | 0.54 |
| PC-3 | 25 kb | refin:all - diff:none | 413 | +0.0088 [+0.0051, +0.0150] | 0.62 |
| PC-3 | 25 kb | refin:all - honest:ctcf | 413 | +0.2045 [+0.1852, +0.2252] | 0.97 |
| PC-3 | 25 kb | refin:all - honest:ctcf_k27ac | 413 | +0.1681 [+0.1486, +0.1943] | 0.96 |
| PC-3 | 25 kb | refin:all - honest:none | 413 | +0.3607 [+0.3128, +0.3917] | 0.99 |
| PC-3 | 25 kb | refin:all - honest_chrho:all | 86 | +0.1614 [+0.1357, +0.1968] | 0.98 |
| PC-3 | 25 kb | refin:all - honest_s1:all | 413 | +0.1475 [+0.1334, +0.1726] | 0.95 |
| PC-3 | 25 kb | refin:all - honest_s2:all | 413 | +0.1288 [+0.1154, +0.1554] | 0.93 |
| PC-3 | 25 kb | refin:all - refin:ctcf | 413 | +0.0106 [+0.0058, +0.0159] | 0.65 |
| PC-3 | 25 kb | refin:all - refin:ctcf_k27ac | 413 | +0.0097 [+0.0059, +0.0130] | 0.68 |
| PC-3 | 25 kb | refin:all - refin:none | 413 | +0.0141 [+0.0087, +0.0177] | 0.71 |
| PC-3 | 25 kb | refin:all - refin_chrho:all | 86 | +0.0015 [-0.0046, +0.0096] | 0.51 |
| PC-3 | 25 kb | refin:all - refin_s1:all | 413 | +0.0008 [-0.0018, +0.0044] | 0.53 |
| PC-3 | 25 kb | refin:all - refin_s2:all | 413 | +0.0017 [-0.0013, +0.0042] | 0.53 |
| PC-3 | 25 kb | refin:all - refres:all | 413 | +0.0011 [-0.0018, +0.0052] | 0.53 |
| PC-3 | 25 kb | refin:all - refres:ctcf | 413 | +0.0110 [+0.0049, +0.0185] | 0.64 |
| PC-3 | 25 kb | refin:all - refres:ctcf_k27ac | 413 | +0.0056 [+0.0017, +0.0118] | 0.61 |
| PC-3 | 25 kb | refin:all - refres:none | 413 | +0.0099 [+0.0052, +0.0141] | 0.61 |
| PC-3 | 25 kb | refin:all - refres_chrho:all | 86 | -0.0009 [-0.0081, +0.0072] | 0.40 |
| Panc1 | 10 kb | A: refin:all - consensus12 | 413 | +0.0020 [-0.0088, +0.0118] | 0.51 |
| Panc1 | 10 kb | B: refin:all - honest:all | 413 | +0.0884 [+0.0637, +0.1053] | 0.84 |
| Panc1 | 10 kb | refin:all - akita | 413 | +0.1809 [+0.1537, +0.2181] | 0.90 |
| Panc1 | 10 kb | refin:all - best_cell | 413 | +0.0376 [+0.0194, +0.0553] | 0.76 |
| Panc1 | 10 kb | refin:all - diff:all | 413 | -0.0086 [-0.0190, +0.0039] | 0.38 |
| Panc1 | 10 kb | refin:all - diff:ctcf | 413 | -0.0000 [-0.0160, +0.0079] | 0.46 |
| Panc1 | 10 kb | refin:all - diff:ctcf_k27ac | 413 | +0.0067 [-0.0046, +0.0161] | 0.56 |
| Panc1 | 10 kb | refin:all - diff:none | 413 | -0.0062 [-0.0162, +0.0058] | 0.42 |
| Panc1 | 10 kb | refin:all - honest:ctcf | 413 | +0.1402 [+0.1159, +0.1689] | 0.87 |
| Panc1 | 10 kb | refin:all - honest:ctcf_k27ac | 413 | +0.1280 [+0.1076, +0.1530] | 0.89 |
| Panc1 | 10 kb | refin:all - honest:none | 413 | +0.1648 [+0.1430, +0.1987] | 0.91 |
| Panc1 | 10 kb | refin:all - honest_chrho:all | 86 | +0.0911 [+0.0382, +0.1495] | 0.83 |
| Panc1 | 10 kb | refin:all - honest_s1:all | 413 | +0.0901 [+0.0677, +0.1124] | 0.86 |
| Panc1 | 10 kb | refin:all - honest_s2:all | 413 | +0.0943 [+0.0768, +0.1172] | 0.86 |
| Panc1 | 10 kb | refin:all - refin:ctcf | 413 | +0.0049 [-0.0101, +0.0148] | 0.48 |
| Panc1 | 10 kb | refin:all - refin:ctcf_k27ac | 413 | +0.0169 [+0.0059, +0.0247] | 0.74 |
| Panc1 | 10 kb | refin:all - refin:none | 413 | -0.0051 [-0.0152, +0.0057] | 0.44 |
| Panc1 | 10 kb | refin:all - refin_chrho:all | 86 | -0.0138 [-0.0244, +0.0067] | 0.33 |
| Panc1 | 10 kb | refin:all - refin_s1:all | 413 | -0.0070 [-0.0153, +0.0038] | 0.49 |
| Panc1 | 10 kb | refin:all - refin_s2:all | 413 | +0.0110 [-0.0022, +0.0191] | 0.59 |
| Panc1 | 10 kb | refin:all - refres:all | 413 | -0.0023 [-0.0106, +0.0076] | 0.47 |
| Panc1 | 10 kb | refin:all - refres:ctcf | 413 | -0.0050 [-0.0162, +0.0076] | 0.47 |
| Panc1 | 10 kb | refin:all - refres:ctcf_k27ac | 413 | +0.0183 [+0.0049, +0.0246] | 0.71 |
| Panc1 | 10 kb | refin:all - refres:none | 413 | -0.0086 [-0.0195, +0.0007] | 0.40 |
| Panc1 | 10 kb | refin:all - refres_chrho:all | 86 | -0.0012 [-0.0100, +0.0135] | 0.50 |
| Panc1 | 25 kb | A: refin:all - consensus12 | 413 | +0.0025 [-0.0038, +0.0137] | 0.52 |
| Panc1 | 25 kb | B: refin:all - honest:all | 413 | +0.1323 [+0.1154, +0.1513] | 0.86 |
| Panc1 | 25 kb | refin:all - akita | 413 | +0.2927 [+0.2498, +0.3493] | 0.94 |
| Panc1 | 25 kb | refin:all - best_cell | 413 | +0.0305 [+0.0191, +0.0412] | 0.69 |
| Panc1 | 25 kb | refin:all - diff:all | 413 | +0.0004 [-0.0054, +0.0081] | 0.46 |
| Panc1 | 25 kb | refin:all - diff:ctcf | 413 | +0.0025 [-0.0060, +0.0104] | 0.53 |
| Panc1 | 25 kb | refin:all - diff:ctcf_k27ac | 413 | +0.0129 [+0.0020, +0.0207] | 0.59 |
| Panc1 | 25 kb | refin:all - diff:none | 413 | -0.0009 [-0.0074, +0.0094] | 0.49 |
| Panc1 | 25 kb | refin:all - honest:ctcf | 413 | +0.1897 [+0.1682, +0.2288] | 0.89 |
| Panc1 | 25 kb | refin:all - honest:ctcf_k27ac | 413 | +0.1880 [+0.1605, +0.2197] | 0.92 |
| Panc1 | 25 kb | refin:all - honest:none | 413 | +0.2405 [+0.2011, +0.2812] | 0.88 |
| Panc1 | 25 kb | refin:all - honest_chrho:all | 86 | +0.1248 [+0.0850, +0.1664] | 0.86 |
| Panc1 | 25 kb | refin:all - honest_s1:all | 413 | +0.1453 [+0.1191, +0.1620] | 0.86 |
| Panc1 | 25 kb | refin:all - honest_s2:all | 413 | +0.1403 [+0.1241, +0.1657] | 0.88 |
| Panc1 | 25 kb | refin:all - refin:ctcf | 413 | +0.0012 [-0.0075, +0.0080] | 0.48 |
| Panc1 | 25 kb | refin:all - refin:ctcf_k27ac | 413 | +0.0053 [+0.0010, +0.0141] | 0.64 |
| Panc1 | 25 kb | refin:all - refin:none | 413 | -0.0044 [-0.0112, +0.0033] | 0.43 |
| Panc1 | 25 kb | refin:all - refin_chrho:all | 86 | +0.0134 [-0.0054, +0.0254] | 0.55 |
| Panc1 | 25 kb | refin:all - refin_s1:all | 413 | +0.0096 [+0.0019, +0.0145] | 0.60 |
| Panc1 | 25 kb | refin:all - refin_s2:all | 413 | +0.0072 [+0.0008, +0.0163] | 0.62 |
| Panc1 | 25 kb | refin:all - refres:all | 413 | +0.0007 [-0.0048, +0.0064] | 0.48 |
| Panc1 | 25 kb | refin:all - refres:ctcf | 413 | +0.0053 [-0.0048, +0.0121] | 0.48 |
| Panc1 | 25 kb | refin:all - refres:ctcf_k27ac | 413 | +0.0099 [+0.0024, +0.0163] | 0.61 |
| Panc1 | 25 kb | refin:all - refres:none | 413 | -0.0001 [-0.0082, +0.0067] | 0.45 |
| Panc1 | 25 kb | refin:all - refres_chrho:all | 86 | -0.0003 [-0.0113, +0.0175] | 0.63 |
| NK cell | 10 kb | A: refin:all - consensus12 | 413 | +0.1083 [+0.0899, +0.1448] | 0.84 |
| NK cell | 10 kb | B: refin:all - honest:all | 413 | +0.0899 [+0.0680, +0.1234] | 0.72 |
| NK cell | 10 kb | refin:all - akita | 413 | +0.3456 [+0.3115, +0.3785] | 0.92 |
| NK cell | 10 kb | refin:all - best_cell | 413 | +0.0717 [+0.0500, +0.1073] | 0.67 |
| NK cell | 10 kb | refin:all - diff:all | 413 | -0.0361 [-0.0507, -0.0191] | 0.30 |
| NK cell | 10 kb | refin:all - diff:ctcf | 413 | +0.1015 [+0.0818, +0.1309] | 0.83 |
| NK cell | 10 kb | refin:all - diff:ctcf_k27ac | 413 | -0.0340 [-0.0471, -0.0143] | 0.30 |
| NK cell | 10 kb | refin:all - diff:none | 413 | +0.1258 [+0.1078, +0.1606] | 0.87 |
| NK cell | 10 kb | refin:all - honest:ctcf | 413 | +0.3119 [+0.2756, +0.3702] | 0.88 |
| NK cell | 10 kb | refin:all - honest:ctcf_k27ac | 413 | +0.1137 [+0.0927, +0.1395] | 0.74 |
| NK cell | 10 kb | refin:all - honest:none | 413 | +0.4480 [+0.4122, +0.4966] | 0.92 |
| NK cell | 10 kb | refin:all - honest_chrho:all | 86 | +0.1676 [+0.1187, +0.3042] | 0.83 |
| NK cell | 10 kb | refin:all - honest_s1:all | 413 | +0.1138 [+0.0837, +0.1395] | 0.74 |
| NK cell | 10 kb | refin:all - honest_s2:all | 413 | +0.0981 [+0.0751, +0.1275] | 0.70 |
| NK cell | 10 kb | refin:all - refin:ctcf | 413 | +0.0892 [+0.0678, +0.1173] | 0.78 |
| NK cell | 10 kb | refin:all - refin:ctcf_k27ac | 413 | -0.0153 [-0.0239, +0.0071] | 0.44 |
| NK cell | 10 kb | refin:all - refin:none | 413 | +0.1221 [+0.0951, +0.1494] | 0.85 |
| NK cell | 10 kb | refin:all - refin_chrho:all | 86 | -0.0073 [-0.0332, +0.0156] | 0.35 |
| NK cell | 10 kb | refin:all - refin_s1:all | 413 | -0.0151 [-0.0237, +0.0031] | 0.39 |
| NK cell | 10 kb | refin:all - refin_s2:all | 413 | -0.0264 [-0.0360, -0.0066] | 0.36 |
| NK cell | 10 kb | refin:all - refres:all | 413 | -0.0165 [-0.0297, -0.0022] | 0.37 |
| NK cell | 10 kb | refin:all - refres:ctcf | 413 | +0.0960 [+0.0800, +0.1279] | 0.84 |
| NK cell | 10 kb | refin:all - refres:ctcf_k27ac | 413 | -0.0274 [-0.0383, -0.0039] | 0.43 |
| NK cell | 10 kb | refin:all - refres:none | 413 | +0.1241 [+0.1007, +0.1579] | 0.85 |
| NK cell | 10 kb | refin:all - refres_chrho:all | 86 | +0.0440 [+0.0072, +0.0652] | 0.64 |
| NK cell | 25 kb | A: refin:all - consensus12 | 413 | +0.1154 [+0.0953, +0.1534] | 0.86 |
| NK cell | 25 kb | B: refin:all - honest:all | 413 | +0.0889 [+0.0619, +0.1242] | 0.70 |
| NK cell | 25 kb | refin:all - akita | 413 | +0.4187 [+0.3676, +0.4578] | 0.93 |
| NK cell | 25 kb | refin:all - best_cell | 413 | +0.0327 [+0.0103, +0.0709] | 0.57 |
| NK cell | 25 kb | refin:all - diff:all | 413 | -0.0347 [-0.0484, -0.0160] | 0.32 |
| NK cell | 25 kb | refin:all - diff:ctcf | 413 | +0.0987 [+0.0830, +0.1320] | 0.81 |
| NK cell | 25 kb | refin:all - diff:ctcf_k27ac | 413 | -0.0196 [-0.0310, -0.0012] | 0.39 |
| NK cell | 25 kb | refin:all - diff:none | 413 | +0.1191 [+0.0983, +0.1514] | 0.86 |
| NK cell | 25 kb | refin:all - honest:ctcf | 413 | +0.3642 [+0.3287, +0.4268] | 0.89 |
| NK cell | 25 kb | refin:all - honest:ctcf_k27ac | 413 | +0.1367 [+0.1086, +0.1638] | 0.78 |
| NK cell | 25 kb | refin:all - honest:none | 413 | +0.4126 [+0.3484, +0.4669] | 0.90 |
| NK cell | 25 kb | refin:all - honest_chrho:all | 86 | +0.1921 [+0.1400, +0.2896] | 0.74 |
| NK cell | 25 kb | refin:all - honest_s1:all | 413 | +0.1260 [+0.0988, +0.1581] | 0.76 |
| NK cell | 25 kb | refin:all - honest_s2:all | 413 | +0.1087 [+0.0842, +0.1488] | 0.73 |
| NK cell | 25 kb | refin:all - refin:ctcf | 413 | +0.1180 [+0.0987, +0.1503] | 0.86 |
| NK cell | 25 kb | refin:all - refin:ctcf_k27ac | 413 | -0.0060 [-0.0188, +0.0121] | 0.39 |
| NK cell | 25 kb | refin:all - refin:none | 413 | +0.1313 [+0.1119, +0.1642] | 0.87 |
| NK cell | 25 kb | refin:all - refin_chrho:all | 86 | -0.0075 [-0.0485, +0.0141] | 0.30 |
| NK cell | 25 kb | refin:all - refin_s1:all | 413 | -0.0285 [-0.0417, -0.0095] | 0.39 |
| NK cell | 25 kb | refin:all - refin_s2:all | 413 | -0.0241 [-0.0364, -0.0096] | 0.37 |
| NK cell | 25 kb | refin:all - refres:all | 413 | -0.0398 [-0.0506, -0.0199] | 0.29 |
| NK cell | 25 kb | refin:all - refres:ctcf | 413 | +0.1089 [+0.0859, +0.1316] | 0.82 |
| NK cell | 25 kb | refin:all - refres:ctcf_k27ac | 413 | -0.0253 [-0.0416, -0.0029] | 0.41 |
| NK cell | 25 kb | refin:all - refres:none | 413 | +0.1288 [+0.1059, +0.1585] | 0.87 |
| NK cell | 25 kb | refin:all - refres_chrho:all | 86 | +0.0127 [-0.0160, +0.0478] | 0.58 |
| Caco-2 | 10 kb | A: refin:all - consensus12 | 413 | +0.0239 [+0.0155, +0.0331] | 0.79 |
| Caco-2 | 10 kb | B: refin:all - honest:all | 413 | +0.1573 [+0.1420, +0.1733] | 0.94 |
| Caco-2 | 10 kb | refin:all - akita | 413 | +0.2057 [+0.1817, +0.2289] | 0.98 |
| Caco-2 | 10 kb | refin:all - best_cell | 413 | +0.0593 [+0.0496, +0.0772] | 0.86 |
| Caco-2 | 10 kb | refin:all - diff:all | 413 | -0.0038 [-0.0098, +0.0014] | 0.34 |
| Caco-2 | 10 kb | refin:all - diff:ctcf | 413 | +0.0148 [+0.0082, +0.0246] | 0.74 |
| Caco-2 | 10 kb | refin:all - diff:ctcf_k27ac | 413 | +0.0006 [-0.0045, +0.0066] | 0.54 |
| Caco-2 | 10 kb | refin:all - diff:none | 413 | +0.0150 [+0.0077, +0.0240] | 0.72 |
| Caco-2 | 10 kb | refin:all - honest:ctcf | 413 | +0.2352 [+0.2119, +0.2662] | 0.98 |
| Caco-2 | 10 kb | refin:all - honest:ctcf_k27ac | 413 | +0.1923 [+0.1733, +0.2075] | 0.95 |
| Caco-2 | 10 kb | refin:all - honest:none | 413 | +0.3801 [+0.3540, +0.4115] | 1.00 |
| Caco-2 | 10 kb | refin:all - honest_chrho:all | 86 | +0.1677 [+0.1341, +0.2286] | 0.94 |
| Caco-2 | 10 kb | refin:all - honest_s1:all | 413 | +0.1576 [+0.1423, +0.1808] | 0.95 |
| Caco-2 | 10 kb | refin:all - honest_s2:all | 413 | +0.1520 [+0.1363, +0.1730] | 0.95 |
| Caco-2 | 10 kb | refin:all - refin:ctcf | 413 | +0.0230 [+0.0145, +0.0311] | 0.79 |
| Caco-2 | 10 kb | refin:all - refin:ctcf_k27ac | 413 | +0.0055 [-0.0011, +0.0100] | 0.60 |
| Caco-2 | 10 kb | refin:all - refin:none | 413 | +0.0233 [+0.0138, +0.0311] | 0.80 |
| Caco-2 | 10 kb | refin:all - refin_chrho:all | 86 | +0.0096 [+0.0010, +0.0233] | 0.69 |
| Caco-2 | 10 kb | refin:all - refin_s1:all | 413 | +0.0028 [-0.0024, +0.0073] | 0.51 |
| Caco-2 | 10 kb | refin:all - refin_s2:all | 413 | -0.0006 [-0.0044, +0.0045] | 0.45 |
| Caco-2 | 10 kb | refin:all - refres:all | 413 | -0.0033 [-0.0073, +0.0011] | 0.31 |
| Caco-2 | 10 kb | refin:all - refres:ctcf | 413 | +0.0207 [+0.0138, +0.0313] | 0.78 |
| Caco-2 | 10 kb | refin:all - refres:ctcf_k27ac | 413 | -0.0008 [-0.0049, +0.0045] | 0.49 |
| Caco-2 | 10 kb | refin:all - refres:none | 413 | +0.0191 [+0.0113, +0.0284] | 0.77 |
| Caco-2 | 10 kb | refin:all - refres_chrho:all | 86 | +0.0110 [-0.0031, +0.0172] | 0.53 |
| Caco-2 | 25 kb | A: refin:all - consensus12 | 413 | +0.0192 [+0.0121, +0.0279] | 0.78 |
| Caco-2 | 25 kb | B: refin:all - honest:all | 413 | +0.1749 [+0.1541, +0.1874] | 0.94 |
| Caco-2 | 25 kb | refin:all - akita | 413 | +0.2457 [+0.2200, +0.2680] | 0.96 |
| Caco-2 | 25 kb | refin:all - best_cell | 413 | +0.0348 [+0.0239, +0.0454] | 0.75 |
| Caco-2 | 25 kb | refin:all - diff:all | 413 | -0.0049 [-0.0108, -0.0018] | 0.36 |
| Caco-2 | 25 kb | refin:all - diff:ctcf | 413 | +0.0251 [+0.0177, +0.0309] | 0.78 |
| Caco-2 | 25 kb | refin:all - diff:ctcf_k27ac | 413 | +0.0045 [-0.0006, +0.0095] | 0.55 |
| Caco-2 | 25 kb | refin:all - diff:none | 413 | +0.0184 [+0.0133, +0.0272] | 0.77 |
| Caco-2 | 25 kb | refin:all - honest:ctcf | 413 | +0.2770 [+0.2439, +0.3098] | 0.98 |
| Caco-2 | 25 kb | refin:all - honest:ctcf_k27ac | 413 | +0.1926 [+0.1704, +0.2216] | 0.94 |
| Caco-2 | 25 kb | refin:all - honest:none | 413 | +0.4149 [+0.3725, +0.4518] | 0.99 |
| Caco-2 | 25 kb | refin:all - honest_chrho:all | 86 | +0.1822 [+0.1532, +0.2365] | 0.94 |
| Caco-2 | 25 kb | refin:all - honest_s1:all | 413 | +0.1889 [+0.1718, +0.2164] | 0.94 |
| Caco-2 | 25 kb | refin:all - honest_s2:all | 413 | +0.1560 [+0.1437, +0.1808] | 0.93 |
| Caco-2 | 25 kb | refin:all - refin:ctcf | 413 | +0.0202 [+0.0140, +0.0254] | 0.79 |
| Caco-2 | 25 kb | refin:all - refin:ctcf_k27ac | 413 | +0.0056 [-0.0009, +0.0093] | 0.61 |
| Caco-2 | 25 kb | refin:all - refin:none | 413 | +0.0210 [+0.0147, +0.0283] | 0.79 |
| Caco-2 | 25 kb | refin:all - refin_chrho:all | 86 | -0.0084 [-0.0179, +0.0031] | 0.47 |
| Caco-2 | 25 kb | refin:all - refin_s1:all | 413 | -0.0033 [-0.0085, -0.0000] | 0.40 |
| Caco-2 | 25 kb | refin:all - refin_s2:all | 413 | -0.0019 [-0.0056, +0.0014] | 0.46 |
| Caco-2 | 25 kb | refin:all - refres:all | 413 | +0.0036 [-0.0009, +0.0063] | 0.55 |
| Caco-2 | 25 kb | refin:all - refres:ctcf | 413 | +0.0245 [+0.0167, +0.0312] | 0.77 |
| Caco-2 | 25 kb | refin:all - refres:ctcf_k27ac | 413 | +0.0078 [+0.0024, +0.0129] | 0.62 |
| Caco-2 | 25 kb | refin:all - refres:none | 413 | +0.0179 [+0.0118, +0.0248] | 0.76 |
| Caco-2 | 25 kb | refin:all - refres_chrho:all | 86 | -0.0013 [-0.0107, +0.0122] | 0.58 |

### Paired contrasts on the whole-genome 1 Mb tiles

| sample | res | contrast | tiles | difference | share of tiles refin better |
|---|---|---|---|---|---|
| PC-3 | 10 kb | A: refin:all - consensus12 | 2922 | +0.0157 [+0.0133, +0.0184] | 0.69 |
| PC-3 | 10 kb | B: refin:all - honest:all | 2922 | +0.1311 [+0.1255, +0.1379] | 0.92 |
| PC-3 | 10 kb | refin:all - akita | 2922 | +0.1975 [+0.1878, +0.2054] | 0.96 |
| PC-3 | 10 kb | refin:all - diff:all | 2922 | -0.0059 [-0.0078, -0.0042] | 0.34 |
| PC-3 | 10 kb | refin:all - honest_chrho:all | 800 | +0.1482 [+0.1394, +0.1647] | 0.94 |
| PC-3 | 10 kb | refin:all - refin_chrho:all | 800 | +0.0023 [+0.0002, +0.0058] | 0.63 |
| PC-3 | 10 kb | refin:all - refres:all | 2922 | -0.0017 [-0.0034, +0.0001] | 0.42 |
| PC-3 | 10 kb | refin:all - refres_chrho:all | 800 | -0.0012 [-0.0041, +0.0013] | 0.47 |
| PC-3 | 25 kb | A: refin:all - consensus12 | 2888 | +0.0080 [+0.0058, +0.0099] | 0.62 |
| PC-3 | 25 kb | B: refin:all - honest:all | 2888 | +0.1346 [+0.1272, +0.1425] | 0.93 |
| PC-3 | 25 kb | refin:all - akita | 2888 | +0.2377 [+0.2282, +0.2485] | 0.97 |
| PC-3 | 25 kb | refin:all - diff:all | 2888 | -0.0061 [-0.0075, -0.0043] | 0.38 |
| PC-3 | 25 kb | refin:all - honest_chrho:all | 793 | +0.1440 [+0.1309, +0.1626] | 0.94 |
| PC-3 | 25 kb | refin:all - refin_chrho:all | 793 | +0.0028 [-0.0001, +0.0057] | 0.61 |
| PC-3 | 25 kb | refin:all - refres:all | 2888 | +0.0006 [-0.0010, +0.0018] | 0.53 |
| PC-3 | 25 kb | refin:all - refres_chrho:all | 793 | +0.0022 [-0.0004, +0.0057] | 0.59 |
| Panc1 | 10 kb | A: refin:all - consensus12 | 2936 | +0.0015 [-0.0022, +0.0050] | 0.51 |
| Panc1 | 10 kb | B: refin:all - honest:all | 2936 | +0.0935 [+0.0873, +0.1008] | 0.82 |
| Panc1 | 10 kb | refin:all - akita | 2936 | +0.1752 [+0.1671, +0.1864] | 0.88 |
| Panc1 | 10 kb | refin:all - diff:all | 2936 | -0.0030 [-0.0063, +0.0018] | 0.42 |
| Panc1 | 10 kb | refin:all - honest_chrho:all | 801 | +0.0949 [+0.0841, +0.1069] | 0.84 |
| Panc1 | 10 kb | refin:all - refin_chrho:all | 801 | -0.0070 [-0.0129, -0.0025] | 0.32 |
| Panc1 | 10 kb | refin:all - refres:all | 2936 | -0.0035 [-0.0060, -0.0001] | 0.45 |
| Panc1 | 10 kb | refin:all - refres_chrho:all | 801 | -0.0033 [-0.0076, +0.0036] | 0.47 |
| Panc1 | 25 kb | A: refin:all - consensus12 | 2900 | +0.0043 [+0.0012, +0.0074] | 0.52 |
| Panc1 | 25 kb | B: refin:all - honest:all | 2900 | +0.1416 [+0.1325, +0.1508] | 0.87 |
| Panc1 | 25 kb | refin:all - akita | 2900 | +0.2895 [+0.2777, +0.3033] | 0.94 |
| Panc1 | 25 kb | refin:all - diff:all | 2900 | -0.0015 [-0.0043, +0.0018] | 0.49 |
| Panc1 | 25 kb | refin:all - honest_chrho:all | 792 | +0.1280 [+0.1071, +0.1409] | 0.86 |
| Panc1 | 25 kb | refin:all - refin_chrho:all | 792 | +0.0046 [-0.0001, +0.0084] | 0.58 |
| Panc1 | 25 kb | refin:all - refres:all | 2900 | -0.0037 [-0.0064, -0.0013] | 0.45 |
| Panc1 | 25 kb | refin:all - refres_chrho:all | 792 | +0.0024 [-0.0021, +0.0078] | 0.58 |
| NK cell | 10 kb | A: refin:all - consensus12 | 2904 | +0.1259 [+0.1166, +0.1347] | 0.82 |
| NK cell | 10 kb | B: refin:all - honest:all | 2904 | +0.1186 [+0.1072, +0.1294] | 0.72 |
| NK cell | 10 kb | refin:all - akita | 2904 | +0.2963 [+0.2823, +0.3094] | 0.91 |
| NK cell | 10 kb | refin:all - diff:all | 2904 | -0.0314 [-0.0385, -0.0260] | 0.31 |
| NK cell | 10 kb | refin:all - honest_chrho:all | 796 | +0.1512 [+0.1348, +0.1679] | 0.77 |
| NK cell | 10 kb | refin:all - refin_chrho:all | 796 | -0.0119 [-0.0206, -0.0017] | 0.38 |
| NK cell | 10 kb | refin:all - refres:all | 2904 | -0.0196 [-0.0262, -0.0141] | 0.35 |
| NK cell | 10 kb | refin:all - refres_chrho:all | 796 | +0.0226 [+0.0124, +0.0328] | 0.58 |
| NK cell | 25 kb | A: refin:all - consensus12 | 2880 | +0.1308 [+0.1235, +0.1393] | 0.84 |
| NK cell | 25 kb | B: refin:all - honest:all | 2880 | +0.1172 [+0.1030, +0.1288] | 0.73 |
| NK cell | 25 kb | refin:all - akita | 2880 | +0.3626 [+0.3482, +0.3779] | 0.93 |
| NK cell | 25 kb | refin:all - diff:all | 2880 | -0.0283 [-0.0328, -0.0236] | 0.32 |
| NK cell | 25 kb | refin:all - honest_chrho:all | 792 | +0.1315 [+0.1074, +0.1522] | 0.76 |
| NK cell | 25 kb | refin:all - refin_chrho:all | 792 | -0.0254 [-0.0346, -0.0143] | 0.31 |
| NK cell | 25 kb | refin:all - refres:all | 2880 | -0.0311 [-0.0354, -0.0256] | 0.31 |
| NK cell | 25 kb | refin:all - refres_chrho:all | 792 | +0.0031 [-0.0064, +0.0117] | 0.53 |
| Caco-2 | 10 kb | A: refin:all - consensus12 | 2908 | +0.0204 [+0.0178, +0.0237] | 0.77 |
| Caco-2 | 10 kb | B: refin:all - honest:all | 2908 | +0.1749 [+0.1667, +0.1824] | 0.95 |
| Caco-2 | 10 kb | refin:all - akita | 2908 | +0.1731 [+0.1632, +0.1820] | 0.96 |
| Caco-2 | 10 kb | refin:all - diff:all | 2908 | -0.0081 [-0.0101, -0.0061] | 0.36 |
| Caco-2 | 10 kb | refin:all - honest_chrho:all | 796 | +0.1672 [+0.1561, +0.1844] | 0.95 |
| Caco-2 | 10 kb | refin:all - refin_chrho:all | 796 | +0.0063 [+0.0038, +0.0105] | 0.68 |
| Caco-2 | 10 kb | refin:all - refres:all | 2908 | -0.0048 [-0.0065, -0.0030] | 0.34 |
| Caco-2 | 10 kb | refin:all - refres_chrho:all | 796 | -0.0011 [-0.0036, +0.0026] | 0.50 |
| Caco-2 | 25 kb | A: refin:all - consensus12 | 2877 | +0.0204 [+0.0179, +0.0227] | 0.75 |
| Caco-2 | 25 kb | B: refin:all - honest:all | 2877 | +0.1945 [+0.1841, +0.2037] | 0.95 |
| Caco-2 | 25 kb | refin:all - akita | 2877 | +0.2109 [+0.2033, +0.2203] | 0.97 |
| Caco-2 | 25 kb | refin:all - diff:all | 2877 | -0.0052 [-0.0070, -0.0034] | 0.37 |
| Caco-2 | 25 kb | refin:all - honest_chrho:all | 789 | +0.1999 [+0.1871, +0.2178] | 0.95 |
| Caco-2 | 25 kb | refin:all - refin_chrho:all | 789 | +0.0005 [-0.0021, +0.0032] | 0.54 |
| Caco-2 | 25 kb | refin:all - refres:all | 2877 | +0.0029 [+0.0015, +0.0044] | 0.56 |
| Caco-2 | 25 kb | refin:all - refres_chrho:all | 789 | +0.0004 [-0.0016, +0.0037] | 0.50 |

### Chromosome-held-out models on chromosomes 2, 3, 11, 14, 17: paired contrasts

| sample | res | windows | contrast | n | difference |
|---|---|---|---|---|---|
| PC-3 | 10 kb | test | refin_chrho:all - consensus12 | 86 | +0.0041 [-0.0053, +0.0131] |
| PC-3 | 10 kb | test | refin_chrho:all - honest_chrho:all | 86 | +0.1604 [+0.1245, +0.1974] |
| PC-3 | 10 kb | test | refin_chrho:all - honest:all | 86 | +0.1340 [+0.1009, +0.1857] |
| PC-3 | 10 kb | test | refres_chrho:all - consensus12 | 86 | +0.0142 [+0.0034, +0.0229] |
| PC-3 | 10 kb | test | refres_chrho:all - honest_chrho:all | 86 | +0.1705 [+0.1339, +0.2046] |
| PC-3 | 10 kb | test | refres_chrho:all - honest:all | 86 | +0.1441 [+0.1113, +0.1939] |
| PC-3 | 25 kb | test | refin_chrho:all - consensus12 | 86 | +0.0043 [-0.0069, +0.0142] |
| PC-3 | 25 kb | test | refin_chrho:all - honest_chrho:all | 86 | +0.1599 [+0.1352, +0.1958] |
| PC-3 | 25 kb | test | refin_chrho:all - honest:all | 86 | +0.1478 [+0.1056, +0.1731] |
| PC-3 | 25 kb | test | refres_chrho:all - consensus12 | 86 | +0.0067 [-0.0046, +0.0158] |
| PC-3 | 25 kb | test | refres_chrho:all - honest_chrho:all | 86 | +0.1623 [+0.1370, +0.1980] |
| PC-3 | 25 kb | test | refres_chrho:all - honest:all | 86 | +0.1502 [+0.1085, +0.1758] |
| Panc1 | 10 kb | test | refin_chrho:all - consensus12 | 86 | +0.0172 [-0.0001, +0.0330] |
| Panc1 | 10 kb | test | refin_chrho:all - honest_chrho:all | 86 | +0.1049 [+0.0482, +0.1612] |
| Panc1 | 10 kb | test | refin_chrho:all - honest:all | 86 | +0.1110 [+0.0629, +0.1495] |
| Panc1 | 10 kb | test | refres_chrho:all - consensus12 | 86 | +0.0047 [-0.0092, +0.0254] |
| Panc1 | 10 kb | test | refres_chrho:all - honest_chrho:all | 86 | +0.0923 [+0.0355, +0.1490] |
| Panc1 | 10 kb | test | refres_chrho:all - honest:all | 86 | +0.0985 [+0.0509, +0.1362] |
| Panc1 | 25 kb | test | refin_chrho:all - consensus12 | 86 | -0.0079 [-0.0198, +0.0188] |
| Panc1 | 25 kb | test | refin_chrho:all - honest_chrho:all | 86 | +0.1114 [+0.0749, +0.1551] |
| Panc1 | 25 kb | test | refin_chrho:all - honest:all | 86 | +0.1017 [+0.0712, +0.1377] |
| Panc1 | 25 kb | test | refres_chrho:all - consensus12 | 86 | +0.0057 [-0.0141, +0.0292] |
| Panc1 | 25 kb | test | refres_chrho:all - honest_chrho:all | 86 | +0.1251 [+0.0838, +0.1600] |
| Panc1 | 25 kb | test | refres_chrho:all - honest:all | 86 | +0.1154 [+0.0784, +0.1394] |
| NK cell | 10 kb | test | refin_chrho:all - consensus12 | 86 | +0.1109 [+0.0747, +0.1650] |
| NK cell | 10 kb | test | refin_chrho:all - honest_chrho:all | 86 | +0.1749 [+0.1225, +0.3024] |
| NK cell | 10 kb | test | refin_chrho:all - honest:all | 86 | +0.1456 [+0.0864, +0.2172] |
| NK cell | 10 kb | test | refres_chrho:all - consensus12 | 86 | +0.0597 [+0.0367, +0.1226] |
| NK cell | 10 kb | test | refres_chrho:all - honest_chrho:all | 86 | +0.1236 [+0.0818, +0.2606] |
| NK cell | 10 kb | test | refres_chrho:all - honest:all | 86 | +0.0943 [+0.0467, +0.1751] |
| NK cell | 25 kb | test | refin_chrho:all - consensus12 | 86 | +0.1475 [+0.0831, +0.1917] |
| NK cell | 25 kb | test | refin_chrho:all - honest_chrho:all | 86 | +0.1996 [+0.1523, +0.2957] |
| NK cell | 25 kb | test | refin_chrho:all - honest:all | 86 | +0.1651 [+0.0897, +0.2415] |
| NK cell | 25 kb | test | refres_chrho:all - consensus12 | 86 | +0.1272 [+0.0535, +0.1619] |
| NK cell | 25 kb | test | refres_chrho:all - honest_chrho:all | 86 | +0.1794 [+0.1223, +0.2750] |
| NK cell | 25 kb | test | refres_chrho:all - honest:all | 86 | +0.1449 [+0.0594, +0.2161] |
| Caco-2 | 10 kb | test | refin_chrho:all - consensus12 | 86 | +0.0189 [+0.0005, +0.0373] |
| Caco-2 | 10 kb | test | refin_chrho:all - honest_chrho:all | 86 | +0.1581 [+0.1218, +0.2173] |
| Caco-2 | 10 kb | test | refin_chrho:all - honest:all | 86 | +0.1631 [+0.1267, +0.2009] |
| Caco-2 | 10 kb | test | refres_chrho:all - consensus12 | 86 | +0.0175 [+0.0027, +0.0393] |
| Caco-2 | 10 kb | test | refres_chrho:all - honest_chrho:all | 86 | +0.1568 [+0.1234, +0.2219] |
| Caco-2 | 10 kb | test | refres_chrho:all - honest:all | 86 | +0.1618 [+0.1278, +0.2035] |
| Caco-2 | 25 kb | test | refin_chrho:all - consensus12 | 86 | +0.0322 [+0.0087, +0.0498] |
| Caco-2 | 25 kb | test | refin_chrho:all - honest_chrho:all | 86 | +0.1906 [+0.1617, +0.2406] |
| Caco-2 | 25 kb | test | refin_chrho:all - honest:all | 86 | +0.1650 [+0.1293, +0.2334] |
| Caco-2 | 25 kb | test | refres_chrho:all - consensus12 | 86 | +0.0250 [-0.0004, +0.0398] |
| Caco-2 | 25 kb | test | refres_chrho:all - honest_chrho:all | 86 | +0.1835 [+0.1532, +0.2323] |
| Caco-2 | 25 kb | test | refres_chrho:all - honest:all | 86 | +0.1579 [+0.1208, +0.2236] |
| PC-3 | 10 kb | tile | refin_chrho:all - consensus12 | 800 | +0.0083 [+0.0047, +0.0118] |
| PC-3 | 10 kb | tile | refin_chrho:all - honest_chrho:all | 800 | +0.1459 [+0.1366, +0.1617] |
| PC-3 | 10 kb | tile | refin_chrho:all - honest:all | 800 | +0.1354 [+0.1235, +0.1489] |
| PC-3 | 10 kb | tile | refres_chrho:all - consensus12 | 800 | +0.0119 [+0.0087, +0.0165] |
| PC-3 | 10 kb | tile | refres_chrho:all - honest_chrho:all | 800 | +0.1494 [+0.1406, +0.1665] |
| PC-3 | 10 kb | tile | refres_chrho:all - honest:all | 800 | +0.1390 [+0.1275, +0.1532] |
| PC-3 | 25 kb | tile | refin_chrho:all - consensus12 | 793 | +0.0013 [-0.0023, +0.0057] |
| PC-3 | 25 kb | tile | refin_chrho:all - honest_chrho:all | 793 | +0.1412 [+0.1276, +0.1596] |
| PC-3 | 25 kb | tile | refin_chrho:all - honest:all | 793 | +0.1302 [+0.1147, +0.1418] |
| PC-3 | 25 kb | tile | refres_chrho:all - consensus12 | 793 | +0.0018 [-0.0019, +0.0059] |
| PC-3 | 25 kb | tile | refres_chrho:all - honest_chrho:all | 793 | +0.1418 [+0.1281, +0.1597] |
| PC-3 | 25 kb | tile | refres_chrho:all - honest:all | 793 | +0.1308 [+0.1144, +0.1421] |
| Panc1 | 10 kb | tile | refin_chrho:all - consensus12 | 801 | +0.0102 [+0.0043, +0.0158] |
| Panc1 | 10 kb | tile | refin_chrho:all - honest_chrho:all | 801 | +0.1018 [+0.0908, +0.1153] |
| Panc1 | 10 kb | tile | refin_chrho:all - honest:all | 801 | +0.0992 [+0.0868, +0.1143] |
| Panc1 | 10 kb | tile | refres_chrho:all - consensus12 | 801 | +0.0065 [-0.0022, +0.0108] |
| Panc1 | 10 kb | tile | refres_chrho:all - honest_chrho:all | 801 | +0.0981 [+0.0860, +0.1096] |
| Panc1 | 10 kb | tile | refres_chrho:all - honest:all | 801 | +0.0954 [+0.0822, +0.1091] |
| Panc1 | 25 kb | tile | refin_chrho:all - consensus12 | 792 | -0.0025 [-0.0078, +0.0050] |
| Panc1 | 25 kb | tile | refin_chrho:all - honest_chrho:all | 792 | +0.1234 [+0.1036, +0.1370] |
| Panc1 | 25 kb | tile | refin_chrho:all - honest:all | 792 | +0.1268 [+0.1072, +0.1451] |
| Panc1 | 25 kb | tile | refres_chrho:all - consensus12 | 792 | -0.0003 [-0.0066, +0.0069] |
| Panc1 | 25 kb | tile | refres_chrho:all - honest_chrho:all | 792 | +0.1256 [+0.1050, +0.1388] |
| Panc1 | 25 kb | tile | refres_chrho:all - honest:all | 792 | +0.1290 [+0.1091, +0.1473] |
| NK cell | 10 kb | tile | refin_chrho:all - consensus12 | 796 | +0.1346 [+0.1171, +0.1508] |
| NK cell | 10 kb | tile | refin_chrho:all - honest_chrho:all | 796 | +0.1631 [+0.1452, +0.1805] |
| NK cell | 10 kb | tile | refin_chrho:all - honest:all | 796 | +0.1215 [+0.1029, +0.1443] |
| NK cell | 10 kb | tile | refres_chrho:all - consensus12 | 796 | +0.1001 [+0.0827, +0.1150] |
| NK cell | 10 kb | tile | refres_chrho:all - honest_chrho:all | 796 | +0.1287 [+0.1154, +0.1442] |
| NK cell | 10 kb | tile | refres_chrho:all - honest:all | 796 | +0.0870 [+0.0707, +0.1093] |
| NK cell | 25 kb | tile | refin_chrho:all - consensus12 | 792 | +0.1611 [+0.1373, +0.1778] |
| NK cell | 25 kb | tile | refin_chrho:all - honest_chrho:all | 792 | +0.1569 [+0.1325, +0.1763] |
| NK cell | 25 kb | tile | refin_chrho:all - honest:all | 792 | +0.1220 [+0.1024, +0.1442] |
| NK cell | 25 kb | tile | refres_chrho:all - consensus12 | 792 | +0.1325 [+0.1127, +0.1481] |
| NK cell | 25 kb | tile | refres_chrho:all - honest_chrho:all | 792 | +0.1284 [+0.1033, +0.1509] |
| NK cell | 25 kb | tile | refres_chrho:all - honest:all | 792 | +0.0935 [+0.0735, +0.1177] |
| Caco-2 | 10 kb | tile | refin_chrho:all - consensus12 | 796 | +0.0174 [+0.0117, +0.0211] |
| Caco-2 | 10 kb | tile | refin_chrho:all - honest_chrho:all | 796 | +0.1609 [+0.1488, +0.1777] |
| Caco-2 | 10 kb | tile | refin_chrho:all - honest:all | 796 | +0.1585 [+0.1426, +0.1740] |
| Caco-2 | 10 kb | tile | refres_chrho:all - consensus12 | 796 | +0.0248 [+0.0195, +0.0284] |
| Caco-2 | 10 kb | tile | refres_chrho:all - honest_chrho:all | 796 | +0.1683 [+0.1564, +0.1851] |
| Caco-2 | 10 kb | tile | refres_chrho:all - honest:all | 796 | +0.1659 [+0.1503, +0.1813] |
| Caco-2 | 25 kb | tile | refin_chrho:all - consensus12 | 789 | +0.0199 [+0.0163, +0.0243] |
| Caco-2 | 25 kb | tile | refin_chrho:all - honest_chrho:all | 789 | +0.1994 [+0.1873, +0.2167] |
| Caco-2 | 25 kb | tile | refin_chrho:all - honest:all | 789 | +0.1831 [+0.1671, +0.2009] |
| Caco-2 | 25 kb | tile | refres_chrho:all - consensus12 | 789 | +0.0200 [+0.0160, +0.0239] |
| Caco-2 | 25 kb | tile | refres_chrho:all - honest_chrho:all | 789 | +0.1995 [+0.1867, +0.2165] |
| Caco-2 | 25 kb | tile | refres_chrho:all - honest:all | 789 | +0.1832 [+0.1669, +0.2008] |
