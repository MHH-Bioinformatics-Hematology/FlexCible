
### Median per-window Pearson, 10 kb, 413 Akita test windows

| predictor | PC-3 | Panc1 | NK cell | Caco-2 |
|---|---|---|---|---|
| consensus12 | 0.883 | 0.613 | 0.658 | 0.870 |
| best_cell | 0.830 | 0.577 | 0.694 | 0.835 |
| akita | 0.673 | 0.434 | 0.420 | 0.688 |
| honest:all | 0.772 | 0.526 | 0.676 | 0.737 |
| refin:all | 0.895 | 0.615 | 0.766 | 0.894 |
| refres:all | 0.896 | 0.617 | 0.782 | 0.897 |
| diff:all | 0.902 | 0.623 | 0.802 | 0.898 |
| rel:all | 0.897 | 0.604 | 0.783 | 0.897 |
| rel:ctcf_k27ac | 0.888 | 0.604 | 0.804 | 0.893 |
| rel:ctcf | 0.876 | 0.619 | 0.669 | 0.869 |
| rel:none | 0.878 | 0.622 | 0.641 | 0.874 |
| rel:all_noref | 0.779 | 0.528 | 0.655 | 0.756 |
| rel:ctcf_k27ac_noref | 0.729 | 0.487 | 0.654 | 0.729 |
| rel:ctcf_noref | 0.695 | 0.478 | 0.424 | 0.641 |
| rel:none_noref | 0.534 | 0.453 | 0.331 | 0.526 |
| split-half ceiling | 0.952 | 0.738 | 0.935 | 0.949 |

### Structure measures, 10 kb (median over windows)

| predictor | sample | pearson | scc | insulation_corr | boundary_f1 | loop_enrich_corr | loop_apa_ratio | compartment_corr | eigenvector_abs_corr |
|---|---|---|---|---|---|---|---|---|---|
| consensus12 | PC-3 | 0.883 | 0.885 | 0.971 | 1.000 | 0.725 | 0.520 | 0.922 | 0.963 |
| consensus12 | Panc1 | 0.613 | 0.517 | 0.936 | 0.000 | 0.204 | 0.114 | 0.762 | 0.899 |
| consensus12 | NK cell | 0.658 | 0.633 | 0.773 | 0.000 | 0.469 | 0.581 | 0.737 | 0.823 |
| consensus12 | Caco-2 | 0.870 | 0.876 | 0.960 | 1.000 | 0.716 | 0.650 | 0.906 | 0.949 |
| best_cell | PC-3 | 0.830 | 0.834 | 0.939 | 0.000 | 0.657 | 0.744 | 0.880 | 0.939 |
| best_cell | Panc1 | 0.577 | 0.485 | 0.911 | 0.000 | 0.167 | 0.157 | 0.760 | 0.888 |
| best_cell | NK cell | 0.694 | 0.674 | 0.866 | 0.000 | 0.361 | 0.549 | 0.746 | 0.836 |
| best_cell | Caco-2 | 0.835 | 0.844 | 0.956 | 0.667 | 0.610 | 0.904 | 0.891 | 0.943 |
| akita | PC-3 | 0.673 | 0.757 | 0.883 | 0.000 | 0.574 | 0.197 | 0.739 | 0.772 |
| akita | Panc1 | 0.434 | 0.461 | 0.863 | 0.000 | 0.181 | 0.048 | 0.591 | 0.659 |
| akita | NK cell | 0.420 | 0.497 | 0.634 | 0.000 | 0.344 | 0.187 | 0.625 | 0.684 |
| akita | Caco-2 | 0.688 | 0.766 | 0.901 | 0.000 | 0.593 | 0.245 | 0.744 | 0.803 |
| refin:all | PC-3 | 0.895 | 0.895 | 0.974 | 1.000 | 0.747 | 0.589 | 0.933 | 0.970 |
| refin:all | Panc1 | 0.615 | 0.519 | 0.929 | 0.000 | 0.206 | 0.120 | 0.769 | 0.905 |
| refin:all | NK cell | 0.766 | 0.750 | 0.890 | 0.000 | 0.546 | 0.591 | 0.819 | 0.898 |
| refin:all | Caco-2 | 0.894 | 0.890 | 0.974 | 1.000 | 0.737 | 0.699 | 0.927 | 0.966 |
| honest:all | PC-3 | 0.772 | 0.758 | 0.894 | 0.000 | 0.579 | 0.256 | 0.843 | 0.916 |
| honest:all | Panc1 | 0.526 | 0.400 | 0.801 | 0.000 | 0.155 | 0.045 | 0.692 | 0.843 |
| honest:all | NK cell | 0.676 | 0.645 | 0.821 | 0.000 | 0.376 | 0.172 | 0.745 | 0.852 |
| honest:all | Caco-2 | 0.737 | 0.725 | 0.879 | 0.000 | 0.528 | 0.252 | 0.820 | 0.894 |
| rel:all | PC-3 | 0.897 | 0.894 | 0.973 | 1.000 | 0.744 | 0.598 | 0.932 | 0.969 |
| rel:all | Panc1 | 0.604 | 0.509 | 0.927 | 0.000 | 0.210 | 0.114 | 0.763 | 0.895 |
| rel:all | NK cell | 0.783 | 0.768 | 0.910 | 0.000 | 0.546 | 0.594 | 0.827 | 0.904 |
| rel:all | Caco-2 | 0.897 | 0.891 | 0.975 | 1.000 | 0.741 | 0.757 | 0.926 | 0.963 |
| rel:all_noref | PC-3 | 0.779 | 0.757 | 0.891 | 0.000 | 0.595 | 0.318 | 0.842 | 0.918 |
| rel:all_noref | Panc1 | 0.528 | 0.400 | 0.804 | 0.000 | 0.163 | 0.049 | 0.687 | 0.853 |
| rel:all_noref | NK cell | 0.655 | 0.643 | 0.813 | 0.000 | 0.410 | 0.254 | 0.721 | 0.832 |
| rel:all_noref | Caco-2 | 0.756 | 0.745 | 0.896 | 0.000 | 0.570 | 0.361 | 0.824 | 0.894 |

### Median per-window Pearson, 25 kb, 413 Akita test windows

| predictor | PC-3 | Panc1 | NK cell | Caco-2 |
|---|---|---|---|---|
| consensus12 | 0.932 | 0.839 | 0.707 | 0.915 |
| best_cell | 0.897 | 0.811 | 0.790 | 0.899 |
| akita | 0.663 | 0.549 | 0.404 | 0.688 |
| honest:all | 0.817 | 0.709 | 0.734 | 0.759 |
| refin:all | 0.942 | 0.841 | 0.823 | 0.934 |
| refres:all | 0.940 | 0.841 | 0.863 | 0.930 |
| diff:all | 0.944 | 0.841 | 0.858 | 0.939 |
| rel:all | 0.936 | 0.824 | 0.830 | 0.933 |
| rel:ctcf_k27ac | 0.931 | 0.825 | 0.857 | 0.929 |
| rel:ctcf | 0.925 | 0.844 | 0.717 | 0.907 |
| rel:none | 0.926 | 0.843 | 0.683 | 0.913 |
| rel:all_noref | 0.827 | 0.718 | 0.690 | 0.785 |
| rel:ctcf_k27ac_noref | 0.780 | 0.670 | 0.696 | 0.754 |
| rel:ctcf_noref | 0.733 | 0.661 | 0.452 | 0.658 |
| rel:none_noref | 0.573 | 0.605 | 0.400 | 0.525 |
| split-half ceiling | 0.978 | 0.857 | 0.977 | 0.979 |

### Structure measures, 25 kb (median over windows)

| predictor | sample | pearson | scc | insulation_corr | boundary_f1 | loop_enrich_corr | loop_apa_ratio | compartment_corr | eigenvector_abs_corr |
|---|---|---|---|---|---|---|---|---|---|
| consensus12 | PC-3 | 0.932 | 0.933 | 0.972 | 1.000 | 0.917 | 0.648 | 0.942 | 0.975 |
| consensus12 | Panc1 | 0.839 | 0.770 | 0.956 | 1.000 | 0.666 | 0.738 | 0.889 | 0.958 |
| consensus12 | NK cell | 0.707 | 0.661 | 0.808 | 0.000 | 0.690 | 0.722 | 0.747 | 0.858 |
| consensus12 | Caco-2 | 0.915 | 0.915 | 0.965 | 1.000 | 0.911 | 0.795 | 0.923 | 0.964 |
| best_cell | PC-3 | 0.897 | 0.899 | 0.958 | 1.000 | 0.867 | 0.628 | 0.915 | 0.965 |
| best_cell | Panc1 | 0.811 | 0.743 | 0.935 | 0.667 | 0.646 | 0.743 | 0.863 | 0.948 |
| best_cell | NK cell | 0.790 | 0.767 | 0.876 | 0.000 | 0.688 | 0.717 | 0.784 | 0.884 |
| best_cell | Caco-2 | 0.899 | 0.900 | 0.958 | 1.000 | 0.884 | 0.996 | 0.914 | 0.953 |
| akita | PC-3 | 0.663 | 0.771 | 0.877 | 0.667 | 0.715 | 0.334 | 0.706 | 0.784 |
| akita | Panc1 | 0.549 | 0.634 | 0.882 | 0.000 | 0.537 | 0.398 | 0.638 | 0.704 |
| akita | NK cell | 0.404 | 0.507 | 0.640 | 0.000 | 0.484 | 0.389 | 0.605 | 0.713 |
| akita | Caco-2 | 0.688 | 0.766 | 0.902 | 1.000 | 0.747 | 0.414 | 0.721 | 0.783 |
| refin:all | PC-3 | 0.942 | 0.930 | 0.974 | 1.000 | 0.920 | 0.665 | 0.951 | 0.979 |
| refin:all | Panc1 | 0.841 | 0.768 | 0.950 | 1.000 | 0.678 | 0.759 | 0.881 | 0.959 |
| refin:all | NK cell | 0.823 | 0.817 | 0.910 | 0.000 | 0.783 | 0.760 | 0.836 | 0.921 |
| refin:all | Caco-2 | 0.934 | 0.929 | 0.975 | 1.000 | 0.922 | 0.820 | 0.940 | 0.975 |
| honest:all | PC-3 | 0.817 | 0.777 | 0.904 | 1.000 | 0.749 | 0.306 | 0.858 | 0.935 |
| honest:all | Panc1 | 0.709 | 0.561 | 0.830 | 0.000 | 0.486 | 0.306 | 0.794 | 0.907 |
| honest:all | NK cell | 0.734 | 0.705 | 0.844 | 0.000 | 0.618 | 0.336 | 0.765 | 0.858 |
| honest:all | Caco-2 | 0.759 | 0.742 | 0.872 | 0.000 | 0.715 | 0.309 | 0.815 | 0.914 |
| rel:all | PC-3 | 0.936 | 0.928 | 0.974 | 1.000 | 0.917 | 0.666 | 0.952 | 0.982 |
| rel:all | Panc1 | 0.824 | 0.745 | 0.941 | 1.000 | 0.668 | 0.738 | 0.874 | 0.948 |
| rel:all | NK cell | 0.830 | 0.820 | 0.925 | 0.000 | 0.784 | 0.759 | 0.837 | 0.912 |
| rel:all | Caco-2 | 0.933 | 0.924 | 0.973 | 1.000 | 0.920 | 0.825 | 0.940 | 0.972 |
| rel:all_noref | PC-3 | 0.827 | 0.786 | 0.895 | 1.000 | 0.765 | 0.386 | 0.853 | 0.933 |
| rel:all_noref | Panc1 | 0.718 | 0.575 | 0.826 | 0.000 | 0.502 | 0.350 | 0.785 | 0.908 |
| rel:all_noref | NK cell | 0.690 | 0.687 | 0.842 | 0.000 | 0.644 | 0.422 | 0.740 | 0.855 |
| rel:all_noref | Caco-2 | 0.785 | 0.748 | 0.875 | 0.667 | 0.734 | 0.397 | 0.825 | 0.911 |

### Paired contrasts (median difference of per-window Pearson, 95% bootstrap interval over windows)

| sample | res | contrast | windows | difference | share of windows x better |
|---|---|---|---|---|---|
| PC-3 | 10 kb | A: rel:all - consensus12 | 413 | +0.0137 [+0.0093, +0.0225] | 0.65 |
| PC-3 | 10 kb | B: rel:all - rel:all_noref | 413 | +0.1180 [+0.1059, +0.1341] | 0.94 |
| PC-3 | 10 kb | rel:all - refin:all | 413 | +0.0021 [-0.0026, +0.0053] | 0.51 |
| PC-3 | 10 kb | rel:all_noref - honest:all | 413 | +0.0072 [-0.0046, +0.0190] | 0.58 |
| PC-3 | 10 kb | rel:all_noref - consensus12 | 413 | -0.1043 [-0.1197, -0.0903] | 0.14 |
| PC-3 | 10 kb | rel:all - akita | 413 | +0.2238 [+0.1995, +0.2561] | 0.98 |
| PC-3 | 10 kb | rel:all_noref - akita | 413 | +0.1058 [+0.0781, +0.1366] | 0.73 |
| PC-3 | 10 kb | rel:all - best_cell | 413 | +0.0666 [+0.0534, +0.0784] | 0.89 |
| PC-3 | 10 kb | refin:all - consensus12 (section 74 A) | 413 | +0.0116 [+0.0072, +0.0206] | 0.71 |
| PC-3 | 10 kb | refin:all - honest:all (section 74 B) | 413 | +0.1231 [+0.1109, +0.1399] | 0.95 |
| PC-3 | 25 kb | A: rel:all - consensus12 | 413 | +0.0042 [-0.0025, +0.0091] | 0.54 |
| PC-3 | 25 kb | B: rel:all - rel:all_noref | 413 | +0.1085 [+0.0983, +0.1253] | 0.92 |
| PC-3 | 25 kb | rel:all - refin:all | 413 | -0.0057 [-0.0093, -0.0025] | 0.41 |
| PC-3 | 25 kb | rel:all_noref - honest:all | 413 | +0.0107 [-0.0044, +0.0279] | 0.51 |
| PC-3 | 25 kb | rel:all_noref - consensus12 | 413 | -0.1043 [-0.1225, -0.0939] | 0.15 |
| PC-3 | 25 kb | rel:all - akita | 413 | +0.2724 [+0.2315, +0.3061] | 0.98 |
| PC-3 | 25 kb | rel:all_noref - akita | 413 | +0.1639 [+0.1210, +0.2016] | 0.72 |
| PC-3 | 25 kb | rel:all - best_cell | 413 | +0.0384 [+0.0281, +0.0449] | 0.79 |
| PC-3 | 25 kb | refin:all - consensus12 (section 74 A) | 413 | +0.0099 [+0.0040, +0.0146] | 0.62 |
| PC-3 | 25 kb | refin:all - honest:all (section 74 B) | 413 | +0.1250 [+0.1074, +0.1487] | 0.91 |
| Panc1 | 10 kb | A: rel:all - consensus12 | 413 | -0.0090 [-0.0208, +0.0017] | 0.44 |
| Panc1 | 10 kb | B: rel:all - rel:all_noref | 413 | +0.0755 [+0.0488, +0.0933] | 0.82 |
| Panc1 | 10 kb | rel:all - refin:all | 413 | -0.0110 [-0.0202, -0.0012] | 0.36 |
| Panc1 | 10 kb | rel:all_noref - honest:all | 413 | +0.0019 [-0.0144, +0.0198] | 0.53 |
| Panc1 | 10 kb | rel:all_noref - consensus12 | 413 | -0.0844 [-0.1047, -0.0573] | 0.19 |
| Panc1 | 10 kb | rel:all - akita | 413 | +0.1699 [+0.1453, +0.2051] | 0.88 |
| Panc1 | 10 kb | rel:all_noref - akita | 413 | +0.0944 [+0.0667, +0.1415] | 0.63 |
| Panc1 | 10 kb | rel:all - best_cell | 413 | +0.0266 [+0.0098, +0.0429] | 0.71 |
| Panc1 | 10 kb | refin:all - consensus12 (section 74 A) | 413 | +0.0020 [-0.0088, +0.0118] | 0.51 |
| Panc1 | 10 kb | refin:all - honest:all (section 74 B) | 413 | +0.0884 [+0.0637, +0.1053] | 0.84 |
| Panc1 | 25 kb | A: rel:all - consensus12 | 413 | -0.0147 [-0.0217, -0.0009] | 0.39 |
| Panc1 | 25 kb | B: rel:all - rel:all_noref | 413 | +0.1059 [+0.0848, +0.1283] | 0.83 |
| Panc1 | 25 kb | rel:all - refin:all | 413 | -0.0172 [-0.0236, -0.0091] | 0.27 |
| Panc1 | 25 kb | rel:all_noref - honest:all | 413 | +0.0092 [-0.0072, +0.0275] | 0.54 |
| Panc1 | 25 kb | rel:all_noref - consensus12 | 413 | -0.1205 [-0.1420, -0.0964] | 0.18 |
| Panc1 | 25 kb | rel:all - akita | 413 | +0.2755 [+0.2331, +0.3316] | 0.93 |
| Panc1 | 25 kb | rel:all_noref - akita | 413 | +0.1696 [+0.1203, +0.2296] | 0.69 |
| Panc1 | 25 kb | rel:all - best_cell | 413 | +0.0133 [+0.0021, +0.0245] | 0.59 |
| Panc1 | 25 kb | refin:all - consensus12 (section 74 A) | 413 | +0.0025 [-0.0038, +0.0137] | 0.52 |
| Panc1 | 25 kb | refin:all - honest:all (section 74 B) | 413 | +0.1323 [+0.1154, +0.1513] | 0.86 |
| NK cell | 10 kb | A: rel:all - consensus12 | 413 | +0.1257 [+0.1081, +0.1614] | 0.86 |
| NK cell | 10 kb | B: rel:all - rel:all_noref | 413 | +0.1282 [+0.1048, +0.1551] | 0.85 |
| NK cell | 10 kb | rel:all - refin:all | 413 | +0.0174 [+0.0034, +0.0287] | 0.66 |
| NK cell | 10 kb | rel:all_noref - honest:all | 413 | -0.0210 [-0.0415, +0.0043] | 0.48 |
| NK cell | 10 kb | rel:all_noref - consensus12 | 413 | -0.0025 [-0.0291, +0.0402] | 0.53 |
| NK cell | 10 kb | rel:all - akita | 413 | +0.3629 [+0.3275, +0.3944] | 0.93 |
| NK cell | 10 kb | rel:all_noref - akita | 413 | +0.2347 [+0.1903, +0.2734] | 0.77 |
| NK cell | 10 kb | rel:all - best_cell | 413 | +0.0891 [+0.0672, +0.1226] | 0.70 |
| NK cell | 10 kb | refin:all - consensus12 (section 74 A) | 413 | +0.1083 [+0.0899, +0.1448] | 0.84 |
| NK cell | 10 kb | refin:all - honest:all (section 74 B) | 413 | +0.0899 [+0.0680, +0.1234] | 0.72 |
| NK cell | 25 kb | A: rel:all - consensus12 | 413 | +0.1223 [+0.1006, +0.1599] | 0.86 |
| NK cell | 25 kb | B: rel:all - rel:all_noref | 413 | +0.1394 [+0.1061, +0.1683] | 0.79 |
| NK cell | 25 kb | rel:all - refin:all | 413 | +0.0069 [-0.0055, +0.0197] | 0.57 |
| NK cell | 25 kb | rel:all_noref - honest:all | 413 | -0.0436 [-0.0664, -0.0107] | 0.41 |
| NK cell | 25 kb | rel:all_noref - consensus12 | 413 | -0.0171 [-0.0511, +0.0295] | 0.53 |
| NK cell | 25 kb | rel:all - akita | 413 | +0.4255 [+0.3749, +0.4656] | 0.94 |
| NK cell | 25 kb | rel:all_noref - akita | 413 | +0.2861 [+0.2259, +0.3423] | 0.76 |
| NK cell | 25 kb | rel:all - best_cell | 413 | +0.0396 [+0.0162, +0.0763] | 0.60 |
| NK cell | 25 kb | refin:all - consensus12 (section 74 A) | 413 | +0.1154 [+0.0953, +0.1534] | 0.86 |
| NK cell | 25 kb | refin:all - honest:all (section 74 B) | 413 | +0.0889 [+0.0619, +0.1242] | 0.70 |
| Caco-2 | 10 kb | A: rel:all - consensus12 | 413 | +0.0272 [+0.0177, +0.0355] | 0.79 |
| Caco-2 | 10 kb | B: rel:all - rel:all_noref | 413 | +0.1413 [+0.1237, +0.1576] | 0.95 |
| Caco-2 | 10 kb | rel:all - refin:all | 413 | +0.0032 [-0.0026, +0.0066] | 0.62 |
| Caco-2 | 10 kb | rel:all_noref - honest:all | 413 | +0.0193 [+0.0043, +0.0342] | 0.60 |
| Caco-2 | 10 kb | rel:all_noref - consensus12 | 413 | -0.1141 [-0.1334, -0.0961] | 0.18 |
| Caco-2 | 10 kb | rel:all - akita | 413 | +0.2089 [+0.1838, +0.2309] | 0.99 |
| Caco-2 | 10 kb | rel:all_noref - akita | 413 | +0.0677 [+0.0397, +0.0922] | 0.63 |
| Caco-2 | 10 kb | rel:all - best_cell | 413 | +0.0625 [+0.0520, +0.0790] | 0.91 |
| Caco-2 | 10 kb | refin:all - consensus12 (section 74 A) | 413 | +0.0239 [+0.0155, +0.0331] | 0.79 |
| Caco-2 | 10 kb | refin:all - honest:all (section 74 B) | 413 | +0.1573 [+0.1420, +0.1733] | 0.94 |
| Caco-2 | 25 kb | A: rel:all - consensus12 | 413 | +0.0182 [+0.0111, +0.0284] | 0.73 |
| Caco-2 | 25 kb | B: rel:all - rel:all_noref | 413 | +0.1485 [+0.1288, +0.1674] | 0.92 |
| Caco-2 | 25 kb | rel:all - refin:all | 413 | -0.0010 [-0.0040, +0.0048] | 0.50 |
| Caco-2 | 25 kb | rel:all_noref - honest:all | 413 | +0.0254 [+0.0088, +0.0406] | 0.61 |
| Caco-2 | 25 kb | rel:all_noref - consensus12 | 413 | -0.1303 [-0.1483, -0.1096] | 0.17 |
| Caco-2 | 25 kb | rel:all - akita | 413 | +0.2447 [+0.2193, +0.2679] | 0.96 |
| Caco-2 | 25 kb | rel:all_noref - akita | 413 | +0.0962 [+0.0663, +0.1243] | 0.64 |
| Caco-2 | 25 kb | rel:all - best_cell | 413 | +0.0339 [+0.0237, +0.0453] | 0.74 |
| Caco-2 | 25 kb | refin:all - consensus12 (section 74 A) | 413 | +0.0192 [+0.0121, +0.0279] | 0.78 |
| Caco-2 | 25 kb | refin:all - honest:all (section 74 B) | 413 | +0.1749 [+0.1541, +0.1874] | 0.94 |
