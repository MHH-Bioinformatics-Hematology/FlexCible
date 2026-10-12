### Structure scores vs Akita, GM12878, Akita test regions (median over regions; pooled values with region-bootstrap 95% interval)

| predictor | regions | Pearson | SCC | ins r 10 kb | ins r 51 kb | ins r 102 kb | ins r 250 kb | bnd F1 | bnd strength r | MoC | stripe r | compart. r | pooled bnd F1 | hFT bnd recall | loop recall | APA ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Akita (GM12878 head) | 413 | 0.548 | 0.510 | 0.391 | 0.688 | 0.761 | 0.847 | 0.500 | 0.608 | 0.625 | 0.697 | 0.710 | 0.436 [0.411, 0.462] | 0.494 [0.471, 0.519] | 0.314 [0.280, 0.346] | 0.264 [0.243, 0.286] |
| ours, ChIP only | 413 | 0.638 | 0.610 | 0.717 | 0.819 | 0.847 | 0.919 | 0.571 | 0.730 | 0.674 | 0.791 | 0.776 | 0.487 [0.457, 0.518] | 0.548 [0.522, 0.573] | 0.383 [0.345, 0.420] | 0.381 [0.356, 0.408] |
| ours, with consensus | 413 | 0.746 | 0.694 | 0.779 | 0.920 | 0.938 | 0.971 | 0.667 | 0.871 | 0.718 | 0.894 | 0.885 | 0.575 [0.545, 0.609] | 0.647 [0.621, 0.672] | 0.571 [0.537, 0.603] | 0.845 [0.814, 0.878] |
| consensus of 4 measured cells | 413 | 0.611 | 0.559 | 0.474 | 0.777 | 0.844 | 0.916 | 0.598 | 0.724 | 0.656 | 0.759 | 0.797 | 0.533 [0.511, 0.556] | 0.589 [0.564, 0.615] | 0.514 [0.479, 0.549] | 1.025 [0.972, 1.076] |
| HFF measured map | 413 | 0.514 | 0.469 | 0.380 | 0.705 | 0.801 | 0.893 | 0.500 | 0.639 | 0.612 | 0.669 | 0.716 | 0.488 [0.469, 0.509] | 0.522 [0.497, 0.548] | 0.418 [0.384, 0.451] | 1.552 [1.453, 1.654] |
| Hi-cGAN (submitted) | 408 | 0.236 | 0.207 | 0.209 | 0.376 | 0.390 | 0.567 | 0.222 | 0.227 | 0.516 | 0.309 | 0.546 | 0.243 [0.225, 0.263] | 0.321 [0.298, 0.344] | 0.137 [0.111, 0.162] | 0.179 [0.146, 0.215] |

### Paired against Akita on the same regions and pixels (difference ours minus Akita, median over regions [region-bootstrap 95% interval]; regions ours is ahead / regions compared)

| metric | ours, ChIP only: difference | ahead (fraction [95%]) | ours, with consensus: difference | ahead (fraction [95%]) |
|---|---|---|---|---|
| Pearson | 0.040 [0.029, 0.050] | 274/413 (0.66 [0.61, 0.71]) | 0.136 [0.124, 0.149] | 401/413 (0.97 [0.95, 0.99]) |
| SCC | 0.061 [0.051, 0.074] | 309/413 (0.75 [0.71, 0.79]) | 0.140 [0.128, 0.153] | 405/413 (0.98 [0.97, 0.99]) |
| ins r 10 kb | 0.272 [0.254, 0.281] | 397/413 (0.96 [0.94, 0.98]) | 0.324 [0.314, 0.343] | 408/413 (0.99 [0.98, 1.00]) |
| ins r 25 kb | 0.176 [0.159, 0.190] | 373/413 (0.90 [0.87, 0.93]) | 0.255 [0.234, 0.268] | 409/413 (0.99 [0.98, 1.00]) |
| ins r 51 kb | 0.096 [0.075, 0.116] | 323/413 (0.78 [0.74, 0.82]) | 0.190 [0.175, 0.213] | 398/413 (0.96 [0.94, 0.98]) |
| ins r 102 kb | 0.058 [0.040, 0.080] | 279/413 (0.68 [0.63, 0.72]) | 0.140 [0.121, 0.168] | 386/413 (0.93 [0.91, 0.96]) |
| ins r 250 kb | 0.026 [0.013, 0.037] | 248/413 (0.60 [0.55, 0.65]) | 0.072 [0.055, 0.097] | 331/413 (0.80 [0.76, 0.84]) |
| bnd F1 | 0.048 [0.000, 0.083] | 223/413 (0.61 [0.56, 0.66]) | 0.127 [0.100, 0.159] | 281/413 (0.78 [0.74, 0.83]) |
| bnd F1 strict (4 kb) | 0.029 [0.000, 0.067] | 214/413 (0.60 [0.54, 0.65]) | 0.167 [0.125, 0.197] | 280/413 (0.77 [0.72, 0.81]) |
| bnd recall | 0.000 [0.000, 0.000] | 184/413 (0.61 [0.55, 0.67]) | 0.125 [0.000, 0.167] | 225/413 (0.82 [0.77, 0.86]) |
| bnd precision | 0.000 [0.000, 0.067] | 199/410 (0.59 [0.54, 0.64]) | 0.150 [0.083, 0.167] | 245/409 (0.76 [0.72, 0.81]) |
| bnd strength r | 0.075 [0.054, 0.095] | 266/413 (0.64 [0.60, 0.69]) | 0.181 [0.157, 0.199] | 365/413 (0.88 [0.85, 0.91]) |
| MoC | 0.026 [0.012, 0.035] | 249/413 (0.60 [0.55, 0.65]) | 0.063 [0.048, 0.081] | 307/413 (0.75 [0.70, 0.79]) |
| stripe r | 0.077 [0.057, 0.101] | 296/413 (0.72 [0.67, 0.76]) | 0.173 [0.146, 0.188] | 389/413 (0.94 [0.92, 0.96]) |
| compart. r | 0.029 [0.016, 0.042] | 263/413 (0.64 [0.59, 0.68]) | 0.099 [0.081, 0.111] | 376/413 (0.91 [0.88, 0.94]) |
| eigenvector |r| | 0.017 [0.006, 0.030] | 238/413 (0.58 [0.53, 0.62]) | 0.062 [0.045, 0.075] | 327/413 (0.79 [0.75, 0.83]) |
| loop enrichment r | 0.024 [0.021, 0.026] | 360/413 (0.87 [0.84, 0.90]) | 0.039 [0.036, 0.042] | 384/413 (0.93 [0.90, 0.95]) |
| hFT bnd recall (per region) | 0.000 | 141/404 (0.61) | 0.000 | 196/404 (0.84) |
| loop recall (per region) | 0.000 | 89/287 (0.62) | 0.000 | 140/287 (0.93) |
| pooled bnd F1 | 0.051 [0.028, 0.075] | pooled | 0.140 [0.118, 0.162] | pooled |
| pooled hFT bnd recall | 0.054 [0.029, 0.080] | pooled | 0.153 [0.129, 0.176] | pooled |
| pooled loop recall | 0.069 [0.025, 0.115] | pooled | 0.257 [0.221, 0.296] | pooled |
| APA ratio | 0.117 [0.086, 0.148] | pooled | 0.581 [0.551, 0.612] | pooled |

### Whole held-out chromosomes at 10 kb (mean over chromosomes; hicFindTADs boundaries, 1 bin tolerance)

| block | predictor | chr | Pearson | SCC | ins r | bnd recall | bnd prec. | bnd F1 | TAD fraction | TAD score MSE | loop recall | APA ratio | compart. r | chromosomes ahead of the competitor: ins r / bnd F1 / loop recall / compart. r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C.Origami imr90 | ours, consensus as residual (refres) | chr2,chr3 | 0.928 | 0.931 | 0.943 | 0.658 | 0.675 | 0.666 | 0.974 | 0.078 | 0.364 | 0.943 | 0.964 | 2/2 / 2/2 / 2/2 / 0/2 |
| C.Origami imr90 | ours, consensus as input (refin) | chr2,chr3 | 0.926 | 0.927 | 0.924 | 0.654 | 0.681 | 0.667 | 0.960 | 0.121 | 0.355 | 0.830 | 0.960 | 2/2 / 2/2 / 2/2 / 0/2 |
| C.Origami imr90 | ours, ChIP only (honest) | chr2,chr3 | 0.808 | 0.812 | 0.823 | 0.497 | 0.542 | 0.518 | 0.917 | 0.256 | 0.192 | 0.409 | 0.885 | 0/2 / 2/2 / 2/2 / 0/2 |
| C.Origami imr90 | consensus of other cells | chr2,chr3 | 0.866 | 0.862 | 0.796 | 0.640 | 0.632 | 0.636 | 1.014 | 0.316 | 0.383 | 0.890 | 0.893 |  |
| C.Origami imr90 | corigami_imr90_8kb (competitor) | chr2,chr3 | 0.920 | 0.913 | 0.866 | 0.474 | 0.526 | 0.498 | 0.900 | 0.173 | 0.162 | 0.729 | 0.972 |  |
| C.Origami imr90 | measured map, balanced (ceiling) | chr2,chr3 |  |  |  | 0.870 | 0.845 | 0.857 | 1.029 | 0.021 |  |  |  |  |
| Epiphany gm12878 | ours, consensus as residual (refres) | chr3,chr11,chr17 | 0.919 | 0.915 | 0.892 | 0.676 | 0.690 | 0.683 | 0.980 | 0.158 | 0.474 | 0.949 | 0.956 | 3/3 / 3/3 / 3/3 / 3/3 |
| Epiphany gm12878 | ours, consensus as input (refin) | chr3,chr11,chr17 | 0.914 | 0.908 | 0.875 | 0.655 | 0.670 | 0.662 | 0.978 | 0.179 | 0.495 | 0.706 | 0.951 | 3/3 / 3/3 / 3/3 / 3/3 |
| Epiphany gm12878 | ours, ChIP only (honest) | chr3,chr11,chr17 | 0.760 | 0.751 | 0.715 | 0.497 | 0.508 | 0.502 | 0.979 | 0.343 | 0.220 | 0.340 | 0.844 | 3/3 / 3/3 / 3/3 / 2/3 |
| Epiphany gm12878 | consensus of other cells | chr3,chr11,chr17 | 0.826 | 0.818 | 0.778 | 0.660 | 0.628 | 0.643 | 1.050 | 0.334 | 0.452 | 0.895 | 0.875 |  |
| Epiphany gm12878 | epiphany_gm12878_10kb (competitor) | chr3,chr11,chr17 | 0.512 | 0.501 | 0.318 | 0.208 | 0.246 | 0.226 | 0.847 | 0.793 | 0.052 | 0.265 | 0.776 |  |
| Epiphany gm12878 | measured map, balanced (ceiling) | chr3,chr11,chr17 |  |  |  | 0.858 | 0.839 | 0.849 | 1.022 | 0.023 |  |  |  |  |
| Hi-cGAN gm12878 | ours, consensus as residual (refres) | chr2,chr14 | 0.924 | 0.921 | 0.890 | 0.640 | 0.640 | 0.640 | 1.000 | 0.119 | 0.407 | 0.945 | 0.960 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN gm12878 | ours, consensus as input (refin) | chr2,chr14 | 0.921 | 0.917 | 0.875 | 0.620 | 0.625 | 0.622 | 0.994 | 0.134 | 0.400 | 0.733 | 0.953 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN gm12878 | ours, ChIP only (honest) | chr2,chr14 | 0.797 | 0.787 | 0.726 | 0.497 | 0.489 | 0.492 | 1.018 | 0.261 | 0.164 | 0.295 | 0.876 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN gm12878 | consensus of other cells | chr2,chr14 | 0.841 | 0.833 | 0.757 | 0.611 | 0.571 | 0.590 | 1.071 | 0.284 | 0.383 | 0.899 | 0.882 |  |
| Hi-cGAN gm12878 | hicgan_gm12878_10kb (competitor) | chr2,chr14 | 0.328 | 0.330 | 0.260 | 0.113 | 0.141 | 0.125 | 0.803 | 0.975 | 0.031 | 0.135 | 0.700 |  |
| Hi-cGAN gm12878 | measured map, balanced (ceiling) | chr2,chr14 |  |  |  | 0.859 | 0.826 | 0.842 | 1.041 | 0.020 |  |  |  |  |
| Hi-cGAN k562 | ours, consensus as residual (refres) | chr2,chr14 | 0.849 | 0.850 | 0.830 | 0.571 | 0.616 | 0.593 | 0.927 | 0.179 | 0.382 | 0.794 | 0.940 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN k562 | ours, consensus as input (refin) | chr2,chr14 | 0.844 | 0.844 | 0.825 | 0.559 | 0.627 | 0.591 | 0.891 | 0.193 | 0.364 | 0.543 | 0.937 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN k562 | ours, ChIP only (honest) | chr2,chr14 | 0.731 | 0.728 | 0.732 | 0.438 | 0.499 | 0.466 | 0.878 | 0.279 | 0.160 | 0.207 | 0.872 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN k562 | consensus of other cells | chr2,chr14 | 0.792 | 0.794 | 0.798 | 0.577 | 0.607 | 0.591 | 0.951 | 0.203 | 0.360 | 0.759 | 0.890 |  |
| Hi-cGAN k562 | hicgan_gm12878_on_k562_10kb (competitor) | chr2,chr14 | 0.125 | 0.141 | 0.339 | 0.108 | 0.156 | 0.127 | 0.678 | 0.927 | 0.008 | 0.088 | 0.519 |  |
| Hi-cGAN k562 | measured map, balanced (ceiling) | chr2,chr14 |  |  |  | 0.833 | 0.823 | 0.828 | 1.012 | 0.056 |  |  |  |  |

### Whole held-out chromosomes at 25 kb (mean over chromosomes; hicFindTADs boundaries, 1 bin tolerance)

| block | predictor | chr | Pearson | SCC | ins r | bnd recall | bnd prec. | bnd F1 | TAD fraction | TAD score MSE | loop recall | APA ratio | compart. r | chromosomes ahead of the competitor: ins r / bnd F1 / loop recall / compart. r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C.Origami imr90 | ours, consensus as residual (refres) | chr2,chr3 | 0.940 | 0.942 | 0.946 | 0.807 | 0.824 | 0.816 | 0.980 | 0.072 | 0.335 | 0.915 | 0.960 | 2/2 / 2/2 / 2/2 / 0/2 |
| C.Origami imr90 | ours, consensus as input (refin) | chr2,chr3 | 0.940 | 0.941 | 0.943 | 0.817 | 0.840 | 0.828 | 0.973 | 0.074 | 0.349 | 1.038 | 0.958 | 2/2 / 2/2 / 2/2 / 0/2 |
| C.Origami imr90 | ours, ChIP only (honest) | chr2,chr3 | 0.804 | 0.800 | 0.775 | 0.647 | 0.639 | 0.643 | 1.011 | 0.284 | 0.206 | 0.541 | 0.862 | 0/2 / 0/2 / 0/2 / 0/2 |
| C.Origami imr90 | consensus of other cells | chr2,chr3 | 0.877 | 0.872 | 0.817 | 0.788 | 0.766 | 0.777 | 1.029 | 0.256 | 0.374 | 0.855 | 0.885 |  |
| C.Origami imr90 | corigami_imr90_8kb (competitor) | chr2,chr3 | 0.942 | 0.933 | 0.898 | 0.752 | 0.727 | 0.739 | 1.035 | 0.108 | 0.313 | 2.930 | 0.980 |  |
| C.Origami imr90 | measured map, balanced (ceiling) | chr2,chr3 |  |  |  | 0.956 | 0.872 | 0.912 | 1.096 | 0.028 |  |  |  |  |
| Epiphany gm12878 | ours, consensus as residual (refres) | chr3,chr11,chr17 | 0.922 | 0.916 | 0.902 | 0.837 | 0.834 | 0.836 | 1.004 | 0.112 | 0.451 | 0.803 | 0.950 | 3/3 / 3/3 / 3/3 / 3/3 |
| Epiphany gm12878 | ours, consensus as input (refin) | chr3,chr11,chr17 | 0.917 | 0.913 | 0.900 | 0.827 | 0.826 | 0.827 | 1.002 | 0.117 | 0.453 | 0.786 | 0.948 | 3/3 / 3/3 / 3/3 / 3/3 |
| Epiphany gm12878 | ours, ChIP only (honest) | chr3,chr11,chr17 | 0.754 | 0.739 | 0.668 | 0.608 | 0.591 | 0.599 | 1.028 | 0.367 | 0.241 | 0.353 | 0.846 | 3/3 / 3/3 / 2/3 / 2/3 |
| Epiphany gm12878 | consensus of other cells | chr3,chr11,chr17 | 0.841 | 0.832 | 0.804 | 0.805 | 0.781 | 0.793 | 1.031 | 0.246 | 0.456 | 0.765 | 0.878 |  |
| Epiphany gm12878 | epiphany_gm12878_10kb (competitor) | chr3,chr11,chr17 | 0.518 | 0.501 | 0.377 | 0.437 | 0.461 | 0.449 | 0.948 | 0.610 | 0.254 | 1.513 | 0.775 |  |
| Epiphany gm12878 | measured map, balanced (ceiling) | chr3,chr11,chr17 |  |  |  | 0.953 | 0.873 | 0.911 | 1.092 | 0.033 |  |  |  |  |
| Hi-cGAN gm12878 | ours, consensus as residual (refres) | chr2,chr14 | 0.932 | 0.927 | 0.893 | 0.810 | 0.798 | 0.804 | 1.016 | 0.105 | 0.373 | 0.856 | 0.957 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN gm12878 | ours, consensus as input (refin) | chr2,chr14 | 0.934 | 0.929 | 0.901 | 0.808 | 0.791 | 0.799 | 1.022 | 0.096 | 0.405 | 0.810 | 0.959 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN gm12878 | ours, ChIP only (honest) | chr2,chr14 | 0.796 | 0.778 | 0.698 | 0.616 | 0.595 | 0.605 | 1.035 | 0.280 | 0.200 | 0.323 | 0.883 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN gm12878 | consensus of other cells | chr2,chr14 | 0.860 | 0.848 | 0.784 | 0.777 | 0.743 | 0.759 | 1.047 | 0.231 | 0.389 | 0.841 | 0.897 |  |
| Hi-cGAN gm12878 | hicgan_gm12878_25kb (competitor) | chr2,chr14 | 0.221 | 0.225 | 0.237 | 0.218 | 0.285 | 0.247 | 0.761 | 0.935 | 0.048 | 0.051 | 0.644 |  |
| Hi-cGAN gm12878 | measured map, balanced (ceiling) | chr2,chr14 |  |  |  | 0.960 | 0.856 | 0.905 | 1.122 | 0.030 |  |  |  |  |
| Hi-cGAN k562 | ours, consensus as residual (refres) | chr2,chr14 | 0.901 | 0.894 | 0.853 | 0.751 | 0.800 | 0.775 | 0.939 | 0.148 | 0.352 | 0.862 | 0.938 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN k562 | ours, consensus as input (refin) | chr2,chr14 | 0.904 | 0.896 | 0.843 | 0.727 | 0.805 | 0.763 | 0.903 | 0.156 | 0.356 | 0.851 | 0.941 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN k562 | ours, ChIP only (honest) | chr2,chr14 | 0.765 | 0.743 | 0.657 | 0.550 | 0.604 | 0.575 | 0.911 | 0.353 | 0.141 | 0.193 | 0.854 | 2/2 / 2/2 / 2/2 / 2/2 |
| Hi-cGAN k562 | consensus of other cells | chr2,chr14 | 0.853 | 0.849 | 0.816 | 0.718 | 0.757 | 0.737 | 0.948 | 0.189 | 0.350 | 0.840 | 0.895 |  |
| Hi-cGAN k562 | hicgan_gm12878_on_k562_25kb (competitor) | chr2,chr14 | 0.143 | 0.147 | 0.218 | 0.213 | 0.317 | 0.254 | 0.672 | 0.986 | 0.050 | 0.156 | 0.656 |  |
| Hi-cGAN k562 | measured map, balanced (ceiling) | chr2,chr14 |  |  |  | 0.953 | 0.873 | 0.911 | 1.092 | 0.033 |  |  |  |  |

### AML patients, 10 kb object windows of 8 donors (median over the 8 donors of the per-donor window median; identical pixels in every window)

| predictor | windows | Pearson | SCC | ins r 50 kb | ins r 250 kb | bnd F1 | bnd strength r | MoC | stripe r | compart. r |
|---|---|---|---|---|---|---|---|---|---|---|
| ours, 12 cells, residual on consensus12 (no AML) | 13064 | 0.761 | 0.688 | 0.735 | 0.869 | 0.500 | 0.628 | 0.651 | 0.724 | 0.821 |
| ours, 12 cells, reference input (no AML) | 13064 | 0.762 | 0.690 | 0.741 | 0.874 | 0.500 | 0.629 | 0.649 | 0.726 | 0.823 |
| consensus12 (12 cell lines, measured) | 13064 | 0.746 | 0.670 | 0.738 | 0.865 | 0.500 | 0.615 | 0.653 | 0.718 | 0.810 |
| Akita (GM12878 head) | 13064 | 0.591 | 0.584 | 0.641 | 0.776 | 0.400 | 0.509 | 0.621 | 0.635 | 0.653 |
| C.Origami (CTCF, ATAC quantile-matched) | 13064 | 0.188 | 0.150 | 0.190 | 0.184 | 0.222 | 0.124 | 0.541 | 0.144 | 0.575 |
| Epiphany (H3K4me3 = mean) | 13064 | 0.248 | 0.235 | 0.342 | 0.366 | 0.000 | 0.102 | 0.468 | 0.385 | 0.578 |
| Epiphany (H3K4me3 = 0) | 13064 | 0.233 | 0.220 | 0.357 | 0.332 | 0.000 | 0.111 | 0.485 | 0.375 | 0.577 |

### AML patients, our AML-inclusive and no-AML models against the measured patient map, 413 held-out regions, 10 donors (folds 10 to 12; median over donors [min, max])

| kb | model | marks | Pearson | SCC | insulation r | boundary F1 (structure.py) | loop enrichment r | compart. r |
|---|---|---|---|---|---|---|---|---|
| 10 | aml_honest | marks | 0.494 | 0.456 | 0.675 [0.582, 0.767] | 0.000 [0.000, 0.000] | 0.176 | 0.639 |
| 10 | aml_refin | marks | 0.813 | 0.804 | 0.961 [0.930, 0.973] | 0.667 [0.000, 1.000] | 0.429 | 0.879 |
| 10 | aml_refin@hspc | marks | 0.760 | 0.753 | 0.951 [0.887, 0.963] | 0.000 [0.000, 1.000] | 0.376 | 0.832 |
| 10 | base_honest | marks | 0.429 | 0.384 | 0.609 [0.513, 0.733] | 0.000 [0.000, 0.000] | 0.175 | 0.645 |
| 10 | base_refin@aml | marks | 0.795 | 0.786 | 0.954 [0.927, 0.971] | 0.583 [0.000, 1.000] | 0.415 | 0.864 |
| 10 | base_refin@c12 | marks | 0.710 | 0.684 | 0.901 [0.844, 0.948] | 0.000 [0.000, 1.000] | 0.356 | 0.786 |
| 25 | aml_honest | marks | 0.595 | 0.538 | 0.667 [0.570, 0.737] | 0.000 [0.000, 0.000] | 0.447 | 0.692 |
| 25 | aml_refin | marks | 0.899 | 0.896 | 0.956 [0.935, 0.971] | 1.000 [1.000, 1.000] | 0.803 | 0.919 |
| 25 | aml_refin@hspc | marks | 0.870 | 0.875 | 0.948 [0.917, 0.960] | 1.000 [1.000, 1.000] | 0.765 | 0.894 |
| 25 | base_honest | marks | 0.533 | 0.488 | 0.604 [0.545, 0.714] | 0.000 [0.000, 0.000] | 0.414 | 0.698 |
| 25 | base_refin@aml | marks | 0.885 | 0.880 | 0.950 [0.933, 0.965] | 1.000 [1.000, 1.000] | 0.789 | 0.909 |
| 25 | base_refin@c12 | marks | 0.817 | 0.801 | 0.906 [0.848, 0.934] | 1.000 [0.667, 1.000] | 0.730 | 0.847 |

## Definitions

Akita table (GM12878, 413 Akita test regions, 2048 bp bins, 448-bin crop = 917 kb, Akita's target space of clipped, smoothed log O/E). One mask per region: upper triangle, at least 2 bins off the diagonal, finite in the target and in every method except Hi-cGAN. Hi-cGAN has no prediction (or a partial one) in 5 regions and is scored in 408.
- Pearson, SCC: Pearson over the mask pixels; HiCRep stratum-adjusted correlation.
- ins r w: Pearson over bins between the insulation profiles of prediction and target; insulation = mean log O/E in the square [i-w, i) x (i, i+w]; w = 5, 12, 25, 50, 122 bins = 10, 25, 51, 102, 250 kb.
- bnd F1, recall, precision: boundaries are local minima of the 51 kb insulation profile whose prominence over the maxima of the 51 kb flanks is at least 0.5 SD of that profile (each map its own SD); minima within the tolerance merged. Reference = boundaries of the measured target on the same pixels. Tolerance 5 bins (10 kb); strict 2 bins (4 kb). Per-region F1 (median) and pooled micro-averaged F1 over all regions.
- bnd strength r: Pearson over bins between the boundary-strength profiles (prominence of the insulation minimum over its flanks) of prediction and target.
- MoC: measure of concordance (Zufferey et al. 2018) between the domain partitions cut at the boundaries above.
- hFT bnd recall: pooled recall of the hicFindTADs boundaries of the measured count map (10 kb) by the K strongest insulation minima of the map, K = number of reference boundaries in the region, tolerance 5 bins; K-matched, so recall equals precision.
- loop recall, APA ratio: hicDetectLoops loops of the measured count map as reference; K strongest donut enrichments of the map (K-matched); APA ratio = (APA of prediction - 1) / (APA of target - 1) at the reference anchors. Pooled over regions with a reference loop.
- stripe r: Pearson over bins between the anchor profiles (mean log O/E of the pixels 20 to 200 kb from the bin) of prediction and target. A proxy; there is no stripe caller in the benchmark.
- compart. r, eigenvector |r|, loop enrichment r: hicmultiome/structure.py (correlation of the within-window correlation matrices; first eigenvector; donut enrichment). Within 917 kb these are local measures: the compartment values are a checkerboard proxy, the 2 kb donut (4 to 10 kb) is smaller than a loop.
- Paired columns: ours minus Akita per region; interval = 2.5 and 97.5 percentiles of 2000 region bootstrap resamples; ahead = regions with ours strictly higher; ties excluded from the fraction.
Chromosome table: values of results/hic_measures_chrho (scripts/hic_measures.py), mean over the held-out chromosomes listed, 10 or 25 kb, own band per block (C.Origami 2 Mb at 25 kb and 1.92 Mb at 10 kb, Epiphany 990 kb, Hi-cGAN 2 Mb). Boundaries are hicFindTADs boundaries of the prediction against those of the measured map (1 bin tolerance), the insulation square is 50 kb (10 kb) or 75 kb (25 kb), loop recall is K-matched against hicDetectLoops of the measured map. These definitions differ from the Akita table (Akita has no whole-chromosome prediction) and the two tables are not comparable in value.
AML windows: 10 kb, 192 bins, windows centred on AML-specific loops and stripes of 8 donors (C.Origami patients), saved predictions of scripts/aml_competitors_loops.py, measured patient map as target; same definitions as the Akita table at 10 kb (insulation squares 50, 100 and 250 kb; boundary tolerance 1 bin; no external reference calls).

## Caveats

- Akita is sequence only, one head per training cell type (GM12878 head here); it sees no cell-specific inputs and is restricted to its five cell types. Our ChIP-only model receives the cell's own ChIP-seq tracks; our with-consensus model additionally receives the Hi-C of other cells (consensus of four measured cells) as input or residual base. The consensus and the HFF map are measured maps of other cells, not predictions.
- Region windows are Akita's test regions, held out of all our trainings (section 18). Hi-cGAN values are the predictions of the submitted model rasterised into Akita's space (akita_testset_rasters.npz), as in section 18.
- Pooled recall and APA are over regions with a reference object; K-matched recall equals precision.
- Chromosome blocks contain 2 or 3 chromosomes: ahead counts are descriptive, there is no interval. Held-out chromosomes per competitor are those of benchmark_competitors.COMPETITORS (C.Origami chr2, chr3; Epiphany chr3, 11, 17; Hi-cGAN even chromosomes), the models of ours are the chromosome-held-out drop_chrho models of section 37.
- AML windows are selected around AML-specific objects, not random windows; there are 8 donors, and Akita is the same prediction for every donor.
