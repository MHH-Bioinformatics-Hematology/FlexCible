# Akita comparison, 413 test regions, GM12878 (median over regions; pooled values with 95% bootstrap interval)

| predictor | regions | pearson | scc | ins_r_10k | ins_r_50k | ins_r_250k | bnd_f1 | bnd_strength_r | moc | stripe_r | compartment_corr | pooled bnd F1 | hFT bnd recall | loop recall | APA ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Akita (GM12878 head) | 413 | 0.548 | 0.510 | 0.391 | 0.688 | 0.847 | 0.500 | 0.608 | 0.625 | 0.697 | 0.710 | 0.436 [0.411, 0.462] | 0.494 [0.471, 0.519] | 0.314 [0.280, 0.346] | 0.264 [0.243, 0.286] |
| Hi-cGAN (submitted) | 408 | 0.236 | 0.207 | 0.209 | 0.376 | 0.567 | 0.222 | 0.227 | 0.516 | 0.309 | 0.546 | 0.243 [0.225, 0.263] | 0.321 [0.298, 0.344] | 0.137 [0.111, 0.162] | 0.179 [0.146, 0.215] |
| ours, ChIP only | 413 | 0.638 | 0.610 | 0.717 | 0.819 | 0.919 | 0.571 | 0.730 | 0.674 | 0.791 | 0.776 | 0.487 [0.457, 0.518] | 0.548 [0.522, 0.573] | 0.383 [0.345, 0.420] | 0.381 [0.356, 0.408] |
| ours, with consensus | 413 | 0.746 | 0.694 | 0.779 | 0.920 | 0.971 | 0.667 | 0.871 | 0.718 | 0.894 | 0.885 | 0.575 [0.545, 0.609] | 0.647 [0.621, 0.672] | 0.571 [0.537, 0.603] | 0.845 [0.814, 0.878] |
| release, marks + Meth only | 413 | 0.595 | 0.598 | 0.713 | 0.820 | 0.903 | 0.571 | 0.754 | 0.676 | 0.764 | 0.729 | 0.504 [0.476, 0.536] | 0.550 [0.525, 0.576] | 0.432 [0.399, 0.466] | 0.624 [0.584, 0.670] |
| release, 13 marks only | 413 | 0.578 | 0.587 | 0.702 | 0.809 | 0.900 | 0.571 | 0.742 | 0.676 | 0.755 | 0.717 | 0.493 [0.465, 0.525] | 0.551 [0.524, 0.578] | 0.428 [0.394, 0.461] | 0.610 [0.572, 0.655] |
| release, with consensus (+ Meth) | 413 | 0.676 | 0.673 | 0.747 | 0.912 | 0.965 | 0.667 | 0.868 | 0.715 | 0.869 | 0.795 | 0.569 [0.541, 0.601] | 0.625 [0.602, 0.648] | 0.530 [0.494, 0.565] | 0.938 [0.890, 0.991] |
| release, with consensus, 13 marks | 413 | 0.674 | 0.672 | 0.735 | 0.907 | 0.963 | 0.667 | 0.864 | 0.704 | 0.869 | 0.784 | 0.572 [0.545, 0.603] | 0.624 [0.601, 0.648] | 0.530 [0.497, 0.564] | 0.922 [0.875, 0.973] |
| control: dedicated 2 kb model (proper targets), with consensus | 413 | 0.670 | 0.684 | 0.779 | 0.919 | 0.968 | 0.667 | 0.883 | 0.715 | 0.862 | 0.802 | 0.606 [0.581, 0.633] | 0.623 [0.596, 0.647] | 0.561 [0.526, 0.595] | 0.890 [0.848, 0.934] |
| control: dedicated 2 kb model (proper targets), ChIP only | 413 | 0.603 | 0.596 | 0.706 | 0.826 | 0.915 | 0.600 | 0.751 | 0.681 | 0.783 | 0.749 | 0.509 [0.480, 0.539] | 0.561 [0.535, 0.587] | 0.414 [0.380, 0.449] | 0.615 [0.578, 0.656] |
| consensus of 4 measured cells | 413 | 0.611 | 0.559 | 0.474 | 0.777 | 0.916 | 0.598 | 0.724 | 0.656 | 0.759 | 0.797 | 0.533 [0.511, 0.556] | 0.589 [0.564, 0.615] | 0.514 [0.479, 0.549] | 1.025 [0.972, 1.076] |
| HFF measured map | 413 | 0.514 | 0.469 | 0.380 | 0.705 | 0.893 | 0.500 | 0.639 | 0.612 | 0.669 | 0.716 | 0.488 [0.469, 0.509] | 0.522 [0.497, 0.548] | 0.418 [0.384, 0.451] | 1.552 [1.453, 1.654] |

## Paired against Akita (ours minus Akita, median over regions [95% interval]; regions ahead)

| predictor | metric | difference | regions ahead of n |
|---|---|---|---|
| ours, ChIP only | pearson | 0.040 [0.029, 0.050] | 274 of 413 |
| ours, ChIP only | scc | 0.061 [0.051, 0.074] | 309 of 413 |
| ours, ChIP only | ins_r_10k | 0.272 [0.254, 0.281] | 397 of 413 |
| ours, ChIP only | ins_r_25k | 0.176 [0.159, 0.190] | 373 of 413 |
| ours, ChIP only | ins_r_50k | 0.096 [0.075, 0.116] | 323 of 413 |
| ours, ChIP only | ins_r_100k | 0.058 [0.040, 0.080] | 279 of 413 |
| ours, ChIP only | ins_r_250k | 0.026 [0.013, 0.037] | 248 of 413 |
| ours, ChIP only | bnd_f1 | 0.048 [0.000, 0.083] | 223 of 413 |
| ours, ChIP only | bnd_f1_strict | 0.029 [0.000, 0.067] | 214 of 413 |
| ours, ChIP only | bnd_recall | 0.000 [0.000, 0.000] | 184 of 413 |
| ours, ChIP only | bnd_precision | 0.000 [0.000, 0.067] | 199 of 410 |
| ours, ChIP only | bnd_strength_r | 0.075 [0.054, 0.095] | 266 of 413 |
| ours, ChIP only | moc | 0.026 [0.012, 0.035] | 249 of 413 |
| ours, ChIP only | stripe_r | 0.077 [0.057, 0.101] | 296 of 413 |
| ours, ChIP only | loop_enrich_corr | 0.024 [0.021, 0.026] | 360 of 413 |
| ours, ChIP only | compartment_corr | 0.029 [0.016, 0.042] | 263 of 413 |
| ours, ChIP only | eigenvector_abs_corr | 0.017 [0.006, 0.030] | 238 of 413 |
| ours, ChIP only | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| ours, ChIP only | hft_bnd_recall_per_region | 0.000 [nan, nan] | 141 of 404 |
| ours, ChIP only | loop_recall_per_region | 0.000 [nan, nan] | 89 of 287 |
| ours, ChIP only | bnd_f1_pooled (pooled) | 0.051 [0.028, 0.075] | |
| ours, ChIP only | bnd_recall_pooled (pooled) | 0.038 [0.017, 0.061] | |
| ours, ChIP only | bnd_precision_pooled (pooled) | 0.075 [0.044, 0.105] | |
| ours, ChIP only | hft_bnd_recall (pooled) | 0.054 [0.029, 0.080] | |
| ours, ChIP only | loop_recall (pooled) | 0.069 [0.025, 0.115] | |
| ours, ChIP only | apa_ratio (pooled) | 0.117 [0.086, 0.148] | |
| ours, with consensus | pearson | 0.136 [0.124, 0.149] | 401 of 413 |
| ours, with consensus | scc | 0.140 [0.128, 0.154] | 405 of 413 |
| ours, with consensus | ins_r_10k | 0.324 [0.315, 0.343] | 408 of 413 |
| ours, with consensus | ins_r_25k | 0.255 [0.235, 0.267] | 409 of 413 |
| ours, with consensus | ins_r_50k | 0.190 [0.175, 0.213] | 398 of 413 |
| ours, with consensus | ins_r_100k | 0.140 [0.121, 0.165] | 386 of 413 |
| ours, with consensus | ins_r_250k | 0.072 [0.055, 0.094] | 331 of 413 |
| ours, with consensus | bnd_f1 | 0.127 [0.100, 0.159] | 281 of 413 |
| ours, with consensus | bnd_f1_strict | 0.167 [0.123, 0.197] | 280 of 413 |
| ours, with consensus | bnd_recall | 0.125 [0.000, 0.167] | 225 of 413 |
| ours, with consensus | bnd_precision | 0.150 [0.083, 0.167] | 245 of 409 |
| ours, with consensus | bnd_strength_r | 0.181 [0.157, 0.202] | 365 of 413 |
| ours, with consensus | moc | 0.063 [0.048, 0.081] | 307 of 413 |
| ours, with consensus | stripe_r | 0.173 [0.148, 0.188] | 389 of 413 |
| ours, with consensus | loop_enrich_corr | 0.039 [0.036, 0.042] | 384 of 413 |
| ours, with consensus | compartment_corr | 0.099 [0.081, 0.111] | 376 of 413 |
| ours, with consensus | eigenvector_abs_corr | 0.062 [0.045, 0.075] | 327 of 413 |
| ours, with consensus | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| ours, with consensus | hft_bnd_recall_per_region | 0.000 [nan, nan] | 196 of 404 |
| ours, with consensus | loop_recall_per_region | 0.000 [nan, nan] | 140 of 287 |
| ours, with consensus | bnd_f1_pooled (pooled) | 0.140 [0.118, 0.162] | |
| ours, with consensus | bnd_recall_pooled (pooled) | 0.122 [0.101, 0.145] | |
| ours, with consensus | bnd_precision_pooled (pooled) | 0.158 [0.132, 0.184] | |
| ours, with consensus | hft_bnd_recall (pooled) | 0.153 [0.129, 0.176] | |
| ours, with consensus | loop_recall (pooled) | 0.257 [0.221, 0.296] | |
| ours, with consensus | apa_ratio (pooled) | 0.581 [0.551, 0.612] | |
| all12one_fullrd_chip | pearson | 0.006 [-0.006, 0.017] | 216 of 413 |
| all12one_fullrd_chip | scc | 0.056 [0.038, 0.072] | 286 of 413 |
| all12one_fullrd_chip | ins_r_10k | 0.264 [0.252, 0.279] | 389 of 413 |
| all12one_fullrd_chip | ins_r_25k | 0.170 [0.155, 0.189] | 356 of 413 |
| all12one_fullrd_chip | ins_r_50k | 0.098 [0.076, 0.117] | 306 of 413 |
| all12one_fullrd_chip | ins_r_100k | 0.042 [0.028, 0.065] | 258 of 413 |
| all12one_fullrd_chip | ins_r_250k | 0.019 [0.006, 0.043] | 237 of 413 |
| all12one_fullrd_chip | bnd_f1 | 0.067 [0.016, 0.100] | 228 of 413 |
| all12one_fullrd_chip | bnd_f1_strict | 0.022 [0.000, 0.061] | 213 of 413 |
| all12one_fullrd_chip | bnd_recall | 0.000 [0.000, 0.000] | 184 of 413 |
| all12one_fullrd_chip | bnd_precision | 0.045 [0.000, 0.083] | 209 of 410 |
| all12one_fullrd_chip | bnd_strength_r | 0.082 [0.063, 0.102] | 268 of 413 |
| all12one_fullrd_chip | moc | 0.032 [0.017, 0.044] | 252 of 413 |
| all12one_fullrd_chip | stripe_r | 0.044 [0.027, 0.069] | 251 of 413 |
| all12one_fullrd_chip | loop_enrich_corr | 0.011 [0.006, 0.015] | 250 of 413 |
| all12one_fullrd_chip | compartment_corr | -0.001 [-0.016, 0.010] | 204 of 413 |
| all12one_fullrd_chip | eigenvector_abs_corr | -0.012 [-0.027, 0.003] | 195 of 413 |
| all12one_fullrd_chip | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 7 |
| all12one_fullrd_chip | hft_bnd_recall_per_region | 0.000 [nan, nan] | 147 of 404 |
| all12one_fullrd_chip | loop_recall_per_region | 0.000 [nan, nan] | 104 of 287 |
| all12one_fullrd_chip | bnd_f1_pooled (pooled) | 0.068 [0.045, 0.094] | |
| all12one_fullrd_chip | bnd_recall_pooled (pooled) | 0.063 [0.041, 0.087] | |
| all12one_fullrd_chip | bnd_precision_pooled (pooled) | 0.069 [0.040, 0.100] | |
| all12one_fullrd_chip | hft_bnd_recall (pooled) | 0.057 [0.029, 0.083] | |
| all12one_fullrd_chip | loop_recall (pooled) | 0.119 [0.076, 0.163] | |
| all12one_fullrd_chip | apa_ratio (pooled) | 0.360 [0.320, 0.406] | |
| all12one_fullrd_chip_nometh | pearson | -0.003 [-0.016, 0.014] | 201 of 413 |
| all12one_fullrd_chip_nometh | scc | 0.049 [0.036, 0.065] | 277 of 413 |
| all12one_fullrd_chip_nometh | ins_r_10k | 0.256 [0.239, 0.266] | 386 of 413 |
| all12one_fullrd_chip_nometh | ins_r_25k | 0.162 [0.148, 0.183] | 354 of 413 |
| all12one_fullrd_chip_nometh | ins_r_50k | 0.082 [0.067, 0.108] | 293 of 413 |
| all12one_fullrd_chip_nometh | ins_r_100k | 0.037 [0.021, 0.057] | 251 of 413 |
| all12one_fullrd_chip_nometh | ins_r_250k | 0.019 [0.005, 0.038] | 233 of 413 |
| all12one_fullrd_chip_nometh | bnd_f1 | 0.063 [0.000, 0.100] | 223 of 413 |
| all12one_fullrd_chip_nometh | bnd_f1_strict | 0.015 [0.000, 0.071] | 209 of 413 |
| all12one_fullrd_chip_nometh | bnd_recall | 0.000 [0.000, 0.000] | 181 of 413 |
| all12one_fullrd_chip_nometh | bnd_precision | 0.000 [0.000, 0.071] | 201 of 409 |
| all12one_fullrd_chip_nometh | bnd_strength_r | 0.080 [0.053, 0.109] | 256 of 413 |
| all12one_fullrd_chip_nometh | moc | 0.026 [0.012, 0.040] | 241 of 413 |
| all12one_fullrd_chip_nometh | stripe_r | 0.037 [0.019, 0.059] | 242 of 413 |
| all12one_fullrd_chip_nometh | loop_enrich_corr | 0.008 [0.003, 0.011] | 241 of 413 |
| all12one_fullrd_chip_nometh | compartment_corr | -0.009 [-0.029, 0.005] | 196 of 413 |
| all12one_fullrd_chip_nometh | eigenvector_abs_corr | -0.022 [-0.047, -0.007] | 180 of 413 |
| all12one_fullrd_chip_nometh | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 7 |
| all12one_fullrd_chip_nometh | hft_bnd_recall_per_region | 0.000 [nan, nan] | 146 of 404 |
| all12one_fullrd_chip_nometh | loop_recall_per_region | 0.000 [nan, nan] | 103 of 287 |
| all12one_fullrd_chip_nometh | bnd_f1_pooled (pooled) | 0.058 [0.034, 0.083] | |
| all12one_fullrd_chip_nometh | bnd_recall_pooled (pooled) | 0.056 [0.034, 0.080] | |
| all12one_fullrd_chip_nometh | bnd_precision_pooled (pooled) | 0.052 [0.021, 0.082] | |
| all12one_fullrd_chip_nometh | hft_bnd_recall (pooled) | 0.058 [0.029, 0.085] | |
| all12one_fullrd_chip_nometh | loop_recall (pooled) | 0.114 [0.071, 0.158] | |
| all12one_fullrd_chip_nometh | apa_ratio (pooled) | 0.346 [0.307, 0.391] | |
| all12one_fullrd_cons | pearson | 0.067 [0.050, 0.080] | 308 of 413 |
| all12one_fullrd_cons | scc | 0.118 [0.106, 0.129] | 376 of 413 |
| all12one_fullrd_cons | ins_r_10k | 0.302 [0.281, 0.315] | 399 of 413 |
| all12one_fullrd_cons | ins_r_25k | 0.232 [0.213, 0.243] | 393 of 413 |
| all12one_fullrd_cons | ins_r_50k | 0.164 [0.152, 0.180] | 382 of 413 |
| all12one_fullrd_cons | ins_r_100k | 0.120 [0.104, 0.146] | 369 of 413 |
| all12one_fullrd_cons | ins_r_250k | 0.067 [0.047, 0.090] | 319 of 413 |
| all12one_fullrd_cons | bnd_f1 | 0.133 [0.111, 0.167] | 287 of 413 |
| all12one_fullrd_cons | bnd_f1_strict | 0.143 [0.102, 0.182] | 278 of 413 |
| all12one_fullrd_cons | bnd_recall | 0.091 [0.000, 0.143] | 223 of 413 |
| all12one_fullrd_cons | bnd_precision | 0.167 [0.150, 0.200] | 260 of 411 |
| all12one_fullrd_cons | bnd_strength_r | 0.170 [0.152, 0.196] | 358 of 413 |
| all12one_fullrd_cons | moc | 0.058 [0.044, 0.077] | 305 of 413 |
| all12one_fullrd_cons | stripe_r | 0.124 [0.106, 0.141] | 346 of 413 |
| all12one_fullrd_cons | loop_enrich_corr | 0.007 [0.002, 0.011] | 239 of 413 |
| all12one_fullrd_cons | compartment_corr | 0.031 [0.021, 0.047] | 270 of 413 |
| all12one_fullrd_cons | eigenvector_abs_corr | 0.005 [0.000, 0.016] | 227 of 413 |
| all12one_fullrd_cons | bnd_f1_struct | 0.000 [0.000, 0.500] | 1 of 6 |
| all12one_fullrd_cons | hft_bnd_recall_per_region | 0.000 [nan, nan] | 185 of 404 |
| all12one_fullrd_cons | loop_recall_per_region | 0.000 [nan, nan] | 127 of 287 |
| all12one_fullrd_cons | bnd_f1_pooled (pooled) | 0.133 [0.113, 0.154] | |
| all12one_fullrd_cons | bnd_recall_pooled (pooled) | 0.116 [0.096, 0.138] | |
| all12one_fullrd_cons | bnd_precision_pooled (pooled) | 0.152 [0.124, 0.183] | |
| all12one_fullrd_cons | hft_bnd_recall (pooled) | 0.131 [0.106, 0.155] | |
| all12one_fullrd_cons | loop_recall (pooled) | 0.216 [0.176, 0.259] | |
| all12one_fullrd_cons | apa_ratio (pooled) | 0.673 [0.624, 0.724] | |
| all12one_fullrd_cons_nometh | pearson | 0.061 [0.050, 0.075] | 305 of 413 |
| all12one_fullrd_cons_nometh | scc | 0.114 [0.100, 0.123] | 375 of 413 |
| all12one_fullrd_cons_nometh | ins_r_10k | 0.288 [0.273, 0.304] | 396 of 413 |
| all12one_fullrd_cons_nometh | ins_r_25k | 0.221 [0.208, 0.236] | 389 of 413 |
| all12one_fullrd_cons_nometh | ins_r_50k | 0.160 [0.143, 0.171] | 377 of 413 |
| all12one_fullrd_cons_nometh | ins_r_100k | 0.118 [0.100, 0.142] | 372 of 413 |
| all12one_fullrd_cons_nometh | ins_r_250k | 0.059 [0.046, 0.084] | 319 of 413 |
| all12one_fullrd_cons_nometh | bnd_f1 | 0.133 [0.109, 0.167] | 289 of 413 |
| all12one_fullrd_cons_nometh | bnd_f1_strict | 0.143 [0.100, 0.182] | 271 of 413 |
| all12one_fullrd_cons_nometh | bnd_recall | 0.100 [0.000, 0.143] | 224 of 413 |
| all12one_fullrd_cons_nometh | bnd_precision | 0.167 [0.100, 0.200] | 253 of 411 |
| all12one_fullrd_cons_nometh | bnd_strength_r | 0.167 [0.148, 0.188] | 359 of 413 |
| all12one_fullrd_cons_nometh | moc | 0.058 [0.045, 0.079] | 295 of 413 |
| all12one_fullrd_cons_nometh | stripe_r | 0.123 [0.101, 0.140] | 344 of 413 |
| all12one_fullrd_cons_nometh | loop_enrich_corr | 0.005 [0.001, 0.010] | 235 of 413 |
| all12one_fullrd_cons_nometh | compartment_corr | 0.031 [0.020, 0.044] | 267 of 413 |
| all12one_fullrd_cons_nometh | eigenvector_abs_corr | 0.006 [-0.002, 0.015] | 224 of 413 |
| all12one_fullrd_cons_nometh | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| all12one_fullrd_cons_nometh | hft_bnd_recall_per_region | 0.000 [nan, nan] | 184 of 404 |
| all12one_fullrd_cons_nometh | loop_recall_per_region | 0.000 [nan, nan] | 127 of 287 |
| all12one_fullrd_cons_nometh | bnd_f1_pooled (pooled) | 0.137 [0.116, 0.158] | |
| all12one_fullrd_cons_nometh | bnd_recall_pooled (pooled) | 0.122 [0.103, 0.144] | |
| all12one_fullrd_cons_nometh | bnd_precision_pooled (pooled) | 0.148 [0.119, 0.178] | |
| all12one_fullrd_cons_nometh | hft_bnd_recall (pooled) | 0.130 [0.104, 0.153] | |
| all12one_fullrd_cons_nometh | loop_recall (pooled) | 0.216 [0.176, 0.258] | |
| all12one_fullrd_cons_nometh | apa_ratio (pooled) | 0.658 [0.610, 0.706] | |
| dedicated2kb_cons | pearson | 0.068 [0.047, 0.086] | 298 of 413 |
| dedicated2kb_cons | scc | 0.129 [0.109, 0.138] | 373 of 413 |
| dedicated2kb_cons | ins_r_10k | 0.331 [0.318, 0.345] | 404 of 413 |
| dedicated2kb_cons | ins_r_25k | 0.247 [0.232, 0.269] | 403 of 413 |
| dedicated2kb_cons | ins_r_50k | 0.182 [0.166, 0.200] | 385 of 413 |
| dedicated2kb_cons | ins_r_100k | 0.127 [0.114, 0.147] | 369 of 413 |
| dedicated2kb_cons | ins_r_250k | 0.064 [0.047, 0.089] | 327 of 413 |
| dedicated2kb_cons | bnd_f1 | 0.179 [0.143, 0.200] | 309 of 413 |
| dedicated2kb_cons | bnd_f1_strict | 0.182 [0.143, 0.222] | 289 of 413 |
| dedicated2kb_cons | bnd_recall | 0.167 [0.125, 0.200] | 242 of 413 |
| dedicated2kb_cons | bnd_precision | 0.167 [0.150, 0.200] | 270 of 411 |
| dedicated2kb_cons | bnd_strength_r | 0.188 [0.166, 0.209] | 360 of 413 |
| dedicated2kb_cons | moc | 0.080 [0.068, 0.097] | 305 of 413 |
| dedicated2kb_cons | stripe_r | 0.130 [0.114, 0.149] | 342 of 413 |
| dedicated2kb_cons | loop_enrich_corr | 0.007 [0.001, 0.011] | 231 of 413 |
| dedicated2kb_cons | compartment_corr | 0.032 [0.022, 0.044] | 274 of 413 |
| dedicated2kb_cons | eigenvector_abs_corr | 0.005 [-0.000, 0.018] | 225 of 413 |
| dedicated2kb_cons | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| dedicated2kb_cons | hft_bnd_recall_per_region | 0.000 [nan, nan] | 180 of 404 |
| dedicated2kb_cons | loop_recall_per_region | 0.167 [nan, nan] | 144 of 287 |
| dedicated2kb_cons | bnd_f1_pooled (pooled) | 0.170 [0.148, 0.192] | |
| dedicated2kb_cons | bnd_recall_pooled (pooled) | 0.169 [0.146, 0.191] | |
| dedicated2kb_cons | bnd_precision_pooled (pooled) | 0.149 [0.117, 0.184] | |
| dedicated2kb_cons | hft_bnd_recall (pooled) | 0.129 [0.102, 0.154] | |
| dedicated2kb_cons | loop_recall (pooled) | 0.247 [0.207, 0.285] | |
| dedicated2kb_cons | apa_ratio (pooled) | 0.626 [0.583, 0.671] | |
| dedicated2kb_chip | pearson | 0.013 [0.002, 0.024] | 231 of 413 |
| dedicated2kb_chip | scc | 0.063 [0.046, 0.076] | 304 of 413 |
| dedicated2kb_chip | ins_r_10k | 0.260 [0.240, 0.276] | 384 of 413 |
| dedicated2kb_chip | ins_r_25k | 0.167 [0.149, 0.189] | 362 of 413 |
| dedicated2kb_chip | ins_r_50k | 0.092 [0.075, 0.112] | 304 of 413 |
| dedicated2kb_chip | ins_r_100k | 0.043 [0.035, 0.060] | 269 of 413 |
| dedicated2kb_chip | ins_r_250k | 0.020 [0.009, 0.040] | 244 of 413 |
| dedicated2kb_chip | bnd_f1 | 0.083 [0.048, 0.111] | 235 of 413 |
| dedicated2kb_chip | bnd_f1_strict | 0.022 [0.000, 0.081] | 213 of 413 |
| dedicated2kb_chip | bnd_recall | 0.000 [0.000, 0.077] | 190 of 413 |
| dedicated2kb_chip | bnd_precision | 0.057 [0.000, 0.100] | 213 of 409 |
| dedicated2kb_chip | bnd_strength_r | 0.073 [0.053, 0.105] | 258 of 413 |
| dedicated2kb_chip | moc | 0.034 [0.023, 0.048] | 261 of 413 |
| dedicated2kb_chip | stripe_r | 0.055 [0.036, 0.070] | 261 of 413 |
| dedicated2kb_chip | loop_enrich_corr | 0.007 [0.004, 0.011] | 248 of 413 |
| dedicated2kb_chip | compartment_corr | 0.009 [-0.004, 0.022] | 219 of 413 |
| dedicated2kb_chip | eigenvector_abs_corr | 0.004 [-0.004, 0.015] | 216 of 413 |
| dedicated2kb_chip | bnd_f1_struct | 0.000 [0.000, 0.000] | 0 of 6 |
| dedicated2kb_chip | hft_bnd_recall_per_region | 0.000 [nan, nan] | 155 of 404 |
| dedicated2kb_chip | loop_recall_per_region | 0.000 [nan, nan] | 107 of 287 |
| dedicated2kb_chip | bnd_f1_pooled (pooled) | 0.073 [0.050, 0.097] | |
| dedicated2kb_chip | bnd_recall_pooled (pooled) | 0.066 [0.045, 0.088] | |
| dedicated2kb_chip | bnd_precision_pooled (pooled) | 0.077 [0.045, 0.105] | |
| dedicated2kb_chip | hft_bnd_recall (pooled) | 0.067 [0.040, 0.094] | |
| dedicated2kb_chip | loop_recall (pooled) | 0.100 [0.057, 0.144] | |
| dedicated2kb_chip | apa_ratio (pooled) | 0.351 [0.314, 0.392] | |

## Paired against the consensus of four measured cells

| predictor | metric | difference | regions ahead of n |
|---|---|---|---|
| ours, with consensus | pearson | 0.098 [0.090, 0.106] | 409 of 413 |
| ours, with consensus | ins_r_50k | 0.123 [0.110, 0.135] | 395 of 413 |
| ours, with consensus | bnd_f1 | 0.050 [0.000, 0.068] | 224 of 413 |
| ours, with consensus | compartment_corr | 0.043 [0.032, 0.054] | 334 of 413 |
| all12one_fullrd_cons | pearson | 0.033 [0.025, 0.040] | 293 of 413 |
| all12one_fullrd_cons | ins_r_50k | 0.106 [0.094, 0.116] | 352 of 413 |
| all12one_fullrd_cons | bnd_f1 | 0.048 [0.000, 0.071] | 227 of 413 |
| all12one_fullrd_cons | compartment_corr | -0.019 [-0.022, -0.006] | 177 of 413 |
