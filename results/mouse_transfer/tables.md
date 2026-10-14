# Mouse transfer: tables (10 kb unless stated; autosome windows; identical pixels)

## Samples

| sample | res | experiment | assay | contacts_M | cis_contacts_M | ice_masked_frac |
|---|---|---|---|---|---|---|
| mesc | 10000 | ENCSR051JZL | hic |  |  | 0.113 |
| mesc | 25000 | ENCSR051JZL | hic | 1176.500 | 1086.100 | 0.092 |
| ch12 | 10000 | ENCSR005UON | hic |  |  | 0.083 |
| ch12 | 25000 | ENCSR005UON | hic | 664.700 | 532.700 | 0.077 |
| heart | 10000 | ENCSR997TTD | intact |  |  | 0.094 |
| heart | 25000 | ENCSR997TTD | intact | 1403.700 | 749.400 | 0.085 |
| mesc_rep | 10000 | ENCSR819XFA | hic |  |  | 0.113 |
| mesc_rep | 25000 | ENCSR819XFA | hic | 473.500 | 414.700 | 0.094 |

## Median per-window Pearson of log O/E


### 10kb

| method (10kb) | ch12 | heart | mesc |
|---|---|---|---|
| distance_decay | 0.000 | 0.000 | 0.000 |
| distance_oracle | 0.376 | 0.380 | 0.425 |
| untrained:all7 | 0.007 | 0.026 | 0.031 |
| honest_shuf:all7 | 0.162 | 0.150 | 0.300 |
| honest:none | 0.386 | 0.452 | 0.536 |
| honest:ctcf | 0.500 | 0.620 | 0.660 |
| honest:ctcf_k27ac | 0.542 | 0.674 | 0.683 |
| honest_xm:all7 | 0.533 | 0.608 | 0.722 |
| honest_b4:all7 |  |  | 0.733 |
| honest:all7 | 0.635 | 0.722 | 0.733 |
| honest:avail | 0.688 |  |  |
| consensus | 0.746 | 0.765 | 0.766 |
| refin_shuf:all7 | 0.643 | 0.692 | 0.754 |
| refin_xm:all7 | 0.740 | 0.765 | 0.779 |
| refin:none | 0.762 | 0.790 | 0.793 |
| refin:all7 | 0.803 | 0.840 | 0.833 |
| refres:all7 | 0.808 | 0.835 | 0.829 |
| diff:all7 | 0.812 | 0.836 | 0.834 |
| akita:Hsieh_mESC_uC |  |  | 0.715 |
| akita:Bonev_ES |  |  | 0.724 |
| akita:Bonev_CN |  |  | 0.671 |
| akita:Bonev_NPC |  |  | 0.702 |
| replicate_experiment |  |  | 0.921 |
| split-half ceiling (Spearman-Brown) | 0.934 | 0.963 | 0.944 |

windows: mesc 5629, ch12 5629, heart 5629

### 25kb

| method (25kb) | ch12 | heart | mesc |
|---|---|---|---|
| distance_decay | 0.000 | 0.000 | 0.000 |
| distance_oracle | 0.511 | 0.448 | 0.473 |
| honest_shuf:all7 | 0.235 | 0.204 | 0.341 |
| honest:none | 0.558 | 0.556 | 0.603 |
| honest:ctcf | 0.623 | 0.654 | 0.679 |
| honest:ctcf_k27ac | 0.661 | 0.725 | 0.715 |
| honest_xm:all7 | 0.658 | 0.671 | 0.753 |
| honest_b4:all7 |  |  | 0.745 |
| honest:all7 | 0.748 | 0.775 | 0.759 |
| honest:avail | 0.771 |  |  |
| consensus | 0.831 | 0.833 | 0.838 |
| refin_shuf:all7 | 0.751 | 0.773 | 0.819 |
| refin_xm:all7 | 0.820 | 0.820 | 0.843 |
| refin:none | 0.835 | 0.847 | 0.854 |
| refin:all7 | 0.874 | 0.882 | 0.878 |
| refres:all7 | 0.863 | 0.875 | 0.876 |
| diff:all7 | 0.881 | 0.870 | 0.879 |
| akita:Hsieh_mESC_uC |  |  | 0.697 |
| akita:Bonev_ES |  |  | 0.712 |
| akita:Bonev_CN |  |  | 0.676 |
| akita:Bonev_NPC |  |  | 0.698 |
| replicate_experiment |  |  | 0.979 |

windows: mesc 5579, ch12 5579, heart 5579

## Paired contrasts of the median Pearson (95 percent bootstrap interval)


### 10kb

| contrast | ch12 | heart | mesc |
|---|---|---|---|
| akita:Bonev_ES - consensus |  |  | -0.043 [-0.055, -0.029] |
| akita:Hsieh_mESC_uC - honest_shuf:all7 |  |  | +0.406 [+0.378, +0.434] |
| consensus - replicate_experiment |  |  | -0.155 [-0.160, -0.150] |
| diff:all7 - consensus | +0.066 [+0.060, +0.072] | +0.071 [+0.067, +0.075] | +0.069 [+0.065, +0.072] |
| honest:all7 - akita:Bonev_CN |  |  | +0.067 [+0.048, +0.082] |
| honest:all7 - akita:Bonev_ES |  |  | +0.014 [-0.001, +0.028] |
| honest:all7 - akita:Bonev_NPC |  |  | +0.036 [+0.021, +0.054] |
| honest:all7 - akita:Hsieh_mESC_uC |  |  | +0.023 [+0.008, +0.038] |
| honest:all7 - consensus | -0.111 [-0.121, -0.103] | -0.043 [-0.050, -0.035] | -0.033 [-0.040, -0.027] |
| honest:all7 - distance_decay | +0.635 [+0.624, +0.643] | +0.722 [+0.715, +0.731] | +0.733 [+0.727, +0.738] |
| honest:all7 - distance_oracle | +0.260 [+0.248, +0.269] | +0.341 [+0.331, +0.353] | +0.308 [+0.298, +0.318] |
| honest:all7 - honest:ctcf | +0.135 [+0.125, +0.142] | +0.101 [+0.097, +0.108] | +0.073 [+0.068, +0.080] |
| honest:all7 - honest:ctcf_k27ac | +0.094 [+0.084, +0.101] | +0.047 [+0.043, +0.052] | +0.050 [+0.045, +0.055] |
| honest:all7 - honest:none | +0.249 [+0.237, +0.259] | +0.270 [+0.259, +0.280] | +0.197 [+0.189, +0.206] |
| honest:all7 - honest_b4:all7 |  |  | -0.000 [-0.005, +0.005] |
| honest:all7 - honest_shuf:all7 | +0.473 [+0.459, +0.485] | +0.571 [+0.559, +0.584] | +0.432 [+0.421, +0.444] |
| honest:all7 - honest_xm:all7 | +0.103 [+0.092, +0.110] | +0.114 [+0.107, +0.124] | +0.011 [+0.006, +0.017] |
| honest:all7 - replicate_experiment |  |  | -0.188 [-0.193, -0.183] |
| honest:all7 - untrained:all7 | +0.628 [+0.617, +0.637] | +0.695 [+0.689, +0.704] | +0.702 [+0.695, +0.707] |
| honest:avail - honest:all7 | +0.053 [+0.047, +0.060] |  |  |
| refin:all7 - akita:Bonev_ES |  |  | +0.106 [+0.095, +0.120] |
| refin:all7 - consensus | +0.057 [+0.052, +0.063] | +0.076 [+0.072, +0.079] | +0.067 [+0.063, +0.071] |
| refin:all7 - honest:all7 | +0.168 [+0.161, +0.178] | +0.119 [+0.111, +0.125] | +0.100 [+0.094, +0.105] |
| refin:all7 - refin:none | +0.041 [+0.036, +0.046] | +0.051 [+0.047, +0.054] | +0.040 [+0.036, +0.042] |
| refin:all7 - refin_shuf:all7 | +0.161 [+0.152, +0.170] | +0.148 [+0.142, +0.154] | +0.079 [+0.075, +0.083] |
| refin:all7 - refin_xm:all7 | +0.063 [+0.059, +0.068] | +0.075 [+0.072, +0.081] | +0.054 [+0.050, +0.057] |
| refin:avail - refin:all7 | +0.014 [+0.012, +0.017] |  |  |
| refres:all7 - consensus | +0.062 [+0.056, +0.067] | +0.070 [+0.066, +0.073] | +0.063 [+0.060, +0.067] |

### 25kb

| contrast | ch12 | heart | mesc |
|---|---|---|---|
| akita:Bonev_ES - consensus |  |  | -0.128 [-0.144, -0.114] |
| akita:Hsieh_mESC_uC - honest_shuf:all7 |  |  | +0.350 [+0.320, +0.388] |
| consensus - replicate_experiment |  |  | -0.141 [-0.146, -0.136] |
| diff:all7 - consensus | +0.050 [+0.046, +0.053] | +0.037 [+0.034, +0.040] | +0.040 [+0.038, +0.044] |
| honest:all7 - akita:Bonev_CN |  |  | +0.096 [+0.078, +0.117] |
| honest:all7 - akita:Bonev_ES |  |  | +0.060 [+0.043, +0.083] |
| honest:all7 - akita:Bonev_NPC |  |  | +0.075 [+0.055, +0.098] |
| honest:all7 - akita:Hsieh_mESC_uC |  |  | +0.075 [+0.056, +0.096] |
| honest:all7 - consensus | -0.084 [-0.091, -0.076] | -0.058 [-0.065, -0.052] | -0.079 [-0.086, -0.071] |
| honest:all7 - distance_decay | +0.748 [+0.742, +0.754] | +0.775 [+0.768, +0.782] | +0.759 [+0.752, +0.767] |
| honest:all7 - distance_oracle | +0.236 [+0.226, +0.247] | +0.327 [+0.317, +0.339] | +0.286 [+0.275, +0.296] |
| honest:all7 - honest:ctcf | +0.124 [+0.117, +0.136] | +0.121 [+0.113, +0.129] | +0.080 [+0.072, +0.087] |
| honest:all7 - honest:ctcf_k27ac | +0.086 [+0.080, +0.096] | +0.049 [+0.045, +0.054] | +0.044 [+0.038, +0.051] |
| honest:all7 - honest:none | +0.189 [+0.179, +0.201] | +0.219 [+0.209, +0.231] | +0.156 [+0.146, +0.164] |
| honest:all7 - honest_b4:all7 |  |  | +0.014 [+0.009, +0.021] |
| honest:all7 - honest_shuf:all7 | +0.513 [+0.497, +0.530] | +0.571 [+0.557, +0.587] | +0.418 [+0.406, +0.435] |
| honest:all7 - honest_xm:all7 | +0.090 [+0.082, +0.098] | +0.104 [+0.094, +0.114] | +0.006 [-0.001, +0.013] |
| honest:all7 - replicate_experiment |  |  | -0.220 [-0.226, -0.213] |
| honest:avail - honest:all7 | +0.023 [+0.019, +0.027] |  |  |
| refin:all7 - akita:Bonev_ES |  |  | +0.166 [+0.154, +0.183] |
| refin:all7 - consensus | +0.043 [+0.039, +0.046] | +0.049 [+0.046, +0.052] | +0.040 [+0.037, +0.043] |
| refin:all7 - honest:all7 | +0.126 [+0.120, +0.132] | +0.107 [+0.101, +0.113] | +0.119 [+0.112, +0.126] |
| refin:all7 - refin:none | +0.039 [+0.035, +0.042] | +0.036 [+0.033, +0.038] | +0.024 [+0.022, +0.026] |
| refin:all7 - refin_shuf:all7 | +0.123 [+0.117, +0.129] | +0.110 [+0.106, +0.114] | +0.059 [+0.056, +0.062] |
| refin:all7 - refin_xm:all7 | +0.054 [+0.049, +0.057] | +0.063 [+0.058, +0.066] | +0.036 [+0.033, +0.038] |
| refin:avail - refin:all7 | +0.008 [+0.006, +0.009] |  |  |
| refres:all7 - consensus | +0.032 [+0.028, +0.036] | +0.042 [+0.039, +0.044] | +0.038 [+0.035, +0.042] |

## Structure classes, honest (7 marks) against controls, 10 kb


### honest minus honest_shuf:all7

| metric | ch12 | heart | mesc |
|---|---|---|---|
| boundary_f1 | +0.187 [+0.173, +0.203] | +0.197 [+0.181, +0.212] | +0.188 [+0.173, +0.203] |
| compartment_corr | +0.198 [+0.188, +0.206] | +0.205 [+0.194, +0.212] | +0.177 [+0.171, +0.185] |
| eigenvector_abs_corr | +0.190 [+0.179, +0.200] | +0.190 [+0.178, +0.200] | +0.153 [+0.142, +0.163] |
| insulation_corr | +0.518 [+0.489, +0.547] | +0.596 [+0.568, +0.625] | +0.461 [+0.435, +0.489] |
| loop_apa_ratio | +0.127 [+0.119, +0.133] | +0.318 [+0.312, +0.323] | +0.227 [+0.219, +0.237] |
| loop_enrich_corr | +0.315 [+0.308, +0.324] | +0.461 [+0.453, +0.468] | +0.337 [+0.330, +0.344] |
| pearson | +0.473 [+0.459, +0.485] | +0.571 [+0.559, +0.584] | +0.432 [+0.421, +0.444] |
| scc | +0.443 [+0.430, +0.455] | +0.560 [+0.546, +0.571] | +0.428 [+0.412, +0.440] |
| stripe_corr | +0.441 [+0.428, +0.458] | +0.559 [+0.545, +0.573] | +0.433 [+0.419, +0.447] |

### honest minus honest:none

| metric | ch12 | heart | mesc |
|---|---|---|---|
| boundary_f1 | +0.106 [+0.092, +0.119] | +0.117 [+0.103, +0.132] | +0.102 [+0.088, +0.117] |
| compartment_corr | +0.129 [+0.120, +0.137] | +0.132 [+0.124, +0.138] | +0.100 [+0.095, +0.108] |
| eigenvector_abs_corr | +0.124 [+0.115, +0.137] | +0.111 [+0.102, +0.122] | +0.080 [+0.072, +0.091] |
| insulation_corr | +0.269 [+0.246, +0.288] | +0.313 [+0.294, +0.332] | +0.236 [+0.219, +0.255] |
| loop_apa_ratio | +0.088 [+0.082, +0.093] | +0.272 [+0.267, +0.277] | +0.159 [+0.151, +0.167] |
| loop_enrich_corr | +0.195 [+0.188, +0.203] | +0.311 [+0.303, +0.318] | +0.205 [+0.199, +0.211] |
| pearson | +0.249 [+0.237, +0.259] | +0.270 [+0.259, +0.280] | +0.197 [+0.189, +0.206] |
| scc | +0.238 [+0.226, +0.250] | +0.320 [+0.309, +0.331] | +0.227 [+0.217, +0.236] |
| stripe_corr | +0.227 [+0.216, +0.242] | +0.305 [+0.292, +0.317] | +0.211 [+0.202, +0.219] |

### honest (7 marks), median per window (boundary F1: mean)

| metric | ch12 | heart | mesc |
|---|---|---|---|
| boundary_f1 | 0.267 | 0.261 | 0.290 |
| compartment_corr | 0.751 | 0.800 | 0.816 |
| eigenvector_abs_corr | 0.843 | 0.876 | 0.896 |
| insulation_corr | 0.754 | 0.808 | 0.794 |
| loop_apa_ratio | 0.150 | 0.352 | 0.295 |
| loop_enrich_corr | 0.384 | 0.527 | 0.452 |
| pearson | 0.635 | 0.722 | 0.733 |
| scc | 0.568 | 0.695 | 0.651 |
| stripe_corr | 0.606 | 0.720 | 0.692 |

### consensus alone (mouse leave-one-out)

| metric | ch12 | heart | mesc |
|---|---|---|---|
| boundary_f1 | 0.380 | 0.369 | 0.369 |
| compartment_corr | 0.817 | 0.832 | 0.834 |
| eigenvector_abs_corr | 0.893 | 0.907 | 0.912 |
| insulation_corr | 0.877 | 0.880 | 0.875 |
| loop_apa_ratio | 0.490 | 0.500 | 0.979 |
| loop_enrich_corr | 0.462 | 0.498 | 0.490 |
| pearson | 0.746 | 0.765 | 0.766 |
| scc | 0.718 | 0.741 | 0.746 |
| stripe_corr | 0.787 | 0.799 | 0.800 |

## Primary endpoint

```
{
 "definition": "honest:all7 10 kb autosome windows above distance_decay and above honest_shuf:all7 in every sample (difference of medians, 95% paired bootstrap interval excludes 0)",
 "samples": {
  "mesc": {
   "distance_decay": {
    "cell": "mesc",
    "res": 10000,
    "m1": "honest:all7",
    "m2": "distance_decay",
    "metric": "pearson",
    "n": 5629,
    "stat_m1": 0.7328,
    "stat_m2": 0.0,
    "diff": 0.7328,
    "lo": 0.72668,
    "hi": 0.73774,
    "frac_windows_m1_above": 0.9877420500977083
   },
   "honest_shuf:all7": {
    "cell": "mesc",
    "res": 10000,
    "m1": "honest:all7",
    "m2": "honest_shuf:all7",
    "metric": "pearson",
    "n": 5629,
    "stat_m1": 0.7328,
    "stat_m2": 0.3004,
    "diff": 0.4324,
    "lo": 0.42089975,
    "hi": 0.44447175,
    "frac_windows_m1_above": 0.925386391899094
   },
   "honest_median": 0.7328,
   "ceiling_full_depth_median": 0.9443728537846935,
   "fraction_of_ceiling": 0.775964701932305
  },
  "ch12": {
   "distance_decay": {
    "cell": "ch12",
    "res": 10000,
    "m1": "honest:all7",
    "m2": "distance_decay",
    "metric": "pearson",
    "n": 5629,
    "stat_m1": 0.63521,
    "stat_m2": 0.0,
    "diff": 0.63521,
    "lo": 0.62385,
    "hi": 0.6432382499999999,
    "frac_windows_m1_above": 0.9596731213359389
   },
   "honest_shuf:all7": {
    "cell": "ch12",
    "res": 10000,
    "m1": "honest:all7",
    "m2": "honest_shuf:all7",
    "metric": "pearson",
    "n": 5629,
    "stat_m1": 0.63521,
    "stat_m2": 0.16177,
    "diff": 0.4734400000000001,
    "lo": 0.4592535,
    "hi": 0.48548025,
    "frac_windows_m1_above": 0.9046011724995559
   },
   "honest_median": 0.63521,
   "ceiling_full_depth_median": 0.9344047163039835,
   "fraction_of_ceiling": 0.6798017913614122
  },
  "heart": {
   "distance_decay": {
    "cell": "heart",
    "res": 10000,
    "m1": "honest:all7",
    "m2": "distance_decay",
    "metric": "pearson",
    "n": 5629,
    "stat_m1": 0.72153,
    "stat_m2": 0.0,
    "diff": 0.72153,
    "lo": 0.71508,
    "hi": 0.7305,
    "frac_windows_m1_above": 0.9694439509682004
   },
   "honest_shuf:all7": {
    "cell": "heart",
    "res": 10000,
    "m1": "honest:all7",
    "m2": "honest_shuf:all7",
    "metric": "pearson",
    "n": 5629,
    "stat_m1": 0.72153,
    "stat_m2": 0.15027,
    "diff": 0.57126,
    "lo": 0.5591397499999999,
    "hi": 0.58444025,
    "frac_windows_m1_above": 0.9379996446971043
   },
   "honest_median": 0.72153,
   "ceiling_full_depth_median": 0.9631660820003638,
   "fraction_of_ceiling": 0.7491231403222601
  }
 },
 "endpoint_met": true
}
```
