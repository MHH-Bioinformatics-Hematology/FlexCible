## 10 kb: median Pearson per cell (413 Akita test regions)

| cell | group | honest | refres | diff | wp_ref | wp_self | wp_both | consensus | P_wp_ref | P_wp_self | n_regions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gm12878 | training_cells | 0.8148 | 0.9299 | 0.9403 | 0.8268 | 0.8234 | 0.9361 | 0.8577 | 0.7184 | 0.7543 | 413 |
| imr90 | training_cells | 0.8557 | 0.9446 | 0.9476 | 0.8605 | 0.8628 | 0.945 | 0.903 | 0.7871 | 0.8163 | 413 |
| hct116 | training_cells | 0.829 | 0.9415 | 0.9514 | 0.8417 | 0.8447 | 0.9456 | 0.9105 | 0.7669 | 0.8043 | 413 |
| hff | training_cells | 0.865 | 0.9536 | 0.9576 | 0.8775 | 0.8665 | 0.9541 | 0.9166 | 0.7953 | 0.8282 | 413 |
| h1 | training_cells | 0.8227 | 0.9244 | 0.9316 | 0.8223 | 0.8092 | 0.9271 | 0.8857 | 0.7671 | 0.7942 | 413 |
| hepg2 | training_cells | 0.8239 | 0.9187 | 0.9249 | 0.8157 | 0.8156 | 0.9195 | 0.8738 | 0.7357 | 0.7543 | 413 |
| a549 | training_cells | 0.8292 | 0.9297 | 0.9378 | 0.8353 | 0.8387 | 0.9337 | 0.9132 | 0.7716 | 0.8006 | 413 |
| nhek | training_cells | 0.7756 | 0.9135 | 0.9162 | 0.8004 | 0.7741 | 0.9144 | 0.8877 | 0.7581 | 0.7224 | 413 |
| gm23248 | training_cells | 0.747 | 0.9404 | 0.9459 | 0.7397 | 0.7497 | 0.941 | 0.925 | 0.6863 | 0.6828 | 413 |
| mcf7 | training_cells | 0.8067 | 0.9014 | 0.9133 | 0.8034 | 0.8046 | 0.906 | 0.8789 | 0.756 | 0.7898 | 413 |
| cd14mono | training_cells | 0.7564 | 0.8454 | 0.85 | 0.7335 | 0.772 | 0.8455 | 0.6261 | 0.5042 | 0.5641 | 413 |
| pc9 | training_cells | 0.77 | 0.8846 | 0.8891 | 0.7742 | 0.7694 | 0.8836 | 0.8517 | 0.732 | 0.7326 | 413 |
| k562 | held_out_cells | 0.7647 | 0.9053 | 0.9047 | 0.792 | 0.7671 | 0.9056 | 0.8586 | 0.7227 | 0.7325 | 413 |
| hmec | held_out_cells | 0.6565 | 0.8089 | 0.8105 | 0.6706 | 0.6561 | 0.8092 | 0.8094 | 0.6436 | 0.6222 | 413 |
| huvec | held_out_cells | 0.6711 | 0.784 | 0.7948 | 0.6729 | 0.6581 | 0.7774 | 0.7689 | 0.6216 | 0.6245 | 413 |
| hmec_deep | held_out_cells | 0.7609 | 0.9081 | 0.9106 | 0.7775 | 0.7537 | 0.9089 | 0.9065 | 0.7476 | 0.7325 | 413 |
| huvec_deep | held_out_cells | 0.7921 | 0.8972 | 0.9188 | 0.8058 | 0.786 | 0.9007 | 0.8836 | 0.7486 | 0.7452 | 413 |
| pc3 | new_samples | 0.7717 | 0.8956 | 0.902 | 0.7654 | 0.7693 | 0.8986 | 0.8831 | nan | nan | 413 |
| panc1 | new_samples | 0.5262 | 0.6169 | 0.6232 | 0.5377 | 0.5269 | 0.6145 | 0.6126 | nan | nan | 413 |
| nk | new_samples | 0.6761 | 0.7824 | 0.8021 | 0.6722 | 0.6934 | 0.814 | 0.6576 | nan | nan | 413 |
| caco2 | new_samples | 0.7365 | 0.8971 | 0.8976 | 0.7699 | 0.7448 | 0.8964 | 0.8699 | nan | nan | 413 |

## Pooled paired contrasts (difference of medians, 95 percent bootstrap interval)

| group | contrast | n_cells | n_windows | median_a | median_b | diff | lo | hi | above | gap_closed | gap_lo | gap_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_cells | wp_ref - honest | 12 | 4956 | 0.8144 | 0.81 | 0.0044 | -0.0001 | 0.008 | False | 0.0361 | -0.0011 | 0.0658 |
| training_cells | wp_ref - diff | 12 | 4956 | 0.8144 | 0.9306 | -0.1162 | -0.1204 | -0.1125 | False | nan | nan | nan |
| training_cells | wp_self - honest | 12 | 4956 | 0.8123 | 0.81 | 0.0022 | -0.0017 | 0.0055 | False | 0.0198 | -0.0148 | 0.0478 |
| training_cells | wp_self - refres | 12 | 4956 | 0.8123 | 0.9238 | -0.1115 | -0.1165 | -0.1074 | False | nan | nan | nan |
| training_cells | wp_both - honest | 12 | 4956 | 0.9252 | 0.81 | 0.1152 | 0.1107 | 0.1197 | True | 0.9551 | 0.9463 | 0.9674 |
| training_cells | wp_both - diff | 12 | 4956 | 0.9252 | 0.9306 | -0.0054 | -0.0065 | -0.004 | False | nan | nan | nan |
| training_cells | S2 wp_both - diff | 12 | 4956 | 0.9252 | 0.9306 | -0.0054 | -0.0065 | -0.004 | False | nan | nan | nan |
| training_cells | refres - honest (the gap) | 12 | 4956 | 0.9238 | 0.81 | 0.1138 | 0.1092 | 0.1181 | True | nan | nan | nan |
| training_cells | diff - honest (the gap) | 12 | 4956 | 0.9306 | 0.81 | 0.1206 | 0.116 | 0.125 | True | nan | nan | nan |
| training_cells:gm12878 | wp_ref - honest | 1 | 413 | 0.8268 | 0.8148 | 0.0119 | -0.0054 | 0.0284 | False | nan | nan | nan |
| training_cells:gm12878 | wp_self - honest | 1 | 413 | 0.8234 | 0.8148 | 0.0086 | -0.0059 | 0.027 | False | nan | nan | nan |
| training_cells:imr90 | wp_ref - honest | 1 | 413 | 0.8605 | 0.8557 | 0.0048 | -0.0066 | 0.0148 | False | nan | nan | nan |
| training_cells:imr90 | wp_self - honest | 1 | 413 | 0.8628 | 0.8557 | 0.0072 | -0.0048 | 0.0181 | False | nan | nan | nan |
| training_cells:hct116 | wp_ref - honest | 1 | 413 | 0.8417 | 0.829 | 0.0126 | 0.0001 | 0.0319 | True | nan | nan | nan |
| training_cells:hct116 | wp_self - honest | 1 | 413 | 0.8447 | 0.829 | 0.0156 | 0.0006 | 0.0292 | True | nan | nan | nan |
| training_cells:hff | wp_ref - honest | 1 | 413 | 0.8775 | 0.865 | 0.0125 | 0.0004 | 0.0223 | True | nan | nan | nan |
| training_cells:hff | wp_self - honest | 1 | 413 | 0.8665 | 0.865 | 0.0015 | -0.0073 | 0.0125 | False | nan | nan | nan |
| training_cells:h1 | wp_ref - honest | 1 | 413 | 0.8223 | 0.8227 | -0.0004 | -0.0102 | 0.0165 | False | nan | nan | nan |
| training_cells:h1 | wp_self - honest | 1 | 413 | 0.8092 | 0.8227 | -0.0135 | -0.0212 | 0.0009 | False | nan | nan | nan |
| training_cells:hepg2 | wp_ref - honest | 1 | 413 | 0.8157 | 0.8239 | -0.0082 | -0.0171 | 0.0076 | False | nan | nan | nan |
| training_cells:hepg2 | wp_self - honest | 1 | 413 | 0.8156 | 0.8239 | -0.0082 | -0.018 | 0.003 | False | nan | nan | nan |
| training_cells:a549 | wp_ref - honest | 1 | 413 | 0.8353 | 0.8292 | 0.0061 | -0.0133 | 0.0132 | False | nan | nan | nan |
| training_cells:a549 | wp_self - honest | 1 | 413 | 0.8387 | 0.8292 | 0.0094 | -0.002 | 0.0186 | False | nan | nan | nan |
| training_cells:nhek | wp_ref - honest | 1 | 413 | 0.8004 | 0.7756 | 0.0248 | 0.0102 | 0.0448 | True | nan | nan | nan |
| training_cells:nhek | wp_self - honest | 1 | 413 | 0.7741 | 0.7756 | -0.0016 | -0.0143 | 0.0165 | False | nan | nan | nan |
| training_cells:gm23248 | wp_ref - honest | 1 | 413 | 0.7397 | 0.747 | -0.0073 | -0.0349 | 0.0154 | False | nan | nan | nan |
| training_cells:gm23248 | wp_self - honest | 1 | 413 | 0.7497 | 0.747 | 0.0027 | -0.0194 | 0.0205 | False | nan | nan | nan |
| training_cells:mcf7 | wp_ref - honest | 1 | 413 | 0.8034 | 0.8067 | -0.0033 | -0.0237 | 0.0104 | False | nan | nan | nan |
| training_cells:mcf7 | wp_self - honest | 1 | 413 | 0.8046 | 0.8067 | -0.0022 | -0.0161 | 0.0065 | False | nan | nan | nan |
| training_cells:cd14mono | wp_ref - honest | 1 | 413 | 0.7335 | 0.7564 | -0.0229 | -0.04 | -0.0022 | False | nan | nan | nan |
| training_cells:cd14mono | wp_self - honest | 1 | 413 | 0.772 | 0.7564 | 0.0156 | 0.0005 | 0.0319 | True | nan | nan | nan |
| training_cells:pc9 | wp_ref - honest | 1 | 413 | 0.7742 | 0.77 | 0.0042 | -0.0073 | 0.0199 | False | nan | nan | nan |
| training_cells:pc9 | wp_self - honest | 1 | 413 | 0.7694 | 0.77 | -0.0005 | -0.0126 | 0.0131 | False | nan | nan | nan |
| held_out_cells | wp_ref - honest | 5 | 2065 | 0.7502 | 0.7273 | 0.0229 | 0.0141 | 0.0317 | True | 0.1558 | 0.0995 | 0.207 |
| held_out_cells | wp_ref - diff | 5 | 2065 | 0.7502 | 0.8742 | -0.124 | -0.1335 | -0.1168 | False | nan | nan | nan |
| held_out_cells | wp_self - honest | 5 | 2065 | 0.722 | 0.7273 | -0.0054 | -0.0121 | 0.0023 | False | -0.039 | -0.0932 | 0.0155 |
| held_out_cells | wp_self - refres | 5 | 2065 | 0.722 | 0.8647 | -0.1427 | -0.1507 | -0.1326 | False | nan | nan | nan |
| held_out_cells | wp_both - honest | 5 | 2065 | 0.8661 | 0.7273 | 0.1388 | 0.129 | 0.1487 | True | 0.945 | 0.9284 | 0.962 |
| held_out_cells | wp_both - diff | 5 | 2065 | 0.8661 | 0.8742 | -0.0081 | -0.0105 | -0.0054 | False | nan | nan | nan |
| held_out_cells | S2 wp_both - diff | 5 | 2065 | 0.8661 | 0.8742 | -0.0081 | -0.0105 | -0.0054 | False | nan | nan | nan |
| held_out_cells | refres - honest (the gap) | 5 | 2065 | 0.8647 | 0.7273 | 0.1373 | 0.1268 | 0.1469 | True | nan | nan | nan |
| held_out_cells | diff - honest (the gap) | 5 | 2065 | 0.8742 | 0.7273 | 0.1469 | 0.1369 | 0.1568 | True | nan | nan | nan |
| held_out_cells:k562 | wp_ref - honest | 1 | 413 | 0.792 | 0.7647 | 0.0273 | 0.0147 | 0.049 | True | nan | nan | nan |
| held_out_cells:k562 | wp_self - honest | 1 | 413 | 0.7671 | 0.7647 | 0.0024 | -0.01 | 0.0187 | False | nan | nan | nan |
| held_out_cells:hmec | wp_ref - honest | 1 | 413 | 0.6706 | 0.6565 | 0.0141 | -0.006 | 0.0384 | False | nan | nan | nan |
| held_out_cells:hmec | wp_self - honest | 1 | 413 | 0.6561 | 0.6565 | -0.0004 | -0.021 | 0.0174 | False | nan | nan | nan |
| held_out_cells:huvec | wp_ref - honest | 1 | 413 | 0.6729 | 0.6711 | 0.0018 | -0.0152 | 0.0191 | False | nan | nan | nan |
| held_out_cells:huvec | wp_self - honest | 1 | 413 | 0.6581 | 0.6711 | -0.0131 | -0.0287 | 0.0062 | False | nan | nan | nan |
| held_out_cells:hmec_deep | wp_ref - honest | 1 | 413 | 0.7775 | 0.7609 | 0.0166 | 0.0001 | 0.0331 | True | nan | nan | nan |
| held_out_cells:hmec_deep | wp_self - honest | 1 | 413 | 0.7537 | 0.7609 | -0.0072 | -0.0228 | 0.0102 | False | nan | nan | nan |
| held_out_cells:huvec_deep | wp_ref - honest | 1 | 413 | 0.8058 | 0.7921 | 0.0138 | 0.0017 | 0.0263 | True | nan | nan | nan |
| held_out_cells:huvec_deep | wp_self - honest | 1 | 413 | 0.786 | 0.7921 | -0.0061 | -0.0208 | 0.0062 | False | nan | nan | nan |
| new_samples | wp_ref - honest | 4 | 1652 | 0.6882 | 0.6772 | 0.0111 | 0.002 | 0.0213 | True | 0.0738 | 0.0146 | 0.1379 |
| new_samples | wp_ref - diff | 4 | 1652 | 0.6882 | 0.8272 | -0.139 | -0.1497 | -0.1291 | False | nan | nan | nan |
| new_samples | wp_self - honest | 4 | 1652 | 0.6772 | 0.6772 | 0.0001 | -0.01 | 0.0087 | False | 0.0004 | -0.0707 | 0.0592 |
| new_samples | wp_self - refres | 4 | 1652 | 0.6772 | 0.8217 | -0.1445 | -0.1579 | -0.132 | False | nan | nan | nan |
| new_samples | wp_both - honest | 4 | 1652 | 0.8318 | 0.6772 | 0.1547 | 0.1426 | 0.1645 | True | 1.0309 | 0.9838 | 1.0533 |
| new_samples | wp_both - diff | 4 | 1652 | 0.8318 | 0.8272 | 0.0046 | -0.0024 | 0.008 | False | nan | nan | nan |
| new_samples | S2 wp_both - diff | 4 | 1652 | 0.8318 | 0.8272 | 0.0046 | -0.0024 | 0.008 | False | nan | nan | nan |
| new_samples | refres - honest (the gap) | 4 | 1652 | 0.8217 | 0.6772 | 0.1446 | 0.1327 | 0.1551 | True | nan | nan | nan |
| new_samples | diff - honest (the gap) | 4 | 1652 | 0.8272 | 0.6772 | 0.1501 | 0.1404 | 0.1609 | True | nan | nan | nan |
| new_samples:pc3 | wp_ref - honest | 1 | 413 | 0.7654 | 0.7717 | -0.0063 | -0.0201 | 0.0167 | False | nan | nan | nan |
| new_samples:pc3 | wp_self - honest | 1 | 413 | 0.7693 | 0.7717 | -0.0024 | -0.0156 | 0.0104 | False | nan | nan | nan |
| new_samples:panc1 | wp_ref - honest | 1 | 413 | 0.5377 | 0.5262 | 0.0115 | -0.0131 | 0.0264 | False | nan | nan | nan |
| new_samples:panc1 | wp_self - honest | 1 | 413 | 0.5269 | 0.5262 | 0.0007 | -0.0222 | 0.0154 | False | nan | nan | nan |
| new_samples:nk | wp_ref - honest | 1 | 413 | 0.6722 | 0.6761 | -0.0039 | -0.0258 | 0.021 | False | nan | nan | nan |
| new_samples:nk | wp_self - honest | 1 | 413 | 0.6934 | 0.6761 | 0.0173 | -0.0085 | 0.0378 | False | nan | nan | nan |
| new_samples:caco2 | wp_ref - honest | 1 | 413 | 0.7699 | 0.7365 | 0.0333 | 0.0154 | 0.048 | True | nan | nan | nan |
| new_samples:caco2 | wp_self - honest | 1 | 413 | 0.7448 | 0.7365 | 0.0082 | -0.01 | 0.0225 | False | nan | nan | nan |

## Primary endpoint

```
{
 "E1_wp_ref": {
  "diff": 0.022885246191766018,
  "lo": 0.01412013298384116,
  "hi": 0.03174227181847681,
  "above": true
 },
 "E2_wp_ref": {
  "diff": 0.011069999999999913,
  "lo": 0.0020496249999999968,
  "hi": 0.021253124999999973,
  "above": true
 },
 "E1_wp_self": {
  "diff": -0.005362609992781375,
  "lo": -0.01213480010478596,
  "hi": 0.00228226885038388,
  "above": false
 },
 "E2_wp_self": {
  "diff": 5.999999999994898e-05,
  "lo": -0.009996374999999974,
  "hi": 0.008653250000000027,
  "above": false
 },
 "met": false
}
```
