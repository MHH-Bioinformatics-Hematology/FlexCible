## 25 kb: median Pearson per cell (413 Akita test regions)

| cell | group | honest | refres | diff | wp_ref | wp_self | wp_both | consensus | P_wp_ref | P_wp_self | n_regions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gm12878 | training_cells | 0.8096 | 0.9313 | 0.9335 | 0.8305 | 0.8003 | 0.9351 | 0.8819 | 0.7485 | 0.7458 | 413 |
| imr90 | training_cells | 0.8576 | 0.9533 | 0.9522 | 0.871 | 0.8662 | 0.9539 | 0.9278 | 0.8139 | 0.8204 | 413 |
| hct116 | training_cells | 0.8227 | 0.9467 | 0.9508 | 0.8488 | 0.8398 | 0.9481 | 0.9319 | 0.7765 | 0.8008 | 413 |
| hff | training_cells | 0.8766 | 0.9585 | 0.9598 | 0.8829 | 0.8745 | 0.9595 | 0.9325 | 0.8145 | 0.823 | 413 |
| h1 | training_cells | 0.8011 | 0.9344 | 0.9387 | 0.8291 | 0.8198 | 0.9404 | 0.9118 | 0.7827 | 0.7865 | 413 |
| hepg2 | training_cells | 0.8077 | 0.9325 | 0.9361 | 0.8417 | 0.8092 | 0.9335 | 0.9057 | 0.7801 | 0.7659 | 413 |
| a549 | training_cells | 0.8436 | 0.945 | 0.9519 | 0.8591 | 0.8448 | 0.9513 | 0.9461 | 0.8132 | 0.8076 | 413 |
| nhek | training_cells | 0.791 | 0.9371 | 0.9388 | 0.8293 | 0.7767 | 0.9372 | 0.9267 | 0.7846 | 0.7382 | 413 |
| gm23248 | training_cells | 0.7783 | 0.945 | 0.9457 | 0.7603 | 0.7467 | 0.9466 | 0.9394 | 0.6849 | 0.6943 | 413 |
| mcf7 | training_cells | 0.8202 | 0.926 | 0.9354 | 0.8293 | 0.8098 | 0.9359 | 0.9179 | 0.7777 | 0.7997 | 413 |
| cd14mono | training_cells | 0.7865 | 0.8704 | 0.8745 | 0.7989 | 0.7761 | 0.8757 | 0.6683 | 0.5714 | 0.5896 | 413 |
| pc9 | training_cells | 0.8048 | 0.9091 | 0.9077 | 0.8106 | 0.8007 | 0.912 | 0.9069 | 0.7781 | 0.7639 | 413 |
| k562 | held_out_cells | 0.7988 | 0.9213 | 0.9236 | 0.8267 | 0.7947 | 0.9223 | 0.8972 | 0.7767 | 0.7407 | 413 |
| hmec | held_out_cells | 0.7112 | 0.8836 | 0.8931 | 0.7586 | 0.701 | 0.894 | 0.8977 | 0.7424 | 0.6818 | 413 |
| huvec | held_out_cells | 0.775 | 0.8802 | 0.897 | 0.7962 | 0.7567 | 0.8944 | 0.8869 | 0.7581 | 0.6949 | 413 |
| hmec_deep | held_out_cells | 0.7517 | 0.9149 | 0.9237 | 0.7982 | 0.7297 | 0.9229 | 0.9324 | 0.7747 | 0.7178 | 413 |
| huvec_deep | held_out_cells | 0.8115 | 0.9072 | 0.9268 | 0.8305 | 0.7949 | 0.9249 | 0.9159 | 0.7808 | 0.7278 | 413 |
| pc3 | new_samples | 0.8165 | 0.9405 | 0.9441 | 0.8422 | 0.8154 | 0.9428 | 0.9316 | nan | nan | 413 |
| panc1 | new_samples | 0.709 | 0.8407 | 0.841 | 0.7422 | 0.6966 | 0.8328 | 0.8388 | nan | nan | 413 |
| nk | new_samples | 0.734 | 0.8627 | 0.8576 | 0.7395 | 0.754 | 0.8561 | 0.7074 | nan | nan | 413 |
| caco2 | new_samples | 0.7591 | 0.9304 | 0.9389 | 0.8161 | 0.7992 | 0.9393 | 0.9149 | nan | nan | 413 |

## Pooled paired contrasts (difference of medians, 95 percent bootstrap interval)

| group | contrast | n_cells | n_windows | median_a | median_b | diff | lo | hi | above | gap_closed | gap_lo | gap_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_cells | wp_ref - honest | 12 | 4956 | 0.8341 | 0.8184 | 0.0157 | 0.0113 | 0.0199 | True | 0.1303 | 0.0968 | 0.1615 |
| training_cells | wp_ref - diff | 12 | 4956 | 0.8341 | 0.9386 | -0.1045 | -0.1086 | -0.1008 | False | nan | nan | nan |
| training_cells | wp_self - honest | 12 | 4956 | 0.8172 | 0.8184 | -0.0013 | -0.0057 | 0.0032 | False | -0.011 | -0.049 | 0.0268 |
| training_cells | wp_self - refres | 12 | 4956 | 0.8172 | 0.9349 | -0.1178 | -0.1226 | -0.1126 | False | nan | nan | nan |
| training_cells | wp_both - honest | 12 | 4956 | 0.9387 | 0.8184 | 0.1202 | 0.1156 | 0.1252 | True | 1.0001 | 0.9921 | 1.0111 |
| training_cells | wp_both - diff | 12 | 4956 | 0.9387 | 0.9386 | 0.0 | -0.001 | 0.0013 | False | nan | nan | nan |
| training_cells | S2 wp_both - diff | 12 | 4956 | 0.9387 | 0.9386 | 0.0 | -0.001 | 0.0013 | False | nan | nan | nan |
| training_cells | refres - honest (the gap) | 12 | 4956 | 0.9349 | 0.8184 | 0.1165 | 0.1119 | 0.1213 | True | nan | nan | nan |
| training_cells | diff - honest (the gap) | 12 | 4956 | 0.9386 | 0.8184 | 0.1202 | 0.1154 | 0.125 | True | nan | nan | nan |
| training_cells:gm12878 | wp_ref - honest | 1 | 413 | 0.8305 | 0.8096 | 0.021 | 0.0061 | 0.0348 | True | nan | nan | nan |
| training_cells:gm12878 | wp_self - honest | 1 | 413 | 0.8003 | 0.8096 | -0.0092 | -0.025 | 0.0132 | False | nan | nan | nan |
| training_cells:imr90 | wp_ref - honest | 1 | 413 | 0.871 | 0.8576 | 0.0134 | -0.0005 | 0.027 | False | nan | nan | nan |
| training_cells:imr90 | wp_self - honest | 1 | 413 | 0.8662 | 0.8576 | 0.0086 | -0.0045 | 0.0188 | False | nan | nan | nan |
| training_cells:hct116 | wp_ref - honest | 1 | 413 | 0.8488 | 0.8227 | 0.0261 | 0.007 | 0.0349 | True | nan | nan | nan |
| training_cells:hct116 | wp_self - honest | 1 | 413 | 0.8398 | 0.8227 | 0.0171 | -0.0011 | 0.0268 | False | nan | nan | nan |
| training_cells:hff | wp_ref - honest | 1 | 413 | 0.8829 | 0.8766 | 0.0063 | -0.0074 | 0.0175 | False | nan | nan | nan |
| training_cells:hff | wp_self - honest | 1 | 413 | 0.8745 | 0.8766 | -0.0021 | -0.012 | 0.0102 | False | nan | nan | nan |
| training_cells:h1 | wp_ref - honest | 1 | 413 | 0.8291 | 0.8011 | 0.028 | 0.0027 | 0.0403 | True | nan | nan | nan |
| training_cells:h1 | wp_self - honest | 1 | 413 | 0.8198 | 0.8011 | 0.0186 | -0.0013 | 0.0341 | False | nan | nan | nan |
| training_cells:hepg2 | wp_ref - honest | 1 | 413 | 0.8417 | 0.8077 | 0.034 | 0.0181 | 0.0472 | True | nan | nan | nan |
| training_cells:hepg2 | wp_self - honest | 1 | 413 | 0.8092 | 0.8077 | 0.0015 | -0.0174 | 0.0122 | False | nan | nan | nan |
| training_cells:a549 | wp_ref - honest | 1 | 413 | 0.8591 | 0.8436 | 0.0155 | -0.0028 | 0.0248 | False | nan | nan | nan |
| training_cells:a549 | wp_self - honest | 1 | 413 | 0.8448 | 0.8436 | 0.0012 | -0.0129 | 0.0151 | False | nan | nan | nan |
| training_cells:nhek | wp_ref - honest | 1 | 413 | 0.8293 | 0.791 | 0.0382 | 0.0185 | 0.0617 | True | nan | nan | nan |
| training_cells:nhek | wp_self - honest | 1 | 413 | 0.7767 | 0.791 | -0.0143 | -0.0313 | 0.0068 | False | nan | nan | nan |
| training_cells:gm23248 | wp_ref - honest | 1 | 413 | 0.7603 | 0.7783 | -0.018 | -0.044 | 0.0054 | False | nan | nan | nan |
| training_cells:gm23248 | wp_self - honest | 1 | 413 | 0.7467 | 0.7783 | -0.0316 | -0.0559 | -0.0072 | False | nan | nan | nan |
| training_cells:mcf7 | wp_ref - honest | 1 | 413 | 0.8293 | 0.8202 | 0.0091 | -0.0063 | 0.0222 | False | nan | nan | nan |
| training_cells:mcf7 | wp_self - honest | 1 | 413 | 0.8098 | 0.8202 | -0.0104 | -0.0258 | 0.0094 | False | nan | nan | nan |
| training_cells:cd14mono | wp_ref - honest | 1 | 413 | 0.7989 | 0.7865 | 0.0123 | -0.0066 | 0.0286 | False | nan | nan | nan |
| training_cells:cd14mono | wp_self - honest | 1 | 413 | 0.7761 | 0.7865 | -0.0104 | -0.0317 | 0.0063 | False | nan | nan | nan |
| training_cells:pc9 | wp_ref - honest | 1 | 413 | 0.8106 | 0.8048 | 0.0058 | -0.0066 | 0.029 | False | nan | nan | nan |
| training_cells:pc9 | wp_self - honest | 1 | 413 | 0.8007 | 0.8048 | -0.004 | -0.0157 | 0.016 | False | nan | nan | nan |
| held_out_cells | wp_ref - honest | 5 | 2065 | 0.803 | 0.7752 | 0.0279 | 0.0209 | 0.0395 | True | 0.2021 | 0.1563 | 0.2686 |
| held_out_cells | wp_ref - diff | 5 | 2065 | 0.803 | 0.913 | -0.11 | -0.1168 | -0.1003 | False | nan | nan | nan |
| held_out_cells | wp_self - honest | 5 | 2065 | 0.7554 | 0.7752 | -0.0197 | -0.027 | -0.0113 | False | -0.152 | -0.2134 | -0.0816 |
| held_out_cells | wp_self - refres | 5 | 2065 | 0.7554 | 0.9047 | -0.1493 | -0.1586 | -0.1416 | False | nan | nan | nan |
| held_out_cells | wp_both - honest | 5 | 2065 | 0.9125 | 0.7752 | 0.1374 | 0.1291 | 0.1506 | True | 0.9963 | 0.9802 | 1.0115 |
| held_out_cells | wp_both - diff | 5 | 2065 | 0.9125 | 0.913 | -0.0005 | -0.0027 | 0.0016 | False | nan | nan | nan |
| held_out_cells | S2 wp_both - diff | 5 | 2065 | 0.9125 | 0.913 | -0.0005 | -0.0027 | 0.0016 | False | nan | nan | nan |
| held_out_cells | refres - honest (the gap) | 5 | 2065 | 0.9047 | 0.7752 | 0.1296 | 0.1216 | 0.1429 | True | nan | nan | nan |
| held_out_cells | diff - honest (the gap) | 5 | 2065 | 0.913 | 0.7752 | 0.1379 | 0.1298 | 0.1508 | True | nan | nan | nan |
| held_out_cells:k562 | wp_ref - honest | 1 | 413 | 0.8267 | 0.7988 | 0.0279 | 0.0103 | 0.0463 | True | nan | nan | nan |
| held_out_cells:k562 | wp_self - honest | 1 | 413 | 0.7947 | 0.7988 | -0.004 | -0.0225 | 0.0111 | False | nan | nan | nan |
| held_out_cells:hmec | wp_ref - honest | 1 | 413 | 0.7586 | 0.7112 | 0.0474 | 0.0272 | 0.0764 | True | nan | nan | nan |
| held_out_cells:hmec | wp_self - honest | 1 | 413 | 0.701 | 0.7112 | -0.0102 | -0.0246 | 0.0119 | False | nan | nan | nan |
| held_out_cells:huvec | wp_ref - honest | 1 | 413 | 0.7962 | 0.775 | 0.0213 | 0.0072 | 0.0384 | True | nan | nan | nan |
| held_out_cells:huvec | wp_self - honest | 1 | 413 | 0.7567 | 0.775 | -0.0182 | -0.0415 | -0.0035 | False | nan | nan | nan |
| held_out_cells:hmec_deep | wp_ref - honest | 1 | 413 | 0.7982 | 0.7517 | 0.0465 | 0.0245 | 0.0714 | True | nan | nan | nan |
| held_out_cells:hmec_deep | wp_self - honest | 1 | 413 | 0.7297 | 0.7517 | -0.022 | -0.0383 | 0.0009 | False | nan | nan | nan |
| held_out_cells:huvec_deep | wp_ref - honest | 1 | 413 | 0.8305 | 0.8115 | 0.019 | 0.0006 | 0.0313 | True | nan | nan | nan |
| held_out_cells:huvec_deep | wp_self - honest | 1 | 413 | 0.7949 | 0.8115 | -0.0166 | -0.0399 | -0.0042 | False | nan | nan | nan |
| new_samples | wp_ref - honest | 4 | 1652 | 0.7852 | 0.7553 | 0.0298 | 0.0189 | 0.0374 | True | 0.2034 | 0.1345 | 0.2503 |
| new_samples | wp_ref - diff | 4 | 1652 | 0.7852 | 0.902 | -0.1168 | -0.1262 | -0.1086 | False | nan | nan | nan |
| new_samples | wp_self - honest | 4 | 1652 | 0.7646 | 0.7553 | 0.0092 | -0.0008 | 0.0166 | False | 0.0644 | -0.0056 | 0.114 |
| new_samples | wp_self - refres | 4 | 1652 | 0.7646 | 0.8987 | -0.1341 | -0.1445 | -0.1252 | False | nan | nan | nan |
| new_samples | wp_both - honest | 4 | 1652 | 0.901 | 0.7553 | 0.1457 | 0.1351 | 0.1533 | True | 0.9937 | 0.9729 | 1.0197 |
| new_samples | wp_both - diff | 4 | 1652 | 0.901 | 0.902 | -0.0009 | -0.004 | 0.0028 | False | nan | nan | nan |
| new_samples | S2 wp_both - diff | 4 | 1652 | 0.901 | 0.902 | -0.0009 | -0.004 | 0.0028 | False | nan | nan | nan |
| new_samples | refres - honest (the gap) | 4 | 1652 | 0.8987 | 0.7553 | 0.1434 | 0.1327 | 0.1514 | True | nan | nan | nan |
| new_samples | diff - honest (the gap) | 4 | 1652 | 0.902 | 0.7553 | 0.1466 | 0.1355 | 0.1547 | True | nan | nan | nan |
| new_samples:pc3 | wp_ref - honest | 1 | 413 | 0.8422 | 0.8166 | 0.0257 | 0.0116 | 0.0434 | True | nan | nan | nan |
| new_samples:pc3 | wp_self - honest | 1 | 413 | 0.8154 | 0.8166 | -0.0011 | -0.0155 | 0.0147 | False | nan | nan | nan |
| new_samples:panc1 | wp_ref - honest | 1 | 413 | 0.7422 | 0.709 | 0.0331 | 0.0168 | 0.0512 | True | nan | nan | nan |
| new_samples:panc1 | wp_self - honest | 1 | 413 | 0.6966 | 0.709 | -0.0124 | -0.0321 | 0.0057 | False | nan | nan | nan |
| new_samples:nk | wp_ref - honest | 1 | 413 | 0.7395 | 0.734 | 0.0055 | -0.02 | 0.031 | False | nan | nan | nan |
| new_samples:nk | wp_self - honest | 1 | 413 | 0.754 | 0.734 | 0.02 | -0.0088 | 0.047 | False | nan | nan | nan |
| new_samples:caco2 | wp_ref - honest | 1 | 413 | 0.8161 | 0.7591 | 0.057 | 0.0345 | 0.0746 | True | nan | nan | nan |
| new_samples:caco2 | wp_self - honest | 1 | 413 | 0.7992 | 0.7591 | 0.04 | 0.0144 | 0.0558 | True | nan | nan | nan |

## Primary endpoint

```
{
 "E1_wp_ref": {
  "diff": 0.027865535555124388,
  "lo": 0.0208783511788989,
  "hi": 0.03948964309844314,
  "above": true
 },
 "E2_wp_ref": {
  "diff": 0.029830000000000023,
  "lo": 0.018914750000000015,
  "hi": 0.03735124999999999,
  "above": true
 },
 "E1_wp_self": {
  "diff": -0.01970448183721929,
  "lo": -0.02697162478090484,
  "hi": -0.01129665438257057,
  "above": false
 },
 "E2_wp_self": {
  "diff": 0.00922999999999996,
  "lo": -0.0007951250000000964,
  "hi": 0.016640499999999985,
  "above": false
 },
 "met": false
}
```
