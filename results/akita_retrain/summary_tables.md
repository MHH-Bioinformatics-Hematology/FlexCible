**Split A chromosomes (2, 3, 11, 14, 17), the 12 training cells**

| method | cells | chromosomes | Pearson, median per 1 Mb region | SCC | insulation r | boundary recall | loop recall | APA ratio | anchor AUROC |
|---|---|---|---|---|---|---|---|---|---|
| ours refin_drop_chrho | 12 | 2, 3, 11, 14, 17 | 0.928 | 0.908 | 0.869 | 0.670 | 0.412 | 0.702 | 0.966 |
| ours honest_drop_chrho (ChIP only) | 12 | 2, 3, 11, 14, 17 | 0.789 | 0.745 | 0.702 | 0.504 | 0.208 | 0.280 | 0.910 |
| Akita retrained, split A | 12 | 2, 3, 11, 14, 17 | 0.461 | 0.446 | 0.509 | 0.339 | 0.077 | 0.194 | 0.796 |
| Akita released | 5 | 2, 3, 11, 14, 17 | 0.682 | 0.587 | 0.651 | 0.551 | 0.273 | 0.589 | 0.942 |
| consensus of the other cells | 12 | 2, 3, 11, 14, 17 | 0.893 | 0.851 | 0.799 | 0.663 | 0.419 | 0.804 | 0.962 |
| best other measured map | 12 | 2, 3, 11, 14, 17 | 0.864 | 0.806 | 0.755 | 0.578 | 0.322 | 0.662 | 0.944 |
| replicate map (ceiling) | 2 | 2, 3, 11, 14, 17 | 0.922 | 0.915 | 0.936 | 0.698 | 0.165 | 1.181 | 0.953 |

**Split A chromosomes, the 5 cells the released Akita has heads for**

| method | cells | chromosomes | Pearson, median per 1 Mb region | SCC | insulation r | boundary recall | loop recall | APA ratio | anchor AUROC |
|---|---|---|---|---|---|---|---|---|---|
| ours refin_drop_chrho | 5 | 2, 3, 11, 14, 17 | 0.942 | 0.916 | 0.871 | 0.684 | 0.463 | 0.766 | 0.974 |
| ours honest_drop_chrho (ChIP only) | 5 | 2, 3, 11, 14, 17 | 0.824 | 0.763 | 0.711 | 0.508 | 0.213 | 0.361 | 0.926 |
| Akita retrained, split A | 5 | 2, 3, 11, 14, 17 | 0.474 | 0.459 | 0.559 | 0.339 | 0.088 | 0.239 | 0.801 |
| Akita released | 5 | 2, 3, 11, 14, 17 | 0.682 | 0.587 | 0.651 | 0.551 | 0.273 | 0.589 | 0.942 |
| consensus of the other cells | 5 | 2, 3, 11, 14, 17 | 0.911 | 0.853 | 0.766 | 0.669 | 0.447 | 0.905 | 0.973 |
| best other measured map | 5 | 2, 3, 11, 14, 17 | 0.873 | 0.817 | 0.758 | 0.598 | 0.334 | 0.822 | 0.951 |
| replicate map (ceiling) | 2 | 2, 3, 11, 14, 17 | 0.922 | 0.915 | 0.936 | 0.698 | 0.165 | 1.181 | 0.953 |

**chr2 and chr14 (even; held out by split A, split B and our models), 12 cells**

| method | cells | chromosomes | Pearson, median per 1 Mb region | SCC | insulation r | boundary recall | loop recall | APA ratio | anchor AUROC |
|---|---|---|---|---|---|---|---|---|---|
| ours refin_drop_chrho | 12 | 2, 14 | 0.931 | 0.915 | 0.877 | 0.655 | 0.402 | 0.707 | 0.966 |
| ours honest_drop_chrho (ChIP only) | 12 | 2, 14 | 0.812 | 0.756 | 0.698 | 0.475 | 0.199 | 0.270 | 0.916 |
| Akita retrained, split A | 12 | 2, 14 | 0.437 | 0.438 | 0.534 | 0.313 | 0.069 | 0.193 | 0.789 |
| Akita retrained, split B | 12 | 2, 14 | 0.217 | 0.339 | 0.394 | 0.194 | 0.019 | 0.002 | 0.643 |
| Akita released | 5 | 2, 14 | 0.671 | 0.586 | 0.637 | 0.552 | 0.279 | 0.582 | 0.939 |
| consensus of the other cells | 12 | 2, 14 | 0.904 | 0.861 | 0.783 | 0.644 | 0.407 | 0.799 | 0.961 |
| best other measured map | 12 | 2, 14 | 0.863 | 0.818 | 0.759 | 0.556 | 0.303 | 0.648 | 0.944 |
| replicate map (ceiling) | 2 | 2, 14 | 0.922 | 0.914 | 0.933 | 0.670 | 0.161 | 1.176 | 0.948 |

**chr2 and chr14, the 5 cells of the released Akita**

| method | cells | chromosomes | Pearson, median per 1 Mb region | SCC | insulation r | boundary recall | loop recall | APA ratio | anchor AUROC |
|---|---|---|---|---|---|---|---|---|---|
| ours refin_drop_chrho | 5 | 2, 14 | 0.945 | 0.922 | 0.867 | 0.661 | 0.448 | 0.740 | 0.974 |
| ours honest_drop_chrho (ChIP only) | 5 | 2, 14 | 0.849 | 0.775 | 0.708 | 0.502 | 0.204 | 0.349 | 0.919 |
| Akita retrained, split A | 5 | 2, 14 | 0.437 | 0.456 | 0.565 | 0.321 | 0.070 | 0.241 | 0.792 |
| Akita retrained, split B | 5 | 2, 14 | 0.264 | 0.351 | 0.461 | 0.200 | 0.021 | 0.019 | 0.643 |
| Akita released | 5 | 2, 14 | 0.671 | 0.586 | 0.637 | 0.552 | 0.279 | 0.582 | 0.939 |
| consensus of the other cells | 5 | 2, 14 | 0.911 | 0.852 | 0.764 | 0.645 | 0.420 | 0.902 | 0.973 |
| best other measured map | 5 | 2, 14 | 0.872 | 0.822 | 0.772 | 0.577 | 0.322 | 0.834 | 0.952 |
| replicate map (ceiling) | 2 | 2, 14 | 0.922 | 0.914 | 0.933 | 0.670 | 0.161 | 1.176 | 0.948 |

**Split B test chromosomes 4, 6, 8, 10, 12, 16, 18, 20, 22 (our models trained on these)**

| method | cells | chromosomes | Pearson, median per 1 Mb region | SCC | insulation r | boundary recall | loop recall | APA ratio | anchor AUROC |
|---|---|---|---|---|---|---|---|---|---|
| Akita retrained, split B | 12 | 4, 6, 8, 10, 12, 16, 18, 20, 22 | 0.225 | 0.328 | 0.354 | 0.191 | 0.019 | -0.011 | 0.630 |
| consensus of the other cells | 12 | 4, 6, 8, 10, 12, 16, 18, 20, 22 | 0.889 | 0.849 | 0.821 | 0.656 | 0.416 | 0.781 | 0.963 |
| best other measured map | 12 | 4, 6, 8, 10, 12, 16, 18, 20, 22 | 0.851 | 0.795 | 0.726 | 0.578 | 0.312 | 0.664 | 0.942 |
| replicate map (ceiling) | 2 | 4, 6, 8, 10, 12, 16, 18, 20, 22 | 0.920 | 0.910 | 0.932 | 0.694 | 0.179 | 1.180 | 0.948 |

**Unseen cell K562 on chr2, 3, 11, 14, 17 (no Akita head exists)**

| method | cells | chromosomes | Pearson, median per 1 Mb region | SCC | insulation r | boundary recall | loop recall | APA ratio | anchor AUROC |
|---|---|---|---|---|---|---|---|---|---|
| ours refin_drop_chrho | 1 | 2, 3, 11, 14, 17 | 0.889 | 0.856 | 0.843 | 0.606 | 0.451 | 0.562 | 0.966 |
| ours honest_drop_chrho (ChIP only) | 1 | 2, 3, 11, 14, 17 | 0.738 | 0.706 | 0.712 | 0.473 | 0.196 | 0.247 | 0.900 |
| consensus of the other cells | 1 | 2, 3, 11, 14, 17 | 0.852 | 0.808 | 0.826 | 0.619 | 0.449 | 0.829 | 0.967 |
| best other measured map | 1 | 2, 3, 11, 14, 17 | 0.802 | 0.732 | 0.634 | 0.523 | 0.349 | 0.623 | 0.952 |
| replicate map (ceiling) | 1 | 2, 3, 11, 14, 17 | 0.912 | 0.861 | 0.786 | 0.663 | 0.370 | 1.297 | 0.967 |

