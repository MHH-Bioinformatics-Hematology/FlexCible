
### detection_rate

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.54 (0.51-0.60) | 0.54 (0.49-0.59) | 0.64 (0.57-0.70) | 0.46 (0.39-0.52) | 0.49 (0.39-0.59) | 0.43 (0.35-0.49) |
| consensus (AML reference map) | reference | 0.32 (0.30-0.35) | 0.27 (0.25-0.30) | 0.66 (0.61-0.74) | 0.26 (0.23-0.30) | 0.21 (0.15-0.28) | 0.29 (0.25-0.32) |
| consensus12 (12 cell lines, no AML) | reference | 0.29 (0.25-0.33) | 0.28 (0.23-0.31) | 0.33 (0.25-0.43) | 0.09 (0.07-0.11) | 0.11 (0.07-0.15) | 0.08 (0.06-0.09) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.42 (0.38-0.45) | 0.36 (0.33-0.39) | 0.77 (0.71-0.82) | 0.23 (0.17-0.30) | 0.24 (0.18-0.31) | 0.21 (0.16-0.28) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.32 (0.26-0.38) | 0.31 (0.25-0.37) | 0.33 (0.28-0.37) | 0.14 (0.11-0.17) | 0.19 (0.13-0.24) | 0.11 (0.07-0.15) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.33 (0.27-0.38) | 0.32 (0.26-0.38) | 0.34 (0.26-0.43) | 0.21 (0.15-0.26) | 0.22 (0.14-0.28) | 0.19 (0.13-0.24) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.27 (0.21-0.33) | 0.26 (0.20-0.32) | 0.32 (0.22-0.42) | 0.15 (0.12-0.20) | 0.17 (0.11-0.27) | 0.12 (0.10-0.15) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.38 (0.34-0.41) | 0.34 (0.31-0.37) | 0.61 (0.55-0.68) | 0.17 (0.13-0.21) | 0.21 (0.14-0.26) | 0.14 (0.11-0.18) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.32 (0.26-0.38) | 0.31 (0.24-0.37) | 0.34 (0.27-0.40) | 0.15 (0.12-0.18) | 0.19 (0.13-0.24) | 0.11 (0.08-0.15) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.18 (0.14-0.23) | 0.18 (0.13-0.22) | 0.15 (0.10-0.20) | 0.11 (0.09-0.13) | 0.13 (0.07-0.19) | 0.09 (0.06-0.13) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.16 (0.11-0.21) | 0.16 (0.12-0.20) | 0.13 (0.07-0.20) | 0.09 (0.07-0.13) | 0.12 (0.07-0.20) | 0.08 (0.05-0.12) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.19 (0.15-0.23) | 0.17 (0.13-0.21) | 0.34 (0.27-0.38) | 0.15 (0.11-0.18) | 0.13 (0.09-0.17) | 0.17 (0.13-0.20) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.09 (0.07-0.12) | 0.09 (0.07-0.12) | 0.07 (0.04-0.12) | 0.01 (0.01-0.02) | 0.03 (0.01-0.05) | 0.01 (0.00-0.01) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.12 (0.09-0.15) | 0.13 (0.10-0.15) | 0.12 (0.06-0.17) | 0.11 (0.07-0.16) | 0.11 (0.08-0.15) | 0.12 (0.07-0.18) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.12 (0.09-0.15) | 0.12 (0.10-0.15) | 0.11 (0.06-0.19) | 0.09 (0.05-0.13) | 0.08 (0.06-0.13) | 0.10 (0.05-0.15) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.40 (0.36-0.43) | 0.34 (0.31-0.38) | 0.71 (0.64-0.80) | 0.25 (0.20-0.30) | 0.26 (0.21-0.32) | 0.25 (0.20-0.29) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.43 (0.39-0.48) | 0.38 (0.34-0.42) | 0.71 (0.66-0.76) | 0.28 (0.22-0.33) | 0.28 (0.22-0.35) | 0.26 (0.20-0.33) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.42 (0.37-0.46) | 0.37 (0.33-0.41) | 0.68 (0.60-0.74) | 0.29 (0.24-0.33) | 0.26 (0.21-0.30) | 0.30 (0.24-0.36) |

### auroc_vs_B_rand

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.87 (0.86-0.88) | 0.87 (0.86-0.88) | 0.88 (0.86-0.90) | 0.77 (0.75-0.80) | 0.79 (0.75-0.83) | 0.75 (0.72-0.78) |
| consensus (AML reference map) | reference | 0.72 (0.71-0.74) | 0.69 (0.68-0.71) | 0.86 (0.84-0.88) | 0.62 (0.59-0.65) | 0.58 (0.53-0.62) | 0.64 (0.60-0.67) |
| consensus12 (12 cell lines, no AML) | reference | 0.70 (0.67-0.72) | 0.69 (0.67-0.71) | 0.65 (0.59-0.72) | 0.52 (0.48-0.56) | 0.52 (0.46-0.57) | 0.52 (0.48-0.56) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.76 (0.75-0.77) | 0.74 (0.73-0.75) | 0.86 (0.83-0.89) | 0.60 (0.57-0.63) | 0.61 (0.57-0.66) | 0.59 (0.56-0.62) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.61 (0.56-0.65) | 0.61 (0.56-0.66) | 0.56 (0.52-0.59) | 0.58 (0.56-0.61) | 0.61 (0.56-0.67) | 0.56 (0.53-0.60) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.64 (0.58-0.68) | 0.64 (0.59-0.68) | 0.62 (0.55-0.69) | 0.64 (0.60-0.68) | 0.65 (0.58-0.72) | 0.63 (0.59-0.68) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.59 (0.54-0.63) | 0.59 (0.53-0.64) | 0.57 (0.47-0.66) | 0.56 (0.54-0.59) | 0.59 (0.54-0.65) | 0.53 (0.51-0.56) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.75 (0.73-0.76) | 0.73 (0.71-0.74) | 0.81 (0.77-0.84) | 0.51 (0.47-0.54) | 0.54 (0.49-0.59) | 0.48 (0.44-0.52) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.63 (0.58-0.66) | 0.63 (0.57-0.68) | 0.55 (0.50-0.60) | 0.58 (0.55-0.61) | 0.61 (0.56-0.67) | 0.55 (0.51-0.59) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.50 (0.45-0.53) | 0.50 (0.46-0.54) | 0.40 (0.35-0.44) | 0.46 (0.43-0.50) | 0.52 (0.48-0.58) | 0.42 (0.38-0.48) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.45 (0.40-0.49) | 0.46 (0.41-0.50) | 0.35 (0.29-0.41) | 0.41 (0.37-0.45) | 0.48 (0.44-0.52) | 0.37 (0.33-0.42) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.56 (0.53-0.59) | 0.55 (0.52-0.57) | 0.63 (0.57-0.68) | 0.53 (0.51-0.56) | 0.52 (0.49-0.58) | 0.55 (0.52-0.57) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.44 (0.39-0.48) | 0.46 (0.41-0.50) | 0.27 (0.23-0.32) | 0.27 (0.25-0.29) | 0.34 (0.30-0.38) | 0.24 (0.22-0.26) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.49 (0.45-0.53) | 0.50 (0.45-0.54) | 0.46 (0.39-0.52) | 0.49 (0.45-0.54) | 0.52 (0.48-0.57) | 0.49 (0.42-0.55) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.48 (0.44-0.52) | 0.49 (0.45-0.54) | 0.44 (0.38-0.51) | 0.45 (0.40-0.51) | 0.49 (0.45-0.54) | 0.45 (0.38-0.51) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.76 (0.75-0.78) | 0.74 (0.73-0.75) | 0.86 (0.83-0.89) | 0.60 (0.56-0.63) | 0.61 (0.57-0.65) | 0.59 (0.55-0.64) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.78 (0.76-0.79) | 0.76 (0.74-0.77) | 0.87 (0.85-0.89) | 0.62 (0.58-0.67) | 0.63 (0.59-0.67) | 0.63 (0.57-0.67) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.77 (0.76-0.78) | 0.75 (0.73-0.76) | 0.88 (0.85-0.90) | 0.62 (0.58-0.66) | 0.61 (0.57-0.64) | 0.64 (0.58-0.69) |

### auroc_vs_A_refrand

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.87 (0.86-0.89) | 0.88 (0.86-0.90) | 0.88 (0.84-0.91) | 0.74 (0.70-0.77) | 0.77 (0.71-0.82) | 0.65 (0.60-0.70) |
| consensus (AML reference map) | reference | 0.58 (0.57-0.59) | 0.57 (0.56-0.58) | 0.65 (0.62-0.68) | 0.51 (0.49-0.53) | 0.47 (0.45-0.49) | 0.52 (0.48-0.56) |
| consensus12 (12 cell lines, no AML) | reference | 0.66 (0.64-0.69) | 0.66 (0.63-0.69) | 0.61 (0.54-0.68) | 0.59 (0.53-0.65) | 0.57 (0.46-0.66) | 0.57 (0.49-0.65) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.67 (0.64-0.68) | 0.66 (0.64-0.68) | 0.70 (0.64-0.76) | 0.57 (0.52-0.62) | 0.59 (0.52-0.64) | 0.53 (0.47-0.58) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.62 (0.57-0.66) | 0.61 (0.57-0.65) | 0.57 (0.49-0.66) | 0.57 (0.52-0.62) | 0.62 (0.53-0.71) | 0.51 (0.48-0.53) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.63 (0.59-0.67) | 0.63 (0.57-0.66) | 0.59 (0.50-0.68) | 0.61 (0.56-0.64) | 0.66 (0.57-0.75) | 0.54 (0.49-0.59) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.59 (0.55-0.63) | 0.59 (0.54-0.63) | 0.60 (0.47-0.73) | 0.54 (0.48-0.59) | 0.60 (0.51-0.71) | 0.45 (0.40-0.50) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.67 (0.65-0.69) | 0.67 (0.64-0.68) | 0.68 (0.61-0.74) | 0.59 (0.53-0.64) | 0.59 (0.53-0.65) | 0.55 (0.50-0.62) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.62 (0.57-0.66) | 0.61 (0.56-0.65) | 0.54 (0.47-0.65) | 0.56 (0.52-0.61) | 0.61 (0.54-0.68) | 0.49 (0.42-0.54) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.52 (0.49-0.55) | 0.52 (0.48-0.54) | 0.47 (0.37-0.59) | 0.48 (0.42-0.52) | 0.53 (0.44-0.63) | 0.44 (0.40-0.48) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.49 (0.46-0.51) | 0.48 (0.45-0.50) | 0.47 (0.38-0.56) | 0.47 (0.42-0.51) | 0.49 (0.41-0.56) | 0.46 (0.39-0.50) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.54 (0.51-0.58) | 0.54 (0.50-0.57) | 0.58 (0.51-0.66) | 0.55 (0.49-0.60) | 0.56 (0.46-0.67) | 0.59 (0.50-0.68) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.52 (0.50-0.53) | 0.52 (0.49-0.54) | 0.43 (0.36-0.50) | 0.52 (0.47-0.56) | 0.54 (0.49-0.59) | 0.48 (0.40-0.56) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.51 (0.49-0.54) | 0.51 (0.48-0.55) | 0.46 (0.39-0.56) | 0.46 (0.41-0.49) | 0.51 (0.46-0.57) | 0.47 (0.42-0.52) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.51 (0.48-0.53) | 0.51 (0.48-0.54) | 0.46 (0.40-0.54) | 0.44 (0.41-0.47) | 0.51 (0.46-0.58) | 0.46 (0.41-0.51) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.69 (0.67-0.70) | 0.68 (0.67-0.69) | 0.75 (0.69-0.80) | 0.58 (0.56-0.60) | 0.58 (0.53-0.62) | 0.57 (0.53-0.59) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.70 (0.68-0.71) | 0.69 (0.67-0.71) | 0.72 (0.66-0.78) | 0.58 (0.53-0.62) | 0.57 (0.52-0.61) | 0.56 (0.50-0.62) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.68 (0.67-0.69) | 0.68 (0.66-0.69) | 0.71 (0.67-0.75) | 0.56 (0.53-0.60) | 0.55 (0.50-0.58) | 0.56 (0.52-0.60) |

### spearman_level_vs_measured

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| consensus (AML reference map) | reference | 0.29 (0.23-0.36) | 0.25 (0.19-0.32) | 0.46 (0.36-0.56) | 0.46 (0.36-0.56) | 0.44 (0.27-0.60) | 0.48 (0.40-0.56) |
| consensus12 (12 cell lines, no AML) | reference | 0.22 (0.15-0.30) | 0.25 (0.17-0.33) | 0.09 (-0.02-0.20) | 0.32 (0.18-0.44) | 0.32 (0.20-0.43) | 0.31 (0.15-0.47) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.30 (0.22-0.38) | 0.27 (0.19-0.35) | 0.43 (0.32-0.55) | 0.49 (0.43-0.55) | 0.46 (0.36-0.57) | 0.48 (0.41-0.56) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.09 (0.03-0.14) | 0.10 (0.03-0.15) | 0.09 (-0.05-0.21) | 0.23 (0.17-0.28) | 0.27 (0.15-0.38) | 0.15 (0.10-0.20) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.10 (0.05-0.15) | 0.10 (0.05-0.15) | 0.19 (0.02-0.31) | 0.23 (0.14-0.35) | 0.29 (0.14-0.42) | 0.18 (0.08-0.28) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.09 (0.05-0.13) | 0.09 (0.05-0.14) | 0.13 (-0.06-0.27) | 0.23 (0.17-0.27) | 0.26 (0.17-0.34) | 0.19 (0.14-0.25) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.29 (0.20-0.37) | 0.26 (0.17-0.34) | 0.41 (0.29-0.54) | 0.51 (0.43-0.57) | 0.47 (0.40-0.58) | 0.49 (0.41-0.57) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.07 (0.02-0.12) | 0.06 (0.01-0.12) | 0.16 (-0.00-0.31) | 0.27 (0.17-0.34) | 0.31 (0.21-0.43) | 0.21 (0.10-0.28) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.04 (0.01-0.07) | 0.05 (0.01-0.08) | 0.00 (-0.09-0.13) | 0.06 (-0.01-0.13) | 0.09 (-0.05-0.23) | 0.02 (-0.07-0.10) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.04 (-0.01-0.07) | 0.06 (0.00-0.09) | -0.04 (-0.16-0.10) | 0.02 (-0.03-0.08) | 0.07 (-0.09-0.23) | -0.04 (-0.09-0.03) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.01 (-0.06-0.07) | 0.01 (-0.04-0.07) | -0.02 (-0.21-0.11) | 0.02 (-0.04-0.10) | 0.00 (-0.09-0.12) | -0.00 (-0.08-0.11) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.09 (0.04-0.13) | 0.10 (0.04-0.16) | 0.12 (0.01-0.23) | 0.10 (0.02-0.19) | 0.09 (-0.04-0.22) | 0.07 (-0.04-0.18) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.05 (-0.01-0.09) | 0.04 (-0.00-0.07) | 0.07 (-0.12-0.24) | 0.00 (-0.08-0.10) | -0.07 (-0.20-0.11) | -0.01 (-0.10-0.09) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.05 (-0.01-0.09) | 0.04 (-0.01-0.07) | 0.06 (-0.14-0.21) | -0.01 (-0.08-0.11) | -0.08 (-0.22-0.09) | -0.01 (-0.10-0.09) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.28 (0.20-0.35) | 0.25 (0.18-0.35) | 0.34 (0.23-0.45) | 0.50 (0.43-0.59) | 0.50 (0.40-0.61) | 0.49 (0.37-0.59) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.26 (0.17-0.35) | 0.24 (0.14-0.35) | 0.37 (0.26-0.47) | 0.49 (0.37-0.60) | 0.47 (0.35-0.59) | 0.48 (0.37-0.60) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.26 (0.17-0.33) | 0.23 (0.16-0.31) | 0.40 (0.29-0.51) | 0.49 (0.38-0.58) | 0.46 (0.33-0.58) | 0.49 (0.38-0.61) |

### detection_rate | C.Origami patients

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.55 (0.51-0.61) | 0.55 (0.48-0.61) | 0.63 (0.56-0.70) | 0.46 (0.40-0.52) | 0.50 (0.36-0.60) | 0.43 (0.34-0.51) |
| consensus (AML reference map) | reference | 0.33 (0.30-0.36) | 0.27 (0.24-0.30) | 0.64 (0.59-0.72) | 0.25 (0.22-0.29) | 0.21 (0.14-0.29) | 0.27 (0.24-0.32) |
| consensus12 (12 cell lines, no AML) | reference | 0.30 (0.27-0.34) | 0.29 (0.26-0.32) | 0.29 (0.22-0.39) | 0.09 (0.07-0.11) | 0.12 (0.08-0.17) | 0.08 (0.06-0.10) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.43 (0.39-0.45) | 0.37 (0.33-0.40) | 0.76 (0.69-0.82) | 0.26 (0.20-0.31) | 0.26 (0.19-0.34) | 0.25 (0.18-0.30) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.33 (0.27-0.39) | 0.32 (0.25-0.38) | 0.31 (0.26-0.37) | 0.14 (0.11-0.18) | 0.18 (0.12-0.24) | 0.11 (0.08-0.17) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.35 (0.29-0.39) | 0.35 (0.28-0.40) | 0.31 (0.23-0.40) | 0.21 (0.16-0.27) | 0.21 (0.13-0.29) | 0.20 (0.15-0.26) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.29 (0.23-0.34) | 0.28 (0.23-0.34) | 0.27 (0.20-0.34) | 0.15 (0.11-0.20) | 0.17 (0.10-0.27) | 0.12 (0.10-0.15) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.39 (0.37-0.41) | 0.35 (0.33-0.37) | 0.59 (0.53-0.64) | 0.18 (0.14-0.23) | 0.22 (0.14-0.28) | 0.16 (0.13-0.20) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.34 (0.29-0.40) | 0.34 (0.28-0.39) | 0.31 (0.25-0.37) | 0.15 (0.11-0.18) | 0.18 (0.12-0.25) | 0.12 (0.08-0.15) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.20 (0.14-0.24) | 0.20 (0.15-0.24) | 0.14 (0.10-0.20) | 0.12 (0.10-0.13) | 0.13 (0.07-0.21) | 0.10 (0.07-0.13) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.17 (0.12-0.22) | 0.18 (0.12-0.22) | 0.15 (0.09-0.23) | 0.10 (0.08-0.13) | 0.13 (0.07-0.22) | 0.08 (0.05-0.12) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.19 (0.15-0.23) | 0.17 (0.13-0.20) | 0.34 (0.27-0.39) | 0.15 (0.11-0.18) | 0.13 (0.08-0.16) | 0.17 (0.13-0.20) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.09 (0.08-0.12) | 0.09 (0.08-0.12) | 0.07 (0.04-0.12) | 0.01 (0.01-0.03) | 0.03 (0.01-0.05) | 0.01 (0.00-0.01) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.12 (0.09-0.15) | 0.13 (0.10-0.15) | 0.12 (0.06-0.17) | 0.11 (0.07-0.16) | 0.11 (0.08-0.15) | 0.12 (0.07-0.17) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.12 (0.09-0.15) | 0.12 (0.10-0.15) | 0.11 (0.07-0.19) | 0.09 (0.06-0.13) | 0.08 (0.05-0.12) | 0.10 (0.05-0.14) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.41 (0.37-0.44) | 0.35 (0.32-0.39) | 0.70 (0.61-0.79) | 0.27 (0.22-0.32) | 0.27 (0.22-0.34) | 0.27 (0.22-0.31) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.44 (0.41-0.49) | 0.39 (0.35-0.43) | 0.69 (0.65-0.75) | 0.29 (0.23-0.35) | 0.28 (0.22-0.35) | 0.28 (0.22-0.35) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.41 (0.37-0.45) | 0.36 (0.33-0.41) | 0.66 (0.57-0.73) | 0.30 (0.25-0.34) | 0.26 (0.21-0.31) | 0.31 (0.25-0.38) |

### detection_rate | Epiphany coverage (reach 990 kb)

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.55 (0.51-0.61) | 0.55 (0.49-0.61) | 0.63 (0.56-0.70) | 0.46 (0.39-0.53) | 0.50 (0.36-0.60) | 0.43 (0.33-0.51) |
| consensus (AML reference map) | reference | 0.33 (0.30-0.36) | 0.27 (0.24-0.30) | 0.64 (0.57-0.71) | 0.26 (0.23-0.30) | 0.21 (0.14-0.29) | 0.28 (0.25-0.32) |
| consensus12 (12 cell lines, no AML) | reference | 0.30 (0.27-0.34) | 0.29 (0.26-0.32) | 0.29 (0.23-0.39) | 0.09 (0.07-0.11) | 0.12 (0.08-0.17) | 0.08 (0.06-0.09) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.43 (0.39-0.45) | 0.37 (0.33-0.40) | 0.76 (0.69-0.81) | 0.26 (0.20-0.31) | 0.27 (0.20-0.34) | 0.25 (0.19-0.30) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.33 (0.27-0.39) | 0.33 (0.25-0.39) | 0.31 (0.26-0.36) | 0.14 (0.11-0.18) | 0.18 (0.12-0.25) | 0.11 (0.07-0.16) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.35 (0.29-0.39) | 0.35 (0.28-0.40) | 0.31 (0.23-0.41) | 0.20 (0.16-0.26) | 0.21 (0.14-0.29) | 0.18 (0.14-0.24) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.29 (0.24-0.34) | 0.28 (0.23-0.34) | 0.27 (0.20-0.34) | 0.15 (0.11-0.20) | 0.17 (0.08-0.26) | 0.12 (0.10-0.15) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.39 (0.37-0.41) | 0.35 (0.33-0.37) | 0.58 (0.53-0.64) | 0.19 (0.14-0.23) | 0.22 (0.15-0.29) | 0.17 (0.14-0.20) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.34 (0.29-0.39) | 0.34 (0.28-0.39) | 0.31 (0.24-0.36) | 0.14 (0.10-0.17) | 0.17 (0.10-0.24) | 0.11 (0.08-0.14) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.20 (0.15-0.24) | 0.20 (0.15-0.24) | 0.14 (0.09-0.20) | 0.12 (0.10-0.13) | 0.12 (0.07-0.21) | 0.10 (0.06-0.13) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.17 (0.13-0.23) | 0.17 (0.12-0.22) | 0.15 (0.09-0.23) | 0.10 (0.08-0.14) | 0.13 (0.07-0.22) | 0.08 (0.05-0.13) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.19 (0.15-0.23) | 0.17 (0.13-0.21) | 0.33 (0.27-0.39) | 0.14 (0.11-0.18) | 0.12 (0.08-0.17) | 0.16 (0.12-0.19) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.09 (0.08-0.12) | 0.09 (0.08-0.12) | 0.07 (0.04-0.12) | 0.02 (0.01-0.03) | 0.03 (0.01-0.05) | 0.01 (0.00-0.01) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.12 (0.10-0.15) | 0.13 (0.10-0.15) | 0.12 (0.06-0.17) | 0.11 (0.07-0.15) | 0.11 (0.08-0.14) | 0.12 (0.07-0.17) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.12 (0.10-0.15) | 0.12 (0.10-0.15) | 0.11 (0.07-0.19) | 0.09 (0.05-0.13) | 0.08 (0.05-0.12) | 0.10 (0.05-0.15) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.41 (0.37-0.44) | 0.35 (0.32-0.39) | 0.70 (0.61-0.79) | 0.27 (0.23-0.32) | 0.27 (0.22-0.35) | 0.27 (0.22-0.31) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.44 (0.41-0.48) | 0.39 (0.36-0.43) | 0.69 (0.64-0.75) | 0.28 (0.23-0.35) | 0.27 (0.22-0.35) | 0.27 (0.21-0.36) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.41 (0.37-0.46) | 0.36 (0.33-0.41) | 0.66 (0.58-0.73) | 0.30 (0.24-0.35) | 0.26 (0.21-0.31) | 0.32 (0.25-0.37) |
