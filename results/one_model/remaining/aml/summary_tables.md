
### detection_rate

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.61 (0.55-0.68) | 0.58 (0.52-0.66) | 0.85 (0.81-0.90) | 0.76 (0.72-0.81) | 0.64 (0.56-0.71) | 0.83 (0.77-0.86) |
| consensus (AML reference map) | reference | 0.44 (0.35-0.52) | 0.37 (0.29-0.47) | 0.84 (0.80-0.89) | 0.68 (0.63-0.76) | 0.48 (0.41-0.61) | 0.80 (0.74-0.86) |
| consensus12 (12 cell lines, no AML) | reference | 0.38 (0.29-0.48) | 0.34 (0.26-0.44) | 0.65 (0.54-0.72) | 0.55 (0.47-0.61) | 0.41 (0.33-0.48) | 0.62 (0.53-0.69) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.48 (0.39-0.56) | 0.42 (0.34-0.50) | 0.85 (0.80-0.91) | 0.67 (0.63-0.73) | 0.49 (0.43-0.59) | 0.77 (0.74-0.83) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.42 (0.34-0.50) | 0.38 (0.31-0.46) | 0.64 (0.57-0.74) | 0.56 (0.49-0.62) | 0.44 (0.35-0.51) | 0.62 (0.54-0.69) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.42 (0.34-0.52) | 0.38 (0.29-0.47) | 0.65 (0.58-0.73) | 0.57 (0.49-0.65) | 0.44 (0.36-0.53) | 0.63 (0.54-0.72) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.40 (0.32-0.50) | 0.37 (0.29-0.47) | 0.62 (0.52-0.71) | 0.52 (0.46-0.59) | 0.41 (0.33-0.51) | 0.58 (0.50-0.64) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.49 (0.38-0.60) | 0.43 (0.35-0.52) | 0.85 (0.79-0.91) | 0.69 (0.65-0.73) | 0.52 (0.46-0.56) | 0.79 (0.74-0.83) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.42 (0.33-0.51) | 0.38 (0.30-0.46) | 0.63 (0.53-0.73) | 0.54 (0.46-0.61) | 0.43 (0.33-0.52) | 0.59 (0.49-0.67) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.30 (0.24-0.35) | 0.28 (0.23-0.33) | 0.37 (0.29-0.44) | 0.29 (0.24-0.35) | 0.27 (0.22-0.32) | 0.30 (0.23-0.37) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.29 (0.25-0.33) | 0.27 (0.23-0.31) | 0.35 (0.28-0.43) | 0.25 (0.19-0.32) | 0.23 (0.17-0.31) | 0.25 (0.18-0.33) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.21 (0.19-0.24) | 0.19 (0.16-0.22) | 0.36 (0.30-0.41) | 0.19 (0.14-0.21) | 0.14 (0.10-0.19) | 0.22 (0.18-0.24) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.13 (0.11-0.14) | 0.13 (0.11-0.14) | 0.12 (0.06-0.17) | 0.17 (0.14-0.20) | 0.16 (0.10-0.20) | 0.19 (0.15-0.22) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.19 (0.14-0.23) | 0.19 (0.15-0.22) | 0.18 (0.07-0.29) | 0.21 (0.16-0.27) | 0.20 (0.14-0.27) | 0.26 (0.16-0.34) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.18 (0.14-0.22) | 0.18 (0.15-0.22) | 0.19 (0.10-0.30) | 0.22 (0.16-0.29) | 0.20 (0.14-0.26) | 0.28 (0.16-0.38) |
| Akita (sequence only, GM12878 head) | competitors (published models, no AML in training) | 0.34 (0.26-0.43) | 0.31 (0.23-0.40) | 0.56 (0.49-0.60) | 0.53 (0.47-0.57) | 0.35 (0.27-0.45) | 0.65 (0.55-0.70) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.49 (0.40-0.61) | 0.42 (0.33-0.54) | 0.89 (0.84-0.93) | 0.70 (0.67-0.74) | 0.53 (0.47-0.59) | 0.80 (0.76-0.83) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.51 (0.40-0.61) | 0.44 (0.34-0.55) | 0.88 (0.85-0.92) | 0.69 (0.66-0.74) | 0.51 (0.46-0.57) | 0.79 (0.76-0.84) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.48 (0.38-0.58) | 0.42 (0.32-0.53) | 0.89 (0.84-0.93) | 0.71 (0.64-0.74) | 0.52 (0.42-0.59) | 0.81 (0.76-0.84) |

### auroc_vs_B_rand

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.89 (0.87-0.91) | 0.88 (0.87-0.90) | 0.95 (0.94-0.96) | 0.92 (0.90-0.93) | 0.88 (0.85-0.90) | 0.94 (0.92-0.95) |
| consensus (AML reference map) | reference | 0.76 (0.72-0.81) | 0.73 (0.68-0.78) | 0.94 (0.92-0.96) | 0.88 (0.86-0.90) | 0.80 (0.76-0.83) | 0.93 (0.92-0.94) |
| consensus12 (12 cell lines, no AML) | reference | 0.76 (0.70-0.81) | 0.74 (0.68-0.78) | 0.89 (0.87-0.91) | 0.84 (0.81-0.86) | 0.75 (0.71-0.79) | 0.88 (0.85-0.90) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.80 (0.75-0.84) | 0.77 (0.72-0.81) | 0.95 (0.93-0.96) | 0.89 (0.87-0.90) | 0.81 (0.77-0.85) | 0.92 (0.91-0.94) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.78 (0.73-0.83) | 0.76 (0.71-0.81) | 0.90 (0.88-0.92) | 0.84 (0.81-0.87) | 0.77 (0.72-0.82) | 0.88 (0.85-0.91) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.78 (0.73-0.82) | 0.76 (0.71-0.81) | 0.90 (0.88-0.93) | 0.85 (0.82-0.87) | 0.78 (0.72-0.82) | 0.89 (0.86-0.91) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.78 (0.72-0.82) | 0.75 (0.70-0.80) | 0.89 (0.87-0.92) | 0.83 (0.80-0.85) | 0.76 (0.71-0.80) | 0.87 (0.84-0.89) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.80 (0.75-0.84) | 0.77 (0.72-0.82) | 0.94 (0.93-0.96) | 0.89 (0.87-0.90) | 0.82 (0.77-0.85) | 0.92 (0.91-0.94) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.78 (0.73-0.83) | 0.76 (0.71-0.80) | 0.90 (0.88-0.92) | 0.84 (0.81-0.86) | 0.77 (0.73-0.81) | 0.87 (0.85-0.90) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.67 (0.64-0.70) | 0.66 (0.63-0.68) | 0.70 (0.67-0.73) | 0.68 (0.65-0.72) | 0.67 (0.62-0.71) | 0.68 (0.64-0.72) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.66 (0.63-0.69) | 0.65 (0.61-0.68) | 0.70 (0.66-0.73) | 0.66 (0.63-0.70) | 0.65 (0.60-0.70) | 0.66 (0.61-0.71) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.60 (0.58-0.62) | 0.59 (0.57-0.61) | 0.67 (0.62-0.72) | 0.60 (0.58-0.62) | 0.57 (0.54-0.61) | 0.63 (0.62-0.64) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.53 (0.51-0.55) | 0.53 (0.51-0.55) | 0.51 (0.47-0.53) | 0.60 (0.56-0.63) | 0.60 (0.55-0.63) | 0.61 (0.56-0.65) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.56 (0.51-0.59) | 0.56 (0.52-0.60) | 0.57 (0.51-0.64) | 0.62 (0.57-0.68) | 0.63 (0.59-0.68) | 0.63 (0.54-0.71) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.55 (0.51-0.59) | 0.55 (0.52-0.59) | 0.57 (0.50-0.63) | 0.61 (0.56-0.67) | 0.63 (0.60-0.68) | 0.63 (0.55-0.71) |
| Akita (sequence only, GM12878 head) | competitors (published models, no AML in training) | 0.74 (0.68-0.79) | 0.72 (0.67-0.77) | 0.82 (0.79-0.85) | 0.84 (0.80-0.86) | 0.76 (0.70-0.82) | 0.88 (0.86-0.90) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.81 (0.76-0.85) | 0.78 (0.74-0.82) | 0.95 (0.93-0.96) | 0.89 (0.87-0.90) | 0.81 (0.78-0.84) | 0.93 (0.91-0.94) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.81 (0.76-0.85) | 0.78 (0.73-0.83) | 0.95 (0.93-0.96) | 0.89 (0.87-0.90) | 0.81 (0.78-0.83) | 0.92 (0.91-0.93) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.80 (0.75-0.85) | 0.78 (0.72-0.83) | 0.95 (0.93-0.96) | 0.89 (0.87-0.90) | 0.81 (0.77-0.83) | 0.93 (0.91-0.94) |

### auroc_vs_A_refrand

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.85 (0.83-0.87) | 0.85 (0.83-0.87) | 0.80 (0.76-0.83) | 0.67 (0.64-0.70) | 0.71 (0.66-0.75) | 0.63 (0.58-0.67) |
| consensus (AML reference map) | reference | 0.55 (0.54-0.56) | 0.55 (0.53-0.56) | 0.57 (0.55-0.59) | 0.51 (0.50-0.53) | 0.50 (0.49-0.51) | 0.53 (0.52-0.54) |
| consensus12 (12 cell lines, no AML) | reference | 0.63 (0.60-0.66) | 0.63 (0.60-0.66) | 0.56 (0.50-0.62) | 0.57 (0.52-0.62) | 0.56 (0.47-0.63) | 0.57 (0.49-0.65) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.64 (0.61-0.66) | 0.63 (0.60-0.65) | 0.60 (0.56-0.65) | 0.55 (0.52-0.58) | 0.57 (0.53-0.60) | 0.54 (0.51-0.57) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.67 (0.64-0.70) | 0.67 (0.63-0.70) | 0.58 (0.50-0.66) | 0.59 (0.54-0.63) | 0.60 (0.52-0.67) | 0.57 (0.48-0.66) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.66 (0.64-0.69) | 0.66 (0.63-0.69) | 0.57 (0.52-0.65) | 0.60 (0.55-0.65) | 0.61 (0.52-0.68) | 0.59 (0.51-0.68) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.66 (0.62-0.68) | 0.66 (0.62-0.68) | 0.57 (0.49-0.66) | 0.56 (0.51-0.61) | 0.59 (0.51-0.66) | 0.53 (0.47-0.61) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.65 (0.62-0.67) | 0.64 (0.61-0.66) | 0.60 (0.55-0.65) | 0.56 (0.53-0.59) | 0.58 (0.54-0.62) | 0.56 (0.52-0.60) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.66 (0.63-0.69) | 0.66 (0.62-0.69) | 0.56 (0.48-0.64) | 0.59 (0.53-0.64) | 0.60 (0.53-0.67) | 0.57 (0.49-0.66) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.60 (0.57-0.63) | 0.60 (0.56-0.63) | 0.53 (0.47-0.61) | 0.51 (0.45-0.57) | 0.57 (0.49-0.64) | 0.45 (0.37-0.56) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.59 (0.56-0.62) | 0.59 (0.56-0.62) | 0.57 (0.52-0.65) | 0.51 (0.46-0.56) | 0.55 (0.48-0.61) | 0.45 (0.37-0.56) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.56 (0.54-0.58) | 0.56 (0.53-0.59) | 0.59 (0.52-0.66) | 0.53 (0.48-0.58) | 0.54 (0.48-0.63) | 0.54 (0.44-0.64) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.51 (0.49-0.54) | 0.51 (0.48-0.54) | 0.42 (0.33-0.52) | 0.52 (0.46-0.59) | 0.57 (0.51-0.64) | 0.48 (0.39-0.58) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.53 (0.49-0.56) | 0.53 (0.49-0.56) | 0.45 (0.39-0.52) | 0.49 (0.46-0.53) | 0.55 (0.47-0.63) | 0.49 (0.41-0.56) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.52 (0.49-0.55) | 0.53 (0.49-0.56) | 0.44 (0.39-0.51) | 0.50 (0.45-0.54) | 0.56 (0.48-0.63) | 0.50 (0.40-0.61) |
| Akita (sequence only, GM12878 head) | competitors (published models, no AML in training) | 0.63 (0.60-0.67) | 0.63 (0.59-0.67) | 0.56 (0.50-0.63) | 0.68 (0.64-0.72) | 0.65 (0.58-0.74) | 0.74 (0.68-0.81) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.67 (0.64-0.68) | 0.66 (0.63-0.68) | 0.64 (0.60-0.69) | 0.55 (0.53-0.57) | 0.55 (0.53-0.58) | 0.55 (0.53-0.58) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.67 (0.64-0.69) | 0.66 (0.63-0.68) | 0.62 (0.58-0.67) | 0.56 (0.53-0.59) | 0.56 (0.53-0.59) | 0.57 (0.53-0.61) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.66 (0.63-0.67) | 0.65 (0.62-0.67) | 0.62 (0.58-0.65) | 0.54 (0.52-0.56) | 0.54 (0.51-0.55) | 0.55 (0.53-0.57) |

### spearman_level_vs_measured

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| consensus (AML reference map) | reference | 0.56 (0.49-0.65) | 0.51 (0.44-0.60) | 0.70 (0.61-0.77) | 0.79 (0.73-0.85) | 0.76 (0.68-0.85) | 0.80 (0.75-0.84) |
| consensus12 (12 cell lines, no AML) | reference | 0.51 (0.41-0.57) | 0.49 (0.40-0.59) | 0.40 (0.31-0.52) | 0.70 (0.61-0.77) | 0.71 (0.64-0.77) | 0.66 (0.55-0.76) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.56 (0.47-0.64) | 0.52 (0.43-0.59) | 0.64 (0.55-0.71) | 0.81 (0.75-0.85) | 0.77 (0.71-0.84) | 0.81 (0.76-0.86) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.49 (0.41-0.57) | 0.48 (0.39-0.57) | 0.36 (0.22-0.51) | 0.71 (0.62-0.78) | 0.71 (0.65-0.78) | 0.68 (0.57-0.78) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.51 (0.43-0.58) | 0.49 (0.41-0.58) | 0.39 (0.27-0.50) | 0.70 (0.62-0.78) | 0.70 (0.64-0.78) | 0.67 (0.56-0.76) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.51 (0.43-0.59) | 0.49 (0.40-0.60) | 0.36 (0.24-0.49) | 0.70 (0.62-0.77) | 0.70 (0.64-0.77) | 0.69 (0.58-0.79) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.55 (0.45-0.63) | 0.50 (0.42-0.59) | 0.63 (0.50-0.72) | 0.81 (0.76-0.87) | 0.77 (0.72-0.84) | 0.82 (0.75-0.87) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.49 (0.40-0.56) | 0.48 (0.38-0.57) | 0.36 (0.23-0.49) | 0.70 (0.60-0.77) | 0.70 (0.63-0.78) | 0.68 (0.57-0.79) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.20 (0.13-0.25) | 0.20 (0.14-0.25) | -0.01 (-0.16-0.12) | 0.42 (0.34-0.50) | 0.45 (0.34-0.55) | 0.44 (0.33-0.53) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.21 (0.15-0.27) | 0.23 (0.16-0.27) | 0.00 (-0.17-0.15) | 0.39 (0.31-0.48) | 0.40 (0.30-0.53) | 0.43 (0.31-0.53) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.10 (0.05-0.14) | 0.09 (0.03-0.15) | 0.02 (-0.07-0.11) | 0.14 (0.09-0.19) | 0.20 (0.15-0.28) | 0.09 (0.01-0.18) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.05 (-0.02-0.12) | 0.05 (-0.03-0.12) | 0.10 (-0.07-0.26) | 0.09 (0.03-0.15) | 0.07 (-0.02-0.18) | 0.08 (0.03-0.14) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.12 (0.08-0.16) | 0.10 (0.05-0.17) | 0.13 (-0.13-0.35) | 0.13 (0.03-0.24) | 0.10 (-0.08-0.27) | 0.17 (0.09-0.26) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.11 (0.07-0.16) | 0.10 (0.05-0.17) | 0.13 (-0.13-0.32) | 0.12 (0.02-0.26) | 0.09 (-0.07-0.27) | 0.16 (0.07-0.26) |
| Akita (sequence only, GM12878 head) | competitors (published models, no AML in training) | 0.41 (0.34-0.49) | 0.40 (0.31-0.48) | 0.23 (0.06-0.39) | 0.33 (0.22-0.45) | 0.41 (0.26-0.53) | 0.22 (0.10-0.35) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.55 (0.46-0.62) | 0.50 (0.42-0.57) | 0.61 (0.51-0.69) | 0.80 (0.76-0.84) | 0.78 (0.73-0.83) | 0.80 (0.74-0.85) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.55 (0.46-0.63) | 0.50 (0.42-0.58) | 0.64 (0.54-0.71) | 0.79 (0.74-0.85) | 0.77 (0.72-0.84) | 0.79 (0.72-0.85) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.55 (0.48-0.63) | 0.50 (0.42-0.58) | 0.65 (0.55-0.74) | 0.80 (0.74-0.85) | 0.77 (0.71-0.84) | 0.79 (0.73-0.86) |

### detection_rate | C.Origami patients

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.63 (0.56-0.70) | 0.61 (0.54-0.68) | 0.86 (0.81-0.91) | 0.77 (0.73-0.82) | 0.66 (0.59-0.73) | 0.84 (0.78-0.88) |
| consensus (AML reference map) | reference | 0.43 (0.33-0.55) | 0.37 (0.26-0.47) | 0.84 (0.79-0.89) | 0.68 (0.63-0.77) | 0.49 (0.44-0.63) | 0.80 (0.74-0.86) |
| consensus12 (12 cell lines, no AML) | reference | 0.38 (0.29-0.49) | 0.35 (0.25-0.46) | 0.58 (0.51-0.71) | 0.55 (0.47-0.62) | 0.43 (0.33-0.51) | 0.62 (0.53-0.70) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.48 (0.36-0.57) | 0.43 (0.33-0.52) | 0.86 (0.78-0.92) | 0.69 (0.65-0.76) | 0.52 (0.47-0.61) | 0.79 (0.75-0.83) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.43 (0.34-0.52) | 0.40 (0.30-0.47) | 0.61 (0.55-0.69) | 0.57 (0.49-0.65) | 0.46 (0.37-0.53) | 0.64 (0.54-0.72) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.43 (0.33-0.54) | 0.40 (0.31-0.50) | 0.63 (0.56-0.70) | 0.57 (0.50-0.66) | 0.46 (0.36-0.55) | 0.64 (0.55-0.74) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.41 (0.32-0.51) | 0.39 (0.29-0.51) | 0.58 (0.47-0.68) | 0.52 (0.47-0.59) | 0.43 (0.35-0.52) | 0.58 (0.50-0.67) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.49 (0.38-0.61) | 0.44 (0.33-0.55) | 0.85 (0.76-0.91) | 0.71 (0.67-0.76) | 0.54 (0.49-0.61) | 0.81 (0.76-0.85) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.42 (0.33-0.53) | 0.40 (0.31-0.49) | 0.59 (0.51-0.69) | 0.54 (0.47-0.63) | 0.45 (0.35-0.53) | 0.60 (0.51-0.69) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.30 (0.25-0.36) | 0.29 (0.24-0.34) | 0.34 (0.26-0.44) | 0.31 (0.26-0.37) | 0.27 (0.22-0.35) | 0.33 (0.25-0.41) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.29 (0.25-0.34) | 0.28 (0.23-0.31) | 0.37 (0.29-0.43) | 0.27 (0.22-0.34) | 0.24 (0.18-0.32) | 0.28 (0.21-0.35) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.21 (0.19-0.24) | 0.19 (0.16-0.22) | 0.36 (0.30-0.41) | 0.19 (0.15-0.22) | 0.14 (0.10-0.20) | 0.22 (0.18-0.24) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.13 (0.12-0.14) | 0.13 (0.11-0.14) | 0.12 (0.07-0.17) | 0.17 (0.14-0.20) | 0.16 (0.10-0.20) | 0.19 (0.16-0.22) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.19 (0.14-0.23) | 0.19 (0.15-0.23) | 0.18 (0.08-0.29) | 0.21 (0.16-0.27) | 0.20 (0.14-0.26) | 0.26 (0.17-0.35) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.18 (0.14-0.22) | 0.18 (0.15-0.22) | 0.19 (0.10-0.32) | 0.22 (0.17-0.29) | 0.20 (0.15-0.26) | 0.28 (0.18-0.36) |
| Akita (sequence only, GM12878 head) | competitors (published models, no AML in training) | 0.36 (0.27-0.46) | 0.33 (0.26-0.43) | 0.55 (0.48-0.60) | 0.56 (0.50-0.59) | 0.37 (0.29-0.48) | 0.67 (0.57-0.72) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.50 (0.40-0.62) | 0.44 (0.34-0.56) | 0.89 (0.83-0.93) | 0.71 (0.68-0.74) | 0.55 (0.50-0.61) | 0.81 (0.76-0.84) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.51 (0.40-0.64) | 0.45 (0.33-0.59) | 0.89 (0.83-0.93) | 0.70 (0.67-0.75) | 0.53 (0.48-0.59) | 0.79 (0.76-0.85) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.48 (0.37-0.62) | 0.42 (0.32-0.56) | 0.88 (0.83-0.94) | 0.71 (0.65-0.75) | 0.53 (0.44-0.61) | 0.81 (0.75-0.85) |

### detection_rate | Akita coverage (object inside the 917 kb window)

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.61 (0.55-0.67) | 0.58 (0.52-0.66) | 0.85 (0.80-0.89) | 0.73 (0.66-0.79) | 0.60 (0.51-0.71) | 0.80 (0.71-0.86) |
| consensus (AML reference map) | reference | 0.43 (0.35-0.52) | 0.37 (0.30-0.47) | 0.84 (0.79-0.89) | 0.63 (0.56-0.72) | 0.45 (0.32-0.58) | 0.75 (0.67-0.85) |
| consensus12 (12 cell lines, no AML) | reference | 0.38 (0.29-0.47) | 0.34 (0.25-0.43) | 0.64 (0.57-0.72) | 0.52 (0.42-0.59) | 0.40 (0.31-0.49) | 0.58 (0.46-0.67) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.47 (0.39-0.55) | 0.42 (0.34-0.49) | 0.85 (0.79-0.91) | 0.64 (0.59-0.72) | 0.47 (0.40-0.57) | 0.75 (0.70-0.83) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.42 (0.35-0.51) | 0.38 (0.30-0.46) | 0.67 (0.60-0.75) | 0.53 (0.44-0.59) | 0.40 (0.30-0.49) | 0.60 (0.49-0.67) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.42 (0.35-0.50) | 0.38 (0.29-0.46) | 0.69 (0.63-0.76) | 0.54 (0.44-0.62) | 0.42 (0.32-0.52) | 0.61 (0.50-0.70) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.40 (0.31-0.49) | 0.37 (0.28-0.46) | 0.64 (0.54-0.72) | 0.48 (0.42-0.57) | 0.40 (0.31-0.50) | 0.53 (0.42-0.64) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.48 (0.39-0.59) | 0.43 (0.34-0.52) | 0.85 (0.78-0.91) | 0.66 (0.59-0.72) | 0.49 (0.40-0.56) | 0.76 (0.69-0.83) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.42 (0.33-0.49) | 0.38 (0.30-0.47) | 0.67 (0.57-0.76) | 0.50 (0.42-0.59) | 0.39 (0.31-0.49) | 0.56 (0.43-0.64) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.31 (0.25-0.35) | 0.29 (0.23-0.33) | 0.39 (0.29-0.46) | 0.26 (0.21-0.34) | 0.30 (0.22-0.39) | 0.24 (0.16-0.34) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.29 (0.25-0.33) | 0.27 (0.23-0.31) | 0.37 (0.29-0.45) | 0.24 (0.19-0.31) | 0.25 (0.18-0.40) | 0.22 (0.16-0.29) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.21 (0.20-0.24) | 0.18 (0.16-0.22) | 0.37 (0.31-0.43) | 0.16 (0.12-0.19) | 0.10 (0.05-0.15) | 0.19 (0.15-0.22) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.13 (0.11-0.15) | 0.13 (0.11-0.14) | 0.12 (0.06-0.17) | 0.20 (0.17-0.24) | 0.17 (0.13-0.22) | 0.24 (0.19-0.29) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.19 (0.15-0.23) | 0.19 (0.15-0.22) | 0.19 (0.08-0.31) | 0.26 (0.18-0.32) | 0.28 (0.19-0.36) | 0.28 (0.17-0.38) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.18 (0.14-0.23) | 0.18 (0.15-0.22) | 0.20 (0.10-0.31) | 0.27 (0.19-0.34) | 0.27 (0.19-0.36) | 0.30 (0.18-0.41) |
| Akita (sequence only, GM12878 head) | competitors (published models, no AML in training) | 0.34 (0.26-0.43) | 0.31 (0.23-0.40) | 0.56 (0.48-0.60) | 0.53 (0.47-0.58) | 0.35 (0.27-0.45) | 0.65 (0.56-0.70) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.49 (0.39-0.58) | 0.42 (0.34-0.53) | 0.89 (0.83-0.93) | 0.68 (0.61-0.72) | 0.53 (0.42-0.60) | 0.77 (0.70-0.82) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.50 (0.40-0.61) | 0.43 (0.35-0.54) | 0.88 (0.84-0.93) | 0.67 (0.62-0.72) | 0.50 (0.44-0.58) | 0.78 (0.72-0.81) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.48 (0.38-0.57) | 0.41 (0.32-0.51) | 0.89 (0.83-0.93) | 0.64 (0.60-0.74) | 0.44 (0.35-0.61) | 0.75 (0.71-0.84) |

### detection_rate | Epiphany coverage (reach 990 kb)

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.63 (0.56-0.70) | 0.60 (0.54-0.68) | 0.86 (0.82-0.90) | 0.76 (0.73-0.81) | 0.65 (0.58-0.73) | 0.83 (0.78-0.87) |
| consensus (AML reference map) | reference | 0.43 (0.32-0.53) | 0.37 (0.27-0.47) | 0.84 (0.80-0.89) | 0.67 (0.62-0.76) | 0.48 (0.42-0.60) | 0.79 (0.73-0.86) |
| consensus12 (12 cell lines, no AML) | reference | 0.38 (0.29-0.49) | 0.34 (0.26-0.46) | 0.58 (0.51-0.72) | 0.54 (0.46-0.60) | 0.41 (0.31-0.49) | 0.61 (0.52-0.69) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.48 (0.36-0.57) | 0.42 (0.32-0.50) | 0.86 (0.78-0.92) | 0.68 (0.65-0.74) | 0.51 (0.47-0.62) | 0.77 (0.74-0.83) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.42 (0.33-0.52) | 0.39 (0.31-0.48) | 0.61 (0.54-0.67) | 0.56 (0.48-0.64) | 0.45 (0.36-0.53) | 0.62 (0.53-0.71) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.43 (0.33-0.53) | 0.40 (0.31-0.50) | 0.63 (0.56-0.71) | 0.56 (0.48-0.64) | 0.45 (0.35-0.55) | 0.62 (0.54-0.73) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.41 (0.32-0.50) | 0.38 (0.30-0.49) | 0.58 (0.48-0.69) | 0.51 (0.45-0.58) | 0.41 (0.34-0.50) | 0.57 (0.49-0.65) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.49 (0.38-0.61) | 0.44 (0.32-0.55) | 0.85 (0.77-0.91) | 0.70 (0.66-0.74) | 0.53 (0.48-0.59) | 0.80 (0.75-0.83) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.42 (0.34-0.52) | 0.39 (0.31-0.49) | 0.59 (0.51-0.68) | 0.53 (0.45-0.60) | 0.43 (0.33-0.52) | 0.59 (0.50-0.67) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.30 (0.25-0.36) | 0.28 (0.24-0.34) | 0.34 (0.25-0.44) | 0.30 (0.24-0.34) | 0.27 (0.21-0.33) | 0.31 (0.24-0.38) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.29 (0.25-0.33) | 0.28 (0.23-0.31) | 0.36 (0.29-0.43) | 0.26 (0.21-0.32) | 0.23 (0.17-0.32) | 0.27 (0.20-0.35) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.21 (0.19-0.24) | 0.18 (0.17-0.22) | 0.36 (0.30-0.41) | 0.18 (0.14-0.21) | 0.14 (0.10-0.19) | 0.21 (0.18-0.23) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.13 (0.12-0.14) | 0.13 (0.11-0.14) | 0.12 (0.07-0.17) | 0.18 (0.16-0.20) | 0.17 (0.12-0.20) | 0.20 (0.17-0.23) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.19 (0.14-0.22) | 0.19 (0.15-0.22) | 0.18 (0.06-0.30) | 0.21 (0.16-0.27) | 0.20 (0.14-0.28) | 0.26 (0.16-0.33) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.18 (0.14-0.23) | 0.18 (0.15-0.22) | 0.19 (0.10-0.30) | 0.22 (0.17-0.29) | 0.20 (0.15-0.25) | 0.28 (0.18-0.38) |
| Akita (sequence only, GM12878 head) | competitors (published models, no AML in training) | 0.36 (0.28-0.44) | 0.33 (0.25-0.43) | 0.55 (0.48-0.59) | 0.56 (0.50-0.59) | 0.37 (0.29-0.48) | 0.67 (0.58-0.71) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.50 (0.39-0.61) | 0.43 (0.34-0.53) | 0.89 (0.84-0.93) | 0.71 (0.67-0.74) | 0.55 (0.50-0.62) | 0.80 (0.75-0.84) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.51 (0.40-0.63) | 0.45 (0.33-0.58) | 0.89 (0.83-0.92) | 0.69 (0.66-0.73) | 0.52 (0.48-0.58) | 0.79 (0.75-0.83) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.48 (0.38-0.61) | 0.42 (0.32-0.55) | 0.88 (0.83-0.94) | 0.69 (0.64-0.74) | 0.51 (0.41-0.61) | 0.79 (0.74-0.85) |

### detection_rate | objects scored by all competitors

| model | block | loop all | loop private | loop recurrent | stripe all | stripe private | stripe recurrent |
|---|---|---|---|---|---|---|---|
| truth (measured) | reference | 0.62 (0.56-0.69) | 0.60 (0.54-0.67) | 0.85 (0.80-0.90) | 0.74 (0.68-0.81) | 0.62 (0.52-0.74) | 0.81 (0.73-0.88) |
| consensus (AML reference map) | reference | 0.43 (0.32-0.54) | 0.37 (0.27-0.46) | 0.84 (0.78-0.88) | 0.63 (0.56-0.76) | 0.46 (0.32-0.61) | 0.74 (0.66-0.86) |
| consensus12 (12 cell lines, no AML) | reference | 0.38 (0.28-0.49) | 0.34 (0.25-0.44) | 0.61 (0.54-0.71) | 0.52 (0.41-0.61) | 0.42 (0.31-0.51) | 0.59 (0.45-0.69) |
| 12-cell model, AML reference as input | no AML in training (paper models) | 0.48 (0.37-0.56) | 0.42 (0.32-0.51) | 0.86 (0.78-0.92) | 0.66 (0.61-0.74) | 0.48 (0.41-0.58) | 0.76 (0.70-0.85) |
| 12-cell model (refin), consensus12 | no AML in training (paper models) | 0.42 (0.33-0.51) | 0.39 (0.31-0.47) | 0.64 (0.58-0.68) | 0.53 (0.44-0.64) | 0.39 (0.31-0.54) | 0.62 (0.50-0.73) |
| 12-cell model (refres), consensus12 | no AML in training (paper models) | 0.43 (0.32-0.52) | 0.40 (0.29-0.49) | 0.66 (0.59-0.73) | 0.55 (0.45-0.64) | 0.41 (0.32-0.54) | 0.63 (0.49-0.74) |
| 12-cell differential-channel model, consensus12 | no AML in training (paper models) | 0.41 (0.31-0.51) | 0.38 (0.30-0.47) | 0.60 (0.50-0.70) | 0.49 (0.41-0.60) | 0.39 (0.31-0.52) | 0.55 (0.43-0.69) |
| release, AML reference input | release checkpoint all12one_fullrd (no AML in training) | 0.49 (0.38-0.60) | 0.43 (0.33-0.54) | 0.85 (0.77-0.91) | 0.68 (0.62-0.74) | 0.49 (0.42-0.60) | 0.79 (0.71-0.85) |
| release, consensus12 | release checkpoint all12one_fullrd (no AML in training) | 0.42 (0.34-0.53) | 0.39 (0.31-0.49) | 0.62 (0.55-0.72) | 0.51 (0.43-0.61) | 0.39 (0.32-0.53) | 0.57 (0.46-0.68) |
| release, marks only (no reference) | release checkpoint all12one_fullrd (no AML in training) | 0.30 (0.25-0.36) | 0.29 (0.24-0.34) | 0.36 (0.28-0.46) | 0.28 (0.22-0.36) | 0.29 (0.21-0.39) | 0.27 (0.19-0.35) |
| dedicated no-reference model (all12honest_drop) | dedicated no-reference model | 0.30 (0.25-0.34) | 0.28 (0.24-0.31) | 0.38 (0.30-0.45) | 0.25 (0.20-0.33) | 0.25 (0.18-0.41) | 0.24 (0.18-0.32) |
| C.Origami (CTCF, ATAC quantile-matched to IMR-90) | competitors (published models, no AML in training) | 0.21 (0.20-0.24) | 0.18 (0.16-0.22) | 0.37 (0.30-0.43) | 0.16 (0.12-0.19) | 0.10 (0.06-0.15) | 0.19 (0.15-0.21) |
| C.Origami (CTCF, ATAC as given) | competitors (published models, no AML in training) | 0.13 (0.12-0.15) | 0.13 (0.11-0.14) | 0.12 (0.06-0.17) | 0.20 (0.17-0.23) | 0.17 (0.13-0.21) | 0.24 (0.19-0.28) |
| Epiphany (ATAC for DNase, H3K4me3 = 0) | competitors (published models, no AML in training) | 0.19 (0.14-0.23) | 0.19 (0.15-0.22) | 0.19 (0.08-0.31) | 0.26 (0.19-0.33) | 0.28 (0.19-0.36) | 0.28 (0.17-0.37) |
| Epiphany (ATAC for DNase, H3K4me3 = genome mean) | competitors (published models, no AML in training) | 0.18 (0.14-0.22) | 0.18 (0.15-0.22) | 0.20 (0.11-0.33) | 0.27 (0.19-0.33) | 0.27 (0.19-0.35) | 0.30 (0.18-0.40) |
| Akita (sequence only, GM12878 head) | competitors (published models, no AML in training) | 0.36 (0.28-0.44) | 0.33 (0.26-0.44) | 0.55 (0.48-0.59) | 0.56 (0.50-0.59) | 0.37 (0.29-0.47) | 0.67 (0.57-0.72) |
| AML-trained, reference input (aml_refin) | trained with AML patients (not comparable to competitors) | 0.49 (0.39-0.62) | 0.43 (0.33-0.53) | 0.89 (0.83-0.93) | 0.69 (0.59-0.75) | 0.53 (0.40-0.62) | 0.78 (0.69-0.84) |
| AML-trained, difference channels (aml_diff) | trained with AML patients (not comparable to competitors) | 0.51 (0.39-0.64) | 0.45 (0.33-0.57) | 0.89 (0.83-0.92) | 0.68 (0.63-0.74) | 0.50 (0.44-0.59) | 0.78 (0.71-0.84) |
| AML-trained, difference channels vs HSPC (aml_diffhspc) | trained with AML patients (not comparable to competitors) | 0.48 (0.38-0.61) | 0.42 (0.32-0.55) | 0.88 (0.82-0.94) | 0.65 (0.60-0.76) | 0.45 (0.37-0.65) | 0.75 (0.70-0.86) |
