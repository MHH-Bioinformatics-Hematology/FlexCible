"""Verify the 4DN-annotated deleted CTCF sites (GRCh38) against the JASPAR MA0139.2 scan used by flexcible."""
import sys, csv, subprocess
import numpy as np, pysam
sys.path.insert(0, "/home/mh-hannover.local/wolffjoa/src/hic-multiome-unet")
from flexcible.motifs import read_jaspar, pwm_logodds, revcomp_pwm
SITES = [  # name, chrom, start(1-based incl), end, orientation from 4DN name
 ("A_fwd","chr8",132875646,132875665,"+"),("B_rev","chr8",133203069,133203088,"-"),("B_fwd","chr8",133209431,133209450,"+"),
 ("E_rev","chr1",180863366,180863385,"-"),("E_fwd","chr1",180882097,180882116,"+"),("H_fwd","chr5",31825672,31825691,"+")]
fa = pysam.FastaFile("/home/mh-hannover.local/wolffjoa/data/hic-multiome/genome/hg38.fa")
pfm = read_jaspar("/home/mh-hannover.local/wolffjoa/data/hic-multiome/motifs/MA0139.2.jaspar")
pwm = pwm_logodds(pfm); rc = revcomp_pwm(pwm); L = pwm.shape[1]; mx = pwm.max(0).sum()
code = {c:i for i,c in enumerate("ACGT")}
def scores(seq, m):
    c = np.array([code.get(x,4) for x in seq.upper()]); lut = np.vstack([m, m.min(0,keepdims=True)])
    n = len(c)-L+1; s = np.zeros(n)
    for k in range(L): s += lut[c[k:k+n],k]
    return s/mx
out = []
for name,ch,s,e,o in SITES:
    pad = 60; a = s-1-pad; seq = fa.fetch(ch, a, e+pad)
    f, r = scores(seq, pwm), scores(seq, rc)
    best = []
    for st,arr in (("+",f),("-",r)):
        i = int(arr.argmax()); best.append((arr[i], st, a+i+1, a+i+L))
    best.sort(reverse=True); fr, st, bs, be = best[0]
    # best hit overlapping the annotated 20-bp window
    ov = [(arr[i], st2, a+i+1, a+i+L) for st2,arr in (("+",f),("-",r)) for i in range(len(arr)) if a+i+1 <= e and a+i+L >= s]
    ov.sort(reverse=True)
    o_fr, o_st, o_s, o_e = ov[0]
    n_hits_pm1kb = 0
    seq2 = fa.fetch(ch, s-1-1000, e+1000); a2 = s-1-1000
    f2, r2 = scores(seq2, pwm), scores(seq2, rc)
    hits = [(a2+i+1, "+", f2[i]) for i in np.nonzero(f2>=0.80)[0]] + [(a2+i+1, "-", r2[i]) for i in np.nonzero(r2>=0.80)[0]]
    out.append(dict(site=name, chrom=ch, start=s, end=e, annotated_strand=o, best_overlap_strand=o_st, best_overlap_start=o_s, best_overlap_end=o_e,
                    best_overlap_score_frac=round(o_fr,3), best_in_pm60bp=f"{st}:{bs}-{be}:{fr:.3f}", n_hits_ge0p80_pm1kb=len(hits),
                    hits_pm1kb=";".join(f"{p}{s2}{sc:.2f}" for p,s2,sc in sorted(hits)), orientation_match=(o==o_st)))
w = csv.DictWriter(open("results/perturb_hap1/site_motif_check.csv","w"), fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out: print({k:r[k] for k in ("site","best_overlap_strand","best_overlap_start","best_overlap_score_frac","best_in_pm60bp","n_hits_ge0p80_pm1kb","orientation_match")})
