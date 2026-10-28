"""Choose, per deleted site, a control CTCF site that was NOT deleted, about 1 Mb away inside the captured region:
the JASPAR MA0139.2 hit (score >= 0.70 of the maximum, >= 50 kb from the region edge) with the highest HAP1 CTCF ChIP fold change within 0.85-1.15 Mb
of the deleted site (falling back to the next-best distance window 0.6-1.4 Mb), on either side, inside the region,
>= 30 kb from every deleted site and not an anchor of any deleted-site loop."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pysam, pyBigWig
from common import *
from lib import *
from analyze import site_loops
sys.path.insert(0, str(Path.home() / "src/hic-multiome-unet"))
from flexcible.motifs import read_jaspar, pwm_logodds, revcomp_pwm
fa = pysam.FastaFile(str(Path.home() / "data/hic-multiome/genome/hg38.fa"))
pwm = pwm_logodds(read_jaspar(Path.home() / "data/hic-multiome/motifs/MA0139.2.jaspar")); rc = revcomp_pwm(pwm); L = pwm.shape[1]; mx = pwm.max(0).sum()
code = {c: i for i, c in enumerate("ACGT")}
bw = pyBigWig.open(str(OUT / "tracks/CTCF.bigwig"))

def scan(ch, a, b):
    seq = fa.fetch(ch, a, b).upper(); c = np.array([code.get(x, 4) for x in seq]); n = len(c) - L + 1
    out = {}
    for st, m in (("+", pwm), ("-", rc)):
        lut = np.vstack([m, m.min(0, keepdims=True)]); s = np.zeros(n)
        for k in range(L): s += lut[c[k:k + n], k]
        out[st] = s / mx
    return out
rows = []
deleted = [(v["chrom"], v["start"], v["end"]) for v in SITES.values()]
for name, d in DELETIONS.items():
    ref = SITES[d["sites"][0]]; r = REGIONS[ref["region"]]
    mid = (ref["start"] + ref["end"]) // 2
    a, b = r["start"], r["end"]
    anchors = [(x, y) for lid, sn, en, rr in site_loops(ref['region'], d['sites']) for x, y in ((rr.x1, rr.x2), (rr.y1, rr.y2))]
    sc = scan(ref["chrom"], a, b)
    cand = []
    for st, s in sc.items():
        for i in np.nonzero(s >= 0.70)[0]:
            pos = a + i + 1; cm = pos + L // 2
            if min(abs(cm - (x[1] + x[2]) // 2) for x in deleted if x[0] == ref["chrom"]) < 30000: continue
            if cm < a + 50000 or cm > b - 50000: continue
            if any(lo_ - 5000 <= cm <= hi_ + 5000 for lo_, hi_ in anchors): continue
            fc = bw.stats(ref["chrom"], pos - 1, pos - 1 + L, type="max")[0] or 0.0
            cand.append((abs(cm - mid), cm, st, float(s[i]), fc, pos))
    for lo, hi in ((850000, 1150000), (600000, 1400000)):
        sel = [c for c in cand if lo <= c[0] <= hi and c[4] > 0]
        if sel: break
    best = max(sel, key=lambda c: c[4])
    rows.append((name, ref["chrom"], best[5], best[5] + L - 1, best[2], round(best[3], 3), round(best[4], 2), best[0], len(sel)))
with open(RES / "control_sites.tsv", "w") as f:
    f.write("deletion\tchrom\tstart\tend\tstrand\tmotif_score_frac\thap1_ctcf_fc_max\tdistance_bp\tn_candidates\n")
    for r_ in rows: f.write("\t".join(map(str, r_)) + "\n")
print(open(RES / "control_sites.tsv").read())
