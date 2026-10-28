"""Lift the Sanborn 2015 HAP1 HiCCUPS loops (hg19, GSE74072) to GRCh38 and keep those in the captured regions."""
import gzip, subprocess, tempfile, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
src = "GSE74072_Hap1_HiCCUPS_looplist.txt.gz"
loops = []
with gzip.open(sys.argv[1], "rt") as f:
    next(f)
    for i, l in enumerate(f):
        p = l.split("\t")
        loops.append((i, "chr" + p[0], int(p[1]), int(p[2]), "chr" + p[3], int(p[4]), int(p[5]), int(p[7]), p[13]))
chain = str(Path.home() / "data/hic-multiome/chain/hg19ToHg38.over.chain.gz")
lo = str(Path.home() / "miniconda3/envs/__ucsc_kent/bin/liftOver")
res = {}
for which, (c, s, e) in (("x", (1, 2, 3)), ("y", (4, 5, 6))):
    with tempfile.TemporaryDirectory() as d:
        bed = Path(d) / "a.bed"; out = Path(d) / "o.bed"; un = Path(d) / "u.bed"
        bed.write_text("".join(f"{l[c]}\t{l[s]}\t{l[e]}\t{l[0]}\n" for l in loops))
        subprocess.run([lo, str(bed), chain, str(out), str(un)], check=True, capture_output=True)
        res[which] = {int(r.split()[3]): (r.split()[0], int(r.split()[1]), int(r.split()[2])) for r in out.read_text().splitlines()}
rows = []
for l in loops:
    i = l[0]
    if i not in res["x"] or i not in res["y"]: continue
    (c1, a1, b1), (c2, a2, b2) = res["x"][i], res["y"][i]
    for rn, r in REGIONS.items():
        if c1 == c2 == r["chrom"] and a1 >= r["start"] and b2 <= r["end"] and a2 > a1:
            rows.append((rn, c1, a1, b1, a2, b2, l[7], l[8], b1 - a1))
with open(RES / "loops_hap1_hiccups_grch38.tsv", "w") as f:
    f.write("region\tchrom\tx1\tx2\ty1\ty2\tobserved\tfdr_donut_or_col\tanchor_width\n")
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
print(len(rows), "loops in captured regions")
from collections import Counter; print(Counter(r[0] for r in rows))
