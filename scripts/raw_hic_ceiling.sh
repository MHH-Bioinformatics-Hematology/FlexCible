#!/bin/bash
# Raw-reads biological-replicate ceiling for HMEC in-situ Hi-C (MboI/GATC).
# fastq (ENCODE) -> minibwa --hic (per mate) -> HiCExplorer v4 C++ hicBuildMatrix 2kb
# -> oe_proper -> per-region Pearson rep1 vs rep2 on the 413 Akita regions.
set -eo pipefail
ENV=/home/mh-hannover.local/wolffjoa/miniconda3/envs/hicdiff/bin
MINIBWA=/home/mh-hannover.local/wolffjoa/src/bwa-mem3/bench/minibwa
HBM=/home/mh-hannover.local/wolffjoa/src/HiCExplorer-v4/cpp/build/tools/hicBuildMatrix
IDX=/mnt/processed/wolffjoa/hic-multiome/minibwa_index/hg38
W=/mnt/processed/wolffjoa/hic-multiome/crosscell_raw
OUT=/mnt/processed/wolffjoa/hic-multiome/crosscell_oe
CODE=/home/mh-hannover.local/wolffjoa/src/flexcible
mkdir -p "$W"
cd "$CODE"

# 1) wait for the minibwa index to finish (file-stability, no process matching)
echo "[wait] minibwa index..."
prev=""; stable=0
while true; do
  cur=$(ls -la /mnt/processed/wolffjoa/hic-multiome/minibwa_index/ 2>/dev/null | md5sum)
  if [ "$cur" = "$prev" ]; then stable=$((stable+1)); else stable=0; fi
  # index done when the directory listing is unchanged for ~2 min AND a .bwt-like file exists
  [ "$stable" -ge 4 ] && ls /mnt/processed/wolffjoa/hic-multiome/minibwa_index/hg38.mbw >/dev/null 2>&1 && break
  prev="$cur"; sleep 30
done
echo "[ok] index ready: $(ls /mnt/processed/wolffjoa/hic-multiome/minibwa_index/)"

dl() { # acc -> $W/acc.fastq.gz
  local acc=$1
  [ -s "$W/$acc.fastq.gz" ] && { echo "have $acc"; return; }
  echo "[dl] $acc"; curl -fSL -C - --retry 5 --retry-delay 10 -o "$W/$acc.fastq.gz.part" \
    "https://www.encodeproject.org/files/$acc/@@download/$acc.fastq.gz"
  mv "$W/$acc.fastq.gz.part" "$W/$acc.fastq.gz"
}
mapmate() { # acc -> $W/acc.bam (fastq order, single-end Hi-C mapping)
  local acc=$1
  [ -s "$W/$acc.bam" ] && { echo "have bam $acc"; return; }
  echo "[map] $acc"; "$MINIBWA" map --hic -t 16 "$IDX" "$W/$acc.fastq.gz" \
    | "$ENV/samtools" view -b -@ 4 -o "$W/$acc.bam.part" -
  mv "$W/$acc.bam.part" "$W/$acc.bam"
}
build() { # rep R1acc R2acc -> $OUT/hmec_<rep>_raw_2kb.cool
  local rep=$1 r1=$2 r2=$3
  [ -s "$OUT/hmec_${rep}_proper_2kb.cool" ] && { echo "have $rep proper cool"; return; }
  echo "[matrix] $rep"
  "$HBM" --samFiles "$W/$r1.bam" "$W/$r2.bam" \
    -o "$W/hmec_${rep}_2kb.cool" --QCfolder "$W/qc_${rep}" \
    --binSize 2000 --restrictionCutFile "$W/gatc_sites.bed" \
    --restrictionSequence GATC --danglingSequence GATC \
    --minMappingQuality 15 --threads 16 --skipDuplicationCheck --noPlot
  "$ENV/python" scripts/oe_proper.py --input "$W/hmec_${rep}_2kb.cool" \
    --output "$OUT/hmec_${rep}_proper_2kb.cool" --binsize 2000
}

# rep1
dl ENCFF055UHA; dl ENCFF416YDZ; mapmate ENCFF055UHA; mapmate ENCFF416YDZ
build rep1 ENCFF055UHA ENCFF416YDZ
# rep2
dl ENCFF126WCO; dl ENCFF244GNV; mapmate ENCFF126WCO; mapmate ENCFF244GNV
build rep2 ENCFF126WCO ENCFF244GNV

# 3) correlate on the 413 Akita regions
"$ENV/python" - <<'PYEOF'
import cooler,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
import sys; sys.path.insert(0,'scripts'); import score_like_hicgan as sc
from flexcible.genome import match_name
OUT="/mnt/processed/wolffjoa/hic-multiome/crosscell_oe"
a=cooler.Cooler(f"{OUT}/hmec_rep1_proper_2kb.cool"); b=cooler.Cooler(f"{OUT}/hmec_rep2_proper_2kb.cool")
reg=pd.read_csv(sc.REGIONS,sep='\t',names=['chrom','start','end']); rs=[]
for _,r in reg.iterrows():
    na=match_name(r.chrom,list(a.chromnames)); nb=match_name(r.chrom,list(b.chromnames))
    if not na or not nb: continue
    try:
        A=np.asarray(a.matrix(balance=False).fetch(f"{na}:{int(r.start)}-{int(r.end)}"),float)
        B=np.asarray(b.matrix(balance=False).fetch(f"{nb}:{int(r.start)}-{int(r.end)}"),float)
    except: continue
    if A.shape!=B.shape or A.shape[0]<50: continue
    iu=np.triu_indices(A.shape[0],2); x=A[iu];y=B[iu]; ok=np.isfinite(x)&np.isfinite(y)&(x!=0)&(y!=0)
    if ok.sum()>100 and x[ok].std()>0 and y[ok].std()>0: rs.append(pearsonr(x[ok],y[ok]).statistic)
print(f"HMEC BIOLOGICAL REPLICATE CEILING (raw reads -> minibwa -> HiCExplorer v4 2kb -> proper O/E):")
print(f"  median per-region Pearson = {np.median(rs):.3f} (mean {np.mean(rs):.3f}, n={len(rs)})")
PYEOF
echo RAW_CEILING_DONE
