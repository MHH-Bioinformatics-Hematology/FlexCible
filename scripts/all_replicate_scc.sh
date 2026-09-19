#!/bin/bash
# HiCRep SCC (plus obs/exp Pearson and stratified) for every replicate pair we
# have, at every resolution both members share.
set -u
cd /home/mh-hannover.local/wolffjoa/src/hic-multiome-unet
export CUDA_VISIBLE_DEVICES=""
D=/mnt/processed/wolffjoa/hic-multiome/replicates
PY=~/miniconda3/envs/hicdiff/bin/python

pair () {  # label file1 file2 resolutions...
  local lab=$1 f1=$2 f3=$3; shift 3
  for r in "$@"; do
    md=$(( 2000000 / r )); [ $md -lt 20 ] && md=20
    [ $md -gt 400 ] && md=400
    echo "----- $lab @ ${r}bp"
    $PY scripts/replicate_ceiling.py --pairs "${lab}=${f1},${f3}" \
        --res $r --n-regions 30 --span 2097152 --max-dist-bins $md \
        --out runs/scc_${lab}_${r}.csv 2>&1 | grep -E "^${lab}" || true
  done
}
pair GSE155971_DMSO $D/gse155971_G1_DMSO_rep1.mcool $D/gse155971_G1_DMSO_rep2.mcool \
     1000 2000 5000 10000 25000 50000 100000
if [ -s $D/gse155971_G1_AUX_rep1.mcool ]; then
pair GSE155971_AUX  $D/gse155971_G1_AUX_rep1.mcool $D/gse155971_G1_AUX_rep2.mcool \
     1000 2000 5000 10000 25000 50000 100000
fi
pair HFFc6_4DN      $D/hff_bio1.mcool $D/hff_bio2.mcool 5000 10000 20000 40000 80000
pair GM12878_4DN    $D/gm_bio1.mcool  $D/gm_bio2.mcool  5000 10000 20000 40000 80000
pair H1_4DN_tech    $D/h1_tec1.mcool  $D/h1_tec2.mcool  5000 10000 20000 40000 80000
echo ALL_SCC_DONE
