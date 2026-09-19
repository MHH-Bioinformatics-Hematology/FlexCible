#!/bin/bash
# Build a 1 kb Hi-C matrix locally from the Galaxy-trimmed, Galaxy-mapped BAMs.
#
# Galaxy's __hicexplorer@3.7.6 has a truncated libgomp.so.1 and dies on import,
# so trimming and mapping stay on Galaxy's compute and only the matrix is built
# here, with the working local hicexplorer 3.7.5.
#
# The BAMs are name-sorted and are NEVER coordinate-sorted: hicBuildMatrix
# pairs the two files by POSITION, so any re-sort would silently produce a
# wrong matrix rather than an error.
set -euo pipefail

ROOT=/mnt/processed/wolffjoa/hic-multiome/local
HIC=~/miniconda3/envs/hicexplorer/bin
FA=/mnt/project/wolffjoa/hg19.fa
THREADS=${THREADS:-28}

label=$1        # e.g. HCT116_rep1
enz=$2          # GATC | AAGCTT
dang=$3         # GATC | AGCT

mkdir -p "$ROOT"/{matrix,logs,qc}

rest="$ROOT/matrix/rest_${enz}.bed"
if [ ! -s "$rest" ]; then
  echo "hicFindRestSite $enz"
  "$HIC/hicFindRestSite" --fasta "$FA" --searchPattern "$enz" -o "$rest" \
    2> "$ROOT/logs/rest_${enz}.log"
fi

r1="$ROOT/bam/${label}_R1.bam"
r2="$ROOT/bam/${label}_R2.bam"
for f in "$r1" "$r2"; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done

out="$ROOT/matrix/${label}_1kb.cool"
if [ -s "$out" ]; then echo "have $out"; exit 0; fi

# hicBuildMatrix picks the output format from the FILE EXTENSION, so the
# usual "write to .part then rename" trick makes it abort with "Please define
# the file extension". The temp name therefore has to keep the .cool suffix.
tmp="$ROOT/matrix/.${label}_1kb.tmp.cool"
echo "hicBuildMatrix 1 kb -> $label (enzyme $enz / dangling $dang)"
"$HIC/hicBuildMatrix" \
  --samFiles "$r1" "$r2" \
  --binSize 1000 \
  --restrictionSequence "$enz" \
  --danglingSequence "$dang" \
  --restrictionCutFile "$rest" \
  --threads "$THREADS" --inputBufferSize 400000 \
  --outFileName "$tmp" \
  --QCfolder "$ROOT/qc/${label}_hicQC" \
  2>&1 | tee "$ROOT/logs/${label}_build.log"
mv "$tmp" "$out"
echo "BUILT $out"
