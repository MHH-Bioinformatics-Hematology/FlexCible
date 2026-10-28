#!/usr/bin/env bash
# Build a HAP1 CTCF fold-change-over-IgG bigWig (GRCh38) from Haarhuis 2017 (GSE94992),
# because ENCODE has no CTCF ChIP-seq for HAP-1.  CTCF = SRX2570608 (SRR5266526), IgG = SRX2570604 (SRR5266522).
set -euo pipefail
export HTTPS_PROXY=http://proxy.mh-hannover.de:8080 https_proxy=http://proxy.mh-hannover.de:8080
W=/mnt/processed/wolffjoa/hic-multiome/perturb/hap1_insilico/work
T=/mnt/processed/wolffjoa/hic-multiome/perturb/hap1_insilico/tracks
IDX=$HOME/data/genomes/hg38/bowtie2/hg38
SRA=$HOME/miniconda3/envs/sratoolkit/bin
BT=$HOME/miniconda3/envs/bowtie2/bin/bowtie2
MACS=${MACS3:-$HOME/venvs/macs3/bin/macs3}
mkdir -p "$W" "$T"; cd "$W"
for r in SRR5266526 SRR5266522; do
  [ -s $r.bam ] && continue
  $SRA/prefetch -O sra $r --max-size 50G
  $SRA/fasterq-dump -e 8 -O fq sra/$r/$r.sra
  $BT -p 8 -x $IDX -U fq/$r.fastq 2> $r.bt2.log | samtools view -@4 -q 30 -F 4 -b - | samtools sort -@4 -o $r.bam -
  samtools index $r.bam
  rm -f fq/$r.fastq
done
$MACS callpeak -t SRR5266526.bam -c SRR5266522.bam -f BAM -g hs -n hap1_ctcf --nomodel --extsize 200 --keep-dup 1 -B -p 1e-2 --outdir macs
$MACS bdgcmp -t macs/hap1_ctcf_treat_pileup.bdg -c macs/hap1_ctcf_control_lambda.bdg -m FE -o macs/hap1_ctcf_FE.bdg
LC_ALL=C sort -k1,1 -k2,2n macs/hap1_ctcf_FE.bdg | awk '$1 ~ /^chr[0-9XY]+$/' > macs/hap1_ctcf_FE.sorted.bdg
cut -f1,2 $HOME/data/hic-multiome/genome/hg38.fa.fai > macs/hg38.sizes
$HOME/miniconda3/envs/__ucsc_kent/bin/bedGraphToBigWig macs/hap1_ctcf_FE.sorted.bdg macs/hg38.sizes $T/CTCF.bigwig
