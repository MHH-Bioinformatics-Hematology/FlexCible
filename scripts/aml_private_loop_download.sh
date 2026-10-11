#!/usr/bin/env bash
# Download the per-patient ATAC-seq (GSE152134) and RNA-seq (GSE152096) files of Xu et al. 2022
# (per-sample suppl folders of the GEO FTP) into aml/primary/atac and aml/primary/rna (about 2.6 GB). Files listed in the series filelist.
set -euo pipefail
P=/mnt/processed/wolffjoa/hic-multiome/aml/primary
fetch() {  # series dir pattern
  local g=$1 d=$2 pat=$3
  mkdir -p "$P/$d"
  awk -F'\t' -v p="$pat" '$1=="File" && $2 ~ p {print $2}' "$P/${g}_filelist.txt" | while read -r f; do
    [ -s "$P/$d/$f" ] && continue
    nice -n 10 curl -sS -f -m 3600 --retry 3 -o "$P/$d/$f.part" \
      "https://ftp.ncbi.nlm.nih.gov/geo/samples/${f:0:7}nnn/${f%%_*}/suppl/$f" && mv "$P/$d/$f.part" "$P/$d/$f"
  done
}
fetch GSE152134 atac 'bigwig$|peaks.txt.gz$'
fetch GSE152096 rna 'quant.genes.txt.gz$'
ls "$P/atac" "$P/rna" | wc -l

# Healthy CD34+ reference: ATAC bigWigs (ChIP-Atlas hg38, three healthy volunteers SRX19490150-52 and
# two control replicates SRX10942129-30) and ENCODE ENCSR830HIN (CD34+ common myeloid progenitor, polyA RNA,
# GRCh38 GENCODE V24 gene quantifications ENCFF690QPA); GENCODE v24 GTF for gene names and TSS.
mkdir -p "$P/atac/hspc" "$P/rna/hspc" "$P/annot"
for s in SRX19490150 SRX19490151 SRX19490152 SRX10942129 SRX10942130; do
  [ -s "$P/atac/hspc/$s.bw" ] || nice -n 10 curl -sS -f -m 3600 -o "$P/atac/hspc/$s.bw" "https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/$s.bw"
done
[ -s "$P/rna/hspc/ENCFF690QPA.tsv" ] || curl -sSL -f -m 600 -o "$P/rna/hspc/ENCFF690QPA.tsv" "https://www.encodeproject.org/files/ENCFF690QPA/@@download/ENCFF690QPA.tsv"
[ -s "$P/annot/gencode.v24.annotation.gtf.gz" ] || curl -sS -f -m 600 -o "$P/annot/gencode.v24.annotation.gtf.gz" "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_24/gencode.v24.annotation.gtf.gz"
