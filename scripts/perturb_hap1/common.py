"""Shared definitions for the HAP1 in-silico CTCF-site deletion test."""
from pathlib import Path
SHARE = Path("/mnt/processed/wolffjoa/hic-multiome/perturb")
MCOOL = SHARE / "hap1_ctcf_site_deletions_sanborn2015"
OUT = SHARE / "hap1_insilico"
RES = Path.home() / "src/hic-multiome-unet/results/perturb_hap1"
RUNS = Path("/mnt/processed/wolffjoa/hic-multiome/runs/multicell_reference")
MODELS = {"10kb": RUNS / "all12honest_drop_10kb.yaml", "5kb": RUNS / "all12honest_drop_5kb.yaml"}
RESOLUTION = {"10kb": 10000, "5kb": 5000}

# captured regions, GRCh38 (cis-coverage of the WT map; 2 Mb each, see README)
REGIONS = {
    "region1": dict(chrom="chr8", start=132_000_000, end=134_000_000),
    "region2": dict(chrom="chr1", start=179_800_000, end=181_800_000),
    "region3": dict(chrom="chr5", start=31_000_000, end=33_000_000),
}
# deleted CTCF motifs (4DN modification records, GRCh38, 1-based inclusive 20 bp)
SITES = {
    "A_fwd": dict(region="region1", chrom="chr8", start=132875646, end=132875665, strand="+"),
    "B_rev": dict(region="region1", chrom="chr8", start=133203069, end=133203088, strand="-"),
    "B_fwd": dict(region="region1", chrom="chr8", start=133209431, end=133209450, strand="+"),
    "E_rev": dict(region="region2", chrom="chr1", start=180863366, end=180863385, strand="-"),
    "E_fwd": dict(region="region2", chrom="chr1", start=180882097, end=180882116, strand="+"),
    "H_fwd": dict(region="region3", chrom="chr5", start=31825672, end=31825691, strand="+"),
}
# deletion experiments -> sites removed together, and the clones (4DN experiment set -> file prefix)
DELETIONS = {
    "A_fwd": dict(region="region1", sites=["A_fwd"], sets=["4DNESG4XBHWB"]),
    "B_rev": dict(region="region1", sites=["B_rev"], sets=["4DNES3CUDIXW"]),
    "B_fwd": dict(region="region1", sites=["B_fwd"], sets=["4DNESYP3AUZS", "4DNESJWXORCI"]),
    "E_fwd": dict(region="region2", sites=["E_fwd"], sets=["4DNESRF2JLP3", "4DNESZ4F7V39", "4DNES5YI4ZCE", "4DNESLBTEN66"]),
    "E_rev": dict(region="region2", sites=["E_rev"], sets=["4DNESNOJ28YT", "4DNESMSJXT8O", "4DNESNCK7FEW"]),
    "E_both": dict(region="region2", sites=["E_fwd", "E_rev"],
                   sets=["4DNES4A1BNPI", "4DNESAVVP47G", "4DNESNI4PPYX", "4DNESVGV5YDR", "4DNESA2R77LP", "4DNES2ZDGCY5"]),
    "H_fwd": dict(region="region3", sites=["H_fwd"], sets=["4DNESTZUCQH5", "4DNES8TLFCMQ"]),
}
WT_SET = "4DNESEYZC9D9"
PAD = 1000  # bp on each side of the deleted motif in the in-silico knockout


def mcool_path(es):
    m = sorted(MCOOL.glob(f"{es}_*.mcool"))
    assert len(m) == 1, es
    return m[0]
