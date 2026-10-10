"""Patient x subtype table for the Xu et al. 2022 cohort (feasibility study, CPU only).

Fusion class: from Suppl. Table 5 SV junctions (hg38) with fixed windows around BCR/ABL1, MYH11/CBFB,
RUNX1T1/RUNX1, KMT2A, DEK/NUP214. Mutations and consensus group: read by hand from Fig. 1c of the paper (Nature Fig. 1 PNG,
https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-022-05365-x/MediaObjects/41586_2022_5365_Fig1_HTML.png;
the row order of the "Consensus mutations" / "Other mutations" columns follows the x-axis sample order).
No tabular per-patient mutation list exists in GEO, the Suppl. Tables or the text.
Output: results/aml_subtype/patient_subtype_table.csv
"""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
S = pd.read_csv(R / "aml_primary/xu_tables/st5_svs.csv", dtype={"patient": str})
S = S[~S.patient.isin(["HL60", "K562", "Kasumi", "THP1"])]
S["patient"] = S.patient.map(lambda x: x.zfill(3) if len(x) < 3 else x)


def near(c, x, chrom, lo, hi):
    return (c == chrom) & (x >= lo * 1e6) & (x <= hi * 1e6)


def flags(g):
    f = set()
    for r in g.itertuples():
        for (c1, x1, c2, x2) in ((r.chrom1, r.pos1, r.chrom2, r.pos2), (r.chrom2, r.pos2, r.chrom1, r.pos1)):
            if near(c1, x1, "chr22", 23.0, 23.6) and near(c2, x2, "chr9", 130.4, 131.2): f.add("BCR-ABL1")
            if r.sv_type == "inversion" and near(c1, x1, "chr16", 15.5, 16.2) and near(c2, x2, "chr16", 66.9, 67.4): f.add("inv16")
            if near(c1, x1, "chr8", 91.9, 92.4) and near(c2, x2, "chr21", 34.5, 36.5): f.add("t(8;21)")
            if near(c1, x1, "chr11", 118.4, 118.6) and c2 != "chr11": f.add("KMT2A-r")
            if near(c1, x1, "chr6", 18.0, 18.5) and near(c2, x2, "chr9", 131.0, 131.4): f.add("t(6;9)")
    return sorted(f)


fl = {p: flags(g) for p, g in S.groupby("patient")}
# all 25 AML Hi-C patients (GEO GSE152135 source name 'Primary AML'); WGS SV table covers 22 of the 25 minus those absent
aml = ["629","270","868","773","1953","566","018","260","546","1021","1916","472","936","1360","168","027","990",
       "1071","1324","424","496","504","798","838","933"]
cuttag = {"018","027","168","270","472","546","629","773","1360","1916"}   # AML with CUT&Tag; 103,673 are healthy donors (GEO source name, Fig. 1c "Normal PBMC")
ctcf = cuttag - {"1360","1916"}
fold0_test = {"027","270","773","1916"}
fold = {"027":0,"1916":0,"270":0,"773":0,"168":1,"472":1,"018":2,"1360":2,"546":2,"629":2}

# (consensus group, consensus mutations, other mutations) per patient, Fig. 1c, hand-read, in figure order
FIG1C = {
 "472": ("STAG2", "STAG2;KRAS;NRAS", "+8;GATA1;IDH1;KMT2C;KMT2D"),
 "936": ("STAG2", "STAG2;KRAS;NRAS", "-7;IDH2;KMT2D"),
 "1071": ("STAG2", "STAG2", "SRSF2;CEBPA"),
 "990": ("RUNX1", "RUNX1", "DNMT3A;KMT2C"),
 "027": ("RUNX1", "RUNX1;t(9;22)", "ASXL1;GATA1;NOTCH1"),
 "168": ("RUNX1", "RUNX1;t(9;22)", "ASXL1;DNMT3A;JAK1;NOTCH1;TP53;KMT2C;KMT2E"),
 "496": ("RUNX1", "RUNX1;t(9;22)", "TP53;DNMT3A"),
 "424": ("TET2", "TET2;NPM1", "IDH1;PTPN1;NRAS/KRAS;CEBPA"),
 "260": ("TET2", "TET2;NPM1", "GATA1;IDH1;MYC;PTPN11;KMT2C"),
 "546": ("TET2", "TET2;FLT3-ITD", ""),
 "018": ("TET2", "TET2;NPM1;FLT3-ITD", "DNMT3A;KMT2C;KMT2E"),
 "1916": ("TET2", "TET2;FLT3-ITD", "+8"),
 "1021": ("TET2", "TET2;FLT3-ITD", "t(6;9)"),
 "1953": ("CEBPA", "CEBPA;KMT2A", ""),
 "566": ("CEBPA", "CEBPA;KMT2A", "WT1;KMT2C"),
 "773": ("CEBPA", "CEBPA", "GATA2;KMT2C"),
 "1324": ("CEBPA", "CEBPA", "GATA2;RUNX1;SRSF2;STAG2"),
 "933": ("CEBPA", "CEBPA;FLT3-ITD;t(8;21)", ""),
 "1360": ("FLT3-ITD only", "FLT3-ITD", ""),
 "504": ("KMT2B", "KMT2B;inv(16)", ""),
 "798": ("KMT2B", "inv(16)", ""),
 "629": ("KMT2B", "KMT2B;inv(16)", "PTPN11;KMT2C;KMT2D"),
 "838": ("KMT2B", "KMT2B", "NRAS;FLT3-TKD;IDH1;NPM1;DNMT3A;TET2"),
 "270": ("KMT2B", "KMT2B", "ASXL1;GATA1;TP53"),
 "868": ("KMT2B", "KMT2B;t(9;22)", ""),
}
def has(p, g):
    c, m, o = FIG1C[p]
    return "yes" if g in (m + ";" + o).split(";") else "no"
rows = []
for p in aml:
    if p in fl:
        f = fl[p]; cls = ";".join(f) if f else "none_detected"; src = "Suppl. Table 5 SV junctions (hg38)"
    else:
        cls = "unknown"; src = "no row in Suppl. Table 5 (no WGS SV calls)"
    rows.append(dict(patient=p, fusion_class=cls, fusion_source=src, consensus_group=FIG1C[p][0], consensus_mutations=FIG1C[p][1], other_mutations=FIG1C[p][2],
                     FLT3_ITD=has(p, "FLT3-ITD"), NPM1=has(p, "NPM1"), TET2=has(p, "TET2") if FIG1C[p][0] != "TET2" else "yes",
                     CEBPA=has(p, "CEBPA"), STAG2_KRAS_NRAS="yes" if any(x in (FIG1C[p][1]+";"+FIG1C[p][2]).split(";") for x in ("STAG2","KRAS","NRAS","NRAS/KRAS")) else "no",
                     mutation_source="Xu 2022 Fig. 1c (consensus + other mutations panel), hand-read; GEO has no mutation fields",
                     in_aml_loop_table=p in {r for r in fl},
                     has_cuttag=p in cuttag, has_ctcf=p in ctcf, aml_train_fold=fold.get(p, ""),
                     fold0_role=("test" if p in fold0_test else "train" if p in cuttag else "not used"),
                     source="GEO GSE152135 sample list; Xu 2022 Suppl. Table 5; results/aml_train/split.json"))
T = pd.DataFrame(rows)
T.to_csv(R / "aml_subtype/patient_subtype_table.csv", index=False)
print(T[["patient","fusion_class","has_cuttag","fold0_role"]].to_string(index=False))
print(T.fusion_class.value_counts())
