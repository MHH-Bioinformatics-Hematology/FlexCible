#!/usr/bin/env python
"""Are Xu et al. 2022 patients 168 and 496 the same donor?

Compares every pair of the Xu AML/HSPC samples on
  sv    breakpoints of Suppl. Table 5 (exact / 10 kb / 100 kb endpoint matches, both-end SV matches)
  cnv   copy-number segments (Hi-C coverage, results/aml_primary/explain_cnv_segments.csv) and the
        500 kb log2 profile (work/cnv_log2_500kb.csv.gz)
  hic   HiCRep SCC per chromosome on the 100 kb ICE-balanced maps of the GEO mcool files
  meth  WGBS CpG methylation (GEO 10 bp bigwigs) in 100 kb windows, raw and cohort-centred
  loops patient-specific loop and stripe-anchor overlap (Suppl. Tables 1 and 4)
  sex   chrX / chrY cis contact density from the raw mcool counts
Output: results/aml_duplicate_check/*.csv and one png. CPU only, single process.
Usage: python scripts/aml_duplicate_check.py [--stages sv,cnv,hic,meth,sex,plot] [--max-dist 10000000]
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "aml_duplicate_check"
PRIM = Path("/mnt/processed/wolffjoa/hic-multiome/aml/primary")
A, B = "168", "496"
CHROMS = [f"chr{i}" for i in range(1, 23)]


def pairkey(a, b):
    return tuple(sorted((a, b), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s)))


def mark_pair(df):
    df["is_168_496"] = [{x, y} == {A, B} for x, y in zip(df.a, df.b)]
    return df


# ------------------------------------------------------------------ SV
def endpoints(sv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in sv.itertuples():
        rows.append((str(r.patient), r.chrom1, int(r.pos1), r.sv_type))
        rows.append((str(r.patient), r.chrom2, int(r.pos2), r.sv_type))
    return pd.DataFrame(rows, columns=["patient", "chrom", "pos", "sv_type"])


def stage_sv():
    sv = pd.read_csv(REPO / "results/aml_primary/xu_tables/st5_svs.csv", dtype={"patient": str})
    sv["patient"] = sv.patient.astype(str)
    sv = sv[sv.patient.str.isdigit()].copy()  # patients only (the table also lists cell lines)
    ep = endpoints(sv)
    pats = sorted(ep.patient.unique(), key=int)
    by = {p: {c: np.sort(g.pos.values) for c, g in ep[ep.patient == p].groupby("chrom")} for p in pats}
    nep = {p: int(sum(len(v) for v in by[p].values())) for p in pats}

    def matched(p, q, tol):
        n = 0
        for c, pos in by[p].items():
            qq = by[q].get(c)
            if qq is None:
                continue
            i = np.searchsorted(qq, pos)
            d = np.full(len(pos), np.inf)
            ok = i < len(qq); d[ok] = np.abs(qq[i[ok]] - pos[ok])
            ok = i > 0; d[ok] = np.minimum(d[ok], np.abs(qq[i[ok] - 1] - pos[ok]))
            n += int((d <= tol).sum())
        return n

    def sv_match(p, q, tol):
        a = sv[sv.patient == p]; b = sv[sv.patient == q]
        n = 0
        for r in a.itertuples():
            m = b[((b.chrom1 == r.chrom1) & (b.chrom2 == r.chrom2) & ((b.pos1 - r.pos1).abs() <= tol) & ((b.pos2 - r.pos2).abs() <= tol))
                  | ((b.chrom1 == r.chrom2) & (b.chrom2 == r.chrom1) & ((b.pos1 - r.pos2).abs() <= tol) & ((b.pos2 - r.pos1).abs() <= tol))]
            n += int(len(m) > 0)
        return n

    rows = []
    for p, q in itertools.combinations(pats, 2):
        a, b = pairkey(p, q)
        r = dict(a=a, b=b, n_ep_a=nep[a], n_ep_b=nep[b], n_sv_a=int((sv.patient == a).sum()), n_sv_b=int((sv.patient == b).sum()))
        for tol, nm in [(0, "exact"), (10_000, "10kb"), (100_000, "100kb")]:
            m1, m2 = matched(a, b, tol), matched(b, a, tol)
            r[f"ep_{nm}"] = (m1 + m2) / 2
            r[f"sv_{nm}"] = (sv_match(a, b, tol) + sv_match(b, a, tol)) / 2
        r["ep_100kb_frac_of_smaller"] = r["ep_100kb"] / max(1, min(nep[a], nep[b]))
        rows.append(r)
    pr = mark_pair(pd.DataFrame(rows))
    pr.to_csv(OUT / "sv_pairs.csv", index=False)

    # breakpoint list of 168 and 496 with recurrence across all other patients
    lst = []
    for p, q in [(A, B), (B, A)]:
        for r in sv[sv.patient == p].itertuples():
            for end, (c, pos) in enumerate([(r.chrom1, r.pos1), (r.chrom2, r.pos2)], 1):
                def cnt(tol, who):
                    return sorted(x for x in pats if x in who and x != p and
                                  (np.abs(by[x].get(c, np.array([10**12])) - pos) <= tol).any())
                others = [x for x in pats if x not in (A, B)]
                lst.append(dict(patient=p, sv_type=r.sv_type, chrom1=r.chrom1, pos1=r.pos1, chrom2=r.chrom2, pos2=r.pos2,
                                orientation=r.orientation, end=end, chrom=c, pos=pos,
                                in_partner_exact=bool((np.abs(by[q].get(c, np.array([10**12])) - pos) == 0).any()),
                                in_partner_10kb=bool((np.abs(by[q].get(c, np.array([10**12])) - pos) <= 10_000).any()),
                                in_partner_100kb=bool((np.abs(by[q].get(c, np.array([10**12])) - pos) <= 100_000).any()),
                                others_exact=";".join(cnt(0, others)), others_10kb=";".join(cnt(10_000, others)),
                                others_100kb=";".join(cnt(100_000, others))))
    ls = pd.DataFrame(lst)
    ls["n_others_exact"] = ls.others_exact.map(lambda s: len(s.split(";")) if s else 0)
    ls["n_others_10kb"] = ls.others_10kb.map(lambda s: len(s.split(";")) if s else 0)
    ls["n_others_100kb"] = ls.others_100kb.map(lambda s: len(s.split(";")) if s else 0)
    ls["class"] = np.where(~ls.in_partner_100kb, "unshared",
                   np.where(ls.n_others_100kb == 0, "private_shared_168_496",
                   np.where(ls.n_others_100kb >= 2, "recurrent_3plus_patients", "shared_with_one_other")))
    ls.to_csv(OUT / "sv_breakpoints_168_496.csv", index=False)
    print(pr[pr.is_168_496].T)
    print(ls.groupby(["patient", "class"]).size())
    rest = pr[~pr.is_168_496]
    for c in ["ep_exact", "ep_10kb", "ep_100kb", "sv_exact", "sv_10kb", "sv_100kb"]:
        v = pr[pr.is_168_496][c].iloc[0]
        print(c, "168-496", v, "others median", rest[c].median(), "max", rest[c].max(), "rank", int((rest[c] >= v).sum()) + 1, "of", len(pr))


# ------------------------------------------------------------------ CNV
def stage_cnv():
    seg = pd.read_csv(REPO / "results/aml_primary/explain_cnv_segments.csv")
    seg["patient"] = seg.patient.str.replace("xu_", "")
    lg = pd.read_csv(PRIM / "work/cnv_log2_500kb.csv.gz")
    lg["patient"] = lg.patient.str.replace("xu_", "")
    wide = lg.pivot_table(index=["chrom", "start"], columns="patient", values="smooth")
    pats = sorted(wide.columns, key=lambda s: int(s) if s.isdigit() else 10**6)
    pats = [p for p in pats if p.isdigit()]
    # state per 500 kb bin
    st = {}
    for p in pats:
        s = pd.Series(0, index=wide.index)
        for r in seg[seg.patient == p].itertuples():
            m = (s.index.get_level_values(0) == r.chrom) & (s.index.get_level_values(1) >= r.start) & (s.index.get_level_values(1) < r.end)
            s[m] = 1 if r.state == "gain" else -1
        st[p] = s
    rows = []
    for p, q in itertools.combinations(pats, 2):
        a, b = pairkey(p, q)
        sa, sb = st[a], st[b]
        alt = (sa != 0) | (sb != 0)
        same = ((sa == sb) & (sa != 0)).sum()
        # segment-edge matches (+-500 kb) among called segments
        ea = seg[seg.patient == a]; eb = seg[seg.patient == b]
        def edges(d):
            return [(r.chrom, x) for r in d.itertuples() for x in (r.start, r.end)]
        Ea, Eb = edges(ea), edges(eb)
        def nmatch(E1, E2, tol):
            return sum(any(c == c2 and abs(x - x2) <= tol for c2, x2 in E2) for c, x in E1)
        ma = nmatch(Ea, Eb, 500_000)
        v = wide[[a, b]].dropna()
        rows.append(dict(a=a, b=b, n_seg_a=len(ea), n_seg_b=len(eb), n_edges_a=len(Ea), n_edges_b=len(Eb),
                         edge_match_500kb=(ma + nmatch(Eb, Ea, 500_000)) / 2,
                         bins_altered_a=int((sa != 0).sum()), bins_altered_b=int((sb != 0).sum()),
                         bins_same_state=int(same), jaccard_state=float(same / alt.sum()) if alt.sum() else np.nan,
                         log2_pearson=float(np.corrcoef(v[a], v[b])[0, 1])))
    pr = mark_pair(pd.DataFrame(rows))
    pr.to_csv(OUT / "cnv_pairs.csv", index=False)
    x = seg[seg.patient.isin([A, B])].sort_values(["patient", "chrom", "start"])
    x.to_csv(OUT / "cnv_segments_168_496.csv", index=False)
    print(pr[pr.is_168_496].T)
    print(pr.sort_values("jaccard_state", ascending=False).head(6)[["a", "b", "jaccard_state", "edge_match_500kb", "log2_pearson"]])
    print(pr.sort_values("log2_pearson", ascending=False).head(6)[["a", "b", "jaccard_state", "edge_match_500kb", "log2_pearson"]])


# ------------------------------------------------------------------ Hi-C
def mcools():
    d = {}
    for f in sorted((PRIM / "hic").glob("*.mcool")):
        m = re.match(r"GSM\d+_(HSPC-\d+|\d+)\.iced", f.name)
        d[m.group(1)] = f
    return d


def band(M, kmax):
    n = M.shape[0]
    out = np.full((kmax + 1, n), np.nan, np.float32)
    for k in range(1, min(kmax, n - 1) + 1):
        out[k, : n - k] = np.diagonal(M, k)
    return out


def scc_band(X, Y):
    num = den = 0.0
    for k in range(1, X.shape[0]):
        a, b = X[k], Y[k]
        ok = np.isfinite(a) & np.isfinite(b)
        m = int(ok.sum())
        if m < 10:
            continue
        a, b = a[ok], b[ok]
        if a.std() == 0 or b.std() == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        w = m * np.sqrt(np.var(rankdata(a) / m) * np.var(rankdata(b) / m))
        num += w * r; den += w
    return num / den if den > 0 else np.nan


def stage_hic(max_dist, res):
    import cooler
    cs = mcools()
    names = list(cs)
    kmax = max_dist // res
    bands = {c: {} for c in CHROMS}
    for n in names:
        cl = cooler.Cooler(f"{cs[n]}::resolutions/{res}")
        for c in CHROMS:
            M = cl.matrix(balance=True, sparse=False).fetch(c)
            M = np.where(np.isfinite(M), M, np.nan)
            bands[c][n] = band(M, kmax)
        print("read", n, flush=True)
    rows = []
    for p, q in itertools.combinations(names, 2):
        a, b = pairkey(p, q) if (p.isdigit() and q.isdigit()) else (p, q)
        for c in CHROMS:
            rows.append(dict(a=a, b=b, chrom=c, scc=scc_band(bands[c][a], bands[c][b])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hic_scc_chrom.csv", index=False)
    summarize_pairs(df, "scc", "hic_scc_pairs.csv")


def summarize_pairs(df, col, fname):
    g = df.groupby(["a", "b"])[col].agg(["mean", "median", "min", "max"]).reset_index()
    g = mark_pair(g)
    aml = lambda s: s.isdigit()
    g["both_aml"] = [aml(x) and aml(y) for x, y in zip(g.a, g.b)]
    g.to_csv(OUT / fname, index=False)
    r = g[g.both_aml]
    t = r[r.is_168_496].iloc[0]
    o = r[~r.is_168_496]
    print(fname, "168-496 mean", round(t["mean"], 4), "median", round(t["median"], 4), "| other AML pairs mean of means", round(o["mean"].mean(), 4),
          "max", round(o["mean"].max(), 4), "rank", int((o["mean"] >= t["mean"]).sum()) + 1, "of", len(r))
    print(r.sort_values("mean", ascending=False).head(6))


# ------------------------------------------------------------------ WGBS
def stage_meth(win):
    import pyBigWig
    files = {}
    for f in sorted((PRIM / "wgbs").glob("*.bw")):
        m = re.search(r"allc_(\d+)\.CGN", f.name)
        files[str(int(m.group(1)))] = f
    sz = {}
    for l in open(PRIM / "hg38.chrom.sizes"):
        c, n = l.split()[:2]
        if c in CHROMS:
            sz[c] = int(n)
    cols = {}
    for p, f in files.items():
        bw = pyBigWig.open(str(f)); v = []
        for c in CHROMS:
            nb = sz[c] // win
            v.append(np.array(bw.stats(c, 0, nb * win, nBins=nb, type="mean"), dtype=float))
        cols[p] = np.concatenate(v)
        print("meth", p, flush=True)
    X = pd.DataFrame(cols)
    ok = X.notna().all(axis=1)
    X = X[ok]
    R = X.sub(X.mean(axis=1), axis=0)  # cohort-centred
    rows = []
    for p, q in itertools.combinations(X.columns, 2):
        a, b = pairkey(p, q)
        rows.append(dict(a=a, b=b, n_windows=len(X), r_raw=np.corrcoef(X[a], X[b])[0, 1], r_centred=np.corrcoef(R[a], R[b])[0, 1],
                         mean_a=X[a].mean(), mean_b=X[b].mean()))
    df = mark_pair(pd.DataFrame(rows))
    df.to_csv(OUT / "meth_pairs.csv", index=False)
    for col in ("r_raw", "r_centred"):
        t = df[df.is_168_496][col].iloc[0]; o = df[~df.is_168_496][col]
        print(col, "168-496", round(t, 4), "others median", round(o.median(), 4), "max", round(o.max(), 4), "rank", int((o >= t).sum()) + 1, "of", len(df))
    print(df.sort_values("r_centred", ascending=False).head(5))


# ------------------------------------------------------------------ sex
def stage_sex(res, samples):
    """chrX cis density from the mcool (chrY is absent from the GEO mcool files) and chrX/chrY pair counts
    streamed from the nodups pairs files of the listed samples."""
    import cooler
    import subprocess
    rows = []
    for n, f in mcools().items():
        cl = cooler.Cooler(f"{f}::resolutions/{res}")
        tot = {}
        for c in ["chrX"] + CHROMS:
            tot[c] = float(cl.matrix(balance=False, sparse=True).fetch(c).sum()) / 2
        auto = sum(tot[c] for c in CHROMS)
        autolen = sum(cl.chromsizes[c] for c in CHROMS)
        rows.append(dict(sample=n, has_chrY_in_mcool="chrY" in cl.chromnames,
                         chrX_cis_density_rel_auto=tot["chrX"] / cl.chromsizes["chrX"] / (auto / autolen)))
    df = pd.DataFrame(rows).set_index("sample")
    pf = {re.match(r"GSM\d+_(HSPC-\d+|\d+)", f.name).group(1): f for f in (PRIM / "pairs").glob("*.pairs.gz")}
    for n in [s for s in samples.split(",") if s]:
        cmd = ("zcat %s | awk '!/^#/{c[$2]++; if($2==\"chrY\"||$4==\"chrY\")y++; if($2==\"chrX\"||$4==\"chrX\")x++; t++} "
               "END{print t, x+0, y+0}'" % pf[n])
        t, x, y = map(int, subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.split())
        df.loc[n, ["pairs_total", "pairs_with_chrX_end", "pairs_with_chrY_end"]] = [t, x, y]
        df.loc[n, "chrY_end_per_million"] = y / t * 1e6
        print(n, t, x, y, flush=True)
        df.reset_index().to_csv(OUT / "sex_coverage.csv", index=False)
    df.reset_index().to_csv(OUT / "sex_coverage.csv", index=False)
    print(df.round(4))


def stage_loops():
    """Overlap of the patient-specific loops (Suppl. Table 1, exact 10 kb anchors) and stripe anchors (Suppl. Table 4)."""
    T = REPO / "results/aml_primary/xu_tables"
    l = pd.read_csv(T / "st1_aml_specific_loops.csv", dtype={"patient": str})
    st = pd.read_csv(T / "st4_aml_specific_stripes.csv", dtype={"patient": str})
    sets = {"loops": {p: set(zip(g.chrom1, g.start1, g.chrom2, g.start2)) for p, g in l.groupby("patient")},
            "stripes": {p: set(zip(g.chrom_anchor, g.start_anchor)) for p, g in st.groupby("patient")}}
    rows = []
    for kind, S in sets.items():
        for p, q in itertools.combinations(sorted(S), 2):
            a, b = pairkey(p, q)
            i = len(S[a] & S[b]); u = len(S[a] | S[b])
            rows.append(dict(kind=kind, a=a, b=b, n_a=len(S[a]), n_b=len(S[b]), shared=i, jaccard=i / u if u else np.nan,
                             frac_of_smaller=i / max(1, min(len(S[a]), len(S[b])))))
    df = mark_pair(pd.DataFrame(rows))
    df.to_csv(OUT / "loops_pairs.csv", index=False)
    for kind, d in df.groupby("kind"):
        t = d[d.is_168_496]; o = d[~d.is_168_496]
        print(kind, t[["n_a", "n_b", "shared", "jaccard"]].to_dict("records"), "others median jaccard", o.jaccard.median(),
              "max", o.jaccard.max(), "rank", int((o.jaccard >= t.jaccard.iloc[0]).sum()) + 1, "of", len(d))
        print(o.sort_values("jaccard", ascending=False).head(3)[["a", "b", "n_a", "n_b", "shared", "jaccard"]])


# ------------------------------------------------------------------ plot
def stage_plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    def hl(ax, df, col, title, both=None):
        d = df if both is None else df[both]
        o = d[~d.is_168_496][col]; t = d[d.is_168_496][col].iloc[0]
        ax.hist(o, bins=30, color="#9aa", edgecolor="w"); ax.axvline(t, color="#c33", lw=2)
        ax.set_title(title, fontsize=10); ax.set_xlabel(col)
    sv = pd.read_csv(OUT / "sv_pairs.csv"); hl(ax[0], sv, "sv_100kb", "SVs matched, both ends, 100 kb")
    cn = pd.read_csv(OUT / "cnv_pairs.csv"); hl(ax[1], cn, "jaccard_state", "CNV 500 kb state Jaccard")
    h = pd.read_csv(OUT / "hic_scc_pairs.csv"); hl(ax[2], h, "mean", "Hi-C SCC, mean over chr1-22", both=h.both_aml)
    m = pd.read_csv(OUT / "meth_pairs.csv"); hl(ax[3], m, "r_centred", "WGBS 100 kb, cohort-centred r")
    for a in ax: a.set_ylabel("sample pairs")
    fig.suptitle("168 vs 496 (red line) against all other pairs (grey)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "duplicate_check.png", dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="sv,cnv,loops,sex,hic,meth,plot")
    ap.add_argument("--max-dist", type=int, default=10_000_000)
    ap.add_argument("--res", type=int, default=100_000)
    ap.add_argument("--sex-samples", default="168,496,027,103,018,270", help="samples whose pairs files are streamed for chrY")
    ap.add_argument("--meth-win", type=int, default=100_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.stages.split(","):
        print("=== stage", s, flush=True)
        {"sv": stage_sv, "cnv": stage_cnv, "plot": stage_plot, "loops": stage_loops,
         "hic": lambda: stage_hic(a.max_dist, a.res), "meth": lambda: stage_meth(a.meth_win),
         "sex": lambda: stage_sex(a.res, a.sex_samples)}[s]()


if __name__ == "__main__":
    main()
