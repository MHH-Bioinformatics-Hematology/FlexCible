"""Copy number as an input and structural variants as a loss mask (registry
section 28).

CNV: per cell, Hi-C read coverage (row sums of the raw contact map, cis and
trans) in 100 kb bins, depth-normalised, as log2 ratio against the median of
all 15 maps (12 training cells, K562, HMEC, HUVEC), median-centred and
corrected for GC content (median per GC class), segmented per autosome by
binary segmentation (segments of at least
MIN_SEG_BINS bins, split while the two-sample t statistic exceeds T_SPLIT).
A segment whose mean |log2| is below CALL_LOG2 counts as copy neutral. The
calls go to two bigWigs per cell under <HM>/cnv/<cell>/: CNVgain = max(log2,
0), CNVloss = max(-log2, 0), so a raw zero (absent track, track dropout, a
cell without the channel) reads as copy neutral. The coverage of a map is a
one-dimensional summary of the measured Hi-C, also for held-out cells; the
channel stands for a copy-number profile that would come from WGS or arrays
in practice, and the segmentation keeps only Mb-scale steps.

SV: EagleC2 predictSV (env eaglec, pretrained models under <HM>/sv) on the
ICE-balanced 10 and 25 kb maps of the five cancer training lines and K562;
both ends of every call are breakpoints; the karyotypically normal cells
carry no mask. A cell where EagleC2 fails falls back to its CNV
change points. The breakpoints go to <HM>/sv/breakpoints.json for
train.sv_mask.

    python scripts/cnv_sv.py --stage coverage cnv          # CPU
    python scripts/cnv_sv.py --stage mcool eaglec breakpoints
    python scripts/cnv_sv.py --stage summary
    python scripts/cnv_sv.py --stage compare     # after all12refres_drop_cnv_10kb
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HM = Path("/mnt/processed/wolffjoa/hic-multiome")
CNV = HM / "cnv"
SV = HM / "sv"
RES_OUT = REPO / "results/cnv_sv"
ENC = HM / "encode_hic"
PP = HM.parent / "restart_of_prediction_project/originalData/4Dnucleosome"
EAGLEC = Path.home() / "miniconda3/envs/eaglec/bin/predictSV"
MODELS = SV / "EagleC2-models"
BIN = 100_000
MIN_SEG_BINS = 10              # 1 Mb
T_SPLIT = 8.0
CALL_LOG2 = 0.3
GC_BINS = 50
FASTA = Path.home() / "data/hic-multiome/genome/hg38.fa"
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
TRACKS = ("CNVgain", "CNVloss")
# cell -> raw contact map: (uri, resolution of that map)
MCOOL = {
    "gm12878": PP / "GM12878_4DNFIXP4QG5B.mcool",
    "h1": PP / "H1_4DNFI6HDY7WZ.mcool",
    "k562": PP / "K562_4DNFI18UHVRO.mcool",
    "imr90": HM.parent / "hyperparameter_publication/imr90.mcool",
    "hff": HM / "matrices/hff_microc.mcool",
    "hct116": HM / "matrices/hct116_untreated.mcool",
}
ENCODE_CELLS = ["hepg2", "a549", "nhek", "gm23248", "mcf7", "cd14mono", "pc9",
                "hmec_deep", "huvec_deep"]
CELLS = list(MCOOL) + ENCODE_CELLS
ALIAS = {"hmec": "hmec_deep", "huvec": "huvec_deep"}
# EagleC2 on the cancer lines (training cells first, then the held-out K562);
# the karyotypically normal cells carry no SV mask. A whole genome takes
# EagleC2 hours on the CPU.
SV_CELLS = ["hepg2", "a549", "mcf7", "pc9", "hct116", "k562"]
SV_RES = [10000, 25000]


def log(msg):
    CNV.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(CNV / "cnv_sv.log", "a") as fh:
        fh.write(line + "\n")


def cnv_dir(cell: str) -> Path:
    return CNV / ALIAS.get(cell, cell)


def has_cnv(cell: str) -> bool:
    return all((cnv_dir(cell) / f"{t}.bigwig").exists() for t in TRACKS)


def raw_uri(cell: str, res: int) -> str:
    if cell in MCOOL:
        return f"{MCOOL[cell]}::resolutions/{res}"
    return str(ENC / f"{cell}_{res // 1000}kb.cool")


# ----------------------------------------------------------------- coverage
def marginal(cell: str) -> pd.Series:
    """Raw coverage per 100 kb bin (both pixel ends; diagonal once)."""
    out = CNV / "coverage" / f"{cell}.csv.gz"
    if out.exists():
        return pd.read_csv(out, index_col=[0, 1])["coverage"]
    import cooler
    import h5py
    res = BIN if cell in MCOOL else 25000
    clr = cooler.Cooler(raw_uri(cell, res))
    bins = clr.bins()[["chrom", "start"]][:]
    tot = np.zeros(len(bins))
    path, grp = (clr.filename, clr.root)
    with h5py.File(path, "r") as f:
        px = f[grp]["pixels"]
        n = px["count"].shape[0]
        step = 50_000_000
        for i in range(0, n, step):
            b1 = px["bin1_id"][i:i + step]
            b2 = px["bin2_id"][i:i + step]
            ct = px["count"][i:i + step].astype(np.float64)
            tot += np.bincount(b1, ct, minlength=len(bins))
            off = b1 != b2
            tot += np.bincount(b2[off], ct[off], minlength=len(bins))
    key = (bins["start"].values // BIN) * BIN
    s = pd.Series(tot).groupby([bins["chrom"].astype(str).values, key]).sum()
    s.index.names = ["chrom", "start"]
    s.name = "coverage"
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(out)
    log(f"[coverage] {cell}: {tot.sum():.3g} contacts ends over {len(s)} bins")
    return s


def stage_coverage(jobs: int):
    with ProcessPoolExecutor(jobs) as ex:
        list(ex.map(marginal, CELLS))


# ---------------------------------------------------------------- segmentation
def _norm_chrom(c: str) -> str:
    return c if c.startswith("chr") else f"chr{c}"


def binary_segment(x: np.ndarray, sd: float, min_len: int = MIN_SEG_BINS,
                   t_split: float = T_SPLIT) -> list[tuple[int, int]]:
    """Recursive binary segmentation of x into [a, b) pieces."""
    out = []

    def rec(a, b):
        n = b - a
        if n < 2 * min_len:
            out.append((a, b))
            return
        cs = np.cumsum(x[a:b])
        k = np.arange(min_len, n - min_len + 1)
        ml = cs[k - 1] / k
        mr = (cs[-1] - cs[k - 1]) / (n - k)
        t = np.abs(ml - mr) / (sd * np.sqrt(1.0 / k + 1.0 / (n - k)))
        j = int(np.argmax(t))
        if t[j] < t_split:
            out.append((a, b))
            return
        rec(a, a + int(k[j]))
        rec(a + int(k[j]), b)

    rec(0, len(x))
    return sorted(out)


def stage_cnv():
    import pyBigWig
    M = pd.DataFrame({c: marginal(c) for c in CELLS})
    M.index = pd.MultiIndex.from_arrays([[_norm_chrom(str(c)) for c in
                                          M.index.get_level_values(0)],
                                         M.index.get_level_values(1)])
    M = M.groupby(level=[0, 1]).sum()
    M = M[M.index.get_level_values(0).isin(AUTOSOMES)].fillna(0)
    M = M / M.sum()
    ref = M.median(axis=1)
    ok = ref > 0.2 * ref[ref > 0].median()
    seg_rows, prof_rows = [], []
    sizes = {}
    gc = gc_content(M.index[ok])
    gq = pd.Series(pd.qcut(gc, GC_BINS, labels=False, duplicates="drop"), index=gc.index)
    for cell in CELLS:
        r = np.log2(M.loc[ok, cell].clip(lower=1e-12) / ref[ok])
        # GC correction: subtract the median log2 ratio of the bin's GC class
        # (coverage of the protocols follows GC by up to r = 0.8 in copy-
        # neutral cells); bins without GC (assembly gaps) dropped
        r = r[gq.notna()]
        r = r - r.groupby(gq[gq.notna()]).transform("median")
        r = r - r.median()
        d = np.diff(r.values)
        sd = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
        calls = {}
        for chrom in AUTOSOMES:
            g = r.loc[chrom] if chrom in r.index.get_level_values(0) else None
            if g is None or len(g) == 0:
                continue
            starts = g.index.values
            for a, b in binary_segment(g.values, sd):
                m = float(np.mean(g.values[a:b]))
                # homozygous deletions reach log2 -8: clipped at +-3
                v = float(np.clip(m, -3, 3)) if abs(m) >= CALL_LOG2 else 0.0
                seg_rows.append({"cell": cell, "chrom": chrom, "start": int(starts[a]),
                                 "end": int(starts[b - 1] + BIN), "bins": b - a,
                                 "mean_log2": m, "call_log2": v})
                calls.setdefault(chrom, []).append((int(starts[a]), int(starts[b - 1] + BIN), v))
            prof_rows.append(pd.DataFrame({"cell": cell, "chrom": chrom, "start": starts,
                                           "log2": g.values}))
        _write_bigwigs(cell, calls, sizes or _chrom_sizes(sizes))
        log(f"[cnv] {cell}: noise sd {sd:.3f}, "
            f"{sum(1 for s in seg_rows if s['cell'] == cell and s['call_log2'] != 0)} "
            "non-neutral segments")
    S = pd.DataFrame(seg_rows)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(RES_OUT / "cnv_segments.csv", index=False, float_format="%.4f")
    pd.concat(prof_rows).to_csv(CNV / "cnv_log2_100kb.csv.gz", index=False,
                                float_format="%.4f")
    return S


def gc_content(index) -> pd.Series:
    """GC fraction of every 100 kb bin (NaN for bins under half sequenced)."""
    f = CNV / "gc_100kb.csv.gz"
    if f.exists():
        g = pd.read_csv(f).set_index(["chrom", "start"])["gc"]
    else:
        import pyfaidx
        fa = pyfaidx.Fasta(str(FASTA))
        rows = []
        for c in AUTOSOMES:
            seq = str(fa[c][:]).upper()
            for st in range(0, len(seq), BIN):
                w = seq[st:st + BIN]
                n = len(w) - w.count("N")
                rows.append((c, st, (w.count("G") + w.count("C")) / n if n > BIN // 2 else np.nan))
        g = pd.DataFrame(rows, columns=["chrom", "start", "gc"])
        g.to_csv(f, index=False)
        g = g.set_index(["chrom", "start"])["gc"]
    return g.reindex(index)


def _chrom_sizes(sizes: dict) -> dict:
    import pyBigWig
    bw = pyBigWig.open(str(HM / "hg38_imr90" / "CTCF.bigwig"))
    sizes.update({c: n for c, n in bw.chroms().items() if c in AUTOSOMES})
    bw.close()
    return sizes


def _write_bigwigs(cell: str, calls: dict, sizes: dict):
    import pyBigWig
    d = CNV / cell
    d.mkdir(parents=True, exist_ok=True)
    for t, sign in zip(TRACKS, (1.0, -1.0)):
        tmp = d / f"{t}.bigwig.part"
        bw = pyBigWig.open(str(tmp), "w")
        bw.addHeader([(c, sizes[c]) for c in AUTOSOMES])
        for c in AUTOSOMES:
            segs = sorted(calls.get(c, []))
            ch, st, en, val = [], [], [], []
            pos = 0
            for a, b, v in segs:
                a, b = max(a, pos), min(b, sizes[c])
                if b <= a:
                    continue
                if a > pos:
                    ch.append(c); st.append(pos); en.append(a); val.append(0.0)
                ch.append(c); st.append(a); en.append(b); val.append(max(sign * v, 0.0))
                pos = b
            if pos < sizes[c]:
                ch.append(c); st.append(pos); en.append(sizes[c]); val.append(0.0)
            bw.addEntries(ch, st, ends=en, values=val)
        bw.close()
        tmp.rename(d / f"{t}.bigwig")


# ------------------------------------------------------------------- EagleC2
def mcool_path(cell: str) -> Path:
    return MCOOL[cell] if cell in MCOOL else SV / "mcool" / f"{cell}.mcool"


def stage_mcool():
    """ENCODE cells: one mcool holding copies of the 10 and 25 kb cools."""
    import h5py
    for cell in SV_CELLS:
        if cell in MCOOL:
            continue
        dst = mcool_path(cell)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        with h5py.File(tmp, "w") as out:
            g = out.create_group("resolutions")
            for res in SV_RES:
                with h5py.File(ENC / f"{cell}_{res // 1000}kb.cool", "r") as src:
                    grp = g.create_group(str(res))
                    for k in src:
                        src.copy(k, grp)
                    for k, v in src.attrs.items():
                        grp.attrs[k] = v
        tmp.rename(dst)
        log(f"[mcool] {cell}: {dst}")


def eaglec_out(cell: str) -> Path:
    return SV / "eaglec" / f"{cell}.SV_calls.txt"


def run_eaglec(cell: str, nproc: int) -> int:
    out = eaglec_out(cell)
    if out.exists() or (SV / "eaglec" / f"{cell}.failed").exists():
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = SV / "eaglec" / cell
    cmd = [str(EAGLEC), "--mcool", str(mcool_path(cell)),
           "--resolutions", ",".join(map(str, SV_RES)), "--high-res", str(SV_RES[0]),
           "--balance-type", "ICE", "--model-path", str(MODELS), "-O", str(prefix),
           "-g", "hg38", "-p", str(nproc), "--cpu"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
               TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
    t0 = time.time()
    with open(SV / "eaglec" / f"{cell}.log", "a") as fh:
        fh.write(f"[cmd] {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=SV, env=env)
    found = sorted(SV.glob(f"eaglec/{cell}*.txt"))
    calls = [f for f in found if f != out and "SV" in f.name.upper()] or \
            [f for f in found if f != out]
    if rc == 0 and not out.exists() and calls:
        calls[0].rename(out)
    if not (rc == 0 and out.exists()):
        (SV / "eaglec" / f"{cell}.failed").write_text(f"rc={rc}\n")
    log(f"[eaglec] {cell} rc={rc} {time.time() - t0:.0f} s -> {[f.name for f in calls]}")
    return rc


def eaglec_finished() -> bool:
    """Every SV cell has its calls or a failure marker (fallback: CNV change points)."""
    return all(eaglec_out(c).exists() or (SV / "eaglec" / f"{c}.failed").exists()
               for c in SV_CELLS)


def stage_eaglec(nproc: int):
    if not MODELS.exists():
        raise SystemExit(f"{MODELS} missing: download EagleC2-models.zip into {SV}")
    for cell in SV_CELLS:
        run_eaglec(cell, nproc)


def read_eaglec(cell: str) -> pd.DataFrame | None:
    f = eaglec_out(cell)
    if not f.exists():
        return None
    d = pd.read_csv(f, sep="\t", comment=None)
    d.columns = [c.lstrip("#").strip() for c in d.columns]
    return d


def cnv_change_points(cell: str, S: pd.DataFrame) -> list[list]:
    """Boundaries between adjacent segments of different called state."""
    out = []
    for chrom, g in S[S.cell == cell].sort_values("start").groupby("chrom"):
        v = g.call_log2.values
        for i in range(1, len(g)):
            if abs(v[i] - v[i - 1]) >= 2 * CALL_LOG2:
                out.append([chrom, int(g.start.values[i])])
    return out


def stage_breakpoints():
    for cell in SV_CELLS:          # stale failure markers next to real calls
        fm = SV / "eaglec" / f"{cell}.failed"
        if eaglec_out(cell).exists() and fm.exists():
            fm.unlink()
    S = pd.read_csv(RES_OUT / "cnv_segments.csv")
    bp, src, rows = {}, {}, []
    for cell in SV_CELLS:
        d = read_eaglec(cell)
        if d is not None:
            pts = []
            for r in d.itertuples(index=False):
                c1, p1, c2, p2 = r[0], int(r[1]), r[2], int(r[3])
                pts += [[_norm_chrom(str(c1)), p1], [_norm_chrom(str(c2)), p2]]
                rows.append({"cell": cell, "source": "eaglec2", "chrom1": c1, "pos1": p1,
                             "chrom2": c2, "pos2": p2,
                             "type": "inter" if c1 != c2 else "intra"})
            src[cell] = "eaglec2"
        else:
            pts = cnv_change_points(cell, S)
            for c, p in pts:
                rows.append({"cell": cell, "source": "cnv_change_point", "chrom1": c,
                             "pos1": p, "chrom2": c, "pos2": p, "type": "cnv"})
            src[cell] = "cnv_change_point"
        bp[cell] = [p for p in pts if p[0] in AUTOSOMES]
    SV.mkdir(parents=True, exist_ok=True)
    (SV / "breakpoints.json").write_text(json.dumps(
        {"source": src, "breakpoints": bp}, indent=1))
    RES_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES_OUT / "sv_calls.csv", index=False)
    log(f"[breakpoints] {({c: len(v) for c, v in bp.items()})} sources {src}")


def stage_summary():
    S = pd.read_csv(RES_OUT / "cnv_segments.csv")
    S["mb"] = (S.end - S.start) / 1e6
    rows = []
    for cell, g in S.groupby("cell"):
        tot = g.mb.sum()
        rows.append({"cell": cell, "genome_mb": tot,
                     "gain_mb": g.mb[g.call_log2 > 0].sum(),
                     "loss_mb": g.mb[g.call_log2 < 0].sum(),
                     "altered_frac": g.mb[g.call_log2 != 0].sum() / tot,
                     "segments": len(g)})
    T = pd.DataFrame(rows)
    sv = RES_OUT / "sv_calls.csv"
    if sv.exists():
        C = pd.read_csv(sv)
        T = T.merge(C.groupby("cell").agg(sv_calls=("pos1", "size"),
                                          sv_source=("source", "first")).reset_index(),
                    on="cell", how="left")
    T.to_csv(RES_OUT / "cnv_sv_summary.csv", index=False, float_format="%.3f")
    print(T.to_string(index=False))


CANCER = ["hct116", "hepg2", "a549", "mcf7", "pc9", "k562"]
COMPARE = {"all12refres_drop_cnv_10kb": "all12refres_drop_10kb"}


def stage_compare():
    """Per cell: the CNV model against the same model without CNV channels
    and mask (crosscell predictions, test regions), cancer and other cells
    apart; held-out cells marked."""
    sys.path.insert(0, str(REPO / "scripts"))
    import multicell_expanded as mx
    rows = []
    for new, old in COMPARE.items():
        dn, do = (mx.out_of(n) / n / "crosscell" for n in (new, old))
        for f in sorted(dn.glob("predict_*.csv")):
            cell = f.stem.replace("predict_", "")
            g = do / f.name
            if not g.exists():
                continue
            a, b = pd.read_csv(f), pd.read_csv(g)
            m = a[["region", "model"]].merge(b[["region", "model"]], on="region",
                                             suffixes=("", "_base"))
            cons = [c for c in a.columns if c.startswith("base_consensus")]
            rows.append({"model": new, "baseline_model": old, "cell": cell,
                         "group": "cancer" if cell in CANCER else "other",
                         "held_out": cell not in mx.TRAIN12, "n_regions": len(m),
                         "median": m.model.median(), "median_base": m.model_base.median(),
                         "delta_median": m.model.median() - m.model_base.median(),
                         "frac_regions_better": float((m.model > m.model_base).mean()),
                         "consensus_median": a[cons[0]].median() if cons else np.nan})
    T = pd.DataFrame(rows)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(RES_OUT / "cnv_compare.csv", index=False, float_format="%.4f")
    print(T.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["coverage", "cnv", "mcool", "eaglec", "breakpoints", "summary",
                             "compare"])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    for s in a.stage:
        if s == "coverage":
            stage_coverage(a.jobs)
        elif s == "cnv":
            stage_cnv()
        elif s == "mcool":
            stage_mcool()
        elif s == "eaglec":
            stage_eaglec(a.nproc)
        elif s == "breakpoints":
            stage_breakpoints()
        elif s == "summary":
            stage_summary()
        elif s == "compare":
            stage_compare()


if __name__ == "__main__":
    sys.exit(main())
