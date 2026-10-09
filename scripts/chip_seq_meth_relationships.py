"""How ChIP-seq peak height relates to DNA sequence and to DNA methylation, and
which relationships are shared (registry section 41). CPU only.

Peak height is the track as the Hi-C model sees it: 10 kb bins, log1p,
z-scored with the track statistics of the training checkpoint (the tables of
scripts/meth_track_prediction.py, section 35 part B, are reused). Cells: the
seven training cells with WGBS (GM12878, IMR-90, H1, HepG2, A549, GM23248,
CD14+ monocyte) plus K562 and HMEC (never in a training set). The lifted hg19
RRBS cells (HCT116, MCF-7, HFF) are excluded: RRBS covers 4 to 5 percent of the
1 kb bins, so their methylation features are the cell mean almost everywhere.
HMEC deep and HMEC share inputs and count as one cell; HUVEC has no methylation.

Honest split: every model is trained on the odd chromosomes and scored on the
even chromosomes; a left-out cell (unseen_cell) is in no training set. K562 and
HMEC are scored with the model trained on all training cells (their seen_cell
and unseen_cell results are the same fit and are reported once, as unseen_cell).

Stages
  seqext   extended sequence features (GC and CpG context, strand-separated CTCF
           motif, YY1, ZNF143 and MAZ motifs) aligned to the section 35 tables
  fit      gradient boosting for 13 marks x 11 feature sets x (all-training fit +
           leave-one-cell-out): variance partition, cross-mark redundancy,
           generalisation, grouped permutation importance
  cpg      CpG-level methylation (ENCODE WGBS beds) per cell and chromosome
  sites    genome-wide CTCF motif sites (PWM >= 0.6 of the maximum), ChIP height
           of CTCF, RAD21, SMC3 and two activity marks at each site, CpG
           methylation inside the motif and in its flanks
  motif    motif-level statistics (even chromosomes) and the odd-to-even models
  summary  tables with bootstrap intervals over cells, and the figures

    python scripts/chip_seq_meth_relationships.py --stage seqext fit cpg sites motif summary
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import meth_track_prediction as mt                                  # noqa: E402
import methylation as me                                            # noqa: E402
import multicell_expanded as mx                                     # noqa: E402

OUT = REPO / "results/chip_seq_meth_relationships"
WORK = mx.HM / "runs/chip_meth_rel"
CELLS, TRAIN_CELLS, HELD_CELLS = mt.CELLS, mt.TRAIN_CELLS, mt.HELD_CELLS
CHROMS, ODD, EVEN = mt.CHROMS, mt.ODD, mt.EVEN
MARKS = mt.MARKS
GROUPS = {
    "ctcf_cohesin": ["CTCF", "RAD21", "SMC3"],
    "active": ["H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac"],
    "gene_body": ["H3K36me3", "H3K79me2", "H4K20me1"],
    "repressive": ["H3K27me3", "H3K9me3"],
}
GROUP_OF = {m: g for g, ms in GROUPS.items() for m in ms}
M_LOCAL, M_CTX, SEQ_BASE = mt.F_LOCAL, mt.F_CONTEXT, mt.F_SEQ
EXTRA_MOTIFS = {"yy1": "MA0095.3", "znf143": "MA0088.2", "maz": "MA1522.1"}
SEQ_EXTRA = (["ctcf_fwd", "ctcf_rev", "gc_100k", "gc_1M", "cpg_100k", "cpg_1M", "ctcf_motif_100k"]
             + [f"{k}{s}" for k in EXTRA_MOTIFS for s in ("", "_100k")])
SEQ_EXT = SEQ_BASE + SEQ_EXTRA
METH = M_LOCAL + M_CTX
RES = 10000
N_FIT = 500_000
N_BOOT = 5000
SEED = 20261009
SETS = ["seq_base", "seq", "meth", "both", "both_base", "O", "O+S", "O+M", "O+S+M", "W", "X"]


def log(msg):
    WORK.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(WORK / "chip_meth_rel.log", "a") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------- seqext
def masked_filter(x, ok, size):
    """Mean of x over a window of `size` bins counting only bins with ok."""
    from scipy.ndimage import uniform_filter1d
    num = uniform_filter1d(np.where(ok, x, 0.0), size, mode="nearest")
    den = uniform_filter1d(ok.astype(float), size, mode="nearest")
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)


def stage_seqext():
    f = WORK / "seq_ext.npz"
    if f.exists():
        return
    from hicmultiome.config import Config
    from hicmultiome.motifs import cached_motif_tracks
    from hicmultiome.sequence import cached_sequence_features
    cfg = Config.load(mx.best_ckpt(mt.NORM) / "config.yaml")
    ctcf = next(t.motif for t in cfg.data.tracks if t.name == "CTCF")
    sizes = me.chrom_sizes()
    cols = {k: [] for k in SEQ_EXTRA}
    for ch in CHROMS:
        nb = -(-sizes[ch] // RES)
        sf = cached_sequence_features(cfg.sequence.fasta, ch, ["gc", "cpg"], nb, RES,
                                      cfg.sequence.cache_dir)
        ok = sf[0] > 0
        d = {}
        d["gc_100k"], d["gc_1M"] = (masked_filter(sf[0], ok, 11), masked_filter(sf[0], ok, 101))
        d["cpg_100k"], d["cpg_1M"] = (masked_filter(sf[1], ok, 11), masked_filter(sf[1], ok, 101))
        m = cached_motif_tracks(cfg.sequence.fasta, ch, ctcf, cfg.motifs.dir, nb, RES,
                                cfg.motifs.score_frac, cfg.sequence.cache_dir)
        d["ctcf_fwd"], d["ctcf_rev"] = m[0], m[1]
        d["ctcf_motif_100k"] = masked_filter(m[0] + m[1], ok, 11)
        for k, mid in EXTRA_MOTIFS.items():
            e = cached_motif_tracks(cfg.sequence.fasta, ch, mid, cfg.motifs.dir, nb, RES,
                                    cfg.motifs.score_frac, cfg.sequence.cache_dir)
            d[k] = e[0] + e[1]
            d[k + "_100k"] = masked_filter(d[k], ok, 11)
        for k in SEQ_EXTRA:
            cols[k].append(np.asarray(d[k], np.float32)[:nb])
        log(f"seqext {ch}")
    # dense per chromosome, indexed by the (chrom, pos) of the section 35 tables
    chrom_arr = {ch: {k: cols[k][i] for k in SEQ_EXTRA} for i, ch in enumerate(CHROMS)}
    ref = mt.load(CELLS[0])
    for c in CELLS[1:]:
        o = mt.load(c)
        assert np.array_equal(o["chrom"], ref["chrom"]) and np.array_equal(o["pos"], ref["pos"])
    out = {k: np.zeros(len(ref["pos"]), np.float32) for k in SEQ_EXTRA}
    for i, ch in enumerate(CHROMS):
        s = ref["chrom"] == i
        for k in SEQ_EXTRA:
            out[k][s] = chrom_arr[ch][k][ref["pos"][s]]
    np.savez_compressed(str(f) + ".tmp.npz", **out)
    os.rename(str(f) + ".tmp.npz", f)


# --------------------------------------------------------------------------- fit
_FM = {}


def feature_matrix(cell):
    """Dict of float32 columns of one cell: methylation, sequence, the 13 tracks (NaN if absent)."""
    if cell in _FM:
        return _FM[cell]
    d = mt.load(cell)
    z = np.load(WORK / "seq_ext.npz")
    fm = {k: d[k].astype(np.float32) for k in METH + SEQ_BASE}
    for k in SEQ_EXTRA:
        fm[k] = z[k]
    for m in MARKS:
        fm["y_" + m] = d["y_" + m].astype(np.float32) if "y_" + m in d else np.full(
            len(d["gc"]), np.nan, np.float32)
    fm["_has"] = {m: ("y_" + m in d) for m in MARKS}
    fm["_chrom"] = d["chrom"]
    _FM[cell] = fm
    return fm


def set_columns(name, mark):
    others = [m for m in MARKS if m != mark]
    ob = ["y_" + m for m in others]
    within = ["y_" + m for m in others if GROUP_OF[m] == GROUP_OF[mark]]
    cross = ["y_" + m for m in others if GROUP_OF[m] != GROUP_OF[mark]]
    return {"seq_base": SEQ_BASE, "seq": SEQ_EXT, "meth": METH, "both": SEQ_EXT + METH,
            "both_base": SEQ_BASE + METH, "O": ob, "O+S": ob + SEQ_EXT, "O+M": ob + METH,
            "O+S+M": ob + SEQ_EXT + METH, "W": within, "X": cross}[name]


def perm_groups(name, mark):
    if name == "both":
        return {"meth_local": M_LOCAL, "meth_context": M_CTX, "sequence": SEQ_EXT}
    if name == "O+S+M":
        g = {"marks_" + gn: ["y_" + m for m in ms if m != mark] for gn, ms in GROUPS.items()}
        g = {k: v for k, v in g.items() if v}
        g.update({"meth_local": M_LOCAL, "meth_context": M_CTX, "sequence": SEQ_EXT})
        return g
    return {}


def stats(y, p):
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    vy, vp = y.var(), p.var()
    r = float(np.corrcoef(y, p)[0, 1]) if vp > 0 and vy > 0 else np.nan
    from scipy.stats import rankdata
    sp = float(np.corrcoef(rankdata(y), rankdata(p))[0, 1]) if vp > 0 else np.nan
    # R2 = r^2 - bias_term - scale_term (exact identity, population moments)
    bias = (p.mean() - y.mean()) ** 2 / vy
    scale = (np.sqrt(vp) - r * np.sqrt(vy)) ** 2 / vy if np.isfinite(r) else np.nan
    return {"r2": mt.r2(y, p), "pearson": r, "spearman": sp, "bias_term": bias,
            "scale_term": scale, "y_mean": y.mean(), "y_sd": np.sqrt(vy),
            "p_mean": p.mean(), "p_sd": np.sqrt(vp)}


def fit_job(args):
    mark, name = args
    from threadpoolctl import threadpool_limits
    threadpool_limits(int(os.environ.get("FIT_THREADS", "2")))
    cols = set_columns(name, mark)
    if not cols:
        return [], []
    rng = np.random.default_rng([SEED, MARKS.index(mark), SETS.index(name)])
    have = [c for c in CELLS if feature_matrix(c)["_has"][mark]]
    train_have = [c for c in TRAIN_CELLS if c in have]
    odd_i = [CHROMS.index(x) for x in ODD]
    even_i = [CHROMS.index(x) for x in EVEN]

    def stack(cells, chroms):
        Xs, ys = [], []
        for c in cells:
            fm = feature_matrix(c)
            sel = np.isin(fm["_chrom"], chroms)
            Xs.append(np.stack([fm[k][sel] for k in cols], 1))
            ys.append(fm["y_" + mark][sel])
        return np.concatenate(Xs), np.concatenate(ys)

    def train(cells):
        X, y = stack(cells, odd_i)
        if len(y) > N_FIT:
            i = rng.choice(len(y), N_FIT, replace=False)
            X, y = X[i], y[i]
        return mt.fit_model(X, y)

    rows, imps = [], []
    groups = perm_groups(name, mark)

    def score(model, c, scheme):
        Xt, yt = stack([c], even_i)
        p = model.predict(Xt)
        r = {"mark": mark, "set": name, "cell": c, "scheme": scheme,
             "held_out_cell": c in HELD_CELLS, "n_train_cells": len(train_have), "n": len(yt)}
        r.update(stats(yt, p))
        rows.append(r)
        if scheme == "unseen_cell" and groups:
            base = mt.r2(yt, p)
            for g, gcols in groups.items():
                gi = [cols.index(k) for k in gcols]
                drops = []
                for rep in range(3):
                    Xp = Xt.copy()
                    Xp[:, gi] = Xp[rng.permutation(len(yt))][:, gi]
                    drops.append(base - mt.r2(yt, model.predict(Xp)))
                imps.append({"mark": mark, "set": name, "cell": c, "group": g,
                             "n_features": len(gi), "r2_drop": float(np.mean(drops))})

    if len(train_have) < 2:
        return [], []
    full = train(train_have)
    for c in have:
        if c in TRAIN_CELLS:
            score(full, c, "seen_cell")
        else:
            score(full, c, "unseen_cell")
    for c in train_have:
        others = [x for x in train_have if x != c]
        score(train(others), c, "unseen_cell")
    log(f"fit {mark} {name}")
    return rows, imps


def stage_fit(workers, marks, sets):
    f = OUT / "partition_fits.csv"
    fi = OUT / "permutation_importance.csv"
    jobs = [(m, s) for m in marks for s in sets]
    done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["mark", "set"])
    dimp = pd.read_csv(fi) if fi.exists() else pd.DataFrame()
    have = set(zip(done["mark"], done["set"]))
    jobs = [j for j in jobs if j not in have]
    log(f"fit: {len(jobs)} jobs, {workers} workers")
    os.environ["FIT_THREADS"] = os.environ.get("FIT_THREADS", "2")
    rows, imps = [], []
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        for r, i in ex.map(fit_job, jobs):
            rows += r
            imps += i
            if r and len(rows) >= 0:
                pd.concat([done, pd.DataFrame(rows)]).to_csv(f, index=False, float_format="%.5f")
                pd.concat([dimp, pd.DataFrame(imps)]).to_csv(fi, index=False, float_format="%.5f")


# --------------------------------------------------------------------------- cpg and sites
def parse_cpg_chunk(df):
    """ENCODE WGBS bed9+2 rows -> chrom, C position of the CpG, reads, methylated reads."""
    df = df[df.iloc[:, 0].isin(CHROMS)]
    pos = df.iloc[:, 1].to_numpy(np.int64)
    strand = df.iloc[:, 2].to_numpy()
    pos = np.where(strand == "-", pos - 1, pos)
    cov = df.iloc[:, 3].to_numpy(np.int32)
    meth = np.rint(cov * df.iloc[:, 4].to_numpy(np.float64) / 100.0).astype(np.int32)
    return df.iloc[:, 0].to_numpy(), pos, cov, meth


def cpg_build(cell):
    f = WORK / "cpg" / f"{cell}.npz"
    if f.exists():
        return
    path = me.raw_path(cell)
    acc = {c: [[], [], []] for c in CHROMS}
    n = 0
    for ch in pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 5, 9, 10], chunksize=4_000_000,
                          dtype={0: str, 1: np.int64, 5: str, 9: np.int32, 10: np.float32}):
        chrom, pos, cov, meth = parse_cpg_chunk(ch)
        n += len(pos)
        for c in np.unique(chrom):
            s = chrom == c
            acc[c][0].append(pos[s]); acc[c][1].append(cov[s]); acc[c][2].append(meth[s])
    out = {}
    for c in CHROMS:
        if not acc[c][0]:
            continue
        pos = np.concatenate(acc[c][0]); cov = np.concatenate(acc[c][1])
        meth = np.concatenate(acc[c][2])
        order = np.argsort(pos, kind="stable")
        pos, cov, meth = pos[order], cov[order], meth[order]
        # the two strands of one CpG share the C position: sum them
        u, first = np.unique(pos, return_index=True)
        out[c + "_pos"] = u.astype(np.int32)
        out[c + "_cov"] = np.add.reduceat(cov, first).astype(np.int32)
        out[c + "_meth"] = np.add.reduceat(meth, first).astype(np.int32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(f) + ".tmp.npz", **out)
    os.rename(str(f) + ".tmp.npz", f)
    log(f"cpg {cell}: {n} records")


SITE_FRAC = 0.6          # PWM >= 0.6 of the maximum (the model channel uses 0.8)
CHANNEL_FRAC = 0.8


def motif_pwm():
    from hicmultiome.config import Config
    from hicmultiome.motifs import pwm_logodds, read_jaspar
    cfg = Config.load(mx.best_ckpt(mt.NORM) / "config.yaml")
    mid = next(t.motif for t in cfg.data.tracks if t.name == "CTCF")
    return cfg, pwm_logodds(read_jaspar(Path(cfg.motifs.dir).expanduser() / f"{mid}.jaspar"))


def sites_chrom(ch):
    """Non-overlapping CTCF motif hits of one chromosome: start, strand, log-odds, fraction of max."""
    from hicmultiome.motifs import revcomp_pwm, scan_hits
    from hicmultiome.sequence import read_chrom_codes
    f = WORK / "sites" / f"motif_{ch}.npz"
    if f.exists():
        z = np.load(f)
        return z["pos"], z["strand"], z["score"], z["frac"]
    cfg, pwm = motif_pwm()
    L = pwm.shape[1]
    codes = read_chrom_codes(cfg.sequence.fasta, ch)
    mx_score = float(pwm.max(0).sum())
    pos, strand, score = [], [], []
    for sgn, w in ((0, pwm), (1, revcomp_pwm(pwm))):
        h = scan_hits(codes, w, SITE_FRAC)
        sc = np.zeros(len(h), np.float32)
        for k in range(L):
            sc += w[codes[h + k], k]
        pos.append(h); strand.append(np.full(len(h), sgn, np.int8)); score.append(sc)
    pos, strand, score = np.concatenate(pos), np.concatenate(strand), np.concatenate(score)
    order = np.argsort(-score, kind="stable")           # greedy: strongest hit first, no overlaps
    import bisect
    kept, taken = [], []
    for i in order:
        p = int(pos[i])
        j = bisect.bisect_left(taken, p - L + 1)
        if j < len(taken) and taken[j] <= p + L - 1:
            continue
        bisect.insort(taken, p)
        kept.append(i)
    kept = np.array(sorted(kept, key=lambda i: pos[i]), dtype=np.int64)
    pos, strand, score = pos[kept], strand[kept], score[kept]
    frac = score / mx_score
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, pos=pos.astype(np.int32), strand=strand, score=score, frac=frac)
    return pos.astype(np.int32), strand, score, frac


def window_sums(cp, lo, hi):
    """Reads, methylated reads and CpG count of CpGs with lo <= C position < hi."""
    pos = cp["pos"]
    a, b = np.searchsorted(pos, lo), np.searchsorted(pos, hi)
    cc = np.concatenate([[0], np.cumsum(cp["cov"], dtype=np.int64)])
    cm = np.concatenate([[0], np.cumsum(cp["meth"], dtype=np.int64)])
    return cc[b] - cc[a], cm[b] - cm[a], b - a


MIN_SITE_COV = 10
HEIGHT_MARKS_100 = ["CTCF", "RAD21", "SMC3"]      # +-100 bp around the motif centre
HEIGHT_MARKS_500 = ["H3K27ac", "H3K4me3"]         # +-500 bp (activity proxies)


def sites_cell(cell):
    import pyBigWig
    f = WORK / "sites" / f"table_{cell}.csv.gz"
    if f.exists():
        return
    cpg = np.load(WORK / "cpg" / f"{cell}.npz")
    have = {p.stem: p for p in Path(mt.cell_dir(cell)).glob("*.bigwig")}
    sizes = me.chrom_sizes()
    L = motif_pwm()[1].shape[1]
    parts = []
    bws = {m: pyBigWig.open(str(have[m])) for m in HEIGHT_MARKS_100 + HEIGHT_MARKS_500 if m in have}
    for ch in CHROMS:
        pos, strand, score, frac = sites_chrom(ch)
        t = pd.DataFrame({"cell": cell, "chrom": ch, "pos": pos, "strand": strand,
                          "score": score, "frac": frac})
        cen = pos + L // 2
        for m in HEIGHT_MARKS_100 + HEIGHT_MARKS_500:
            if m not in bws:
                t["h_" + m] = np.nan
                continue
            w = 100 if m in HEIGHT_MARKS_100 else 500
            n = sizes[ch]
            v = [bws[m].stats(ch, int(max(c - w, 0)), int(min(c + w, n)), type="mean")[0] for c in cen]
            v = np.array([np.nan if x is None else x for x in v], np.float64)
            t["h_" + m] = np.log1p(np.clip(np.nan_to_num(v, nan=0.0), 0, None))
        cp = {"pos": cpg[ch + "_pos"], "cov": cpg[ch + "_cov"], "meth": cpg[ch + "_meth"]}
        cov, me_, nc = window_sums(cp, pos, pos + L - 1)          # CpGs wholly inside the motif
        fcov, fme, fn = window_sums(cp, pos - 250, pos + L + 250)
        fcov, fme, fn = fcov - cov, fme - me_, fn - nc            # flanks only
        t["cpg_n_motif"], t["cov_motif"] = nc, cov
        t["meth_motif"] = np.where(cov >= MIN_SITE_COV, me_ / np.maximum(cov, 1), np.nan)
        t["cpg_n_flank"], t["cov_flank"] = fn, fcov
        t["meth_flank"] = np.where(fcov >= MIN_SITE_COV, fme / np.maximum(fcov, 1), np.nan)
        parts.append(t)
    for b in bws.values():
        b.close()
    t = pd.concat(parts, ignore_index=True)
    t["split"] = np.where(t.chrom.isin(ODD), "odd", "even")
    t.to_csv(f, index=False, float_format="%.5f")
    log(f"sites {cell}: {len(t)} sites, {int((t.meth_motif.notna()).sum())} with motif CpG coverage")


def stage_cpg_sites(workers, do_cpg, do_sites):
    if do_cpg:
        with ProcessPoolExecutor(min(workers, 4)) as ex:
            list(ex.map(cpg_build, CELLS))
    if do_sites:
        for ch in CHROMS:
            sites_chrom(ch)
        with ProcessPoolExecutor(min(workers, 9)) as ex:
            list(ex.map(sites_cell, CELLS))


# --------------------------------------------------------------------------- motif analysis
def rank_resid(a, b_cols):
    """Residual of the ranks of a after OLS on the ranks of b_cols (cubic in the first)."""
    from scipy.stats import rankdata
    X = [np.ones(len(a))]
    for j, b in enumerate(b_cols):
        r = (rankdata(b) - 0.5) / len(b)
        X += [r, r ** 2, r ** 3] if j == 0 else [r]
    X = np.stack(X, 1)
    ra = (rankdata(a) - 0.5) / len(a)
    beta, *_ = np.linalg.lstsq(X, ra, rcond=None)
    return ra - X @ beta


def partial_spearman(h, m, covs):
    from scipy.stats import rankdata
    rh = rank_resid(h, covs)
    rm = rank_resid(m, covs)
    return float(np.corrcoef(rh, rm)[0, 1])


def chrom_bootstrap(t, fn, rng, B=500):
    """Percentile interval of fn(table) when whole chromosomes are resampled."""
    ch = t.chrom.unique()
    groups = {c: t[t.chrom == c] for c in ch}
    vals = []
    for _ in range(B):
        s = rng.choice(ch, len(ch), replace=True)
        try:
            vals.append(fn(pd.concat([groups[c] for c in s])))
        except Exception:
            continue
    return tuple(np.nanpercentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def load_sites():
    out = {}
    for c in CELLS:
        f = WORK / "sites" / f"table_{c}.csv.gz"
        if f.exists():
            out[c] = pd.read_csv(f)
    return out


def stage_motif():
    from scipy.stats import spearmanr
    S = load_sites()
    rng = np.random.default_rng(SEED)
    rows = []
    mark_cells = {m: [c for c in S if ("y_" + m) in np.load(mt.OUT / "tables" / f"{c}.npz").files]
                  for m in HEIGHT_MARKS_100}
    for mark in HEIGHT_MARKS_100:
        for cell in mark_cells[mark]:
            t = S[cell]
            t = t[t.split == "even"].copy()
            t["h"] = t["h_" + mark]
            sub = t[t.meth_motif.notna()].copy()
            sub = sub.dropna(subset=["meth_flank", "h_H3K27ac", "h_H3K4me3"])
            if len(sub) < 200:
                continue

            def stat(d, kind):
                if kind == "rho_strength":
                    return float(spearmanr(d.h, d.frac)[0])
                if kind == "rho_meth":
                    return float(spearmanr(d.h, d.meth_motif)[0])
                if kind == "partial_strength":
                    return partial_spearman(d.h.to_numpy(), d.meth_motif.to_numpy(), [d.frac.to_numpy()])
                if kind == "partial_strength_flank":
                    return partial_spearman(d.h.to_numpy(), d.meth_motif.to_numpy(),
                                            [d.frac.to_numpy(), d.meth_flank.to_numpy()])
                if kind == "partial_strength_activity":
                    return partial_spearman(d.h.to_numpy(), d.meth_motif.to_numpy(),
                                            [d.frac.to_numpy(), d.h_H3K27ac.to_numpy(),
                                             d.h_H3K4me3.to_numpy()])
                if kind == "partial_flank_given_strength":
                    return partial_spearman(d.h.to_numpy(), d.meth_flank.to_numpy(), [d.frac.to_numpy()])
                raise KeyError(kind)

            for kind in ("rho_strength", "rho_meth", "partial_strength", "partial_strength_flank",
                         "partial_strength_activity", "partial_flank_given_strength"):
                d_all = t if kind == "rho_strength" else sub
                v = stat(d_all, kind)
                lo, hi = chrom_bootstrap(d_all, lambda d, k=kind: stat(d, k), rng, B=300)
                rows.append({"mark": mark, "cell": cell, "held_out_cell": cell in HELD_CELLS,
                             "statistic": kind, "value": v, "ci_lo": lo, "ci_hi": hi,
                             "n_sites": len(d_all), "n_chrom": d_all.chrom.nunique()})
            # share of sites by methylation level and the median z-height
            for lvl, (a, b) in {"low": (0, 0.2), "mid": (0.2, 0.8), "high": (0.8, 1.01)}.items():
                s = sub[(sub.meth_motif >= a) & (sub.meth_motif < b)]
                rows.append({"mark": mark, "cell": cell, "held_out_cell": cell in HELD_CELLS,
                             "statistic": f"median_height_{lvl}", "value": float(s.h.median()) if len(s) else np.nan,
                             "ci_lo": np.nan, "ci_hi": np.nan, "n_sites": len(s), "n_chrom": s.chrom.nunique()})
            log(f"motif {mark} {cell}: {len(t)} even sites, {len(sub)} with CpG coverage")
    pd.DataFrame(rows).to_csv(OUT / "motif_statistics.csv", index=False, float_format="%.5f")
    motif_models(S, mark_cells)


def motif_models(S, mark_cells):
    """Odd-chromosome training, even-chromosome scoring: does motif methylation add to motif strength."""
    from scipy.stats import rankdata, spearmanr
    feat_sets = {
        "strength": ["frac"],
        "strength+motif_meth": ["frac", "meth_motif"],
        "strength+motif_meth+flank_meth": ["frac", "meth_motif", "meth_flank"],
        "strength+activity": ["frac", "h_H3K27ac", "h_H3K4me3"],
        "strength+activity+motif_meth+flank_meth": ["frac", "h_H3K27ac", "h_H3K4me3",
                                                     "meth_motif", "meth_flank"],
        "strength+flank_meth": ["frac", "meth_flank"],
    }
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    for mark in HEIGHT_MARKS_100:
        have = mark_cells[mark]
        train_have = [c for c in TRAIN_CELLS if c in have]
        Z = {}
        for c in have:
            t = S[c].copy()
            h = t["h_" + mark]
            odd = t.split == "odd"
            # z-score within the cell on its own sites of the SAME split (scale-free evaluation)
            for sp in ("odd", "even"):
                m_ = t.split == sp
                t.loc[m_, "z"] = (h[m_] - h[m_].mean()) / h[m_].std()
            Z[c] = t
        for c in have:
            trn = [x for x in train_have if x != c]
            if len(trn) < 2:
                continue
            tr = pd.concat([Z[x][Z[x].split == "odd"] for x in trn])
            te = Z[c][Z[c].split == "even"]
            te_cpg = te[te.meth_motif.notna()]
            for fs, cols in feat_sets.items():
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                                  min_samples_leaf=50, random_state=0)
                m.fit(tr[cols].to_numpy(), tr["z"].to_numpy())
                for scope, d in (("all_even_sites", te), ("even_sites_with_motif_CpG", te_cpg)):
                    p = m.predict(d[cols].to_numpy())
                    y = d["z"].to_numpy()
                    rows.append({"mark": mark, "cell": c, "held_out_cell": c in HELD_CELLS,
                                 "scheme": "unseen_cell", "features": fs, "scope": scope,
                                 "n_sites": len(y), "n_train_cells": len(trn),
                                 "pearson": float(np.corrcoef(y, p)[0, 1]),
                                 "spearman": float(spearmanr(y, p)[0]),
                                 "r2_affine": float(np.corrcoef(y, p)[0, 1] ** 2)})
        log(f"motif models {mark}")
    pd.DataFrame(rows).to_csv(OUT / "motif_models.csv", index=False, float_format="%.5f")


# --------------------------------------------------------------------------- summary
def boot_median(x, rng, B=N_BOOT):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    idx = rng.integers(0, len(x), (B, len(x)))
    med = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5)), len(x)


def partition_table(R, rng):
    """Per cell and median over cells: variance partition from the sets seq, meth, both (and the
    section 35 style 3-feature sequence set)."""
    out_cell, out_sum = [], []
    for seqset, bothset, tag in (("seq", "both", "seq_ext"), ("seq_base", "both_base", "seq_base")):
        W = {}
        for s in (seqset, "meth", bothset):
            d = R[R["set"] == s].set_index(["mark", "scheme", "cell"])
            W[s] = d
        idx = W["meth"].index.intersection(W[seqset].index).intersection(W[bothset].index)
        for basis in ("r2", "pearson2"):
            def val(s):
                d = W[s].loc[idx]
                return (d["r2"] if basis == "r2" else np.sign(d["pearson"]) * d["pearson"] ** 2)
            S_, M_, B_ = val(seqset), val("meth"), val(bothset)
            t = pd.DataFrame({"sequence": S_, "methylation": M_, "both": B_})
            t["unique_sequence"] = t["both"] - t["methylation"]
            t["unique_methylation"] = t["both"] - t["sequence"]
            t["shared"] = t["sequence"] + t["methylation"] - t["both"]
            t["unexplained"] = 1 - t["both"]
            t["held_out_cell"] = W["meth"].loc[idx, "held_out_cell"]
            t = t.reset_index()
            t["basis"], t["sequence_set"] = basis, tag
            out_cell.append(t)
    C = pd.concat(out_cell)
    parts = ["sequence", "methylation", "both", "unique_sequence", "unique_methylation", "shared"]
    for (mark, scheme, basis, tag), g in C.groupby(["mark", "scheme", "basis", "sequence_set"]):
        r = {"mark": mark, "scheme": scheme, "basis": basis, "sequence_set": tag, "n_cells": len(g)}
        for p in parts:
            m, lo, hi, _ = boot_median(g[p].to_numpy(), rng)
            r[p], r[p + "_lo"], r[p + "_hi"] = m, lo, hi
        gh = g[g.held_out_cell]
        for p in parts:
            r[p + "_heldout_mean"] = gh[p].mean() if len(gh) else np.nan
        out_sum.append(r)
    return C, pd.DataFrame(out_sum)


def stage_summary():
    rng = np.random.default_rng(SEED)
    R = pd.read_csv(OUT / "partition_fits.csv")
    R["pearson2"] = np.sign(R.pearson) * R.pearson ** 2
    C, P = partition_table(R, rng)
    C.to_csv(OUT / "variance_partition_per_cell.csv", index=False, float_format="%.4f")
    P.to_csv(OUT / "variance_partition_summary.csv", index=False, float_format="%.4f")

    # all sets: median and bootstrap over cells of every metric
    rows = []
    for (mark, s, scheme), g in R.groupby(["mark", "set", "scheme"]):
        r = {"mark": mark, "set": s, "scheme": scheme, "n_cells": g.cell.nunique(),
             "n_cells_heldout": int(g.held_out_cell.sum())}
        for k in ("r2", "pearson", "spearman", "pearson2", "bias_term", "scale_term"):
            m, lo, hi, _ = boot_median(g[k].to_numpy(), rng)
            r[k], r[k + "_lo"], r[k + "_hi"] = m, lo, hi
        r["frac_cells_r2_negative"] = float((g.r2 < 0).mean())
        r["frac_cells_r2_negative_pearson_positive"] = float(((g.r2 < 0) & (g.pearson > 0)).mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "metrics_summary.csv", index=False, float_format="%.4f")

    # generalisation: seen vs unseen on the same training cells
    gen = []
    for (mark, s), g in R.groupby(["mark", "set"]):
        a = g[g.scheme == "seen_cell"].set_index("cell")
        b = g[(g.scheme == "unseen_cell") & ~g.held_out_cell].set_index("cell")
        cells = a.index.intersection(b.index)
        if len(cells) < 2:
            continue
        d = {"mark": mark, "set": s, "n_cells": len(cells)}
        for k in ("r2", "pearson", "spearman"):
            d[k + "_seen"] = float(a.loc[cells, k].median())
            d[k + "_unseen"] = float(b.loc[cells, k].median())
            dl = (b.loc[cells, k] - a.loc[cells, k]).to_numpy()
            m, lo, hi, _ = boot_median(dl, rng)
            d[k + "_delta"], d[k + "_delta_lo"], d[k + "_delta_hi"] = m, lo, hi
        gh = g[g.held_out_cell]
        d["r2_heldout_mean"] = float(gh.r2.mean()) if len(gh) else np.nan
        d["pearson_heldout_mean"] = float(gh.pearson.mean()) if len(gh) else np.nan
        d["bias_term_unseen"] = float(b.loc[cells, "bias_term"].median())
        d["scale_term_unseen"] = float(b.loc[cells, "scale_term"].median())
        gen.append(d)
    pd.DataFrame(gen).to_csv(OUT / "generalisation.csv", index=False, float_format="%.4f")

    # permutation importance
    I = pd.read_csv(OUT / "permutation_importance.csv")
    rows = []
    for (mark, s, grp), g in I.groupby(["mark", "set", "group"]):
        m, lo, hi, n = boot_median(g.r2_drop.to_numpy(), rng)
        rows.append({"mark": mark, "set": s, "group": grp, "r2_drop": m, "lo": lo, "hi": hi, "n_cells": n})
    pd.DataFrame(rows).to_csv(OUT / "permutation_importance_summary.csv", index=False, float_format="%.4f")

    # redundancy: gains between sets, per cell, bootstrap over cells
    pairs = [("O", "O+S", "sequence added to other marks"), ("O", "O+M", "methylation added to other marks"),
             ("O+S", "O+S+M", "methylation added to other marks + sequence"),
             ("O+M", "O+S+M", "sequence added to other marks + methylation"),
             ("seq", "O+S", "other marks added to sequence"),
             ("both", "O+S+M", "other marks added to sequence + methylation"),
             ("X", "O", "own-group marks added to cross-group marks"),
             ("W", "O", "cross-group marks added to own-group marks")]
    rows = []
    for mark in MARKS:
        for scheme in ("unseen_cell",):
            for a, b, label in pairs:
                A = R[(R.mark == mark) & (R["set"] == a) & (R.scheme == scheme)].set_index("cell")
                B = R[(R.mark == mark) & (R["set"] == b) & (R.scheme == scheme)].set_index("cell")
                cells = A.index.intersection(B.index)
                if len(cells) < 2:
                    continue
                for k in ("r2", "pearson2"):
                    m, lo, hi, n = boot_median((B.loc[cells, k] - A.loc[cells, k]).to_numpy(), rng)
                    rows.append({"mark": mark, "from_set": a, "to_set": b, "gain": label, "metric": k,
                                 "median_gain": m, "lo": lo, "hi": hi, "n_cells": n})
    pd.DataFrame(rows).to_csv(OUT / "redundancy_gains.csv", index=False, float_format="%.4f")
    figures(R, P)


def figures(R, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    col = {"unique_sequence": "#0072B2", "shared": "#999999", "unique_methylation": "#D55E00"}
    order = (P[(P.scheme == "unseen_cell") & (P.basis == "r2") & (P.sequence_set == "seq_ext")]
             .sort_values("both", ascending=False).mark.tolist())

    def panel(ax, scheme, basis, title):
        d = P[(P.scheme == scheme) & (P.basis == basis) & (P.sequence_set == "seq_ext")].set_index("mark")
        d = d.reindex([m for m in order if m in d.index])
        x = np.arange(len(d))
        pos = np.zeros(len(d)); neg = np.zeros(len(d))
        for p in ("unique_sequence", "shared", "unique_methylation"):
            v = d[p].to_numpy()
            base = np.where(v >= 0, pos, neg)
            ax.bar(x, v, bottom=base, color=col[p], width=0.75, label=p.replace("_", " "))
            pos = pos + np.clip(v, 0, None); neg = neg + np.clip(v, None, 0)
        ax.errorbar(x, d["both"], yerr=[d["both"] - d["both_lo"], d["both_hi"] - d["both"]],
                    fmt="o", color="k", ms=3, lw=1, label="both (median, 95% CI over cells)")
        ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=60, ha="right")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title, fontsize=9)
        for i, (m, n) in enumerate(zip(d.index, d.n_cells)):
            ax.text(i, ax.get_ylim()[0], f"{int(n)}", ha="center", va="bottom", fontsize=6, color="#555")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    panel(axs[0], "unseen_cell", "r2", "unseen_cell, R2 partition")
    panel(axs[1], "seen_cell", "r2", "seen_cell, R2 partition")
    panel(axs[2], "unseen_cell", "pearson2", "unseen_cell, signed Pearson^2 partition")
    axs[0].set_ylabel("variance explained (median over cells)")
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_variance_partition.png", dpi=170)
    plt.close(fig)

    # cross-mark
    M = pd.read_csv(OUT / "metrics_summary.csv")
    M = M[M.scheme == "unseen_cell"]
    sets = [("meth", "methylation only"), ("seq", "sequence only"), ("both", "sequence + methylation"),
            ("W", "own-group marks"), ("X", "other-group marks"), ("O", "all 12 other marks"),
            ("O+S", "other marks + sequence"), ("O+S+M", "other marks + sequence + methylation")]
    T = M.pivot(index="mark", columns="set", values="pearson2").reindex(order)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(T[[s for s, _ in sets]].to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([l for _, l in sets], rotation=40, ha="right")
    ax.set_yticks(range(len(T))); ax.set_yticklabels(T.index)
    for i in range(T.shape[0]):
        for j, (s, _) in enumerate(sets):
            v = T.iloc[i][s]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if v < 0.55 else "k")
    fig.colorbar(im, label="median squared Pearson, unseen_cell")
    fig.tight_layout()
    fig.savefig(OUT / "fig_crossmark_predictability.png", dpi=170)
    plt.close(fig)

    # R2 against Pearson per mark and cell
    d = R[(R["set"] == "both") & (R.scheme == "unseen_cell")]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.scatter(d.pearson, d.r2, c=np.where(d.held_out_cell, "#D55E00", "#0072B2"), s=14)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, xs ** 2, color="k", lw=0.8, label="R2 = Pearson^2 (cell scale known)")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("Pearson"); ax.set_ylabel("R2")
    ax.set_title("unseen_cell, sequence + methylation\nblue training cells, orange K562 and HMEC", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scale_problem.png", dpi=170)
    plt.close(fig)
    motif_figure()


def motif_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = load_sites()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"low": "#0072B2", "mid": "#999999", "high": "#D55E00"}
    levels = {"low": (0, 0.2), "mid": (0.2, 0.8), "high": (0.8, 1.01)}
    panels = [("training cells pooled (even chromosomes)", [c for c in TRAIN_CELLS if c in S and c != "gm23248"]),
              ("K562 (held out)", ["k562"]), ("HMEC (held out)", ["hmec"])]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (title, cells) in zip(axs, panels):
        parts = []
        for c in cells:
            t = S[c]
            t = t[(t.split == "even") & t.meth_motif.notna()].copy()
            if "h_CTCF" not in t or t.h_CTCF.isna().all():
                continue
            full = S[c][S[c].split == "even"].h_CTCF
            t["z"] = (t.h_CTCF - full.mean()) / full.std()
            parts.append(t)
        d = pd.concat(parts)
        edges = np.quantile(d.frac, np.linspace(0, 1, 7))
        d["bin"] = np.clip(np.searchsorted(edges, d.frac, side="right") - 1, 0, 5)
        for lvl, (a, b) in levels.items():
            s = d[(d.meth_motif >= a) & (d.meth_motif < b)]
            g = s.groupby("bin")
            mu, se, xm, n = g.z.mean(), g.z.sem(), g.frac.mean(), g.z.size()
            ok = n >= 15
            ax.errorbar(xm[ok], mu[ok], yerr=1.96 * se[ok], color=colors[lvl], marker="o", ms=3, lw=1,
                        label=f"motif CpG methylation {a:.1f} to {min(b, 1):.1f} (n={len(s)})")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("motif strength (fraction of maximum PWM score)")
        ax.legend(fontsize=6.5, frameon=False)
    axs[0].set_ylabel("CTCF height at the motif (z within cell)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ctcf_height_vs_motif_strength_by_methylation.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["seqext", "fit", "cpg", "sites", "motif", "summary"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--marks", nargs="+", default=MARKS)
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if "seqext" in a.stage:
        stage_seqext()
    if "fit" in a.stage:
        stage_fit(a.workers, a.marks, a.sets)
    if "cpg" in a.stage or "sites" in a.stage:
        stage_cpg_sites(a.workers, "cpg" in a.stage, "sites" in a.stage)
    if "motif" in a.stage:
        stage_motif()
    if "summary" in a.stage:
        stage_summary()


if __name__ == "__main__":
    main()
