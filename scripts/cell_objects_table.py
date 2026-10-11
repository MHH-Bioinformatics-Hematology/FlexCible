"""Cell-specific loops and stripes of every training cell line, with matched shared controls (registry 66).

All-cell generalisation of the AML objects table (scripts/aml_objects_table.py, registry section 63):
the same table columns, so hicmultiome/aml_objects.py and the trainer read it unchanged
(train.object_scope "all": data.cell_objects_table beside data.aml_objects_table).

The rule "this chromatin pattern leads to a loop / stripe" is cell-type independent, so the objects
of a cell are defined by one cell-agnostic procedure from its own Hi-C and its reference, the map the
model also reads (leave-one-out consensus of the other 11 training cells; the 12-cell consensus for a
held-out cell). Definitions follow scripts/aml_loop_classifier.py (donut enrichment of
hicmultiome.bioscore, 10 kb, inner 1 / outer 4, local maxima, 6 to 150 bins; stripe rectangle of an
anchor and an extent block, mean log O/E) with the thresholds below. The functions are separable on
purpose: `loop_peaks`, `stripe_scores`, `classify_*` and `match_controls` take arrays, so that a
unified object definition (scripts/crosscell_rule_transfer.py, being developed elsewhere; not present
when this file was written) can replace `call_cell` without touching the table assembly.

Two sources, one schema (`--source`). `local` (this file's own Hi-C calls, below) is self-contained.
`rule_transfer` converts the cell objects of scripts/crosscell_rule_transfer.py (stage objects, written to
$HM/crosscell_rules/cell_objects.csv.gz: the same operational definitions as the AML objects, candidates on
peak bins of H3K27ac / CTCF, specific = e_cell >= 0.5 and gain over the consensus >= 0.30, stripes 4 x 40 bins
at an offset of 50 bins, shared controls A_ref matched on distance, orientation and the consensus' CTCF and
H3K27ac) into this table: training cells to the training table with the controls A_ref, K562 / HMEC / HUVEC
(deep maps) to the evaluation table, `n_train_other` counted over the specific objects of the other training
cells. `auto` (default) takes `rule_transfer` when that file exists and `local` otherwise, so that one
definition of a cell-specific object serves the rule analysis and the training once that stage has run. The
generated run configs read the same output path either way.

Thresholds of `local` (documented here, tested on synthetic maps in tests/test_diffmarks.py)
  loops    local maximum of the log2 donut enrichment of the cell, >= E_PEAK (0.4); distance 6 to 150 bins
           cell-specific   e_cell >= E_MIN (0.6) and the 3 x 3 maximum of the reference enrichment <= E_REF_MAX (0.2)
           shared          e_cell >= E_MIN and the reference 3 x 3 maximum >= E_MIN (the classifier's rule)
  stripes  anchor of RA = 4 bins (40 kb), extent block of EB = 20 bins (200 kb) adjacent to the anchor,
           to the right and to the left; score = mean log O/E of the rectangle (anchor rows x extent
           columns), at least half of its pixels measured; local maxima along the chromosome (+-RA bins)
           cell-specific   s_cell >= S_MIN (0.30) and s_cell - s_ref >= S_DELTA (0.25)
           shared          s_cell >= S_MIN and s_ref >= S_MIN
  depth   maps of different depth give very different numbers of candidates (noise peaks of a shallow map
           are "specific"): per cell the TOP_LOOPS (400) strongest specific loops (by e_cell) and the
           TOP_STRIPES (200) strongest specific stripes (by s_cell) are kept, the order of magnitude of
           the AML patients' private loops (about 230 per patient)
  controls one shared object per specific object, within the cell, never within 2 bins of a specific
           object: loops matched on log distance (+-15 percent) and on the anchors' CTCF and H3K27ac
           (each within 0.3 sd of the pool, the cell's own ChIP); stripes on the anchor's CTCF and
           H3K27ac and the direction. Orientation of the CTCF motif is not matched (the AML classifier
           matches it); unmatched specific objects keep no control.
  n_train_other  number of other training cells carrying the object (loops: a peak with e >= E_MIN within one bin in
           both anchors; stripes: s >= S_MIN within one bin at the same direction); 0 = private

Leakage. A training table holds the 12 training cells only (`assert_no_leak`, run before writing and in
the tests); the reference of a cell never contains the cell (reference.assert_reference_excludes). The
held-out cells (K562, HMEC, HUVEC) are written to a separate evaluation table
(cell_objects_heldout.csv.gz, reference = consensus12) that the trainer never reads (`load_objects`
rejects cells that are not training samples of the run).

    python scripts/cell_objects_table.py --stage calls build [--cells gm12878 ...] [--heldout]
Run with the hicdiff conda env. Output: results/cell_objects/, share HM/cell_objects/.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

HM = Path("/mnt/processed/wolffjoa/hic-multiome")
SHARE = HM / "cell_objects"
WORK = SHARE / "work"
OUT = REPO / "results/cell_objects"
RES = 10000
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
MIN_BINS, MAX_BINS = 6, 150
E_PEAK, E_MIN, E_REF_MAX = 0.4, 0.6, 0.2
RA, EB = 4, 20
S_MIN, S_DELTA = 0.30, 0.25
WIN, STRIDE, MARGIN = 420, 250, 10          # loop windows (as scripts/aml_loop_classifier.py)
SWIN, SOWN = 420, 300                       # stripe windows; anchors a in [30, 330) are own
SPAD = 30
DIST_TOL = 0.15
MARK_TOL = 0.3
NEAR = 2
TOP_LOOPS, TOP_STRIPES = 400, 200          # specific objects kept per cell, strongest first
SEED = 0
COLS = ["fold", "patient", "cell", "kind", "role", "set", "pair_id", "chrom", "a_start", "a_end",
        "b_start", "b_end", "pair_type", "n_train_other"]
HELDOUT = ["k562", "hmec_deep", "huvec_deep"]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def table_path(heldout: bool = False) -> Path:
    """Table the generated configs read (on the share, identical on both machines)."""
    return SHARE / ("cell_objects_heldout.csv.gz" if heldout else "cell_objects.csv.gz")


# =============================================================== pure functions (arrays in, arrays out)
def donut_log2(M: np.ndarray) -> np.ndarray:
    """log2 donut enrichment of a log O/E window (the classifier's setting: no coarsening, ring 1 to 4)."""
    from hicmultiome.bioscore import BioParams, local_enrichment
    prm = BioParams(loop_coarsen=1, donut_inner=1, donut_outer=4)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log2(local_enrichment(np.exp(M), prm))


def loop_peaks(Lc: np.ndarray, Lr: np.ndarray, s: int, n: int, first: bool, e_peak: float = E_PEAK,
               margin: int = MARGIN, stride: int = STRIDE) -> pd.DataFrame:
    """Local maxima of the cell's enrichment window Lc (distance MIN_BINS..MAX_BINS) with the reference
    enrichment Lr at the same place (3 x 3 maximum), the rows the window owns, global bins."""
    from scipy.ndimage import maximum_filter
    mp = maximum_filter(np.nan_to_num(Lc, nan=-9), size=3)
    mr = maximum_filter(np.nan_to_num(Lr, nan=-9), size=3)
    r0 = 0 if first else margin
    is_max = (np.nan_to_num(Lc, nan=-9) == mp) & (Lc > e_peak)
    ii, jj = np.nonzero(np.triu(is_max, MIN_BINS))
    keep = (jj - ii <= MAX_BINS) & (ii >= r0) & (ii < margin + stride) & (s + jj < n)
    ii, jj = ii[keep], jj[keep]
    return pd.DataFrame({"i": s + ii, "j": s + jj, "e_cell": Lc[ii, jj], "e_ref": mr[ii, jj]})


def stripe_scores(M: np.ndarray, ra: int = RA, eb: int = EB) -> tuple[np.ndarray, np.ndarray]:
    """Mean log O/E of the stripe rectangle for every anchor start a of a window matrix M (NaN =
    unmeasured): (right, left), right = rows a..a+ra-1 x columns a+ra..a+ra+eb-1, left = the same rows
    x columns a-eb..a-1. NaN where the rectangle leaves the window or fewer than half of its pixels
    are measured."""
    n = M.shape[0]
    ok = np.isfinite(M)
    S = np.zeros((n + 1, n + 1))
    C = np.zeros((n + 1, n + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(np.where(ok, M, 0.0), 0), 1)
    C[1:, 1:] = np.cumsum(np.cumsum(ok.astype(float), 0), 1)

    def rect(r0, r1, c0, c1):
        return (S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0], C[r1, c1] - C[r0, c1] - C[r1, c0] + C[r0, c0])

    a = np.arange(n)
    right, left = np.full(n, np.nan), np.full(n, np.nan)
    m = a + ra + eb <= n
    s_, c_ = rect(a[m], a[m] + ra, a[m] + ra, a[m] + ra + eb)
    right[m] = np.where(c_ >= 0.5 * ra * eb, s_ / np.maximum(c_, 1), np.nan)
    m = (a - eb >= 0) & (a + ra <= n)
    s_, c_ = rect(a[m], a[m] + ra, a[m] - eb, a[m])
    left[m] = np.where(c_ >= 0.5 * ra * eb, s_ / np.maximum(c_, 1), np.nan)
    return right, left


def local_max_1d(x: np.ndarray, rad: int = RA) -> np.ndarray:
    """True where x is the maximum of its +-rad neighbourhood (NaN never)."""
    from scipy.ndimage import maximum_filter1d
    v = np.nan_to_num(x, nan=-1e9)
    return (v == maximum_filter1d(v, size=2 * rad + 1)) & np.isfinite(x)


def classify_loops(P: pd.DataFrame) -> pd.DataFrame:
    """Adds `cls` = specific | shared | none to loop peaks (columns e_cell, e_ref)."""
    P = P.copy()
    P["cls"] = np.where((P.e_cell >= E_MIN) & (P.e_ref <= E_REF_MAX), "specific",
                        np.where((P.e_cell >= E_MIN) & (P.e_ref >= E_MIN), "shared", "none"))
    return P


def classify_stripes(s_cell: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
    """Per anchor: 'specific' | 'shared' | 'none' from the rectangle scores of the cell and the reference."""
    peak = local_max_1d(s_cell)
    spec = peak & (s_cell >= S_MIN) & (s_cell - s_ref >= S_DELTA)
    shared = peak & (s_cell >= S_MIN) & (s_ref >= S_MIN)
    return np.where(spec, "specific", np.where(shared, "shared", "none"))


def match_controls(spec_f: np.ndarray, pool_f: np.ndarray, spec_d: np.ndarray | None,
                   pool_d: np.ndarray | None, forbidden: np.ndarray, rng) -> list[int]:
    """One shared object (index into the pool, -1 = none) per specific object, no pool object twice.
    spec_f / pool_f: (n, k) matching features (anchor marks; NaN columns are ignored); spec_d / pool_d:
    log distances to match within DIST_TOL (None: no distance constraint, stripes); forbidden: pool
    objects that may not be used (near a specific object). Mark tolerance: MARK_TOL times the pool sd."""
    k = spec_f.shape[1] if spec_f.ndim == 2 else 0
    sd = np.nanstd(pool_f, axis=0) if len(pool_f) and k else np.zeros(k)
    used = np.zeros(len(pool_f), bool) | forbidden
    out = []
    for q in rng.permutation(len(spec_f)):
        ok = ~used
        if spec_d is not None:
            ok &= np.abs(pool_d - spec_d[q]) <= DIST_TOL
        cost = np.zeros(len(pool_f))
        for c in range(k):
            if not np.isfinite(spec_f[q, c]) or sd[c] <= 0:
                continue
            d = np.abs(pool_f[:, c] - spec_f[q, c])
            ok &= np.isfinite(d) & (d <= MARK_TOL * sd[c])
            cost += np.nan_to_num(d / (MARK_TOL * sd[c]), nan=0.0)
        if spec_d is not None:
            cost += np.abs(pool_d - spec_d[q]) / DIST_TOL
        idx = np.nonzero(ok)[0]
        if not len(idx):
            out.append((q, -1))
            continue
        pick = int(idx[np.argmin(cost[idx])])
        used[pick] = True
        out.append((q, pick))
    res = [-1] * len(spec_f)
    for q, p in out:
        res[q] = p
    return res


def near_any(chrom_bins: np.ndarray, centers: np.ndarray, rad: int = NEAR) -> np.ndarray:
    """True for entries of chrom_bins within rad bins of any value of centers (both 1-D, one chromosome)."""
    if not len(centers):
        return np.zeros(len(chrom_bins), bool)
    c = np.sort(centers)
    k = np.clip(np.searchsorted(c, chrom_bins), 1, len(c))
    lo = c[np.maximum(k - 1, 0)]
    hi = c[np.minimum(k, len(c) - 1)]
    return np.minimum(np.abs(chrom_bins - lo), np.abs(chrom_bins - hi)) <= rad


# =============================================================== Hi-C stage (one cell)
def _fetch(clr, chrom: str, s: int, e: int, size_bins: int) -> np.ndarray:
    from hicmultiome.genome import match_name
    name = match_name(chrom, list(clr.chromnames))
    size = int(clr.chromsizes[name])
    M = np.asarray(clr.matrix(balance=False).fetch(f"{name}:{s * RES}-{min(e * RES, size)}"), np.float64)
    M = np.where(M == 0, np.nan, M)
    if M.shape[0] < size_bins:
        P = np.full((size_bins, size_bins), np.nan)
        P[:M.shape[0], :M.shape[1]] = M
        M = P
    return M


def call_cell(target: str, reference: str, chroms: list[str] = AUTOSOMES) -> dict:
    """Loop peaks and stripe rectangle scores of one cell against its reference, genome-wide
    (autosomes). Returns {"loops": DataFrame(chrom, i, j, e_cell, e_ref),
    "stripes": {chrom: (s_cell_right, s_cell_left, s_ref_right, s_ref_left)}}."""
    import cooler
    from hicmultiome.genome import match_name
    ct, cr = cooler.Cooler(str(target)), cooler.Cooler(str(reference))
    loops, stripes = [], {}
    for chrom in chroms:
        name = match_name(chrom, list(ct.chromnames))
        if name is None:
            continue
        n = int(ct.chromsizes[name]) // RES + 1
        s = 0
        first = True
        while s < n - 2 * MARGIN:
            Mc, Mr = _fetch(ct, chrom, s, s + WIN, WIN), _fetch(cr, chrom, s, s + WIN, WIN)
            P = loop_peaks(donut_log2(Mc), donut_log2(Mr), s, n, first)
            P.insert(0, "chrom", chrom)
            loops.append(P)
            s += STRIDE
            first = False
        arr = np.full((4, n), np.nan, np.float32)
        s = 0
        while s < n:
            Mc, Mr = _fetch(ct, chrom, s, s + SWIN, SWIN), _fetch(cr, chrom, s, s + SWIN, SWIN)
            rc, lc = stripe_scores(Mc)
            rr, lr = stripe_scores(Mr)
            lo = 0 if s == 0 else SPAD
            hi = SPAD + SOWN if s + SWIN < n else SWIN
            a = np.arange(lo, hi)
            a = a[s + a < n]
            for k, v in enumerate((rc, lc, rr, lr)):
                arr[k, s + a] = v[a]
            if s + SWIN >= n:
                break
            s += SOWN
        stripes[chrom] = arr
    return {"loops": pd.concat(loops, ignore_index=True), "stripes": stripes}


def calls_path(cell: str, work: Path = WORK) -> Path:
    return work / f"calls_{cell}.pkl"


# =============================================================== tables
def marks_of(cell: str, chrom: str, nb: int, bw_dir: Path) -> dict:
    """10 kb means of the cell's CTCF and H3K27ac (NaN arrays when a mark is absent)."""
    from hicmultiome.tracks import read_track_binned
    out = {}
    for m in ("CTCF", "H3K27ac"):
        f = Path(bw_dir) / f"{m}.bigwig"
        out[m] = (read_track_binned(str(f), chrom, nb, RES).astype(float) if f.exists()
                  else np.full(nb, np.nan))
    return out


def recurrence_loops(spec: pd.DataFrame, others: dict) -> np.ndarray:
    """Per specific loop: the number of other cells whose peaks (e_cell >= E_MIN) lie within one bin in both anchors."""
    sets = {c: set(zip(P.chrom, P.i, P.j)) for c, P in others.items()}
    out = np.zeros(len(spec), int)
    for k, r in enumerate(spec.itertuples()):
        for c, S in sets.items():
            if any((r.chrom, r.i + di, r.j + dj) in S for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                out[k] += 1
    return out


def build_cell(cell: str, calls: dict, other_calls: dict, bw_dir: Path, fold: int = 0) -> pd.DataFrame:
    """Rows of the objects table of one cell (specific objects and their matched shared controls)."""
    rng = np.random.default_rng(SEED + sum(map(ord, cell)))
    rows = []
    # ---- loops
    P = classify_loops(calls["loops"])
    P = P[P.cls != "none"].reset_index(drop=True)
    other_peaks = {c: o["loops"][o["loops"].e_cell >= E_MIN] for c, o in other_calls.items()}
    spec_all = P[P.cls == "specific"].nlargest(TOP_LOOPS, "e_cell")
    for chrom, g in P.groupby("chrom"):
        spec = spec_all[spec_all.chrom == chrom]
        if not len(spec):
            continue
        sh = g[g.cls == "shared"]
        nb = int(max(g.j.max(), g.i.max()) + 2) + 1
        mk = marks_of(cell, chrom, nb + 100, bw_dir)
        spec = spec.assign(n_other=recurrence_loops(spec, {c: p[p.chrom == chrom] for c, p in other_peaks.items()}))

        def feat(d):
            return np.column_stack([(mk[m][d.i.to_numpy()] + mk[m][d.j.to_numpy()]) / 2 for m in ("CTCF", "H3K27ac")])
        near = near_any_pairs(sh, spec)
        pick = match_controls(feat(spec), feat(sh) if len(sh) else np.zeros((0, 2)),
                              np.log10((spec.j - spec.i).to_numpy()),
                              np.log10((sh.j - sh.i).to_numpy()) if len(sh) else None,
                              near if len(sh) else np.zeros(0, bool), rng)
        for k, r in enumerate(spec.itertuples()):
            pid = f"{cell}:L{chrom}:{r.i}:{r.j}"
            rows.append(_row(cell, "loop", "pos", "pos", pid, chrom, r.i, r.i + 1, r.j, r.j + 1, r.n_other, fold))
            if pick[k] >= 0:
                c = sh.iloc[pick[k]]
                rows.append(_row(cell, "loop", "ctrl", "A_ref", pid, chrom, c.i, c.i + 1, c.j, c.j + 1,
                                 np.nan, fold))
    # ---- stripes
    cls_of, cands = {}, []
    for chrom, arr in calls["stripes"].items():
        for d, dn in ((0, "right"), (1, "left")):
            cls = classify_stripes(arr[d], arr[2 + d])
            cls_of[(chrom, dn)] = (cls, arr[d])
            for a in np.nonzero(cls == "specific")[0]:
                cands.append((float(arr[d][a]), chrom, dn, int(a)))
    keep = {(c, d): [] for c, d in cls_of}
    for sc, chrom, dn, a in sorted(cands, reverse=True)[:TOP_STRIPES]:
        keep[(chrom, dn)].append(a)
    for chrom, arr in calls["stripes"].items():
        n = arr.shape[1]
        spec_a = np.concatenate([np.asarray(keep[(chrom, d)], int) for d in ("right", "left")])
        if not len(spec_a):
            continue
        mk = marks_of(cell, chrom, n + EB + RA + 10, bw_dir)
        a_ix = np.arange(n)
        anchor_f = np.column_stack([np.array([np.nanmean(mk[m][a:a + RA]) if a + RA <= len(mk[m]) else np.nan
                                              for a in a_ix]) for m in ("CTCF", "H3K27ac")])
        for dn in ("right", "left"):
            cls, _ = cls_of[(chrom, dn)]
            sa = np.asarray(keep[(chrom, dn)], int)
            ha = a_ix[cls == "shared"]
            if not len(sa):
                continue
            other_hits = _stripe_recurrence(chrom, dn, sa, other_calls)
            near = near_any(ha, spec_a)
            pick = match_controls(anchor_f[sa], anchor_f[ha] if len(ha) else np.zeros((0, 2)), None, None,
                                  near if len(ha) else np.zeros(0, bool), rng)
            for k, a in enumerate(sa):
                pid = f"{cell}:S{chrom}:{dn}:{a}"
                rows.append(_stripe_row(cell, "pos", "pos", pid, chrom, int(a), dn, other_hits[k], fold))
                if pick[k] >= 0:
                    rows.append(_stripe_row(cell, "ctrl", "A_ref", pid, chrom, int(ha[pick[k]]), dn, np.nan, fold))
    return pd.DataFrame(rows, columns=COLS)


def near_any_pairs(pool: pd.DataFrame, spec: pd.DataFrame, rad: int = NEAR) -> np.ndarray:
    """Pool loops within rad bins (both anchors) of a specific loop."""
    if not len(pool) or not len(spec):
        return np.zeros(len(pool), bool)
    S = set(zip(spec.i, spec.j))
    return np.array([any((i + di, j + dj) in S for di in range(-rad, rad + 1) for dj in range(-rad, rad + 1))
                     for i, j in zip(pool.i, pool.j)])


def _stripe_recurrence(chrom: str, dn: str, anchors: np.ndarray, others: dict) -> np.ndarray:
    d = 0 if dn == "right" else 1
    out = np.zeros(len(anchors), int)
    for c, o in others.items():
        s = o["stripes"].get(chrom)
        if s is None:
            continue
        sc = s[d]
        for k, a in enumerate(anchors):
            seg = sc[max(0, a - 1):a + 2]
            if np.isfinite(seg).any() and np.nanmax(seg) >= S_MIN:
                out[k] += 1
    return out


def _row(cell, kind, role, st, pid, chrom, a0, a1, b0, b1, n_other, fold) -> dict:
    return {"fold": fold, "patient": cell, "cell": cell, "kind": kind, "role": role, "set": st,
            "pair_id": pid, "chrom": chrom, "a_start": int(a0) * RES, "a_end": int(a1) * RES,
            "b_start": int(b0) * RES, "b_end": int(b1) * RES, "pair_type": "other", "n_train_other": n_other}


def _stripe_row(cell, role, st, pid, chrom, a, dn, n_other, fold) -> dict:
    b0, b1 = (a + RA, a + RA + EB) if dn == "right" else (a - EB, a)
    return _row(cell, "stripe", role, st, pid, chrom, a, a + RA, b0, b1, n_other, fold)


def assert_no_leak(d: pd.DataFrame, heldout: bool = False) -> None:
    """The training table holds training cells only; the evaluation table held-out cells only."""
    import multicell_expanded as mx
    got = set(d.cell)
    if heldout:
        assert not got & set(mx.TRAIN12), f"held-out table with training cells {sorted(got & set(mx.TRAIN12))}"
    else:
        assert got <= set(mx.TRAIN12), f"training table with non-training cells {sorted(got - set(mx.TRAIN12))}"
        from hicmultiome.reference import cell_identity
        held = {cell_identity(c) for c in mx.TEST}
        assert not {cell_identity(c) for c in got} & held, "training table with a held-out cell identity"
    assert set(d.role) <= {"pos", "ctrl"}
    assert (d.groupby("pair_id").role.apply(lambda r: (r == "pos").sum()) <= 1).all()


# =============================================================== source: crosscell_rule_transfer
RULE_TRANSFER = HM / "crosscell_rules/cell_objects.csv.gz"
HELD_NAME = {"hmec": "hmec_deep", "huvec": "huvec_deep"}       # crosscell_rule_transfer names the deep maps hmec / huvec


EVAL_LIMIT = 1000                           # held-out cells: specific objects per cell and kind (seeded random draw)


def cap_objects(d: pd.DataFrame, limit_loops: int, limit_stripes: int, how: str) -> pd.DataFrame:
    """Per cell and kind at most `limit` specific objects (those with a matched control first, then the strongest by `gain`; or a seeded random draw);
    controls keep only the pairs of the kept objects. d: rule-transfer rows with sets pos / A_ref."""
    keep = []
    matched = set(zip(d[d["set"] != "pos"].cell, d[d["set"] != "pos"].kind, d[d["set"] != "pos"].pair_id))
    for (cell, kind), g in d[d["set"] == "pos"].groupby(["cell", "kind"]):
        lim = limit_loops if kind == "loop" else limit_stripes
        if lim is None or len(g) <= lim:
            keep.append(g)
        elif how == "gain":
            # objects with a matched control first (the ranking pairs), the strongest by gain within each group
            has = pd.Series([(cell, kind, p) in matched for p in g.pair_id], index=g.index)
            col = "gain" if "gain" in g else "e_cell"
            keep.append(pd.concat([g[has].nlargest(lim, col), g[~has].nlargest(lim, col)]).head(lim))
        else:
            keep.append(g.sample(lim, random_state=SEED + sum(map(ord, cell))))
    pos = pd.concat(keep) if keep else d.iloc[:0]
    ids = set(zip(pos.cell, pos.kind, pos.pair_id))
    ctrl = d[(d["set"] != "pos") & pd.Series([(c, k, p) in ids for c, k, p in zip(d.cell, d.kind, d.pair_id)],
                                             index=d.index)]
    return pd.concat([pos, ctrl])


def from_rule_transfer(d: pd.DataFrame, train_cells: list[str], heldout: bool = False,
                       limit_loops: int | None = TOP_LOOPS, limit_stripes: int | None = TOP_STRIPES,
                       how: str = "gain") -> pd.DataFrame:
    """Rows of scripts/crosscell_rule_transfer.py's cell objects (columns cell, kind, set, pair_id, chrom,
    a_lo, a_hi, b_lo, b_hi, ...) in this table's schema: sets pos and A_ref only; the training table keeps the
    training cells, the evaluation table the held-out ones (renamed to the cell names of
    scripts/multicell_expanded.py). n_train_other = number of other TRAINING cells with a specific object of the
    same kind within one bin of the anchor (loops: both anchors; stripes: the same direction), counted before the
    cap. The cap (TOP_LOOPS / TOP_STRIPES per cell: those with a matched control first, the strongest by gain; the evaluation table EVAL_LIMIT per
    cell and kind, seeded random) keeps the number of objects at the scale of the AML patients' private objects
    and the oversampled window count of the trainer at the section 63 level."""
    # the specific objects of a cell: `pos_all` (every one) when the source keeps it beside its capped `pos`
    src = d[d["set"] == ("pos_all" if (d["set"] == "pos_all").any() else "pos")].copy()
    src["cell"] = src.cell.replace(HELD_NAME)
    d = d[d["set"].isin(["pos", "A_ref"])].copy()
    d["cell"] = d.cell.replace(HELD_NAME)
    train = set(train_cells)
    pos = src[src.cell.isin(train)]
    key = {}
    for r in pos.itertuples():
        right = r.kind == "stripe" and r.b_lo > r.a_lo
        key.setdefault(r.cell, set()).add((r.kind, r.chrom, r.a_lo, r.b_lo if r.kind == "loop" else int(right)))

    def n_other(r):
        right = r.kind == "stripe" and r.b_lo > r.a_lo
        k2 = r.b_lo if r.kind == "loop" else int(right)
        n = 0
        for c, S in key.items():
            if c == r.cell:
                continue
            if any((r.kind, r.chrom, r.a_lo + da, k2 + (db if r.kind == "loop" else 0)) in S
                   for da in (-1, 0, 1) for db in ((-1, 0, 1) if r.kind == "loop" else (0,))):
                n += 1
        return n
    d = d[d.cell.isin(set(HELD_NAME.values()) | {"k562"} if heldout else train)].copy()
    d = cap_objects(d, limit_loops, limit_stripes, how)
    d["role"] = np.where(d["set"] == "pos", "pos", "ctrl")
    d["n_train_other"] = np.nan
    pm = d["set"] == "pos"
    d.loc[pm, "n_train_other"] = [n_other(r) for r in d[pm].itertuples()]
    d["fold"], d["patient"], d["pair_type"] = 0, d.cell, "other"
    d["a_start"], d["a_end"] = d.a_lo * RES, d.a_hi * RES
    d["b_start"], d["b_end"] = d.b_lo * RES, d.b_hi * RES
    return d.sort_values(["cell", "kind", "set", "pair_id"]).reset_index(drop=True)[COLS]


# =============================================================== driver
def main():
    import multicell_expanded as mx
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["calls", "build"])
    ap.add_argument("--cells", nargs="*", default=None, help="default: the 12 training cells")
    ap.add_argument("--heldout", action="store_true",
                    help=f"the evaluation table of the held-out cells {HELDOUT} (reference consensus12)")
    ap.add_argument("--no-share", action="store_true")
    ap.add_argument("--source", choices=["auto", "local", "rule_transfer"], default="auto",
                    help="where the objects come from (see the module doc); auto: rule_transfer when "
                         f"{RULE_TRANSFER} exists")
    a = ap.parse_args()
    source = a.source if a.source != "auto" else ("rule_transfer" if RULE_TRANSFER.exists() else "local")
    log(f"object source: {source}")
    train = list(mx.TRAIN12)
    cells = a.cells or (HELDOUT if a.heldout else train)
    WORK.mkdir(parents=True, exist_ok=True)

    def ref_of(c):
        return mx.consensus_loo(c, RES) if c in train else mx.consensus12(RES)
    if source == "rule_transfer":
        d = from_rule_transfer(pd.read_csv(RULE_TRANSFER), train, a.heldout,
                               *( (EVAL_LIMIT, EVAL_LIMIT, "random") if a.heldout else (TOP_LOOPS, TOP_STRIPES, "gain")))
        if a.cells:
            d = d[d.cell.isin(a.cells)]
        assert_no_leak(d, a.heldout)
        OUT.mkdir(parents=True, exist_ok=True)
        name = "cell_objects_heldout.csv.gz" if a.heldout else "cell_objects.csv.gz"
        d.to_csv(OUT / name, index=False)
        (OUT / (name + ".source")).write_text(f"rule_transfer {RULE_TRANSFER} {time.strftime('%Y-%m-%d %H:%M')}\n")
        if not a.no_share:
            SHARE.mkdir(parents=True, exist_ok=True)
            d.to_csv(SHARE / name, index=False)
        s_ = d.groupby(["cell", "kind", "set"]).size().rename("n").reset_index()
        s_.to_csv(OUT / ("summary_heldout.csv" if a.heldout else "summary.csv"), index=False)
        print(s_.pivot_table(index="cell", columns=["kind", "set"], values="n", fill_value=0).to_string())
        return
    if "calls" in a.stage:
        for c in dict.fromkeys(list(cells) + (train if a.heldout else [])):
            if calls_path(c).exists():
                continue
            log(f"calls {c}")
            pickle.dump(call_cell(mx.target(c, RES), ref_of(c)), open(calls_path(c), "wb"))
    if "build" in a.stage:
        calls = {c: pickle.load(open(calls_path(c), "rb")) for c in dict.fromkeys(list(cells) + train)}
        parts = []
        for c in cells:
            others = {o: calls[o] for o in train if o != c}
            d = build_cell(c, calls[c], others, mx.CELLS[c][1])
            log(f"{c}: {len(d)} rows ({int((d.role == 'pos').sum())} specific)")
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        assert_no_leak(d, a.heldout)
        OUT.mkdir(parents=True, exist_ok=True)
        name = "cell_objects_heldout.csv.gz" if a.heldout else "cell_objects.csv.gz"
        d.to_csv(OUT / name, index=False)
        (OUT / (name + ".source")).write_text(f"local scripts/cell_objects_table.py {time.strftime('%Y-%m-%d %H:%M')}\n")
        if not a.no_share:
            SHARE.mkdir(parents=True, exist_ok=True)
            d.to_csv(SHARE / name, index=False)
        s = (d.groupby(["cell", "kind", "set"]).size().rename("n").reset_index())
        s.to_csv(OUT / ("summary_heldout.csv" if a.heldout else "summary.csv"), index=False)
        print(s.pivot_table(index="cell", columns=["kind", "set"], values="n", fill_value=0).to_string())


if __name__ == "__main__":
    main()
