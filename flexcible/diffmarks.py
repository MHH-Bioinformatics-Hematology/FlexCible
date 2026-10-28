"""Differential mark channels: a cell's chromatin relative to the reference whose Hi-C is also given.

Rule behind the design (registry section 66): "this chromatin-mark / methylation pattern leads to a
loop or a stripe" is cell-type independent biology. A model that gets, besides the sample's own
marks, the marks of the reference whose Hi-C it also reads, and the difference of the two, can learn
ONE function f with  residual Hi-C ~ f(delta marks)  for every cell type: cell lines (reference =
leave-one-out consensus of the other training cells, consensus of all 12 for a held-out cell), AML
patients (reference = mean of the training patients of the fold, leave-one-out) and the HSPC donors.

Rows (appended after the other sequence-feature rows, before the motif rows; `data.diff_marks`):

  dref:<mark>          the reference's mark: mean over the reference set of the per-sample normalised
                       track, at the native bin
  diff:<mark>:s<r>     sample minus reference, smoothed with a box of +-r bp (r = 0: the native
                       bin; default scales 0, 20 kb, 100 kb), then standardised per sample, mark
                       and scale: (d - median) / (1.4826 MAD) over the autosomal bins, clipped to +-8

for every mark of the model (all ChIP tracks and Meth; copy-number and annotation tracks have no
reference). A mark the sample lacks (missing_tracks / zeroed_tracks), or that no member of its
reference set has, gives 0 in all of its rows ("no information"); track dropout does the same in the
trainer (flexcible.train, diff_coupling).

Per-sample normalisation of a mark (quantile anchoring): v = log1p(max(raw, 0)); z = (v - q50) / (q99 - q50)
with the quantiles over the autosomal bins of the sample, so that a deep and a shallow track of the same
mark sit on one scale (q50 -> 0, q99 -> 1). The model's own TrackNormalizer (log1p, global z-score) is a
different, shared normalisation and is not used here; the diff rows bypass it (TrackNormalizer.passthrough).

Reference sets (`data.diff_reference_sets`, name -> list of sample names of data.samples; the pool of a
sample is `samples[].diff_set`, default `data.diff_default_set`). The reference of a sample is its pool
without the sample itself and without every sample of the same cell identity
(flexcible.reference.cell_identity); the sample's own track never enters its reference
(`reference_members`, tested). No label (Hi-C) is read, so held-out chromosomes cannot leak through the
per-sample statistics, which use marks only.

Everything is off unless data.diff_marks is set.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .reference import cell_identity

CLIP = 8.0
MAD_TO_SD = 1.4826
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
NO_REF_TYPES = ("annotation", "copynumber")
DREF = "dref"
DIFF = "diff"
REF_KINDS = ("consensus", "hspc", "mean_aml")


# --------------------------------------------------------------------------- names
def marks_of(cfg) -> list[str]:
    """Tracks that get reference and difference rows (data.diff_tracks, else every track that is
    not an annotation or copy-number track), in track order."""
    types = {t.name: t.type for t in cfg.data.tracks}
    want = cfg.data.diff_tracks
    return [t for t in cfg.track_names
            if types[t] not in NO_REF_TYPES and (want is None or t in want)]


def scale_tag(r: int) -> str:
    return f"s{int(r)}"


def diff_row_names(cfg) -> list[str]:
    """Names of the rows, in row order: all dref rows (when data.diff_ref_channels), then per mark
    its scales. Empty when data.diff_marks is off."""
    if not cfg.data.diff_marks:
        return []
    m = marks_of(cfg)
    names = [f"{DREF}:{t}" for t in m] if cfg.data.diff_ref_channels else []
    names += [f"{DIFF}:{t}:{scale_tag(r)}" for t in m for r in cfg.data.diff_scales]
    return names


def row_mark(name: str) -> str | None:
    """Mark a diff row belongs to, None for any other row name."""
    p = name.split(":")
    return p[1] if p[0] in (DREF, DIFF) and len(p) >= 2 else None


def is_diff_row(name: str) -> bool:
    return row_mark(name) is not None


def radius_bins(r_bp: int, res: int) -> int:
    """Half-width in bins of the smoothing box of radius r_bp (0 = native bin)."""
    return int(round(int(r_bp) / int(res))) if r_bp > 0 else 0


def smooth_box(x: np.ndarray, rad: int) -> np.ndarray:
    """Box mean over +-rad bins along the last axis ('nearest' at the ends)."""
    if rad <= 0:
        return x
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, size=2 * rad + 1, axis=-1, mode="nearest")


# --------------------------------------------------------------------------- reference sets
def set_key(kind: str, group: str) -> str:
    """Reference set a sample of `group` ('cell' line, 'aml' patient or AML line, 'hspc' donor) reads
    under data.diff_reference `kind`: cell lines always the 12-cell consensus, HSPC donors always the
    HSPC training donors, AML samples the set of the kind (consensus, hspc, or mean_aml = the training
    patients of the fold)."""
    if kind not in REF_KINDS:
        raise ValueError(f"unknown data.diff_reference {kind!r} ({REF_KINDS})")
    if group == "cell":
        return "consensus"
    if group == "hspc":
        return "hspc"
    if group == "aml":
        return {"consensus": "consensus", "hspc": "hspc", "mean_aml": "mean_aml"}[kind]
    raise ValueError(f"unknown sample group {group!r} (cell|aml|hspc)")


def build_sets(cells: list[str], aml: list[str], hspc: list[str]) -> dict[str, list[str]]:
    """data.diff_reference_sets from the training samples of each group (AML: the patients, without
    the AML lines such as MOLM-13, which have no reference of their own beyond the mean)."""
    out = {"consensus": list(cells)}
    if aml:
        out["mean_aml"] = list(aml)
    if hspc:
        out["hspc"] = list(hspc)
    return out


def reference_members(pool: list[str], sample: str) -> list[str]:
    """The pool without `sample` and without any sample of its cell identity: a sample never sees
    its own track (or the other condition of its cell) in its reference."""
    ident = cell_identity(sample)
    return [n for n in pool if n != sample and cell_identity(n) != ident]


def sample_pool(cfg, sample) -> list[str]:
    """Pool (set of names) the sample's reference is drawn from, before the leave-one-out."""
    key = sample.diff_set or cfg.data.diff_default_set
    sets = cfg.data.diff_reference_sets
    if key not in sets:
        raise KeyError(f"sample {sample.name!r}: diff set {key!r} not in data.diff_reference_sets "
                       f"{sorted(sets)}")
    return list(sets[key])


# --------------------------------------------------------------------------- per-sample marks
def chrom_key(chrom: str) -> str:
    return chrom if str(chrom).startswith("chr") else f"chr{chrom}"


class SampleMarks:
    """Per-sample normalised marks z (M, nb) for every stat chromosome, plus which marks are valid."""

    def __init__(self, z: dict, valid: np.ndarray):
        self.z = z
        self.valid = valid


def quantile_anchor(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """z = (log1p(max(raw, 0)) - q_lo) / (q_hi - q_lo) with q the 50th / 99th percentile of the
    log1p values handed in as (lo, hi)."""
    v = np.log1p(np.clip(raw, 0, None))
    return ((v - lo) / (hi - lo)).astype(np.float32)


class DiffMarks:
    """Builds the diff rows of a configuration. One instance per Config (`get`); memoises the
    normalised marks of every sample it touches (and keeps them on disk beside the window cache)."""

    _instances: dict = {}

    @classmethod
    def get(cls, cfg) -> "DiffMarks":
        key = id(cfg)
        inst = cls._instances.get(key)
        if inst is None or inst.cfg is not cfg:
            inst = cls._instances[key] = cls(cfg)
        return inst

    def __init__(self, cfg):
        self.cfg = cfg
        self.res = int(cfg.data.resolution)
        self.marks = marks_of(cfg)
        self.names = diff_row_names(cfg)
        self.mi = {t: cfg.track_names.index(t) for t in self.marks}
        want = cfg.data.chroms
        self.chroms = [c for c in AUTOSOMES if want is None or c in {chrom_key(x) for x in want}]
        self._marks: dict = {}
        self._stats: dict = {}
        self._byname = {s.name: s for s in cfg.data.samples}
        self.cache_dir = Path(cfg.train.cache_dir) / "diffmarks"

    # ---- raw tracks of a sample
    def _files(self, sample):
        from .precompute import sample_track_files
        return sample_track_files(self.cfg, sample)

    def _dead(self, sample) -> set:
        return set(sample.missing_tracks) | set(sample.zeroed_tracks)

    def _signature(self, sample, files) -> str:
        h = hashlib.sha1()
        h.update(f"{sample.diff_identity or sample.name}|{self.res}|{','.join(self.chroms)}|{','.join(self.marks)}".encode())
        dead = self._dead(sample)
        for name, p in files:
            if name in self.marks and name not in dead and p is not None:
                st = os.stat(p)
                h.update(f"{name}|{p}|{st.st_size}|{st.st_mtime_ns}".encode())
        return h.hexdigest()[:12]

    def _chrom_bins(self, files) -> dict:
        import pyBigWig
        for name, p in files:
            if p is None:
                continue
            bw = pyBigWig.open(p)
            try:
                ln = {chrom_key(k): int(v) for k, v in bw.chroms().items()}
            finally:
                bw.close()
            return {c: -(-ln[c] // self.res) for c in self.chroms if c in ln}
        return {}

    def sample_marks(self, sample) -> SampleMarks:
        key = (sample.diff_identity or sample.name, sample.bigwig_dir, tuple(sample.extra_bigwig_dirs),
               tuple(sorted(self._dead(sample))))
        if key in self._marks:
            return self._marks[key]
        files = self._files(sample)
        sig = self._signature(sample, files)
        f = self.cache_dir / f"{sample.diff_identity or sample.name}_{sig}.npz"
        sm = None
        if f.exists():
            try:
                with np.load(f) as zf:
                    sm = SampleMarks({c: zf[c] for c in zf.files if c != "_valid"},
                                     zf["_valid"].astype(bool))
            except Exception:
                sm = None
        if sm is None:
            sm = self._compute(sample, files)
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = f.with_name(f.name + f".{os.getpid()}.tmp.npz")
                np.savez(tmp, _valid=sm.valid, **sm.z)
                os.replace(tmp, f)
            except OSError:
                pass                                   # a read-only cache folder: recompute next time
        self._marks[key] = sm
        return sm

    def _compute(self, sample, files) -> SampleMarks:
        from .tracks import read_track_binned
        dead = self._dead(sample)
        nbs = self._chrom_bins(files)
        fmap = dict(files)
        M = len(self.marks)
        raw = {c: np.zeros((M, nb), np.float32) for c, nb in nbs.items()}
        have = np.zeros(M, bool)
        for k, t in enumerate(self.marks):
            p = fmap.get(t)
            if t in dead or p is None:
                continue
            have[k] = True
            for c, nb in nbs.items():
                raw[c][k] = read_track_binned(p, c, nb, self.res)
        z = {c: np.zeros_like(a) for c, a in raw.items()}
        valid = np.zeros(M, bool)
        for k in range(M):
            if not have[k]:
                continue
            v = np.concatenate([np.log1p(np.clip(raw[c][k], 0, None)) for c in raw])
            lo, hi = np.percentile(v, 50), np.percentile(v, 99)
            if hi - lo < 1e-6:
                continue                               # a flat track carries no signal
            valid[k] = True
            for c in raw:
                z[c][k] = quantile_anchor(raw[c][k], lo, hi)
        return SampleMarks(z, valid)

    # ---- reference of a sample
    def members(self, sample) -> list[str]:
        mem = reference_members(sample_pool(self.cfg, sample), sample.diff_identity or sample.name)
        miss = [n for n in mem if n not in self._byname]
        if miss:
            raise KeyError(f"sample {sample.name!r}: reference members {miss} are not in data.samples")
        return mem

    def reference(self, sample, chrom: str):
        """(mean z of the members over the stat chromosome, count of members with the mark (M,))."""
        c = chrom_key(chrom)
        mem = self.members(sample)
        acc, cnt = None, np.zeros(len(self.marks))
        for n in mem:
            sm = self.sample_marks(self._byname[n])
            if c not in sm.z:
                continue
            z = sm.z[c] * sm.valid[:, None]
            acc = z if acc is None else acc + z
            cnt += sm.valid
        if acc is None:
            return None, cnt
        return acc / np.maximum(cnt, 1)[:, None], cnt

    # ---- standardisation statistics
    def stats(self, sample) -> dict:
        """{(mark index, scale): (median, sd)} of the smoothed difference over the autosomal bins."""
        key = (sample.name, sample.diff_identity, sample.diff_set or self.cfg.data.diff_default_set,
               sample.bigwig_dir, tuple(sample.extra_bigwig_dirs), tuple(sorted(self._dead(sample))))
        if key in self._stats:
            return self._stats[key]
        sm = self.sample_marks(sample)
        out = {}
        diffs = {}
        for c in sm.z:
            ref, cnt = self.reference(sample, c)
            if ref is None:
                continue
            ok = sm.valid & (cnt > 0)
            diffs[c] = (sm.z[c] - ref, ok)
        for k in range(len(self.marks)):
            for r in self.cfg.data.diff_scales:
                rad = radius_bins(r, self.res)
                vals = [smooth_box(d[k], rad) for c, (d, ok) in diffs.items() if ok[k]]
                if not vals:
                    out[(k, r)] = None
                    continue
                v = np.concatenate(vals)
                med = float(np.median(v))
                sd = MAD_TO_SD * float(np.median(np.abs(v - med)))
                out[(k, r)] = (med, max(sd, 1e-3))
        self._stats[key] = out
        return out

    # ---- the rows
    def rows(self, sample, chrom: str, nb: int) -> np.ndarray:
        """(len(names), nb) float32 diff rows of the chromosome, in the order of diff_row_names."""
        cfg = self.cfg
        c = chrom_key(chrom)
        sm = self.sample_marks(sample)
        out = np.zeros((len(self.names), nb), np.float32)
        if c not in sm.z:
            return out
        ref, cnt = self.reference(sample, c)
        if ref is None:
            return out
        st = self.stats(sample)
        n = min(nb, sm.z[c].shape[1])
        pos = {nm: i for i, nm in enumerate(self.names)}
        for k, t in enumerate(self.marks):
            if not (sm.valid[k] and cnt[k] > 0):
                continue                                # missing mark or reference: 0, "no information"
            if f"{DREF}:{t}" in pos:
                out[pos[f"{DREF}:{t}"], :n] = ref[k, :n]
            d = sm.z[c][k] - ref[k]
            for r in cfg.data.diff_scales:
                s = st[(k, r)]
                if s is None:
                    continue
                v = np.clip((smooth_box(d, radius_bins(r, self.res)) - s[0]) / s[1], -CLIP, CLIP)
                out[pos[f"{DIFF}:{t}:{scale_tag(r)}"], :n] = v[:n]
        return out


def configure_unseen(cfg, sample, cell: str, group: str | None) -> None:
    """Point `sample` (a copy of a configured sample used for an unseen cell) at the reference set of
    the cell's group ('cell' line, 'aml', 'hspc'; None = 'cell') and leave `cell` out of it."""
    sample.diff_identity = cell
    sample.diff_set = set_key(cfg.data.diff_reference, group or "cell")


def blank_diff_rows(sym: np.ndarray, cfg, tracks: list[str], seq_features: list[str], track: str) -> None:
    """In place: the reference/difference rows of `track` in sym (full cache rows) set to 0, the
    state of a mark that is absent (inference-time counterpart of track dropout)."""
    if not cfg.data.diff_marks or track not in tracks:
        return
    for k in coupling(cfg, tracks, seq_features).get(tracks.index(track), []):
        sym[len(tracks) + k] = 0.0


def diff_rows(cfg, sample, chrom: str, nb: int) -> np.ndarray:
    """Rows of `sample` on `chrom` (see module doc); shape (len(diff_row_names(cfg)), nb)."""
    return DiffMarks.get(cfg).rows(sample, chrom, nb)


def coupling(cfg, tracks: list[str], seq_features: list[str]) -> dict[int, list[int]]:
    """{index of the track (in `tracks`) -> indices, within the seq-feature block, of its diff
    rows}. Track dropout of a mark blanks these rows too."""
    out: dict[int, list[int]] = {}
    for k, nm in enumerate(seq_features):
        m = row_mark(nm)
        if m is not None and m in tracks:
            out.setdefault(tracks.index(m), []).append(k)
    return out


def validate(cfg) -> None:
    """Configuration checks (called from Config.validate)."""
    d = cfg.data
    if not d.diff_marks:
        return
    if d.diff_reference not in REF_KINDS:
        raise ValueError(f"data.diff_reference {d.diff_reference!r} is not one of {REF_KINDS}")
    if not d.diff_scales or any(int(r) < 0 for r in d.diff_scales):
        raise ValueError("data.diff_scales: radii in bp, >= 0 (0 = the native bin)")
    if len(set(d.diff_scales)) != len(d.diff_scales):
        raise ValueError("data.diff_scales: duplicate radii")
    if not d.diff_reference_sets:
        raise ValueError("data.diff_marks needs data.diff_reference_sets (name -> sample names)")
    snames = {s.name for s in d.samples}
    for k, v in d.diff_reference_sets.items():
        bad = [n for n in v if n not in snames]
        if bad:
            raise ValueError(f"data.diff_reference_sets[{k!r}]: {bad} are not samples")
    if d.diff_default_set not in d.diff_reference_sets:
        raise ValueError(f"data.diff_default_set {d.diff_default_set!r} is not a reference set")
    for s in d.samples:
        key = s.diff_set or d.diff_default_set
        if key not in d.diff_reference_sets:
            raise ValueError(f"sample {s.name!r}: diff_set {key!r} is not a reference set")
        if not reference_members(d.diff_reference_sets[key], s.name):
            raise ValueError(f"sample {s.name!r}: empty reference set {key!r} after leaving it out")
    if not marks_of(cfg):
        raise ValueError("data.diff_marks: no tracks to difference")
    if d.diff_tracks is not None:
        bad = [t for t in d.diff_tracks if t not in cfg.track_names]
        if bad:
            raise ValueError(f"data.diff_tracks: unknown tracks {bad}")
