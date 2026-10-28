"""Loop-centric training for the AML-inclusive model (registry section 63).

The pixel loss is dominated by background: AML-specific loops and stripes are rare,
small-amplitude features. This module gives the trainer three levers on them, all driven by an
objects table (scripts/aml_objects_table.py) of annotated objects of the TRAINING patients of one
fold:

  * rows_of_objects / select_pairs / object_batches: which training windows hold objects, which
    of them are oversampled (AML-specific windows and an equal number of matched shared-control
    windows, one to one) and how batches are composed so that a batch holds both kinds;
  * batch_geometry: the neighbourhood mask and the object index arrays of a batch;
  * ObjectLoss: extra weight on the donut / stripe neighbourhoods of the objects in the residual
    (prediction minus reference against target minus reference), a match of the object-level
    enrichment, and a pairwise ranking term that penalises the same predicted enrichment at an
    AML-specific object and at a matched shared control.

Everything is off unless train.aml_object_* are set; with the defaults the trainer is unchanged.

Table columns: fold, patient, cell (sample name of the config, e.g. amlxu018), kind (loop|stripe),
role (pos|ctrl), set (pos|A_hspc|A_full|A_do|B_rand), pair_id, chrom, a_start, a_end, b_start,
b_end (bp, end exclusive; a loop is one bin on each axis, a stripe is anchor rows x extent
columns), n_train_other (training donors other than the patient's own that carry the object).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REQUIRED = ("cell", "kind", "role", "set", "pair_id", "chrom", "a_start", "a_end", "b_start", "b_end")
KINDS = ("loop", "stripe")
ROLE_POS, ROLE_CTRL, ROLE_WEAK = 1, 2, 3


# --------------------------------------------------------------------------- table
def load_objects(path: str, samples, controls=("A_hspc",), weak=()) -> pd.DataFrame:
    """Objects of the table whose set is 'pos', one of `controls` (role 2: sampling, ranking, all
    terms) or one of `weak` (role 3: the neighbourhood and enrichment terms only, at a reduced weight;
    never ranked, never oversampled). Raises when an object
    belongs to a cell that is not among `samples` (the config's training samples): a table of
    another fold, or one that carries test-patient objects, cannot be used by this run."""
    df = pd.read_csv(path, dtype={"patient": str})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"{path}: objects table lacks columns {miss}")
    foreign = sorted(set(df.cell) - set(samples))
    if foreign:
        raise ValueError(f"{path}: objects of cells {foreign} that are not training samples of "
                         f"this run (table of another fold, or test patients)")
    unknown = sorted(set(df.kind) - set(KINDS))
    if unknown:
        raise ValueError(f"{path}: unknown kinds {unknown}")
    df = df[df["set"].isin(["pos", *controls, *weak])].reset_index(drop=True)
    df["role_id"] = np.where(df["set"] == "pos", ROLE_POS,
                             np.where(df["set"].isin(list(controls)), ROLE_CTRL, ROLE_WEAK))
    return df


def to_bins(df: pd.DataFrame, res: int) -> pd.DataFrame:
    """Object extents in bins of `res` (start floored, end ceiled; a loop at a finer resolution
    than `res` keeps at least one bin)."""
    out = df.copy()
    for c in ("a", "b"):
        out[c + "0"] = (df[c + "_start"].to_numpy() // res).astype(np.int64)
        out[c + "1"] = np.maximum(-(-df[c + "_end"].to_numpy() // res),
                                  out[c + "0"].to_numpy() + 1).astype(np.int64)
    return out


# --------------------------------------------------------------------------- windows
def rows_of_objects(objs: pd.DataFrame, coords, window: int, margin: int):
    """Objects inside every window row of `coords` = (chrom list, start bins, sample list).

    A window holds an object when the object's own sample matches the row's sample, the object
    (both anchors, the whole stripe rectangle) lies at least `margin` bins inside the window.
    Returns a list (one entry per row) of tuples (kind_id, role_id, a0, a1, b0, b1, obj_index)
    in window-local bins."""
    ch, st, sm = coords
    n = len(ch)
    rows: list[list] = [[] for _ in range(n)]
    by_group: dict = {}
    for i in range(n):
        by_group.setdefault((sm[i], ch[i]), []).append(i)
    grp = {k: (np.asarray(v, dtype=np.int64), np.asarray([st[i] for i in v], dtype=np.int64))
           for k, v in by_group.items()}
    for g, sub in objs.groupby(["cell", "chrom"], sort=False):
        hit = grp.get((g[0], g[1]))
        if hit is None:
            continue
        ids, starts = hit
        order = np.argsort(starts, kind="stable")
        ids, starts = ids[order], starts[order]
        lo = np.minimum(sub.a0.to_numpy(), sub.b0.to_numpy())
        hi = np.maximum(sub.a1.to_numpy(), sub.b1.to_numpy())
        first = np.searchsorted(starts, hi + margin - window, side="left")
        last = np.searchsorted(starts, lo - margin, side="right")
        kind = sub.kind.map({k: j for j, k in enumerate(KINDS)}).to_numpy()
        role = sub.role_id.to_numpy()
        a0, a1, b0, b1 = (sub[c].to_numpy() for c in ("a0", "a1", "b0", "b1"))
        for k, oi in enumerate(sub.index.to_numpy()):
            for r in range(first[k], last[k]):
                s = int(starts[r])
                rows[int(ids[r])].append((int(kind[k]), int(role[k]), int(a0[k] - s), int(a1[k] - s),
                                          int(b0[k] - s), int(b1[k] - s), int(oi)))
    return rows


def row_roles(row_objs) -> np.ndarray:
    """Per row: 1 when it holds an AML-specific object, 2 when it holds only controls, else 0."""
    out = np.zeros(len(row_objs), dtype=np.int8)
    for i, o in enumerate(row_objs):
        if not o:
            continue
        if any(x[1] == ROLE_POS for x in o):
            out[i] = ROLE_POS
        elif any(x[1] == ROLE_CTRL for x in o):
            out[i] = ROLE_CTRL
    return out


def select_pairs(row_objs, objs: pd.DataFrame, rng, fallback: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one (AML-specific row, control row) pairs, equal in number.

    First choice: the control row holds a control object whose (cell, kind, pair_id) equals that
    of an AML-specific object of the row (the classifier's matched control: same patient, matched
    on distance, orientation and marks). The matched control is often outside the cached training
    windows (validation, region filters); with `fallback`, an AML-specific row without a matched
    control row then takes an unused control row of the same cell holding a control of the same
    kind (loop or stripe). Each row is used once, pos and control rows are disjoint."""
    key = (objs.cell + "|" + objs.kind + "|" + objs.pair_id).to_numpy()
    ctrl_rows: dict = {}
    by_cell: dict = {}
    for i, o in enumerate(row_objs):
        for x in o:
            if x[1] == ROLE_CTRL:
                ctrl_rows.setdefault(key[x[6]], []).append(i)
                by_cell.setdefault((objs.cell.iat[x[6]], x[0]), set()).add(i)
    pos_rows = [i for i, o in enumerate(row_objs) if any(x[1] == ROLE_POS for x in o)]
    used: set = set()
    pos_sel, ctl_sel = [], []
    todo = []
    for i in rng.permutation(pos_rows):
        i = int(i)
        if i in used:
            continue
        cand = []
        for x in row_objs[i]:
            if x[1] == ROLE_POS:
                cand += ctrl_rows.get(key[x[6]], [])
        cand = [c for c in cand if c not in used and c != i]
        if not cand:
            todo.append(i)
            continue
        c = int(cand[int(rng.integers(len(cand)))])
        used.update((c, i))
        pos_sel.append(i)
        ctl_sel.append(c)
    if fallback:
        for i in todo:
            if i in used:
                continue
            cand = set()
            for x in row_objs[i]:
                if x[1] == ROLE_POS:
                    cand |= by_cell.get((objs.cell.iat[x[6]], x[0]), set())
            cand = sorted(c for c in cand if c not in used and c != i)
            if not cand:
                continue
            c = int(cand[int(rng.integers(len(cand)))])
            used.update((c, i))
            pos_sel.append(i)
            ctl_sel.append(c)
    return np.asarray(pos_sel, dtype=np.int64), np.asarray(ctl_sel, dtype=np.int64)


def object_batches(perm: np.ndarray, roles: np.ndarray, bs: int, rng) -> np.ndarray:
    """Reorder an epoch permutation so that object batches hold both kinds: batches of
    bs//2 AML-specific rows and bs - bs//2 control rows (all rows of the epoch are kept), the
    remaining rows in random batches, all batches in random order. Same length as `perm`."""
    if bs < 2:
        return perm
    r = roles[perm]
    pos, ctl, rest = perm[r == ROLE_POS], perm[r == ROLE_CTRL], perm[r == 0]
    hp = bs // 2
    hc = bs - hp
    nb = min(len(pos) // hp, len(ctl) // hc)
    batches = [np.concatenate([pos[k * hp:(k + 1) * hp], ctl[k * hc:(k + 1) * hc]]) for k in range(nb)]
    left = np.concatenate([pos[nb * hp:], ctl[nb * hc:], rest])
    left = left[rng.permutation(len(left))]
    nfull = len(left) // bs
    batches += [left[k * bs:(k + 1) * bs] for k in range(nfull)]
    order = rng.permutation(len(batches))
    out = [batches[k] for k in order] + [left[nfull * bs:]]
    return np.concatenate(out).astype(perm.dtype)


# --------------------------------------------------------------------------- batch geometry
def flip_object(o, window: int):
    """The object after the genomic reversal of its window (both axes flipped, then transposed
    back to the upper triangle): (a0, a1, b0, b1) -> (W - b1, W - b0, W - a1, W - a0)."""
    k, role, a0, a1, b0, b1, oi = o
    return (k, role, window - b1, window - b0, window - a1, window - a0, oi)


class Geometry:
    """Objects of one batch: neighbourhood mask (B, W, W) and index arrays."""

    def __init__(self, mask, loops, stripes):
        self.mask = mask                  # float (B, W, W), 1 in the neighbourhoods
        self.loops = loops                # (b, i, j, role) int64 tensors, or None
        self.stripes = stripes            # list of (b, a0, a1, b0, b1, role)

    def has_pairs(self) -> bool:
        """The batch holds an AML-specific and a control object of one kind (a ranking pair)."""
        rl = [] if self.loops is None else self.loops[3].tolist()
        rs = [t[5] for t in self.stripes]
        return any(ROLE_POS in r and ROLE_CTRL in r for r in (rl, rs))

    @property
    def n_objects(self) -> int:
        return (0 if self.loops is None else int(self.loops[0].numel())) + len(self.stripes)


def batch_geometry(objs_per_window, rev, window: int, outer: int, device, weak_weight: float = 0.2) -> Geometry:
    """`objs_per_window`: per batch element the tuples of rows_of_objects; `rev`: per element
    True when the window was reversed (or None). Weak objects (role 3) enter the mask at `weak_weight`."""
    B = len(objs_per_window)
    mask = torch.zeros(B, window, window)
    lb, li, lj, lr = [], [], [], []
    stripes = []
    for b, objs in enumerate(objs_per_window):
        for o in objs:
            if rev is not None and rev[b]:
                o = flip_object(o, window)
            k, role, a0, a1, b0, b1, _ = o
            mv = weak_weight if role == ROLE_WEAK else 1.0
            if k == 0:                                 # loop: pixel (a0, b0)
                i, j = a0, b0
                lb.append(b), li.append(i), lj.append(j), lr.append(role)
                r0, r1 = max(0, i - outer), min(window, i + outer + 1)
                c0, c1 = max(0, j - outer), min(window, j + outer + 1)
                mask[b, r0:r1, c0:c1] = torch.clamp(mask[b, r0:r1, c0:c1], min=mv)
                mask[b, c0:c1, r0:r1] = torch.clamp(mask[b, c0:c1, r0:r1], min=mv)
            else:                                      # stripe rectangle, one bin of flank
                stripes.append((b, a0, a1, b0, b1, role))
                r0, r1 = max(0, a0 - 1), min(window, a1 + 1)
                c0, c1 = max(0, b0 - 1), min(window, b1 + 1)
                mask[b, r0:r1, c0:c1] = torch.clamp(mask[b, r0:r1, c0:c1], min=mv)
                mask[b, c0:c1, r0:r1] = torch.clamp(mask[b, c0:c1, r0:r1], min=mv)
    loops = None
    if lb:
        loops = tuple(torch.tensor(v, dtype=torch.long, device=device) for v in (lb, li, lj, lr))
    return Geometry(mask.to(device), loops, stripes)


# --------------------------------------------------------------------------- loss
def _ring(inner: int, outer: int, device, dtype):
    r = torch.arange(-outer, outer + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    k = ((dy.abs() > inner) | (dx.abs() > inner)).to(dtype)
    return (k / k.sum())[None, None]


def donut_enrichment(m: torch.Tensor, inner: int, outer: int) -> torch.Tensor:
    """(B, W, W): pixel minus the mean of its donut ring (zero padded; objects keep a margin of
    outer + 1 bins to the window edge)."""
    k = _ring(inner, outer, m.device, m.dtype)
    x = m.unsqueeze(1)
    return (x - F.conv2d(x, k, padding=outer)).squeeze(1)


_OFF = torch.tensor([-1, 0, 1])


def object_scores(maps: torch.Tensor, geo: Geometry, inner: int, outer: int, space_full: bool = False):
    """Object-level scores of a batch of maps (B, W, W): loops, the maximum of the donut
    enrichment over the 3 x 3 pixels around the annotated pixel (as the evaluation); stripes, the
    mean over the rectangle. Returns (loop scores, stripe scores), 1-D tensors (possibly empty)."""
    W = maps.shape[-1]
    if geo.loops is not None:
        b, i, j, _ = geo.loops
        enr = donut_enrichment(maps, inner, outer)
        off = _OFF.to(maps.device)
        ii = (i[:, None, None] + off[None, :, None]).clamp(0, W - 1).expand(-1, 3, 3)
        jj = (j[:, None, None] + off[None, None, :]).clamp(0, W - 1).expand(-1, 3, 3)
        bb = b[:, None, None].expand(-1, 3, 3)
        ls = enr[bb, ii, jj].reshape(len(b), 9).amax(1)
    else:
        ls = maps.new_zeros(0)
    if geo.stripes:
        ss = torch.stack([maps[b, a0:a1, b0:b1].mean() for b, a0, a1, b0, b1, _ in geo.stripes])
    else:
        ss = maps.new_zeros(0)
    return ls, ss


class ObjectLoss:
    """weight * (neighbourhood L1 + object enrichment L1) + rank * pairwise ranking
    + marks_margin * marks-dependence margin.

    pred, tgt, ref: (B, W, W) maps of the Hi-C channel in the encoded space; the residual
    (pred - ref) against (tgt - ref) is what the terms see (rank_space "residual"; "full" scores
    the maps themselves). The ranking term is a logistic loss on the score difference between
    every AML-specific object and every control object (role 2) of the same kind in the batch:
    softplus(-(s_pos - s_ctrl) / temp); with `consistent`, only pairs whose measured scores
    point the same way (target score of the AML-specific object above the control's) count.
    Weak controls (role 3, shared loops) enter the neighbourhood and enrichment terms at
    `weak_weight` and never the ranking.

    Marks dependence (registry section 70): `pred_none` is the prediction of the same windows with the
    sample's own marks replaced by the 'absent' encoding (no gradient). The pair margin without marks,
    d_none = s_pos - s_ctrl of that pass, must stay below the margin with marks by `delta`:
    relu(d_none - d_with + delta) over the ranking pairs, weighted by `marks_margin`. A margin that the
    reference / HSPC shortcut produces alone is the same in both passes and is penalised; only the part
    that needs the marks passes. The mean gap d_with - d_none is logged (obj_gap) whenever pred_none is given."""

    def __init__(self, weight=0.0, rank=0.0, temp=0.05, consistent=True, space="residual",
                 donut=(1, 4), weak_weight=0.2, marks_margin=0.0, delta=0.02):
        if space not in ("residual", "full"):
            raise ValueError(f"unknown aml_object_rank_space {space!r}")
        self.weight, self.rank, self.temp = float(weight), float(rank), float(temp)
        self.consistent, self.space = bool(consistent), space
        self.inner, self.outer = int(donut[0]), int(donut[1])
        self.weak_weight, self.marks_margin, self.delta = float(weak_weight), float(marks_margin), float(delta)
        self.last: dict = {}

    def _pair_terms(self, A, Tm, geo):
        """Per kind (score matrix of pairs, measured pair weights) of the maps A against the target Tm."""
        lp, sp = object_scores(A, geo, self.inner, self.outer)
        lt, st = object_scores(Tm.detach(), geo, self.inner, self.outer)
        roles_l = geo.loops[3] if geo.loops is not None else A.new_zeros(0).long()
        roles_s = torch.tensor([s[5] for s in geo.stripes], dtype=torch.long, device=A.device)
        out = []
        for s, t, r in ((lp, lt, roles_l), (sp, st, roles_s)):
            pos, ctl = r == ROLE_POS, r == ROLE_CTRL
            if not pos.any() or not ctl.any():
                continue
            d = s[pos][:, None] - s[ctl][None, :]
            w = ((t[pos][:, None] - t[ctl][None, :]) > 0).to(d.dtype) if self.consistent else torch.ones_like(d)
            out.append((d, w))
        return out

    def __call__(self, pred, tgt, ref, geo: Geometry, pred_none=None):
        zero = pred.new_zeros(())
        self.last = {"obj_pix": 0.0, "obj_enr": 0.0, "obj_rank": 0.0, "obj_margin": 0.0, "obj_gap": 0.0,
                     "obj_n": float(geo.n_objects)}
        if geo.n_objects == 0:
            return zero
        P, T = pred - ref, tgt - ref
        total = zero
        if self.weight > 0:
            m = geo.mask
            pix = ((P - T).abs() * m).sum() / m.sum().clamp(min=1.0)
            lp, sp = object_scores(P, geo, self.inner, self.outer)
            lt, st = object_scores(T.detach(), geo, self.inner, self.outer)
            roles = torch.cat([geo.loops[3] if geo.loops is not None else pred.new_zeros(0).long(),
                               torch.tensor([x[5] for x in geo.stripes], dtype=torch.long, device=pred.device)])
            wt = torch.where(roles == ROLE_WEAK, torch.full_like(roles, self.weak_weight, dtype=pred.dtype),
                             torch.ones_like(roles, dtype=pred.dtype))
            diffs = torch.cat([(lp - lt).abs(), (sp - st).abs()])
            enr = (diffs * wt).sum() / wt.sum().clamp(min=1e-6) if diffs.numel() else zero
            total = total + self.weight * (pix + enr)
            self.last["obj_pix"], self.last["obj_enr"] = float(pix.detach()), float(enr.detach())
        if self.rank > 0 or pred_none is not None:
            A, Tm = (P, T) if self.space == "residual" else (pred, tgt)
            pairs = self._pair_terms(A, Tm, geo)
            if self.rank > 0 and pairs:
                rk = torch.stack([(F.softplus(-d / self.temp) * w).sum() / w.sum()
                                  for d, w in pairs if w.sum() > 0] or [zero]).mean()
                total = total + self.rank * rk
                self.last["obj_rank"] = float(rk.detach())
            if pred_none is not None and pairs:
                An = (pred_none - ref) if self.space == "residual" else pred_none
                with torch.no_grad():
                    none_pairs = self._pair_terms(An, Tm, geo)
                gaps, mg = [], []
                for (d, w), (dn, _) in zip(pairs, none_pairs):
                    if w.sum() <= 0:
                        continue
                    gaps.append(((d - dn) * w).sum() / w.sum())
                    mg.append((F.relu(dn - d + self.delta) * w).sum() / w.sum())
                if gaps:
                    self.last["obj_gap"] = float(torch.stack(gaps).mean().detach())
                    if self.marks_margin > 0:
                        mm = torch.stack(mg).mean()
                        total = total + self.marks_margin * mm
                        self.last["obj_margin"] = float(mm.detach())
        return total
