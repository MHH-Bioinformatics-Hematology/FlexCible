"""Cohesin-loss augmentation (registry section 55).

A degron pair (control and 6 h auxin Hi-C of one cell) gives one measured
cohesin-depleted condition. `DegronOperator` is a small translation-invariant
network fitted on that pair (scripts/cohesin_operator.py): it maps a control
log O/E window to the change that the depletion causes, in log O/E units.
`augment_cohesin` applies it on the fly to the training windows of the intact
samples, together with the matching change of the RAD21/SMC3 rows and of the
cohesin:fraction row, so that the cohesin signal and the fraction row, and not
the identity of one cell or one ChIP batch, carry the depletion.

All functions work on batches already on the training device.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


class DegronOperator(nn.Module):
    """Control log O/E window (B, W, W) -> change of the window under cohesin
    depletion (B, W, W), log O/E units: dilated 3x3 convolutions on the map and
    the genomic distance, symmetrised, zero within 2 bins of the diagonal."""

    def __init__(self, width: int = 24, dilations=(1, 2, 4, 8, 1)):
        super().__init__()
        layers, c = [], 2
        for d in dilations:
            layers += [nn.Conv2d(c, width, 3, padding=d, dilation=d), nn.SiLU()]
            c = width
        layers.append(nn.Conv2d(c, 1, 1))
        self.net = nn.Sequential(*layers)
        self.width, self.dilations = width, tuple(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, _ = x.shape
        idx = torch.arange(w, device=x.device, dtype=x.dtype)
        dist = (idx[:, None] - idx[None, :]).abs()
        d = (dist / w)[None, None].expand(b, 1, w, w)
        o = self.net(torch.cat([x[:, None] / 2.0, d], 1))[:, 0]
        o = 0.5 * (o + o.transpose(-1, -2))
        return o * (dist >= 2).to(o.dtype)[None]


def save_operator(path, op: DegronOperator, fraction_ref: float, resolution: int, meta=None):
    """fraction_ref: cohesin fraction of the condition the operator was fitted on."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": op.state_dict(), "width": op.width, "dilations": op.dilations,
                "fraction_ref": float(fraction_ref), "resolution": int(resolution),
                "meta": meta or {}}, path)


def load_operator(path, device="cpu") -> tuple[DegronOperator, float]:
    d = torch.load(path, map_location=device)
    op = DegronOperator(d["width"], d["dilations"]).to(device)
    op.load_state_dict(d["state"])
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op, float(d["fraction_ref"])


def fraction_z(f: torch.Tensor | float) -> torch.Tensor | float:
    """Normalised value of the cohesin:fraction row, 0 for f = 1, -1 for f = 0
    (same as hicmultiome.precompute.fraction_row_value)."""
    if isinstance(f, torch.Tensor):
        return torch.log1p(f) / math.log(2.0) - 1.0
    return math.log1p(f) / math.log(2.0) - 1.0


class CohesinAugmenter:
    """Replaces a random share of the intact windows of a batch by synthetic
    cohesin-depleted ones.

    rows: cond-row positions in the (possibly reduced) training cond of the
    RAD21/SMC3 tracks present (`coh_rows`), with their normaliser mean/std and
    absent codes, and of the cohesin:fraction row (`frac_row`).
    """

    def __init__(self, op: DegronOperator, fraction_ref: float, prob: float,
                 fractions, p_absent: float, coh_rows, coh_mean, coh_std, coh_absent,
                 frac_row: int, device):
        if fraction_ref >= 1.0:
            raise ValueError("the operator must be fitted on a depleted condition")
        self.op, self.f_ref, self.prob = op, float(fraction_ref), float(prob)
        self.fractions = torch.tensor(list(fractions), dtype=torch.float32, device=device)
        self.p_absent = float(p_absent)
        self.coh_rows = list(coh_rows)
        self.mean = torch.tensor(coh_mean, dtype=torch.float32, device=device)[:, None]
        self.std = torch.tensor(coh_std, dtype=torch.float32, device=device)[:, None]
        self.absent = torch.tensor(coh_absent, dtype=torch.float32, device=device)[:, None]
        self.frac_row = int(frac_row)
        self.device = device

    @torch.no_grad()
    def __call__(self, cond: torch.Tensor, tgt: torch.Tensor, scale: torch.Tensor,
                 generator: torch.Generator | None = None):
        """cond (B, R, W) float, tgt (B, 1, W, W) in [-1, 1] (x / scale), scale (B,)
        contact scale of each window's sample. Returns (cond, tgt, n_aug)."""
        bs = cond.shape[0]
        intact = cond[:, self.frac_row].mean(-1) > -1e-6          # fraction row 0: f = 1
        pick = (torch.rand(bs, device=cond.device, generator=generator) < self.prob) & intact
        if not bool(pick.any()):
            return cond, tgt, 0
        f = self.fractions[torch.randint(len(self.fractions), (bs,), device=cond.device,
                                         generator=generator)]
        absent = torch.rand(bs, device=cond.device, generator=generator) < self.p_absent
        cond = cond.clone()
        tgt = tgt.clone()
        sel = pick.nonzero().squeeze(1)                           # the operator only runs on these
        sc = scale[sel].to(tgt.dtype)[:, None, None]
        x = tgt[sel, 0] * sc
        delta = self.op(x) * (tgt[sel, 0] != 0).to(x.dtype)       # empty pixels stay empty
        fac = ((1.0 - f[sel]) / (1.0 - self.f_ref))[:, None, None]
        tgt[sel, 0] = torch.clamp((x + fac * delta) / sc, -1.0, 1.0)
        for k, r in enumerate(self.coh_rows):
            z = cond[:, r]
            raw = torch.expm1(z * self.std[k] + self.mean[k]).clamp(min=0)
            med = raw.median(dim=-1, keepdim=True).values
            raw2 = (med + f[:, None] * (raw - med)).clamp(min=0)
            z2 = (torch.log1p(raw2) - self.mean[k]) / self.std[k]
            z2 = torch.where(absent[:, None], self.absent[k].expand_as(z2), z2)
            cond[:, r] = torch.where(pick[:, None], z2, z)
        fz = fraction_z(f)[:, None].expand_as(cond[:, self.frac_row])
        cond[:, self.frac_row] = torch.where(pick[:, None], fz, cond[:, self.frac_row])
        return cond, tgt, int(pick.sum())
