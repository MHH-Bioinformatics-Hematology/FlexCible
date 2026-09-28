"""Spatial-conditioning U-Net for multi-omic Hi-C prediction. No adversary:
the earlier cGAN experiments showed the adversarial term does not help this
task, so this is a plain regression U-Net (the approach that reached SCC 0.84
from CTCF alone).

Conditioning is built from three ingredient groups, all per-bin 1-D signals:
  - protein ChIP tracks (each DNA-binding protein with its required motif)
  - DNA sequence features (GC, CpG density)
  - motif-hit tracks from PWM-scanning the genome (fwd / rev strand separate)

For symmetric channels t_c the 2-D conditioning image at pixel (i, j) is
    mean_c(i,j), prod_c(i,j), |diff_c(i,j)|              (3 planes per channel)
plus one |i-j|/W distance plane. For every motif, two *directed* planes are
added from its strand-separated hit tracks:
    fwd[i] * rev[j]   and   rev[i] * fwd[j]
so a convergently oriented motif pair (the CTCF loop-anchor geometry) lights
up exactly at its loop pixel -- something no symmetric combination can encode.

Output: 1 channel (Hi-C) or 2 channels (Hi-C + capture-Hi-C auxiliary), each
symmetrised, in [-1, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn


COND_MODES = ("full", "pair_dist", "pair", "tile_dist", "tile")


def build_spatial_cond(
    sym: torch.Tensor, pairs: torch.Tensor | None = None,
    planes: torch.Tensor | None = None, mode: str = "full",
) -> torch.Tensor:
    """sym: (B, T, W) symmetric 1-D channels; pairs: (B, M, 2, W) strand pairs;
    planes: (B, P, W, W) ready-made 2-D conditioning planes (e.g. an optional
    capture-Hi-C map + its bait-pair mask), appended verbatim.

    This is a fixed, PARAMETER-FREE tensor op applied to each window on every
    forward pass: the rule mapping input positions to pixel (i, j) is the same
    for every window and every genome position, only the values differ.

    `mode` controls how much of that 1D->2D wiring is handed to the network,
    which is what the conditioning ablation measures:
      full      mean/prod/|diff| per track + directed motif fwd_i*rev_j planes
                + distance -- the pairing AND the interaction are given
      pair_dist t_i and t_j broadcast (both endpoints present at the pixel)
                + distance -- pairing given, interaction must be learned
      pair      t_i and t_j only -- pairing given, nothing else
      tile_dist t_i broadcast only (+ distance) -- the tensor is CONSTANT
                along j, so the network must discover the correspondence
                itself, as a 1D->2D generator has to
      tile      t_i broadcast only -- hardest, no hints at all
    """
    if mode not in COND_MODES:
        raise ValueError(f"cond mode {mode!r} not in {COND_MODES}")
    b, t, w = sym.shape
    ti = sym[:, :, :, None].expand(b, t, w, w)
    if mode == "full":
        tj = sym[:, :, None, :].expand(b, t, w, w)
        feats = [0.5 * (ti + tj), ti * tj, (ti - tj).abs()]
        if pairs is not None and pairs.shape[1] > 0:
            fwd, rev = pairs[:, :, 0, :], pairs[:, :, 1, :]      # (B, M, W)
            feats += [fwd[:, :, :, None] * rev[:, :, None, :],   # fwd_i * rev_j
                      rev[:, :, :, None] * fwd[:, :, None, :]]   # rev_i * fwd_j
    elif mode.startswith("pair"):
        feats = [ti, sym[:, :, None, :].expand(b, t, w, w)]
    else:                                    # tile: pixel (i, j) never sees j
        feats = [ti]
    if planes is not None and planes.shape[1] > 0:
        feats.append(planes)
    if mode in ("full", "pair_dist", "tile_dist"):
        idx = torch.arange(w, device=sym.device, dtype=sym.dtype)
        dist = (idx[:, None] - idx[None, :]).abs()[None, None] / w
        feats.append(dist.expand(b, 1, w, w))
    return torch.cat(feats, dim=1)


def n_cond_channels(n_sym: int, n_motifs: int, n_planes: int = 0,
                    mode: str = "full") -> int:
    if mode not in COND_MODES:
        raise ValueError(f"cond mode {mode!r} not in {COND_MODES}")
    if mode == "full":
        base = 3 * n_sym + 2 * n_motifs
    elif mode.startswith("pair"):
        base = 2 * n_sym
    else:
        base = n_sym
    return base + n_planes + (1 if mode.endswith("dist") or mode == "full" else 0)


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class HiCUNet(nn.Module):
    """Small symmetric U-Net: (B, in_ch, W, W) -> (B, out_ch, W, W) in [-1, 1]."""

    def __init__(self, in_ch: int, out_ch: int = 1, base: int = 48, depth: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        chans = [base * (2 ** i) for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
            self.downs.append(_Block(chans[i], chans[i]))
            self.pools.append(nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1))
        self.mid = _Block(chans[-1], chans[-1])
        self.ups = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 4, stride=2, padding=1))
            self.upblocks.append(_Block(chans[i] * 2, chans[i]))
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for blk, pool in zip(self.downs, self.pools):
            x = blk(x)
            skips.append(x)
            x = pool(x)
        x = self.mid(x)
        for up, blk, skip in zip(self.ups, self.upblocks, reversed(skips)):
            x = up(x)
            x = blk(torch.cat([x, skip], dim=1))
        out = torch.tanh(self.head(x))
        return 0.5 * (out + out.transpose(-1, -2))     # enforce contact symmetry


# --------------------------------------------------------------------------- #
# alternative backbones
#
# The U-Net was the first thing that worked and has been the only Hi-C
# backbone since. These give the same interface -- (B, in_ch, W, W) ->
# (B, out_ch, W, W), symmetrised -- so the conditioning, the loss and the
# merge stay fixed and only the architecture varies. `dilated` is the family
# Akita uses, which is the relevant comparison at fine resolution.
# --------------------------------------------------------------------------- #
def _sym(out: torch.Tensor) -> torch.Tensor:
    out = torch.tanh(out)
    return 0.5 * (out + out.transpose(-1, -2))


class PixelMLP(nn.Module):
    """1x1 convolutions only: every pixel judged from its own features, with
    no spatial context at all. The control that says how much context buys."""

    def __init__(self, in_ch: int, out_ch: int = 1, width: int = 128, depth: int = 3):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(depth):
            layers += [nn.Conv2d(c, width, 1), nn.SiLU()]
            c = width
        layers.append(nn.Conv2d(c, out_ch, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return _sym(self.net(x))


class _DilBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DilatedCNN(nn.Module):
    """Residual dilated convolutions at full resolution -- no downsampling, so
    fine detail is never blurred away. This is Akita's trunk family."""

    def __init__(self, in_ch: int, out_ch: int = 1, width: int = 64,
                 dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1)
        self.blocks = nn.Sequential(*[_DilBlock(width, d) for d in dilations])
        self.head = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        return _sym(self.head(self.blocks(self.stem(x))))


class _AxialAttn(nn.Module):
    def __init__(self, ch: int, heads: int = 4, axis: int = -1):
        super().__init__()
        self.axis = axis
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)
        if self.axis == -1:
            y = y.permute(0, 2, 3, 1).reshape(b * h, w, c)
        else:
            y = y.permute(0, 3, 2, 1).reshape(b * w, h, c)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = (y.reshape(b, h, w, c).permute(0, 3, 1, 2) if self.axis == -1
             else y.reshape(b, w, h, c).permute(0, 3, 2, 1))
        return x + y


class AxialNet(nn.Module):
    """Row and column attention: every pixel can reach any other on its own
    row or column, so long-range anchors are visible without downsampling."""

    def __init__(self, in_ch: int, out_ch: int = 1, width: int = 64, depth: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1)
        blocks = []
        for _ in range(depth):
            blocks += [_AxialAttn(width, axis=-1), _AxialAttn(width, axis=-2),
                       nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                     nn.GroupNorm(8, width), nn.SiLU())]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        return _sym(self.head(self.blocks(self.stem(x))))


class SeqEncoder(nn.Module):
    """Learned DNA sequence trunk, Akita-style: one-hot bases (B, 4, L) over a
    window's genomic span are reduced to a per-bin embedding (B, embed_dim, W)
    by a stack of strided convolutions (bp -> bins) followed by dilated
    convolutions for a wide receptive field, then adaptive-pooled to exactly W
    bins so it works at any bin size. The embedding is concatenated to the
    per-bin conditioning, so the model reads the actual sequence rather than
    only the GC/CpG summary."""

    def __init__(self, embed_dim: int = 8, width: int = 48, n_down: int = 5,
                 dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.stem = nn.Conv1d(4, width, 7, padding=3)
        self.downs = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(width, width, 7, stride=4, padding=3),
                          nn.GroupNorm(8, width), nn.SiLU())
            for _ in range(n_down)])
        self.dil = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(width, width, 3, padding=d, dilation=d),
                          nn.GroupNorm(8, width), nn.SiLU())
            for d in dilations])
        self.head = nn.Conv1d(width, embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, onehot: torch.Tensor, n_bins: int) -> torch.Tensor:
        x = self.dil(self.downs(self.stem(onehot)))
        x = torch.nn.functional.adaptive_avg_pool1d(x, n_bins)
        return self.head(x)                                   # (B, embed_dim, W)


class HiCModel(nn.Module):
    """Backbone plus an optional learned sequence encoder, as one module so DDP
    and checkpointing cover both. The encoder's per-bin embedding is appended to
    the symmetric conditioning rows (after the motif rows, so the motif fwd/rev
    view keeps its offset), and the fixed 1D->2D conditioning is built here."""

    def __init__(self, backbone: nn.Module, cond_mode: str,
                 encoder: SeqEncoder | None = None, dist_bias: int = 0,
                 reference: str = "none"):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.cond_mode = cond_mode
        # reference map (config model.reference): "input" appends it as one
        # 2D conditioning plane (after any other planes); "residual" also adds
        # it to the Hi-C output channel, so the backbone predicts the deviation
        # from the reference. No parameters, so "none" checkpoints load as before.
        if reference not in ("none", "input", "residual"):
            raise ValueError(f"unknown reference mode {reference!r}")
        self.reference = reference
        # optional learned per-distance offset b[|i-j|] added to the Hi-C
        # output channel (symmetric by construction); 0 = off, so checkpoints
        # without it load unchanged
        self.dist_bias = (nn.Parameter(torch.zeros(dist_bias)) if dist_bias
                          else None)

    def forward(self, cond, pairs=None, planes=None, seq_onehot=None, ref=None):
        """ref: (B, W, W) encoded reference map; required unless reference is
        "none" (then ignored)."""
        sym = cond
        if self.encoder is not None and seq_onehot is not None:
            emb = self.encoder(seq_onehot, cond.shape[-1])
            sym = torch.cat([cond, emb], dim=1)
        if self.reference != "none":
            if ref is None:
                raise ValueError(f"reference={self.reference!r} model called without ref")
            ref = ref.to(cond.dtype)[:, None]
            planes = ref if planes is None else torch.cat([planes, ref], dim=1)
        out = self.backbone(build_spatial_cond(sym, pairs, planes, self.cond_mode))
        if self.dist_bias is not None:
            w = out.shape[-1]
            idx = torch.arange(w, device=out.device)
            b = self.dist_bias[:w][(idx[:, None] - idx[None, :]).abs()]
            out = torch.cat([out[:, :1] + b.to(out.dtype), out[:, 1:]], dim=1)
        if self.reference == "residual":
            out = torch.cat([out[:, :1] + ref.to(out.dtype), out[:, 1:]], dim=1)
        return out


BACKBONES = ("unet", "dilated", "axial", "pixelmlp")


def build_backbone(kind: str, in_ch: int, out_ch: int = 1, base: int = 48,
                   depth: int = 4) -> nn.Module:
    if kind == "unet":
        return HiCUNet(in_ch, out_ch=out_ch, base=base, depth=depth)
    if kind == "dilated":
        return DilatedCNN(in_ch, out_ch=out_ch, width=max(base, 64))
    if kind == "axial":
        return AxialNet(in_ch, out_ch=out_ch, width=max(base, 64), depth=depth)
    if kind == "pixelmlp":
        return PixelMLP(in_ch, out_ch=out_ch, width=max(base, 64))
    raise ValueError(f"unknown backbone {kind!r} (one of {BACKBONES})")
