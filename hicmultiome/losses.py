"""Objectives that match how Hi-C predictions are actually judged.

Naming, because these are easy to confuse:

  HiCRep   a reproducibility MEASURE; the number it produces is the SCC.
  SCC      stratum-adjusted correlation coefficient: correlate predicted and
           measured contacts separately within each genomic-distance stratum,
           then average with HiCRep's variance weights. Stratifying is the
           point -- it removes the distance decay, which otherwise dominates
           any plain correlation and flatters every model.
  HiC-Reg  not a measure at all: a random-forest PREDICTION tool (Zhang 2019),
           one of the competitors.
  SSIM     structural similarity, an image measure used by the Hi-C
           super-resolution literature; rewards matching local means,
           variances and covariance rather than exact values.

Training has been L1+MSE on log-normalised contacts while evaluation is SCC,
so the objective and the metric disagree: L1 is dominated by the near-diagonal
pixels that carry most of the signal, exactly the ones SCC discounts. These
losses are differentiable versions of the metrics, so the model can optimise
what it is scored on.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


_STRATA_CACHE: dict = {}


def _strata_index(w: int, dmax: int, stride: int, device):
    """Flat upper-triangle positions and the stratum each one belongs to.

    Looping over the W diagonals in Python costs one handful of tiny CUDA
    kernels per stratum and dominated the step time (36 ms against 0.2 ms for
    L1+MSE at W=128). Instead every pixel is labelled with its stratum once and
    all strata are reduced together by a single scatter-add.
    """
    key = (w, dmax, stride, str(device))
    hit = _STRATA_CACHE.get(key)
    if hit is None:
        i, j = torch.triu_indices(w, w, device=device)     # i <= j
        d = j - i
        keep = (d < dmax) & (d % stride == 0)
        i, j, d = i[keep], j[keep], d[keep]
        nbins = int(d.max().item()) + 1 if d.numel() else 1
        hit = _STRATA_CACHE[key] = (i, j, d, nbins)
    return hit


def scc(pred: torch.Tensor, target: torch.Tensor, max_dist: int | None = None,
        stride: int = 1, eps: float = 1e-8, weighted: bool = True) -> torch.Tensor:
    """Differentiable stratum-adjusted correlation for a batch of windows.

    pred/target: (B, W, W). Each genomic-distance stratum is pooled across the
    batch before correlating, which keeps the per-stratum sample size large
    enough to be stable at small batch sizes.

    `weighted` selects HiCRep's variance weight n_d*sx_d*sy_d (True) or a plain
    mean over strata (False). The choice depends on the target space and it
    matters -- see `dcorr` below.
    """
    assert pred.shape == target.shape and pred.dim() == 3
    b, _, w = pred.shape
    dmax = min(max_dist or w, w)
    i, j, d, nbins = _strata_index(w, dmax, stride, pred.device)

    x = pred[:, i, j].reshape(-1)
    y = target[:, i, j].reshape(-1)
    idx = d.repeat(b)

    def total(v):
        return torch.zeros(nbins, dtype=v.dtype, device=v.device).index_add(0, idx, v)

    n = total(torch.ones_like(x))
    n = n.clamp(min=1.0)
    mx, my = total(x) / n, total(y) / n
    # E[xy] - E[x]E[y] and the two variances, all per stratum
    cov = total(x * y) / n - mx * my
    vx = (total(x * x) / n - mx * mx).clamp(min=0)
    vy = (total(y * y) / n - my * my).clamp(min=0)
    # sqrt has infinite derivative at 0 and a fresh model emits near-constant
    # maps, so the variance is smoothed inside the root, not after it.
    sx, sy = (vx + eps).sqrt(), (vy + eps).sqrt()
    r = cov / (sx * sy)
    weight = n * sx * sy if weighted else torch.ones_like(n)
    weight = weight * (n > 1).to(weight.dtype)
    return (weight * r).sum() / (weight.sum() + eps)


def dcorr(pred: torch.Tensor, target: torch.Tensor, max_dist: int | None = None,
          stride: int = 1) -> torch.Tensor:
    """Distance-stratified correlation with every stratum weighted equally.

    Which of this and `scc` is the right objective depends on the target space,
    and the two runs here are in different ones.

    In COUNT space (log1p targets) the distance decay is still present and
    dominates any plain correlation, so stratifying is essential; HiCRep's
    n_d*sx_d*sy_d weight then concentrates on the near-diagonal strata, which
    is where the counts and therefore the reliable signal are. `scc` is right.

    In OBS/EXP space (the `prescaled` Akita targets: log observed/expected)
    the decay has already been divided out, so SCC's main justification is
    gone. Worse, the variance weight becomes actively harmful: in obs/exp
    space the far strata have the LARGEST variance because they are the
    sparsest and noisiest, so weighting by sx*sy up-weights exactly the strata
    whose values are least trustworthy, pushing the model to fit noise. An
    unweighted mean over strata is the honest analogue there, and it is also
    the metric checkpoints are already selected on.
    """
    return scc(pred, target, max_dist=max_dist, stride=stride, weighted=False)


def dcorr_loss(pred: torch.Tensor, target: torch.Tensor,
               max_dist: int | None = None, stride: int = 1) -> torch.Tensor:
    return 1.0 - dcorr(pred, target, max_dist=max_dist, stride=stride)


def scc_loss(pred: torch.Tensor, target: torch.Tensor, max_dist: int | None = None,
             stride: int = 1) -> torch.Tensor:
    """1 - SCC, so minimising it maximises the metric we report."""
    return 1.0 - scc(pred, target, max_dist=max_dist, stride=stride)


def _gaussian(window: int, sigma: float, device, dtype):
    x = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2
    g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(1, 1, window, window)


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11,
         sigma: float = 1.5, data_range: float = 2.0) -> torch.Tensor:
    """Structural similarity on (B, W, W) maps, values assumed in [-1, 1]."""
    p = pred.unsqueeze(1)
    t = target.unsqueeze(1)
    k = _gaussian(window, sigma, p.device, p.dtype)
    pad = window // 2
    mu_p = F.conv2d(p, k, padding=pad)
    mu_t = F.conv2d(t, k, padding=pad)
    pp, tt, pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    s_p = F.conv2d(p * p, k, padding=pad) - pp
    s_t = F.conv2d(t * t, k, padding=pad) - tt
    s_pt = F.conv2d(p * t, k, padding=pad) - pt
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * pt + c1) * (2 * s_pt + c2)) / ((pp + tt + c1) * (s_p + s_t + c2))
    return m.mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(pred, target)


def _gradients(x: torch.Tensor):
    """First differences along both axes; (B, H-1, W) and (B, H, W-1)."""
    return x[:, 1:, :] - x[:, :-1, :], x[:, :, 1:] - x[:, :, :-1]


def grad_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between the spatial gradients of prediction and target.

    L1/MSE alone are minimised by the conditional mean, so wherever the model
    is unsure of fine texture it hedges toward the local average and the map
    comes out blurry -- at no cost in correlation, which is why the reported
    scores never flagged it. Measured against the truth's fine-scale energy,
    our predictions carry 0.057 of it and Hi-cGAN 0.239.

    Matching gradients makes that hedging expensive: a smooth patch where the
    target has structure now has a gradient deficit. Note it penalises
    INVENTED texture just as much, which matters because the truth's
    high-frequency content is partly measurement noise and reproducing noise
    is not the goal.
    """
    px, py = _gradients(pred)
    tx, ty = _gradients(target)
    return 0.5 * (torch.nn.functional.l1_loss(px, tx)
                  + torch.nn.functional.l1_loss(py, ty))


def hf_energy_loss(pred: torch.Tensor, target: torch.Tensor,
                   sigma: float = 1.5) -> torch.Tensor:
    """Match how much energy sits above a blur, without dictating where.

    Softer than grad_loss: it asks the prediction to be as textured as the
    target on average rather than to match texture pixel by pixel, so it does
    not force the model to reproduce noise at specific positions.
    """
    k = max(3, int(2 * round(3 * sigma) + 1))
    g = _gaussian(k, sigma, pred.device, pred.dtype)
    pad = k // 2
    p = pred.unsqueeze(1)
    t = target.unsqueeze(1)
    hp = p - F.conv2d(p, g, padding=pad)
    ht = t - F.conv2d(t, g, padding=pad)
    return (hp.std() - ht.std()).abs()


def build_loss(spec: str, max_dist: int | None = None, stride: int = 1):
    """`spec` is a '+'-joined list with optional weights, e.g.
    'l1+mse', 'l1+mse+0.5*scc', 'scc', '0.8*l1+0.2*ssim'."""
    terms = []
    for part in spec.split("+"):
        part = part.strip()
        if not part:
            continue
        w, _, name = part.rpartition("*")
        terms.append((float(w) if w else 1.0, name.strip().lower()))
    known = {"l1", "mse", "scc", "dcorr", "ssim", "grad", "hf",
             "insul", "donut", "compart"}
    for _, n in terms:
        if n not in known:
            raise ValueError(f"unknown loss term {n!r} (of {sorted(known)})")

    def loss_fn(pred: torch.Tensor, target: torch.Tensor,
                pixel_weight: torch.Tensor | None = None) -> torch.Tensor:
        """pixel_weight (B, W, W), mean 1 over the batch: weights the l1 and
        mse terms per pixel (other terms ignore it)."""
        total = pred.new_zeros(())
        parts = {}
        for weight, name in terms:
            if name == "l1":
                v = (F.l1_loss(pred, target) if pixel_weight is None
                     else ((pred - target).abs() * pixel_weight).mean())
            elif name == "mse":
                v = (F.mse_loss(pred, target) if pixel_weight is None
                     else ((pred - target).pow(2) * pixel_weight).mean())
            elif name == "scc":
                v = scc_loss(pred, target, max_dist=max_dist, stride=stride)
            elif name == "dcorr":
                v = dcorr_loss(pred, target, max_dist=max_dist, stride=stride)
            elif name == "grad":
                v = grad_loss(pred, target)
            elif name == "hf":
                v = hf_energy_loss(pred, target)
            elif name == "insul":
                v = insulation_loss(pred, target)
            elif name == "donut":
                v = donut_loss(pred, target)
            elif name == "compart":
                v = compartment_loss(pred, target)
            else:
                v = ssim_loss(pred, target)
            parts[name] = v.detach()
            total = total + weight * v
        loss_fn.last_terms = parts          # unweighted value of every term
        return total

    loss_fn.last_terms = {}
    loss_fn.term_names = [n for _, n in terms]

    return loss_fn


# --------------------------------------------------------------------------- #
# Structure terms.
#
# The feature-level score showed our margin over Akita is wider on the biology
# (boundary F1 0.582 vs 0.388, loop F1 0.553 vs 0.465) than on pixels (0.646 vs
# 0.548). So the biology is worth optimising directly -- but F1 involves
# thresholding and discrete matching and has no gradient. What DOES have a
# gradient is the continuous quantity each F1 is computed from: the insulation
# profile behind boundary calls, and the local enrichment behind loop calls.
# --------------------------------------------------------------------------- #

def insulation_profile_t(m: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Mean contact in the square just upstream x downstream of each bin.

    This is the quantity TAD boundaries are called from: it dips where a
    boundary blocks contacts across it. Computed as a strided box-sum so it
    stays a couple of convolutions rather than a Python loop over bins.
    """
    b, n, _ = m.shape
    k = torch.ones(1, 1, window, window, device=m.device, dtype=m.dtype)
    s = F.conv2d(m.unsqueeze(1), k, padding=0).squeeze(1)   # (B, n-w+1, n-w+1)
    # the square [i-w, i) x (i, i+w] has its top-left corner at (i-w, i+1)
    idx = torch.arange(window, s.shape[1], device=m.device)
    col = torch.clamp(idx + 1, max=s.shape[2] - 1)
    prof = s[:, idx - window, col] / (window * window)
    return prof


def insulation_loss(pred: torch.Tensor, target: torch.Tensor,
                    window: int = 10) -> torch.Tensor:
    """Match the insulation profile, in shape and in position.

    Half L1 on the profile itself and half 1-correlation: the L1 keeps the
    absolute level honest, the correlation is what boundary calling actually
    depends on, since a profile offset by a constant calls the same boundaries.
    """
    pp = insulation_profile_t(pred, window)
    tt = insulation_profile_t(target, window)
    l1 = F.l1_loss(pp, tt)
    p = pp - pp.mean(dim=1, keepdim=True)
    t = tt - tt.mean(dim=1, keepdim=True)
    r = (p * t).mean(dim=1) / ((p.pow(2).mean(dim=1) + 1e-8).sqrt()
                               * (t.pow(2).mean(dim=1) + 1e-8).sqrt())
    return 0.5 * l1 + 0.5 * (1.0 - r.mean())


def _donut_kernel(inner: int, outer: int, device, dtype):
    r = torch.arange(-outer, outer + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    ring = ((dy.abs() > inner) | (dx.abs() > inner)).to(dtype)
    ring = ring / ring.sum()
    return ring[None, None]


def donut_loss(pred: torch.Tensor, target: torch.Tensor,
               inner: int = 2, outer: int = 5) -> torch.Tensor:
    """Match local enrichment: pixel minus its surrounding ring.

    This is what a loop caller thresholds. Matching it pushes the model to put
    focal enrichment where the measurement has it, which plain L1 does not --
    L1 is equally happy with a smooth bump of the same mass.
    """
    k = _donut_kernel(inner, outer, pred.device, pred.dtype)
    pad = outer
    p = pred.unsqueeze(1)
    t = target.unsqueeze(1)
    ep = p - F.conv2d(p, k, padding=pad)
    et = t - F.conv2d(t, k, padding=pad)
    return F.l1_loss(ep, et)


def corr_matrix_t(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pearson correlation between the rows of each window: (B, W, W)."""
    x = m - m.mean(dim=2, keepdim=True)
    x = x / (x.pow(2).mean(dim=2, keepdim=True) + eps).sqrt()
    return x @ x.transpose(1, 2) / m.shape[-1]


def compartment_loss(pred: torch.Tensor, target: torch.Tensor,
                     min_d: int = 2) -> torch.Tensor:
    """1 - Pearson between the row-correlation matrices of prediction and
    target, per window, averaged over the batch.

    The row-correlation matrix of an O/E map is what compartment calling
    takes its first eigenvector from; matching the whole matrix instead of
    its eigenvector avoids the sign ambiguity and the eigen-decomposition.
    Windows are about 2 Mb, so this sees the compartment checkerboard inside
    2 Mb only (a within-window proxy for A/B compartments). Pixels within
    min_d bins of the diagonal are left out."""
    cp, ct = corr_matrix_t(pred), corr_matrix_t(target)
    w = pred.shape[-1]
    i, j = torch.triu_indices(w, w, offset=min_d, device=pred.device)
    a, b = cp[:, i, j], ct[:, i, j]
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    r = (a * b).mean(dim=1) / ((a.pow(2).mean(dim=1) + 1e-8).sqrt()
                               * (b.pow(2).mean(dim=1) + 1e-8).sqrt())
    return 1.0 - r.mean()
