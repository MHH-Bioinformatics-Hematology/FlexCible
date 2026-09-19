import numpy as np
import pytest
import torch

from hicmultiome.losses import build_loss, dcorr, scc, scc_loss, ssim


def _reference_scc(pred, target, max_dist=None):
    """Independent numpy implementation of HiCRep's weighting, to check the
    torch version against something that was not written the same way."""
    w = pred.shape[-1]
    num = den = 0.0
    for d in range(min(max_dist or w, w)):
        x = np.concatenate([np.diagonal(p, offset=d) for p in pred])
        y = np.concatenate([np.diagonal(t, offset=d) for t in target])
        if x.size < 2:
            continue
        sx, sy = x.std(), y.std()
        r = np.corrcoef(x, y)[0, 1] if sx > 0 and sy > 0 else 0.0
        wt = x.size * sx * sy
        num += wt * r
        den += wt
    return num / den


def _maps(seed=0, b=3, w=32):
    g = np.random.default_rng(seed)
    a = g.normal(size=(b, w, w)).astype(np.float32)
    a = 0.5 * (a + a.transpose(0, 2, 1))
    return a


def test_scc_matches_numpy_reference():
    p, t = _maps(0), _maps(1)
    got = scc(torch.from_numpy(p), torch.from_numpy(t)).item()
    assert got == pytest.approx(_reference_scc(p, t), abs=1e-4)


def test_scc_is_one_for_identical_maps():
    p = torch.from_numpy(_maps(2))
    assert scc(p, p).item() == pytest.approx(1.0, abs=1e-4)
    assert scc_loss(p, p).item() == pytest.approx(0.0, abs=1e-4)


def test_scc_ignores_a_pure_distance_decay():
    """A prediction that only reproduces the distance decay -- no structure --
    is what a plain correlation rewards. Stratifying must not reward it."""
    w = 48
    idx = np.arange(w)
    decay = np.exp(-np.abs(idx[:, None] - idx[None, :]) / 8.0).astype(np.float32)
    truth = decay + 0.3 * _maps(3, b=1, w=w)[0]
    p = torch.from_numpy(decay[None])
    t = torch.from_numpy(truth[None])
    plain = np.corrcoef(decay.ravel(), truth.ravel())[0, 1]
    assert plain > 0.75                     # flattering
    assert abs(scc(p, t).item()) < 0.2      # honest


def test_scc_gradient_flows_and_improves_the_metric():
    torch.manual_seed(0)
    t = torch.from_numpy(_maps(4))
    p = torch.zeros_like(t).requires_grad_(True)
    opt = torch.optim.SGD([p], lr=0.5)
    before = scc(p.detach(), t).item()
    for _ in range(50):
        opt.zero_grad()
        loss = scc_loss(p, t)
        loss.backward()
        assert torch.isfinite(p.grad).all()
        opt.step()
    assert scc(p.detach(), t).item() > before + 0.2


def test_max_dist_and_stride_restrict_the_strata():
    p, t = torch.from_numpy(_maps(5)), torch.from_numpy(_maps(6))
    near = scc(p, t, max_dist=8).item()
    assert near == pytest.approx(_reference_scc(p.numpy(), t.numpy(), 8), abs=1e-4)
    assert np.isfinite(scc(p, t, stride=4).item())


def test_ssim_peaks_on_identity():
    p = torch.from_numpy(_maps(7))
    assert ssim(p, p).item() == pytest.approx(1.0, abs=1e-3)
    assert ssim(p, torch.from_numpy(_maps(8))).item() < 0.5


def test_build_loss_parses_weights_and_terms():
    p, t = torch.from_numpy(_maps(9)), torch.from_numpy(_maps(10))
    l1_mse = build_loss("l1+mse")(p, t)
    expected = (torch.nn.functional.l1_loss(p, t)
                + torch.nn.functional.mse_loss(p, t))
    assert l1_mse.item() == pytest.approx(expected.item(), abs=1e-6)

    half = build_loss("0.5*scc")(p, t)
    assert half.item() == pytest.approx(0.5 * scc_loss(p, t).item(), abs=1e-6)

    combo = build_loss("l1+mse+0.3*scc")(p, t)
    assert combo.item() == pytest.approx(
        (expected + 0.3 * scc_loss(p, t)).item(), abs=1e-6)

    with pytest.raises(ValueError):
        build_loss("l1+bogus")


def test_loss_works_under_bf16_autocast():
    if not torch.cuda.is_available():
        pytest.skip("no gpu")
    p = torch.from_numpy(_maps(11)).cuda().requires_grad_(True)
    t = torch.from_numpy(_maps(12)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = p * 1.0
    loss = build_loss("l1+mse+0.5*scc")(out.float(), t)
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_dcorr_weights_strata_equally():
    p, t = torch.from_numpy(_maps(13)), torch.from_numpy(_maps(14))
    ref = _reference_scc(p.numpy(), t.numpy())
    assert scc(p, t).item() == pytest.approx(ref, abs=1e-4)
    assert build_loss("dcorr")(p, t).item() == pytest.approx(
        1.0 - dcorr(p, t).item(), abs=1e-6)
    assert abs(dcorr(p, t).item() - scc(p, t).item()) > 1e-6   # genuinely differ


def test_hicrep_weighting_chases_noise_in_obs_exp_space():
    """In obs/exp space the far strata are the sparsest and so the noisiest.
    HiCRep's n*sx*sy weight scales with their variance, so it hands them the
    most influence; an unweighted mean over strata does not."""
    g = np.random.default_rng(0)
    w, b = 64, 4
    idx = np.arange(w)
    d = np.abs(idx[:, None] - idx[None, :])
    # obs/exp: no decay left in the mean, but variance grows with distance
    noise_scale = (1.0 + 0.25 * d).astype(np.float32)
    signal = np.zeros((b, w, w), np.float32)
    truth = np.zeros((b, w, w), np.float32)
    for k in range(b):
        s = g.normal(size=(w, w)).astype(np.float32)
        s = 0.5 * (s + s.T)
        signal[k] = s
        n = g.normal(size=(w, w)).astype(np.float32) * noise_scale
        truth[k] = s + 0.5 * (n + n.T)          # far strata dominated by noise

    P, T = torch.from_numpy(signal), torch.from_numpy(truth)
    near = 12
    # how much of each metric's total weight sits in the trustworthy near band
    i, j, dd, nb = _strata_index_for_test(w, P.device)
    x, y = P[:, i, j].reshape(-1), T[:, i, j].reshape(-1)
    idxb = dd.repeat(b)
    n_d = torch.zeros(nb).index_add(0, idxb, torch.ones_like(x)).clamp(min=1)
    mx, my = (torch.zeros(nb).index_add(0, idxb, x) / n_d,
              torch.zeros(nb).index_add(0, idxb, y) / n_d)
    vx = (torch.zeros(nb).index_add(0, idxb, x * x) / n_d - mx * mx).clamp(min=0)
    vy = (torch.zeros(nb).index_add(0, idxb, y * y) / n_d - my * my).clamp(min=0)
    wt = n_d * vx.sqrt() * vy.sqrt()
    share_hicrep = (wt[:near].sum() / wt.sum()).item()
    share_uniform = near / nb
    assert share_hicrep < share_uniform      # HiCRep starves the reliable band


def _strata_index_for_test(w, device):
    from hicmultiome.losses import _strata_index
    return _strata_index(w, w, 1, device)


def test_grad_loss_punishes_blur():
    """A blurred version of the target must cost more than the target itself."""
    from scipy.ndimage import gaussian_filter
    from hicmultiome.losses import grad_loss, hf_energy_loss
    t = torch.from_numpy(_maps(20))
    blurred = torch.from_numpy(
        np.stack([gaussian_filter(m, 2.0) for m in t.numpy()]))
    assert grad_loss(t, t).item() < 1e-6
    assert grad_loss(blurred, t).item() > 0.05
    # and the energy term sees the deficit too
    assert hf_energy_loss(blurred, t).item() > hf_energy_loss(t, t).item()


def test_grad_loss_also_punishes_invented_texture():
    """Not just a blur detector: adding noise the target does not have must
    also cost, or the model would be rewarded for hallucinating detail."""
    from hicmultiome.losses import grad_loss
    t = torch.from_numpy(_maps(21))
    g = np.random.default_rng(0)
    noisy = t + torch.from_numpy(g.normal(scale=0.3, size=t.shape).astype("float32"))
    assert grad_loss(noisy, t).item() > 0.05


def test_grad_term_composes_and_is_differentiable():
    from hicmultiome.losses import build_loss
    p = torch.from_numpy(_maps(22)).requires_grad_(True)
    t = torch.from_numpy(_maps(23))
    loss = build_loss("l1+mse+0.5*grad+0.2*hf")(p, t)
    loss.backward()
    assert torch.isfinite(p.grad).all()


def _tad_map(n=64, bnd=(20, 42), seed=0):
    """A map with real TAD structure: contacts suppressed across boundaries."""
    g = np.random.default_rng(seed)
    i = np.arange(n)
    m = np.exp(-np.abs(i[:, None] - i[None, :]) / 12.0)
    dom = np.digitize(i, bnd)
    m = m * np.where(dom[:, None] == dom[None, :], 1.0, 0.25)
    m = m + 0.02 * g.normal(size=(n, n))
    m = 0.5 * (m + m.T)
    return m.astype(np.float32)


def test_insulation_profile_dips_at_a_boundary():
    from hicmultiome.losses import insulation_profile_t
    m = torch.from_numpy(_tad_map()[None])
    prof = insulation_profile_t(m, window=8)[0].numpy()
    # boundaries sit at bins 20 and 42; the profile is offset by `window`
    lo = prof.argmin() + 8
    assert min(abs(lo - 20), abs(lo - 42)) <= 3, lo


def test_insulation_loss_zero_on_identity_and_positive_otherwise():
    from hicmultiome.losses import insulation_loss
    a = torch.from_numpy(_tad_map(seed=1)[None])
    b = torch.from_numpy(_tad_map(bnd=(10, 55), seed=2)[None])
    assert insulation_loss(a, a).item() < 1e-5
    assert insulation_loss(b, a).item() > 0.05


def test_donut_loss_sees_a_focal_peak_that_l1_barely_notices():
    """A sharp loop and a smooth bump of equal mass look similar to L1 but
    must not look similar to the donut term."""
    from scipy.ndimage import gaussian_filter
    from hicmultiome.losses import donut_loss
    base = _tad_map(seed=3)
    focal = base.copy()
    focal[30, 50] += 2.0
    focal[50, 30] += 2.0
    smooth = base + (gaussian_filter(focal - base, 2.0))
    t = torch.from_numpy(focal[None])
    p = torch.from_numpy(smooth[None])
    l1 = torch.nn.functional.l1_loss(p, t).item()
    dn = donut_loss(p, t).item()
    assert dn > l1, (dn, l1)


def test_structure_terms_compose_and_backprop():
    from hicmultiome.losses import build_loss
    p = torch.from_numpy(_tad_map(seed=4)[None]).requires_grad_(True)
    t = torch.from_numpy(_tad_map(seed=5)[None])
    loss = build_loss("l1+mse+0.3*insul+0.3*donut")(p, t)
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
