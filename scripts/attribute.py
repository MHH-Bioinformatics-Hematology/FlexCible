"""Per-locus and global attribution for the one model all12one at 10 kb (integrated gradients).

Question: which input tracks, and which positions along the window, carry a predicted loop and a
predicted domain border, with and without the consensus map as input? Method: integrated gradients
(Sundararajan, Taly, Yan, ICML 2017) of a scalar of the predicted map with respect to the model
inputs, 50 midpoint Riemann steps along the straight path from a baseline to the input.

Model and cell: runs/multicell_reference/all12one_10kb/checkpoint-best, held-out K562 with all 13
marks, 192-bin windows (1.92 Mb) centred on the Akita test regions as in crosscell_predict.py, no
test-time augmentation. "With consensus" gives the reference plane the 12-cell consensus (encoded
as in the dataset), "without consensus" sets that plane to 0 (the expected level, the keep sets
named *_noref of crosscell_predict.py).

Inputs attributed (HiCModel.forward): the 23 per-bin rows `cond` (13 marks, GC, CpG, fwd and rev
hit tracks of four motifs, all z-scored) and the 2D reference plane. The pairing of the rows into
the mean, product and |diff| planes and the directed motif planes happens INSIDE the model
(build_spatial_cond), so the gradients are taken with respect to the 1D rows before pairing; the
motif strand pairs are a differentiable view of the same `cond` rows, so a motif row collects its
effect through both the symmetric planes and the directed planes. No 2D back-summation is needed.
Baselines (what the model sees for a missing signal):
  marks            TrackNormalizer.absent_values(), the z-score of a raw-zero track (the encoding
                   dropout and absent assays use)
  motif rows       the same absent value (raw zero = no motif hit)
  GC, CpG          0 in z units (the genome-average composition; a raw zero base content is not a
                   state the model has seen)
  consensus plane  0 (encoded space, the expected level)
Without consensus the reference plane is 0 at the input and at the baseline, so it is not attributed.

Targets (scalars of the decoded prediction P = output * scale, log obs/exp, symmetric):
  loop    sum of P over the 3x3 box at the strongest measured loop pixel of the window
  border  minus the mean of P over [b-5, b) x (b, b+5] at the strongest measured border b (insulation
          square of the paper, negated so a stronger border is positive)
  global  sum of P over the upper triangle (k = 2), the whole predicted map
Measured loop pixel: maximum over pixels at 100 to 600 kb of (mean of the 3x3 box) minus (mean of the
donut, 11x11 without the central 5x5) of the measured log O/E map. Measured border: maximum over bins
of 0.5 * (mean of the two 5x5 intra-domain squares) minus the cross square mean. Both are located on
the measured map of the Akita region and mapped to the model window by absolute bin index.

Completeness (sum of attributions = f(x) - f(baseline)) is recorded for every target and setting.

Stages:
  python scripts/attribute.py select                    # candidate regions from results/flexible_inputs
  python scripts/attribute.py locus --regions R1 R2 R3  # ig_locus_<region>.npz
  python scripts/attribute.py glob --n 40               # ig_global.csv, 40 random test regions
  python scripts/attribute.py report                    # tables.md (completeness, coordinates, enrichments)
Environment as in flexible_inputs.py: HICMULTI_HM (data root), FLEX_REGIONS (regions TSV) for hosts
without the share. Outputs in results/attribution/.

Other checkpoints (release candidate with methylation and the cohesin-fraction row):
  python scripts/attribute.py locus --model all12one_fullrd --meth 1 --regions ...  (also glob, report)
--model names the run whose 10 kb view runs/multicell_reference/<model>_10kb is used; --meth 1 gives
K562 its methylation track "Meth" (an attributed row, baseline = absent value), --meth 0 sets Meth absent
in the input (its row is held at the absent value, attribution 0). The cohesin:fraction row is a context
row (K562 intact, 1.0): it is held at its input value in the baseline, so it receives no attribution and the
path changes only the tracks. Outputs: results/attribution/<model>/ (--meth 1) and
results/attribution/<model>/nometh/ (--meth 0); the default model all12one writes results/attribution/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import score_like_hicgan as sc                                           # noqa: E402
from flexcible.config import Config                                      # noqa: E402
from flexcible.genome import match_name                                  # noqa: E402
from flexcible.matrix import MatrixNormalizer                            # noqa: E402
from flexcible.precompute import (assemble_chrom, chrom_nbins_of,        # noqa: E402
                                  sample_track_files)
from flexcible.reference import open_reference, reference_block          # noqa: E402
from flexcible.tracks import TrackNormalizer                             # noqa: E402

HM = Path(os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome"))
OUT = REPO / "results/attribution"
CELL, KB = "k562", 10
RES = KB * 1000
CKPT = HM / f"runs/multicell_reference/all12one_{KB}kb/checkpoint-best"
BIGWIG = HM / f"hg38_{CELL}"
TARGET = HM / f"crosscell_oe/{CELL}_proper_{KB}kb.cool"
REFERENCE = HM / f"crosscell_oe/consensus12_proper_{KB}kb.cool"
STEPS = 50
SETTINGS = ("cons", "nocons")
INS = 5                       # insulation square side, bins
GC_ROWS = ("seq:gc", "seq:cpg")
FLEX = REPO / "results/flexible_inputs/per_region.csv"


MODEL, METH = "all12one", True
HELD_ROWS = ("cohesin:fraction",)       # context rows: baseline = input (no attribution)


def configure(model, meth):
    global MODEL, METH, OUT, CKPT
    MODEL, METH = model, bool(meth)
    CKPT = HM / f"runs/multicell_reference/{model}_{KB}kb/checkpoint-best"
    if model != "all12one":
        OUT = REPO / "results/attribution" / model / ("" if METH else "nometh")


def parse_region(s):
    c, r = s.split(":")
    a, b = r.split("-")
    return c, int(a), int(b)


def regions_list():
    p = os.environ.get("FLEX_REGIONS")
    path = Path(p) if p else sc.REGIONS
    if path.exists():
        d = pd.read_csv(path, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
        return [f"{r.chrom}:{r.start}-{r.end}" for r in d.itertuples()]
    d = pd.read_csv(FLEX)
    return sorted(d[d.cell == CELL].region.unique())


# --------------------------------------------------------------------------- #
# measured-map measures used to place the targets
# --------------------------------------------------------------------------- #
def box_mean(A, k):
    """Mean of the valid (finite) values in the k x k box centred on every pixel (NaN edges)."""
    from scipy.ndimage import uniform_filter
    v = np.isfinite(A).astype(float)
    s = uniform_filter(np.where(v > 0, A, 0.0), k, mode="constant") * k * k
    n = uniform_filter(v, k, mode="constant") * k * k
    return np.where(n > 0.5 * k * k, s / np.maximum(n, 1e-9), np.nan)


def loop_map(T, dmin=10, dmax=60, margin=6):
    """Loop strength of every pixel: 3x3 box mean minus donut mean (11x11 without the central 5x5)."""
    n = T.shape[0]
    inner = box_mean(T, 3)
    s11 = box_mean(T, 11) * 121
    s5 = box_mean(T, 5) * 25
    donut = (s11 - s5) / 96.0
    S = inner - donut
    ii, jj = np.indices(T.shape)
    d = jj - ii
    ok = (d >= dmin) & (d <= dmax) & (ii >= margin) & (jj < n - margin)
    return np.where(ok, S, np.nan)


def border_scores(T, margin=INS + 1):
    """Border strength per bin: mean of the two intra-domain squares minus the cross square."""
    n = T.shape[0]
    s = np.full(n, np.nan)
    for b in range(margin, n - margin):
        cross = np.nanmean(T[b - INS:b, b + 1:b + 1 + INS])
        left = np.nanmean(T[b - INS:b, b - INS:b])
        right = np.nanmean(T[b + 1:b + 1 + INS, b + 1:b + 1 + INS])
        s[b] = 0.5 * (left + right) - cross
    return s


def measured_region(tgt, chrom, start, end):
    name = match_name(chrom, list(tgt.chromnames))
    T = np.asarray(tgt.matrix(balance=False).fetch(f"{name}:{start}-{end}"), float)
    return np.where(T == 0, np.nan, T)


def select(a):
    """Candidate regions: K562, all 13 marks with consensus, r > 0.9; loop and border strength."""
    d = pd.read_csv(FLEX)
    d = d[(d.cell == CELL) & (d.res == KB) & (d.set_name == "all13") & (d.with_reference == 1)]
    d = d[d.r > 0.9]
    tgt = cooler.Cooler(str(TARGET))
    rows = []
    for r in d.itertuples():
        c, s, e = parse_region(r.region)
        T = measured_region(tgt, c, s, e)
        L = loop_map(T)
        if not np.isfinite(L).any():
            continue
        i, j = np.unravel_index(np.nanargmax(L), L.shape)
        B = border_scores(T)
        b = int(np.nanargmax(B))
        rows.append({"region": r.region, "r": r.r, "loop_score": L[i, j], "loop_i": i, "loop_j": j,
                     "loop_span_kb": (j - i) * KB, "border_score": B[b], "border_bin": b})
    t = pd.DataFrame(rows).sort_values("loop_score", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(OUT / "candidates.csv", index=False)
    print(t.head(25).round(3).to_string())
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = t.head(a.n)
        cols = 6
        fig, axs = plt.subplots(-(-len(top) // cols), cols, figsize=(3 * cols, 3 * -(-len(top) // cols)))
        for ax, r in zip(axs.ravel(), top.itertuples()):
            c, s, e = parse_region(r.region)
            T = measured_region(tgt, c, s, e)
            ax.imshow(T, cmap="RdYlBu_r", vmin=-0.8, vmax=0.8)
            ax.plot(r.loop_j, r.loop_i, "ks", mfc="none", ms=7)
            ax.axvline(r.border_bin, color="k", lw=0.5, ls=":")
            ax.axhline(r.border_bin, color="k", lw=0.5, ls=":")
            ax.set_title(f"{r.region}\nr={r.r:.2f} L={r.loop_score:.2f} B={r.border_score:.2f}", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(a.png, dpi=110)


# --------------------------------------------------------------------------- #
# model, windows, integrated gradients
# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self, device):
        self.device = device
        cfg = Config.load(str(CKPT / "config.yaml"))
        self.cfg = cfg
        self.model, ch = sc.load_model(cfg, CKPT, device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.channels = ch
        self.size = cfg.window.size
        self.res = cfg.data.resolution
        assert self.res == RES
        self.rows = list(ch["sym_rows"])
        self.n_motifs = len(ch["motif_ids"])
        self.offset = len(ch["tracks"]) + len(ch["seq_features"])
        self.mat_norm = MatrixNormalizer.load(CKPT / f"norm_matrix_{cfg.get_sample(None).name}.json")
        self.scale = float(self.mat_norm.scale)
        assert self.mat_norm.transform == "prescaled"
        self.track_norm = TrackNormalizer.load(CKPT / "norm_tracks.json")
        self.absent = self.track_norm.absent_values()
        sample = cfg.get_sample(None)
        sample.bigwig_dir = str(BIGWIG)
        has_meth_row = "Meth" in ch["tracks"]
        sample.extra_bigwig_dirs = [str(HM / "meth" / CELL)] if (has_meth_row and METH) else []
        sample.missing_tracks = ["Meth"] if (has_meth_row and not METH) else []
        self.held = [k for k, nm in enumerate(self.rows)
                     if nm in HELD_ROWS or (nm == "Meth" and not METH)]
        self.sample = sample
        self.files = sample_track_files(cfg, sample)
        self.tgt = cooler.Cooler(str(TARGET))
        self.ref_clr = open_reference(str(REFERENCE), self.res)
        self._chrom = {}
        # baseline of the 1D rows
        b = self.absent.astype(np.float32).copy()
        for k, nm in enumerate(self.rows):
            if nm in GC_ROWS:
                b[k] = 0.0
        self.baseline_row = b

    def chrom_rows(self, chrom):
        if chrom not in self._chrom:
            sname = match_name(chrom, list(self.tgt.chromnames))
            nb = chrom_nbins_of(self.tgt, sname)
            self._chrom[chrom] = self.track_norm.apply(
                assemble_chrom(self.cfg, self.sample, self.files, sname, nb))
        return self._chrom[chrom]

    def window(self, region):
        chrom, s, e = parse_region(region)
        sym = self.chrom_rows(chrom)
        nb = sym.shape[1]
        centre = (s + e) // 2 // self.res
        st = max(0, min(nb - self.size, centre - self.size // 2))
        cond = sym[:, st:st + self.size].astype(np.float32).copy()
        ref = reference_block(self.ref_clr, chrom, st, self.size, self.mat_norm).astype(np.float32)
        T = measured_region(self.tgt, chrom, s, e)
        off = s // self.res - st            # window bin of the first row of T
        Tw = np.full((self.size, self.size), np.nan)
        n = T.shape[0]
        lo, hi = max(0, -off), min(n, self.size - off)
        Tw[off + lo:off + hi, off + lo:off + hi] = T[lo:hi, lo:hi]
        return dict(region=region, chrom=chrom, start=s, end=e, st=st, cond=cond, ref=ref,
                    T=T, Tw=Tw, off=off)

    def forward(self, cond, ref):
        pairs = cond[:, self.offset:self.offset + 2 * self.n_motifs].reshape(
            cond.shape[0], self.n_motifs, 2, cond.shape[-1])
        out = self.model(cond, pairs, None, None, ref)
        P = out[:, 0] * self.scale
        return 0.5 * (P + P.transpose(-1, -2))

    def predict(self, w, consensus=True):
        cond = torch.from_numpy(w["cond"][None]).to(self.device)
        ref = torch.from_numpy(w["ref"][None]).to(self.device)
        if not consensus:
            ref = torch.zeros_like(ref)
        with torch.no_grad():
            return self.forward(cond, ref)[0].float().cpu().numpy()


def target_fns(loop=None, border=None, size=192):
    """name -> function of a batch of decoded maps (B, W, W) giving (B,) scalars."""
    fns = {}
    if loop is not None:
        i, j = loop
        fns["loop"] = lambda P: P[:, i - 1:i + 2, j - 1:j + 2].sum(dim=(1, 2))
    if border is not None:
        b = border
        fns["border"] = lambda P: -P[:, b - INS:b, b + 1:b + 1 + INS].mean(dim=(1, 2))
    return fns


def upper_sum_fn(size):
    iu = torch.triu_indices(size, size, offset=2)
    return {"global": lambda P: P[:, iu[0], iu[1]].sum(dim=1)}


def integrated_gradients(ctx, w, fns, consensus, steps=STEPS, chunk=10):
    """IG of every target in `fns` for one window. Returns {target: dict(cond, ref, fx, fb, total)}."""
    dev = ctx.device
    x = torch.from_numpy(w["cond"]).to(dev)
    b = torch.from_numpy(ctx.baseline_row[:, None].repeat(x.shape[1], 1)).to(dev)
    for k in ctx.held:                          # context rows: baseline = input
        b[k] = x[k]
    r = torch.from_numpy(w["ref"]).to(dev) if consensus else torch.zeros(x.shape[1], x.shape[1], device=dev)
    acc = {k: [torch.zeros_like(x), torch.zeros_like(r)] for k in fns}
    alphas = (torch.arange(steps, device=dev, dtype=torch.float32) + 0.5) / steps
    for c0 in range(0, steps, chunk):
        al = alphas[c0:c0 + chunk]
        ck = (b[None] + al[:, None, None] * (x - b)[None]).requires_grad_(True)
        rk = (al[:, None, None] * r[None]).requires_grad_(consensus)
        P = ctx.forward(ck, rk)
        for name, fn in fns.items():
            f = fn(P).sum()
            ins = [ck] + ([rk] if consensus else [])
            g = torch.autograd.grad(f, ins, retain_graph=True)
            acc[name][0] += g[0].sum(0)
            if consensus:
                acc[name][1] += g[1].sum(0)
    res = {}
    with torch.no_grad():
        Px = ctx.forward(x[None], r[None])
        Pb = ctx.forward(b[None], torch.zeros_like(r)[None])
    for name, fn in fns.items():
        a_c = (acc[name][0] / steps * (x - b)).cpu().numpy()
        a_r = (acc[name][1] / steps * r).cpu().numpy()
        fx, fb = float(fn(Px)[0]), float(fn(Pb)[0])
        res[name] = dict(cond=a_c, ref=a_r, fx=fx, fb=fb, total=float(a_c.sum() + a_r.sum()))
    return res, Px[0].cpu().numpy(), Pb[0].cpu().numpy()


def locate(w):
    """Strongest measured loop pixel and border in WINDOW bins."""
    T = w["T"]
    L = loop_map(T)
    li, lj = np.unravel_index(np.nanargmax(L), L.shape)
    B = border_scores(T)
    b = int(np.nanargmax(B))
    off = w["off"]
    return dict(loop=(int(li) + off, int(lj) + off), loop_score=float(L[li, lj]),
                border=b + off, border_score=float(B[b]))


def locus(a):
    ctx = Ctx(a.device)
    steps = a.steps
    out_dir = Path(a.out) if a.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for region in a.regions:
        w = ctx.window(region)
        loc = locate(w)
        fns = target_fns(loc["loop"], loc["border"])
        arrs = {}
        meta = dict(region=region, cell=CELL, kb=KB, st=w["st"], window_start_bp=w["st"] * ctx.res,
                    window_bins=ctx.size, region_offset_bins=w["off"], steps=steps,
                    loop_i=loc["loop"][0], loop_j=loc["loop"][1], loop_score=loc["loop_score"],
                    border_bin=loc["border"], border_score=loc["border_score"], complete={})
        for cons in (True, False):
            tag = "cons" if cons else "nocons"
            res, Px, Pb = integrated_gradients(ctx, w, fns, cons, steps)
            arrs[f"pred_{tag}"] = Px.astype(np.float32)
            arrs[f"pred_baseline_{tag}"] = Pb.astype(np.float32)
            for tname, d in res.items():
                arrs[f"ig_{tname}_{tag}"] = d["cond"].astype(np.float32)
                if cons:
                    arrs[f"ig_{tname}_consensus2d"] = d["ref"].astype(np.float32)
                delta = d["fx"] - d["fb"]
                meta["complete"][f"{tname}_{tag}"] = dict(
                    f_x=d["fx"], f_baseline=d["fb"], delta=delta, sum_attr=d["total"],
                    rel_err=abs(d["total"] - delta) / max(abs(delta), 1e-9))
                print(f"{region} {tname:6s} {tag:6s} f(x)={d['fx']:.4f} f(b)={d['fb']:.4f} "
                      f"sumIG={d['total']:.4f} relerr={meta['complete'][f'{tname}_{tag}']['rel_err']:.3f}",
                      flush=True)
        name = region.replace(":", "_").replace("-", "_")
        np.savez_compressed(out_dir / f"ig_locus_{name}.npz", rows=np.array(ctx.rows), cond=w["cond"],
                            consensus=w["ref"], measured=w["Tw"].astype(np.float32),
                            meta=np.array(json.dumps(meta)), **arrs)


def glob(a):
    ctx = Ctx(a.device)
    steps = a.steps
    out_dir = Path(a.out) if a.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    regs = regions_list()
    rng = np.random.default_rng(a.seed)
    pick = []
    for idx in rng.permutation(len(regs)):
        try:
            w = ctx.window(regs[idx])
        except Exception:
            continue
        if np.isfinite(w["T"]).mean() < 0.5:
            continue
        pick.append(regs[idx])
        if len(pick) == a.n:
            break
    fns = upper_sum_fn(ctx.size)
    rows = ctx.rows + ["consensus"]
    raw = {s: [] for s in SETTINGS}
    comp = []
    for region in pick:
        w = ctx.window(region)
        for cons in (True, False):
            tag = "cons" if cons else "nocons"
            res, _, _ = integrated_gradients(ctx, w, fns, cons, steps)
            d = res["global"]
            v = np.abs(d["cond"]).sum(1)
            v = np.append(v, np.abs(d["ref"]).sum())
            raw[tag].append(v)
            dl = d["fx"] - d["fb"]
            comp.append({"region": region, "setting": tag, "f_x": d["fx"], "f_baseline": d["fb"],
                         "sum_attr": d["total"], "rel_err": abs(d["total"] - dl) / max(abs(dl), 1e-9)})
            print(f"{region} {tag} sumIG={d['total']:.2f} delta={d['fx'] - d['fb']:.2f}", flush=True)
    recs = []
    for tag in SETTINGS:
        M = np.array(raw[tag])                         # (regions, rows+1)
        if tag == "nocons":
            M[:, -1] = 0.0
        share = M / M.sum(1, keepdims=True)
        for k, tr in enumerate(rows):
            if tag == "nocons" and tr == "consensus":
                continue
            rec = {"track": tr, "setting": tag, "mean_abs_ig": M[:, k].mean(),
                   "mean_share": share[:, k].mean(), "sd_share": share[:, k].std()}
            rec.update({f"share:{r}": share[i, k] for i, r in enumerate(pick)})
            recs.append(rec)
    df = pd.DataFrame(recs)
    df.round(6).to_csv(out_dir / "ig_global.csv", index=False)
    pd.DataFrame(comp).round(5).to_csv(out_dir / "ig_global_completeness.csv", index=False)
    print(df[["track", "setting", "mean_share"]].pivot(index="track", columns="setting", values="mean_share")
          .sort_values("cons", ascending=False).round(4).to_string())


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
GROUPS = {"cohesin-CTCF": ["CTCF", "RAD21", "SMC3"],
          "active": ["H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac"],
          "repressive": ["H3K27me3", "H3K9me3"]}


def mass_share(A, rows, names, sel):
    """Fraction of the absolute attribution of the rows `names` that lies in the bin mask `sel`."""
    idx = [rows.index(n) for n in names]
    m = np.abs(A[idx]).sum(0)
    return float(m[sel].sum() / max(m.sum(), 1e-12))


def report(a):
    out = ["# Attribution tables (generated by scripts/attribute.py report)\n"]
    comp, coords, enr = [], [], []
    for f in sorted(OUT.glob("ig_locus_*.npz")):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        rows = [str(x) for x in z["rows"]]
        W = meta["window_bins"]
        reg = meta["region"]
        i, j, b = meta["loop_i"], meta["loop_j"], meta["border_bin"]
        st = meta["window_start_bp"]
        coords.append(f"| {reg} | {i} / {j} | {st + i * RES:,} / {st + j * RES:,} | {(j - i) * KB} | "
                      f"{meta['loop_score']:.2f} | {b} | {st + b * RES:,} | {meta['border_score']:.2f} |")
        for k, v in meta["complete"].items():
            comp.append(f"| {reg} | {k} | {v['f_x']:.3f} | {v['f_baseline']:.3f} | {v['delta']:.3f} | "
                        f"{v['sum_attr']:.3f} | {100 * v['rel_err']:.1f} |")
        pos = np.arange(W)
        near_anchor = (np.abs(pos - i) <= 2) | (np.abs(pos - j) <= 2)
        inside = (pos > i + 2) & (pos < j - 2)
        near_b = np.abs(pos - b) <= 2
        for tag in SETTINGS:
            A = z[f"ig_loop_{tag}"]
            for g in ("cohesin-CTCF", "active", "repressive"):
                enr.append(f"| {reg} | loop | {tag} | {g} | "
                           f"{100 * mass_share(A, rows, GROUPS[g], near_anchor):.1f} "
                           f"({100 * near_anchor.mean():.1f}) | "
                           f"{100 * mass_share(A, rows, GROUPS[g], inside):.1f} ({100 * inside.mean():.1f}) | |")
            B = z[f"ig_border_{tag}"]
            for g in ("cohesin-CTCF", "active", "repressive"):
                enr.append(f"| {reg} | border | {tag} | {g} | "
                           f"{100 * mass_share(B, rows, GROUPS[g], near_b):.1f} ({100 * near_b.mean():.1f}) "
                           f"| | |")
    out += ["## Chosen windows\n",
            "| region | loop bins i / j | loop bp i / j | span kb | measured loop score | border bin | border bp | measured border score |",
            "|---|---|---|---|---|---|---|---|", *coords, "",
            "## Completeness (sum of attributions against f(x) - f(baseline), 50 steps)\n",
            "| region | target_setting | f(x) | f(baseline) | f(x) - f(b) | sum IG | relative error % |",
            "|---|---|---|---|---|---|---|", *comp, "",
            "## Share of the absolute attribution of a mark group near the anchors (percent; in parentheses the share of bins)\n",
            "Loop: columns near anchors (within 2 bins of either anchor) and inside the loop (between the anchors, "
            "2 bins excluded). Border: near the border (within 2 bins) in the first of the two columns.\n",
            "| region | target | setting | group | near anchors / border | inside loop | |", "|---|---|---|---|---|---|---|", *enr]
    (OUT / "tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="stage", required=True)
    s = sp.add_parser("select")
    s.add_argument("--png", default=None)
    s.add_argument("--n", type=int, default=24)
    s.set_defaults(f=select)
    s = sp.add_parser("locus")
    s.add_argument("--regions", nargs="+", required=True)
    s.add_argument("--steps", type=int, default=STEPS)
    s.add_argument("--out", default=None, help="output folder (default results/attribution)")
    s.add_argument("--device", default="cuda")
    s.add_argument("--model", default="all12one")
    s.add_argument("--meth", type=int, choices=[0, 1], default=1)
    s.set_defaults(f=locus)
    s = sp.add_parser("glob")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, default=STEPS)
    s.add_argument("--out", default=None, help="output folder (default results/attribution)")
    s.add_argument("--device", default="cuda")
    s.add_argument("--model", default="all12one")
    s.add_argument("--meth", type=int, choices=[0, 1], default=1)
    s.set_defaults(f=glob)
    s = sp.add_parser("report")
    s.add_argument("--model", default="all12one")
    s.add_argument("--meth", type=int, choices=[0, 1], default=1)
    s.set_defaults(f=report)
    a = ap.parse_args()
    configure(getattr(a, "model", "all12one"), getattr(a, "meth", 1))
    a.f(a)


if __name__ == "__main__":
    main()
