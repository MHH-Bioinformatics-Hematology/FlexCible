"""Structural variants in the prediction: derivative alleles of the AML patients.

For every SV junction of a patient with CUT&Tag (Xu et al. 2022,
Supplementary Table 5: WGS plus hic_breakfinder; every row is one junction
with its orientation), the local derivative allele is assembled from the
patient's input rows (CUT&Tag tracks, sequence features, motif fwd/rev
planes) and, for reference models, the reference map:

  orientation of (pos1 on chrom1, pos2 on chrom2), NeoLoopFinder convention
  ('+' joins the part upstream of the breakpoint, '-' the part downstream):
    '+-'  L = chrom1[pos1 - h, pos1)          R = chrom2[pos2, pos2 + h)
    '++'  L = chrom1[pos1 - h, pos1)          R = reverse(chrom2[pos2 - h, pos2))
    '--'  L = reverse(chrom1[pos1, pos1 + h)) R = chrom2[pos2, pos2 + h)
    '-+'  L = chrom2[pos2 - h, pos2)          R = chrom1[pos1, pos1 + h)
  so deletions ('+-', one chromosome) join the flanks, tandem duplications
  ('-+', one chromosome) repeat the segment, inversion junctions reverse a
  piece and translocations fuse two chromosomes. A reversed piece has its rows
  reversed and each motif's forward and reverse planes swapped
  (flexcible.dataset.reverse_cond). The reference plane of the derivative
  holds each piece's own reference block (reversed with it) and the expected
  level (log O/E 0) between pieces.

The model predicts the derivative window (h = half the model window on each
side, TTA as crosscell_predict). Contacts are mapped back to reference
coordinates: within a piece the derivative distance equals the reference
distance; between pieces of one chromosome the contact
exp(P_der) E(d_der) is placed at the reference pixel and expressed against
E(d_ref); between chromosomes it forms an inter-chromosomal block. SVs are
mostly heterozygous and the tracks cannot be phased (the CUT&Tag signal is
the sum of both alleles and is used as is on both): the combined prediction
is 0.5 x normal allele + 0.5 x derivative allele in contact space, converted
to log O/E against the reference-coordinate expected E(d_ref); the
derivative-only version is reported as well. E(d) is the patient's own cis
expected (mean balanced contact per diagonal).

Evaluation per junction at 10 kb, on the reference-coordinate window of
WIN bins centred on each breakpoint (patient proper O/E map) and, for
inter-chromosomal junctions, on the balanced inter-chromosomal block
(patient's own ICE-balanced 10 kb map, which holds trans contacts):
Pearson of the naive (SV-unaware) prediction, the SV-aware prediction,
consensus12 and, for the trans block, the distance-only junction model
log E(d_der); contact enrichment across the junction; neo-loops of
Supplementary Table 6 inside a window: donut enrichment at the
gene-enhancer pixel in the measured block and in each prediction.

    python scripts/aml_sv_predict.py --device 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aml_primary as ap                                          # noqa: E402
import aml_primary_explain as ax                                  # noqa: E402

RES = 10000
MODELS = ["all12refres_drop_10kb", "all12honest_drop_10kb"]
OUT = ap.PRIM / "sv"
log = ap.log


def bundle(model: str, device: str, cell: str):
    import hard_split as hs
    return hs.load_bundle(ap.PMODELS[model][1], device, cell, RES)


def patient_rows(b, p: str):
    """sym rows per chromosome for patient p (crosscell_predict assembly)."""
    import cooler
    from flexcible.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
    from flexcible.tracks import TrackNormalizer
    cfg = b["cfg"]
    tn = TrackNormalizer.load(b["ck"] / "norm_tracks.json")
    absent = tn.absent_values()
    sample = cfg.get_sample(None)
    sample.bigwig_dir = str(ap.INP / p)
    have = {f.stem.upper() for f in (ap.INP / p).glob("*.bigwig")}
    missing = [t for t in b["channels"]["tracks"] if t.upper() not in have]
    sample.missing_tracks = list(missing)
    tf = sample_track_files(cfg, sample)
    row_of = {t: i for i, t in enumerate(b["channels"]["tracks"])}
    grid = cooler.Cooler(str(ap.grid_cool(RES)))
    cache = {}

    def get(chrom):
        if chrom not in cache:
            nb = chrom_nbins_of(grid, chrom)
            s = tn.apply(assemble_chrom(cfg, sample, tf, chrom, nb))
            for t in missing:
                s[row_of[t]] = absent[row_of[t]]
            cache[chrom] = s if b["sel"] is None else s[b["sel"]]
        return cache[chrom]
    return get


def pieces(r, h: int):
    """(chrom, start_bin, end_bin, reversed) of L and R for one junction."""
    b1, b2 = int(r.pos1) // RES, int(r.pos2) // RES
    o = str(r.orientation)
    if o == "+-":
        return (r.chrom1, b1 - h, b1, False), (r.chrom2, b2, b2 + h, False)
    if o == "++":
        return (r.chrom1, b1 - h, b1, False), (r.chrom2, b2 - h, b2, True)
    if o == "--":
        return (r.chrom1, b1, b1 + h, True), (r.chrom2, b2, b2 + h, False)
    if o == "-+":
        return (r.chrom2, b2 - h, b2, False), (r.chrom1, b1, b1 + h, False)
    return None


def piece_cols(rows_of, piece, b):
    chrom, s, e, rev = piece
    X = rows_of(chrom)
    if s < 0 or e > X.shape[1]:
        return None
    x = X[:, s:e].copy()
    if rev:
        x = x[:, ::-1].copy()
        off, nm = b["offset"], b["n_motifs"]
        for m in range(nm):
            f = off + 2 * m
            x[[f, f + 1]] = x[[f + 1, f]]
    return x


def piece_ref(cons, piece, b):
    from flexcible.matrix import fetch_diagonal_window
    chrom, s, e, rev = piece
    m = fetch_diagonal_window(cons, chrom, s, e - s)
    if rev:
        m = m[::-1, ::-1]
    return m


def run_model(b, cond_np, ref_np, device):
    import torch
    from flexcible.dataset import reverse_cond, split_cond
    off, nm, size = b["offset"], b["n_motifs"], b["size"]
    cond = torch.from_numpy(np.ascontiguousarray(cond_np)[None]).to(device)
    pairs = cond[:, off:off + 2 * nm].reshape(1, nm, 2, size) if nm else None
    ref = (torch.from_numpy(np.ascontiguousarray(ref_np)[None]).to(device)
           if b["ref"] else None)
    with torch.no_grad():
        o = b["model"](cond, pairs, None, None, ref)
        cr = reverse_cond(cond, off, nm)
        sr, pr = split_cond(cr, off, nm)
        rr = torch.flip(ref, dims=[-1, -2]) if ref is not None else None
        o = 0.5 * (o + torch.flip(b["model"](sr, pr, None, None, rr), dims=[-1, -2]))
    P = b["mat_norm"].decode(o[0, 0].float().cpu().numpy())
    return 0.5 * (P + P.T)


def derivative_prediction(b, rows_of, cons, L, R, device):
    """Predicted log O/E of the derivative window L|R (size x size)."""
    xl, xr = piece_cols(rows_of, L, b), piece_cols(rows_of, R, b)
    if xl is None or xr is None:
        return None
    cond = np.concatenate([xl, xr], 1)
    ref = None
    if b["ref"]:
        h = xl.shape[1]
        enc0 = b["mat_norm"].encode(np.zeros((2 * h, 2 * h), np.float32))
        ref = enc0.copy()
        ref[:h, :h] = b["mat_norm"].encode(piece_ref(cons, L, b))
        ref[h:, h:] = b["mat_norm"].encode(piece_ref(cons, R, b))
    return run_model(b, cond, ref, device)


def normal_prediction(b, rows_of, cons, chrom, centre_bin, device):
    """Naive prediction of the reference window centred on a bin."""
    from flexcible.matrix import fetch_diagonal_window
    X = rows_of(chrom)
    size = b["size"]
    st = max(0, min(X.shape[1] - size, centre_bin - size // 2))
    ref = b["mat_norm"].encode(fetch_diagonal_window(cons, chrom, st, size)) if b["ref"] else None
    return run_model(b, X[:, st:st + size], ref, device), st


def expected_curve(p: str, band: int) -> np.ndarray:
    """Mean balanced cis contact per diagonal (0..band) of the patient's
    ICE-balanced 10 kb map, chr1-22 and X."""
    import cooler
    f = OUT / f"expected_{p}_{RES // 1000}kb.npy"
    if f.exists():
        return np.load(f)
    c = cooler.Cooler(str(ap.WORK / f"{p}_{RES // 1000}kb.cool"))
    s = np.zeros(band + 1)
    n = np.zeros(band + 1)
    for ch in ap.MAIN:
        if ch not in c.chromnames:
            continue
        w = c.bins().fetch(ch)["weight"].values
        good = np.isfinite(w)
        lo = c.offset(ch)
        px = c.pixels().fetch(ch)
        px = px[px.bin2_id < lo + len(w)]
        d = (px.bin2_id - px.bin1_id).values
        keep = d <= band
        b1 = px.bin1_id.values[keep] - lo
        b2 = px.bin2_id.values[keep] - lo
        v = px["count"].values[keep] * w[b1] * w[b2]
        ok = np.isfinite(v)
        np.add.at(s, d[keep][ok], v[ok])
        g = good.astype(float)
        for dd in range(band + 1):
            n[dd] += (g[:len(g) - dd] * g[dd:]).sum() if dd < len(g) else 0
    E = s / np.maximum(n, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(f, E)
    return E


def ref_window(clr, chrom, centre_bin, n):
    """Measured patient log O/E around a bin (n x n, NaN where unmeasured)."""
    from aml_eval import fetch_region
    st = max(0, centre_bin - n // 2)
    M = fetch_region(clr, f"{chrom}:{st * RES}-{(st + n) * RES}", {})
    return M, st


def trans_block(raw, piece_a, piece_b):
    """Balanced contacts between two reference intervals (any chromosomes),
    oriented as the derivative pieces."""
    ca, sa, ea, ra = piece_a
    cb, sb, eb, rb = piece_b
    M = np.asarray(raw.matrix(balance=True).fetch(f"{ca}:{sa * RES}-{ea * RES}",
                                                  f"{cb}:{sb * RES}-{eb * RES}"), float)
    if ra:
        M = M[::-1]
    if rb:
        M = M[:, ::-1]
    return M


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def run_patient(p: str, device: str) -> list[dict]:
    import cooler
    sv = ax.sv_table()
    sv = sv[sv.patient == p]
    if not len(sv):
        return []
    tclr = cooler.Cooler(str(ap.map_path(p, RES)))
    cclr = cooler.Cooler(str(ap.consensus12(RES)))
    raw = cooler.Cooler(str(ap.WORK / f"{p}_{RES // 1000}kb.cool"))
    neo = pd.read_csv(ax.XT / "st6_neoloop_hijacking.csv", dtype={"patient": str})
    neo["patient"] = "xu_" + neo.patient.map(ax.norm_id)
    neo = neo[neo.patient == p]
    rows, saved = [], {}
    for m in MODELS:
        b = bundle(m, device, ap.guard_name(p))
        rows_of = patient_rows(b, p)
        size = b["size"]
        h = size // 2
        E = expected_curve(p, size)
        for r in sv.itertuples():
            pc = pieces(r, h)
            if pc is None:
                continue
            L, R = pc
            try:
                Pd = derivative_prediction(b, rows_of, cclr, L, R, device)
            except Exception as e:                       # noqa: BLE001
                log(f"[skip] {p} {r.Index}: {e}")
                continue
            if Pd is None:
                continue
            rec = {"patient": p, "model": m, "sv_index": int(r.Index), "sv_type": r.sv_type,
                   "chrom1": r.chrom1, "pos1": int(r.pos1), "chrom2": r.chrom2,
                   "pos2": int(r.pos2), "orientation": r.orientation}
            d = np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
            Ed = E[np.minimum(d, len(E) - 1)]
            # cross-junction block (L rows x R columns) in the derivative
            cross_der = Pd[:h, h:]
            rec["pred_der_cross_mean_logoe"] = float(np.nanmean(cross_der))
            # measured junction block, oriented as the derivative pieces
            M = trans_block(raw, L, R)
            if M.shape == (h, h):
                Mexp = np.log(np.where(M > 0, M, np.nan) / Ed[:h, h:])
                rec["meas_cross_mean_logoe"] = float(np.nanmean(Mexp))
                rec["r_cross|sv_aware"] = corr(cross_der.ravel(), Mexp.ravel())
                rec["r_cross|distance_only"] = corr(np.log(Ed[:h, h:]).ravel(),
                                                    np.log(np.where(M > 0, M, np.nan)).ravel())
                rec["r_cross_counts|sv_aware"] = corr((cross_der + np.log(Ed[:h, h:])).ravel(),
                                                      np.log(np.where(M > 0, M, np.nan)).ravel())
            # cis window around breakpoint 1 (reference coordinates)
            c1 = int(r.pos1) // RES
            Tm, st = ref_window(tclr, r.chrom1, c1, size)
            Pn, stn = normal_prediction(b, rows_of, cclr, r.chrom1, c1, device)
            if Tm is not None and Tm.shape == (size, size) and stn == st:
                # SV-aware cis window: pixels of the joined piece mixed 0.5/0.5
                comb = Pn.copy()
                piece1 = L if L[0] == r.chrom1 and L[1] <= c1 <= L[2] else R
                ch, s0, e0, rev = piece1
                idx_ref = np.arange(s0, e0) - st
                idx_der = (np.arange(h) if piece1 is L else np.arange(h, 2 * h))
                if rev:
                    idx_der = idx_der[::-1]
                ok = (idx_ref >= 0) & (idx_ref < size)
                ir, idd = idx_ref[ok], idx_der[ok]
                blk_n = Pn[np.ix_(ir, ir)]
                blk_d = Pd[np.ix_(idd, idd)]
                comb[np.ix_(ir, ir)] = np.log(0.5 * np.exp(blk_n) + 0.5 * np.exp(blk_d))
                C = ref_window(cclr, r.chrom1, c1, size)[0]
                tri = np.triu(np.ones((size, size), bool), 2)
                mk = tri & np.isfinite(Tm)
                if C is not None and C.shape == Tm.shape:
                    mk &= np.isfinite(C)
                    rec["r_cis|consensus12"] = corr(C[mk], Tm[mk])
                rec["r_cis|naive"] = corr(Pn[mk], Tm[mk])
                rec["r_cis|sv_aware"] = corr(comb[mk], Tm[mk])
                saved[f"{m}|{r.Index}"] = {"Pd": Pd.astype(np.float32), "Pn": Pn.astype(np.float32),
                                           "comb": comb.astype(np.float32), "T": Tm.astype(np.float32),
                                           "M": M.astype(np.float32) if M.shape == (h, h) else np.zeros(1)}
            # neo-loops of Suppl. Table 6 whose gene and enhancer fall in L and R
            hits = []
            for x in neo.itertuples():
                gch, gse = str(x.gene_region).split(":")
                ech, ese = str(x.enhancer).split(":")
                g0 = int(gse.split("-")[0]) // RES
                e0_ = int(ese.split("-")[0]) // RES

                def pos_in(piece, ch_, bin_):
                    c_, s_, e_, rv = piece
                    if c_ != ch_ or not (s_ <= bin_ < e_):
                        return None
                    k = bin_ - s_
                    return (e_ - s_ - 1 - k) if rv else k
                for A_, B_ in ((L, R), (R, L)):
                    ia, ib = pos_in(A_, gch, g0), pos_in(B_, ech, e0_)
                    if ia is not None and ib is not None:
                        ia = ia if A_ is L else ia + h
                        ib = ib if B_ is L else ib + h
                        hits.append((min(ia, ib), max(ia, ib), x.gene))
            if hits:
                from flexcible.bioscore import local_enrichment
                from aml_eval import loop_params
                prm = loop_params(RES)
                eP = local_enrichment(np.exp(Pd), prm)
                full = np.full((size, size), np.nan)
                if M.shape == (h, h):
                    Mo = np.log(np.where(M > 0, M, np.nan) / Ed[:h, h:])
                    full[:h, h:] = Mo
                    full[h:, :h] = Mo.T
                eM = local_enrichment(np.exp(np.nan_to_num(full)), prm)
                vals_p = [eP[i, j] for i, j, _ in hits]
                vals_m = [eM[i, j] for i, j, _ in hits]
                rec["neoloops_n"] = len(hits)
                rec["neoloops_genes"] = ";".join(sorted({g for *_, g in hits}))
                rec["neoloop_pred_enrich_median"] = float(np.nanmedian(np.log2(vals_p)))
                rec["neoloop_meas_enrich_median"] = float(np.nanmedian(np.log2(vals_m)))
            rows.append(rec)
        del b
    if saved:
        np.savez_compressed(OUT / f"windows_{p}.npz", **{k_ + "|" + kk: v for k_, d_ in saved.items()
                                                         for kk, v in d_.items()})
    return rows


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--device", default="0")
    a.add_argument("--patients", nargs="*", default=None)
    a = a.parse_args()
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.device)
    dev = "cuda" if a.device != "cpu" else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    pats = a.patients or ax.target_patients()
    rows = []
    for p in pats:
        f = OUT / f"junctions_{p}.csv"
        if f.exists():
            rows.append(pd.read_csv(f))
            continue
        d = pd.DataFrame(run_patient(p, dev))
        d.to_csv(f, index=False)
        rows.append(d)
        log(f"[sv] {p}: {len(d)} junction x model rows")
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(ap.RES_DIR / "sv_junctions.csv", index=False)
    cols = [c for c in D.columns if c.startswith(("r_", "pred_", "meas_", "neoloop_"))]
    S = D.groupby(["model", "sv_type"])[cols].median().reset_index()
    S.insert(2, "n", D.groupby(["model", "sv_type"]).size().values)
    S.to_csv(ap.RES_DIR / "sv_summary.csv", index=False)
    log("[sv] wrote sv_junctions.csv, sv_summary.csv")


if __name__ == "__main__":
    main()
