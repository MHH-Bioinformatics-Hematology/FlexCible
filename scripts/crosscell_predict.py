"""Predict a cell's Hi-C from its own ChIP and score against its real map.

Runs the (dropout-trained) model on one cell's ChIP bigwigs, predicts each test
region at 2 kb, and scores against that cell's measured O/E cool on the SAME 2 kb
grid (no resampling). Missing marks (e.g. HMEC/HUVEC have no cohesin) are filled
with the absent value, so the dropout model predicts from whatever subset exists.
Also scores every OTHER cell's real map as a prediction for this cell (the
measured-map baseline), so model-vs-baseline is on identical regions/pixels.

    python scripts/crosscell_predict.py --config configs/akita_native_trackdrop.yaml \
        --checkpoint runs/akita_native_trackdrop/checkpoint-best \
        --cell imr90 --bigwig-dir /mnt/processed/wolffjoa/hic-multiome/hg38_imr90 \
        --target crosscell_oe/imr90_2kb_oe.cool \
        --others gm12878=.../gm12878_2kb_oe.cool k562=.../k562_2kb_oe.cool
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_like_hicgan as sc
from flexcible import motifmeth as mm
from flexcible.config import Config
from flexcible.genome import match_name
from flexcible.matrix import MatrixNormalizer, open_cool
from flexcible.reference import assert_reference_excludes
from flexcible.precompute import assemble_chrom, chrom_nbins_of, sample_track_files
from flexcible.tracks import TrackNormalizer


def coarse_score(M, start, end, fine_res, cres, cclr, chrom):
    """Pearson of the fine prediction M (region grid, fine_res bins starting at
    `start`), averaged into the bins of the coarser cooler `cclr`, against that
    cooler's values on the same region. None if too few bins or pixels."""
    name = match_name(chrom, list(cclr.chromnames))
    if name is None:
        return None
    c0 = (start // cres) * cres
    c1 = -(-end // cres) * cres
    try:
        T = np.asarray(cclr.matrix(balance=False).fetch(f"{name}:{c0}-{c1}"), float)
    except Exception:
        return None
    m = T.shape[0]
    if m < 20:
        return None
    centres = start + (np.arange(M.shape[0]) + 0.5) * fine_res
    idx = ((centres - c0) // cres).astype(int)
    keep = (idx >= 0) & (idx < m)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    ok = np.isfinite(M) & keep[:, None] & keep[None, :]
    s = np.zeros((m, m)); n = np.zeros((m, m))
    np.add.at(s, (ii[ok], jj[ok]), M[ok])
    np.add.at(n, (ii[ok], jj[ok]), 1.0)
    P = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    iu = np.triu_indices(m, k=2)
    x, y = P[iu], np.where(T == 0, np.nan, T)[iu]
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 100 or x[good].std() == 0 or y[good].std() == 0:
        return None
    return float(pearsonr(x[good], y[good]).statistic)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--bigwig-dir", required=True)
    ap.add_argument("--extra-bigwig-dir", nargs="*", default=[],
                    help="further folders of this cell's track bigWigs (e.g. its methylation "
                         "folder), searched after --bigwig-dir")
    ap.add_argument("--target", required=True, help="this cell's measured O/E cool (2kb)")
    ap.add_argument("--others", nargs="*", default=[], help="cell=cool of OTHER cells")
    ap.add_argument("--binsize", type=int, default=2000)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--assay", choices=["hic", "intact", "microc"], default=None,
                    help="contact assay of this cell's map, for models trained "
                         "with data.assay_indicator (default: keep the config's)")
    ap.add_argument("--lab", default=None,
                    help="models trained with data.lab_classes: the laboratory / protocol "
                         "class of this cell's map (default: the config sample's, i.e. none)")
    ap.add_argument("--coarse", nargs="*", default=[],
                    help="res=cool: also block-average the native prediction onto "
                         "that coarser grid and score it against that target")
    ap.add_argument("--regions", default=None,
                    help="chrom/start/end TSV of regions to score (default: the "
                         "413 Akita test regions)")
    ap.add_argument("--reference", default=None,
                    help="reference-map models (model.reference): the reference O/E "
                         "cool at the model resolution (default: the cell's own "
                         "training reference, else the map embedded in the "
                         "checkpoint, else data.reference_cool). It must not "
                         "contain this cell's map")
    ap.add_argument("--prior", default=None,
                    help="cascade models (model.prior): the coarser model's prediction "
                         "for this cell (default: the cell's own samples[].prior_cool)")
    ap.add_argument("--structure", action="store_true",
                    help="also score TAD, loop and compartment measures per region "
                         "(flexcible.structure), for the model (struct_*) and for "
                         "every --others map named consensus* (struct_<name>_*)")
    ap.add_argument("--save-pred", default=None,
                    help="also write every region's prediction (target grid) to this npz")
    ap.add_argument("--keep-tracks", default=None,
                    help="comma-separated ChIP tracks given at prediction; every other "
                         "track is filled with its absent value ('' = motifs and "
                         "sequence only). Default: every track the cell has")
    ap.add_argument("--keep-sets", nargs="*", default=None,
                    help="several keep sets in one pass, each name=TRACK,TRACK (name= for "
                         "none): one output folder <out-dir>/<name> per set; --save-pred "
                         "then needs a {set} placeholder")
    ap.add_argument("--impute", nargs="*", default=[],
                    help="name=npz: imputed ChIP tracks (scripts/twostage_cnn_imputation.py; "
                         "arrays <chrom> of shape (marks, bins) in log1p signal units, key "
                         "'marks' lists the track names, NaN = no imputation). A keep set "
                         "named <set>@<name> gets those tracks in place of the absent value "
                         "for every imputable track it does not keep (also tracks the cell "
                         "lacks); the other tracks behave as in <set>")
    ap.add_argument("--cohesin-unknown", choices=["auto", "0", "1"], default="auto",
                    help="models trained with data.cohesin_unknown: 'auto' sets the "
                         "indicator to 1 where no cohesin track (RAD21, SMC3) is given "
                         "(absent in the cell or dropped by a keep set), '0'/'1' force it "
                         "(0 with absent RAD21 tests the model without the indicator)")
    ap.add_argument("--cohesin-fraction", type=float, default=1.0,
                    help="models trained with data.cohesin_fraction: the fraction of cohesin "
                         "left in the cell (1 intact, 0 none), the value of the cohesin:fraction "
                         "row. A keep set named *_f<value> (e.g. minus_RAD21_f0) overrides it. "
                         "A missing RAD21/SMC3 track alone does not change it: the row states "
                         "the biology, not what was measured")
    ap.add_argument("--diff-group", choices=["cell", "aml", "hspc"], default="cell",
                    help="models trained with data.diff_marks: the group of the cell, which picks "
                         "its reference set (cell lines: the 12-cell consensus; aml: the training "
                         "patients or the HSPC donors per data.diff_reference; hspc: the training "
                         "HSPC donors); the cell itself never enters it")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=os.environ.get("HICMULTI_HM", "/mnt/processed/wolffjoa/hic-multiome") + "/runs/crosscell")
    a = ap.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    cfg = Config.load(a.config)
    ckpt = Path(a.checkpoint)
    model, channels = sc.load_model(cfg, ckpt, device)
    size = cfg.window.size
    res = cfg.data.resolution                    # model native res == target binsize (2000)
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])
    n_chip = len(channels["tracks"])
    row_of = {t: i for i, t in enumerate(channels["tracks"])}
    sel_rows = None                              # model trained on a subset of the cond rows
    if cfg.train.select_tracks is not None or cfg.train.select_motifs is not None:
        from flexcible.train import _select_rows
        sel_rows, offset, n_motifs, _ = _select_rows(cfg, channels)
    mat_norm = MatrixNormalizer.load(ckpt / f"norm_matrix_{cfg.get_sample(None).name}.json")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    absent = track_norm.absent_values()
    mm_rows = (mm.coupling_rows(channels["sym_rows"], cfg.motifs.motif_meth_id)
               if cfg.motif_meth_active else None)
    use_seq = cfg.sequence.uses_encoder
    fasta = cfg.get_sample(None).fasta or cfg.sequence.fasta

    # this cell's sample: point tracks at its bigwig dir; which marks it has
    sample = cfg.get_sample(None)
    sample.bigwig_dir = a.bigwig_dir
    sample.extra_bigwig_dirs = list(a.extra_bigwig_dir)    # this cell's, never the config's
    if a.assay is not None:
        sample.assay = a.assay
    if a.lab is not None:
        sample.lab = a.lab
    have = {p.stem.upper(): p for d in [*reversed(a.extra_bigwig_dir), a.bigwig_dir]
            for p in Path(d).glob("*.bigwig")}
    missing = [t for t in channels["tracks"] if t.upper() not in have]
    if a.keep_tracks is not None:
        keep = {t.strip().upper() for t in a.keep_tracks.split(",") if t.strip()}
        missing = [t for t in channels["tracks"] if t in missing or t.upper() not in keep]
    print(f"[{a.cell}] {n_chip-len(missing)}/{n_chip} marks; missing={missing}", flush=True)
    # keep sets: extra tracks set to their absent value on top of the cell's own gaps
    sets = {"": []}
    nomm_sets = set()          # keep sets naming the pseudo track NOMM: methylation of the motif planes removed
    if a.keep_sets:
        sets = {}
        for item in a.keep_sets:
            nm, _, lst = item.partition("=")
            kp = {t.strip().upper() for t in lst.split(",") if t.strip()}
            sets[nm] = [t for t in channels["tracks"] if t not in missing and t.upper() not in kp]
            if "NOMM" in kp:
                nomm_sets.add(nm)
        if a.save_pred and "{set}" not in a.save_pred:
            raise SystemExit("--keep-sets with --save-pred needs a {set} placeholder")
    imp_files = {}
    for kv in a.impute:
        k, v = kv.split("=", 1)
        imp_files[k] = np.load(v)
    if imp_files:
        assert track_norm.transform == "log1p" and not track_norm.smooth_bins, \
            "--impute expects log1p track statistics without smoothing"
    imp_drop = {}          # keep set -> (imputation name, tracks taken from it)
    for nm, drop in list(sets.items()):
        if "@" in nm:
            base, _, var = nm.partition("@")
            kp = {t.strip().upper() for t in dict(i.partition("=")[::2] for i in a.keep_sets)[nm].split(",") if t.strip()}
            marks_ = [str(x) for x in imp_files[var]["marks"]]
            imp_drop[nm] = (var, [t for t in channels["tracks"] if t in marks_ and t.upper() not in kp])
    sample.missing_tracks = list(missing)        # declared absent -> raw zero -> absent value
    coh_i = None                                 # cohesin-unknown indicator row (full sym rows)
    if "cohesin_unknown" in channels["seq_features"]:
        coh_i = len(channels["tracks"]) + channels["seq_features"].index("cohesin_unknown")
        coh_v = [float(track_norm.apply(np.full((len(track_norm.mean), 1), v, np.float32))[coh_i, 0])
                 for v in (0.0, 1.0)]
        coh_tracks = [t for t in ("RAD21", "SMC3") if t in row_of]
    frac_i = None                                # cohesin-fraction row (full sym rows)
    if "cohesin_fraction" in channels["seq_features"]:
        frac_i = len(channels["tracks"]) + channels["seq_features"].index("cohesin_fraction")

    def frac_z(f):
        return float(track_norm.apply(np.full((len(track_norm.mean), 1), f, np.float32))[frac_i, 0])
    import flexcible.diffmarks as dmk
    if cfg.data.diff_marks:
        dmk.configure_unseen(cfg, sample, a.cell, a.diff_group)

    def blank_diff(sy, t):
        """a dropped / imputed mark also loses its reference and difference rows (diff models)"""
        dmk.blank_diff_rows(sy, cfg, channels["tracks"], channels["seq_features"], t)
    track_files = sample_track_files(cfg, sample)

    ref_clr = None
    if cfg.model.reference != "none":
        # --reference, else the cell's own training reference (a leave-one-out
        # map for a training cell), else the map embedded in the checkpoint
        # (model.embed_reference), else data.reference_cool; never a map that
        # contains this cell (flexcible.reference.resolve_reference)
        from flexcible.reference import open_reference, resolve_reference
        try:
            ref_path = resolve_reference(cfg, ckpt, a.cell, a.reference)
        except ValueError as e:
            raise SystemExit(str(e))
        # a training cell's reference was encoded with its own normaliser
        own_norm = ckpt / f"norm_matrix_{a.cell}.json"
        if own_norm.exists():
            mat_norm = MatrixNormalizer.load(own_norm)
        ref_clr = open_reference(ref_path, res)
        print(f"[{a.cell}] reference map {ref_path} ({cfg.model.reference})", flush=True)

    prior_clr = None
    if cfg.model.prior:
        own = next((s for s in cfg.data.samples if s.name == a.cell), None)
        pp = a.prior or (cfg.sample_prior(own) if own else cfg.data.prior_cool)
        if not pp:
            raise SystemExit(f"{a.cell}: cascade model needs --prior")
        prior_clr = cooler.Cooler(pp)
        print(f"[{a.cell}] prior map {pp}", flush=True)

    tgt = cooler.Cooler(a.target)
    src = tgt                                    # chrom sizes at the model/target resolution
    coarse = {}
    for kv in a.coarse:
        k, v = kv.split("=", 1)
        coarse[int(k)] = cooler.Cooler(v)
    others = {}
    for kv in a.others:
        k, v = kv.split("=", 1)
        assert_reference_excludes(v, a.cell)    # consensus baselines must exclude the cell
        others[k] = cooler.Cooler(v)

    regions = pd.read_csv(a.regions or sc.REGIONS, sep="\t", usecols=[0, 1, 2],
                          names=["chrom", "start", "end"])
    rows = {k: [] for k in sets}
    saved = {k: {} for k in sets}
    for chrom, grp in regions.groupby("chrom"):
        tname = match_name(chrom, list(tgt.chromnames))     # target fetch (2 kb grid)
        sname = match_name(chrom, list(src.chromnames))     # nb + ChIP binning (2048)
        if tname is None or sname is None:
            continue
        nb = chrom_nbins_of(src, sname)
        sym_full = track_norm.apply(assemble_chrom(cfg, sample, track_files, sname, nb))
        # fill any missing mark rows with the absent value
        for t in missing:
            sym_full[row_of[t]] = absent[row_of[t]]
            blank_diff(sym_full, t)
        sym_sets, coh_report = {}, {}
        imp_z = {}
        for var, f in imp_files.items():
            key = match_name(sname, [x for x in f.files if x != "marks"])
            L = f[key] if key is not None else None
            marks_ = [str(x) for x in f["marks"]]
            z = np.full((len(marks_), nb), np.nan, np.float32)
            if L is not None:
                n = min(nb, L.shape[1])
                for j, t in enumerate(marks_):
                    if t in row_of:
                        z[j, :n] = ((L[j, :n] - track_norm.mean[row_of[t]])
                                    / track_norm.std[row_of[t]])
            imp_z[var] = (marks_, z)
        for k, drop in sets.items():
            sy = sym_full.copy() if (drop or coh_i is not None or frac_i is not None
                                     or k in nomm_sets or k in imp_drop) else sym_full
            if frac_i is not None:
                mf = re.search(r"_f(\d+(?:\.\d+)?)$", k)
                sy[frac_i] = frac_z(float(mf.group(1)) if mf else a.cohesin_fraction)
            for t in drop:
                sy[row_of[t]] = absent[row_of[t]]
                blank_diff(sy, t)
            if k in imp_drop:
                var, take = imp_drop[k]
                marks_, z = imp_z[var]
                for t in take:
                    zt = z[marks_.index(t)]
                    sy[row_of[t]] = np.where(np.isfinite(zt), zt, absent[row_of[t]])
                    blank_diff(sy, t)
            if mm_rows is not None and ("Meth" in drop or k in nomm_sets):
                mm.drop_methylation(sy, track_norm, mm_rows)      # motif planes lose it too
            if coh_i is not None:
                gone = set(missing) | set(drop)
                mode = ("0" if k.endswith("_ind0") else "1" if k.endswith("_ind1")
                        else a.cohesin_unknown)       # a keep set named *_ind0/_ind1 forces it
                unk = (mode == "1" if mode != "auto"
                       else all(t in gone for t in coh_tracks))
                sy[coh_i] = coh_v[int(unk)]
                coh_report[k] = int(unk)
            sym_sets[k] = sy[sel_rows] if sel_rows is not None else sy
        if coh_i is not None and chrom == regions.chrom.iloc[0]:
            print(f"[{a.cell}] cohesin_unknown indicator per keep set: {coh_report}", flush=True)
        print(f"  {chrom}: {len(grp)} regions", flush=True)
        for _, r in grp.iterrows():
            try:
                T = np.asarray(tgt.matrix(balance=False).fetch(
                    f"{tname}:{int(r.start)}-{int(r.end)}"), float)
            except Exception:
                continue
            n_out = T.shape[0]
            centre = (int(r.start) + int(r.end)) // 2 // res
            st = max(0, min(nb - size, centre - size // 2))
            ref = None
            if ref_clr is not None:
                from flexcible.reference import reference_block
                ref = torch.from_numpy(
                    reference_block(ref_clr, chrom, st, size, mat_norm)[None]).to(device)
            prior = None
            if prior_clr is not None:
                from flexcible.reference import prior_block
                prior = torch.from_numpy(
                    prior_block(prior_clr, chrom, st, size, res, mat_norm)[None]).to(device)
            seq = (sc.region_onehot(fasta, tname, st, size, res, device) if use_seq else None)
            seqr = (sc.region_onehot(fasta, tname, st, size, res, device, revcomp=True)
                    if (use_seq and a.tta) else None)
            # measured-map baselines: other cells' REAL map vs this cell's real map
            base_rec = {}
            iu = np.triu_indices(n_out, k=2)
            for oc, oclr in others.items():
                onm = match_name(chrom, list(oclr.chromnames))
                try:
                    O = np.asarray(oclr.matrix(balance=False).fetch(
                        f"{onm}:{int(r.start)}-{int(r.end)}"), float)
                except Exception:
                    continue
                if O.shape == T.shape and a.structure and oc.startswith("consensus"):
                    from flexcible.structure import structure_metrics
                    sm_o = structure_metrics(np.where(O == 0, np.nan, O),
                                             np.where(T == 0, np.nan, T), tgt.binsize)
                    base_rec.update({f"struct_{oc}_{k}": v for k, v in sm_o.items()})
                if O.shape == T.shape:
                    x, y = O[iu], T[iu]
                    m2 = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
                    if m2.sum() > 100 and x[m2].std() > 0 and y[m2].std() > 0:
                        base_rec[f"base_{oc}"] = float(pearsonr(x[m2], y[m2]).statistic)
            for skey, sym_set in sym_sets.items():
                ct = sym_set[:, st:st + size]
                if ct.shape[1] < size:
                    ct = np.concatenate([ct, np.zeros((ct.shape[0], size - ct.shape[1]),
                                                      np.float32)], 1)
                cond = torch.from_numpy(ct[None]).to(device)
                pairs = (cond[:, offset:offset + 2 * n_motifs].reshape(1, n_motifs, 2, size)
                         if n_motifs else None)
                with torch.no_grad():
                    o = model(cond, pairs, None, seq, ref, prior=prior)
                    if a.tta:
                        from flexcible.dataset import reverse_cond, split_cond
                        cr = reverse_cond(cond, offset, n_motifs)
                        srr, prr = split_cond(cr, offset, n_motifs)
                        refr = torch.flip(ref, dims=[-1, -2]) if ref is not None else None
                        prr_ = torch.flip(prior, dims=[-1, -2]) if prior is not None else None
                        o = 0.5 * (o + torch.flip(model(srr, prr, None, seqr, refr, prior=prr_),
                                                  dims=[-1, -2]))
                P = mat_norm.decode(o[0, 0].float().cpu().numpy())
                P = 0.5 * (P + P.T)
                P = sc.resample(P, st * res, res, int(r.start), n_out, tgt.binsize)
                if a.save_pred:
                    saved[skey][f"{chrom}:{int(r.start)}-{int(r.end)}"] = P.astype(np.float32)
                live = np.abs(P).sum(1) > 0
                if live.mean() < 0.5:
                    continue
                M = np.where(live[:, None] & live[None, :], P, np.nan)
                Tm = np.where(T == 0, np.nan, T)
                ok = np.isfinite(M) & np.isfinite(Tm)
                rec = {"region": f"{chrom}:{int(r.start)}-{int(r.end)}"}
                if ok.sum() > 100 and M[ok].std() > 0 and Tm[ok].std() > 0:
                    rec["model"] = float(pearsonr(M[ok], Tm[ok]).statistic)
                if a.structure:
                    from flexcible.structure import structure_metrics
                    rec.update({f"struct_{k}": v for k, v in
                                structure_metrics(M, Tm, tgt.binsize).items()})
                # the same prediction averaged onto coarser grids, scored there; only
                # bins inside the model window count (resample extends the edges of
                # a window shorter than the region by nearest-value copies)
                if coarse:
                    cen = int(r.start) + (np.arange(n_out) + 0.5) * tgt.binsize
                    cov = (cen >= st * res) & (cen < (st + size) * res)
                    Mc = np.where(cov[:, None] & cov[None, :], M, np.nan)
                for cres, cclr in coarse.items():
                    r_c = coarse_score(Mc, int(r.start), int(r.end), tgt.binsize, cres, cclr,
                                       chrom)
                    if r_c is not None:
                        rec[f"model_{cres // 1000}kb"] = r_c
                rec.update(base_rec)
                rows[skey].append(rec)

    for skey in sets:
        out = Path(a.out_dir) / skey if skey else Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if a.save_pred:
            np.savez_compressed(a.save_pred.replace("{set}", skey), **saved[skey])
        df = pd.DataFrame(rows[skey])
        df.to_csv(out / f"predict_{a.cell}.csv", index=False)
    print(f"\n=== {a.cell}: model prediction vs measured-map baselines ===", flush=True)
    cols = [c for c in df.columns if c != "region"]
    summ = df[cols].median().sort_values(ascending=False)
    for k, v in summ.items():
        tag = ("MODEL (from ChIP)" if k == "model" else
               f"MODEL averaged to {k[6:]}" if k.startswith("model_") else
               k.replace("base_", "real ") + " -> " + a.cell)
        print(f"  {v:.4f}  {tag}", flush=True)
    print(f"  n={df['model'].notna().sum()} regions; wrote predict_{a.cell}.csv")


if __name__ == "__main__":
    main()
