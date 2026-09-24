"""Predict 3-D structure for an ALTERED DNA sequence.

An edit is any replacement of a reference interval by a new sequence, which
covers the clinically relevant cases: an insertion (empty interval, non-empty
replacement), a deletion (non-empty interval, empty replacement), a tandem
duplication such as FLT3-ITD (replacement = the interval twice), and an
arbitrary substitution.

What actually reaches the model:

* sequence features (GC, CpG) are recomputed on the edited sequence;
* motifs are RE-SCANNED on the edited sequence, so an edit that destroys or
  creates a binding site changes the conditioning even when it is far smaller
  than a bin -- this is the route by which a short variant can move the
  prediction at all;
* ChIP tracks are carried through the coordinate map. Duplicated sequence
  inherits the signal of the segment it was copied from (the honest choice
  for a tandem duplication); genuinely novel inserted sequence has no
  measurement, so it is marked and filled with the "absent" level rather than
  silently reusing a neighbour's signal.

Resolution caveat, stated because it decides what is worth asking: at 25 kb
bins a 30-300 bp FLT3-ITD shifts coordinates by a fraction of one bin, so any
change in the predicted map comes from the motif/sequence channels, not from
the geometry. Edits large enough to move bins (tens of kb and up) change the
map structurally. Both are supported; only the second rearranges the matrix.

CLI:
  python -m hicmultiome.variants --config configs/multiome_25kb.yaml \
      --region chr13:28000000-28600000 \
      --edit chr13:28034000-28034060:dup --out runs/flt3_itd.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .genome import match_name
from .utils import get_logger

log = get_logger()

_BASES = np.frombuffer(b"ACGTN", dtype=np.uint8)


@dataclass
class Edit:
    """Replace `chrom:start-end` (0-based, half-open) by `replacement`."""

    chrom: str
    start: int
    end: int
    replacement: str = ""
    label: str = ""

    @classmethod
    def parse(cls, spec: str, ref_seq_fn=None) -> "Edit":
        """`chrom:start-end:KIND[:SEQ]` with KIND in del|dup|ins|sub.

        dup duplicates the interval in place (tandem), ins inserts SEQ at
        `start` (end is ignored), del removes the interval, sub replaces it
        with SEQ.
        """
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"edit {spec!r}: expected chrom:start-end:kind[:SEQ]")
        chrom, span, kind = parts[0], parts[1], parts[2].lower()
        start, end = (int(x) for x in span.split("-"))
        seq = parts[3].upper() if len(parts) > 3 else ""
        if kind == "del":
            return cls(chrom, start, end, "", f"del {end - start} bp")
        if kind == "ins":
            if not seq:
                raise ValueError("ins needs a sequence: chrom:pos-pos:ins:ACGT")
            return cls(chrom, start, start, seq, f"ins {len(seq)} bp")
        if kind == "dup":
            ref = ref_seq_fn(chrom, start, end) if ref_seq_fn else None
            if ref is None:
                raise ValueError("dup needs the reference sequence")
            return cls(chrom, start, end, ref + ref, f"dup {end - start} bp")
        if kind == "sub":
            return cls(chrom, start, end, seq, f"sub {end - start}->{len(seq)} bp")
        raise ValueError(f"unknown edit kind {kind!r} (del|dup|ins|sub)")

    @property
    def delta(self) -> int:
        return len(self.replacement) - (self.end - self.start)


def read_chrom_seq(fasta: str, chrom: str) -> str:
    from pyfaidx import Fasta

    fa = Fasta(str(Path(fasta).expanduser()), rebuild=False)
    name = match_name(chrom, list(fa.keys()))
    if name is None:
        raise KeyError(f"{chrom} not in {fasta}")
    return str(fa[name][:].seq).upper()


def apply_edits(seq: str, edits: list[Edit]) -> tuple[str, np.ndarray, np.ndarray]:
    """Edited sequence, a reference->edited position map, and a SOURCE map.

    pos_map[reference_position] = index in the edited sequence, or -1 if the
    base was deleted.

    src_map[edited_position] = the reference position that base came from, or
    -1 for genuinely new sequence. This is what lets a tandem duplication
    inherit the chromatin state of the segment it copied: every copy points
    back at the same source, so the second copy carries the same marks rather
    than appearing as unmeasured emptiness -- which would put a false cliff at
    the junction and invite the model to call a boundary there.
    """
    edits = sorted(edits, key=lambda e: e.start)
    for a, b in zip(edits, edits[1:]):
        if b.start < a.end:
            raise ValueError("overlapping edits are not supported")
    out: list[str] = []
    pos_map = np.full(len(seq), -1, dtype=np.int64)
    src_chunks: list[np.ndarray] = []
    ref_i = cur = 0
    for e in edits:
        keep = seq[ref_i : e.start]
        out.append(keep)
        pos_map[ref_i : e.start] = np.arange(cur, cur + len(keep))
        src_chunks.append(np.arange(ref_i, e.start))
        cur += len(keep)

        src = seq[e.start : e.end]
        rep = e.replacement
        out.append(rep)
        if (src and rep and len(rep) % len(src) == 0
                and rep == src * (len(rep) // len(src))):
            # a tandem repeat of the source: EVERY copy inherits the source's
            # coordinates, so all of them carry its chromatin state
            k = len(rep) // len(src)
            for c in range(k):
                src_chunks.append(np.arange(e.start, e.end))
            pos_map[e.start : e.end] = np.arange(cur, cur + len(src))
        else:
            if rep.startswith(src) and src:
                pos_map[e.start : e.end] = np.arange(cur, cur + len(src))
                src_chunks.append(np.arange(e.start, e.end))
                src_chunks.append(np.full(len(rep) - len(src), -1, dtype=np.int64))
            else:
                src_chunks.append(np.full(len(rep), -1, dtype=np.int64))
        cur += len(rep)
        ref_i = e.end
    out.append(seq[ref_i:])
    pos_map[ref_i:] = np.arange(cur, cur + len(seq) - ref_i)
    src_chunks.append(np.arange(ref_i, len(seq)))
    edited = "".join(out)
    src_map = (np.concatenate(src_chunks) if src_chunks
               else np.arange(len(seq), dtype=np.int64))
    return edited, pos_map, src_map


def _codes(seq: str) -> np.ndarray:
    from .sequence import _CODE

    return _CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def edited_rows(cfg: Config, sample, chrom: str, edits: list[Edit],
                ref_rows: np.ndarray, n_bins_ref: int):
    """Per-bin conditioning rows on the EDITED coordinate system.

    ref_rows: the reference rows (R, n_bins_ref) as assemble_chrom produced
    them. Returns (rows_edited, n_bins_edited, novel_bins).
    """
    from .motifs import pwm_logodds, read_jaspar, revcomp_pwm, scan_hits
    from .precompute import sym_row_names
    from .sequence import sequence_features

    res = cfg.data.resolution
    fasta = cfg.sample_fasta(sample)
    seq = read_chrom_seq(fasta, chrom)
    edited, pos_map, src_map = apply_edits(seq, edits)
    nb_ed = int(np.ceil(len(edited) / res))
    rows_out = np.zeros((ref_rows.shape[0], nb_ed), dtype=np.float32)

    names = sym_row_names(cfg)
    n_chip = len(cfg.track_names)

    # 1) ChIP tracks: every edited base pulls its value from the reference base
    # it came from, so each copy of a duplication carries the same marks.
    ed_bin = np.arange(len(edited)) // res
    ref_pos = src_map
    have = (ref_pos >= 0) & (ed_bin < nb_ed)
    ref_bin_of = np.clip(ref_pos[have] // res, 0, n_bins_ref - 1)
    tgt_bin = ed_bin[have]
    cnt = np.bincount(tgt_bin, minlength=nb_ed).astype(np.float64)
    for r in range(n_chip):
        acc = np.bincount(tgt_bin, weights=ref_rows[r][ref_bin_of].astype(np.float64),
                          minlength=nb_ed)
        rows_out[r] = (acc / np.maximum(cnt, 1)).astype(np.float32)
    novel_mask = np.zeros(len(edited), dtype=bool)
    novel_mask[src_map < 0] = True

    # 2) sequence features recomputed on the edited sequence
    codes = _codes(edited)
    off = n_chip
    if cfg.sequence.features:
        feats = sequence_features(codes, cfg.sequence.features, nb_ed, res)
        rows_out[off : off + feats.shape[0]] = feats
        off += feats.shape[0]

    # 3) motifs RE-SCANNED on the edited sequence
    for mid in cfg.active_motif_ids:
        pwm = pwm_logodds(read_jaspar(Path(cfg.motifs.dir).expanduser() /
                                      f"{mid}.jaspar"))
        for pwm_s in (pwm, revcomp_pwm(pwm)):
            hits = scan_hits(codes, pwm_s, cfg.motifs.score_frac)
            b = hits // res
            b = b[b < nb_ed]
            rows_out[off] = np.bincount(b, minlength=nb_ed).astype(np.float32)
            off += 1

    novel_bins = np.zeros(nb_ed, dtype=bool)
    if novel_mask.any():
        nb_idx = np.nonzero(novel_mask)[0] // res
        novel_bins[np.clip(nb_idx, 0, nb_ed - 1)] = True
        # only sequence with NO reference source is unmeasured; duplicated
        # copies keep the signal they inherited above
        rows_out[:n_chip, novel_bins] = 0.0
    log.info("edited %s: %d -> %d bp (%+d), %d bins, %d novel bins; rows %s",
             chrom, len(seq), len(edited), len(edited) - len(seq), nb_ed,
             int(novel_bins.sum()), len(names))
    return rows_out, nb_ed, novel_bins


def predict_edited(cfg: Config, chrom: str, edits: list[Edit], out_cool: str,
                   sample_name: str | None = None, checkpoint=None,
                   batch: int = 8,
                   zero_regions: list[tuple[str, str, int, int]] | None = None,
                   renormalize_tracks: bool = False):
    """Predict the contact map of an EDITED chromosome and write it as a cool.

    The three perturbation kinds compose here: `zero_regions` can remove a
    protein (assay row) or a binding site (motif rows) over an interval, and
    `edits` change the DNA itself, after which sequence features and motifs are
    re-scanned and coordinates shift. The output cool carries the edited
    chromosome's own bin table, so an insertion or deletion is visible as a
    changed length rather than being silently squeezed back onto the reference.
    """
    import torch

    from .matrix import (BandedAccumulator, MatrixNormalizer, hann2d, open_cool,
                         plan_windows, write_cool_custom)
    from .precompute import assemble_chrom, chrom_nbins_of, sample_track_files
    from .predict import _resolve_ckpt, build_model, load_norm
    from .tracks import TrackNormalizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_ckpt(cfg, checkpoint)
    sample = cfg.get_sample(sample_name)
    if not cfg.sample_fasta(sample):
        raise ValueError(
            f"sample {sample.name!r} has no genome; sequence edits need a FASTA")
    track_norm = TrackNormalizer.load(ckpt / "norm_tracks.json")
    mat_norm = load_norm(ckpt, "norm_matrix", sample.name, MatrixNormalizer)
    model, channels = build_model(cfg, ckpt, device)
    n_motifs = len(channels["motif_ids"])
    offset = len(channels["tracks"]) + len(channels["seq_features"])

    clr = open_cool(sample.cool_path)
    chrom = match_name(chrom, list(clr.chromnames)) or chrom
    res = cfg.data.resolution
    nb = chrom_nbins_of(clr, chrom)
    tf = sample_track_files(cfg, sample)
    raw = assemble_chrom(cfg, sample, tf, chrom, nb)
    if renormalize_tracks:
        log.info("re-fitting track statistics on this sample's own signal")
        track_norm = TrackNormalizer(
            transform=track_norm.transform, norm=track_norm.norm,
            smooth_bins=track_norm.smooth_bins).fit([raw])

    for t, ch, a, b in (zero_regions or []):     # protein or site removal first
        rows = ([i for i, r in enumerate(channels["sym_rows"]) if r.startswith(t)]
                if t.startswith("motif:") else [channels["tracks"].index(t)])
        lo, hi = max(0, a // res), min(nb, -(-b // res))
        for r in rows:
            raw[r, lo:hi] = 0.0

    rows_ed, nb_ed, novel = edited_rows(cfg, sample, chrom, edits, raw, nb)
    sym_full = track_norm.apply(rows_ed)
    size = cfg.window.size
    acc = BandedAccumulator(nb_ed, band=size)
    w2d = hann2d(size)
    wins = plan_windows(nb_ed, size, cfg.window.stride, chrom)
    for i in range(0, len(wins), batch):
        bw = wins[i : i + batch]
        conds = []
        for w in bw:
            ct = sym_full[:, w.start : w.start + size]
            if ct.shape[1] < size:
                ct = np.concatenate(
                    [ct, np.zeros((ct.shape[0], size - ct.shape[1]), np.float32)], 1)
            conds.append(ct)
        cond = torch.from_numpy(np.stack(conds)).to(device)
        pairs = None
        if n_motifs:
            pairs = cond[:, offset : offset + 2 * n_motifs].reshape(
                cond.shape[0], n_motifs, 2, size)
        planes = None
        if channels.get("capture_input"):
            planes = torch.full((cond.shape[0], 2, size, size), -1.0, device=device)
            planes[:, 1] = 0.0
        with torch.no_grad():
            # HiCModel builds the 2-D conditioning itself (as in training)
            imgs = model(cond, pairs, planes)[:, 0]
        for w, img in zip(bw, imgs.float().cpu().numpy()):
            block = np.maximum(mat_norm.decode(img), 0.0)
            acc.add_block(w.start, 0.5 * (block + block.T), w2d)
    b1, b2, c = acc.to_pixels(0)
    write_cool_custom(out_cool, chrom, nb_ed, res, b1, b2, c)
    log.info("wrote %s (%d bins after %d edit(s); reference had %d)",
             out_cool, nb_ed, len(edits), nb)
    return {"out": out_cool, "n_bins": nb_ed, "n_bins_reference": nb,
            "novel_bins": [int(x) for x in np.nonzero(novel)[0]]}


def main():
    ap = argparse.ArgumentParser(
        description="Predict folding under DNA edits, protein knockouts and "
                    "binding-site removal (the three compose).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--edit", action="append", default=[],
                    help="chrom:start-end:del|dup|ins|sub[:SEQ] (repeatable)")
    ap.add_argument("--zero-region", action="append", default=[],
                    help="TRACK:chrom:start-end removes the PROTEIN; "
                         "motif:ID:chrom:start-end removes the BINDING SITE")
    ap.add_argument("--renormalize-tracks", action="store_true",
                    help="re-fit per-track statistics on this sample")
    ap.add_argument("--out", required=True, help="output .cool")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    sample = cfg.get_sample(args.sample)
    fasta = cfg.sample_fasta(sample)

    def ref_seq(chrom, a, b):
        return read_chrom_seq(fasta, chrom)[a:b]

    edits = [Edit.parse(spec, ref_seq) for spec in args.edit]
    zr = []
    for spec in args.zero_region:
        parts = spec.split(":")
        if parts[0] == "motif":                       # motif:ID:chrom:start-end
            name, chrom, span = f"motif:{parts[1]}", parts[2], parts[3]
        else:                                         # TRACK:chrom:start-end
            name, chrom, span = parts[0], parts[1], parts[2]
        a, b = (int(x) for x in span.replace(",", "").split("-"))
        zr.append((name, chrom, a, b))

    if edits:
        info = predict_edited(cfg, args.chrom, edits, args.out, args.sample,
                              args.checkpoint, args.batch_size, zr or None,
                              args.renormalize_tracks)
        print(info)
    else:
        from .predict import predict as run_predict

        run_predict(cfg, args.chrom, args.out, args.checkpoint, args.batch_size,
                    sample_name=args.sample, zero_regions=zr or None,
                    renormalize_tracks=args.renormalize_tracks)
        print({"out": args.out})


if __name__ == "__main__":
    main()
