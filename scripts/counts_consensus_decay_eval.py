"""Count-space correlations on chr2:18-22 Mb (10 kb, band 1.5 Mb) for the two back-transforms of a prediction.

(a) measured map vs prediction back-transformed with the decay of the measured map of the same cell
(b) measured map vs prediction back-transformed with the consensus decay (scripts/oe_to_counts_consdecay.py)
(c) the two reconstructions with each other (they differ only by a per-diagonal factor)
Pixels as in the manuscript's fig_counts.py: mappable bins, 2 to 150 bins from the diagonal, upper triangle, present in the
predicted cool, measured count > 0. 'diag-removed' = Pearson of log10 counts after subtracting the per-diagonal mean
(over the same pixels) from each map; 'raw' = Pearson of log10 counts.

    python scripts/counts_consensus_decay_eval.py [out.csv]
"""
import sys

import cooler
import numpy as np

sys.path.insert(0, "/home/mh-hannover.local/wolffjoa/src/hic-multiome-unet/scripts")
import oe_to_counts as OC  # noqa: E402

R = "/mnt/processed/wolffjoa/hic-multiome/runs/dense_region_fullrd_chrho"
M4 = "/mnt/processed/wolffjoa/restart_of_prediction_project/originalData/4Dnucleosome"
CELLS = {"gm12878": ("GM12878", f"{M4}/GM12878_4DNFIXP4QG5B.mcool"),
         "imr90": ("IMR-90", "/mnt/processed/wolffjoa/hyperparameter_publication/imr90.mcool"),
         "k562": ("K562", f"{M4}/K562_4DNFI18UHVRO.mcool")}
CH, S, E, RES, SHOW = "chr2", 18_000_000, 22_000_000, 10_000, 150
NB = (E - S) // RES


def dense(cool_path):
    c = cooler.Cooler(cool_path)
    px = c.matrix(balance=False, as_pixels=True, join=False).fetch(f"{CH}:{S}-{E}")
    b0 = c.offset(CH) + S // RES
    M = np.full((NB, NB), np.nan)
    i, j = px.bin1_id.values - b0, px.bin2_id.values - b0
    M[i, j] = px["count"].values
    M[j, i] = px["count"].values
    return M


def pear(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def demean(v, d):
    v = v.copy()
    for q in np.unique(d):
        m = d == q
        v[m] -= v[m].mean()
    return v


def main(out):
    rows = []
    for key, (name, mc) in CELLS.items():
        meas = OC.OP.open_at(mc, RES)
        reg = f"{CH}:{S}-{E}"
        M = np.asarray(meas.matrix(balance="weight").fetch(reg), float)
        mp = np.isfinite(meas.bins()[meas.offset(CH) + S // RES: meas.offset(CH) + E // RES]["weight"].values)
        A = dense(f"{R}/{key}_counts_chr2.cool")
        B = dense(f"{R}/consensus_decay/{key}_counts_chr2_consdecay.cool")
        d = np.abs(np.arange(NB)[:, None] - np.arange(NB)[None, :])
        ok = (mp[:, None] & mp[None, :] & (d >= 2) & (d <= SHOW) & np.triu(np.ones((NB, NB), bool))
              & np.isfinite(A) & np.isfinite(B) & (M > 0) & (A > 0) & (B > 0))
        lm, la, lb, dd = np.log10(M[ok]), np.log10(A[ok]), np.log10(B[ok]), d[ok]
        dm, da, db = demean(lm, dd), demean(la, dd), demean(lb, dd)
        delta = np.array([np.mean(lb[dd == q] - la[dd == q]) for q in range(2, SHOW + 1)])
        rows.append(dict(cell=name, n=int(ok.sum()),
                         a_diag=pear(dm, da), b_diag=pear(dm, db), c_diag=pear(da, db),
                         a_raw=pear(lm, la), b_raw=pear(lm, lb), c_raw=pear(la, lb),
                         delta_median=float(np.median(delta)), delta_min=float(delta.min()), delta_max=float(delta.max()),
                         delta_at_20kb=float(delta[0]), delta_at_1500kb=float(delta[-1])))
        print(rows[-1], flush=True)
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/one_model/counts_consensus_decay.csv")
