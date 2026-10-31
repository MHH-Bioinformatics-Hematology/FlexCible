"""Score predicted against measured changes. Writes, per region, results/perturb_hap1/<region>_{map,loop,insulation}_scores.csv"""
import sys, os, itertools; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from lib import *

LOOPS = pd.read_csv(RES / "loops_hap1_hiccups_grch38.tsv", sep="\t")
TOL = 5000          # an anchor belongs to the deleted site when its interval, widened by TOL, contains the motif
INS_W = {"100kb": 100_000, "200kb": 200_000}


def site_loops(region, sites):
    """Loops of the Sanborn HAP1 list with an anchor at one of the deleted motifs; returns (loop id, site, end 'x'|'y', row)."""
    out = []
    L = LOOPS[LOOPS.region == region]
    for sname in sites:
        s = SITES[sname]
        for k, r in L.iterrows():
            for end, (a, b) in (("x", (r.x1, r.x2)), ("y", (r.y1, r.y2))):
                if a - TOL <= (s["start"] + s["end"]) // 2 <= b + TOL:
                    out.append((f"{int(r.x1)}-{int(r.y1)}", sname, end, r))
    return out


def loop_pix(r, reg, res):
    xi = int(((r.x1 + r.x2) / 2 - reg["start"]) // res); yi = int(((r.y1 + r.y2) / 2 - reg["start"]) // res)
    return min(xi, yi), max(xi, yi)


def mean_change(clones, wt):
    return np.nanmean([c - wt for c in clones], 0)


def pooled_logoe(es_list, rn, rl, z):
    c = sum(z[f"counts_{e}"].astype(np.float64) for e in es_list)
    return region_logoe(c)


def main(models):
    for rn, reg in REGIONS.items():
        mrows, lrows, irows = [], [], []
        for rl in models:
            res = RESOLUTION[rl]; n = (reg["end"] - reg["start"]) // res
            z = np.load(RES / "maps" / f"measured_{rn}_{rl}.npz"); wt = z[WT_SET]
            wt_pred = pred_region(COOLS / rl / f"wt_{rn}.cool", reg["chrom"], reg["start"], reg["end"], res)
            I, J = pixel_mask(n, 2)
            I1, J1 = pixel_mask(n, 2, int(1_000_000 // res))
            for dn, d in DELETIONS.items():
                if d["region"] != rn: continue
                pdel = pred_region(COOLS / rl / f"del_{dn}.cool", reg["chrom"], reg["start"], reg["end"], res)
                pctl = pred_region(COOLS / rl / f"ctrl_{dn}.cool", reg["chrom"], reg["start"], reg["end"], res)
                pchg, cchg = pdel - wt_pred, pctl - wt_pred
                units = {f"clone:{s}": z[s] for s in d["sets"]}
                units["mean_of_clones"] = None
                units["pooled_counts"] = pooled_logoe(d["sets"], rn, rl, z)
                chg = {k: (v - wt if v is not None else mean_change([z[s] for s in d["sets"]], wt)) for k, v in units.items()}
                # ---- map-wide change correlation
                for u, m in chg.items():
                    mrows.append(dict(model=rl, region=rn, deletion=dn, unit=u,
                                      pearson_pred_vs_meas=pearson(pchg, m, I, J), pearson_le1Mb=pearson(pchg, m, I1, J1),
                                      spearman=spearman(pchg, m, I, J),
                                      pearson_control_vs_meas=pearson(cchg, m, I, J),
                                      pred_change_sd=float(np.nanstd(pchg[I, J])), ctrl_change_sd=float(np.nanstd(cchg[I, J])),
                                      meas_change_sd=float(np.nanstd(m[I, J]))))
                # noise floor: agreement between independent clones of the same deletion
                for a, b in itertools.combinations(d["sets"], 2):
                    mrows.append(dict(model=rl, region=rn, deletion=dn, unit=f"clone_vs_clone:{a}|{b}",
                                      pearson_pred_vs_meas=pearson(chg[f"clone:{a}"], chg[f"clone:{b}"], I, J),
                                      pearson_le1Mb=pearson(chg[f"clone:{a}"], chg[f"clone:{b}"], I1, J1),
                                      spearman=spearman(chg[f"clone:{a}"], chg[f"clone:{b}"], I, J)))
                # specificity: this deletion's measured change vs the prediction for the other deletions of the region
                for dn2, d2 in DELETIONS.items():
                    if d2["region"] == rn and dn2 != dn:
                        p2 = pred_region(COOLS / rl / f"del_{dn2}.cool", reg["chrom"], reg["start"], reg["end"], res) - wt_pred
                        mrows.append(dict(model=rl, region=rn, deletion=dn, unit=f"mean_of_clones_vs_prediction_of:{dn2}",
                                          pearson_pred_vs_meas=pearson(p2, chg["mean_of_clones"], I, J),
                                          pearson_le1Mb=pearson(p2, chg["mean_of_clones"], I1, J1)))
                # ---- loops anchored at the deleted site(s)
                for lid, sname, end, r in site_loops(rn, d["sites"]):
                    i, j = loop_pix(r, reg, res)
                    pw, pd_, pc = win_max(wt_pred, i, j), win_max(pdel, i, j), win_max(pctl, i, j)
                    for u in units:
                        if u == "mean_of_clones":
                            ms = [win_max(z[s], i, j) for s in d["sets"]]; mw = win_max(wt, i, j)
                            mdel = float(np.nanmean(ms))
                        elif u == "pooled_counts":
                            mdel, mw = win_max(units[u], i, j), win_max(wt, i, j)
                        else:
                            mdel, mw = win_max(units[u], i, j), win_max(wt, i, j)
                        mc = mdel - mw
                        lrows.append(dict(model=rl, region=rn, deletion=dn, unit=u, loop=lid, site=sname, anchor_end=end, convergent_anchor=bool((end == 'x' and SITES[sname]['strand'] == '+') or (end == 'y' and SITES[sname]['strand'] == '-')),
                                          loop_size_kb=int(abs(j - i) * res / 1000), hiccups_observed=int(r.observed),
                                          meas_wt=mw, meas_del=mdel, meas_change=mc,
                                          pred_wt=pw, pred_del=pd_, pred_change=pd_ - pw,
                                          ctrl_pred_change=pc - pw, sign_agree=bool(np.sign(mc) == np.sign(pd_ - pw)) if np.isfinite(mc) else np.nan))
                # ---- insulation across the deleted site(s)
                for sname in d["sites"]:
                    s = SITES[sname]; b = int(((s["start"] + s["end"]) // 2 - reg["start"]) // res)
                    for wn, wb in INS_W.items():
                        w = wb // res
                        pw_, pd2, pc2 = insulation(wt_pred, b, w), insulation(pdel, b, w), insulation(pctl, b, w)
                        for u in units:
                            if u == "mean_of_clones":
                                mdel = float(np.nanmean([insulation(z[s_], b, w) for s_ in d["sets"]]))
                            else:
                                mdel = insulation(units[u], b, w)
                            mw = insulation(wt, b, w)
                            irows.append(dict(model=rl, region=rn, deletion=dn, unit=u, site=sname, window=wn,
                                              meas_wt=mw, meas_del=mdel, meas_change=mdel - mw,
                                              pred_wt=pw_, pred_del=pd2, pred_change=pd2 - pw_, ctrl_pred_change=pc2 - pw_,
                                              sign_agree=bool(np.sign(mdel - mw) == np.sign(pd2 - pw_))))
        pd.DataFrame(mrows).to_csv(RES / f"{PFX}{rn}_map_scores.csv", index=False, float_format="%.4f")
        pd.DataFrame(lrows).to_csv(RES / f"{PFX}{rn}_loop_scores.csv", index=False, float_format="%.4f")
        pd.DataFrame(irows).to_csv(RES / f"{PFX}{rn}_insulation_scores.csv", index=False, float_format="%.4f")
        print("wrote", rn, len(mrows), len(lrows), len(irows))


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["10kb", "5kb"])
