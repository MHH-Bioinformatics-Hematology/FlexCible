"""Side-by-side statistics of the HAP1 CTCF-site deletions for the dedicated model of section 85 (no consensus),
the release model all12one_fullrd with the consensus plane and with --no-reference (registry section 99).
Writes results/perturb_hap1/all12one_fullrd_comparison.csv and prints markdown tables."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from common import RES
V = {"dedicated (section 85)": "", "release, consensus": "all12one_fullrd_consensus_", "release, no reference": "all12one_fullrd_noref_",
     "release, consensus, CTCF signal zero only": "all12one_fullrd_consensus_sigzero_",
     "release, no reference, CTCF signal zero only": "all12one_fullrd_noref_sigzero_"}
rows = []
for name, pf in V.items():
    if not (RES / f"{pf}region1_map_scores.csv").exists():
        continue
    M = pd.concat([pd.read_csv(RES / f"{pf}region{i}_map_scores.csv") for i in (1, 2, 3)])
    L = pd.concat([pd.read_csv(RES / f"{pf}region{i}_loop_scores.csv") for i in (1, 2, 3)])
    I = pd.concat([pd.read_csv(RES / f"{pf}region{i}_insulation_scores.csv") for i in (1, 2, 3)])
    for rl in ("10kb", "5kb"):
        l = L[(L.model == rl) & (L.unit == "mean_of_clones") & L.convergent_anchor]
        la = L[(L.model == rl) & (L.unit == "mean_of_clones")]      # all loops at a deleted site, convergent or not
        m = M[(M.model == rl) & (M.unit == "mean_of_clones")]
        cc = M[(M.model == rl) & M.unit.str.startswith("clone_vs_clone")].pearson_pred_vs_meas
        cl = M[(M.model == rl) & M.unit.str.startswith("clone:")].pearson_pred_vs_meas
        r = dict(model=name, res=rl, loops_all=len(la), loops_all_pred_decrease=int((la.pred_change < 0).sum()),
                 loops_all_sign_agree=int(la.sign_agree.sum()), loops_all_ctrl_abs_max=la.ctrl_pred_change.abs().max(),
                 loops_convergent=len(l), loops_convergent_pred_decrease=int((l.pred_change < 0).sum()), loops_convergent_sign_agree=int(l.sign_agree.sum()),
                 loop_pred_change_mean=l.pred_change.mean(), loop_pred_change_min=l.pred_change.min(), loop_pred_change_max=l.pred_change.max(),
                 loop_meas_change_mean=l.meas_change.mean(), loop_ratio_meas_over_pred=l.meas_change.mean() / l.pred_change.mean(),
                 loop_ratio_median_per_loop=float(np.median(l.meas_change / l.pred_change)),
                 loop_ctrl_abs_max=l.ctrl_pred_change.abs().max(),
                 pearson_mean_of_clones_min=m.pearson_pred_vs_meas.min(), pearson_mean_of_clones_median=m.pearson_pred_vs_meas.median(),
                 pearson_mean_of_clones_max=m.pearson_pred_vs_meas.max(), pearson_single_clone_min=cl.min(), pearson_single_clone_max=cl.max(),
                 pearson_ctrl_min=m.pearson_control_vs_meas.min(), pearson_ctrl_max=m.pearson_control_vs_meas.max(),
                 clone_vs_clone_min=cc.min(), clone_vs_clone_max=cc.max(),
                 pred_sd_median=m.pred_change_sd.median(), meas_sd_median=m.meas_change_sd.median())
        for w in ("100kb", "200kb"):
            i = I[(I.model == rl) & (I.unit == "mean_of_clones") & (I.window == w)]
            r[f"ins{w}_n"] = len(i); r[f"ins{w}_sign_agree"] = int(i.sign_agree.sum())
            r[f"ins{w}_pred_mean"] = i.pred_change.mean(); r[f"ins{w}_meas_mean"] = i.meas_change.mean()
            r[f"ins{w}_ctrl_abs_max"] = i.ctrl_pred_change.abs().max()
        r["ins_total_sign_agree"] = f"{r['ins100kb_sign_agree'] + r['ins200kb_sign_agree']}/{r['ins100kb_n'] + r['ins200kb_n']}"
        rows.append(r)
D = pd.DataFrame(rows); D.to_csv(RES / "all12one_fullrd_comparison.csv", index=False, float_format="%.4f")
print(D.T.to_string())
