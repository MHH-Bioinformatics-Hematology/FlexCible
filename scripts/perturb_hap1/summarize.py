import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from common import RES
M = pd.concat([pd.read_csv(RES / f"region{i}_map_scores.csv") for i in (1, 2, 3)])
L = pd.concat([pd.read_csv(RES / f"region{i}_loop_scores.csv") for i in (1, 2, 3)])
I = pd.concat([pd.read_csv(RES / f"region{i}_insulation_scores.csv") for i in (1, 2, 3)])
rows = []
for (rl, rn, dn), g in M.groupby(["model", "region", "deletion"], sort=False):
    mc = g[g.unit == "mean_of_clones"].iloc[0]
    nf = g[g.unit.str.startswith("clone_vs_clone")].pearson_pred_vs_meas
    cl = g[g.unit.str.startswith("clone:")].pearson_pred_vs_meas
    l = L[(L.model == rl) & (L.deletion == dn) & (L.unit == "mean_of_clones") & L.convergent_anchor]
    i = I[(I.model == rl) & (I.deletion == dn) & (I.unit == "mean_of_clones") & (I.window == "100kb")]
    rows.append(dict(model=rl, region=rn, deletion=dn, n_clones=len(cl), pearson_mean_of_clones=mc.pearson_pred_vs_meas,
                     pearson_clone_min=cl.min(), pearson_clone_max=cl.max(), pearson_control_site=mc.pearson_control_vs_meas,
                     clone_vs_clone_pearson_mean=nf.mean() if len(nf) else float("nan"),
                     loops_n=len(l), loops_sign_agree=int(l.sign_agree.sum()), loops_meas_change_mean=l.meas_change.mean(),
                     loops_pred_change_mean=l.pred_change.mean(), loops_control_pred_change_mean=l.ctrl_pred_change.mean(),
                     insulation100kb_sites=len(i), insulation100kb_sign_agree=int(i.sign_agree.sum()),
                     insulation100kb_meas_change_mean=i.meas_change.mean(), insulation100kb_pred_change_mean=i.pred_change.mean(),
                     insulation100kb_control_pred_change_mean=i.ctrl_pred_change.mean()))
S = pd.DataFrame(rows); S.to_csv(RES / "summary.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(S.round(2).to_string(index=False))
