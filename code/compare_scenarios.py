import glob
import os
import re

import pandas as pd

ORDER = ["allon", "elastictree", "et_hedera", "smartfct", "wang22",
         "ours_core", "ours_net", "ours_joint", "ours_adaptive", "ours_mlp", "oracle"]
rows, extra = [], []
pat = re.compile(r"results_(\w+?)_(\d+)g_(\d+)m$")
dirs = sorted((d for d in glob.glob("results_*_*g_*m") if pat.match(d)),
              key=lambda d: (pat.match(d).group(1), int(pat.match(d).group(2)), int(pat.match(d).group(3))))
for d in dirs:
    f = os.path.join(d, "day_results.csv")
    if not os.path.exists(f):
        continue
    scen, gpus, pmin = pat.match(d).groups()
    df = pd.read_csv(f)
    w = df.period_h / 1000
    df["E_scaleout"] = (df.P_total - df.P_nvlink) * w
    df["E_total"] = df.P_total * w
    df["E_core"] = (df.P_spine + df.P_leaf_uplink) * w
    e = df.groupby("policy").E_scaleout.sum()
    tot = df.groupby("policy").E_total.sum()
    core = df.groupby("policy").E_core.sum()
    viol = df.groupby("policy").slo_ok.apply(lambda s: int((~s).sum()))
    for p in ORDER:
        if p in e:
            rows.append(dict(scenario=scen, gpus=int(gpus), period_min=int(pmin), policy=p,
                             scaleout_kWh=round(e[p], 1), saving=f"{(1 - e[p] / e['allon']) * 100:.1f}%",
                             saving_incl_nvlink=f"{(1 - tot[p] / tot['allon']) * 100:.1f}%",
                             spine_uplink_kWh=round(core[p], 1), violation_periods=viol[p]))
    for p, nm in (("ours_adaptive", "CoNap-Sim"), ("ours_mlp", "CoNap")):
        if "oracle" in e and p in e:
            gap = (e["allon"] - e[p]) / (e["allon"] - e["oracle"]) * 100
            print(f"[{scen} {gpus}g {pmin}m] {nm} reaches {gap:.1f}% of the Oracle energy saving, with {viol[p]} SLO-violating periods")
    if "ours_net" in e:
        for p in ("ours_joint", "ours_adaptive"):
            if p in e:
                extra.append(dict(scenario=scen, gpus=int(gpus), period_min=int(pmin), vs_ours_net=p,
                                  extra_saving=f"{(e['ours_net'] - e[p]) / e['ours_net'] * 100:.1f}%",
                                  spine_uplink_reduction=f"{(1 - core[p] / core['ours_net']) * 100:.1f}%",
                                  violation_periods=viol[p]))

pd.set_option("display.width", 220)
if not rows:
    print("No results_<scenario>_<GPUs>g_<period>m/ folders found")
else:
    print("== Policy comparison per scenario, scale and slow-layer period (energy saving based on scale-out network energy) ==")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== Additional gains of the placement-enabled policies over CoNap-Sim w/o placement (ours_net, network only) ==")
    print(pd.DataFrame(extra).to_string(index=False))
