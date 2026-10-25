import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MAIN = ["allon", "elastictree", "et_hedera", "smartfct", "wang22", "ours_mlp", "oracle"]
ABL = ["ours_mlp", "ours_adaptive", "ours_joint", "ours_net", "ours_core", "et_hedera"]
LAB = {"allon": "All-on", "elastictree": "ElasticTree", "et_hedera": "ET+Hedera", "smartfct": "SmartFCT-like",
       "wang22": "Wang22-like", "ours_mlp": "CoNap", "oracle": "Oracle", "ours_adaptive": "CoNap-Sim",
       "ours_joint": "  w/o adaptive pairing", "ours_net": "  w/o placement",
       "ours_core": "  w/o placement, rate, NIC"}


def load(argv):
    if len(argv) > 2 and argv[1] == "--from-runs":
        d = argv[2]
        return pd.concat([pd.read_csv(os.path.join(d, f"results_{s}_1024g_60m", "day_results.csv"))
                          for s in ("base", "kvheavy")], ignore_index=True)
    path = argv[1] if len(argv) > 1 else os.path.join(ROOT, "results", "hourly_1024gpu_60min.csv")
    return pd.read_csv(path)


def summarize(h):
    h = h.copy()
    dur = h["period_h"] if "period_h" in h else 1.0
    h["E"] = (h.P_total - h.P_nvlink) / 1000 * dur
    h["E_wu"] = (h.P_spine + h.P_leaf_uplink) / 1000 * dur
    h["E_nic"] = h.P_edge / 1000 * dur
    h["spines"] = h.cfg.str.extract(r"^(\d+)sp")[0].astype(int)
    h["viol"] = ~h.slo_ok.astype(str).str.lower().eq("true")
    g = h.groupby(["scenario", "policy"]).agg(energy_kwh=("E", "sum"), wu_kwh=("E_wu", "sum"), nic_kwh=("E_nic", "sum"),
                                              violations=("viol", "sum"), worst_sd=("kv_sd_p99", "max"),
                                              min_bg=("bg_tput", "min"), mean_spines=("spines", "mean"),
                                              periods=("viol", "size")).reset_index()
    allon = g[g.policy == "allon"].set_index("scenario").energy_kwh
    g["saving_pct"] = 100 * (1 - g.energy_kwh / g.scenario.map(allon))
    return g


def show(g, pols, cols, title):
    print(f"\n=== {title} ===")
    rows = []
    for sc in ("base", "kvheavy"):
        for p in pols:
            r = g[(g.scenario == sc) & (g.policy == p)]
            if len(r):
                rows.append(dict(scenario=sc, method=LAB.get(p, p), **{c: r.iloc[0][c] for c in cols}))
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(t.round(dict(energy_kwh=1, wu_kwh=1, nic_kwh=1, saving_pct=1, worst_sd=2, min_bg=3, mean_spines=2)).to_string(index=False))
    return t


if __name__ == "__main__":
    g = summarize(load(sys.argv))
    assert (g.periods == 24).all(), "every policy must have 24 periods"
    main = show(g, MAIN, ["energy_kwh", "saving_pct", "violations", "worst_sd", "min_bg", "mean_spines"],
                "Table 3: full-day results (1024 GPUs, 60-min slow layer)")
    abl = show(g, ABL, ["saving_pct", "wu_kwh", "violations", "worst_sd", "mean_spines"], "Table 4: ablation")
    out = sys.argv[2] if (len(sys.argv) > 2 and sys.argv[1] == "--from-runs") else os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    main.to_csv(os.path.join(out, "table_main.csv"), index=False)
    abl.to_csv(os.path.join(out, "table_ablation.csv"), index=False)
    for sc in ("base", "kvheavy"):
        o = g[(g.scenario == sc) & (g.policy == "oracle")].saving_pct.iloc[0]
        c = g[(g.scenario == sc) & (g.policy == "ours_mlp")].saving_pct.iloc[0]
        print(f"{sc}: CoNap reaches {100 * c / o:.1f}% of the oracle saving")
