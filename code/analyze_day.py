import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netsim as n

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def save(fig, name):
    fig.savefig(P(name + ".pdf"))
    plt.close(fig)
ORDER = ["allon", "elastictree", "et_hedera", "smartfct", "wang22",
         "ours_core", "ours_net", "ours_joint", "ours_adaptive", "ours_mlp", "oracle"]
NAMES = {"allon": "All-on", "elastictree": "ElasticTree", "et_hedera": "ET+Hedera",
         "smartfct": "SmartFCT-like", "wang22": "Wang22-like",
         "ours_core": "CoNap-Sim w/o placement, rate, NIC", "ours_net": "CoNap-Sim w/o placement",
         "ours_joint": "CoNap-Sim w/o adaptive pairing", "ours_adaptive": "CoNap-Sim", "ours_mlp": "CoNap",
         "oracle": "Oracle"}
BASELINES = {"elastictree", "et_hedera", "smartfct", "wang22"}
PLOT_ALL = ["allon", "elastictree", "et_hedera", "smartfct", "wang22", "ours_adaptive", "oracle"]
STYLE = {"allon": ("#8a8984", "-"), "elastictree": ("#eb6834", "--"), "et_hedera": ("#e34948", "--"),
         "smartfct": ("#e87ba4", "--"), "wang22": ("#eda100", "--"),
         "ours_core": ("#6250d6", "-"), "ours_net": ("#2a78d6", "-"), "ours_joint": ("#1baf7a", "-"),
         "ours_adaptive": ("#008300", "-"), "ours_mlp": ("#1baf7a", "-."), "oracle": ("#0b0b0b", ":")}
C = ["#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#eda100", "#e87ba4"]

SCEN = sys.argv[1] if len(sys.argv) > 1 else "base"
GPUS = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
PMIN = int(sys.argv[3]) if len(sys.argv) > 3 else 60
OUT = f"results_{SCEN}_{GPUS}g_{PMIN}m"
PRM = n.scenario_prm(SCEN, GPUS)
TAG = f"{SCEN}, {GPUS} GPUs, {PMIN}-min slow layer"


def P(name):
    return os.path.join(OUT, name)


df = pd.read_csv(P("day_results.csv"))
df = df[df.policy.isin(ORDER)]
ORDER = [p for p in ORDER if p in set(df.policy)]
PLOT = [p for p in PLOT_ALL if p in ORDER]
SLO_SD, SLO_BG = df.slo_kv_sd.iloc[0], df.slo_bg.iloc[0]

df["P_scaleout"] = df.P_total - df.P_nvlink
for col in ["P_scaleout", "P_total", "P_spine", "P_leaf_uplink", "P_edge", "P_nvlink", "P_leaf_base"]:
    df["E_" + col[2:]] = df[col] * df.period_h / 1000

base_e = df[df.policy == "allon"].E_scaleout.sum()
base_tot = df[df.policy == "allon"].E_total.sum()

rows = []
for p in ORDER:
    q = df[df.policy == p].sort_values("period")
    e, tot = q.E_scaleout.sum(), q.E_total.sum()
    viol = int((~q.slo_ok).sum())
    rows.append(dict(
        policy=NAMES[p], scaleout_kWh=round(e, 1), saving_pct=round((1 - e / base_e) * 100, 1),
        saving_incl_nvlink_pct=round((1 - tot / base_tot) * 100, 1),
        spine_kWh=round(q.E_spine.sum(), 1), uplink_kWh=round(q.E_leaf_uplink.sum(), 1),
        edge_kWh=round(q.E_edge.sum(), 1),
        violation_periods=viol, violation_hours=round(viol * PMIN / 60, 2),
        worst_kv_sd_p99=round(q.kv_sd_p99.max(), 2), min_bg_tput=round(q.bg_tput.min(), 3),
        spine_transitions=int(np.abs(np.diff(q.n_spine.values)).sum()),
        cfg_changes=int((q.cfg.values[1:] != q.cfg.values[:-1]).sum()),
        avg_spines_on=round(q.n_spine.mean(), 2),
        kv_local_frac=round(q.kv_local_frac.mean(), 2) if "kv_local_frac" in q else np.nan,
    ))
summ = pd.DataFrame(rows)
summ.to_csv(P("day_summary.csv"), index=False)

pd.set_option("display.width", 240)
print(f"Scenario: {SCEN}, {GPUS} GPUs, slow-layer period {PMIN} min ({df.period.nunique()} periods)")
print(f"SLO: KV P99 slowdown ≤ {SLO_SD}, background throughput ratio ≥ {SLO_BG} (calibrated against all-on)\n")
print("== Daily summary (energy saving based on scale-out network energy, excluding the always-on NVLink) ==")
print(summ.to_string(index=False))

if df.period.nunique() <= 24:
    print("\n== Configuration chosen by each policy in each period ==")
    piv = df.pivot(index="hour", columns="policy", values="cfg")[ORDER]
    print(piv.to_string())
else:
    print(f"\n({df.period.nunique()} periods; see {OUT}/day_results.csv for the per-period configurations)")

viol = df[(~df.slo_ok) & (~df.policy.isin(BASELINES))]
counts = ", ".join(f"{NAMES[p]} {int((~df[df.policy == p].slo_ok).sum())}" for p in ORDER if p in BASELINES)
print(f"\n== SLO violations (baselines: number of violating periods only: {counts}) ==")
if len(viol):
    v = viol[["policy", "hour", "cfg", "kv_sd_p99", "bg_tput", "ar_pf_p99", "ar_dec_p99"]].copy()
    v["violated"] = [",".join(x for x, bad in [
        ("KV", r.kv_sd_p99 > SLO_SD), ("bg", r.bg_tput < SLO_BG),
        ("AR-P", r.ar_pf_p99 > PRM.slo_ar_prefill_p99), ("AR-D", r.ar_dec_p99 > PRM.slo_ar_decode_p99)] if bad)
        for r in v.itertuples()]
    print(v.to_string(index=False))
else:
    print("All non-baseline policies meet the SLO")

fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 2]))
q = df[df.policy == "allon"].sort_values("period")
a0.plot(q.hour, q.lam / q.lam.max(), color=C[0], linewidth=2, label="Inference load (period mean, norm.)")
a0.plot(q.hour, q.bg / q.bg.max(), color=C[1], linewidth=2, label="Background traffic (period mean, norm.)")
a0.set_ylabel("Normalized")
a0.set_ylim(0, 1.1)
a0.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
a0.set_title(f"Traffic over the day and scale-out network power  [{TAG}]", loc="left")
for i, p in enumerate(PLOT):
    q = df[df.policy == p].sort_values("period")
    col, ls = STYLE[p]
    a1.step(q.hour, q.P_scaleout / 1000, where="post", color=col, linestyle=ls, linewidth=1.8, label=NAMES[p])
    bad = q[~q.slo_ok]
    a1.scatter(bad.hour + q.period_h.iloc[0] / 2, bad.P_scaleout / 1000, s=60, facecolors="none",
               edgecolors="#e34948", linewidths=1.4, zorder=5)
a1.set_ylabel("Scale-out network power (kW)")
a1.set_xlabel("Hour of day  (red ring = SLO violated in that period)")
a1.set_ylim(0, None)
a1.set_xticks(range(0, 25, 2))
a1.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
fig.tight_layout()
save(fig, "fig1_power_vs_hour")

fig, ax = plt.subplots(figsize=(9.2, 4.0))
for i, p in enumerate(PLOT):
    q = df[df.policy == p].sort_values("period")
    col, ls = STYLE[p]
    ax.plot(q.hour + q.period_h / 2, q.kv_sd_p99, color=col, linestyle=ls, linewidth=1.6, marker="o",
            markersize=3, label=NAMES[p])
ax.axhline(SLO_SD, color="#52514e", linestyle="--", linewidth=1)
ax.text(24, SLO_SD, " SLO", va="center", fontsize=9, color="#52514e")
ax.set_xlabel("Hour of day")
ax.set_ylabel("KV coflow P99 slowdown")
ax.set_title(f"KV-transfer coflow tail slowdown (actual / ideal CCT)  [{TAG}]", loc="left")
ax.set_xticks(range(0, 25, 2))
ax.set_ylim(0, None)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
fig.tight_layout()
save(fig, "fig2_kv_slowdown_vs_hour")

parts = [("spine_kWh", "Spine switches"), ("uplink_kWh", "Leaf uplink ports"), ("edge_kWh", "NIC + leaf edge ports")]
fig, ax = plt.subplots(figsize=(10.5, 4.8))
sp = summ[summ.policy.isin([NAMES[p] for p in PLOT])].reset_index(drop=True)
x = np.arange(len(PLOT))
leaf_base = df[df.policy == "allon"].E_leaf_base.sum()
bottom = np.full(len(PLOT), leaf_base)
ax.bar(x, bottom, width=0.55, color="#b9b8b2", label="Leaf base (always on)", edgecolor="#fcfcfb", linewidth=2)
for i, (k, lab) in enumerate(parts):
    v = sp[k].values
    ax.bar(x, v, bottom=bottom, width=0.55, color=C[i], label=lab, edgecolor="#fcfcfb", linewidth=2)
    bottom = bottom + v
for xi, (_, r) in zip(x, sp.iterrows()):
    lab = f"{r.scaleout_kWh:.0f} kWh" + (f"\n(−{r.saving_pct:.0f}%)" if r.saving_pct > 0 else "")
    ax.text(xi, bottom[xi] * 1.01, lab, ha="center", va="bottom", fontsize=9)
ax.set_xticks(x, [NAMES[p].replace(" (", "\n(").replace(" + ", "\n+ ") for p in PLOT], fontsize=8)
ax.set_ylabel("Scale-out network energy per day (kWh)")
ax.set_title(f"Daily scale-out network energy (NVLink excluded, always on)  [{TAG}]", loc="left")
ax.set_ylim(0, bottom.max() * 1.15)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig3_energy_breakdown")

fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9.2, 7.4), sharex=True)
for i, p in enumerate(PLOT):
    q = df[df.policy == p].sort_values("period")
    off = (i - len(PLOT) / 2) * 0.05
    col, ls = STYLE[p]
    a0.step(q.hour, q.n_spine + off, where="post", color=col, linestyle=ls, linewidth=1.6, label=NAMES[p])
    a1.step(q.hour, q.edge_ports + off * 2, where="post", color=col, linestyle=ls, linewidth=1.6)
    if p in ("ours_adaptive", "oracle"):
        a2.plot(q.hour + q.period_h / 2, q.kv_local_frac, color=col, linestyle=ls, linewidth=1.6, marker="o",
                markersize=3, label=NAMES[p])
a0.set_ylabel("Spines on")
a0.set_yticks(range(1, PRM.n_spine + 1, max(1, PRM.n_spine // 8)))
a0.set_title(f"Devices kept on (slow layer) and KV pairing (fast layer)  [{TAG}]", loc="left")
a0.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
a1.set_ylabel("NIC ports / server")
a1.set_yticks([2, 4, 8])
a2.set_ylabel("Share of KV\nkept in leaf")
a2.set_ylim(0, 1.05)
a2.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
a2.set_xlabel("Hour of day")
a2.set_xticks(range(0, 25, 2))
fig.tight_layout()
save(fig, "fig4_devices_vs_hour")

print(f"\nWrote {OUT}/day_summary.csv and fig1-fig4 (PDF)")
