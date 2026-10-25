import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import controller as ctl
import netsim as n

SCALES_TRAIN = (128, 256, 512, 1024)
MODES = [
    (("seg", "lb", "explicit"), 0.30),
    (("seg", "lb", "ecmp"), 0.15),
    (("mixed", "local", "explicit"), 0.15),
    (("mixed", "adaptive", "explicit"), 0.30),
    (("seg", "lb", "pack"), 0.10),
]


def sample_group(gid, scales=SCALES_TRAIN):
    rng = np.random.default_rng(900_000 + gid)
    gpus = int(rng.choice(scales))
    scen = str(rng.choice(list(n.SCENARIOS)))
    prm = n.scenario_prm(scen, gpus)
    lam_peak, bg_peak = n.scenario_load(scen, prm)
    hour = float(rng.uniform(0, 24))
    lam, bg = n.diurnal_at(hour, lam_peak, bg_peak)
    scale = float(rng.uniform(0.7, 1.3))
    lam, bg = float(lam) * scale, float(bg) * scale
    modes, w = zip(*MODES)
    layout, pairing, routing = modes[rng.choice(len(modes), p=np.array(w) / sum(w))]
    margin = float(rng.choice(ctl.SMARTFCT_MARGINS)) if routing == "pack" else 0.0
    return dict(gid=gid, gpus=gpus, scenario=scen, hour=hour, load_scale=scale, lam=lam, bg=bg,
                layout=layout, pairing=pairing, routing=routing, pack_margin=margin,
                seed=int(rng.integers(1e9)))


def sample_configs(prm, g, k, rng):
    need = ctl.leaf_egress_demand(prm, g["lam"], g["bg"], g["layout"])
    if prm.leaf_bg_w:
        w = np.asarray(prm.leaf_bg_w)
        need += g["bg"] / prm.n_leaf * (w.max() / w.mean() - 1)
    allc = [c for c in ctl.candidates(prm, rates=(200, 400), edges=(2, 4, 8))]
    ratio = np.array([ctl.uplink_capacity(c) / max(need, 1.0) for c in allc])
    near = [c for c, r in zip(allc, ratio) if 0.8 <= r <= 4.0]
    out = []
    for _ in range(k):
        pool = near if (near and rng.random() < 0.8) else allc
        c = pool[rng.integers(len(pool))]
        out.append(n.NetCfg(c.n_spine, c.up_per_spine, c.rate, c.edge_ports, pack_margin=g["pack_margin"]))
    return list(dict.fromkeys(out))


def run_group(args):
    gid, k, scales, asym_frac = args
    g = sample_group(gid, scales)
    prm = n.scenario_prm(g["scenario"], g["gpus"])
    rng_a = np.random.default_rng(2_900_000 + gid)
    if asym_frac > 0 and rng_a.random() < asym_frac:
        prm = n.asym_prm(prm, rng_a)
    f_str, w_str = n.asym_to_str(prm)
    g.update(asym=bool(prm.failed_up or prm.leaf_bg_w), failed_up=f_str, leaf_bg_w=w_str,
             n_failed=len(prm.failed_up), n_hot=int(sum(1 for x in prm.leaf_bg_w if x > 1)))
    rng = np.random.default_rng(1_900_000 + gid)
    rows = []
    dur = ctl.window_len(g["lam"], ctl.PROBE_MIN_DUR, ctl.PROBE_MIN_KV, prm.warmup)
    for cfg in sample_configs(prm, g, k, rng):
        t0 = time.time()
        r = n.simulate_window(prm, cfg, g["layout"], g["routing"], g["pairing"], g["lam"], g["bg"], dur, g["seed"])
        pw = n.network_power(cfg, prm, g["layout"])
        rows.append(dict(**g, n_spine=cfg.n_spine, up_per_spine=cfg.up_per_spine, rate=cfg.rate,
                         edge_ports=cfg.edge_ports, P_scaleout=pw["total"] - pw["nvlink"], duration=dur,
                         sim_seconds=time.time() - t0,
                         **{k_: r[k_] for k_ in ("kv_sd_p50", "kv_sd_p99", "kv_sd_p995", "bg_tput",
                                                 "kv_local_frac", "n_kv", "kv_cct_p99", "truncated")}))
    return rows


if __name__ == "__main__":
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out = sys.argv[4] if len(sys.argv) > 4 else "gnn_data.csv"
    scales = tuple(int(x) for x in sys.argv[5].split(",")) if len(sys.argv) > 5 else SCALES_TRAIN
    gid0 = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    asym_frac = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    assert all(s in n.SCALES for s in scales), f"GPU scale must be one of {list(n.SCALES)}"
    t0 = time.time()
    rows = []
    srv = [s // 8 for s in scales]
    print(f"Generating {n_groups} groups x up to {k} configurations, scales {srv} servers, {n_proc} processes, "
          f"asymmetric fraction {asym_frac:.0%} -> {out}", flush=True)
    tasks = [(gid0 + g, k, scales, asym_frac) for g in range(n_groups)]
    with Pool(n_proc) as pool:
        for i, rs in enumerate(pool.imap_unordered(run_group, tasks, chunksize=1), 1):
            rows.extend(rs)
            if i % max(1, n_groups // 20) == 0 or i == n_groups:
                el = time.time() - t0
                print(f"  {i}/{n_groups} groups, {len(rows)} samples, elapsed {el/60:.1f} min, "
                      f"est. remaining {el/i*(n_groups-i)/60:.1f} min (rough: group runtimes vary widely)", flush=True)
                pd.DataFrame(rows).to_csv(out + ".partial", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if os.path.exists(out + ".partial"):
        os.remove(out + ".partial")
    print(f"Samples hitting early overload abort: {int(df.truncated.sum())} / {len(df)}; asymmetric groups: {int(df.groupby('gid').asym.first().sum())} / {df.gid.nunique()}")
    by_srv = {f"{k // 8} servers": v for k, v in df.groupby('gpus').size().to_dict().items()}
    print(f"Done: {len(df)} samples, {df.gid.nunique()} groups; samples per scale: {by_srv}")
