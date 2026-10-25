from __future__ import annotations

import dataclasses
import itertools
import os

import numpy as np

import netsim as n

POLICIES = {
    "allon": dict(layout="seg", routing="ecmp", pairing="lb"),
    "elastictree": dict(layout="seg", routing="ecmp", pairing="lb"),
    "ours_core": dict(layout="seg", routing="explicit", pairing="lb",
                      rates=(400,), edges=(8,), tune_nvlink=False),
    "ours_net": dict(layout="seg", routing="explicit", pairing="lb",
                     rates=(200, 400), edges=(2, 4, 8), tune_nvlink=True),
    "ours_joint": dict(layout="mixed", routing="explicit", pairing="local",
                       rates=(200, 400), edges=(2, 4, 8), tune_nvlink=True),
    "ours_adaptive": dict(layout="mixed", routing="explicit", pairing="adaptive",
                          rates=(200, 400), edges=(2, 4, 8), tune_nvlink=True),
    "oracle": dict(layout="mixed", routing="explicit", pairing="adaptive",
                   rates=(200, 400), edges=(2, 4, 8), tune_nvlink=False),
    "et_hedera": dict(layout="seg", routing="explicit", pairing="lb"),
    "smartfct": dict(layout="seg", routing="pack", pairing="lb"),
    "wang22": dict(layout="seg", routing="explicit", pairing="lb"),
}

EVALUATOR = os.environ.get("CONAP_EVALUATOR", "conap_evaluator.pt")
LEARNED_POLICIES = {"ours_mlp"}
if os.path.exists(EVALUATOR):
    POLICIES["ours_mlp"] = dict(POLICIES["ours_adaptive"])

SMARTFCT_MARGINS = (0.1, 0.2, 0.3, 0.4, 0.5)
WANG_UTIL = 0.85
WANG_TREND = 0.5

NVLINK_KNOB = False
MIN_SPINES = int(os.environ.get("CONAP_MIN_SPINES", 2))
MIN_EDGE_PORTS = 2
ET_MARGIN = 0.75
FORECAST_ERR = 0.10
ROBUST_K = 1.0
SD_MARGIN = 0.9
BG_EXTRA = 0.03
AR_MARGIN = 0.8
HYSTERESIS = 0.05
SPINE_TOL = 0.05

PROBE_MIN_DUR, PROBE_MIN_KV = 2.0, 300
PROBE_MIN_KV_LOW_EDGE = 800
EVAL_MIN_DUR, EVAL_MIN_KV = 4.0, 600


def window_len(lam, min_dur, min_kv, warmup):
    return max(min_dur, min_kv / max(lam, 1e-6) + warmup)


def with_slo(prm: n.NP, slo: dict | None):
    return dataclasses.replace(prm, **slo) if slo else prm


def up_options(prm: n.NP):
    return tuple(sorted({max(1, prm.max_up_per_spine // 2), prm.max_up_per_spine}))


def candidates(prm: n.NP, nvl_p="18p/4sw", nvl_d="18p/4sw", rates=(200, 400), edges=(2, 4, 8)):
    w_min = min(MIN_SPINES, prm.n_spine)
    out = []
    for ns, up, r, e in itertools.product(range(w_min, prm.n_spine + 1), up_options(prm), rates, edges):
        if e >= MIN_EDGE_PORTS:
            out.append(n.NetCfg(ns, up, r, e, nvl_p, nvl_d))
    return out


def mean_kv_bytes(prm: n.NP):
    return prm.lin_median * np.exp(prm.lin_sigma ** 2 / 2) * prm.kv_bytes_per_token


def leaf_egress_demand(prm: n.NP, lam, bg_rate, layout):
    kv = lam * mean_kv_bytes(prm)
    if layout == "seg":
        kv_leaf = kv / (prm.n_leaf // 2)
    else:
        kv_leaf = 0.15 * kv / prm.n_leaf
    return bg_rate / prm.n_leaf + kv_leaf


def uplink_capacity(cfg: n.NetCfg):
    return cfg.n_spine * cfg.up_per_spine * cfg.rate / 8 * 1e9


def forecast(lam_true, bg_true, hour, err=FORECAST_ERR):
    rng = np.random.default_rng(10_000 + hour)
    return lam_true * (1 + err * rng.standard_normal()), bg_true * (1 + err * rng.standard_normal())


def robust_load(lam_hat, bg_hat):
    f = 1 + ROBUST_K * FORECAST_ERR
    return lam_hat * f, bg_hat * f


def robust_ok(prm: n.NP, r, cfg: n.NetCfg):
    ok = (r["kv_sd_p99"] <= prm.slo_kv_sd_p99 * SD_MARGIN
          and r["bg_tput"] >= prm.slo_bg_tput + BG_EXTRA
          and r["ar_pf_p99"] <= prm.slo_ar_prefill_p99
          and r["ar_dec_p99"] <= prm.slo_ar_decode_p99 * AR_MARGIN)
    if cfg.edge_ports < 8:
        ok = ok and r["kv_sd_p995"] <= prm.slo_kv_sd_p99 * SD_MARGIN
    return ok


def prefill_batch_tokens_max(prm: n.NP):
    return max(prm.prefill_token_budget, prm.lin_max)


def nvl_prefill_worst_case(prm: n.NP):
    nbytes = prefill_batch_tokens_max(prm) * prm.hidden * 2
    order = sorted(n.NVL_CONFIGS, key=lambda k: n.NVL_CONFIGS[k][0])
    return next((k for k in order if n.allreduce_cct(nbytes, k, prm) <= prm.slo_ar_prefill_p99), "18p/4sw")


def power(cfg, prm, policy):
    pw = n.network_power(cfg, prm, POLICIES[policy]["layout"])
    return pw["total"] - pw["nvlink"]


def probe_seed(hour):
    return 50_000 + hour * 100


def probe(prm, policy, hour, cfg, lam, bg, seed, log, tag="search"):
    pol = POLICIES[policy]
    min_kv = PROBE_MIN_KV if cfg.edge_ports >= 8 else PROBE_MIN_KV_LOW_EDGE
    dur = window_len(lam, PROBE_MIN_DUR, min_kv, prm.warmup)
    r = n.simulate_window(prm, cfg, pol["layout"], pol["routing"], pol["pairing"], lam, bg, dur, seed)
    log.append(dict(policy=policy, hour=hour, tag=tag, lam=lam, bg=bg, layout=pol["layout"],
                    routing=pol["routing"], n_spine=cfg.n_spine, up_per_spine=cfg.up_per_spine,
                    rate=cfg.rate, edge_ports=cfg.edge_ports, nvl_p=cfg.nvl_p, nvl_d=cfg.nvl_d,
                    power=power(cfg, prm, policy), duration=dur, passed=robust_ok(prm, r, cfg),
                    **{k: v for k, v in r.items() if not k.endswith("_all")}))
    return r


def decide_allon(prm, hour, lam, bg):
    return n.full_cfg(prm), []


def decide_elastictree(prm, hour, lam_prev, bg_prev):
    need = leaf_egress_demand(prm, lam_prev, bg_prev, "seg") / ET_MARGIN
    cands = [c for c in candidates(prm, rates=(400,), edges=(8,)) if uplink_capacity(c) >= need]
    if not cands:
        return n.full_cfg(prm), []
    best = min(cands, key=lambda c: power(c, prm, "elastictree"))
    return best, []


def eval_seed(k):
    return 1000 + k


def cheapest_by_capacity(prm, need, policy, rates=(400,), edges=(8,), margin=0.0):
    cands = [dataclasses.replace(c, pack_margin=margin)
             for c in candidates(prm, rates=rates, edges=edges) if uplink_capacity(c) >= need]
    if not cands:
        return dataclasses.replace(n.full_cfg(prm), pack_margin=margin)
    return min(cands, key=lambda c: power(c, prm, policy))


def decide_et_hedera(prm, hour, lam_prev, bg_prev):
    need = leaf_egress_demand(prm, lam_prev, bg_prev, "seg") / ET_MARGIN
    return cheapest_by_capacity(prm, need, "et_hedera"), []


def decide_smartfct(prm, hour, lam_prev, bg_prev):
    log = []
    kp = hour - 1
    for m in SMARTFCT_MARGINS:
        need = leaf_egress_demand(prm, lam_prev, bg_prev, "seg") / (1 - m)
        cfg = cheapest_by_capacity(prm, need, "smartfct", margin=m)
        r = probe(prm, "smartfct", hour, cfg, lam_prev, bg_prev, eval_seed(kp), log, tag=f"margin={m}")
        if n.slo_check(prm, r):
            return cfg, log
    return dataclasses.replace(n.full_cfg(prm), pack_margin=SMARTFCT_MARGINS[-1]), log


def decide_wang22(prm, hour, lam_hist, bg_hist):
    (l2, l1), (b2, b1) = lam_hist, bg_hist
    lam_f = max(0.0, l1 + WANG_TREND * (l1 - l2))
    bg_f = max(0.0, b1 + WANG_TREND * (b1 - b2))
    need = leaf_egress_demand(prm, lam_f, bg_f, "seg") / WANG_UTIL
    return cheapest_by_capacity(prm, need, "wang22"), []


def decide_oracle(prm, hour, lam_true, bg_true):
    pol = POLICIES["oracle"]
    log = []
    need = leaf_egress_demand(prm, lam_true, bg_true, pol["layout"])
    cands = [c for c in candidates(prm, rates=pol["rates"], edges=pol["edges"]) if uplink_capacity(c) >= need]
    cands.sort(key=lambda c: power(c, prm, "oracle"))
    dur = window_len(lam_true, EVAL_MIN_DUR, EVAL_MIN_KV, prm.warmup)
    for c in cands:
        r = n.simulate_window(prm, c, pol["layout"], pol["routing"], pol["pairing"],
                              lam_true, bg_true, dur, seed=eval_seed(hour))
        if n.slo_check(prm, r):
            return c, log
    return n.full_cfg(prm), log


def decide_ours(prm, hour, lam_hat, bg_hat, policy):
    pol = POLICIES[policy]
    lam_u, bg_u = robust_load(lam_hat, bg_hat)
    log = []
    seed = probe_seed(hour)

    nvl_p = nvl_d = "18p/4sw"
    if pol["tune_nvlink"] and NVLINK_KNOB:
        nvl_p = nvl_prefill_worst_case(prm)
        r0 = probe(prm, policy, hour, n.full_cfg(prm), lam_u, bg_u, seed, log, tag="nvlink")
        nvl_order = sorted(n.NVL_CONFIGS, key=lambda k: n.NVL_CONFIGS[k][0])
        nvl_d = next((k for k in nvl_order if r0["ar_dec_all"][k] <= prm.slo_ar_decode_p99 * AR_MARGIN), "18p/4sw")

    need = leaf_egress_demand(prm, lam_u, bg_u, pol["layout"])
    cores = [c for c in candidates(prm, nvl_p, nvl_d, pol["rates"], (8,)) if uplink_capacity(c) >= need]
    cores.sort(key=lambda c: power(c, prm, policy))
    core = None
    for c in cores:
        r = probe(prm, policy, hour, c, lam_u, bg_u, seed, log)
        if robust_ok(prm, r, c):
            core = c
            break
    if core is None:
        return n.NetCfg(prm.n_spine, prm.max_up_per_spine, 400, 8, nvl_p, nvl_d), log
    p_core = power(core, prm, policy)
    fewer = [c for c in cores if c.n_spine < core.n_spine and power(c, prm, policy) <= p_core * (1 + SPINE_TOL)]
    fewer.sort(key=lambda c: (c.n_spine, power(c, prm, policy)))
    for c in fewer:
        r = probe(prm, policy, hour, c, lam_u, bg_u, seed, log, tag="fewer_spines")
        if robust_ok(prm, r, c):
            core = c
            break
    for e in sorted(x for x in pol["edges"] if x < 8):
        c = dataclasses.replace(core, edge_ports=e)
        r = probe(prm, policy, hour, c, lam_u, bg_u, seed, log, tag="edge")
        if robust_ok(prm, r, c):
            return c, log
    return core, log


def decide_ours_learned(prm, scen, gpus, hour, lam_hat, bg_hat, policy):
    import learned_eval as le
    pol = POLICIES[policy]
    lam_u, bg_u = robust_load(lam_hat, bg_hat)
    need = leaf_egress_demand(prm, lam_u, bg_u, pol["layout"])
    cands = [c for c in candidates(prm, "18p/4sw", "18p/4sw", pol["rates"], pol["edges"]) if uplink_capacity(c) >= need]
    log = []
    if not cands:
        return n.full_cfg(prm), log
    ok, up99, lowbg, t_inf = le.judge(EVALUATOR, scen, gpus, prm, cands, pol, lam_u, bg_u)
    log.append(dict(policy=policy, hour=hour, tag="learned", n_candidates=len(cands), n_ok=int(sum(ok)),
                    infer_seconds=t_inf))
    feas = [c for c, o in zip(cands, ok) if o]
    if not feas:
        return n.full_cfg(prm), log
    p_min = min(power(c, prm, policy) for c in feas)
    near = [c for c in feas if power(c, prm, policy) <= p_min * (1 + SPINE_TOL)]
    near.sort(key=lambda c: (c.n_spine, power(c, prm, policy)))
    return near[0], log


def learned_ok(prm, scen, gpus, policy, cfg, lam_u, bg_u):
    import learned_eval as le
    ok, _, _, _ = le.judge(EVALUATOR, scen, gpus, prm, [cfg], POLICIES[policy], lam_u, bg_u)
    return ok[0]


def apply_hysteresis(prm, policy, decisions, lam_true, bg_true, hours, scen=None, gpus=None):
    out, log, kept = {}, [], 0
    for h in hours:
        new = decisions[h]
        if h == hours[0] or out[h - 1] == new:
            out[h] = new
            continue
        prev = out[h - 1]
        p_prev, p_new = power(prev, prm, policy), power(new, prm, policy)
        if p_prev <= p_new * (1 + HYSTERESIS):
            lam_hat, bg_hat = forecast(lam_true[h], bg_true[h], h)
            lam_u, bg_u = robust_load(lam_hat, bg_hat)
            if policy in LEARNED_POLICIES:
                still_ok = learned_ok(prm, scen, gpus, policy, prev, lam_u, bg_u)
            else:
                r = probe(prm, policy, h, prev, lam_u, bg_u, probe_seed(h), log, tag="hysteresis")
                still_ok = robust_ok(prm, r, prev)
            if still_ok:
                out[h] = prev
                kept += 1
                continue
        out[h] = new
    return out, log, kept


def no_abort(prm):
    return dataclasses.replace(prm, abort_sd=float("inf"), abort_bg=0.0, drain_max=float("inf"),
                               max_active_flows=10 ** 9)


def evaluate(prm, cfg, policy, hour, lam_true, bg_true):
    prm = no_abort(prm)
    pol = POLICIES[policy]
    dur = window_len(lam_true, EVAL_MIN_DUR, EVAL_MIN_KV, prm.warmup)
    r = n.simulate_window(prm, cfg, pol["layout"], pol["routing"], pol["pairing"],
                          lam_true, bg_true, dur, seed=eval_seed(hour))
    pw = n.network_power(cfg, prm, pol["layout"])
    return dict(policy=policy, hour=hour, lam=lam_true, bg=bg_true, cfg=cfg.label(),
                n_spine=cfg.n_spine, up_per_spine=cfg.up_per_spine, rate=cfg.rate,
                edge_ports=cfg.edge_ports, nvl_p=cfg.nvl_p, nvl_d=cfg.nvl_d, pack_margin=cfg.pack_margin,
                duration=dur,
                **{f"P_{k}": v for k, v in pw.items()},
                **{k: v for k, v in r.items() if not k.endswith("_all")},
                slo_kv_sd=prm.slo_kv_sd_p99, slo_bg=prm.slo_bg_tput,
                slo_ok=n.slo_check(prm, r))
