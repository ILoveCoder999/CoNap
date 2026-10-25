from __future__ import annotations

import dataclasses
import heapq
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NP:
    n_leaf: int = 4
    servers_per_leaf: int = 4
    gpus_per_server: int = 8
    n_spine: int = 4
    max_up_per_spine: int = 8
    nic_bw: float = 50e9

    p_switch_base: float = 500.0
    p_sw_port: dict = field(default_factory=lambda: {400: 15.0, 200: 9.0})
    p_nic_port: float = 18.0
    p_leaf_edge_port: float = 15.0
    nvl_ports_per_gpu: int = 18
    nvl_port_bw: float = 25e9
    nvl_alpha: float = 1.5e-6
    p_nvl_port_gpu: float = 2.0
    p_nvl_port_sw: float = 1.5
    p_nvswitch_base: float = 30.0

    n_param: float = 70e9
    n_layer: int = 80
    hidden: int = 8192
    kv_bytes_per_token: float = 327_680
    peak_flops: float = 989e12
    mfu_prefill: float = 0.55
    prefill_token_budget: int = 8192
    lin_median: float = 3000
    lin_sigma: float = 0.6
    lin_max: int = 16384
    lout_mean: float = 250
    tpot_est: float = 0.015

    bg_flow_cap: float = 25e9
    bg_size_mean: float = 25e9

    slo_kv_sd_p99: float = 3.0
    slo_bg_tput: float = 0.95
    slo_ar_prefill_p99: float = 1.6e-3
    slo_ar_decode_p99: float = 30e-6

    fluct_on: bool = True
    fluct_n_seg: int = 8
    fluct_sigma: float = 0.25
    fluct_burst_p: float = 0.15
    fluct_burst_x: float = 1.8
    fluct_bg_sigma: float = 0.15

    adapt_u_low: float = 0.3
    adapt_u_high: float = 0.6
    adapt_slack: tuple = (0, 4, 12)

    warmup: float = 0.5
    drain_max: float = 1.0
    max_active_flows: int = 30000
    failed_up: tuple = ()
    leaf_bg_w: tuple = ()
    abort_sd: float = 50.0
    abort_frac: float = 0.02
    abort_bg: float = 0.75


@dataclass(frozen=True)
class NetCfg:
    n_spine: int
    up_per_spine: int
    rate: int
    edge_ports: int
    nvl_p: str = "18p/4sw"
    nvl_d: str = "18p/4sw"
    pack_margin: float = 0.0

    def label(self):
        return f"{self.n_spine}sp×{self.up_per_spine}@{self.rate}G e{self.edge_ports}"


FULL = NetCfg(4, 8, 400, 8)


def full_cfg(prm: "NP"):
    return NetCfg(prm.n_spine, prm.max_up_per_spine, 400, 8)

PORT_PLANE = [i % 4 for i in range(18)]
NVL_CONFIGS = {
    "18p/4sw": (18, (0, 1, 2, 3)),
    "12p/4sw": (12, (0, 1, 2, 3)),
    "10p/2sw": (10, (0, 1)),
    "6p/2sw": (6, (0, 1)),
    "4p/1sw": (4, (0,)),
}


def nvl_active_ports(k, planes):
    by_plane = {p: [i for i in range(18) if PORT_PLANE[i] == p] for p in planes}
    chosen = []
    while len(chosen) < k:
        for p in planes:
            if by_plane[p] and len(chosen) < k:
                chosen.append(by_plane[p].pop(0))
    return chosen


def nvl_effective_bw(cfg_name, prm: NP):
    k, _ = NVL_CONFIGS[cfg_name]
    return k * prm.nvl_port_bw


def allreduce_cct(nbytes, cfg_name, prm: NP, n=8):
    return 2 * (n - 1) * (prm.nvl_alpha + (nbytes / n) / nvl_effective_bw(cfg_name, prm))


def opened_ups(prm, cfg, lf, sp):
    if not prm.failed_up:
        return list(range(cfg.up_per_spine))
    bad = _failed_set(prm)
    return [u for u in range(prm.max_up_per_spine) if (lf, sp, u) not in bad][:cfg.up_per_spine]


_FAILED_CACHE = {}


def _failed_set(prm):
    key = prm.failed_up
    if key not in _FAILED_CACHE:
        _FAILED_CACHE[key] = frozenset(tuple(x) for x in key)
    return _FAILED_CACHE[key]


class Topo:

    def __init__(self, prm: NP):
        self.prm = prm
        self.n_srv = prm.n_leaf * prm.servers_per_leaf
        self.idx = {}
        L = 0
        for s in range(self.n_srv):
            for p in range(prm.gpus_per_server):
                self.idx[("tx", s, p)] = L; L += 1
                self.idx[("rx", s, p)] = L; L += 1
        for lf in range(prm.n_leaf):
            for sp in range(prm.n_spine):
                for u in range(prm.max_up_per_spine):
                    self.idx[("up", lf, sp, u)] = L; L += 1
                    self.idx[("dn", sp, lf, u)] = L; L += 1
        self.L = L

    def leaf_of(self, s):
        return s // self.prm.servers_per_leaf

    def capacity(self, cfg: NetCfg):
        cap = np.zeros(self.L)
        prm = self.prm
        for s in range(self.n_srv):
            for p in range(cfg.edge_ports):
                cap[self.idx[("tx", s, p)]] = prm.nic_bw
                cap[self.idx[("rx", s, p)]] = prm.nic_bw
        r = cfg.rate / 8 * 1e9
        for lf in range(prm.n_leaf):
            for sp in range(cfg.n_spine):
                for u in opened_ups(prm, cfg, lf, sp):
                    cap[self.idx[("up", lf, sp, u)]] = r
                    cap[self.idx[("dn", sp, lf, u)]] = r
        return cap


def network_power(cfg: NetCfg, prm: NP, layout: str):
    n_srv = prm.n_leaf * prm.servers_per_leaf
    port_up = prm.p_sw_port[cfg.rate]
    n_up_per_leaf = cfg.n_spine * cfg.up_per_spine
    spine = cfg.n_spine * (prm.p_switch_base + prm.n_leaf * cfg.up_per_spine * port_up)
    leaf_base = prm.n_leaf * prm.p_switch_base
    leaf_up = prm.n_leaf * n_up_per_leaf * port_up
    edge = n_srv * cfg.edge_ports * (prm.p_nic_port + prm.p_leaf_edge_port)
    roles = server_roles(layout, prm)
    nvl = 0.0
    for s in range(n_srv):
        name = cfg.nvl_p if roles[s] == "P" else cfg.nvl_d
        k, planes = NVL_CONFIGS[name]
        ports = nvl_active_ports(k, planes)
        per_plane = {p: sum(1 for x in ports if PORT_PLANE[x] == p) for p in planes}
        nvl += prm.gpus_per_server * k * prm.p_nvl_port_gpu
        nvl += sum(prm.p_nvswitch_base + prm.gpus_per_server * n * prm.p_nvl_port_sw for n in per_plane.values())
    out = dict(spine=spine, leaf_base=leaf_base, leaf_uplink=leaf_up, edge=edge, nvlink=nvl)
    out["total"] = sum(out.values())
    return out


def server_roles(layout: str, prm: NP):
    n_srv = prm.n_leaf * prm.servers_per_leaf
    roles = {}
    for s in range(n_srv):
        lf, j = divmod(s, prm.servers_per_leaf)
        if layout == "seg":
            roles[s] = "P" if lf < prm.n_leaf // 2 else "D"
        else:
            roles[s] = "P" if j < prm.servers_per_leaf // 2 else "D"
    return roles


class FluidNet:

    def __init__(self, cap):
        self.cap = np.asarray(cap, dtype=float)
        self.t = 0.0
        self.paths: list[np.ndarray] = []
        self.meta: list[tuple] = []
        self.rem = np.zeros(0)
        self.dem = np.zeros(0)
        self.is_bg = np.zeros(0, dtype=bool)
        self.rate = np.zeros(0)
        self.link_cnt = np.zeros(len(cap))
        self.link_load = np.zeros(len(cap))
        self.bg_moved = 0.0
        self.bg_demand_time = 0.0
        self.measure_from = 0.0
        self.measure_to = math.inf

    def _recompute(self):
        n = len(self.paths)
        if n == 0:
            self.rate = np.zeros(0)
            self.link_load = np.zeros(len(self.cap))
            return
        lens = np.fromiter((len(p) for p in self.paths), dtype=np.int64, count=n)
        flow_of = np.repeat(np.arange(n), lens)
        links = np.concatenate(self.paths)
        uniq, inv = np.unique(links, return_inverse=True)
        m = len(uniq)
        cap = self.cap[uniq]
        rem = cap.copy()
        dem = self.dem
        rate = np.zeros(n)
        frozen = np.zeros(n, dtype=bool)
        tol = 1e-9 * np.maximum(cap, 1.0) + 1e-3
        for _ in range(n + 1):
            act = ~frozen
            if not act.any():
                break
            act_pair = act[flow_of]
            cnt = np.bincount(inv[act_pair], minlength=m)
            used = cnt > 0
            share = np.min(rem[used] / cnt[used]) if used.any() else math.inf
            inc = max(min(share, np.min(dem[act] - rate[act])), 0.0)
            rate[act] += inc
            rem -= inc * cnt
            sat = used & (rem <= tol)
            newly = np.zeros(n, dtype=bool)
            newly[flow_of[act_pair & sat[inv]]] = True
            newly |= act & (rate >= dem - 1e-6)
            if not newly.any():
                break
            frozen |= newly
        self.rate = rate
        self.link_load = np.bincount(links, weights=rate[flow_of], minlength=len(self.cap))

    def advance(self, t):
        dt = t - self.t
        if dt > 0 and self.paths:
            self.rem -= self.rate * dt
            eff = min(t, self.measure_to) - max(self.t, self.measure_from)
            if eff > 0 and self.is_bg.any():
                self.bg_moved += float(self.rate[self.is_bg].sum()) * eff
                self.bg_demand_time += float(self.dem[self.is_bg].sum()) * eff
        self.t = max(self.t, t)

    def add(self, t, path, size, demand, meta):
        self.advance(t)
        p = np.asarray(path, dtype=np.int64)
        self.paths.append(p)
        self.meta.append(meta)
        self.rem = np.append(self.rem, size)
        self.dem = np.append(self.dem, demand)
        self.is_bg = np.append(self.is_bg, meta[0] == "bg")
        self.link_cnt[p] += 1
        self._recompute()

    def add_many(self, t, paths, sizes, demands, metas):
        self.advance(t)
        for p in paths:
            p = np.asarray(p, dtype=np.int64)
            self.paths.append(p)
            self.link_cnt[p] += 1
        self.meta.extend(metas)
        self.rem = np.append(self.rem, sizes)
        self.dem = np.append(self.dem, demands)
        self.is_bg = np.append(self.is_bg, [m[0] == "bg" for m in metas])
        self._recompute()

    def next_completion(self):
        if not self.paths:
            return math.inf, -1
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = np.where(self.rate > 0, np.maximum(self.rem, 0) / self.rate, math.inf)
        i = int(np.argmin(tt))
        return self.t + tt[i], i

    def complete(self, i, t):
        self.advance(t)
        path = self.paths.pop(i)
        meta = self.meta.pop(i)
        self.rem = np.delete(self.rem, i)
        self.dem = np.delete(self.dem, i)
        self.is_bg = np.delete(self.is_bg, i)
        self.rate = np.delete(self.rate, i)
        self.link_cnt[path] -= 1
        self._recompute()
        return meta


class Router:

    PACK_FLOW_EST = 25e9

    def __init__(self, topo: Topo, cfg: NetCfg, mode: str, rng, net=None):
        self.topo, self.cfg, self.mode, self.rng, self.net = topo, cfg, mode, rng, net
        I = topo.idx
        prm = topo.prm
        ups = {(lf, sp): opened_ups(prm, cfg, lf, sp) for lf in range(prm.n_leaf) for sp in range(cfg.n_spine)}
        self.up = {(lf, sp): [I[("up", lf, sp, u)] for u in ups[(lf, sp)]] for (lf, sp) in ups}
        self.dn = {(sp, lf): [I[("dn", sp, lf, u)] for u in ups[(lf, sp)]] for (lf, sp) in ups}

    def _spines(self, lf_s, lf_d):
        return [sp for sp in range(self.cfg.n_spine) if self.up[(lf_s, sp)] and self.dn[(sp, lf_d)]]

    def core_path(self, lf_s, lf_d, link_cnt):
        sps = self._spines(lf_s, lf_d)
        if self.mode == "ecmp":
            sp = sps[int(self.rng.integers(len(sps)))]
            ups, dns = self.up[(lf_s, sp)], self.dn[(sp, lf_d)]
            return [ups[int(self.rng.integers(len(ups)))], dns[int(self.rng.integers(len(dns)))]]
        if self.mode == "pack":
            load, cap = self.net.link_load, self.net.cap
            lim = 1.0 - self.cfg.pack_margin
            for sp in sps:
                ups, dns = self.up[(lf_s, sp)], self.dn[(sp, lf_d)]
                u = min(ups, key=lambda x: load[x])
                d = min(dns, key=lambda x: load[x])
                if ((load[u] + self.PACK_FLOW_EST) / cap[u] <= lim and
                        (load[d] + self.PACK_FLOW_EST) / cap[d] <= lim):
                    return [u, d]
            best, bp = math.inf, None
            for sp in sps:
                ups, dns = self.up[(lf_s, sp)], self.dn[(sp, lf_d)]
                u = min(ups, key=lambda x: load[x] / cap[x])
                d = min(dns, key=lambda x: load[x] / cap[x])
                c = max(load[u] / cap[u], load[d] / cap[d])
                if c < best:
                    best, bp = c, [u, d]
            return bp
        best, bp = math.inf, None
        for sp in sps:
            ups, dns = self.up[(lf_s, sp)], self.dn[(sp, lf_d)]
            u = min(ups, key=lambda x: link_cnt[x])
            d = min(dns, key=lambda x: link_cnt[x])
            c = link_cnt[u] + link_cnt[d]
            if c < best:
                best, bp = c, [u, d]
        return bp

    def gpu_path(self, s_src, g, s_dst, link_cnt):
        I = self.topo.idx
        p_s = g % self.cfg.edge_ports
        p_d = g % self.cfg.edge_ports
        path = [I[("tx", s_src, p_s)]]
        lf_s, lf_d = self.topo.leaf_of(s_src), self.topo.leaf_of(s_dst)
        if lf_s != lf_d:
            path += self.core_path(lf_s, lf_d, link_cnt)
        path.append(I[("rx", s_dst, p_d)])
        return path


def prefill_service(tokens, nvl_name, prm: NP):
    t_comp = 2 * prm.n_param * tokens / (prm.gpus_per_server * prm.peak_flops * prm.mfu_prefill)
    t_comm = 2 * prm.n_layer * allreduce_cct(tokens * prm.hidden * 2, nvl_name, prm)
    return t_comp + 0.7 * t_comm


def simulate_window(prm: NP, cfg: NetCfg, layout: str, routing: str, pairing: str,
                    lam: float, bg_rate: float, duration: float, seed: int):
    rng = np.random.default_rng(seed)
    topo = Topo(prm)
    net = FluidNet(topo.capacity(cfg))
    net.measure_from = prm.warmup
    net.measure_to = duration
    router = Router(topo, cfg, routing, rng, net)
    roles = server_roles(layout, prm)
    P_srv = [s for s in roles if roles[s] == "P"]
    D_srv = [s for s in roles if roles[s] == "D"]

    ev, seq = [], 0

    def push(t, kind, data):
        nonlocal seq
        heapq.heappush(ev, (t, seq, kind, data))
        seq += 1

    n_seg = prm.fluct_n_seg if prm.fluct_on else 1
    seg_len = duration / n_seg
    if prm.fluct_on:
        m_inf = rng.lognormal(-prm.fluct_sigma ** 2 / 2, prm.fluct_sigma, n_seg)
        m_inf *= np.where(rng.random(n_seg) < prm.fluct_burst_p, prm.fluct_burst_x, 1.0)
        m_inf /= m_inf.mean()
        m_bg = rng.lognormal(-prm.fluct_bg_sigma ** 2 / 2, prm.fluct_bg_sigma, n_seg)
        m_bg /= m_bg.mean()
    else:
        m_inf = m_bg = np.ones(1)

    def poisson_arrivals(rate_per_seg):
        out = []
        for k, rk in enumerate(rate_per_seg):
            t, t_end = k * seg_len, (k + 1) * seg_len
            while rk > 0:
                t += rng.exponential(1 / rk)
                if t > t_end:
                    break
                out.append(t)
        return out

    for t in poisson_arrivals(lam * m_inf):
        lin = int(np.clip(rng.lognormal(math.log(prm.lin_median), prm.lin_sigma), 256, prm.lin_max))
        lout = max(16, int(rng.exponential(prm.lout_mean)))
        push(t, "arr", (lin, lout))
    for t in poisson_arrivals(bg_rate / prm.bg_size_mean * m_bg):
        push(t, "bg", None)

    pq = {s: [] for s in P_srv}
    p_busy = {s: False for s in P_srv}
    d_active = {s: 0 for s in D_srv}
    coflows = {}
    kv_cct, kv_fct, kv_sd = [], [], []
    pf_batch_tokens, dec_batch = [], []
    cid = 0

    def start_prefill(s, t):
        nonlocal cid
        if p_busy[s] or not pq[s]:
            return
        batch, tok = [], 0
        while pq[s] and (not batch or tok + pq[s][0][0] <= prm.prefill_token_budget):
            r = pq[s].pop(0)
            batch.append(r)
            tok += r[0]
        p_busy[s] = True
        if t >= prm.warmup:
            pf_batch_tokens.append(tok)
        push(t + prefill_service(tok, cfg.nvl_p, prm), "pf_done", (s, batch))

    local_D = {lf: [d for d in D_srv if topo.leaf_of(d) == lf] for lf in range(prm.n_leaf)}
    leaf_up = {lf: np.array([x for (l, sp), lst in router.up.items() if l == lf for x in lst], dtype=np.int64)
               for lf in range(prm.n_leaf)}
    leaf_dn = {lf: np.array([x for (sp, l), lst in router.dn.items() if l == lf for x in lst], dtype=np.int64)
               for lf in range(prm.n_leaf)}
    n_local_pick = [0, 0]

    def util(idx):
        c = net.cap[idx].sum()
        return float(net.link_load[idx].sum() / c) if c > 0 else 1.0

    def pick_decode(s_src):
        lf = topo.leaf_of(s_src)
        g_best = min(D_srv, key=lambda d: d_active[d])
        local = local_D[lf]
        if pairing == "lb" or not local:
            choice = g_best
        else:
            l_best = min(local, key=lambda d: d_active[d])
            if pairing == "local":
                slack = 4
            else:
                u = max(util(leaf_up[lf]), util(leaf_dn[topo.leaf_of(g_best)]))
                slack = (prm.adapt_slack[0] if u < prm.adapt_u_low else
                         prm.adapt_slack[1] if u < prm.adapt_u_high else prm.adapt_slack[2])
            choice = l_best if d_active[l_best] <= d_active[g_best] + slack else g_best
        n_local_pick[0 if topo.leaf_of(choice) == lf else 1] += 1
        return choice

    t_cut = None
    n_iter, n_bad_done = 0, [0]
    n_kv_expect = lam * max(duration - prm.warmup, 1e-9)
    while True:
        t_net, i_net = net.next_completion()
        t_ev = ev[0][0] if ev else math.inf
        if t_net == math.inf and t_ev == math.inf:
            break
        if min(t_net, t_ev) > duration and not coflows:
            break
        if min(t_net, t_ev) > duration + prm.drain_max:
            t_cut = duration + prm.drain_max
            break
        if len(net.rem) > prm.max_active_flows:
            t_cut = max(net.t, prm.warmup)
            break
        n_iter += 1
        if n_iter % 200 == 0 and net.t > prm.warmup:
            tn = net.t
            n_bad = n_bad_done[0] + sum(1 for cf in coflows.values()
                                        if cf["t0"] >= prm.warmup and (tn - cf["t0"]) / cf["ideal"] > prm.abort_sd)
            if n_bad >= max(20, prm.abort_frac * n_kv_expect):
                t_cut = tn
                break
            if (tn > prm.warmup + 0.25 * (duration - prm.warmup) and net.bg_demand_time > 0
                    and net.bg_moved / net.bg_demand_time < prm.abort_bg):
                t_cut = tn
                break
        if t_net <= t_ev:
            meta = net.complete(i_net, t_net)
            kind, c, t0, size = meta
            if kind == "kv":
                cf = coflows[c]
                cf["left"] -= 1
                if cf["t0"] >= prm.warmup:
                    kv_fct.append(t_net - t0)
                if cf["left"] == 0:
                    if cf["t0"] >= prm.warmup:
                        kv_cct.append(t_net - cf["t0"])
                        kv_sd.append((t_net - cf["t0"]) / cf["ideal"])
                        if kv_sd[-1] > prm.abort_sd:
                            n_bad_done[0] += 1
                    del coflows[c]
            continue
        t_now, _, kind, data = heapq.heappop(ev)
        if t_now > duration and kind in ("arr", "bg"):
            continue
        if kind == "arr":
            s = min(P_srv, key=lambda x: (p_busy[x], sum(r[0] for r in pq[x])))
            pq[s].append(data)
            start_prefill(s, t_now)
        elif kind == "pf_done":
            s, batch = data
            p_busy[s] = False
            batch_tok = sum(lin for lin, _ in batch)
            ideal = batch_tok * prm.kv_bytes_per_token / prm.gpus_per_server / prm.nic_bw
            for lin, lout in batch:
                d = pick_decode(s)
                d_active[d] += 1
                if t_now >= prm.warmup:
                    dec_batch.append(d_active[d])
                push(t_now + lout * prm.tpot_est, "dec_done", d)
                if t_now > duration:
                    continue
                size = lin * prm.kv_bytes_per_token / prm.gpus_per_server
                coflows[cid] = dict(left=prm.gpus_per_server, t0=t_now, ideal=ideal)
                paths = []
                for g in range(prm.gpus_per_server):
                    p = router.gpu_path(s, g, d, net.link_cnt)
                    paths.append(p)
                    net.link_cnt[p] += 1
                    net.link_load[p] += Router.PACK_FLOW_EST
                for p in paths:
                    net.link_cnt[p] -= 1
                    net.link_load[p] -= Router.PACK_FLOW_EST
                k = prm.gpus_per_server
                net.add_many(t_now, paths, [size] * k, [prm.nic_bw] * k,
                             [("kv", cid, t_now, size)] * k)
                cid += 1
            start_prefill(s, t_now)
        elif kind == "dec_done":
            d_active[data] -= 1
        elif kind == "bg":
            if prm.leaf_bg_w:
                w = np.asarray(prm.leaf_bg_w, dtype=float)
                lf_s = int(rng.choice(prm.n_leaf, p=w / w.sum()))
                w2 = w.copy(); w2[lf_s] = 0.0
                lf_d = int(rng.choice(prm.n_leaf, p=w2 / w2.sum()))
            else:
                lf_s = int(rng.integers(prm.n_leaf))
                lf_d = int((lf_s + 1 + rng.integers(prm.n_leaf - 1)) % prm.n_leaf)
            size = rng.exponential(prm.bg_size_mean)
            path = router.core_path(lf_s, lf_d, net.link_cnt)
            net.add(t_now, path, size, prm.bg_flow_cap, ("bg", -1, t_now, size))
    if t_cut is not None:
        for cf in coflows.values():
            if cf["t0"] >= prm.warmup:
                kv_cct.append(t_cut - cf["t0"])
                kv_sd.append((t_cut - cf["t0"]) / cf["ideal"])
        if net.t < t_cut:
            net.advance(min(t_cut, duration))
    else:
        net.advance(min(net.t, duration))
    kv_cct = np.array(kv_cct) if kv_cct else np.array([0.0])
    kv_fct = np.array(kv_fct) if kv_fct else np.array([0.0])
    kv_sd = np.array(kv_sd) if kv_sd else np.array([1.0])
    pf = np.array(pf_batch_tokens) if pf_batch_tokens else np.array([0])
    db = np.array(dec_batch) if dec_batch else np.array([0])

    ar_pf = {k: allreduce_cct(np.percentile(pf, 99) * prm.hidden * 2, k, prm) for k in NVL_CONFIGS}
    ar_dec = {k: allreduce_cct(np.percentile(db, 99) * prm.hidden * 2, k, prm) for k in NVL_CONFIGS}

    bg_tput = net.bg_moved / net.bg_demand_time if net.bg_demand_time > 0 else 1.0
    return dict(
        kv_cct_p50=float(np.percentile(kv_cct, 50)), kv_cct_p99=float(np.percentile(kv_cct, 99)),
        kv_fct_p99=float(np.percentile(kv_fct, 99)), n_kv=int(len(kv_cct)),
        kv_sd_p50=float(np.percentile(kv_sd, 50)), kv_sd_p99=float(np.percentile(kv_sd, 99)),
        kv_sd_p995=float(np.percentile(kv_sd, 99.5)),
        kv_local_frac=n_local_pick[0] / max(1, sum(n_local_pick)),
        bg_tput=float(bg_tput),
        pf_tokens_p99=float(np.percentile(pf, 99)), dec_batch_p99=float(np.percentile(db, 99)),
        ar_pf_p99=ar_pf[cfg.nvl_p], ar_dec_p99=ar_dec[cfg.nvl_d],
        ar_pf_all=ar_pf, ar_dec_all=ar_dec,
        kv_cct_all=kv_cct, kv_sd_all=kv_sd,
        truncated=t_cut is not None,
    )


def slo_check(prm: NP, res, margin=1.0):
    return (res["kv_sd_p99"] <= prm.slo_kv_sd_p99 * margin
            and res["bg_tput"] >= 1 - (1 - prm.slo_bg_tput) * margin
            and res["ar_pf_p99"] <= prm.slo_ar_prefill_p99 * margin
            and res["ar_dec_p99"] <= prm.slo_ar_decode_p99 * margin)


SCALES = {
    128: dict(n_leaf=4, servers_per_leaf=4, n_spine=4, max_up_per_spine=8),
    256: dict(n_leaf=4, servers_per_leaf=8, n_spine=8, max_up_per_spine=8),
    512: dict(n_leaf=8, servers_per_leaf=8, n_spine=8, max_up_per_spine=8),
    1024: dict(n_leaf=16, servers_per_leaf=8, n_spine=8, max_up_per_spine=8),
    2048: dict(n_leaf=32, servers_per_leaf=8, n_spine=16, max_up_per_spine=4),
    4096: dict(n_leaf=64, servers_per_leaf=8, n_spine=32, max_up_per_spine=2),
    8192: dict(n_leaf=128, servers_per_leaf=8, n_spine=64, max_up_per_spine=1),
}

SCENARIOS = {
    "base": dict(prm={}, lam_per_pserver=7.5, bg_per_server=75e9),
    "kvheavy": dict(prm=dict(n_param=8e9, n_layer=32, hidden=4096, kv_bytes_per_token=131_072,
                             lin_median=8000, lin_max=32768, prefill_token_budget=16384),
                    lam_per_pserver=17.5, bg_per_server=25e9),
}


def scenario_prm(name, gpus=1024):
    return NP(**SCENARIOS[name]["prm"], **SCALES[gpus])


def asym_prm(prm: NP, rng, hot_w=3.0):
    L, W, U = prm.n_leaf, prm.n_spine, prm.max_up_per_spine
    failed = []
    if U >= 2:
        for lf in rng.choice(L, size=int(rng.integers(1, max(1, L // 4) + 1)), replace=False):
            for sp in rng.choice(W, size=max(1, W // 2), replace=False):
                for u in rng.choice(U, size=U // 2, replace=False):
                    failed.append((int(lf), int(sp), int(u)))
    w = np.ones(L)
    w[rng.choice(L, size=int(rng.integers(1, max(1, L // 4) + 1)), replace=False)] = hot_w
    return dataclasses.replace(prm, failed_up=tuple(sorted(failed)), leaf_bg_w=tuple(float(x) for x in w))


def asym_to_str(prm: NP):
    f = ";".join(f"{a}-{b}-{c}" for a, b, c in prm.failed_up)
    w = ";".join(f"{x:g}" for x in prm.leaf_bg_w)
    return f, w


def asym_from_str(prm: NP, failed: str, weights: str):
    failed = tuple(tuple(int(v) for v in x.split("-")) for x in str(failed).split(";") if x and x != "nan")
    w = tuple(float(x) for x in str(weights).split(";") if x and x != "nan")
    return dataclasses.replace(prm, failed_up=failed, leaf_bg_w=w)


def scenario_load(name, prm: NP):
    n_srv = prm.n_leaf * prm.servers_per_leaf
    sc = SCENARIOS[name]
    return sc["lam_per_pserver"] * n_srv / 2, sc["bg_per_server"] * n_srv


def diurnal_at(t_hours, lam_peak, bg_peak):
    h = np.asarray(t_hours, dtype=float)
    inf = 0.30 + 0.70 * (0.5 * (1 - np.cos(2 * np.pi * (h - 2) / 24))) ** 1.5
    bg = 0.40 + 0.60 * (0.5 * (1 + np.cos(2 * np.pi * (h - 3) / 24))) ** 2
    return lam_peak * inf, bg_peak * bg


def diurnal(prm: NP, lam_peak=60.0, bg_peak=1.2e12, hours=24):
    h = np.arange(hours)
    inf = 0.30 + 0.70 * (0.5 * (1 - np.cos(2 * np.pi * (h - 2) / 24))) ** 1.5
    bg = 0.40 + 0.60 * (0.5 * (1 + np.cos(2 * np.pi * (h - 3) / 24))) ** 2
    return lam_peak * inf, bg_peak * bg
