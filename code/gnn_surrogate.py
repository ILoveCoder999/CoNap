import glob
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import controller as ctl
import netsim as n

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ARGV = sys.argv if __name__ == "__main__" else [""]
DATA = _ARGV[1] if len(_ARGV) > 1 else "gnn_data.csv"
EPOCHS = int(_ARGV[2]) if len(_ARGV) > 2 else 200
EXTRA = [f for f in _ARGV[3].split(",") if f] if len(_ARGV) > 3 else []
SEEDS = int(_ARGV[4]) if len(_ARGV) > 4 else 1
GNN_HIDDEN = int(os.environ.get("GNN_HIDDEN", 64))
GNN_LAYERS = int(os.environ.get("GNN_LAYERS", 3))
PATIENCE = int(os.environ.get("PATIENCE", 0))
ONLY_SPLITS = [x for x in os.environ.get("ONLY_SPLITS", "").split(",") if x]
MODELS = [x for x in os.environ.get("MODELS", "gnn,mlp,hyb").split(",") if x]
OUTDIR = os.environ.get("OUTDIR", ".")
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)


CRC_ALPHAS = [float(x) for x in os.environ.get("CRC_ALPHAS", "0.05,0.10").split(",") if x]
MODEL_NAMES = {"gnn": "GNN", "mlp": "MLP", "hyb": "Hybrid"}

SLO = {"base": (8.792, 0.927), "kvheavy": (7.049, 0.964)}
for scen in SLO:
    for f in sorted(glob.glob(f"results_{scen}_1024g_*m/slo.json")):
        d = json.load(open(f))
        SLO[scen] = (d["slo_kv_sd_p99"], d["slo_bg_tput"])
        break
SD_MARGIN, BG_EXTRA = 0.9, 0.02

PAIRINGS, ROUTINGS = ["lb", "local", "adaptive"], ["ecmp", "explicit", "pack"]
NODE_F, EDGE_F = 13, 2


_prm_cache = {}


def get_prm(scen, gpus):
    key = (scen, gpus)
    if key not in _prm_cache:
        _prm_cache[key] = n.scenario_prm(scen, gpus)
    return _prm_cache[key]


def build(row):
    prm = get_prm(row.scenario, int(row.gpus))
    if getattr(row, "asym", False) and isinstance(getattr(row, "leaf_bg_w", None), str):
        prm = n.asym_from_str(prm, row.failed_up, row.leaf_bg_w)
    n_leaf, spl, n_srv = prm.n_leaf, prm.servers_per_leaf, prm.n_leaf * prm.servers_per_leaf
    W, U, ns, u = prm.n_spine, prm.max_up_per_spine, int(row.n_spine), int(row.up_per_spine)
    cfg = n.NetCfg(ns, u, int(row.rate), int(row.edge_ports))
    roles = n.server_roles(row.layout, prm)
    n_p = sum(1 for s in roles if roles[s] == "P")
    n_d = n_srv - n_p
    e_kv = ctl.mean_kv_bytes(prm)
    nic = prm.nic_bw
    edge_cap = row.edge_ports * nic
    link_bw = row.rate / 8 * 1e9
    up_cap_leaf = ns * u * link_bw
    kv_p = row.lam / n_p * e_kv
    kv_d = row.lam / n_d * e_kv
    w = np.asarray(prm.leaf_bg_w, dtype=float) if prm.leaf_bg_w else np.ones(n_leaf)
    bg_lf = row.bg * w / w.sum()
    heat = w / w.mean()
    eff = np.array([[len(n.opened_ups(prm, cfg, lf, sp)) for sp in range(ns)] for lf in range(n_leaf)])
    eff_cap = np.maximum(eff.sum(1), 1) * link_bw
    health = eff.sum(1) / (ns * u)

    x, etype = [], []
    for s in range(n_srv):
        isP = roles[s] == "P"
        x.append([1, 0, 0, float(isP), float(not isP), row.edge_ports / 8,
                  (kv_p if isP else kv_d) / edge_cap, 0, 0, 0, 0, 0, 0])
        etype.append(0)
    kv_out_l, kv_in_l = [], []
    for lf in range(n_leaf):
        srvs = range(lf * spl, (lf + 1) * spl)
        kv_out = sum(kv_p for s in srvs if roles[s] == "P")
        kv_in = sum(kv_d for s in srvs if roles[s] == "D")
        kv_out_l.append(kv_out); kv_in_l.append(kv_in)
        x.append([0, 1, 0, 0, 0, 0, 0, bg_lf[lf] / eff_cap[lf], kv_out / eff_cap[lf], kv_in / eff_cap[lf],
                  eff[lf].sum() / (W * U), heat[lf], health[lf]])
        etype.append(1)
    for sp in range(ns):
        x.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, eff[:, sp].mean() / u, u / U, row.rate / 400])
        etype.append(2)
    src, dst, ea = [], [], []

    def link(a, b, cap_norm, core):
        src.extend([a, b]); dst.extend([b, a]); ea.extend([[cap_norm, core], [cap_norm, core]])

    for s in range(n_srv):
        link(s, n_srv + s // spl, row.edge_ports / 8, 0.0)
    for lf in range(n_leaf):
        for sp in range(ns):
            link(n_srv + lf, n_srv + n_leaf + sp, eff[lf, sp] * row.rate / (U * 400), 1.0)

    g = [float(row.pairing == p) for p in PAIRINGS] + [float(row.routing == r) for r in ROUTINGS] + [
        row.pack_margin, float(row.layout == "mixed"), prm.kv_bytes_per_token / 327_680, prm.lin_median / 8000]
    kv_out_l, kv_in_l = np.array(kv_out_l), np.array(kv_in_l)
    flat = g + [ns / W, u / U, row.rate / 400, row.edge_ports / 8,
                row.bg / n_leaf / up_cap_leaf, kv_out_l.max() / up_cap_leaf, kv_in_l.max() / up_cap_leaf,
                kv_p / edge_cap, kv_d / edge_cap,
                heat.max(), heat.std(), health.min(), health.mean()]
    need = ctl.leaf_egress_demand(prm, row.lam, row.bg, row.layout)
    kv_part = need - row.bg / n_leaf
    analytic_ok = (float(((bg_lf + kv_part) / eff_cap).max()) <= 0.75) and (max(kv_p, kv_d) / edge_cap <= 0.75)
    return dict(x=np.array(x, dtype=np.float32), etype=np.array(etype), ei=np.array([src, dst]),
                ea=np.array(ea, dtype=np.float32), g=np.array(g, dtype=np.float32),
                flat=np.array(flat, dtype=np.float32), analytic_ok=analytic_ok)


def collate(graphs):
    xs, ets, eis, eas, bs, off = [], [], [], [], [], 0
    for i, gr in enumerate(graphs):
        xs.append(gr["x"]); ets.append(gr["etype"]); eas.append(gr["ea"])
        eis.append(gr["ei"] + off); bs.append(np.full(len(gr["x"]), i)); off += len(gr["x"])
    parts = (np.concatenate(xs), np.concatenate(ets), np.concatenate(eis, axis=1), np.concatenate(eas),
             np.concatenate(bs), np.stack([gr["g"] for gr in graphs]), np.stack([gr["flat"] for gr in graphs]))
    return tuple(torch.from_numpy(p).to(DEVICE, non_blocking=True) for p in parts)


def seg_mean(v, idx, n_seg):
    out = torch.zeros(n_seg, v.shape[1], device=v.device).index_add_(0, idx, v)
    cnt = torch.zeros(n_seg, device=v.device).index_add_(0, idx, torch.ones(len(idx), device=v.device)).clamp(min=1)
    return out / cnt[:, None]


def seg_max(v, idx, n_seg):
    out = torch.full((n_seg, v.shape[1]), -1e9, device=v.device)
    out = out.scatter_reduce(0, idx[:, None].expand_as(v), v, reduce="amax", include_self=True)
    return torch.where(out < -1e8, torch.zeros_like(out), out)


class GNN(nn.Module):
    def __init__(self, h=64, layers=3, n_glob=10, n_out=3, pool="meanmax"):
        super().__init__()
        self.pool = pool
        self.inp = nn.Linear(NODE_F, h)
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(h + EDGE_F, h), nn.ReLU()) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])
        n_pool = 6 if pool == "meanmax" else 3
        self.head = nn.Sequential(nn.Linear(n_pool * h + n_glob, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, n_out))

    def forward(self, x, et, ei, ea, b, g, flat=None):
        h = torch.relu(self.inp(x))
        src, dst = ei
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            m = msg(torch.cat([h[src], ea], 1))
            agg = seg_mean(m, dst, len(h))
            h = norm(h + torch.relu(upd(torch.cat([h, agg], 1))))
        B = g.shape[0]
        pooled = []
        for t in (0, 1, 2):
            mask = et == t
            pooled.append(seg_mean(h[mask], b[mask], B))
            if self.pool == "meanmax":
                pooled.append(seg_max(h[mask], b[mask], B))
        return self.head(torch.cat(pooled + [g], 1))


class MLP(nn.Module):
    def __init__(self, n_in, n_out=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, flat):
        return self.net(flat)


class Hybrid(nn.Module):

    def __init__(self, n_flat, h=64, layers=3, n_glob=10, n_out=3):
        super().__init__()
        self.mlp = MLP(n_flat, n_out)
        self.gnn = GNN(h=h, layers=layers, n_glob=n_glob, n_out=n_out, pool="mean")
        last = self.gnn.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, et, ei, ea, b, g, flat):
        return self.mlp(flat) + self.gnn(x, et, ei, ea, b, g)


SD_CAP = 50.0


def targets(df):
    return np.stack([np.log(df.kv_sd_p99.clip(1.0, SD_CAP)), np.log(df.kv_sd_p995.clip(1.0, SD_CAP)),
                     df.bg_tput.clip(0, 1)], 1).astype(np.float32)


def train(kind, graphs, Y, tr, va, epochs, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Yn = torch.from_numpy((Y - mu) / sd).to(DEVICE)
    if kind == "gnn":
        model = GNN(h=GNN_HIDDEN, layers=GNN_LAYERS, n_glob=len(graphs[0]["g"])).to(DEVICE)
    elif kind == "hyb":
        model = Hybrid(len(graphs[0]["flat"]), h=GNN_HIDDEN, layers=GNN_LAYERS, n_glob=len(graphs[0]["g"])).to(DEVICE)
    else:
        flat = torch.from_numpy(np.stack([gr["flat"] for gr in graphs])).to(DEVICE)
        model = MLP(flat.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-5)

    def predict(idx, bs=256):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]
                outs.append(model(*collate([graphs[k] for k in j])) if kind != "mlp" else model(flat[j]))
        return torch.cat(outs).cpu()

    best, best_state, bad = math.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            out = model(*collate([graphs[k] for k in j])) if kind != "mlp" else model(flat[j])
            loss = ((out - Yn[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        vl = ((predict(va) - Yn[va].cpu()) ** 2).mean().item()
        sched.step(vl)
        if PATIENCE <= 0:
            continue
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if PATIENCE > 0:
        model.load_state_dict(best_state)

    def pred_fn(idx):
        p = predict(idx).numpy() * sd + mu
        return np.exp(p[:, 0]), np.exp(p[:, 1]), p[:, 2]
    pred_fn.model, pred_fn.mu, pred_fn.sd = model, mu, sd
    return pred_fn, ep + 1


def thresholds(df):
    sd = df.scenario.map(lambda s: SLO[s][0]).values
    bg = df.scenario.map(lambda s: SLO[s][1]).values
    return sd, bg


def decide(d, ok_pred, true_ok):
    d = d.assign(true_ok=true_ok, pred_ok=ok_pred)
    viol, regret, nofeas, groups = 0, [], 0, 0
    saving, oracle = [], []
    for _, gdf in d.groupby("gid"):
        if not gdf.true_ok.any():
            continue
        groups += 1
        best_true = gdf[gdf.true_ok].P_scaleout.min()
        p_full = gdf.P_full.iloc[0]
        oracle.append(1 - best_true / p_full)
        cand = gdf[gdf.pred_ok]
        if cand.empty:
            nofeas += 1
            regret.append(p_full / best_true - 1)
            saving.append(0.0)
            continue
        chosen = cand.loc[cand.P_scaleout.idxmin()]
        saving.append(1 - chosen.P_scaleout / p_full)
        if not chosen.true_ok:
            viol += 1
        else:
            regret.append(chosen.P_scaleout / best_true - 1)
    return groups, viol, regret, nofeas, dict(saving=saving, oracle=oracle)


def intuitive(groups, viol, extra):
    return dict(slo_ok_pct=100 - viol / max(groups, 1) * 100,
                saving_pct=float(np.mean(extra["saving"]) * 100) if extra["saving"] else float("nan"),
                oracle_saving_pct=float(np.mean(extra["oracle"]) * 100) if extra["oracle"] else float("nan"))


CRC_LEVELS = np.r_[np.linspace(0.0, 0.9, 46), np.linspace(0.91, 0.999, 30), 1.0]


def crc_bounds(p99, pbg, qs, qb):
    return p99 * np.exp(qs), pbg - qb


def crc_calibrate(df, cal, p99c, pbgc, alpha):
    d = df.iloc[cal]
    sd_thr, bg_thr = thresholds(d)
    y99 = d.kv_sd_p99.clip(1.0, SD_CAP).values
    ybg = d.bg_tput.clip(0, 1).values
    true_ok = (y99 <= sd_thr) & (ybg >= bg_thr)
    s_sd = np.log(y99) - np.log(np.clip(p99c, 1.0, SD_CAP))
    s_bg = pbgc - ybg
    for q in CRC_LEVELS:
        qs, qb = float(np.quantile(s_sd, q)), float(np.quantile(s_bg, q))
        up99, lowbg = crc_bounds(p99c, pbgc, qs, qb)
        n, v, _, _, _ = decide(d, (up99 <= sd_thr) & (lowbg >= bg_thr), true_ok)
        if n and (v + 1) / (n + 1) <= alpha:
            return q, qs, qb
    return 1.0, float(s_sd.max()), float(s_bg.max())


def crc_evaluate(df, idx, p99, pbg, qs, qb):
    d = df.iloc[idx]
    sd_thr, bg_thr = thresholds(d)
    true_ok = (d.kv_sd_p99.values <= sd_thr) & (d.bg_tput.values >= bg_thr)
    up99, lowbg = crc_bounds(p99, pbg, qs, qb)
    ok = (up99 <= sd_thr) & (lowbg >= bg_thr)
    n, v, reg, nf, ex = decide(d, ok, true_ok)
    return dict(n_test=len(idx), feasible_acc=float(np.mean(ok == true_ok) * 100), decision_groups=n,
                decision_violation_pct=v / max(n, 1) * 100,
                decision_regret_pct=float(np.mean(reg) * 100) if reg else float("nan"),
                fallback_pct=nf / max(n, 1) * 100, **intuitive(n, v, ex))


def evaluate(df, idx, pred99, pred995, predbg, name, split):
    d = df.iloc[idx]
    sd_thr, bg_thr = thresholds(d)
    true_ok = (d.kv_sd_p99.values <= sd_thr) & (d.bg_tput.values >= bg_thr)
    pred_ok_raw = (pred99 <= sd_thr) & (predbg >= bg_thr)
    pred_ok_safe = (pred99 <= sd_thr * SD_MARGIN) & (predbg >= bg_thr + BG_EXTRA) & (pred995 <= sd_thr)
    m = ~d.truncated.values
    y = d.kv_sd_p99.clip(1.0, SD_CAP).values[m]
    p = np.clip(pred99, 1.0, SD_CAP)[m]
    res = dict(split=split, model=name, n_test=len(idx), n_truncated=int((~m).sum()),
               sd99_mape=float(np.mean(np.abs(p - y) / y) * 100),
               sd99_r2=float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)),
               bg_mae=float(np.mean(np.abs(predbg[m] - d.bg_tput.values[m]))),
               feasible_acc=float(np.mean(pred_ok_raw == true_ok) * 100))
    groups, viol, regret, nofeas, ex = decide(d, pred_ok_safe, true_ok)
    res.update(decision_groups=groups, decision_violation_pct=viol / max(groups, 1) * 100,
               decision_regret_pct=float(np.mean(regret) * 100) if regret else float("nan"),
               fallback_pct=nofeas / max(groups, 1) * 100, **intuitive(groups, viol, ex))
    return res


def analytic_eval(df, idx, graphs, split):
    d = df.iloc[idx]
    sd_thr, bg_thr = thresholds(d)
    true_ok = (d.kv_sd_p99.values <= sd_thr) & (d.bg_tput.values >= bg_thr)
    pred_ok = np.array([graphs[i]["analytic_ok"] for i in idx])
    groups, viol, regret, nofeas, ex = decide(d, pred_ok, true_ok)
    return dict(split=split, model="Analytic (ET rule)", n_test=len(idx), sd99_mape=np.nan, sd99_r2=np.nan,
                bg_mae=np.nan, feasible_acc=float(np.mean(pred_ok == true_ok) * 100), decision_groups=groups,
                decision_violation_pct=viol / max(groups, 1) * 100,
                decision_regret_pct=float(np.mean(regret) * 100) if regret else float("nan"),
                fallback_pct=nofeas / max(groups, 1) * 100, **intuitive(groups, viol, ex))


def load_csv(f, source):
    d = pd.read_csv(f, low_memory=False)
    if "asym" not in d:
        d["asym"], d["failed_up"], d["leaf_bg_w"] = False, "", ""
    d["asym"] = d.asym.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    d["failed_up"] = d.failed_up.fillna("").astype(str)
    d["leaf_bg_w"] = d.leaf_bg_w.fillna("").astype(str)
    if "truncated" not in d:
        d["truncated"] = False
    d["truncated"] = d.truncated.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    d = d[(d.n_kv >= 50) | d.truncated].copy()
    d.loc[d.truncated, ["kv_sd_p99", "kv_sd_p995"]] = SD_CAP
    d["source"] = source
    return d


def prepare(df):
    df = df.reset_index(drop=True)
    df["servers"] = df.gpus // 8
    _pf = {}
    for key in df[["scenario", "gpus", "layout"]].drop_duplicates().itertuples(index=False):
        prm_ = get_prm(key.scenario, int(key.gpus))
        pw = n.network_power(n.full_cfg(prm_), prm_, key.layout)
        _pf[(key.scenario, int(key.gpus), key.layout)] = pw["total"] - pw["nvlink"]
    df["P_full"] = [_pf[(a, int(b), c)] for a, b, c in zip(df.scenario, df.gpus, df.layout)]
    return df


def group_split(df, mask_train, rng, val_frac=0.1):
    gids_tr = df[mask_train].gid.unique()
    rng.shuffle(gids_tr)
    n_va = max(1, int(len(gids_tr) * val_frac))
    va_g = set(gids_tr[:n_va])
    tr = np.where(mask_train & ~df.gid.isin(va_g))[0]
    va = np.where(df.gid.isin(va_g))[0]
    return tr, va


if __name__ == "__main__":
    t0 = time.time()

    df = prepare(pd.concat([load_csv(DATA, "main")] + [load_csv(f, "xl") for f in EXTRA], ignore_index=True))
    m_main = (df.source == "main").values
    print(f"Asymmetric samples: training data {int(df[m_main].asym.sum())} / {int(m_main.sum())}, "
          f"test data {int(df[~m_main].asym.sum())} / {int((~m_main).sum())}")
    print(f"Training data: {m_main.sum()} samples, {df[m_main].gid.nunique()} groups; by server count "
          f"{df[m_main].groupby('servers').size().to_dict()}; scenarios {df[m_main].groupby('scenario').size().to_dict()}")
    if EXTRA:
        print(f"Large-scale test data: {(~m_main).sum()} samples; by server count {df[~m_main].groupby('servers').size().to_dict()}")
    print(f"SLO thresholds: {SLO}")
    print(f"Training device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    graphs = [build(r) for r in df.itertuples()]
    Y = targets(df)
    rng = np.random.default_rng(0)

    splits = {}
    gids = df[m_main].gid.unique().copy()
    rng.shuffle(gids)
    test_g = set(gids[: int(0.2 * len(gids))])
    m_test = m_main & df.gid.isin(test_g).values
    splits["random"] = (*group_split(df, m_main & ~m_test, rng), {"random": np.where(m_test)[0]})
    sv = df.servers.values
    if (m_main & (sv == 128)).any() and (m_main & (sv < 128)).any():
        m = m_main & (sv < 128)
        splits["scale"] = (*group_split(df, m, rng),
                           {"scale (train 16-64 servers, test 128 servers)": np.where(m_main & (sv == 128))[0]})
    if df[m_main].scenario.nunique() > 1:
        m = m_main & (df.scenario == "base").values
        splits["scenario"] = (*group_split(df, m, rng),
                              {"scenario (train base, test kvheavy)": np.where(m_main & (df.scenario == "kvheavy").values)[0]})
    if EXTRA:
        tests = {f"scale-XL (train 16-128 servers, test {k} servers)": np.where((~m_main) & (sv == k))[0]
                 for k in sorted(set(sv[~m_main]))}
        splits["scale-XL"] = (*group_split(df, m_main, rng), tests)

    if ONLY_SPLITS:
        splits = {k: v for k, v in splits.items() if k in ONLY_SPLITS}
    print(f"Models: {[MODEL_NAMES[k] for k in MODELS]}; GNN: hidden dim {GNN_HIDDEN}, {GNN_LAYERS} layers; "
          + (f"early-stopping patience {PATIENCE} epochs" if PATIENCE > 0 else f"no early stopping (all {EPOCHS} epochs, last model used)")
          + f"; number of seeds {SEEDS}", flush=True)
    results, preds_for_plot = [], {}
    asym_all = df.asym.values
    for split, (tr, va, tests) in splits.items():
        tests = dict(tests)
        for tname, te in list(tests.items()):
            a = te[asym_all[te]]
            if 0 < len(a) < len(te):
                tests[tname + " | asym only"] = a
        va_g = np.unique(df.gid.values[va])
        if CRC_ALPHAS and len(va_g) >= 4:
            cal_g = set(va_g[1::2])
            cal = va[np.isin(df.gid.values[va], list(cal_g))]
            va = va[~np.isin(df.gid.values[va], list(cal_g))]
        else:
            cal = va[:0]
        print(f"\n== Split: {split}  train {len(tr)}, validation {len(va)}, calibration {len(cal)}, "
              f"test {sum(len(t) for t in tests.values())} ==", flush=True)
        for kind, name, seed in [(k, MODEL_NAMES[k], sd) for sd in range(SEEDS) for k in MODELS]:
            ts = time.time()
            pred_fn, ep = train(kind, graphs, Y, tr, va, EPOCHS, seed=seed)
            t_train = time.time() - ts
            crc = {}
            if CRC_ALPHAS and len(cal):
                p99c, _, pbgc = pred_fn(cal)
                for a in CRC_ALPHAS:
                    crc[a] = crc_calibrate(df, cal, p99c, pbgc, a)
            for tname, te in tests.items():
                ts = time.time()
                p99, p995, pbg = pred_fn(te)
                t_inf = (time.time() - ts) / len(te)
                r = evaluate(df, te, p99, p995, pbg, name, tname)
                r.update(seed=seed, epochs=ep, train_s=round(t_train, 1), infer_ms_per_sample=t_inf * 1000,
                         sim_ms_per_sample=float(df.sim_seconds.values[te].mean() * 1000))
                results.append(r)
                preds_for_plot[(tname, name)] = (df.kv_sd_p99.values[te], p99)
                for a, (q, qs, qb) in crc.items():
                    rc = crc_evaluate(df, te, p99, pbg, qs, qb)
                    rc.update(split=tname, model=f"{name}+CRC{a:.0%}", seed=seed, epochs=ep, crc_level=q,
                              crc_sd_margin=float(np.exp(qs)), crc_bg_margin=qb)
                    results.append(rc)
                    print(f"    {name}+CRC (target SLO violation {a:.0%}, quantile level {q:.3f}, slowdown upper bound = prediction x {np.exp(qs):.2f}, "
                          f"background lower bound = prediction - {qb:.3f}) [{tname}]: SLO satisfaction rate {rc['slo_ok_pct']:.1f}%, "
                          f"energy saving {rc['saving_pct']:.1f}% (oracle {rc['oracle_saving_pct']:.1f}%)", flush=True)
                print(f"  {name} (seed {seed}) [{tname}]: trained {ep} epochs in {t_train:.0f}s; P99 slowdown mean relative error {r['sd99_mape']:.1f}%, "
                      f"R² {r['sd99_r2']:.3f}; background throughput error {r['bg_mae']:.3f}; feasibility accuracy {r['feasible_acc']:.1f}%; "
                      f"SLO satisfaction rate {r['slo_ok_pct']:.1f}%, energy saving {r['saving_pct']:.1f}% (oracle {r['oracle_saving_pct']:.1f}%)", flush=True)
        for tname, te in tests.items():
            r = analytic_eval(df, te, graphs, tname)
            r.update(sim_ms_per_sample=float(df.sim_seconds.values[te].mean() * 1000))
            results.append(r)
            print(f"  Analytic [{tname}]: feasibility accuracy {r['feasible_acc']:.1f}%; "
                  f"SLO satisfaction rate {r['slo_ok_pct']:.1f}%, energy saving {r['saving_pct']:.1f}% (oracle {r['oracle_saving_pct']:.1f}%)")

    res = pd.DataFrame(results)
    sim_ms = df.sim_seconds.mean() * 1000
    res["speedup_vs_sim"] = res.sim_ms_per_sample / res.infer_ms_per_sample
    res.to_csv(out("gnn_results.csv"), index=False)
    pd.set_option("display.width", 240)
    print("\n== Summary ==")
    cols = ["split", "model", "n_test", "sd99_mape", "sd99_r2", "bg_mae", "feasible_acc",
            "decision_violation_pct", "decision_regret_pct", "fallback_pct", "infer_ms_per_sample",
            "sim_ms_per_sample", "speedup_vs_sim"]
    print(res[cols + (["seed"] if "seed" in res else [])].round(3).to_string(index=False))
    if SEEDS > 1:
        key = ["slo_ok_pct", "saving_pct", "oracle_saving_pct", "decision_violation_pct", "decision_regret_pct",
               "fallback_pct", "feasible_acc", "sd99_mape", "sd99_r2"]
        agg = res[~res.model.str.startswith("Analytic")].groupby(["split", "model"], sort=False)[key].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.round(2).to_csv(out("gnn_results_seeds.csv"))
        print(f"\n== Mean and standard deviation over {SEEDS} random seeds (saved to gnn_results_seeds.csv) ==")
        for (sp, mdl), r in agg.iterrows():
            print(f"  {mdl} [{sp}]: SLO satisfaction rate {r.slo_ok_pct_mean:.1f}±{r.slo_ok_pct_std:.1f}%, "
                  f"energy saving {r.saving_pct_mean:.1f}±{r.saving_pct_std:.1f}% (oracle {r.oracle_saving_pct_mean:.1f}%), "
                  f"feasibility accuracy {r.feasible_acc_mean:.1f}±{r.feasible_acc_std:.1f}%, "
                  f"MAPE {r.sd99_mape_mean:.1f}±{r.sd99_mape_std:.1f}%, R² {r.sd99_r2_mean:.3f}±{r.sd99_r2_std:.3f}")
    print(f"\nMean simulation time per sample {sim_ms:.0f} ms (single core); by server count:",
          {f"{k} servers": v for k, v in (df.groupby("servers").sim_seconds.mean() * 1000).round(0).to_dict().items()})

    plain = res.groupby(["split", "model"], sort=False)[["slo_ok_pct", "saving_pct", "oracle_saving_pct"]].mean().round(1)
    plain.columns = ["SLO satisfaction rate (%)", "Energy saving vs all-on (%)", "Oracle energy saving (%)"]
    plain.to_csv(out("gnn_plain_summary.csv"))
    print("\n== Intuitive summary (saved to gnn_plain_summary.csv): how often the chosen configuration meets the SLO, and how much energy it saves vs all-on ==")
    print(plain.to_string())

    test_names = list(dict.fromkeys(k[0] for k in preds_for_plot))
    ncol = len(test_names)
    names = [MODEL_NAMES[k] for k in MODELS]
    colors = {"GNN": "#2a78d6", "MLP": "#eb6834", "Hybrid": "#1baf7a"}
    fig, axes = plt.subplots(len(names), ncol, figsize=(4.2 * ncol, 3.8 * len(names)), squeeze=False)
    for c, split in enumerate(test_names):
        for r_, name in enumerate(names):
            ax = axes[r_][c]
            y, p = preds_for_plot[(split, name)]
            ax.scatter(y, p, s=6, alpha=0.4, color=colors[name])
            lim = [1, max(y.max(), p.max()) * 1.1]
            ax.plot(lim, lim, color="#52514e", linewidth=1, linestyle="--")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_title(f"{name} — {split}", fontsize=7.5, loc="left")
            ax.set_xlabel("Simulated KV P99 slowdown")
            ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(out("gnn_pred_vs_true.pdf"))
    print(f"\nWrote {out('gnn_results.csv')} and {out('gnn_pred_vs_true.pdf')}, total time {(time.time()-t0)/60:.1f} min")
