import os
import time
import types

import numpy as np
import torch

import netsim as n

_CACHE = {}


def _gs():
    import gnn_surrogate as gs
    gs.DEVICE = torch.device("cpu")
    torch.set_num_threads(1)
    return gs


def load(path):
    if path not in _CACHE:
        gs = _gs()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["kind"] == "mlp":
            model = gs.MLP(ck["n_flat"])
        elif ck["kind"] == "gnn":
            model = gs.GNN(h=ck["gnn_hidden"], layers=ck["gnn_layers"], n_glob=ck["n_glob"])
        else:
            model = gs.Hybrid(ck["n_flat"], h=ck["gnn_hidden"], layers=ck["gnn_layers"], n_glob=ck["n_glob"])
        model.load_state_dict(ck["state"])
        model.eval()
        _CACHE[path] = (gs, model, ck)
    return _CACHE[path]


def judge(path, scen, gpus, prm, cfgs, pol, lam, bg):
    gs, model, ck = load(path)
    t0 = time.time()
    failed, weights = n.asym_to_str(prm)
    rows = [types.SimpleNamespace(scenario=scen, gpus=int(gpus), asym=bool(prm.failed_up or prm.leaf_bg_w),
                                  failed_up=failed, leaf_bg_w=weights, n_spine=c.n_spine,
                                  up_per_spine=c.up_per_spine, rate=c.rate, edge_ports=c.edge_ports,
                                  layout=pol["layout"], routing=pol["routing"], pairing=pol["pairing"],
                                  pack_margin=c.pack_margin, lam=float(lam), bg=float(bg)) for c in cfgs]
    graphs = [gs.build(r) for r in rows]
    with torch.no_grad():
        if ck["kind"] == "mlp":
            out = model(torch.from_numpy(np.stack([g["flat"] for g in graphs])))
        else:
            out = model(*gs.collate(graphs))
    p = out.numpy() * ck["sd"] + ck["mu"]
    p99 = np.clip(np.exp(p[:, 0]), 1.0, ck["sd_cap"])
    up99 = p99 * np.exp(ck["crc_sd_log"])
    lowbg = p[:, 2] - ck["crc_bg"]
    ok = (up99 <= prm.slo_kv_sd_p99) & (lowbg >= prm.slo_bg_tput)
    return ok.tolist(), up99.tolist(), lowbg.tolist(), time.time() - t0


def available(path):
    return bool(path) and os.path.exists(path)
