import os
import sys
import time

import numpy as np
import torch

import gnn_surrogate as gs

DATA = sys.argv[1] if len(sys.argv) > 1 else "gnn_data_all.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "conap_evaluator.pt"
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
gs.PATIENCE = int(os.environ.get("PATIENCE", 40))
SEED = int(os.environ.get("SEED", 0))
KIND = os.environ.get("MODEL", "mlp")


def main():
    t0 = time.time()
    df = gs.prepare(gs.pd.concat([gs.load_csv(f, "main") for f in DATA.split(",") if f], ignore_index=True))
    print(f"Training data: {len(df)} samples, {df.gid.nunique()} groups; by server count {df.groupby('servers').size().to_dict()}; "
          f"asymmetric samples {int(df.asym.sum())}; SLO thresholds {gs.SLO}", flush=True)
    graphs = [gs.build(r) for r in df.itertuples()]
    Y = gs.targets(df)

    rng = np.random.default_rng(SEED)
    gids = df.gid.unique()
    rng.shuffle(gids)
    n_g = len(gids)
    va_g, cal_g = set(gids[:n_g // 10]), set(gids[n_g // 10: n_g // 5])
    g = df.gid.values
    va = np.where(np.isin(g, list(va_g)))[0]
    cal = np.where(np.isin(g, list(cal_g)))[0]
    tr = np.where(~np.isin(g, list(va_g | cal_g)))[0]
    print(f"Train {len(tr)}, validation {len(va)}, calibration {len(cal)} samples; model {KIND}; device {gs.DEVICE}", flush=True)

    pred_fn, ep = gs.train(KIND, graphs, Y, tr, va, EPOCHS, seed=SEED)
    print(f"Training done: {ep} epochs, {time.time() - t0:.0f} s", flush=True)

    p99c, _, pbgc = pred_fn(cal)
    q, qs, qb = gs.crc_calibrate(df, cal, p99c, pbgc, ALPHA)
    rc = gs.crc_evaluate(df, cal, p99c, pbgc, qs, qb)
    print(f"Conformal calibration (target SLO violation {ALPHA:.0%}): quantile level {q:.3f}, slowdown upper bound = prediction x {np.exp(qs):.3f}, "
          f"background-throughput lower bound = prediction - {qb:.4f}; on the calibration set: SLO satisfaction rate {rc['slo_ok_pct']:.1f}%, "
          f"energy saving vs all-on {rc['saving_pct']:.1f}% (oracle {rc['oracle_saving_pct']:.1f}%)", flush=True)

    model = pred_fn.model
    torch.save(dict(kind=KIND, state=model.state_dict(), n_flat=len(graphs[0]["flat"]), n_glob=len(graphs[0]["g"]),
                    gnn_hidden=gs.GNN_HIDDEN, gnn_layers=gs.GNN_LAYERS,
                    mu=pred_fn.mu, sd=pred_fn.sd, alpha=ALPHA, crc_level=q, crc_sd_log=qs, crc_bg=qb,
                    slo=gs.SLO, sd_cap=gs.SD_CAP, data=DATA, epochs=ep), OUT)
    print(f"Saved {OUT}, total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
