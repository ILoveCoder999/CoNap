import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = {"random": "Same scale", "random | asym only": "Asym. only",
          "scale-XL (train 16-128 servers, test 256 servers)": "256 servers",
          "scale-XL (train 16-128 servers, test 512 servers)": "512 servers"}
ROWS = {"MLP+CRC5%": "MLP (default)", "GNN+CRC5%": "GNN", "Hybrid+CRC5%": "Hybrid (GNN+MLP)",
        "Analytic (ET rule)": "Analytic rule"}
PAPER = {"MLP (default)": (93.5, 92.5, 93.8, 93.1), "GNN": (95.0, 94.5, 95.5, 87.4),
         "Hybrid (GNN+MLP)": (95.1, 94.6, 89.6, 88.7), "Analytic rule": (46.7, 48.9, 56.3, 52.8)}
PAPER_TEXT = {("MLP", "slo"): 95.9, ("MLP", "regret"): 14.1, ("MLP+CRC5%", "regret"): 8.2,
              ("MLP+CRC10%", "slo"): 90.1, ("MLP+CRC10%", "regret"): 5.5,
              ("GNN+CRC5%", "regret"): 9.0, ("Hybrid+CRC5%", "regret"): 10.6}


def r1(x):
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "eval_study", "gnn_results.csv")
    d = pd.read_csv(path)
    d["slo"] = 100 - d.decision_violation_pct
    d["regret"] = d.decision_regret_pct
    mean = d.groupby(["model", "split"]).agg(slo=("slo", "mean"), regret=("regret", "mean"), seeds=("slo", "size"))

    t = pd.DataFrame({lab: [r1(mean.loc[(m, s), "slo"]) for s in SPLITS] for m, lab in ROWS.items()},
                     index=list(SPLITS.values())).T
    print("Table 6: SLO satisfaction rate (%) of the configurations selected by each evaluator")
    print("(learned models calibrated to alpha = 5%; analytic rule uncalibrated; mean over seeds)\n")
    print(t.to_string())
    bad = 0
    for lab, ref in PAPER.items():
        for col, e in zip(t.columns, ref):
            if t.loc[lab, col] != e:
                bad += 1
                print(f"  DIFFERS  {lab} / {col}: paper {e} vs {t.loc[lab, col]}")
    print("\nSection 6.7, same-scale split:")
    for (m, k), e in PAPER_TEXT.items():
        v = r1(mean.loc[(m, "random"), k])
        ok = v == e
        bad += not ok
        print(f"  {m:12s} {'SLO satisfaction' if k == 'slo' else 'excess energy':16s} {v:5.1f}%"
              + ("" if ok else f"   DIFFERS from paper ({e}%)"))
    sc = r1(mean.loc[("MLP+CRC5%", "scale (train 16-64 servers, test 128 servers)"), "slo"])
    print(f"  MLP+CRC5% on 128 servers (trained on 16-64): {sc}% (paper: 'more than 97%')")
    bad += not sc > 97
    out = os.path.join(os.path.dirname(path), "table5.csv")
    t.to_csv(out)
    print("\nALL VALUES MATCH THE PAPER" if bad == 0 else f"\n{bad} value(s) differ from the paper")
    sys.exit(1 if bad else 0)
