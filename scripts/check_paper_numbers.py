import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_MAIN = {
    ("base", "All-on"): (1836.3, 0.0, 0, 5.86, 0.947, 8.00),
    ("base", "ElasticTree"): (1195.4, 34.9, 24, 13.00, 0.576, 2.92),
    ("base", "ET+Hedera"): (1195.4, 34.9, 1, 6.46, 0.900, 2.92),
    ("base", "SmartFCT-like"): (1182.3, 35.6, 5, 14.00, 0.798, 2.62),
    ("base", "Wang22-like"): (1160.6, 36.8, 12, 9.10, 0.806, 2.25),
    ("base", "CoNap"): (581.7, 68.3, 0, 8.00, 0.950, 2.46),
    ("base", "Oracle"): (553.3, 69.9, 0, 8.00, 0.950, 2.33),
    ("kvheavy", "All-on"): (1836.3, 0.0, 0, 4.70, 0.984, 8.00),
    ("kvheavy", "ElasticTree"): (1119.2, 39.1, 24, 21.04, 0.627, 2.00),
    ("kvheavy", "ET+Hedera"): (1119.2, 39.1, 24, 10.56, 0.770, 2.00),
    ("kvheavy", "SmartFCT-like"): (1781.4, 3.0, 2, 8.08, 0.977, 7.58),
    ("kvheavy", "Wang22-like"): (1119.2, 39.1, 24, 10.56, 0.770, 2.00),
    ("kvheavy", "CoNap"): (717.5, 60.9, 0, 4.76, 0.975, 2.00),
    ("kvheavy", "Oracle"): (713.7, 61.1, 0, 4.76, 0.975, 2.00),
}
PAPER_ABL = {
    ("base", "CoNap"): (68.3, 186.9, 0, 8.00, 2.46),
    ("base", "CoNap-Sim"): (58.7, 177.6, 0, 8.00, 2.00),
    ("base", "w/o adaptive pairing"): (58.0, 173.8, 0, 7.87, 2.00),
    ("base", "w/o placement"): (54.0, 212.7, 0, 8.00, 2.04),
    ("base", "w/o placement, rate, NIC"): (33.8, 212.7, 0, 4.44, 2.04),
    ("base", "ET+Hedera"): (34.9, 192.4, 1, 6.46, 2.92),
    ("kvheavy", "CoNap"): (60.9, 120.0, 0, 4.76, 2.00),
    ("kvheavy", "CoNap-Sim"): (59.0, 139.2, 0, 4.00, 2.00),
    ("kvheavy", "w/o adaptive pairing"): (57.3, 169.9, 0, 4.74, 2.00),
    ("kvheavy", "w/o placement"): (48.4, 299.5, 0, 4.22, 2.88),
    ("kvheavy", "w/o placement, rate, NIC"): (29.5, 291.1, 0, 4.19, 3.46),
    ("kvheavy", "ET+Hedera"): (39.1, 116.2, 24, 10.56, 2.00),
}
TOL = (0.051, 0.051, 0, 0.0051, 0.00051, 0.0051)


def check(path, paper, cols, tols, name):
    t = pd.read_csv(path)
    t["method"] = t.method.str.strip()
    bad = 0
    for (sc, m), ref in paper.items():
        r = t[(t.scenario == sc) & (t.method == m)]
        if r.empty:
            print(f"  MISSING  {name} {sc} {m}")
            bad += 1
            continue
        got = [float(r.iloc[0][c]) for c in cols]
        diff = [c for c, g, e, tl in zip(cols, got, ref, tols) if abs(g - e) > tl]
        if diff:
            bad += 1
            print(f"  DIFFERS  {name} {sc:8s} {m:26s} " + ", ".join(f"{c}: paper {e} vs {g:.4g}" for c, g, e in zip(cols, got, ref) if c in diff))
    print(f"{name}: {len(paper) - bad}/{len(paper)} rows match the paper")
    return bad


if __name__ == "__main__":
    main = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "table_main.csv")
    abl = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "table_ablation.csv")
    bad = check(main, PAPER_MAIN, ["energy_kwh", "saving_pct", "violations", "worst_sd", "min_bg", "mean_spines"], TOL, "Table 4")
    bad += check(abl, PAPER_ABL, ["saving_pct", "wu_kwh", "violations", "worst_sd", "mean_spines"],
                 (0.051, 0.051, 0, 0.0051, 0.0051), "Table 5")
    print("ALL VALUES MATCH THE PAPER" if bad == 0 else f"{bad} row(s) differ from the paper")
    sys.exit(1 if bad else 0)
