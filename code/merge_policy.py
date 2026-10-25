import json
import os
import shutil
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
s_slo = json.load(open(os.path.join(src, "slo.json")))
d_slo = json.load(open(os.path.join(dst, "slo.json")))
for k in ("slo_kv_sd_p99", "slo_bg_tput", "min_spines"):
    if s_slo.get(k) != d_slo.get(k):
        sys.exit(f"{k} differs between the two folders: {s_slo.get(k)} vs {d_slo.get(k)}; make sure both were produced with the same code version and settings")
new = pd.read_csv(os.path.join(src, "day_results.csv"))
old = pd.read_csv(os.path.join(dst, "day_results.csv"))
pols = sorted(new.policy.unique())
shutil.copy(os.path.join(dst, "day_results.csv"), os.path.join(dst, "day_results_before_merge.csv"))
merged = pd.concat([old[~old.policy.isin(pols)], new], ignore_index=True)
merged.to_csv(os.path.join(dst, "day_results.csv"), index=False)
if os.path.exists(os.path.join(src, "decide_time.csv")):
    shutil.copy(os.path.join(src, "decide_time.csv"), os.path.join(dst, "decide_time_" + "_".join(pols) + ".csv"))
print(f"Merged the results of {pols} ({len(new)} rows) into {dst}/day_results.csv (original backed up as day_results_before_merge.csv)")
