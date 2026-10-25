import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
out = []
for name in ("code", "conv"):
    path = os.path.join(HERE, f"AzureLLMInferenceTrace_{name}_1week.csv")
    for ch in pd.read_csv(path, chunksize=4_000_000, dtype={"ContextTokens": "int64", "GeneratedTokens": "int64"}):
        t = pd.to_datetime(ch.TIMESTAMP.str.slice(0, 15) + "0", format="%Y-%m-%d %H:%M")
        g = (pd.DataFrame({"t": t, "ctx": ch.ContextTokens, "gen": ch.GeneratedTokens})
             .groupby("t").agg(n=("ctx", "size"), ctx=("ctx", "sum"), gen=("gen", "sum")))
        g["svc"] = name
        out.append(g.reset_index())
d = pd.concat(out).groupby(["t", "svc"]).sum().reset_index()
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
d.to_csv(os.path.join(HERE, "data", "azure2024_10min.csv"), index=False)
print(d.groupby("svc").agg(first=("t", "min"), last=("t", "max"), requests=("n", "sum")))
