# CoNap — reproducibility package

Code, data and scripts for the paper

> **CoNap: Coflow-Aware Network Napping for Energy-Efficient LLM Inference Clusters**
> X. Fan, Y. Zhang, Y. Gao, S. Wang, P. Cao

CoNap saves energy in the scale-out network of prefill–decode (PD) disaggregated LLM inference clusters.
A slow layer (SDN controller) chooses which spines, uplinks, link rates and NIC ports stay on, using the
inference scheduler's traffic forecast and a learned tail-latency evaluator with conformal risk control.
A fast layer (scheduler plug-in) chooses the decode server of every KV-cache transfer.
Everything is evaluated with a flow-level simulator included here.

Reviewers can verify the paper at three levels of effort:

| Level | What it covers |
|---|---|
| **A. Tables and figures from the shipped results** | every number of Tables 3–5 and every data figure is recomputed from the raw results and checked against the paper |
| **B. Smoke test** | the complete pipeline (simulator, all baselines, dataset generation, evaluator training with conformal calibration, CoNap with the learned evaluator, evaluator comparison) runs end to end at toy scale |
| **C. Full reproduction** | re-runs all experiments of the paper (1024 GPUs, 24 h) |

## 0. One-click reproduction

```bash
pip install -r requirements.txt
./reproduce.sh              # levels A + B
```
The script recomputes Tables 3, 4 and 5 from the raw results and checks every value against the paper,
regenerates all figures (and compiles them into
`paper_figures/figures_preview.pdf` if `pdflatex` is available), and runs the whole pipeline at toy scale.
Other modes: `./reproduce.sh tables | smoke | full | evaluator` (`full` and `evaluator` are level C;
set `NPROC` to the number of cores to use).

---

## 1. Contents

```
CoNap-artifact/
├── README.md
├── reproduce.sh               one-click entry point (see Section 0)
├── requirements.txt
├── code/                      simulator, controller, evaluator (all Python)
│   ├── netsim.py              topology, power model, max-min fluid network, routing, PD placement/pairing,
│   │                          single-window simulation, scenarios, 24-hour load curves
│   ├── controller.py          all policies (CoNap, CoNap-Sim and ablations, baselines, oracle),
│   │                          candidate generation with N+1 redundancy, hysteresis, evaluation
│   ├── run_day.py             one full day: SLO calibration -> decisions -> hysteresis -> evaluation
│   ├── merge_policy.py        merge a separately run policy (e.g. CoNap) into a results folder
│   ├── analyze_day.py         per-scenario summary table and diagnostic plots
│   ├── compare_scenarios.py   summary over all scenarios / scales / periods that were run
│   ├── gen_dataset.py         training/test data for the learned evaluator ("scenario groups")
│   ├── train_evaluator.py     trains the MLP evaluator + conformal risk control -> conap_evaluator.pt
│   ├── learned_eval.py        loads the evaluator and judges a batch of candidates (used by the controller)
│   └── gnn_surrogate.py       evaluator study: MLP vs GNN vs Hybrid vs analytic rule (Table 5)
├── scripts/
│   ├── make_tables.py         Tables 3 and 4 from per-hour results
│   ├── check_paper_numbers.py compares recomputed Tables 3-4 with the values printed in the paper
│   ├── make_table5.py         Table 5 and the evaluator numbers of Sec. 6.7, checked against the paper
│   ├── smoke_test.sh          level B
│   ├── run_evaluator_study.sh level C, part 1 (evaluator data, Table 5, evaluator for CoNap)
│   └── run_full_day.sh        level C, part 2 (full-day experiments, Tables 3-4)
├── results/
│   ├── hourly_1024gpu_60min.csv   raw per-hour results behind Tables 3-4 and Figs. 4-8
│   │                              (all 11 policies x 2 scenarios x 24 hours)
│   ├── conap_evaluator.pt         evaluator used for the full-day CoNap runs (MLP, alpha = 5%)
│   └── eval_study/gnn_results.csv raw results of the evaluator comparison behind Table 5
│                                  (MLP / GNN / Hybrid / analytic rule, with and without conformal calibration, 3 seeds)
└── paper_figures/
    ├── gen_figures.py         regenerates figures/*.tex (pgfplots) from the results
    ├── prepare_azure_trace.py builds data/azure2024_10min.csv from the public Azure trace (Fig. 1)
    ├── data/azure2024_10min.csv
    ├── figures/*.tex          TikZ/pgfplots sources of all figures of the paper
    └── figures_preview.tex    compiles all figures on their own (pdflatex)
```

## 2. Setup

```bash
conda create -n conap python=3.10 -y && conda activate conap
pip install -r requirements.txt          # numpy, pandas, matplotlib, torch
```
A GPU is optional (it only speeds up evaluator training). For the figures, any TeX Live ≥ 2020 with
`pgfplots` (compat 1.18) is enough.

All scripts are run from the repository root; the shell scripts set `PYTHONPATH` to `code/` themselves.

## 3. Level A — tables and figures from the shipped results

```bash
python scripts/make_tables.py                 # prints Tables 3 and 4, writes results/table_*.csv
cd paper_figures && python gen_figures.py     # rewrites figures/*.tex from the data
pdflatex figures_preview.tex                  # all figures in one PDF (run twice for the legends)
```
`make_tables.py` recomputes daily energy, saving vs. All-on, SLO-violation hours, worst P99 slowdown,
minimum background satisfaction and mean number of active spines directly from the per-hour rows,
and prints the fraction of the oracle saving reached by CoNap.
`gen_figures.py` asserts that the daily energy and the number of violations of every curve match the tables
before writing a figure.

## 4. Level B — smoke test

```bash
NPROC=2 bash scripts/smoke_test.sh            # outputs in ./smoke
```
Runs every component at toy scale (128 GPUs, 3 hours, 60 scenario groups, 30 training epochs).
It checks that the pipeline works; the numbers are **not** those of the paper.

## 5. Level C — full reproduction

### 5.1 Learned evaluator (Section 4.3, Section 6.7, Table 5)

```bash
NPROC=48 bash scripts/run_evaluator_study.sh  # outputs in ./evaluator
```
1. generates the evaluator data with the simulator: symmetric scenario groups for 16–128 servers, asymmetric
   groups (link failures + traffic hotspots) for 16–64 servers, and asymmetric test sets with 256 and 512 servers;
2. compares MLP, GNN, Hybrid and the ElasticTree-style analytic rule, each calibrated with conformal risk control
   at α = 5% and 10%, over 3 seeds (`evaluator/eval_study/gnn_results.csv`, `gnn_results_seeds.csv`,
   `gnn_plain_summary.csv`), then prints Table 5 with `scripts/make_table5.py`;
3. trains the evaluator used by CoNap (MLP, α = 5%) → `evaluator/conap_evaluator.pt`.

The evaluator shipped in `results/conap_evaluator.pt` is the one used for the paper (MLP, α = 5%, trained on the
combined symmetric + asymmetric data).

`gen_dataset.py` saves `<output>.partial` regularly, so an interrupted run keeps the finished groups.

### 5.2 Full-day experiments (Tables 3–4, Figs. 4–8)

```bash
NPROC=48 bash scripts/run_full_day.sh         # outputs in ./runs
```
For each scenario (`base`, `kvheavy`) it runs all policies on the 1024-GPU cluster for 24 hours with a 60-minute
slow layer, runs CoNap with the learned evaluator, merges the two, and finally prints Tables 3 and 4 with
`make_tables.py --from-runs runs`. The evaluator is taken from `$EVALUATOR`, else `results/conap_evaluator.pt`,
else `evaluator/conap_evaluator.pt`.

Single steps, if preferred:
```bash
export PYTHONPATH=$PWD/code
python code/run_day.py 48 24 kvheavy 1024 60          # args: processes, hours, scenario, GPUs, slow-layer minutes
CONAP_EVALUATOR=evaluator/conap_evaluator.pt ONLY_POLICIES=ours_mlp RUN_OUTDIR=results_kvheavy_1024g_60m_mlp \
  python code/run_day.py 48 24 kvheavy 1024 60
python code/merge_policy.py results_kvheavy_1024g_60m_mlp results_kvheavy_1024g_60m
python code/analyze_day.py kvheavy 1024 60
```
Other scales: GPUs ∈ {128, 256, 512, 1024, 2048, 4096, 8192}; slow-layer period: any divisor of 60 minutes
(e.g. 15).

## 6. Where each result of the paper comes from

| Paper | Produced by | Output |
|---|---|---|
| Fig. 1 (Azure trace) | `paper_figures/prepare_azure_trace.py` → `gen_figures.py` | `figures/fig_trace.tex` |
| Figs. 2–3 (topology, architecture) | drawn by hand in TikZ | `figures/fig_topology.tex`, architecture in the paper source |
| Table 3, Fig. 4 (saving + violations) | `run_full_day.sh` → `make_tables.py` / `gen_figures.py` | `results/table_main.csv`, `fig_saving.tex` |
| Fig. 5 (SLO metrics per hour) | same per-hour results | `fig_tail.tex` |
| Fig. 6 (load + power per hour) | same per-hour results + load curves in `netsim.py` | `fig_power.tex` |
| Table 4, Fig. 7 (ablation) | same per-hour results (CoNap-Sim variants) | `results/table_ablation.csv`, `fig_ablation.tex` |
| Fig. 8 (energy breakdown) | same per-hour results | `fig_breakdown.tex` |
| Table 5, evaluator numbers of Sec. 6.7 | shipped: `scripts/make_table5.py`; regenerate: `run_evaluator_study.sh` (`gnn_surrogate.py`) | `results/eval_study/table5.csv` |
| CoNap decision time per period | `run_day.py` with `ours_mlp` | `decide_time.csv` + console |

Policy names in the code and in the paper:

| Code | Paper |
|---|---|
| `ours_mlp` | **CoNap** (learned evaluator + conformal calibration) |
| `ours_adaptive` | CoNap-Sim (simulation-based evaluator) |
| `ours_joint` | CoNap-Sim w/o adaptive pairing (fixed leaf-local pairing) |
| `ours_net` | CoNap-Sim w/o placement |
| `ours_core` | CoNap-Sim w/o placement, rate and NIC ports |
| `allon`, `elastictree`, `et_hedera`, `smartfct`, `wang22`, `oracle` | All-on, ElasticTree, ET+Hedera, SmartFCT-like, Wang22-like, Oracle |

## 7. Key parameters (Table 1 of the paper)

| Parameter | Location |
|---|---|
| topology presets, power model, scenario loads, intra-period fluctuation, fast-layer thresholds | `NP` and `SCENARIOS` in `code/netsim.py` |
| N+1 redundancy (`MIN_SPINES = 2`, `MIN_EDGE_PORTS = 2`, uplinks ∈ {U/2, U}) | `code/controller.py` (env. `CONAP_MIN_SPINES=1` disables it) |
| forecast error 10%, robust probing ×1.1, hysteresis 5%, fewer-spines tolerance 5%, CoNap-Sim margins | `code/controller.py` |
| SLO calibration (1.5 × worst all-on P99 slowdown; worst all-on background satisfaction − 0.02) | `SD_FACTOR`, `BG_SLACK` in `code/run_day.py` |
| evaluator architecture, features, conformal risk control | `code/gnn_surrogate.py` (`MLP`, `build`, `crc_calibrate`) |
| conformal target α | 3rd argument of `train_evaluator.py` |

## 8. Notes on determinism

All randomness is seeded. Re-running a configuration reproduces the per-hour results exactly; the evaluator
training is deterministic on CPU with a fixed seed (on GPU, cuDNN may introduce tiny floating-point differences).
Wall-clock columns (`sim_seconds`, decision times) naturally differ between machines. `gen_dataset.py` writes
its rows sorted by scenario group (`gid`), so the generated data and the train/validation/calibration splits
do not depend on the number of worker processes.

## 9. Third-party data

`paper_figures/data/azure2024_10min.csv` is derived from the *Azure LLM inference trace 2024*
(CC-BY 4.0; J. Stojkovic et al., "DynamoLLM", HPCA 2025), https://github.com/Azure/AzurePublicDataset.

## 10. Contact

Xi Fan — s328208@studenti.polito.it
