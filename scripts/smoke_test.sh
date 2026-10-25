#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$ROOT/smoke}"
NPROC="${NPROC:-2}"
mkdir -p "$WORK" && cd "$WORK"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
C="$ROOT/code"

echo "== 1/5 baselines, oracle and simulation-based CoNap-Sim variants (kvheavy, 128 GPUs, 3 periods)"
python "$C/run_day.py" "$NPROC" 3 kvheavy 128 60

echo "== 2/5 tiny evaluator dataset (60 scenario groups, 16 servers, half asymmetric)"
python "$C/gen_dataset.py" "$NPROC" 60 4 tiny_data.csv 128 0 0.5

echo "== 3/5 train the MLP evaluator + conformal calibration (alpha = 0.10, 30 epochs)"
python "$C/train_evaluator.py" tiny_data.csv conap_evaluator.pt 0.10 30

echo "== 4/5 CoNap with the learned evaluator, merged into the same results folder"
CONAP_EVALUATOR=conap_evaluator.pt ONLY_POLICIES=ours_mlp RUN_OUTDIR=results_kvheavy_128g_60m_mlp \
  python "$C/run_day.py" "$NPROC" 3 kvheavy 128 60
python "$C/merge_policy.py" results_kvheavy_128g_60m_mlp results_kvheavy_128g_60m
python "$C/analyze_day.py" kvheavy 128 60

echo "== 5/5 evaluator comparison (MLP / GNN / Hybrid / analytic rule), 10 epochs"
OUTDIR=eval_study python "$C/gnn_surrogate.py" tiny_data.csv 10

echo "Smoke test finished. Outputs are in $WORK"
