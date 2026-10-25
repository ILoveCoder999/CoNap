#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$ROOT/evaluator}"
NPROC="${NPROC:-$(nproc)}"
mkdir -p "$WORK" && cd "$WORK"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
C="$ROOT/code"

python "$C/gen_dataset.py" "$NPROC" 3000 8 data_sym.csv 128,256,512,1024 0 0.0
python "$C/gen_dataset.py" "$NPROC" 2000 8 data_asym.csv 128,256,512 200000 1.0
python "$C/gen_dataset.py" "$NPROC" 100 8 test_256srv_asym.csv 2048 400000 1.0
python "$C/gen_dataset.py" "$NPROC" 100 8 test_512srv_asym.csv 4096 500000 1.0

python -c "import pandas as pd; pd.concat([pd.read_csv(f) for f in ('data_sym.csv','data_asym.csv')], ignore_index=True).to_csv('data_all.csv', index=False)"

MODELS=mlp,gnn,hyb CRC_ALPHAS=0.05,0.10 OUTDIR=eval_study \
  python "$C/gnn_surrogate.py" data_all.csv 400 test_256srv_asym.csv,test_512srv_asym.csv 3

python "$ROOT/scripts/make_table5.py" eval_study/gnn_results.csv || true

python "$C/train_evaluator.py" data_all.csv conap_evaluator.pt 0.05 400
echo "Evaluator written to $WORK/conap_evaluator.pt"
