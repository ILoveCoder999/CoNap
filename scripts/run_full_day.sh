#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$ROOT/runs}"
NPROC="${NPROC:-$(nproc)}"
if [ -z "${EVALUATOR:-}" ]; then
  if [ -f "$ROOT/results/conap_evaluator.pt" ]; then EVALUATOR="$ROOT/results/conap_evaluator.pt"
  else EVALUATOR="$ROOT/evaluator/conap_evaluator.pt"; fi
fi
mkdir -p "$WORK" && cd "$WORK"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
C="$ROOT/code"
[ -f "$EVALUATOR" ] || { echo "missing $EVALUATOR: run scripts/run_evaluator_study.sh first (or copy the provided evaluator)"; exit 1; }

for s in base kvheavy; do
  python "$C/run_day.py" "$NPROC" 24 "$s" 1024 60
  CONAP_EVALUATOR="$EVALUATOR" ONLY_POLICIES=ours_mlp RUN_OUTDIR="results_${s}_1024g_60m_mlp" \
    python "$C/run_day.py" "$NPROC" 24 "$s" 1024 60
  python "$C/merge_policy.py" "results_${s}_1024g_60m_mlp" "results_${s}_1024g_60m"
  python "$C/analyze_day.py" "$s" 1024 60
done
python "$C/compare_scenarios.py"
python "$ROOT/scripts/make_tables.py" --from-runs "$WORK"
python "$ROOT/scripts/check_paper_numbers.py" "$WORK/table_main.csv" "$WORK/table_ablation.csv" || true
