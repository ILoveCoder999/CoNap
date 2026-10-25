#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
export NPROC="${NPROC:-$(nproc 2>/dev/null || echo 2)}"
PY="${PYTHON:-python3}"
line() { printf '\n==================== %s ====================\n' "$1"; }

line "checking Python dependencies"
if ! $PY -c "import numpy, pandas, matplotlib, torch" 2>/dev/null; then
  echo "Missing packages. Install them with:  $PY -m pip install -r requirements.txt"; exit 1
fi
$PY -c "import sys, numpy, pandas, torch; print(f'python {sys.version.split()[0]}, numpy {numpy.__version__}, pandas {pandas.__version__}, torch {torch.__version__}')"

tables() {
  line "1/3 Tables 4 and 5 from the raw per-hour results (results/hourly_1024gpu_60min.csv)"
  $PY scripts/make_tables.py
  echo
  $PY scripts/check_paper_numbers.py
  line "Table 6 from the raw evaluator-study results (results/eval_study/gnn_results.csv)"
  $PY scripts/make_table5.py
  line "2/3 figures (pgfplots sources in paper_figures/figures/)"
  (cd paper_figures && $PY gen_figures.py)
  if command -v pdflatex >/dev/null 2>&1; then
    (cd paper_figures && pdflatex -interaction=nonstopmode figures_preview.tex >/dev/null && \
     pdflatex -interaction=nonstopmode figures_preview.tex >/dev/null && rm -f figures_preview.aux figures_preview.log) \
     && echo "all figures compiled into paper_figures/figures_preview.pdf" \
     || echo "pdflatex failed; the .tex figure sources are still up to date"
  else
    echo "pdflatex not found: figure sources regenerated, PDF preview skipped"
  fi
}

smoke() {
  line "3/3 smoke test of the full pipeline at toy scale (outputs in ./smoke)"
  bash scripts/smoke_test.sh "$ROOT/smoke"
}

case "$MODE" in
  quick)     tables; smoke ;;
  tables)    tables ;;
  smoke)     smoke ;;
  full)      line "full-day experiments (outputs in ./runs)"; bash scripts/run_full_day.sh "$ROOT/runs" ;;
  evaluator) line "evaluator study (outputs in ./evaluator)"; bash scripts/run_evaluator_study.sh "$ROOT/evaluator" ;;
  *) echo "unknown mode '$MODE' (use: quick | tables | smoke | full | evaluator)"; exit 1 ;;
esac
line "done"
