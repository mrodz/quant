#!/usr/bin/env bash
# run_pipeline.sh — Run the full credit factor pipeline in order
# Usage: bash run_pipeline.sh
# From:  ~/quant (project root)

set -e  # stop on first error

PIPELINE="research/credit_factor_pipeline"
PYTHON=".venv/bin/python"

run() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP $1: $2"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    $PYTHON $PIPELINE/$2
}

START=$(date +%s)

run 1 01_ingest.py
run 2 02_features.py
run 3 03_targets.py
run 4 04_model.py
run 5 05_shap.py
run 6 06_signal.py
run 7 07_validate.py

END=$(date +%s)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PIPELINE COMPLETE  ($(( (END - START) / 60 ))m $(( (END - START) % 60 ))s)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"