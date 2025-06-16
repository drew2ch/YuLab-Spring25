#!/usr/bin/env bash
set -euo pipefail

# Running Foldseek across 5,196 unique proteins identified.
# Using batch size of 4, with parallel processing of 4 agents enabled.
PYTHON="/c/Users/hychu/AppData/Local/Programs/Python/Python312/python"

TOTAL_BATCHES=4
for (( i=0; i<TOTAL_BATCHES; i++ )); do
    echo "Starting batch $i of $TOTAL_BATCHES..."
    "$PYTHON" generate_hits.py \
    --total_batches "$TOTAL_BATCHES" \
    --current_batch_index "$i" \
    --force_rerun &
done

wait
