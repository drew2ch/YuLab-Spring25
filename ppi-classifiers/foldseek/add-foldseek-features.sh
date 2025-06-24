#!/usr/bin/env bash
set -euo pipefail

# prescribes input features from foldseek (.tsv) results
# using process_foldseek_results.py
# Andrew Chung, hc893; 6/23/2025

types=('train' 'test' 'validation')
DIR='/c/Users/hychu/OneDrive/Desktop/Summer25/github/ppi-classifiers/data'
PYTHON='/c/Users/hychu/AppData/Local/Programs/Python/Python312/python'

for type in "${types[@]}"; do
    echo "Processing $type data..."
    $PYTHON process_foldseek_results.py \
        --input_file "${DIR}/${type}_set.txt" \
        --output_file "${DIR}/${type}_data.csv" \
        --is_txt
done
