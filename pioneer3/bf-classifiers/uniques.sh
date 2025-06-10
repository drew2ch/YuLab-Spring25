#!/usr/bin/env bash
set -euo pipefail

# Andrew Chung, hc893, 6/10/2025

DATA="af3_preppi_features_combined_jun10.csv"
TXT="unique_monomers.txt"

# extract all common dimer pairs, then identify unique 
tail -n +2 "$DATA" | cut -d',' -f2 | tr ':' '\n' | sort | uniq > "$TXT"
