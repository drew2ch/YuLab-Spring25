#!/usr/bin/env bash
set -euo pipefail

# Bash script to compile unique individual proteins in PrePPI-Genomic combined PPI
#   pairs, as well as extracting the associated .fasta files for each protein monomer.
# Then, it calls the generate_hits.py script that runs Foldseek and generates .tsv
#   files for each UniProt ID, containing information on their identified homologs.
# Andrew Chung, hc893, 6/13/2025

# import features data file
DATA="features.csv"
TXT="monomers.txt"

# identify unique individual proteins from features file
echo "Parsing features data file, iterating through PPI pairs..."
unique_monomers=()
mapfile -t unique_monomers < <(tail -n +2 "$DATA" | cut -d',' -f2 | tr ':' '\n' | sort | uniq)
# first, save up the list of monomers into .txt
printf "%s\n" "${unique_monomers[@]}" > monomers.txt
echo "Unique proteins successfully identified and saved to 'monomers.txt'."
echo "${#unique_monomers[@]} distinct monomers were found."

# import .FASTA files for all monomers, derived from rest.uniprot.org
# then save 'copies' in /fasta
mkdir -p ../../output/fasta
uniprot_url="https://rest.uniprot.org/uniprotkb/"
echo "Accessing .FASTA files from rest.uniprot.org..."
for monomer in "${unique_monomers[@]}"; do
  if [[ -z "${monomer}" ]]; then continue; fi # check if length zero, skip if so
  # echo "Processing ${monomer}..."
  status=$(curl -s -L -w "%{http_code}" -o "../../output/fasta/${monomer}.fasta" "${uniprot_url}${monomer}.fasta")
  if [[ "${status}" -eq 200 ]]; then
    { # integrate multi-line sequences into a single line
        head -n1 "../../output/fasta/${monomer}.fasta"
        tail -n +2 "../../output/fasta/${monomer}.fasta" | paste -sd ''
    } > "../../output/fasta/${monomer}.fasta.tmp" && \
    mv "../../output/fasta/${monomer}.fasta.tmp" "../../output/fasta/${monomer}.fasta"
    echo "Successfully imported ${monomer}.fasta"
  else
    echo "Failed to import ${monomer}.fasta, HTTP status ${status}"
    rm -f "fasta/${monomer}.fasta"
  fi
done

# now, run foldseek on each individual query protein, then store the search (hit)
# results into .tsv files

# first, 
mkdir -p ../../output/tsv
DEFAULT_USER_BASE_DIR="C:/Users/hychu/OneDrive/Desktop/Summer 2025/"
# REPOSITORY_DIR="github/ppi-classifiers"
# define some paths
FASTA_INPUT_PATH="${DEFAULT_USER_BASE_DIR}/output/fasta"
TSV_OUTPUT_PATH="${DEFAULT_USER_BASE_DIR}/output/tsv"
FOLDSEEK_INTERNAL_TEMP_DIR="${DEFAULT_USER_BASE_DIR}/transient/foldseek_internal_temp"

# run python script
python3 generate_hits.py \
  --fasta_input_dir "${FASTA_INPUT_PATH}" \
  --tsv_output_dir "${TSV_OUTPUT_PATH}" \
  --foldseek_internal_tmp_dir "${FOLDSEEK_INTERNAL_TEMP_DIR}" \
  --pdb_db_foldseek "${DEFAULT_USER_BASE_DIR}/databases/pdb_" \
  --prostt5_weights "${DEFAULT_USER_BASE_DIR}/databases/weights" \
  --force_rerun --total_batches 1 --current_batch_index 0

# docker run --rm -v ${PWD}:/data foldseek-binary
