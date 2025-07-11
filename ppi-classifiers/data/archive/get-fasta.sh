#!/usr/bin/env bash
set -euo pipefail

# Bash script to compile unique individual proteins in PrePPI-Genomic combined PPI
#   pairs, as well as extracting the associated .fasta files for each protein monomer.
# Andrew Chung, hc893, 6/13/2025

# import features data file
# DATA="features.csv"
INFILE="monomers.txt"
OUTFILE="uniprot_seqs_both_sets.txt"

# identify unique individual proteins from features file
# echo "Parsing (training) features data file, iterating through PPI pairs..."
# unique_monomers=()
# mapfile -t unique_monomers < <(tail -n +2 "$DATA" | cut -d',' -f2 | tr ':' '\n' | sort | uniq)
# first, save up the list of monomers into .txt
# printf "%s\n" "${unique_monomers[@]}" > training_monomers.txt
# echo "Unique proteins successfully identified and saved to 'training_monomers.txt'."
# echo "${#unique_monomers[@]} distinct monomers were found."

# import .FASTA files for all monomers, derived from rest.uniprot.org
# then save 'copies' in /fasta
# mkdir -p ../../output/fasta

unique_monomers=()
mapfile -t unique_monomers < ${INFILE}
uniprot_url="https://rest.uniprot.org/uniprotkb/"
> "${OUTFILE}"

unsuccessful=()
echo "Accessing .FASTA files from rest.uniprot.org..."
for monomer in "${unique_monomers[@]}"; do

  if [[ -z "${monomer}" ]]; then continue; fi # check if length zero, skip if so
  echo "Processing ${monomer}..."

  temp_file=$(mktemp)
  status=$(curl -s -L -w "%{http_code}" -o "${temp_file}" "${uniprot_url}${monomer}.fasta")

  if [[ "${status}" -eq 200 ]]; then

    # get uniprot ID and single-line sequence
    uniprot_id=$(head -n1 "${temp_file}" | cut -d'|' -f2)
    sequence=$(tail -n +2 "${temp_file}" | tr -d '\n')
    echo "${uniprot_id} ${sequence}" >> "${OUTFILE}"
    
    echo "Successfully imported ${monomer}."

  else
    echo "  -> Failed to import ${monomer}, HTTP status ${status}"
  fi

  rm "${temp_file}"
done
echo "Successfully loaded ${#unique_monomers[@]} FASTA sequences."
