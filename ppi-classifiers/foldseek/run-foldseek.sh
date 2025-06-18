#!/usr/bin/env bash
#set -euo pipefail

# Running Foldseek across 5,196 unique proteins identified.
# Using batch size of 4, with parallel processing of 4 agents enabled.
# Andrew Chung, hc893; 6/17/2025

PYTHON="/c/Users/hychu/AppData/Local/Programs/Python/Python312/python"
DEFAULT_USER_BASE_DIR="C:/Users/hychu/OneDrive/Desktop/Summer25"
TOTAL_BATCHES=4

for (( i=0; i<TOTAL_BATCHES; i++ )); do
    echo "Starting batch $i of $TOTAL_BATCHES..."
    "$PYTHON" generate_hits.py \
    --total_batches "$TOTAL_BATCHES" \
    --current_batch_index "$i" \
    --force_rerun &
done
wait

# apparently, foldseek results for many proteins among the 5,196 were missing
# so let's identify them and selectively re-run foldseek

echo "Compiling list of proteins with unsuccessful Foldseek results..."

TXT="training_monomers.txt"
all_proteins=()
while IFS= read -r line; do
    uniprot_id="${line%.tsv}"
    all_proteins+=("$uniprot_id")
done < "$TXT"

cd ../../output/tsv
n_files=$(( 5196-$(ls -1 | wc -l) ))
successful_hits=()
mapfile -t successful_hits < <(ls)
for ind in "${!successful_hits[@]}"; do
    successful_hits[ind]="${successful_hits[ind]%.tsv}"
done

# find difference between 2 sets: full protein set vs. successfully run
cd -
printf "%s\n" "${all_proteins[@]}" | sort > all_proteins.tmp
printf "%s\n" "${successful_hits[@]}" | sort > successful_hits.tmp
diff=()
mapfile -t diff < <(comm -23 all_proteins.tmp successful_hits.tmp)

echo "Foldseek was ran unsuccessfully for ${#diff[@]} proteins. .tsv file missing for ${n_files} proteins."
if [[ ${#diff[@]} -ne ${n_files} ]]; then
    echo "Error: mismatch between values." >&2
    exit 1
fi
echo "Saving ${#diff[@]} proteins for re-run..."
: > unsuccesful_hits.txt
printf "%s\n" "${diff[@]}" | sort > unsuccesful_hits.txt
rm *.tmp

# identify FASTA files of unsuccessful hits
cd ../../output
mkdir -p remaining_fasta

copied_count=0
for protein_id in "${diff[@]}"; do
    [[ -z "$protein_id" ]] && continue
    fasta_file="fasta/${protein_id}.fasta"
    if [[ -f $fasta_file ]]; then
        cp $fasta_file remaining_fasta
        echo "${protein_id}.fasta copied."
        ((copied_count++))
    else
        echo "${protein_id}.fasta not found."
    fi
done
echo "${copied_count}/${#diff[@]} FASTA files copied."


TOTAL_BATCHES=4
for (( i=0; i<TOTAL_BATCHES; i++ )); do
    echo "Starting batch $i of $TOTAL_BATCHES..."
    "$PYTHON" generate_hits.py \
    --fasta_input_dir "${DEFAULT_USER_BASE_DIR}/output/remaining_fasta" \
    --total_batches "$TOTAL_BATCHES" \
    --current_batch_index "$i" \
    --force_rerun &
done
wait
