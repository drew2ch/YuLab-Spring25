#!/usr/bin/env bash
# example script to run BLASTP query search

set -euo pipefail

# re-write FASTA file with a here-document
cat > example_sequence.fasta << 'EOF'
>partial_alpha_chain
VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR
EOF
echo "FASTA file 'example_sequence.fasta' created."

# run BLASTP on 'nr' database
echo "Running BLASTP search on 'nr' database..."
blastp \
  -query example_sequence.fasta \
  -db nr \
  -evalue 1e-5 \
  -outfmt 6 \   # tabular; aseqid sseqid pident length mismatch gapopen
  -max_target_seqs 20 \
  | tee blastp_output.txt   # save ooutput to plain text
echo "BLASTP search completed. Output saved to 'blastp_output.txt'."
