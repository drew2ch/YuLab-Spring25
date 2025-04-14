# BLAST Notes

Andrew Chung, hc893, 4/14/2025

The Basic Local Alignment Search Tool (BLAST) finds regions of local similarity between protein or nucleotide sequences. The program compares nucleotide or protein sequences to sequence in a database and calculates the statistical significance of the matches. Below is a comprehensive run-through of the `blastp` (Protein BLAST) workflow.

Information acquired from [NCBI website](https://www.ncbi.nlm.nih.gov/books/NBK1734/) and ChatGPT.

## Query Sequence Pre-processing

Query sequences must be loaded into **FASTA format**, and pre-processing steps such as masking of low-complexity regions (e.g. repetitive sequences) may be performed. Example FASTA format below:

```fasta
>SequenceName Description
MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYKGQMQDY
```

## Setting up Database

Specifically for local (custom-sequence) databases, use `makeblastdb` command to format:

```bash
makeblastdb -in query.fasta -dbtype prot -out mydb
```

`prot` for Blastp, and `nucl` for Blastn.

## Running a BLAST search

### Command Line

```bash
blastp -query query.fasta -db mydb -out results.txt -outfmt 5
```

### Remote Search with Python

Using the `Biopython` package.

```py
from Bio.Blast import NCBIWWW, NCBIXML

query = """ExampleQuery
MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAYKGQMQDY
"""

result = NCBIWWW.qblast(program = "blastp", database = "nr", sequence = query)

with open("results.xml", "w") as handle:
  handle.write(result.read())
handle.close()
```

## Algorithm Internals and Mathematical Basis

### Seed-and-Extend

The full sequence (query) is broken down into consecutive, overlapping **words** (usually of size 3). For example, the sequence $\text{MTEITAAMVKELR}$ can be decomposed into $\{\text{MTE, TEI, EIT, ITA, TAA, ...}\}$. Among them, certain words are selected as **seeds** either by 

1. An exact match in a protein database sequence, or
2. Neighboring words, with high similarity scores according to a **substitution matrix** (e.g. `BLOSUM62`)

For identified seeds, the alignment is **extended** in either direction, using substitution matrices (like `BLOSUM62`) to score amino acid substitutions until it falls below some threshld (**ungapped alignments**). Some mismatches, insertions, or deletions are tolerated to optimize the alignment score.

> Consider the seed $\text{AYGHKQL}$, with the extension sequences $\text{...AYGHKQLMVW...}$ (query) and $\text{...AYGHKQLIFP...}$ (database). 
>
> If, say, the substitution scores of $\text{M/I}$ and $\text{V/F}$ are positive (to varying degrees), the extension is accepted; once a negative query-to-database substitution score (e.g. $\text{W/P}$) is detected (cumulative alignment score drops) the extension stops in that direction.

**Gapped alignments**, which permit insertions and deletions (indels), can be computed using dynamic programming.

### Alignment Score Computation

For an alignment with lengtb $L$, the cumulative **alignment score** is computed using the following formula:

$$S = \sum_{i=1}^{L}s(q_i,d_i)-G$$

where each $s(q_i,d_i)$ measures the substitution score for aligning query amino acid $q_i$ with database amino acid $d_i$, usually acquired from `BLOSUM62`, and $G$ is the gap penalty to represent the cost of initiating a gap (indels), reflecting the biological reality that indels are less frequent than simple substitutions.

### E-value

The **E-value** quantifies the statistical significance of a particular alignment:

$$E=Kmn\operatorname{exp}(-\lambda S)$$

where

- $m$ is the effective query length, close to the actual query length (save for edge effects/statistical modeling),
- $n$ is the effective database length, similar to $m$; not directly calculated, but estimated from NCBI sizes (e.g. `nr`: $n \approx 5\times10^8$),
- $S$ is the alignment score, computed as above,
- $K,\lambda$ are calculated internally by BLAST's scoring system using Karlin-Altschul statistics, accounting for the substitution matrix, gap penalties, and sequence composition.

Like $p$ values, **lower** $E$ scores indicate **more significant** matches; it represents the number of alignments with scores equal to or better than the observed score expected to occur by random chance in a database search. As a rule of thumb, in varying scales, E-values **close to 0** indicate **strong homology**.

## Pseudocode

Acquired from ChatGPT.

```plaintext
Input: Protein query sequence Q, Protein database D, Word length W (typically 3), Score Threshold T
Output: List of significant high-scoring segment pairs (HSPs)

1. Preprocessing:
   a. Convert Q to overlapping words of length W.
   b. Index each sequence in D by their overlapping words.

2. For each word "w" in Q:
   a. Retrieve all occurrences of "w" in D using the index.
   b. For every hit:
      i. Extend ungapped from the word in both directions.
      ii. If the extended score > T, perform gapped extension.
      iii. Compute the alignment score S:
             S = Sum(s(q_i, d_i)) - (Gap Opening + Number of Gaps * Gap Extension)
      iv. Compute the E-value:
             E = K * m * n * exp(-lambda * S)
      v. If E < significance cutoff (e.g., 0.01), record as a high-scoring segment pair.

3. Return all significant HSPs.
```

## Parsing BLASTP Results with Python

Using the `Biopython` `NCBIXML` package to parse an exemplary alignment result file (`.xml`).

```py
from Bio.Blast import NCBIXML

with open("result.xml", "r") as file:
  record = NCBIXML.read(file)

print("Total Alignments: %d", len(record.alignments))
for alignment in record.alignments:
  print("Title: %s", alignment.title)

  # print attributes: Alignment score, E-value, Query/Subject Sequences for 
  # High-Scoring Segment Pairs (HSPs) - fully-extended local alignments.
  for hsp in alignment.hsps:
    print(f" Score: {hsp.score:.2f}, E-value: {hsp.expect:.2e}")
    print("  Query:", hsp.query[:50], "...")
    print("  Subject:", hsp.sbjct[:50], "...")
```