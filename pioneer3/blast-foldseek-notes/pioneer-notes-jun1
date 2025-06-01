# PIONEER Notes

**Andrew Chung**, hc893; 6/1/2025

Some notes on BLAST/Foldseek and general PIONEER/PIONEER2 architecture ahead of the ML Subgroup meeting on 6/2. Information acquired from Juheon's personal notes, NCBI, the PIONEER repository, and ChatGPT.

## Recap: BLAST and Foldseek

**BLAST and Foldseek** are two software tools used in bioinformatics workflows to identify **homologs** of a given protein sequence/structure set. Both accept a **protein sequence** (conventionally in `FASTA` format) as input and runs a query search against a corpus of existing protein databases, ultimately returning a **set of protein sequences** believed to be homologous to the input protein. Each candidate homolog is identified with an **e-value** and **sequence identity**:

- **Sequence Identity** simply measures the *fraction* of aligned residues with matching amino acids between the input protein and the candidate homolog.
- **E-value** is a somewhat abstract concept that quantifies the **expected number of random hits**. That is, “If I repeated this search against a database of this size many times, about $e$ (e.g. one every thousand) hits of this quality would pop up by pure random.” Somewhat analogous to a p-value in statistics; **lower** E-values indicate more significant homolog pairs.

A query search on either BLAST/Foldseek usually returns a large set of candidate homologs; rather than accepting every output sequence, an **E-value upper bound** is usually imposed to filter for "significant" homolog candidates (BLAST: $10^{-5}$, Foldseek: $10^{-3}$).

The main distinction between the two tools lies in the biological dimensionality of input data and the types of homologies detected. 
- **BLAST** is compatible with **nucleotide or amino acid sequences** (i.e. *primary* (I) structure) and is primarily focused on **sequence similarity**, convenient for identifying closely related genes/proteins and initial functional annotation. 
- **Foldseek** accepts **protein 3D structures** (i.e. *tertiary* (III) structure) and emphasizes **structural similarity**, especially in identifying **remote homologs** (proteins that have diverged at the sequential level but whose evolutionary/functional relationships can be structurally inferred) that may not be apparent from sequence alone.

Please refer to `blast-notes.md` and `foldseek-notes.md` for detailed notes on algorithmic/mathematical internals.

## The Bit-Score

Earlier, I discussed the **E-value** as a numeric standard for evaluating the quality of an identified homology. The E-value is used in tandem with **structural identity**; *usable homologs*, vetted through E-value and identity thresholds, are compared in E-values to assess relative "significance" of a homology. Let $S$ denote the raw alignment score; the formula for computing the E-value is thus (details on individual arguments can be found in my other notes)

$$E=Kmn\cdot \exp(-\lambda S)$$

What the E-value does not take into account, though, is **database size**. E-values obtained from a *specific database search* are readily comparable, but not as much so when comparing alignments across diverse search contexts. The **Bit-Score** provides an alternative measure of alignment quality by standardizing for **database size** and other search contexts like the **type of substitution matrix** (e.g. BLOSUM) used to compute $S$. Simply, the formula to compute the Bit-Score is

$$S'=\frac{\lambda S-\ln K}{\ln2}$$

where $S$ is the raw alignment score and $\lambda, K$ are derived from Karlin-Altschul Statistics. Opposite to the E-value, **higher** Bit-scores point towards homolog significance.

[Substitution Matrix](https://en.wikipedia.org/wiki/Substitution_matrix), [BLOSUM](https://en.wikipedia.org/wiki/BLOSUM), [K-A Statistics](https://www.cbcb.umd.edu/confcour/Statistics_Talk_UMD.pdf)

## PIONEER Basics

**PIONEER** (and **PIONEER 2.0**) were developed as deep-learning model to predict **interface residues** of a given **target protein**, based on information derived from its **partner protein**. As such, it accepts a pair of protein sequences **(target, partner)** as input, as well as a set of qualitative features that capture sequence and structural homology and are known to predict protein binary interactions (PPI).

Building from the original PIONEER model, PIONEER 2.0 incorporates **PrePPI-derived features** (structural neighborhood). The *guiding principle* for both is

> If a pair of proteins A–B (in your training set) has homologs A′–B′ that are known to interact in some structural database (e.g., a solved PDB complex), then that gives you high confidence that A–B will behave similarly.

The workflow for **interface residue discovery** can be summarized as follows.
1. Consider an original training pair $(A,B)$.
2. Using BLAST/Foldseek, identify distinct sets of protein sequences $(A_1',A_2',...,A_m')$ homologous to $A$ and $(B_1',B_2',...,B_n')$ homologous to $B$, adhering to some cutoff values (see above). *OPTIONAL: you can prune the homolog data by keeping only the top $N$ hits by E-value or Bit-score.*
3. With each of the $m\times n$ possible combinations $(A_i', B_j'),\forall i\in\mathbb{Z}_m^+,\forall j\in\mathbb{Z}_n^+$, check if there is a known interface (e.g. PBD, AFM Database); skip homolog pairs without valid interfaces.
4. For each verified interface, extract interface residue indices (e.g. all $A_i'$ residues within $5\text{\r{A}}$ of any $B_j'$ atom), and map the interface residues back (by alignment) to the original pair $(A,B)$ (that becomes your “ground truth interface” for training PIONEER). Incorporate it as **structural neighborhood information**.
5. Thus, “When A interacts with B, the putative interface on A is [these residues], and on B is [these residues].”

Once interface residues are identified for viable homolog-pair candidates, the next logical step would be to embed these findings as **structural-neighborhood features**, which can be embedded in a variety of ways. NOTE: to my knowledge we haven't gone over how to specifically encode these features yet, so for now I am going off of what ChatGPT has given me.

- For each $A$ and $B$, take a union of the sets of interface residues from each homolog (e.g. given $A_1': \{17,20,22,...\}, A_2':\{25,28,30,...\},...$, find $A_1' \cup A_2'$; repeat for $B$), and apply a *binary mask* along the original sequence (i.e. $0$ for no interface, $1$ for interface).
- Additionally, homolog pairs (and binary masks) can be *weighted* by their bit-scores to account for relative interface strength (combined bit-scores can be sums or max-values).

Now the model has a per‐residue feature that says “in known homologous complexes, this residue is frequently observed at an interface.”
