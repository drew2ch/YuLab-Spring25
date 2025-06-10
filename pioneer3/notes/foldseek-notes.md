# FoldSeek Notes

Andrew Chung, hc893, 4/14/2025

Foldseek enables fast and sensitive comparisons of large protein structure sets, supporting monomer and multimer searches, as well as clustering. It does so by representing 3D protein structures in a simplified search space. It runs on CPU, supports GPU acceleration for faster searches, and optionally allows ultra-fast and sensitive comparisons directly from protein sequence inputs using a language model, bypassing the need for structures.

Information acquired from [Steinegger Lab Github](https://github.com/steineggerlab/foldseek) and ChatGPT.

## Query Sequence Pre-Processing

Foldseek accepts protein sequences in standard (e.g. `PDB` or `mmCIF`) file formats, but `FASTA` format is required for sequence-based search with a language model.

Input structures must be pre-processed to extract fundamental information such as backbone coordinates and secondary structure annotations. This involves

- Parsing `PDB` or `mmCIF` files
- Removing heteroatoms or alternative conformations
- Flattening 3D structure into a "linearized" representation

Optionally, a pre-trained embedding model (e.g. ESM-2 or similar transformer) can be used to parse sequences into structures.

## Setting up Database

Foldseek's `createdb` command can be used to create a database:

```bash
foldseek createdb structure\ structures_db
```

To verify that the data base is set up correctly, you can run a test search query:

```bash
foldseek search query_structure.pdb my_database.db result.out --threads 8
```

## Running a Query Submission

```bash
foldseek search query_structure.pdb target_database.db result.out --threads 8 --gpu 1
```

- `--threads` specifies # of CPU threads for parallel processing
- `--gpu` activates GPU acceleration if a compatible CUDA device is available

## Parsing Results

The outputted results file (`result.out`) should contain, for each hit:

- **Alignment Scores** (accounting for structural similarity and statistical significance, E-values)
- **Query/Target ID**
- **E-value** or p-value
- **Geometry** Information (e.g. RMSD, Alignemnt length)

Hits are **ranked** by their alignment scores and presented with **clustering** information to group similar hits. Below is a python script to parse results:

```py
import pandas as pd

results = pd.read_csv("result.out", sep = "\t", header = None, names = [
  "query_id", "target_id", "score", "alignment_length", "E_value", "RMSD"
])

# E-value cutoff of 0.01
significant_hits = results[results["E-value"] < 0.01]
```

## Algorithm Internals and Mathematical Basis

### 3Di Structure Encoding

Protein structures are converted to a **discrete representation**, encoding key **geometric features** (like secondary structure elements, distances, and angles) into a compact form. In some instances, an internal distance matrix (or graphs) is computed to capture inter-residue distances for fast comparison.

### Seed-and-Extend Fast Alignment

Just as in BLAST (refer to BLAST notes), seeds are identified as anchors for alignment extension. Dynamic programming may be incorporated to account for spatial and angular deviations.

### Alignment Score Computation

The overall alignment score combines several elements.

#### Root Mean Square Deviation (RMSD)

The RMSD, used to quantify the average distance between consecutive alpha carbons ($C_\alpha$) after optimal superposition:

$$\text{RMSD} = \sqrt{N^{-1}\sum_{i=1}^{N}||\mathbf{x}_i - \mathbf{y}_i||^2}$$

where $\mathbf{x}_i$ and $\mathbf{y}_i$ are, respectively, the coordinates of the $i$th residue in the query and target structures, and $N$ the number of aligned residues.

#### Statistical Significance: E-value

The E-value is computed identically as in BLAST; refer to the BLAST notes for detailed description of individual terms.

$$E=Kmn\cdot\exp(-\lambda S)$$

### Algorithmic Optimization

- Advanced data structures (k-d trees, suffix arrays, hash tables) are used to index pre-processed representations and rapidly look up seed regions.
- Libraries like OpenMP and CUDA exploit multi-threading on CPU/GPU acceleration.
- Heuristic pre-filters are employed to discard candidate alignments with low preliminary scores, known as the "filtering" stage.
- Algorithms for superposition and best-fit transformation (often solved via SVD) are optimized using well-tested numerical libraries.
- The Kabsch algorithm computes the optimal rotation matrix that minimizes the RMSD. For two sets of points $\mathbf{X}$ and $\mathbf{Y}$, the algorithm finds the rotation matrix $\mathbf{R}$ s.t.

$$\mathbf{R} = \underset{\mathbf{R}}{\argmin}\sum_{i} ||\mathbf{Y}_i - \mathbf{RX}_i||^2$$

## Pseudocode

Acquired from ChatGPT.

```
// Pseudo-code: Seed-and-Extend Alignment
for (auto seed : identify_seeds(query_representation, database_index)) {
    // Initialize dynamic programming matrix.
    DPMatrix dp = initialize_dp_matrix(seed);
    // Extend alignment in both directions.
    for (int i = seed.start_query, j = seed.start_target; i < query_length && j < target_length; i++, j++) {
        double geom_score = compute_geometric_score(query[i], target[j]);
        double dp_value = max({
            dp[i-1][j-1] + geom_score,
            dp[i-1][j] - gap_penalty,
            dp[i][j-1] - gap_penalty
        });
        dp[i][j] = dp_value;
    }
    double alignment_score = dp[query_length][target_length];
    // Compute E-value using pre-calibrated lambda and K.
    double e_value = compute_evalue(alignment_score, query_length, target_length, lambda, K);
    if (e_value < threshold)
        report_alignment(seed, alignment_score, e_value);
}
```