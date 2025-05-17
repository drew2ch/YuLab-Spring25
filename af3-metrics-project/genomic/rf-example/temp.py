import pandas as pd
from itertools import combinations
import random
import numpy as np
from goatools.obo_parser import GODag
from goatools.semantic import semantic_similarity
import pickle

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_transcript(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df

def load_annotations(file_path):
    columns = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]

    annotation_data = pd.read_csv(
        file_path,
        sep="\t",
        comment="#",  # Skip comment lines starting with #
        names=columns,
        header=None,  # GFF3 has no header row
        low_memory=False
    )
    return annotation_data

def load_go_ontology(file_path):
    go_dag = GODag(file_path)
    print(f"Loaded GO ontology with {len(go_dag)} terms!")
    return go_dag

def transcript_to_gene_map(gff3_df):
    # Attributes column: ID=ENST00000456328.2;Parent=ENSG00000223972.5;
    gff3_df['ID'] = gff3_df['attributes'].str.extract(r'ID=([^;]+)')
    gff3_df['Parent'] = gff3_df['attributes'].str.extract(r'Parent=([^;]+)')

    # Filter rows for transcripts and create the map (e.g. {'ENST00000456328.2': 'ENSG00000223972.5'})
    transcript_to_gene_map = (
        gff3_df[gff3_df['type'] == 'transcript'][['ID', 'Parent']]
        .set_index('ID')['Parent']
        .to_dict()
    )

    # Output the mapping
    return transcript_to_gene_map

def parse_gtf(gtf_file):
    """
    Parse a GTF file to extract mappings for transcript IDs to gene symbols.
    """

    gene_to_symbol = {}

    with open(gtf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines

            columns = line.strip().split('\t')

            if columns[2] == 'gene':  # Only parse gene features
                attributes = columns[8]

                # Extract relevant attributes
                gene_id = extract_attribute(attributes, 'gene_id')
                gene_name = extract_attribute(attributes, 'gene_name')

                # Populate the mappings
                if gene_id and gene_name:
                    gene_to_symbol[gene_id] = gene_name

    return gene_to_symbol

def extract_attribute(attributes, key):
    """
    Extract the value of a key from the GTF attributes column.
    """
    for attr in attributes.split(';'):
        if attr.strip().startswith(key):
            return attr.split('"')[1]  # Extract the value between quotes
    return None

def parse_go_annotations(gaf_file, ontology_file):
    """
    Parses a GAF file to extract GO annotations and organizes them by BP, CC, and MF.
    
    Args:
        gaf_file (str): Path to the GAF annotation file.
        ontology_file (str): Path to the GO ontology file.

    Returns:
        dict: Mapping {Gene Symbol → {'BP': [...], 'CC': [...], 'MF': [...]}}
        GODag: Parsed GO ontology object.
    """
    go_dag = GODag(ontology_file)  # Load GO Ontology
    go_annotations = {}

    with open(gaf_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith("!"):  # Skip header lines
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue  # Skip malformed lines

            gene = parts[2]  # Gene Symbol
            go_term = parts[4]  # GO Term ID
            namespace = parts[8]  # BP, CC, or MF (not parts[8]!)

            if gene not in go_annotations:
                go_annotations[gene] = {"BP": [], "CC": [], "MF": []}

            if namespace == "P":  # Biological Process (BP) → Co-functionality
                go_annotations[gene]["BP"].append(go_term)
            elif namespace == "C":  # Cellular Component (CC) → Co-localization
                go_annotations[gene]["CC"].append(go_term)
            elif namespace == "F":  # Molecular Function (MF) → Negative control
                go_annotations[gene]["MF"].append(go_term)

    print(f"✅ Parsed {len(go_annotations)} genes with GO annotations.")
    return go_annotations, go_dag


def generate_coexpression_matrix(gene_expression, valid_genes):
    """
    Generate a co-expression matrix for the given valid genes.

    Parameters:
        gene_expression (DataFrame): Gene expression data indexed by gene symbols.
        valid_genes (list): List of valid gene symbols.

    Returns:
        DataFrame: Co-expression matrix (gene-by-gene correlation matrix).
    """
    # Filter expression data to include only valid genes
    filtered_expression = gene_expression.loc[valid_genes]

    # Compute the pairwise Pearson correlation matrix.
    coexpression_mat = filtered_expression.T.corr(method='pearson')
    
    # Select a random 5x5 submatrix
    random_genes = random.sample(valid_genes, 5)  # Select 5 random genes
    submatrix = coexpression_mat.loc[random_genes, random_genes]

    # Create and save visualization unconditionally
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        submatrix,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmin=-1.0,
        vmax=1.0,
        square=True,
        annot=True  # Add numerical values in heatmap
    )
    plt.title('Random 5x5 Gene Co-expression Matrix')
    plt.savefig("/home/jc3668/ppi-evaluation/analysis/genetic_analysis/random_5x5_correlation_matrix.png", bbox_inches='tight')

    return coexpression_mat

def filter_valid_genes(gene_expression, go_annotations):
    """
    Filters and returns a list of valid genes that exist in both gene_expression
    and go_annotations.

    Parameters:
        gene_expression (DataFrame): Gene expression data indexed by gene symbols.
        go_annotations (dict): A dictionary of GO annotations with gene symbols as keys.

    Returns:
        list: Valid gene symbols.
    """
    # Intersect gene symbols from gene_expression and go_annotations
    valid_genes = set(gene_expression.index).intersection(go_annotations.keys())
    print(f"Filtered valid genes count: {len(valid_genes)}")
    return list(valid_genes)

def sample_pairs_with_labels(labeled_pairs_df, total_samples, ratio):
    """
    Sample gene pairs with a controlled ratio of positive-to-negative pairs.

    Parameters:
        labeled_pairs_df (DataFrame): DataFrame containing all gene pairs and their labels.
        total_samples (int): Total number of gene pairs to sample.
        ratio (float): Desired ratio of positive-to-total pairs (e.g., 1/10).

    Returns:
        DataFrame: Sampled gene pairs DataFrame.
    """

    # Separate positive and negative pairs
    positive_pairs = labeled_pairs_df[labeled_pairs_df['label'] == 1]
    negative_pairs = labeled_pairs_df[labeled_pairs_df['label'] == 0]
    
    print(f"Available positive pairs: {len(positive_pairs)}")
    print(f"Available negative pairs: {len(negative_pairs)}")

    num_positives = int(total_samples * ratio)
    num_negatives = total_samples - num_positives

    # Sample positive and negative pairs
    sampled_positive = positive_pairs.sample(n=min(num_positives, len(positive_pairs)), random_state=42)
    sampled_negative = negative_pairs.sample(n=min(num_negatives, len(negative_pairs)), random_state=42)

    # Combine and shuffle
    sampled_pairs = pd.concat([sampled_positive, sampled_negative]).sample(frac=1, random_state=42).reset_index(drop=True)

    return sampled_pairs



def calculate_functional_similarity(go_annotations, go_dag, gene_pairs):
    """
    Calculate similarity metrics for BP, CC, and MF for each gene pair.

    If only one GO term exists, calculates semantic similarity directly.
    Otherwise, computes separate scores for BP, CC, and MF.
    """

    similarities = {}

    for gene1, gene2 in gene_pairs:
        terms1 = go_annotations.get(gene1, {})
        terms2 = go_annotations.get(gene2, {})

        bp_similarities = [
            semantic_similarity(term1, term2, go_dag)
            for term1 in terms1.get("BP", [])
            for term2 in terms2.get("BP", [])
            if semantic_similarity(term1, term2, go_dag) is not None
        ]

        cc_similarities = [
            semantic_similarity(term1, term2, go_dag)
            for term1 in terms1.get("CC", [])
            for term2 in terms2.get("CC", [])
            if semantic_similarity(term1, term2, go_dag) is not None
        ]

        mf_similarities = [
            semantic_similarity(term1, term2, go_dag)
            for term1 in terms1.get("MF", [])
            for term2 in terms2.get("MF", [])
            if semantic_similarity(term1, term2, go_dag) is not None
        ]

        similarities[(gene1, gene2)] = {
            "BP": max(bp_similarities) if bp_similarities else None,
            "CC": max(cc_similarities) if cc_similarities else None,
            "MF": max(mf_similarities) if mf_similarities else None
        }

    return similarities
            
def create_feature_matrix(coexpression_matrix, functional_similarities, valid_sampled_pairs, labeled_pairs_df):
    """
    Create a feature matrix by matching valid sampled pairs with co-expression
    and co-functionality values, separating BP, CC, and MF.

    Parameters:
        coexpression_matrix (DataFrame): Gene-by-gene co-expression matrix.
        functional_similarities (dict): Dictionary of similarity metrics (BP, CC, MF) for gene pairs.
        valid_sampled_pairs (list): List of valid sampled pairs (tuples of gene symbols).
        labeled_pairs_df (DataFrame): DataFrame containing all gene pairs with their labels.

    Returns:
        DataFrame: Feature matrix containing gene1, gene2, co-expression, BP, CC, MF, and label.
    """
    
    valid_sampled_set = set(valid_sampled_pairs)  # Convert to set for faster lookup
    filtered_pairs = labeled_pairs_df[
        labeled_pairs_df.apply(
            lambda row: (row['gene1'], row['gene2']) in valid_sampled_set or
                        (row['gene2'], row['gene1']) in valid_sampled_set,
            axis=1
        )
    ]

    # Prepare the feature matrix
    feature_data = []
    for _, row in filtered_pairs.iterrows():
        gene1, gene2, label = row['gene1'], row['gene2'], row['label']

        # Fetch co-expression
        coexp = (
            coexpression_matrix.loc[gene1, gene2]
            if gene1 in coexpression_matrix.index and gene2 in coexpression_matrix.columns
            else None
        )

        # Fetch BP, CC, MF similarities
        similarities = functional_similarities.get((gene1, gene2), {})
        bp_sim = similarities.get("BP")
        cc_sim = similarities.get("CC")
        mf_sim = similarities.get("MF")

        # Only include pairs with valid metrics
        if coexp is not None or any([bp_sim, cc_sim, mf_sim]):
            feature_data.append({
                "gene1": gene1,
                "gene2": gene2,
                "co-expression": coexp,
                "BP": bp_sim,
                "CC": cc_sim,
                "MF": mf_sim,
                "label": label
            })

    return pd.DataFrame(feature_data)



def evaluate_and_visualize(df):
    """
    Train a Random Forest Classifier, evaluate metrics, and create visualizations for different ratios.
    """
    print(f"Number of positives: {len(df[df['label'] == 1])}")
    print(f"Number of negatives: {len(df[df['label'] == 0])}")

    ratios = [1, 1/10, 1/100, 1/1000]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)  # Use constrained layout to handle overlaps
    axes = axes.flatten()
    
    for i, ratio in enumerate(ratios):
        df_temp = df.copy()
        
        positives = df_temp[df_temp['label'] == 1]
        negatives = df_temp[df_temp['label'] == 0]
        
        n_positives = 100
        n_negatives = int(n_positives * (1000 * ratio))
        
        if ratio < 1:
            negatives_sampled = negatives.sample(n=n_negatives, random_state=42)
            df_sampled = pd.concat([positives, negatives_sampled])
        else:
            df_sampled = df_temp
        
        # Train/test split
        X = df_sampled[['co-expression', 'BP', 'CC']].fillna(0.0)
        y = df_sampled['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train Random Forest Model
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        # Compute Metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = auc(recall, precision)

        # Check uniqueness of protein pairs in training and test sets
        unique_train = set(X_train.index)
        unique_test = set(X_test.index)
        overlap = unique_train.intersection(unique_test)
        print(f"Overlap between training and test sets: {len(overlap)}")
        
        
        # Plot ROC Curve
        axes[i].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        axes[i].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        axes[i].set_title(f"Interacting ({n_positives}) vs non-interacting ({n_negatives})", fontsize=12)
        axes[i].set_xlabel("False Positive Rate", fontsize=10)
        axes[i].set_ylabel("True Positive Rate", fontsize=10)
        axes[i].legend(fontsize=8, loc='lower right')
        
        # Plot PR Curve
        axes[i + 4].plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
        axes[i + 4].set_title(f"Interacting ({n_positives}) vs non-interacting ({n_negatives})", fontsize=12)
        axes[i + 4].set_xlabel("Recall", fontsize=10)
        axes[i + 4].set_ylabel("Precision", fontsize=10)
        axes[i + 4].legend(fontsize=8, loc='lower left')
        
    plt.savefig("roc_pr_curves.png", bbox_inches='tight')
    plt.close()

"""
def evaluate_and_visualize(df):
    print(f"Number of positives: {df[df['label'] == 1]}")
    print(f"Number of negatives: {df[df['label'] == 0]}")


    ratios = [1, 1/10, 1/100, 1/1000]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns
    axes = axes.flatten()

    for i, ratio in enumerate(ratios):
        
        df_temp = df.copy()

        positives = df_temp[df_temp['label'] == 1]
        negatives = df_temp[df_temp['label'] == 0]

        n_positives = len(positives)
        n_negatives = int(n_positives * (1000 * ratio))
        
        if ratio < 1:
            negatives_sampled = negatives.sample(n=n_negatives, random_state=42)
            df_sampled = pd.concat([positives, negatives_sampled])
        else:
            df_sampled = df_temp

        df_sampled['pair'] = df_sampled[['gene1', 'gene2']].apply(frozenset, axis=1)
        unique_pairs = df_sampled['pair'].unique()
        train_pairs, test_pairs = train_test_split(unique_pairs, test_size=0.2, random_state=42)

        # Map back to DataFrame
        train_df = df_sampled[df_sampled['pair'].isin(train_pairs)]
        
        test_df = df_sampled[df_sampled['pair'].isin(test_pairs)]

        # Train/test split
        X_train = train_df[['co-expression', 'BP', 'CC']].fillna(0.0)
        y_train = train_df['label']
        X_test = test_df[['co-expression', 'BP', 'CC']].fillna(0.0)
        y_test = test_df['label']

        # Train full model (BP (cofunction) + CC (colocalization) + co-expression)
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_prob = rf.predict_proba(X_test)[:, 1]

            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        pos_len = df_sampled[df_sampled['label']==1]
        neg_len = df_sampled[df_sampled['label']==0]

        # Plot ROC Curve
        axes[i].plot(fpr, tpr, label=f"Random Forest 1:{int(1000 * ratio)} (AUC = {roc_auc:.2f})")
        axes[i].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
        axes[i].set_title(f"Interacting ({pos_len}) vs non-interacting ({neg_len})", fontsize=14)
        axes[i].set_xlabel("False Positive Rate", fontsize=12)
        axes[i].set_ylabel("True Positive Rate", fontsize=12)
        axes[i].legend(fontsize=10)

        # Plot PR Curve
        axes[i + 4].plot(recall, precision, label=f"Random Forest 1:{int(1000 * ratio)} (AUC = {pr_auc:.2f})")
        axes[i + 4].set_title(f"Interacting ({pos_len}) vs non-interacting ({neg_len})", fontsize=14)
        axes[i + 4].set_xlabel("Recall", fontsize=12)
        axes[i + 4].set_ylabel("Precision", fontsize=12)
        axes[i + 4].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("roc_pr_curves.png")
"""


# From Yilin's code
def assign_ppi_label(df_ppi_raw, nonstr_pos, nonstr_ex, str_pos, ppi_in_pdb):
    df_ppi = df_ppi_raw.copy()
    df_ppi['ppi'] = df_ppi_raw['ppi'].apply(lambda x: ':'.join(sorted(x.split(':'))))  # make sure proteins are sorted
    # non-structural label
    df_ppi['non_struct_label'] = -1
    df_ppi.loc[~df_ppi['ppi'].isin(nonstr_ex), 'non_struct_label'] = 0
    df_ppi.loc[df_ppi['ppi'].isin(nonstr_pos), 'non_struct_label'] = 1

    # structural label
    df_ppi['str_label'] = -1  # interactions not found in the same complex structure / no structural evidence
    df_ppi.loc[df_ppi['ppi'].isin(ppi_in_pdb), 'str_label'] = 0  # found in same complex
    df_ppi.loc[df_ppi['ppi'].isin(str_pos), 'str_label'] = 1  # structural evidence

    label_dict = {-1: 'unclear (exclude for analysis)', 0: 'non-interacting pairs', 1: 'HINT-binary-HQ-LC'}
    df_ppi['nonstr_label_name'] = df_ppi['non_struct_label'].apply(lambda x: label_dict.get(x, x))

    label_dict = {-1: 'not found in same PDB complex', 0: 'no physical interaction', 1: 'direct physical interaction'}
    df_ppi['str_label_name'] = df_ppi['str_label'].apply(lambda x: label_dict[x])

    return df_ppi


def perform_eda(transcript_df, gene_expression):
    """
    Perform exploratory data analysis (EDA) on transcript and gene expression data.
    """
    sns.set(style="white")  # Set seaborn style without grid

    # ========================================================================
    # 1. Conditions per Transcript (Filtered) - Converted to Percentage
    # ========================================================================
    print("\n[1/3] Generating conditions per transcript distribution...")
    num_gene_symbols = transcript_df.shape[0]
    
    non_zero_counts = transcript_df.drop(columns=['transcript', 'gene_id', 'normalized_gene', 'gene_symbol']).astype(bool).sum(axis=1)
    total_transcripts = len(non_zero_counts)
    
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(non_zero_counts, bins=50, color='#2ecc71', edgecolor='#27ae60', alpha=0.8)
    
    plt.title(f"Number of Samples with Non-Zero TPM per Transcript\n(Total Genes: {num_gene_symbols})", fontsize=16)
    plt.xlabel("Conditions", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.savefig("conditions_per_transcript_filtered.png", bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 2. Pearson Correlation Distribution (Log-Transformed) - Percentage Based
    # ========================================================================
    print("\n[2/3] Generating Pearson correlation distribution...")
    log_expression = np.log1p(gene_expression)  # Log transform (log1p to avoid log(0))
    corr_matrix = log_expression.T.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_values = corr_matrix.mask(mask).stack().dropna()
    
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(corr_values, bins=100, color='#3498db', edgecolor='#2980b9', alpha=0.8, stat='percent', binrange=(-1, 1))
    
    plt.title(f"Gene Expression Correlation Distribution (Log-Transformed TPM)\n(Total Genes: {num_gene_symbols})", fontsize=16)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.savefig("pearson_correlation_distribution_log.png", bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 3. Random Gene Expression Patterns - Clean Visualization
    # ========================================================================
    print("\n[3/3] Plotting random gene expression patterns...")
    np.random.seed(42)
    random_genes = gene_expression.sample(n=5).index.tolist()
    sample_indices = np.arange(1, gene_expression.shape[1] + 1)
    
    plt.figure(figsize=(14, 8))
    colors = ['#e74c3c', '#9b59b6', '#f1c40f', '#2ecc71', '#e67e22']
    
    for i, gene in enumerate(random_genes):
        plt.plot(sample_indices, gene_expression.loc[gene], marker='o', markersize=4, linestyle='-',
                 linewidth=1.5, alpha=0.8, color=colors[i % len(colors)], label=gene)
    
    plt.title(f"Expression Patterns for Randomly Selected Genes\n(Total Genes: {num_gene_symbols})", fontsize=16)
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("Log Normalized TPM", fontsize=12)
    plt.yscale('log')
    
    max_samples = gene_expression.shape[1]
    plt.xticks(np.arange(0, max_samples + 1, 10))
    plt.xlim(0, max_samples + 1)
    
    plt.legend(title="Gene Symbols", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig("random_gene_expression_clean.png", bbox_inches='tight')
    plt.close()

    print("\n✅ EDA visualizations saved:")
    print("- conditions_per_transcript_filtered.png")
    print("- pearson_correlation_distribution_log.png")
    print("- random_gene_expression_clean.png")



def map_genes_to_uniprot(go_annotations_file):
    """
    Parse the GO annotations file to extract a mapping of gene symbols to UniProt IDs.
    Args:
        go_annotations_file (str): Path to the GO annotations file (.gaf format).
    Returns:
        dict: Mapping of Gene Symbols to UniProt IDs.
    """
    gene_to_uniprot = {}

    with open(go_annotations_file, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue  # Skip comment lines
            parts = line.strip().split('\t')

            uniprot_id = parts[1]  # UniProtKB ID (e.g., O15530)
            gene_symbol = parts[2]  # Gene Symbol (e.g., PDPK1)

            gene_to_uniprot[gene_symbol] = uniprot_id

    print(f"Extracted {len(gene_to_uniprot)} gene-to-UniProt mappings.")
    return gene_to_uniprot

def generate_labeled_pairs(valid_genes, gene_to_uniprot):
    """
    Generate all possible gene pairs with their UniProt mappings.
    """
    # Filter genes with UniProt mappings
    genes_with_uniprot = [gene for gene in valid_genes if gene in gene_to_uniprot]
    print(f"Genes with UniProt mappings: {len(genes_with_uniprot)}")

    # Generate all combinations of valid genes
    all_pairs = list(combinations(genes_with_uniprot, 2))

    # Create a DataFrame of pairs with UniProt mappings
    labeled_pairs = [
        {
            "gene1": pair[0],
            "gene2": pair[1],
            "uniprot_gene1": gene_to_uniprot[pair[0]],
            "uniprot_gene2": gene_to_uniprot[pair[1]],
        }
        for pair in all_pairs
    ]
    labeled_pairs_df = pd.DataFrame(labeled_pairs)
    print(f"Generated {len(labeled_pairs_df)} gene pairs.")
    return labeled_pairs_df

def validate_sampled_pairs(sampled_pairs, coexpression_matrix, go_annotations):
    """
    Ensure pairs have valid co-expression values and GO annotations.
    """
    valid_pairs = []
    for gene1, gene2 in sampled_pairs:
        # Check if genes exist in coexpression matrix and have GO terms
        if (gene1 in coexpression_matrix.index and 
            gene2 in coexpression_matrix.columns and 
            gene1 in go_annotations and 
            gene2 in go_annotations):
            valid_pairs.append((gene1, gene2))
    return valid_pairs

def add_ppi_and_assign_labels(labeled_pairs_df, ppi_label_ref_dict):
    # Create 'ppi' column
    labeled_pairs_df['ppi'] = labeled_pairs_df.apply(
        lambda row: ':'.join(sorted([row['uniprot_gene1'], row['uniprot_gene2']])), axis=1
    )
    print("Created 'ppi' column.")

    # Assign labels using PPI reference
    labeled_pairs_df = assign_ppi_label(
        labeled_pairs_df,
        nonstr_pos=ppi_label_ref_dict['nonstr_pos'],
        nonstr_ex=ppi_label_ref_dict['nonstr_exclusion'],
        str_pos=ppi_label_ref_dict['true_ppi_in_pdb'],
        ppi_in_pdb=ppi_label_ref_dict['all_pair_in_pdb']
    )
    print("Labels assigned to pairs.")
    print(f"Label distribution: {labeled_pairs_df['non_struct_label'].value_counts().to_dict()}")
    return labeled_pairs_df

def main():
    

    # Step 1: Load transcript data
    print("Step 1: Loading transcript data...")
    transcript_df = load_transcript('/home/jc3668/ppi-evaluation/data/quantification_gencode.tpm.txt')
    print(f"# Transcript_df rows: {len(transcript_df)}")
    print(transcript_df.head())

    # Step 2: Load annotations
    print("\n Step 2: Loading annotations..")
    gff3_df = load_annotations('/home/jc3668/ppi-evaluation/data/gencode.v47.chr_patch_hapl_scaff.annotation.gff3')
    print(gff3_df.head())
    gtf_file = '/home/jc3668/ppi-evaluation/data/Homo_sapiens.GRCh38.113.gtf'

    # Step 3: Create transcript-to-gene mapping
    print("\nStep 3: Creating transcript-to-gene mapping...")
    transcript_to_gene = transcript_to_gene_map(gff3_df)
    print("Transcript-to-gene mappings (sample):", list(transcript_to_gene.items())[:10])

    # Step 4: Create gene-to-symbol mapping
    print("\nStep 4: Creating gene-to-symbol mapping...")
    gene_to_symbol = parse_gtf(gtf_file)
    print("Gene-to-symbol mappings (sample):", list(gene_to_symbol.items())[:10])

    # Step 5: Map transcripts to gene IDs and symbols
    print(f"\nStep 5: Mapping transcripts to gene IDs and symbols..")
    transcript_df['gene_id'] = transcript_df['transcript'].map(transcript_to_gene)
    transcript_df = transcript_df.dropna(subset=['gene_id'])
    transcript_df['normalized_gene'] = transcript_df['gene_id'].str.split('.').str[0]
    transcript_df['gene_symbol'] = transcript_df['normalized_gene'].map(gene_to_symbol)
    transcript_df = transcript_df.dropna(subset=['gene_symbol'])
    print(f"Filtered transcript_df: {len(transcript_df)} rows")


    # Step 6: Aggregate expression to gene level
    print(f"\nStep 6: Aggregating expression to gene level...")
    gene_expression = transcript_df.groupby('gene_symbol').sum(numeric_only=True)
    print(f"Number of genes after aggregation: {len(gene_expression)} rows")
    print(gene_expression.head())

    
    
    # Perform EDA
    print("Performing EDA...")
    perform_eda(transcript_df, gene_expression)    

    # Step 7: Load GO annotations
    print(f"\nStep 7: Loading GO annotations...")
    ontology_file = '/home/jc3668/ppi-evaluation/data/go-basic.obo'
    gaf_file = '/home/jc3668/ppi-evaluation/data/goa_human.gaf'
    go_annotations, go_dag = parse_go_annotations(gaf_file, ontology_file)
   
    # Debug go_annotations
    for gene, annotations in list(go_annotations.items())[:5]:
        print(f"Gene: {gene}")
        print(f"  BP: {annotations.get('BP', [])}")  # Biological Process
        print(f"  CC: {annotations.get('CC', [])}")  # Cellular Component
        print(f"  MF: {annotations.get('MF', [])}")  # Molecular Function
        print("-" * 50)


    # Print the first 10 terms in go_dag
    for i, (go_id, go_term) in enumerate(go_dag.items()):
        print(f"GO ID: {go_id}")
        print(f"Name: {go_term.name}")
        print(f"Namespace: {go_term.namespace}")
        print(f"Is_a Relationships: {[parent.id for parent in go_term.parents]}")  # Get parent GO terms
        print(f"Children: {[child.id for child in go_term.children]}")  # Get child GO terms
        print("-" * 50)
        if i == 9:  # Stop after printing 10 terms
            break

    print("\n Step 8: Mapping genes to UniProt IDs...")
    gene_to_uniprot = map_genes_to_uniprot(gaf_file)
    print(f"Extracted {len(gene_to_uniprot)} gene-to-UniProt mappings")

    print("\nStep 9: Remove gene symbols that do not intersect with those from go_annotation...")
    valid_genes = filter_valid_genes(gene_expression, go_annotations)
    print(f"Number of valid genes: {len(valid_genes)}")
    print(f"Sample valid genes: {valid_genes[:5]}")

    print(f"\nStep 10: Generate co-expression matrix..")
    coexpression_matrix = generate_coexpression_matrix(gene_expression, valid_genes)
    print(f"Generated coexpression matrix with shape: {coexpression_matrix.shape}\n")
    print(coexpression_matrix.head())

    print("\nStep 11: Loading PPI label reference...")
     # Load PPI label reference
    with open('/home/yl986/data/protein_interaction/parsed/ppi_label_reference.pkl', 'rb') as f:
        ppi_label_ref_dict = pickle.load(f)
    print("PPI label reference loaded.")

    
    # Redefine labels to valid gene_pairs(e.g. labeld_pairs_df) - gene A, gene B, but use their UniProt IDs when extracting PPI labels. 
    labeled_pairs_df = generate_labeled_pairs(valid_genes, gene_to_uniprot)
    labeled_pairs_df = add_ppi_and_assign_labels(labeled_pairs_df, ppi_label_ref_dict)

    print(f"Labeled pairs: {labeled_pairs_df['non_struct_label'].value_counts().to_dict()}") 
    print(f"Number of labeled pairs: {len(labeled_pairs_df)}")
    print(labeled_pairs_df.head())

    labeled_pairs_df['label'] = labeled_pairs_df['non_struct_label']
    
    sample_size = 110000 # Total size for 1:1000 (100 pos vs 100000 neg) ratio
    print(f"Sample size: {sample_size}")

    sampled_pairs_df = sample_pairs_with_labels(labeled_pairs_df, total_samples=sample_size, ratio=1/1000)
    print(f"Sampled pairs: {len(sampled_pairs_df)}")

    # Validate sampled pairs
    valid_sampled_pairs = validate_sampled_pairs(
        sampled_pairs_df[['gene1', 'gene2']].values.tolist(),
        coexpression_matrix,
        go_annotations
    )

    print(f"Valid sampled pairs: {len(valid_sampled_pairs)}\n")
    print('------------------------------------------------------------------')
    # Calculate functional similarity
    functional_similarities = calculate_functional_similarity(go_annotations, go_dag, valid_sampled_pairs)

    # Create feature matrix
    df = create_feature_matrix(coexpression_matrix, functional_similarities, valid_sampled_pairs, labeled_pairs_df)
    df.to_csv('/home/jc3668/ppi-evaluation/analysis/genetic_analysis/data.csv', index=False)


    df = pd.read_csv('/home/jc3668/ppi-evaluation/analysis/genetic_analysis/data.csv')
    print(f"Evaluating!!!!")
    

    # Evaluate and visualize results
    evaluate_and_visualize(df)

if __name__ == "__main__":
    main()
