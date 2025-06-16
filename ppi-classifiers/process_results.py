""" Script to process Foldseek results for 5,196 unique protein monomers.
# --- Parsing of .tsv query results and extraction of relevant metrics.
# --- Performs template search for each pair in the data file and impute newly prescribed features.
# Andrew Chung, hc893; 6/16/2025
"""

import os
import pandas as pd
import numpy as np
import argparse
import tqdm
import re

# define some global objects
DEFAULT_USER_BASE_DIR = "C:/Users/hychu/OneDrive/Desktop/Summer25"
DEFAULT_TSV_PATH = os.path.join(DEFAULT_USER_BASE_DIR, "output/tsv")
NUM_PROTEINS = 5196

# full format: "query,target,pident,fident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qaln,taln,qseq,tseq"
CONVERTALIS_FORMAT = np.array(
    "query", "target", "pident", "fident"
)
pattern = re.compile("([a-z0-9]{4})-assembly\d+\.cif\.gz_(.+)", re.IGNORECASE)

def parse_tsv_file(tsv_name, path) -> np.array:
    """ function to parse .tsv files that store hit (homolog) information for a given protein.
        - tsv_name: name of the .tsv file
        - path: home path for the .tsv file
        - params: list of params (by index) to consider from the results. DEFAULT = ['query', 'target', 'pident', 'fident']
    """
    # verify that the file exists first and foremost
    FILE_PATH = os.path.join(path, tsv_name)
    if not os.path.exists(FILE_PATH):
        print(f"Error: cannot find file {tsv_name} in path {path}. Exiting.")
        return

    homologs = np.array([])
    try:
        with open(FILE_PATH, "r", newline = '', encoding = 'utf-8') as infile:
            if not infile.read(): # file is empty; no homologs detected by Foldseek
                print(f"Empty .tsv file detected: {tsv_name} contains no homologs.")
                return np.array([])
            
            n_homologs = len(infile.readline()) # for assertion
            for index, line in enumerate(infile):
                # for each line, identify parameters of interest and append it to list of homologs
                items = line.strip().split('\t')[:4]
                assert len(items) == len(CONVERTALIS_FORMAT), \
                    print(f"Error: dimension mismatch: expected {len(CONVERTALIS_FORMAT)} params, got {len(items)}")
                homologs.append(items)
            # convert from nested list structure to np.2darray
            homologs = np.array(homologs)
            assert len(homologs) == n_homologs, \
                f"Error: homolog counts do not match: {n_homologs} lines vs. {len(homologs)} parsed."
            print(f"Homologs detected and parsed for file {tsv_name}.")
    except Exception as e:
        print(f"Error parsing {tsv_name}: {e}")
    
    return homologs

def templates_with_ident(
        homologs_a: np.array, 
        homologs_b: np.array, 
        rank_by = 'fident', 
        top_n = 1
    ) -> np.array:
    """ Identifies candidate template pairs between two proteins A and B and returns the top n results.
    Definition: Each protein contains a set of homologs (or lack thereof). Concerning a protein-protein interaction pair,
    template pairs are selected from the combination of all possible homolog interaction pairs by the following criteria:

    - identical pdb_id (e.g. '8ink-assembly1.cif.gz_z' -> '8ink') pdb_id(A) == pdb_id(B)
    - non-identical chain id (e.g. '8ink-assembly1.cif.gz_z' -> 'z') chain_id(A) != chain_id(B)
    - Template pairs are ranked by the 'rank_by' argument (either 'pident' or 'fident'); by default, 'fident' is used.
    - In the case of a tie between two 'fident' scores, the 'pident' is used as a tiebraker.
    - The 'pident' and 'fident' scores for a template pair are taken as the average of individual respective scores. 
    - RETURNS: a np.2darray of the format (*[pdb-id, pident, fident])
    """

    # either protein lacks homologs, so no template pairs can be found
    if not homologs_a or not homologs_b:
        return np.array([])
    
    template_pairs = []
    for i in range(len(homologs_a)):
        target_a = homologs_a[i][1]
        for j in range(len(homologs_b)):
            target_b = homologs_b[j][1]
            # ensure that both target (pdb_id + chain_id sequence) outputs are consistent
            if (not pattern.fullmatch(target_a)) or (not pattern.fullmatch(target_b)):
                print(f"Error: at least one of {target_a} and {target_b} do not conform to proper target format.")
                continue
            
            # template condition: pdb_id match, chain_id do NOT match
            match_a, match_b = pattern.search(target_a), pattern.search(target_b)
            if match_a.group(1) == match_b.group(1) and match_a.group(2) != match_b.group(2):
                template_pident, template_fident = np.mean([homologs_a[i][2], homologs_b[j][2]]), \
                    np.mean([homologs_a[i][3], homologs_b[j][3]])
                template_pairs.append([
                    f"{match_a.group(1)}-{match_a.group(2)}/{match_b.group(2)}", template_pident, template_fident
                ])
    
    # convert to np.2darray
    template_pairs = np.array(template_pairs)

    # sort by specified parameter
    if rank_by == 'fident': # 'fident' as primary criteria, 'pident' as tiebraker
        template_pairs = template_pairs[
            np.lexsort((-template_pairs[:, 2], -template_pairs[:, 3]))
        ]
    elif rank_by == 'pident': # 'pident' as primary criteria, 'fident' as tiebraker
        template_pairs = template_pairs[
            np.lexsort((-template_pairs[:, 3], -template_pairs[:, 2]))
        ]
    else: # invalid rank method; returning hits without ranking
        print(f"Invalid rank parameter: {rank_by}. Returning templates without sorting.")
    
    top_templates = template_pairs[:top_n]
    assert top_templates.shape == (top_n, 3), \
        print(f"Error: top_templates dimension is invalid.")
    
    return top_templates

def main():

    parser = argparse.ArgumentParser(description = "Foldseek Result Parsing and Template Identification among Identified Homolog Pairs.")
    parser.add_argument("--tsv_path", default = DEFAULT_TSV_PATH, help = "Path for output .tsv files")
    parser.add_argument("--output_file", default = "features_fs.csv", help = "Output data file name")
    parser.add_argument("--rank_by", default = "fident", help = "Foldseek output metric to rank template pairs")
    parser.add_argument("--top_n", default = 1, help = "Top n template pairs")
    args = parser.parse_args()

    # === identify and parse through .tsv results ===
    print("=== Begin Workflow ===")
    print("Parsing .tsv query results...")

    # === tally the filenames of all .tsv files ===
    # the total count should be 5,196
    tsv_files = np.array([])
    try:
        for dirpath, dirnames, filenames in os.walk(args.tsv_path):
            for filename in tqdm.tqdm(filenames):
                if filename.endswith('.tsv'):
                    tsv_files = np.append(tsv_files, [os.path.join(dirpath, filename)])
    except Exception as e:
        print(f"Error extracting .tsv file names: {e}")
    assert len(tsv_files) == NUM_PROTEINS, \
        f"Error: mismatch in .tsv file count. Expected {NUM_PROTEINS}, got {len(tsv_files)}."
    
    print("All .tsv files were successfully identified.")

    # === compile homolog parameters ===
    print("==================================================")
    print("Extracting homolog information for all proteins...")

    homologs_tree = {}
    proteins = [tsv.replace('.tsv', '') for tsv in tsv_files]
    for protein_id in tqdm.tqdm(proteins):
        homologs_tree[protein_id] = parse_tsv_file(f"{protein_id}.tsv", args.tsv_path)
    assert len(homologs_tree) == NUM_PROTEINS, \
        f"Error: incomplete homolog compilation. Expected {NUM_PROTEINS}, got {len(homologs_tree)}."
    
    print("Homolog information successfully transcribed for all proteins.")

    # === Read in features data file ===
    print("==================================================")
    print("Reading in features.csv...")

    try: # omit index column
        data = pd.read_csv("features.csv").iloc[:, 1:].assign(
            has_templates = 0, 
            pident = 0, 
            fident = 0
        )
    except FileNotFoundError as e:
        print(f"Error: features.csv not found: {e}. Exiting.")
        return
    if 'ppi' not in data.columns:
        print("Critical Error: column 'ppi' missing in data file. Exiting.")
        return
    print(f"features.csv contains {len(data)} distinct PPIs.")

    # === Evaluate Template Pairs and impute new features ===
    print("==================================================")
    print("Evaluating template pairs for all PPIs...")

    for index, row in tqdm.tqdm(data.iterrows()):
        proteins = row['ppi'].strip().split(':')
        if len(proteins) < 2:
            print(f"Error: faulty PPI pair at index {index}: {row['ppi']}")
            continue
        if proteins[0] not in proteins or proteins[1] not in proteins:
            print(f"Error: one or more proteins not found. Check your output path.")
            continue

        # find top 1 template pair
        candidate_template = templates_with_ident(
            homologs_a = homologs_tree[proteins[0]],
            homologs_b = homologs_tree[proteins[1]],
            rank_by = args.rank_by, top_n = args.top_n
        )
        if not candidate_template: # returned no template pairs
            continue
        row['has_templates'] = 1
        row[['pident', 'fident']] = candidate_template[0][1:]
        
    print(f"Template Evaluation complete. {data['has_templates'].value_counts()[1]} pairs were identified with a viable template pair.")

    # === Save results ===
    try:
        data.to_csv(args.output_file)
        print(f"Data saved as {args.output_file}.")
    except Exception as e:
        print(f"Error saving file: {e}")

    print("=== Job Finished. ===")

if __name__ == "__main__":
    main()
