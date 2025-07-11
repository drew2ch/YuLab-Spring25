""" Script to process Foldseek (and PSI-BLAST, 6/18) results for 5,196+ unique protein monomers.
# --- Parsing of .tsv query results and extraction of relevant metrics.
# --- Performs template search for each pair in the data file and impute newly prescribed features.
# Andrew Chung, hc893; 6/16/2025
"""

import os
import re
import numpy as np
import pandas as pd
import argparse
import tqdm
import pickle

# define some global objects
DEFAULT_USER_BASE_DIR = "C:/Users/hychu/OneDrive/Desktop/Summer25"
DEFAULT_FOLDSEEK_PATH = os.path.join(DEFAULT_USER_BASE_DIR, "homolog-results/foldseek")
DEFAULT_INPUT_FILE = os.path.join(DEFAULT_USER_BASE_DIR, "github/ppi-classifiers/data/features.csv")
DEFAULT_CACHE_FILE = os.path.join(DEFAULT_USER_BASE_DIR, "github/ppi-classifiers/data/archive/homologs_tree.pkl")   
NUM_PROTEINS = 12343 # based on total count of tsv output files

# full convertalis format: "query, target, pident, fident, alnlen, mismatch, gapopen, qstart, qend, qlen, tstart, tend, tlen, evalue, bits, qaln, taln, qseq, tseq"
FOLDSEEK_FEATURES = np.array([
    "query", "target", "pident", "fident", "evalue", "bits"
])
# set homolog target format
pattern = re.compile("([a-z0-9]{4})-assembly\\d+\\.cif\\.gz_(.+)", re.IGNORECASE)

def parse_tsv_file(tsv_name, path) -> np.ndarray: 
    """ function to parse .tsv files that store Foldseek-identified hit (homolog) information for a given protein.
        - tsv_name: name of the .tsv file
        - path: home path for the .tsv file
        - ~params: list of params (by index) to consider from the results. DEFAULT = ['query', 'target', 'pident', 'fident']~
    """
    # verify that the file exists first and foremost
    FILE_PATH = os.path.join(path, tsv_name)
    if not os.path.exists(FILE_PATH):
        print(f"Error: cannot find file {tsv_name} in path {path}. Exiting.")
        return np.empty((0, 4))

    homologs_info = []
    try:
        with open(FILE_PATH, "r", newline = '', encoding = 'utf-8') as infile:
            lines = infile.read().splitlines()
            if not lines: # file is empty; no homologs detected by Foldseek
                # print(f"Empty .tsv file detected: {tsv_name} contains no homologs.")
                return np.empty((0, 4))
            
            n_homologs = len(lines) # for assertion
            for line in lines:
                # for each line, identify parameters of interest and append it to list of homologs
                items = line.strip().split('\t')
                query, target = items[0], items[1]
                try:
                    pident, fident = float(items[2]), float(items[3])
                    evalue, bits = float(items[13]), float(items[14])
                except ValueError:
                    continue
                homologs_info.append([query, target, pident, fident, evalue, bits])
            # convert from nested list structure to np.2darray
            homologs_info = np.array(homologs_info, dtype = object)
            assert len(homologs_info) == n_homologs, \
                f"Error: homolog counts do not match: {n_homologs} lines vs. {len(homologs_info)} parsed."
            # print(f"Homologs detected and parsed for file {tsv_name}.")
    except Exception as e:
        print(f"Error parsing {tsv_name}: {e}")
    
    return homologs_info

def templates_with_ident(
        homologs_a: np.ndarray, 
        homologs_b: np.ndarray, 
        rank_by: str = 'fident', 
        top_n: int = 1
    ) -> np.array:
    """ Identifies candidate template pairs between two proteins A and B and returns the top n results.
    Definition: Each protein contains a set of homologs (or lack thereof). Concerning a protein-protein interaction pair,
    template pairs are selected from the combination of all possible homolog interaction pairs by the following criteria:

    - identical pdb_id (e.g. '8ink-assembly1.cif.gz_z' -> '8ink') pdb_id(A) == pdb_id(B)
    - non-identical chain id (e.g. '8ink-assembly1.cif.gz_z' -> 'z') chain_id(A) != chain_id(B)
    - Template pairs are ranked by the 'rank_by' argument (either 'pident' or 'fident'); by default, 'fident' is used.
        - 'fident', 'pident', and 'bits' of homolog pairs are averaged to represent respective values of the template pair.
        - In the case of a tie between two 'fident' scores, the 'pident' is used as a tiebraker.
        - The 'pident' and 'fident' scores for a template pair are taken as the average of individual respective scores. 
        - e-values are tricky, since they aren't additive. Some ideas are (1) product or geometric/harmonic mean (2) Fisher's method (3) minimum
        - here, I will go with the minimum (Tippett's) method
    - RETURNS: a np.2darray of the format (*[pdb-id, pident, fident])
    """

    # either protein lacks homologs, so no template pairs can be found
    if (not homologs_a.size) or (not homologs_b.size):
        return np.array([])
    
    template_pairs = []
    for query_a, target_a, pident_a, fident_a, evalue_a, bits_a in homologs_a:
        if not pattern.fullmatch(target_a):
            continue
        pdb_a, chain_a = pattern.search(target_a).groups()

        for query_b, target_b, pident_b, fident_b, evalue_b, bits_b in homologs_b:
            if not pattern.fullmatch(target_b):
                continue
            pdb_b, chain_b = pattern.search(target_b).groups()

            if pdb_a == pdb_b and chain_a != chain_b:
                template_pident = np.mean([pident_a, pident_b])
                template_fident = np.mean([fident_a, fident_b])
                template_evalue = np.min([evalue_a, evalue_b])
                template_bits = np.mean([bits_a, bits_b])
                template_pairs.append([
                    # f"{pdb_a}-{chain_a}/{chain_b}", 
                    template_pident, template_fident, 
                    template_evalue, template_bits,
                    pdb_a, chain_a, chain_b
                ])

    if not template_pairs: # no template pairs found
        return np.empty((0, 5))
    
    # sort by specified parameter
    # may add evalue/bits as additional ranking criteria options in the future
    if rank_by == 'fident': # 'fident' as primary criteria, 'pident' as tiebraker
        template_pairs.sort(key = lambda row: (row[2], row[1]), reverse = True)
    elif rank_by == 'pident': # 'pident' as primary criteria, 'fident' as tiebraker
        template_pairs.sort(key = lambda row: (row[1], row[2]), reverse = True)
    else: # invalid rank method; returning hits without ranking
        print(f"Invalid rank parameter: {rank_by}. Returning templates without sorting.")
    
    return np.array(template_pairs[:top_n], dtype = object)

def complete_features(
        data: pd.DataFrame, 
        tree: dict,
        rank_by: str = 'fident', 
        top_n: int = 1
    ) -> pd.DataFrame:
    """ Takes a data file of distinct PPI pairs (also conventionally tagged with genomic features),
        as well as a homolog-tree summarizing the list of homologs for each query protein.
        It then accordingly imputes the following Foldseek-specific features:
    --- has_templates: whether a viable template pair exists between the dimer pair.
    --- pident: percent sequence identity score of the pair, calculated as the average of the pident scores of its constituent monomers.
    --- fident: structural identity score of the pair, calculated as the average of the fident scores of its constituent monomers.
        RETURNS: a pd.DataFrame object with the three aforementioned features appended at the end.
    """
    assert 'ppi' in data.columns, "Critical Error: no 'ppi' column detected."

    has_templates, pident, fident, evalue, bits, pdb_id, chain_a, chain_b = [], [], [], [], [], [], [], []
    for index, row in tqdm.tqdm(data.iterrows()):
        proteins = str(row['ppi']).strip().split(':')
        if len(proteins) < 2:
            print(f"Error: faulty PPI pair at index {index}: {row['ppi']}")
            continue
        homA =  tree.get(proteins[0], np.empty((0, 6)))
        homB =  tree.get(proteins[1], np.empty((0, 6)))
        candidate_template = templates_with_ident(
            homologs_a = homA,
            homologs_b = homB,
            rank_by = rank_by, top_n = top_n
        )
        if not candidate_template.size: # returned no template pairs
            has_templates.append(0)
            pident.append(0.0)
            fident.append(0.0)
            evalue.append(0.0)
            bits.append(0.0)
            pdb_id.append("NULL")
            chain_a.append("NULL")
            chain_b.append("NULL")
        else:
            has_templates.append(1)
            pident.append(candidate_template[0, 0])
            fident.append(candidate_template[0, 1])
            evalue.append(candidate_template[0, 2])
            bits.append(candidate_template[0, 3])
            pdb_id.append(candidate_template[0, 4])
            chain_a.append(candidate_template[0, 5])
            chain_b.append(candidate_template[0, 6])
    
    data_ = data.copy()
    data_['has_templates'] = has_templates
    data_['pident'] = pident
    data_['fident'] = fident
    data_['evalue'] = evalue
    data_['bits'] = bits
    data_['pdb_id'] = pdb_id
    data_['chain_a'] = chain_a
    data_['chain_b'] = chain_b

    return data_

def load_homologs_tree(
        foldseek_path: str, 
        cache_file: str = None
    ) -> dict:
    """ Loads the homologs tree from a specified path.
        If a cache file is provided, it will load the tree from the cache.
        Otherwise, it will parse all .tsv files in the path and build the tree.
    """
    # check if cache file exists and load it
    if cache_file and os.path.exists(cache_file):
        try:
            print(f"Loading homologs tree from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache file: {e}. Rebuilding homologs tree.")

    # if cache file does not exist or is invalid, parse .tsv files 
    # and newly build the homologs tree
    print(f"No valid cache file found. (Re)Building homologs tree from {foldseek_path}...")
    tsv_files = []
    try:
        for dirpath, dirnames, filenames in os.walk(foldseek_path):
            for filename in tqdm.tqdm(filenames):
                if filename.endswith('.tsv'):
                    tsv_files.append(os.path.join(dirpath, filename))
    except Exception as e:
        print(f"Error extracting .tsv file names: {e}")
    assert len(tsv_files) == NUM_PROTEINS, \
        f"Error: mismatch in .tsv file count. Expected {NUM_PROTEINS}, got {len(tsv_files)}."

    homologs_tree = {}
    proteins = [tsv.replace('.tsv', '').strip().split('\\')[-1] for tsv in tsv_files]
    for protein_id in tqdm.tqdm(proteins):
        homologs_tree[protein_id] = parse_tsv_file(f"{protein_id}.tsv", foldseek_path)
    assert len(homologs_tree) == NUM_PROTEINS, \
        f"Error: incomplete homolog compilation. Expected {NUM_PROTEINS}, got {len(homologs_tree)}."
    
    # dump homolog tree into a cached pickle file
    print(f"Caching homologs tree to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(homologs_tree, f)

    return homologs_tree

def main():

    parser = argparse.ArgumentParser(description = "Foldseek Result Parsing and Template Identification among Identified Homolog Pairs.")
    parser.add_argument("--foldseek_path", default = DEFAULT_FOLDSEEK_PATH, help = "Path for Foldseek result files (.tsv)")
    parser.add_argument("--input_file", default = DEFAULT_INPUT_FILE, help = "Input/Output data file name: .csv or .txt")
    parser.add_argument("--output_file", default = DEFAULT_INPUT_FILE, help = "Output data file name: .csv")
    parser.add_argument("--rank_by", default = "fident", help = "Foldseek output metric to rank template pairs")
    parser.add_argument("--top_n", default = 1, help = "Top n template pairs")
    # parser.add_argument("--encode_template_id", action = "store_true", default = False, help = "Process as .txt file instead of .csv")
    args = parser.parse_args()

    # === identify and parse through .tsv results ===
    print("=== Begin Workflow ===")
    print("Parsing .tsv query results and extracting homolog data...")

    homologs_tree = load_homologs_tree(
        foldseek_path = args.foldseek_path, 
        cache_file = DEFAULT_CACHE_FILE
    )
    
    print("Homolog information successfully transcribed for all proteins.")

    # === Read in features data file ===
    DATA_PATH = args.input_file.strip().split('/')[-1]
    print("==================================================")
    print(f"Reading in {DATA_PATH}...")

    try:
        data = pd.read_csv(
            args.input_file, index_col = False
        )
    except FileNotFoundError as e:
        print(f"Error: {DATA_PATH} not found: {e}. Exiting.")
        return
    if 'ppi' not in data.columns:
        print("Critical Error: column 'ppi' missing in data file. Exiting.")
        return

    print(f"{DATA_PATH} contains {len(data)} distinct PPI pairs.")
    print(data.head())
    input("Press Enter to continue...")

    # === Evaluate Template Pairs and impute new features ===
    print("==================================================")
    print("Evaluating template pairs for all PPIs...")

    data = complete_features(data, homologs_tree, rank_by = args.rank_by, top_n = args.top_n)
    print(data.value_counts(subset = ['has_templates']))
    print(f"Template Evaluation complete.")

    # === Save results ===
    try:
        if not args.output_file:
            print("No output file specified.")
            args.output_file = input("Please specify an output name: ")
        if not args.output_file.endswith('.csv'):
            print("Warning: output file should be a .csv file. Appending '.csv' to the output name.")
            args.output_file += '.csv'
        data.to_csv(args.output_file, index = False)
        print(f"Data saved to {args.output_file}.")
    except Exception as e:
        print(f"Error saving file: {e}")

    # Delete pickle file to prevent Github Commit-Push obstruction
    try:
        os.remove(DEFAULT_CACHE_FILE)
        print("Deleted cached file.")
    except OSError as e:
        print(f"Error deleting cached file: {e}")


    print("=== Job Finished. ===")

if __name__ == "__main__":
    main()
