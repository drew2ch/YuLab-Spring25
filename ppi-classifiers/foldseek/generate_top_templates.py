""" Script to generate SIZ/COV features as defined by the PrePPI interaction classifier model.
    Incorporates Juheon's code plus custom machine-level modifications by Andrew.
"""

import tempfile
import urllib.request
import sys

from Bio.Blast import NCBIXML
from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
import re
import os
import numpy as np
import pandas as pd
import gzip
import tarfile
import json
import tempfile
import shutil
import argparse
import logging
import urllib.request

import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# --- Configuration ---
logging.basicConfig(
    level = logging.INFO, 
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
warnings.simplefilter('ignore', PDBConstructionWarning)

DEFAULT_USER_BASE_DIR = "C:/Users/hychu/OneDrive/Desktop/Summer25"

IRES_FILE = os.path.join(DEFAULT_USER_BASE_DIR, "pdb/ires_perpdb_alltax.txt")
FASTA_FILE_PATH = os.path.join(DEFAULT_USER_BASE_DIR, "github/ppi-classifiers/data/uniprot_seqs_both_sets.txt")
LOCAL_PDB_DIR = os.path.join(DEFAULT_USER_BASE_DIR, "pdb/templates/")
FOLDSEEK_TSV_DIR = os.path.join(DEFAULT_USER_BASE_DIR, "homolog-results/foldseek/")

BASE_OUTPUT_JSONL_PATH = os.path.join(DEFAULT_USER_BASE_DIR, "github/ppi-classifiers/data/")
BASE_SCRATCH_PATH = os.path.join(DEFAULT_USER_BASE_DIR, "scratch")
DATAFILE_PATH = os.path.join(DEFAULT_USER_BASE_DIR, "github/ppi-classifiers/data/features_with_tempid.csv")

# RESIDUE_MAP as provided in your original script context (second script in first prompt)
RESIDUE_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XAA": "X",
    "MLY": "K", "HIC": "H", "TPO": "T"
}

# full directory of Foldseek output parameters
FOLDSEEK_COLUMNS = [
    'query', 'target', 'pident', 'fident', 'alnlen', 'mismatch', 'gapopen',
    'qstart', 'qend', 'qlen', 'tstart', 'tend', 'tlen', 'evalue', 'bits',
    'qaln', 'taln', 'qseq', 'tseq'
]

def parse_bundle_chain_mapping(chainmapping_file):
    """Parse chain mapping file for PDB bundle."""
    chain_mapping = {}
    current_key = None
    skip_head = True
    with open(chainmapping_file, "r") as file:
        for line in file:
            line = line.strip()
            if skip_head:
                skip_head = False
                continue
            if not line:
                continue
            if line.endswith(".pdb:"):
                current_key = line.rstrip(":")
                chain_mapping[current_key] = {}
            else:
                new_chain_id, original_chain_id = line.split()
                chain_mapping[current_key][new_chain_id] = original_chain_id
    return chain_mapping

def get_structure_from_file(pdb_id, pdb_dir):
    """Loads structure from a .cif or .pdb file."""
    pdb_id_lower = pdb_id.lower()
    cif_path = os.path.join(pdb_dir, f"{pdb_id_lower}.cif")
    pdb_path = os.path.join(pdb_dir, f"{pdb_id_lower}.pdb")

    structure = None
    if os.path.exists(cif_path):
        parser = MMCIFParser(Quiet = True)
        structure = parser.get_structure(pdb_id_lower, cif_path)
    elif os.path.exists(pdb_path):
        parser = PDBParser(QUIET = True)
        structure = parser.get_structure(pdb_id_lower, pdb_path)
    else:
        logging.warning(f"Structure file for {pdb_id} not found in {pdb_dir}.")
        return None
    
    return structure[0]

def unzip_res_range(res_range):
	'''Converts ranges in the form: [2-210] or [3-45,47A,47B,51-67] into lists of strings including all numbers in these ranges in order'''
	res_ranges = res_range.strip()[1:-1].split(',')
	index_list = []
	for r in res_ranges:
		if re.match('.+-.+', r):
			a, b = r.split('-')
			index_list += [str(n) for n in range(int(a), int(b)+1)]
		else:
			index_list.append(r)
	
	if index_list == ['']:
		return []
	else:
		return index_list
    
def download_pdb(pdbcode, datadir, downloadurl = "https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    pdbcode_lower = pdbcode.lower()
    
    # 1. Try to download the modern .cif format first
    cif_fn = pdbcode_lower + ".cif"
    cif_url = downloadurl + cif_fn
    out_cif = os.path.join(datadir, cif_fn)
    
    try:
        urllib.request.urlretrieve(cif_url, out_cif)
        print(f"  -> Successfully downloaded {cif_fn}")
        return out_cif
    except urllib.error.HTTPError as e:
        # If the .cif file doesn't exist (404), try the legacy .pdb format
        if e.code == 404:
            print(f"  -> .cif not found for {pdbcode}, trying .pdb...")
        else:
            print(f"  -> Error downloading {cif_fn}: {e}")
            return None
    
    # 2. If .cif failed with a 404, try the legacy .pdb format
    pdb_fn = pdbcode_lower + ".pdb"
    pdb_url = downloadurl + pdb_fn
    out_pdb = os.path.join(datadir, pdb_fn)
    
    try:
        urllib.request.urlretrieve(pdb_url, out_pdb)
        print(f"  -> Successfully downloaded {pdb_fn}")
        return out_pdb
    except Exception as e:
        print(f"  -> Error downloading {pdb_fn} as well: {e}")
        return None

def get_mapped_binding_sites(alignment):
    """Map binding sites from template to query by aligning query to PDB-extracted sequence using Foldseek."""

    # Extract PDB ID and chain from the Foldseek hit
    pdb, chain = alignment["target"].split("_")
    pdb = pdb.split('-')[0].lower()  # Standard PDB ID in lowercase

    # Get structure model from local .cif or .pdb files
    t_protein_model = get_structure_from_file(pdb, LOCAL_PDB_DIR)
    if t_protein_model is None or chain not in t_protein_model:
        logging.warning(f"Could not get chain '{chain}' for PDB '{pdb}' from local files.")
        return {}
    
    t_chain = t_protein_model[chain]
    scratch_dir = tempfile.mkdtemp(dir = BASE_SCRATCH_PATH)

    try:
        """ Perform 3-way sequence aligment.
            - A   = the original query protein sequence
            - A'  = the Foldseek-derived template sequence
            - A'' = the PDB structure's sequence
            First, map PDB sequence to Foldseek sequence
            Second, map the Foldseek sequence to the query sequence
        """
        # Extract sequence (A'') and residue IDs from PDB structure
        t_residue_id_pos_mapping = {res.id[1]: i + 1 for i, res in enumerate(t_chain.get_residues())}
        t_pdb_seq = ''.join([RESIDUE_MAP.get(res.resname, "X") for res in t_chain.get_residues()])
        print(f"DEBUG: For PDB {pdb}_{chain}, t_residue_id_pos_mapping (first 10): {t_residue_id_pos_mapping[:10]}")
        print(f"Extracted sequence for chain {chain}: {t_pdb_seq}")

        # Get foldseek sequence (A') from alignment
        t_foldseek_seq = alignment['tseq']

        # Align PDB sequence (A'') with Foldseek sequence (A') using BLAST
        t_pdb_seq_file = os.path.join(scratch_dir, "t_pdb_seq.fasta")
        with open(t_pdb_seq_file, 'w') as f:
            f.write(f">{pdb}_{chain}\n{t_pdb_seq}\n")
        t_foldseek_seq_file = os.path.join(scratch_dir, "t_foldseek_seq.fasta")
        with open(t_foldseek_seq_file, 'w') as f:
            f.write(f">{alignment['target']}\n{t_foldseek_seq}\n")
        
        ## perform BLAST alignment of A' to A''
        blast_result_file = os.path.join(scratch_dir, "blast_result.xml")
        os.system(f"blastp -query {t_foldseek_seq_file} -subject {t_pdb_seq_file} -out {blast_result_file} -outfmt 5")
        if not os.path.exists(blast_result_file) or os.path.getsize(blast_result_file) == 0:
            logging.warning(f"BLAST alignment failed for {alignment['target']}")
            return {}
        with open(blast_result_file) as result:
            blast_record = NCBIXML.read(result)
        if not blast_record.alignments or blast_record.alignments[0].hsps:
            logging.warning(f"No BLAST HSP found for {alignment['target']}")
            return {}
        hsp = blast_record.alignments[0].hsps[0] # take top HSP

        # create mapping dictionaries: PDB -> Foldseek -> Query
        # Map PDB positions (A'') to Foldseek sequence positions (A')
        pdb_to_foldseek_map = {}
        q_pos, s_pos = hsp.query_start, hsp.sbjct_start

        for i in range(len(hsp.sbjct)):
            if hsp.sbjct[i] != "-" and hsp.query[i] != "-":
                pdb_to_foldseek_map[s_pos] = q_pos
            if hsp.sbjct[i] != "-": s_pos += 1
            if hsp.query[i] != "-": q_pos += 1
        
        # Map Foldseek sequence positions (A') to Query sequence positions (A)
        foldseek_to_query_map = {}
        q_pos, t_pos = alignment["qstart"], alignment["tstart"]
        for i in range(len(alignment["taln"])):
            if alignment["taln"][i] != "-" and alignment["qaln"][i] != "-":
                foldseek_to_query_map[t_pos] = q_pos
            if alignment["taln"][i] != "-": t_pos += 1
            if alignment["qaln"][i] != "-": q_pos += 1

        print(f"BLAST command: blastp -query {t_foldseek_seq_file} -subject {t_pdb_seq_file} -outfmt 5")
        print(f"Alignment: query = {hsp.query}, sbjct = {hsp.sbjct}")
        print(f"Identity: {hsp.identities / len(hsp.query) * 100:.2f}%, E-value: {hsp.expect}")
        print(f"Foldseek alignment: qaln = {alignment['qaln']}, taln = {alignment['taln']}")

        # chain maps together to create PDB-Query map
        final_residue_map = {}
        for res_id, pdb_pos in t_residue_id_pos_mapping.items():
            foldseek_pos = pdb_to_foldseek_map.get(pdb_pos)
            if foldseek_pos:
                query_pos = foldseek_to_query_map.get(foldseek_pos)
                if query_pos:
                    # convert from 1-based to 0-based index for final query position
                    final_residue_map[str(res_id)] = query_pos - 1

        return final_residue_map

    finally:
        # need to clean up temporary directory for subsequent use
        shutil.rmtree(scratch_dir)

def parse_foldseek_tsv(tsv_file):
    if not os.path.exists(tsv_file) or os.path.getsize(tsv_file) == 0:
        return []
    df = pd.read_csv(tsv_file, sep = '\t', header = None, names = FOLDSEEK_COLUMNS)
    # Convert numeric columns
    for col in ['pident', 'fident', 'evalue', 'bits']:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')
    for col in ['alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'qlen', 'tstart', 'tend', 'tlen']:
        df[col] = pd.to_numeric(df[col], errors = 'coerce', downcast = 'integer')
    return df.to_dict('records')

def load_fasta_sequences(fasta_file_path):
    # return dict formatted as UniprotID : sequence.
    fasta_dict = {}
    with open(fasta_file_path, 'r') as infile:
        for line in infile:
            line_list = line.strip().split()
            fasta_dict[line_list[0]] = line_list[1]
    return fasta_dict

def get_sequence_from_dict(uniprot_id, fasta_dict):
    ''' Helper function to get a sequence of a UniProt ID'''
    seq = fasta_dict.get(uniprot_id)
    return seq

def process_single_pair(p1_id, p2_id, dataset_name, fasta_sequences_dict, ires_df, pair_output_idx, rerun_xml = False, out_dir = None, chunk_id = None):

    pair_key = f"{p1_id}_{p2_id}"
    logging.info(f"Processing pair: {pair_key}")

    # SETUP: Define paths to Foldseek results
    alignments1 = parse_foldseek_tsv(os.path.join(FOLDSEEK_TSV_DIR, f"{p1_id}.tsv"))
    alignments2 = parse_foldseek_tsv(os.path.join(FOLDSEEK_TSV_DIR, f"{p2_id}.tsv"))
    # if either query protein does not return any homologs, or if for whatever reason the
    # Foldseek-returned .tsv files are empty, there is no point in searching for template pairs.
    if not alignments1 or not alignments2:
        logging.warning(f"No Foldseek hits found for one or both proteins in {pair_key}. Skipping.")
        return

    # STAGE 1: Collect all valid template candidates
    candidate_templates = []
    for align1 in alignments1:
        for align2 in alignments2:
            pdb1_full, chain1 = align1['target'].rsplit('_', 1)
            pdb2_full, chain2 = align2['target'].rsplit('_', 1)
            pdb1_id = pdb1_full.split('-')[0]
            pdb2_id = pdb2_full.split('-')[0]

            if pdb1_id != pdb2_id or chain1 == chain2:
                continue

            # Check if this PDB chain pair is a known interface
            ires_row = ires_df[
                ((ires_df["PDB"] == pdb1_id.upper()) & (ires_df["ChainA"] == chain1) & (ires_df["ChainB"] == chain2)) | 
                ((ires_df["PDB"] == pdb1_id.upper()) & (ires_df["ChainA"] == chain2) & (ires_df["ChainB"] == chain1))
            ]
            if ires_row.empty:
                continue
            
            # This is a valid candidate
            # modify/add/delete foldseek output parameters as needed
            score = (align1['fident'] + align2['fident']) / 2.0
            template_data = {
                "pdb_id": pdb1_id,
                "template_pair": (align1['target'], align2['target']),
                "alignment_data": (align1, align2),
                "score": score,
                "fident": (align1['fident'], align2['fident']),
                "pident": (align1['pident'], align2['pident']),
                "e_value": (align1['evalue'], align2['evalue']),
                "ires_row_index": ires_row.index[0]
            }
            candidate_templates.append(template_data)

    # if no valid template pairs found, structural data cannot be encoded
    if not candidate_templates:
        logging.info(f"No valid templates found for {pair_key}.")
        return
    
    # --- STAGE 2: CATEGORIZE THE FINAL, FILTERED TEMPLATES ---
    ## LOGIC: This is the key categorization from your BLAST script, now applied to Foldseek results.
    best_templates = {"condition_1": [], "condition_2": []}
    for template_data in candidate_templates:
        e_val_1, e_val_2 = template_data["e_value"]
        # Convert fident (0-1 scale) to percent for comparison (e.g., 0.8 -> 80)
        pident_1 = template_data["pident"][0] 
        pident_2 = template_data["pident"][1]

        # Conditions to remove self templates
        if e_val_1 > 1e-5 or e_val_2 > 1e-5 or pident_1 < 80 or pident_2 < 80:
            best_templates["condition_1"].append(template_data)

        if e_val_1 > 1e-5 or e_val_2 > 1e-5 or pident_1 < 50 or pident_2 < 50:
            best_templates["condition_2"].append(template_data)

    for condition, templates in best_templates.items():

        if not templates:
            continue

        # Sort templates for this specific condition by score
        templates.sort(key = lambda x: x["score"], reverse = True)

        top_template = templates[0]
        templates_with_mapping = []
        logging.info(f"Processing {len(templates)} templates for {pair_key} under {condition}")
        align1, align2 = top_template['alignment_data']
        map1, map2 = get_mapped_binding_sites(align1), get_mapped_binding_sites(align2)
        if not map1 or not map2:
            logging.warning(f"Mapping failed for {template_data['template_pair']}. Skipping.")
            continue

        
        SIZ, COV = 0, 0.0
        pdb_id = top_template['pdb_id']
        structure = get_structure_from_file(pdb_id, LOCAL_PDB_DIR)

        if structure:
            # get ires data for this template from ires_df
            ires_row_data = ires_df.loc[top_template['ires_row_index']]
            _, template_chain_A = top_template['template_pair'][0].rsplit('_', 1)
            _, template_chain_B = top_template['template_pair'][1].rsplit('_', 1)

            # determine which column, PDBIResA, PDBIResB, corresponds to which chain
            if ires_row_data['ChainA'] == template_chain_A:
                ires_list_A = unzip_res_range(ires_row_data['PDBIresA'])
                ires_list_B = unzip_res_range(ires_row_data['PDBIresB'])
            else: # The chains are swapped in the ires file
                ires_list_A = unzip_res_range(ires_row_data['PDBIresB'])
                ires_list_B = unzip_res_range(ires_row_data['PDBIresA'])

            chainA_obj, chainB_obj = structure[template_chain_A], structure[template_chain_B]

            # create list of actual Bio.PDB.Residue objects from ires lists
            residues_A = [chainA_obj[int(res_id)] for res_id in ires_list_A if res_id.isdigit() and int(res_id) in chainA_obj]
            residues_B = [chainB_obj[int(res_id)] for res_id in ires_list_B if res_id.isdigit() and int(res_id) in chainB_obj]

            # use NeighborSearch to find template IRes pairs
            all_atoms_B = [atom for res in residues_B for atom in res.get_atoms() if atom.name == 'CA']
            ns = NeighborSearch(all_atoms_B)

            template_interacting_pairs = []
            for res_a in residues_A:
                if 'CA' in res_a:
                    ca_a = res_a['CA']
                    # find all CA atoms in chain B <=6.05A of this CA in chain A
                    nearby_b_atoms = ns.search(ca_a.coord, 6.05, 'A')
                    for atom_b in nearby_b_atoms:
                        res_b = atom_b.get_parent()
                        template_interacting_pairs.append((str(res_a.id[1]), str(res_b.id[1])))
            n_template_pairs = len(template_interacting_pairs)

            # < TODO >
            mapped_query_pairs = []
            for template_res_a, template_res_b in template_interacting_pairs:
                query_pos_a = map1.get(template_res_a)
                query_pos_b = map2.get(template_res_b)

                if query_pos_a is not None and query_pos_b is not None:
                    mapped_query_pairs.append((query_pos_a, query_pos_b))

            seq_a = get_sequence_from_dict(p1_id, fasta_sequences_dict)
            seq_b = get_sequence_from_dict(p2_id, fasta_sequences_dict)

            if seq_a is None or seq_b is None:
                logging.warning(f"Could not load sequence for {p1_id} or {p2_id}")
                SIZ, COV = 0, 0.0
            elif len(mapped_query_pairs) == 0:
                logging.warning(f"No template pairs could be mapped to query for {pair_key}")
                SIZ, COV = 0, 0.0
            else:
                # compute distances between query residues
                surviving_pairs = []

                query_structure_a = 0 # TODO
                query_structure_b = 0 # TODO

                

        # Define a unique output file for this specific condition
        output_file = os.path.join(out_dir, f"best_templates_{condition}_{dataset_name}_{pair_key}.jsonl")
    
        # Create the dictionary to dump
        data_to_dump = {pair_key: templates_with_mapping}
        
        with open(output_file, "w") as f:
            json.dump(data_to_dump, f, indent = 4)

def main():
    """Main function to drive the pipeline"""

    parser = argparse.ArgumentParser(description = "Process a slice of a dataset using Foldseek results.")
    parser.add_argument('--data_file', default = DATAFILE_PATH, help = "data file containing experimental protein pairs and all features.")
    parser.add_argument('--task-id', required = True, type = int)
    parser.add_argument('--total-tasks', required = True, type = int)
    args = parser.parse_args()

    # Load shared resources
    fasta_sequences_dict = load_fasta_sequences(FASTA_FILE_PATH)
    ires_df = pd.read_csv(IRES_FILE, sep = "\t")
    data = pd.read_csv(DATAFILE_PATH)

    # Read and partition the pair list
    all_pairs = [tuple(ppi_string.split(':')) for ppi_string in data['ppi'].dropna() if len(ppi_string.split(':')) == 2]
    assert len(all_pairs) == len(data), "Error: some PPI pairs were not successfully accounted for."
    
    num_all_pairs = len(all_pairs)
    pairs_per_task = (num_all_pairs + args.total_tasks - 1) // args.total_tasks
    start_index = args.task_id * pairs_per_task
    end_index = min(start_index + pairs_per_task, num_all_pairs)
    my_slice_of_pairs = all_pairs[start_index:end_index]
    
    logging.info(f"Task {args.task_id}: Processing {len(my_slice_of_pairs)} pairs from index {start_index} to {end_index-1}")

    output_dir = os.path.join(BASE_OUTPUT_JSONL_PATH, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each pair in the assigned slice
    for pair_idx, (p1, p2) in enumerate(my_slice_of_pairs):
        process_single_pair(
            p1, p2, args.dataset, fasta_sequences_dict, ires_df, pair_idx,
            out_dir = output_dir, chunk_id = f"task_{args.task_id}"
        )

if __name__ == "__main__":
    main()





