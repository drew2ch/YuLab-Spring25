""" Script to generate SIZ/COV features as defined by the PrePPI interaction classifier model.
    Incorporates Juheon Chu's code plus custom machine-level modifications by Andrew Chung.
--- Run 1: 8/1/2025 -- full comprehensive run for initial Genomic/PrePPI intersection pairs
--- Run 2: 8/21/2025 -- augmentative run to accommodate expanded training set
"""

import gzip
import tarfile
import tempfile
import urllib.request

from pathlib import Path
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch
import re
import os
import glob
import numpy as np
import pandas as pd
import json
import shutil
import argparse
import logging
import subprocess

import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# --- Configuration ---
logging.basicConfig(
    level = logging.INFO, 
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
warnings.simplefilter('ignore', PDBConstructionWarning)

DEFAULT_USER_BASE_DIR = "C:/Users/hychu/OneDrive/Desktop/Summer25"
DATA_BASE_DIR = "F:/Research/ppi-cdata" # utilizing external drive for data storage

IRES_TEMPLATE_FILE = os.path.join(DATA_BASE_DIR, "ires_perpdb_alltax.txt")
# IRES_QUERY_FILE = os.path.join(DATA_BASE_DIR, "ires_all.txt")
# FASTA_FILE_PATH = os.path.join(DATA_BASE_DIR, "uniprot_seqs_both_sets.txt")
YU_PDB_DIR = "/share/yu/resources/pdb/"
YU_PDB_BUNDLE_DIR = "/share/yu/resources/pdb_like/"
LOCAL_PDB_DIR = os.path.join(DATA_BASE_DIR, "pdb-templates/")
FOLDSEEK_TSV_DIR = os.path.join(DATA_BASE_DIR, "foldseek-homologs/")
AF3_MODELS_DIR = os.path.join(DATA_BASE_DIR, "af3-models/")
BASE_OUTPUT_JSONL_PATH = os.path.join(DATA_BASE_DIR, "template-data")
BASE_SCRATCH_PATH = os.path.join(DATA_BASE_DIR, "scratch")
CANDIDATE_TEMPLATES_PATH = os.path.join(DATA_BASE_DIR, "candidate-templates-top20")
DATAFILE_PATH = os.path.join(DEFAULT_USER_BASE_DIR, "github/ppi-classifiers/data/features.csv")

# Define Data Files
IRES_TEMPLATE_DATA = pd.read_csv(IRES_TEMPLATE_FILE, sep = "\t")\
    .dropna(subset = ['UniProtA', 'UniProtB', 'ChainA', 'ChainB'])
DATA = pd.read_csv(DATAFILE_PATH)
CA_DISTANCE_THRESHOLD = 6.05 # Angstroms, as defined by PrePPI model

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
    """Loads structure (template and query) from a .cif or .pdb file."""
    for ext in ['.cif', '.pdb']:
        path = os.path.join(pdb_dir, f"{pdb_id.lower()}{ext}")
        if os.path.exists(path):
            try:
                parser = MMCIFParser(QUIET = True) if ext == '.cif' else PDBParser(QUIET = True)
                structure = parser.get_structure(pdb_id, path)[0]
                logging.info(f"Successfully loaded structure {pdb_id} from {path}.")
                return structure
            except Exception as e:
                logging.error(f"Could not parse query structure {path}: {e}")
                return None
    logging.warning(f"Query structure not found for {pdb_id} in {pdb_dir}.")
    return None

def get_structure_from_archive(pdb_id, archive_path):
    """ Loads .cif/.pdb structure (template) from a .ent.gz file. Adapted in concordance with .tar.gz based database."""
    pdb_id_lower = pdb_id.lower()
    subdir = pdb_id_lower[1:3] # middle 2 characters of PDB ID
    filename = f"pdb{pdb_id_lower}.ent.gz"
    filepath = os.path.join(archive_path, subdir, filename)
    if not os.path.exists(filepath):
        logging.warning(f"Structure file {filepath} not found.")
        return None
    
    try:
        with gzip.open(filepath, 'rt') as structure_handle:
            # --- Tier 1: Try parsing as a modern mmCIF file first ---
            try:
                parser = MMCIFParser(QUIET = True)
                structure = parser.get_structure(pdb_id, structure_handle)[0]
                logging.info(f"Successfully parsed {filepath} as mmCIF format.")
                return structure
            except Exception as e:
                logging.warning(f"Could not parse {filepath} as mmCIF ({e}). Attempting legacy PDB format...")
                # CRITICAL: Rewind the file handle to the beginning for the next parser.
                structure_handle.seek(0)
                
                # --- Tier 2: If mmCIF fails, try parsing as a legacy PDB file ---
                try:
                    parser = PDBParser(QUIET = True)
                    structure = parser.get_structure(pdb_id, structure_handle)[0]
                    logging.info(f"Successfully parsed {filepath} as legacy PDB format.")
                    return structure
                except Exception as final_e:
                    logging.error(f"Failed to parse {filepath} as either mmCIF or PDB format. Final error: {final_e}")
                    return None
    except Exception as e:
        logging.error(f"Error reading structure from {filepath}: {e}")
        return None
    
def get_structure_from_bundle(pdb_id, bundle_base_dir):
    """ Loads .cif/.pdb structure (template) from a .tar.gz file. Adapted in concordance with .tar.gz based database."""
    pdb_id_lower = pdb_id.lower()
    subdir = pdb_id_lower[1:3] # middle 2 characters of PDB ID
    filename = f"{pdb_id_lower}-pdb-bundle.tar.gz"
    filepath = os.path.join(bundle_base_dir, subdir, filename)
    if not os.path.exists(filepath):
        logging.warning(f"Structure file {filepath} not found.")
        return None
    
    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            # --- Tier 1: Try parsing as a modern mmCIF file first ---
            cif_member_name = f"{pdb_id_lower}.cif"
            try:
                cif_file_handle = tar.extractfile(cif_member_name)
                if cif_file_handle:
                    parser = MMCIFParser(QUIET = True)
                    structure = parser.get_structure(pdb_id, cif_file_handle)[0]
                    cif_file_handle.close()
                    logging.info(f"Successfully parsed {cif_member_name} from bundle {filepath}.")
                    return structure
            except KeyError:
                logging.warning(f"Member file '{cif_member_name}' not found in bundle {filepath}.")
                return None
    except Exception as e:
        logging.error(f"Error reading structure from {filepath}: {e}")
        return None
    
def read_cif(structure):
    """ Read a .cif file and return a data structure containing all chains.
    --- For the express purpose of computing inter-Ca distances, the pLDDT module won't be necessary.
    --- Instead, we will extract the coordinates of the C-alpha atoms (CA) for each chain.
    """

    chain_coords = {}

    for model in structure:
        for chain in model:
            chain_coords[chain.id] = []
            for residue in chain:
                if residue.has_id('CA'):
                    ca_atom = residue['CA']
                    chain_coords[chain.id].append(ca_atom.get_coord())
    
    # convert to arrays
    for chain in chain_coords:
        chain_coords[chain] = np.array(chain_coords[chain])
    
    return chain_coords

"""def unzip_res_range(res_range):
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
		return index_list"""
    
def unzip_res_range(res_range: str) -> list[str]:
    """
    Converts ranges like '[2-10,45A,51-53B]' into a list of individual residue ID strings.
    It expands purely numeric ranges but keeps alphanumeric parts as-is.
    """
    if not isinstance(res_range, str) or not res_range.startswith('[') or not res_range.endswith(']'):
        return []

    content = res_range.strip()[1:-1]
    if not content:
        return []
    
    res_ranges = content.split(',')
    index_list = []
    
    for r in res_ranges:
        if '-' in r:
            parts = r.split('-')
            if len(parts) == 2:
                start, end = parts
                # Only expand if both start and end are purely numeric
                if start.isdigit() and end.isdigit():
                    index_list.extend(str(n) for n in range(int(start), int(end) + 1))
                else:
                    # If not purely numeric (e.g., '101A-101C'), don't expand.
                    # This is a safe fallback, as iterating these is complex and rare.
                    # Your `parse_pdb_residue_id` will handle the individual IDs if they appear alone.
                    index_list.append(r) 
            else: # Malformed range
                index_list.append(r)
        else:
            index_list.append(r)
            
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

def get_mapped_binding_sites(
        alignment: dict, # foldseek alignment data (query, target, ...)
        t_protein_model: object # pre-loaded Bio.PDB structure object for the template
    ) -> dict[str, int]:
    # Map binding sites from template to query by aligning query to PDB-extracted sequence using Foldseek.
    # 8/21/2025: updated in concordance with .ent.gz data format adaptation.

    # Verify that the alignment contains all required keys
    required_keys = ['target', 'qaln', 'taln', 'qstart', 'tstart']
    if not all(key in alignment for key in required_keys):
        logging.error(f"Missing required alignment keys: {required_keys}")
        return {}
    
    # Extract PDB ID and chain from the Foldseek hit
    pdb, chain = alignment["target"].split("_")
    pdb = pdb.split('-')[0].lower()  # Standard PDB ID in lowercase
    
    if chain not in t_protein_model:
        logging.error(f"Chain '{chain}' not found in PDB structure {pdb}.")
        return {}
    
    t_chain = t_protein_model[chain]
    os.makedirs(BASE_SCRATCH_PATH, exist_ok = True)
    scratch_dir = tempfile.mkdtemp(dir = BASE_SCRATCH_PATH)
    logging.debug(f"Created temporary directory: {scratch_dir}")

    try:
        """Perform 3-way sequence aligment.
            - A   = the original query protein sequence
            - A'  = the Foldseek-derived template sequence
            - A'' = the PDB structure's sequence
            First, map PDB sequence to Foldseek sequence
            Second, map the Foldseek sequence to the query sequence"""
        
        # --- 1. Extract sequence (A'') and residue IDs from PDB structure ---
        t_residue_id_pos_mapping = {f"{res.id[1]}{res.id[2].strip()}": i + 1 for i, res in enumerate(t_chain.get_residues())}
        t_pdb_seq = ''.join([RESIDUE_MAP.get(res.resname, "X") for res in t_chain.get_residues()])
        if not t_pdb_seq:
            logging.warning(f"No valid residues found to extract sequence for PDB {pdb}_{chain}.")
            return {}
        if len(t_pdb_seq) < 10:
            logging.warning(f"Extracted sequence for PDB {pdb}_{chain} is too short ({len(t_pdb_seq)} residues).")
            return {}
        print(f"DEBUG: For PDB {pdb}_{chain}, t_residue_id_pos_mapping (first 10): {list(t_residue_id_pos_mapping.items())[:10]}")
        print(f"Extracted sequence for chain {chain}: {t_pdb_seq}")

        # Get foldseek sequence (A') from alignment
        t_foldseek_seq = alignment['taln'].replace('-', '')

        # --- 2. Align PDB sequence (A'') with Foldseek sequence (A') using BLAST ---
        t_pdb_seq_file = os.path.join(scratch_dir, "t_pdb_seq.fasta")
        with open(t_pdb_seq_file, 'w') as f:
            f.write(f">{pdb}_{chain}\n{t_pdb_seq}\n")

        t_foldseek_seq_file = os.path.join(scratch_dir, "t_foldseek_seq.fasta")
        with open(t_foldseek_seq_file, 'w') as f:
            f.write(f">{alignment['target']}\n{t_foldseek_seq}\n")
        
        ## perform BLAST alignment: A' to A''
        blast_result_file = os.path.join(scratch_dir, "blast_result.xml")
        blastp_cline = NcbiblastpCommandline(
            query = t_foldseek_seq_file,
            subject = t_pdb_seq_file,
            out = blast_result_file,
            outfmt = 5
        )
        logging.debug(f"Executing BLAST Command: {blastp_cline}")
        # os.system(f"blastp -query {t_foldseek_seq_file} -subject {t_pdb_seq_file} -out {blast_result_file} -outfmt 5")

        try:
            stdout, stderr = blastp_cline()
        except subprocess.CalledProcessError as e:
            logging.error(f"BLAST alignment failed for {alignment['target']}. Stderr: {e.stderr}")
            return {}
        except FileNotFoundError:
            logging.error(f"BLAST+ executables not found. Please ensure BLAST+ installed and in system's PATH.")
            return {}
        if not os.path.exists(blast_result_file) or os.path.getsize(blast_result_file) == 0:
            logging.warning(f"BLAST generated empty result file for {alignment['target']}")
            return {}
        
        with open(blast_result_file) as result:
            blast_record = NCBIXML.read(result)
        if not blast_record.alignments or \
                not blast_record.alignments[0].hsps:
            logging.warning(f"No BLAST HSP found for {alignment['target']} vs PDB {pdb}_{chain}.")
            return {}
        
        hsp = blast_record.alignments[0].hsps[0] # take top HSP
        logging.debug(f"BLAST HSP found. Identity: {hsp.identities/len(hsp.query) * 100:.2f}%, E-value = {hsp.expect}")
        logging.debug(f"BLAST alignment query start: {hsp.query_start}, subject start: {hsp.sbjct_start}")
        logging.debug(f"Coordinate ranges - PDB: 1-{len(t_pdb_seq)}, "
                     f"Foldseek: {alignment['tstart']}-{alignment['tend']}, "
                     f"Query: {alignment['qstart']}-{alignment['qend']}")

        # --- 3. create mapping dictionaries: PDB -> Foldseek -> Query ---

        # -- Map 3a. PDB positions (A'') to Foldseek sequence positions (A') --
        pdb_to_foldseek_map = {}
        q_pos, s_pos = hsp.query_start, hsp.sbjct_start

        for i in range(len(hsp.sbjct)):
            if hsp.sbjct[i] != "-" and hsp.query[i] != "-":
                pdb_to_foldseek_map[s_pos] = q_pos
            if hsp.sbjct[i] != "-": s_pos += 1
            if hsp.query[i] != "-": q_pos += 1
        logging.debug(f"PDB to Foldseek map size = {len(pdb_to_foldseek_map)}")
        
        # -- Map 3b. Map Foldseek sequence positions (A') to Query sequence positions (A) --
        foldseek_to_query_map = {}
        q_pos, t_pos = alignment.get("qstart"), alignment.get("tstart")
        for i in range(len(alignment["taln"])):
            if alignment["taln"][i] != "-" and alignment["qaln"][i] != "-":
                foldseek_to_query_map[t_pos] = q_pos
            if alignment["taln"][i] != "-": t_pos += 1
            if alignment["qaln"][i] != "-": q_pos += 1
        logging.debug(f"Foldseek to Query map size = {len(foldseek_to_query_map)}")

        # -- Map 3c. (Bridge) Gapless taln position -> Full PDB residue number; fix coordinate system mismatch --
        gapless_to_full_target_map = {}
        gapless_pos = 1
        full_target_pos = alignment.get('tstart')
        for residue in alignment['taln']:
            if residue != '-':
                gapless_to_full_target_map[gapless_pos] = full_target_pos
                gapless_pos += 1
                full_target_pos += 1

        print(f"BLAST command: blastp -query {t_foldseek_seq_file} -subject {t_pdb_seq_file} -outfmt 5")
        print(f"Alignment: query = {hsp.query}, sbjct = {hsp.sbjct}")
        print(f"Identity: {hsp.identities / len(hsp.query) * 100:.2f}%, E-value: {hsp.expect}")
        print(f"Foldseek alignment: qaln = {alignment['qaln']}, taln = {alignment['taln']}")

        # --- 4. chain maps together to create final PDB-Query map ---
        final_residue_map = {}
        for res_id, pdb_pos in t_residue_id_pos_mapping.items():

            # Step 6a. PDB sequential pos -> Gapless taln position
            gapless_taln_pos = pdb_to_foldseek_map.get(pdb_pos)
            if not gapless_taln_pos:
                continue

            # Step 6b. Gapless taln position -> Full PDB residue number (bridge)\
            full_target_pos = gapless_to_full_target_map.get(gapless_taln_pos)
            if not full_target_pos:
                continue

            # Step 6c. Full PDB residue number -> Query position
            query_pos = foldseek_to_query_map.get(full_target_pos)
            
            if query_pos is not None:
                final_residue_map[str(res_id)] = query_pos - 1 # convert to 0-based index

        # output mapping quality metrics
        mapped_residues = len(final_residue_map)
        total_residues = len(t_residue_id_pos_mapping)
        mapping_coverage = (mapped_residues / total_residues) * 100 if total_residues > 0 else 0
        logging.info(f"Mapping coverage: {mapped_residues}/{total_residues} ({mapping_coverage:.1f}%)")
        
        if mapping_coverage < 30:  # warn if coverage is very low
            logging.warning(f"Low mapping coverage ({mapping_coverage:.1f}%) - results may be unreliable")
        
        return final_residue_map

    finally:
        # need to clean up temporary directory for subsequent use
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)
            logging.debug(f"Cleaned up temporary directory: {scratch_dir}")

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

def parse_pdb_residue_id(res_id: str) -> tuple:
    """Parst a PDB residue ID string ('101', '101A') into a Bio.PDB compatible tuple."""
    match = re.match(r'^(\d+)([A-Za-z]?)$', res_id)
    if not match: return None
    res_num = int(match.group(1))
    ins_code = match.group(2).upper() if match.group(2) else ' '
    return (' ', res_num, ins_code)  # PDB format requires a tuple (hetfield, resseq, icode)

def process_single_pair(p1_id, p2_id):
    """
    Process a single protein pair to compute SIZE and COV metrics by mapping 
    template interface residues to query complexes.
    
    Args:
    --- p1_id, p2_id: UniProt IDs for the protein pair
    --- ires_df: DataFrame containing interface residue data
    --- index: Index for logging purposes
    
    Returns:
    --- dict: Contains pair data with SIZE and COV metrics for each condition
    """

    pair_key = f"{p1_id}_{p2_id}"
    pair_data = {
        "pair_key": pair_key,
        "pdb_id": None, "chain_ids": None, 
        "fident": 0.0,
        "pident": 0.0,
        "e_value": 0.0,
        "SIZE": 0, "COV": 0.0
    }
    # SETUP: Define paths to Foldseek results
    try:
        alignments1 = parse_foldseek_tsv(os.path.join(FOLDSEEK_TSV_DIR, f"{p1_id}.tsv"))
        alignments2 = parse_foldseek_tsv(os.path.join(FOLDSEEK_TSV_DIR, f"{p2_id}.tsv"))
        if not alignments1 or not alignments2:
            logging.warning(f"No Foldseek hits for {pair_key}.")
            return pair_data # Return no data
    except Exception as e:
        logging.error(f"Error parsing Foldseek TSV files for {pair_key}: {e}")
        return pair_data
    
    # --- STAGE 1: Collect all valid template candidates ---
    candidate_templates = []
    templates_by_pdb = {} # collect PDB IDs from protein 1
    for align1 in alignments1:
        try:
            pdb1_id = align1['target'].split('-')[0].split('_')[0]
            if pdb1_id not in templates_by_pdb:
                templates_by_pdb[pdb1_id] = []
            templates_by_pdb[pdb1_id].append(align1)
        except (KeyError, IndexError):
            continue # Skip invalid targets

    # Now iterate through alignments2 and find matches in templates_by_pdb
    for align2 in alignments2:
        try:
            pdb2_id = align2['target'].split('-')[0].split('_')[0]
            if pdb2_id in templates_by_pdb:
                for align1 in templates_by_pdb[pdb2_id]:
                    chain1, chain2 = align1['target'].rsplit('_', 1)[1], align2['target'].rsplit('_', 1)[1]
                    if chain1 == chain2: continue

                    ires_row = IRES_TEMPLATE_DATA[
                        ((IRES_TEMPLATE_DATA["PDB"] == pdb2_id.upper()) & (IRES_TEMPLATE_DATA["ChainA"] == chain1) & (IRES_TEMPLATE_DATA["ChainB"] == chain2)) | 
                        ((IRES_TEMPLATE_DATA["PDB"] == pdb2_id.upper()) & (IRES_TEMPLATE_DATA["ChainA"] == chain2) & (IRES_TEMPLATE_DATA["ChainB"] == chain1))
                    ]
                    if ires_row.empty: continue

                    score = (align1['fident'] + align2['fident']) / 2.0
                    align1_native = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in align1.items()}
                    align2_native = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in align2.items()}

                    template_data = {
                        "pdb_id": pdb2_id,
                        "template_pair": (align1_native['target'], align2_native['target']),
                        # Store the cleaned alignment data
                        "alignment_data": (align1_native, align2_native), 
                        "score": float(score),
                        "fident": (float(align1_native['fident']), float(align2_native['fident'])),
                        "pident": (float(align1_native['pident']), float(align2_native['pident'])),
                        "e_value": (float(align1_native['evalue']), float(align2_native['evalue'])),
                        # Cast the index value to a standard Python integer
                        "ires_row_index": int(ires_row.index[0]) 
                    }
                    candidate_templates.append(template_data)
        except (ValueError, KeyError, IndexError) as e:
            logging.warning(f"Error processing alignment for {pair_key}: {e}")
            continue

    # if no valid template pairs found, structural data cannot be encoded
    if not candidate_templates:
        logging.info(f"No valid templates found for {pair_key}.")
        for cond in range(1, 4):
            output_jsonl_path = os.path.join(CANDIDATE_TEMPLATES_PATH, f"best_templates_condition_{cond}_train_{pair_key}.jsonl")
            os.makedirs(os.path.dirname(output_jsonl_path), exist_ok = True)
            with open(output_jsonl_path, 'w') as jsonl_file:
                json.dump({}, jsonl_file)
                jsonl_file.write('\n')
            logging.info(f"Saved empty template file for {pair_key} under condition '{cond}' to {output_jsonl_path}.")
        return pair_data
    else:
        logging.info(f"Found {len(candidate_templates)} candidate templates for {pair_key}.")
    candidate_templates.sort(key = lambda x: x["score"], reverse = True)

    # --- STAGE 1.5: COMPILE AND SAVE CANDIDATE TEMPLATES TO JSONL --- (09/15/2025)

    # Save top 20 candidates to JSONL for record-keeping, 3 distinct conditions
    # 1. e(A) > 1e-5 || e(B) > 1e-5 || f(A) < 0.8 || f(B) < 0.8
    # 2. e(A) > 1e-5 || e(B) > 1e-5 || f(A) < 0.5 || f(B) < 0.5
    # 3. Default (no filter); keep all templates
    
    for cond in range(1, 4):
        if cond == 1:
            filtered_templates = [
                t for t in candidate_templates 
                if t['e_value'][0] > 1e-5 or t['e_value'][1] > 1e-5 or t['fident'][0] < 0.8 or t['fident'][1] < 0.8
            ]
        elif cond == 2:
            filtered_templates = [
                t for t in candidate_templates 
                if t['e_value'][0] > 1e-5 or t['e_value'][1] > 1e-5 or t['fident'][0] < 0.5 or t['fident'][1] < 0.5
            ]
        else:
            filtered_templates = candidate_templates

        # re-sort filtered template pairs by score
        # filtered_templates.sort(key = lambda x: x["score"], reverse = True)
        top_n_templates = filtered_templates[:20]
        output_jsonl_path = os.path.join(CANDIDATE_TEMPLATES_PATH, f"best_templates_condition_{cond}_train_{pair_key}.jsonl")
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok = True)

        with open(output_jsonl_path, 'w') as jsonl_file:
            for template in top_n_templates:
                json.dump(template, jsonl_file)
                jsonl_file.write('\n')
        logging.info(f"Saved top {len(top_n_templates)} templates for {pair_key} under condition '{cond}' to {output_jsonl_path}.")

    # --- STAGE 2: FOR EACH CONDITION, MAP TEMPLATE PAIRS TO QUERY COMPLEX ---

    # Sort candidate templates for this specific condition by score, and then select top template
    top_template = candidate_templates[0]
    pdb_id = top_template['pdb_id']
    
    logging.info(f"Processing top template for {pair_key}.")
    align1, align2 = top_template['alignment_data']

    # --- Centralized Structure Loading ---
    # a. load from main Yu Lab PDB mirror first
    structure = get_structure_from_archive(pdb_id, YU_PDB_DIR)
    # b. if not found, try the Yu Lab PDB_like mirror
    if not structure:
        structure = get_structure_from_bundle(pdb_id, YU_PDB_BUNDLE_DIR)
    # c. if still not found, try the download cache
    if not structure:
        structure = get_structure_from_file(pdb_id, LOCAL_PDB_DIR)
    # d. if all local sources fail, download from RCSB PDB
    if not structure:
        logging.warning(f"Structure {pdb_id} not found in local cache. Attempting to download.")
        download_pdb(pdb_id, LOCAL_PDB_DIR)
        structure = get_structure_from_file(pdb_id, LOCAL_PDB_DIR)
    # e. if all attempts fail, log error and skip
    if not structure:
        logging.error(f"Failed to load structure for {pdb_id} after all attempts.")
        return pair_data
    else:
        logging.info(f"Structure for {pdb_id} was successfully procured.")
    
    map1, map2 = get_mapped_binding_sites(align1, structure), get_mapped_binding_sites(align2, structure)
    if not map1 or not map2:
        logging.warning(f"Mapping failed for {top_template['template_pair']}. Skipping.")
        return pair_data
    
    # get ires data for this template from ires_df
    ires_row_data = IRES_TEMPLATE_DATA.loc[top_template['ires_row_index']]
    _, template_chain_A = top_template['template_pair'][0].rsplit('_', 1)
    _, template_chain_B = top_template['template_pair'][1].rsplit('_', 1)

    pair_data = {
        "pair_key": pair_key,
        "pdb_id": pdb_id, "chain_ids": f"{template_chain_A},{template_chain_B}", 
        "fident": top_template['score'],
        "pident": np.mean(top_template['pident']),
        "e_value": np.mean(top_template['e_value']),
        "SIZE": 0, "COV": 0.0
    }

    # determine which column, PDBIResA, PDBIResB, corresponds to which chain
    if ires_row_data['ChainA'] == template_chain_A and ires_row_data['ChainB'] == template_chain_B:
        ires_list_A = unzip_res_range(ires_row_data['PDBIresA'])
        ires_list_B = unzip_res_range(ires_row_data['PDBIresB'])
    elif ires_row_data['ChainA'] == template_chain_B and ires_row_data['ChainB'] == template_chain_A:
        ires_list_A = unzip_res_range(ires_row_data['PDBIresB'])
        ires_list_B = unzip_res_range(ires_row_data['PDBIresA'])
    else:
        logging.error(f"Chain mismatch for {pair_key} in template {pdb_id}. Skipping.")
        return pair_data
    chainA_obj, chainB_obj = structure[template_chain_A], structure[template_chain_B]

    # create list of actual Bio.PDB.Residue objects from ires lists
    # residues_A = [chainA_obj[int(res_id)] for res_id in ires_list_A if res_id.isdigit() and int(res_id) in chainA_obj]
    # residues_B = [chainB_obj[int(res_id)] for res_id in ires_list_B if res_id.isdigit() and int(res_id) in chainB_obj]
    residues_A, residues_B = [], []
    for res_id in ires_list_A:
        pdb_res_id = parse_pdb_residue_id(res_id)
        if pdb_res_id and pdb_res_id in chainA_obj:
            residues_A.append(chainA_obj[pdb_res_id])
    for res_id in ires_list_B:
        pdb_res_id = parse_pdb_residue_id(res_id)
        if pdb_res_id and pdb_res_id in chainB_obj:
            residues_B.append(chainB_obj[pdb_res_id])
    
    # use NeighborSearch to find template IRes pairs
    all_atoms_B = [atom for res in residues_B for atom in res.get_atoms() if atom.name == 'CA']
    if not all_atoms_B:
        logging.warning(f"No CA atoms found in chain B residues for {pdb_id}")
        return pair_data
    ns = NeighborSearch(all_atoms_B)

    template_interacting_pairs = []
    for res_a in residues_A:
        if 'CA' in res_a:
            ca_a = res_a['CA']
            # find all CA atoms in chain B <=6.05A of this CA in chain A
            nearby_b_atoms = ns.search(ca_a.coord, CA_DISTANCE_THRESHOLD, 'A')
            for atom_b in nearby_b_atoms:
                res_b = atom_b.get_parent()
                template_res_a_id = f"{res_a.id[1]}{res_a.id[2].strip()}"
                template_res_b_id = f"{res_b.id[1]}{res_b.id[2].strip()}"
                template_interacting_pairs.append((template_res_a_id, template_res_b_id))
    n_template_pairs = len(template_interacting_pairs)
    if n_template_pairs == 0: 
        logging.warning(f"No interacting residue pairs found for {pair_key} in template {pdb_id}.")
        return pair_data
    logging.info(f"Found {n_template_pairs} interacting residue pairs in template {pdb_id} for {pair_key}.")
    
    """ With template residue pairs identified, we must now map them onto the pertinent query complex.
    """

    # A. Mapping of template residue pairs to query residue pairs
    mapped_query_pairs = []
    for template_res_a, template_res_b in template_interacting_pairs:
        query_pos_a = map1.get(template_res_a)
        query_pos_b = map2.get(template_res_b)
        if query_pos_a is not None and query_pos_b is not None:
            mapped_query_pairs.append((query_pos_a, query_pos_b))
    if not mapped_query_pairs:
        logging.warning(f"No template pairs could be mapped for {pair_key}.")
        return pair_data
    
    # B. Extract unique active residues for each protein; 0-based indexing
    # active_residues_a = sorted(list(set([pair[0] for pair in mapped_query_pairs])))
    # active_residues_b = sorted(list(set([pair[1] for pair in mapped_query_pairs])))

    # C. Import Query Structures (AFDB Generated Models)
    query_specific_file = f"fold_{p1_id.lower()}_{p2_id.lower()}_model_0.cif"
    logging.info(f"Recursively searching for {query_specific_file}...")
    
    search_pattern = os.path.join(AF3_MODELS_DIR, '**', query_specific_file)
    cif_files = glob.glob(search_pattern, recursive = True)
    if not cif_files:
        logging.warning(f"File {query_specific_file} not found in {AF3_MODELS_DIR}.")
        return pair_data
    logging.info(f"Found evidence of AF3 Model {query_specific_file}.")
    
    # D. Load and process the top query structure.
    # Note: This assumes that the AF3 model are stored in a specific directory structure.
    # cif_files.sort()
    top_model_file = cif_files[0]
    # --- FIX: Use os.path to split the path string ---
    parent_dir = os.path.dirname(top_model_file)
    filename = os.path.basename(top_model_file)
    model_stem, _ = os.path.splitext(filename) # Splits 'model.cif' into ('model', '.cif')
    
    try:
        query_structure = get_structure_from_file(model_stem, parent_dir)
        if not query_structure:
            logging.warning(f"Failed to load query structure {filename}")
            return pair_data
            
        if 'A' not in query_structure or 'B' not in query_structure:
            logging.warning(f"Missing chains A/B in query structure")
            return pair_data
        
        # Extract constituent chains A and B and their residues
        residues_list_A, residues_list_B = list(query_structure['A'].get_residues()), list(query_structure['B'].get_residues())
    
        if len(residues_list_A) == 0 or len(residues_list_B) == 0:
            logging.warning(f"No residues found in chains A/B for query structure")
            return pair_data

        inter_ca_distances = []
        invalid_pairs = 0

        for i1, i2 in mapped_query_pairs:
            if i1 < len(residues_list_A) and i2 < len(residues_list_B):
                res1, res2 = residues_list_A[i1], residues_list_B[i2]
                if 'CA' in res1 and 'CA' in res2:
                    distance = np.linalg.norm(res1['CA'].coord - res2['CA'].coord)
                    inter_ca_distances.append(distance)
                else: invalid_pairs += 1
            else: invalid_pairs += 1
        
        if invalid_pairs > 0:
            logging.debug(f"Invalid residue pairs found in model: {invalid_pairs} pairs skipped.")
        
        # Tally # of pairs conserved
        conserved_amount = sum(1 for d in inter_ca_distances if d <= CA_DISTANCE_THRESHOLD)
        logging.debug(f"Model has {conserved_amount}/{len(inter_ca_distances)} conserved pairs.")
    
    except Exception as e:
        logging.error(f"Error processing model for {pair_key}: {e}")
        return pair_data
    
    # E. compute SIZE and COV metrics across the models
    # - SIZE is defined as the number of conserved residue pairs
    # - COV is defined as the fraction of conserved residue pairs
    #   relative to the total number of template residue pairs
    #   The initial count of template residue pairs is n_template_pairs, held constant.
    pair_data['SIZE'] = conserved_amount
    pair_data['COV'] = conserved_amount / n_template_pairs if n_template_pairs > 0 else 0.0
    logging.info(f"Final metrics for {pair_key}: SIZE = {pair_data['SIZE']}, COV = {pair_data['COV']:.2f}")

    return pair_data

def main():
    """Main function to drive the pipeline"""

    parser = argparse.ArgumentParser(description = "Process a slice of a dataset using Foldseek results.")
    parser.add_argument('--ires_template_file', default = IRES_TEMPLATE_FILE, help = ".txt file containing 3D interfacial structures for TEMPLATE pairs.")
    # parser.add_argument('--ires_query_file', default = IRES_QUERY_FILE, help = ".txt file containing 3D interfacial structures for QUERY pairs.")
    # parser.add_argument('--fasta_dir', default = FASTA_FILE_PATH, help = "path containing .FASTA files for query proteins.")
    parser.add_argument('--tsv_dir', default = FOLDSEEK_TSV_DIR, help = "path containing .tsv formatted Foldseek results (homologs).")
    parser.add_argument('--pdb_dir', default = LOCAL_PDB_DIR, help = "local path storing 3D template PDB structures.")
    parser.add_argument('--datafile_path', default = DATAFILE_PATH, help = "data file containing experimental protein pairs and all features.")
    parser.add_argument('--scratch_path', default = BASE_SCRATCH_PATH, help = "scratch path directory.")
    parser.add_argument('--candidate_templates_dir', default = CANDIDATE_TEMPLATES_PATH, help = "output path to save candidate templates in .jsonl format.")
    parser.add_argument('--json_output_path', default = BASE_OUTPUT_JSONL_PATH, help = "specify output path of .jsonl files containing structural features.")
    parser.add_argument('--task-id', default = 0, type = int)
    parser.add_argument('--total-tasks', default = 1, type = int)
    args = parser.parse_args()

    # Load shared resources
    # Read and partition the pair list
    all_pairs = [tuple(ppi_string.split(':')) for ppi_string in DATA['ppi'].dropna() if len(ppi_string.split(':')) == 2]
    assert len(all_pairs) == len(DATA), "Error: some PPI pairs were not successfully accounted for."
    
    num_all_pairs = len(all_pairs)
    pairs_per_task = (num_all_pairs + args.total_tasks - 1) // args.total_tasks
    start_index = args.task_id * pairs_per_task
    end_index = min(start_index + pairs_per_task, num_all_pairs)
    my_slice_of_pairs = all_pairs[start_index:end_index]
    
    logging.info(f"Task {args.task_id}: Processing {len(my_slice_of_pairs)} pairs from index {start_index} to {end_index - 1}")

    # output_dir = os.path.join(args.json_output_path, args.dataset)
    os.makedirs(args.json_output_path, exist_ok = True)
    
    # Process each pair in the assigned slice
    for index, (p1, p2) in enumerate(my_slice_of_pairs):
        logging.info(f"Processing pair {index + 1}: {p1}_{p2}")
        pair_data = process_single_pair(p1, p2)
        with open(os.path.join(args.json_output_path, f"{p1}_{p2}.json"), "w") as f:
            json.dump(pair_data, f, indent = 4)
        print("-"*60)
        # logging.info(f"Processed pair {pair_idx + 1}/{len(my_slice_of_pairs)}: {p1} - {p2}")
    logging.info(f"Task {args.task_id} completed. Processed {len(my_slice_of_pairs)} pairs.")

if __name__ == "__main__":
    main()
