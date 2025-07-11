import os
import requests
import logging
import pandas as pd

# --- Configuration ---
TSV_DIR = "/workdir/jc3668/foldseek_search_test/scratch/tsv/"
DATASET_FILES = [
    "/workdir/jc3668/blast_search_test/data/train/train_set.txt",
    "/workdir/jc3668/blast_search_test/data/test/test_set.txt",
    "/workdir/jc3668/blast_search_test/data/validation/validation_set.txt"
]
# IMPORTANT: This is where the files will be saved.
# Create this directory if it doesn't exist.
CIF_OUTPUT_DIR = "/local/storage/jc3668/data/cif"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_all_pdb_ids_from_foldseek(unique_protein_ids):
    """Parses all TSV files to find every unique PDB ID mentioned."""
    all_pdb_ids = set()
    logging.info(f"Scanning TSV files for {len(unique_protein_ids)} proteins to find all needed PDBs...")
    for i, protein_id in enumerate(sorted(list(unique_protein_ids))):
        if (i+1) % 1000 == 0: logging.info(f"Scanned {i+1}/{len(unique_protein_ids)} files...")
        tsv_file_path = os.path.join(TSV_DIR, f"{protein_id}.tsv")
        if not os.path.exists(tsv_file_path): continue
        try:
            df = pd.read_csv(tsv_file_path, sep='\t', header=None, usecols=[1])
            for target in df[1]:
                pdb_id = target.split('-')[0].split('_')[0]
                all_pdb_ids.add(pdb_id)
        except Exception:
            continue
    return all_pdb_ids

def get_unique_protein_ids(file_paths_list):
    """Reads protein pairs from multiple files and returns a set of unique protein IDs."""
    unique_ids = set()
    for file_path in file_paths_list:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2: unique_ids.update(parts[:2])
    return unique_ids

def download_cif_file(pdb_id, output_dir):
    """Downloads a single .cif.gz file for a given PDB ID."""
    pdb_id_lower = pdb_id.lower()
    middle_chars = pdb_id_lower[1:3]
    final_dir = os.path.join(output_dir, middle_chars)
    os.makedirs(final_dir, exist_ok=True)

    output_path = os.path.join(final_dir, f"{pdb_id_lower}.cif.gz")
    if os.path.exists(output_path):
        return # Skip if it already exists

    url = f"https://files.rcsb.org/download/{pdb_id_lower}.cif.gz"
    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(res.content)
        logging.info(f"Successfully downloaded {pdb_id}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {pdb_id}: {e}")

if __name__ == "__main__":
    protein_ids = get_unique_protein_ids(DATASET_FILES)
    pdb_ids_to_download = get_all_pdb_ids_from_foldseek(protein_ids)
    logging.info(f"Found {len(pdb_ids_to_download)} unique PDB IDs to download.")

    for i, pdb_id in enumerate(sorted(list(pdb_ids_to_download))):
        if (i+1) % 100 == 0: logging.info(f"Downloading PDB {i+1}/{len(pdb_ids_to_download)}...")
        download_cif_file(pdb_id, CIF_OUTPUT_DIR)

    logging.info("Download process complete.")
