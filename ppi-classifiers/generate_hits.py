""" 6/11/25 code by Juheon Chu, to run FASTA for generating hits (homologs) from individual protein query sequences.
---    Annotations and pertinent workflow modifications made by Andrew.
---    To run Foldseek directly on shell script, paste the following command:
---    docker run --rm -v  C:/Users/hychu/OneDrive/Desktop/Summer25/:/data quay.io/biocontainers/foldseek:10.941cd33--h5021889_1 foldseek

    Juheon Chu (jc3668) and Andrew Chung (hc893)
"""

import os
import numpy as np
import subprocess
import argparse
import logging
import tempfile
import pathlib
import shutil
import tqdm

# Configure logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)

# === Path Configuration ===
DEFAULT_USER_BASE_DIR = "C:/Users/hychu/OneDrive/Desktop/Summer25"
# REPOSITORY_DIR = "github/ppi-classifiers"

# === Default Directories (Host Paths); Overridable ===
DEFAULT_FASTA_INPUT_DIR = os.path.join(DEFAULT_USER_BASE_DIR, "output/fasta")
DEFAULT_TSV_OUTPUT_DIR = os.path.join(DEFAULT_USER_BASE_DIR, "output/tsv")
DEFAULT_FOLDSEEK_INTERNAL_TMP_DIR = os.path.join(DEFAULT_USER_BASE_DIR, "transient/foldseek_internal_temp")
DEFAULT_PDB_DB_FOLDSEEK = os.path.join(DEFAULT_USER_BASE_DIR, "databases/pdb/pdb")
DEFAULT_PROSTT5_WEIGHTS = os.path.join(DEFAULT_USER_BASE_DIR, "databases/weights/prostt5-f16.gguf")

# determine host-to-container mount; Build Docker Image for Foldseek
DOCKER_IMAGE = "quay.io/biocontainers/foldseek:10.941cd33--h5021889_1"
HOST_CWD = pathlib.Path(DEFAULT_USER_BASE_DIR).resolve()
CONTAINER_MOUNT_POINT = "/data"
CONTAINER_MOUNT = f"{HOST_CWD}:{CONTAINER_MOUNT_POINT}"

def to_container_path(host_path) -> str:
    # Convert a host absolute path to Docker container equivalent
    p = str(host_path).replace('\\', '/')
    host_prefix = str(DEFAULT_USER_BASE_DIR).replace('\\', '/')
    return p.replace(host_prefix, CONTAINER_MOUNT_POINT)

def load_fasta_sequences(fasta_file_path) -> dict:
    """Load protein sequences from the FASTA-like files # (ID sequence per line).
    Assume each FASTA file has a header (UniProt ID + Description) and a sequence line.
    """
    # key: UniProt ID; Value: lettered protein sequences (variable length)
    assert os.path.isdir(fasta_file_path), f"FASTA path does not exist: {fasta_file_path}"
    fasta_dict = {}
    for filename in tqdm.tqdm(os.listdir(fasta_file_path)):
        if filename.endswith('.fasta'):
            monomer_id = os.path.splitext(filename)[0]
            filepath = os.path.join(fasta_file_path, filename)
            with open(filepath, "r") as infile:
                infile.readline() # skip header
                sequence = infile.readline().strip().rstrip('\n')
                fasta_dict[monomer_id] = sequence
    logging.info(f"Successfully loaded {len(fasta_dict)} sequences from {fasta_file_path}.")
    return fasta_dict

def run_foldseek_for_single_protein(
    # sequence and foldseek_path are redundant
    protein_id: str,
    fasta_input_path: str,
    tsv_output_path: str,
    pdb_db_path: str,
    prostt5_host: str,
    foldseek_internal_tmp_dir: str
):
    """
    Runs Foldseek for a single protein.
    Saves query FASTA and detailed TSV output. Uses assert for validation.
    """

    # since I already imported and saved query FASTA files in /fasta,
    # that particular step may not be necessary here.
    #os.makedirs(os.path.dirname(fasta_output_path), exist_ok=True)
    #with open(fasta_output_path, "w") as f:
    #    f.write(f">{protein_id}\n{sequence}\n")
    #logging.info(f"Query FASTA for {protein_id} saved to {fasta_output_path}")

    # create/ensure output path
    os.makedirs(os.path.dirname(fasta_input_path), exist_ok = True)
    os.makedirs(os.path.dirname(tsv_output_path), exist_ok = True)
    os.makedirs(foldseek_internal_tmp_dir, exist_ok = True)

    # unique temp directory for protein
    host_tmp = tempfile.mkdtemp(dir = foldseek_internal_tmp_dir, prefix = f"fs_tmp_{protein_id}_")
    db_host = os.path.join(host_tmp, f"{protein_id}_querydb")
    res_host = os.path.join(host_tmp, f"{protein_id}_search_result")

    # convert host path to Docker-recognizable paths
    fasta_docker = to_container_path(fasta_input_path)
    tsv_file_docker = to_container_path(tsv_output_path)
    pdb_db_docker = to_container_path(pdb_db_path)
    prostt5_docker = to_container_path(prostt5_host)
    tmp_dir_docker = to_container_path(foldseek_internal_tmp_dir)
    db_docker = to_container_path(db_host)
    res_docker = to_container_path(res_host)

    docker_base = [
        "docker", "run", "--rm", "-v", CONTAINER_MOUNT, DOCKER_IMAGE, "foldseek"
    ]

    # 1. createdb: Query Database (with the single protein FASTA)
    cmd_cdb = docker_base + ["createdb", "--db-extraction-mode", "1", fasta_docker, db_docker]
    if prostt5_host and os.path.exists(os.path.dirname(prostt5_host)):
        cmd_cdb.extend(["--prostt5-model", prostt5_docker])
    logging.debug(f"Running Foldseek createdb for {protein_id}: {' '.join(cmd_cdb)}")
    cp = subprocess.run(cmd_cdb, capture_output = True, text = True, check = False)
    assert cp.returncode == 0, \
        f"Foldseek createdb failed for {protein_id}.\nExit Code: {cp.returncode}\nStdout: {cp.stdout}\nStderr: {cp.stderr}"

    # 2. Foldseek Query Search
    cmd_search = docker_base + [
        "search", db_docker, pdb_db_docker, res_docker, tmp_dir_docker,
        "-e", "0.001", "--sort-by-structure-bits", "0", "-a"
    ]
    logging.debug(f"Running Foldseek search for {protein_id}: {' '.join(cmd_search)}")
    sp = subprocess.run(cmd_search, capture_output = True, text = True, check = False)
    assert sp.returncode == 0, \
        f"Foldseek search failed for {protein_id}.\nExit Code: {sp.returncode}\nStdout: {sp.stdout}\nStderr: {sp.stderr}"

    # If search returned 0, Foldseek expects the result file to exist for convertalis,
    # even if it's empty (no hits). We'll let convertalis handle an empty result if that's Foldseek's behavior.
    # An explicit check for tmp_result_path's existence might be redundant if convertalis itself fails clearly.

    # 3. Convert alignments -- Convertalis
    convertalis_format = "query,target,pident,fident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qaln,taln,qseq,tseq"
    convertalis_cmd_parts = docker_base + [
        "convertalis", db_docker, pdb_db_docker,
        res_docker, tsv_file_docker,
        "--format-output", convertalis_format
    ]
    logging.debug(f"Running Foldseek convertalis for {protein_id}: {' '.join(convertalis_cmd_parts)}")
    xp = subprocess.run(convertalis_cmd_parts, capture_output = True, text = True, check = False)
    assert xp.returncode == 0, \
        f"Foldseek convertalis failed for {protein_id}.\nExit Code: {xp.returncode}\nStdout: {xp.stdout}\nStderr: {xp.stderr}"

    logging.info(f"Foldseek TSV output for {protein_id} saved to {tsv_output_path}")

    # Cleanup temporary directory
    if os.path.exists(tmp_dir_docker):
        shutil.rmtree(tmp_dir_docker)
    logging.info(f"Finished {protein_id}: TSV saved to {tsv_output_path}")
    return True # Return True on success (all assertions passed)

def main():
    parser = argparse.ArgumentParser(description = "Run Foldseek for unique individual proteins from specified dataset files.")

    # === Docker container paths converted from host defaults ===
    # root = pathlib.Path().absolute().parent.parent
    # DOCKER_FASTA_INPUT_DIR = to_container_path(DEFAULT_FASTA_INPUT_DIR)
    # DOCKER_TSV_OUTPUT_DIR = to_container_path(DEFAULT_TSV_OUTPUT_DIR)

    """Slight modification to workflow: since I have already identified unique protein IDs from the data, I will prioritize 
    generating hits (homologs) on the already identified unique monomers first and perform train-test splitting later."""

    # === generate arguments to parser ===
    parser.add_argument('--fasta_input_dir', default = DEFAULT_FASTA_INPUT_DIR, help = f"Directory of saved query FASTA files. Default: {DEFAULT_FASTA_INPUT_DIR}")
    parser.add_argument('--tsv_output_dir', default = DEFAULT_TSV_OUTPUT_DIR, help = f"Directory to save Foldseek TSV results. Default: {DEFAULT_TSV_OUTPUT_DIR}")
    parser.add_argument('--foldseek_internal_tmp_dir', default = DEFAULT_FOLDSEEK_INTERNAL_TMP_DIR, help = f"Base temporary directory for Foldseek's own intermediate files. Default: {DEFAULT_FOLDSEEK_INTERNAL_TMP_DIR}")
    parser.add_argument('--pdb_db_foldseek', default = DEFAULT_PDB_DB_FOLDSEEK, help = f"Path to Foldseek target PDB database. Default: {DEFAULT_PDB_DB_FOLDSEEK}")
    parser.add_argument('--prostt5_weights', default = DEFAULT_PROSTT5_WEIGHTS, help = f"Path to ProstT5 model weights. Default: '{DEFAULT_PROSTT5_WEIGHTS}'") # switch on/off as needed
    parser.add_argument('--force_rerun', action = 'store_true') 
    parser.add_argument('--total_batches', type = int, default = 1, help = "Total number of batches the protein list is divided into.")
    parser.add_argument('--current_batch_index', type = int, default = 0, help = "Index of the current batch to process (0-indexed).")
    args = parser.parse_args()

    logging.info("--- Starting Unique Protein Foldseek Processing (with asserts) ---")
    for arg, value in vars(args).items():
        logging.info(f"Argument --{arg}: {value}")

    # === Load (FASTA) queries, assert ===
    all_sequences = load_fasta_sequences(args.fasta_input_dir) # Will assert if file not found or malformed
    all_unique_protein_ids_sorted = np.sort(np.array(list(all_sequences.keys()))) # list of unique IDs, sorted
    num_total_proteins = len(all_unique_protein_ids_sorted)
    assert num_total_proteins > 0, "No unique protein IDs were found. Exiting."
    logging.info(f"Total unique proteins identified overall: {num_total_proteins}")

    # === Validate batch arguments ===
    assert args.total_batches > 0, "--total_batches must be greater than 0."
    assert 0 <= args.current_batch_index < args.total_batches, \
        f"Invalid --current_batch_index {args.current_batch_index}. Must be between 0 and {args.total_batches - 1}."
    batch_size = (num_total_proteins + args.total_batches - 1) // args.total_batches
    start_index = args.current_batch_index * batch_size
    end_index = min((args.current_batch_index + 1) * batch_size, num_total_proteins)
    proteins_to_process_this_batch = all_unique_protein_ids_sorted[start_index:end_index]
    logging.info(f"Calculated batch size: {batch_size}")
    logging.info(f"This batch (index {args.current_batch_index}) will process {len(proteins_to_process_this_batch)} proteins (from overall index {start_index} to {end_index -1}).")

    # === Ensure host output directories ===
    # os.makedirs(args.fasta_input_dir, exist_ok=True)
    os.makedirs(args.tsv_output_dir, exist_ok=True)
    os.makedirs(args.foldseek_internal_tmp_dir, exist_ok=True)

    processed = 0; successful = 0; skipped = 0

    # === Loop Processing of Foldseek Workflow for Single Proteins ===
    for protein_id in proteins_to_process_this_batch:
        processed +=1
        logging.info(f"--- Processing protein {processed}/{len(proteins_to_process_this_batch)}: {protein_id} ---")

        expected_fasta_path = os.path.join(args.fasta_input_dir, f"{protein_id}.fasta")
        expected_tsv_path = os.path.join(args.tsv_output_dir, f"{protein_id}.tsv")

        if not args.force_rerun and \
           os.path.exists(expected_fasta_path) and \
           os.path.exists(expected_tsv_path):
            logging.info(f"Output FASTA and non-empty TSV for {protein_id} already exist. Skipping.")
            skipped +=1
            successful +=1
            continue
        sequence = all_sequences.get(protein_id) # extract corresponding sequence
        assert sequence is not None, f"Sequence for protein ID {protein_id} not found in main FASTA file."

        # run_foldseek_for_single_protein will now assert on failure
        logging.info(f"--- Running Foldseek for Protein {protein_id} ---")
        run_foldseek_for_single_protein(
            protein_id = protein_id,
            fasta_input_path = expected_fasta_path,
            tsv_output_path = expected_tsv_path,
            pdb_db_path = args.pdb_db_foldseek,
            prostt5_host = args.prostt5_weights,
            foldseek_internal_tmp_dir = args.foldseek_internal_tmp_dir
        )

        successful += 1

    # Use print for final summary of this batch job to distinguish it in logs
    print("--- Batch Processing Summary for this job ---")
    print(f"Batch Index Processed: {args.current_batch_index}")
    print(f"Proteins assigned to this batch: {len(proteins_to_process_this_batch)}")
    print(f"Successfully processed/found existing in this batch: {successful}")
    print(f"Skipped in this batch (outputs existed): {skipped}")
    print(f"Query FASTA files for this batch saved in: {args.fasta_output_dir}")
    print(f"TSV result files for this batch saved in: {args.tsv_output_dir}")
    print("Processing complete for this batch job.")

if __name__ == "__main__":
    main()
