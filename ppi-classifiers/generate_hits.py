import os
import numpy as np
import subprocess
import argparse
import logging
import tempfile
import shutil
import tqdm

# 6/11/25 code by Juheon, to generate hits from individual protein query sequences.
# annotations and pertinent workflow modifications made by Andrew.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Outputs to console
    ]
)

# --- Default Constants (can be overridden by command-line arguments) ---
# may not need this, if running shell script locally
# DEFAULT_USER_BASE_DIR = "/home/jc3668/projects" 
DEFAULT_USER_BASE_DIR = "C:/Users/hychu/OneDrive/Desktop/Summer 2025"
REPOSITORY_DIR="github/ppi-classifiers"

# DEFAULT_FOLDSEEK_PATH = os.path.join(DEFAULT_USER_BASE_DIR, "foldseek")
DEFAULT_PDB_DB_FOLDSEEK = os.path.join(DEFAULT_USER_BASE_DIR, "pdb")
DEFAULT_PROSTT5_WEIGHTS = os.path.join(DEFAULT_USER_BASE_DIR, "prostt5_out")

"""DEFAULT_FASTA_FILE_INPUT = "/home/jc3668/projects/foldseek_search_test/data/uniprot_seqs_both_sets.txt"
DEFAULT_UNIPROT_FILE_INPUT = f"{DEFAULT_USER_BASE_DIR}/{REPOSITORY_DIR}/monomers.txt"
"""
DEFAULT_FASTA_INPUT_DIR = "/home/jc3668/projects/foldseek_search_test/scratch/fasta"
DEFAULT_TSV_OUTPUT_DIR = "/home/jc3668/projects/foldseek_search_test/scratch/tsv"
DEFAULT_FOLDSEEK_INTERNAL_TMP_DIR = os.path.join(DEFAULT_USER_BASE_DIR, REPOSITORY_DIR, 'transient', 'foldseek_internal_tmp')

# data sets
"""
DEFAULT_TRAIN_SET_FILE = "/home/jc3668/projects/foldseek_search_test/data/train/train_set.txt"
DEFAULT_TEST_SET_FILE = "/home/jc3668/projects/foldseek_search_test/data/test/test_set.txt"
DEFAULT_VALIDATION_SET_FILE = "/home/jc3668/projects/foldseek_search_test/data/validation/validation_set.txt"
"""

def load_fasta_sequences(fasta_file_path):
    """Load protein sequences from the FASTA-like files #(ID sequence per line)."""
    # key: UniProt ID; Value: lettered protein sequences (variable length)
    assert os.path.exists(fasta_file_path), f"FASTA path does not exist: {fasta_file_path}"
    fasta_dict = {}
    for filename in tqdm.tqdm(os.listdir(fasta_file_path)):
        if filename.endswith('.fasta'):
            monomer_id = os.path.splitext(filename)[0]
            filepath = os.path.join(fasta_file_path, filename)
            with open(filepath, "r") as infile:
                infile.readline() # skip header
                sequence = infile.readline().rstrip('\n')
                fasta_dict[monomer_id] = sequence
    logging.info(f"Successfully loaded {len(fasta_dict)} sequences from {fasta_file_path}")
    return fasta_dict

def run_foldseek_for_single_protein(
    protein_id,
    sequence,
    # foldseek_executable_path,
    target_pdb_db_path,
    prostt5_model_path,
    fasta_input_path,
    tsv_output_path,
    foldseek_internal_tmp_dir
):
    """
    Runs Foldseek for a single protein.
    Saves query FASTA and detailed TSV output. Uses assert for validation.
    """

    # create output path
    #os.makedirs(os.path.dirname(fasta_output_path), exist_ok=True)
    #os.makedirs(os.path.dirname(tsv_output_path), exist_ok=True)

    # temp directory
    os.makedirs(foldseek_internal_tmp_dir, exist_ok=True)

    # since I already imported and saved query FASTA files in /fasta,
    # that particular step may not be necessary here.
    """
    with open(fasta_output_path, "w") as f:
        f.write(f">{protein_id}\n{sequence}\n")
    logging.info(f"Query FASTA for {protein_id} saved to {fasta_output_path}")
    """

    current_run_tmp_dir = tempfile.mkdtemp(dir=foldseek_internal_tmp_dir, prefix=f"fs_tmp_{protein_id}_")
    tmp_db_path = os.path.join(current_run_tmp_dir, f"{protein_id}_querydb")
    tmp_result_path = os.path.join(current_run_tmp_dir, f"{protein_id}_search_result")

    # 1. Create query database
    # Note: I'm running Foldseek as a Docker image.
    docker_elements = ["docker", "run", "--rm", "-v", "${PWD}:/data", "foldseek-binary"]
    createdb_cmd_parts = docker_elements + [
        "createdb", 
        fasta_input_path, 
        tmp_db_path
    ]
    if prostt5_model_path and os.path.exists(os.path.dirname(prostt5_model_path)):
        createdb_cmd_parts.extend(["--prostt5-model", prostt5_model_path])
    logging.debug(f"Running Foldseek createdb for {protein_id}: {' '.join(createdb_cmd_parts)}")
    db_process = subprocess.run(createdb_cmd_parts, capture_output=True, text=True, check=False)
    assert db_process.returncode == 0, \
        f"Foldseek createdb failed for {protein_id}.\nExit Code: {db_process.returncode}\nStdout: {db_process.stdout}\nStderr: {db_process.stderr}"

    # 2. Search
    search_cmd_parts = docker_elements + [
        "search", tmp_db_path, target_pdb_db_path,
        tmp_result_path, current_run_tmp_dir,
        "-e", "0.001", # e-value threshold
        "--sort-by-structure-bits", "0",
        "-a"
    ]
    logging.debug(f"Running Foldseek search for {protein_id}: {' '.join(search_cmd_parts)}")
    search_process = subprocess.run(search_cmd_parts, capture_output=True, text=True, check=False)
    assert search_process.returncode == 0, \
        f"Foldseek search failed for {protein_id}.\nExit Code: {search_process.returncode}\nStdout: {search_process.stdout}\nStderr: {search_process.stderr}"

    # If search returned 0, Foldseek expects the result file to exist for convertalis,
    # even if it's empty (no hits). We'll let convertalis handle an empty result if that's Foldseek's behavior.
    # An explicit check for tmp_result_path's existence might be redundant if convertalis itself fails clearly.

    # 3. Convert alignments
    convertalis_format = "query,target,pident,fident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qaln,taln,qseq,tseq"
    convertalis_cmd_parts = docker_elements + [
        "convertalis", tmp_db_path, target_pdb_db_path,
        tmp_result_path, tsv_output_path,
        "--format-output", convertalis_format
    ]
    logging.debug(f"Running Foldseek convertalis for {protein_id}: {' '.join(convertalis_cmd_parts)}")
    convert_process = subprocess.run(convertalis_cmd_parts, capture_output=True, text=True, check=False)
    assert convert_process.returncode == 0, \
        f"Foldseek convertalis failed for {protein_id}.\nExit Code: {convert_process.returncode}\nStdout: {convert_process.stdout}\nStderr: {convert_process.stderr}"

    logging.info(f"Foldseek TSV output for {protein_id} saved to {tsv_output_path}")

    # Cleanup temporary directory
    if os.path.exists(current_run_tmp_dir):
        shutil.rmtree(current_run_tmp_dir)
    return True # Return True on success (all assertions passed)

"""
def get_unique_ids_from_files(file_paths_list):
    # Reads protein pairs from multiple files and returns a set of unique protein IDs.
    unique_ids = set()
    for file_path in file_paths_list:
        assert os.path.exists(file_path), f"Dataset file not found: {file_path}"
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1): # humans count from 1, not 0
                stripped_line = line.strip()
                if not stripped_line: # Skip empty lines
                    continue
                parts = stripped_line.split("\t")
                assert len(parts) >= 2, \
                    f"Malformed line {line_num} in {os.path.basename(file_path)}: '{stripped_line}'. Expected tab-separated pair."
                unique_ids.add(parts[0])
                unique_ids.add(parts[1])
        logging.info(f"Finished reading IDs from {file_path}.")
    return unique_ids
"""

def main():
    parser = argparse.ArgumentParser(description="Run Foldseek for unique individual proteins from specified dataset files.")
    """Slight modification to workflow: since I have already identified unique protein IDs from the data, I will prioritize 
    generating hits (homologs) on the already identified unique monomers first and perform train-test splitting later."""

    # parser.add_argument('--train_set_file', default=DEFAULT_TRAIN_SET_FILE, help=f"Path to the training set file. Default: {DEFAULT_TRAIN_SET_FILE}")
    # parser.add_argument('--test_set_file', default=DEFAULT_TEST_SET_FILE, help=f"Path to the test set file. Default: {DEFAULT_TEST_SET_FILE}")
    # parser.add_argument('--validation_set_file', default=DEFAULT_VALIDATION_SET_FILE, help=f"Path to the validation set file. Default: {DEFAULT_VALIDATION_SET_FILE}")
    # parser.add_argument('--uniprot_id_file', default=DEFAULT_UNIPROT_FILE_INPUT, help=f"Path to the main UniProt ID file. Default: {DEFAULT_UNIPROT_FILE_INPUT}")
    # parser.add_argument('--fasta_file', default=DEFAULT_FASTA_FILE_INPUT, help=f"Path to the main FASTA file. Default: {DEFAULT_FASTA_FILE_INPUT}")
    parser.add_argument('--fasta_input_dir', default=DEFAULT_FASTA_INPUT_DIR, help=f"Directory of saved query FASTA files. Default: {DEFAULT_FASTA_INPUT_DIR}")
    parser.add_argument('--tsv_output_dir', default=DEFAULT_TSV_OUTPUT_DIR, help=f"Directory to save Foldseek TSV results. Default: {DEFAULT_TSV_OUTPUT_DIR}")
    parser.add_argument('--foldseek_internal_tmp_dir', default=DEFAULT_FOLDSEEK_INTERNAL_TMP_DIR, help=f"Base temporary directory for Foldseek's own intermediate files. Default: {DEFAULT_FOLDSEEK_INTERNAL_TMP_DIR}")
    # parser.add_argument('--foldseek_path', default=DEFAULT_FOLDSEEK_PATH, help=f"Path to Foldseek executable. Default: {DEFAULT_FOLDSEEK_PATH}")
    parser.add_argument('--pdb_db_foldseek', default=DEFAULT_PDB_DB_FOLDSEEK, help=f"Path to Foldseek target PDB database. Default: {DEFAULT_PDB_DB_FOLDSEEK}")
    parser.add_argument('--prostt5_weights', default=DEFAULT_PROSTT5_WEIGHTS, help=f"Path to ProstT5 model weights. Default: '{DEFAULT_PROSTT5_WEIGHTS}'")
    parser.add_argument('--force_rerun', action='store_true') 
    parser.add_argument('--total_batches', type=int, default=1, help="Total number of batches the protein list is divided into.")
    parser.add_argument('--current_batch_index', type=int, default=0, help="Index of the current batch to process (0-indexed).")

    args = parser.parse_args()

    logging.info("--- Starting Unique Protein Foldseek Processing (with asserts) ---")
    # Log all arguments
    for arg, value in vars(args).items():
        logging.info(f"Argument --{arg}: {value}")

    all_sequences = load_fasta_sequences(args.fasta_input_dir) # Will assert if file not found or malformed
    all_unique_protein_ids_sorted = np.sort(np.array(list(all_sequences.keys()))) # list of unique IDs, sorted

    """
    dataset_files = [args.train_set_file, args.test_set_file, args.validation_set_file]
    #unique_protein_ids = get_unique_ids_from_files(dataset_files) # Will assert if files not found or malformed
    all_unique_protein_ids_sorted = sorted(list(get_unique_ids_from_files(dataset_files)))
    """

    num_total_proteins = len(all_unique_protein_ids_sorted)
    assert num_total_proteins > 0, "No unique protein IDs were found. Exiting."
    logging.info(f"Total unique proteins identified overall: {num_total_proteins}")

    # Validate batch arguments
    assert args.total_batches > 0, "--total_batches must be greater than 0."
    assert 0 <= args.current_batch_index < args.total_batches, \
        f"Invalid --current_batch_index {args.current_batch_index}. Must be between 0 and {args.total_batches - 1}."

    batch_size = (num_total_proteins + args.total_batches - 1) // args.total_batches
    start_index = args.current_batch_index * batch_size
    end_index = min((args.current_batch_index + 1) * batch_size, num_total_proteins)

    proteins_to_process_this_batch = all_unique_protein_ids_sorted[start_index:end_index]

    logging.info(f"Calculated batch size: {batch_size}")
    logging.info(f"This batch (index {args.current_batch_index}) will process {len(proteins_to_process_this_batch)} proteins (from overall index {start_index} to {end_index -1}).")

    #os.makedirs(args.fasta_input_dir, exist_ok=True)
    os.makedirs(args.tsv_output_dir, exist_ok=True)
    os.makedirs(args.foldseek_internal_tmp_dir, exist_ok=True)

    processed_in_this_batch = 0
    successful_in_this_batch = 0
    skipped_in_this_batch = 0

    for protein_id in proteins_to_process_this_batch:
        processed_in_this_batch +=1
        logging.info(f"--- Processing protein {processed_in_this_batch}/{len(proteins_to_process_this_batch)}: {protein_id} ---")

        expected_fasta_path = os.path.join(args.fasta_input_dir, f"{protein_id}.fasta")
        expected_tsv_path = os.path.join(args.tsv_output_dir, f"{protein_id}.tsv")

        if not args.force_rerun and \
           os.path.exists(expected_fasta_path) and \
           os.path.exists(expected_tsv_path): # force_rerun = True, so I can ignore this
            logging.info(f"Output FASTA and non-empty TSV for {protein_id} already exist. Skipping.")
            skipped_in_this_batch +=1
            successful_in_this_batch +=1
            continue
        sequence = all_sequences.get(protein_id) # extract corresponding sequence
        assert sequence is not None, f"Sequence for protein ID {protein_id} not found in main FASTA file. Check {args.fasta_file}."

        # run_foldseek_for_single_protein will now assert on failure
        run_foldseek_for_single_protein(
            protein_id=protein_id,
            sequence=sequence,
            # foldseek_executable_path=args.foldseek_path,
            target_pdb_db_path=args.pdb_db_foldseek,
            prostt5_model_path=args.prostt5_weights,
            fasta_input_path=expected_fasta_path,
            tsv_output_path=expected_tsv_path,
            foldseek_internal_tmp_dir=args.foldseek_internal_tmp_dir
        )

        successful_in_this_batch += 1

    # Use print for final summary of this batch job to distinguish it in logs
    print("--- Batch Processing Summary for this job ---")
    print(f"Batch Index Processed: {args.current_batch_index}")
    print(f"Proteins assigned to this batch: {len(proteins_to_process_this_batch)}")
    print(f"Successfully processed/found existing in this batch: {successful_in_this_batch}")
    print(f"Skipped in this batch (outputs existed): {skipped_in_this_batch}")
    print(f"Query FASTA files for this batch saved in: {args.fasta_output_dir}")
    print(f"TSV result files for this batch saved in: {args.tsv_output_dir}")
    print("Processing complete for this batch job.")
if __name__ == "__main__":
    main()
