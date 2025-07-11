import os
import requests
import argparse
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

def get_unique_uniprot_ids(file_paths_list):
    """Reads your train/test/validation files and gets a set of all unique UniProt IDs."""
    unique_ids = set()
    for file_path in file_paths_list:
        if not os.path.exists(file_path):
            logging.warning(f"Dataset file not found, skipping: {file_path}")
            continue
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    unique_ids.add(parts[0])
                    unique_ids.add(parts[1])
    logging.info(f"Found {len(unique_ids)} unique UniProt IDs in dataset files.")
    return list(unique_ids) # Return as a list for stable ordering

def download_structure(uniprot_id, output_dir):
    """
    Downloads the latest AlphaFold structure for a single UniProt ID.
    The URL points to model v4, the latest version available via this API.
    """
    # Construct the download URL and the local output path
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
    output_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1-model_v4.cif")

    # If the file already exists, skip the download
    if os.path.exists(output_path):
        return uniprot_id, "skipped"

    try:
        # Make the web request to download the file
        response = requests.get(url, stream=True, timeout=60)
        
        # Check if the request was successful (e.g., HTTP 200 OK)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return uniprot_id, "downloaded"
        else:
            # If the file doesn't exist on the server (e.g., 404 Not Found)
            return uniprot_id, f"failed_http_{response.status_code}"
            
    except requests.exceptions.RequestException as e:
        # Handle network errors (e.g., timeout, connection error)
        return uniprot_id, f"failed_network_{e.__class__.__name__}"


def main():
    """parser = argparse.ArgumentParser(description="Download latest AlphaFold structures for a list of UniProt IDs.")
    parser.add_argument('--dataset_files', nargs='+', required=True,
                        help="One or more paths to your train/test/validation set files.")
    parser.add_argument('--output_dir', default="/share/yu/jc3668/data",
                        help="The directory where downloaded .cif files will be saved.")
    parser.add_argument('--threads', type=int, default=10,
                        help="Number of parallel downloads.")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the list of IDs to process
    uniprot_ids = get_unique_uniprot_ids(args.dataset_files)
    total_ids = len(uniprot_ids)
    
    if total_ids == 0:
        logging.error("No UniProt IDs found to process. Exiting.")
        return"""
        

    QUERY_DATA_PATH = "C:/Users/hychu/OneDrive/Desktop/Summer25/github/ppi-classifiers/data/query_pairs_for_3dmodel.csv"
    OUTPUT_DIR = "C:/Users/hychu/OneDrive/Desktop/Summer25/pdb/query-singles"
    query_pairs = pd.read_csv(QUERY_DATA_PATH)['id'].to_numpy()
    uniprot_ids = set(np.array([
        query_pair.split('_') for query_pair in query_pairs
    ]).flatten().tolist())
    THREADS = 10
    logging.info(f"Starting download for {len(uniprot_ids)} proteins with {THREADS} parallel threads...")


    # Use a ThreadPoolExecutor for parallel downloads
    success_count = 0
    skipped_count = 0
    failed_ids = []

    with ThreadPoolExecutor(max_workers = THREADS) as executor:
        # Create a future for each download task
        future_to_id = {executor.submit(download_structure, uid, OUTPUT_DIR): uid for uid in uniprot_ids}
        
        for i, future in enumerate(as_completed(future_to_id)):
            uniprot_id, status = future.result()
            
            if status == "downloaded":
                success_count += 1
            elif status == "skipped":
                skipped_count += 1
            else: # Any failure status
                failed_ids.append(f"{uniprot_id} ({status})")
            
            # Log progress
            if (i + 1) % 100 == 0:
                logging.info(f"  ...processed {i+1}/{len(uniprot_ids)}")

    print("\n" + "="*50)
    print("      DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total proteins to process: {len(uniprot_ids)}")
    print(f"Successfully downloaded:   {success_count}")
    print(f"Skipped (already exist):   {skipped_count}")
    print(f"Failed to download:        {len(failed_ids)}")
    if failed_ids:
        print("\nFailed IDs (check if they are valid or exist in AlphaFold DB):")
        # Print first 20 failed IDs for brevity
        for item in failed_ids[:20]:
            print(f"  - {item}")
        if len(failed_ids) > 20:
            print(f"  ... and {len(failed_ids) - 20} more.")
    print("="*50)

if __name__ == "__main__":
    main()
