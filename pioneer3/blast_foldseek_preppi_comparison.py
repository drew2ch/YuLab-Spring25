""" Modified version of Juheon's (jc3668) code, originally writte to compare BLAST 
    and PrePPI template hits for PIONEER3 target PPI pairs, integrating both BLAST
    and Foldseek comparisons.

    Andrew Chung (hc893)
    10/28/2025
"""

import os
import pandas as pd
import xml.etree.ElementTree as ET
# from tqdm import tqdm
from matplotlib import pyplot as plt

# modify paths
xlsx_path = "/home/hc893/data/PrePPI_for_PIONEER_8k_PDB.xlsx"
xml_dir = "/share/yu/jc3668/data/pioneer2/blast/scratch_pioneer_preppi/xml"
tsv_dir = "/share/yu/jc3668/data/pioneer2/foldseek/tsv_prostT5_5000_pioneer_preppi/"

def has_hit_blast(xml_path, pdbid, chain):
    if not os.path.exists(xml_path):
        return False
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pdb_lower = pdbid.lower()

    for hit in root.findall(".//Hit"):
        hit_id = hit.find("Hit_id")
        if hit_id is not None and hit_id.text:
            parts = hit_id.text.split('|')
            # <Hit_id> format: "pdb|<pdb_id>|<chain_id>"
            if len(parts) >= 3: 
                # pdb ID is case insensitive,
                # chain ID is case sensitive
                if parts[1].lower() == pdb_lower and \
                    parts[2] == chain: return True
    return False

def has_hit_foldseek(tsv_path, pdbid, chain):
    if not os.path.exists(tsv_path):
        return False
    try:
        data = pd.read_csv(tsv_path, sep = '\t', header = None)
    except pd.errors.EmptyDataError:
        return False
    pdb_lower = pdbid.lower()

    for index, hit in data.iterrows():
        hit = hit[1]
        if hit is not None:
            pdb_id = hit.split('-')[0]
            chain_id = hit.split('_')[-1]
            if pdb_id.lower() == pdb_lower and \
            chain_id == chain: return True
    return False

def make_piechart(
        sizes, 
        explode = (0.1, 0, 0),
        labels = ['Both Hits', 'One Hit', 'No Hits'], 
        colors = ['#66b3ff', '#ffcc99', '#ff88cc'],
        title = None
    ):

    fig, ax = plt.subplots(figsize = (8, 6))
    ax.pie(
        sizes, explode = explode, labels = labels, colors = colors,
        autopct = '%1.1f%%', shadow = True, startangle = 90
    )
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(title)

def main():
    # PrePPI template hits
    data = pd.read_excel(xlsx_path, sheet_name = 'Sheet2')
    total = len(data)
    count_both_blast, count_one_blast, count_neither_blast = 0, 0, 0
    count_both_fs, count_one_fs, count_neither_fs = 0, 0, 0

    for index, row in data.iterrows():
        model = row.get('model')
        if pd.isna(model) or not model:
            count_neither_blast += 1
            count_neither_fs += 1
            continue
        parts = model.split(':')
        if len(parts) < 2:
            count_neither_blast += 1
            count_neither_fs += 1
            continue
        pdbid, chains = parts[0], parts[1]
        if len(chains) != 2:
            count_neither_blast += 1
            count_neither_fs += 1
            continue

        prot1, prot2 = row['uniprot1'], row['uniprot2']
        chain1, chain2 = chains # condition checked above

        # BLAST
        xml1, xml2 = os.path.join(xml_dir, f"{prot1}.xml"), \
                    os.path.join(xml_dir, f"{prot2}.xml")
        hit1_blast, hit2_blast = has_hit_blast(xml1, pdbid, chain1), has_hit_blast(xml2, pdbid, chain2)

        if hit1_blast and hit2_blast: count_both_blast += 1
        elif hit1_blast or hit2_blast: count_one_blast += 1
        else: count_neither_blast += 1

        # Foldseek
        tsv1, tsv2 = os.path.join(tsv_dir, f"{prot1}.tsv"), \
                    os.path.join(tsv_dir, f"{prot2}.tsv")
        hit1_fs, hit2_fs = has_hit_foldseek(tsv1, pdbid, chain1), has_hit_foldseek(tsv2, pdbid, chain2)

        if hit1_fs and hit2_fs: count_both_fs += 1
        elif hit1_fs or hit2_fs: count_one_fs += 1
        else: count_neither_fs += 1

    make_piechart(
        [count_both_blast, count_one_blast, count_neither_blast], 
        'figures/blast_preppi_comparison_pioneer3.png'
    )
    make_piechart(
        [count_both_fs, count_one_fs, count_neither_fs], 
        'figures/foldseek_preppi_comparison_pioneer3.png'
    )

if __name__ == "__main__":
    main()
