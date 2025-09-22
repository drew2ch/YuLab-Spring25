""" Script to generate P-R scores from individual protein pair residues.

    Adapted from Juheon Chu's Code
    Andrew Chung, hc893; 09/18/2025
"""

import json
import glob
import os
import re
import sys
import pandas as pd
import numpy as np
import time
import itertools
import seaborn as sns
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.lines import Line2D
from collections import defaultdict

BASE = "/home/hc893/"
CONDITIONS = ['condition_1', 'condition_2', 'condition_3']
JSONL_PATH = os.path.join(BASE, 'data/candidate-templates')
GROUND_TRUTH = os.path.join(BASE, 'data/ires_all.txt')

""" Load the ground truth data file.
    Format: Protein 1, Protein 2, IRes 1, IRes 2
"""
def load_ground_truth_query_ires(path):
    ground_truth_data = {}
    print(f"Loading ground truth IRES from: {path}")
    with open(path, 'r') as f:
        for line in f:
            p1, p2, ires1, ires2 = line.strip().split('\t')
            ires1_set = {idx for idx, val in enumerate(ires1.split(';')) if val == '1'}
            ires2_set = {idx for idx, val in enumerate(ires2.split(';')) if val == '1'}
            pair_key_tuple = tuple(sorted([p1, p2]))
            if pair_key_tuple[0] == p1:
                ground_truth_data[pair_key_tuple] = (ires1_set, ires2_set)
            else: ground_truth_data[pair_key_tuple] = (ires2_set, ires1_set)
    
    return ground_truth_data

""" The main analysis function.
--- For each condition, find all .jsonl files (i.e. covering all PPI pairs) and
--- compute P-R scores for each condition/PPI.
--- P-R scores are computed based on predicted vs. ground truth interface residues.
"""
def run_computation(ground_truth):
    start_time = time.time()

    # store P-R scores for each condition: condition_1, condition_2, condition_3
    # the innermost dict stores aggregate precision and recall scores across all PPI pairs for a given condition.
    results = {
        'condition_1': defaultdict(list, {'precision': [], 'recall': []}),
        'condition_2': defaultdict(list, {'precision': [], 'recall': []}),
        'condition_3': defaultdict(list, {'precision': [], 'recall': []})
    }
    # results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    pairs_in_gt = 0 # count of PPI pairs found in ground truth file
    print(f"Processing Data File...")
    for condition in CONDITIONS: # each of the 3 conditions
        print(f"=== Processing {condition} ===")
        # find all .jsonl files under the specified condition
        jsonl_files = glob.glob(os.path.join(JSONL_PATH, f"best_templates_{condition}_train_*.jsonl"))
        if not jsonl_files:
            print(f"Error: No .jsonl files found for {condition}. Skipping.")
            continue
        # per_ppi_scores = defaultdict(np.array) # store P-R scores for each PPI pair

        # iterate through each PPI pair file
        
        for path in jsonl_files:
            # extract protein Uniprot IDs
            m = re.match(f"best_templates_{condition}_train_([A-Z0-9]+)_([A-Z0-9]+).jsonl", os.path.basename(path))
            if m:
                p1, p2 = m.groups()
            else: 
                print(f"Error: Filename {os.path.basename(path)} does not match expected pattern. Skipping.")
                continue

            templates = []
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        templates.append(json.loads(line))
            if not templates: # no templates found for condition/PPI
                print(f"No template pairs found for {condition} {p1}-{p2}. Skipping.")
                # per_ppi_scores[(p1, p2)] = 0.0
                continue

            pair_tuple = tuple(sorted((p1, p2)))
            if pair_tuple not in ground_truth.keys():
                print(f"Warning: Ground truth missing for PPI pair {p1}-{p2}. Skipping.")
                continue
            else: pairs_in_gt += 1
            ground_truth_ires1, ground_truth_ires2 = ground_truth[pair_tuple]
            print(f"Processing PPI Pair {p1}-{p2} with {len(templates)} template pairs.")

            for template in templates:
                pred_set1 = set(template['mapped_binding_sites'][0])
                pred_set2 = set(template['mapped_binding_sites'][1])

                # compute true positives, false positives, false negatives
                TP = len(pred_set1.intersection(ground_truth_ires1)) + len(pred_set2.intersection(ground_truth_ires2))
                FP = len(pred_set1.difference(ground_truth_ires1)) + len(pred_set2.difference(ground_truth_ires2))
                FN = len(ground_truth_ires1.difference(pred_set1)) + len(ground_truth_ires2.difference(pred_set2))
                
                # compute precision, recall scores
                PRECISION = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                RECALL = TP / (TP + FN) if (TP + FN) > 0 else 0.0

                # store precision, recall scores for the condition
                results[condition]['precision'].append(PRECISION)
                results[condition]['recall'].append(RECALL)

                print(f"Precision: {PRECISION:.4f}, Recall: {RECALL:.4f}")

        print(f"Completed processing for {condition}.")
        print('='*60)
        print(f"Total PPI pairs found in ground truth: {pairs_in_gt}")
        pairs_in_gt = 0 # reset for next condition

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds")
    return results

def plot_pr_distributions(results):

    fig, ax = plt.subplots(1, len(results), figsize = (18, 6), sharey = True)
    fig.suptitle("P-R Distributions Across IRes Prediction Conditions", fontsize = 16)

    unique_handles = {}
    for index, (condition, scores) in enumerate(results.items()):
        precision_scores = scores['precision']
        recall_scores = scores['recall']

        # Plot Precision Distribution
        sns.histplot(precision_scores, bins = 20, kde = True, color = 'blue', ax = ax[index], label = 'Precision', stat = 'density', alpha = 0.3)
        # Plot Recall Distribution
        sns.histplot(recall_scores, bins = 20, kde = True, color = 'orange', ax = ax[index], label = 'Recall', stat = 'density', alpha = 0.3)

        ax[index].set_title(f"{condition.replace('_', ' ').title()}")
        ax[index].set_xlabel("Score")
        ax[index].set_ylabel("Density" if index == 0 else "")
        
        # Only grab the handles and labels from the first subplot to avoid duplicates
        if index == 0:
            handles, labels = ax[index].get_legend_handles_labels()
            # Store them in a dictionary to ensure uniqueness
            for handle, label in zip(handles, labels):
                if label not in unique_handles:
                    unique_handles[label] = handle
        ax[index].legend().remove()  # Remove individual legends

    fig.legend(
        unique_handles.values(), unique_handles.keys(),
        loc = 'upper right',
        bbox_to_anchor = (1, 1),  # The key change: place the legend anchor at the top-right corner of the figure
        bbox_transform = fig.transFigure,
        title = ""
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave right margin for legend

    # save figure
    plt.savefig("figures/pr_distributions.png", dpi = 300)

if __name__ == "__main__":

    ground_truth_db = load_ground_truth_query_ires(GROUND_TRUTH)
    results = run_computation(ground_truth_db)
    plot_pr_distributions(results)

    print("Final Results Summary:")
    for condition, metrics in results.items():
        avg_precision = np.mean(metrics['precision']) if metrics['precision'] else 0.0
        avg_recall = np.mean(metrics['recall']) if metrics['recall'] else 0.0
        print(f"{condition}: Average Precision = {avg_precision:.4f}, Average Recall = {avg_recall:.4f}")
