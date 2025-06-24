""" Training Random Forest Classifier Algorithm on PPI feature space with Foldseek-integrated features.
--- Classifier 1: Traditional (coexp, BP, CC, MF)
--- Classifier 2: coexp, BP, CC, MF + has_templates, fident
--- Classifier 3: coexp, BP, CC, MF + has_templates, pident
--- Classifier 4: coexp, BP, CC, MF + has_templates, fident, pident
--- Classifier 5: has_templates, pident
--- Classifier 6: has_templates, fident
--- Classifier 7: has_templates, pident, fident
=== Andrew Chung, hc893; 6/16/2025 ===
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))
)
from util.classifier_util import *

# Configure logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)

def train_rf(X_train, y_train, X_test, y_test, n_estimators = 500) -> dict:
    """ Train a Random Forest Classifier on the train data and return 
        a structured output of classification metrics as a dict object.
    --- fpr: False Positive Rate (FPR)
    --- tpr: True Positive Rate (TPR)
    --- precision: Precision
    --- recall: Recall
    --- roc_auc: ROC curve AUC
    --- pr_auc: PR curve AUC
    --- ap_score: Average Precision
        A helper function, compute_metrics(), is defined inside to calculate the above metrics.
    """
    def compute_metrics(y_test, y_prob) -> dict:
        if len(y_test) == 0:
            print("Warning: y_test is empty, cannot compute metrics.")
            return {
                'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 
                'precision': np.array([0,1]), 'recall': np.array([1,0]),
                'roc_auc': 0.0, 'pr_auc': 0.0, 'ap_score': 0.0
            }
        if len(np.unique(y_test)) < 2:
            num_actual_positives = np.sum(y_test == 1)
            baseline_pr = num_actual_positives / len(y_test) if len(y_test) > 0 else 0.0
            print(f"Warning: y_test has only one class ({np.unique(y_test)[0]})")
            return {
                'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 
                'precision': np.array([baseline_pr, baseline_pr]), 'recall': np.array([1,0]),
                'roc_auc': 0.5, 'pr_auc': baseline_pr, 'ap_score': baseline_pr
            }
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ap_score = average_precision_score(y_test, y_prob)
        return {
            'fpr': fpr, 'tpr': tpr,
            'precision': precision, 'recall': recall,
            'roc_auc': roc_auc, 'pr_auc': ap_score, 'ap_score': ap_score
        }

    rfc = RandomForestClassifier(
        n_estimators = n_estimators, 
        oob_score = True,
        random_state = 893
    )
    rfc.fit(X_train, y_train)
    y_prob = rfc.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_prob), y_prob

def plot_roc_pr(results: dict, axes, classifier_name: str):
    """ Plot ROC/PR curves for the given classifier results.
        NOTE: this function does not handle class ratios. 
        see classifier_util.plot_roc_pr_with_ratios() for that.
    --- 'axes' must conform to a 1x2 grid, respectively for ROC and PR curves.
    """

    # plot ROC curve
    sns.lineplot(
        x = results['fpr'], y = results['tpr'], ax = axes[0],
        label = f'{classifier_name} ({results['roc_auc']:.3f})', linewidth = 1.5, ci = None
    )
    axes[0].set_title('ROC Curve', fontsize = 14)
    axes[0].set_xlabel('False Positive Rate', fontsize = 12)
    axes[0].set_ylabel('True Positive Rate', fontsize = 12)
    axes[0].legend(loc = 'lower right', facecolor = 'white', fontsize = 10, frameon = True, fancybox = True)
    axes[0].plot([0, 1], [0, 1], linestyle = '--', color = 'gray', alpha = 0.5) # diagonal line

    # plot PR curve
    sns.lineplot(
        x = results['recall'], y = results['precision'], ax = axes[1],
        label = f'{classifier_name} ({results['pr_auc']:.3f})', linewidth = 1.5, ci = None
    )
    axes[1].set_title('Precision-Recall Curve', fontsize = 14)
    axes[1].set_xlabel('Recall', fontsize = 12)
    axes[1].set_ylabel('Precision', fontsize = 12)
    axes[1].legend(loc = 'best', facecolor = 'white', fontsize = 10, frameon = True, fancybox = True)
    axes[1].set_ylim([0.0, 1.05])

    # figure aesthetics
    axes[0].tick_params(axis = 'both', which = 'major', labelsize = 12)
    axes[1].tick_params(axis = 'both', which = 'major', labelsize = 12)

def main():

    parser = argparse.ArgumentParser(description = "Foldseek Classifier Training")
    parser.add_argument('--training_data_path', type = str, required = True, help = "Training data path")
    parser.add_argument('--test_data_path', type = str, required = True, help = "Test data path")
    parser.add_argument('--output_figures_dir', type = str, required = True, help = "Output directory for result figures")
    args = parser.parse_args()

    print("=== Foldseek Classifier Training ===")
    print("Loading and processing data...")

    # Load training and test data
    try:
        training_data = pd.read_csv(args.training_data_path).set_index('ppi')
        test_data = pd.read_csv(args.test_data_path).set_index('ppi')
    except FileNotFoundError:
        logging.error(f"Data file(s) not found.")
        return

    assert ('label' in training_data.columns) and ('label' in test_data.columns), \
        logging.error("Error: 'label' column is missing from either data set.")
    
    # Provide comprehensive overview of training and test data sets
    print(f"=== DATASETS OVERVIEW ===")

    logging.info(f"Training data: {len(training_data)} pairs")
    logging.info(f"Test data: {len(test_data)} pairs")

    print(f"--- Training Data ---")
    total_neg_train, total_pos_train = training_data['label'].value_counts()
    if total_pos_train == 0 or len(training_data) < 2:
        logging.error("ERROR: No positive samples or insufficient data. Exiting.")
        return
    print(f"Total samples for splitting: {len(training_data)} ({total_pos_train} P, {total_neg_train} N)")
    print(f"Overall P:N ratio: 1:{total_neg_train/total_pos_train:.1f}" if total_pos_train > 0 else "N/A (0 positives)")
    print(f"Overall Baseline precision: {total_pos_train/(total_pos_train + total_neg_train):.4f}" if (total_pos_train + total_neg_train) > 0 else "N/A")

    print(f"--- Test Data ---")
    total_neg_test, total_pos_test = test_data['label'].value_counts()
    if total_pos_test == 0 or len(test_data) < 2:
        logging.error("ERROR: No positive samples or insufficient data. Exiting.")
        return
    print(f"Total samples for splitting: {len(test_data)} ({total_pos_test} P, {total_neg_test} N)")
    print(f"Overall P:N ratio: 1:{total_neg_test/total_pos_test:.1f}" if total_pos_test > 0 else "N/A (0 positives)")
    print(f"Overall Baseline precision: {total_pos_test/(total_pos_test + total_neg_test):.4f}" if (total_pos_test + total_neg_test) > 0 else "N/A")

    input("Press Enter to continue...")

    # train-test split is unnecessary here since we already have a prescribed holdout test set.
    # instead, let's define sets of features for each classifier, with the order above
    genomic_features = ['coexp', 'BP', 'CC', 'MF']
    feature_sets = {
        'genomic': genomic_features,
        'genomic + HT/fident': genomic_features + ['has_templates', 'fident'],
        'genomic + HT/pident': genomic_features + ['has_templates', 'pident'],
        'genomic + HT/fident/pident': genomic_features + ['has_templates', 'fident', 'pident'],
        'HT/fident': ['has_templates', 'fident'],
        'HT/pident': ['has_templates', 'pident'],
        'HT/fident/pident': ['has_templates', 'pident', 'fident']
    }

    """ Train each classifier and collect metrics
        Plotting region is defined as a 1x2 grid, exhibiting ROC/PR performances for each classifier.
    """
    fig, ax = plt.subplots(1, 2, figsize = (10, 6))
    sns.set_theme(style = "darkgrid")

    for clf_id, features in feature_sets.items():

        logging.info(f"\n=== Training Classifier {clf_id} with features {','.join(features)} ===")

        # organize train and test data
        X_train, X_test = training_data[features], test_data[features]
        y_train, y_test = training_data.label, test_data.label
        assert (X_train.columns == features) and (X_test.columns == features), \
            logging.error(f"Error: certain features are missing from either train or test set.")
        
        RF_RESULTS, y_prob = train_rf(X_train, y_train, X_test, y_test)

        logging.info(f"Model training complete. Classifier {clf_id} results:")
        analyze_predictions(
            y_test, y_prob, 
            model_name = clf_id, 
            test_set_desc = ','.join(features)
        )
        logging.info("Plotting ROC/PR curves...")
        plot_roc_pr(
            results = RF_RESULTS, axes = ax, classifier_name = clf_id
        )
        print()
    
    # finalize figures, save to output directory
    print(f"\n{'='*60}")
    logging.info(f"All classifiers have been trained/evaluated. Saving figures to output directory {args.output_figures_dir}...")
    try:
        plt.savefig('foldseek_classifiers.png', bbox_inches = 'tight', dpi = 300)
        plt.close()
        logging.info("Plots saved as 'foldseek_classifiers.png'.")
    except Exception as e:
        logging.error(f"Error saving plots: {e}")
        return
    logging.info(f"Job Finished. \n{'='*60}")

if __name__ == "__main__":
    main()
