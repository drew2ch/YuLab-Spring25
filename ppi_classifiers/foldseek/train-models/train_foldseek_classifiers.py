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
from cycler import cycler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))
)
from ...util.classifier_util import *

DEFAULT_USER_BASE_DIR = "C:/Users/hychu/OneDrive/Desktop/Summer25/github/ppi-classifiers"
DEFAULT_DATA_PATH = f"{DEFAULT_USER_BASE_DIR}/data/features.csv"
DEFAULT_FIGURES_DIR = f"{DEFAULT_USER_BASE_DIR}/foldseek/"
RANDOM_STATE = np.random.RandomState(893)

# Configure logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)

# refine figure aesthetics
plt.style.use('bmh')
plt.rcParams.update({
    # figure
    'figure.facecolor': 'white',
    'figure.dpi': 150,

    # axes
    'axes.facecolor': 'white',
    'axes.edgecolor': '0.2',         # dark grey axis lines
    'axes.grid': True,
    'grid.color': '0.8',             # light grey grid
    'grid.linestyle': '--',
    'grid.linewidth': 1,
    'axes.axisbelow': True,          # grid below the plot
    'axes.prop_cycle': cycler(
        'color', sns.color_palette("twilight_shifted", n_colors = 10)
    ),

    # lines
    'lines.linewidth': 2,
    'lines.markersize': 6,

    # fonts
    'font.family': 'serif',
    'font.serif': ['Palatino Linotype'],
    'font.size': 12,

    # ticks
    'xtick.color': '0.2',
    'ytick.color': '0.2',
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    # legend
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '0.2',
    'legend.fontsize': 10,
})

def compute_metrics(y_test, y_prob) -> dict:
    """ Given either y_test and y_prob from a trained RFC or pre-computed 
        PrePPI probabilistic scores, compute the following metrics:
        --- fpr: False Positive Rate (FPR)
        --- tpr: True Positive Rate (TPR)
        --- precision: Precision
        --- recall: Recall
        --- roc_auc: ROC curve AUC
        --- pr_auc: PR curve AUC
        --- ap_score: Average Precision
    """
    if len(y_test) == 0:
        logging.warning("Warning: y_test is empty, cannot compute metrics.")
        return {
            'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 
            'precision': np.array([0,1]), 'recall': np.array([1,0]),
            'roc_auc': 0.0, 'pr_auc': 0.0, 'ap_score': 0.0
        }
    if len(np.unique(y_test)) < 2:
        num_actual_positives = np.sum(y_test == 1)
        baseline_pr = num_actual_positives / len(y_test) if len(y_test) > 0 else 0.0
        logging.warning(f"Warning: y_test has only one class ({np.unique(y_test)[0]})")
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

def train_rf(X_train, y_train, X_test, y_test, n_estimators: int = 500, random_state = RANDOM_STATE) -> dict:
    """ Train a Random Forest Classifier on the train data and return 
        a structured output of classification metrics (see above) as a dict object.
        A helper function, compute_metrics(), is defined above to calculate aforementioned metrics.
    """

    rfc = RandomForestClassifier(
        n_estimators = n_estimators, 
        oob_score = True,
        random_state = random_state
    )
    rfc.fit(X_train, y_train)
    y_prob = rfc.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_prob), y_prob

def plot_roc_pr(
        results: dict, 
        axes, 
        classifier_name: str
    ):
    """ Plot ROC/PR curves for the given classifier results.
        NOTE: this function does not handle class ratios. 
        see classifier_util.plot_roc_pr_with_ratios() for that.
    --- 'axes' must conform to a 1x2 grid, respectively for ROC and PR curves.
    """

    # plot ROC curve
    sns.lineplot(
        x = results['fpr'], y = results['tpr'], ax = axes[0],
        label = f'{classifier_name} (AUROC = {results['roc_auc']:.3f})', linewidth = 1.5, ci = None
    )
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize = 12)
    axes[0].set_xlabel('False Positive Rate', fontsize = 12)
    axes[0].set_ylabel('True Positive Rate', fontsize = 12)
    axes[0].legend(loc = 'lower right', facecolor = 'white', fontsize = 8, frameon = True, fancybox = True)
    axes[0].plot([0, 1], [0, 1], linestyle = '--', color = 'gray', alpha = 0.5) # diagonal line

    # plot PR curve
    sns.lineplot(
        x = results['recall'], y = results['precision'], ax = axes[1],
        label = f'{classifier_name} (AUPR = {results['pr_auc']:.3f})', linewidth = 1.5, ci = None
    )
    axes[1].set_title('Precision Recall (PR)', fontsize = 12)
    axes[1].set_xlabel('Recall', fontsize = 12)
    axes[1].set_ylabel('Precision', fontsize = 12)
    axes[1].legend(loc = 'best', facecolor = 'white', fontsize = 8, frameon = True, fancybox = True)
    axes[1].set_ylim([0.0, 1.05])

    # figure aesthetics
    axes[0].tick_params(axis = 'both', which = 'major', labelsize = 12)
    axes[1].tick_params(axis = 'both', which = 'major', labelsize = 12)

def main():

    parser = argparse.ArgumentParser(description = "Foldseek Classifier Training")
    parser.add_argument('--data_path', type = str, default = DEFAULT_DATA_PATH, help = "Training data path")
    parser.add_argument('--figures_dir', type = str, default = DEFAULT_FIGURES_DIR, help = "Output directory for result figures")
    args = parser.parse_args()

    logging.info("=== Foldseek Classifier Training ===")
    logging.info("Loading and processing data...")

    # Load comprehensive PPI data set, retain only relevant columns
    try:
        data = pd.read_csv(args.data_path).set_index('ppi')[[
            'Total', 'co-expression', 'BP', 'CC', 'MF',
            'has_templates', 'fident', 'pident', 'evalue', 'bits', 'label'
        ]]
    except FileNotFoundError:
        logging.error(f"Data file(s) not found.")
        return
    assert 'label' in data.columns, \
        logging.error("Error: 'label' column is missing.")
    
    # Provide comprehensive overview of training and test data sets
    print(f"{'='*60}\n=== DATASETS OVERVIEW ===")

    print(f"Selected features: {', '.join(data.columns[:-1])}")
    print("Head of PPI Data File:")
    print(data.head())

    logging.info(f"{len(data)} PPI pairs found.")
    total_neg, total_pos = data['label'].value_counts()
    if total_pos == 0 or len(data) < 2:
        logging.error("ERROR: No positive samples or insufficient data. Exiting.")
        return
    print(f"Total samples for splitting: {len(data)} ({total_pos} P, {total_neg} N)")
    print(f"Overall P:N ratio: 1:{total_neg/total_pos:.1f}" if total_pos > 0 else "N/A (0 positives)")
    print(f"Overall Baseline precision: {total_pos/(total_pos + total_neg):.4f}" if (total_pos + total_neg) > 0 else "N/A")
    input("Press Enter to continue...")

    genomic_features = ['co-expression', 'BP', 'CC', 'MF']
    feature_sets = {
        'PrePPI -- total': ['Total'],
        'RF: Genetic Features': genomic_features,
        'RF: Genetic Features + HT/fident': genomic_features + ['has_templates', 'fident'],
        'RF: Genetic Features + HT/pident': genomic_features + ['has_templates', 'pident'],
        'RF: Genetic Features + HT/fident/pident': genomic_features + ['has_templates', 'fident', 'pident'],
        'RF: Genetic Features + HT/fident/pident/e-value': genomic_features + ['has_templates', 'fident', 'pident', 'evalue'],
        'RF: Genetic Features + HT/fident/pident/bit-score': genomic_features + ['has_templates', 'fident', 'pident', 'bits']
        # 'HT/fident': ['has_templates', 'fident'],
        # 'HT/pident': ['has_templates', 'pident'],
        # 'HT/fident/pident': ['has_templates', 'pident', 'fident']
    }

    """ Part 1: Full Raw Data File (P:N Ratio 1:4.5). Use all existing PPI pairs to train RFC.
        80-20 Train-Test Split, stratified by label proportions.
    """
    logging.info("=== Part 1: Training Classifiers on Full Data ===")
    fig, ax = plt.subplots(1, 2, figsize = (13, 6))
    fig.suptitle(
        f"PrePPI (Total) vs. Random Forest Classifier with Genomic and Foldseek Features (P:N Ratio = 1:{(total_neg/total_pos):.1f})", 
        fontsize = 16
    )

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns = 'label'), data['label'],
        test_size = 0.2, random_state = RANDOM_STATE, stratify = data['label']
    )

    for clf_id, features in feature_sets.items():

        assert all(feature in data.columns for feature in features), \
            logging.error(f"Error: One or more features {features} not found in data columns.")
        logging.info(f"\n=== Training Classifier {clf_id} with features {features} ===")
        
        """
        if clf_id == 'PrePPI (total)':
            RESULTS, y_prob = compute_metrics(y_test, X_test[features]), X_test[features].squeeze()
        else:
            RESULTS, y_prob = train_rf(X_train[features], y_train, X_test[features], y_test)
        """

        RESULTS, y_prob = compute_metrics(
            y_test, X_test[features]
        ), X_test[features].squeeze() if clf_id == 'PrePPI (total)' else train_rf(
            X_train[features], y_train, X_test[features], y_test
        )

        logging.info(f"Model training complete. Classifier {clf_id} results:")
        analyze_predictions(
            y_test, y_prob, 
            model_name = clf_id, 
            test_set_desc = ','.join(features)
        )
        logging.info("Plotting ROC/PR curves...")
        plot_roc_pr(
            results = RESULTS, axes = ax, classifier_name = clf_id
        )
    
    # finalize figures, save to output directory
    print(f"\n{'='*60}")
    logging.info(f"All classifiers have been trained/evaluated. Saving figures to output directory {args.figures_dir}...")
    try:
        plt.savefig('foldseek_classifiers_fulldata.png', bbox_inches = 'tight', dpi = 300)
        plt.close()
        logging.info("Plots saved as 'foldseek_classifiers_fulldata.png'.")
    except Exception as e:
        logging.error(f"Error saving plots: {e}")
        return
    
    input("Press Enter to continue...")
    print(f"\n{'='*60}\n")
          
    """ Part 2: Class Ratio-Based Resampling. Systematically oversample negative pairs to 
        simulate real-world protein interactions.
        Ratios (P:N): 1:1, 1:10, 1:100, 1:1000
        In the same fashion as Part 1, 80-20 Train-Test Split, stratified by label proportions.
    """
    logging.info("=== Part 2: Training Classifiers with Class Ratios ===")
    fig, ax = plt.subplots(2, 4, figsize = (20, 8))
    fig.suptitle(
        f"PrePPI (Total) vs. Random Forest Classifier with Genomic and Foldseek Features with Class Ratios", 
        fontsize = 16
    )
    # First, train random forest classifiers on a balanced (constant) 1:1 training set
    X_train_bal, y_train_bal = manual_resample_training_data(
        X_train, y_train, 
        n_pos_target = sum(y_train == 1), n_neg_target = sum(y_train == 1),
        random_state = RANDOM_STATE
    )
    logging.info(f"Balanced training set size: P: {sum(y_train_bal == 1)}, N: {sum(y_train_bal == 0)}")
    
    # train each RFC on the balanced training set
    # need to store trained RFC objects for test set evaluation
    rfc_models = {clf_id: None for clf_id in feature_sets.keys() if clf_id != 'PrePPI -- total'}
    for i, (clf_id, features) in enumerate(list(feature_sets.items())[1:]): # omit PrePPI (not RF)

        logging.info(f"Training Classifier {clf_id} with balanced 1:1 train set...")
        assert all(feature in X_train_bal.columns for feature in features), \
            logging.error(f"Error: One or more features {features} not found in data columns.")

        rfc = RandomForestClassifier(
            n_estimators = 500, oob_score = True, random_state = RANDOM_STATE
        )
        rfc.fit(X_train_bal[features], y_train_bal)
        rfc_models[clf_id] = rfc
    logging.info("All classifiers trained on balanced 1:1 training set.")

    """ Now, systematically create test sets with varying class ratios.
        For each ratio, oversample the negative class to achieve the desired ratio.
        Then, evaluate each classifier on the test set and plot ROC/PR curves.
    """
    ratios = np.power(10, np.arange(0, 4))  # 1:1, 1:10, 1:100, 1:1000
    for i, ratio in enumerate(ratios):

        print(f"\n{'='*60}\n=== Target Class Ratio 1:{ratio} ===")
        # Test Set Resampling
        X_test_ratio, y_test_ratio = create_ratioed_test_set_oversample_neg(
            X_test, y_test, target_neg_multiplier = ratio, random_state = RANDOM_STATE + i # slight seed variations
        )
        n_p_test_ratio, n_n_test_ratio = sum(y_test_ratio == 1), sum(y_test_ratio == 0)
        logging.info(f"Test set size: P: {n_p_test_ratio}, N: {n_n_test_ratio}")
        logging.info(f"Overall Baseline Precision: {n_p_test_ratio/(n_p_test_ratio + n_n_test_ratio):.4f}")

        for clf_id, features in feature_sets.items():

            logging.info(f"\n=== Training Classifier {clf_id} with features {features} ===")
            if clf_id == 'PrePPI (total)':
                RESULTS, y_prob = compute_metrics(y_test_ratio, X_test_ratio[features]), X_test_ratio[features].squeeze()
            else:
                clf = rfc_models[clf_id]
                y_prob = clf.predict_proba(X_test_ratio[features])[:, 1]
                RESULTS = compute_metrics(y_test_ratio, y_prob)
            
            logging.info(f"Model training complete. Classifier {clf_id} results:")
            analyze_predictions(
                y_test_ratio, y_prob, model_name = clf_id, test_set_desc = ','.join(features)
            )
            logging.info("Plotting ROC/PR curves...")
            plot_roc_pr_with_ratios(
                results = RESULTS, axes = ax, cind = i, 
                plot_title_detail = None,
                clf_type = clf_id
            )
        
        ax[0, i].set_title(f"Test Set: P:N ~ 1:{ratio}\n({n_p_test_ratio}P:{n_n_test_ratio}N)")

    # finalize figures, save to output directory
    print(f"\n{'='*60}")
    logging.info(f"All classifiers have been trained/evaluated throughout all ratios. Saving figures to output directory {args.figures_dir}...")
    try:
        plt.savefig('foldseek_classifiers_ratio.png', bbox_inches = 'tight', dpi = 300)
        plt.close()
        logging.info("Plots saved as 'foldseek_classifiers_ratio.png'.")
    except Exception as e:
        logging.error(f"Error saving plots: {e}")
        return

if __name__ == "__main__":
    main()
    logging.info(f"Job Finished. \n{'='*60}")
