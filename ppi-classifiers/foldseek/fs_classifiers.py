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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
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
    return compute_metrics(y_test, y_prob)

def plot_roc_pr(metrics: dict, axes, cind, plot_title_detail, data_type = None) -> None:
    """ Plot ROC/PR curves and refines figure aesthetics. 
    --- metrics should be a dict object with the key:value pairs prescribed in compute_metrics above.
    --- axes: the axes on which your plots are superimposed.
    --- cind: (for grid-arranged axes) column index
    --- plot_title_detail: customizable
    --- data_type: nature of the data set (e.g. features used, data source)
    """
    # plot ROC curve
    sns.lineplot(
        x = metrics['fpr'], y = metrics['tpr'], ax = axes[0, cind], 
        label = f'{data_type} ({metrics['roc_auc']:.3f})',
        linewidth = 1.5, ci = None
    )
    axes[0, cind].set_title(f"Test set: {plot_title_detail}", fontsize = 11)
    axes[0, cind].set_xlabel("False Positive Rate", fontsize = 9)
    axes[0, cind].set_ylabel("True Positive Rate", fontsize = 9)
    axes[0, cind].legend(loc = 'lower left', facecolor = 'white', fontsize = 7, frameon = True, fancybox = True)
    axes[0, cind].plot([0, 1], [0, 1], linestyle = '--', color = 'gray', alpha = 0.5)

    # plot PR curve
    sns.lineplot(
        x = metrics['recall'], y = metrics['precision'], ax = axes[1, cind],
        label = f'{data_type} ({metrics['pr_auc']:.3f})',
        linewidth = 1.5, ci = None
    )
    axes[1, cind].set_xlabel('Recall', fontsize = 9)
    axes[1, cind].set_xlabel('Precision', fontsize = 9)
    axes[1, cind].legend(loc = 'best', facecolor = 'white', fontsize = 7, frameon = True, fancybox = True)
    axes[1, cind].set_ylim([0.0, 1.05])

    # refine figure aesthetics
    axes[0, cind].tick_params(axis = 'both', which = 'major', labelsize = 8)
    axes[1, cind].tick_params(axis = 'both', which = 'major', labelsize = 8)

# def 
