""" Script to define some common utility functions for classifiers.
--- manual_resample_training_data: re-sampling of X and y training data to target positive/negative counts.
--- analyze_predictions: detailed analysis of classifier predictions, baseline precision, and score distributions.
--- crreate_ratioed_test_set_oversample_neg: create test sets by keeping all positive samples and oversampling 
    negative samoples (with replacement) to achieve a given target negative multiplier.
--- plot_roc_pr: plot ROC and PR curves.
Much of the helper functions were sourced from Juheon's original code for PrePPI comparisons (compare.py).
=== Andrew Chung, hc893; 6/20/2025 ===
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def manual_resample_training_data(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        n_pos_target: int, 
        n_neg_target: int, 
        random_state: int = 893
    ):
    # manual re-sampling of X and y training data to target positive/negative counts.
    rng = np.random.RandomState(random_state)
    X_data_df = X_train.reset_index(drop = True)
    y_data_series = y_train.reset_index(drop = True)

    pos_df = X_data_df[y_data_series == 1]
    neg_df = X_data_df[y_data_series == 0]
    n_pos_current = len(pos_df)
    n_neg_current = len(neg_df)

    selected_dfs = []

    # positive samples
    if n_pos_target > 0:
        if n_pos_current == 0:
            print(f"Warning: No positive samples in input data to manual_resample_training_data for target {n_pos_target}.")
        else:
            selected_pos_df = pos_df.sample(
                n = n_pos_target,
                random_state = rng,
                replace = n_pos_target > n_pos_current # oversample only if n_pos_target > n_pos_current
            )
            selected_dfs.append(selected_pos_df)
    
    # negative samples
    if n_neg_target > 0:
        if n_neg_current == 0:
            print(f"Warning: No negative samples in input data to manual_resample_training_data for target {n_neg_target}.")
        else:
            selected_neg_df = neg_df.sample(
                n = n_neg_target,
                random_state = rng,
                replace = n_neg_target > n_neg_current # oversample only if n_neg_target > n_neg_current
            )
            selected_dfs.append(selected_neg_df)

    if not selected_dfs: # no samples available, or both n_pos_target and n_neg_target are 0
        return pd.DataFrame(columns = X_data_df.columns), pd.Series(dtype = int)
    
    X_resampled = pd.concat(selected_dfs)
    y_resampled = y_data_series[X_resampled.index]

    # shuffle combined data
    shuffled_indices = rng.permutation(len(X_resampled))
    X_resampled = X_resampled.iloc[shuffled_indices].reset_index(drop = True)
    y_resampled = y_resampled.iloc[shuffled_indices].reset_index(drop = True)
    return X_resampled, y_resampled

def analyze_predictions(
        y_test: pd.Series, 
        y_prob: pd.Series, 
        model_name, 
        test_set_desc
    ):
    n_pos_test = sum(y_test == 1)
    n_neg_test = sum(y_test == 0)

    if len(y_test) == 0:
        print("Test set is empty for analysis.")
        return
    
    baseline_precision = n_pos_test / len(y_test) if len(y_test) > 0 else 0.0
    pos_scores = y_prob[y_test == 1] if n_pos_test > 0 else np.array([])
    neg_scores = y_prob[y_test == 0] if n_neg_test > 0 else np.array([])

    print(f"\n=== {model_name} - Test Set Details: {test_set_desc} ===")
    print(f"Actual Test Composition: {n_pos_test} P, {n_neg_test} N (Baseline Precision: {baseline_precision:.4f})")
    
    print("Positive scores - {}".format(
        f"Mean: {pos_scores.mean():.4f}, Med: {np.median(pos_scores):.4f}, SD: {pos_scores.std():.4f}" if len(pos_scores) > 0 else "No positive samples in this test set."
    ))
    print("Negative scores - {}".format(
        f"Mean: {neg_scores.mean():.4f}, Med: {np.median(neg_scores):.4f}, SD: {neg_scores.std():.4f}" if len(neg_scores) > 0 else "No negative samples in this test set."
    ))
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        print(f"Score separation (mean(P) - mean(N)) = {pos_scores.mean() - neg_scores.mean():.4f}")

    for threshold in np.array([0.1, 0.5, 0.9]):
        n_pred_over_thr = sum(y_prob >= threshold)
        if n_pred_over_thr > 0:
            tp_at_thr = sum((y_prob >= threshold) & (y_test == 1))
            precision_at_thr = tp_at_thr / n_pred_over_thr
            recall_at_thr = tp_at_thr / n_pos_test if n_pos_test > 0 else 0.0
            print(f"Thr {threshold:.1f}: {n_pred_over_thr} preds, Prec = {precision_at_thr:.4f}, Rec = {recall_at_thr:.4f}")
        else: 
            print(f"Thr {threshold:.1f}: 0 preds")

def create_ratioed_test_set_oversample_neg(X_orig_test, y_orig_test, target_neg_multiplier, random_state = 893):
    """
    Creates test sets by keeping all positive samples from X_orig_test and 
    oversampling negative samples (with replacement) from X_orig_test 
    to achieve the target_neg_multiplier (negatives = positives * multiplier).
    """
    rng = np.random.RandomState(random_state)
    
    X_orig_test_df = X_orig_test.reset_index(drop = True)
    y_orig_test_series = y_orig_test.reset_index(drop = True)

    X_pos_all_from_orig_test = X_orig_test_df[y_orig_test_series == 1]
    y_pos_all_from_orig_test = y_orig_test_series[y_orig_test_series == 1]
    
    X_neg_unique_from_orig_test = X_orig_test_df[y_orig_test_series == 0]
    # y_neg_unique_from_orig_test = y_orig_test_series[y_orig_test_series == 0] # Not directly used for sampling y

    n_pos_final = len(X_pos_all_from_orig_test) # Keep all original test positives
    
    if n_pos_final == 0:
        print(f"  Warning: No positive samples in original test data. Cannot create targeted ratio 1:{target_neg_multiplier}.")
        # Return empty or just a few negatives if that's meaningful.
        if len(X_neg_unique_from_orig_test) > 0:
            n_neg_sample = min(10, len(X_neg_unique_from_orig_test)) # Sample a few negatives for consistency
            X_neg_final_df = X_neg_unique_from_orig_test.sample(n = n_neg_sample, random_state = rng, replace = False)
            y_neg_final_series = y_orig_test_series.loc[X_neg_final_df.index] # Get original labels for these negatives
            print(f"  Returning {n_neg_sample} unique negative samples as positives are zero.")
            return X_neg_final_df, y_neg_final_series
        return pd.DataFrame(columns = X_orig_test_df.columns), pd.Series(dtype=int)

    n_neg_target = int(round(n_pos_final * target_neg_multiplier)) # Calculate target number of negatives

    # Final positive samples are all original test positives
    X_pos_final_df = X_pos_all_from_orig_test
    y_pos_final_series = y_pos_all_from_orig_test

    # Handle negative samples
    X_neg_final_df = pd.DataFrame(columns = X_orig_test_df.columns) # Initialize empty
    y_neg_final_series = pd.Series(dtype = int)

    if n_neg_target > 0:
        if len(X_neg_unique_from_orig_test) == 0: # No unique negatives to sample from
            print(f"  Warning: No negative samples in original test data to oversample for target ratio 1:{target_neg_multiplier}.")
            # X_neg_final_df remains empty
        else:
            # Oversample from unique negatives with replacement
            # We need to sample indices to correctly get X and y
            sampled_neg_indices = rng.choice(X_neg_unique_from_orig_test.index, size = n_neg_target, replace = True)
            X_neg_final_df = X_neg_unique_from_orig_test.loc[sampled_neg_indices]
            y_neg_final_series = y_orig_test_series.loc[sampled_neg_indices] # Get original labels for these negatives

    # Combine positives and (potentially oversampled) negatives
    X_test_ratioed = pd.concat([X_pos_final_df, X_neg_final_df])
    y_test_ratioed = pd.concat([y_pos_final_series, y_neg_final_series])
    
    # Shuffle the combined test set
    if not X_test_ratioed.empty:
        shuffled_indices = rng.permutation(len(X_test_ratioed))
        X_test_ratioed = X_test_ratioed.iloc[shuffled_indices].reset_index(drop = True)
        y_test_ratioed = y_test_ratioed.iloc[shuffled_indices].reset_index(drop = True)
    
    actual_n_p, actual_n_n = sum(y_test_ratioed == 1), sum(y_test_ratioed == 0)
    actual_final_ratio_str = f"1:{actual_n_n/actual_n_p:.1f}" if actual_n_p > 0 else ("0P" if actual_n_n == 0 else "0P, ManyN")
    print(f"  Target Pos:Neg Ratio 1:{target_neg_multiplier} -> Actual Test Set: "
          f"{actual_n_p}P, {actual_n_n}N (Ratio: {actual_final_ratio_str}). "
          f"Negatives created from {len(X_neg_unique_from_orig_test)} unique negatives (oversampled if needed).")
          
    return X_test_ratioed, y_test_ratioed

def plot_roc_pr_with_ratios(metrics: dict, axes, cind, plot_title_detail, data_type = None) -> None:
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
    axes[0, cind].legend(loc = 'lower right', facecolor = 'white', fontsize = 7, frameon = True, fancybox = True)
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
