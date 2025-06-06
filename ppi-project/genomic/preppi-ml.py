# PPI Prediction Comparison with PrePPI (GO terms) Genomic ML Features
# Andrew Chung, hc893; 5/20/2025

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
  roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier

def compute_metrics(y_test, y_prob):
  if len(y_test) == 0:
    print("Warning: y_test is empty, cannot compute metrics.")
    return {'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 'precision': np.array([0,1]), 'recall': np.array([1,0]),
                'roc_auc': 0.0, 'pr_auc': 0.0, 'ap_score': 0.0}
  if len(np.unique(y_test)) < 2: # no distinct classes
    num_actual_positives = np.sum(y_test == 1)
    baseline_pr = num_actual_positives / len(y_test) if len(y_test) > 0 else 0.0
    print(f"Warning: y_test has only one class ({np.unique(y_test)[0]}). ROC-AUC set to 0.5; PR-AUC set to baseline ({baseline_pr:.3f}).")
    return {'fpr': np.array([0,1]), 'tpr': np.array([0,1]), 'precision': np.array([baseline_pr, baseline_pr]), 'recall': np.array([1,0]),
                'roc_auc': 0.5, 'pr_auc': baseline_pr, 'ap_score': baseline_pr}
  
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  precision, recall, _ = precision_recall_curve(y_test, y_prob)
  roc_auc = roc_auc_score(y_test, y_prob)
  ap_score = average_precision_score(y_test, y_prob)
  return {
    'fpr': fpr, 'tpr': tpr,
    'precision': precision, 'recall': recall,
    'roc_auc': roc_auc, 'pr_auc': ap_score, 'ap_score': ap_score
  }

def manual_resample_training_data(X_train, y_train, n_pos_target, n_neg_target, random_state = 893):
  # manual re-sampling of X and y training data to target positive/negative counts.
  rng = np.random.RandomState(random_state)
  # X_train: dataframe, y_train: series
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

def train_rf(X_train, y_train, X_test, y_test):
  rf = RandomForestClassifier(
    n_estimators = 500, 
    random_state = 893
  )
  rf.fit(X_train, y_train)
  y_prob = rf.predict_proba(X_test)[:, 1]
  return compute_metrics(y_test, y_prob), y_prob

def train_xgb(X_train, y_train, X_test, y_test):
  xgb = XGBClassifier(
    n_estimators = 500, 
    random_state = 893
  )
  xgb.fit(X_train, y_train)
  y_prob = xgb.predict_proba(X_test)[:, 1]
  return compute_metrics(y_test, y_prob), y_prob

def analyze_predictions(y_test, y_prob, model_name, test_set_desc):
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
    else: print(f"Thr {threshold:.1f}: 0 preds")

def check_feature_separability(data, feature_cols_rf, preppi_score_cols_present):
  # see R code in repository for Cohen's d values per-feature.
  # also can refer to Juheon's code compare.py for full details.
  pass

def plot_roc_pr(metrics, axes, col, plot_title_detail, data_type = None):
  # plot ROC curve
  sns.lineplot(x = metrics['fpr'], y = metrics['tpr'], ax = axes[0, col], label = f'{data_type} ({metrics["roc_auc"]:.3f})', linewidth = 1.5, ci = None)
  axes[0, col].set_title(f'Test set: {plot_title_detail}', fontsize = 11)
  axes[0, col].set_xlabel('False Positive Rate', fontsize = 9)
  axes[0, col].set_ylabel('True Positive Rate', fontsize = 9)
  axes[0, col].legend(loc = 'lower right', facecolor = 'white', fontsize = 11, frameon = True, fancybox = True)
  axes[0, col].plot([0, 1], [0, 1], linestyle = '--', color = 'gray', alpha = 0.5) # ROC diagonal

  # plot PR curve
  sns.lineplot(x = metrics['recall'], y = metrics['precision'], ax = axes[1, col], label = f'{data_type} ({metrics["pr_auc"]:.3f})', linewidth = 1.5, ci = None)
  axes[1, col].set_xlabel('Recall', fontsize = 9)
  axes[1, col].set_ylabel('Precision', fontsize = 9)

  # set labels, figure aesthetics
  axes[0, col].legend(loc = 'lower left', facecolor = 'white', fontsize = 7, frameon = True, fancybox = True)
  axes[1, col].legend(loc = 'best', facecolor = 'white', fontsize = 7, frameon = True, fancybox = True)
  axes[1, col].set_ylim([0.0, 1.05])
  axes[0, col].tick_params(axis = 'both', which = 'major', labelsize = 8)
  axes[1, col].tick_params(axis = 'both', which = 'major', labelsize = 8)

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

def main():

  print("Loading and processing data...")
  # Read in data sets
  try:
    af3 = pd.read_csv("ppi_features_max_avg.csv")
    preppi = pd.ExcelFile("Yilin_PrePPI_LRs_allclues.xlsx").parse(1)
  except FileNotFoundError as e:
    print(f"Error loading data files: {e}. Exiting.")
    return
  
  for name, input in [('af3', af3), ('preppi', preppi)]:
    df = input.copy()
    if 'ppi' not in df.columns:
      print(f"CRITICAL ERROR: 'ppi' column missing in {name}. Exiting.")
      return
    try:
      split_ppi = df['ppi'].astype(str).str.split(':', expand = True)
      df['protein1'] = split_ppi[0]
      df['protein2'] = split_ppi[1] if split_ppi.shape[1] > 1 else split_ppi[0]  # handle single-column case
      df['ppi_std'] = df.apply(lambda row: ':'.join(sorted([str(row['protein1']), str(row['protein2'])])), axis = 1)
      df.drop(columns = ['protein1', 'protein2', 'ppi'], inplace = True, errors = 'ignore')
      if name == 'af3': af3 = df
      else: preppi = df
    except Exception as e:
      print(f"Error standardizing PPIs in {name}: {e}")
      return
    
  data = pd.merge(af3, preppi, on = 'ppi_std', how = 'inner', suffixes = ('_af3', '_preppi')).rename(columns = {'ppi_std': 'ppi'})

  feature_cols_rf = ['co-expression', 'BP', 'CC', 'MF']
  preppi_score_cols = ['SM', 'PrP', 'max(SM,PrP)', 'PR', 'OR', 'PP', 'GO', 'EP', 'Total']

  # Handle label column after merge (prefer _af3 if suffixes exist)
  if 'label_af3' in data.columns: data['label'] = data['label_af3']
  elif 'label_preppi' in data.columns and 'label' not in data.columns : data['label'] = data['label_preppi']
  elif 'label' not in data.columns:
    print("CRITICAL ERROR: 'label' column is definitively missing. Exiting.")
    return
  
  all_expected_cols = ['ppi'] + feature_cols_rf + preppi_score_cols + ['label']
  final_cols_to_use = []
  for col_name in all_expected_cols:
    if col_name in data.columns:
      final_cols_to_use.append(col_name)
    elif col_name not in ['ppi', 'label']:
      print(f"Warning: Data column '{col_name}' not found. Will be created as 0.0.")
      data[col_name] = 0.0
      final_cols_to_use.append(col_name)
  data = data[final_cols_to_use]

  numeric_cols_to_clean = [col for col in feature_cols_rf + preppi_score_cols if col in data.columns]
  data[numeric_cols_to_clean] = data[numeric_cols_to_clean].replace('NULL', np.nan).fillna(0.0)
  for col in numeric_cols_to_clean:
    data[col] = pd.to_numeric(data[col], errors = 'coerce').fillna(0.0)

  data.dropna(subset=['label'], inplace=True) # Should not be needed if label handling is robust
  data['label'] = pd.to_numeric(data['label'], errors = 'coerce').fillna(-1).astype(int)
  data = data.query('label > -1')

  print(f"Data # rows after initial merge & type conversion: {len(data)}")
  if 'ppi' in data.columns:
    temp_ppi_split = data['ppi'].astype(str).str.split(':', expand=True)
    col1 = temp_ppi_split[0]
    col2 = temp_ppi_split[1] if temp_ppi_split.shape[1] > 1 else temp_ppi_split[0]
    homo_dimers = (col1 == col2)
    print(f"Homo-dimers found: {sum(homo_dimers)}")
    # data = data[~homo_dimers] # Uncomment to exclude

  present_preppi_score_cols = [col for col in preppi_score_cols if col in data.columns]
  if present_preppi_score_cols:
    data = data[~np.isinf(data[present_preppi_score_cols]).any(axis=1)]

  # initial dataset review
  print(f"\n=== DATASET OVERVIEW (After all cleaning & filtering) ===")
  total_pos = sum(data['label'] == 1)
  total_neg = sum(data['label'] == 0)

  if total_pos == 0 or len(data) < 2: # need 2+ samples, 1+ positives
    print("ERROR: No positive samples or insufficient data after all filtering. Exiting.")
    return
  print(f"Total samples for splitting: {len(data)} ({total_pos} P, {total_neg} N)")
  print(f"Overall P:N ratio: 1:{total_neg/total_pos:.1f}" if total_pos > 0 else "N/A (0 positives)")
  print(f"Overall Baseline precision: {total_pos/(total_pos + total_neg):.4f}" if (total_pos + total_neg) > 0 else "N/A")

  # check separability (Cohen's d, etc.) on final dataset -- pass (see R code.)
  # check_feature_separability(data, feature_cols_rf, present_preppi_score_cols)

  # train test split on original data set
  print("Performing 80:20 train test split...")
  feature_and_score_cols_present = [col for col in feature_cols_rf + present_preppi_score_cols if col in data.columns]
  if not feature_and_score_cols_present:
    print("ERROR: No feature or score columns available for X. Exiting.")
    return
  
  X_orig = data[feature_and_score_cols_present]
  y_orig = data['label']

  if len(X_orig) < 2:
    print("Error: Too few samples to perform train/test split. Exiting.")
    return
  
  try:
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
      X_orig, y_orig, test_size = 0.2, stratify = y_orig, random_state = 893
    )
  except ValueError as e:
    print(f"Error during train_test_split (likely due to too few samples for stratification): {e}. Exiting.")
    return
  
  print(f"Original Training set: {sum(y_orig_train == 1)} P, {sum(y_orig_train == 0)} N")
  print(f"Original Holdout Test set ({len(X_orig_test)} samples): {sum(y_orig_test == 1)} P, {sum(y_orig_test == 0)} N")

  print("\nCreating 1:1 balanced training set for Random Forest (undersampling negatives)...")
  n_pos_train_orig = sum(y_orig_train == 1)
  # Ensure feature_cols_rf are actually in X_orig_train before selecting
  X_orig_train_rf_features_only = X_orig_train[[col for col in feature_cols_rf if col in X_orig_train.columns]]

  if n_pos_train_orig == 0:
    print("ERROR: No positive samples in the original training set for RF. Cannot balance. Exiting.")
    return
  if X_orig_train_rf_features_only.empty:
    print("ERROR: No RF features columns found in X_orig_train. Exiting.")
    return
  if len(X_orig_train_rf_features_only) == 0: # If dataframe exists but is empty
    print("ERROR: RF feature DataFrame for training is empty. Exiting.")
    return
  
  # resampling step -- balanced training data set
  # sample random forest model training for RF features (coexp, BP, CC, MF).
  X_rf_train_balanced, y_rf_train_balanced = manual_resample_training_data(
    X_orig_train_rf_features_only, y_orig_train,
    n_pos_target = n_pos_train_orig, n_neg_target = n_pos_train_orig, 
    random_state = 893
  )
  print(f"Balanced RF training set: {sum(y_rf_train_balanced == 1)} P, {sum(y_rf_train_balanced == 0)} N")

  print("Training SINGLE Random Forest model on this balanced set...")
  if X_rf_train_balanced.empty or y_rf_train_balanced.empty:
      print("ERROR: Balanced training set is empty. Cannot train RF. Exiting")
      return
  single_rf_model = RandomForestClassifier(n_estimators = 500, random_state = 893)
  single_rf_model.fit(X_rf_train_balanced, y_rf_train_balanced)
  print("Single RF model trained.")

  # Now, train classifiers with varying ratios and feature spaces
  ratios = np.power(10, np.arange(4))  # 1:1, 1:10, 1:100, 1:1000
  
  sns.set_theme(style = 'darkgrid')
  fig, axes = plt.subplots(2, len(ratios), figsize = (6.5 * len(ratios), 10), squeeze = False, constrained_layout = True)
  fig.suptitle("RF (1:1 Balanced Train) vs PrePPI - Evaluated on Ratioed Test Sets (Negatives Oversampled)", fontsize = 14, fontweight = 'bold')

  for i, ratio in enumerate(ratios):

    plot_idx = i
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE ON TEST SET: Target Pos:Neg Ratio ~1:{ratio} (by Oversampling Negatives)")
    print(f"{'='*60}")

    X_test_ratioed, y_test_ratioed = create_ratioed_test_set_oversample_neg(
      X_orig_test, y_orig_test, target_neg_multiplier = ratio, random_state = 893 + i # slight seed variations
    )
    n_p_test_ratioed, n_n_test_ratioed = sum(y_test_ratioed == 1), sum(y_test_ratioed == 0)

    # Clear previous subplot content and redraw ROC diagonal
    axes[0, plot_idx].cla() 
    axes[1, plot_idx].cla() 
    axes[0, plot_idx].plot([0, 1], [0, 1], linestyle = '--', color = 'gray', alpha = 0.5)
    # plot aesthetics, objective information
    plot_title_detail_base = f"P:N ~1:{ratio}"
    plot_title_detail_counts = f"({n_p_test_ratioed}P:{n_n_test_ratioed}N)"
    plot_title_detail = f"{plot_title_detail_base}\n{plot_title_detail_counts}"

    # if generated test set is empty, skip this ratio
    if len(y_test_ratioed) == 0 or (n_p_test_ratioed == 0 and n_n_test_ratioed == 0):
      axes[0, plot_idx].set_title(f'Test Set: {plot_title_detail_base} (No Data)', fontsize=11)
      axes[0, plot_idx].text(0.5,0.5,'No Test Data',ha='center',va='center', fontsize=9)
      axes[1, plot_idx].text(0.5,0.5,'No Test Data',ha='center',va='center', fontsize=9)
      print(f"  Skipping ratio ~1:{ratio} as the generated test set is empty.")
      continue

    # split featur sets for ratioed test set
    current_test_rf_features = X_test_ratioed[[col for col in feature_cols_rf if col in X_test_ratioed.columns]]
    current_test_preppi_scores = X_test_ratioed[[col for col in present_preppi_score_cols if col in X_test_ratioed.columns]]

    print(f"\nEvaluating RF (trained on 1:1 data) on this test set...")
    if not current_test_rf_features.empty and not y_test_ratioed.empty:
      if len(np.unique(y_test_ratioed)) < 2 and not (n_p_test_ratioed == 0 and n_n_test_ratioed == 0) : # Only one class present but not empty
        print(f"  RF Evaluation: Test set for ratio 1:{ratio} has only one class. Metrics will be trivial.")
      y_prob_rf = single_rf_model.predict_proba(current_test_rf_features)[:, 1]
      metrics_rf = compute_metrics(y_test_ratioed, y_prob_rf)
      plot_roc_pr(metrics_rf, axes, plot_idx, plot_title_detail, data_type='RF (1:1 Train)')
      analyze_predictions(y_test_ratioed, y_prob_rf, 'RF (1:1 Train)', f"Test P:N ~1:{ratio}")
      print(f"AUPR: {metrics_rf['pr_auc']:.4f}")
    else: print("  Skipping RF evaluation: No RF features or empty test set for this ratio.")

    preppi_eval_components = [('PrePPI (Total)', 'Total'), ('PrePPI (GO)', 'GO'), ('PrePPI (EP)', 'EP'), ('PrePPI (avg(GO,EP))', None)]
    for label, col_name in preppi_eval_components:
      y_prob_preppi_component_eval = None
      valid_preppi_eval = True
      if label == 'PrePPI (avg(GO,EP))':
        if not ('GO' in current_test_preppi_scores.columns and 'EP' in current_test_preppi_scores.columns and \
          not current_test_preppi_scores['GO'].empty and not current_test_preppi_scores['EP'].empty):
          print(f"Skipping {label} for {plot_title_detail}: GO or EP scores missing/empty."); valid_preppi_eval = False
        else: y_prob_preppi_component_eval = (current_test_preppi_scores['GO'] + current_test_preppi_scores['EP']) / 2
      elif not(col_name and col_name in current_test_preppi_scores.columns and not current_test_preppi_scores[col_name].empty):
        print(f"Skipping {label} for {plot_title_detail}: score column '{col_name}' missing or empty."); valid_preppi_eval = False
      else: y_prob_preppi_component_eval = current_test_preppi_scores[col_name]
      
      if not valid_preppi_eval or y_prob_preppi_component_eval is None or y_prob_preppi_component_eval.empty or y_test_ratioed.empty:
        if valid_preppi_eval: print(f"Skipping {label} for {plot_title_detail}: No scores or test labels."); 
        continue

      print(f"\nEvaluating {label} on this test set ({plot_title_detail})...")
      if len(np.unique(y_test_ratioed)) < 2 and not (n_p_test_ratioed == 0 and n_n_test_ratioed == 0) :
        print(f"  {label} Evaluation: Test set for ratio 1:{ratio} has only one class. Metrics will be trivial.")
      metrics_preppi = compute_metrics(y_test_ratioed, y_prob_preppi_component_eval)
      plot_roc_pr(metrics_preppi, axes, plot_idx, plot_title_detail, data_type = label)
      analyze_predictions(y_test_ratioed, y_prob_preppi_component_eval, label, f"Test P:N ~1:{ratio}")
      print(f"AUPR: {metrics_preppi['pr_auc']:.4f}")
        
    # Finalize legends for this subplot column after all lines are added
    axes[0, plot_idx].legend(loc = 'lower right', facecolor = 'white', fontsize = 7, frameon = True, fancybox = True)
    axes[1, plot_idx].legend(loc = 'best', facecolor = 'white', fontsize = 7, frameon = True, fancybox = True)

  print(f"\n{'='*60}")
  print("Analysis complete. Saving plots...")
  try:
    plt.savefig('rf_vs_ppi_final.png', bbox_inches = 'tight', dpi = 300)
    plt.close()
    print("Plots saved as 'rf_vs_ppi_final.png'")
  except Exception as e: print(f"Error saving plot: {e}")

  print("\nComparison Summary:")
  print("- Single Random Forest: Trained on a 1:1 balanced training set (undersampled negatives).")
  print("- Evaluation: RF and PrePPI scores tested on four different test sets.")
  print("  Test sets created by keeping all original test positives and OVERSAMPLING original test negatives (with replacement)")
  print("  to achieve approximate Pos:Neg ratios of 1:1, 1:10, 1:100, and 1:1000.")
  print("- CAUTION: Oversampling test sets means evaluation includes duplicate negative instances for higher ratios.")

if __name__ == "__main__":
  main()