# PPI Prediction Comparison with PrePPI (GO terms) Genomic ML Features
# Andrew Chung, hc893; 5/20/2025

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
  roc_auc_score, roc_curve, precision_recall_curve, auc
)
from xgboost import XGBClassifier

def compute_metrics(y_test, y_prob):
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  precision, recall, _ = precision_recall_curve(y_test, y_prob)
  roc_auc = roc_auc_score(y_test, y_prob)
  pr_auc = auc(recall, precision)
  return {
    'fpr': fpr, 'tpr': tpr,
    'precision': precision, 'recall': recall,
    'roc_auc': roc_auc, 'pr_auc': pr_auc
  }

def train_rf(X_train, y_train, X_test, y_test, class_weight = 1):
  rf = RandomForestClassifier(
    n_estimators = 500, 
    random_state = 893,
    class_weight = {0: class_weight, 1: 1}
  )
  rf.fit(X_train, y_train)
  y_prob = rf.predict_proba(X_test)[:, 1]
  return compute_metrics(y_test, y_prob)

def train_xgb(X_train, y_train, X_test, y_test, class_weight = 1):
  xgb = XGBClassifier(
    n_estimators = 500, 
    random_state = 893,
    scale_pos_weight = class_weight
  )
  xgb.fit(X_train, y_train)
  y_prob = xgb.predict_proba(X_test)[:, 1]
  return compute_metrics(y_test, y_prob)

# below classifiers are not used
'''
def train_svc(X_train, y_train, X_test, y_test):
  svc = SVC(
    probability = True, 
    random_state = 893
  )
  svc.fit(StandardScaler().fit_transform(X_train), y_train)
  y_prob = svc.predict_proba(X_test)[:, 1]
  return compute_metrics(y_test, y_prob)

def train_knn(X_train, y_train, X_test, y_test, cv = False):
  if not cv:
    knn = KNeighborsClassifier(n_neighbors = 5) # gridcv ?
    knn.fit(
      StandardScaler().fit_transform(X_train), y_train
    )
    y_prob = knn.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_prob)
  else:
    knn = GridSearchCV(
      estimator = KNeighborsClassifier(),
      param_grid = {
        'n_neighbors': np.arange(3, 13, 2)
      },
      scoring = 'roc_auc', cv = 5
    )
    knn.fit(
      StandardScaler().fit_transform(X_train), y_train
    )
    y_prob = knn.best_estimator_.predict_proba(
      StandardScaler().fit_transform(X_test)
    )[:, 1]
    return compute_metrics(y_test, y_prob)
  
def train_nb(X_train, y_train, X_test, y_test):
  nb = GaussianNB()
  nb.fit(X_train, y_train)
  y_prob = nb.predict_proba(X_test)[:, 1]
  return compute_metrics(y_test, y_prob)
'''

def plot_roc_pr(metrics, axes, col, n_pos, n_neg, data_type = None):
  # plot ROC curve
  sns.lineplot(x = metrics['fpr'], y = metrics['tpr'], ax = axes[0, col], label = f'{data_type} ({metrics["roc_auc"]:.2f})')
  axes[0, col].set_title(f'Interacting ({n_pos}) vs Non-interacting ({n_neg})', fontsize = 12)
  axes[0, col].set_xlabel('False Positive Rate', fontsize = 10)
  axes[0, col].set_ylabel('True Positive Rate', fontsize = 10)
  axes[0, col].legend(loc = 'lower right', fontsize = 8)

  # plot PR curve
  sns.lineplot(x = metrics['recall'], y = metrics['precision'], ax = axes[1, col], label = f'{data_type} ({metrics["pr_auc"]:.2f})')
  axes[1, col].set_title(f'Interacting ({n_pos}) vs Non-interacting ({n_neg})', fontsize = 12)
  axes[1, col].set_xlabel('Recall', fontsize = 10)
  axes[1, col].set_ylabel('Precision', fontsize = 10)
  axes[1, col].legend(loc = 'lower right', fontsize = 8)


def main():

  print("Loading and processing data...")
  # Read in data sets
  af3_data = pd.read_csv("ppi_features_max_avg.csv")
  preppi_data = pd.ExcelFile("Yilin_PrePPI_LRs_allclues.xlsx").parse(1)

  # Before training any classifiers, I need to identify common PPI pairs between the two data sets
  # To ensure uniqueness of PPI pairs (i.e. no ordered duplicates), I will check for lexicographical ordering
  af3_data[['protein1', 'protein2']] = af3_data['ppi'].str.split(':', expand = True)
  assert (af3_data.protein1 <= af3_data.protein2).all() & (preppi_data.protein1 <= preppi_data.protein2).all(), "Error: inconsistent order pairs"
  af3_data = af3_data.drop(columns = ['protein1', 'protein2'])
  preppi_data = preppi_data.drop(columns = ['protein1', 'protein2'])

  # merge data sets, preserve only PPI pairs common to both data sets.
  data = pd.merge(af3_data, preppi_data, on = 'ppi', how = 'inner').query('label > -1')[[
    'ppi', 'co-expression', 'BP', 'CC', 'MF', 'SM', 'PrP', 'max(SM,PrP)', 
    'PR', 'OR', 'PP', 'GO', 'EP', 'Total', 'label'
  ]].replace('NULL', 0).fillna(0.0) # ignore label = -1

  preppi_labels = np.array(['SM', 'PrP', 'max(SM,PrP)', 'PR', 'OR', 'PP', 'GO', 'EP', 'Total'])
  data = data[~np.isinf(data[preppi_labels]).any(axis = 1)]

  print("Training classifiers...")

  # define figure, axes
  fig, axes = plt.subplots(2, 4, figsize = (20, 10), constrained_layout = True)
  fig.suptitle("Traditional ML Classifiers vs. PrePPI", fontsize = 16, fontweight = 'bold')
  sns.set_theme(style = 'darkgrid')

  # Train Models, plot ROC/PR curves
  # ratios: 1:1, 1:10, 1:100, 1:1000
  ratios = np.power(10, np.arange(4))
  for i, ratio in enumerate(ratios):

    print(f"Training on ratio 1:{ratio:.0f}...\n-----------------------------------")

    positives = data[data['label'] == 1]
    negatives = data[data['label'] == 0]

    n_pos = len(positives)
    n_neg = int(n_pos * ratio)

    if ratio == 1: # no class weighting
      neg_sample = negatives.sample(n = n_neg, replace = False, random_state = 893)
      class_weight = 1
    else: # need class weighting
      neg_sample = negatives
      class_weight = n_neg/len(negatives)
    data_sample = pd.concat([positives, neg_sample])

    # train test split
    X = data_sample[[
      'co-expression', 'BP', 'CC', 'MF', 'SM', 'PrP', 'max(SM,PrP)', 
      'PR', 'OR', 'PP', 'GO', 'EP', 'Total'
    ]]
    y = data_sample['label']
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size = 0.2, stratify = y, random_state = 893
    )

    # partition data sets into 3 groups
    # 1. Co-expression, BP, CC, MF
    # 2. GO, EP; 3. Total
    train1, train2, train3 = X_train[['co-expression', 'BP', 'CC', 'MF']], X_train[['GO', 'EP']], X_train[preppi_labels]
    test1, test2, test3 = X_test[['co-expression', 'BP', 'CC', 'MF']], X_test[['GO', 'EP']], X_test[preppi_labels]
    train_test_pairs = {
      'Co-exp, BP, MF, CC': [train1, test1],
      'PrePPI (GO, EP)': [train2, test2],
      'PrePPI (overall)': [train3, test3]
    }
    axes[0, i].plot([0, 1], [0, 1], linestyle = '--', color = 'gray', label = 'Random Guess')

    for mod, sets in train_test_pairs.items():
      
      # Random Forest Classifier
      print(f"Training on {mod}")
      metrics_rf = train_rf(sets[0], y_train, sets[1], y_test, class_weight = class_weight)
      # XGBoost Classifier
      print(f"Training on {mod}")
      metrics_xgb = train_xgb(sets[0], y_train, sets[1], y_test, class_weight = class_weight)

      '''
      # train Naive Bayes Classifier
      print(f"Naive Bayes: {mod}")
      metrics_nb = train_nb(sets[0], y_train, sets[1], y_test)
      '''

      # Plot ROC/PR
      plot_roc_pr(metrics_rf, axes, i, n_pos, n_neg, data_type = mod)
      plot_roc_pr(metrics_xgb, axes, i, n_pos, n_neg, data_type = mod)
      # plot_roc_pr(metrics_nb, axes, i, n_pos, n_neg, data_type = mod, classifier = 'Naive Bayes')

    print("-----------------------------------")

  print("Training classifiers complete. Saving plots...")
  plt.savefig('preppi_ml.png', bbox_inches = 'tight')
  plt.close()
  print("Plots saved.")

if __name__ == "__main__":
  main()