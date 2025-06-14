# PPI Predictions with ML Classifiers
# Andrew Chung, hc893; 5/17/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
  roc_curve, precision_recall_curve, auc
)
from xgboost import XGBClassifier

def main():

  # importing PPI Features Data
  print("Importing PPI Features Data...")
  data = pd.read_csv("ppi_features_max_avg.csv")
  data = data[data['label'] > -1]
  positives = data[data['label'] == 1]
  negatives = data[data['label'] == 0]
  print("Positives: {}, Negatives: {}".format(
    len(positives), len(negatives)
  ))

  fig, axes = plt.subplots(1, 2, figsize = (20, 10), constrained_layout = True)
  axes = axes.flatten()

  # Train Test Split
  X = data[['co-expression', 'BP', 'CC', 'MF']].fillna(0.0)
  y = data['label']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

  # Model 1 = Random Forest
  print("Training Random Forest...")
  rf = RandomForestClassifier(
    n_estimators = 500,
    random_state = 893
  )
  rf.fit(X_train, y_train)
  y_prob_rf = rf.predict_proba(X_test)[:, 1]

  # Model 2 = SVM
  print("Training SVM...")
  svc = SVC(
    probability = True,
    random_state = 893
  )
  svc.fit(
    StandardScaler().fit_transform(X_train), y_train
  )
  y_prob_svc = svc.predict_proba(X_test)[:, 1]

  # Model 3 = XGBoost
  print("Training XGBoost...")
  xgb = XGBClassifier(
    n_estimators = 100,
    random_state = 893
  )
  xgb.fit(X_train, y_train)
  y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

  # Model 4 = kNN (with GridSearchCV)
  print("Training kNN...")
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
  y_prob_knn = knn.best_estimator_.predict_proba(
    StandardScaler().fit_transform(X_test)
  )[:, 1]

  # Model 5 = Naive Bayes
  print("Training Naive Bayes...")
  nb = GaussianNB()
  nb.fit(X_train, y_train)
  y_prob_nb = nb.predict_proba(X_test)[:, 1]
  
  # Compile Predictions, Compute Metrics
  print("Computing Metrics...")
  y_probs = {
    'Random Forest': y_prob_rf,
    'SVM': y_prob_svc,
    'XGBoost': y_prob_xgb,
    'kNN': y_prob_knn,
    'Naive Bayes': y_prob_nb
  }
  metrics = {
    pred : {
      'fpr': None, 'tpr': None,
      'precision': None, 'recall': None,
      'roc_auc': None, 'pr_auc': None
    } for pred in y_probs.keys()
  }
  for pred in y_probs.keys():
    fpr, tpr, _ = roc_curve(y_test, y_probs[pred])
    precision, recall, _ = precision_recall_curve(y_test, y_probs[pred])
    roc_auc = roc_auc_score(y_test, y_probs[pred])
    pr_auc = auc(recall, precision)
    metrics[pred].update({
      'fpr': fpr, 'tpr': tpr,
      'precision': precision, 'recall': recall,
      'roc_auc': roc_auc, 'pr_auc': pr_auc
    })

  # Check Uniqueness of Protein Pairs in Training and Test Sets
  unique_train = set(X_train.index)
  unique_test = set(X_test.index)
  print("Overlap between train and test sets: {}".format(
    len(unique_train.intersection(unique_test))
  ))

  # Plot ROC Curves
  print("Plotting ROC Curves...")
  axes[0].plot(
    [0, 1], [0, 1], linestyle = '--', color = 'gray'
  )
  for pred in y_probs.keys():
    axes[0].plot(
      metrics[pred]['fpr'], metrics[pred]['tpr'],
      label = '{} (AUC = {:.3f})'.format(
        pred, metrics[pred]['roc_auc']
      )
    )
  axes[0].set_title("ROC Curve")
  axes[0].set_xlabel("False Positive Rate", fontsize = 10)
  axes[0].set_ylabel("True Positive Rate", fontsize = 10)
  axes[0].legend(fontsize = 8, loc = 'lower right')
  
  # Plot PR Curves
  print("Plotting PR Curves...")
  for pred in y_probs.keys():
    axes[1].plot(
      metrics[pred]['recall'], metrics[pred]['precision'],
      label = '{} (AUC = {:.3f})'.format(
        pred, metrics[pred]['pr_auc']
      )
    )
  axes[1].set_title("PR Curve")
  axes[1].set_xlabel("Recall", fontsize = 10)
  axes[1].set_ylabel("Precision", fontsize = 10)
  axes[1].legend(fontsize = 8, loc = 'lower left')
  
  plt.savefig("roc_pr_curves_may17.png", bbox_inches = 'tight')
  plt.close()

if __name__ == "__main__":
  main()