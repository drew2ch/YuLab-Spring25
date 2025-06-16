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

