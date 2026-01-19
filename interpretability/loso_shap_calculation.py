#!/usr/bin/env python3
"""
SHAP Value Calculation for LOSO Setting
========================================
Calculate SHAP values for ElasticNet, Random Forest, and TabPFN models
using Leave-One-Study-Out (LOSO) cross-validation on genus-level data.

For each cohort:
- Train on all other cohorts
- Calculate SHAP values for all samples in the test cohort
- Save results as matrices

Preprocessing: None (uses raw data as-is)
"""

import os
os.environ["SCIPY_ARRAY_API"] = "1"

import argparse
import warnings
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
import torch

# ---------- Configuration ----------
RANDOM_STATE = 42
BG_MAX = 128         # Background samples for SHAP
NSAMPLES = 50        # Kernel SHAP sampling strength

# Random Forest hyperparameter grid (same as classicML.py)
RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "max_features": ["sqrt", 0.5],
    "min_samples_split": [2, 5],
}

# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description="Calculate SHAP values for LOSO setting")
parser.add_argument("--disease", type=str, required=True, help="Disease name, e.g., 'PD'")
parser.add_argument("--data_type", type=str, required=True, help="Data type, e.g., 'Amplicon' or 'Metagenomics'")
parser.add_argument("--tax_level", type=str, default="genus", help="Taxonomic level (default: genus)")
args = parser.parse_args()

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

# ---------- Paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "filterd_data")
OUT_DIR = os.path.join(PROJECT_DIR, "interpretability", "shap_results",
                       f"{args.disease}_{args.data_type}_{args.tax_level}_loso")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"SHAP Calculation for {args.disease} - {args.data_type} - {args.tax_level}")
print(f"{'='*60}\n")

# ---------- Utils ----------
def get_features_and_labels(df: pd.DataFrame, label_col: str = "Group", label_encoder=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract feature columns and labels."""
    feature_cols = [c for c in df.columns if c.startswith("ncbi")]
    X = df[feature_cols].copy()
    y_raw = df[label_col].values

    # Encode labels if encoder is provided
    if label_encoder is not None:
        y = label_encoder.transform(y_raw)
    else:
        y = y_raw

    return X, y


# ---------- Load Data ----------
filename = f"{args.disease}_{args.data_type}_{args.tax_level}.csv"
csv_path = os.path.join(DATA_DIR, filename)
print(f"Loading data from: {csv_path}")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Data file not found: {csv_path}")

data = pd.read_csv(csv_path, index_col=0)
print(f"Data shape: {data.shape}")

# Get cohort information
pair_ranks = np.sort(pd.unique(data["pair_rank"]))
print(f"Found {len(pair_ranks)} cohorts: {pair_ranks}")

# Check classification type
unique_labels = data["Group"].unique()
n_classes = len(unique_labels)
print(f"Number of classes: {n_classes} ({unique_labels})")

if n_classes != 2:
    raise ValueError(f"Expected binary classification, but found {n_classes} classes. "
                     f"This script is designed for binary classification only.")

# Encode labels to 0/1
le = LabelEncoder()
le.fit(data["Group"])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Label mapping: {label_mapping}")

# Define target class for SHAP - always use "Case"
POS_LABEL = "Case"
if POS_LABEL not in le.classes_:
    raise ValueError(f"Target label '{POS_LABEL}' not found in classes: {le.classes_}")

# Get the encoded value for "Case"
pos_label_encoded = le.transform([POS_LABEL])[0]
print(f"Target class for SHAP: '{POS_LABEL}' (encoded as {pos_label_encoded})")

# GPU check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------- LOSO SHAP Calculation ----------
# Storage for results
all_shap_elasticnet = []
all_shap_rf = []
all_shap_tabpfn = []
all_samples_info = []
all_feature_names = None

rng = np.random.default_rng(RANDOM_STATE)

for test_rank in tqdm(pair_ranks, desc="LOSO cohorts"):
    print(f"\n{'─'*60}")
    print(f"Test Cohort: {test_rank}")
    print(f"{'─'*60}")

    # Split data
    train_df = data[data["pair_rank"] != test_rank]
    test_df = data[data["pair_rank"] == test_rank]

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Extract features and labels (with encoding)
    X_train_raw, y_train = get_features_and_labels(train_df, label_encoder=le)
    X_test_raw, y_test = get_features_and_labels(test_df, label_encoder=le)

    # Store feature names (should be same across all folds)
    if all_feature_names is None:
        all_feature_names = np.array(X_train_raw.columns)

    # Use raw data (no transformation)
    X_train = X_train_raw.values.astype(np.float64)
    X_test = X_test_raw.values.astype(np.float64)

    # Create background samples for SHAP (from training set)
    bg_size = min(BG_MAX, len(X_train))
    bg_idx = rng.choice(len(X_train), size=bg_size, replace=False)
    background = X_train[bg_idx]

    # We will explain ALL test samples
    X_explain = X_test
    n_explain = len(X_explain)
    print(f"Explaining {n_explain} test samples with {bg_size} background samples")

    # Store sample information
    for idx in test_df.index:
        all_samples_info.append({
            "sample_id": idx,
            "cohort": test_rank,
            "true_label": test_df.loc[idx, "Group"]
        })

    # ============================================================
    # ElasticNet
    # ============================================================
    print("\n[1/3] Training ElasticNet...")
    try:
        elasticnet = LogisticRegressionCV(
            cv=3, penalty="elasticnet", solver="saga", scoring="roc_auc",
            max_iter=1000, l1_ratios=[0.5], random_state=RANDOM_STATE
        )
        elasticnet.fit(X_train, y_train)

        # Find the index of POS_LABEL in this model's classes
        pos_idx_enet = int(np.where(elasticnet.classes_ == pos_label_encoded)[0][0])
        print(f"  ElasticNet class order: {elasticnet.classes_}, target index: {pos_idx_enet}")

        # SHAP using LinearExplainer
        print("  Calculating SHAP values...")
        enet_explainer = shap.LinearExplainer(
            elasticnet, background, feature_perturbation="interventional"
        )
        shap_enet_raw = enet_explainer.shap_values(X_explain)

        # For binary classification, select target class
        if isinstance(shap_enet_raw, list):
            shap_enet = np.atleast_2d(shap_enet_raw[pos_idx_enet])
        else:
            shap_enet = np.atleast_2d(shap_enet_raw)
        all_shap_elasticnet.append(shap_enet)
        print(f"  ✓ ElasticNet SHAP shape: {shap_enet.shape}")

    except Exception as e:
        print(f"  ✗ ElasticNet failed: {e}")
        all_shap_elasticnet.append(np.full((n_explain, len(all_feature_names)), np.nan))

    # ============================================================
    # Random Forest
    # ============================================================
    print("\n[2/3] Training Random Forest...")
    try:
        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        rf_search = RandomizedSearchCV(
            rf, RF_PARAM_GRID, scoring="roc_auc", n_iter=10, cv=3,
            n_jobs=-1, random_state=RANDOM_STATE
        )
        rf_search.fit(X_train, y_train)

        # Use the best estimator for SHAP
        best_rf = rf_search.best_estimator_

        # Find the index of POS_LABEL in this model's classes
        pos_idx_rf = int(np.where(best_rf.classes_ == pos_label_encoded)[0][0])
        print(f"  Random Forest class order: {best_rf.classes_}, target index: {pos_idx_rf}")

        # SHAP using TreeExplainer
        print("  Calculating SHAP values...")
        rf_explainer = shap.TreeExplainer(
            best_rf, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        rf_shap_raw = rf_explainer.shap_values(X_explain)

        # For binary classification, select target class
        if isinstance(rf_shap_raw, list):
            shap_rf = np.atleast_2d(rf_shap_raw[pos_idx_rf])
        elif hasattr(rf_shap_raw, 'ndim') and rf_shap_raw.ndim == 3:
            # Handle 3D output: (n_samples, n_features, n_classes)
            shap_rf = np.atleast_2d(rf_shap_raw[:, :, pos_idx_rf])
        else:
            shap_rf = np.atleast_2d(rf_shap_raw)
        all_shap_rf.append(shap_rf)
        print(f"  ✓ Random Forest SHAP shape: {shap_rf.shape}")

    except Exception as e:
        print(f"  ✗ Random Forest failed: {e}")
        all_shap_rf.append(np.full((n_explain, len(all_feature_names)), np.nan))

    # ============================================================
    # TabPFN
    # ============================================================
    print("\n[3/3] Training TabPFN...")
    try:
        tabpfn = TabPFNClassifier(
            random_state=RANDOM_STATE, device=device, ignore_pretraining_limits=True
        )
        tabpfn.fit(X_train, y_train)

        # Find the index of POS_LABEL in this model's classes
        pos_idx_tabpfn = int(np.where(tabpfn.classes_ == pos_label_encoded)[0][0])
        print(f"  TabPFN class order: {tabpfn.classes_}, target index: {pos_idx_tabpfn}")

        # SHAP using KernelExplainer
        print("  Calculating SHAP values (this may take a while)...")

        def prob_target_tabpfn(X_in):
            P = tabpfn.predict_proba(np.asarray(X_in, dtype=np.float64))
            return P[:, pos_idx_tabpfn]

        tabpfn_explainer = shap.KernelExplainer(
            prob_target_tabpfn, background, link="logit"
        )
        shap_tabpfn = tabpfn_explainer.shap_values(
            X_explain, nsamples=NSAMPLES, feature_selection="none"
        )

        shap_tabpfn = np.atleast_2d(shap_tabpfn)
        all_shap_tabpfn.append(shap_tabpfn)
        print(f"  ✓ TabPFN SHAP shape: {shap_tabpfn.shape}")

    except Exception as e:
        print(f"  ✗ TabPFN failed: {e}")
        all_shap_tabpfn.append(np.full((n_explain, len(all_feature_names)), np.nan))

# ---------- Concatenate All Results ----------
print(f"\n{'='*60}")
print("Concatenating results...")
print(f"{'='*60}")

# Stack all SHAP values into single matrices
shap_elasticnet_all = np.vstack(all_shap_elasticnet)  # (n_total_samples, n_features)
shap_rf_all = np.vstack(all_shap_rf)
shap_tabpfn_all = np.vstack(all_shap_tabpfn)

print(f"ElasticNet SHAP matrix: {shap_elasticnet_all.shape}")
print(f"Random Forest SHAP matrix: {shap_rf_all.shape}")
print(f"TabPFN SHAP matrix: {shap_tabpfn_all.shape}")

# Create sample info DataFrame
samples_df = pd.DataFrame(all_samples_info)
print(f"Sample info: {samples_df.shape}")

# ---------- Save Results ----------
print(f"\nSaving results to: {OUT_DIR}")

# Save SHAP values as numpy arrays
np.save(os.path.join(OUT_DIR, "shap_elasticnet.npy"), shap_elasticnet_all)
np.save(os.path.join(OUT_DIR, "shap_rf.npy"), shap_rf_all)
np.save(os.path.join(OUT_DIR, "shap_tabpfn.npy"), shap_tabpfn_all)

# Save feature names
np.save(os.path.join(OUT_DIR, "feature_names.npy"), all_feature_names)

# Save sample information
samples_df.to_csv(os.path.join(OUT_DIR, "sample_info.csv"), index=False)

# Save SHAP values as CSV for easy inspection
shap_elasticnet_df = pd.DataFrame(shap_elasticnet_all, columns=all_feature_names)
shap_rf_df = pd.DataFrame(shap_rf_all, columns=all_feature_names)
shap_tabpfn_df = pd.DataFrame(shap_tabpfn_all, columns=all_feature_names)

shap_elasticnet_df.to_csv(os.path.join(OUT_DIR, "shap_elasticnet.csv"), index=False)
shap_rf_df.to_csv(os.path.join(OUT_DIR, "shap_rf.csv"), index=False)
shap_tabpfn_df.to_csv(os.path.join(OUT_DIR, "shap_tabpfn.csv"), index=False)

# ---------- Global Feature Importance ----------
def compute_global_importance(shap_matrix: np.ndarray, feature_names: np.ndarray) -> pd.DataFrame:
    """Compute global feature importance from SHAP values."""
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp_df

imp_elasticnet = compute_global_importance(shap_elasticnet_all, all_feature_names)
imp_rf = compute_global_importance(shap_rf_all, all_feature_names)
imp_tabpfn = compute_global_importance(shap_tabpfn_all, all_feature_names)

imp_elasticnet.to_csv(os.path.join(OUT_DIR, "global_importance_elasticnet.csv"), index=False)
imp_rf.to_csv(os.path.join(OUT_DIR, "global_importance_rf.csv"), index=False)
imp_tabpfn.to_csv(os.path.join(OUT_DIR, "global_importance_tabpfn.csv"), index=False)

# ---------- Summary ----------
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"Total samples processed: {len(samples_df)}")
print(f"Number of features: {len(all_feature_names)}")
print(f"Number of cohorts: {len(pair_ranks)}")
print(f"\nTop 10 features by mean |SHAP| (ElasticNet):")
print(imp_elasticnet.head(10))

# Save metadata
meta_path = os.path.join(OUT_DIR, "metadata.txt")
with open(meta_path, "w") as f:
    f.write(f"LOSO SHAP Calculation Results\n")
    f.write(f"{'='*60}\n")
    f.write(f"Disease: {args.disease}\n")
    f.write(f"Data type: {args.data_type}\n")
    f.write(f"Taxonomic level: {args.tax_level}\n")
    f.write(f"Number of cohorts: {len(pair_ranks)}\n")
    f.write(f"Cohorts: {pair_ranks}\n")
    f.write(f"Total samples: {len(samples_df)}\n")
    f.write(f"Number of features: {len(all_feature_names)}\n")
    f.write(f"Label mapping: {label_mapping}\n")
    f.write(f"Target class: '{POS_LABEL}' (encoded as {pos_label_encoded})\n")
    f.write(f"Preprocessing: None (raw data)\n")
    f.write(f"Random state: {RANDOM_STATE}\n")
    f.write(f"Background samples: {BG_MAX}\n")
    f.write(f"Kernel SHAP samples: {NSAMPLES}\n")
    f.write(f"\nOutput files:\n")
    f.write(f"  - shap_elasticnet.npy / .csv\n")
    f.write(f"  - shap_rf.npy / .csv\n")
    f.write(f"  - shap_tabpfn.npy / .csv\n")
    f.write(f"  - feature_names.npy\n")
    f.write(f"  - sample_info.csv\n")
    f.write(f"  - global_importance_*.csv\n")

print(f"\n✅ All results saved to: {OUT_DIR}")
print(f"   - SHAP matrices (numpy): shap_*.npy")
print(f"   - SHAP matrices (CSV): shap_*.csv")
print(f"   - Feature names: feature_names.npy")
print(f"   - Sample info: sample_info.csv")
print(f"   - Global importance: global_importance_*.csv")
print(f"   - Metadata: metadata.txt")
print(f"\n{'='*60}\n")
