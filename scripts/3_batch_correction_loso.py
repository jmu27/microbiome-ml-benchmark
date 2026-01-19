#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Effect Correction for LOSO (Leave-One-Study-Out) Cross-Validation
Compares: ComBat, MMUPHin, DebiasM
Models: ElasticNet, RandomForest, TabPFN
Adapted for Micro_bench project structure
"""

import os
os.environ['SCIPY_ARRAY_API'] = '1'

import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import tempfile
from tabpfn import TabPFNClassifier
from debiasm import DebiasMClassifier

warnings.filterwarnings("ignore")

# Check GPU availability for TabPFN
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- Config ----------
parser = argparse.ArgumentParser(description="Batch correction for LOSO validation")
parser.add_argument("--disease", type=str, required=True, help="Disease name, e.g., 'IBD'")
parser.add_argument("--data_type", type=str, required=True, help="Data type, e.g., 'Metagenomics'")
parser.add_argument("--tax_level", type=str, required=True, help="Taxonomic level, e.g., 'genus' or 'species'")
parser.add_argument("--batch_col", type=str, default="pair_rank", help="Batch column name (default: 'pair_rank')")
parser.add_argument("--label_col", type=str, default="Group", help="Label column name (default: 'Group')")
parser.add_argument("--methods", type=str, default="DebiasM,ComBat,MMUPHin", help="Batch correction methods, comma-separated (default: 'DebiasM,ComBat,MMUPHin')")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs for models (default: -1, all CPUs)")
parser.add_argument("--random_state", type=int, default=42, help="Random state (default: 42)")
parser.add_argument("--preprocess_method", type=str, default="none", help="Preprocessing method: 'none', 'log_std', 'CLR', 'ALR' (default: 'none')")
args = parser.parse_args()

# Constants
disease = args.disease
data_type = args.data_type
tax_level = args.tax_level
batch_col = args.batch_col
label_col = args.label_col
n_jobs = args.n_jobs
random_state = args.random_state
methods = [m.strip() for m in args.methods.split(',')]
preprocess_method = args.preprocess_method

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
rawdata_dir = PROJECT_DIR / "data" / "filterd_data"
output_dir = PROJECT_DIR / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# Use the batch_correct.R script from the scripts directory
batch_correct_script = SCRIPT_DIR / "batch_correct.R"
if not batch_correct_script.exists():
    raise FileNotFoundError(f"batch_correct.R not found at {batch_correct_script}")

# RandomForest parameter grid (same as classicML.py)
RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "max_features": ["sqrt", 0.5],
    "min_samples_split": [2, 5],
}

# ---------- Load Data ----------
filename = f"{disease}_{data_type}_{tax_level}.csv"
csv_path = rawdata_dir / filename
print(f"\n{'='*60}")
print(f"Loading: {filename}")
print(f"{'='*60}\n")

if not csv_path.exists():
    raise FileNotFoundError(f"Data file not found: {csv_path}")

data = pd.read_csv(csv_path, index_col=0)

# Check required columns
if batch_col not in data.columns:
    raise ValueError(f"Batch column '{batch_col}' not found in data. Available: {data.columns.tolist()}")
if label_col not in data.columns:
    raise ValueError(f"Label column '{label_col}' not found in data. Available: {data.columns.tolist()}")

# Get feature columns
feature_cols = [col for col in data.columns if col.startswith("ncbi")]
if not feature_cols:
    raise ValueError("No feature columns starting with 'ncbi' found in data")

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(data)}")
print(f"Batch column: {batch_col}")
print(f"Label column: {label_col}")
print(f"Device: {DEVICE}")
print(f"Methods: {', '.join(methods)}")

# Get unique studies (batches)
studies = sorted(data[batch_col].unique())
n_studies = len(studies)
print(f"Number of studies: {n_studies}")
print(f"Studies: {studies}\n")

# Check if LOSO is feasible
if n_studies < 3:
    raise ValueError(f"LOSO requires at least 3 studies, but found {n_studies}")

# ---------- Preprocessing Functions ----------

def log_std_normalization(X: pd.DataFrame, log_n0: float = 1e-6, sd_min_q: float = 0.1) -> np.ndarray:
    """
    Log-transform and standardize following SIAMCAT's log.std method.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × features)
    log_n0 : float
        Pseudocount to add before log transformation (default: 1e-6)
    sd_min_q : float
        Quantile of standard deviations to use as minimum denominator (default: 0.1)

    Returns
    -------
    np.ndarray
        Normalized feature matrix
    """
    # Log transform with pseudocount
    feat_log = np.log10(X.values + log_n0)

    # Calculate mean and std for each feature (across samples)
    m = feat_log.mean(axis=0)  # mean of each feature
    s = feat_log.std(axis=0, ddof=1)  # std of each feature

    # Calculate quantile of standard deviations
    q = np.quantile(s, sd_min_q)

    # Ensure q > 0
    if q <= 0:
        q = 1e-8
        print(f"  ⚠️  Warning: Quantile of std is {np.quantile(s, sd_min_q):.2e}, using q={q:.2e}")

    # Normalize: (feat.log - m) / (s + q)
    feat_norm = (feat_log - m) / (s + q)

    return feat_norm


def clr_transform(X: pd.DataFrame, pseudocount: float = 1e-5) -> np.ndarray:
    """
    Centered log-ratio (CLR) transform.
    """
    X_rel = X.values + pseudocount
    X_log = np.log(X_rel)
    gm = np.exp(np.mean(X_log, axis=1, keepdims=True))  # geometric mean per sample
    return np.log(X_rel / gm)


def alr_transform(X: pd.DataFrame, reference_index: int = -1, pseudocount: float = 1e-5) -> np.ndarray:
    """
    Additive log-ratio (ALR) transform relative to a reference component.
    """
    x = X.values.astype(np.float64) + pseudocount  # avoid log(0)
    ref = x[:, reference_index].reshape(-1, 1)
    numerator = np.delete(x, reference_index, axis=1)
    return np.log(numerator / ref)


def apply_preprocessing(X: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply the selected preprocessing method.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    method : str
        Preprocessing method: 'none', 'log_std', 'CLR', 'ALR'

    Returns
    -------
    pd.DataFrame
        Preprocessed feature matrix
    """
    if method == "none":
        return X.copy()
    elif method == "log_std":
        X_prep = log_std_normalization(X)
        return pd.DataFrame(X_prep, columns=X.columns, index=X.index)
    elif method == "CLR":
        X_prep = clr_transform(X)
        return pd.DataFrame(X_prep, columns=X.columns, index=X.index)
    elif method == "ALR":
        X_prep = alr_transform(X)
        # ALR reduces dimension by 1 (removes reference)
        new_cols = [col for i, col in enumerate(X.columns) if i != len(X.columns) - 1]
        return pd.DataFrame(X_prep, columns=new_cols, index=X.index)
    else:
        raise ValueError(f"Unsupported preprocessing method: {method}")


# ---------- Helper Functions ----------

def run_batch_correction(method, train_csv, test_csv, batch_col, label_col, temp_dir):
    """
    Run batch_correct.R using conda environment and return corrected train/test dataframes
    """
    train_out = temp_dir / f"train_{method}.csv"
    test_out = temp_dir / f"test_{method}.csv"

    cmd = [
        "Rscript", str(batch_correct_script),
        "--method", method,
        "--train", str(train_csv),
        "--test", str(test_csv),
        "--batch_col", batch_col,
        "--label_col", label_col,
        "--out_train", str(train_out),
        "--out_test", str(test_out)
    ]

    try:
        # Use conda environment for R dependencies
        # Need to source conda first and run in the batch_correction environment
        conda_cmd = f"source /data/jmu27/miniconda3/etc/profile.d/conda.sh && conda run -n batch_correction {' '.join(cmd)}"
        result = subprocess.run(
            conda_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            executable='/bin/bash'
        )
        # Print R script output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Batch correction failed for method '{method}':")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

    # Read corrected data
    train_corrected = pd.read_csv(train_out)
    test_corrected = pd.read_csv(test_out)

    return train_corrected, test_corrected


def run_debiasm_correction(train_data, test_data, batch_col, label_col, feature_cols):
    """
    Run DebiasM correction (Python-based, not R)
    DebiasM requires batch ID as the first column of the feature matrix
    """
    # Get features and labels
    X_train = train_data[feature_cols].values.astype(np.float64)
    X_test = test_data[feature_cols].values.astype(np.float64)

    # Convert labels to binary (0/1)
    # Assuming 'Case' = 1, 'Control' = 0 (or anything != 'Case' = 0)
    y_train = (train_data[label_col].values == 'Case').astype(np.int64)
    y_test = (test_data[label_col].values == 'Case').astype(np.int64)

    # Get batch IDs (pair_rank) - ensure they are integers
    batch_train = train_data[batch_col].values.astype(np.int64).reshape(-1, 1)
    batch_test = test_data[batch_col].values.astype(np.int64).reshape(-1, 1)

    # Concatenate batch ID as first column
    X_train_with_batch = np.hstack((batch_train, X_train))
    X_test_with_batch = np.hstack((batch_test, X_test))

    # Initialize and fit DebiasMClassifier
    # x_val parameter is used for validation during training
    dmc = DebiasMClassifier(x_val=X_test_with_batch)
    dmc.fit(X_train_with_batch, y_train)

    # Transform both train and test data
    # Note: DebiasM transform automatically removes the batch ID column (first column)
    # and returns only the corrected features
    X_train_corrected = dmc.transform(X_train_with_batch)
    X_test_corrected = dmc.transform(X_test_with_batch)

    # Convert back to DataFrame with original feature names
    train_corrected = pd.DataFrame(X_train_corrected, columns=feature_cols)
    test_corrected = pd.DataFrame(X_test_corrected, columns=feature_cols)

    return train_corrected, test_corrected


def evaluate_models(X_train, y_train, X_test, y_test, use_tabpfn=True):
    """
    Train and evaluate ElasticNet, RandomForest, and TabPFN
    Returns dict of AUC scores
    """
    results = {}

    # Ensure no NaN/inf
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

    # Check if binary classification is possible
    if len(np.unique(y_test)) < 2:
        print(f"[WARNING] Test set has only one class. Returning NaN AUC.")
        return {'ElasticNet': np.nan, 'RandomForest': np.nan, 'TabPFN': np.nan}

    # ElasticNet (using LogisticRegressionCV with elasticnet penalty)
    try:
        enet = LogisticRegressionCV(
            cv=3, penalty="elasticnet", solver="saga", scoring="roc_auc",
            max_iter=1000, l1_ratios=[0.5], random_state=random_state, n_jobs=n_jobs
        )
        enet.fit(X_train, y_train)
        y_prob_enet = enet.predict_proba(X_test)[:, 1]
        auc_enet = roc_auc_score(y_test, y_prob_enet)
        results['ElasticNet'] = auc_enet
    except Exception as e:
        print(f"[WARNING] ElasticNet failed: {e}")
        results['ElasticNet'] = np.nan

    # RandomForest
    try:
        rf = RandomForestClassifier(random_state=random_state)
        rf_search = RandomizedSearchCV(
            rf, RF_PARAM_GRID, scoring="roc_auc", n_iter=10, cv=3,
            n_jobs=n_jobs, random_state=random_state
        )
        rf_search.fit(X_train, y_train)
        y_prob_rf = rf_search.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, y_prob_rf)
        results['RandomForest'] = auc_rf
    except Exception as e:
        print(f"[WARNING] RandomForest failed: {e}")
        results['RandomForest'] = np.nan

    # TabPFN
    if use_tabpfn:
        try:
            tabpfn = TabPFNClassifier(
                random_state=random_state, device=DEVICE,
                ignore_pretraining_limits=True
            )
            tabpfn.fit(X_train, y_train)
            y_prob_tabpfn = tabpfn.predict_proba(X_test)[:, 1]
            auc_tabpfn = roc_auc_score(y_test, y_prob_tabpfn)
            results['TabPFN'] = auc_tabpfn
        except Exception as e:
            print(f"[WARNING] TabPFN failed: {e}")
            results['TabPFN'] = np.nan
    else:
        results['TabPFN'] = np.nan

    return results


# ---------- LOSO Cross-Validation ----------
print(f"\n{'='*60}")
print(f"Starting LOSO Cross-Validation")
print(f"{'='*60}\n")

results_list = []

for test_study in tqdm(studies, desc="LOSO folds"):
    print(f"\n--- Test Study: {test_study} ---")

    # Split data
    train_data = data[data[batch_col] != test_study].copy()
    test_data = data[data[batch_col] == test_study].copy()

    print(f"Train: {len(train_data)} samples from {train_data[batch_col].nunique()} studies")
    print(f"Test:  {len(test_data)} samples from 1 study")

    # Create temporary directory for this fold
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # IMPORTANT: Apply preprocessing BEFORE batch correction
        # This is the correct order for microbiome data:
        # 1. Preprocess raw abundance data (log_std/CLR/ALR)
        # 2. Apply batch correction on preprocessed data
        # 3. Train models on batch-corrected data

        print(f"    Preprocessing: {preprocess_method}")

        # Extract features and apply preprocessing
        train_features_raw = train_data[feature_cols]
        test_features_raw = test_data[feature_cols]

        train_features_prep = apply_preprocessing(train_features_raw, preprocess_method)
        test_features_prep = apply_preprocessing(test_features_raw, preprocess_method)

        # Reconstruct full dataframes with preprocessed features + metadata
        train_data_prep = train_features_prep.copy()
        train_data_prep[batch_col] = train_data[batch_col].values
        train_data_prep[label_col] = train_data[label_col].values

        test_data_prep = test_features_prep.copy()
        test_data_prep[batch_col] = test_data[batch_col].values
        test_data_prep[label_col] = test_data[label_col].values

        # Save preprocessed train/test for batch correction
        train_csv = temp_dir / "train_preprocessed.csv"
        test_csv = temp_dir / "test_preprocessed.csv"
        train_data_prep.to_csv(train_csv, index=False)
        test_data_prep.to_csv(test_csv, index=False)

        # Get feature columns after preprocessing (ALR reduces dimensionality)
        feature_cols_prep = list(train_features_prep.columns)

        # Test each correction method
        for method in methods:
            print(f"\n  Method: {method}")

            # Run batch correction on PREPROCESSED data
            try:
                # DebiasM is Python-based, not R-based
                if method == "DebiasM":
                    train_corrected, test_corrected = run_debiasm_correction(
                        train_data_prep, test_data_prep, batch_col, label_col, feature_cols_prep
                    )
                else:
                    # R-based methods (ComBat, MMUPHin, ConQuR)
                    train_corrected, test_corrected = run_batch_correction(
                        method, train_csv, test_csv, batch_col, label_col, temp_dir
                    )
                train_features = train_corrected
                test_features = test_corrected
            except Exception as e:
                print(f"[ERROR] Batch correction failed for {method}: {e}")
                # Record failure
                for model_name in ['ElasticNet', 'RandomForest', 'TabPFN']:
                    results_list.append({
                        'test_study': test_study,
                        'method': method,
                        'model': model_name,
                        'auc': np.nan
                    })
                continue

            # Get labels
            y_train = train_data[label_col]
            y_test = test_data[label_col]

            # NO additional preprocessing - already done before batch correction!
            # Use batch-corrected features directly for model training

            # Evaluate models
            model_aucs = evaluate_models(train_features, y_train, test_features, y_test, use_tabpfn=True)

            # Record results
            for model_name, auc in model_aucs.items():
                results_list.append({
                    'test_study': test_study,
                    'method': method,
                    'model': model_name,
                    'auc': auc
                })
                print(f"    {model_name}: AUC = {auc:.4f}")

# ---------- Save Results ----------
results_df = pd.DataFrame(results_list)

# Create output in the same format as existing classicML.py results
# Combine all methods and models into a single file
all_rows = []

for method in methods:
    for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
        model_method_results = results_df[(results_df['model'] == model) & (results_df['method'] == method)]

        # Create a row with study results
        # Model name format: ModelName_Method (e.g., "ElasticNet_ComBat")
        row_data = {'model': f"{model}_{method}"}

        # Add LODO row with results for each study
        for i, study in enumerate(studies, 1):
            study_result = model_method_results[model_method_results['test_study'] == study]
            if len(study_result) > 0:
                row_data[str(i)] = study_result['auc'].values[0]
            else:
                row_data[str(i)] = np.nan

        all_rows.append(row_data)

# Combine all rows into a single DataFrame
combined_df = pd.DataFrame(all_rows)

# Save to single output file
# Include preprocessing method in filename
output_file = output_dir / f"{disease}_{data_type}_{tax_level}_{preprocess_method}_batchcorrection_result.csv"
combined_df.to_csv(output_file, index=False)

print(f"\n✅ Results saved to: {output_file}")

# ---------- Summary ----------
print(f"\n{'='*60}")
print(f"LOSO Cross-Validation Complete")
print(f"{'='*60}\n")
print(f"Mean AUC Summary")
print(f"{'='*60}\n")

for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
    model_results = results_df[results_df['model'] == model]
    print(f"\n{model}:")
    for method in methods:
        method_results = model_results[model_results['method'] == method]
        mean_auc = method_results['auc'].mean()
        std_auc = method_results['auc'].std()
        print(f"  {method:15s}: {mean_auc:.4f} ± {std_auc:.4f}")
