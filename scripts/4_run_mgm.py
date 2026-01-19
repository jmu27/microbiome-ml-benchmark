#!/usr/bin/env python3
# ---------- Imports ----------
import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
import subprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ---------- Config ----------
parser = argparse.ArgumentParser(description="Run MGM foundation model on microbiome classification benchmarks")
parser.add_argument("--disease", type=str, required=True, help="Disease name, e.g., 'PD'")
parser.add_argument("--data_type", type=str, required=True, help="Data type, e.g., 'Amplicon' or 'Metagenomics'")
args = parser.parse_args()

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_FOLDS = 5
TAX_LEVEL = "genus"  # Only process genus level data

# ---------- Paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "filterd_data")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
TEMP_DIR = "/data/jmu27"  # Intermediate results to avoid filling up /ua/jmu27

# Load genus names mapping
GENUS_NAMES_PATH = os.path.join(PROJECT_DIR, "data", "gpt_embedding", "genus_id_names.csv")
genus_names = pd.read_csv(GENUS_NAMES_PATH)

# Create result directory if not exists
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------- Data Load ----------
filename = f"{args.disease}_{args.data_type}_{TAX_LEVEL}.csv"
csv_path = os.path.join(DATA_DIR, filename)

if not os.path.exists(csv_path):
    print(f"❌ Error: Data file not found: {csv_path}")
    sys.exit(1)

print(f"📂 Loading data from: {csv_path}")
data = pd.read_csv(csv_path, index_col=0)

label_col = "Group"
pair_ranks = np.sort(pd.unique(data["pair_rank"]))
print(f"📊 Found {len(pair_ranks)} cohorts: {list(pair_ranks)}")

# ---------- Helper Functions ----------
def get_features_and_labels(df):
    """Extract features and labels from dataframe."""
    feature_cols = [col for col in df.columns if col.startswith("ncbi")]
    X = df[feature_cols]

    # Convert ncbi IDs to genus names
    taxids = [int(col.replace('ncbi_', '')) for col in X.columns]
    taxid_to_name = genus_names.set_index('taxid')['taxname'].to_dict()
    new_columns = [taxid_to_name.get(tid, f"Unknown_{tid}") for tid in taxids]
    X.columns = new_columns
    X.columns = ["g__" + col for col in X.columns]

    y = df[label_col]
    return X, y

def prepare_data_for_mgm(X, y, train_idx, test_idx):
    """Prepare training and test data for MGM."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Add 'G' prefix to sample indices (required by MGM)
    X_train.columns = X_train.columns.astype(str)
    X_train.index = "G" + X_train.index.astype(str)
    y_train.index = "G" + y_train.index.astype(str)

    X_test.columns = X_test.columns.astype(str)
    X_test.index = "G" + X_test.index.astype(str)
    y_test.index = "G" + y_test.index.astype(str)

    return X_train, X_test, y_train, y_test

def prepare_data_for_mgm_full(X, y):
    """Prepare full dataset for MGM (for cross-cohort or LODO)."""
    X = X.copy()
    y = y.copy()

    # Add 'G' prefix to sample indices (required by MGM)
    X.columns = X.columns.astype(str)
    X.index = "G" + X.index.astype(str)
    y.index = "G" + y.index.astype(str)

    return X, y

def run_mgm_pipeline(X_train, X_test, y_train, y_test):
    """Run MGM construct, finetune, and predict pipeline."""
    # Save data to temporary directory
    train_data_path = os.path.join(TEMP_DIR, "train_data.csv")
    test_data_path = os.path.join(TEMP_DIR, "test_data.csv")
    train_label_path = os.path.join(TEMP_DIR, "train_label.csv")
    test_label_path = os.path.join(TEMP_DIR, "test_label.csv")

    X_train.T.to_csv(train_data_path, index=True)
    X_test.T.to_csv(test_data_path, index=True)
    y_train.to_csv(train_label_path, index=True)
    y_test.to_csv(test_label_path, index=True)

    # Construct command with conda environment activation
    # IMPORTANT: cd to TEMP_DIR first so MGM_log is created there, not in /ua/jmu27/
    mgm_cmd = (
        f"cd {TEMP_DIR} && "
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate /data/jmu27/envs/MGM && "
        f"mgm construct -i {train_data_path} -o {TEMP_DIR}/train_corpus.pkl && "
        f"mgm construct -i {test_data_path} -o {TEMP_DIR}/test_corpus.pkl && "
        f"mgm finetune -i {TEMP_DIR}/train_corpus.pkl -l {train_label_path} "
        f"-o {TEMP_DIR}/model_finetune --seed {RANDOM_STATE} && "
        f"mgm predict -E -i {TEMP_DIR}/test_corpus.pkl -l {test_label_path} "
        f"-m {TEMP_DIR}/model_finetune -o {TEMP_DIR}/predictions"
    )

    # Run MGM pipeline using bash
    # Allow output to flow through so we can see progress in logs
    result = subprocess.run(
        mgm_cmd,
        shell=True,
        executable='/bin/bash'
    )

    if result.returncode != 0:
        print(f"  ⚠️  MGM pipeline failed with exit code {result.returncode}")
        return np.nan

    # Read results
    try:
        result_path = os.path.join(TEMP_DIR, "predictions", "evaluation", "avg.csv")
        results = pd.read_csv(result_path)
        auc = results.loc[0, "ROC-AUC"]

        # Cleanup model files and logs
        subprocess.run(f"rm -rf {TEMP_DIR}/model_finetune", shell=True, executable='/bin/bash')
        subprocess.run(f"rm -rf {TEMP_DIR}/MGM_log", shell=True, executable='/bin/bash')

        return auc
    except Exception as e:
        print(f"  ⚠️  Failed to read MGM results: {e}")
        return np.nan

# ---------- Intra-Cohort Cross-Validation ----------
print(f"\n{'='*60}")
print("🔄 Running Intra-Cohort Cross-Validation")
print(f"{'='*60}")

results_intra = []
for rank in pair_ranks:
    print(f"\n▶️  Cohort {rank}")
    subset = data[data["pair_rank"] == rank]
    X, y = get_features_and_labels(subset)

    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"  📍 Fold {fold_idx}/{N_FOLDS}...", end=" ")
        X_train, X_test, y_train, y_test = prepare_data_for_mgm(X, y, train_idx, test_idx)
        auc = run_mgm_pipeline(X_train, X_test, y_train, y_test)
        aucs.append(auc)
        print(f"AUC = {auc:.4f}" if not np.isnan(auc) else "FAILED")

    mean_auc = np.nanmean(aucs)
    results_intra.append({
        'pair_rank': rank,
        'mgm_auc': mean_auc,
    })
    print(f"  ✅ Mean AUC: {mean_auc:.4f}")

# ---------- Cross-Cohort Evaluation ----------
print(f"\n{'='*60}")
print("🔀 Running Cross-Cohort Evaluation")
print(f"{'='*60}")

mgm_cross = pd.DataFrame(index=pair_ranks, columns=pair_ranks, dtype=float)

for i in tqdm(pair_ranks, desc="Training cohorts"):
    # Prepare training data once for cohort i
    train_df = data[data['pair_rank'] == i]
    X_train, y_train = get_features_and_labels(train_df)
    X_train, y_train = prepare_data_for_mgm_full(X_train, y_train)

    for j in pair_ranks:
        if i == j:
            # Use intra-cohort result
            row = next((r for r in results_intra if r['pair_rank'] == i), None)
            if row is not None:
                mgm_cross.loc[i, j] = row['mgm_auc']
            else:
                mgm_cross.loc[i, j] = np.nan
            continue

        # Prepare test data for cohort j
        test_df = data[data['pair_rank'] == j]
        X_test, y_test = get_features_and_labels(test_df)
        X_test, y_test = prepare_data_for_mgm_full(X_test, y_test)

        # Run MGM pipeline
        auc = run_mgm_pipeline(X_train, X_test, y_train, y_test)
        mgm_cross.loc[i, j] = auc

print(f"\n✅ Cross-cohort evaluation completed")

# ---------- Leave-One-Dataset-Out (LODO) ----------
mgm_cross.loc['lodo'] = np.nan

if len(pair_ranks) >= 3:
    print(f"\n{'='*60}")
    print("🚪 Running Leave-One-Dataset-Out (LODO)")
    print(f"{'='*60}")

    for test_rank in pair_ranks:
        print(f"\n▶️  Test cohort: {test_rank}")

        # Train on all cohorts except test_rank
        train_df = data[data['pair_rank'] != test_rank]
        test_df = data[data['pair_rank'] == test_rank]

        X_train, y_train = get_features_and_labels(train_df)
        X_test, y_test = get_features_and_labels(test_df)

        X_train, y_train = prepare_data_for_mgm_full(X_train, y_train)
        X_test, y_test = prepare_data_for_mgm_full(X_test, y_test)

        # Run MGM pipeline
        auc = run_mgm_pipeline(X_train, X_test, y_train, y_test)
        mgm_cross.loc['lodo', test_rank] = auc
        print(f"  ✅ LODO AUC: {auc:.4f}" if not np.isnan(auc) else "  ⚠️  LODO FAILED")

    print(f"\n✅ LODO evaluation completed")
else:
    print(f"\n⚠️  Skipping LODO: Need at least 3 cohorts (found {len(pair_ranks)})")

# ---------- Save Results ----------
mgm_cross['model'] = 'MGM'
result_filename = f"{args.disease}_{args.data_type}_{TAX_LEVEL}_mgm_result.csv"
result_path = os.path.join(RESULT_DIR, result_filename)
mgm_cross.to_csv(result_path, index=True)

print(f"\n{'='*60}")
print(f"✅ Results saved to: {result_path}")
print(f"{'='*60}")

# Clean up temporary files
print("\n🧹 Cleaning up temporary files...")
subprocess.run(f"rm -f {TEMP_DIR}/train_data.csv {TEMP_DIR}/test_data.csv", shell=True, executable='/bin/bash')
subprocess.run(f"rm -f {TEMP_DIR}/train_label.csv {TEMP_DIR}/test_label.csv", shell=True, executable='/bin/bash')
subprocess.run(f"rm -f {TEMP_DIR}/train_corpus.pkl {TEMP_DIR}/test_corpus.pkl", shell=True, executable='/bin/bash')
subprocess.run(f"rm -rf {TEMP_DIR}/predictions {TEMP_DIR}/model_finetune {TEMP_DIR}/MGM_log", shell=True, executable='/bin/bash')
print("✅ Cleanup completed")
