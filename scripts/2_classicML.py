# ---------- Imports ----------
import os
# Set environment variable before importing sklearn/scipy (required for TabPFN)
os.environ['SCIPY_ARRAY_API'] = '1'

import argparse
import warnings
import subprocess
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier

# ---------- GPT Embedding Data ----------
EMBED_DIM = 1536  # embedding dim from GPT-3.5
# Use path relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EMBED_DATA_DIR = os.path.join(PROJECT_DIR, "data", "gpt_embedding")

# Load taxid to name mappings
genus_names = pd.read_csv(os.path.join(EMBED_DATA_DIR, "genus_id_names.csv"))
species_names = pd.read_csv(os.path.join(EMBED_DATA_DIR, "species_id_names.csv"))

# Load embedding pickle files
with open(os.path.join(EMBED_DATA_DIR, "combined_embedding_genus.pkl"), "rb") as f:
    gpt_embedding_genus = pickle.load(f)
with open(os.path.join(EMBED_DATA_DIR, "combined_embedding_species.pkl"), "rb") as f:
    gpt_embedding_species = pickle.load(f)

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Run classic ML models on microbiome classification benchmarks.")
parser.add_argument("--disease", type=str, required=True, help="Disease name, e.g., 'PD'.")
parser.add_argument("--data_type", type=str, required=True, help="Data type, e.g., 'Amplicon'.")
parser.add_argument("--tax_level", type=str, required=True, help="Taxonomic level, e.g., 'genus'.")
parser.add_argument("--preprocess_method", type=str, required=True, help="Preprocessing method, e.g., 'none', 'binary', 'log_std', 'CLR', 'ALR', 'gptemb'.")
args = parser.parse_args()

# ---------- Config ----------
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_FOLDS = 5

# Check GPU availability for TabPFN
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device for TabPFN: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "max_features": ["sqrt", 0.5],
    "min_samples_split": [2, 5],
}

# ---------- Data Load ----------
filename = f"{args.disease}_{args.data_type}_{args.tax_level}.csv"
csv_path = os.path.join(PROJECT_DIR, "data", "filterd_data", filename)
data = pd.read_csv(csv_path, index_col=0)

label_col = "Group"
pair_ranks = np.sort(pd.unique(data["pair_rank"]))  

# ---------- Preprocessing ----------
def binarize_abundance(x: pd.DataFrame) -> pd.DataFrame:
    """Convert abundances to binary presence/absence."""
    return (x > 0).astype(int)


def log_std_normalization(X: pd.DataFrame, log_n0: float = 1e-6, sd_min_q: float = 0.1) -> np.ndarray:
    """
    Log-transform and standardize following SIAMCAT's log.std method.

    This matches the R code:
        feat.log <- log10(feat.red + norm.param$log.n0)
        m <- rowMeans(feat.log)
        s <- rowSds(feat.log)
        q <- quantile(s, norm.param$sd.min.q, names = FALSE)
        stopifnot(q > 0)
        feat.norm <- (feat.log - m)/(s + q)

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × features)
    log_n0 : float
        Pseudocount to add before log transformation (default: 1e-6, SIAMCAT default)
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
    # In pandas/numpy: axis=0 means across rows (samples)
    m = feat_log.mean(axis=0)  # mean of each feature
    s = feat_log.std(axis=0, ddof=1)  # std of each feature (ddof=1 for sample std)

    # Calculate quantile of standard deviations
    q = np.quantile(s, sd_min_q)

    # Ensure q > 0 (if q is 0 or negative, use a small value to avoid division by zero)
    # This can happen when many features have zero variance
    if q <= 0:
        q = 1e-8
        print(f"  ⚠️  Warning: Quantile of std is {np.quantile(s, sd_min_q):.2e}, using q={q:.2e}")

    # Normalize: (feat.log - m) / (s + q)
    # Note: R code uses (s + q), not max(s, q)
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


def alr_transform(x: pd.DataFrame | np.ndarray, reference_index: int = -1, pseudocount: float = 1e-5) -> np.ndarray:
    """
    Additive log-ratio (ALR) transform relative to a reference component.
    """
    x = np.asarray(x, dtype=np.float64) + pseudocount  # avoid log(0)
    ref = x[:, reference_index].reshape(-1, 1)
    numerator = np.delete(x, reference_index, axis=1)
    return np.log(numerator / ref)


def gpt_embedding_transform(X: pd.DataFrame, tax_level: str = "genus") -> np.ndarray:
    """
    GPT embedding transform: maps taxonomic features to GPT-3.5 embeddings.

    Computes weighted average of embeddings and applies StandardScaler normalization.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with ncbi_* columns
    tax_level : str
        Taxonomic level, either "genus" or "species"

    Returns
    -------
    np.ndarray
        Standardized weighted embedding matrix of shape (n_samples, 1536)
    """
    # Extract ncbi feature columns and their taxids
    feature_cols_list = [col for col in X.columns if col.startswith("ncbi")]
    X_features = X[feature_cols_list]
    taxids = [int(col.replace('ncbi_', '')) for col in X_features.columns]

    # Get taxid to name mapping
    if tax_level == "genus":
        taxid_to_name = genus_names.set_index('taxid')['taxname'].to_dict()
    else:
        taxid_to_name = species_names.set_index('taxid')['taxname'].to_dict()

    # Map ncbi columns to taxonomic names
    new_columns = [taxid_to_name.get(tid, f"Unknown_{tid}") for tid in taxids]
    X_features.columns = new_columns

    # Create embedding lookup table
    lookup_embed = np.zeros(shape=(len(new_columns), EMBED_DIM))
    count_missing = 0

    # Get corresponding GPT embedding data
    if tax_level == "genus":
        gpt_embedding = gpt_embedding_genus.copy()
        gpt_embedding = gpt_embedding.drop_duplicates(subset='Genus').set_index('Genus')
    else:
        gpt_embedding = gpt_embedding_species.copy()
        gpt_embedding = gpt_embedding.drop_duplicates(subset='Species').set_index('Species')

    # Fill embedding matrix
    for i, gene in enumerate(new_columns):
        if gene in gpt_embedding.index:
            lookup_embed[i, :] = gpt_embedding.loc[gene].values
        else:
            count_missing += 1

    if count_missing > 0:
        print(f"  ⚠️  {count_missing}/{len(new_columns)} taxa missing embeddings")

    # Compute weighted embedding: weighted average across all taxa
    genePT_w_embed = np.dot(X_features.values, lookup_embed) / len(new_columns)

    # Apply StandardScaler normalization
    scaler = StandardScaler()
    genePT_w_embed_scaled = scaler.fit_transform(genePT_w_embed)

    return genePT_w_embed_scaled


def apply_preprocessing(X: pd.DataFrame, method: str, tax_level: str = "genus") -> np.ndarray | pd.DataFrame:
    """Apply the selected preprocessing method."""
    if method == "none":
        return X.copy()
    if method == "binary":
        return binarize_abundance(X)
    if method == "log_std":
        return log_std_normalization(X)
    if method == "CLR":
        return clr_transform(X)
    if method == "ALR":
        return alr_transform(X.values)
    if method == "gptemb":
        return gpt_embedding_transform(X, tax_level)
    raise ValueError(f"Unsupported preprocessing method: {method}")


# ---------- Helpers ----------
def get_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Select ncbi* feature columns and the label vector."""
    feature_cols = [c for c in df.columns if c.startswith("ncbi")]
    X = df[feature_cols]
    y = df[label_col].values
    return X, y


def evaluate_model(model, X_train, y_train, X_test, y_test) -> float:
    """Fit model and return ROC AUC on the test split."""
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        prob = model.decision_function(X_test)
    return roc_auc_score(y_test, prob)


# ---------- Intra-Cohort CV ----------
results_intra: list[dict] = []
for rank in pair_ranks:
    print(f"\n▶️  Cohort {rank}")
    subset = data[data["pair_rank"] == rank]
    outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    X, y = get_features_and_labels(subset)
    aucs_enet, aucs_rf, aucs_tabpfn = [], [], []

    # Intra-cohort cross-validation
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        # Preprocessing
        X_train_prep = apply_preprocessing(X_train_fold, args.preprocess_method, args.tax_level)
        X_test_prep = apply_preprocessing(X_test_fold, args.preprocess_method, args.tax_level)

        # ----- Elastic Net -----
        try:
            enet = LogisticRegressionCV(
                cv=3, penalty="elasticnet", solver="saga", scoring="roc_auc",
                max_iter=1000, l1_ratios=[0.5], random_state=RANDOM_STATE
            )
            auc_enet = evaluate_model(enet, X_train_prep, y_train_fold, X_test_prep, y_test_fold)
            aucs_enet.append(auc_enet)
        except Exception as e:
            print(f"  ⚠️  Elastic Net failed: {e}")
            aucs_enet.append(np.nan)

        # ----- Random Forest -----
        try:
            rf = RandomForestClassifier(random_state=RANDOM_STATE)
            rf_search = RandomizedSearchCV(
                rf, RF_PARAM_GRID, scoring="roc_auc", n_iter=10, cv=3, n_jobs=-1, random_state=RANDOM_STATE
            )
            rf_search.fit(X_train_prep, y_train_fold)
            prob_rf = rf_search.predict_proba(X_test_prep)[:, 1]
            auc_rf = roc_auc_score(y_test_fold, prob_rf)
            aucs_rf.append(auc_rf)
        except Exception as e:
            print(f"  ⚠️  Random Forest failed: {e}")
            aucs_rf.append(np.nan)

        # ----- TabPFN -----
        try:
            tabpfn = TabPFNClassifier(random_state=RANDOM_STATE, device=DEVICE, ignore_pretraining_limits=True)
            tabpfn.fit(X_train_prep, y_train_fold)
            prob_tabpfn = tabpfn.predict_proba(X_test_prep)[:, 1]
            auc_tabpfn = roc_auc_score(y_test_fold, prob_tabpfn)
            aucs_tabpfn.append(auc_tabpfn)
        except Exception as e:
            print(f"  ⚠️  TabPFN failed: {e}")
            aucs_tabpfn.append(np.nan)

    results_intra.append(
        {
            "pair_rank": rank,
            "enet_auc": np.nanmean(aucs_enet) if aucs_enet else np.nan,
            "rf_auc": np.nanmean(aucs_rf) if aucs_rf else np.nan,
            "tabpfn_auc": np.nanmean(aucs_tabpfn) if aucs_tabpfn else np.nan,
        }
    )
    print(f"  Elastic Net AUC: {np.nanmean(aucs_enet):.3f}, RF AUC: {np.nanmean(aucs_rf):.3f}, TabPFN AUC: {np.nanmean(aucs_tabpfn):.3f}")

# ---------- Cross-Cohort ----------
enet_cross, rf_cross, tabpfn_cross = [pd.DataFrame(index=pair_ranks, columns=pair_ranks) for _ in range(3)]

for i in tqdm(pair_ranks):
    for j in pair_ranks:
        if i == j:
            row = next((r for r in results_intra if r["pair_rank"] == i), None)
            if row is not None:
                enet_cross.loc[i, j] = row["enet_auc"]
                rf_cross.loc[i, j] = row["rf_auc"]
                tabpfn_cross.loc[i, j] = row["tabpfn_auc"]
            else:
                enet_cross.loc[i, j] = np.nan
                rf_cross.loc[i, j] = np.nan
                tabpfn_cross.loc[i, j] = np.nan
        else:
            # Train on cohort i, test on cohort j
            train_df = data[data["pair_rank"] == i]
            test_df = data[data["pair_rank"] == j]
            X_train, y_train = get_features_and_labels(train_df)
            X_test, y_test = get_features_and_labels(test_df)

            # Preprocessing
            X_train_prep = apply_preprocessing(X_train, args.preprocess_method, args.tax_level)
            X_test_prep = apply_preprocessing(X_test, args.preprocess_method, args.tax_level)

            # ----- Elastic Net -----
            try:
                enet = LogisticRegressionCV(
                    cv=3, penalty="elasticnet", solver="saga", scoring="roc_auc",
                    max_iter=1000, l1_ratios=[0.5], random_state=RANDOM_STATE
                )
                enet_cross.loc[i, j] = evaluate_model(enet, X_train_prep, y_train, X_test_prep, y_test)
            except Exception as e:
                print(f"  ⚠️  Elastic Net failed (train={i}, test={j}): {e}")
                enet_cross.loc[i, j] = np.nan

            # ----- Random Forest -----
            try:
                rf = RandomForestClassifier(random_state=RANDOM_STATE)
                rf_search = RandomizedSearchCV(
                    rf, RF_PARAM_GRID, scoring="roc_auc", n_iter=10, cv=3, n_jobs=-1, random_state=RANDOM_STATE
                )
                rf_search.fit(X_train_prep, y_train)
                prob_rf = rf_search.predict_proba(X_test_prep)[:, 1]
                rf_cross.loc[i, j] = roc_auc_score(y_test, prob_rf)
            except Exception as e:
                print(f"  ⚠️  Random Forest failed (train={i}, test={j}): {e}")
                rf_cross.loc[i, j] = np.nan

            # ----- TabPFN -----
            try:
                tabpfn = TabPFNClassifier(random_state=RANDOM_STATE, device=DEVICE, ignore_pretraining_limits=True)
                tabpfn.fit(X_train_prep, y_train)
                prob_tabpfn = tabpfn.predict_proba(X_test_prep)[:, 1]
                tabpfn_cross.loc[i, j] = roc_auc_score(y_test, prob_tabpfn)
            except Exception as e:
                print(f"  ⚠️  TabPFN failed (train={i}, test={j}): {e}")
                tabpfn_cross.loc[i, j] = np.nan

# ---------- Leave-One-Dataset-Out ----------
for df in (enet_cross, rf_cross, tabpfn_cross):
    df.loc["lodo"] = np.nan

# Only run LODO if at least 3 cohorts exist
if len(pair_ranks) >= 3:
    for test_rank in pair_ranks:
        print(f"\n🚪 Leave-One-Domain-Out: Test = Cohort {test_rank}")

        train_df = data[data["pair_rank"] != test_rank]
        test_df = data[data["pair_rank"] == test_rank]

        X_train, y_train = get_features_and_labels(train_df)
        X_test, y_test = get_features_and_labels(test_df)

        # Preprocessing
        X_train_prep = apply_preprocessing(X_train, args.preprocess_method, args.tax_level)
        X_test_prep = apply_preprocessing(X_test, args.preprocess_method, args.tax_level)

        # ----- Elastic Net -----
        try:
            enet = LogisticRegressionCV(
                cv=3, penalty="elasticnet", solver="saga", scoring="roc_auc",
                max_iter=1000, l1_ratios=[0.5], random_state=RANDOM_STATE
            )
            auc_enet = evaluate_model(enet, X_train_prep, y_train, X_test_prep, y_test)
            enet_cross.loc["lodo", test_rank] = auc_enet
            print(f"  ✔ Elastic Net done: AUC = {auc_enet:.3f}")
        except Exception as e:
            print(f"  ⚠️  Elastic Net failed: {e}")
            enet_cross.loc["lodo", test_rank] = np.nan

        # ----- Random Forest -----
        try:
            rf = RandomForestClassifier(random_state=RANDOM_STATE)
            rf_search = RandomizedSearchCV(
                rf, RF_PARAM_GRID, scoring="roc_auc", n_iter=10, cv=3, n_jobs=-1, random_state=RANDOM_STATE
            )
            rf_search.fit(X_train_prep, y_train)
            prob_rf = rf_search.predict_proba(X_test_prep)[:, 1]
            auc_rf = roc_auc_score(y_test, prob_rf)
            rf_cross.loc["lodo", test_rank] = auc_rf
            print(f"  ✔ Random Forest done: AUC = {auc_rf:.3f}")
        except Exception as e:
            print(f"  ⚠️  Random Forest failed: {e}")
            rf_cross.loc["lodo", test_rank] = np.nan

        # ----- TabPFN -----
        try:
            tabpfn = TabPFNClassifier(random_state=RANDOM_STATE, device=DEVICE, ignore_pretraining_limits=True)
            tabpfn.fit(X_train_prep, y_train)
            prob_tabpfn = tabpfn.predict_proba(X_test_prep)[:, 1]
            auc_tabpfn = roc_auc_score(y_test, prob_tabpfn)
            tabpfn_cross.loc["lodo", test_rank] = auc_tabpfn
            print(f"  ✔ TabPFN done: AUC = {auc_tabpfn:.3f}")
        except Exception as e:
            print(f"  ⚠️  TabPFN failed: {e}")
            tabpfn_cross.loc["lodo", test_rank] = np.nan

# ---------- Save Result ----------
enet_cross["model"] = "ElasticNet"
rf_cross["model"] = "RandomForest"
tabpfn_cross["model"] = "TabPFN"

combined_df = pd.concat([enet_cross, rf_cross, tabpfn_cross], ignore_index=True)
result_filename = f"{args.disease}_{args.data_type}_{args.tax_level}_{args.preprocess_method}_result.csv"
out_dir = os.path.join(PROJECT_DIR, "results")
os.makedirs(out_dir, exist_ok=True)
combined_df.to_csv(os.path.join(out_dir, result_filename), index=False)
print(f"\n✅ Results saved to {os.path.join(out_dir, result_filename)}")
