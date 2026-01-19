#!/usr/bin/env python3
"""
MGM Attention-based Feature Importance for LOSO Setting
========================================================
Extract attention weights from fine-tuned MGM models as feature importance scores.
Uses Leave-One-Study-Out (LOSO) cross-validation on genus-level data.

Preprocessing: None (uses raw data)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import subprocess
from pickle import load

# Import MGM modules
import sys
sys.path.insert(0, '/ua/jmu27/mgm')
from mgm.CLI.CLI_utils import find_pkg_resource
from mgm.src.MicroCorpus import SequenceClassificationDataset
from mgm.src.utils import seed_everything, CustomUnpickler
from transformers import GPT2ForSequenceClassification, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ---------- Configuration ----------
RANDOM_STATE = 42
POS_LABEL = "Case"  # Always target "Case" class

# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description="Calculate MGM attention weights for LOSO setting")
parser.add_argument("--disease", type=str, required=True, help="Disease name, e.g., 'PD'")
parser.add_argument("--data_type", type=str, required=True, help="Data type, e.g., 'Amplicon' or 'Metagenomics'")
parser.add_argument("--tax_level", type=str, default="genus", help="Taxonomic level (default: genus)")
args = parser.parse_args()

warnings.filterwarnings("ignore")
seed_everything(RANDOM_STATE)

# ---------- Paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "filterd_data")
OUT_DIR = os.path.join(PROJECT_DIR, "interpretability", "mgm_results",
                       f"{args.disease}_{args.data_type}_{args.tax_level}_loso")
TEMP_DIR = Path(f"/data/jmu27/mgm_temp_{args.disease}_{args.data_type}")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"MGM Attention Calculation for {args.disease} - {args.data_type} - {args.tax_level}")
print(f"{'='*60}\n")

# ---------- Utils ----------
# Load genus names mapping
GENUS_NAMES_PATH = os.path.join(PROJECT_DIR, "data", "gpt_embedding", "genus_id_names.csv")
genus_names = pd.read_csv(GENUS_NAMES_PATH)

def get_features_and_labels(df: pd.DataFrame, label_col: str = "Group") -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature columns and labels"""
    feature_cols = [c for c in df.columns if c.startswith("ncbi")]
    X = df[feature_cols].copy()

    # Map taxids to genus names
    taxids = [int(col.replace('ncbi_', '')) for col in X.columns]
    taxid_to_name = genus_names.set_index('taxid')['taxname'].to_dict()
    new_columns = [taxid_to_name.get(tid, f"Unknown_{tid}") for tid in taxids]
    X.columns = ["g__" + col for col in new_columns]

    y = df[label_col].copy()
    return X, y


def prepare_mgm_data(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> tuple[Path, Path]:
    """Prepare data in MGM format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Format data with G-prefixed indices
    X_formatted = X.copy()
    X_formatted.columns = X_formatted.columns.astype(str)
    X_formatted = X_formatted.reset_index(drop=True)
    X_formatted.index = "G" + X_formatted.index.astype(str)

    y_formatted = y.copy()
    y_formatted = y_formatted.reset_index(drop=True)
    y_formatted.index = "G" + y_formatted.index.astype(str)
    y_formatted = pd.DataFrame(y_formatted, columns=['label'])

    # Save as CSV
    data_path = output_dir / "data.csv"
    label_path = output_dir / "label.csv"
    X_formatted.T.to_csv(data_path, index=True)
    y_formatted.to_csv(label_path, index=True)

    return data_path, label_path


def train_mgm_model(corpus_path: Path, label_path: Path, model_output_dir: Path, seed: int = 42):
    """Train MGM model using CLI"""
    cmd = (
        f"cd {TEMP_DIR} && "
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate /data/jmu27/envs/MGM && "
        f"mgm finetune -i {corpus_path} -l {label_path} -o {model_output_dir} --seed {seed}"
    )
    result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"MGM finetune failed: {result.stderr.decode()}")


def construct_corpus(data_path: Path, corpus_path: Path):
    """Construct MGM corpus"""
    cmd = (
        f"cd {TEMP_DIR} && "
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate /data/jmu27/envs/MGM && "
        f"mgm construct -i {data_path} -o {corpus_path}"
    )
    result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"MGM construct failed: {result.stderr.decode()}")


def compute_attention_weights(model, corpus, device="cuda"):
    """
    Extract attention weights from fine-tuned model

    Returns:
        attention_df: DataFrame with shape (n_samples, n_features)
    """
    model.eval()
    model.to(device)

    tokens = corpus.tokens
    attention_values = []

    for sample_idx in tqdm(range(len(corpus)), desc="Computing attention"):
        token = tokens[sample_idx]
        token_nonzero = token[token != 0]

        # Get model input
        input_dict = {k: v.to(device) for k, v in corpus[sample_idx:sample_idx+1].items()}

        # Forward pass with attention output
        with torch.no_grad():
            outputs = model(**input_dict, output_attentions=True)
            attention = outputs[-1]

        # Sum attention across layers
        attention_sum = sum(list(attention))

        # Sum over heads and tokens: (batch, heads, seq, seq) -> (batch, seq)
        attention_agg = attention_sum.sum(axis=1).sum(axis=1)

        # Create attention vector for this sample
        sample_attention = torch.zeros(corpus.tokenizer.vocab_size, device=device)
        sample_attention[token_nonzero] = attention_agg[0][:len(token_nonzero)]

        attention_values.append(sample_attention.cpu().numpy())

    # Create DataFrame with proper column names
    attention_df = pd.DataFrame(
        np.array(attention_values),
        index=corpus.data.index,
        columns=list(corpus.tokenizer.vocab.keys()),
        dtype=np.float32
    )

    # Filter out zero columns and special tokens
    attention_df = attention_df.loc[:, ~(attention_df == 0).all(axis=0)]
    attention_df = attention_df.loc[:, ~attention_df.columns.isin(['<bos>', '<eos>', '<pad>'])]

    # Normalize attention weights per sample
    attention_df = attention_df.div(attention_df.sum(axis=1), axis=0)

    return attention_df


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
    raise ValueError(f"Expected binary classification, but found {n_classes} classes.")

# Encode labels to 0/1
le = LabelEncoder()
le.fit(data["Group"])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Label mapping: {label_mapping}")

# Check target class
if POS_LABEL not in le.classes_:
    raise ValueError(f"Target label '{POS_LABEL}' not found in classes: {le.classes_}")

pos_label_encoded = le.transform([POS_LABEL])[0]
print(f"Target class for attention: '{POS_LABEL}' (encoded as {pos_label_encoded})")

# GPU check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load MGM tokenizer
print("\nLoading MGM tokenizer...")
with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
    unpickler = CustomUnpickler(f)
    tokenizer = unpickler.load()

# ---------- LOSO Loop ----------
all_attention_weights = []
all_samples_info = []

for test_rank in tqdm(pair_ranks, desc="LOSO cohorts"):
    print(f"\n{'─'*60}")
    print(f"Test Cohort: {test_rank}")
    print(f"{'─'*60}")

    # Split data
    train_df = data[data["pair_rank"] != test_rank]
    test_df = data[data["pair_rank"] == test_rank]

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Extract features and labels
    X_train_raw, y_train_raw = get_features_and_labels(train_df)
    X_test_raw, y_test_raw = get_features_and_labels(test_df)

    # Encode labels
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    y_train = pd.Series(y_train, index=y_train_raw.index)
    y_test = pd.Series(y_test, index=y_test_raw.index)

    # Store sample information
    for idx in test_df.index:
        all_samples_info.append({
            "sample_id": idx,
            "cohort": test_rank,
            "true_label": test_df.loc[idx, "Group"]
        })

    # Prepare training data
    fold_train_dir = TEMP_DIR / f"fold_{test_rank}_train"
    train_data_path, train_label_path = prepare_mgm_data(X_train_raw, y_train, fold_train_dir)

    # Construct training corpus
    print("  Constructing training corpus...")
    train_corpus_path = fold_train_dir / "corpus.pkl"
    construct_corpus(train_data_path, train_corpus_path)

    # Train MGM model
    print("  Training MGM model...")
    model_path = TEMP_DIR / f"fold_{test_rank}_model"
    train_mgm_model(train_corpus_path, train_label_path, model_path, seed=RANDOM_STATE)

    # Load fine-tuned model
    print("  Loading fine-tuned model...")
    train_corpus = load(open(train_corpus_path, "rb"))
    model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=n_classes)

    # Prepare test data
    print("  Preparing test corpus...")
    fold_test_dir = TEMP_DIR / f"fold_{test_rank}_test"
    test_data_path, test_label_path = prepare_mgm_data(X_test_raw, y_test, fold_test_dir)

    test_corpus_path = fold_test_dir / "corpus.pkl"
    construct_corpus(test_data_path, test_corpus_path)
    test_corpus = load(open(test_corpus_path, "rb"))

    # Align y_test with test_corpus
    y_test_aligned = y_test.copy()
    y_test_aligned.index = "G" + pd.Series(range(len(y_test))).astype(str)
    y_test_aligned = y_test_aligned.loc[test_corpus.data.index]

    # Evaluate model
    dataset = SequenceClassificationDataset(
        test_corpus[:]["input_ids"],
        test_corpus[:]["attention_mask"],
        torch.tensor(y_test_aligned.values)
    )
    trainer = Trainer(model=model)
    predictions = trainer.predict(dataset)

    if n_classes == 2:
        proba = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
        auc = roc_auc_score(y_test_aligned.values, proba[:, pos_label_encoded])
        print(f"  Test AUC: {auc:.4f}")

    # Extract attention weights
    print("  Extracting attention weights...")
    attention_df = compute_attention_weights(model, test_corpus, device=device)
    all_attention_weights.append(attention_df)

    print(f"  ✓ Attention shape: {attention_df.shape}")

    # Cleanup
    del model, train_corpus, test_corpus
    torch.cuda.empty_cache()

# ---------- Concatenate All Results ----------
print(f"\n{'='*60}")
print("Concatenating results...")
print(f"{'='*60}")

# Concatenate all attention weights
attention_all = pd.concat(all_attention_weights, axis=0)
print(f"Total attention matrix: {attention_all.shape}")

# Create sample info DataFrame
samples_df = pd.DataFrame(all_samples_info)
print(f"Sample info: {samples_df.shape}")

# ---------- Save Results ----------
print(f"\nSaving results to: {OUT_DIR}")

# Save attention weights
np.save(os.path.join(OUT_DIR, "attention_mgm.npy"), attention_all.values)
attention_all.to_csv(os.path.join(OUT_DIR, "attention_mgm.csv"), index=False)

# Save feature names
np.save(os.path.join(OUT_DIR, "feature_names.npy"), attention_all.columns.values)

# Save sample information
samples_df.to_csv(os.path.join(OUT_DIR, "sample_info.csv"), index=False)

# ---------- Global Feature Importance ----------
def compute_global_importance(attention_matrix: np.ndarray, feature_names: np.ndarray) -> pd.DataFrame:
    """Compute global feature importance from attention values"""
    mean_attention = np.mean(attention_matrix, axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_attention  # Use same column name as SHAP for consistency
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp_df

imp_mgm = compute_global_importance(attention_all.values, attention_all.columns.values)
imp_mgm.to_csv(os.path.join(OUT_DIR, "global_importance_mgm.csv"), index=False)

# ---------- Summary ----------
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"Total samples processed: {len(samples_df)}")
print(f"Number of features: {len(attention_all.columns)}")
print(f"Number of cohorts: {len(pair_ranks)}")
print(f"\nTop 10 features by mean attention (MGM):")
print(imp_mgm.head(10))

# Save metadata
meta_path = os.path.join(OUT_DIR, "metadata.txt")
with open(meta_path, "w") as f:
    f.write(f"LOSO MGM Attention Results\n")
    f.write(f"{'='*60}\n")
    f.write(f"Disease: {args.disease}\n")
    f.write(f"Data type: {args.data_type}\n")
    f.write(f"Taxonomic level: {args.tax_level}\n")
    f.write(f"Number of cohorts: {len(pair_ranks)}\n")
    f.write(f"Cohorts: {pair_ranks}\n")
    f.write(f"Total samples: {len(samples_df)}\n")
    f.write(f"Number of features: {len(attention_all.columns)}\n")
    f.write(f"Label mapping: {label_mapping}\n")
    f.write(f"Target class: '{POS_LABEL}' (encoded as {pos_label_encoded})\n")
    f.write(f"Preprocessing: None (raw data)\n")
    f.write(f"Random state: {RANDOM_STATE}\n")
    f.write(f"\nOutput files:\n")
    f.write(f"  - attention_mgm.npy / .csv\n")
    f.write(f"  - feature_names.npy\n")
    f.write(f"  - sample_info.csv\n")
    f.write(f"  - global_importance_mgm.csv\n")

print(f"\n✅ All results saved to: {OUT_DIR}")
print(f"   - Attention matrix (numpy): attention_mgm.npy")
print(f"   - Attention matrix (CSV): attention_mgm.csv")
print(f"   - Feature names: feature_names.npy")
print(f"   - Sample info: sample_info.csv")
print(f"   - Global importance: global_importance_mgm.csv")
print(f"   - Metadata: metadata.txt")
print(f"\n{'='*60}\n")

# Cleanup temp directory
import shutil
shutil.rmtree(TEMP_DIR, ignore_errors=True)
