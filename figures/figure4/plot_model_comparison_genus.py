#!/usr/bin/env python3
"""
Figure 4 plotting script for model comparison at genus level
Creates boxplot comparing ElasticNet/RandomForest/TabPFN/MGM across all diseases
with genus-level data, showing intra-cohort, cross-cohort, and LOSO performance
"""

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------ Paths ------------------ #
results_dir = "/ua/jmu27/Micro_bench/results"
output_dir = "/ua/jmu27/Micro_bench/figures/figure4"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "model_comparison_genus_boxplot.png")

# ------------------ Helper Functions ------------------ #
def parse_filename(fname: str):
    """Extract disease name from filename"""
    stem = os.path.basename(fname).replace("_result.csv", "")
    parts = stem.split("_")
    disease = parts[0] if len(parts) > 0 else None
    return disease

def analyze_matrix_df(df: pd.DataFrame):
    """
    Extract intra (diagonal), cross (off-diagonal), and loso values from result matrix.

    Structure:
    - Rows: training cohorts (last row is LOSO/LODO)
    - Columns: test cohorts (numeric columns)
    - Intra: diagonal elements (same train/test cohort)
    - Cross: off-diagonal elements (different train/test cohort)
    - LOSO: last row values
    """
    # Get numeric columns (test cohorts)
    numeric_cols = [col for col in df.columns if col not in ['model', 'Unnamed: 0']]
    numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(df[col]) or
                    all(pd.to_numeric(df[col], errors='coerce').notna())]

    n = len(numeric_cols)
    if n == 0:
        return [], [], []

    intra_vals = []
    cross_vals = []
    loso_vals = []

    try:
        # Identify last row (LOSO/LODO)
        # Check if there's a 'model' column or if last row contains 'lodo'/'loso'
        last_row_idx = None
        if 'model' in df.columns:
            model_col_vals = df['model'].astype(str).str.lower()
            loso_mask = model_col_vals.isin(['lodo', 'loso'])
            if loso_mask.any():
                last_row_idx = df[loso_mask].index[0]

        if last_row_idx is None:
            # Assume last row is LOSO
            last_row_idx = df.index[-1]

        # Extract LOSO values (last row)
        loso_row = df.loc[last_row_idx, numeric_cols]
        for val in loso_row:
            if pd.notna(val):
                loso_vals.append(float(val))

        # Extract intra and cross values (all rows except last)
        other_rows = df[df.index != last_row_idx]

        # Take first n rows as the square matrix
        if len(other_rows) >= n:
            matrix_rows = other_rows.iloc[:n]

            for i in range(n):
                for j in range(n):
                    val = matrix_rows.iloc[i][numeric_cols[j]]
                    if pd.notna(val):
                        if i == j:
                            # Diagonal - intra cohort
                            intra_vals.append(float(val))
                        else:
                            # Off-diagonal - cross cohort
                            cross_vals.append(float(val))
    except Exception as e:
        print(f"Error analyzing matrix: {e}")

    return intra_vals, cross_vals, loso_vals

def load_results_from_file(file_path: str, model_name: str):
    """
    Load results from a single file and extract intra/cross/loso values.
    Returns list of records with phenotype, model, and evaluation values.
    """
    disease = parse_filename(file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    # For MGM files, there's only one model
    # For regular files, filter by model name
    if 'model' in df.columns:
        if model_name != 'MGM':
            # Filter for specific model
            model_data = df[df['model'].astype(str) == model_name]
            if len(model_data) == 0:
                return []
            df = model_data

    intra, cross, loso = analyze_matrix_df(df)

    records = []

    # Add intra records
    for val in intra:
        records.append({
            'disease': disease,
            'model': model_name,
            'eval_type': 'Intra',
            'AUC': val
        })

    # Add cross records
    for val in cross:
        records.append({
            'disease': disease,
            'model': model_name,
            'eval_type': 'Cross',
            'AUC': val
        })

    # Add loso records
    for val in loso:
        records.append({
            'disease': disease,
            'model': model_name,
            'eval_type': 'LOSO',
            'AUC': val
        })

    return records

# ------------------ Gather files ------------------ #
print("Gathering genus-level result files...")

# Get all genus-level files with "none" preprocessing (for ElasticNet, RF, TabPFN)
classic_files = glob.glob(os.path.join(results_dir, "*_*_genus_none_result.csv"))

# Get all genus-level MGM files
mgm_files = glob.glob(os.path.join(results_dir, "*_*_genus_mgm_result.csv"))

print(f"Found {len(classic_files)} classic model result files")
print(f"Found {len(mgm_files)} MGM result files")

# ------------------ Load data ------------------ #
records = []

# Load ElasticNet, RandomForest, TabPFN from classic files
for file_path in classic_files:
    for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
        records.extend(load_results_from_file(file_path, model))

# Load MGM from mgm files
for file_path in mgm_files:
    records.extend(load_results_from_file(file_path, 'MGM'))

# Create DataFrame
df_all = pd.DataFrame.from_records(records)

print(f"\nTotal records collected: {len(df_all)}")
print(f"Diseases included: {df_all['disease'].nunique()}")
print(f"Models: {df_all['model'].unique()}")

# Check data distribution
print("\nData distribution:")
print(df_all.groupby(['model', 'eval_type']).size())

# ------------------ Create box plot ------------------ #
# Set style
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'

# Define colors for each model (matching reference)
model_colors = {
    'ElasticNet': '#aec7e8',    # Light blue
    'RandomForest': '#ffbb78',  # Light orange
    'TabPFN': '#98df8a',        # Light green
    'MGM': '#ff9896'            # Light red
}

# Model abbreviations for display
model_display_names = {
    'ElasticNet': 'Enet',
    'RandomForest': 'RF',
    'TabPFN': 'TabPFN',
    'MGM': 'MGM'
}

# Rename models in dataframe for display
df_all['model_display'] = df_all['model'].map(model_display_names)

# Order of evaluation types and models
eval_order = ['Intra', 'Cross', 'LOSO']
model_order = ['Enet', 'RF', 'TabPFN', 'MGM']

# Create figure (square shape matching reference)
fig, ax = plt.subplots(figsize=(8, 8))

# Create box plot with styling to match reference
sns.boxplot(data=df_all, x='eval_type', y='AUC', hue='model_display',
            order=eval_order, hue_order=model_order,
            palette={model_display_names[k]: v for k, v in model_colors.items()},
            ax=ax,
            linewidth=1.5,
            fliersize=4,
            showcaps=True,
            boxprops=dict(edgecolor='black', linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5),
            medianprops=dict(color='black', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='black',
                           markersize=4, linestyle='none',
                           markeredgecolor='black'))

# Styling
ax.set_xlabel('', fontsize=16)
ax.set_ylabel('AUC', fontsize=18)
ax.tick_params(axis='both', labelsize=16)

# Update x-axis labels
x_labels = ['Intra-cohort', 'Cross-cohort', 'LOSO']
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, fontsize=16)

# Legend styling - move to lower left
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=14, loc='lower left',
          frameon=True, fancybox=False, shadow=False,
          edgecolor='black', framealpha=1, facecolor='white')

# Add y-axis grid
ax.grid(True, axis='y', alpha=0.3)
ax.set_axisbelow(True)

# Set y-axis limits from 0 to 1
ax.set_ylim(0, 1.0)

# Background and spines
ax.set_facecolor('white')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
fig.patch.set_facecolor('white')

# Add white box title "Genus" at the top
title_box = mpatches.FancyBboxPatch(
    (0.0, 1.01), 1.0, 0.06,  # (x, y), width, height in axes coordinates
    boxstyle="square,pad=0.0",
    linewidth=1.5,
    edgecolor='black',
    facecolor='white',
    transform=ax.transAxes,
    clip_on=False,
    zorder=10
)
ax.add_patch(title_box)
ax.text(0.5, 1.04, 'Genus',
        transform=ax.transAxes,
        ha='center', va='center',
        fontsize=18, fontweight='bold',
        zorder=11)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white', pad_inches=0.5)
plt.close()

print(f"\n✅ Box plot saved to: {output_path}")
