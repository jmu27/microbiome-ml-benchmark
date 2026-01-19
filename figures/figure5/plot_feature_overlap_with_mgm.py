#!/usr/bin/env python3
"""
Plot heatmap comparing Ridge, RF, TabPFN, and MGM attention
Model pairs on Y-axis, disease categories on X-axis
Adapted for /ua/jmu27/Micro_bench folder structure
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

# Disease type mapping
disease_type_map = {
    'CRC': 'Intestinal', 'T2D': 'Metabolic', 'Obesity': 'Metabolic', 'Overweight': 'Metabolic',
    'Adenoma': 'Intestinal', 'CDI': 'Intestinal', 'AD': 'Mental', 'MCI': 'Mental', 'PD': 'Mental',
    'RA': 'Autoimmun', 'MS': 'Autoimmun', 'ASD': 'Mental', 'CD': 'Intestinal', 'UC': 'Intestinal',
    'IBD': 'Intestinal', 'AS': 'Autoimmun', 'IBS': 'Intestinal', 'CFS': 'Mental',
    'JA': 'Autoimmun', 'NAFLD': 'Liver'
}

# Directories
script_dir = Path(__file__).parent
project_dir = script_dir.parent.parent
shap_dir = project_dir / "interpretability" / "shap_results"
mgm_dir = project_dir / "interpretability" / "mgm_results"
output_dir = script_dir

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True, parents=True)

# Load taxid to genus name mapping
genus_names_path = project_dir / "data" / "gpt_embedding" / "genus_id_names.csv"
genus_names = pd.read_csv(genus_names_path)
taxid_to_name = genus_names.set_index('taxid')['taxname'].to_dict()

# Find all genus LOSO datasets
print("="*80)
print("FEATURE OVERLAP ANALYSIS WITH MGM")
print("="*80)
print(f"\nSearching for datasets...")
print(f"  SHAP directory: {shap_dir}")
print(f"  MGM directory:  {mgm_dir}")

shap_datasets = [d.name for d in shap_dir.glob("*_genus_loso") if d.is_dir()]
mgm_datasets = [d.name for d in mgm_dir.glob("*_genus_loso") if d.is_dir()]

# Get intersection of datasets (those that have both SHAP and MGM results)
datasets = sorted(set(shap_datasets) & set(mgm_datasets))
print(f"\nFound {len(datasets)} datasets with both SHAP and MGM results:")
for ds in datasets:
    print(f"  - {ds}")

# Function to load top N features
def load_top_features(dataset, model, top_n=50):
    """Load top N features from global importance file and convert to genus names"""
    if model == 'mgm':
        file_path = mgm_dir / dataset / "global_importance_mgm.csv"
    else:
        file_path = shap_dir / dataset / f"global_importance_{model}.csv"

    if not file_path.exists():
        return set()

    df = pd.read_csv(file_path)
    top_features_raw = df.head(top_n)['feature'].values

    # Convert SHAP features from taxid format to genus names
    if model != 'mgm':
        converted_features = set()
        for feat in top_features_raw:
            if feat.startswith('ncbi_'):
                taxid = int(feat.replace('ncbi_', ''))
                genus_name = taxid_to_name.get(taxid, None)
                if genus_name:
                    converted_features.add(f"g__{genus_name}")
            else:
                converted_features.add(feat)
        return converted_features
    else:
        # MGM features already in genus name format
        return set(top_features_raw)

# Function to calculate overlap percentage
def calculate_overlap(features1, features2):
    """Calculate Jaccard index (percentage overlap)"""
    if len(features1) == 0 or len(features2) == 0:
        return np.nan
    intersection = len(features1 & features2)
    union = len(features1 | features2)
    return 100 * intersection / union

# Calculate overlap for each dataset
print("\n" + "="*80)
print("Calculating feature overlaps...")
print("="*80)
overlap_results = []

for dataset in datasets:
    disease = dataset.split('_')[0]

    if disease not in disease_type_map:
        print(f"  Warning: {disease} not in disease_type_map, skipping {dataset}")
        continue

    disease_type = disease_type_map[disease]

    # Load top 50 features for each model
    ridge_features = load_top_features(dataset, 'elasticnet', top_n=50)
    rf_features = load_top_features(dataset, 'rf', top_n=50)
    tabpfn_features = load_top_features(dataset, 'tabpfn', top_n=50)
    mgm_features = load_top_features(dataset, 'mgm', top_n=50)

    if len(ridge_features) == 0 or len(rf_features) == 0 or len(tabpfn_features) == 0 or len(mgm_features) == 0:
        print(f"  ⚠ Skipping {dataset} - missing model results")
        print(f"     Ridge: {len(ridge_features)}, RF: {len(rf_features)}, TabPFN: {len(tabpfn_features)}, MGM: {len(mgm_features)}")
        continue

    # Calculate all pairwise overlaps
    ridge_rf = calculate_overlap(ridge_features, rf_features)
    ridge_tabpfn = calculate_overlap(ridge_features, tabpfn_features)
    ridge_mgm = calculate_overlap(ridge_features, mgm_features)
    rf_tabpfn = calculate_overlap(rf_features, tabpfn_features)
    rf_mgm = calculate_overlap(rf_features, mgm_features)
    tabpfn_mgm = calculate_overlap(tabpfn_features, mgm_features)

    overlap_results.append({
        'disease_type': disease_type,
        'dataset': dataset,
        'Ridge vs RF': ridge_rf,
        'Ridge vs TabPFN': ridge_tabpfn,
        'Ridge vs MGM': ridge_mgm,
        'RF vs TabPFN': rf_tabpfn,
        'RF vs MGM': rf_mgm,
        'TabPFN vs MGM': tabpfn_mgm
    })

    print(f"  ✓ {dataset:40s} ({disease_type:12s})")

# Convert to DataFrame
df = pd.DataFrame(overlap_results)

if len(df) == 0:
    print("\n❌ ERROR: No datasets processed successfully!")
    exit(1)

# Group by disease type and calculate mean
print("\n" + "="*80)
print("Averaging by Disease Type")
print("="*80)

model_pairs = ['Ridge vs RF', 'Ridge vs TabPFN', 'Ridge vs MGM',
               'RF vs TabPFN', 'RF vs MGM', 'TabPFN vs MGM']
disease_type_order = ['Autoimmun', 'Intestinal', 'Liver', 'Mental', 'Metabolic']

# Calculate mean for each disease type
category_means = []
for dtype in disease_type_order:
    datasets_in_type = df[df['disease_type'] == dtype]
    if len(datasets_in_type) == 0:
        continue

    row_data = {'disease_type': dtype, 'n_datasets': len(datasets_in_type)}
    for model_pair in model_pairs:
        mean_val = datasets_in_type[model_pair].mean()
        row_data[model_pair] = mean_val

    category_means.append(row_data)

    print(f"\n{dtype} (n={len(datasets_in_type)}):")
    for model_pair in model_pairs:
        print(f"  {model_pair:20s}: {row_data[model_pair]:5.1f}%")

category_df = pd.DataFrame(category_means)

# Prepare data for heatmap
heatmap_data = category_df[model_pairs].T
heatmap_data.columns = category_df['disease_type'].values

# Convert to 0-1 scale
heatmap_data = heatmap_data / 100.0

# Create figure
print("\n" + "="*80)
print("Creating heatmap visualization...")
print("="*80)

fig, ax = plt.subplots(figsize=(8, 8))

# Create heatmap with vlag color palette (professional diverging colormap)
# vlag: blue -> white -> red, excellent for showing contrasts
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.3f',
    cmap='vlag',
    center=0.5,  # Center the colormap at 0.5 for overlap data
    vmin=0,
    vmax=1,
    ax=ax,
    linewidths=1,
    linecolor='white',
    cbar=False,
    annot_kws={'fontsize': 16, 'color': 'black', 'weight': 'normal'}
)

# Remove axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# Set tick labels
ax.set_xticklabels(category_df['disease_type'].values, rotation=0, ha='center', fontsize=12)
ax.set_yticklabels(model_pairs, rotation=0, fontsize=12)

# Remove title
ax.set_title('')

# Add white box title that spans the exact width of the heatmap
n_cols = len(category_df)
rect = patches.Rectangle(
    (0, -0.55),
    n_cols,
    0.55,
    linewidth=1.5,
    edgecolor='black',
    facecolor='white',
    clip_on=False,
    zorder=10
)
ax.add_patch(rect)
ax.text(n_cols / 2, -0.275, "Overlap of Top 50 Features",
       ha='center', va='center', fontsize=16, fontweight='bold',
       fontfamily='sans-serif', zorder=11)

# Background colors
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()

# Save figure
output_path = output_dir / "feature_overlap_with_mgm_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white', pad_inches=0.5)
print(f"\n✓ Heatmap saved to: {output_path}")

output_path_pdf = output_dir / "feature_overlap_with_mgm_heatmap.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', pad_inches=0.5)
print(f"✓ PDF saved to: {output_path_pdf}")

plt.close()

# Save category means to CSV
csv_path = output_dir / "feature_overlap_with_mgm_category_means.csv"
category_df.to_csv(csv_path, index=False)
print(f"✓ Category means saved to: {csv_path}")

# Save detailed results to CSV
detailed_csv_path = output_dir / "feature_overlap_with_mgm_detailed.csv"
df.to_csv(detailed_csv_path, index=False)
print(f"✓ Detailed results saved to: {detailed_csv_path}")

# Print overall statistics
print("\n" + "="*80)
print("Overall Statistics Across All Disease Categories")
print("="*80)

for model_pair in model_pairs:
    values = category_df[model_pair].values
    print(f"\n{model_pair}:")
    print(f"  Mean: {values.mean():5.1f}%")
    print(f"  Std:  {values.std():5.1f}%")
    print(f"  Min:  {values.min():5.1f}% ({category_df.loc[category_df[model_pair].idxmin(), 'disease_type']})")
    print(f"  Max:  {values.max():5.1f}% ({category_df.loc[category_df[model_pair].idxmax(), 'disease_type']})")

print("\n" + "="*80)
print("✅ COMPLETED!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print(f"  - feature_overlap_with_mgm_heatmap.png")
print(f"  - feature_overlap_with_mgm_heatmap.pdf")
print(f"  - feature_overlap_with_mgm_category_means.csv")
print(f"  - feature_overlap_with_mgm_detailed.csv")
print()
