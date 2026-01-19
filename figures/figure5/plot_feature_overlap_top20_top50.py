#!/usr/bin/env python3
"""
Plot dual heatmap comparing Ridge, RF, TabPFN, and MGM attention
Two panels: Top 20 features (left) and Top 50 features (right)
Model pairs on Y-axis, disease categories on X-axis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
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
print("FEATURE OVERLAP ANALYSIS WITH MGM - TOP 20 vs TOP 50")
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

# Function to process datasets for a given top_n
def process_datasets(top_n):
    """Calculate overlaps for all datasets using top N features"""
    overlap_results = []

    for dataset in datasets:
        disease = dataset.split('_')[0]

        if disease not in disease_type_map:
            continue

        disease_type = disease_type_map[disease]

        # Load top N features for each model
        ridge_features = load_top_features(dataset, 'elasticnet', top_n=top_n)
        rf_features = load_top_features(dataset, 'rf', top_n=top_n)
        tabpfn_features = load_top_features(dataset, 'tabpfn', top_n=top_n)
        mgm_features = load_top_features(dataset, 'mgm', top_n=top_n)

        if len(ridge_features) == 0 or len(rf_features) == 0 or len(tabpfn_features) == 0 or len(mgm_features) == 0:
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
            'ENet vs RF': ridge_rf,
            'ENet vs TabPFN': ridge_tabpfn,
            'ENet vs MGM': ridge_mgm,
            'RF vs TabPFN': rf_tabpfn,
            'RF vs MGM': rf_mgm,
            'TabPFN vs MGM': tabpfn_mgm
        })

    return pd.DataFrame(overlap_results)

# Calculate overlaps for both top 20 and top 50
print("\n" + "="*80)
print("Calculating feature overlaps...")
print("="*80)

print("\nProcessing Top 20 features...")
df_top20 = process_datasets(top_n=20)
print(f"  ✓ Processed {len(df_top20)} datasets for Top 20")

print("\nProcessing Top 50 features...")
df_top50 = process_datasets(top_n=50)
print(f"  ✓ Processed {len(df_top50)} datasets for Top 50")

if len(df_top20) == 0 or len(df_top50) == 0:
    print("\n❌ ERROR: No datasets processed successfully!")
    exit(1)

# Define model pairs and disease type order
model_pairs = ['ENet vs RF', 'ENet vs TabPFN', 'ENet vs MGM',
               'RF vs TabPFN', 'RF vs MGM', 'TabPFN vs MGM']
disease_type_order = ['Autoimmun', 'Intestinal', 'Liver', 'Mental', 'Metabolic']

# Function to create category means
def create_category_means(df):
    """Calculate mean overlap for each disease type"""
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

    return pd.DataFrame(category_means)

# Create category means for both
category_df_top20 = create_category_means(df_top20)
category_df_top50 = create_category_means(df_top50)

# Print statistics
print("\n" + "="*80)
print("TOP 20 FEATURES - Averaging by Disease Type")
print("="*80)
for _, row in category_df_top20.iterrows():
    dtype = row['disease_type']
    n_datasets = int(row['n_datasets'])
    print(f"\n{dtype} (n={n_datasets}):")
    for model_pair in model_pairs:
        print(f"  {model_pair:20s}: {row[model_pair]:5.1f}%")

print("\n" + "="*80)
print("TOP 50 FEATURES - Averaging by Disease Type")
print("="*80)
for _, row in category_df_top50.iterrows():
    dtype = row['disease_type']
    n_datasets = int(row['n_datasets'])
    print(f"\n{dtype} (n={n_datasets}):")
    for model_pair in model_pairs:
        print(f"  {model_pair:20s}: {row[model_pair]:5.1f}%")

# Prepare data for heatmaps
heatmap_data_top20 = category_df_top20[model_pairs].T
heatmap_data_top20.columns = category_df_top20['disease_type'].values
heatmap_data_top20 = heatmap_data_top20 / 100.0

heatmap_data_top50 = category_df_top50[model_pairs].T
heatmap_data_top50.columns = category_df_top50['disease_type'].values
heatmap_data_top50 = heatmap_data_top50 / 100.0

# Create dual heatmap figure
print("\n" + "="*80)
print("Creating dual heatmap visualization...")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Create custom divergent colormap matching figure2 (blue to white to red)
macaron_divergent = [
    "#B8E5FA",  # Light blue - low value
    "#D9F2FD",  # Lighter blue
    "#FFFFFF",  # White - center
    "#FBD3D6",  # Light pink
    "#F7A6AC"   # Red/pink - high value
]
custom_cmap = LinearSegmentedColormap.from_list('macaron_divergent', macaron_divergent)

# Common heatmap parameters (matching figure2 style)
heatmap_params = {
    'annot': True,
    'fmt': '.3f',
    'cmap': custom_cmap,
    'center': 0.5,
    'vmin': 0,
    'vmax': 1,
    'linewidths': 0.5,
    'linecolor': 'lightgray',
    'cbar': False,
    'annot_kws': {'fontsize': 20, 'color': 'black', 'fontfamily': 'sans-serif'}
}

# Left panel: Top 20
sns.heatmap(heatmap_data_top20, ax=ax1, **heatmap_params)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_xticklabels(category_df_top20['disease_type'].values, rotation=15, ha='right', fontsize=18)
ax1.set_yticklabels(model_pairs, rotation=0, fontsize=18)
ax1.set_title('')

# Add title box for Top 20
n_cols_20 = len(category_df_top20)
rect1 = patches.Rectangle(
    (0, -0.55), n_cols_20, 0.55,
    linewidth=1.5, edgecolor='black', facecolor='white',
    clip_on=False, zorder=10, transform=ax1.transData
)
ax1.add_patch(rect1)
ax1.text(n_cols_20 / 2, -0.275, "Overlap of Top 20 Features",
        ha='center', va='center', fontsize=20, fontweight='bold',
        fontfamily='sans-serif', zorder=11, transform=ax1.transData)

ax1.set_facecolor('white')

# Right panel: Top 50
sns.heatmap(heatmap_data_top50, ax=ax2, **heatmap_params)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_xticklabels(category_df_top50['disease_type'].values, rotation=15, ha='right', fontsize=18)
ax2.set_yticklabels([])  # Remove y-tick labels - only show on left panel
ax2.set_title('')

# Add title box for Top 50
n_cols_50 = len(category_df_top50)
rect2 = patches.Rectangle(
    (0, -0.55), n_cols_50, 0.55,
    linewidth=1.5, edgecolor='black', facecolor='white',
    clip_on=False, zorder=10, transform=ax2.transData
)
ax2.add_patch(rect2)
ax2.text(n_cols_50 / 2, -0.275, "Overlap of Top 50 Features",
        ha='center', va='center', fontsize=20, fontweight='bold',
        fontfamily='sans-serif', zorder=11, transform=ax2.transData)

ax2.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()

# Save figure
output_path = output_dir / "feature_overlap_top20_top50_combined.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white', pad_inches=0.5)
print(f"\n✓ Combined heatmap saved to: {output_path}")

output_path_pdf = output_dir / "feature_overlap_top20_top50_combined.pdf"
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', pad_inches=0.5)
print(f"✓ PDF saved to: {output_path_pdf}")

plt.close()

# Save CSV files
csv_path_top20 = output_dir / "feature_overlap_top20_category_means.csv"
category_df_top20.to_csv(csv_path_top20, index=False)
print(f"✓ Top 20 category means saved to: {csv_path_top20}")

csv_path_top50 = output_dir / "feature_overlap_top50_category_means.csv"
category_df_top50.to_csv(csv_path_top50, index=False)
print(f"✓ Top 50 category means saved to: {csv_path_top50}")

detailed_csv_top20 = output_dir / "feature_overlap_top20_detailed.csv"
df_top20.to_csv(detailed_csv_top20, index=False)
print(f"✓ Top 20 detailed results saved to: {detailed_csv_top20}")

detailed_csv_top50 = output_dir / "feature_overlap_top50_detailed.csv"
df_top50.to_csv(detailed_csv_top50, index=False)
print(f"✓ Top 50 detailed results saved to: {detailed_csv_top50}")

# Print comparison statistics
print("\n" + "="*80)
print("COMPARISON: Top 20 vs Top 50")
print("="*80)

for model_pair in model_pairs:
    values_top20 = category_df_top20[model_pair].values
    values_top50 = category_df_top50[model_pair].values

    print(f"\n{model_pair}:")
    print(f"  Top 20 - Mean: {values_top20.mean():5.1f}%  Std: {values_top20.std():5.1f}%")
    print(f"  Top 50 - Mean: {values_top50.mean():5.1f}%  Std: {values_top50.std():5.1f}%")
    print(f"  Difference: {values_top50.mean() - values_top20.mean():+5.1f}%")

print("\n" + "="*80)
print("✅ COMPLETED!")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print(f"  - feature_overlap_top20_top50_combined.png")
print(f"  - feature_overlap_top20_top50_combined.pdf")
print(f"  - feature_overlap_top20_category_means.csv")
print(f"  - feature_overlap_top50_category_means.csv")
print(f"  - feature_overlap_top20_detailed.csv")
print(f"  - feature_overlap_top50_detailed.csv")
print()
