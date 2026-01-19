#!/usr/bin/env python3
"""
Plot LOSO AUC and sample sizes for MGM, TabPFN, and RandomForest at genus level
Left panel: Sample sizes per disease
Right panels: LOSO AUROC values with range and mean for MGM, TabPFN, and RandomForest
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------ Configuration ------------------ #
results_dir = "/ua/jmu27/Micro_bench/results"
rawdata_dir = "/ua/jmu27/Micro_bench/data/rawdata"
output_dir = "/ua/jmu27/Micro_bench/figures/figure4"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "loso_sample_auc_genus_mgm.png")

# Disease type mapping
disease_type_map = {
    "CRC": "Intestinal", "T2D": "Metabolic", "Obesity": "Metabolic", "Overweight": "Metabolic",
    "Adenoma": "Intestinal", "CDI": "Intestinal", "AD": "Mental", "MCI": "Mental", "PD": "Mental",
    "RA": "Autoimmun", "MS": "Autoimmun", "ASD": "Mental", "CD": "Intestinal", "UC": "Intestinal",
    "IBD": "Intestinal", "AS": "Autoimmun", "IBS": "Intestinal", "CFS": "Mental",
    "JA": "Autoimmun", "NAFLD": "Liver"
}

# Order of disease types for plotting
disease_type_order = ["Autoimmun", "Intestinal", "Liver", "Mental", "Metabolic"]

# ------------------ Helper Functions ------------------ #
def parse_filename(fname: str):
    """Extract disease name from filename"""
    stem = os.path.basename(fname).replace("_result.csv", "")
    parts = stem.split("_")
    disease = parts[0] if len(parts) > 0 else None
    return disease

def extract_loso_values_with_cohorts(df: pd.DataFrame, model_name=None):
    """Extract LOSO test AUC values with cohort names from result file

    Args:
        df: DataFrame with results
        model_name: For classic models (ElasticNet, RandomForest, TabPFN), specify which model to extract
    """
    # Get numeric columns (test cohorts)
    numeric_cols = [col for col in df.columns if col not in ['model', 'Unnamed: 0', '']]
    numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(df[col]) or
                    all(pd.to_numeric(df[col], errors='coerce').notna())]

    if len(numeric_cols) == 0:
        return {}, []

    # Find LOSO row (labeled as 'lodo' or 'loso')
    loso_row_idx = None
    if 'model' in df.columns:
        model_col_vals = df['model'].astype(str).str.lower()

        # If model_name is specified (for classic models), filter by model first
        if model_name:
            model_mask = df['model'].astype(str) == model_name
            df_filtered = df[model_mask]
            if len(df_filtered) == 0:
                return {}, []
            model_col_vals = df_filtered['model'].astype(str).str.lower()
        else:
            df_filtered = df

        loso_mask = model_col_vals.isin(['lodo', 'loso'])
        if loso_mask.any():
            loso_row_idx = df_filtered[loso_mask].index[0]

    # If not found by label, assume last row is LOSO
    if loso_row_idx is None:
        if model_name and 'model' in df.columns:
            model_mask = df['model'].astype(str) == model_name
            df_filtered = df[model_mask]
            if len(df_filtered) > 0:
                loso_row_idx = df_filtered.index[-1]
            else:
                return {}, []
        else:
            loso_row_idx = df.index[-1]

    # Extract LOSO values
    loso_row = df.loc[loso_row_idx, numeric_cols]
    loso_dict = {}
    loso_values = []

    for cohort, val in zip(numeric_cols, loso_row):
        val_float = pd.to_numeric(val, errors="coerce")
        if pd.notna(val_float):
            loso_dict[cohort] = val_float
            loso_values.append(val_float)

    return loso_dict, loso_values

def get_sample_size_per_cohort(disease: str):
    """Get sample size for each cohort of a disease from rawdata"""
    # Try to find rawdata file matching disease_*_genus.csv
    pattern = os.path.join(rawdata_dir, f"{disease}_*_genus.csv")
    files = glob.glob(pattern)

    if not files:
        return {}

    # Use the first matching file
    try:
        df = pd.read_csv(files[0])

        # Use pair_rank column to identify cohorts
        if 'pair_rank' in df.columns:
            cohort_sizes = {}
            for pair_rank in sorted(df['pair_rank'].unique()):
                cohort_df = df[df['pair_rank'] == pair_rank]
                # Count unique samples
                if 'Row.names' in df.columns:
                    size = cohort_df['Row.names'].nunique()
                elif 'sample_id' in df.columns:
                    size = cohort_df['sample_id'].nunique()
                elif 'SampleID' in df.columns:
                    size = cohort_df['SampleID'].nunique()
                else:
                    size = len(cohort_df)
                # Store with pair_rank as string (matching column names in result files)
                cohort_sizes[str(int(pair_rank))] = size
            return cohort_sizes

        # Fallback: return total
        if 'Row.names' in df.columns:
            return {'All': df['Row.names'].nunique()}
        elif 'sample_id' in df.columns:
            return {'All': df['sample_id'].nunique()}
        else:
            return {'All': len(df)}

    except Exception as e:
        print(f"Error reading {files[0]}: {e}")
        return {}

# ------------------ Load Data ------------------ #
# Get all MGM genus result files
mgm_files = glob.glob(os.path.join(results_dir, "*_*_genus_mgm_result.csv"))
# Get all classic model result files (log_std normalization)
classic_files = glob.glob(os.path.join(results_dir, "*_*_genus_log_std_result.csv"))

print(f"Found {len(mgm_files)} MGM genus result files")
print(f"Found {len(classic_files)} classic model result files")

# Collect LOSO AUC values and sample sizes for each cohort in each disease
# Store separately for each model
model_data = {'MGM': [], 'TabPFN': [], 'RandomForest': []}

# Process MGM files
for file_path in mgm_files:
    disease = parse_filename(file_path)

    if disease not in disease_type_map:
        continue

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

    # Extract LOSO values with cohort information
    loso_dict, loso_values = extract_loso_values_with_cohorts(df)

    if len(loso_dict) > 0:
        # Get sample sizes per cohort
        cohort_sizes = get_sample_size_per_cohort(disease)

        # Create entry for each cohort
        for cohort, auc in loso_dict.items():
            cohort_size = cohort_sizes.get(cohort, np.nan)
            model_data['MGM'].append({
                'disease': disease,
                'cohort': cohort,
                'disease_type': disease_type_map[disease],
                'sample_size': cohort_size,
                'loso_auc': auc,
                'all_loso_aucs': loso_values  # Store all values for range
            })

# Process classic model files (TabPFN and RandomForest)
for file_path in classic_files:
    disease = parse_filename(file_path)

    if disease not in disease_type_map:
        continue

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

    # Extract LOSO values for TabPFN and RandomForest
    for model_name in ['TabPFN', 'RandomForest']:
        loso_dict, loso_values = extract_loso_values_with_cohorts(df, model_name=model_name)

        if len(loso_dict) > 0:
            # Get sample sizes per cohort
            cohort_sizes = get_sample_size_per_cohort(disease)

            # Create entry for each cohort
            for cohort, auc in loso_dict.items():
                cohort_size = cohort_sizes.get(cohort, np.nan)
                model_data[model_name].append({
                    'disease': disease,
                    'cohort': cohort,
                    'disease_type': disease_type_map[disease],
                    'sample_size': cohort_size,
                    'loso_auc': auc,
                    'all_loso_aucs': loso_values  # Store all values for range
                })

print(f"Collected data for MGM: {len(model_data['MGM'])} cohorts")
print(f"Collected data for TabPFN: {len(model_data['TabPFN'])} cohorts")
print(f"Collected data for RandomForest: {len(model_data['RandomForest'])} cohorts")

# Aggregate by disease for each model
disease_aggregated_by_model = {}
for model_name in ['MGM', 'TabPFN', 'RandomForest']:
    cohort_df = pd.DataFrame(model_data[model_name])

    disease_aggregated = []
    for disease in cohort_df['disease'].unique():
        disease_data = cohort_df[cohort_df['disease'] == disease]

        # Collect all LOSO AUC values from all cohorts
        all_loso_values = []
        for val_list in disease_data['all_loso_aucs'].values:
            # Each entry already contains all LOSO values, just take the first one
            all_loso_values = val_list
            break

        # Sum sample sizes across all cohorts
        total_sample_size = disease_data['sample_size'].sum()

        # Calculate mean AUC
        mean_auc = np.mean(all_loso_values) if len(all_loso_values) > 0 else np.nan

        disease_aggregated.append({
            'disease': disease,
            'disease_type': disease_data['disease_type'].iloc[0],
            'total_sample_size': total_sample_size,
            'all_loso_values': all_loso_values,
            'mean_auc': mean_auc,
            'n_cohorts': len(disease_data)
        })

    disease_aggregated_by_model[model_name] = pd.DataFrame(disease_aggregated)

# Use MGM data to determine disease order (since sample sizes are the same)
disease_df = disease_aggregated_by_model['MGM']

# Sort by disease type and disease name
disease_df['disease_type'] = pd.Categorical(disease_df['disease_type'],
                                           categories=disease_type_order, ordered=True)
disease_df = disease_df.sort_values(['disease_type', 'disease'])

print(f"Aggregated into {len(disease_df)} diseases")
print(f"Sample sizes - NA count: {disease_df['total_sample_size'].isna().sum()}, Valid count: {disease_df['total_sample_size'].notna().sum()}")
if disease_df['total_sample_size'].notna().any():
    print(f"Sample size range: {disease_df['total_sample_size'].min():.0f} - {disease_df['total_sample_size'].max():.0f}")

# ------------------ Create Figure ------------------ #
fig = plt.figure(figsize=(18, 0.5 * len(disease_df) + 2))
gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 2, 2, 2], wspace=0.05)
ax1 = fig.add_subplot(gs[0])  # Sample sizes
ax2 = fig.add_subplot(gs[1])  # MGM
ax3 = fig.add_subplot(gs[2])  # TabPFN
ax4 = fig.add_subplot(gs[3])  # RandomForest

# Get disease list in order
disease_names = disease_df['disease'].values
n_diseases = len(disease_df)
y_positions = np.arange(n_diseases)

# Color mapping for disease types
disease_type_colors = {
    'Autoimmun': '#3B63B8',    # Dark Blue
    'Intestinal': '#F76889',   # Pink/Red
    'Liver': '#991B46',        # Dark Red/Burgundy
    'Mental': '#F4C756',       # Yellow/Gold
    'Metabolic': '#8EA8DE'     # Light Blue
}

# Left panel: Sample sizes (horizontal bar chart)
for i, (idx, row) in enumerate(disease_df.iterrows()):
    color = disease_type_colors.get(row['disease_type'], 'gray')

    if pd.notna(row['total_sample_size']):
        # Create hollow bar with colored edge (matching reference style)
        ax1.barh(i, row['total_sample_size'], height=0.6, align='center',
                edgecolor=color, color='none', linewidth=2.5, zorder=3)

        # Add text label for sample size
        if row['total_sample_size'] > 100:
            ax1.text(row['total_sample_size'] - max(disease_df['total_sample_size'].dropna()) * 0.05,
                    i, int(row['total_sample_size']),
                    va='center', ha='right', fontsize=10)

ax1.set_yticks(y_positions)
ax1.set_yticklabels(disease_names, fontsize=11)
ax1.set_ylim(-0.5, n_diseases - 0.5)
ax1.set_xlabel('N samples', fontsize=14)
ax1.invert_xaxis()  # Invert x-axis to match reference
ax1.xaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(True)
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.tick_params(axis='y', pad=5)

# Right panels: LOSO AUC values for MGM, TabPFN, RandomForest
for ax, model_name in [(ax2, 'MGM'), (ax3, 'TabPFN'), (ax4, 'RandomForest')]:
    model_df = disease_aggregated_by_model[model_name]
    # Sort to match disease_df order
    model_df['disease_type'] = pd.Categorical(model_df['disease_type'],
                                               categories=disease_type_order, ordered=True)
    model_df = model_df.sort_values(['disease_type', 'disease'])

    for i, (idx, row) in enumerate(model_df.iterrows()):
        color = disease_type_colors.get(row['disease_type'], 'gray')
        all_aucs = row['all_loso_values']

        if len(all_aucs) > 0:
            # Plot range line (min to max)
            if len(all_aucs) > 1:
                min_auc = np.min(all_aucs)
                max_auc = np.max(all_aucs)
                ax.plot([min_auc, max_auc], [i, i],
                        color=color, linewidth=2.5, solid_capstyle='butt', zorder=2)

            # Plot all individual LOSO values as scatter points
            ax.scatter(all_aucs, [i] * len(all_aucs),
                       color=color, alpha=0.3, s=40, edgecolors='none',
                       linewidths=0, zorder=2)

            # Highlight mean with diamond marker
            mean_auc = row['mean_auc']
            ax.scatter(mean_auc, i, marker='D', s=60,
                       facecolors=color, edgecolors='black', linewidths=1, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([])  # No labels on right panels
    ax.set_ylim(-0.5, n_diseases - 0.5)
    ax.set_xlabel('AUROC', fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.xaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

# Add legend for disease categories on MGM panel lower left
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='none', edgecolor=disease_type_colors['Autoimmun'],
          label='Autoimmun', linewidth=2.5),
    Patch(facecolor='none', edgecolor=disease_type_colors['Intestinal'],
          label='Intestinal', linewidth=2.5),
    Patch(facecolor='none', edgecolor=disease_type_colors['Liver'],
          label='Liver', linewidth=2.5),
    Patch(facecolor='none', edgecolor=disease_type_colors['Mental'],
          label='Mental', linewidth=2.5),
    Patch(facecolor='none', edgecolor=disease_type_colors['Metabolic'],
          label='Metabolic', linewidth=2.5)
]
ax2.legend(handles=legend_elements, loc='lower left', fontsize=11,
          frameon=True, fancybox=False, edgecolor='black')

# Add white header bars at the top of each model panel
for ax, model_name in [(ax2, 'MGM'), (ax3, 'TabPFN'), (ax4, 'RandomForest')]:
    rect_top = patches.Rectangle((0, 1.01), 1, 0.06, transform=ax.transAxes,
                                 facecolor='white', edgecolor='black', linewidth=1.5, clip_on=False)
    ax.add_patch(rect_top)
    ax.text(0.5, 1.04, model_name, transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='black')

plt.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.08)
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ Figure saved to: {save_path}")
