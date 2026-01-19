#!/usr/bin/env python3
"""
Figure 4 plotting script for model comparison at genus level with batch correction
Creates boxplot comparing ElasticNet/RandomForest/TabPFN/MGM across all diseases
with genus-level data, showing intra-cohort, cross-cohort, and LOSO performance
with batch correction methods in LOSO panel
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ------------------ Paths ------------------ #
results_dir = "/ua/jmu27/Micro_bench/results"
output_dir = "/ua/jmu27/Micro_bench/figures/figure4"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "model_comparison_genus_batchcorrection_boxplot.png")

# ------------------ Statistical Testing Functions ------------------ #
def perform_pairwise_model_tests(data_df, models):
    """
    Perform pairwise Wilcoxon tests between models

    Args:
        data_df: DataFrame with 'model' and 'AUC' columns
        models: list of model names to compare

    Returns:
        dict: {(model1, model2): {'p_value', 'p_adjusted', 'significant'}}
    """
    results = {}
    p_values = []
    comparisons = []

    # Pairwise comparisons between all models
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]

            data1 = data_df[data_df['model'] == model1]['AUC'].values
            data2 = data_df[data_df['model'] == model2]['AUC'].values

            if len(data1) < 3 or len(data2) < 3:
                continue

            # Ensure same length for paired test
            min_len = min(len(data1), len(data2))
            if min_len < 3:
                continue

            data1_paired = data1[:min_len]
            data2_paired = data2[:min_len]

            try:
                # Perform Wilcoxon signed-rank test (two-sided)
                stat, p_value = wilcoxon(data1_paired, data2_paired, alternative='two-sided')
                p_values.append(p_value)
                comparisons.append((model1, model2))
            except Exception as e:
                print(f"Warning: Wilcoxon test failed for {model1} vs {model2}: {e}")
                continue

    # Apply Benjamini-Hochberg correction
    if p_values:
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        for i, (model1, model2) in enumerate(comparisons):
            results[(model1, model2)] = {
                'p_value': p_values[i],
                'p_adjusted': p_adjusted[i],
                'significant': reject[i]
            }

    return results

def perform_loso_batchcorrection_tests(data_df, models):
    """
    Perform Wilcoxon tests comparing batch correction methods to baseline for LOSO

    Args:
        data_df: DataFrame with 'model', 'method', and 'AUC' columns (LOSO data only)
        models: list of model names

    Returns:
        dict: {(model, method): {'p_value', 'p_adjusted', 'significant'}}
    """
    results = {}
    p_values = []
    comparisons = []

    # For each model, compare batch correction methods against baseline
    for model in models:
        if model == 'MGM':  # MGM doesn't have batch correction
            continue

        baseline_data = data_df[
            (data_df['model'] == model) &
            (data_df['method'] == 'baseline')
        ]['AUC'].values

        if len(baseline_data) < 3:
            continue

        for method in ['DebiasM', 'ComBat', 'MMUPHin']:
            method_data = data_df[
                (data_df['model'] == model) &
                (data_df['method'] == method)
            ]['AUC'].values

            if len(method_data) < 3:
                continue

            # Ensure same length (paired test)
            min_len = min(len(baseline_data), len(method_data))
            if min_len < 3:
                continue

            baseline_paired = baseline_data[:min_len]
            method_paired = method_data[:min_len]

            try:
                # Perform Wilcoxon signed-rank test (paired, one-tailed)
                stat, p_value = wilcoxon(method_paired, baseline_paired, alternative='greater')
                p_values.append(p_value)
                comparisons.append((model, method))
            except Exception as e:
                print(f"Warning: Wilcoxon test failed for {model} + {method}: {e}")
                continue

    # Apply Benjamini-Hochberg correction
    if p_values:
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        for i, (model, method) in enumerate(comparisons):
            results[(model, method)] = {
                'p_value': p_values[i],
                'p_adjusted': p_adjusted[i],
                'significant': reject[i]
            }

    return results

def draw_significance_bracket(ax, x1, x2, y, p_adjusted, height=0.02):
    """
    Draw a bracket line between two positions with significance stars (below boxplot)
    """
    # Determine significance marker
    if p_adjusted < 0.001:
        marker = '***'
    elif p_adjusted < 0.01:
        marker = '**'
    elif p_adjusted < 0.05:
        marker = '*'
    else:
        return

    # Draw inverted bracket (U-shape below boxplot)
    ax.plot([x1, x1, x2, x2], [y+height, y, y, y+height],
            '-', linewidth=1.5, color='black', zorder=5)

    # Add significance marker below the bracket
    ax.text((x1 + x2) / 2, y - 0.015, marker,
            ha='center', va='top', fontsize=14, fontweight='bold', color='black', zorder=5)

def add_pairwise_significance_brackets(ax, positions, data_df, model_order, stats_results, use_q1=False, base_offset=-0.06):
    """
    Add significance brackets for pairwise model comparisons (below boxplots)
    Only show comparisons with MGM

    Args:
        use_q1: if True, position brackets below Q1 (lower quantile) instead of minimum value
        base_offset: offset from reference point (default -0.06)
    """
    if not stats_results:
        return

    # Get data for positioning brackets
    ref_values = []
    for model in model_order:
        model_data = data_df[data_df['model'] == model]['AUC'].values
        if len(model_data) > 0:
            if use_q1:
                # Use 25th percentile (Q1 / lower quantile)
                ref_values.append(np.percentile(model_data, 25))
            else:
                # Use minimum value
                ref_values.append(min(model_data))
        else:
            ref_values.append(0)

    # Base y position for brackets (below reference point)
    base_y = min(ref_values) + base_offset

    # Draw brackets for significant comparisons with MGM only
    # Use different y levels for different comparisons to avoid overlap
    bracket_levels = {}
    level = 0
    for (model1, model2) in stats_results.keys():
        # Only include comparisons involving MGM
        if 'MGM' in [model1, model2]:
            bracket_levels[(model1, model2)] = level
            level += 1

    for (model1, model2), result in stats_results.items():
        # Only draw brackets for comparisons involving MGM
        if 'MGM' not in [model1, model2]:
            continue

        if result['significant']:
            # Get positions
            try:
                idx1 = model_order.index(model1)
                idx2 = model_order.index(model2)
            except ValueError:
                continue

            x1 = positions[idx1]
            x2 = positions[idx2]

            # Determine bracket height level
            level_offset = bracket_levels.get((model1, model2), 0) * -0.06
            y = base_y + level_offset

            # Draw bracket
            draw_significance_bracket(ax, x1, x2, y, result['p_adjusted'])

def add_loso_significance_brackets(ax, positions_dict, data_df, model_order, methods, stats_results):
    """
    Add significance brackets in LOSO panel connecting batch correction methods to baseline
    """
    if not stats_results:
        return

    for model in model_order:
        if model == 'MGM':  # Skip MGM
            continue

        if model not in positions_dict:
            continue

        model_positions = positions_dict[model]
        baseline_pos = model_positions[0]  # First position is baseline

        # Get minimum value for positioning brackets below
        model_loso_data = data_df[data_df['model'] == model]['AUC'].values
        if len(model_loso_data) == 0:
            continue

        base_y = min(model_loso_data) - 0.06

        # Draw brackets from baseline to each significant batch correction method
        bracket_offset = 0
        for idx, (method_key, method_display) in enumerate(methods):
            if method_key == 'baseline':
                continue

            if (model, method_key) in stats_results:
                result = stats_results[(model, method_key)]

                if result['significant']:
                    method_pos = model_positions[idx]

                    # Calculate y position with stacking offset
                    y = base_y + bracket_offset
                    bracket_offset -= 0.05  # Stack next bracket lower

                    # Draw bracket from baseline to this method
                    draw_significance_bracket(ax, baseline_pos, method_pos, y, result['p_adjusted'])

# ------------------ Helper Functions ------------------ #
def parse_filename(fname: str):
    """Extract disease name from filename"""
    stem = os.path.basename(fname).replace("_result.csv", "").replace("_batchcorrection_result.csv", "")
    parts = stem.split("_")
    disease = parts[0] if len(parts) > 0 else None
    return disease

def analyze_matrix_df(df: pd.DataFrame):
    """
    Extract intra (diagonal), cross (off-diagonal), and loso values from result matrix.
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
                            intra_vals.append(float(val))
                        else:
                            cross_vals.append(float(val))
    except Exception as e:
        print(f"Error analyzing matrix: {e}")

    return intra_vals, cross_vals, loso_vals

def parse_batchcorrection_result(filepath):
    """
    Parse batch correction result file (LOSO data with batch correction methods)
    Returns dict: {model_method: [values]}
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

    results = {}

    # Each row is a model_method combination
    for idx, row in df.iterrows():
        model_method = row['model']

        # Get all numeric columns (study results)
        numeric_cols = [col for col in df.columns if col != 'model']
        values = []

        for col in numeric_cols:
            if pd.notna(row[col]):
                values.append(float(row[col]))

        if values:
            results[model_method] = values

    return results

def load_regular_results(file_path: str, model_name: str):
    """
    Load results from regular result file and extract intra/cross/loso baseline values.
    """
    disease = parse_filename(file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    # For MGM files, there's only one model
    # For regular files, filter by model name
    if 'model' in df.columns and model_name != 'MGM':
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
            'eval_type': 'intra',
            'method': 'baseline',
            'AUC': val
        })

    # Add cross records
    for val in cross:
        records.append({
            'disease': disease,
            'model': model_name,
            'eval_type': 'cross',
            'method': 'baseline',
            'AUC': val
        })

    # Add loso baseline records
    for val in loso:
        records.append({
            'disease': disease,
            'model': model_name,
            'eval_type': 'loso',
            'method': 'baseline',
            'AUC': val
        })

    return records

def load_batchcorrection_results(file_path: str):
    """
    Load batch correction results (LOSO with batch correction methods)
    """
    disease = parse_filename(file_path)

    batch_results = parse_batchcorrection_result(file_path)

    records = []

    for model_method, values in batch_results.items():
        # Parse model and method
        # Format: "ElasticNet_DebiasM", "RandomForest_ComBat", etc.
        parts = model_method.split('_')
        if len(parts) >= 2:
            model = parts[0]
            method = parts[1]

            for val in values:
                records.append({
                    'disease': disease,
                    'model': model,
                    'eval_type': 'loso',
                    'method': method,
                    'AUC': val
                })

    return records

# ------------------ Gather files ------------------ #
print("Gathering genus-level result files...")

# Get all genus-level files with "none" preprocessing (for ElasticNet, RF, TabPFN)
classic_files = glob.glob(os.path.join(results_dir, "*_*_genus_none_result.csv"))

# Get batch correction files
batch_files = glob.glob(os.path.join(results_dir, "*_*_genus_none_batchcorrection_result.csv"))

# Get all genus-level MGM files
mgm_files = glob.glob(os.path.join(results_dir, "*_*_genus_mgm_result.csv"))

print(f"Found {len(classic_files)} classic model result files")
print(f"Found {len(batch_files)} batch correction result files")
print(f"Found {len(mgm_files)} MGM result files")

# ------------------ Load data ------------------ #
records = []

# Load ElasticNet, RandomForest, TabPFN from classic files (intra/cross/loso baseline)
print("\nLoading classic model results...")
for file_path in classic_files:
    for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
        records.extend(load_regular_results(file_path, model))

# Load batch correction results (LOSO with batch correction methods)
print("Loading batch correction results...")
for file_path in batch_files:
    records.extend(load_batchcorrection_results(file_path))

# Load MGM from mgm files
print("Loading MGM results...")
for file_path in mgm_files:
    records.extend(load_regular_results(file_path, 'MGM'))

# Create DataFrame
df_all = pd.DataFrame.from_records(records)

print(f"\nTotal records collected: {len(df_all)}")
print(f"Diseases included: {df_all['disease'].nunique()}")
print(f"Models: {df_all['model'].unique()}")

# Check data distribution
print("\nData distribution:")
print(df_all.groupby(['model', 'eval_type', 'method']).size())

# ------------------ Create plot with 3 panels ------------------ #
# Set matplotlib parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

# Color scheme for models (matching Figure 3)
model_colors = {
    'ElasticNet': '#C5B0D5',   # Light purple/lavender
    'RandomForest': '#AEC7E8',  # Light blue
    'TabPFN': '#FFBB78',        # Light orange
    'MGM': '#FF9896'            # Light red
}

# Model abbreviations
model_abbrev = {
    'ElasticNet': 'Enet',
    'RandomForest': 'RF',
    'TabPFN': 'TabPFN',
    'MGM': 'MGM'
}

# Perform statistical tests
print("\nPerforming statistical tests...")

# Define model list
all_models = ['ElasticNet', 'RandomForest', 'TabPFN', 'MGM']

# Intra and Cross: pairwise model comparisons
intra_data_df = df_all[df_all['eval_type'] == 'intra']
cross_data_df = df_all[df_all['eval_type'] == 'cross']
loso_data_df = df_all[df_all['eval_type'] == 'loso']

stats_intra = perform_pairwise_model_tests(intra_data_df, all_models)
stats_cross = perform_pairwise_model_tests(cross_data_df, all_models)

# LOSO: batch correction vs baseline
stats_loso = perform_loso_batchcorrection_tests(loso_data_df, all_models)

# Print significant results
print(f"\nIntra-cohort significant comparisons (BH-corrected p < 0.05):")
if stats_intra:
    for (model1, model2), result in stats_intra.items():
        if result['significant']:
            print(f"  {model1} vs {model2}: p={result['p_value']:.4f}, p_adj={result['p_adjusted']:.4f} {'***' if result['p_adjusted']<0.001 else '**' if result['p_adjusted']<0.01 else '*'}")
else:
    print("  None")

print(f"\nCross-cohort significant comparisons (BH-corrected p < 0.05):")
if stats_cross:
    for (model1, model2), result in stats_cross.items():
        if result['significant']:
            print(f"  {model1} vs {model2}: p={result['p_value']:.4f}, p_adj={result['p_adjusted']:.4f} {'***' if result['p_adjusted']<0.001 else '**' if result['p_adjusted']<0.01 else '*'}")
else:
    print("  None")

print(f"\nLOSO batch correction significant comparisons (BH-corrected p < 0.05):")
if stats_loso:
    for (model, method), result in stats_loso.items():
        if result['significant']:
            print(f"  {model} + {method}: p={result['p_value']:.4f}, p_adj={result['p_adjusted']:.4f} {'***' if result['p_adjusted']<0.001 else '**' if result['p_adjusted']<0.01 else '*'}")
else:
    print("  None")

# Create figure with 3 panels: Intra, Cross, LOSO
# Adjust subplot widths to match figure3
fig = plt.figure(figsize=(22, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 0.8, 2.4])
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

# Panel 1: Intra-cohort
ax_intra = axes[0]
intra_data_df = df_all[df_all['eval_type'] == 'intra']

intra_data = []
intra_positions = []
intra_colors = []
intra_labels = []

pos = 1
for model in ['ElasticNet', 'RandomForest', 'TabPFN', 'MGM']:
    model_data = intra_data_df[intra_data_df['model'] == model]['AUC'].values
    if len(model_data) > 0:
        intra_data.append(model_data)
        intra_positions.append(pos)
        intra_colors.append(model_colors[model])
        intra_labels.append(model_abbrev[model])
        pos += 1

if intra_data:
    bp = ax_intra.boxplot(intra_data, positions=intra_positions, widths=0.6,
                         patch_artist=True, showfliers=True,
                         boxprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='black'),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, color in zip(bp['boxes'], intra_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax_intra.set_ylabel('AUC', fontsize=16)
ax_intra.set_ylim(0.0, 1.0)  # Extended lower limit for brackets
ax_intra.set_xticks(intra_positions)
ax_intra.set_xticklabels(intra_labels, rotation=0, ha='center', fontsize=16)
ax_intra.grid(axis='y', alpha=0.3, linestyle='--')

# Add significance brackets for Intra panel
if stats_intra and intra_data:
    model_order = [m for m in all_models if m in intra_data_df['model'].values and len(intra_data_df[intra_data_df['model'] == m]) > 0]
    add_pairwise_significance_brackets(ax_intra, intra_positions, intra_data_df,
                                      model_order, stats_intra)

# Add title with white box
rect = patches.Rectangle(
    (0, 1.02), 1.0, 0.08,
    linewidth=1.5, edgecolor='black', facecolor='white',
    clip_on=False, zorder=10, transform=ax_intra.transAxes
)
ax_intra.add_patch(rect)
ax_intra.text(0.5, 1.06, 'Intra-cohort',
             ha='center', va='center', fontsize=18, fontweight='bold',
             zorder=11, transform=ax_intra.transAxes)

# Panel 2: Cross-cohort
ax_cross = axes[1]
cross_data_df = df_all[df_all['eval_type'] == 'cross']

cross_data = []
cross_positions = []
cross_colors = []
cross_labels = []

pos = 1
for model in ['ElasticNet', 'RandomForest', 'TabPFN', 'MGM']:
    model_data = cross_data_df[cross_data_df['model'] == model]['AUC'].values
    if len(model_data) > 0:
        cross_data.append(model_data)
        cross_positions.append(pos)
        cross_colors.append(model_colors[model])
        cross_labels.append(model_abbrev[model])
        pos += 1

if cross_data:
    bp = ax_cross.boxplot(cross_data, positions=cross_positions, widths=0.6,
                         patch_artist=True, showfliers=True,
                         boxprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='black'),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5),
                         flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, color in zip(bp['boxes'], cross_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax_cross.set_ylabel('')
ax_cross.set_ylim(0.0, 1.0)  # Extended lower limit for brackets
ax_cross.set_xticks(cross_positions)
ax_cross.set_xticklabels(cross_labels, rotation=0, ha='center', fontsize=16)
ax_cross.grid(axis='y', alpha=0.3, linestyle='--')

# Add significance brackets for Cross panel (higher position)
if stats_cross and cross_data:
    model_order = [m for m in all_models if m in cross_data_df['model'].values and len(cross_data_df[cross_data_df['model'] == m]) > 0]
    add_pairwise_significance_brackets(ax_cross, cross_positions, cross_data_df,
                                      model_order, stats_cross, use_q1=False, base_offset=0.05)

# Add title with white box
rect = patches.Rectangle(
    (0, 1.02), 1.0, 0.08,
    linewidth=1.5, edgecolor='black', facecolor='white',
    clip_on=False, zorder=10, transform=ax_cross.transAxes
)
ax_cross.add_patch(rect)
ax_cross.text(0.5, 1.06, 'Cross-cohort',
             ha='center', va='center', fontsize=18, fontweight='bold',
             zorder=11, transform=ax_cross.transAxes)

# Panel 3: LOSO with batch correction methods
ax_loso = axes[2]

loso_data = []
loso_positions = []
loso_colors = []
loso_labels = []
loso_positions_dict = {}  # Track positions for each model: {model: [positions]}

pos = 1
methods = [
    ('baseline', None),  # baseline - only show model name
    ('DebiasM', 'DebiasM'),
    ('ComBat', 'ComBat'),
    ('MMUPHin', 'MMUPHin')
]

# Changed order: iterate by method first, then by model (matching figure3)
for method_key, method_display in methods:
    for model in ['ElasticNet', 'RandomForest', 'TabPFN', 'MGM']:
        # For MGM, only show baseline (no batch correction methods)
        if model == 'MGM' and method_key != 'baseline':
            continue

        model_method_data = loso_data_df[
            (loso_data_df['model'] == model) &
            (loso_data_df['method'] == method_key)
        ]['AUC'].values

        if len(model_method_data) > 0:
            loso_data.append(model_method_data)
            loso_positions.append(pos)
            loso_colors.append(model_colors[model])

            # Track positions for each model
            if model not in loso_positions_dict:
                loso_positions_dict[model] = []
            loso_positions_dict[model].append(pos)

            # Create label
            if method_display is None:
                # Baseline: only model name
                loso_labels.append(model_abbrev[model])
            else:
                # Batch correction: model + method
                loso_labels.append(f"{model_abbrev[model]}\n{method_display}")

            pos += 3.0  # Spacing within method group

    # Spacing between method groups
    pos += 1.5

if loso_data:
    bp = ax_loso.boxplot(loso_data, positions=loso_positions, widths=2.5,
                        patch_artist=True, showfliers=True,
                        boxprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='black'),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for patch, color in zip(bp['boxes'], loso_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax_loso.set_ylabel('')
ax_loso.set_ylim(0.0, 1.0)  # Extended lower limit for brackets
ax_loso.set_xticks(loso_positions)
ax_loso.set_xticklabels(loso_labels, rotation=0, ha='center', fontsize=9)
ax_loso.grid(axis='y', alpha=0.3, linestyle='--')

# Add significance brackets for LOSO panel
if stats_loso and loso_data:
    add_loso_significance_brackets(ax_loso, loso_positions_dict, loso_data_df,
                                  all_models, methods, stats_loso)

# Add title with white box
rect = patches.Rectangle(
    (0, 1.02), 1.0, 0.08,
    linewidth=1.5, edgecolor='black', facecolor='white',
    clip_on=False, zorder=10, transform=ax_loso.transAxes
)
ax_loso.add_patch(rect)
ax_loso.text(0.5, 1.06, 'Leave-One-Study-Out',
             ha='center', va='center', fontsize=18, fontweight='bold',
             zorder=11, transform=ax_loso.transAxes)

plt.tight_layout()

# Save figure
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
plt.close()

print(f"\n✅ Box plot with batch correction saved to: {output_path}")
