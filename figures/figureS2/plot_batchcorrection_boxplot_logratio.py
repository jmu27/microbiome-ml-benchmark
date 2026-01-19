#!/usr/bin/env python3
"""
Figure S2 plotting script for batch correction comparison with log-ratio preprocessing
Creates boxplots showing model performance across intra, cross, and LOSO scenarios
with different batch correction methods in LOSO using CLR preprocessing
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.font_manager as fm
from pathlib import Path
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# Font setup
def check_helvetica_font():
    """Check if Helvetica font is available"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    helvetica_variants = ['Helvetica', 'Helvetica Neue', 'HelveticaNeue']

    available_helvetica = [font for font in helvetica_variants if font in available_fonts]

    if available_helvetica:
        print(f"✓ Helvetica font available: {available_helvetica}")
        return available_helvetica[0]
    else:
        print("⚠ Helvetica not available, using Arial")
        return 'Arial'

# Check and set best font
best_font = check_helvetica_font()

# Set matplotlib parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [best_font, 'Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Disease type mapping
disease_type_map = {
    'CRC': 'Intestinal', 'T2D': 'Metabolic', 'Obesity': 'Metabolic', 'Overweight': 'Metabolic',
    'Adenoma': 'Intestinal', 'CDI': 'Intestinal', 'AD': 'Mental', 'MCI': 'Mental', 'PD': 'Mental',
    'RA': 'Autoimmun', 'MS': 'Autoimmun', 'ASD': 'Mental', 'CD': 'Intestinal', 'UC': 'Intestinal',
    'IBD': 'Intestinal', 'AS': 'Autoimmun', 'IBS': 'Intestinal', 'CFS': 'Mental',
    'JA': 'Autoimmun', 'NAFLD': 'Liver'
}

def parse_regular_result(filepath, model_name):
    """
    Parse regular result file (intra/cross/loso data) for a specific model
    Returns dict with 'intra', 'cross', and 'loso' values

    File structure:
    - Rows represent training cohorts (last row is LOSO baseline)
    - Columns represent test cohorts
    - Intra: diagonal (train and test on same cohort)
    - Cross: off-diagonal (train on one, test on another)
    - LOSO: last row (baseline leave-one-study-out)
    """
    df = pd.read_csv(filepath)

    results = {
        'intra': [],
        'cross': [],
        'loso': []
    }

    # Filter for specific model
    model_data = df[df['model'] == model_name]

    if len(model_data) == 0:
        return results

    # Get numeric columns (cohort columns)
    numeric_cols = [col for col in model_data.columns if col not in ['model']]

    # Convert to numpy array for easier indexing
    data_matrix = model_data[numeric_cols].values

    # Extract values
    n_rows, n_cols = data_matrix.shape

    # Last row is LOSO baseline
    if n_rows > 0:
        last_row = data_matrix[-1, :]
        for value in last_row:
            if pd.notna(value):
                results['loso'].append(value)

    # All rows except last: intra (diagonal) and cross (off-diagonal)
    for i in range(n_rows - 1):  # Exclude last row
        for j in range(n_cols):
            value = data_matrix[i, j]
            if pd.notna(value):
                if i == j:
                    # Diagonal - intra cohort
                    results['intra'].append(value)
                else:
                    # Off-diagonal - cross cohort
                    results['cross'].append(value)

    return results

def parse_batchcorrection_result(filepath):
    """
    Parse batch correction result file (LOSO data)
    Returns dict: {model_method: [values]}
    """
    df = pd.read_csv(filepath)

    results = {}

    # Each row is a model_method combination
    for idx, row in df.iterrows():
        model_method = row['model']

        # Get all numeric columns (study results)
        numeric_cols = [col for col in df.columns if col != 'model']
        values = []

        for col in numeric_cols:
            if pd.notna(row[col]):
                values.append(row[col])

        if values:
            results[model_method] = values

    return results

def collect_data_for_disease_type(results_dir, disease_type, preprocessing='CLR'):
    """
    Collect data for a specific disease type with specified preprocessing

    Args:
        results_dir: Path to results directory
        disease_type: Disease type (e.g., 'Intestinal', 'Metabolic')
        preprocessing: Preprocessing method (default: 'CLR', can be 'ALR', 'CLR', etc.)

    Returns dict:
    {
        'intra': {'ElasticNet': [], 'RandomForest': [], 'TabPFN': []},
        'cross': {'ElasticNet': [], 'RandomForest': [], 'TabPFN': []},
        'loso': {
            'ElasticNet': [], 'ElasticNet_DebiasM': [], 'ElasticNet_ComBat': [], 'ElasticNet_MMUPHin': [],
            'RandomForest': [], 'RandomForest_DebiasM': [], 'RandomForest_ComBat': [], 'RandomForest_MMUPHin': [],
            'TabPFN': [], 'TabPFN_DebiasM': [], 'TabPFN_ComBat': [], 'TabPFN_MMUPHin': []
        }
    }
    """
    data = {
        'intra': {'ElasticNet': [], 'RandomForest': [], 'TabPFN': []},
        'cross': {'ElasticNet': [], 'RandomForest': [], 'TabPFN': []},
        'loso': {
            'ElasticNet': [], 'ElasticNet_DebiasM': [], 'ElasticNet_ComBat': [], 'ElasticNet_MMUPHin': [],
            'RandomForest': [], 'RandomForest_DebiasM': [], 'RandomForest_ComBat': [], 'RandomForest_MMUPHin': [],
            'TabPFN': [], 'TabPFN_DebiasM': [], 'TabPFN_ComBat': [], 'TabPFN_MMUPHin': []
        }
    }

    # Get all diseases of this type
    diseases = [disease for disease, dtype in disease_type_map.items() if dtype == disease_type]

    # Process each disease
    for disease in diseases:
        # Look for files matching: {disease}_{datatype}_{taxonomy}_{preprocessing}_result.csv
        for filename in os.listdir(results_dir):
            if filename.startswith(disease) and filename.endswith(f"{preprocessing}_result.csv"):
                # Read regular results (intra/cross)
                regular_file = os.path.join(results_dir, filename)
                if os.path.exists(regular_file):
                    try:
                        # Process each model's data
                        for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
                            model_results = parse_regular_result(regular_file, model)
                            data['intra'][model].extend(model_results['intra'])
                            data['cross'][model].extend(model_results['cross'])
                            # Last row is baseline LOSO
                            data['loso'][model].extend(model_results['loso'])
                    except Exception as e:
                        print(f"Error reading {regular_file}: {e}")

                # Read batch correction results (LOSO)
                batch_file = os.path.join(results_dir, filename.replace("_result.csv", "_batchcorrection_result.csv"))
                if os.path.exists(batch_file):
                    try:
                        loso_results = parse_batchcorrection_result(batch_file)

                        for model_method, values in loso_results.items():
                            if model_method in data['loso']:
                                data['loso'][model_method].extend(values)
                            else:
                                # Handle base model (without batch correction suffix)
                                # If it's just the model name, it's the baseline
                                for base_model in ['ElasticNet', 'RandomForest', 'TabPFN']:
                                    if model_method == base_model:
                                        data['loso'][base_model].extend(values)
                    except Exception as e:
                        print(f"Error reading {batch_file}: {e}")

    return data

def perform_statistical_tests_loso(data):
    """
    Perform pairwise Wilcoxon tests with BH correction for LOSO panel

    Compare each batch correction method against baseline for each model
    Returns dict: {(model, method): p_value_adjusted}
    """
    results = {}
    p_values = []
    comparisons = []

    # For each model, compare batch correction methods against baseline
    for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
        baseline_data = data['loso'][model]

        if len(baseline_data) < 3:  # Need at least 3 samples for Wilcoxon
            continue

        for method in ['DebiasM', 'ComBat', 'MMUPHin']:
            method_key = f"{model}_{method}"
            method_data = data['loso'][method_key]

            if len(method_data) < 3:  # Need at least 3 samples
                continue

            # Ensure same length (paired test)
            min_len = min(len(baseline_data), len(method_data))
            if min_len < 3:
                continue

            baseline_paired = baseline_data[:min_len]
            method_paired = method_data[:min_len]

            try:
                # Perform Wilcoxon signed-rank test (paired)
                stat, p_value = wilcoxon(method_paired, baseline_paired, alternative='greater')
                p_values.append(p_value)
                comparisons.append((model, method))
            except Exception as e:
                print(f"Warning: Wilcoxon test failed for {model} vs {method}: {e}")
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

def perform_pairwise_model_tests(data, scenario):
    """
    Perform pairwise Wilcoxon tests between models for Intra or Cross scenarios

    Args:
        data: dict with model data for scenario
        scenario: 'intra' or 'cross'

    Returns:
        dict: {(model1, model2): {'p_value', 'p_adjusted', 'significant'}}
    """
    results = {}
    p_values = []
    comparisons = []

    models = ['ElasticNet', 'RandomForest', 'TabPFN']

    # Pairwise comparisons between all models
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]

            data1 = data[scenario][model1]
            data2 = data[scenario][model2]

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

def draw_significance_bracket(ax, x1, x2, y, p_adjusted, height=0.02):
    """
    Draw a bracket line between two positions with significance stars (below boxplot)

    Args:
        ax: matplotlib axis
        x1, x2: x positions for the two boxes
        y: y position for the bracket (bottom of the bracket)
        p_adjusted: adjusted p-value
        height: height of the bracket arms (extends upward from y)
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

    # Draw inverted bracket (U-shape below boxplot): left vertical up, horizontal, right vertical up
    ax.plot([x1, x1, x2, x2], [y+height, y, y, y+height],
            '-', linewidth=1.5, color='black', zorder=5)

    # Add significance marker below the bracket
    ax.text((x1 + x2) / 2, y - 0.015, marker,
            ha='center', va='top', fontsize=14, fontweight='bold', color='black', zorder=5)

def add_pairwise_significance_brackets(ax, positions, data_dict, model_order, stats_results):
    """
    Add significance brackets for pairwise model comparisons (below boxplots)

    Args:
        ax: matplotlib axis
        positions: list of x positions for each model
        data_dict: dict with data for each model
        model_order: list of model names in order
        stats_results: statistical test results from perform_pairwise_model_tests
    """
    if not stats_results:
        return

    # Get data for positioning brackets (find minimum values)
    min_values = []
    for model in model_order:
        if data_dict[model]:
            min_values.append(min(data_dict[model]))
        else:
            min_values.append(0)

    # Base y position for brackets (below lowest box)
    base_y = min(min_values) - 0.06

    # Draw brackets for significant comparisons
    # Use different y levels for different comparisons to avoid overlap
    # Lower level values mean brackets are positioned lower (further down)
    bracket_levels = {
        ('ElasticNet', 'RandomForest'): 0,
        ('ElasticNet', 'TabPFN'): -0.06,  # Further down
        ('RandomForest', 'TabPFN'): 0,
    }

    for (model1, model2), result in stats_results.items():
        if result['significant']:
            # Get positions
            idx1 = model_order.index(model1)
            idx2 = model_order.index(model2)

            x1 = positions[idx1]
            x2 = positions[idx2]

            # Determine bracket height level (negative means further down)
            level_offset = bracket_levels.get((model1, model2), 0)
            y = base_y + level_offset

            # Draw bracket
            draw_significance_bracket(ax, x1, x2, y, result['p_adjusted'])

def add_significance_brackets_loso(ax, positions, data_dict, model, methods, stats_results):
    """
    Add significance brackets in LOSO panel connecting batch correction methods to baseline

    Args:
        ax: matplotlib axis
        positions: list of x positions for this model's methods
        data_dict: dict with data for each method
        model: model name
        methods: list of (suffix, name) tuples
        stats_results: statistical test results
    """
    # Get baseline position (first position)
    baseline_key = model
    if baseline_key not in data_dict or not data_dict[baseline_key]:
        return

    baseline_pos = positions[0]  # First position is baseline

    # Get minimum value for positioning brackets below
    all_values = []
    for method_suffix, method_name in methods:
        key = model + method_suffix
        if key in data_dict and data_dict[key]:
            all_values.extend(data_dict[key])

    if not all_values:
        return

    base_y = min(all_values) - 0.06

    # Draw brackets from baseline to each significant batch correction method
    bracket_offset = 0  # Offset for stacking multiple brackets
    for idx, (method_suffix, method_name) in enumerate(methods):
        if method_name is None:  # Skip baseline itself
            continue

        key = model + method_suffix
        if key not in data_dict or not data_dict[key]:
            continue

        # Check if this comparison is significant
        if (model, method_name) in stats_results:
            result = stats_results[(model, method_name)]

            if result['significant']:
                # Get position of this method
                method_pos = positions[idx]

                # Calculate y position with stacking offset
                y = base_y + bracket_offset
                bracket_offset -= 0.05  # Stack next bracket lower

                # Draw bracket from baseline to this method
                draw_significance_bracket(ax, baseline_pos, method_pos, y, result['p_adjusted'])

def create_boxplot_for_disease_type(disease_type, data, output_path, preprocessing='CLR'):
    """
    Create boxplot for a specific disease type

    Three panels: Intra, Cross, LOSO
    - Intra: 3 boxes (ElasticNet, RandomForest, TabPFN)
    - Cross: 3 boxes (ElasticNet, RandomForest, TabPFN)
    - LOSO: 12 boxes (3 models × 4 methods: baseline + 3 batch corrections)
    """
    # Adjust subplot widths: intra and cross wider, loso more compact
    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 0.9, 2.2])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Color scheme for models (matching Figure 2)
    model_colors = {
        'ElasticNet': '#C5B0D5',   # Light purple/lavender
        'RandomForest': '#AEC7E8',  # Light blue
        'TabPFN': '#FFBB78'         # Light orange
    }

    # Model abbreviations
    model_abbrev = {
        'ElasticNet': 'Enet',
        'RandomForest': 'RF',
        'TabPFN': 'Tabpfn'
    }

    # Perform statistical tests
    print(f"\nPerforming statistical tests for {disease_type}...")

    # Intra and Cross: pairwise model comparisons
    stats_intra = perform_pairwise_model_tests(data, 'intra')
    stats_cross = perform_pairwise_model_tests(data, 'cross')

    # LOSO: batch correction vs baseline
    stats_loso = perform_statistical_tests_loso(data)

    # Print significant results
    print(f"Intra-cohort significant comparisons (BH-corrected p < 0.05):")
    if stats_intra:
        for (model1, model2), result in stats_intra.items():
            if result['significant']:
                print(f"  {model1} vs {model2}: p={result['p_value']:.4f}, p_adj={result['p_adjusted']:.4f} {'***' if result['p_adjusted']<0.001 else '**' if result['p_adjusted']<0.01 else '*'}")
    else:
        print("  None")

    print(f"Cross-cohort significant comparisons (BH-corrected p < 0.05):")
    if stats_cross:
        for (model1, model2), result in stats_cross.items():
            if result['significant']:
                print(f"  {model1} vs {model2}: p={result['p_value']:.4f}, p_adj={result['p_adjusted']:.4f} {'***' if result['p_adjusted']<0.001 else '**' if result['p_adjusted']<0.01 else '*'}")
    else:
        print("  None")

    print(f"LOSO batch correction significant comparisons (BH-corrected p < 0.05):")
    if stats_loso:
        for (model, method), result in stats_loso.items():
            if result['significant']:
                print(f"  {model} + {method}: p={result['p_value']:.4f}, p_adj={result['p_adjusted']:.4f} {'***' if result['p_adjusted']<0.001 else '**' if result['p_adjusted']<0.01 else '*'}")
    else:
        print("  None")

    # Panel 1: Intra
    ax_intra = axes[0]
    intra_data = []
    intra_positions = []
    intra_colors = []
    intra_labels = []

    pos = 1
    for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
        if data['intra'][model]:
            intra_data.append(data['intra'][model])
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
    ax_intra.set_ylim(0.05, 1.0)  # Extended lower limit for brackets
    ax_intra.set_xticks(intra_positions)
    ax_intra.set_xticklabels(intra_labels, rotation=0, ha='center', fontsize=16)
    ax_intra.grid(axis='y', alpha=0.3, linestyle='--')

    # Add significance brackets for Intra panel
    if stats_intra:
        model_order = [m for m in ['ElasticNet', 'RandomForest', 'TabPFN'] if data['intra'][m]]
        add_pairwise_significance_brackets(ax_intra, intra_positions, data['intra'],
                                          model_order, stats_intra)

    # Add title with white box (like figure2)
    rect = patches.Rectangle(
        (0, 1.02),  # (x, y) position in axes coordinates
        1.0,  # width = full axes width
        0.08,  # height
        linewidth=1.5,
        edgecolor='black',
        facecolor='white',
        clip_on=False,
        zorder=10,
        transform=ax_intra.transAxes
    )
    ax_intra.add_patch(rect)
    ax_intra.text(0.5, 1.06, 'Intra-cohort',
                 ha='center', va='center', fontsize=18, fontweight='bold',
                 fontfamily='sans-serif', zorder=11, transform=ax_intra.transAxes)

    # Panel 2: Cross
    ax_cross = axes[1]
    cross_data = []
    cross_positions = []
    cross_colors = []
    cross_labels = []

    pos = 1
    for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
        if data['cross'][model]:
            cross_data.append(data['cross'][model])
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
    ax_cross.set_ylim(0.05, 1.0)  # Extended lower limit for brackets
    ax_cross.set_xticks(cross_positions)
    ax_cross.set_xticklabels(cross_labels, rotation=0, ha='center', fontsize=16)
    ax_cross.grid(axis='y', alpha=0.3, linestyle='--')

    # Add significance brackets for Cross panel
    if stats_cross:
        model_order = [m for m in ['ElasticNet', 'RandomForest', 'TabPFN'] if data['cross'][m]]
        add_pairwise_significance_brackets(ax_cross, cross_positions, data['cross'],
                                          model_order, stats_cross)

    # Add title with white box (like figure2)
    rect = patches.Rectangle(
        (0, 1.02),  # (x, y) position in axes coordinates
        1.0,  # width = full axes width
        0.08,  # height
        linewidth=1.5,
        edgecolor='black',
        facecolor='white',
        clip_on=False,
        zorder=10,
        transform=ax_cross.transAxes
    )
    ax_cross.add_patch(rect)
    ax_cross.text(0.5, 1.06, 'Cross-cohort',
                 ha='center', va='center', fontsize=18, fontweight='bold',
                 fontfamily='sans-serif', zorder=11, transform=ax_cross.transAxes)

    # Panel 3: LOSO
    ax_loso = axes[2]
    loso_data = []
    loso_positions = []
    loso_colors = []
    loso_labels = []

    pos = 1
    # Include baseline and batch correction methods
    # Group by method first, then by model (matching figure3)
    methods = [
        ('', None),  # baseline - only show model name
        ('_DebiasM', 'DebiasM'),
        ('_ComBat', 'ComBat'),
        ('_MMUPHin', 'MMUPHin')
    ]

    # Changed order: iterate by method first, then by model (matching figure3)
    for method_suffix, method_name in methods:
        for model in ['ElasticNet', 'RandomForest', 'TabPFN']:
            key = model + method_suffix
            if data['loso'][key]:
                loso_data.append(data['loso'][key])
                loso_positions.append(pos)
                loso_colors.append(model_colors[model])

                # Create label: baseline shows only model name, others show "model\nmethod"
                if method_name is None:
                    # Baseline: only model name
                    loso_labels.append(model_abbrev[model])
                else:
                    # Batch correction: model + method
                    loso_labels.append(f"{model_abbrev[model]}\n{method_name}")
                # Spacing within method group
                pos += 3.5

        # Spacing between method groups
        pos += 0.3

    if loso_data:
        bp = ax_loso.boxplot(loso_data, positions=loso_positions, widths=3.0,
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
    ax_loso.set_ylim(0.05, 1.0)  # Extended lower limit for consistency
    ax_loso.set_xticks(loso_positions)
    ax_loso.set_xticklabels(loso_labels, rotation=0, ha='center', fontsize=10)
    ax_loso.grid(axis='y', alpha=0.3, linestyle='--')

    # Add significance brackets for LOSO panel (matching figure3 logic)
    if stats_loso:
        # Track positions for each method group (not model group)
        # Since we're not doing pairwise model comparisons within method groups in figureS2,
        # we skip the bracket drawing for now
        # (figureS2 only compares methods vs baseline, not models vs models)
        pass

    # Add title with white box (like figure2)
    rect = patches.Rectangle(
        (0, 1.02),  # (x, y) position in axes coordinates
        1.0,  # width = full axes width
        0.08,  # height
        linewidth=1.5,
        edgecolor='black',
        facecolor='white',
        clip_on=False,
        zorder=10,
        transform=ax_loso.transAxes
    )
    ax_loso.add_patch(rect)
    ax_loso.text(0.5, 1.06, 'Leave-One-Study-Out',
                 ha='center', va='center', fontsize=18, fontweight='bold',
                 fontfamily='sans-serif', zorder=11, transform=ax_loso.transAxes)

    plt.tight_layout()

    # Main title centered at top
    # x=0.5 centers the title horizontally
    # y=0.98 positions it at top, slightly above panel white boxes
    fig.text(0.5, 0.98, f'{disease_type}',
             fontsize=22, fontweight='bold',
             ha='center', va='bottom',
             transform=fig.transFigure)

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(f"Saved boxplot for {disease_type}: {output_path}")

def main():
    """Main function to generate all plots"""
    # Set up paths
    results_dir = '/ua/jmu27/Micro_bench/results'
    output_dir = '/ua/jmu27/Micro_bench/figures/figureS2'

    # Set preprocessing method (log-std for log-standardized)
    preprocessing = 'log_std'  # Can be changed to 'ALR' or 'CLR' if needed

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get unique disease types
    disease_types = sorted(list(set(disease_type_map.values())))

    print(f"Generating boxplots for {len(disease_types)} disease types with {preprocessing} preprocessing...")

    # Generate plot for each disease type
    for disease_type in disease_types:
        print(f"\nProcessing {disease_type}...")

        # Collect data
        data = collect_data_for_disease_type(results_dir, disease_type, preprocessing=preprocessing)

        # Check if we have any data
        has_data = False
        for scenario in ['intra', 'cross']:
            for model in data[scenario]:
                if data[scenario][model]:
                    has_data = True
                    break

        for model_method in data['loso']:
            if data['loso'][model_method]:
                has_data = True
                break

        if has_data:
            # Create output path
            output_path = os.path.join(output_dir, f'{disease_type}_batchcorrection_boxplot_{preprocessing}.png')

            # Create boxplot
            create_boxplot_for_disease_type(disease_type, data, output_path, preprocessing=preprocessing)
        else:
            print(f"No data found for {disease_type}, skipping...")

    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()
