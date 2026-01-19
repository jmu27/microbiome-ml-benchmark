#!/usr/bin/env python3
"""Generate Figure 2 heatmap: 3 square heatmaps (one per model) comparing transformations"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# Font setup
def check_helvetica_font():
    """Check if Helvetica or similar font is available"""
    fm._load_fontmanager(try_read_cache=False)
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_preferences = ['Helvetica', 'Helvetica Neue', 'HelveticaNeue', 'Liberation Sans', 'Arial']

    for font in font_preferences:
        if font in available_fonts:
            print(f"✓ Using font: {font}")
            return font

    print("⚠ No preferred font available, using default sans-serif")
    return 'DejaVu Sans'

best_font = check_helvetica_font()

# Set matplotlib parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [best_font, 'Liberation Sans', 'Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Configuration
preprocessing_methods = ['RA', 'Binary', 'Log-Std', 'ALR', 'CLR', 'GPTemb']
preprocessing_files = ['none', 'binary', 'log_std', 'ALR', 'CLR', 'gptemb']
models = ['ElasticNet', 'RandomForest', 'TabPFN']

def calculate_metrics(df, model_name):
    """Calculate intra, cross, LOSO mean values for a specific model"""
    metrics = {}

    # Filter for this specific model
    model_df = df[df['model'] == model_name].copy()

    # Get study columns (all columns except 'model')
    study_cols = [col for col in model_df.columns if col != 'model']

    # Convert to matrix format
    matrix_df = model_df[study_cols]

    # Intra: diagonal elements
    n_studies = len(study_cols)
    intra_values = []
    for i in range(min(n_studies, len(matrix_df))):
        col_name = study_cols[i]
        val = matrix_df.iloc[i][col_name]
        if not pd.isna(val):
            intra_values.append(val)
    metrics['Intra'] = np.mean(intra_values) if intra_values else np.nan

    # Cross: off-diagonal elements
    cross_values = []
    for i in range(min(n_studies, len(matrix_df))):
        for j, col_name in enumerate(study_cols):
            if i != j:
                val = matrix_df.iloc[i][col_name]
                if not pd.isna(val):
                    cross_values.append(val)
    metrics['Cross'] = np.mean(cross_values) if cross_values else np.nan

    # LOSO: last row (lodo row)
    if len(matrix_df) > n_studies:
        loso_row = matrix_df.iloc[-1]
        loso_values = []
        for col_name in study_cols:
            val = loso_row[col_name]
            if not pd.isna(val):
                loso_values.append(val)
        metrics['LOSO'] = np.mean(loso_values) if loso_values else np.nan
    else:
        metrics['LOSO'] = np.nan

    return metrics

def collect_all_data(results_dir):
    """Collect all data for each preprocessing method and model (aggregated across scenarios)"""
    # Structure: {model: {prep_method: [all_values]}}
    data_collection = {}

    for model in models:
        data_collection[model] = {}
        for prep in preprocessing_methods:
            data_collection[model][prep] = []

    # Iterate through all files
    for filename in os.listdir(results_dir):
        # Skip batch correction and debias files
        if 'batchcorrection' in filename or 'debias' in filename.lower():
            continue

        if filename.endswith('_result.csv'):
            parts = filename.replace('_result.csv', '').split('_')
            if len(parts) >= 4:
                # Parse filename
                file_tax = parts[-2] if parts[-1] in preprocessing_files else parts[-3]

                # Get preprocessing method
                if parts[-1] == 'std' and parts[-2] == 'log':
                    file_prep = 'log_std'
                    file_tax = parts[-3]
                else:
                    file_prep = parts[-1]
                    file_tax = parts[-2]

                # Check if this is a valid combination
                if (file_tax in ['genus', 'species'] and
                    file_prep in preprocessing_files):

                    # Find display name for preprocessing method
                    prep_display = preprocessing_methods[preprocessing_files.index(file_prep)]

                    file_path = os.path.join(results_dir, filename)
                    try:
                        df = pd.read_csv(file_path)

                        # Process each model
                        for model in models:
                            if model in df['model'].values:
                                metrics = calculate_metrics(df, model)

                                # Add ALL values (intra, cross, loso) to the collection
                                for scenario in ['Intra', 'Cross', 'LOSO']:
                                    if not pd.isna(metrics[scenario]):
                                        data_collection[model][prep_display].append(metrics[scenario])
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    return data_collection

def perform_pairwise_comparisons(data_collection, model):
    """Perform pairwise statistical comparisons between preprocessing methods for a model"""
    n_methods = len(preprocessing_methods)

    # Create matrices for mean difference and p-values
    mean_diff_matrix = np.zeros((n_methods, n_methods))
    pval_matrix = np.full((n_methods, n_methods), np.nan)

    # Store all comparisons for FDR correction
    all_pvals = []
    pval_positions = []

    for i, prep1 in enumerate(preprocessing_methods):
        for j, prep2 in enumerate(preprocessing_methods):
            if i == j:
                # Diagonal: no comparison needed
                mean_diff_matrix[i, j] = 0.0
                pval_matrix[i, j] = np.nan
            else:
                values1 = data_collection[model][prep1]
                values2 = data_collection[model][prep2]

                if len(values1) > 0 and len(values2) > 0:
                    # Calculate mean difference (prep1 - prep2)
                    mean_diff = np.mean(values1) - np.mean(values2)
                    mean_diff_matrix[i, j] = mean_diff

                    # Perform Wilcoxon signed-rank test (paired)
                    if len(values1) == len(values2):
                        try:
                            _, pval = stats.wilcoxon(values1, values2, alternative='two-sided')
                            pval_matrix[i, j] = pval
                            all_pvals.append(pval)
                            pval_positions.append((i, j))
                        except:
                            pval_matrix[i, j] = np.nan
                    else:
                        # Use Mann-Whitney U test for unpaired data
                        try:
                            _, pval = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                            pval_matrix[i, j] = pval
                            all_pvals.append(pval)
                            pval_positions.append((i, j))
                        except:
                            pval_matrix[i, j] = np.nan
                else:
                    mean_diff_matrix[i, j] = np.nan
                    pval_matrix[i, j] = np.nan

    # Apply FDR correction
    fdr_matrix = np.full((n_methods, n_methods), np.nan)
    if len(all_pvals) > 0:
        _, fdr_corrected, _, _ = multipletests(all_pvals, method='fdr_bh')
        for idx, (i, j) in enumerate(pval_positions):
            fdr_matrix[i, j] = fdr_corrected[idx]

    return mean_diff_matrix, pval_matrix, fdr_matrix

def create_three_heatmap_figure(all_results_dict, output_path):
    """Create 3 square heatmaps (one per model) showing transformation1 vs transformation2"""

    # Create figure with 3 subplots (1 row, 3 columns)
    # Increase figure size to accommodate larger text
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Create custom divergent colormap: blue to white to red
    macaron_divergent = [
        "#B8E5FA",  # Light blue - negative/low value
        "#D9F2FD",  # Lighter blue
        "#FFFFFF",  # White - center/zero
        "#FBD3D6",  # Light pink
        "#F7A6AC"   # Red/pink - positive/high value
    ]
    custom_cmap = LinearSegmentedColormap.from_list('macaron_divergent', macaron_divergent)

    # Find global vmin/vmax for consistent color scaling
    all_mean_diffs = []
    for model in models:
        mean_diff_matrix, _, _ = all_results_dict[model]
        all_mean_diffs.extend(mean_diff_matrix[~np.isnan(mean_diff_matrix)].flatten())

    vmax = np.max(np.abs(all_mean_diffs)) if len(all_mean_diffs) > 0 else 0.1
    vmin = -vmax

    # Create heatmap for each model
    for idx, model in enumerate(models):
        ax = axes[idx]

        mean_diff_matrix, pval_matrix, fdr_matrix = all_results_dict[model]

        # Create annotation matrix with significance stars
        annot_matrix = []
        for i in range(len(preprocessing_methods)):
            row_annots = []
            for j in range(len(preprocessing_methods)):
                if i == j:
                    row_annots.append('')
                else:
                    mean_diff = mean_diff_matrix[i, j]
                    fdr = fdr_matrix[i, j]
                    pval = pval_matrix[i, j]

                    if np.isnan(mean_diff):
                        row_annots.append('')
                    else:
                        # Determine significance
                        if not np.isnan(fdr) and fdr <= 0.05:
                            sig = '**'
                        elif not np.isnan(pval) and pval <= 0.05:
                            sig = '*'
                        else:
                            sig = ''

                        row_annots.append(f'{mean_diff:.3f}{sig}')
            annot_matrix.append(row_annots)

        # Create DataFrame
        df = pd.DataFrame(mean_diff_matrix,
                         index=preprocessing_methods,
                         columns=preprocessing_methods)

        # Plot heatmap (show full matrix, no mask)
        sns.heatmap(df,
                   annot=annot_matrix,
                   fmt='',
                   cmap=custom_cmap,
                   center=0,
                   vmin=vmin,
                   vmax=vmax,
                   cbar=False,  # No colorbar
                   linewidths=0.5,
                   linecolor='lightgray',
                   ax=ax,
                   annot_kws={'fontsize': 20, 'fontfamily': 'sans-serif', 'color': 'black'})

        # Set labels with larger font sizes
        ax.set_title(model, fontsize=24, fontweight='bold', pad=20)
        ax.set_xlabel('Transformation 2', fontsize=20)

        # Only show y-axis label on the first (leftmost) subplot
        if idx == 0:
            ax.set_ylabel('Transformation 1', fontsize=20)
            ax.set_yticklabels(preprocessing_methods, rotation=0, fontsize=18)
        else:
            ax.set_ylabel('', fontsize=20)
            ax.set_yticklabels([])  # Hide y-axis labels for 2nd and 3rd plots

        # Set x-axis labels horizontal (no rotation)
        ax.set_xticklabels(preprocessing_methods, rotation=0, ha='center', fontsize=18)

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved combined heatmap: {output_path}")
    print(f"Saved combined heatmap: {output_path.replace('.png', '.pdf')}")

def main():
    """Main function to generate combined heatmap"""

    RESULTS_DIR = "/ua/jmu27/Micro_bench/results"
    OUTPUT_DIR = "/ua/jmu27/Micro_bench/figures/figure2"

    print("=" * 80)
    print("Generating Figure 2: 3 Heatmaps (One per Model)")
    print("=" * 80)

    # Collect all data (aggregated across all scenarios)
    print("\nCollecting data from results...")
    data_collection = collect_all_data(RESULTS_DIR)

    # Print data summary
    print("\nData summary (aggregated across Intra/Cross/LOSO):")
    for model in models:
        print(f"\n{model}:")
        for prep in preprocessing_methods:
            n_values = len(data_collection[model][prep])
            if n_values > 0:
                mean_val = np.mean(data_collection[model][prep])
                print(f"  {prep}: {n_values} values, mean={mean_val:.4f}")

    # Perform pairwise comparisons for each model
    all_results_dict = {}

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"Processing {model}...")
        print('=' * 80)

        mean_diff_matrix, pval_matrix, fdr_matrix = perform_pairwise_comparisons(data_collection, model)
        all_results_dict[model] = (mean_diff_matrix, pval_matrix, fdr_matrix)

        # Save matrices as CSV
        mean_diff_df = pd.DataFrame(mean_diff_matrix,
                                    index=preprocessing_methods,
                                    columns=preprocessing_methods)
        pval_df = pd.DataFrame(pval_matrix,
                              index=preprocessing_methods,
                              columns=preprocessing_methods)
        fdr_df = pd.DataFrame(fdr_matrix,
                             index=preprocessing_methods,
                             columns=preprocessing_methods)

        mean_diff_df.to_csv(f"{OUTPUT_DIR}/{model}_mean_diff_matrix.csv")
        pval_df.to_csv(f"{OUTPUT_DIR}/{model}_pval_matrix.csv")
        fdr_df.to_csv(f"{OUTPUT_DIR}/{model}_fdr_matrix.csv")

        print(f"Saved matrices for {model}")

    # Create combined heatmap figure
    print(f"\n{'=' * 80}")
    print("Creating combined heatmap figure...")
    print('=' * 80)
    output_path = f"{OUTPUT_DIR}/figure2_heatmap_combined.png"
    create_three_heatmap_figure(all_results_dict, output_path)

    print("\n" + "=" * 80)
    print("Combined heatmap generated successfully!")
    print(f"Output: {output_path}")
    print(f"Output: {output_path.replace('.png', '.pdf')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
