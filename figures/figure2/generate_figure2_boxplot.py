#!/usr/bin/env python3
"""Generate Figure 2 as boxplots showing preprocessing method performance"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# Font setup
def check_helvetica_font():
    """Check if Helvetica or similar font is available"""
    # Reload font manager to pick up newly installed fonts
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
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Configuration
preprocessing_methods = ['RA', 'Binary', 'Log-Std', 'ALR', 'CLR', 'GPTemb']
preprocessing_files = ['none', 'binary', 'log_std', 'ALR', 'CLR', 'gptemb']
models = ['ElasticNet', 'RandomForest', 'TabPFN']

def calculate_metrics(df, model_name):
    """Calculate intra, cross, LOSO mean values for a specific model

    df format: columns are study IDs (1, 2, 3, ...) and 'model'
               rows are train-test combinations with last row being 'lodo'
    """
    metrics = {}

    # Filter for this specific model
    model_df = df[df['model'] == model_name].copy()

    # Get study columns (all columns except 'model')
    study_cols = [col for col in model_df.columns if col != 'model']

    # Convert to matrix format
    matrix_df = model_df[study_cols]

    # Intra: diagonal elements (first n rows where n = number of studies)
    n_studies = len(study_cols)
    intra_values = []
    for i in range(min(n_studies, len(matrix_df))):
        col_name = study_cols[i]
        val = matrix_df.iloc[i][col_name]
        if not pd.isna(val):
            intra_values.append(val)
    metrics['Intra'] = np.mean(intra_values) if intra_values else np.nan

    # Cross: off-diagonal elements (first n rows)
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

def collect_boxplot_data(results_dir):
    """Collect all data for boxplots"""
    # Structure: {scenario: {prep_method: {model: [values]}}}
    # Combine genus and species data together
    data_collection = {}

    for scenario in ['Intra', 'Cross', 'LOSO']:
        data_collection[scenario] = {}
        for prep in preprocessing_methods:
            data_collection[scenario][prep] = {model: [] for model in models}

    # Iterate through all files
    for filename in os.listdir(results_dir):
        # Skip batch correction and debias files
        if 'batchcorrection' in filename or 'debias' in filename.lower():
            continue

        if filename.endswith('_result.csv'):
            parts = filename.replace('_result.csv', '').split('_')
            if len(parts) >= 4:
                # Parse filename: {disease}_{dtype}_{taxlevel}_{preprocess}_result.csv
                # Handle multi-part preprocessing method names like 'log_std'
                file_tax = parts[-2] if parts[-1] in preprocessing_files else parts[-3]

                # Get preprocessing method (might be multi-part like log_std)
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

                                # Add values to collection (combine genus and species)
                                for scenario in ['Intra', 'Cross', 'LOSO']:
                                    if not pd.isna(metrics[scenario]):
                                        data_collection[scenario][prep_display][model].append(metrics[scenario])
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    return data_collection

def create_boxplot_figure(data_collection, output_path):
    """Create 3-panel boxplot figure (1 row × 3 columns)"""

    # Create 1x3 subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Color scheme for models
    colors = {
        'ElasticNet': '#C5B0D5',   # Light purple/lavender
        'RandomForest': '#AEC7E8',  # Light blue
        'TabPFN': '#FFBB78'         # Light orange
    }

    # Scenarios
    scenarios = ['Intra', 'Cross', 'LOSO']

    for scenario_idx, scenario in enumerate(scenarios):
        ax = axes[scenario_idx]

        # Prepare boxplot data
        positions = []
        box_data = []
        box_colors = []

        pos = 1
        for prep_idx, prep in enumerate(preprocessing_methods):
            for model_idx, model in enumerate(models):
                values = data_collection[scenario][prep][model]
                if values:  # Only add if there's data
                    box_data.append(values)
                    positions.append(pos)
                    box_colors.append(colors[model])
                    pos += 1

            # Add spacing between preprocessing methods (not between models)
            pos += 1.5

        # Draw boxplots
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.9,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(linewidth=1.2),
                           medianprops=dict(linewidth=2, color='black'),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2),
                           flierprops=dict(marker='o', markersize=3, alpha=0.5))

            # Set box colors
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        # Set y-axis label only on leftmost panel
        if scenario_idx == 0:
            ax.set_ylabel('AUC', fontsize=18, fontfamily='sans-serif')
        else:
            ax.set_ylabel('', fontfamily='sans-serif')

        # Set y-axis limits
        ax.set_ylim(0.2, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

        # Set x-axis ticks and labels
        prep_positions = []
        pos = 1
        for prep_idx, prep in enumerate(preprocessing_methods):
            # Calculate center position for each preprocessing method
            n_models = len(models)
            center_pos = pos + (n_models - 1) / 2
            prep_positions.append(center_pos)
            pos += n_models + 1.5  # Match the spacing used above

        ax.set_xticks(prep_positions)
        ax.set_xticklabels(preprocessing_methods, fontsize=18, fontfamily='sans-serif', rotation=0)

        # Set tight x-axis limits to remove blank space
        if positions:
            ax.set_xlim(0.5, max(positions) + 0.5)

        # Set title with white box (AFTER setting xlim)
        title = f'{scenario}'

        # Draw white background rectangle using axes coordinates
        rect = patches.Rectangle(
            (0, 1.02),  # (x, y) position in axes coordinates
            1.0,  # width = full axes width
            0.08,  # height (increased for larger box)
            linewidth=1.5,
            edgecolor='black',
            facecolor='white',
            clip_on=False,
            zorder=10,
            transform=ax.transAxes
        )
        ax.add_patch(rect)

        # Add text in white box
        ax.text(0.5, 1.06, title,
               ha='center', va='center', fontsize=18, fontweight='bold',
               fontfamily='sans-serif', zorder=11, transform=ax.transAxes)

        # Add legend only to leftmost panel
        if scenario_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[model], alpha=0.7,
                                    edgecolor='black', linewidth=1.2, label=model)
                             for model in models]
            ax.legend(handles=legend_elements, loc='lower left',
                     fontsize=14, frameon=True, fancybox=False,
                     edgecolor='black', framealpha=1)

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved boxplot figure: {output_path}")
    print(f"Saved boxplot figure: {output_path.replace('.png', '.pdf')}")

def main():
    """Main function"""

    RESULTS_DIR = "/ua/jmu27/Micro_bench/results"
    OUTPUT_PATH = "/ua/jmu27/Micro_bench/figures/figure2/figure2_boxplot.png"

    print("=" * 80)
    print("Generating Figure 2 Boxplot")
    print("=" * 80)

    # Collect data
    print("\nCollecting data from results...")
    data_collection = collect_boxplot_data(RESULTS_DIR)

    # Print summary
    print("\nData summary (genus + species combined):")
    for scenario in ['Intra', 'Cross', 'LOSO']:
        print(f"\n{scenario}:")
        for prep in preprocessing_methods:
            n_datasets = len(data_collection[scenario][prep]['ElasticNet'])
            if n_datasets > 0:
                print(f"  {prep}: {n_datasets} datasets")

    # Create boxplot figure
    print("\nCreating boxplot figure...")
    create_boxplot_figure(data_collection, OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("Figure 2 boxplot generated successfully!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Output: {OUTPUT_PATH.replace('.png', '.pdf')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
