#!/usr/bin/env python3
"""Generate Figure S1: Genus vs Species comparison for Metagenomics data
Boxplots showing preprocessing method performance across taxonomic levels
"""

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
taxonomic_levels = ['genus', 'species']

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
    """Collect all data for boxplots - only Metagenomics data, separated by taxonomic level"""
    # Structure: {scenario: {prep_method: {tax_level: {model: [values]}}}}
    data_collection = {}

    for scenario in ['Intra', 'Cross', 'LOSO']:
        data_collection[scenario] = {}
        for prep in preprocessing_methods:
            data_collection[scenario][prep] = {}
            for tax_level in taxonomic_levels:
                data_collection[scenario][prep][tax_level] = {model: [] for model in models}

    # Iterate through all files
    for filename in os.listdir(results_dir):
        # Skip batch correction and debias files
        if 'batchcorrection' in filename or 'debias' in filename.lower():
            continue

        # Only process Metagenomics files
        if 'Metagenomics' not in filename:
            continue

        if filename.endswith('_result.csv'):
            parts = filename.replace('_result.csv', '').split('_')
            if len(parts) >= 4:
                # Parse filename: {disease}_Metagenomics_{taxlevel}_{preprocess}_result.csv
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
                if (file_tax in taxonomic_levels and
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

                                # Add values to collection (keep genus and species separate)
                                for scenario in ['Intra', 'Cross', 'LOSO']:
                                    if not pd.isna(metrics[scenario]):
                                        data_collection[scenario][prep_display][file_tax][model].append(metrics[scenario])
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    return data_collection

def create_boxplot_figure(data_collection, output_path):
    """Create 3-panel boxplot figure (3 rows × 1 column)
    Each preprocessing method has 6 boxes: 2 taxonomic levels × 3 models
    """

    # Create 3x1 subplot layout
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    # Color scheme for models (same as figure2)
    model_colors = {
        'ElasticNet': '#C5B0D5',   # Light purple/lavender
        'RandomForest': '#AEC7E8',  # Light blue
        'TabPFN': '#FFBB78'         # Light orange
    }

    # Pattern or alpha to distinguish taxonomic levels
    # Genus: solid (alpha=0.7), Species: lighter (alpha=0.4)
    tax_alpha = {
        'genus': 0.7,
        'species': 0.4
    }

    # Scenarios
    scenarios = ['Intra', 'Cross', 'LOSO']

    for scenario_idx, scenario in enumerate(scenarios):
        ax = axes[scenario_idx]

        # Prepare boxplot data
        positions = []
        box_data = []
        box_colors = []
        box_alphas = []

        pos = 1
        for prep_idx, prep in enumerate(preprocessing_methods):
            # For each preprocessing method, add 6 boxes (3 models × genus/species)
            # Group by model: ElasticNet(genus, species), RF(genus, species), TabPFN(genus, species)
            for model_idx, model in enumerate(models):
                for tax_level in taxonomic_levels:
                    values = data_collection[scenario][prep][tax_level][model]
                    if values:  # Only add if there's data
                        box_data.append(values)
                        positions.append(pos)
                        box_colors.append(model_colors[model])
                        box_alphas.append(tax_alpha[tax_level])
                        pos += 1

                # Add small spacing between different models (within same preprocessing)
                if model_idx < len(models) - 1:
                    pos += 0.5

            # Add spacing between preprocessing methods
            pos += 2

        # Draw boxplots
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.9,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(linewidth=1.2),
                           medianprops=dict(linewidth=2, color='black'),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2),
                           flierprops=dict(marker='o', markersize=3, alpha=0.5))

            # Set box colors and alphas
            for patch, color, alpha in zip(bp['boxes'], box_colors, box_alphas):
                patch.set_facecolor(color)
                patch.set_alpha(alpha)

        # Set y-axis label on all panels (vertical layout)
        ax.set_ylabel('AUC', fontsize=18, fontfamily='sans-serif')

        # Set y-axis limits
        ax.set_ylim(0.2, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

        # Set x-axis ticks and labels
        prep_positions = []
        pos = 1
        for prep_idx, prep in enumerate(preprocessing_methods):
            # Calculate center position for each preprocessing method
            # 6 boxes per preprocessing (2 tax_levels × 3 models) + spacing between models
            n_boxes = len(taxonomic_levels) * len(models)
            n_model_gaps = len(models) - 1  # Number of gaps between models
            total_width = n_boxes - 1 + n_model_gaps * 0.5
            center_pos = pos + total_width / 2
            prep_positions.append(center_pos)
            pos += n_boxes + n_model_gaps * 0.5 + 2  # Match the spacing used above

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

        # Add legend only to topmost panel (Intra)
        if scenario_idx == 0:
            from matplotlib.patches import Patch

            # Create first legend for models
            model_legend_elements = []
            model_legend_elements.append(
                Patch(facecolor=model_colors['ElasticNet'], alpha=0.7,
                     edgecolor='black', linewidth=1.2, label='ElasticNet')
            )
            model_legend_elements.append(
                Patch(facecolor=model_colors['RandomForest'], alpha=0.7,
                     edgecolor='black', linewidth=1.2, label='RandomForest')
            )
            model_legend_elements.append(
                Patch(facecolor=model_colors['TabPFN'], alpha=0.7,
                     edgecolor='black', linewidth=1.2, label='TabPFN')
            )

            # Create second legend for taxonomic levels
            tax_legend_elements = []
            tax_legend_elements.append(
                Patch(facecolor='gray', alpha=0.7,
                     edgecolor='black', linewidth=1.2, label='Genus')
            )
            tax_legend_elements.append(
                Patch(facecolor='gray', alpha=0.4,
                     edgecolor='black', linewidth=1.2, label='Species')
            )

            # Add first legend (models) - left side
            legend1 = ax.legend(handles=model_legend_elements, loc='lower left',
                               fontsize=14, frameon=True, fancybox=False,
                               edgecolor='black', framealpha=1, ncol=1,
                               bbox_to_anchor=(0, 0))

            # Add second legend (taxonomic levels) - next to first legend
            legend2 = ax.legend(handles=tax_legend_elements, loc='lower left',
                               fontsize=14, frameon=True, fancybox=False,
                               edgecolor='black', framealpha=1, ncol=1,
                               bbox_to_anchor=(0.25, 0))

            # Add first legend back (because the second legend call replaces it)
            ax.add_artist(legend1)

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
    OUTPUT_PATH = "/ua/jmu27/Micro_bench/figures/figureS1/figureS1_genus_vs_species.png"

    print("=" * 80)
    print("Generating Figure S1: Genus vs Species Comparison (Metagenomics only)")
    print("=" * 80)

    # Collect data
    print("\nCollecting data from results (Metagenomics only)...")
    data_collection = collect_boxplot_data(RESULTS_DIR)

    # Print summary
    print("\nData summary (genus and species separate):")
    for scenario in ['Intra', 'Cross', 'LOSO']:
        print(f"\n{scenario}:")
        for prep in preprocessing_methods:
            n_genus = len(data_collection[scenario][prep]['genus']['ElasticNet'])
            n_species = len(data_collection[scenario][prep]['species']['ElasticNet'])
            if n_genus > 0 or n_species > 0:
                print(f"  {prep}: genus={n_genus} datasets, species={n_species} datasets")

    # Create boxplot figure
    print("\nCreating boxplot figure...")
    create_boxplot_figure(data_collection, OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("Figure S1 generated successfully!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Output: {OUTPUT_PATH.replace('.png', '.pdf')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
