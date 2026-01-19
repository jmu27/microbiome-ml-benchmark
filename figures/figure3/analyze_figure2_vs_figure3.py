#!/usr/bin/env python3
"""Compare data collection between figure2 and figure3"""

import os
import pandas as pd
import numpy as np

disease_type_map = {
    'CRC': 'Intestinal', 'T2D': 'Metabolic', 'Obesity': 'Metabolic', 'Overweight': 'Metabolic',
    'Adenoma': 'Intestinal', 'CDI': 'Intestinal', 'AD': 'Mental', 'MCI': 'Mental', 'PD': 'Mental',
    'RA': 'Autoimmun', 'MS': 'Autoimmun', 'ASD': 'Mental', 'CD': 'Intestinal', 'UC': 'Intestinal',
    'IBD': 'Intestinal', 'AS': 'Autoimmun', 'IBS': 'Intestinal', 'CFS': 'Mental',
    'JA': 'Autoimmun', 'NAFLD': 'Liver'
}

def calculate_metrics_figure2_style(df, model_name):
    """Calculate metrics like figure2 does"""
    model_df = df[df['model'] == model_name].copy()
    
    if len(model_df) == 0:
        return {'Intra': np.nan, 'Cross': np.nan, 'LOSO': np.nan}
    
    study_cols = [col for col in model_df.columns if col != 'model']
    matrix_df = model_df[study_cols]
    
    n_studies = len(study_cols)
    
    # Intra: diagonal elements
    intra_values = []
    for i in range(min(n_studies, len(matrix_df))):
        col_name = study_cols[i]
        val = matrix_df.iloc[i][col_name]
        if not pd.isna(val):
            intra_values.append(val)
    
    # Cross: off-diagonal elements
    cross_values = []
    for i in range(min(n_studies, len(matrix_df))):
        for j, col_name in enumerate(study_cols):
            if i != j:
                val = matrix_df.iloc[i][col_name]
                if not pd.isna(val):
                    cross_values.append(val)
    
    # LOSO: last row
    if len(matrix_df) > n_studies:
        loso_row = matrix_df.iloc[-1]
        loso_values = []
        for col_name in study_cols:
            val = loso_row[col_name]
            if not pd.isna(val):
                loso_values.append(val)
    else:
        loso_values = []
    
    return {
        'Intra': np.mean(intra_values) if intra_values else np.nan,
        'Cross': np.mean(cross_values) if cross_values else np.nan,
        'LOSO': np.mean(loso_values) if loso_values else np.nan
    }

results_dir = '/ua/jmu27/Micro_bench/results'

print("Checking how figure2 aggregates Mental disease data")
print("=" * 80)

# Check RA preprocessing (none)
preprocessing = 'none'
mental_diseases = [d for d, t in disease_type_map.items() if t == 'Mental']

all_cross_values = []
all_cross_means = []

for disease in mental_diseases:
    for filename in os.listdir(results_dir):
        if filename.startswith(disease) and filename.endswith(f"{preprocessing}_result.csv"):
            filepath = os.path.join(results_dir, filename)
            df = pd.read_csv(filepath)
            
            # Calculate like figure2 (takes MEAN per file)
            metrics = calculate_metrics_figure2_style(df, 'TabPFN')
            
            if not pd.isna(metrics['Cross']):
                print(f"{disease} ({filename}): Cross mean = {metrics['Cross']:.4f}")
                all_cross_means.append(metrics['Cross'])
            
            # Also get raw cross values
            model_df = df[df['model'] == 'TabPFN'].copy()
            if len(model_df) > 0:
                study_cols = [col for col in model_df.columns if col != 'model']
                matrix_df = model_df[study_cols]
                n_studies = len(study_cols)
                
                for i in range(min(n_studies, len(matrix_df))):
                    for j, col_name in enumerate(study_cols):
                        if i != j:
                            val = matrix_df.iloc[i][col_name]
                            if not pd.isna(val):
                                all_cross_values.append(val)

print("\n" + "=" * 80)
print("Summary:")
print(f"  Total raw cross-cohort values: {len(all_cross_values)}")
print(f"  Min raw value: {min(all_cross_values):.4f}")
print(f"  Max raw value: {max(all_cross_values):.4f}")
print(f"\n  Total per-file means: {len(all_cross_means)}")
print(f"  Min per-file mean: {min(all_cross_means):.4f}")
print(f"  Max per-file mean: {max(all_cross_means):.4f}")

print("\n" + "=" * 80)
print("Explanation:")
print("  - Figure2 calculates MEAN per file, then plots all file means")
print("  - This aggregation reduces the number of points")
print("  - Extreme outliers (like 0.1689) are averaged with other values")
print("  - So the minimum shown in figure2 is the minimum of MEANS, not raw values")
