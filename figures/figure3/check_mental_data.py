#!/usr/bin/env python3
"""Check Mental disease type data for Cross-cohort TabPFN"""

import os
import pandas as pd
import numpy as np

# Disease type mapping
disease_type_map = {
    'CRC': 'Intestinal', 'T2D': 'Metabolic', 'Obesity': 'Metabolic', 'Overweight': 'Metabolic',
    'Adenoma': 'Intestinal', 'CDI': 'Intestinal', 'AD': 'Mental', 'MCI': 'Mental', 'PD': 'Mental',
    'RA': 'Autoimmun', 'MS': 'Autoimmun', 'ASD': 'Mental', 'CD': 'Intestinal', 'UC': 'Intestinal',
    'IBD': 'Intestinal', 'AS': 'Autoimmun', 'IBS': 'Intestinal', 'CFS': 'Mental',
    'JA': 'Autoimmun', 'NAFLD': 'Liver'
}

def parse_regular_result(filepath, model_name):
    """Parse regular result file"""
    df = pd.read_csv(filepath)
    
    results = {
        'intra': [],
        'cross': [],
        'loso': []
    }
    
    model_data = df[df['model'] == model_name]
    
    if len(model_data) == 0:
        return results
    
    numeric_cols = [col for col in model_data.columns if col not in ['model']]
    data_matrix = model_data[numeric_cols].values
    
    n_rows, n_cols = data_matrix.shape
    
    # Last row is LOSO baseline
    if n_rows > 0:
        last_row = data_matrix[-1, :]
        for value in last_row:
            if pd.notna(value):
                results['loso'].append(value)
    
    # All rows except last: intra (diagonal) and cross (off-diagonal)
    for i in range(n_rows - 1):
        for j in range(n_cols):
            value = data_matrix[i, j]
            if pd.notna(value):
                if i == j:
                    results['intra'].append(value)
                else:
                    results['cross'].append(value)
    
    return results

results_dir = '/ua/jmu27/Micro_bench/results'

# Get all diseases of Mental type
mental_diseases = [disease for disease, dtype in disease_type_map.items() if dtype == 'Mental']

print("Mental diseases:", mental_diseases)
print("\n" + "=" * 80)

all_cross_tabpfn = []

for disease in mental_diseases:
    print(f"\nDisease: {disease}")
    
    for filename in os.listdir(results_dir):
        if filename.startswith(disease) and filename.endswith("none_result.csv"):
            filepath = os.path.join(results_dir, filename)
            
            print(f"  File: {filename}")
            
            # Get TabPFN cross values
            tabpfn_results = parse_regular_result(filepath, 'TabPFN')
            
            if tabpfn_results['cross']:
                print(f"    TabPFN Cross values: {tabpfn_results['cross']}")
                print(f"    Min: {min(tabpfn_results['cross']):.4f}")
                print(f"    Max: {max(tabpfn_results['cross']):.4f}")
                all_cross_tabpfn.extend(tabpfn_results['cross'])

print("\n" + "=" * 80)
print("ALL Mental Cross-cohort TabPFN values:")
if all_cross_tabpfn:
    print(f"  Count: {len(all_cross_tabpfn)}")
    print(f"  Min: {min(all_cross_tabpfn):.4f}")
    print(f"  Max: {max(all_cross_tabpfn):.4f}")
    print(f"  Mean: {np.mean(all_cross_tabpfn):.4f}")
    print(f"  Values < 0.2: {[v for v in all_cross_tabpfn if v < 0.2]}")
else:
    print("  No data found")
