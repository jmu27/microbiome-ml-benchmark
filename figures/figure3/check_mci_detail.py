#!/usr/bin/env python3
"""Check MCI data in detail"""

import pandas as pd

filepath = '/ua/jmu27/Micro_bench/results/MCI_Amplicon_genus_none_result.csv'

print("Reading:", filepath)
print("\n" + "=" * 80)

df = pd.read_csv(filepath)
print("\nFull data:")
print(df)

print("\n" + "=" * 80)

# Filter for TabPFN
tabpfn_data = df[df['model'] == 'TabPFN']
print("\nTabPFN data:")
print(tabpfn_data)

print("\n" + "=" * 80)
print("\nData matrix (excluding model column):")
numeric_cols = [col for col in tabpfn_data.columns if col != 'model']
print("Numeric columns:", numeric_cols)
data_matrix = tabpfn_data[numeric_cols].values
print("Matrix shape:", data_matrix.shape)
print("\nMatrix values:")
print(data_matrix)

print("\n" + "=" * 80)
print("\nCross-cohort values (off-diagonal, excluding last row):")
n_rows, n_cols = data_matrix.shape
for i in range(n_rows - 1):  # exclude last row
    for j in range(n_cols):
        value = data_matrix[i, j]
        if pd.notna(value) and i != j:
            print(f"  Row {i}, Col {j} ({numeric_cols[j]}): {value:.4f}")
