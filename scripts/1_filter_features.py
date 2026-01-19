#!/usr/bin/env python3
"""
Filter uncommon features from microbiome count tables.

This script processes all CSV files in the rawdata directory and filters out
features with low relative abundance across all samples.
"""

import os
import pandas as pd
from pathlib import Path


def filter_features_by_relative_abundance(
    table: pd.DataFrame,
    min_rel_abundance: float = 1e-4,
) -> pd.DataFrame:
    """
    Drop features whose maximum relative abundance is below a threshold.

    This is useful for removing extremely rare taxa.

    Parameters
    ----------
    table : pd.DataFrame
        Count table (samples × features).
    min_rel_abundance : float
        Minimum relative abundance in at least one sample.

    Returns
    -------
    pd.DataFrame
        Filtered table (same scale as input, counts).
    """
    # compute relative abundances per sample
    rel = table.div(table.sum(axis=1), axis=0)
    max_rel = rel.max(axis=0)
    keep_features = max_rel >= min_rel_abundance
    return table.loc[:, keep_features]


def identify_feature_columns(df: pd.DataFrame) -> list:
    """
    Identify which columns contain feature data (ncbi_* columns).

    Parameters
    ----------
    df : pd.DataFrame
        The full dataframe with both features and metadata.

    Returns
    -------
    list
        List of column names that are features.
    """
    # Feature columns typically start with 'ncbi_'
    feature_cols = [col for col in df.columns if col.startswith('ncbi_')]
    return feature_cols


def process_csv_file(input_path: str, output_path: str, min_rel_abundance: float = 1e-4):
    """
    Process a single CSV file: load, filter features, and save.

    Parameters
    ----------
    input_path : str
        Path to input CSV file.
    output_path : str
        Path to output filtered CSV file.
    min_rel_abundance : float
        Minimum relative abundance threshold.
    """
    print(f"Processing {os.path.basename(input_path)}...")

    # Load the data
    df = pd.read_csv(input_path, index_col=0)

    # Identify feature and metadata columns
    feature_cols = identify_feature_columns(df)
    metadata_cols = [col for col in df.columns if col not in feature_cols]

    print(f"  - Total columns: {len(df.columns)}")
    print(f"  - Feature columns: {len(feature_cols)}")
    print(f"  - Metadata columns: {len(metadata_cols)}")

    # Extract features and metadata
    features = df[feature_cols]
    metadata = df[metadata_cols]

    # Filter features by relative abundance
    filtered_features = filter_features_by_relative_abundance(
        features,
        min_rel_abundance=min_rel_abundance
    )

    print(f"  - Features after filtering: {len(filtered_features.columns)}")
    print(f"  - Features removed: {len(feature_cols) - len(filtered_features.columns)}")

    # Combine filtered features with metadata
    filtered_df = pd.concat([filtered_features, metadata], axis=1)

    # Save the filtered data
    filtered_df.to_csv(output_path)
    print(f"  - Saved to {os.path.basename(output_path)}")
    print()

    return len(feature_cols), len(filtered_features.columns)


def main():
    """Main function to process all CSV files."""
    # Define paths
    rawdata_dir = Path("/ua/jmu27/Micro_bench/data/rawdata")
    filtered_dir = Path("/ua/jmu27/Micro_bench/data/filterd_data")

    # Create output directory if it doesn't exist
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # Set minimum relative abundance threshold
    min_rel_abundance = 0.001

    print(f"Filtering features with minimum relative abundance: {min_rel_abundance}\n")
    print("=" * 70)

    # Get all CSV files in rawdata directory
    csv_files = sorted(rawdata_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {rawdata_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process\n")

    # Track statistics
    total_stats = []

    # Process each CSV file
    for csv_file in csv_files:
        output_file = filtered_dir / csv_file.name
        original_count, filtered_count = process_csv_file(
            str(csv_file),
            str(output_file),
            min_rel_abundance=min_rel_abundance
        )
        total_stats.append((csv_file.name, original_count, filtered_count))

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'File':<40} {'Original':<12} {'Filtered':<12} {'Removed':<12}")
    print("-" * 70)
    for filename, original, filtered in total_stats:
        removed = original - filtered
        print(f"{filename:<40} {original:<12} {filtered:<12} {removed:<12}")

    print("-" * 70)
    total_original = sum(s[1] for s in total_stats)
    total_filtered = sum(s[2] for s in total_stats)
    total_removed = total_original - total_filtered
    print(f"{'TOTAL':<40} {total_original:<12} {total_filtered:<12} {total_removed:<12}")
    print("=" * 70)
    print(f"\nAll filtered files saved to: {filtered_dir}")


if __name__ == "__main__":
    main()
