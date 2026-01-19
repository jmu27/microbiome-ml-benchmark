#!/usr/bin/env python
"""
Run batch effect correction for all data combinations.
Supports parallel execution, logging, and resume from failure.
Adapted from run_all_experiments.py for batch correction tasks.
"""
import os
# Set environment variable before importing anything (required for TabPFN)
os.environ['SCIPY_ARRAY_API'] = '1'

import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import argparse
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
RAWDATA_DIR = PROJECT_DIR / "data" / "rawdata"
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs" / "batch_correction"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Batch correction methods
BATCH_METHODS = ['DebiasM', 'ComBat', 'MMUPHin']

# Preprocessing methods
PREPROCESS_METHODS = ['none', 'log_std', 'CLR', 'ALR']


def get_data_combinations():
    """Extract all data combinations from rawdata directory.
    Only returns datasets with >=3 studies (required for LOSO)."""
    csv_files = sorted(RAWDATA_DIR.glob("*.csv"))
    combinations = []
    skipped = []

    for csv_file in csv_files:
        basename = csv_file.stem
        parts = basename.split('_')

        if len(parts) >= 3:
            tax_level = parts[-1]  # genus or species
            data_type = parts[-2]  # Amplicon or Metagenomics
            disease = '_'.join(parts[:-2])  # Handle multi-part disease names

            # Check if dataset has enough studies for LOSO (need >=3)
            try:
                data = pd.read_csv(csv_file, index_col=0)
                if 'pair_rank' in data.columns:
                    n_studies = data['pair_rank'].nunique()
                    if n_studies >= 3:
                        combinations.append((disease, data_type, tax_level, n_studies))
                    else:
                        skipped.append((disease, data_type, tax_level, n_studies))
                else:
                    print(f"Warning: {basename} missing 'pair_rank' column")
            except Exception as e:
                print(f"Error reading {basename}: {e}")

    return combinations, skipped


def check_if_completed(disease, data_type, tax_level, preprocess_method, batch_method):
    """Check if this job has already been completed."""
    result_file = RESULTS_DIR / f"{disease}_{data_type}_{tax_level}_{preprocess_method}_batchcorrection_result.csv"

    # If file doesn't exist, job is not completed
    if not result_file.exists():
        return False

    # If file exists, check if it contains results for this method
    try:
        df = pd.read_csv(result_file)
        # Check if any model contains this batch method (e.g., "ElasticNet_ComBat")
        method_exists = df['model'].str.contains(f"_{batch_method}$", regex=True).any()
        return method_exists
    except Exception:
        # If there's an error reading the file, assume not completed
        return False


def run_single_job(job_num, total_jobs, disease, data_type, tax_level, preprocess_method, batch_methods, skip_existing=True):
    """Run a single batch correction job (all batch methods together)."""

    # Check if already completed (check if file exists and has all methods)
    if skip_existing:
        result_file = RESULTS_DIR / f"{disease}_{data_type}_{tax_level}_{preprocess_method}_batchcorrection_result.csv"
        if result_file.exists():
            try:
                df = pd.read_csv(result_file)
                # Check if all batch methods are present
                all_methods_present = all(
                    df['model'].str.contains(f"_{method}$", regex=True).any()
                    for method in batch_methods
                )
                if all_methods_present:
                    print(f"[{job_num}/{total_jobs}] ⏭️  Skipping {disease}_{data_type}_{tax_level} - {preprocess_method} - ALL (already completed)")
                    return True, 0
            except Exception:
                pass

    # Prepare command - pass all batch methods as comma-separated
    batch_methods_str = ','.join(batch_methods)
    cmd = [
        "python", "scripts/batch_correction_loso.py",
        "--disease", disease,
        "--data_type", data_type,
        "--tax_level", tax_level,
        "--preprocess_method", preprocess_method,
        "--methods", batch_methods_str  # All methods together
    ]

    # Log file for this job
    log_file = LOGS_DIR / f"{disease}_{data_type}_{tax_level}_{preprocess_method}_ALL.log"

    # Print progress
    print(f"\n{'='*60}")
    print(f"[{job_num}/{total_jobs}] Running: {disease}_{data_type}_{tax_level} - {preprocess_method} - {batch_methods_str}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start_time = time.time()

    # Run command
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_DIR,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Job {job_num} completed successfully in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"❌ Job {job_num} failed with exit code {result.returncode}")
            print(f"   Check log: {log_file}")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Job {job_num} failed with exception: {e}")
        print(f"   Check log: {log_file}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all batch effect correction experiments")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip jobs that already have results (default: True)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-run all jobs even if results exist")
    parser.add_argument("--methods", type=str, nargs='+', default=BATCH_METHODS,
                        help=f"Batch correction methods to run (default: {BATCH_METHODS})")
    parser.add_argument("--preprocess", type=str, nargs='+', default=PREPROCESS_METHODS,
                        help=f"Preprocessing methods to run (default: {PREPROCESS_METHODS})")
    args = parser.parse_args()

    # Get all combinations (only those with >=3 studies for LOSO)
    data_combinations, skipped_combinations = get_data_combinations()
    print(f"Found {len(data_combinations)} eligible datasets (>=3 studies for LOSO)")
    print(f"Skipped {len(skipped_combinations)} datasets (<3 studies)")

    if skipped_combinations:
        print("\nSkipped datasets:")
        for disease, data_type, tax_level, n_studies in skipped_combinations:
            print(f"  {disease}_{data_type}_{tax_level}: {n_studies} studies")

    print(f"\nBatch correction methods: {args.methods}")
    print(f"Preprocessing methods: {args.preprocess}")

    # Generate all jobs: preprocessing × datasets (all batch methods run together)
    jobs = []
    for disease, data_type, tax_level, n_studies in data_combinations:
        for preprocess_method in args.preprocess:
            # Run all batch methods together for each preprocessing method
            jobs.append((disease, data_type, tax_level, preprocess_method, args.methods))

    total_jobs = len(jobs)
    print(f"\nTotal jobs: {total_jobs}")
    print(f"Skip existing: {args.skip_existing}")

    # Main log file
    main_log = LOGS_DIR / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"Main log file: {main_log}\n")

    # Redirect stdout to both console and log file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    with open(main_log, 'w') as log_f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_f)

        print(f"{'='*60}")
        print(f"Starting all batch correction experiments")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Run all jobs
        start_time = time.time()
        successful = 0
        failed = 0
        skipped = 0

        for i, (disease, data_type, tax_level, preprocess_method, batch_methods) in enumerate(jobs, 1):
            success, elapsed = run_single_job(
                i, total_jobs, disease, data_type, tax_level, preprocess_method, batch_methods,
                skip_existing=args.skip_existing
            )

            if success:
                # Check if it was skipped (elapsed == 0)
                if elapsed == 0:
                    skipped += 1
                else:
                    successful += 1
            else:
                failed += 1

        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"All batch correction experiments completed!")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Summary:")
        print(f"  Total jobs: {total_jobs}")
        print(f"  Successful: {successful}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print(f"  Results directory: {RESULTS_DIR}")
        print(f"  Main log file: {main_log}")
        print(f"{'='*60}")

        sys.stdout = original_stdout

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
