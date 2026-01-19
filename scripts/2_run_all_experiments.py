#!/usr/bin/env python
"""
Run all combinations of data and preprocessing methods.
Supports parallel execution, logging, and resume from failure.
"""
import os
# Set environment variable before importing anything (required for TabPFN)
os.environ['SCIPY_ARRAY_API'] = '1'

import sys
import glob
import subprocess
import time
from datetime import datetime
from pathlib import Path
import argparse

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
RAWDATA_DIR = PROJECT_DIR / "data" / "rawdata"
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Preprocessing methods
PREPROCESS_METHODS = ['none', 'binary', 'log_std', 'CLR', 'ALR', 'gptemb']


def get_data_combinations():
    """Extract all data combinations from rawdata directory."""
    csv_files = sorted(RAWDATA_DIR.glob("*.csv"))
    combinations = []

    for csv_file in csv_files:
        basename = csv_file.stem
        parts = basename.split('_')

        if len(parts) >= 3:
            tax_level = parts[-1]  # genus or species
            data_type = parts[-2]  # Amplicon or Metagenomics
            disease = '_'.join(parts[:-2])  # Handle multi-part disease names
            combinations.append((disease, data_type, tax_level))

    return combinations


def check_if_completed(disease, data_type, tax_level, preprocess):
    """Check if this job has already been completed."""
    result_file = RESULTS_DIR / f"{disease}_{data_type}_{tax_level}_{preprocess}_result.csv"
    return result_file.exists()


def run_single_job(job_num, total_jobs, disease, data_type, tax_level, preprocess, skip_existing=True):
    """Run a single experiment job."""

    # Check if already completed
    if skip_existing and check_if_completed(disease, data_type, tax_level, preprocess):
        print(f"[{job_num}/{total_jobs}] ⏭️  Skipping {disease}_{data_type}_{tax_level} - {preprocess} (already completed)")
        return True, 0

    # Prepare command
    cmd = [
        "python", "scripts/classicML.py",
        "--disease", disease,
        "--data_type", data_type,
        "--tax_level", tax_level,
        "--preprocess_method", preprocess
    ]

    # Log file for this job
    log_file = LOGS_DIR / f"{disease}_{data_type}_{tax_level}_{preprocess}.log"

    # Print progress
    print(f"\n{'='*60}")
    print(f"[{job_num}/{total_jobs}] Running: {disease}_{data_type}_{tax_level} - {preprocess}")
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
    parser = argparse.ArgumentParser(description="Run all microbiome ML experiments")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip jobs that already have results (default: True)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-run all jobs even if results exist")
    parser.add_argument("--preprocess", type=str, nargs='+', default=PREPROCESS_METHODS,
                        help=f"Preprocessing methods to run (default: all)")
    args = parser.parse_args()

    # Get all combinations
    data_combinations = get_data_combinations()
    print(f"Found {len(data_combinations)} data combinations")
    print(f"Preprocessing methods: {args.preprocess}")

    # Generate all jobs
    jobs = []
    for disease, data_type, tax_level in data_combinations:
        for preprocess in args.preprocess:
            jobs.append((disease, data_type, tax_level, preprocess))

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
        print(f"Starting all experiments")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Run all jobs
        start_time = time.time()
        successful = 0
        failed = 0
        skipped = 0

        for i, (disease, data_type, tax_level, preprocess) in enumerate(jobs, 1):
            success, elapsed = run_single_job(
                i, total_jobs, disease, data_type, tax_level, preprocess,
                skip_existing=args.skip_existing
            )

            if success:
                if check_if_completed(disease, data_type, tax_level, preprocess) and elapsed == 0:
                    skipped += 1
                else:
                    successful += 1
            else:
                failed += 1

        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"All experiments completed!")
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
