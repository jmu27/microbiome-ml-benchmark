#!/usr/bin/env python3
"""
Batch script to run MGM model on all genus-level datasets.
"""
import os
import subprocess
from pathlib import Path

# Define the script directory and data directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "filterd_data"
LOG_DIR = PROJECT_DIR / "logs"

# Create log directory if not exists
LOG_DIR.mkdir(exist_ok=True)

# Get all genus-level CSV files
genus_files = list(DATA_DIR.glob("*_genus.csv"))

if not genus_files:
    print("❌ No genus-level data files found!")
    exit(1)

print(f"📂 Found {len(genus_files)} genus-level datasets:")
for f in genus_files:
    print(f"   - {f.name}")

# Extract disease and data_type from filenames
tasks = []
for file in genus_files:
    # Format: {disease}_{data_type}_genus.csv
    filename = file.stem  # Remove .csv
    parts = filename.split('_')

    if len(parts) >= 3 and parts[-1] == 'genus':
        disease = parts[0]
        data_type = '_'.join(parts[1:-1])  # Handle cases like "data_type" with underscores
        tasks.append((disease, data_type))

print(f"\n{'='*60}")
print(f"🚀 Running MGM on {len(tasks)} datasets")
print(f"{'='*60}\n")

# Run each task
for idx, (disease, data_type) in enumerate(tasks, 1):
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(tasks)}] Processing: {disease} - {data_type}")
    print(f"{'='*60}")

    # Construct command
    cmd = [
        "python3",
        str(SCRIPT_DIR / "run_mgm.py"),
        "--disease", disease,
        "--data_type", data_type
    ]

    # Define log file
    log_file = LOG_DIR / f"mgm_{disease}_{data_type}_genus.log"

    print(f"📝 Logging to: {log_file}")
    print(f"▶️  Running: {' '.join(cmd)}\n")

    # Run the command and capture output
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        if result.returncode == 0:
            print(f"✅ Completed: {disease} - {data_type}")
        else:
            print(f"❌ Failed: {disease} - {data_type} (exit code: {result.returncode})")
            print(f"   Check log: {log_file}")

    except Exception as e:
        print(f"❌ Error running {disease} - {data_type}: {e}")
        continue

print(f"\n{'='*60}")
print("🎉 All MGM tasks completed!")
print(f"{'='*60}")
print(f"\n📊 Results saved in: {PROJECT_DIR / 'results'}")
print(f"📝 Logs saved in: {LOG_DIR}")
