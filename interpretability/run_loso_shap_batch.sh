#!/bin/bash
# Batch script to run LOSO SHAP calculation for multiple datasets

# Set error handling
set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Python script path
PYTHON_SCRIPT="loso_shap_calculation.py"

# Logging
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Function to run SHAP calculation
run_shap() {
    local disease=$1
    local data_type=$2
    local tax_level=$3
    local log_file="$LOG_DIR/${disease}_${data_type}_${tax_level}_$(date +%Y%m%d_%H%M%S).log"

    echo "============================================"
    echo "Running: $disease - $data_type - $tax_level"
    echo "Log file: $log_file"
    echo "============================================"

    python "$PYTHON_SCRIPT" \
        --disease "$disease" \
        --data_type "$data_type" \
        --tax_level "$tax_level" \
        2>&1 | tee "$log_file"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Success: $disease - $data_type - $tax_level"
    else
        echo "✗ Failed: $disease - $data_type - $tax_level"
        return 1
    fi
    echo ""
}

# Example: Run for multiple datasets
# Uncomment the lines you want to run

echo "Starting LOSO SHAP calculations..."
echo "Started at: $(date)"
echo ""

# Example configurations - uncomment and modify as needed:

# run_shap "PD" "Amplicon" "genus"
# run_shap "PD" "Metagenomics" "genus"
# run_shap "IBD" "Amplicon" "genus"
# run_shap "IBD" "Metagenomics" "genus"
# run_shap "CRC" "Amplicon" "genus"
# run_shap "CRC" "Metagenomics" "genus"

# Default: If no specific runs are uncommented, show usage
if [ $# -eq 3 ]; then
    # If arguments provided, run with those
    run_shap "$1" "$2" "$3"
else
    echo "Usage:"
    echo "  1. Edit this script and uncomment the datasets you want to run"
    echo "  2. Or run directly with arguments:"
    echo "     ./run_loso_shap_batch.sh DISEASE DATA_TYPE TAX_LEVEL"
    echo ""
    echo "Example:"
    echo "     ./run_loso_shap_batch.sh PD Amplicon genus"
    exit 1
fi

echo ""
echo "All SHAP calculations completed!"
echo "Finished at: $(date)"
