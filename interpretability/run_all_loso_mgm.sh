#!/bin/bash
# Batch script to run LOSO MGM attention extraction for all qualifying genus datasets

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Python script
MGM_SCRIPT="loso_mgm_attention.py"

# Logging
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Master log
MASTER_LOG="$LOG_DIR/master_mgm_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================================" | tee -a "$MASTER_LOG"
echo "LOSO MGM Attention Extraction - Batch Run" | tee -a "$MASTER_LOG"
echo "Started at: $(date)" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Dataset list (14 datasets with >= 3 cohorts at genus level)
declare -a DATASETS=(
    "AD Amplicon"
    "ASD Amplicon"
    "ASD Metagenomics"
    "Adenoma Metagenomics"
    "CD Amplicon"
    "CD Metagenomics"
    "CRC Metagenomics"
    "IBS Amplicon"
    "NAFLD Amplicon"
    "PD Amplicon"
    "RA Metagenomics"
    "T2D Amplicon"
    "UC Amplicon"
    "UC Metagenomics"
)

# Counters
TOTAL=${#DATASETS[@]}
SUCCESS=0
FAILED=0

# Function to run MGM attention extraction
run_mgm() {
    local disease=$1
    local data_type=$2
    local log_file="$LOG_DIR/${disease}_${data_type}_genus_MGM_$(date +%Y%m%d_%H%M%S).log"

    echo "----------------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo "[$((SUCCESS+FAILED+1))/$TOTAL] MGM: $disease - $data_type - genus" | tee -a "$MASTER_LOG"
    echo "Log file: $log_file" | tee -a "$MASTER_LOG"
    echo "----------------------------------------------------------------------" | tee -a "$MASTER_LOG"

    # Run with timeout (12 hours = 43200 seconds, MGM may be slower)
    # Need to activate MGM conda environment
    if timeout 43200 bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate /data/jmu27/envs/MGM && python -u $MGM_SCRIPT --disease $disease --data_type $data_type --tax_level genus" \
        > "$log_file" 2>&1; then

        echo "✓ SUCCESS (MGM): $disease - $data_type - genus" | tee -a "$MASTER_LOG"
        SUCCESS=$((SUCCESS+1))
        return 0
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "✗ TIMEOUT (MGM): $disease - $data_type - genus (exceeded 12 hours)" | tee -a "$MASTER_LOG"
        else
            echo "✗ FAILED (MGM): $disease - $data_type - genus (exit code: $EXIT_CODE)" | tee -a "$MASTER_LOG"
        fi
        FAILED=$((FAILED+1))
        return 1
    fi
    echo "" | tee -a "$MASTER_LOG"
}

# Run all datasets
for dataset in "${DATASETS[@]}"; do
    read -r disease data_type <<< "$dataset"
    run_mgm "$disease" "$data_type"
done

echo "" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "Batch Run Summary" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"
echo "Total datasets: $TOTAL" | tee -a "$MASTER_LOG"
echo "Successful:     $SUCCESS" | tee -a "$MASTER_LOG"
echo "Failed:         $FAILED" | tee -a "$MASTER_LOG"
echo "Finished at:    $(date)" | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
