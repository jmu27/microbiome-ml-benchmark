#!/bin/bash
# Quick script to check SHAP calculation progress

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="$SCRIPT_DIR/shap_results"
LOG_DIR="$SCRIPT_DIR/logs"

echo "========================================================================"
echo "LOSO SHAP Calculation - Progress Check"
echo "========================================================================"
echo ""

# Check completed results
if [ -d "$RESULTS_DIR" ]; then
    echo "Completed datasets:"
    completed=$(find "$RESULTS_DIR" -type d -name "*_loso" | wc -l)
    echo "  Total: $completed"
    echo ""
    echo "  Details:"
    find "$RESULTS_DIR" -type d -name "*_loso" -exec basename {} \; | sed 's/_loso$//' | sort | sed 's/^/    /'
    echo ""
else
    echo "No results directory found yet."
    echo ""
fi

# Check running processes
echo "Running processes:"
running=$(ps aux | grep -E "loso_shap_calculation.py" | grep -v grep | wc -l)
if [ $running -gt 0 ]; then
    echo "  Active: $running"
    ps aux | grep -E "loso_shap_calculation.py" | grep -v grep | awk '{print "    " $11, $12, $13, $14, $15}'
else
    echo "  None"
fi
echo ""

# Check latest log
if [ -d "$LOG_DIR" ]; then
    latest_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "Latest log activity:"
        echo "  File: $(basename $latest_log)"
        echo "  Last 5 lines:"
        tail -5 "$latest_log" | sed 's/^/    /'
    fi
fi

echo ""
echo "========================================================================"
