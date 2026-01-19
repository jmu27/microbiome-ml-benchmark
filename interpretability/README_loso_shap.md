# LOSO SHAP Calculation

This script calculates SHAP values for ElasticNet, Random Forest, and TabPFN models using Leave-One-Study-Out (LOSO) cross-validation.

## Overview

For each cohort in the dataset:
1. Train models on all other cohorts
2. Calculate SHAP values for all samples in the test cohort
3. Concatenate results into large matrices

## Usage

```bash
# Basic usage
python loso_shap_calculation.py --disease PD --data_type Amplicon --tax_level genus

# For metagenomics data
python loso_shap_calculation.py --disease IBD --data_type Metagenomics --tax_level genus
```

## Arguments

- `--disease`: Disease name (e.g., 'PD', 'IBD', 'CRC')
- `--data_type`: Data type ('Amplicon' or 'Metagenomics')
- `--tax_level`: Taxonomic level (default: 'genus')

## Input

The script expects filtered data files in:
```
/ua/jmu27/Micro_bench/data/filterd_data/{disease}_{data_type}_{tax_level}.csv
```

Data must contain:
- Feature columns starting with "ncbi"
- "Group" column for labels
- "pair_rank" column for cohort identification

## Output

Results are saved to:
```
/ua/jmu27/Micro_bench/interpretability/shap_results/{disease}_{data_type}_{tax_level}_loso/
```

### Files Generated

1. **SHAP Value Matrices**
   - `shap_elasticnet.npy` / `.csv` - SHAP values for ElasticNet (n_samples × n_features)
   - `shap_rf.npy` / `.csv` - SHAP values for Random Forest
   - `shap_tabpfn.npy` / `.csv` - SHAP values for TabPFN

2. **Feature Information**
   - `feature_names.npy` - Array of feature names
   - `global_importance_elasticnet.csv` - Mean |SHAP| per feature (ElasticNet)
   - `global_importance_rf.csv` - Mean |SHAP| per feature (RF)
   - `global_importance_tabpfn.csv` - Mean |SHAP| per feature (TabPFN)

3. **Sample Information**
   - `sample_info.csv` - Sample IDs, cohorts, and true labels

4. **Metadata**
   - `metadata.txt` - Configuration and run information

## SHAP Matrix Format

Each SHAP matrix has shape `(n_total_samples, n_features)` where:
- Rows correspond to test samples (across all LOSO folds)
- Columns correspond to features
- Values are SHAP attributions

The row order matches `sample_info.csv`.

## Configuration

Key parameters (modify in script if needed):
- `RANDOM_STATE = 42` - Random seed
- `BG_MAX = 128` - Background samples for SHAP
- `NSAMPLES = 100` - Kernel SHAP sampling strength

## Notes

- Uses raw data (no preprocessing or transformation)
- ElasticNet uses LinearExplainer (fast)
- Random Forest uses TreeExplainer (fast)
- TabPFN uses KernelExplainer (slow, may take hours for large datasets)
- Requires GPU for TabPFN (CUDA)

## Example: Loading Results

```python
import numpy as np
import pandas as pd

# Load SHAP values
shap_enet = np.load("shap_elasticnet.npy")
shap_rf = np.load("shap_rf.npy")
shap_tabpfn = np.load("shap_tabpfn.npy")

# Load feature names
features = np.load("feature_names.npy")

# Load sample information
samples = pd.read_csv("sample_info.csv")

# Load global importance
imp_enet = pd.read_csv("global_importance_elasticnet.csv")
print(imp_enet.head(10))  # Top 10 features
```

## Troubleshooting

- **Out of memory**: Reduce `BG_MAX` or process cohorts separately
- **TabPFN slow**: Reduce `NSAMPLES` (accuracy tradeoff)
- **GPU not found**: TabPFN will use CPU (much slower)
