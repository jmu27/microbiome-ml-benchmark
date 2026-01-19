# MGM Attention-based Feature Importance (LOSO)

This script extracts attention weights from fine-tuned MGM models to compute feature importance using Leave-One-Study-Out (LOSO) cross-validation.

## Overview

For each cohort in the dataset:
1. Train (fine-tune) MGM model on all other cohorts
2. Extract attention weights for all samples in the test cohort
3. Concatenate attention weights into a large matrix
4. Compute global feature importance by averaging attention across all samples

## Usage

```bash
# Basic usage
python loso_mgm_attention.py --disease PD --data_type Amplicon --tax_level genus

# For metagenomics data
python loso_mgm_attention.py --disease IBD --data_type Metagenomics --tax_level genus
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
/ua/jmu27/Micro_bench/interpretability/mgm_results/{disease}_{data_type}_{tax_level}_loso/
```

### Files Generated

1. **Attention Weight Matrices**
   - `attention_mgm.npy` / `.csv` - Attention weights (n_samples × n_features)

2. **Feature Information**
   - `feature_names.npy` - Array of feature names (genus names with g__ prefix)
   - `global_importance_mgm.csv` - Mean attention per feature

3. **Sample Information**
   - `sample_info.csv` - Sample IDs, cohorts, and true labels

4. **Metadata**
   - `metadata.txt` - Configuration and run information

## Attention Matrix Format

The attention matrix has shape `(n_total_samples, n_features)` where:
- Rows correspond to test samples (across all LOSO folds)
- Columns correspond to features (genus names)
- Values are normalized attention weights (sum to 1 per sample)

The row order matches `sample_info.csv`.

## Method Details

### Attention Extraction
For each sample, attention weights are extracted from the fine-tuned MGM model:
1. Sum attention across all transformer layers
2. Sum attention across all attention heads
3. Sum attention across all query token positions
4. Normalize per sample (sum to 1)

### Global Importance
Global feature importance is computed by:
1. Concatenating attention weights from all LOSO folds
2. Averaging attention across all test samples

### Key Differences from SHAP
- **MGM Attention**: Intrinsic model interpretability from transformer attention
- **SHAP**: Model-agnostic post-hoc explanations
- **Output format**: Same structure for easy comparison (mean_abs_shap column)

## Configuration

Key parameters:
- `RANDOM_STATE = 42` - Random seed
- `POS_LABEL = "Case"` - Target class for binary classification

## Requirements

- MGM conda environment: `/data/jmu27/envs/MGM`
- GPU recommended (CUDA)
- Sufficient disk space in `/data/jmu27/` for temporary files

## Notes

- Uses raw data (no preprocessing or transformation)
- Fine-tunes a separate MGM model for each LOSO fold
- Requires MGM CLI tools (mgm construct, mgm finetune)
- Temp files automatically cleaned up after completion
- May take several hours per dataset depending on size

## Example: Loading Results

```python
import numpy as np
import pandas as pd

# Load attention weights
attention_mgm = np.load("attention_mgm.npy")

# Load feature names
features = np.load("feature_names.npy")

# Load sample information
samples = pd.read_csv("sample_info.csv")

# Load global importance
imp_mgm = pd.read_csv("global_importance_mgm.csv")
print(imp_mgm.head(10))  # Top 10 features by attention
```

## Comparison with SHAP

Both methods produce similar output formats:
- `global_importance_mgm.csv` (from MGM attention)
- `global_importance_elasticnet.csv` (from SHAP)
- `global_importance_rf.csv` (from SHAP)
- `global_importance_tabpfn.csv` (from SHAP)

All use the column name `mean_abs_shap` for consistency, making cross-method comparisons straightforward.

## Troubleshooting

- **MGM finetune failed**: Check MGM environment is activated
- **Out of GPU memory**: Reduce batch size or use CPU (slower)
- **Corpus construction error**: Check data format and feature names
- **Missing features**: MGM tokenizer may not recognize all genera
