# Figure S2: Batch Correction Comparison with Log-Ratio Preprocessing

This directory contains scripts to generate Figure S2, which shows batch correction comparison using log-ratio preprocessing methods (CLR or ALR).

## Files

1. **plot_batchcorrection_boxplot_logratio.py** - Main plotting script that generates boxplots for each disease type
2. **combine_plots.py** - Script to combine individual disease type plots into a single figure

## Key Differences from Figure 3

- **Preprocessing**: Uses log-ratio preprocessing (CLR by default) instead of 'none'
- **Input Data**: Reads from `*_CLR_result.csv` and `*_CLR_batchcorrection_result.csv` files
- **Output**: Includes preprocessing method in filename and title

## Usage

### Generate individual disease type plots:
```bash
python plot_batchcorrection_boxplot_logratio.py
```

This will create separate PNG files for each disease type:
- `Autoimmun_batchcorrection_boxplot_CLR.png`
- `Intestinal_batchcorrection_boxplot_CLR.png`
- `Liver_batchcorrection_boxplot_CLR.png`
- `Mental_batchcorrection_boxplot_CLR.png`
- `Metabolic_batchcorrection_boxplot_CLR.png`

### Combine plots into a single figure:
```bash
python combine_plots.py
```

This will create:
- `combined_batchcorrection_boxplot_CLR.png`

## Changing Preprocessing Method

To use ALR instead of CLR, modify the `preprocessing` variable in both scripts:

**In plot_batchcorrection_boxplot_logratio.py (line 779):**
```python
preprocessing = 'ALR'  # Changed from 'CLR' to 'ALR'
```

**In combine_plots.py (line 65):**
```python
preprocessing = 'ALR'  # Changed from 'CLR' to 'ALR'
```

## Output Structure

Each disease type plot contains three panels:
1. **Intra-cohort**: Performance when training and testing on the same cohort
2. **Cross-cohort**: Performance when training on one cohort and testing on another
3. **Leave-One-Study-Out (LOSO)**: Performance with batch correction methods
   - Baseline (no batch correction)
   - DebiasM
   - ComBat
   - MMUPHin

Statistical significance is shown using brackets with asterisks:
- `*` : p < 0.05
- `**` : p < 0.01
- `***` : p < 0.001

All p-values are adjusted using Benjamini-Hochberg correction.
