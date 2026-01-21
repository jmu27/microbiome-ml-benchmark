# Microbiome-ML-Benchmark
This repository provides a systematic benchmarking framework for evaluating classical machine learning and modern foundation models in microbiome-based disease prediction.

## Project Highlights
- Large-Scale Data: Utilization of a curated compendium encompassing 83 independent cohorts, covering 20 diseases across both 16S rRNA and WGS sequencing platforms.
- Diverse Benchmarking: Comparison of four major paradigms: Elastic Net, Random Forest, TabPFN, and Microbiome Generalist Models (MGM).
- Robust Evaluation: Validation strategy encompassing Intra-cohort, Cross-cohort and Leave-One-Study-Out (LOSO) Setting.

## Repository Structure
The repository is organized to ensure reproducibility of the benchmarking results:
- data: Contains raw datasets, pre-processed (filtered) data, and GPT-generated embeddings
- scripts: Modular implementation of the core pipeline, including feature selection, transformation, predictive modeling, and batch effect correction.
- interpretability: Scripts and associated data for evaluating biomarker concordance and signal stability between model pairs.
- figures: Visualizations of benchmarking results, including performance boxplot, heatmaps and biomarker overlap distributions.

