# Fonda-scikit

Notebook-first machine learning workflow for disturbance/deforestation modeling with yearly training, unified evaluation, and concept drift analysis.

## Overview

This repository contains four connected workflows:

1. Data preparation from private zarr datasets
2. Year-wise model training (main path: SGD with incremental scaler and monthly features)
3. Unified evaluation across model years and evaluation years
4. Concept drift detection (discriminator, distribution-distance, and synthetic drift validation)

The project is primarily notebook-driven. Historical and experimental variants are also present, but this README focuses on a reproducible canonical path.

## Repository Structure

- `Data-prep.ipynb`: canonical base data preparation
- `Data-prep_monthly.ipynb`: preparation variant with monthly-oriented features
- `merge monthly features.ipynb`: merges monthly indices into prepared data
- `SGD Classifier_prevyears and monthly features-incremental scaler.ipynb`: primary training path
- `Experience Replay-SGD Classifier_prevyears and monthly features-incremental scaler.ipynb`: replay variant
- `XGBoost.ipynb`: yearly independent XGBoost training
- `Evaluations.ipynb`: unified evaluation across model families
- `concept_drift_detection.ipynb`: discriminator drift baseline
- `concept_drift_detection-discriminator.ipynb`: year vs all previous years drift analysis
- `concept_drift_detection-distribution.ipynb`: MMD and Energy Distance drift analysis
- `data drift testing on syntetic data.ipynb`: synthetic drift dataset generation
- `src/`: shared code for preparation, loading, and utilities
- `models_*/`: trained model artifacts
- `eval_outputs/unified_eval/`: evaluation outputs
- `synthetic_drift_data/`: generated synthetic drift datasets

## Data Requirements

This repository expects local private datasets and does not include raw data downloads.

Typical inputs used by notebooks include:

- `full_dataset_resizedv2.zarr`
- `full_dataset_20m_monthly_with_indices.zarr`

Common generated artifacts consumed by downstream notebooks:

- `data_split.npz`
- `training_data.zarr`
- `training_data_enriched.zarr`
- `training_data_with_features.zarr`
- `training_data_with_features_plus_monthly_indices.zarr`
- `training_data_with_neighbourhood_features.zarr` (variant-specific)

If local paths differ, update path cells in the notebooks before running.

## Environment Setup

No pinned environment file is currently included, so start with a practical baseline:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas xarray scikit-learn scipy xgboost torch torchvision matplotlib zarr jupyter tqdm
```

## Canonical Pipeline

## 1) Data Preparation

Primary sequence:

1. Run `Data-prep.ipynb`
2. Run `merge monthly features.ipynb` (if you want monthly indices merged)

Optional variants:

- `Data-prep_monthly.ipynb`
- `NOTEBOOKS/Data-prep-all.ipynb`
- `NOTEBOOKS/Data-prep-neigbours.ipynb`

What this stage does:

1. Loads source zarr data
2. Filters/selects valid disturbance-relevant samples
3. Builds feature-ready arrays
4. Produces train/validation/test split indices
5. Writes zarr artifacts used by training and drift notebooks

Key outputs:

- `data_split.npz`
- `training_data_with_features.zarr`
- `training_data_with_features_plus_monthly_indices.zarr` (if monthly merge is run)

Referenced helper modules:

- `src/cube/prepare.py`
- `src/dataset/cubeloader.py`

## 2) Model Training

### Main Recommended Path: SGD Incremental Scaler + Monthly Features

Notebook:

- `SGD Classifier_prevyears and monthly features-incremental scaler.ipynb`

Typical behavior:

1. Trains incrementally by year (2017-2022 style workflow)
2. Uses per-year feature extraction from prepared zarr data
3. Fits and updates scaler/model over yearly batches
4. Saves yearly model and scaler artifacts

Expected output directory pattern:

- `models_prevyears_monthly_features/`

Typical file naming:

- `model_year_<YEAR>.pkl`
- `scaler_year_<YEAR>.pkl`

### Replay Variant (Optional)

Notebook:

- `Experience Replay-SGD Classifier_prevyears and monthly features-incremental scaler.ipynb`

Expected output directory pattern:

- `models_prevyears_monthly_features_experience_replay/`

### Secondary Baseline: XGBoost

Notebook:

- `XGBoost.ipynb`

Expected output directory pattern:

- `models_xgb/`

Typical file naming:

- `model_year_<YEAR>.json`

## 3) Unified Evaluation

Notebook:

- `Evaluations.ipynb`

What it does:

1. Loads trained model/scaler artifacts by family and year
2. Reconstructs family-specific features per evaluation year
3. Computes metrics such as F1, precision, recall, ROC-AUC, and PR-AUC
4. Writes consolidated outputs to evaluation folders

Output location:

- `eval_outputs/unified_eval/`

Common outputs:

- `all_families_combined.csv`
- `all_families_status.csv`
- per-family and per-scheme CSV files

If some model families are not trained, skip corresponding blocks or run those training notebooks first.

## 4) Concept Drift Detection

The repository includes three drift analysis styles.

### A) Discriminator-Based Drift

Notebooks:

- `concept_drift_detection.ipynb` (adjacent-year comparisons)
- `concept_drift_detection-discriminator.ipynb` (year vs all prior years)

Method:

1. Build binary classification tasks where label indicates source period
2. Train classifier to separate year distributions
3. Use separability metrics (for example ROC-AUC/F1) as drift indicators

Interpretation:

- Higher separability usually indicates stronger shift between compared periods
- Compare drift scores with model performance trends to identify operational risk

### B) Distribution-Distance Drift

Notebook:

- `concept_drift_detection-distribution.ipynb`

Method:

1. Computes distribution distances/statistics between year slices
2. Uses methods such as MMD (RBF/Laplacian variants) and Energy Distance
3. Applies permutation tests to estimate statistical significance

Interpretation:

- Significant distance suggests measurable covariate shift
- Use alongside classifier-based drift and performance curves for stronger conclusions

### C) Synthetic Drift Validation

Notebook:

- `data drift testing on syntetic data.ipynb`

Method:

1. Creates controlled perturbations (for example mean/covariance/noise shifts)
2. Writes synthetic zarr datasets into `synthetic_drift_data/`
3. Enables stress-testing drift detection methods against known injected drift

## How To Run The Pipeline

Use this sequence for a practical end-to-end run.

### Quick Start (Minimal)

1. Activate environment and install dependencies
2. Run `Data-prep.ipynb`
3. Run `merge monthly features.ipynb`
4. Run `SGD Classifier_prevyears and monthly features-incremental scaler.ipynb`
5. Run `Evaluations.ipynb`
6. Run `concept_drift_detection-discriminator.ipynb`

### Script-Style Checklist

```powershell
# 1) Environment
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas xarray scikit-learn scipy xgboost torch torchvision matplotlib zarr jupyter tqdm

# 2) Open notebooks
jupyter notebook

# 3) Execute in order (inside Jupyter)
#    Data-prep.ipynb
#    merge monthly features.ipynb
#    SGD Classifier_prevyears and monthly features-incremental scaler.ipynb
#    Evaluations.ipynb
#    concept_drift_detection-discriminator.ipynb
```

## Troubleshooting

### Missing data files

- Ensure private zarr datasets exist at paths referenced in notebook cells.

### Evaluation cannot find model files

- Confirm the expected model family directory contains `model_year_*` and `scaler_year_*` artifacts.

### Path errors between machines

- Update notebook path cells to your local directory layout.

### Import errors in notebooks

- Confirm the active Jupyter kernel is the same environment where dependencies were installed.

## Methodological Caveats

Current project notes indicate active work around:

- Threshold selection strategy in evaluation (retrospective vs production-realistic thresholding)
- Temporal leakage risks in some preprocessing/evaluation choices
- Ongoing MLP artifact refresh and replay experiments

Treat reported metrics according to the notebook configuration used in each run.

## Suggested Next Improvements

1. Add a pinned `requirements.txt` or `environment.yml`
2. Move feature-preparation logic into shared reusable modules used by both training and evaluation
3. Add a lightweight orchestration script for reproducible stage-by-stage execution