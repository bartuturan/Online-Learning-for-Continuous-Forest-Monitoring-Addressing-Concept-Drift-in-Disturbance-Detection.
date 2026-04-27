# Sciikit-git

Notebook-first machine learning pipeline for deforestation/disturbance modeling with yearly training, multi-family evaluation, and concept drift analysis.

## What This Repository Covers

- Data preparation from private zarr datasets
- Training multiple model families (SGD variants and XGBoost) by year
- Unified evaluation across model-year and eval-year combinations
- Concept drift detection using discriminator and distribution-based methods

## Repository Map

- `Data-prep.ipynb`, `Data-prep_monthly.ipynb`: prepare feature datasets and splits
- `XGBoost.ipynb`: train yearly XGBoost models
- `Evaluations.ipynb`: unified evaluation for all model families
- `concept_drift_detection*.ipynb`: drift analysis workflows
- `NOTEBOOKS/`: additional SGD variants and historical experiment notebooks
- `src/`: shared modules for preparation, dataset loading, modeling, and visualization
- `models_*/`: trained model artifacts and scalers
- `eval_outputs/unified_eval/`: final evaluation CSV outputs

## Data Access Assumption (Private Dataset)

This project expects private local zarr data and does not include a public download link.

Expected input datasets used by notebooks include:

- `full_dataset_resizedv2.zarr`
- `training_data_with_features.zarr`
- `training_data_with_features_plus_monthly_indices.zarr`

If your local file names differ, update the dataset path cells in the notebooks.

## Environment Setup (pip)

This repository currently has no pinned dependency file. The commands below provide a practical setup baseline.

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas xarray scikit-learn xgboost torch torchvision matplotlib zarr jupyter tqdm
```

## Pipeline At A Glance

Run notebooks in this order:

1. Data preparation
2. Model training (one or more families)
3. Unified evaluation
4. Concept drift detection (optional but recommended)

## 1) Data Preparation

Primary notebooks:

- `Data-prep.ipynb`
- `Data-prep_monthly.ipynb`

Typical preparation flow:

1. Load source zarr dataset
2. Filter/select valid pixels/cubes
3. Build feature-ready datasets
4. Create train/validation/test pixel index splits

Expected artifacts:

- `data_split.npz` (pixel indices for train/val/test)
- prepared zarr datasets used by training/evaluation notebooks

Related reusable code:

- `src/cube/prepare.py` (split and prep helpers)
- `src/dataset/cubeloader.py` (dataset loader logic)

## 2) Model Training

### XGBoost

- Notebook: `XGBoost.ipynb`
- Output pattern: `models_xgb/model_year_<YEAR>.json`

### SGD Family Variants

Key notebooks are in root and `NOTEBOOKS/`, including variants for:

- baseline SGD
- huber loss
- lagged features
- neighbourhood features
- previous-years + monthly features
- incremental scaler
- experience replay
- RBF variants

Typical output pattern across families:

- `models_<family>/model_year_<YEAR>.pkl`
- `models_<family>/scaler_year_<YEAR>.pkl`

Practical recommendation:

- Start with one family first (for example, XGBoost or one SGD notebook)
- Expand to additional families after confirming outputs and paths

## 3) Unified Evaluation

Primary notebook:

- `Evaluations.ipynb`

Evaluation behavior:

1. Loads trained models and scalers by family and year
2. Prepares family-specific features for each eval year
3. Computes metrics for each model-year vs eval-year combination
4. Writes comparable CSV outputs

Expected outputs:

- `eval_outputs/unified_eval/<family>_results.csv`

Note: if some families were not trained yet, either skip those sections in the notebook or train those families first.

## 4) Concept Drift Detection

Main notebooks:

- `concept_drift_detection.ipynb`
- `concept_drift_detection-discriminator.ipynb`
- `concept_drift_detection-distribution.ipynb`
- `concept_drift_detection_with_syntetic_data-disc.ipynb`
- `data drift testing on syntetic data.ipynb`

Drift approaches implemented:

- Discriminator-based drift: train a classifier to separate years; stronger separability can indicate drift
- Distribution-based drift: statistical comparison (for example, MMD-style analysis)
- Synthetic drift validation: controlled perturbation experiments

Expected outputs for discriminator workflow:

- `concept_drift_discriminator_all_prev/results.csv`
- `concept_drift_discriminator_all_prev/results.json`
- `concept_drift_discriminator_all_prev/checkpoint.json`
- `concept_drift_discriminator_all_prev/models/`
- `concept_drift_discriminator_all_prev/scalers/`

## Minimal End-To-End Run (Recommended First Pass)

1. Activate your virtual environment
2. Run `Data-prep.ipynb` to produce dataset splits and prepared features
3. Run one training notebook (`XGBoost.ipynb` is a straightforward start)
4. Run `Evaluations.ipynb` for that trained family
5. Run one concept drift notebook (`concept_drift_detection-discriminator.ipynb`)

After this first pass succeeds, scale up to additional model families and full cross-family evaluation.

## Common Issues

- Missing zarr files
  - Ensure private dataset files exist at the paths used inside notebooks.

- Notebook path mismatches
  - Update dataset/model/output path cells for your local environment.

- Missing model/scaler files during evaluation
  - Train that model family first, or disable the corresponding evaluation block.

- Package import errors
  - Re-check your virtual environment activation and install commands.

## Current Gaps And Suggested Follow-Ups

- No pinned `requirements.txt` or `environment.yml`
- No single script/CLI entrypoint for full orchestration

Suggested follow-up improvements:

1. Add pinned dependency file for reproducibility.
2. Add a lightweight orchestration script to run the pipeline stages in order.
3. Add a quick smoke-test profile (single family, short run) for onboarding.