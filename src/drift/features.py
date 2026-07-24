"""Feature extraction for the concept-drift notebooks.

Deliberately separate from src/mlp_replay/data.py: drift detection compares raw
spectral distributions between years, so it uses S2 bands only, takes a pre-sliced
dataset or plain arrays, and returns (X, y) with no scaler. Folding that into the
training feature function would give one function with two unrelated branches.

Imputation here fills S2 NaNs with the per-pixel/per-band mean across *all* years,
matching what the notebooks did inline. That mixes future years into past rows; see
the impute_window discussion in src/mlp_replay/data.py. Changing it is a separate
decision from moving the code, so it is preserved exactly as-is.
"""

import numpy as np


def prepare_features_from_arrays(s2_arr, dist_arr, year_idx, x_dtype=None, y_dtype=None):
    """Build S2-band features for one year from pre-loaded numpy arrays.

    Returns (None, None) for year 0 (skipped by design) and when no rows survive
    masking, matching the notebooks' existing contract.

    x_dtype/y_dtype cast the outputs when given; the class-conditional HGB notebook
    needs (np.float32, np.uint8), the others pass nothing and keep the input dtypes.
    """
    if year_idx == 0:
        return None, None

    valid_mask = dist_arr[:, year_idx] != 255
    s2_t = s2_arr[:, year_idx, :].copy()

    nan_mask = np.isnan(s2_t)
    if np.any(nan_mask):
        band_means = np.nanmean(s2_arr, axis=1)
        s2_t[nan_mask] = band_means[nan_mask]

    X = s2_t
    y = dist_arr[:, year_idx].copy()

    keep_mask = valid_mask & np.isin(y, [0, 1]) & np.all(np.isfinite(X), axis=1)
    X = X[keep_mask]
    y = y[keep_mask]

    if len(X) == 0:
        return None, None

    if x_dtype is not None:
        X = X.astype(x_dtype)
    if y_dtype is not None:
        y = y.astype(y_dtype)

    return X, y


def prepare_features_for_year(ds_subset, year_idx, x_dtype=None, y_dtype=None):
    """Same as prepare_features_from_arrays, but takes an already-sliced xarray subset.

    ds_subset is expected to carry `s2_bands` (pixel, year, band) and `disturbances`
    (pixel, year) for the pixels of interest.
    """
    return prepare_features_from_arrays(
        ds_subset.s2_bands.values,
        ds_subset.disturbances.values,
        year_idx,
        x_dtype=x_dtype,
        y_dtype=y_dtype,
    )
