from pathlib import Path

import numpy as np
import xarray as xr
from tqdm.notebook import tqdm


def load_dataset_and_splits(ds_path, split_path):
    ds_path = Path(ds_path)
    split_path = Path(split_path)

    print(f'Loading data from {ds_path}...')
    ds = xr.open_dataset(ds_path, engine='zarr')
    print('Data loaded')

    print(f'Loading split from {split_path}...')
    split_data = np.load(split_path)
    train_pixel_indices = split_data['train_pixel_indices']
    val_pixel_indices = split_data['val_pixel_indices']
    test_pixel_indices = split_data['test_pixel_indices']
    print('Split loaded')

    print('Dataset info:')
    print(f'  Total pixels: {len(ds.pixel)}')
    print(f'  Total years: {len(ds.year)}')
    print(f'  Train pixels: {len(train_pixel_indices)}')
    print(f'  Val pixels: {len(val_pixel_indices)}')
    print(f'  Test pixels: {len(test_pixel_indices)}')

    return ds, train_pixel_indices, val_pixel_indices, test_pixel_indices


def prepare_raw_features_for_year(ds, pixel_indices, year_idx, s2_mean_per_pixel=None, dtype=np.float32):
    """Extract and clean raw (unscaled) features for one year. Year 0 is skipped by design."""
    if year_idx == 0:
        return np.empty((0, 0), dtype=dtype), np.empty((0,), dtype=np.int64)

    if s2_mean_per_pixel is None:
        # Only years up to and including year_idx are used, so imputation never
        # leaks future-year spectral values into past-year feature rows.
        s2_all_years = ds['s2_bands'].isel(pixel=pixel_indices, year=slice(0, year_idx + 1)).values
        s2_mean_per_pixel = np.nanmean(s2_all_years, axis=1)

    ds_subset = ds.isel(pixel=pixel_indices, year=year_idx)
    s2_features = ds_subset['s2_bands'].values
    if np.isnan(s2_features).any():
        s2_features = np.where(np.isnan(s2_features), s2_mean_per_pixel, s2_features)

    dem_features = ds_subset['dem'].values.reshape(-1, 1)
    ndvi_features = ds_subset['ndvi'].values.reshape(-1, 1)
    ndwi_features = ds_subset['ndwi'].values.reshape(-1, 1)

    ds_prev = ds.isel(pixel=pixel_indices, year=year_idx - 1)
    ndvi_last_year = np.where(np.isnan(ds_prev['ndvi'].values.reshape(-1, 1)), 0, ds_prev['ndvi'].values.reshape(-1, 1))
    ndwi_last_year = np.where(np.isnan(ds_prev['ndwi'].values.reshape(-1, 1)), 0, ds_prev['ndwi'].values.reshape(-1, 1))

    required_last_year_bands = ['B04', 'B03', 'B06']
    band_to_idx = {band: i for i, band in enumerate(ds['s2_band'].values)}
    missing_bands = [band for band in required_last_year_bands if band not in band_to_idx]
    if missing_bands:
        raise ValueError(f"Missing required S2 bands for last-year features: {missing_bands}")

    last_year_s2_features = []
    for band in required_last_year_bands:
        band_values = ds_prev['s2_bands'].sel(s2_band=band).values
        band_values = np.where(np.isnan(band_values), 0, band_values)
        last_year_s2_features.append(band_values.reshape(-1, 1))

    # Keep feature order aligned with MLP training notebook.
    features_list = [
        s2_features,
        dem_features,
        ndvi_features,
        ndwi_features,
        ndvi_last_year,
        ndwi_last_year,
        *last_year_s2_features,
    ]

    if 'nbr' in ds.data_vars:
        features_list.append(ds_subset['nbr'].values.reshape(-1, 1))

    if year_idx > 0 and 'ndvi_delta' in ds.data_vars:
        delta_year_idx = year_idx - 1
        ds_delta = ds.isel(pixel=pixel_indices, year=delta_year_idx)
        features_list.append(ds_delta['ndvi_delta'].values.reshape(-1, 1))
        features_list.append(ds_delta['ndwi_delta'].values.reshape(-1, 1))
        if 'nbr_delta' in ds.data_vars:
            features_list.append(ds_delta['nbr_delta'].values.reshape(-1, 1))

    for var_name in ['years_since_last_disturbance', 'log_years_since_last_disturbance', 'ever_disturbed']:
        if var_name in ds.data_vars:
            features_list.append(ds_subset[var_name].values.reshape(-1, 1))

    yearly_index_feature_names = [
        'ndvi_cv_year',
        'ndvi_max_m2m_drop_year',
        'ndvi_max_year',
        'ndvi_min_year',
        'ndvi_std_year',
        'ndwi_cv_year',
        'ndwi_max_m2m_drop_year',
        'ndwi_max_year',
        'ndwi_min_year',
        'ndwi_std_year',
    ]
    for feature_name in yearly_index_feature_names:
        if feature_name in ds.data_vars:
            features_list.append(ds_subset[feature_name].values.reshape(-1, 1))

    X = np.concatenate(features_list, axis=1)
    y = ds_subset['disturbances'].values

    valid_label_mask = np.isin(y, [0, 1])
    X = X[valid_label_mask]
    y = y[valid_label_mask]

    nan_mask = ~np.isnan(X).any(axis=1)
    X_clean = X[nan_mask]
    y_clean = y[nan_mask]

    if len(X_clean) == 0:
        feature_dim = X.shape[1] if X.ndim == 2 and X.shape[0] > 0 else 0
        return np.empty((0, feature_dim), dtype=dtype), np.empty((0,), dtype=np.int64)

    return X_clean.astype(dtype, copy=False), y_clean.astype(np.int64, copy=False)


def prepare_features_for_year(ds, pixel_indices, year_idx, scaler=None, scaler_mode='auto', s2_mean_per_pixel=None, dtype=np.float32):
    """Extract, clean, and optionally scale features for one year. Year 0 is skipped by design."""
    from sklearn.preprocessing import StandardScaler

    valid_scaler_modes = {'auto', 'fit', 'partial_fit', 'transform', 'none'}
    if scaler_mode not in valid_scaler_modes:
        raise ValueError(f"Invalid scaler_mode '{scaler_mode}'. Valid options: {sorted(valid_scaler_modes)}")

    X_clean, y_clean = prepare_raw_features_for_year(
        ds,
        pixel_indices,
        year_idx,
        s2_mean_per_pixel=s2_mean_per_pixel,
        dtype=dtype,
    )

    if len(X_clean) == 0:
        return X_clean, y_clean, scaler

    if scaler_mode == 'none':
        return X_clean, y_clean, scaler

    if scaler is None:
        scaler = StandardScaler()

    if scaler_mode == 'auto':
        if hasattr(scaler, 'mean_'):
            X_clean = scaler.transform(X_clean)
        else:
            X_clean = scaler.fit_transform(X_clean)
    elif scaler_mode == 'fit':
        X_clean = scaler.fit_transform(X_clean)
    elif scaler_mode == 'partial_fit':
        scaler.partial_fit(X_clean)
        X_clean = scaler.transform(X_clean)
    elif scaler_mode == 'transform':
        if not hasattr(scaler, 'mean_'):
            raise ValueError("Scaler must be fitted before using scaler_mode='transform'.")
        X_clean = scaler.transform(X_clean)

    return X_clean, y_clean, scaler


def precompute_yearly_raw_cache(ds, pixel_indices, n_years, split_name):
    cache = {}
    empty_years = 0
    for year_idx in tqdm(range(1, n_years), desc=f'Precompute {split_name}'):
        # Recomputed per year_idx (expanding window) so no year's imputation
        # uses S2 data from years that haven't happened yet.
        s2_all_years = ds['s2_bands'].isel(pixel=pixel_indices, year=slice(0, year_idx + 1)).values
        s2_mean_per_pixel = np.nanmean(s2_all_years, axis=1)

        X_raw, y_raw = prepare_raw_features_for_year(
            ds,
            pixel_indices,
            year_idx,
            s2_mean_per_pixel=s2_mean_per_pixel,
            dtype=np.float32,
        )
        cache[year_idx] = (X_raw, y_raw)
        if len(y_raw) == 0:
            empty_years += 1

    print(
        f"{split_name}: cached {len(cache)} years, empty years={empty_years}, "
        f"sample feature dim={next((x.shape[1] for x, y in cache.values() if len(y) > 0), 0)}"
    )
    return cache
