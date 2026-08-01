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


VALID_IMPUTE_WINDOWS = ('expanding', 'all_years')


def prepare_raw_features_for_year(
    ds,
    pixel_indices,
    year_idx,
    s2_mean_per_pixel=None,
    dtype=np.float32,
    include_last_year=True,
    include_monthly=None,
    include_neighbourhood=False,
    impute_window='expanding',
    return_feature_names=False,
):
    """Extract and clean raw (unscaled) features for one year. Year 0 is skipped by design.

    include_last_year: when False, drop the previous-year NDVI/NDWI and B04/B03/B06 block.
        Serves the evaluation baseline family, which trains without lag features.
    include_monthly: None (default) includes the yearly NDVI/NDWI index features whenever the
        dataset carries them, which is what the replay notebooks rely on. Pass False to exclude
        them even on a dataset that has them — needed to score a model trained without them
        against the monthly-enriched dataset. True demands they be present and raises if not.
    include_neighbourhood: opt in to the neighbour-aggregate columns, appended after `nbr` to
        match the evaluation notebook's ordering. Only the neighbourhood dataset carries them.
    impute_window: 'expanding' fills S2 NaNs using only years <= year_idx, so no future-year
        spectral value ever reaches a past-year row. 'all_years' is the legacy full-window
        behavior still used by the evaluation and drift notebooks; it leaks and exists only so
        those notebooks can migrate without their numbers moving.
    return_feature_names: also return the column names, so callers can select or reorder by
        name instead of by position.
    dtype: cast the feature matrix to this type (and labels to int64). Pass None to leave both
        exactly as the dataset produced them, which is what the evaluation notebook expects.
    """
    if impute_window not in VALID_IMPUTE_WINDOWS:
        raise ValueError(f"Invalid impute_window '{impute_window}'. Valid options: {list(VALID_IMPUTE_WINDOWS)}")

    if year_idx == 0:
        empty = (np.empty((0, 0), dtype=dtype), np.empty((0,), dtype=np.int64))
        return (*empty, []) if return_feature_names else empty

    if s2_mean_per_pixel is None:
        if impute_window == 'expanding':
            # Only years up to and including year_idx are used, so imputation never
            # leaks future-year spectral values into past-year feature rows.
            s2_all_years = ds['s2_bands'].isel(pixel=pixel_indices, year=slice(0, year_idx + 1)).values
        else:
            s2_all_years = ds['s2_bands'].isel(pixel=pixel_indices).values
        s2_mean_per_pixel = np.nanmean(s2_all_years, axis=1)

    ds_subset = ds.isel(pixel=pixel_indices, year=year_idx)
    s2_features = ds_subset['s2_bands'].values
    if np.isnan(s2_features).any():
        s2_features = np.where(np.isnan(s2_features), s2_mean_per_pixel, s2_features)

    dem_features = ds_subset['dem'].values.reshape(-1, 1)
    ndvi_features = ds_subset['ndvi'].values.reshape(-1, 1)
    ndwi_features = ds_subset['ndwi'].values.reshape(-1, 1)

    # Keep feature order aligned with MLP training notebook.
    features_list = [
        s2_features,
        dem_features,
        ndvi_features,
        ndwi_features,
    ]
    feature_names = [f's2_{band}' for band in ds['s2_band'].values] + ['dem', 'ndvi', 'ndwi']

    if include_last_year:
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

        features_list += [ndvi_last_year, ndwi_last_year, *last_year_s2_features]
        feature_names += ['ndvi_last_year', 'ndwi_last_year'] + [f's2_last_year_{b}' for b in required_last_year_bands]

    if 'nbr' in ds.data_vars:
        features_list.append(ds_subset['nbr'].values.reshape(-1, 1))
        feature_names.append('nbr')

    if include_neighbourhood:
        for var_name in ['ndvi_neighbour', 'ndwi_neighbour', 's2_b03_neighbour', 's2_b04_neighbour']:
            if var_name in ds.data_vars:
                features_list.append(ds_subset[var_name].values.reshape(-1, 1))
                feature_names.append(var_name)

    if year_idx > 0 and 'ndvi_delta' in ds.data_vars:
        delta_year_idx = year_idx - 1
        ds_delta = ds.isel(pixel=pixel_indices, year=delta_year_idx)
        features_list.append(ds_delta['ndvi_delta'].values.reshape(-1, 1))
        features_list.append(ds_delta['ndwi_delta'].values.reshape(-1, 1))
        feature_names += ['ndvi_delta', 'ndwi_delta']
        if 'nbr_delta' in ds.data_vars:
            features_list.append(ds_delta['nbr_delta'].values.reshape(-1, 1))
            feature_names.append('nbr_delta')

    for var_name in ['years_since_last_disturbance', 'log_years_since_last_disturbance', 'ever_disturbed']:
        if var_name in ds.data_vars:
            features_list.append(ds_subset[var_name].values.reshape(-1, 1))
            feature_names.append(var_name)

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
    if include_monthly is not False:
        missing_monthly = [n for n in yearly_index_feature_names if n not in ds.data_vars]
        if include_monthly is True and missing_monthly:
            raise ValueError(f'include_monthly=True but the dataset is missing: {missing_monthly}')
        for feature_name in yearly_index_feature_names:
            if feature_name in ds.data_vars:
                features_list.append(ds_subset[feature_name].values.reshape(-1, 1))
                feature_names.append(feature_name)

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
        X_clean = np.empty((0, feature_dim), dtype=dtype or X.dtype)
        y_clean = np.empty((0,), dtype=np.int64 if dtype is not None else y.dtype)
    elif dtype is not None:
        X_clean = X_clean.astype(dtype, copy=False)
        y_clean = y_clean.astype(np.int64, copy=False)

    if return_feature_names:
        return X_clean, y_clean, feature_names
    return X_clean, y_clean


def prepare_features_for_year(
    ds,
    pixel_indices,
    year_idx,
    scaler=None,
    scaler_mode='auto',
    s2_mean_per_pixel=None,
    dtype=np.float32,
    include_last_year=True,
    include_monthly=None,
    include_neighbourhood=False,
    impute_window='expanding',
    ds_path=None,
    split_name=None,
    cache_root=None,
):
    """Extract, clean, and optionally scale features for one year. Year 0 is skipped by design.

    See prepare_raw_features_for_year for the include_* flags and impute_window.

    ds_path: when given, routes the raw-feature step through the disk-backed
    cross-process cache in disk_feature_cache.py instead of recomputing from the
    zarr store every time (see that module's docstring). s2_mean_per_pixel is
    ignored on the cached path -- a miss recomputes prepare_raw_features_for_year's
    own expanding mean, which is what happens on every call anyway when ds_path is
    omitted. Omitting ds_path (the default) leaves this function's behavior
    unchanged from before the cache existed.
    """
    from sklearn.preprocessing import StandardScaler

    valid_scaler_modes = {'auto', 'fit', 'partial_fit', 'transform', 'none'}
    if scaler_mode not in valid_scaler_modes:
        raise ValueError(f"Invalid scaler_mode '{scaler_mode}'. Valid options: {sorted(valid_scaler_modes)}")

    if ds_path is not None:
        from src.mlp_replay.disk_feature_cache import get_or_compute_year_features

        X_clean, y_clean = get_or_compute_year_features(
            ds,
            ds_path,
            pixel_indices,
            year_idx,
            split_name=split_name,
            dtype=dtype,
            include_last_year=include_last_year,
            include_monthly=include_monthly,
            include_neighbourhood=include_neighbourhood,
            impute_window=impute_window,
            cache_root=cache_root,
        )
    else:
        X_clean, y_clean = prepare_raw_features_for_year(
            ds,
            pixel_indices,
            year_idx,
            s2_mean_per_pixel=s2_mean_per_pixel,
            dtype=dtype,
            include_last_year=include_last_year,
            include_monthly=include_monthly,
            include_neighbourhood=include_neighbourhood,
            impute_window=impute_window,
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


def precompute_yearly_raw_cache(
    ds, pixel_indices, n_years, split_name, include_last_year=True, include_monthly=None,
    impute_window='expanding', ds_path=None, cache_root=None,
):
    """See disk_feature_cache.py's docstring for what ds_path enables.

    When ds_path is given, each year is fetched through the disk cache instead of
    the running-sum expanding-mean optimization below -- that optimization only
    ever pays for itself on a cold miss, which becomes rare once the cache is warm,
    so it's dropped on this path in favor of a much simpler cache-then-recompute
    call. Omitting ds_path (the default) leaves this function byte-for-byte
    unchanged.
    """
    cache = {}
    empty_years = 0

    if ds_path is not None:
        from src.mlp_replay.disk_feature_cache import get_or_compute_year_features

        for year_idx in tqdm(range(1, n_years), desc=f'Precompute {split_name}'):
            X_raw, y_raw = get_or_compute_year_features(
                ds,
                ds_path,
                pixel_indices,
                year_idx,
                split_name=split_name,
                dtype=np.float32,
                include_last_year=include_last_year,
                include_monthly=include_monthly,
                impute_window=impute_window,
                cache_root=cache_root,
            )
            cache[year_idx] = (X_raw, y_raw)
            if len(y_raw) == 0:
                empty_years += 1

        print(
            f"{split_name}: cached {len(cache)} years (disk-cache-backed), empty years={empty_years}, "
            f"sample feature dim={next((x.shape[1] for x, y in cache.values() if len(y) > 0), 0)}"
        )
        return cache

    if impute_window == 'expanding':
        # Running sum/count of non-NaN S2 values, updated with only the newest
        # year's data each iteration below. Equivalent to nanmean(years <= t) but
        # without reloading and re-averaging the whole growing window from year 0
        # on every iteration.
        s2_year0 = ds['s2_bands'].isel(pixel=pixel_indices, year=0).values
        valid0 = ~np.isnan(s2_year0)
        s2_sum = np.where(valid0, s2_year0, 0.0)
        s2_count = valid0.astype(np.int64)
    else:
        # Legacy full-window behavior: independent of year_idx, so compute once
        # rather than reloading it fresh on every iteration below.
        s2_all_years = ds['s2_bands'].isel(pixel=pixel_indices).values
        s2_mean_per_pixel = np.nanmean(s2_all_years, axis=1)

    for year_idx in tqdm(range(1, n_years), desc=f'Precompute {split_name}'):
        if impute_window == 'expanding':
            s2_year = ds['s2_bands'].isel(pixel=pixel_indices, year=year_idx).values
            valid = ~np.isnan(s2_year)
            s2_sum = s2_sum + np.where(valid, s2_year, 0.0)
            s2_count = s2_count + valid.astype(np.int64)
            s2_mean_per_pixel = np.divide(
                s2_sum, s2_count, out=np.full_like(s2_sum, np.nan, dtype=np.float64), where=s2_count > 0
            )

        X_raw, y_raw = prepare_raw_features_for_year(
            ds,
            pixel_indices,
            year_idx,
            s2_mean_per_pixel=s2_mean_per_pixel,
            dtype=np.float32,
            include_last_year=include_last_year,
            include_monthly=include_monthly,
            impute_window=impute_window,
        )
        cache[year_idx] = (X_raw, y_raw)
        if len(y_raw) == 0:
            empty_years += 1

    print(
        f"{split_name}: cached {len(cache)} years, empty years={empty_years}, "
        f"sample feature dim={next((x.shape[1] for x, y in cache.values() if len(y) > 0), 0)}"
    )
    return cache
