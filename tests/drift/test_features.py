import numpy as np
import pytest
import xarray as xr

from src.drift.features import prepare_features_for_year, prepare_features_from_arrays

N_PIXELS, N_YEARS, N_BANDS = 40, 5, 4


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    s2 = rng.normal(loc=0.3, scale=0.05, size=(N_PIXELS, N_YEARS, N_BANDS))
    dist = (rng.random(size=(N_PIXELS, N_YEARS)) < 0.3).astype(np.int64)
    return s2, dist


@pytest.fixture
def dataset(arrays):
    s2, dist = arrays
    return xr.Dataset(
        {
            "s2_bands": (("pixel", "year", "s2_band"), s2),
            "disturbances": (("pixel", "year"), dist),
        },
        coords={
            "pixel": np.arange(N_PIXELS),
            "year": np.arange(2000, 2000 + N_YEARS),
            "s2_band": [f"B{i:02d}" for i in range(N_BANDS)],
        },
    )


def test_year_zero_returns_none(arrays):
    s2, dist = arrays
    assert prepare_features_from_arrays(s2, dist, 0) == (None, None)


def test_s2_bands_only(arrays):
    s2, dist = arrays
    X, y = prepare_features_from_arrays(s2, dist, 1)
    assert X.shape == (len(y), N_BANDS), "drift features are S2 bands only, no DEM/NDVI/NDWI"
    assert not np.isnan(X).any()


def test_label_255_is_dropped(arrays):
    s2, dist = arrays
    dist = dist.copy()
    dist[:5, 1] = 255
    X, y = prepare_features_from_arrays(s2, dist, 1)
    assert 255 not in set(np.unique(y).tolist())
    assert len(y) == N_PIXELS - 5


def test_all_rows_dropped_returns_none(arrays):
    s2, dist = arrays
    dist = dist.copy()
    dist[:, 1] = 255
    assert prepare_features_from_arrays(s2, dist, 1) == (None, None)


def test_nan_imputed_from_per_pixel_band_mean_across_all_years(arrays):
    s2, dist = arrays
    s2 = s2.copy()
    dist = dist.copy()
    dist[:, 1] = 0  # keep every row so indexing below is stable
    s2[3, 1, 2] = np.nan

    X, _ = prepare_features_from_arrays(s2, dist, 1)
    expected = np.nanmean(s2[3, :, 2])
    assert X[3, 2] == pytest.approx(expected)


def test_pixel_all_nan_in_band_is_dropped(arrays):
    s2, dist = arrays
    s2 = s2.copy()
    dist = dist.copy()
    dist[:, 1] = 0
    s2[7, :, 1] = np.nan  # no year to impute from, so the row cannot be completed

    X, y = prepare_features_from_arrays(s2, dist, 1)
    assert len(y) == N_PIXELS - 1


def test_dtype_casts_are_opt_in(arrays):
    """The class-conditional HGB notebook needs float32/uint8; the others keep input dtypes."""
    s2, dist = arrays
    X_plain, y_plain = prepare_features_from_arrays(s2, dist, 1)
    assert X_plain.dtype == s2.dtype and y_plain.dtype == dist.dtype

    X_cast, y_cast = prepare_features_from_arrays(s2, dist, 1, x_dtype=np.float32, y_dtype=np.uint8)
    assert X_cast.dtype == np.float32 and y_cast.dtype == np.uint8
    assert np.allclose(X_cast, X_plain.astype(np.float32))


def test_dataset_wrapper_matches_array_form(dataset, arrays):
    s2, dist = arrays
    for year_idx in range(N_YEARS):
        from_ds = prepare_features_for_year(dataset, year_idx)
        from_arr = prepare_features_from_arrays(s2, dist, year_idx)
        if from_arr[0] is None:
            assert from_ds[0] is None
            continue
        assert np.array_equal(from_ds[0], from_arr[0])
        assert np.array_equal(from_ds[1], from_arr[1])
