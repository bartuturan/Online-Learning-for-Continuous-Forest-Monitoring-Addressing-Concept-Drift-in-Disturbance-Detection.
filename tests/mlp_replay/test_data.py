import numpy as np
import pytest
import xarray as xr

from src.mlp_replay.data import (
    load_dataset_and_splits,
    precompute_yearly_raw_cache,
    prepare_features_for_year,
    prepare_raw_features_for_year,
)


def test_year_zero_is_skipped(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    X, y = prepare_raw_features_for_year(synthetic_dataset, train_idx, year_idx=0)
    assert X.shape == (0, 0)
    assert y.shape == (0,)


def test_prepare_raw_features_shapes_and_dtypes(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    X, y = prepare_raw_features_for_year(synthetic_dataset, train_idx, year_idx=1)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] <= len(train_idx)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert set(np.unique(y).tolist()).issubset({0, 1})
    assert not np.isnan(X).any()


def test_prepare_raw_features_missing_band_raises(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    ds_missing_band = synthetic_dataset.sel(s2_band=["B02", "B03"])  # drops required B04/B06
    with pytest.raises(ValueError, match="Missing required S2 bands"):
        prepare_raw_features_for_year(ds_missing_band, train_idx, year_idx=1)


def test_prepare_raw_features_drops_invalid_labels(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    ds = synthetic_dataset.copy(deep=True)
    ds["disturbances"].values[train_idx[0], 1] = 9  # invalid label, must be filtered out
    X, y = prepare_raw_features_for_year(ds, train_idx, year_idx=1)
    assert 9 not in set(np.unique(y).tolist())
    assert len(y) <= len(train_idx) - 1


def test_prepare_raw_features_nan_rows_dropped(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    ds = synthetic_dataset.copy(deep=True)
    ds["dem"].values[train_idx[0], 1] = np.nan
    X_before, y_before = prepare_raw_features_for_year(synthetic_dataset, train_idx, year_idx=1)
    X_after, y_after = prepare_raw_features_for_year(ds, train_idx, year_idx=1)
    assert len(y_after) == len(y_before) - 1


def test_prepare_raw_features_empty_result_shape(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    ds = synthetic_dataset.copy(deep=True)
    ds["disturbances"].values[:, 1] = 9  # every label invalid -> nothing survives
    X, y = prepare_raw_features_for_year(ds, train_idx, year_idx=1)
    assert X.shape[0] == 0
    assert y.shape[0] == 0
    assert X.ndim == 2


@pytest.mark.parametrize("scaler_mode", ["auto", "fit", "partial_fit", "none"])
def test_prepare_features_for_year_scaler_modes(synthetic_dataset, synthetic_pixel_indices, scaler_mode):
    train_idx, _ = synthetic_pixel_indices
    X, y, scaler = prepare_features_for_year(synthetic_dataset, train_idx, year_idx=1, scaler_mode=scaler_mode)
    assert X.shape[0] == y.shape[0]
    if scaler_mode == "none":
        assert scaler is None
    else:
        assert hasattr(scaler, "mean_")


def test_prepare_features_for_year_transform_requires_fitted_scaler(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    with pytest.raises(ValueError, match="must be fitted"):
        prepare_features_for_year(synthetic_dataset, train_idx, year_idx=1, scaler_mode="transform")


def test_prepare_features_for_year_invalid_scaler_mode(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    with pytest.raises(ValueError, match="Invalid scaler_mode"):
        prepare_features_for_year(synthetic_dataset, train_idx, year_idx=1, scaler_mode="bogus")


def test_prepare_features_for_year_auto_mode_fits_then_transforms(synthetic_dataset, synthetic_pixel_indices):
    train_idx, _ = synthetic_pixel_indices
    X1, y1, scaler = prepare_features_for_year(synthetic_dataset, train_idx, year_idx=1, scaler_mode="auto")
    X2, y2, scaler2 = prepare_features_for_year(synthetic_dataset, train_idx, year_idx=2, scaler_mode="auto", scaler=scaler)
    assert scaler2 is scaler
    # scaler was already fitted, so year-2 features are transform-only (not fit again)
    assert not np.allclose(scaler.mean_, X2.mean(axis=0))


def test_precompute_yearly_raw_cache(monkeypatch, synthetic_dataset, synthetic_pixel_indices):
    import src.mlp_replay.data as data_mod

    monkeypatch.setattr(data_mod, "tqdm", lambda it, **kw: it)
    train_idx, _ = synthetic_pixel_indices
    n_years = synthetic_dataset.sizes["year"]

    cache = precompute_yearly_raw_cache(synthetic_dataset, train_idx, n_years, "train")

    assert set(cache.keys()) == set(range(1, n_years))
    for year_idx, (X, y) in cache.items():
        assert X.shape[0] == y.shape[0]


def test_load_dataset_and_splits(synthetic_zarr_and_split, capsys):
    zarr_path, split_path = synthetic_zarr_and_split
    ds, train_idx, val_idx, test_idx = load_dataset_and_splits(zarr_path, split_path)

    assert isinstance(ds, xr.Dataset)
    assert len(train_idx) + len(val_idx) + len(test_idx) == ds.sizes["pixel"]
    captured = capsys.readouterr()
    assert "Data loaded" in captured.out
    assert "Split loaded" in captured.out
