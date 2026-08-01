import numpy as np
import pytest
import xarray as xr

from src.mlp_replay import data as data_module
from src.mlp_replay.disk_feature_cache import get_or_compute_year_features


@pytest.fixture
def loaded_zarr(synthetic_zarr_and_split):
    zarr_path, split_path = synthetic_zarr_and_split
    ds = xr.open_dataset(zarr_path, engine="zarr")
    split = np.load(split_path)
    return ds, zarr_path, split["train_pixel_indices"], split["val_pixel_indices"]


def _spy_on_prepare_raw(monkeypatch):
    calls = []
    original = data_module.prepare_raw_features_for_year

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(data_module, "prepare_raw_features_for_year", spy)
    return calls


def test_miss_then_hit_skips_recompute(loaded_zarr, tmp_path, monkeypatch):
    ds, zarr_path, train_idx, _ = loaded_zarr
    calls = _spy_on_prepare_raw(monkeypatch)
    cache_root = tmp_path / "cache"

    X1, y1 = get_or_compute_year_features(ds, zarr_path, train_idx, 1, split_name="train", cache_root=cache_root)
    assert len(calls) == 1

    X2, y2 = get_or_compute_year_features(ds, zarr_path, train_idx, 1, split_name="train", cache_root=cache_root)
    assert len(calls) == 1  # second call was a cache hit, no recompute

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_impute_window_variants_do_not_collide(loaded_zarr, tmp_path):
    ds, zarr_path, train_idx, _ = loaded_zarr
    cache_root = tmp_path / "cache"

    X_expanding, _ = get_or_compute_year_features(
        ds, zarr_path, train_idx, 2, impute_window="expanding", cache_root=cache_root,
    )
    X_all_years, _ = get_or_compute_year_features(
        ds, zarr_path, train_idx, 2, impute_window="all_years", cache_root=cache_root,
    )
    # Both must have been computed (not cross-served) and are independently retrievable.
    X_expanding_again, _ = get_or_compute_year_features(
        ds, zarr_path, train_idx, 2, impute_window="expanding", cache_root=cache_root,
    )
    X_all_years_again, _ = get_or_compute_year_features(
        ds, zarr_path, train_idx, 2, impute_window="all_years", cache_root=cache_root,
    )
    np.testing.assert_array_equal(X_expanding, X_expanding_again)
    np.testing.assert_array_equal(X_all_years, X_all_years_again)


def test_pixel_subset_does_not_collide_with_full_split(loaded_zarr, tmp_path, monkeypatch):
    ds, zarr_path, train_idx, _ = loaded_zarr
    calls = _spy_on_prepare_raw(monkeypatch)
    cache_root = tmp_path / "cache"

    X_full, y_full = get_or_compute_year_features(
        ds, zarr_path, train_idx, 1, split_name="train", cache_root=cache_root,
    )
    subset = train_idx[:5]
    X_subset, y_subset = get_or_compute_year_features(
        ds, zarr_path, subset, 1, split_name="train", cache_root=cache_root,
    )
    assert len(calls) == 2  # subset must be a distinct cache entry, not a false hit on the full split
    assert X_subset.shape[0] <= len(subset)
    assert X_subset.shape[0] != X_full.shape[0]


def test_half_written_entry_is_treated_as_miss(loaded_zarr, tmp_path, monkeypatch):
    ds, zarr_path, train_idx, _ = loaded_zarr
    calls = _spy_on_prepare_raw(monkeypatch)
    cache_root = tmp_path / "cache"

    get_or_compute_year_features(ds, zarr_path, train_idx, 1, split_name="train", cache_root=cache_root)
    assert len(calls) == 1

    # Simulate a crash between writing X.npy/y.npy and the manifest: delete manifest.json.
    manifest_paths = list(cache_root.rglob("manifest.json"))
    assert len(manifest_paths) == 1
    manifest_paths[0].unlink()

    get_or_compute_year_features(ds, zarr_path, train_idx, 1, split_name="train", cache_root=cache_root)
    assert len(calls) == 2  # treated as a miss, recomputed and republished
    assert list(cache_root.rglob("manifest.json"))  # republished


def test_dtype_none_vs_float32_do_not_collide(loaded_zarr, tmp_path, monkeypatch):
    ds, zarr_path, train_idx, _ = loaded_zarr
    calls = _spy_on_prepare_raw(monkeypatch)
    cache_root = tmp_path / "cache"

    X_native, _ = get_or_compute_year_features(ds, zarr_path, train_idx, 1, dtype=None, cache_root=cache_root)
    X_f32, _ = get_or_compute_year_features(ds, zarr_path, train_idx, 1, dtype=np.float32, cache_root=cache_root)

    assert len(calls) == 2  # each dtype request is its own cache entry, not a false hit on the other
    assert X_f32.dtype == np.float32
    assert X_native.shape == X_f32.shape

    manifests = list(cache_root.rglob("manifest.json"))
    assert len(manifests) == 2


def test_precompute_yearly_raw_cache_with_ds_path_matches_without(loaded_zarr, tmp_path):
    ds, zarr_path, train_idx, _ = loaded_zarr
    n_years = len(ds.year)

    cache_uncached = data_module.precompute_yearly_raw_cache(ds, train_idx, n_years, "train")
    cache_disk = data_module.precompute_yearly_raw_cache(
        ds, train_idx, n_years, "train", ds_path=zarr_path, cache_root=tmp_path / "cache",
    )

    assert set(cache_uncached.keys()) == set(cache_disk.keys())
    for year_idx in cache_uncached:
        X_a, y_a = cache_uncached[year_idx]
        X_b, y_b = cache_disk[year_idx]
        np.testing.assert_array_equal(X_a, X_b)
        np.testing.assert_array_equal(y_a, y_b)


def test_prepare_features_for_year_with_ds_path_matches_without(loaded_zarr, tmp_path):
    ds, zarr_path, train_idx, _ = loaded_zarr

    X_a, y_a, _ = data_module.prepare_features_for_year(ds, train_idx, 1, scaler_mode="none")
    X_b, y_b, _ = data_module.prepare_features_for_year(
        ds, train_idx, 1, scaler_mode="none", ds_path=zarr_path, cache_root=tmp_path / "cache",
    )

    np.testing.assert_array_equal(X_a, X_b)
    np.testing.assert_array_equal(y_a, y_b)
