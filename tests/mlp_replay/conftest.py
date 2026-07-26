import numpy as np
import pytest

from tests.conftest import build_synthetic_dataset

# synthetic_dataset and synthetic_pixel_indices are defined in tests/conftest.py and
# apply here automatically -- pytest resolves fixtures from parent conftest files.
# This file keeps only the fixture that is genuinely mlp_replay-specific: writing to
# an actual zarr store, which load_dataset_and_splits reads from disk.


@pytest.fixture
def synthetic_zarr_and_split(tmp_path):
    """Writes a synthetic dataset + split file to disk, for load_dataset_and_splits tests."""
    ds = build_synthetic_dataset()
    n_pixels = ds.sizes["pixel"]
    rng = np.random.default_rng(2)
    perm = rng.permutation(n_pixels)

    zarr_path = tmp_path / "training_data.zarr"
    ds.to_zarr(zarr_path, mode="w")

    split_path = tmp_path / "data_split.npz"
    np.savez(
        split_path,
        train_pixel_indices=perm[: n_pixels // 2],
        val_pixel_indices=perm[n_pixels // 2 : n_pixels * 3 // 4],
        test_pixel_indices=perm[n_pixels * 3 // 4 :],
    )
    return zarr_path, split_path
