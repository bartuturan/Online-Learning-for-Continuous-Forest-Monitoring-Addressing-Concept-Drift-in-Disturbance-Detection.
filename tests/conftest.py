"""Fixtures shared across every test package.

Promoted from tests/mlp_replay/conftest.py so tests/eval/ (and any future package)
can use the same in-memory synthetic dataset without depending on a sibling
directory. tests/mlp_replay/conftest.py keeps only the zarr-writing fixture, which
is genuinely mlp_replay-specific (load_dataset_and_splits reads from disk).
"""

import numpy as np
import pytest
import xarray as xr


def build_synthetic_dataset(n_pixels=60, n_years=5, seed=0):
    rng = np.random.default_rng(seed)
    bands = ["B02", "B03", "B04", "B06"]

    s2_bands = rng.normal(loc=0.3, scale=0.05, size=(n_pixels, n_years, len(bands))).astype(np.float32)
    dem = rng.normal(loc=500, scale=50, size=(n_pixels, n_years)).astype(np.float32)
    ndvi = rng.normal(loc=0.5, scale=0.1, size=(n_pixels, n_years)).astype(np.float32)
    ndwi = rng.normal(loc=0.1, scale=0.1, size=(n_pixels, n_years)).astype(np.float32)
    disturbances = (rng.random(size=(n_pixels, n_years)) < 0.3).astype(np.int64)

    return xr.Dataset(
        {
            "s2_bands": (("pixel", "year", "s2_band"), s2_bands),
            "dem": (("pixel", "year"), dem),
            "ndvi": (("pixel", "year"), ndvi),
            "ndwi": (("pixel", "year"), ndwi),
            "disturbances": (("pixel", "year"), disturbances),
        },
        coords={
            "pixel": np.arange(n_pixels),
            "year": np.arange(2000, 2000 + n_years),
            "s2_band": bands,
        },
    )


@pytest.fixture
def synthetic_dataset():
    """An in-memory xr.Dataset matching the schema src.mlp_replay.data expects."""
    return build_synthetic_dataset()


@pytest.fixture
def synthetic_pixel_indices(synthetic_dataset):
    n_pixels = synthetic_dataset.sizes["pixel"]
    rng = np.random.default_rng(1)
    perm = rng.permutation(n_pixels)
    split = n_pixels // 2
    return perm[:split], perm[split:]
