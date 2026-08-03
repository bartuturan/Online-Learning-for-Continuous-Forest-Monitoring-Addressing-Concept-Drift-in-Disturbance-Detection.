import json

import numpy as np
import pytest
import xarray as xr

from src.cube.manifest import (
    SPLIT_HASH_KEY,
    compute_pixel_identity_hash,
    manifest_path_for,
    pixel_identity_hash_from_dataset,
    read_manifest,
    read_split_hash,
    verify_dataset_matches_manifest,
    verify_split_matches_dataset,
    write_manifest,
)


def build_pixels(n_cubes=3, per_cube=4):
    """A pixel population shaped like Data-prep's output: cube-major, (y, x) sorted within cube."""
    cube_idx, cube_names, y, x = [], [], [], []
    for c in range(n_cubes):
        for i in range(per_cube):
            cube_idx.append(c)
            cube_names.append(f'mc_test_cube_{c}')
            y.append(i)
            x.append((i * 7) % 128)
    return (
        np.array(cube_idx, dtype=np.int64),
        np.array(y, dtype=np.int64),
        np.array(x, dtype=np.int64),
        np.array(cube_names),
    )


def build_dataset(cube_idx, y, x, cube_names, n_years=3):
    n_pixels = len(cube_idx)
    return xr.Dataset(
        {
            'cube_idx': (('pixel',), cube_idx.astype(np.int32)),
            'cube_name': (('pixel',), cube_names),
            'x': (('pixel',), x.astype(np.int32)),
            'y': (('pixel',), y.astype(np.int32)),
            'disturbances': (('pixel', 'year'), np.zeros((n_pixels, n_years), dtype=np.uint8)),
        },
        coords={'pixel': np.arange(n_pixels), 'year': np.arange(2016, 2016 + n_years)},
    )


def sampler_config(seed=42):
    return {'random_seed': seed, 'neighbourhood_radius': 3, 'negatives_per_disturbed_pixel': 5}


def test_hash_is_stable_across_integer_widths():
    """The notebook builds int64 in memory; zarr hands back int32. Both must agree."""
    cube_idx, y, x, names = build_pixels()
    wide = compute_pixel_identity_hash(cube_idx, y, x, names)
    narrow = compute_pixel_identity_hash(
        cube_idx.astype(np.int32), y.astype(np.int32), x.astype(np.int32), names
    )
    assert wide == narrow


def test_hash_changes_when_pixel_order_changes():
    """Reordering keeps the same pixel set but breaks every positional index into it."""
    cube_idx, y, x, names = build_pixels()
    original = compute_pixel_identity_hash(cube_idx, y, x, names)

    perm = np.array([1, 0] + list(range(2, len(cube_idx))))
    reordered = compute_pixel_identity_hash(cube_idx[perm], y[perm], x[perm], names[perm])

    assert reordered != original


def test_hash_changes_when_a_pixel_moves():
    cube_idx, y, x, names = build_pixels()
    original = compute_pixel_identity_hash(cube_idx, y, x, names)

    moved = x.copy()
    moved[5] += 1
    assert compute_pixel_identity_hash(cube_idx, y, moved, names) != original


def test_hash_changes_when_a_cube_is_renamed():
    """cube_idx is positional within the filtered set, so the names are what pin it to reality."""
    cube_idx, y, x, names = build_pixels()
    original = compute_pixel_identity_hash(cube_idx, y, x, names)

    renamed = names.copy()
    renamed[renamed == 'mc_test_cube_1'] = 'mc_other_cube'
    assert compute_pixel_identity_hash(cube_idx, y, x, renamed) != original


def test_hash_rejects_misaligned_arrays():
    cube_idx, y, x, names = build_pixels()
    with pytest.raises(ValueError, match='aligned along pixel'):
        compute_pixel_identity_hash(cube_idx[:-1], y, x, names)


def test_dataset_hash_matches_array_hash():
    cube_idx, y, x, names = build_pixels()
    ds = build_dataset(cube_idx, y, x, names)
    assert pixel_identity_hash_from_dataset(ds) == compute_pixel_identity_hash(cube_idx, y, x, names)


def test_dataset_hash_ignores_added_features():
    """Features are regenerable; pixel positions are not. Only the latter define identity."""
    cube_idx, y, x, names = build_pixels()
    ds = build_dataset(cube_idx, y, x, names)
    before = pixel_identity_hash_from_dataset(ds)

    ds['ndvi'] = (('pixel', 'year'), np.ones((ds.sizes['pixel'], ds.sizes['year'])))
    assert pixel_identity_hash_from_dataset(ds) == before


def test_dataset_hash_requires_identity_variables():
    cube_idx, y, x, names = build_pixels()
    ds = build_dataset(cube_idx, y, x, names).drop_vars('cube_name')
    with pytest.raises(KeyError, match='cube_name'):
        pixel_identity_hash_from_dataset(ds)


def test_write_and_read_manifest_round_trip(tmp_path):
    cube_idx, y, x, names = build_pixels()
    dataset_path = tmp_path / 'training_data.zarr'

    manifest = write_manifest(
        dataset_path,
        cube_idx=cube_idx,
        y=y,
        x=x,
        cube_names=names,
        config=sampler_config(),
        source_dataset='full_dataset_resizedv2.zarr',
        extra={'n_pixels': len(cube_idx)},
    )

    assert manifest_path_for(dataset_path) == tmp_path / 'training_data.manifest.json'
    on_disk = read_manifest(dataset_path)
    assert on_disk == manifest
    assert on_disk['pixel_identity_sha256'] == compute_pixel_identity_hash(cube_idx, y, x, names)
    assert on_disk['config']['random_seed'] == 42
    assert on_disk['stats']['n_pixels'] == len(cube_idx)
    assert json.loads(manifest_path_for(dataset_path).read_text(encoding='utf-8'))


def test_write_manifest_coerces_numpy_config_values(tmp_path):
    """np.int64 from a notebook cell must not make the manifest unserialisable."""
    cube_idx, y, x, names = build_pixels()
    dataset_path = tmp_path / 'training_data.zarr'

    write_manifest(
        dataset_path,
        cube_idx=cube_idx,
        y=y,
        x=x,
        cube_names=names,
        config={'random_seed': np.int64(42), 'test_mode': np.bool_(False)},
    )

    config = read_manifest(dataset_path)['config']
    assert config == {'random_seed': 42, 'test_mode': False}


def test_read_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='predates manifest support'):
        read_manifest(tmp_path / 'training_data.zarr')


def test_verify_dataset_matches_manifest(tmp_path):
    cube_idx, y, x, names = build_pixels()
    dataset_path = tmp_path / 'training_data.zarr'
    write_manifest(
        dataset_path, cube_idx=cube_idx, y=y, x=x, cube_names=names, config=sampler_config()
    )

    ds = build_dataset(cube_idx, y, x, names)
    assert verify_dataset_matches_manifest(ds, dataset_path) == read_manifest(dataset_path)[
        'pixel_identity_sha256'
    ]


def test_verify_dataset_detects_a_resampled_store(tmp_path):
    """The failure this whole module exists for: same schema, different pixels, no other symptom."""
    cube_idx, y, x, names = build_pixels()
    dataset_path = tmp_path / 'training_data.zarr'
    write_manifest(
        dataset_path, cube_idx=cube_idx, y=y, x=x, cube_names=names, config=sampler_config()
    )

    resampled_x = x.copy()
    resampled_x[3] += 1  # a different draw from the unseeded sampler
    ds = build_dataset(cube_idx, y, resampled_x, names)

    with pytest.raises(ValueError, match='Pixel identity mismatch'):
        verify_dataset_matches_manifest(ds, dataset_path)


def save_split(split_path, n_pixels, pixel_identity_sha256=None):
    arrays = {
        'train_pixel_indices': np.arange(n_pixels // 2),
        'val_pixel_indices': np.arange(n_pixels // 2, n_pixels),
    }
    if pixel_identity_sha256 is not None:
        arrays[SPLIT_HASH_KEY] = np.array(pixel_identity_sha256)
    np.savez(split_path, **arrays)


def test_split_hash_survives_an_npz_round_trip(tmp_path):
    """A 0-d unicode array is what the notebook writes; it has to come back as the same string."""
    cube_idx, y, x, names = build_pixels()
    digest = compute_pixel_identity_hash(cube_idx, y, x, names)
    split_path = tmp_path / 'data_split.npz'
    save_split(split_path, len(cube_idx), digest)

    assert read_split_hash(split_path) == digest


def test_verify_split_matches_dataset(tmp_path):
    cube_idx, y, x, names = build_pixels()
    ds = build_dataset(cube_idx, y, x, names)
    split_path = tmp_path / 'data_split.npz'
    save_split(split_path, len(cube_idx), pixel_identity_hash_from_dataset(ds))

    assert verify_split_matches_dataset(split_path, ds) is not None


def test_verify_split_detects_a_stale_split(tmp_path):
    cube_idx, y, x, names = build_pixels()
    split_path = tmp_path / 'data_split.npz'
    save_split(split_path, len(cube_idx), compute_pixel_identity_hash(cube_idx, y, x, names))

    perm = np.array([1, 0] + list(range(2, len(cube_idx))))
    reordered = build_dataset(cube_idx[perm], y[perm], x[perm], names[perm])

    with pytest.raises(ValueError, match='different pixel population'):
        verify_split_matches_dataset(split_path, reordered)


def test_verify_split_without_a_recorded_hash_returns_none(tmp_path):
    """Pre-manifest splits cannot be checked, only regenerated -- that is not an error here."""
    cube_idx, y, x, names = build_pixels()
    split_path = tmp_path / 'data_split.npz'
    save_split(split_path, len(cube_idx))

    assert verify_split_matches_dataset(split_path, build_dataset(cube_idx, y, x, names)) is None
