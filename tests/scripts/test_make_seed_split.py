import numpy as np
import pytest

from scripts.make_seed_split import (
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    pixel_indices_for,
    resolve_dataset_path,
    split_cubes,
)


@pytest.fixture
def cubes():
    return np.arange(200)


# ---------------------------------------------------------------------------
# split_cubes -- the part that actually consumes the seed
# ---------------------------------------------------------------------------

def test_ratios_sum_to_one():
    assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == pytest.approx(1.0)


def test_same_seed_gives_an_identical_split(cubes):
    assert all(
        np.array_equal(a, b)
        for a, b in zip(split_cubes(cubes, 1), split_cubes(cubes, 1))
    )


def test_different_seeds_give_different_splits(cubes):
    """The whole point of the exercise -- if this held, every 'seed' would be a
    rerun of the same partition."""
    for other in (2, 3, 42):
        assert not np.array_equal(split_cubes(cubes, 1)[2], split_cubes(cubes, other)[2])


def test_the_three_sets_are_disjoint_and_cover_every_cube(cubes):
    train, val, test = split_cubes(cubes, 1)
    assert set(train) | set(val) | set(test) == set(cubes)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_proportions_match_the_configured_ratios(cubes):
    train, val, test = split_cubes(cubes, 1)
    n = len(cubes)
    assert len(test) / n == pytest.approx(TEST_RATIO, abs=0.02)
    assert len(val) / n == pytest.approx(VAL_RATIO, abs=0.02)
    assert len(train) / n == pytest.approx(TRAIN_RATIO, abs=0.02)


def test_seed_42_reproduces_the_notebooks_split(cubes):
    """Data-prep.ipynb cell 31 hardcodes random_state=42 in both calls. This script
    must reproduce it exactly at seed 42, or the existing data_split.npz would not
    be the seed-42 member of the family we are comparing against."""
    from sklearn.model_selection import train_test_split
    expected_tv, expected_test = train_test_split(cubes, test_size=TEST_RATIO, random_state=42)
    expected_train, expected_val = train_test_split(
        expected_tv, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), random_state=42,
    )
    train, val, test = split_cubes(cubes, 42)
    assert np.array_equal(train, expected_train)
    assert np.array_equal(val, expected_val)
    assert np.array_equal(test, expected_test)


# ---------------------------------------------------------------------------
# pixel_indices_for -- cube membership -> positional pixel indices
# ---------------------------------------------------------------------------

def test_pixel_indices_follow_cube_membership():
    cube_idx = np.array([0, 0, 1, 1, 2, 2, 0])
    assert np.array_equal(pixel_indices_for(cube_idx, [0]), [0, 1, 6])
    assert np.array_equal(pixel_indices_for(cube_idx, [1, 2]), [2, 3, 4, 5])


def test_pixel_split_is_a_partition_of_every_pixel():
    """Cube-wise splitting must never drop or duplicate a pixel -- a positional
    index into the zarr store is only meaningful if it is exhaustive."""
    rng = np.random.default_rng(0)
    cube_idx = rng.integers(0, 50, size=5000)
    train, val, test = split_cubes(np.unique(cube_idx), 1)
    parts = [pixel_indices_for(cube_idx, c) for c in (train, val, test)]
    combined = np.concatenate(parts)
    assert len(combined) == len(cube_idx)
    assert np.array_equal(np.sort(combined), np.arange(len(cube_idx)))


def test_no_cube_is_split_across_two_sets():
    """The anti-leakage invariant: adjacent pixels share a cube, so a cube landing
    in both train and test would leak neighbours across the boundary."""
    rng = np.random.default_rng(1)
    cube_idx = rng.integers(0, 50, size=5000)
    train, val, test = split_cubes(np.unique(cube_idx), 3)
    seen = [set(cube_idx[pixel_indices_for(cube_idx, c)]) for c in (train, val, test)]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])


# ---------------------------------------------------------------------------
# resolve_dataset_path
# ---------------------------------------------------------------------------

def test_prefers_the_first_candidate_that_exists(tmp_path):
    (tmp_path / "training_data_with_features_plus_monthly_indices.zarr").mkdir()
    resolved = resolve_dataset_path(tmp_path)
    assert resolved.name == "training_data_with_features_plus_monthly_indices.zarr"

    # Kaggle only has the monthly store; a local checkout has both and should pick
    # the base one, matching what the notebook opens.
    (tmp_path / "training_data_with_features.zarr").mkdir()
    assert resolve_dataset_path(tmp_path).name == "training_data_with_features.zarr"


def test_explicit_dataset_wins(tmp_path):
    (tmp_path / "training_data_with_features.zarr").mkdir()
    (tmp_path / "other.zarr").mkdir()
    assert resolve_dataset_path(tmp_path, "other.zarr").name == "other.zarr"


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_dataset_path(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_dataset_path(tmp_path, "nope.zarr")
