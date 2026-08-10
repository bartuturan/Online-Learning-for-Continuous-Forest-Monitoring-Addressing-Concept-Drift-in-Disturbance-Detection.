"""Generate a cube-wise train/val/test split at a given seed.

This is Data-prep.ipynb cells 31 + 33 lifted into a script. The notebook hardcodes
random_state=42 in both train_test_split calls, so the only split that has ever
existed is data_split.npz. Multi-seed runs need one split per seed, and a script
is testable and re-runnable in a way a notebook cell is not.

The split is CUBE-wise, not pixel-wise, and that must not change: pixels from one
cube are spatially adjacent, so splitting at pixel level would leak neighbouring
pixels across the train/test boundary and inflate every metric.

Cheap enough to run anywhere -- it reads one variable (cube_idx) and does no
feature computation. That is what makes varying the split across seeds practical:
the expensive part of a seed run is retraining, not re-splitting.

Usage:
    python scripts/make_seed_split.py --seed 1
    python scripts/make_seed_split.py --seed 1 --dataset training_data_with_features.zarr
    python scripts/make_seed_split.py --seed 1 --dry-run    # print the split, write nothing
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cube.manifest import (  # noqa: E402
    SPLIT_HASH_KEY,
    manifest_path_for,
    pixel_identity_hash_from_dataset,
    read_manifest,
)
from src.seed_run import seed_split_path  # noqa: E402

# Same ratios as Data-prep.ipynb cell 31. Changing these would make seed runs
# incomparable to the existing results, so they are constants, not flags.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Preference order for the source store. Kaggle sessions only have the monthly
# one attached; a local checkout has both. Either works -- only cube_idx, cube_name,
# x and y are read, and those are identical across the derived stores.
DATASET_CANDIDATES = (
    "training_data_with_features.zarr",
    "training_data_with_features_plus_monthly_indices.zarr",
)


def resolve_dataset_path(root, explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"No dataset at {path}")
        return path
    for name in DATASET_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"None of {DATASET_CANDIDATES} found under {root}. Pass --dataset explicitly."
    )


def split_cubes(unique_cubes, seed):
    """Cube-wise 70/15/15. Mirrors cell 31's two-step split exactly, including the
    adjusted second ratio -- val is 15% of the whole, which is a larger fraction of
    the 85% that survives the first split."""
    cubes_train_val, cubes_test = train_test_split(
        unique_cubes, test_size=TEST_RATIO, random_state=seed,
    )
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    cubes_train, cubes_val = train_test_split(
        cubes_train_val, test_size=val_ratio_adjusted, random_state=seed,
    )
    return cubes_train, cubes_val, cubes_test


def pixel_indices_for(cube_idx_values, cubes):
    return np.where(np.isin(cube_idx_values, cubes))[0]


def describe_split(ds, name, pixel_indices):
    """Class balance for one split. Worth printing for every seed: a partition that
    lands most of the positives in one split makes that seed's metrics incomparable,
    and it is far cheaper to notice here than after a full training run."""
    labels = ds["disturbances"].values[pixel_indices].flatten()
    # int() first: np.sum on a bool array can return a fixed-width int that overflows
    # when multiplied by 100 at this pixel count (the same trap cell 31 documents).
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    total = n_pos + n_neg
    pct = (100.0 * n_pos / total) if total else float("nan")
    print(f"  {name:<11} {len(pixel_indices):>10,} pixels   "
          f"disturbed {n_pos:>9,} ({pct:.2f}%)   undisturbed {n_neg:>10,}")
    return {"n_pixels": len(pixel_indices), "n_pos": n_pos, "n_neg": n_neg, "pos_rate": pct}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed for both train_test_split calls.")
    parser.add_argument("--dataset", default=None,
                        help=f"Source zarr store. Default: first of {DATASET_CANDIDATES} that exists.")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="Report the split, write nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing split file for this seed.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_path = seed_split_path(root, seed=args.seed)
    if out_path.exists() and not args.force and not args.dry_run:
        print(f"{out_path.name} already exists. Refusing to overwrite -- a seed's split must stay "
              f"fixed once its models are trained against it, or every artifact keyed to it "
              f"silently means something else. Pass --force if you really mean to.")
        return 1

    ds_path = resolve_dataset_path(root, args.dataset)
    print(f"Seed {args.seed}   source {ds_path.name}")
    ds = xr.open_dataset(ds_path, engine="zarr")

    cube_idx_values = ds["cube_idx"].values
    unique_cubes = np.unique(cube_idx_values)
    cubes_train, cubes_val, cubes_test = split_cubes(unique_cubes, args.seed)
    print(f"\nCubes: {len(unique_cubes)} total -> "
          f"{len(cubes_train)} train / {len(cubes_val)} val / {len(cubes_test)} test")

    train_pixel_indices = pixel_indices_for(cube_idx_values, cubes_train)
    val_pixel_indices = pixel_indices_for(cube_idx_values, cubes_val)
    test_pixel_indices = pixel_indices_for(cube_idx_values, cubes_test)

    print()
    for name, idx in (("train", train_pixel_indices), ("val", val_pixel_indices),
                      ("test", test_pixel_indices)):
        describe_split(ds, name, idx)

    covered = len(train_pixel_indices) + len(val_pixel_indices) + len(test_pixel_indices)
    if covered != len(ds.pixel):
        raise ValueError(
            f"Split covers {covered:,} pixels but the dataset has {len(ds.pixel):,}. "
            f"Every pixel belongs to exactly one cube, so this should be impossible."
        )

    # Positional indices only mean anything against the exact pixel population they
    # were built from. Stamp that population's identity into the npz so a later
    # mismatch is detectable rather than silently addressing the wrong rows.
    print("\nComputing pixel identity hash...")
    pixel_identity_sha256 = pixel_identity_hash_from_dataset(ds)
    print(f"  {pixel_identity_sha256}")

    sampled_dataset_path = root / "training_data.zarr"
    try:
        sampled_manifest = read_manifest(sampled_dataset_path)
    except FileNotFoundError:
        # Expected on Kaggle, where only the derived store is attached. The hash is
        # still recorded, so the split remains verifiable against the store it was
        # actually built from -- only the cross-check against the sampler is skipped.
        print(f"  (no {manifest_path_for(sampled_dataset_path).name} -- skipping sampler "
              f"cross-check; the hash above is still recorded)")
    else:
        if sampled_manifest["pixel_identity_sha256"] != pixel_identity_sha256:
            raise ValueError(
                f"Pixel identity mismatch -- refusing to write a split that cannot be trusted.\n"
                f"  {manifest_path_for(sampled_dataset_path).name}: "
                f"{sampled_manifest['pixel_identity_sha256']}\n"
                f"  {ds_path.name}: {pixel_identity_sha256}\n"
                f"The feature dataset does not hold the pixels the sampler produced."
            )
        print(f"  matches {manifest_path_for(sampled_dataset_path).name}")

    if args.dry_run:
        print(f"\n--dry-run: would write {out_path.name}")
        return 0

    np.savez(
        out_path,
        train_pixel_indices=train_pixel_indices,
        val_pixel_indices=val_pixel_indices,
        test_pixel_indices=test_pixel_indices,
        train_cube_indices=cubes_train,
        val_cube_indices=cubes_val,
        test_cube_indices=cubes_test,
        **{SPLIT_HASH_KEY: np.array(pixel_identity_sha256)},
    )
    print(f"\nWrote {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
